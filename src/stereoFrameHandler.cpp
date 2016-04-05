#include <stereoFrameHandler.h>
#include <future>

#pragma message("TODO: introduce in the general class, with the rest of auxiliar functions")
struct sort_descriptor_by_queryIdx
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b){
        return ( a[0].queryIdx < b[0].queryIdx );
    }
};

double vector_stdv_mad( VectorXf residues)
{
    // Return the standard deviation of vector with MAD estimation
    int n_samples = residues.size();
    sort( residues.derived().data(),residues.derived().data()+residues.size());
    double median = residues( n_samples/2 );
    residues << ( residues - VectorXf::Constant(n_samples,median) ).cwiseAbs();
    sort(residues.derived().data(),residues.derived().data()+residues.size());
    double MAD = residues( n_samples/2 );
    return 1.4826 * MAD;
}

StereoFrameHandler::StereoFrameHandler( PinholeStereoCamera *cam_ ) : cam(cam_) {}

StereoFrameHandler::~StereoFrameHandler(){}

#pragma message("TODO: not pass the camera as argument")

void StereoFrameHandler::initialize(const Mat &img_l_, const Mat &img_r_ , const int idx_)
{
    prev_frame = new StereoFrame( img_l_, img_r_, idx_, cam, Matrix4d::Identity() );
    prev_frame->extractStereoFeatures();
}

void StereoFrameHandler::insertStereoPair(const Mat &img_l_, const Mat &img_r_ , const int idx_)
{
    curr_frame = new StereoFrame( img_l_, img_r_, idx_, cam );
    curr_frame->extractStereoFeatures();
    f2fTracking();
}

void StereoFrameHandler::insertStereoPair(const Mat &img_l_, const Mat &img_r_ , const int idx_, Matrix4d DT_ini)
{
    curr_frame = new StereoFrame( img_l_, img_r_, idx_, cam, DT_ini );
    curr_frame->extractStereoFeatures();
    f2fTracking();
}

void StereoFrameHandler::f2fTracking()
{

    // points f2f tracking
    matched_pt.clear();
    if( Config::hasPoints() && !(curr_frame->stereo_pt.size()==0) && !(prev_frame->stereo_pt.size()==0)  )
    {
        BFMatcher* bfm = new BFMatcher( NORM_HAMMING, false );    // cross-check
        Mat pdesc_l1, pdesc_l2;
        vector<vector<DMatch>> pmatches_12, pmatches_21;
        // 12 and 21 matches
        pdesc_l1 = prev_frame->pdesc_l;
        pdesc_l2 = curr_frame->pdesc_l;        
        if( Config::bestLRMatches() )
        {
            if( Config::lrInParallel() )
            {
                auto match_l = async( launch::async, &StereoFrameHandler::matchPointFeatures, this, bfm, pdesc_l1, pdesc_l2, ref(pmatches_12) );
                auto match_r = async( launch::async, &StereoFrameHandler::matchPointFeatures, this, bfm, pdesc_l2, pdesc_l1, ref(pmatches_21) );
                match_l.wait();
                match_r.wait();
            }
            else
            {
                bfm->knnMatch( pdesc_l1, pdesc_l2, pmatches_12, 2);
                bfm->knnMatch( pdesc_l2, pdesc_l1, pmatches_21, 2);
            }
        }
        else
            bfm->knnMatch( pdesc_l1, pdesc_l2, pmatches_12, 2);

        // ---------------------------------------------------------------------
        // sort matches by the distance between the best and second best matches
        #pragma message("TODO: try robust standard deviation (MAD)")
        double nn_dist_th, nn12_dist_th;
        curr_frame->pointDescriptorMAD( pmatches_12, nn_dist_th, nn12_dist_th );
        nn_dist_th    = nn_dist_th   * Config::descThP();
        nn12_dist_th  = nn12_dist_th * Config::descThP();
        // ---------------------------------------------------------------------

        // resort according to the queryIdx
        sort( pmatches_12.begin(), pmatches_12.end(), sort_descriptor_by_queryIdx() );
        if( Config::bestLRMatches() )
            sort( pmatches_21.begin(), pmatches_21.end(), sort_descriptor_by_queryIdx() );

        // bucle around pmatches
        for( int i = 0; i < pmatches_12.size(); i++ )
        {
            // check if they are mutual best matches
            int lr_qdx = pmatches_12[i][0].queryIdx;
            int lr_tdx = pmatches_12[i][0].trainIdx;
            int rl_tdx;
            if( Config::bestLRMatches() )
                rl_tdx = pmatches_21[lr_tdx][0].trainIdx;
            else
                rl_tdx = lr_qdx;
            // check if they are mutual best matches and the minimum distance
            double dist_nn = pmatches_12[i][0].distance;
            double dist_12 = pmatches_12[i][1].distance - pmatches_12[i][0].distance;
            if( lr_qdx == rl_tdx  && dist_12 > nn12_dist_th && dist_nn < nn_dist_th )
            {
                #pragma message("TODO: check f2f consistency")
                PointFeature* point_ = prev_frame->stereo_pt[lr_qdx];
                point_->pl_obs = curr_frame->stereo_pt[lr_tdx]->pl;
                point_->inlier = true;
                matched_pt.push_back( point_ );
            }
        }
    }

    // line segments f2f tracking
    matched_ls.clear();
    if( Config::hasLines() && !(curr_frame->stereo_ls.size()==0) && !(prev_frame->stereo_ls.size()==0)  )
    {
        Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
        Mat ldesc_l1, ldesc_l2;
        vector<vector<DMatch>> lmatches_12, lmatches_21;
        // 12 and 21 matches
        ldesc_l1 = prev_frame->ldesc_l;
        ldesc_l2 = curr_frame->ldesc_l;
        if( Config::bestLRMatches() )
        {
            if( Config::lrInParallel() )
            {
                auto match_l = async( launch::async, &StereoFrameHandler::matchLineFeatures, this, bdm, ldesc_l1, ldesc_l2, ref(lmatches_12) );
                auto match_r = async( launch::async, &StereoFrameHandler::matchLineFeatures, this, bdm, ldesc_l2, ldesc_l1, ref(lmatches_21) );
                match_l.wait();
                match_r.wait();
            }
            else
            {
                bdm->knnMatch( ldesc_l1, ldesc_l2, lmatches_12, 2);
                bdm->knnMatch( ldesc_l2, ldesc_l1, lmatches_21, 2);
            }
        }
        else
            bdm->knnMatch( ldesc_l1, ldesc_l2, lmatches_12, 2);

        // ---------------------------------------------------------------------
        // sort matches by the distance between the best and second best matches
        #pragma message("TODO: try robust standard deviation (MAD)")
        double nn_dist_th, nn12_dist_th;
        curr_frame->pointDescriptorMAD( lmatches_12, nn_dist_th, nn12_dist_th );
        nn_dist_th    = nn_dist_th   * Config::descThL();
        nn12_dist_th  = nn12_dist_th * Config::descThL();
        // ---------------------------------------------------------------------

        // resort according to the queryIdx
        sort( lmatches_12.begin(), lmatches_12.end(), sort_descriptor_by_queryIdx() );
        if( Config::bestLRMatches() )
            sort( lmatches_21.begin(), lmatches_21.end(), sort_descriptor_by_queryIdx() );
        // bucle around pmatches
        for( int i = 0; i < lmatches_12.size(); i++ )
        {
            // check if they are mutual best matches
            int lr_qdx = lmatches_12[i][0].queryIdx;
            int lr_tdx = lmatches_12[i][0].trainIdx;
            int rl_tdx;
            if( Config::bestLRMatches() )
                rl_tdx = lmatches_21[lr_tdx][0].trainIdx;
            else
                rl_tdx = lr_qdx;
            // check if they are mutual best matches and the minimum distance
            double dist_nn = lmatches_12[i][0].distance;
            double dist_12 = lmatches_12[i][1].distance - lmatches_12[i][0].distance;
            if( lr_qdx == rl_tdx  && dist_12 > nn12_dist_th && dist_nn < nn_dist_th )
            {
                #pragma message("TODO: check f2f consistency")
                LineFeature* line_ = prev_frame->stereo_ls[lr_qdx];
                line_->spl_obs = curr_frame->stereo_ls[lr_tdx]->spl;
                line_->epl_obs = curr_frame->stereo_ls[lr_tdx]->epl;
                line_->le_obs  = curr_frame->stereo_ls[lr_tdx]->le;
                line_->inlier  = true;
                matched_ls.push_back( line_ );
            }
        }
    }

    n_inliers_pt = matched_pt.size();
    n_inliers_ls = matched_ls.size();
    n_inliers    = n_inliers_pt + n_inliers_ls;

}

void StereoFrameHandler::matchPointFeatures(BFMatcher* bfm, Mat pdesc_1, Mat pdesc_2, vector<vector<DMatch>> &pmatches_12  )
{
    bfm->knnMatch( pdesc_1, pdesc_2, pmatches_12, 2);
}

void StereoFrameHandler::matchLineFeatures(Ptr<BinaryDescriptorMatcher> bdm, Mat ldesc_1, Mat ldesc_2, vector<vector<DMatch>> &lmatches_12  )
{
    bdm->knnMatch( ldesc_1, ldesc_2, lmatches_12, 2);
}

void StereoFrameHandler::updateFrame()
{
    matched_pt.clear();
    matched_ls.clear();
    prev_frame = curr_frame;
    curr_frame = NULL;
}

void StereoFrameHandler::optimizePose()
{

    // definitions
    Matrix6d DT_cov;
    Matrix4d DT, DT_;
    double err, err_prev = 999999999.9;

    #pragma message("TODO: implement some logic to select the initial pose")
    // set init pose    (depending on the values of DT_cov_eig)
    if( true )
    {
        DT     = prev_frame->DT;
        DT_cov = prev_frame->DT_cov;
    }

    // Gauss-Newton solver
    if( n_inliers > Config::minFeatures() )
    {
        // optimize
        DT_ = DT;
        gaussNewtonOptimization(DT_,DT_cov);

        // remove outliers (implement some logic based on the covariance's eigenvalues)
        if( is_finite(DT_) )
            removeOutliers(DT_);

        // refine without outliers
        if( n_inliers > Config::minFeatures() )
            gaussNewtonOptimization(DT,DT_cov);
        else
        {
            DT     = Matrix4d::Identity();
            DT_cov = Matrix6d::Zero();
        }

    }
    else
    {
        DT     = Matrix4d::Identity();
        DT_cov = Matrix6d::Zero();
    }

    // set estimated pose
    curr_frame->DT     = inverse_transformation( DT );  //check what's best
    curr_frame->DT_cov = DT_cov;
    SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
    curr_frame->DT_cov_eig = eigensolver.eigenvalues();
    curr_frame->err_norm   = err;

}

void StereoFrameHandler::gaussNewtonOptimization(Matrix4d &DT, Matrix6d &DT_cov)
{
    Matrix6d H;
    Vector6d g, DT_inc;
    double err, err_prev = 999999999.9;
    for( int iters = 0; iters < Config::maxIters(); iters++)
    {
        // estimate hessian and gradient (select)
        optimizeFunctions_nonweighted( DT, H, g, err );
        // if the difference is very small stop
        if( ( abs(err-err_prev) < Config::minErrorChange() ) || ( err < Config::minError()) )
            break;
        // update step
        LDLT<Matrix6d> solver(H);
        DT_inc = solver.solve(g);
        DT  << DT * inverse_transformation( transformation_expmap(DT_inc) );
        // if the parameter change is small stop (TODO: change with two parameters, one for R and another one for t)
        if( DT_inc.norm() < numeric_limits<double>::epsilon() )
            break;
        // update previous values
        err_prev = err;
    }
    DT_cov = H.inverse();
}

void StereoFrameHandler::removeOutliers(Matrix4d DT)
{
    VectorXf residuals( n_inliers );

    // point features
    int iter = 0;
    for( list<PointFeature*>::iterator it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
    {
        // projection error
        Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
        Vector2d pl_proj = cam->projection( P_ );
        residuals(iter) = ( pl_proj - (*it)->pl_obs ).norm();
    }

    // line segment features
    for( list<LineFeature*>::iterator it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
    {
        // projection error
        Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);
        Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);
        Vector2d spl_proj = cam->projection( sP_ );
        Vector2d epl_proj = cam->projection( eP_ );
        Vector3d l_obs    = (*it)->le_obs;
        Vector2d err_li;
        err_li(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
        err_li(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
        residuals(iter) = err_li.norm();
    }

    // estimate mad standard deviation
    double inlier_th =  Config::inlierK() * vector_stdv_mad( residuals );

    // filter outliers
    iter      = 0;
    for( list<PointFeature*>::iterator it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
    {
        if( residuals(iter) > inlier_th )
        {
            (*it)->inlier = false;
            n_inliers--;
            n_inliers_pt--;
        }
    }
    for( list<LineFeature*>::iterator it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
    {
        if( residuals(iter) > inlier_th )
        {
            (*it)->inlier = false;
            n_inliers--;
            n_inliers_ls--;
        }
    }
}

void StereoFrameHandler::optimizeFunctions_nonweighted(Matrix4d DT, Matrix6d &H, Vector6d &g, double &err )
{

    // define hessian, gradient, and residuals
    H   = Matrix6d::Zero();
    g   = Vector6d::Zero();
    err = 0.0;

    // point features
    for( list<PointFeature*>::iterator it = matched_pt.begin(); it!=matched_pt.end(); it++)
    {
        if( (*it)->inlier )
        {
            Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
            Vector2d pl_proj = cam->projection( P_ );
            // projection error
            Vector2d err_i    = pl_proj - (*it)->pl_obs;
            double err_i_norm = err_i.norm();
            // check inverse of err_i_norm
            if( err_i_norm > Config::homogTh() )
            {
                double gx   = P_(0);
                double gy   = P_(1);
                double gz   = P_(2);
                double gz2  = gz*gz;
                double fgz2 = cam->getFx() / std::max(0.0000001,gz2);
                double dx   = err_i(0);
                double dy   = err_i(1);
                // jacobian
                Vector6d J_aux;
                J_aux << + fgz2 * dx * gz,
                         + fgz2 * dy * gz,
                         - fgz2 * ( gx*dx + gy*dy ),
                         - fgz2 * ( gx*gy*dx + gy*gy*dy + gz*gz*dy ),
                         + fgz2 * ( gx*gx*dx + gz*gz*dx + gx*gy*dy ),
                         + fgz2 * ( gx*gz*dy - gy*gz*dx );
                J_aux = J_aux / std::max(0.0000001,err_i_norm);
                // if employing robust cost function
                double w = 1.0;
                if( Config::robustCost() )
                    w = 1.0 / ( 1.0 + err_i_norm );
                // update hessian, gradient, and error
                H   += J_aux * J_aux.transpose() * w;
                g   += J_aux * err_i_norm * w;
                err += err_i_norm * err_i_norm * w;
            }
            else
                (*it)->inlier = false;
        }
    }

    // line segment features
    for( list<LineFeature*>::iterator it = matched_ls.begin(); it!=matched_ls.end(); it++)
    {

        if( (*it)->inlier )
        {
            Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);
            Vector2d spl_proj = cam->projection( sP_ );
            Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);
            Vector2d epl_proj = cam->projection( eP_ );
            Vector3d l_obs = (*it)->le_obs;
            // projection error
            Vector2d err_i;
            err_i(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
            err_i(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
            double err_i_norm = err_i.norm();
            // check inverse of err_i_norm
            if( err_i_norm > Config::homogTh() )
            {
                // start point
                double gx   = sP_(0);
                double gy   = sP_(1);
                double gz   = sP_(2);
                double gz2  = gz*gz;
                double fgz2 = cam->getFx() / std::max(0.0000001,gz2);
                double ds   = err_i(0);
                double de   = err_i(1);
                double lx   = l_obs(0);
                double ly   = l_obs(1);
                Vector6d Js_aux;
                Js_aux << + fgz2 * lx * gz,
                          + fgz2 * ly * gz,
                          - fgz2 * ( gx*lx + gy*ly ),
                          - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                          + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                          + fgz2 * ( gx*gz*ly - gy*gz*lx );
                // end point
                gx   = eP_(0);
                gy   = eP_(1);
                gz   = eP_(2);
                gz2  = gz*gz;
                fgz2 = cam->getFx() / std::max(0.0000001,gz2);
                Vector6d Je_aux, J_aux;
                Je_aux << + fgz2 * lx * gz,
                          + fgz2 * ly * gz,
                          - fgz2 * ( gx*lx + gy*ly ),
                          - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                          + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                          + fgz2 * ( gx*gz*ly - gy*gz*lx );
                // jacobian
                J_aux = ( Js_aux * ds + Je_aux * de ) / std::max(0.0000001,err_i_norm);
                // if employing robust cost function
                double w = 1.0;
                if( Config::robustCost() )
                    w = 1.0 / ( 1.0 + err_i_norm );
                // update hessian, gradient, and error
                H   += J_aux * J_aux.transpose() * w;
                g   += J_aux * err_i_norm * w;
                err += err_i_norm * err_i_norm * w;
            }
            else
                (*it)->inlier = false;
        }

    }

    // normalize error
    err /= n_inliers;

}

/*void StereoFrameHandler::optimizeFunctions_uncweighted(Matrix4d DT, Matrix6d &H, Vector6d &g, double &err )
{

    // define hessian, gradient, and residuals
    H   = Matrix6d::Zero();
    g   = Vector6d::Zero();
    err = 0.0;

    // assign cam parameters
    double f     = cam->getFx();   // we assume fx == fy
    double cx    = cam->getCx();
    double cy    = cam->getCy();
    double sigma = Config::sigmaPx();

    // estimate sigma parameters
    double bsigma     = f * cam->getB() * sigma;
    double bsigma_inv = 1.f / bsigma;
    double sigma2     = sigma * sigma;

    // point features
    Matrix3d R  = DT.block(0,0,3,3);
    int n_inliers_ = 0;
    for( list<PointFeature*>::iterator it = matched_pt.begin(); it!=matched_pt.end(); it++)
    {
        if( (*it)->inlier )
        {
            Vector3d P_ = R * (*it)->P + DT.col(3).head(3);
            Vector2d pl_proj = cam->projection( P_ );
            // projection error
            Vector2d err_i    = pl_proj - (*it)->pl_obs;
            double err_i_norm = err_i.norm();
            // check inverse of err_i_norm
            if( err_i_norm > Config::homogTh() )
            {
                n_inliers_++;
                double gx   = P_(0);
                double gy   = P_(1);
                double gz   = P_(2);
                double gz2  = gz*gz;
                double fgz2 = f / std::max(0.0000001,gz2);
                double dx   = err_i(0);
                double dy   = err_i(1);
                // jacobian
                Vector6d J_aux;
                J_aux << + fgz2 * dx * gz,
                         + fgz2 * dy * gz,
                         - fgz2 * ( gx*dx + gy*dy ),
                         - fgz2 * ( gx*gy*dx + gy*gy*dy + gz*gz*dy ),
                         + fgz2 * ( gx*gx*dx + gz*gz*dx + gx*gy*dy ),
                         + fgz2 * ( gx*gz*dy - gy*gz*dx );
                J_aux = J_aux / std::max(0.0000001,err_i_norm);
                // uncertainty
                double px_hat = (*it)->pl(0) - cx;
                double py_hat = (*it)->pl(1) - cy;
                double disp   = (*it)->disp;
                double disp2  = disp * disp;
                Matrix3d covP_an;
                covP_an(0,0) = disp2+2.f*px_hat*px_hat;
                covP_an(0,1) = 2.f*px_hat*py_hat;
                covP_an(0,2) = 2.f*f*px_hat;
                covP_an(1,1) = disp2+2.f*py_hat*py_hat;
                covP_an(1,2) = 2.f*f*py_hat;
                covP_an(2,2) = 2.f*f*f;
                covP_an(1,0) = covP_an(0,1);
                covP_an(2,0) = covP_an(0,2);
                covP_an(2,1) = covP_an(1,2);
                covP_an << covP_an / (disp2*disp2);
                MatrixXd Jhg(2,3), covp(2,2), covp_inv(2,2);
                Jhg  << gz, 0.f, -gx, 0.f, gz, -gy;
                Jhg  << Jhg * R;
                covp << Jhg * covP_an * Jhg.transpose();
                covp << covp / (gz2*gz2);
                covp = bsigma * covp;
                covp(0,0) = covp(0,0) + sigma2;
                covp(1,1) = covp(1,1) + sigma2;
                covp_inv = covp.inverse();
                // update the weights matrix
                double wunc = err_i.transpose() * covp_inv * err_i;
                wunc = wunc / (dx*dx+dy*dy);                
                // if employing robust cost function
                double w = 1.0;
                if( Config::robustCost() )
                    w = 1.0 / ( 1.0 + err_i_norm );
                // update hessian, gradient, and error
                H   += J_aux * J_aux.transpose() * wunc * w / err_i_norm ;
                g   += J_aux * w;
                err += err_i_norm * err_i_norm * wunc * w ;
            }
            else
                (*it)->inlier = false;
        }
    }

    // line segment features
    for( list<LineFeature*>::iterator it = matched_ls.begin(); it!=matched_ls.end(); it++)
    {

        if( (*it)->inlier )
        {
            Vector3d sP_ = DT.block(0,0,3,3) * (*it)->sP + DT.col(3).head(3);
            Vector2d spl_proj = cam->projection( sP_ );
            Vector3d eP_ = DT.block(0,0,3,3) * (*it)->eP + DT.col(3).head(3);
            Vector2d epl_proj = cam->projection( eP_ );
            Vector3d l_obs = (*it)->le_obs;
            // projection error
            Vector2d err_i;
            err_i(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
            err_i(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
            double err_i_norm = err_i.norm();
            // check inverse of err_i_norm
            if( err_i_norm > Config::homogTh() )
            {
                // -- start point
                double gx   = sP_(0);
                double gy   = sP_(1);
                double gz   = sP_(2);
                double gz2  = gz*gz;
                double fgz2 = f / std::max(0.0000001,gz2);
                double ds   = err_i(0);
                double de   = err_i(1);
                double lx   = l_obs(0);
                double ly   = l_obs(1);
                Vector6d Js_aux;
                Js_aux << + fgz2 * lx * gz,
                          + fgz2 * ly * gz,
                          - fgz2 * ( gx*lx + gy*ly ),
                          - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                          + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                          + fgz2 * ( gx*gz*ly - gy*gz*lx );
                // uncertainty
                double px_hat = (*it)->spl(0) - cx;
                double py_hat = (*it)->spl(1) - cy;
                double disp   = (*it)->sdisp;
                double disp2  = disp * disp;
                Matrix3d covP_an;
                covP_an(0,0) = disp2+2.f*px_hat*px_hat;
                covP_an(0,1) = 2.f*px_hat*py_hat;
                covP_an(0,2) = 2.f*f*px_hat;
                covP_an(1,1) = disp2+2.f*py_hat*py_hat;
                covP_an(1,2) = 2.f*f*py_hat;
                covP_an(2,2) = 2.f*f*f;
                covP_an(1,0) = covP_an(0,1);
                covP_an(2,0) = covP_an(0,2);
                covP_an(2,1) = covP_an(1,2);
                covP_an << covP_an / (disp2*disp2);
                Vector3d spl_proj_ = cam->projectionNH( sP_ );
                MatrixXd J_ep(1,3);
                double lxpz = lx * spl_proj_(2);
                double lypz = ly * spl_proj_(2);
                J_ep << lxpz*f, lypz*f, lxpz*cx+lypz*cy-lx*spl_proj_(0)-ly*spl_proj_(1);
                J_ep << J_ep * R;
                double p4 = pow(spl_proj_(2),4);
                double cov_p;
                VectorXd cov_aux(1);
                cov_aux << J_ep * covP_an * J_ep.transpose();
                cov_p = cov_aux(0);
                cov_p = 1.f/cov_p;
                cov_p = p4 * cov_p * 0.5f * bsigma_inv;

                // -- end point
                gx   = eP_(0);
                gy   = eP_(1);
                gz   = eP_(2);
                gz2  = gz*gz;
                fgz2 = cam->getFx() / std::max(0.0000001,gz2);
                Vector6d Je_aux, J_aux;
                Je_aux << + fgz2 * lx * gz,
                          + fgz2 * ly * gz,
                          - fgz2 * ( gx*lx + gy*ly ),
                          - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly ),
                          + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly ),
                          + fgz2 * ( gx*gz*ly - gy*gz*lx );
                // uncertainty
                px_hat = (*it)->epl(0) - cx;
                py_hat = (*it)->epl(1) - cy;
                disp   = (*it)->edisp;
                disp2  = disp * disp;
                Matrix3d covQ_an;
                covQ_an(0,0) = disp2+2.f*px_hat*px_hat;
                covQ_an(0,1) = 2.f*px_hat*py_hat;
                covQ_an(0,2) = 2.f*f*px_hat;
                covQ_an(1,1) = disp2+2.f*py_hat*py_hat;
                covQ_an(1,2) = 2.f*f*py_hat;
                covQ_an(2,2) = 2.f*f*f;
                covQ_an(1,0) = covQ_an(0,1);
                covQ_an(2,0) = covQ_an(0,2);
                covQ_an(2,1) = covQ_an(1,2);
                covQ_an << covQ_an / (disp2*disp2);
                Vector3d epl_proj_ = cam->projectionNH( eP_ );
                lxpz = lx * epl_proj_(2);
                lypz = ly * epl_proj_(2);
                J_ep << lxpz*f, lypz*f, lxpz*cx+lypz*cy-lx*epl_proj_(0)-ly*epl_proj_(1);
                J_ep << J_ep * R;
                p4 = pow(epl_proj_(2),4);
                double cov_q;
                cov_aux << J_ep * covQ_an * J_ep.transpose();
                cov_q = cov_aux(0);
                cov_q = 1.f / cov_q;
                cov_q = p4 * cov_q * 0.5f * bsigma_inv;

                cout << endl << cov_p << " " << cov_q ;
                //cout << endl << p4 << " " << bsigma_inv << "\t" << epl_proj_.transpose() << "\t" << spl_proj_.transpose() ;

                if( !std::isinf(cov_p) && !std::isnan(cov_p) && !std::isinf(cov_q) && !std::isnan(cov_q) )
                {
                    n_inliers_++;
                    // update the weights matrix
                    double wunc = err_i(0) * err_i(0) * cov_p + err_i(1) * err_i(1) * cov_q;
                    wunc = wunc / ( err_i(0)*err_i(0) + err_i(1)*err_i(1) );
                    // jacobian
                    J_aux = ( Js_aux * ds + Je_aux * de ) / std::max(0.0000001,err_i_norm);
                    // if employing robust cost function
                    double w = 1.0;
                    if( Config::robustCost() )
                        w = 1.0 / ( 1.0 + err_i_norm );
                    // update hessian, gradient, and error
                    H   += J_aux * J_aux.transpose() * wunc * w / err_i_norm ;
                    g   += J_aux * w;
                    err += err_i_norm * err_i_norm * wunc * w ;
                }
                else
                    (*it)->inlier = false;
            }
            else
                (*it)->inlier = false;
        }

    }

    //cout << endl;

    // normalize error
    err /= n_inliers_;

}
*/
