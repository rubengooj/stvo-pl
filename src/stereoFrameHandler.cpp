#include <stereoFrameHandler.h>

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
        BFMatcher* bfm = new BFMatcher( NORM_L2, false );    // cross-check
        Mat pdesc_l1, pdesc_l2;
        vector<vector<DMatch>> pmatches_12, pmatches_21;
        // 12 and 21 matches
        pdesc_l1 = prev_frame->pdesc_l;
        pdesc_l2 = curr_frame->pdesc_l;
        bfm->knnMatch( pdesc_l1, pdesc_l2, pmatches_12, 2);
        bfm->knnMatch( pdesc_l2, pdesc_l1, pmatches_21, 2);

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
        sort( pmatches_21.begin(), pmatches_21.end(), sort_descriptor_by_queryIdx() );
        // bucle around pmatches
        for( int i = 0; i < pmatches_12.size(); i++ )
        {
            // check if they are mutual best matches
            int lr_qdx = pmatches_12[i][0].queryIdx;
            int lr_tdx = pmatches_12[i][0].trainIdx;
            //int rl_qdx = pmatches_rl[lr_tdx][0].queryIdx;
            int rl_tdx = pmatches_21[lr_tdx][0].trainIdx;
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
        bdm->knnMatch( ldesc_l1, ldesc_l2, lmatches_12, 2);
        bdm->knnMatch( ldesc_l2, ldesc_l1, lmatches_21, 2);

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
        sort( lmatches_21.begin(), lmatches_21.end(), sort_descriptor_by_queryIdx() );
        // bucle around pmatches
        for( int i = 0; i < lmatches_12.size(); i++ )
        {
            // check if they are mutual best matches
            int lr_qdx = lmatches_12[i][0].queryIdx;
            int lr_tdx = lmatches_12[i][0].trainIdx;
            //int rl_qdx = pmatches_rl[lr_tdx][0].queryIdx;
            int rl_tdx = lmatches_21[lr_tdx][0].trainIdx;
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

void StereoFrameHandler::updateFrame()
{
    matched_pt.clear();
    matched_ls.clear();
    prev_frame = curr_frame;
    curr_frame = NULL;
}

void StereoFrameHandler::optimizePose()
{
    // TODO: remove...
    if(true)
        optimizeGN();
    else
        optimizeLM();
}

void StereoFrameHandler::optimizeGN()
{

    // definitions
    Matrix6d H, DT_cov;
    Vector6d g, DT_inc;
    Matrix4d DT;
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

        // remove outliers (implement some logic based on the covariance's eigenvalues)
        if( is_finite(DT) )
            removeOutliers( DT );

        // refine without outliers
        if( n_inliers > Config::minFeatures() )
        {
            for( int iters = 0; iters < Config::maxItersRef(); iters++)
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

void StereoFrameHandler::optimizeLM()
{

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
