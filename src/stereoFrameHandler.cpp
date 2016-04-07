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

double vector_stdv_mad( vector<double> residues)
{
    // Return the standard deviation of vector with MAD estimation
    int n_samples = residues.size();
    sort( residues.begin(),residues.end() );
    double median = residues[ n_samples/2 ];
    for( int i = 0; i < n_samples; i++)
        residues[i] = fabsf( residues[i] - median );
    sort( residues.begin(),residues.end() );
    double MAD = residues[ n_samples/2 ];
    return 1.4826 * MAD;
}

namespace StVO{

StereoFrameHandler::StereoFrameHandler( PinholeStereoCamera *cam_ ) : cam(cam_) {}

StereoFrameHandler::~StereoFrameHandler(){}

#pragma message("TODO: avoid passing the camera as argument")

void StereoFrameHandler::initialize(const Mat &img_l_, const Mat &img_r_ , const int idx_)
{
    prev_frame = new StereoFrame( img_l_, img_r_, idx_, cam, Matrix4d::Identity() );
    prev_frame->extractInitialStereoFeatures();
    prev_frame->Tfw = Matrix4d::Identity();
    max_idx_pt = prev_frame->stereo_pt.size();  max_idx_pt_prev_kf = max_idx_pt;
    max_idx_ls = prev_frame->stereo_ls.size();  max_idx_ls_prev_kf = max_idx_ls;
    prev_keyframe = prev_frame;
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
                curr_frame->stereo_pt[lr_tdx]->idx = prev_frame->stereo_pt[lr_qdx]->idx; // prev idx
            }
            else
            {
                curr_frame->stereo_pt[lr_tdx]->idx = max_idx_pt;
                max_idx_pt++;
            }
        }

        // put index on the rest of the features
        for( int i = 0; i < curr_frame->stereo_pt.size(); i++)
        {
            if( curr_frame->stereo_pt[i]->idx == -1 )
            {
                curr_frame->stereo_pt[i]->idx = max_idx_pt;
                max_idx_pt++;
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
                curr_frame->stereo_ls[lr_tdx]->idx = prev_frame->stereo_ls[lr_qdx]->idx; // prev idx
            }
            else
            {
                curr_frame->stereo_ls[lr_tdx]->idx = max_idx_ls;
                max_idx_ls++;
            }
        }

        // put index on the rest of the features
        for( int i = 0; i < curr_frame->stereo_ls.size(); i++)
        {
            if( curr_frame->stereo_ls[i]->idx == -1 )
            {
                curr_frame->stereo_ls[i]->idx = max_idx_ls;
                max_idx_ls++;
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
    double   err;

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
        if( Config::useLevMarquardt() )
            levMarquardtOptimization(DT_,DT_cov,err);
        else
            gaussNewtonOptimization(DT_,DT_cov,err);
        // remove outliers (implement some logic based on the covariance's eigenvalues and optim error)
        if( is_finite(DT_) )
        {
            removeOutliers(DT_);
            // refine without outliers
            if( n_inliers > Config::minFeatures() )
            {
                if( Config::useLevMarquardt() )
                    levMarquardtOptimization(DT,DT_cov,err);
                else
                    gaussNewtonOptimization(DT,DT_cov,err);
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
    }
    else
    {
        DT     = Matrix4d::Identity();
        DT_cov = Matrix6d::Zero();
    }

    // set estimated pose
    if( is_finite(DT_) && err < 1.0 )
    {
        curr_frame->DT     = inverse_transformation( DT );  //check what's best
        curr_frame->Tfw    = prev_frame->Tfw * curr_frame->DT;
        curr_frame->DT_cov = DT_cov;
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
        curr_frame->err_norm   = err;
    }
    else
    {
        curr_frame->DT     = Matrix4d::Identity();
        curr_frame->Tfw    = prev_frame->Tfw;
        curr_frame->DT_cov = Matrix6d::Zero();
        SelfAdjointEigenSolver<Matrix6d> eigensolver(DT_cov);
        curr_frame->DT_cov_eig = eigensolver.eigenvalues();
        curr_frame->err_norm   = -1.0;
    }

}

void StereoFrameHandler::gaussNewtonOptimization(Matrix4d &DT, Matrix6d &DT_cov, double &err_)
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
    err_   = err;
}

void StereoFrameHandler::levMarquardtOptimization(Matrix4d &DT, Matrix6d &DT_cov, double &err_)
{
    Matrix6d H;
    Vector6d g, DT_inc;
    Matrix4d DT_;
    double err, err_prev = 999999999.9;
    double lambda = Config::lambdaLM(), lambda_k = Config::lambdaK();
    for( int iters = 0; iters < Config::maxIters(); iters++)
    {
        // estimate hessian and gradient (select)
        optimizeFunctions_nonweighted( DT, H, g, err );
        // if the difference is very small stop
        if( ( abs(err-err_prev) < Config::minErrorChange() ) || ( err < Config::minError()) )
            break;
        // update step
        H += lambda * H.diagonal().asDiagonal();
        LDLT<Matrix6d> solver(H);
        DT_inc = solver.solve(g);
        DT_  << DT * inverse_transformation( transformation_expmap(DT_inc) );
        // update lambda
        if( err > err_prev )
            lambda /= lambda_k;
        else
        {
            lambda *= lambda_k;
            DT = DT_;
        }
        // if the parameter change is small stop (TODO: change with two parameters, one for R and another one for t)
        if( DT_inc.norm() < numeric_limits<double>::epsilon() )
            break;
        // update previous values
        err_prev = err;
    }
    DT_cov = H.inverse();
    err_   = err;
}

void StereoFrameHandler::removeOutliers(Matrix4d DT)
{
    //VectorXf residuals( n_inliers );

    vector<double> res_p, res_l;

    // point features
    int iter = 0;
    for( list<PointFeature*>::iterator it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
    {
        // projection error
        Vector3d P_ = DT.block(0,0,3,3) * (*it)->P + DT.col(3).head(3);
        Vector2d pl_proj = cam->projection( P_ );
        //residuals(iter) = ( pl_proj - (*it)->pl_obs ).norm();
        res_p.push_back( ( pl_proj - (*it)->pl_obs ).norm() );
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
        //residuals(iter) = err_li.norm();
        res_l.push_back( err_li.norm() );
    }

    // estimate mad standard deviation
    double inlier_th_p =  Config::inlierK() * vector_stdv_mad( res_p );
    double inlier_th_l =  Config::inlierK() * vector_stdv_mad( res_l );

    // filter outliers
    iter = 0;
    for( list<PointFeature*>::iterator it = matched_pt.begin(); it!=matched_pt.end(); it++, iter++)
    {
        if( res_p[iter] > inlier_th_p )
        {
            (*it)->inlier = false;
            n_inliers--;
            n_inliers_pt--;
        }
    }
    iter = 0;
    for( list<LineFeature*>::iterator it = matched_ls.begin(); it!=matched_ls.end(); it++, iter++)
    {
        if( res_l[iter] > inlier_th_l )
        {
            (*it)->inlier = false;
            n_inliers--;
            n_inliers_ls--;
        }
    }
}

void StereoFrameHandler::optimizeFunctions_nonweighted(Matrix4d DT, Matrix6d &H, Vector6d &g, double &e )
{

    // define hessians, gradients, and residuals
    Matrix6d H_l, H_p;
    Vector6d g_l, g_p;
    double   e_l = 0.0, e_p = 0.0, S_l, S_p;
    H   = Matrix6d::Zero(); H_l = H; H_p = H;
    g   = Vector6d::Zero(); g_l = g; g_p = g;
    e   = 0.0;

    // point features
    int N_p = 0;
    vector<double> r_p;
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
                H_p += J_aux * J_aux.transpose() * w;
                g_p += J_aux * err_i_norm * w;
                e_p += err_i_norm * err_i_norm * w;
                N_p++;
                if( Config::scalePointsLines() )
                    r_p.push_back( err_i_norm * err_i_norm * w );
            }
            else
                (*it)->inlier = false;
        }
    }
    if( Config::scalePointsLines() )
        S_p = vector_stdv_mad(r_p);

    // line segment features
    int N_l = 0;
    vector<double> r_l;
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
                H_l += J_aux * J_aux.transpose() * w;
                g_l += J_aux * err_i_norm * w;
                e_l += err_i_norm * err_i_norm * w;
                N_l++;
                if( Config::scalePointsLines() )
                    r_l.push_back( err_i_norm * err_i_norm * w );
            }
            else
                (*it)->inlier = false;
        }

    }
    if( Config::scalePointsLines() )
        S_l = vector_stdv_mad(r_l);

    // sum H, g and err from both points and lines
    if( Config::scalePointsLines() && S_l > Config::homogTh() && S_p > Config::homogTh() &&
        Config::hasPoints() && Config::hasLines() )
    {
        double S_l_inv = 1.0 / S_l;
        double S_p_inv = 1.0 / S_p;
        double S_l_ = (S_p_inv+S_l_inv) / S_p_inv;
        double S_p_ = (S_p_inv+S_l_inv) / S_l_inv;
        H = H_p * S_p_ + H_l * S_l_;
        g = g_p * S_p_ + g_l * S_l_;
        e = e_p * S_p_ + e_l * S_l_;
    }
    else
    {
        H = H_p + H_l;
        g = g_p + g_l;
        e = e_p + e_l;
    }

    // normalize error
    e /= (N_l+N_p);

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

/*  slam functions  */

void StereoFrameHandler::currFrameIsKF()
{
    // update KF
    prev_keyframe = curr_frame;

    // restart point indices
    int idx_pt = 0;
    for( vector<PointFeature*>::iterator it = prev_keyframe->stereo_pt.begin(); it!=prev_keyframe->stereo_pt.end(); it++)
    {
        (*it)->idx = idx_pt;
        idx_pt++;
    }
    max_idx_pt = prev_keyframe->stereo_pt.size();
    max_idx_pt_prev_kf = max_idx_pt;

    // restart line indices
    int idx_ls = 0;
    for( vector<LineFeature*>::iterator it = prev_keyframe->stereo_ls.begin(); it!=prev_keyframe->stereo_ls.end(); it++)
    {
        (*it)->idx = idx_ls;
        idx_ls++;
    }
    max_idx_ls = prev_keyframe->stereo_ls.size();
    max_idx_ls_prev_kf = max_idx_ls;

}

void StereoFrameHandler::checkKFCommonCorrespondences(double p_th, double l_th)
{

    // estimates pose increment
    Matrix4d DT = inverse_transformation( curr_frame->Tfw ) * prev_keyframe->Tfw;

    // points f2f tracking
    matched_pt.clear();
    int common_pt = 0;
    if( Config::hasPoints() && !(curr_frame->stereo_pt.size()==0) && !(prev_keyframe->stereo_pt.size()==0)  )
    {
        BFMatcher* bfm = new BFMatcher( NORM_HAMMING, false );    // cross-check
        Mat pdesc_l1, pdesc_l2;
        vector<vector<DMatch>> pmatches_12, pmatches_21;
        // 12 and 21 matches
        pdesc_l1 = prev_keyframe->pdesc_l;
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
                // check epipolar constraint
                Vector3d P_ = DT.block(0,0,3,3) * prev_keyframe->stereo_pt[lr_qdx]->P + DT.col(3).head(3);
                Vector2d pl_proj = cam->projection( P_ );
                double   error = ( pl_proj - curr_frame->stereo_pt[lr_tdx]->pl ).norm();
                if( error < p_th )
                    common_pt++;
            }
        }

    }

    // line segments f2f tracking
    matched_ls.clear();
    int common_ls = 0;
    if( Config::hasLines() && !(curr_frame->stereo_ls.size()==0) && !(prev_keyframe->stereo_ls.size()==0)  )
    {
        Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
        Mat ldesc_l1, ldesc_l2;
        vector<vector<DMatch>> lmatches_12, lmatches_21;
        // 12 and 21 matches
        ldesc_l1 = prev_keyframe->ldesc_l;
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
        // bucle around lmatches
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
                Vector3d sP_ = DT.block(0,0,3,3) * prev_keyframe->stereo_ls[lr_qdx]->sP + DT.col(3).head(3);
                Vector2d spl_proj = cam->projection( sP_ );
                Vector3d eP_ = DT.block(0,0,3,3) * prev_keyframe->stereo_ls[lr_qdx]->eP + DT.col(3).head(3);
                Vector2d epl_proj = cam->projection( eP_ );
                Vector3d l_obs = prev_keyframe->stereo_ls[lr_qdx]->le;
                // projection error
                Vector2d err_ls;
                err_ls(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
                err_ls(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
                if( err_ls.norm() < l_th )
                {
                    common_ls++;
                }
            }
        }

    }

    cout << endl << "Common points: " << common_pt << " \t Common lines: " << common_ls << endl << endl;

}

cv::Mat StereoFrameHandler::checkKFCommonCorrespondencesPlot(double p_th, double l_th)
{

    // representation variables
    int lowest = 100, highest = 255;
    int range  = (highest-lowest) + 1 ;
    unsigned int r, g, b; //the color of lines
    int radius  = 3;
    float thick = 1.5f;
    cv::Mat img_1, img_2;

    // convert images to color
    cvtColor(prev_keyframe->img_l,img_1,CV_GRAY2BGR);
    cvtColor(curr_frame->img_l,img_2,CV_GRAY2BGR);

    // estimates pose increment
    Matrix4d DT = inverse_transformation( curr_frame->Tfw ) * prev_keyframe->Tfw;

    // points f2f tracking
    matched_pt.clear();
    int common_pt = 0;
    if( Config::hasPoints() && !(curr_frame->stereo_pt.size()==0) && !(prev_keyframe->stereo_pt.size()==0)  )
    {
        BFMatcher* bfm = new BFMatcher( NORM_HAMMING, false );    // cross-check
        Mat pdesc_l1, pdesc_l2;
        vector<vector<DMatch>> pmatches_12, pmatches_21;
        // 12 and 21 matches
        pdesc_l1 = prev_keyframe->pdesc_l;
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
                // check epipolar constraint
                Vector3d P_ = DT.block(0,0,3,3) * prev_keyframe->stereo_pt[lr_qdx]->P + DT.col(3).head(3);
                Vector2d pl_proj = cam->projection( P_ );
                double   error = ( pl_proj - curr_frame->stereo_pt[lr_tdx]->pl ).norm();
                if( error < p_th )
                {
                    common_pt++;
                    r = lowest+int(rand()%range);
                    g = lowest+int(rand()%range);
                    b = lowest+int(rand()%range);
                    // first image
                    cv::Point2f P(prev_keyframe->stereo_pt[lr_qdx]->pl(0),prev_keyframe->stereo_pt[lr_qdx]->pl(1));
                    cv::circle(img_1, P, radius, Scalar(b,g,r), thick);
                    // second image
                    P = cv::Point2f(curr_frame->stereo_pt[lr_tdx]->pl(0),curr_frame->stereo_pt[lr_tdx]->pl(1));
                    cv::circle(img_2, P, radius, Scalar(b,g,r), thick);
                }
            }
        }
    }

    // line segments f2f tracking
    matched_ls.clear();
    int common_ls = 0;
    if( Config::hasLines() && !(curr_frame->stereo_ls.size()==0) && !(prev_keyframe->stereo_ls.size()==0)  )
    {
        Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
        Mat ldesc_l1, ldesc_l2;
        vector<vector<DMatch>> lmatches_12, lmatches_21;
        // 12 and 21 matches
        ldesc_l1 = prev_keyframe->ldesc_l;
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
        // bucle around lmatches
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
                Vector3d sP_ = DT.block(0,0,3,3) * prev_keyframe->stereo_ls[lr_qdx]->sP + DT.col(3).head(3);
                Vector2d spl_proj = cam->projection( sP_ );
                Vector3d eP_ = DT.block(0,0,3,3) * prev_keyframe->stereo_ls[lr_qdx]->eP + DT.col(3).head(3);
                Vector2d epl_proj = cam->projection( eP_ );
                Vector3d l_obs = prev_keyframe->stereo_ls[lr_qdx]->le;
                // projection error
                Vector2d err_ls;
                err_ls(0) = l_obs(0) * spl_proj(0) + l_obs(1) * spl_proj(1) + l_obs(2);
                err_ls(1) = l_obs(0) * epl_proj(0) + l_obs(1) * epl_proj(1) + l_obs(2);
                if( err_ls.norm() < l_th )
                {
                    common_ls++;
                    r = lowest+int(rand()%range);
                    g = lowest+int(rand()%range);
                    b = lowest+int(rand()%range);
                    // first image
                    cv::Point2f P(prev_keyframe->stereo_ls[lr_qdx]->spl(0),prev_keyframe->stereo_ls[lr_qdx]->spl(1));
                    cv::Point2f Q(prev_keyframe->stereo_ls[lr_qdx]->epl(0),prev_keyframe->stereo_ls[lr_qdx]->epl(1));
                    cv::line( img_1, P, Q, cv::Scalar(b,g,r), thick );
                    // second image
                    P = cv::Point2f(curr_frame->stereo_ls[lr_tdx]->spl(0),curr_frame->stereo_ls[lr_tdx]->spl(1));
                    Q = cv::Point2f(curr_frame->stereo_ls[lr_tdx]->epl(0),curr_frame->stereo_ls[lr_tdx]->epl(1));
                    cv::line( img_2, P, Q, cv::Scalar(b,g,r), thick );
                }
            }
        }

    }

    // plot the two left images together
    int rows = img_1.rows;
    int cols = img_1.cols;
    cv::Mat img_12(rows*2,cols,img_1.type());

    img_1.copyTo( img_12( cv::Rect(0,rows,cols,rows) ) );
    img_2.copyTo( img_12( cv::Rect(0,0,cols,rows)    ) );

    cout << endl << "Common points: " << common_pt << " \t Common lines: " << common_ls << endl << endl;

    return img_12;

}

}
