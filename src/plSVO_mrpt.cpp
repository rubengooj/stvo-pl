/*****************************************************************************
**   Stereo Visual Odometry by combining point and line segment features	**
******************************************************************************
**																			**
**	Copyright(c) 2015, Ruben Gomez-Ojeda, University of Malaga              **
**	Copyright(c) 2015, MAPIR group, University of Malaga					**
**																			**
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.								**
**																			**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of				**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			**
**  GNU General Public License for more details.							**
**																			**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**																			**
*****************************************************************************/

#include <plSVO_mrpt.h>

unsigned int nFrame = 0;

float delta = 0.00000000000001f;

/* Auxiliar functions */

float vector_stdv_MAD(VectorXf residues){

    int n_samples = residues.size();
    VectorXf     residue_sort(n_samples), residue_aux(n_samples);
    residue_sort = residues;
    residue_aux  = residues;
    sort(residue_sort.derived().data(),residue_sort.derived().data()+residue_sort.size());
    float median = residue_sort( (int)(0.5f*n_samples) );
    residue_aux << (residue_aux - VectorXf::Constant(n_samples,median)).cwiseAbs();
    sort(residue_aux.derived().data(),residue_aux.derived().data()+residue_aux.size());
    float MAD = residue_aux( (int)(0.5f*n_samples) );
    return 1.4826f * MAD;

}

/* Constructors and destructor */

plSVO::plSVO(){

    // Point Tracking Parameters
    minDist12       = 0.1f;             // min descriptor distance (threshold)
    maxStDist       = 1.f;              // max error in the y coordinate for the stereo correspondences
    maxF2FDisp      = 0.06f;            // max disparity (in proportion to the resolution) between the L-L and R-R correspondences in the F2F tracking
    nFeatures       = 1500;             // max number of detected features
    nFeaturesB      = 90;               // max number of detected features
    nLevels         = 1;                // number of levels
    scaleFactor     = 1.2f;             // pyramid decimation ratio
    dispMin         = 1.f;              // min disparity value (avoid points in the infinite)

    // Line Tracking Parameters
    scale           = 1.2f;             // scale for the LSD detector
    nOctaves        = 1;                // number of octaves where the lines are detected
    dispMax         = 1000.f;           // Max. disparity to filter outliers in the stereo matching (pixels)
    alphaMax        = 10.f;             // Max. angle to filter outliers in the stereo matching (degrees)
    horizTh         = 0.1f;             // Parameter to remove the horizontal lines (pixels)
    flowTh          = 1000.f;           // Flow Optical Vector threshold for the frame2frame matching (pixels)
    alphaTh         = 10.f;             // Angle threshold for the frame2frame matching (degrees)
    matchPr         = 0.4f;             // Probability of inlier

    // Uncertainty Parameters
    sigmaL          = 1.f;              // Standard Deviation
    sigmaP          = 1.f;              // Standard Deviation

    // Optimization Parameters
    maxIters        = 4;                // Max number of iterations
    minError        = 0.0001f;          // Min error to leave the iterative optimization
    minErrorChange  = 0.0001f;          // Min error change to leave the iterative optimization
    prevErrNorm     = 99999999.9f;      // Previous error norm (initialized to high value)
    homogTh         = 0.05f;            // Parameter to control the bad-conditioned divisions when transforming a point to homogeneous coordinates

    // Robust Optimization Parameters
    maxItersH       = 4;                // Max number of robust iterations
    minErrorH       = 0.0001f;          // Min error to leave the iterative robust optimization
    minErrorChangeH = 0.0001f;          // Min error change to leave the iterative robust optimization
    huberTh         = 0.5f;             // Parameter to obtain the Huber threshold and filter the outliers
    huberK          = 2.f;              // Parameter to obtain the Huber threshold and filter the outliers
    b               = 1.5f;             // Pseudo-Huber Loss Function parameter
    b2              = b * b;

    // Flags
    parallel        = false;            // detect the features in two separated threads for each image
    bucketing       = false;            // detect the features in buckets from the image (16 by now)
    points          = true;             // introduce points in the system
    lines           = true;             // introduce line segments in the system
    uncertainty     = true;             // include uncertainty in the optimization
    overlap         = false;            // include the overlap measurement for the line segments (similarly to those employed in Witt's ICML)
    erode           = true;             // include the ERODE outlier rejection
    gamma           = true;             // if true gamma weights, else t-student weights
    motionprior     = true;             // if true we employ the motion prior as extra terms in the optimization
    optimMode       = 0;                // Set the optimization mode

    // Convert to radians
    alphaMax    = alphaMax * CV_PI / 180.f;
    alphaTh     = alphaTh  * CV_PI / 180.f;

    // Creation of the objects
    bfm = BFMatcher(NORM_HAMMING,false);                // Brute Force Matching declaration

    // Size of the point and line structures
    Np   = 16;
    Nl   = 24;
    Nmin = 8;

}

plSVO::plSVO(string configFile){

    // Config File object
    CConfigFile config(configFile);

    // Point Tracking Parameters
    minDist12       = config.read_float("PointTracking","minDist12",0.1f);
    maxStDist       = config.read_float("PointTracking","maxStDist",1.f);
    maxF2FDisp      = config.read_float("PointTracking","maxF2FDisp",0.06f);
    nFeatures       = config.read_float("PointTracking","nFeatures",1500);
    nFeaturesB      = config.read_float("PointTracking","nFeaturesB",90);
    nLevels         = config.read_float("PointTracking","nLevels",1);
    scaleFactor     = config.read_float("PointTracking","scaleFactor",1.2f);
    dispMin         = config.read_float("PointTracking","dispMin",1.f);

    // Line Tracking Parameters
    scale           = config.read_float("LineTracking","scale",1.2f);
    nOctaves        = config.read_uint64_t("LineTracking","nOctaves",1);
    dispMax         = config.read_float("LineTracking","dispMax",1000.f);
    alphaMax        = config.read_float("LineTracking","alphaMax",10.f);
    horizTh         = config.read_float("LineTracking","horizTh",0.1f);
    flowTh          = config.read_float("LineTracking","flowTh",1000.f);
    alphaTh         = config.read_float("LineTracking","alphaTh",10.f);
    matchPr         = config.read_float("LineTracking","matchPr",0.4f);

    // Flags
    parallel        = config.read_bool("SVO","parallel",true);
    bucketing       = config.read_bool("SVO","bucketing",true);
    points          = config.read_bool("SVO","points",true);
    lines           = config.read_bool("SVO","lines",true);
    uncertainty     = config.read_bool("SVO","uncertainty",false);
    overlap         = config.read_bool("SVO","overlap",false);
    erode           = config.read_bool("SVO","erode",true);
    gamma           = config.read_bool("SVO","gamma",true);
    motionprior     = config.read_bool("SVO","motionprior",true);
    optimMode       = config.read_uint64_t("SVO","optimMode",0);

    // Uncertainty Parameters
    sigmaP          = config.read_float("PoseOptimization","sigmaP",1.f);
    sigmaL          = config.read_float("PoseOptimization","sigmaL",1.f);

    // Optimization Parameters
    maxIters        = config.read_uint64_t("PoseOptimization","maxIters",4);
    minError        = config.read_float("PoseOptimization","minError",0.0001f);
    minErrorChange  = config.read_float("PoseOptimization","minErrorChange",0.0001f);
    prevErrNorm     = config.read_float("PoseOptimization","prevErrNorm",999999999.9f);
    homogTh         = config.read_float("PoseOptimization","homogTh",0.05f);

    // Robust Optimization Parameters
    maxItersH       = config.read_uint64_t("ERODE","maxItersH",4);
    minErrorH       = config.read_float("ERODE","minErrorH",0.0001f);
    minErrorChangeH = config.read_float("ERODE","minErrorChangeH",0.0001f);
    huberK          = config.read_float("ERODE","huberK",0.5f);
    huberTh         = config.read_float("ERODE","huberTh",0.5f);
    huberKL         = config.read_float("ERODE","huberKL",1.5f);
    huberKP         = config.read_float("ERODE","huberKP",2.5f);
    b               = config.read_float("ERODE","b",1.5f);
    b2              = b * b;

    // Convert to radians
    alphaMax    = alphaMax * CV_PI / 180.f;
    alphaTh     = alphaTh  * CV_PI / 180.f;

    // Creation of the objects
    bfm = BFMatcher(NORM_HAMMING,false);                // Brute Force Matching declaration

    // Size of the point and line structures
    Np   = 16;
    Nl   = 24;
    Nmin = 8;

    prevCov = MatrixXf::Zero(6,6);
    prevPoseInc = VectorXf::Zero(6);

}

plSVO::~plSVO(){

}

/* Stereo Visual Odometry */

Matrix4f plSVO::SVO(Mat imgL, Mat imgR, Matrix4f x_ini){
    // Stereo Visual Odometry
    Matrix4f x_svo;
    readImages(imgL,imgR);
    detectLR();
    stereoMutualMatching();
    f2fMutualTracking();
    setInitialValue(x_ini.inverse());
    svoOptim(x_svo);
    if(!is_finite(x_svo))
        return Matrix4f::Identity();
    else
        return x_svo;
}

/* Setters and Getters - JOURNAL16 */

void plSVO::getFeaturesL(vector<KeyPoint> &kp, Mat &dp, vector<KeyLine> &kl, Mat &dl){
    kp = pointsThird;
    dp = pdescThird;
    kl = linesThird;
    dl = ldescThird;
}

void plSVO::getFeaturesR(vector<KeyPoint> &kp, Mat &dp, vector<KeyLine> &kl, Mat &dl){
    kp = pointsFourth;
    dp = pdescFourth;
    kl = linesFourth;
    dl = ldescFourth;
}

void plSVO::getErrorNorm(float &errNorm_){
    errNorm_    = errNorm;
}

void plSVO::getItersNErrors(unsigned int &iters_, float &errNorm_){
    iters_      = iters;
    errNorm_    = errNorm;
}

void plSVO::getMatches(unsigned int &pMatches, unsigned int &pMatchesH, unsigned int &lMatches, unsigned int &lMatchesH){
    pMatches  = nPoints;
    pMatchesH = nPointsH;
    lMatches  = nLines;
    lMatchesH = nLinesH;
}

void plSVO::getCovariance(MatrixXf &cov_){
    cov_ = cov;
}

vector<bool> plSVO::getInliers(){
    return inliers;
}

MatrixXf plSVO::getLOptimData(){
    return lData;
}

MatrixXf plSVO::getPOptimData(){
    return pData;
}

void plSVO::setPOptimData(MatrixXf pData_){
    pData      = pData_;
    pData_full = pData_;
    nPoints  = pData.cols();
    nPointsH = nPoints;
    nLines   = 0;
    nLinesH  = 0;
    n = nPoints;
}

void plSVO::setInitialValue(Matrix4f X){
    x_0 = X;
}

void plSVO::setInitialValueNPrior(Matrix4f x_0_, MatrixXf prevPoseInc_, MatrixXf prevCov_){
    x_0 = x_0_;
    prevCov     = prevCov_;
    prevPoseInc = logarithm_map(prevPoseInc_);
}

void plSVO::setFeaturesL(vector<KeyPoint> kp, Mat dp, vector<KeyLine> kl, Mat dl){
    pointsThird = kp;
    pdescThird = dp;
    linesThird = kl;
    ldescThird = dl;
}

void plSVO::setFeaturesR(vector<KeyPoint> kp, Mat dp, vector<KeyLine> kl, Mat dl){
    pointsFourth = kp;
    pdescFourth = dp;
    linesFourth = kl;
    ldescFourth = dl;
}

void plSVO::setUncertainty(string uncertainty_){
    if(uncertainty_ == "true")
        uncertainty = true;
    else if(uncertainty_ == "false")
        uncertainty = false;
}

void plSVO::setNLevels(string nLevels_){
    nLevels = stoi(nLevels_);
}

void plSVO::setBucketing(bool bucketing_){
    bucketing = bucketing_;
}

void plSVO::setMinDist12(float minDist12_){
    minDist12 = minDist12_;
}

void plSVO::setMaxStDist(float maxStDist_){
    maxStDist = maxStDist_;
}

void plSVO::setMaxF2FDisp(float maxF2FDisp_){
    maxF2FDisp = maxF2FDisp_;
}

void plSVO::setDispMin(float dispMin_){
    dispMin = dispMin_;
}

void plSVO::setScale(float scale_){
    scale = scale_;
}

void plSVO::setAlphaMax(float alphaMax_){
    alphaMax = alphaMax_;
}

void plSVO::setAlphaTh(float alphaTh_){
    alphaTh = alphaTh_;
}

void plSVO::setHorizTh(float horizTh_){
    horizTh = horizTh_;
}

void plSVO::setMatchPr(float matchPr_){
    matchPr = matchPr_;
}

void plSVO::setWeights(bool gamma_){
    gamma = gamma_;
}

void plSVO::setOptimMode(unsigned int optimMode_){
    optimMode = optimMode_;
    pData = pData_full;
    lData = lData_full;
    nPointsH = nPoints;
    nLinesH  = nLines;
    n = nPoints + nLines;
}

/* Tracking functions */

void plSVO::setStereo(Matrix3f K_, float baseline_){
    /* Initialize the parameters of the stereo camera, assumed to be in an ideal configuration */
    baseline  = baseline_;  // Horizontal baseline
    f         = K_(0,0);    // Focal length (fx = fy)
    cx        = K_(0,2);    // Principal point x
    cy        = K_(1,2);    // Principal point y
    K         = K_;         // Calibration matrix K
    // Initialize the ampliated calibration matrix K_amp
    K_amp      = MatrixXf::Zero(3,4);
    K_amp(0,0) = f;
    K_amp(1,1) = f;
    K_amp(0,2) = cx;
    K_amp(1,2) = cy;
    K_amp(2,2) = 1.f;
    // Point Uncertainty constants
    bsigmaP   = f * baseline * sigmaP;
    bsigmaP   = bsigmaP * bsigmaP;
    bsigmaP_inv   = 1.f / bsigmaP;
    sigmaP2       = sigmaP * sigmaP;
    sigmaP2_inv   = 1.f / sigmaP2;
    // Line Uncertainty constants
    bsigmaL   = baseline * sigmaL;
    bsigmaL   = bsigmaL * bsigmaL;
    bsigmaL_inv   = 1.f / bsigmaL;
}

void plSVO::initialize(Mat imgL, Mat imgR){
    /* Initialize the points and line segments tracking system */
    imgL.copyTo(imgLeft);                   // Copy the input images
    imgR.copyTo(imgRight);
    maxF2FDist = maxF2FDisp * imgL.cols;    // Max disparity (in proportion to the resolution) between the L-L and R-R correspondences in the F2F tracking
    // Initialize variables for bucketing
    c = imgLeft.cols / 4;
    C = 2 * c;
    r = imgLeft.rows / 4;
    R = 2 * r;
    // Detect the features in the first frame and save them in the structures
    detectLR();
    stereoMutualMatching();
    updateTracking();
}

void plSVO::setInitialize(Mat imgL, Mat imgR, vector<KeyPoint> kpL, Mat dpL, vector<KeyLine> klL, Mat dlL, vector<KeyPoint> kpR, Mat dpR, vector<KeyLine> klR, Mat dlR){
    /* Initialize the points and line segments tracking system */
    imgL.copyTo(imgLeft);                   // Copy the input images
    imgR.copyTo(imgRight);
    maxF2FDist = maxF2FDisp * imgL.cols;    // Max disparity (in proportion to the resolution) between the L-L and R-R correspondences in the F2F tracking
    // Initialize variables for bucketing
    c = imgLeft.cols / 4;
    C = 2 * c;
    r = imgLeft.rows / 4;
    R = 2 * r;
    // Detect the features in the first frame and save them in the structures
    //detectLR();
    setFeaturesL(kpL,dpL,klL,dlL);
    setFeaturesR(kpR,dpR,klR,dlR);
    stereoMutualMatching();
    updateTracking();
}

void plSVO::getInitialize(Mat imgL, Mat imgR, vector<KeyPoint> &kpL, Mat &dpL, vector<KeyLine> &klL, Mat &dlL, vector<KeyPoint> &kpR, Mat &dpR, vector<KeyLine> &klR, Mat &dlR){
    /* Initialize the points and line segments tracking system */
    imgL.copyTo(imgLeft);                   // Copy the input images
    imgR.copyTo(imgRight);
    maxF2FDist = maxF2FDisp * imgL.cols;    // Max disparity (in proportion to the resolution) between the L-L and R-R correspondences in the F2F tracking
    // Initialize variables for bucketing
    c = imgLeft.cols / 4;
    C = 2 * c;
    r = imgLeft.rows / 4;
    R = 2 * r;
    // Detect the features in the first frame and save them in the structures
    detectLR();
    getFeaturesL(kpL,dpL,klL,dlL);
    getFeaturesR(kpR,dpR,klR,dlR);
    stereoMutualMatching();
    updateTracking();
}

void plSVO::readImages(Mat imgL, Mat imgR){
    /* Introduce the input images */
    imgL.copyTo(imgLeft);
    imgR.copyTo(imgRight);
}

void plSVO::updateTracking(){
    /* Update the tracking */

    // Line features
    linesFirst.clear(); linesSecond.clear();
    linesFirst  = linesThird;
    linesSecond = linesFourth;
    linesThird.clear(); linesFourth.clear();
    l1aux.clear();
    l2aux.clear();

    // Line descriptors
    ldescThird.copyTo(ldescFirst);
    ldescFourth.copyTo(ldescSecond);
    ldescThird.release();
    ldescFourth.release();    
    ld1aux.release();
    ld2aux.release();

    // Point features
    pointsFirst.clear(); pointsSecond.clear();
    pointsFirst  = pointsThird;
    pointsSecond = pointsFourth;
    pointsThird.clear(); pointsFourth.clear();
    p1aux.clear();
    p2aux.clear();

    // Point descriptors
    pdescThird.copyTo(pdescFirst);
    pdescFourth.copyTo(pdescSecond);
    pdescThird.release();
    pdescFourth.release();
    pd1aux.release();
    pd2aux.release();

    // Update the images
    imgLeft.copyTo(imgFirst);
    imgRight.copyTo(imgSecond);

}

void plSVO::discardTracking(){
    /* Discard the current frame */

    // Line features
    linesThird.clear();
    linesFourth.clear();
    l1aux.clear();
    l2aux.clear();

    // Line descriptors
    ldescThird.release();
    ldescFourth.release();
    ld1aux.release();
    ld2aux.release();

    // Point features
    pointsThird.clear();
    pointsFourth.clear();
    p1aux.clear();
    p2aux.clear();

    // Point descriptors
    pdescThird.release();
    pdescFourth.release();
    pd1aux.release();
    pd2aux.release();

}

void plSVO::detectLR(){

    /* Detect and descript the features (if parallel we detect the features in separated threads for each image)*/
    if(parallel){
        thread first(&plSVO::detectL, this);
        thread second(&plSVO::detectR, this);
        first.join();
        second.join();
    }
    else{
        detectL();
        detectR();
    }

}

void plSVO::detect1L(){

    Mat img = imgLeft( Rect(0,0,C,R) );

    if(points){
        Ptr<ORB>         orb = ORB::create(nFeaturesB,scaleFactor,nLevels);
        vector<KeyPoint> p1, p2, p3, p4;
        Mat              d1, d2, d3, d4;
        orb->detectAndCompute(img(Rect(0,0,c,r)), cv::Mat(),p1, d1,false);
        orb->detectAndCompute(img(Rect(c,0,c,r)), cv::Mat(),p2, d2,false);
        orb->detectAndCompute(img(Rect(0,r,c,r)), cv::Mat(),p3, d3,false);
        orb->detectAndCompute(img(Rect(c,r,c,r)), cv::Mat(),p4, d4,false);
        points1L.clear();
        pdesc1L.release();
        for(unsigned int i = 0; i < p1.size(); i++){
            p1[i].pt.x += 0;
            p1[i].pt.y += 0;
            points1L.push_back(p1[i]);
            pdesc1L.push_back(d1.row(i));
        }
        for(unsigned int i = 0; i < p2.size(); i++){
            p2[i].pt.x += c;
            p2[i].pt.y += 0;
            points1L.push_back(p2[i]);
            pdesc1L.push_back(d2.row(i));
        }
        for(unsigned int i = 0; i < p3.size(); i++){
            p3[i].pt.x += 0;
            p3[i].pt.y += r;
            points1L.push_back(p3[i]);
            pdesc1L.push_back(d3.row(i));
        }
        for(unsigned int i = 0; i < p4.size(); i++){
            p4[i].pt.x += c;
            p4[i].pt.y += r;
            points1L.push_back(p4[i]);
            pdesc1L.push_back(d4.row(i));
        }
    }

    if(false){
        Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
        Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
        vector<KeyLine>         l1, l2, l3, l4;
        Mat                     d1l, d2l, d3l, d4l;
        lines1L.clear();
        ldesc1L.release();
        lsd->detect( img(Rect(0,0,c,r)),l1,scale,nOctaves );
        lbd->compute(img(Rect(0,0,c,r)),l1,d1l);
        lsd->detect( img(Rect(c,0,c,r)),l2,scale,nOctaves );
        lbd->compute(img(Rect(c,0,c,r)),l2,d2l);
        lsd->detect( img(Rect(0,r,c,r)),l3,scale,nOctaves );
        lbd->compute(img(Rect(0,r,c,r)),l3,d3l);
        lsd->detect( img(Rect(c,r,c,r)),l4,scale,nOctaves );
        lbd->compute(img(Rect(c,r,c,r)),l4,d4l);
        for(unsigned int i = 0; i < l1.size(); i++){
            l1[i].startPointX += 0;
            l1[i].startPointY += 0;
            l1[i].endPointX   += 0;
            l1[i].endPointY   += 0;
            lines1L.push_back(l1[i]);
            ldesc1L.push_back(d1l.row(i));
        }
        for(unsigned int i = 0; i < l2.size(); i++){
            l2[i].startPointX += c;
            l2[i].startPointY += 0;
            l2[i].endPointX   += c;
            l2[i].endPointY   += 0;
            lines1L.push_back(l2[i]);
            ldesc1L.push_back(d2l.row(i));
        }
        for(unsigned int i = 0; i < l3.size(); i++){
            l3[i].startPointX += 0;
            l3[i].startPointY += r;
            l3[i].endPointX   += 0;
            l3[i].endPointY   += r;
            lines1L.push_back(l3[i]);
            ldesc1L.push_back(d3l.row(i));
        }
        for(unsigned int i = 0; i < l4.size(); i++){
            l4[i].startPointX += c;
            l4[i].startPointY += r;
            l4[i].endPointX   += c;
            l4[i].endPointY   += r;
            lines1L.push_back(l4[i]);
            ldesc1L.push_back(d4l.row(i));
        }
    }
}

void plSVO::detect1R(){

    Mat img = imgRight(Rect(0,0,C,R));

    if(points){
        Ptr<ORB>         orb = ORB::create(nFeaturesB,scaleFactor,nLevels);
        vector<KeyPoint> p1, p2, p3, p4;
        Mat              d1, d2, d3, d4;
        orb->detectAndCompute(img(Rect(0,0,c,r)), cv::Mat(),p1, d1,false);
        orb->detectAndCompute(img(Rect(c,0,c,r)), cv::Mat(),p2, d2,false);
        orb->detectAndCompute(img(Rect(0,r,c,r)), cv::Mat(),p3, d3,false);
        orb->detectAndCompute(img(Rect(c,r,c,r)), cv::Mat(),p4, d4,false);
        points1R.clear();
        pdesc1R.release();
        for(unsigned int i = 0; i < p1.size(); i++){
            p1[i].pt.x += 0;
            p1[i].pt.y += 0;
            points1R.push_back(p1[i]);
            pdesc1R.push_back(d1.row(i));
        }
        for(unsigned int i = 0; i < p2.size(); i++){
            p2[i].pt.x += c;
            p2[i].pt.y += 0;
            points1R.push_back(p2[i]);
            pdesc1R.push_back(d2.row(i));
        }
        for(unsigned int i = 0; i < p3.size(); i++){
            p3[i].pt.x += 0;
            p3[i].pt.y += r;
            points1R.push_back(p3[i]);
            pdesc1R.push_back(d3.row(i));
        }
        for(unsigned int i = 0; i < p4.size(); i++){
            p4[i].pt.x += c;
            p4[i].pt.y += r;
            points1R.push_back(p4[i]);
            pdesc1R.push_back(d4.row(i));
        }
    }

    if(false){
        Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
        Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
        vector<KeyLine>         l1, l2, l3, l4;
        Mat                     d1l, d2l, d3l, d4l;
        lines1R.clear();
        ldesc1R.release();
        lsd->detect( img(Rect(0,0,c,r)),l1,scale,nOctaves );
        lbd->compute(img(Rect(0,0,c,r)),l1,d1l);
        lsd->detect( img(Rect(c,0,c,r)),l2,scale,nOctaves );
        lbd->compute(img(Rect(c,0,c,r)),l2,d2l);
        lsd->detect( img(Rect(0,r,c,r)),l3,scale,nOctaves );
        lbd->compute(img(Rect(0,r,c,r)),l3,d3l);
        lsd->detect( img(Rect(c,r,c,r)),l4,scale,nOctaves );
        lbd->compute(img(Rect(c,r,c,r)),l4,d4l);
        for(unsigned int i = 0; i < l1.size(); i++){
            l1[i].startPointX += 0;
            l1[i].startPointY += 0;
            l1[i].endPointX   += 0;
            l1[i].endPointY   += 0;
            lines1R.push_back(l1[i]);
            ldesc1R.push_back(d1l.row(i));
        }
        for(unsigned int i = 0; i < l2.size(); i++){
            l2[i].startPointX += c;
            l2[i].startPointY += 0;
            l2[i].endPointX   += c;
            l2[i].endPointY   += 0;
            lines1R.push_back(l2[i]);
            ldesc1R.push_back(d2l.row(i));
        }
        for(unsigned int i = 0; i < l3.size(); i++){
            l3[i].startPointX += 0;
            l3[i].startPointY += r;
            l3[i].endPointX   += 0;
            l3[i].endPointY   += r;
            lines1R.push_back(l3[i]);
            ldesc1R.push_back(d3l.row(i));
        }
        for(unsigned int i = 0; i < l4.size(); i++){
            l4[i].startPointX += c;
            l4[i].startPointY += r;
            l4[i].endPointX   += c;
            l4[i].endPointY   += r;
            lines1R.push_back(l4[i]);
            ldesc1R.push_back(d4l.row(i));
        }
    }
}

void plSVO::detect2L(){

    Mat img = imgLeft(Rect(C,0,C,R));

    if(points){
        Ptr<ORB>         orb = ORB::create(nFeaturesB,scaleFactor,nLevels);
        vector<KeyPoint> p1, p2, p3, p4;
        Mat              d1, d2, d3, d4;
        orb->detectAndCompute(img(Rect(0,0,c,r)), cv::Mat(),p1, d1,false);
        orb->detectAndCompute(img(Rect(c,0,c,r)), cv::Mat(),p2, d2,false);
        orb->detectAndCompute(img(Rect(0,r,c,r)), cv::Mat(),p3, d3,false);
        orb->detectAndCompute(img(Rect(c,r,c,r)), cv::Mat(),p4, d4,false);
        points2L.clear();
        pdesc2L.release();
        for(unsigned int i = 0; i < p1.size(); i++){
            p1[i].pt.x += 0 + C;
            p1[i].pt.y += 0;
            points2L.push_back(p1[i]);
            pdesc2L.push_back(d1.row(i));
        }
        for(unsigned int i = 0; i < p2.size(); i++){
            p2[i].pt.x += c + C;
            p2[i].pt.y += 0;
            points2L.push_back(p2[i]);
            pdesc2L.push_back(d2.row(i));
        }
        for(unsigned int i = 0; i < p3.size(); i++){
            p3[i].pt.x += 0 + C;
            p3[i].pt.y += r;
            points2L.push_back(p3[i]);
            pdesc2L.push_back(d3.row(i));
        }
        for(unsigned int i = 0; i < p4.size(); i++){
            p4[i].pt.x += c + C;
            p4[i].pt.y += r;
            points2L.push_back(p4[i]);
            pdesc2L.push_back(d4.row(i));
        }
    }

    if(false){
        Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
        Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
        vector<KeyLine>         l1, l2, l3, l4;
        Mat                     d1l, d2l, d3l, d4l;
        lines2L.clear();
        ldesc2L.release();
        lsd->detect( img(Rect(0,0,c,r)),l1,scale,nOctaves );
        lbd->compute(img(Rect(0,0,c,r)),l1,d1l);
        lsd->detect( img(Rect(c,0,c,r)),l2,scale,nOctaves );
        lbd->compute(img(Rect(c,0,c,r)),l2,d2l);
        lsd->detect( img(Rect(0,r,c,r)),l3,scale,nOctaves );
        lbd->compute(img(Rect(0,r,c,r)),l3,d3l);
        lsd->detect( img(Rect(c,r,c,r)),l4,scale,nOctaves );
        lbd->compute(img(Rect(c,r,c,r)),l4,d4l);
        for(unsigned int i = 0; i < l1.size(); i++){
            l1[i].startPointX += 0 + C;
            l1[i].startPointY += 0;
            l1[i].endPointX   += 0 + C;
            l1[i].endPointY   += 0;
            lines2L.push_back(l1[i]);
            ldesc2L.push_back(d1l.row(i));
        }
        for(unsigned int i = 0; i < l2.size(); i++){
            l2[i].startPointX += c + C;
            l2[i].startPointY += 0;
            l2[i].endPointX   += c + C;
            l2[i].endPointY   += 0;
            lines2L.push_back(l2[i]);
            ldesc2L.push_back(d2l.row(i));
        }
        for(unsigned int i = 0; i < l3.size(); i++){
            l3[i].startPointX += 0 + C;
            l3[i].startPointY += r;
            l3[i].endPointX   += 0 + C;
            l3[i].endPointY   += r;
            lines2L.push_back(l3[i]);
            ldesc2L.push_back(d3l.row(i));
        }
        for(unsigned int i = 0; i < l4.size(); i++){
            l4[i].startPointX += c + C;
            l4[i].startPointY += r;
            l4[i].endPointX   += c + C;
            l4[i].endPointY   += r;
            lines2L.push_back(l4[i]);
            ldesc2L.push_back(d4l.row(i));
        }
    }

}

void plSVO::detect2R(){

    Mat img = imgRight(Rect(C,0,C,R));

    if(points){
        Ptr<ORB>         orb = ORB::create(nFeaturesB,scaleFactor,nLevels);
        vector<KeyPoint> p1, p2, p3, p4;
        Mat              d1, d2, d3, d4;
        orb->detectAndCompute(img(Rect(0,0,c,r)), cv::Mat(),p1, d1,false);
        orb->detectAndCompute(img(Rect(c,0,c,r)), cv::Mat(),p2, d2,false);
        orb->detectAndCompute(img(Rect(0,r,c,r)), cv::Mat(),p3, d3,false);
        orb->detectAndCompute(img(Rect(c,r,c,r)), cv::Mat(),p4, d4,false);
        points2R.clear();
        pdesc2R.release();
        for(unsigned int i = 0; i < p1.size(); i++){
            p1[i].pt.x += 0 + C;
            p1[i].pt.y += 0;
            points2R.push_back(p1[i]);
            pdesc2R.push_back(d1.row(i));
        }
        for(unsigned int i = 0; i < p2.size(); i++){
            p2[i].pt.x += c + C;
            p2[i].pt.y += 0;
            points2R.push_back(p2[i]);
            pdesc2R.push_back(d2.row(i));
        }
        for(unsigned int i = 0; i < p3.size(); i++){
            p3[i].pt.x += 0 + C;
            p3[i].pt.y += r;
            points2R.push_back(p3[i]);
            pdesc2R.push_back(d3.row(i));
        }
        for(unsigned int i = 0; i < p4.size(); i++){
            p4[i].pt.x += c + C;
            p4[i].pt.y += r;
            points2R.push_back(p4[i]);
            pdesc2R.push_back(d4.row(i));
        }
    }

    if(false){
        Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
        Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
        vector<KeyLine>         l1, l2, l3, l4;
        Mat                     d1l, d2l, d3l, d4l;
        lines2R.clear();
        ldesc2R.release();
        lsd->detect( img(Rect(0,0,c,r)),l1,scale,nOctaves );
        lbd->compute(img(Rect(0,0,c,r)),l1,d1l);
        lsd->detect( img(Rect(c,0,c,r)),l2,scale,nOctaves );
        lbd->compute(img(Rect(c,0,c,r)),l2,d2l);
        lsd->detect( img(Rect(0,r,c,r)),l3,scale,nOctaves );
        lbd->compute(img(Rect(0,r,c,r)),l3,d3l);
        lsd->detect( img(Rect(c,r,c,r)),l4,scale,nOctaves );
        lbd->compute(img(Rect(c,r,c,r)),l4,d4l);
        for(unsigned int i = 0; i < l1.size(); i++){
            l1[i].startPointX += 0 + C;
            l1[i].startPointY += 0;
            l1[i].endPointX   += 0 + C;
            l1[i].endPointY   += 0;
            lines2R.push_back(l1[i]);
            ldesc2R.push_back(d1l.row(i));
        }
        for(unsigned int i = 0; i < l2.size(); i++){
            l2[i].startPointX += c + C;
            l2[i].startPointY += 0;
            l2[i].endPointX   += c + C;
            l2[i].endPointY   += 0;
            lines2R.push_back(l2[i]);
            ldesc2R.push_back(d2l.row(i));
        }
        for(unsigned int i = 0; i < l3.size(); i++){
            l3[i].startPointX += 0 + C;
            l3[i].startPointY += r;
            l3[i].endPointX   += 0 + C;
            l3[i].endPointY   += r;
            lines2R.push_back(l3[i]);
            ldesc2R.push_back(d3l.row(i));
        }
        for(unsigned int i = 0; i < l4.size(); i++){
            l4[i].startPointX += c + C;
            l4[i].startPointY += r;
            l4[i].endPointX   += c + C;
            l4[i].endPointY   += r;
            lines2R.push_back(l4[i]);
            ldesc2R.push_back(d4l.row(i));
        }
    }

}

void plSVO::detect3L(){

    Mat img = imgLeft(Rect(0,R,C,R));

    if(points){
        Ptr<ORB>         orb = ORB::create(nFeaturesB,scaleFactor,nLevels);
        vector<KeyPoint> p1, p2, p3, p4;
        Mat              d1, d2, d3, d4;
        orb->detectAndCompute(img(Rect(0,0,c,r)), cv::Mat(),p1, d1,false);
        orb->detectAndCompute(img(Rect(c,0,c,r)), cv::Mat(),p2, d2,false);
        orb->detectAndCompute(img(Rect(0,r,c,r)), cv::Mat(),p3, d3,false);
        orb->detectAndCompute(img(Rect(c,r,c,r)), cv::Mat(),p4, d4,false);
        points3L.clear();
        pdesc3L.release();
        for(unsigned int i = 0; i < p1.size(); i++){
            p1[i].pt.x += 0;
            p1[i].pt.y += 0 + R;
            points3L.push_back(p1[i]);
            pdesc3L.push_back(d1.row(i));
        }
        for(unsigned int i = 0; i < p2.size(); i++){
            p2[i].pt.x += c;
            p2[i].pt.y += 0 + R;
            points3L.push_back(p2[i]);
            pdesc3L.push_back(d2.row(i));
        }
        for(unsigned int i = 0; i < p3.size(); i++){
            p3[i].pt.x += 0;
            p3[i].pt.y += r + R;
            points3L.push_back(p3[i]);
            pdesc3L.push_back(d3.row(i));
        }
        for(unsigned int i = 0; i < p4.size(); i++){
            p4[i].pt.x += c;
            p4[i].pt.y += r + R;
            points3L.push_back(p4[i]);
            pdesc3L.push_back(d4.row(i));
        }
    }

    if(false){
        Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
        Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
        vector<KeyLine>         l1, l2, l3, l4;
        Mat                     d1l, d2l, d3l, d4l;
        lines3L.clear();
        ldesc3L.release();
        lsd->detect( img(Rect(0,0,c,r)),l1,scale,nOctaves );
        lbd->compute(img(Rect(0,0,c,r)),l1,d1l);
        lsd->detect( img(Rect(c,0,c,r)),l2,scale,nOctaves );
        lbd->compute(img(Rect(c,0,c,r)),l2,d2l);
        lsd->detect( img(Rect(0,r,c,r)),l3,scale,nOctaves );
        lbd->compute(img(Rect(0,r,c,r)),l3,d3l);
        lsd->detect( img(Rect(c,r,c,r)),l4,scale,nOctaves );
        lbd->compute(img(Rect(c,r,c,r)),l4,d4l);
        for(unsigned int i = 0; i < l1.size(); i++){
            l1[i].startPointX += 0;
            l1[i].startPointY += 0 + R;
            l1[i].endPointX   += 0;
            l1[i].endPointY   += 0 + R;
            lines3L.push_back(l1[i]);
            ldesc3L.push_back(d1l.row(i));
        }
        for(unsigned int i = 0; i < l2.size(); i++){
            l2[i].startPointX += c;
            l2[i].startPointY += 0 + R;
            l2[i].endPointX   += c;
            l2[i].endPointY   += 0 + R;
            lines3L.push_back(l2[i]);
            ldesc3L.push_back(d2l.row(i));
        }
        for(unsigned int i = 0; i < l3.size(); i++){
            l3[i].startPointX += 0;
            l3[i].startPointY += r + R;
            l3[i].endPointX   += 0;
            l3[i].endPointY   += r + R;
            lines3L.push_back(l3[i]);
            ldesc3L.push_back(d3l.row(i));
        }
        for(unsigned int i = 0; i < l4.size(); i++){
            l4[i].startPointX += c;
            l4[i].startPointY += r + R;
            l4[i].endPointX   += c;
            l4[i].endPointY   += r + R;
            lines3L.push_back(l4[i]);
            ldesc3L.push_back(d4l.row(i));
        }
    }
}

void plSVO::detect3R(){

    Mat img = imgRight(Rect(0,R,C,R));

    if(points){
        Ptr<ORB>         orb = ORB::create(nFeaturesB,scaleFactor,nLevels);
        vector<KeyPoint> p1, p2, p3, p4;
        Mat              d1, d2, d3, d4;
        orb->detectAndCompute(img(Rect(0,0,c,r)), cv::Mat(),p1, d1,false);
        orb->detectAndCompute(img(Rect(c,0,c,r)), cv::Mat(),p2, d2,false);
        orb->detectAndCompute(img(Rect(0,r,c,r)), cv::Mat(),p3, d3,false);
        orb->detectAndCompute(img(Rect(c,r,c,r)), cv::Mat(),p4, d4,false);
        points3R.clear();
        pdesc3R.release();;
        for(unsigned int i = 0; i < p1.size(); i++){
            p1[i].pt.x += 0;
            p1[i].pt.y += 0 + R;
            points3R.push_back(p1[i]);
            pdesc3R.push_back(d1.row(i));
        }
        for(unsigned int i = 0; i < p2.size(); i++){
            p2[i].pt.x += c;
            p2[i].pt.y += 0 + R;
            points3R.push_back(p2[i]);
            pdesc3R.push_back(d2.row(i));
        }
        for(unsigned int i = 0; i < p3.size(); i++){
            p3[i].pt.x += 0;
            p3[i].pt.y += r + R;
            points3R.push_back(p3[i]);
            pdesc3R.push_back(d3.row(i));
        }
        for(unsigned int i = 0; i < p4.size(); i++){
            p4[i].pt.x += c;
            p4[i].pt.y += r + R;
            points3R.push_back(p4[i]);
            pdesc3R.push_back(d4.row(i));
        }
    }

    if(false){
        Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
        Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
        vector<KeyLine>         l1, l2, l3, l4;
        Mat                     d1l, d2l, d3l, d4l;
        lines3R.clear();
        ldesc3R.release();;
        lsd->detect( img(Rect(0,0,c,r)),l1,scale,nOctaves );
        lbd->compute(img(Rect(0,0,c,r)),l1,d1l);
        lsd->detect( img(Rect(c,0,c,r)),l2,scale,nOctaves );
        lbd->compute(img(Rect(c,0,c,r)),l2,d2l);
        lsd->detect( img(Rect(0,r,c,r)),l3,scale,nOctaves );
        lbd->compute(img(Rect(0,r,c,r)),l3,d3l);
        lsd->detect( img(Rect(c,r,c,r)),l4,scale,nOctaves );
        lbd->compute(img(Rect(c,r,c,r)),l4,d4l);
        for(unsigned int i = 0; i < l1.size(); i++){
            l1[i].startPointX += 0;
            l1[i].startPointY += 0 + R;
            l1[i].endPointX   += 0;
            l1[i].endPointY   += 0 + R;
            lines3R.push_back(l1[i]);
            ldesc3R.push_back(d1l.row(i));
        }
        for(unsigned int i = 0; i < l2.size(); i++){
            l2[i].startPointX += c;
            l2[i].startPointY += 0 + R;
            l2[i].endPointX   += c;
            l2[i].endPointY   += 0 + R;
            lines3R.push_back(l2[i]);
            ldesc3R.push_back(d2l.row(i));
        }
        for(unsigned int i = 0; i < l3.size(); i++){
            l3[i].startPointX += 0;
            l3[i].startPointY += r + R;
            l3[i].endPointX   += 0;
            l3[i].endPointY   += r + R;
            lines3R.push_back(l3[i]);
            ldesc3R.push_back(d3l.row(i));
        }
        for(unsigned int i = 0; i < l4.size(); i++){
            l4[i].startPointX += c;
            l4[i].startPointY += r + R;
            l4[i].endPointX   += c;
            l4[i].endPointY   += r + R;
            lines3R.push_back(l4[i]);
            ldesc3R.push_back(d4l.row(i));
        }
    }
}

void plSVO::detect4L(){

    Mat img = imgLeft(Rect(C,R,C,R));

    if(points){
        Ptr<ORB>         orb = ORB::create(nFeaturesB,scaleFactor,nLevels);
        vector<KeyPoint> p1, p2, p3, p4;
        Mat              d1, d2, d3, d4;
        orb->detectAndCompute(img(Rect(0,0,c,r)), cv::Mat(),p1, d1,false);
        orb->detectAndCompute(img(Rect(c,0,c,r)), cv::Mat(),p2, d2,false);
        orb->detectAndCompute(img(Rect(0,r,c,r)), cv::Mat(),p3, d3,false);
        orb->detectAndCompute(img(Rect(c,r,c,r)), cv::Mat(),p4, d4,false);
        points4L.clear();
        pdesc4L.release();;
        for(unsigned int i = 0; i < p1.size(); i++){
            p1[i].pt.x += 0 + C;
            p1[i].pt.y += 0 + R;
            points4L.push_back(p1[i]);
            pdesc4L.push_back(d1.row(i));
        }
        for(unsigned int i = 0; i < p2.size(); i++){
            p2[i].pt.x += c + C;
            p2[i].pt.y += 0 + R;
            points4L.push_back(p2[i]);
            pdesc4L.push_back(d2.row(i));
        }
        for(unsigned int i = 0; i < p3.size(); i++){
            p3[i].pt.x += 0 + C;
            p3[i].pt.y += r + R;
            points4L.push_back(p3[i]);
            pdesc4L.push_back(d3.row(i));
        }
        for(unsigned int i = 0; i < p4.size(); i++){
            p4[i].pt.x += c + C;
            p4[i].pt.y += r + R;
            points4L.push_back(p4[i]);
            pdesc4L.push_back(d4.row(i));
        }
    }

    if(false){
        Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
        Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
        vector<KeyLine>         l1, l2, l3, l4;
        Mat                     d1l, d2l, d3l, d4l;
        lines4L.clear();
        ldesc4L.release();;
        lsd->detect( img(Rect(0,0,c,r)),l1,scale,nOctaves );
        lbd->compute(img(Rect(0,0,c,r)),l1,d1l);
        lsd->detect( img(Rect(c,0,c,r)),l2,scale,nOctaves );
        lbd->compute(img(Rect(c,0,c,r)),l2,d2l);
        lsd->detect( img(Rect(0,r,c,r)),l3,scale,nOctaves );
        lbd->compute(img(Rect(0,r,c,r)),l3,d3l);
        lsd->detect( img(Rect(c,r,c,r)),l4,scale,nOctaves );
        lbd->compute(img(Rect(c,r,c,r)),l4,d4l);
        for(unsigned int i = 0; i < l1.size(); i++){
            l1[i].startPointX += 0 + C;
            l1[i].startPointY += 0 + R;
            l1[i].endPointX   += 0 + C;
            l1[i].endPointY   += 0 + R;
            lines4L.push_back(l1[i]);
            ldesc4L.push_back(d1l.row(i));
        }
        for(unsigned int i = 0; i < l2.size(); i++){
            l2[i].startPointX += c + C;
            l2[i].startPointY += 0 + R;
            l2[i].endPointX   += c + C;
            l2[i].endPointY   += 0 + R;
            lines4L.push_back(l2[i]);
            ldesc4L.push_back(d2l.row(i));
        }
        for(unsigned int i = 0; i < l3.size(); i++){
            l3[i].startPointX += 0 + C;
            l3[i].startPointY += r + R;
            l3[i].endPointX   += 0 + C;
            l3[i].endPointY   += r + R;
            lines4L.push_back(l3[i]);
            ldesc4L.push_back(d3l.row(i));
        }
        for(unsigned int i = 0; i < l4.size(); i++){
            l4[i].startPointX += c + C;
            l4[i].startPointY += r + R;
            l4[i].endPointX   += c + C;
            l4[i].endPointY   += r + R;
            lines4L.push_back(l4[i]);
            ldesc4L.push_back(d4l.row(i));
        }
    }
}

void plSVO::detect4R(){

    Mat img = imgRight(Rect(C,R,C,R));

    if(points){
        Ptr<ORB>         orb = ORB::create(nFeaturesB,scaleFactor,nLevels);
        vector<KeyPoint> p1, p2, p3, p4;
        Mat              d1, d2, d3, d4;
        orb->detectAndCompute(img(Rect(0,0,c,r)), cv::Mat(),p1, d1,false);
        orb->detectAndCompute(img(Rect(c,0,c,r)), cv::Mat(),p2, d2,false);
        orb->detectAndCompute(img(Rect(0,r,c,r)), cv::Mat(),p3, d3,false);
        orb->detectAndCompute(img(Rect(c,r,c,r)), cv::Mat(),p4, d4,false);
        points4R.clear();
        pdesc4R.release();
        for(unsigned int i = 0; i < p1.size(); i++){
            p1[i].pt.x += 0 + C;
            p1[i].pt.y += 0 + R;
            points4R.push_back(p1[i]);
            pdesc4R.push_back(d1.row(i));
        }
        for(unsigned int i = 0; i < p2.size(); i++){
            p2[i].pt.x += c + C;
            p2[i].pt.y += 0 + R;
            points4R.push_back(p2[i]);
            pdesc4R.push_back(d2.row(i));
        }
        for(unsigned int i = 0; i < p3.size(); i++){
            p3[i].pt.x += 0 + C;
            p3[i].pt.y += r + R;
            points4R.push_back(p3[i]);
            pdesc4R.push_back(d3.row(i));
        }
        for(unsigned int i = 0; i < p4.size(); i++){
            p4[i].pt.x += c + C;
            p4[i].pt.y += r + R;
            points4R.push_back(p4[i]);
            pdesc4R.push_back(d4.row(i));
        }
    }

    if(false){
        Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
        Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
        vector<KeyLine>         l1, l2, l3, l4;
        Mat                     d1l, d2l, d3l, d4l;
        lines4R.clear();
        ldesc4R.release();;
        lsd->detect( img(Rect(0,0,c,r)),l1,scale,nOctaves );
        lbd->compute(img(Rect(0,0,c,r)),l1,d1l);
        lsd->detect( img(Rect(c,0,c,r)),l2,scale,nOctaves );
        lbd->compute(img(Rect(c,0,c,r)),l2,d2l);
        lsd->detect( img(Rect(0,r,c,r)),l3,scale,nOctaves );
        lbd->compute(img(Rect(0,r,c,r)),l3,d3l);
        lsd->detect( img(Rect(c,r,c,r)),l4,scale,nOctaves );
        lbd->compute(img(Rect(c,r,c,r)),l4,d4l);
        for(unsigned int i = 0; i < l1.size(); i++){
            l1[i].startPointX += 0 + C;
            l1[i].startPointY += 0 + R;
            l1[i].endPointX   += 0 + C;
            l1[i].endPointY   += 0 + R;
            lines4R.push_back(l1[i]);
            ldesc4R.push_back(d1l.row(i));
        }
        for(unsigned int i = 0; i < l2.size(); i++){
            l2[i].startPointX += c + C;
            l2[i].startPointY += 0 + R;
            l2[i].endPointX   += c + C;
            l2[i].endPointY   += 0 + R;
            lines4R.push_back(l2[i]);
            ldesc4R.push_back(d2l.row(i));
        }
        for(unsigned int i = 0; i < l3.size(); i++){
            l3[i].startPointX += 0 + C;
            l3[i].startPointY += r + R;
            l3[i].endPointX   += 0 + C;
            l3[i].endPointY   += r + R;
            lines4R.push_back(l3[i]);
            ldesc4R.push_back(d3l.row(i));
        }
        for(unsigned int i = 0; i < l4.size(); i++){
            l4[i].startPointX += c + C;
            l4[i].startPointY += r + R;
            l4[i].endPointX   += c + C;
            l4[i].endPointY   += r + R;
            lines4R.push_back(l4[i]);
            ldesc4R.push_back(d4l.row(i));
        }
    }
}

void plSVO::detectL(){

    // Declarations
    Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
    Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
    Ptr<ORB>                orb = ORB::create(nFeatures,scaleFactor,nLevels);

    // Feature detection and description
    if(lines){
        lsd->detect(imgLeft,linesThird,scale,nOctaves );
        lbd->compute(imgLeft,linesThird,ldescThird);
    }
    if(points){
        if(bucketing){
            thread first(&plSVO::detect1L, this);
            thread second(&plSVO::detect2L,this);
            thread third(&plSVO::detect3L, this);
            thread fourth(&plSVO::detect4L,this);
            first.join();
            second.join();
            third.join();
            fourth.join();
            if(points){
                for(unsigned int i = 0; i < points1L.size(); i++){
                    pointsThird.push_back(points1L[i]);
                    pdescThird.push_back(pdesc1L.row(i));
                }
                for(unsigned int i = 0; i < points2L.size(); i++){
                    pointsThird.push_back(points2L[i]);
                    pdescThird.push_back(pdesc2L.row(i));
                }
                for(unsigned int i = 0; i < points3L.size(); i++){
                    pointsThird.push_back(points3L[i]);
                    pdescThird.push_back(pdesc3L.row(i));
                }
                for(unsigned int i = 0; i < points4L.size(); i++){
                    pointsThird.push_back(points4L[i]);
                    pdescThird.push_back(pdesc4L.row(i));
                }
            }
        }
        else
            orb->detectAndCompute(imgLeft, cv::Mat(),pointsThird, pdescThird,false);
    }

}

void plSVO::detectR(){

    // Declarations
    Ptr<LSDDetector>        lsd = LSDDetector::createLSDDetector();
    Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
    Ptr<ORB>                orb = ORB::create(nFeatures,scaleFactor,nLevels);

    // Feature detection and description
    if(lines){
        lsd->detect(imgRight,linesFourth,scale,nOctaves );
        lbd->compute(imgRight,linesFourth,ldescFourth);
    }
    if(points){
        if(bucketing){
            thread first(&plSVO::detect1R, this);
            thread second(&plSVO::detect2R,this);
            thread third(&plSVO::detect3R, this);
            thread fourth(&plSVO::detect4R,this);
            first.join();
            second.join();
            third.join();
            fourth.join();
            if(points){
                for(unsigned int i = 0; i < points1R.size(); i++){
                    pointsFourth.push_back(points1R[i]);
                    pdescFourth.push_back(pdesc1R.row(i));
                }
                for(unsigned int i = 0; i < points2R.size(); i++){
                    pointsFourth.push_back(points2R[i]);
                    pdescFourth.push_back(pdesc2R.row(i));
                }
                for(unsigned int i = 0; i < points3R.size(); i++){
                    pointsFourth.push_back(points3R[i]);
                    pdescFourth.push_back(pdesc3R.row(i));
                }
                for(unsigned int i = 0; i < points4R.size(); i++){
                    pointsFourth.push_back(points4R[i]);
                    pdescFourth.push_back(pdesc4R.row(i));
                }
            }
        }
        else
            orb->detectAndCompute(imgRight, cv::Mat(),pointsFourth, pdescFourth,false);
    }

}

void plSVO::stereoMatching(){

    /* Points Stereo Matching */
    vector<vector<DMatch>> pmatches;
    int m1, m2;
    // Match the stereo points
    if( (pointsThird.size()>0) && (pointsFourth.size()>0) ){
        bfm.knnMatch(pdescThird,pdescFourth,pmatches,2);
        // Filter the bad matches
        for(unsigned int i = 0; i < pmatches.size(); i++){
            // TODO: ALSO CHECK THE MEDIAN VALUE OF THE DESCRIPTOR TO FILTER THE OUTLIERS
            if(pmatches[i][0].distance > pmatches[i][1].distance * minDist12 ){             //min distance between the 2NN
                m1 = pmatches[i][0].queryIdx;
                m2 = pmatches[i][0].trainIdx;
                if( abs(pointsThird[m1].pt.y-pointsFourth[m2].pt.y) <= maxStDist ){         //stereo constraint
                    if( (pointsThird[m1].pt.x-pointsFourth[m2].pt.x) >= dispMin ){          //disparity > dispMin
                        pd1aux.push_back(pdescThird.row(m1));
                        pd2aux.push_back(pdescFourth.row(m2));
                        p1aux.push_back(pointsThird[m1]);
                        p2aux.push_back(pointsFourth[m2]);
                    }
                }
            }
        }
        // Copy the valid points
        pointsThird.clear(); pointsFourth.clear();
        pointsThird  = p1aux;
        pointsFourth = p2aux;
        // Copy the valid descriptors
        pd1aux.copyTo(pdescThird);
        pd2aux.copyTo(pdescFourth);
    }
    /* Lines Stereo Matching */
    Ptr<BinaryDescriptorMatcher>    bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    vector<vector<DMatch>>  lmatches;
    Vector3f                sp, ep, ll, rl;
    // Match the stereo lines
    if( (linesThird.size()>0)&&(linesFourth.size()>0) ){
        bdm->knnMatch( ldescThird, ldescFourth, lmatches, 2 );    
        // Calculate the thresholds with the median values
        int nMatch = lmatches.size();
        if(nMatch > 0){
            VectorXf dist(nMatch), dist_sort(nMatch), diff(nMatch), diff_sort(nMatch);
            for(int i = 0; i < nMatch; i++){
                dist(i) = lmatches[i][0].distance;
                diff(i) = lmatches[i][1].distance-lmatches[i][0].distance;
            }
            dist_sort = dist; diff_sort = diff;
            sort(dist_sort.derived().data(),dist_sort.derived().data()+dist_sort.size());
            sort(diff_sort.derived().data(),diff_sort.derived().data()+diff_sort.size());
            float distTh = dist_sort((int)(matchPr*nMatch));
            float diffTh = diff_sort((int)((1.0f-matchPr)*nMatch));
            // Filter the bad matches
            for(unsigned int i=0; i<nMatch; i++){
                if ( (dist(i) < distTh ) && ( diff(i) > diffTh ) ){             // Description condition
                    m1 = lmatches[i][0].queryIdx;
                    m2 = lmatches[i][0].trainIdx;
                    sp << linesThird[m1].startPointX,  linesThird[m1].startPointY,  1.f;
                    ep << linesThird[m1].endPointX,    linesThird[m1].endPointY,    1.f;
                    ll << sp.cross(ep);
                    sp << linesFourth[m2].startPointX, linesFourth[m2].startPointY, 1.f;
                    ep << linesFourth[m2].endPointX,   linesFourth[m2].endPointY,   1.f;
                    rl << sp.cross(ep);
                    if(abs(ll(0)) > horizTh && abs(rl(0)) > horizTh){                                                                                           // Avoid horizontal lines
                        if( abs(linesThird[m1].endPointX+linesThird[m1].startPointX-linesFourth[m2].startPointX-linesFourth[m2].endPointX) < 2.f * dispMax ){   // Max disparity condition
                            if( abs(angDiff(linesThird[m1].angle,linesFourth[m2].angle)) < alphaMax ){                                                          // Angular difference condition
                                l1aux.push_back(linesThird[m1]);
                                l2aux.push_back(linesFourth[m2]);
                                ld1aux.push_back(ldescThird.row(m1));
                                ld2aux.push_back(ldescFourth.row(m2));
                            }
                        }
                    }
                }
            }
            // Copy the valid lines
            linesThird.clear(); linesFourth.clear();
            linesThird = l1aux; linesFourth = l2aux;
            // Copy the valid descriptors
            ld1aux.copyTo(ldescThird);
            ld2aux.copyTo(ldescFourth);
        }
    }
}

void plSVO::stereoMutualMatching(){

    /* Points Stereo Matching */
    vector<vector<DMatch>> pmatches, pmatches_;
    int m1, m2, m1_, m2_;
    // Match the stereo points
    if( (pointsThird.size()>0) && (pointsFourth.size()>0) ){
        bfm.knnMatch(pdescThird,pdescFourth,pmatches,2);
        bfm.knnMatch(pdescFourth,pdescThird,pmatches_,2);
        // Filter the bad matches
        for(unsigned int i = 0; i < pmatches.size(); i++){
            // TODO: ALSO CHECK THE MEDIAN VALUE OF THE DESCRIPTOR TO FILTER THE OUTLIERS
            if(pmatches[i][0].distance > pmatches[i][1].distance * minDist12 ){             //min distance between the 2NN
                m1 = pmatches[i][0].queryIdx;
                m2 = pmatches[i][0].trainIdx;
                for(unsigned int j = 0; j < pmatches_.size(); j++){
                    m1_ = pmatches_[j][0].queryIdx;
                    m2_ = pmatches_[j][0].trainIdx;
                    if(m2_ == m1){
                        if(m1_ == m2){
                            j = pmatches_.size();
                            if( abs(pointsThird[m1].pt.y-pointsFourth[m2].pt.y) <= maxStDist ){         //stereo constraint
                                if( (pointsThird[m1].pt.x-pointsFourth[m2].pt.x) >= dispMin ){          //disparity > dispMin
                                    pd1aux.push_back(pdescThird.row(m1));
                                    pd2aux.push_back(pdescFourth.row(m2));
                                    p1aux.push_back(pointsThird[m1]);
                                    p2aux.push_back(pointsFourth[m2]);
                                }
                            }
                        }
                    }

                }
            }
        }
        // Copy the valid points
        pointsThird.clear(); pointsFourth.clear();
        pointsThird  = p1aux;
        pointsFourth = p2aux;
        // Copy the valid descriptors
        pd1aux.copyTo(pdescThird);
        pd2aux.copyTo(pdescFourth);
    }

    /* Lines Stereo Matching */
    Ptr<BinaryDescriptorMatcher>    bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    vector<vector<DMatch>>  lmatches, lmatches_;
    Vector3f                sp, ep, ll, rl;
    // Match the stereo lines
    if( (linesThird.size()>0)&&(linesFourth.size()>0) ){
        bdm->knnMatch( ldescThird, ldescFourth, lmatches, 2 );
        bdm->knnMatch( ldescFourth, ldescThird, lmatches_, 2 );
        // Calculate the thresholds with the median values
        int nMatch = lmatches.size();
        if(nMatch > 0){
            VectorXf dist(nMatch), dist_sort(nMatch), diff(nMatch), diff_sort(nMatch);
            for(int i = 0; i < nMatch; i++){
                dist(i) = lmatches[i][0].distance;
                diff(i) = lmatches[i][1].distance-lmatches[i][0].distance;
            }
            dist_sort = dist; diff_sort = diff;
            sort(dist_sort.derived().data(),dist_sort.derived().data()+dist_sort.size());
            sort(diff_sort.derived().data(),diff_sort.derived().data()+diff_sort.size());
            float distTh = dist_sort((int)(matchPr*nMatch));
            float diffTh = diff_sort((int)((1.0f-matchPr)*nMatch));
            // Filter the bad matches
            for(unsigned int i=0; i<nMatch; i++){
                if ( (dist(i) < distTh ) && ( diff(i) > diffTh ) ){             // Description condition
                    m1 = lmatches[i][0].queryIdx;
                    m2 = lmatches[i][0].trainIdx;
                    for(unsigned int j=0; j<lmatches_.size(); j++){
                        m1_ = lmatches_[j][0].queryIdx;
                        m2_ = lmatches_[j][0].trainIdx;
                        if(m2_ == m1){
                            if(m1_ == m2){
                                j = lmatches_.size();
                                sp << linesThird[m1].startPointX,  linesThird[m1].startPointY,  1.f;
                                ep << linesThird[m1].endPointX,    linesThird[m1].endPointY,    1.f;
                                ll << sp.cross(ep);
                                sp << linesFourth[m2].startPointX, linesFourth[m2].startPointY, 1.f;
                                ep << linesFourth[m2].endPointX,   linesFourth[m2].endPointY,   1.f;
                                rl << sp.cross(ep);
                                if(abs(ll(0)) > horizTh && abs(rl(0)) > horizTh){                                                                                           // Avoid horizontal lines
                                    if( abs(linesThird[m1].endPointX+linesThird[m1].startPointX-linesFourth[m2].startPointX-linesFourth[m2].endPointX) < 2.f * dispMax ){   // Max disparity condition
                                        if( abs(angDiff(linesThird[m1].angle,linesFourth[m2].angle)) < alphaMax ){                                                          // Angular difference condition
                                            l1aux.push_back(linesThird[m1]);
                                            l2aux.push_back(linesFourth[m2]);
                                            ld1aux.push_back(ldescThird.row(m1));
                                            ld2aux.push_back(ldescFourth.row(m2));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Copy the valid lines
            linesThird.clear(); linesFourth.clear();
            linesThird = l1aux; linesFourth = l2aux;
            // Copy the valid descriptors
            ld1aux.copyTo(ldescThird);
            ld2aux.copyTo(ldescFourth);
        }
    }
}

void plSVO::f2fTracking(){

    /* Points Tracking */
    vector<vector<DMatch>>  matchesL, matchesR;
    int                     m1, m2, m3, m4, k = 0;
    MatrixXf                pData_ = MatrixXf::Zero(Np,pointsFirst.size());
    float                   bd;
    Vector3f                pointL, pointR, point3D;
    // Match the stereo points (we discard checking the L-R consistency since we performs a robust optimization and otherwise the number of points decrease critically)
    if( (pointsFirst.size()>0) && (pointsThird.size()>0) ){
        // TODO: MUTUAL BEST MATCHES (IF IT IS FAST...)
        bfm.knnMatch(pdescFirst,pdescThird,matchesL,2);
        // Filter the bad matches
        for(unsigned int i = 0; i<matchesL.size(); i++){
            // TODO: ALSO CHECK THE MEDIAN VALUE OF THE DESCRIPTOR TO FILTER THE OUTLIERS
            m1 = matchesL[i][0].queryIdx;   m2 = m1;
            m3 = matchesL[i][0].trainIdx;   m4 = m3;
            if( (abs(pointsFirst[m1].pt.x-pointsThird[m3].pt.x)<=maxF2FDist) && (abs(pointsSecond[m2].pt.x-pointsFourth[m4].pt.x)<=maxF2FDist) ){      //f2f consistency
                // ------- Data Formation
                // Form the 2D point vectors
                pointL << pointsFirst[m1].pt.x,  pointsFirst[m1].pt.y,  1.f;
                pointR << pointsSecond[m2].pt.x, pointsSecond[m2].pt.y, 1.f;
                // Back-projection to 3D points
                bd = baseline / ( pointL(0) - pointR(0) );
                point3D <<  bd * (pointL(0) - cx) ,  bd * (pointL(1) - cy) , bd * f ;
                // Data formation
                pData_(0,k)  = point3D(0);
                pData_(1,k)  = point3D(1);
                pData_(2,k)  = point3D(2);
                pData_(3,k)  = pointL(0);
                pData_(4,k)  = pointL(1);
                pData_(5,k)  = pointR(0);
                pData_(6,k)  = pointR(1);
                pData_(7,k)  = pointsThird[m3].pt.x;
                pData_(8,k)  = pointsThird[m3].pt.y;
                pData_(9,k)  = pointsFourth[m4].pt.x;
                pData_(10,k) = pointsFourth[m4].pt.y;
                k++;
            }
        }
    }
    // Form the data
    pData = pData_.block(0,0,Np,k);
    nPoints  = k;
    nPointsH = k;
    k = 0;

    /* Line Segments Tracking */
    Ptr<BinaryDescriptorMatcher>    bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    vector<vector<DMatch>>  lmatches;
    vector<KeyLine>         l1_, l2_, l3_, l4_;
    Vector2f                sp, ep, mp1, mp2, x1, x2;
    Vector3f                line, spoint_L, epoint_L, spoint_R, epoint_R, point3D_s, point3D_e;
    float                   a1, a2, b_d, distTh = 0.f, diffTh = 0.f;
    unsigned int            nMatch;
    VectorXf                dist, dist_sort, diff, diff_sort;
    MatrixXf                lData_ = MatrixXf::Zero(Nl,linesFirst.size());
    // Match the f2f points (to reduce time it is only performed with the left images)
    if( (linesFirst.size()>0) && (linesThird.size()>0) ){
        bdm->knnMatch(ldescFirst,ldescThird,lmatches,2);
        if(lmatches.size()>0){
            // Calculate the thresholds with the median values
            nMatch    = lmatches.size();
            dist      = VectorXf::Zero(nMatch);
            diff      = VectorXf::Zero(nMatch);
            dist_sort = VectorXf::Zero(nMatch);
            diff_sort = VectorXf::Zero(nMatch);
            for(int i = 0; i < nMatch; i++){
                dist(i) = lmatches[i][0].distance;
                diff(i) = lmatches[i][1].distance-lmatches[i][0].distance;
            }
            dist_sort = dist; diff_sort = diff;
            sort(dist_sort.derived().data(),dist_sort.derived().data()+dist_sort.size());
            sort(diff_sort.derived().data(),diff_sort.derived().data()+diff_sort.size());
            distTh = dist_sort((int)(matchPr*nMatch));
            diffTh = diff_sort((int)((1.0f - matchPr)*nMatch));
            // Filter the bad matches
            for(unsigned int i=0; i<nMatch; i++){
                if ( (dist(i) < distTh ) && ( diff(i) > diffTh ) ){
                    m1 = lmatches[i][0].queryIdx;   m2 = m1;
                    m3 = lmatches[i][0].trainIdx;   m4 = m3;
                    a1 = linesFirst[m1].angle - linesSecond[m2].angle;
                    a2 = linesThird[m3].angle - linesFourth[m4].angle;
                    if( abs(angDiff(a1,a2)) < alphaTh ){
                        sp  << linesFirst[m1].startPointX,  linesFirst[m1].startPointY;
                        ep  << linesFirst[m1].endPointX,    linesFirst[m1].endPointY;
                        mp1 << (sp+ep)/2.f;
                        sp  << linesSecond[m2].startPointX,  linesSecond[m2].startPointY;
                        ep  << linesSecond[m2].endPointX,    linesSecond[m2].endPointY;
                        mp2 << (sp+ep)/2.f;
                        x1  << mp2 - mp1;
                        sp  << linesThird[m3].startPointX,  linesThird[m3].startPointY;
                        ep  << linesThird[m3].endPointX,    linesThird[m3].endPointY;
                        mp1 << (sp+ep)/2.f;
                        sp  << linesFourth[m4].startPointX,  linesFourth[m4].startPointY;
                        ep  << linesFourth[m4].endPointX,    linesFourth[m4].endPointY;
                        mp2 << (sp+ep)/2.f;
                        x2  << mp2 - mp1;
                        if( (x2-x1).norm() < flowTh ){                            
                            // Transform to the new line
                            spoint_L << linesFirst[m1].startPointX , linesFirst[m1].startPointY , 1.f;
                            epoint_L << linesFirst[m1].endPointX   , linesFirst[m1].endPointY   , 1.f;
                            spoint_R << linesSecond[m2].startPointX, linesSecond[m2].startPointY, 1.f;
                            epoint_R << linesSecond[m2].endPointX  , linesSecond[m2].endPointY  , 1.f;
                            line << spoint_R.cross(epoint_R);
                            spoint_R << - (line(2)+line(1)*spoint_L(1) )/line(0) , spoint_L(1) ,  1.f;
                            epoint_R << - (line(2)+line(1)*epoint_L(1) )/line(0) , epoint_L(1) ,  1.f;
                            if( ( (spoint_L(0) - spoint_R(0)) > dispMin ) && ( (epoint_L(0) - epoint_R(0)) > dispMin ) ){
                                // ------- Data formation
                                b_d = baseline / ( f * (cx - spoint_R(0)) + f * (spoint_L(0) - cx) );
                                point3D_s <<  b_d * f * (spoint_L(0) - cx) ,  b_d * f * (spoint_L(1) - cy) , b_d * f * f;
                                b_d = baseline / ( f * (cx - epoint_R(0)) + f * (epoint_L(0) - cx) );
                                point3D_e <<  b_d * f * (epoint_L(0) - cx) ,  b_d * f * (epoint_L(1) - cy) , b_d * f * f;
                                for(unsigned int l = 0; l < 3; l++){
                                    lData_(l,k)   = point3D_s(l);
                                    lData_(l+3,k) = point3D_e(l);
                                }
                                lData_(13,k) = spoint_L(0);
                                lData_(14,k) = spoint_L(1);
                                lData_(15,k) = spoint_R(0);
                                lData_(16,k) = epoint_L(0);
                                lData_(17,k) = epoint_L(1);
                                lData_(18,k) = epoint_R(0);
                                // Second pair
                                spoint_L << linesThird[m3].startPointX , linesThird[m3].startPointY , 1.f;
                                epoint_L << linesThird[m3].endPointX   , linesThird[m3].endPointY ,   1.f;
                                line << spoint_L.cross(epoint_L);
                                lData_(20,k) = line(0);
                                lData_(21,k) = line(1);
                                lData_(22,k) = line(2);
                                line = line / sqrt(line(0)*line(0)+line(1)*line(1));
                                lData_(6,k)  = line(0);
                                lData_(7,k)  = line(1);
                                lData_(8,k)  = line(2);
                                lData_(9,k)  = linesThird[m3].startPointX;
                                lData_(10,k) = linesThird[m3].startPointY;
                                lData_(11,k) = linesThird[m3].endPointX;
                                lData_(12,k) = linesThird[m3].endPointY;
                                lData_(19,k) = linesThird[m3].angle;
                                k++;
                            }
                        }
                    }
                }
            }

        }
    }
    // Form the data
    lData = lData_.block(0,0,Nl,k);
    nLines  = k;
    nLinesH = k;

    n = nLines + nPoints;

}

void plSVO::f2fMutualTracking(){

    /* Points Tracking */
    vector<vector<DMatch>>  matchesL, matchesL_, matchesR;
    int                     m1, m2, m3, m4, k = 0, m1_, m3_;
    MatrixXf                pData_ = MatrixXf::Zero(Np,pointsFirst.size());
    float                   bd;
    Vector3f                pointL, pointR, point3D;
    // Match the stereo points (we discard checking the L-R consistency since we performs a robust optimization and otherwise the number of points decrease critically)
    if( (pointsFirst.size()>0) && (pointsThird.size()>0) ){
        // TODO: MUTUAL BEST MATCHES (IF IT IS FAST...)
        bfm.knnMatch(pdescFirst,pdescThird,matchesL,2);
        bfm.knnMatch(pdescThird,pdescFirst,matchesL_,2);
        // Filter the bad matches
        for(unsigned int i = 0; i<matchesL.size(); i++){
            //if(matchesL[i][0].distance > matchesL[i][1].distance * minDist12 ){             //min distance between the 2NN
                // TODO: ALSO CHECK THE MEDIAN VALUE OF THE DESCRIPTOR TO FILTER THE OUTLIERS
                m1 = matchesL[i][0].queryIdx;   m2 = m1;
                m3 = matchesL[i][0].trainIdx;   m4 = m3;
                for(unsigned int j = 0; j<matchesL_.size();j++){
                    m1_ = matchesL_[j][0].queryIdx;
                    m3_ = matchesL_[j][0].trainIdx;
                    if(m3_ == m1){
                        if(m1_ == m3){
                            j = matchesL_.size();
                            if( (abs(pointsFirst[m1].pt.x-pointsThird[m3].pt.x)<=maxF2FDist) && (abs(pointsSecond[m2].pt.x-pointsFourth[m4].pt.x)<=maxF2FDist) ){      //f2f consistency
                                // ------- Data Formation
                                // Form the 2D point vectors
                                pointL << pointsFirst[m1].pt.x,  pointsFirst[m1].pt.y,  1.f;
                                pointR << pointsSecond[m2].pt.x, pointsSecond[m2].pt.y, 1.f;
                                // Back-projection to 3D points
                                bd = baseline / ( pointL(0) - pointR(0) );
                                point3D <<  bd * (pointL(0) - cx) ,  bd * (pointL(1) - cy) , bd * f ;
                                // Data formation
                                pData_(0,k)  = point3D(0);
                                pData_(1,k)  = point3D(1);
                                pData_(2,k)  = point3D(2);
                                pData_(3,k)  = pointL(0);
                                pData_(4,k)  = pointL(1);
                                pData_(5,k)  = pointR(0);
                                pData_(6,k)  = pointR(1);
                                pData_(7,k)  = pointsThird[m3].pt.x;
                                pData_(8,k)  = pointsThird[m3].pt.y;
                                pData_(9,k)  = pointsFourth[m4].pt.x;
                                pData_(10,k) = pointsFourth[m4].pt.y;
                                // ------------------------
                                // New weights - 05/10/2015
                                // ------------------------
                                pData_(11,k) = sqrt( point3D(0)*point3D(0) + point3D(1)*point3D(1) + point3D(2)*point3D(2) );   // depth
                                pData_(12,k) = matchesL[i][0].distance;
                                pData_(13,k) = matchesL[i][1].distance;
                                pData_(14,k) = abs(pointsFirst[m1].pt.x-pointsThird[m3].pt.x);
                                pData_(15,k) = abs(pointsSecond[m2].pt.x-pointsFourth[m4].pt.x);
                                // ------------------------
                                k++;
                            }
                        }
                    }
                }
            //}
        }
    }
    // Form the data
    pData = pData_.block(0,0,Np,k);
    pData_full = pData;
    nPoints  = k;
    nPointsH = k;
    k = 0;

    /* Line Segments Tracking */
    Ptr<BinaryDescriptorMatcher>    bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    vector<vector<DMatch>>  lmatches, lmatches_;
    vector<KeyLine>         l1_, l2_, l3_, l4_;
    Vector2f                sp, ep, mp1, mp2, x1, x2;
    Vector3f                line, spoint_L, epoint_L, spoint_R, epoint_R, point3D_s, point3D_e;
    float                   a1, a2, b_d, distTh = 0.f, diffTh = 0.f;
    unsigned int            nMatch;
    VectorXf                dist, dist_sort, diff, diff_sort;
    MatrixXf                lData_ = MatrixXf::Zero(Nl,linesFirst.size());
    // Match the f2f points (to reduce time it is only performed with the left images)
    if( (linesFirst.size()>0) && (linesThird.size()>0) ){
        bdm->knnMatch(ldescFirst,ldescThird,lmatches,2);
        bdm->knnMatch(ldescThird,ldescFirst,lmatches_,2);
        if(lmatches.size()>0){
            // Calculate the thresholds with the median values
            nMatch    = lmatches.size();
            dist      = VectorXf::Zero(nMatch);
            diff      = VectorXf::Zero(nMatch);
            dist_sort = VectorXf::Zero(nMatch);
            diff_sort = VectorXf::Zero(nMatch);
            for(int i = 0; i < nMatch; i++){
                dist(i) = lmatches[i][0].distance;
                diff(i) = lmatches[i][1].distance-lmatches[i][0].distance;
            }
            dist_sort = dist; diff_sort = diff;
            sort(dist_sort.derived().data(),dist_sort.derived().data()+dist_sort.size());
            sort(diff_sort.derived().data(),diff_sort.derived().data()+diff_sort.size());
            distTh = dist_sort((int)(matchPr*nMatch));
            diffTh = diff_sort((int)((1.0f - matchPr)*nMatch));
            // Filter the bad matches
            for(unsigned int i=0; i<nMatch; i++){
                if ( (dist(i) < distTh ) && ( diff(i) > diffTh ) ){
                    m1 = lmatches[i][0].queryIdx;   m2 = m1;
                    m3 = lmatches[i][0].trainIdx;   m4 = m3;
                    for(unsigned int j=0; j<lmatches_.size(); j++){
                        m1_ = lmatches_[j][0].queryIdx;
                        m3_ = lmatches_[j][0].trainIdx;
                        if(m3_==m1){
                            if(m1_==m3){
                                j = lmatches_.size();
                                a1 = linesFirst[m1].angle - linesSecond[m2].angle;
                                a2 = linesThird[m3].angle - linesFourth[m4].angle;
                                if( abs(angDiff(a1,a2)) < alphaTh ){
                                    sp  << linesFirst[m1].startPointX,  linesFirst[m1].startPointY;
                                    ep  << linesFirst[m1].endPointX,    linesFirst[m1].endPointY;
                                    mp1 << (sp+ep)/2.f;
                                    sp  << linesSecond[m2].startPointX,  linesSecond[m2].startPointY;
                                    ep  << linesSecond[m2].endPointX,    linesSecond[m2].endPointY;
                                    mp2 << (sp+ep)/2.f;
                                    x1  << mp2 - mp1;
                                    sp  << linesThird[m3].startPointX,  linesThird[m3].startPointY;
                                    ep  << linesThird[m3].endPointX,    linesThird[m3].endPointY;
                                    mp1 << (sp+ep)/2.f;
                                    sp  << linesFourth[m4].startPointX,  linesFourth[m4].startPointY;
                                    ep  << linesFourth[m4].endPointX,    linesFourth[m4].endPointY;
                                    mp2 << (sp+ep)/2.f;
                                    x2  << mp2 - mp1;
                                    if( (x2-x1).norm() < flowTh ){
                                        // Transform to the new line
                                        spoint_L << linesFirst[m1].startPointX , linesFirst[m1].startPointY , 1.f;
                                        epoint_L << linesFirst[m1].endPointX   , linesFirst[m1].endPointY   , 1.f;
                                        spoint_R << linesSecond[m2].startPointX, linesSecond[m2].startPointY, 1.f;
                                        epoint_R << linesSecond[m2].endPointX  , linesSecond[m2].endPointY  , 1.f;
                                        line << spoint_R.cross(epoint_R);
                                        spoint_R << - (line(2)+line(1)*spoint_L(1) )/line(0) , spoint_L(1) ,  1.f;
                                        epoint_R << - (line(2)+line(1)*epoint_L(1) )/line(0) , epoint_L(1) ,  1.f;
                                        if( ( (spoint_L(0) - spoint_R(0)) > dispMin ) && ( (epoint_L(0) - epoint_R(0)) > dispMin ) ){
                                            // ------- Data formation
                                            b_d = baseline / ( f * (cx - spoint_R(0)) + f * (spoint_L(0) - cx) );
                                            point3D_s <<  b_d * f * (spoint_L(0) - cx) ,  b_d * f * (spoint_L(1) - cy) , b_d * f * f;
                                            b_d = baseline / ( f * (cx - epoint_R(0)) + f * (epoint_L(0) - cx) );
                                            point3D_e <<  b_d * f * (epoint_L(0) - cx) ,  b_d * f * (epoint_L(1) - cy) , b_d * f * f;
                                            for(unsigned int l = 0; l < 3; l++){
                                                lData_(l,k)   = point3D_s(l);
                                                lData_(l+3,k) = point3D_e(l);
                                            }
                                            lData_(13,k) = spoint_L(0);
                                            lData_(14,k) = spoint_L(1);
                                            lData_(15,k) = spoint_R(0);
                                            lData_(16,k) = epoint_L(0);
                                            lData_(17,k) = epoint_L(1);
                                            lData_(18,k) = epoint_R(0);
                                            // Second pair
                                            spoint_L << linesThird[m3].startPointX , linesThird[m3].startPointY , 1.f;
                                            epoint_L << linesThird[m3].endPointX   , linesThird[m3].endPointY ,   1.f;
                                            line << spoint_L.cross(epoint_L);
                                            lData_(20,k) = line(0);
                                            lData_(21,k) = line(1);
                                            lData_(22,k) = line(2);
                                            line = line / sqrt(line(0)*line(0)+line(1)*line(1));
                                            lData_(6,k)  = line(0);
                                            lData_(7,k)  = line(1);
                                            lData_(8,k)  = line(2);
                                            lData_(9,k)  = linesThird[m3].startPointX;
                                            lData_(10,k) = linesThird[m3].startPointY;
                                            lData_(11,k) = linesThird[m3].endPointX;
                                            lData_(12,k) = linesThird[m3].endPointY;
                                            lData_(19,k) = linesThird[m3].angle;
                                            // ------------------------
                                            // New weights - 05/10/2015
                                            // ------------------------
                                            lData_(23,k) = max( point3D_s.norm(), point3D_e.norm() );   // depth
                                            // ------------------------
                                            k++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Form the data
    lData = lData_.block(0,0,Nl,k);
    lData_full = lData;
    nLines  = k;
    nLinesH = k;

    n = nLines + nPoints;

}

/* Optimization functions */

void plSVO::svoOptim(Matrix4f &pose_){
    if(erode)
        pose_ = erodeOptimizationGN();
    else
        pose_ = optimizeGN();
}

MatrixXf plSVO::optimizeGN(){
    Matrix4f x;
    MatrixXf J(2*n,6);
    VectorXf E(2*n);
    epsInc = VectorXf::Zero(6);
    //Initial value in SE(3) space
    prevParam = x_0;
    param     = x_0;
    prevErrNorm = 99999999.9f;
    if ( n >= Nmin ){
        for(iters = 0; iters < maxIters; iters++){
            x = param;
            T = param;
            optimFunctions(JtJ,JtErr,errNorm,E);
            // If the difference is very small stop
            if( ( abs(errNorm-prevErrNorm) < minErrorChange ) || ( errNorm < minError) )
                break;
            // Update step
            LDLT<MatrixXf> solver(JtJ);
            epsInc = solver.solve(JtErr);   //epsInc << JtJ.inverse() * JtErr;
            param  << prevParam * transformation_expmap(epsInc).inverse();
            // If the parameter change is small stop, else if the parameter is big update weights
            if( diffManifoldError(param,prevParam) < minChange )
                break;
            // Update values
            prevErrNorm = errNorm;
            prevParam   = param;
        }
        cov = JtJ.inverse();
        return param;
    }
    else{
        cov = MatrixXf::Zero(6,6);
        return Matrix4f::Identity();
    }
}

MatrixXf plSVO::erodeOptimizationGN(){

    unsigned int maxIters_ = maxIters;
    bool previous_uncertainty;
    VectorXf residue_;
    MatrixXf x_h, x_f;
    maxIters = maxItersH;

    previous_uncertainty = uncertainty; // 10sept
    uncertainty = false;                // 10sept
    x_h = optimizeRobustGN(residue_);

    if( is_finite(x_h) && (n>=Nmin) ){
        removeERODE(residue_);
        maxIters   = maxIters_;
        //setInitialValue(x_h.inverse());
        uncertainty = previous_uncertainty; // 10sept
        x_f = optimizeRobustGN(residue_);
    }
    else{
        maxIters   = maxIters_;
        setInitialValue(x_0);
        uncertainty = previous_uncertainty; // 10sept
        x_f = optimizeGN();
    }

    return x_f;
}

MatrixXf plSVO::optimizeRobustGN(VectorXf &residue_){
    Matrix4f x;
    MatrixXf J(2*n,6);
    VectorXf E(n);
    epsInc = VectorXf::Zero(6);
    //Initial value in SE(3) space
    prevParam = x_0;
    param     = x_0;
    if ( n >= Nmin ){  // Check
        for(iters = 0; iters < maxIters; iters++){
            x = param;
            T = param;
            optimFunctionsH(JtJ,JtErr,errNorm,E);
            // If the difference is very small stop
            if( ( abs(errNorm-prevErrNorm) < minErrorChange ) || ( errNorm < minError) )
                break;
            // Update step
            LDLT<MatrixXf> solver(JtJ);
            epsInc = solver.solve(JtErr);   //epsInc << JtJ.inverse() * JtErr;
            //param  << transformation_expmap(epsInc).inverse() * prevParam;
            param  << prevParam * transformation_expmap(epsInc).inverse();
            //param  << prevParam * inverse_transformation( transformation_expmap(epsInc) );
            // If the parameter change is small stop, else if the parameter is big update weights
            if( diffManifoldError(param,prevParam) < minChange )
                break;
            // Update values
            prevErrNorm = errNorm;
            prevParam   = param;
        }
        cov = JtJ.inverse();
        residue_ = E;
        //return param.inverse();
        return param;
    }
    else{
        cov = MatrixXf::Zero(6,6);
        residue_ = VectorXf::Zero(n);
        return Matrix4f::Identity();
    }
}

void plSVO::huberFunction(float &s, float &eNorm, float &dH ){

    // Pseudo-Huber Loss Function
    float sb = s / b2;
    float sq = sqrt(1.f+sb);
    dH = 1.f / sq;
    s  = 2.f * b2 * (sq-1.f);
    eNorm += s ;

}

void plSVO::removeERODE(VectorXf residue_){

    unsigned int k = 0;
    float        huberTh_, huberPTh_, huberLTh_, q1, q3;
    VectorXf     residue_sort(n);
    inliers.clear();

    residue_sort = residue_;
    sort(residue_sort.derived().data(),residue_sort.derived().data()+residue_sort.size());
    huberLTh_ = huberKL * residue_sort((int)(huberTh*n));
    huberPTh_ = huberKP * residue_sort((int)(huberTh*n));

    // Points Features
    if(nPoints>0){
        MatrixXf pDataAux = MatrixXf::Zero(Np,nPoints);
        for(unsigned int i = 0; i < nPoints; i++){
            if( (residue_(i) < huberPTh_) || (residue_(i) == 0) ) {
//                // Epipolar condition to remove F2F (L and R) inconsistent matches
//                Matrix3f R, skew_tran, F;
//                Vector3f t, x1, x2, x3, x4;
//                R << param.block(0,0,3,3);
//                t << param(0,3), param(1,3), param(2,3);
//                skew_tran = skew(t);
//                F = skew_tran * R;
//                F = R * skew_tran;
//                x1 << pData(3,i),  pData(4,i),  1.f;
//                x2 << pData(5,i),  pData(6,i),  1.f;
//                x3 << pData(7,i),  pData(8,i),  1.f;
//                x4 << pData(9,i),  pData(10,i), 1.f;
//                float epip1 = abs( x3.transpose() * F * x1 );
//                float epip2 = abs( x4.transpose() * F * x2 );
//                cout << endl << x3.transpose() << " " << x1.transpose() << endl << endl << F << endl;
//                if( ( epip1 < maxStDist ) && ( epip2 < maxStDist ) ){
                    inliers.push_back(true);
                    for(unsigned int j = 0; j < Np; j++){
                        pDataAux(j,k) = pData(j,i);
                    }
                    k++;
//                }
//                else
//                    inliers.push_back(false);
            }
            else
                inliers.push_back(false);
        }
        pData_full = pData;
        pData     = pDataAux.block(0,0,Np,k);
        nPointsH  = k;
        k = 0;
    }

    // Line Features
    if(nLines>0){
        MatrixXf lDataAux = MatrixXf::Zero(Nl,nLines);
        for(unsigned int i = nPoints; i < nLines+nPoints; i++){
            if( (residue_(i) < huberLTh_) || (residue_(i) == 0) ){
                inliers.push_back(true);
                for(unsigned int j = 0; j < Nl; j++){
                    lDataAux(j,k) = lData(j,i-nPoints);
                }
                k++;
            }
            else
                inliers.push_back(false);
        }
        lData_full = lData;
        lData     = lDataAux.block(0,0,Nl,k);
        nLinesH   = k;
    }

    n = nPointsH + nLinesH;

}

void plSVO::optimFunctions(MatrixXf &JtJ_, VectorXf &JtE_, float &errNorm_, VectorXf &E){

    E    = VectorXf::Zero(2*n);
    MatrixXf J    = MatrixXf::Zero(2*n,6);
    JtJ_ = MatrixXf::Zero(6,6);
    JtE_ = VectorXf::Zero(6);
    MatrixXf JtW(6,2*n), R(3,3);
    errNorm_ = 0.f;

    // ------ Point Features
    Vector3f p, ph;
    Vector4f P, TP;
    float    dx, dy, gx, gy, gz, gz2, fgz2, s, dH;
    MatrixXf Jhg(2,3), covp(2,2), covp_inv(2,2);
    R << param.block(0,0,3,3);
    for(unsigned int i = 0; i < nPointsH; i++){
        // Project the 3D point to the Left image in the second frame
        P  << pData(0,i), pData(1,i), pData(2,i), 1.f;
        TP = param * P;
        p << K_amp * TP;
        if( abs(p(2)) > homogTh ){
            ph << p / p(2);
            gx = TP(0);
            gy = TP(1);
            gz = TP(2);
            gz2 = gz*gz;
            fgz2 = -f/gz2;
            // Error function
            dx  = pData(7,i)-ph(0);  E(2*i)   = -dx;
            dy  = pData(8,i)-ph(1);  E(2*i+1) = -dy;
            // Uncertainty function
            if(uncertainty){
                float px_hat = pData(3,i) - cx;
                float py_hat = pData(4,i) - cy;
                float disp   = pData(3,i) - pData(5,i);
                float disp2  = disp * disp;
                Matrix3f covP_an;
                covP_an(0,0) = disp2+2.f*px_hat*px_hat;
                covP_an(0,1) = 2.f*px_hat*py_hat;
                covP_an(0,2) = 2.f*f*px_hat;
                covP_an(1,1) = disp2+2.f*py_hat*py_hat;
                covP_an(1,2) = 2.f*f*py_hat;
                covP_an(2,2) = 2.f*f*f;
                covP_an(1,0) = covP_an(0,1);
                covP_an(2,0) = covP_an(0,2);
                covP_an(2,1) = covP_an(1,2);
                covP_an << covP_an / (disp2*disp2);     // Covariance of the 3D point P up to b2*sigma2
                Jhg  << gz, 0.f, -gx, 0.f, gz, -gy;
                Jhg  << Jhg * R;
                covp << Jhg * covP_an * Jhg.transpose();
                covp << covp / (gz2*gz2);               // Covariance of the 3D projection \hat{p} up to f2*b2*sigma2
                covp = bsigmaP * covp;
                covp(0,0) = covp(0,0) + sigmaP2;
                covp(1,1) = covp(1,1) + sigmaP2;
                covp_inv = covp.inverse();
            }
            else
                covp_inv = Matrix2f::Identity();
            // Error norm
            errNorm_ += E(2*i) * covp_inv(0,0) * E(2*i) + E(2*i+1) * covp_inv(1,1) * E(2*i+1) + E(2*i) * E(2*i+1) * (covp_inv(0,1)+covp_inv(1,0));
            // Jacobian function
            MatrixXf Jt(6,2), J_(2,6);
            J_(0,0) = - fgz2 * gz;
            J_(0,1) =   0.f;
            J_(0,2) =   fgz2 * gx;
            J_(0,3) =   fgz2 * gx*gy;
            J_(0,4) = - fgz2 * (gx*gx+gz2);
            J_(0,5) =   fgz2 * gy*gz;
            J_(1,0) =   0.f;
            J_(1,1) = - fgz2 * gz;
            J_(1,2) =   fgz2 * gy;
            J_(1,3) =   fgz2 * (gz2+gy*gy);
            J_(1,4) = - fgz2 * gx*gy;
            J_(1,5) = - fgz2 * gx*gz;
            Jt = J_.transpose();
            Jt = Jt * covp_inv;
            for(int k = 0; k < 2; k++){
                for(int j = 0; j < 6; j++){
                    JtW(j,2*i+k) = Jt(j,k);
                    J(2*i+k,j)   = J_(k,j);
                }
            }
        }
    }

    // ------ Line Features
    Vector2f sl, el, sp, ep, l, v, ph_, qh_;
    Vector3f L, Ln, q, qh, r1, r2, r3, t1, v_aux;
    Vector4f Q, TQ;
    Matrix2f u, v_, rot;
    Matrix3f sr1, sr2, sr3, st1, m_aux;
    float    d_p, d_q, lxx, lxy, lyy, lxy2, seno, coseno, lnorm, sln, eln, spn, epn, length, overlap, cov_p, cov_q;
    r1 << param(0,0), param(1,0), param(2,0);   sr1 = skew(r1);
    r2 << param(0,1), param(1,1), param(2,1);   sr2 = skew(r2);
    r3 << param(0,2), param(1,2), param(2,2);   sr3 = skew(r3);
    t1 << param(0,3), param(1,3), param(2,3);   st1 = skew(t1);
    unsigned int j;
    for(unsigned int i = 0; i < nLinesH; i++){
        j = i + nPointsH;
        P  << lData(0,i), lData(1,i), lData(2,i), 1.f;
        Q  << lData(3,i), lData(4,i), lData(5,i), 1.f;
        Ln << lData(6,i), lData(7,i), lData(8,i);
        TP = param * P;
        TQ = param * Q;
        p << K_amp * TP;
        q << K_amp * TQ;
        if((abs(p(2))>homogTh)&&(abs(q(2))>homogTh)){
            ph << p / p(2);
            qh << q / q(2);
            // Overlap
            if(overlap){
                sl << lData(9,i),  lData(10,i);
                el << lData(11,i), lData(12,i);
                l << sl - el;
                lxx  = l(0)*l(0);
                lyy  = l(1)*l(1);
                lxy  = l(0)*l(1);
                lxy2 = 1.f/(lxx+lyy);
                u  << lxx, lxy, lxy, lyy;   u << u * lxy2;
                v_ << lyy, -lxy, -lxy, lxx; v_ << v_ * lxy2;
                v << v_ * sl;
                ph_ << ph(0), ph(1);
                qh_ << qh(0), qh(1);
                sp << u * ph_ + v;
                ep << u * qh_ + v;
                lnorm  = 1.f / l.norm();
                seno   = -l(0)*lnorm;
                coseno = -l(1)*lnorm;
                rot    << coseno, -seno, seno, coseno;
                sl     << rot * sl;
                el     << rot * el;
                sp     << rot * sp;
                ep     << rot * ep;
                sln    = min(sl(1), el(1));
                eln    = max(sl(1), el(1));
                spn    = min(sp(1), ep(1));
                epn    = max(sp(1), ep(1));
                length = eln-spn;
                if ( (epn < sln) || (spn > eln) )
                    overlap = 0.f;
                else{
                    if ( (epn>eln) && (spn<sln) )
                        overlap = eln-sln;
                    else
                        overlap = min(eln,epn) - max(sln,spn);
                }
                if(length>0.01f){
                    overlap = overlap / length;
                    overlap = sqrt(overlap);
                }
                else
                    overlap = 0.f;
            }
            else
                overlap = 1.f;
            // Uncertainty function
            if(uncertainty){
                float px_hat = lData(13,i) - cx;
                float py_hat = lData(14,i) - cy;
                float disp   = lData(13,i) - lData(15,i);
                float disp2  = disp * disp;
                Matrix3f covP_an;
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
                MatrixXf J_ep(1,3);
                float lxpz = Ln(0)*p(2);
                float lypz = Ln(1)*p(2);
                J_ep << lxpz*f, lypz*f, lxpz*cx+lypz*cy-Ln(0)*p(0)-Ln(1)*p(1);
                J_ep << J_ep * R;
                float p4 = pow(p(2),4);
                VectorXf cov_aux(1);
                cov_aux << J_ep * covP_an * J_ep.transpose();
                cov_p = cov_aux(0);
                cov_p = 1.f/cov_p;
                cov_p = p4 * cov_p * 0.5f * bsigmaL_inv;
                px_hat = lData(16,i) - cx;
                py_hat = lData(17,i) - cy;
                disp   = lData(16,i) - lData(18,i);
                disp2  = disp * disp;
                Matrix3f covQ_an;
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
                lxpz = Ln(0)*q(2);
                lypz = Ln(1)*q(2);
                J_ep << lxpz*f, lypz*f, lxpz*cx+lypz*cy-Ln(0)*q(0)-Ln(1)*q(1);
                J_ep << J_ep * R;
                float q4 = pow(q(2),4);
                cov_aux << J_ep * covQ_an * J_ep.transpose();
                cov_q = cov_aux(0);
                cov_q = 1.f/cov_q;
                cov_q = q4 * cov_q * 0.5f * bsigmaL_inv;
                if( isinf(cov_p) || isnan(cov_p) )  cov_p = 0.f;
                if( isinf(cov_q) || isnan(cov_q) )  cov_q = 0.f;
            }
            else{
                cov_p = 1.f;
                cov_q = 1.f;
            }
            // Error function
            d_p  = Ln(0) * ph(0) + Ln(1) * ph(1) + Ln(2) ;  E(2*j)   = overlap * d_p;
            d_q  = Ln(0) * qh(0) + Ln(1) * qh(1) + Ln(2) ;  E(2*j+1) = overlap * d_q;
            errNorm_ += E(2*j)*cov_p*E(2*j) + E(2*j+1)*cov_q*E(2*j+1);
            // Jacobian function
            m_aux << TP(0) * sr1 + TP(1) * sr2 + TP(2) * sr3 + st1 ;
            v_aux << Ln(0) * f, Ln(1) * f, Ln(0) * cx + Ln(1) * cy + Ln(2) - d_p;
            v_aux << overlap * v_aux / p(2);
            for(unsigned int k = 0; k < 3; k++){
                J(2*j,k)   = v_aux(k);
                JtW(k,2*j) = v_aux(k) * cov_p;
            }
            v_aux << -(v_aux.transpose() * m_aux).transpose();
            for(unsigned int k = 0; k < 3; k++){
                J(2*j,k+3)   = v_aux(k);
                JtW(k+3,2*j) = v_aux(k) * cov_p;
            }
            m_aux << TQ(0) * sr1 + TQ(1) * sr2 + TQ(2) * sr3 + st1 ;
            v_aux << Ln(0) * f, Ln(1) * f, Ln(0) * cx + Ln(1) * cy + Ln(2) - d_q;
            v_aux << overlap * v_aux / q(2);
            for(unsigned int k = 0; k < 3; k++){
                J(2*j+1,k)   = v_aux(k);
                JtW(k,2*j+1) = v_aux(k) * cov_q;
            }
            v_aux << -(v_aux.transpose() * m_aux).transpose();
            for(unsigned int k = 0; k < 3; k++){
                J(2*j+1,k+3)   = v_aux(k);
                JtW(k+3,2*j+1) = v_aux(k) * cov_q;
            }
        }
    }

    // ------ Final Functions
    JtJ_ << JtW * J;
    JtE_ << JtW * E;
    errNorm_ = errNorm_ * 0.5f / (float)n;

}

void plSVO::optimFunctionsH(MatrixXf &JtJ_, VectorXf &JtE_, float &errNorm_, VectorXf &E){

    E    = VectorXf::Zero(2*n);
    VectorXf Eh(n);
    MatrixXf J    = MatrixXf::Zero(2*n,6);
    JtJ_ = MatrixXf::Zero(6,6);
    JtE_ = VectorXf::Zero(6);
    errNorm_ = 0.f;
    MatrixXf JtW(6,2*n), R(3,3);

    // ------ Point Features
    Vector3f p, ph;
    Vector4f P, TP;
    float    dx, dy, gx, gy, gz, gz2, fgz2, s, dH;
    MatrixXf Jhg(2,3), covp(2,2), covp_inv(2,2);
    R << param.block(0,0,3,3);
    for(unsigned int i = 0; i < nPointsH; i++){
        // Project the 3D point to the Left image in the second frame
        P  << pData(0,i), pData(1,i), pData(2,i), 1.f;
        TP = param * P;
        p << K_amp * TP;
        if( abs(p(2)) > homogTh ){
            ph << p / p(2);
            gx = TP(0);
            gy = TP(1);
            gz = TP(2);
            gz2 = gz*gz;
            fgz2 = f/gz2;
            // Error function
            dx  = pData(7,i)-ph(0);
            dy  = pData(8,i)-ph(1);
            // Uncertainty function
            if(uncertainty){
                float px_hat = pData(3,i) - cx;
                float py_hat = pData(4,i) - cy;
                float disp   = pData(3,i) - pData(5,i);
                float disp2  = disp * disp;
                Matrix3f covP_an;
                covP_an(0,0) = disp2+2.f*px_hat*px_hat;
                covP_an(0,1) = 2.f*px_hat*py_hat;
                covP_an(0,2) = 2.f*f*px_hat;
                covP_an(1,1) = disp2+2.f*py_hat*py_hat;
                covP_an(1,2) = 2.f*f*py_hat;
                covP_an(2,2) = 2.f*f*f;
                covP_an(1,0) = covP_an(0,1);
                covP_an(2,0) = covP_an(0,2);
                covP_an(2,1) = covP_an(1,2);
                covP_an << covP_an / (disp2*disp2);     // Covariance of the 3D point P up to b2*sigma2
                Jhg  << gz, 0.f, -gx, 0.f, gz, -gy;
                Jhg  << Jhg * R;
                covp << Jhg * covP_an * Jhg.transpose();
                covp << covp / (gz2*gz2);               // Covariance of the 3D projection \hat{p} up to f2*b2*sigma2
                covp = bsigmaP * covp;
                covp(0,0) = covp(0,0) + sigmaP2;
                covp(1,1) = covp(1,1) + sigmaP2;
                covp_inv = covp.inverse();
            }
            else
                covp_inv = Matrix2f::Identity();
            // Classic Huber Loss Function
            s = dx*dx*covp_inv(0,0)+dx*dy*(covp_inv(0,1)+covp_inv(1,0))+dy*dy*covp_inv(1,1);
            huberFunction(s,errNorm,dH);
            Eh(i) = s;
            E(2*i)   = dH * dx;
            E(2*i+1) = dH * dy;
            // Jacobian function
            MatrixXf Jt(6,2), J_(2,6);
            Jt(0,0) = - fgz2 * gz;
            Jt(1,0) =   0.f;
            Jt(2,0) =   fgz2 * gx;
            Jt(3,0) =   fgz2 * gx*gy;
            Jt(4,0) = - fgz2 * (gx*gx+gz2);
            Jt(5,0) =   fgz2 * gy*gz;
            Jt(0,1) =   0.f;
            Jt(1,1) = - fgz2 * gz;
            Jt(2,1) =   fgz2 * gy;
            Jt(3,1) =   fgz2 * (gz2+gy*gy);
            Jt(4,1) = - fgz2 * gx*gy;
            Jt(5,1) = - fgz2 * gx*gz;
            J_ = Jt.transpose();
            Jt = Jt * covp_inv;
            for(int k = 0; k < 2; k++){
                for(int j = 0; j < 6; j++){
                    JtW(j,2*i+k) = Jt(j,k);
                    J(2*i+k,j)   = J_(k,j);
                }
            }
        }
        else
            Eh(i) = 9999.f;
    }

    // ------ Line Features
    Vector2f sl, el, sp, ep, l, v, ph_, qh_;
    Vector3f L, Ln, q, qh, r1, r2, r3, t1, v_aux;
    Vector4f Q, TQ;
    Matrix2f u, v_, rot;
    Matrix3f sr1, sr2, sr3, st1, m_aux;
    float    d_p, d_q, lxx, lxy, lyy, lxy2, seno, coseno, lnorm, sln, eln, spn, epn, length, overlap, cov_p, cov_q;
    r1 << param(0,0), param(1,0), param(2,0);   sr1 = skew(r1);
    r2 << param(0,1), param(1,1), param(2,1);   sr2 = skew(r2);
    r3 << param(0,2), param(1,2), param(2,2);   sr3 = skew(r3);
    t1 << param(0,3), param(1,3), param(2,3);   st1 = skew(t1);
    unsigned int j;
    for(unsigned int i = 0; i < nLinesH; i++){
        j = i + nPointsH;
        P  << lData(0,i), lData(1,i), lData(2,i), 1.f;
        Q  << lData(3,i), lData(4,i), lData(5,i), 1.f;
        Ln << lData(6,i), lData(7,i), lData(8,i);
        TP = param * P;
        TQ = param * Q;
        p << K_amp * TP;
        q << K_amp * TQ;
        if((abs(p(2))>homogTh)&&(abs(q(2))>homogTh)){
            ph << p / p(2);
            qh << q / q(2);
            // Error function
            d_p  = Ln(0) * ph(0) + Ln(1) * ph(1) + Ln(2) ;
            d_q  = Ln(0) * qh(0) + Ln(1) * qh(1) + Ln(2) ;
            // Uncertainty function
            if(uncertainty){
                float px_hat = lData(13,i) - cx;
                float py_hat = lData(14,i) - cy;
                float disp   = lData(13,i) - lData(15,i);
                float disp2  = disp * disp;
                Matrix3f covP_an;
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
                MatrixXf J_ep(1,3);
                float lxpz = Ln(0)*p(2);
                float lypz = Ln(1)*p(2);
                J_ep << lxpz*f, lypz*f, lxpz*cx+lypz*cy-Ln(0)*p(0)-Ln(1)*p(1);
                J_ep << J_ep * R;
                float p4 = pow(p(2),4);
                VectorXf cov_aux(1);
                cov_aux << J_ep * covP_an * J_ep.transpose();
                cov_p = cov_aux(0);
                cov_p = 1.f/cov_p;
                cov_p = p4 * cov_p * 0.5f * bsigmaL_inv;
                px_hat = lData(16,i) - cx;
                py_hat = lData(17,i) - cy;
                disp   = lData(16,i) - lData(18,i);
                disp2  = disp * disp;
                Matrix3f covQ_an;
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
                lxpz = Ln(0)*q(2);
                lypz = Ln(1)*q(2);
                J_ep << lxpz*f, lypz*f, lxpz*cx+lypz*cy-Ln(0)*q(0)-Ln(1)*q(1);
                J_ep << J_ep * R;
                float q4 = pow(q(2),4);
                cov_aux << J_ep * covQ_an * J_ep.transpose();
                cov_q = cov_aux(0);
                cov_q = 1.f/cov_q;
                cov_q = q4 * cov_q * 0.5f * bsigmaL_inv;
                if( isinf(cov_p) || isnan(cov_p) )  cov_p = 0.f;
                if( isinf(cov_q) || isnan(cov_q) )  cov_q = 0.f;
            }
            else{
                cov_p = 1.f;
                cov_q = 1.f;
            }
            // Overlap
            if(overlap){
                sl << lData(9,i),  lData(10,i);
                el << lData(11,i), lData(12,i);
                l << sl - el;
                lxx  = l(0)*l(0);
                lyy  = l(1)*l(1);
                lxy  = l(0)*l(1);
                lxy2 = 1/(lxx+lyy);
                u  << lxx, lxy, lxy, lyy;   u << u * lxy2;
                v_ << lyy, -lxy, -lxy, lxx; v_ << v_ * lxy2;
                v << v_ * sl;
                ph_ << ph(0), ph(1);
                qh_ << qh(0), qh(1);
                sp << u * ph_ + v;
                ep << u * qh_ + v;
                lnorm  = 1 / l.norm();
                seno   = -l(0)*lnorm;
                coseno = -l(1)*lnorm;
                rot    << coseno, -seno, seno, coseno;
                sl     << rot * sl;
                el     << rot * el;
                sp     << rot * sp;
                ep     << rot * ep;
                sln    = min(sl(1), el(1));
                eln    = max(sl(1), el(1));
                spn    = min(sp(1), ep(1));
                epn    = max(sp(1), ep(1));
                length = eln-spn;
                if ( (epn < sln) || (spn > eln) )
                    overlap = 0.f;
                else{
                    if ( (epn>eln) && (spn<sln) )
                        overlap = eln-sln;
                    else
                        overlap = min(eln,epn) - max(sln,spn);
                }
                if(length>0.01f){
                    overlap = overlap / length;
                    overlap = sqrt(overlap);
                }
                else
                    overlap = 0.f;
            }
            else
                overlap = 1.f;
            // Clasic Huber Loss Function
            float d_p_ = overlap * d_p;
            float d_q_ = overlap * d_q;
            s = d_p_*d_p_*cov_p + d_q_*d_q_*cov_q;
            huberFunction(s,errNorm,dH);
            Eh(j) = s;
            E(2*j)   = dH * d_p_;
            E(2*j+1) = dH * d_q_;
            // Jacobian function
            m_aux << TP(0) * sr1 + TP(1) * sr2 + TP(2) * sr3 + st1 ;
            v_aux << Ln(0) * f, Ln(1) * f, Ln(0) * cx + Ln(1) * cy + Ln(2) - d_p;
            v_aux << overlap * v_aux / p(2);
            for(unsigned int k = 0; k < 3; k++){
                J(2*j,k)   = v_aux(k);
                JtW(k,2*j) = v_aux(k) * cov_p;
            }
            v_aux << -(v_aux.transpose() * m_aux).transpose();
            for(unsigned int k = 0; k < 3; k++){
                J(2*j,k+3)   = v_aux(k);
                JtW(k+3,2*j) = v_aux(k) * cov_p;
            }
            m_aux << TQ(0) * sr1 + TQ(1) * sr2 + TQ(2) * sr3 + st1 ;
            v_aux << Ln(0) * f, Ln(1) * f, Ln(0) * cx + Ln(1) * cy + Ln(2) - d_q;
            v_aux << overlap * v_aux / q(2);
            for(unsigned int k = 0; k < 3; k++){
                J(2*j+1,k)   = v_aux(k);
                JtW(k,2*j+1) = v_aux(k) * cov_q;
            }
            v_aux << -(v_aux.transpose() * m_aux).transpose();
            for(unsigned int k = 0; k < 3; k++){
                J(2*j+1,k+3)   = v_aux(k);
                JtW(k+3,2*j+1) = v_aux(k) * cov_q;
            }
        }
        else
            Eh(j) = 9999.f;
    }

    // ------ Final Functions
    JtJ_ << JtW * J;
    JtE_ << JtW * E;
    errNorm_ = errNorm_ * 0.5f / (float)n;

    E = VectorXf::Zero(n);
    E = Eh;

    if(iters==0){
        ofstream fileL("./results/residues/"+to_string(nFrame)+"L.txt");
        ofstream fileP("./results/residues/"+to_string(nFrame)+"P.txt");
        for(unsigned int i = 0; i < nPointsH; i++){
            fileP << E(i) << endl;
        }
        fileP.close();
        for(unsigned int i = 0; i < nLinesH; i++){
            fileL << E(i+nPointsH) << endl;
        }
        fileL.close();
        nFrame++;
    }

}

/* New optimization functions */

void plSVO::svoInliers(Matrix4f pose_inc, string &inliers_ratio){

    int n_inliers = inliers.size();
    VectorXf residuals_sort, residuals_aux, residuals_ = VectorXf::Zero(n_inliers);

    // ------ Point Features
    Vector3f p, ph;
    Vector4f P, TP;
    float    dx, dy, rP;
    for(unsigned int i = 0; i < nPoints; i++){
        // Project the 3D point to the Left image in the second frame
        P  << pData_full(0,i), pData_full(1,i), pData_full(2,i), 1.f;
        TP = pose_inc * P;
        p << K_amp * TP;
        if( abs(p(2)) > homogTh ){
            ph << p / p(2);
            dx  = ph(0)-pData_full(7,i);
            dy  = ph(1)-pData_full(8,i);
            rP  = sqrt(dx*dx+dy*dy);
            residuals_(i) = rP;
        }
        else{
            residuals_(i) = 9999.f;
        }
    }

    // ------ Line Features
    Vector3f Ln, q, qh;
    Vector4f Q, TQ;
    float    d_p, d_q, rL;
    unsigned int j = 0;
    for(unsigned int i = 0; i < nLines; i++){
        j = i + nPoints;
        P  << lData_full(0,i), lData_full(1,i), lData_full(2,i), 1.f;
        Q  << lData_full(3,i), lData_full(4,i), lData_full(5,i), 1.f;
        Ln << lData_full(6,i), lData_full(7,i), lData_full(8,i);
        TP = pose_inc * P;
        TQ = pose_inc * Q;
        p << K_amp * TP;
        q << K_amp * TQ;
        if((abs(p(2))>homogTh)&&(abs(q(2))>homogTh)){
            ph << p / p(2);
            qh << q / q(2);
            // Error function
            d_p  = Ln(0) * ph(0) + Ln(1) * ph(1) + Ln(2) ;
            d_q  = Ln(0) * qh(0) + Ln(1) * qh(1) + Ln(2) ;
            rL  = sqrt(d_p*d_p+d_q*d_q);
            residuals_(j) = rL;
        }
        else{
            residuals_(j) = 9999.f;
        }
    }

    // ------ Inliers ratio
    residuals_sort = residuals_;
    residuals_aux  = residuals_;
    sort(residuals_sort.derived().data(),residuals_sort.derived().data()+residuals_sort.size());
    float median = residuals_sort( (int)(0.5f*n_inliers) );
    residuals_aux << (residuals_aux - VectorXf::Constant(n_inliers,median)).cwiseAbs();
    sort(residuals_aux.derived().data(),residuals_aux.derived().data()+residuals_aux.size());
    float MAD = residuals_aux( (int)(0.5f*n_inliers) );
    float stdv = 1.4826f * MAD;

    int fp = 0, tp = 0, fn = 0, tn = 0;
    for(unsigned int i = 0; i < n_inliers; i++){
        if( inliers[i] ){
            if( residuals_(i)>3.f*stdv )
                fn++;
            else
                tp++;
        }
        else{
            if( residuals_(i)>3.f*stdv )
                tn++;
            else
                fp++;
        }
    }
    inliers_ratio = to_string(tp) + " " + to_string(fp) + " " + to_string(tn) + " " + to_string(fn);
    //cout << endl << residuals_sort << endl << endl;

}

void plSVO::svoOptim_new_(Matrix4f &x_f){

    unsigned int maxIters_ = maxIters, optimMode_ = optimMode;
    MatrixXf x_ini_ = x_0;
    MatrixXf x_h;
    VectorXf residue_;

    maxIters = maxItersH;
    x_h = optimizeGN_new(residue_);
    if( is_finite(x_h) && (n>=Nmin) ){
        removeERODE_new(residue_);
        maxIters   = maxIters_;
        //setInitialValue(x_h);
        x_f = optimizeGN_new(residue_);
    }
    else{
        optimMode = 0;
        x_h = optimizeRobustGN_new(residue_);
        if( is_finite(x_h) && (n>=Nmin) ){
            removeERODE_new(residue_);
            maxIters   = maxIters_;
            //setInitialValue(x_h);
            x_f = optimizeRobustGN_new(residue_);
        }
        else{
            inliers = vector<bool>(n,false);
            pData_full = pData;
            lData_full = lData;
            x_f        = Matrix4f::Identity();
            cov        = MatrixXf::Zero(6,6);
        }
    }

}


void plSVO::svoOptim_new(Matrix4f &pose_){

    if(erode){
        // Change the maxIters value (non-necessary if both are equal)
        unsigned int maxIters_  = maxIters;
        VectorXf residue_;
        MatrixXf x_h, x_f;
        maxIters = maxItersH;
        x_h = optimizeGN_new(residue_);
        // If we achieve a proper solution, we remove the outliers and refine the pose estimation
        if( is_finite(x_h) && (n>=Nmin) ){
            removeERODE_new(residue_);
            maxIters   = maxIters_;
            setInitialValue(x_h);
            pose_ = optimizeGN_new(residue_);
        }
        // If not, we optimize without robust function
        else{
            maxIters   = maxIters_;
            setInitialValue( pose_ );
            pose_ = optimizeGN_new(residue_);
            if( !is_finite(x_h) || (n<Nmin) ){
                inliers = vector<bool>(n,false);
                pData_full = pData;
                lData_full = lData;
                pose_ = Matrix4f::Identity();
                cov   = MatrixXf::Zero(6,6);
            }
        }
    }
    else{
        VectorXf residue_;
        pose_ = optimizeGN_new(residue_);
        inliers = vector<bool>(n,true);
        pData_full = pData;
        lData_full = lData;
    }

}

MatrixXf plSVO::optimizeGN_new(VectorXf &residue_){
    Matrix4f x;
    MatrixXf J(n,6);
    VectorXf residue(n);
    epsInc = VectorXf::Zero(6);
    //Initial value in SE(3) space
    prevParam   = x_0;
    param       = x_0;
    prevErrNorm = 99999999.9f;
    if ( n >= Nmin ){  // Check
        for(iters = 0; iters < maxIters; iters++){
            x = param;
            T = param;
            optimFunctions_new(JtJ,JtErr,errNorm,residue);
            // If the difference is very small stop
            if( ( abs(errNorm-prevErrNorm) < minErrorChange ) || ( errNorm < minError) )
                break;
            // Update step
            if(motionprior){
                JtJ   = JtJ + prevCov;
                JtErr = JtErr + prevCov * (prevPoseInc-logarithm_map(param));
                LDLT<MatrixXf> solver(JtJ);
                epsInc = solver.solve(JtErr);
                param  << prevParam * inverse_transformation( transformation_expmap(epsInc) );
            }
            else{
                LDLT<MatrixXf> solver(JtJ);
                epsInc = solver.solve(JtErr);                                   //epsInc << JtJ.inverse() * JtErr;
                param  << prevParam * inverse_transformation( transformation_expmap(epsInc) );
            }
            // If the parameter change is small stop, else if the parameter is big update weights
            if( diffManifoldError(param,prevParam) < minChange )
                break;
            // Update values
            prevErrNorm = errNorm;
            prevParam   = param;
        }
        cov = JtJ.inverse();
        residue_ = residue;
        return param;
    }
    else{
        cov = MatrixXf::Zero(6,6);
        residue_ = VectorXf::Zero(n);
        return Matrix4f::Identity();
    }
}

MatrixXf plSVO::optimizeRobustGN_new(VectorXf &residue_){
    Matrix4f x;
    MatrixXf J(n,6);
    VectorXf E(n), Eh(n);
    epsInc = VectorXf::Zero(6);
    //Initial value in SE(3) space
    prevParam   = x_0;
    param       = x_0;
    prevErrNorm = 99999999.9f;
    if ( n >= Nmin ){  // Check
        for(iters = 0; iters < maxIters; iters++){
            x = param;
            T = param;
            optimFunctionsRobust_new(JtJ,JtErr,errNorm,E,Eh);
            // If the difference is very small stop
            if( ( abs(errNorm-prevErrNorm) < minErrorChange ) || ( errNorm < minError) )
                break;
            // Update step
            if(motionprior){
                JtJ   = JtJ + prevCov;
                JtErr = JtErr + prevCov * (prevPoseInc-logarithm_map(param));
                LDLT<MatrixXf> solver(JtJ);
                epsInc = solver.solve(JtErr);
                param  << prevParam * inverse_transformation( transformation_expmap(epsInc) );
            }
            else{
                LDLT<MatrixXf> solver(JtJ);
                epsInc = solver.solve(JtErr);
                param  << prevParam * inverse_transformation( transformation_expmap(epsInc) );
            }
            // If the parameter change is small stop, else if the parameter is big update weights
            if( diffManifoldError(param,prevParam) < minChange )
                break;
            // Update values
            prevErrNorm = errNorm;
            prevParam   = param;
        }
        cov = JtJ.inverse();
        residue_ = E;
        return param;
    }
    else{
        cov = MatrixXf::Zero(6,6);
        residue_ = VectorXf::Zero(n);
        return Matrix4f::Identity();
    }
}

void plSVO::optimFunctions_new(MatrixXf &JtJ_, VectorXf &JtE_, float &errNorm_, VectorXf &E_){

    W = VectorXf::Zero(n);
    MatrixXf J = MatrixXf::Zero(n,6), JtW(6,n);
    E_   = VectorXf::Zero(n);
    JtJ_ = MatrixXf::Zero(6,6);
    JtE_ = VectorXf::Zero(6);
    errNorm_ = 0.f;

    // ------ Point Features
    Vector3f p, ph;
    Vector4f P, TP;
    float    dx, dy, gx, gy, gz, gz2, fgz2, rP, rPH, dH;
    for(unsigned int i = 0; i < nPointsH; i++){
        // Project the 3D point to the Left image in the second frame
        P  << pData(0,i), pData(1,i), pData(2,i), 1.f;
        TP = param * P;
        p << K_amp * TP;
        if( abs(p(2)) > homogTh ){
            ph << p / p(2);
            gx = TP(0);
            gy = TP(1);
            gz = TP(2);
            gz2 = gz*gz;
            fgz2 = f/gz2;
            // Error function
            dx  = ph(0)-pData(7,i);
            dy  = ph(1)-pData(8,i);
            rP  = sqrt(dx*dx+dy*dy);
            errNorm_ += rP*rP;
            E_(i)  = rP;
            // Jacobian function
            if(rP>homogTh){
                MatrixXf Jt(6,1);
                Jt(0,0) = + fgz2 * dx * gz;
                Jt(1,0) = + fgz2 * dy * gz;
                Jt(2,0) = - fgz2 * ( gx*dx + gy*dy );
                Jt(3,0) = - fgz2 * ( gx*gy*dx + gy*gy*dy + gz*gz*dy );
                Jt(4,0) = + fgz2 * ( gx*gx*dx + gz*gz*dx + gx*gy*dy );
                Jt(5,0) = + fgz2 * ( gx*gz*dy - gy*gz*dx );
                for(int j = 0; j < 6; j++){ // Write this way to introduce the weights in JtW
                    J(i,j)   = Jt(j,0) / rP;
                }
            }
        }
        else{
            E_(i)  = 9999.f;
        }
    }

    // ------ Line Features
    Vector3f Ln, q, qh;
    Vector4f Q, TQ;
    float    d_p, d_q, rL, lx, ly, rLH;
    unsigned int j;
    for(unsigned int i = 0; i < nLinesH; i++){
        j = i + nPointsH;
        P  << lData(0,i), lData(1,i), lData(2,i), 1.f;
        Q  << lData(3,i), lData(4,i), lData(5,i), 1.f;
        Ln << lData(6,i), lData(7,i), lData(8,i);
        TP = param * P;
        TQ = param * Q;
        p << K_amp * TP;
        q << K_amp * TQ;
        lx = Ln(0);
        ly = Ln(1);
        if((abs(p(2))>homogTh)&&(abs(q(2))>homogTh)){
            ph << p / p(2);
            qh << q / q(2);
            // Error function
            d_p  = Ln(0) * ph(0) + Ln(1) * ph(1) + Ln(2) ;
            d_q  = Ln(0) * qh(0) + Ln(1) * qh(1) + Ln(2) ;
            rL  = sqrt(d_p*d_p+d_q*d_q);
            errNorm_ += rL*rL;
            E_(j)  = rL;
            // Jacobian function
            MatrixXf Jt_p(6,1),Jt_q(6,1);
            gx = TP(0);
            gy = TP(1);
            gz = TP(2);
            gz2 = gz*gz;
            fgz2 = f/gz2;
            Jt_p(0,0) = + fgz2 * lx * gz;
            Jt_p(1,0) = + fgz2 * ly * gz;
            Jt_p(2,0) = - fgz2 * ( gx*lx + gy*ly );
            Jt_p(3,0) = - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly );
            Jt_p(4,0) = + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly );
            Jt_p(5,0) = + fgz2 * ( gx*gz*ly - gy*gz*lx );
            gx = TQ(0);
            gy = TQ(1);
            gz = TQ(2);
            gz2 = gz*gz;
            fgz2 = f/gz2;
            Jt_q(0,0) = + fgz2 * lx * gz;
            Jt_q(1,0) = + fgz2 * ly * gz;
            Jt_q(2,0) = - fgz2 * ( gx*lx + gy*ly );
            Jt_q(3,0) = - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly );
            Jt_q(4,0) = + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly );
            Jt_q(5,0) = + fgz2 * ( gx*gz*ly - gy*gz*lx );
            for(int j = 0; j < 6; j++){ // Write this way to introduce the weights in JtW
                J(i,j)   = (Jt_p(j,0)*d_p + Jt_q(j,0)*d_q) / rL;
            }
        }
        else{
            E_(j)  = 9999.f;
        }
    }

    // ------ Estimate weights (not in the first iteration)
    J   = 2.f * J;
    JtW = J.transpose();

    switch(optimMode){
        case 0:
            W = VectorXf::Constant(n,1.f);  // Non-Weighted mode (all weights equal to 1.f)
            break;
        case 1:
            estimateGaussianWeights2(E_,W,JtW,errNorm_);
            break;
        case 2:
            estimateTStudentWeights(E_,W,JtW,errNorm_);
            break;
        case 3:
            estimateGammaWeights3_edo(E_,W,JtW,errNorm_);
            break;
        case 4:
            estimateGammaWeights4_edo(E_,W,JtW,errNorm_);
            break;
        default:
            cout << endl << endl << "Error: unknown optimization mode" << endl << endl;
            break;
    }

    // ------ Final Functions
    JtJ_ << JtW * J;
    JtE_ << JtW * E_;
    errNorm_ = errNorm_ / (float)n;

}

void plSVO::optimFunctionsRobust_new(MatrixXf &JtJ_, VectorXf &JtE_, float &errNorm_, VectorXf &E_, VectorXf &Eh){

    W = VectorXf::Zero(n);
    MatrixXf J = MatrixXf::Zero(n,6), JtW(6,n);
    E_   = VectorXf::Zero(n);
    Eh   = VectorXf::Zero(n);
    JtJ_ = MatrixXf::Zero(6,6);
    JtE_ = VectorXf::Zero(6);
    errNorm_ = 0.f;

    // ------ Point Features
    Vector3f p, ph;
    Vector4f P, TP;
    float    dx, dy, gx, gy, gz, gz2, fgz2, rP, rPH, dH;
    for(unsigned int i = 0; i < nPointsH; i++){
        // Project the 3D point to the Left image in the second frame
        P  << pData(0,i), pData(1,i), pData(2,i), 1.f;
        TP = param * P;
        p << K_amp * TP;
        if( abs(p(2)) > homogTh ){
            ph << p / p(2);
            gx = TP(0);
            gy = TP(1);
            gz = TP(2);
            gz2 = gz*gz;
            fgz2 = f/gz2;
            // Error function
            dx  = ph(0)-pData(7,i);
            dy  = ph(1)-pData(8,i);
            rP  = sqrt(dx*dx+dy*dy);
            huberFunction2(rP,rPH,dH);
            errNorm_ += rP*rP;
            E_(i)  = rPH;
            Eh(i) = rP; // Variable to remove outliers (outliers will have small robust errors but big residual errors)
            // Jacobian function
            if(rP>homogTh){
                MatrixXf Jt(6,1);
                Jt(0,0) = + fgz2 * dx * gz;
                Jt(1,0) = + fgz2 * dy * gz;
                Jt(2,0) = - fgz2 * ( gx*dx + gy*dy );
                Jt(3,0) = - fgz2 * ( gx*gy*dx + gy*gy*dy + gz*gz*dy );
                Jt(4,0) = + fgz2 * ( gx*gx*dx + gz*gz*dx + gx*gy*dy );
                Jt(5,0) = + fgz2 * ( gx*gz*dy - gy*gz*dx );
                for(int j = 0; j < 6; j++){ // Write this way to introduce the weights in JtW
                    J(i,j)   = dH * Jt(j,0) / rP;
                }
            }
        }
        else{
            E_(i)  = 9999.f;
            Eh(i) = 9999.f;
        }
    }

    // ------ Line Features
    Vector3f Ln, q, qh;
    Vector4f Q, TQ;
    float    d_p, d_q, rL, lx, ly, rLH;
    unsigned int j;
    for(unsigned int i = 0; i < nLinesH; i++){
        j = i + nPointsH;
        P  << lData(0,i), lData(1,i), lData(2,i), 1.f;
        Q  << lData(3,i), lData(4,i), lData(5,i), 1.f;
        Ln << lData(6,i), lData(7,i), lData(8,i);
        TP = param * P;
        TQ = param * Q;
        p << K_amp * TP;
        q << K_amp * TQ;
        lx = Ln(0);
        ly = Ln(1);
        if((abs(p(2))>homogTh)&&(abs(q(2))>homogTh)){
            ph << p / p(2);
            qh << q / q(2);
            // Error function
            d_p  = Ln(0) * ph(0) + Ln(1) * ph(1) + Ln(2) ;
            d_q  = Ln(0) * qh(0) + Ln(1) * qh(1) + Ln(2) ;
            rL  = sqrt(d_p*d_p+d_q*d_q);
            huberFunction2(rL,rLH,dH);
            errNorm_ += rL*rL;
            E_(j)  = rLH;
            Eh(j) = rL;
            // Jacobian function
            MatrixXf Jt_p(6,1),Jt_q(6,1);
            gx = TP(0);
            gy = TP(1);
            gz = TP(2);
            gz2 = gz*gz;
            fgz2 = f/gz2;
            Jt_p(0,0) = + fgz2 * lx * gz;
            Jt_p(1,0) = + fgz2 * ly * gz;
            Jt_p(2,0) = - fgz2 * ( gx*lx + gy*ly );
            Jt_p(3,0) = - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly );
            Jt_p(4,0) = + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly );
            Jt_p(5,0) = + fgz2 * ( gx*gz*ly - gy*gz*lx );
            gx = TQ(0);
            gy = TQ(1);
            gz = TQ(2);
            gz2 = gz*gz;
            fgz2 = f/gz2;
            Jt_q(0,0) = + fgz2 * lx * gz;
            Jt_q(1,0) = + fgz2 * ly * gz;
            Jt_q(2,0) = - fgz2 * ( gx*lx + gy*ly );
            Jt_q(3,0) = - fgz2 * ( gx*gy*lx + gy*gy*ly + gz*gz*ly );
            Jt_q(4,0) = + fgz2 * ( gx*gx*lx + gz*gz*lx + gx*gy*ly );
            Jt_q(5,0) = + fgz2 * ( gx*gz*ly - gy*gz*lx );
            for(int j = 0; j < 6; j++){ // Write this way to introduce the weights in JtW
                J(i,j)   = dH * (Jt_p(j,0)*d_p + Jt_q(j,0)*d_q) / rL;
            }
        }
        else{
            E_(j)  = 9999.f;
            Eh(j) = 9999.f;
        }
    }

    // ------ Estimate weights (not in the first iteration)
    J   = 2.f * J;
    JtW = J.transpose();

    switch(optimMode){
        case 0:
            W = VectorXf::Constant(n,1.f);  // Non-Weighted mode (all weights equal to 1.f)
            break;
        case 1:
            estimateGaussianWeights2(E_,W,JtW,errNorm_);
            break;
        case 2:
            estimateTStudentWeights(E_,W,JtW,errNorm_);
            break;
        case 3:
            estimateGammaWeights3_edo(E_,W,JtW,errNorm_);
            break;
        case 4:
            estimateGammaWeights4_edo(E_,W,JtW,errNorm_);
            break;
        default:
            cout << endl << endl << "Error: unknown optimization mode" << endl << endl;
            break;
    }

    // ------ Final Functions
    JtJ_ << JtW * J;
    JtE_ << JtW * E_;
    errNorm_ = errNorm_ / (float)n;

}



void plSVO::fastFilter(VectorXf residue_){

    unsigned int k = 0;
    float        median, stdv, MAD;
    VectorXf     residue_sort(n);
    inliers.clear();

    // Automatic estimation of the threshold thanks to the MAD
    residue_sort = residue_;
    sort(residue_sort.derived().data(),residue_sort.derived().data()+residue_sort.size());
    median = residue_sort( (int)(0.5f*n) );
    stdv = 1.4826f * vector_stdv_MAD(residue_);

    // Points Features
    if(nPoints>0){
        MatrixXf pDataAux = MatrixXf::Zero(Np,nPoints);
        for(unsigned int i = 0; i < nPoints; i++){
            if(residue_(i) < huberK * stdv){
            //if(residue_(i) < huberK * median ){
                    inliers.push_back(true);
                    for(unsigned int j = 0; j < Np; j++){
                        pDataAux(j,k) = pData(j,i);
                    }
                    k++;
            }
            else
                inliers.push_back(false);
        }
        pData_full = pData;
        pData     = pDataAux.block(0,0,Np,k);
        nPointsH  = k;
        nPoints   = k;
        k = 0;
    }

    // Line Features
    if(nLines>0){
        MatrixXf lDataAux = MatrixXf::Zero(Nl,nLines);
        for(unsigned int i = nPoints; i < nLines+nPoints; i++){
            if(residue_(i) < huberK * stdv ){
                inliers.push_back(true);
                for(unsigned int j = 0; j < Nl; j++){
                    lDataAux(j,k) = lData(j,i-nPoints);
                }
                k++;
            }
            else
                inliers.push_back(false);
        }
        lData_full = lData;
        lData     = lDataAux.block(0,0,Nl,k);
        nLinesH   = k;
        nLines    = k;
    }

    n = nPointsH + nLinesH;

}

void plSVO::removeERODE_new(VectorXf residue_){

    unsigned int k = 0;
    float        median, stdv, MAD;
    VectorXf     residue_sort(n);
    inliers.clear();

    // Automatic estimation of the threshold thanks to the MAD
    residue_sort = residue_;
    sort(residue_sort.derived().data(),residue_sort.derived().data()+residue_sort.size());
    median = residue_sort( (int)(0.5f*n) );
    stdv = 1.4826f * vector_stdv_MAD(residue_);

    // Points Features
    if(nPoints>0){
        MatrixXf pDataAux = MatrixXf::Zero(Np,nPoints);
        for(unsigned int i = 0; i < nPointsH; i++){
            if(residue_(i) < huberK * stdv){
            //if(residue_(i) < huberK * median ){
                    inliers.push_back(true);
                    for(unsigned int j = 0; j < Np; j++){
                        pDataAux(j,k) = pData(j,i);
                    }
                    k++;
            }
            else
                inliers.push_back(false);
        }
        //pData_full = pData;
        pData     = pDataAux.block(0,0,Np,k);
        nPointsH  = k;
        k = 0;
    }

    // Line Features
    if(nLines>0){
        MatrixXf lDataAux = MatrixXf::Zero(Nl,nLines);
        for(unsigned int i = nPointsH; i < nLinesH+nPoints; i++){
            if(residue_(i) < huberK * stdv ){
                inliers.push_back(true);
                for(unsigned int j = 0; j < Nl; j++){
                    lDataAux(j,k) = lData(j,i-nPointsH);
                }
                k++;
            }
            else
                inliers.push_back(false);
        }
        //lData_full = lData;
        lData     = lDataAux.block(0,0,Nl,k);
        nLinesH   = k;
    }

    n = nPointsH + nLinesH;

}



void plSVO::estimateTStudentWeights(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_){
    // Variables
    VectorXf res2 = VectorXf::Zero(n);
    VectorXf res4 = VectorXf::Zero(n);
    weights  = VectorXf::Zero(n);

    // Parameters
    float mu    = 5.f, mu1;
    int maxit   = 25;
    int iter    = 0;
    float xini  = 1.f, x_last, x_next = 0.f, delta_error = 0.0001f;
    int samples = 0;
    mu1 = mu + 1.f;

    // First estimation
    x_last = xini;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)!=9999.f){
            res2(i) = residues(i) * residues(i);
            x_next += res2(i) / (mu + res2(i)/x_last);
            samples++;
            /*res4(i) = res2(i) * res2(i);
            x_next += res4(i) / (mu + res4(i)/x_last);*/
        }
    }
    x_next *= mu1 / ( float(samples) );
    samples = 0;
    float stdv = vector_stdv_MAD(residues);
    // Recursive estimation
    while( (abs(x_next-x_last) > delta_error) && (iter < maxit) ){
        x_last = x_next;
        x_next = 0.f;
        for(unsigned int i = 0; i < n; i++){
            if(residues(i)<3.f*stdv){
            //if(residues(i)!=9999.f){
                x_next += res2(i) / (mu + res2(i)/x_last);
                samples++;
            }
        }
        x_next *= mu1 / ( float(samples) );
        samples = 0;
        iter++;
    }

    // Weights asignation
    errNorm_ = 0.f;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)!=9999.f){
            weights(i)  = mu1 / ( mu + res2(i)/x_next );
        }
        else{
            weights(i)  = 0.f;
        }
        errNorm_ += weights(i)*res2(i);
        for(unsigned int j = 0; j < 6; j++){
            JtW_(j,i) = weights(i) * JtW_(j,i);
        }
    }

}

void plSVO::estimateTStudentWeights2(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_){
    // Variables
    VectorXf res2 = VectorXf::Zero(n);
    weights  = VectorXf::Zero(n);
    // Parameters
    float mu    = 5.f, mu1;
    int maxit   = 15;
    int iter    = 0;
    float xini  = 1.f, x_last, x_next = 0.f;
    float beta;
    mu1 = mu + 1.f;
    beta  = mu1/float(n);
    // Sort the residues vector, and estimate the median threshold
    int   samples = 0;
    float median_th, threshold = 0.5f, factor = 1.5f;
    VectorXf     residue_sort(n);
    residue_sort = residues;
    sort(residue_sort.derived().data(),residue_sort.derived().data()+residue_sort.size());
    median_th = factor * residue_sort((int)(threshold*n));
    // First estimation
    x_last = xini;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)<median_th){
            samples++;
            res2(i) = residues(i) * residues(i);
            x_next += res2(i) / (mu + res2(i)/x_last);
        }
    }
    beta  = mu1/float(samples);
    x_next *= beta;
    // Recursive estimation
    while( (abs(x_next-x_last) > delta) && (iter < maxit) ){
        x_last = x_next;
        x_next = 0.f;
        samples = 0;
        for(unsigned int i = 0; i < n; i++){
            if(residues(i)<median_th){
                samples++;
                x_next += res2(i) / ( mu + res2(i)/x_last );
            }
        }
        beta    = mu1/float(samples);
        x_next *= beta;
        iter++;
    }
    // Weights asignation
    errNorm_ = 0.f;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)!=9999.f)
            weights(i) = mu1 / ( mu + res2(i)/x_next );
        else{
            weights(i) = 0.f;
        }
        errNorm_ += weights(i)*res2(i);
        for(unsigned int j = 0; j < 6; j++){
            JtW_(j,i) = weights(i) * JtW_(j,i);
        }
    }

}



void plSVO::estimateGammaWeights(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_){

    // Variables
    int   samples = 0;
    float alpha, theta, mean, var, wmax = 0.f, res = 0.f, res2 = 0.f, lres = 0.f;
    weights  = VectorXf::Zero(n);
    VectorXf residues2 = VectorXf::Zero(n);
    // Robust estimation of the distribution - Method of Moments Estimation
    for(unsigned int i = 0; i < n; i++){
        residues2(i) = residues(i)*residues(i);
        if(residues(i)!=9999.f){
            samples++;
            res  += residues(i);
            res2 += residues2(i);
        }
    }
    if(samples!=0){                                     // TODO: check samples&mea&theta are not 0
        mean    = res  / (float)(samples);
        var     = res2 / (float)(samples) - mean*mean;
        theta = var / (mean+delta);
        alpha = var/(theta*theta+delta) - 1.f;              // Since we only use it to estimate weights, we estimate alpha-1
    }

    // Weights asignation
    errNorm_ = 0.f;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)!=9999.f){
            weights(i) = (residues(i)-theta*alpha)/(theta*residues2(i)+delta);
            if( isinf(weights(i)) || isnan(weights(i)) )
                weights(i) = 0.f;
        }
        else{
            weights(i) = 0.f;
        }
        errNorm_ += weights(i)*residues2(i);
        for(unsigned int j = 0; j < 6; j++){
            JtW_(j,i) = weights(i) * JtW_(j,i);
        }
    }

}

void plSVO::estimateGammaWeights2(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_){

    // Variables
    float threshold = 0.5f, median_th;
    int   samples = 0;
    float alpha, theta, mean, var, res = 0.f, res2 = 0.f, lres = 0.f, wmax = 0.f;
    weights  = VectorXf::Zero(n);
    VectorXf residues2 = VectorXf::Zero(n);

    // Robust estimation of the distribution - Method of Moments Estimation
    VectorXf     residue_sort(n);
    residue_sort = residues;
    sort(residue_sort.derived().data(),residue_sort.derived().data()+residue_sort.size());
    median_th = huberK * residue_sort((int)(threshold*n));
    for(unsigned int i = 0; i < n; i++){
        residues2(i) = residues(i)*residues(i);
        if(residues(i)<median_th){
            samples++;
            res  += residues(i);
            res2 += residues2(i);
            //lres += log(residues(i));
        }
    }
    if(samples!=0){                                     // TODO: check samples&mea&theta are not 0
        mean    = res  / (float)(samples);
        var     = res2 / (float)(samples) - mean*mean;
        theta = var / (mean+delta);
        alpha = var/(theta*theta+delta) - 1.f;              // Since we only use it to estimate weights, we estimate alpha-1
    }

    // Weights asignation
    errNorm_ = 0.f;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)!=9999.f){
            weights(i) = (residues(i)-theta*alpha)/(theta*residues2(i)+delta);
            if( isinf(weights(i)) || isnan(weights(i)) )
                weights(i) = 0.f;
        }
        else{
            weights(i) = 0.f;
        }
        errNorm_ += weights(i)*residues2(i);
        for(unsigned int j = 0; j < 6; j++){
            JtW_(j,i) = weights(i) * JtW_(j,i);
        }
    }

}

void plSVO::estimateGammaWeights3(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_){

    // Variables
    float threshold = 0.5f, median, MAD ;
    int   samples = 0;
    float alpha, theta, mean, var, stdv, res = 0.f, res2 = 0.f, lres = 0.f, wmax = 0.f;
    weights  = VectorXf::Zero(n);
    VectorXf residues2 = VectorXf::Zero(n);

    // Robust estimation of the distribution - Method of Moments Estimation
    VectorXf     residue_sort(n), residue_aux;
    residue_sort = residues;
    residue_aux  = residues;
    sort(residue_sort.derived().data(),residue_sort.derived().data()+residue_sort.size());
    median = residue_sort( (int)(0.5f*n) );
    residue_aux << (residue_aux - VectorXf::Constant(n,median)).cwiseAbs();
    sort(residue_aux.derived().data(),residue_aux.derived().data()+residue_aux.size());
    MAD = residue_aux( (int)(0.5f*n) );
    stdv = 1.4826f * MAD;
    var  = stdv*stdv;
    for(unsigned int i = 0; i < n; i++){
        residues2(i) = residues(i)*residues(i);
        if(residues(i)<huberK*stdv){
            samples++;
            res  += residues(i);
            res2 += residues2(i);
        }
    }
    if(samples!=0){                                     // TODO: check samples&mea&theta are not 0
        mean    = res  / (float)(samples);
        //var     = res2 / (float)(samples) - mean*mean;
        //cout << endl << var << " \t " << res2 / (float)(samples) - mean*mean << endl ;
        theta = var / (mean+delta);
        alpha = var/(theta*theta+delta) - 1.f;                // Since we only use it to estimate weights, we estimate alpha-1
    }

    // Weights asignation
    errNorm_ = 0.f;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)!=9999.f){
            weights(i) = (residues(i)-theta*alpha)/(theta*residues2(i)+delta);
            if( isinf(weights(i)) || isnan(weights(i)) )
                weights(i) = 0.f;
        }
        else{
            weights(i) = 0.f;
        }
        errNorm_ += weights(i)*residues2(i);
        for(unsigned int j = 0; j < 6; j++){
            JtW_(j,i) = weights(i) * JtW_(j,i);
        }
    }

}



void plSVO::estimateGammaWeights_edo(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_){

    // Variables
    int   samples = 0;
    float alpha, theta, mean, var, wmax = 0.f, res = 0.f, res2 = 0.f, lres = 0.f;
    weights  = VectorXf::Zero(n);
    VectorXf residues2 = VectorXf::Zero(n);
    // Robust estimation of the distribution - Method of Moments Estimation
    for(unsigned int i = 0; i < n; i++){
        residues2(i) = residues(i)*residues(i);
        if(residues(i)!=9999.f){
            samples++;
            res  += residues(i);
            res2 += residues2(i);
            //lres += log(residues(i));
        }
    }
    if(samples!=0){                                     // TODO: check samples&mea&theta are not 0
        mean    = res  / (float)(samples);
        var     = res2 / (float)(samples) - mean*mean;
        theta = var / mean;
        alpha = var/(theta*theta) - 1.f;              // Since we only use it to estimate weights, we estimate alpha-1
    }

    // Weights asignation
    alpha = 1.6448f;
    theta = 1.3920f;
    errNorm_ = 0.f;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)!=9999.f){
            weights(i) = ( residues(i) - theta * alpha * log(residues(i)+delta)  ) / (theta*residues2(i)+delta);
            if( (weights(i)>wmax) && (!isinf(weights(i))) && (!isnan(weights(i))) )
                wmax = weights(i);
            else
                weights(i) = 0.f;
        }
        else{
            weights(i) = 0.f;
        }
        /*if( (weights(i)<1.f) && (weights(i)!=0.f) )
            weights(i) = 1.f;*/
        errNorm_ += weights(i)*residues2(i);
        for(unsigned int j = 0; j < 6; j++){
            JtW_(j,i) = weights(i) * JtW_(j,i);
        }
    }

    /*weights  = weights  / wmax;
    errNorm_ = errNorm_ / wmax;
    JtW_     = JtW_     / wmax;*/

}

void plSVO::estimateGammaWeights2_edo(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_){

    // Variables
    float threshold = 0.5f, factor = 2.f, median_th;
    int   samples = 0;
    float alpha, theta, mean, var, res = 0.f, res2 = 0.f, lres = 0.f, wmax = 0.f;
    weights  = VectorXf::Zero(n);
    VectorXf residues2 = VectorXf::Zero(n);

    // Robust estimation of the distribution - Method of Moments Estimation
    VectorXf     residue_sort(n);
    residue_sort = residues;
    sort(residue_sort.derived().data(),residue_sort.derived().data()+residue_sort.size());
    median_th = factor * residue_sort((int)(threshold*n));
    for(unsigned int i = 0; i < n; i++){
        residues2(i) = residues(i)*residues(i);
        if(residues(i)<median_th){
            samples++;
            res  += residues(i);
            res2 += residues2(i);
            //lres += log(residues(i));
        }
    }
    if(samples!=0){                                     // TODO: check samples&mea&theta are not 0
        mean    = res  / (float)(samples);
        var     = res2 / (float)(samples) - mean*mean;
        theta = var / mean;
        alpha = var/(theta*theta) - 1.f;              // Since we only use it to estimate weights, we estimate alpha-1
    }

    // Weights asignation
    errNorm_ = 0.f;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)!=9999.f){
            weights(i) = ( residues(i) - theta * alpha * log(residues(i)+delta)  ) / (theta*residues2(i)+delta);
            if( (weights(i)>wmax) && (!isinf(weights(i))) && (!isnan(weights(i))) )
                wmax = weights(i);
            else
                weights(i) = 0.f;
        }
        else{
            weights(i) = 0.f;
        }
        errNorm_ += weights(i)*residues2(i);
        for(unsigned int j = 0; j < 6; j++){
            JtW_(j,i) = weights(i) * JtW_(j,i);
        }
    }

    /*weights  = weights  / wmax;
    errNorm_ = errNorm_ / wmax;
    JtW_     = JtW_     / wmax;*/

}

void plSVO::estimateGammaWeights3_edo(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_){

    // Variables
    float threshold = 0.5f, median, MAD ;
    int   samples = 0;
    float alpha, theta, mean, var, stdv, res = 0.f, res2 = 0.f, lres = 0.f, wmax = 0.f;
    weights  = VectorXf::Zero(n);
    VectorXf residues2 = VectorXf::Zero(n);

    // Robust estimation of the distribution - Method of Moments Estimation
    VectorXf     residue_sort(n), residue_aux;
    residue_sort = residues;
    residue_aux  = residues;
    sort(residue_sort.derived().data(),residue_sort.derived().data()+residue_sort.size());
    median = residue_sort( (int)(0.5f*n) );
    residue_aux << (residue_aux - VectorXf::Constant(n,median)).cwiseAbs();
    sort(residue_aux.derived().data(),residue_aux.derived().data()+residue_aux.size());
    MAD = residue_aux( (int)(0.5f*n) );
    stdv = 1.4826f * MAD;
    var  = stdv*stdv;

    for(unsigned int i = 0; i < n; i++){
        residues2(i) = residues(i)*residues(i);
        if(residues(i)<huberK*stdv){
            samples++;
            res  += residues(i);
            res2 += residues2(i);
        }
    }
    if(samples!=0){
        mean    = res  / (float)(samples);
        var     = res2 / (float)(samples) - mean*mean;
        theta = var / (mean);
        alpha = var/(theta*theta) - 1.f;                // Since we only use it to estimate weights, we estimate alpha-1
    }

    // Weights asignation
    errNorm_ = 0.f;
    for(unsigned int i = 0; i < n; i++){        
        if(residues(i)!=9999.f){
            weights(i) = ( residues(i) - theta * alpha * log(residues(i)+delta)  ) / (theta*residues2(i)+delta);
            if( isinf(weights(i)) || isnan(weights(i)) )
                weights(i) = 0.f;
        }
        else{
            weights(i) = 0.f;
        }
        if( (weights(i) < 1.f) && (weights(i)!=0.f) )
            weights(i) = 1.f;
        errNorm_ += weights(i)*residues2(i);
        for(unsigned int j = 0; j < 6; j++){
            JtW_(j,i) = weights(i) * JtW_(j,i);
        }
    }

}

void plSVO::estimateGammaWeights4_edo(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_){

    // Variables
    float threshold = 0.5f, median, MAD ;
    int   samples = 0;
    float alpha, theta, mean, var, stdv, res = 0.f, res2 = 0.f, lres = 0.f, wmax = 0.f, wmin = 99999999999999999999999.f;
    weights  = VectorXf::Zero(n);
    VectorXf residues2 = VectorXf::Zero(n);

    // Robust estimation of the distribution - Method of Moments Estimation
    VectorXf     residue_sort(n), residue_aux;
    residue_sort = residues;
    residue_aux  = residues;
    sort(residue_sort.derived().data(),residue_sort.derived().data()+residue_sort.size());
    median = residue_sort( (int)(0.5f*n) );
    residue_aux << (residue_aux - VectorXf::Constant(n,median)).cwiseAbs();
    sort(residue_aux.derived().data(),residue_aux.derived().data()+residue_aux.size());
    MAD = residue_aux( (int)(0.5f*n) );
    stdv = 1.4826f * MAD;
    var  = stdv*stdv;

    for(unsigned int i = 0; i < n; i++){
        residues2(i) = residues(i)*residues(i);
        if(residues(i)<3.f*stdv){
            samples++;
            res  += residues(i);
            res2 += residues2(i);
        }
    }
    if(samples!=0){
        mean    = res  / (float)(samples);
        //var     = res2 / (float)(samples) - mean*mean;
        theta = var / (mean);
        alpha = var/(theta*theta) - 1.f;  // Since we only use it to estimate weights, we estimate alpha-1
    }

    // Weights asignation
    errNorm_ = 0.f;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)!=9999.f){
            weights(i) = ( residues(i) - theta * alpha * log(residues(i)+delta)  ) / (theta*residues2(i)+delta);
            if( isinf(weights(i)) || isnan(weights(i)) )
                weights(i) = 0.f;
            else{
                if(weights(i)>wmax)
                    wmax = weights(i);
                if(weights(i)<wmin)
                    wmin = weights(i);
            }
        }
        else{
            weights(i) = 0.f;
        }
    }

    for(unsigned int i = 0; i < n; i++){
        if(weights(i)!=0.f){
            weights(i)  = ( weights(i) - wmin + delta ) / wmax;
        }
        errNorm_ += weights(i)*residues2(i);
        for(unsigned int j = 0; j < 6; j++){
            JtW_(j,i) = weights(i) * JtW_(j,i);
        }
    }

}



void plSVO::estimateGaussianWeights(VectorXf &weights, MatrixXf &JtW_, float &errNorm_){

    weights = VectorXf::Zero(n);
    Matrix3f R;
    errNorm_ = 0.f;

    // ------ Point Features
    Vector3f p, ph;
    Vector4f P, TP;
    float    gx, gy, gz, gz2, dx, dy, dxdy;
    R << param.block(0,0,3,3);
    for(unsigned int i = 0; i < nPointsH; i++){
        // Project the 3D point to the Left image in the second frame
        P  << pData(0,i), pData(1,i), pData(2,i), 1.f;
        TP = param * P;
        p << K_amp * TP;
        if( abs(p(2)) > homogTh ){
            ph << p / p(2);
            gx = TP(0);
            gy = TP(1);
            gz = TP(2);
            gz2 = gz*gz;
            dx  = ph(0)-pData(7,i);
            dy  = ph(1)-pData(8,i);
            dxdy= dx*dx+dy*dy;
            // Uncertainty function
            float px_hat = pData(3,i) - cx;
            float py_hat = pData(4,i) - cy;
            float disp   = pData(3,i) - pData(5,i);
            float disp2  = disp * disp;
            Matrix3f covP_an;
            covP_an(0,0) = disp2+2.f*px_hat*px_hat;
            covP_an(0,1) = 2.f*px_hat*py_hat;
            covP_an(0,2) = 2.f*f*px_hat;
            covP_an(1,1) = disp2+2.f*py_hat*py_hat;
            covP_an(1,2) = 2.f*f*py_hat;
            covP_an(2,2) = 2.f*f*f;
            covP_an(1,0) = covP_an(0,1);
            covP_an(2,0) = covP_an(0,2);
            covP_an(2,1) = covP_an(1,2);
            covP_an << covP_an / (disp2*disp2);     // Covariance of the 3D point P up to b2*sigma2
            MatrixXf Jhg(2,3), covp(2,2), covp_inv(2,2);
            Jhg  << gz, 0.f, -gx, 0.f, gz, -gy;
            Jhg  << Jhg * R;
            covp << Jhg * covP_an * Jhg.transpose();
            covp << covp / (gz2*gz2);               // Covariance of the 3D projection \hat{p} up to f2*b2*sigma2
            covp = bsigmaP * covp;
            covp(0,0) = covp(0,0) + sigmaP2;
            covp(1,1) = covp(1,1) + sigmaP2;
            covp_inv = covp.inverse();
            // Update the weights matrix
            Vector2f aux;
            aux << dx, dy;
            weights(i) = aux.transpose() * covp_inv * aux;
            weights(i) = weights(i) / dxdy;
            // Update errNorm and JtW
            errNorm_ += aux.transpose() * covp_inv * aux;
            for(unsigned int j = 0; j < 6; j++){
                JtW_(j,i) = weights(i) * JtW_(j,i);
            }
        }
    }

    // ------ Line Features
    Vector3f L, Ln, q, qh;
    Vector4f Q, TQ;
    unsigned int j;
    float dp, dq, dpdq;
    for(unsigned int i = 0; i < nLinesH; i++){
        j = i + nPointsH;
        P  << lData(0,i), lData(1,i), lData(2,i), 1.f;
        Q  << lData(3,i), lData(4,i), lData(5,i), 1.f;
        Ln << lData(6,i), lData(7,i), lData(8,i);
        TP = param * P;
        TQ = param * Q;
        p << K_amp * TP;
        q << K_amp * TQ;
        if((abs(p(2))>homogTh)||(abs(q(2))>homogTh)){
            ph << p / p(2);
            qh << q / q(2);
            dp  = Ln(0) * ph(0) + Ln(1) * ph(1) + Ln(2) ;
            dq  = Ln(0) * qh(0) + Ln(1) * qh(1) + Ln(2) ;
            dpdq= dp*dp+dq*dq;
            // Uncertainty function
            float px_hat = lData(13,i) - cx;
            float py_hat = lData(14,i) - cy;
            float disp   = lData(13,i) - lData(15,i);
            float disp2  = disp * disp;
            Matrix3f covP_an;
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
            MatrixXf J_ep(1,3);
            float lxpz = Ln(0)*p(2);
            float lypz = Ln(1)*p(2);
            J_ep << lxpz*f, lypz*f, lxpz*cx+lypz*cy-Ln(0)*p(0)-Ln(1)*p(1);
            J_ep << J_ep * R;
            float p4 = pow(p(2),4);
            float cov_p;
            VectorXf cov_aux(1);
            cov_aux << J_ep * covP_an * J_ep.transpose();
            cov_p = cov_aux(0);
            cov_p = 1.f/cov_p;
            cov_p = p4 * cov_p * 0.5f * bsigmaL_inv;

            px_hat = lData(16,i) - cx;
            py_hat = lData(17,i) - cy;
            disp   = lData(16,i) - lData(18,i);
            disp2  = disp * disp;
            Matrix3f covQ_an;
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
            lxpz = Ln(0)*q(2);
            lypz = Ln(1)*q(2);
            J_ep << lxpz*f, lypz*f, lxpz*cx+lypz*cy-Ln(0)*q(0)-Ln(1)*q(1);
            J_ep << J_ep * R;
            float q4 = pow(q(2),4);
            float cov_q;
            cov_aux << J_ep * covQ_an * J_ep.transpose();
            cov_q = cov_aux(0);
            cov_q = 1.f/cov_q;
            cov_q = q4 * cov_q * 0.5f * bsigmaL_inv;

            if( isinf(cov_p) || isnan(cov_p) )  cov_p = 0.f;
            if( isinf(cov_q) || isnan(cov_q) )  cov_q = 0.f;

            weights(j) = dp*dp*cov_p+dq*dq*cov_q;
            weights(i) = weights(i) / dpdq;
            // Update errNorm and JtW
            errNorm_ += dp*dp*cov_p+dq*dq*cov_q;
            for(unsigned int j = 0; j < 6; j++){
                JtW_(j,i) = weights(i) * JtW_(j,i);
            }

        }

    }

}


void plSVO::estimateGaussianWeights2(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_){

    // Variables
    float threshold = 0.5f, median, MAD, stdv, var, var_inv ;
    int   samples = 0;
    weights  = VectorXf::Zero(n);

    stdv = 1.4826f * vector_stdv_MAD(residues);
    var  = stdv*stdv;
    var_inv = 1.f / var;

    // Weights asignation
    errNorm_ = 0.f;
    for(unsigned int i = 0; i < n; i++){
        if(residues(i)!=9999.f)
            weights(i) = var_inv;
        else
            weights(i) = 0.f;
    }
    for(unsigned int i = 0; i < n; i++){
        errNorm_ += weights(i)*residues(i)*residues(i);
        for(unsigned int j = 0; j < 6; j++){
            JtW_(j,i) = weights(i) * JtW_(j,i);
        }
    }

}




void plSVO::huberFunction2(float r, float &rH, float &dH ){
    // Function: log(1+s)
    float r1 = 1.f+r;
    rH = log(r1);
    dH = 1.f / r1;
    //// Function: log(1+s2)
    //float r1 = 1.f+r*r;
    //rH = log(r1);
    //dH = 2.f * r / r1;
}

void plSVO::huberFunction3(float r, float &rH, float &dH ){
    /*// Function: sqrt(1+s)
    float r1 = 1.f+r;
    float sq = sqrt(r1);
    rH = sq-1.f;
    dH = 0.5f / sq;*/
    // Function: sqrt(1+s2)
    float r1 = 1.f+r*r;
    float sq = sqrt(r1);
    rH = sq-1.f;
    dH = r / sq;
}

/* Representation functions */

Mat plSVO::imageInliersWeights(){

    // Variables
    unsigned int    r, g, b = 0;
    Point2f         P;
    Point           sP, eP, mP;
    float           maxP = 0.f, minP = 999999.f, aux, invP, thick = 1.5f;
    int             k = 0, radius  = 3;

    // Estimates the maximum and minimum weight values
    for(unsigned int i = 0; i < nPointsH + nLinesH; i++){
        if(inliers[i]){
            aux = W(i);
            if(aux>maxP)
                maxP = aux;
            if(aux<minP)
                minP = aux;
            k++;
        }
    }
    invP = maxP - minP;
    invP = 1.f/invP;

    // Represent stereo points
    k = 0;
    for(unsigned int i = 0; i < nPoints; i++){
        if(inliers[i]){
            g = 255 * invP * ( W(k) - minP );
            r = 255-g;
            P = cv::Point(int(pData(3,k)),int(pData(4,k)));
            circle(imgLeft, P, radius, Scalar(b,g,r), thick);
            k++;
        }
    }

    // Represent stereo line segments
    k = 0;
    for(unsigned int i = nPoints; i < nPoints + nLines; i++)    {
        if(inliers[i]){
            g = 255 * invP * ( W(k) - minP );
            r = 255-g;
            sP = cv::Point(int(lData(13,k)),int(lData(14,k)));
            eP = cv::Point(int(lData(16,k)),int(lData(17,k)));
            mP   = cv::Point((sP.x+eP.x)/2,(sP.y+eP.y)/2);
            cv::line( imgLeft,sP,eP,Scalar(b,g,r));
            k++;
        }
    }

    return imgLeft;

}

Mat plSVO::imageTracking(){

    int lowest=100, highest=255;
    int range=(highest-lowest)+1;
    unsigned int r, g, b; //the color of lines
    string  text;
    Point2f P, Poff;
    Point   sP, eP, mP;

    // Represent stereo points
    int radius  = 3;
    float thick = 1.5f;
    Poff = cv::Point2f(0.1,0.1);
    for(unsigned int i = 0; i < pointsFirst.size(); i++){
        r = lowest+int(rand()%range);
        g = lowest+int(rand()%range);
        b = lowest+int(rand()%range);
        text = "P" + to_string(i);
        P = pointsFirst[i].pt;
        circle(imgFirst, P, radius, Scalar(b,g,r), thick);
        cv::putText( imgFirst, text , P+Poff , FONT_HERSHEY_PLAIN , 0.8 , 1 , 1.8 , LINE_8 , false);
        P = pointsSecond[i].pt;
        circle(imgSecond, P, radius, Scalar(b,g,r), thick);
        cv::putText( imgSecond, text , P+Poff , FONT_HERSHEY_PLAIN , 0.8 , 1 , 1.8 , LINE_8 , false);
        P = pointsThird[i].pt;
        circle(imgLeft, P, radius, Scalar(b,g,r), thick);
        cv::putText( imgLeft, text , P+Poff , FONT_HERSHEY_PLAIN , 0.8 , 1 , 1.8 , LINE_8 , false);
        P = pointsFourth[i].pt;
        circle(imgRight, P, radius, Scalar(b,g,r), thick);
        cv::putText( imgRight, text , P+Poff , FONT_HERSHEY_PLAIN , 0.8 , 1 , 1.8 , LINE_8 , false);
    }

    // Represent stereo line segments
    for(unsigned int i = 0; i < linesFirst.size(); i++)    {
        r = lowest+int(rand()%range);
        g = lowest+int(rand()%range);
        b = lowest+int(rand()%range);
        text = "L" + to_string(i);
        sP   = cv::Point(int(linesFirst[i].startPointX),int(linesFirst[i].startPointY));
        eP   = cv::Point(int(linesFirst[i].endPointX),  int(linesFirst[i].endPointY));
        mP   = cv::Point((sP.x+eP.x)/2,(sP.y+eP.y)/2);
        cv::line( imgFirst,sP,eP,Scalar(b,g,r));
        cv::putText( imgFirst, text , mP , FONT_HERSHEY_PLAIN , 0.8 , 1 , 1.8 , LINE_8 , false);
        sP   = cv::Point(int(linesSecond[i].startPointX),int(linesSecond[i].startPointY));
        eP   = cv::Point(int(linesSecond[i].endPointX),  int(linesSecond[i].endPointY));
        mP   = cv::Point((sP.x+eP.x)/2,(sP.y+eP.y)/2);
        cv::line( imgSecond,sP,eP,Scalar(b,g,r));
        cv::putText( imgSecond, text , mP , FONT_HERSHEY_PLAIN , 0.8 , 1 , 1.8 , LINE_8 , false);
        sP   = cv::Point(int(linesThird[i].startPointX),int(linesThird[i].startPointY));
        eP   = cv::Point(int(linesThird[i].endPointX),  int(linesThird[i].endPointY));
        mP   = cv::Point((sP.x+eP.x)/2,(sP.y+eP.y)/2);
        cv::line( imgLeft,sP,eP,Scalar(b,g,r));
        cv::putText( imgLeft, text , mP , FONT_HERSHEY_PLAIN , 0.8 , 1 , 1.8 , LINE_8 , false);
        sP   = cv::Point(int(linesFourth[i].startPointX),int(linesFourth[i].startPointY));
        eP   = cv::Point(int(linesFourth[i].endPointX),  int(linesFourth[i].endPointY));
        mP   = cv::Point((sP.x+eP.x)/2,(sP.y+eP.y)/2);
        cv::line( imgRight,sP,eP,Scalar(b,g,r));
        cv::putText( imgRight, text , mP , FONT_HERSHEY_PLAIN , 0.8 , 1 , 1.8 , LINE_8 , false);
    }

    // Mount the output image and represent it
    int rows = imgFirst.rows;
    int cols = imgFirst.cols;
    Mat imgStereo(rows*2,cols*2,imgFirst.type());
    imgFirst.copyTo( imgStereo( Rect(0,rows,cols,rows) ) );
    imgSecond.copyTo( imgStereo( Rect(cols,rows,cols,rows) ) );
    imgLeft.copyTo( imgStereo( Rect(0,0,cols,rows) ) );
    imgRight.copyTo( imgStereo( Rect(cols,0,cols,rows) ) );

    return imgStereo;

}

Mat plSVO::imageInliers(){

    // Variables
    unsigned int    r, g, b = 0;
    Point2f         P, Poff = cv::Point2f(0.1,0.1);
    Point           sP, eP, mP;
    Matrix2f        w;
    float           maxP = 0.f, minP = 999999.f, aux, invP, invL, maxL = 0.f, minL = 999999.f, thick = 1.5f;
    int             k = 0, radius  = 3;

    // Represent stereo points
    for(unsigned int i = 0; i < nPoints; i++){
        if(inliers[i]){
            g = 255;
            r = 0;
            P = cv::Point(int(pData_full(3,i)),int(pData_full(4,i)));
            circle(imgFirst, P, radius, Scalar(b,g,r), thick);
        }
        else{
            g = 0;
            r = 255;
            P = cv::Point(int(pData_full(3,i)),int(pData_full(4,i)));
            circle(imgFirst, P, radius, Scalar(b,g,r), thick);
        }
    }

    // Represent stereo line segments
    k = 0;
    for(unsigned int i = nPoints; i < nPoints + nLines; i++)    {
        if(inliers[i]){
            g = 255;
            r = 0;
            sP = cv::Point(int(lData_full(13,k)),int(lData_full(14,k)));
            eP = cv::Point(int(lData_full(16,k)),int(lData_full(17,k)));
            mP   = cv::Point((sP.x+eP.x)/2,(sP.y+eP.y)/2);
            cv::line( imgFirst,sP,eP,Scalar(b,g,r));
            k++;
        }
        else{
            g = 0;
            r = 255;
            sP = cv::Point(int(lData_full(13,k)),int(lData_full(14,k)));
            eP = cv::Point(int(lData_full(16,k)),int(lData_full(17,k)));
            mP   = cv::Point((sP.x+eP.x)/2,(sP.y+eP.y)/2);
            cv::line( imgFirst,sP,eP,Scalar(b,g,r));
            k++;
        }
    }

    Mat imgFirst_half;
    resize(imgFirst, imgFirst_half, cv::Size(621,188) );

    return imgFirst;

}
