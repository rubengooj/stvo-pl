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

#include "plSVO.h"

/* Constructors and destructor */

plSVO::plSVO(){

    // Point Tracking Parameters
    minDist12       = 0.1f;             // min descriptor distance (threshold)
    maxStDist       = 0.f;              // max error in the y coordinate for the stereo correspondences
    maxF2FDisp      = 0.04f;            // max disparity (in proportion to the resolution) between the L-L and R-R correspondences in the F2F tracking
    nFeatures       = 500;              // max number of detected features
    nLevels         = 1;                // number of levels
    scaleFactor     = 1.2f;             // pyramid decimation ratio
    dispMin         = 1.f;              // min disparity value (avoid points in the infinite)

    // Line Tracking Parameters
    scale           = 1;                // scale for the LSD detector
    nOctaves        = 1;                // number of octaves where the lines are detected
    matchTh         = 40;               // match threshold
    matchDist       = 10;               // min distance between the best match and the next one
    dispMax         = 1000.f;           // Max. disparity to filter outliers in the stereo matching (pixels)
    alphaMax        = 10.f;             // Max. angle to filter outliers in the stereo matching (degrees)
    horizTh         = 0.5f;             // Parameter to remove the horizontal lines (pixels)
    flowTh          = 1000.f;           // Flow Optical Vector threshold for the frame2frame matching (pixels)
    alphaTh         = 10.f;             // Angle threshold for the frame2frame matching (degrees)
    matchPr         = 0.6f;             // Probability of inlier

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
    huberKL         = 1.5f;             // Parameter to obtain the Huber threshold and filter the outliers
    huberKP         = 2.5f;             // Parameter to obtain the Huber threshold and filter the outliers
    b               = 1.5f;             // Pseudo-Huber Loss Function parameter
    b2              = b * b;

    // Flags
    parallel        = false;            // detect the features in two separated threads for each image
    points          = true;             // introduce points in the system
    lines           = true;             // introduce line segments in the system
    uncertainty     = true;             // include uncertainty in the optimization
    overlap         = false;            // include the overlap measurement for the line segments (similarly to those employed in Witt's ICML)
    erode           = true;             // include the ERODE outlier rejection

    // Convert to radians
    alphaMax    = alphaMax * CV_PI / 180.f;
    alphaTh     = alphaTh  * CV_PI / 180.f;

    // Creation of the objects
    bfm = BFMatcher(NORM_HAMMING,false);                // Brute Force Matching declaration

    // Size of the point and line structures
    Np   = 11;
    Nl   = 23;
    Nmin = 8;

}

plSVO::~plSVO(){

}

/* Stereo Visual Odometry */

Matrix4f plSVO::SVO(Mat imgL, Mat imgR, Matrix4f x_ini){
    // Stereo Visual Odometry
    Matrix4f x_svo;
    readImages(imgL,imgR);
    detectLR();
    stereoMatching();
    f2fTracking();
    setInitialValue(x_ini.inverse());
    svoOptim(x_svo);
    if(!is_finite(x_svo)){
        return Matrix4f::Identity();
        cov = MatrixXf::Zero(6,6);
    }
    else
        return x_svo;
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
    // Detect the features in the first frame and save them in the structures
    detectLR();
    stereoMatching();
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
    if(points)
        orb->detectAndCompute(imgLeft, cv::Mat(),pointsThird, pdescThird,false);

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
    if(points)
        orb->detectAndCompute(imgRight, cv::Mat(),pointsFourth, pdescFourth,false);

}

void plSVO::stereoMatching(){

    /* Points Stereo Matching */
    vector<vector<DMatch>> pmatches;
    int m1, m2;
    // Match the stereo points
    if( (pointsThird.size()>0)&&(pointsFourth.size()>0) ){
        bfm.knnMatch(pdescThird,pdescFourth,pmatches,2);
        // Filter the bad matches
        for(unsigned int i = 0; i < pmatches.size(); i++){
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

void plSVO::f2fTracking(){

    /* Points Tracking */
    vector<vector<DMatch>>  matchesL, matchesR;
    int                     m1, m2, m3, m4, k = 0;
    MatrixXf                pData_ = MatrixXf::Zero(Np,pointsFirst.size());
    float                   bd;
    Vector3f                pointL, pointR, point3D;
    // Match the stereo points (we discard checking the L-R consistency since we performs a robust optimization and otherwise the number of points decrease critically)
    if( (pointsFirst.size()>0) && (pointsThird.size()>0) ){
        bfm.knnMatch(pdescFirst,pdescThird,matchesL,2);
        // Filter the bad matches
        for(unsigned int i = 0; i<matchesL.size(); i++){
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
                            epoint_L << linesFirst[m1].endPointX   , linesFirst[m1].endPointY ,   1.f;
                            spoint_R << linesSecond[m2].startPointX, linesSecond[m2].startPointY ,1.f;
                            epoint_R << linesSecond[m2].endPointX  , linesSecond[m2].endPointY ,  1.f;
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
    lData = lData_;
    nLines  = k;
    nLinesH = k;

    n = nLines + nPoints;

}

/* Optimization functions */

void plSVO::setInitialValue(Matrix4f X){
    x_0 = X;
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
            param  << transformation_expmap(epsInc).inverse() * prevParam;
            // If the parameter change is small stop, else if the parameter is big update weights
            if( diffManifoldError(param,prevParam) < minChange )
                break;
            // Update values
            prevErrNorm = errNorm;
            prevParam   = param;
        }
        cov = JtJ.inverse();
        return param.inverse();
    }
    else{
        cov = MatrixXf::Zero(6,6);
        return Matrix4f::Identity();
    }
}

MatrixXf plSVO::erodeOptimizationGN(){

    unsigned int maxIters_ = maxIters;
    VectorXf residue_;
    MatrixXf x_h, x_f;
    maxIters = maxItersH;
    x_h = optimizeRobustGN(residue_);

    if( is_finite(x_h) && (n>=Nmin) ){
        removeERODE(residue_);
        maxIters   = maxIters_;
        setInitialValue(x_h.inverse());
        x_f = optimizeRobustGN(residue_);
    }
    else{
        maxIters   = maxIters_;
        setInitialValue(x_0);
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
            param  << transformation_expmap(epsInc).inverse() * prevParam;
            // If the parameter change is small stop, else if the parameter is big update weights
            if( diffManifoldError(param,prevParam) < minChange )
                break;
            // Update values
            prevErrNorm = errNorm;
            prevParam   = param;
        }
        cov = JtJ.inverse();
        residue_ = E;
        return param.inverse();
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
                inliers.push_back(true);
                for(unsigned int j = 0; j < Np; j++){
                    pDataAux(j,k) = pData(j,i);
                }
                k++;
            }
            else
                inliers.push_back(false);
        }
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
        if((abs(p(2))>homogTh)||(abs(q(2))>homogTh)){
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
        if((abs(p(2))>homogTh)||(abs(q(2))>homogTh)){
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
            Eh(i) = 9999.f;
    }

    // ------ Final Functions
    JtJ_ << JtW * J;
    JtE_ << JtW * E;
    errNorm_ = errNorm_ * 0.5f / (float)n;

    E = VectorXf::Zero(n);
    E = Eh;
}

/* Representation functions */

void plSVO::calculateWeights(Matrix4f x_optim){

    W = MatrixXf::Identity(2*n,2*n);
    Matrix3f R;

    // ------ Point Features
    Vector3f p, ph;
    Vector4f P, TP;
    float    gx, gy, gz, gz2;
    R << x_optim.block(0,0,3,3);
    for(unsigned int i = 0; i < nPointsH; i++){
        // Project the 3D point to the Left image in the second frame
        P  << pData(0,i), pData(1,i), pData(2,i), 1.f;
        TP = x_optim * P;
        p << K_amp * TP;
        if( abs(p(2)) > homogTh ){
            ph << p / p(2);
            gx = TP(0);
            gy = TP(1);
            gz = TP(2);
            gz2 = gz*gz;
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
            W(2*i,2*i)     = covp_inv(0,0);
            W(2*i,2*i+1)   = covp_inv(0,1);
            W(2*i+1,2*i+1) = covp_inv(1,1);
            W(2*i+1,2*i)   = covp_inv(1,0);
        }
    }

    // ------ Line Features
    Vector3f L, Ln, q, qh;
    Vector4f Q, TQ;
    unsigned int j;
    for(unsigned int i = 0; i < nLinesH; i++){
        j = i + nPointsH;
        P  << lData(0,i), lData(1,i), lData(2,i), 1.f;
        Q  << lData(3,i), lData(4,i), lData(5,i), 1.f;
        Ln << lData(6,i), lData(7,i), lData(8,i);
        TP = x_optim * P;
        TQ = x_optim * Q;
        p << K_amp * TP;
        q << K_amp * TQ;
        if((abs(p(2))>homogTh)||(abs(q(2))>homogTh)){
            ph << p / p(2);
            qh << q / q(2);
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
            W(2*j,2*j) = cov_p;
            W(2*j+1,2*j+1) = cov_q;



        }

    }

}

Mat plSVO::imageInliersWeights(Matrix4f x_optim){

    // Variables
    unsigned int    r, g, b = 0;
    Point2f         P, Poff = cv::Point2f(0.1,0.1);
    Point           sP, eP, mP;
    Matrix2f        w;
    float           maxP = 0.f, minP = 999999.f, aux, invP, invL, maxL = 0.f, minL = 999999.f, thick = 1.5f;
    int             k = 0, radius  = 3;
    vector<float>   weights;

    // Estimates the weights
    calculateWeights(x_optim);

    // Estimates the maximum and minimum weight values
    for(unsigned int i = 0; i < nPointsH; i++){
        if(inliers[i]){
            w = W.block(2*k,2*k,2,2);
            aux = abs(w.determinant());
            weights.push_back(aux);
            if(aux>maxP)
                maxP = aux;
            if(aux<minP)
                minP = aux;
            k++;
        }
    }
    for(unsigned int i = nPointsH; i < nPointsH + nLinesH; i++){
        if(inliers[i]){
            w = W.block(2*k,2*k,2,2);
            aux = abs(w.determinant());
            weights.push_back(aux);
            if(aux>maxL)
                maxL = aux;
            if(aux<minL)
                minL = aux;
            k++;
        }
    }
    invP = maxP - minP;
    invP = 1.f/invP;
    invL = maxL - minL;
    invL = 1.f/invL;

    // Represent stereo points
    k = 0;
    for(unsigned int i = 0; i < nPoints; i++){
        if(inliers[i]){
            g = 255 * invP * ( weights[k] - minP );
            r = 255-g;
            P = cv::Point(int(pData(3,k)),int(pData(4,k)));
            circle(imgFirst, P, radius, Scalar(b,g,r), thick);
            k++;
        }
    }

    // Represent stereo line segments
    k = 0;
    for(unsigned int i = nPoints; i < nPoints + nLines; i++)    {
        if(inliers[i]){
            g = 255 * invL * ( weights[k] - minL );
            r = 255-g;
            sP = cv::Point(int(lData(13,k)),int(lData(14,k)));
            eP = cv::Point(int(lData(16,k)),int(lData(17,k)));
            mP   = cv::Point((sP.x+eP.x)/2,(sP.y+eP.y)/2);
            cv::line( imgFirst,sP,eP,Scalar(b,g,r));
            k++;
        }
    }

    return imgFirst;

}

