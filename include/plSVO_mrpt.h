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

#include <mutex>
#include <thread>
#include <iomanip>
using namespace std;

#include <mrpt/opengl.h>
#include <mrpt/gui.h>
#include <mrpt/utils/CConfigFile.h>
#include <mrpt/utils/CConfigFileBase.h>
using namespace mrpt;
using namespace mrpt::gui;
using namespace mrpt::poses;
using namespace mrpt::utils;
using namespace mrpt::math;
using namespace mrpt::opengl;

#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/line_descriptor.hpp>
using namespace cv;
using namespace cv::line_descriptor;

#include <eigen3/Eigen/Core>
using namespace Eigen;

#include <kinematics.h>

class plSVO{

public:

    // Constructors and destructor
    plSVO();
    plSVO(string configFile);
    ~plSVO();

    // Stereo Visual Odometry
    Matrix4f SVO(Mat imgL, Mat imgR, Matrix4f x_ini);

    // Tracking Functions
    void setStereo(Matrix3f K_, float baseline_);
    void initialize(Mat imgL, Mat imgR);
    void setInitialize(Mat imgL, Mat imgR, vector<KeyPoint> kpL, Mat dpL, vector<KeyLine> klL, Mat dlL, vector<KeyPoint> kpR, Mat dpR, vector<KeyLine> klR, Mat dlR);
    void getInitialize(Mat imgL, Mat imgR, vector<KeyPoint> &kpL, Mat &dpL, vector<KeyLine> &klL, Mat &dlL, vector<KeyPoint> &kpR, Mat &dpR, vector<KeyLine> &klR, Mat &dlR);
    void readImages(Mat imgL, Mat imgR);
    void detectLR();
    void detectL();
    void detectR();
    void stereoMatching();
    void stereoMutualMatching();
    void f2fTracking();
    void f2fMutualTracking();
    void updateTracking();
    void discardTracking();
    Mat  imageInliersWeights();
    Mat  imageTracking();
    Mat  imageInliers();

    // Setters ICRA16
    void setUncertainty(string uncertainty_);
    void setNLevels(string nLevels_);    
    void setBucketing(bool bucketing_);
    void setMinDist12(float minDist12_);
    void setMaxStDist(float maxStDisp_);
    void setMaxF2FDisp(float maxF2FDisp_);
    void setDispMin(float dispMin_);
    void setScale(float scale_);
    void setAlphaMax(float alphaMax_);
    void setAlphaTh(float alphaTh_);
    void setHorizTh(float horizTh_);
    void setMatchPr(float matchPr_);
    void setWeights(bool gamma_);
    void setOptimMode(unsigned int optimMode_);

    void getFeaturesL(vector<KeyPoint> &kp, Mat &dp, vector<KeyLine> &kl, Mat &dl);
    void getFeaturesR(vector<KeyPoint> &kp, Mat &dp, vector<KeyLine> &kl, Mat &dl);
    void setFeaturesL(vector<KeyPoint> kp, Mat dp, vector<KeyLine> kl, Mat dl);
    void setFeaturesR(vector<KeyPoint> kp, Mat dp, vector<KeyLine> kl, Mat dl);

    // Optimization Functions
    void setInitialValue(Matrix4f X);
    void setInitialValueNPrior(Matrix4f x_0_, MatrixXf prevPoseInc_, MatrixXf prevCov_);
    void getErrorNorm(float &errNorm_);
    void getItersNErrors(unsigned int &iters_, float &errNorm_);
    void getMatches(unsigned int &pMatches, unsigned int &pMatchesH, unsigned int &lMatches, unsigned int &lMatchesH);
    void getCovariance(MatrixXf &cov_);    
    void detect1L();
    void detect2L();
    void detect3L();
    void detect4L();
    void detect1R();
    void detect2R();
    void detect3R();
    void detect4R();
    vector<bool> getInliers();
    MatrixXf getPOptimData();
    void     setPOptimData(MatrixXf pData_);
    MatrixXf getLOptimData();
    void     svoOptim(Matrix4f &pose_);
    void     svoOptim_new(Matrix4f &pose_);
    void     svoOptim_new_(Matrix4f &x_f);
    MatrixXf optimizeGN();
    MatrixXf optimizeGN_new(VectorXf &residue_);
    MatrixXf optimizeRobustGN(VectorXf &residue_);
    MatrixXf erodeOptimizationGN();
    MatrixXf optimizeRobustGN_new(VectorXf &residue_);
    void     huberFunction(float &d, float &eNorm, float &res );
    void     huberFunction2(float r, float &rH, float &dH );
    void     huberFunction3(float r, float &rH, float &dH );
    void     removeERODE(VectorXf pResidue_);
    void     removeERODE_new(VectorXf pResidue_);
    void     fastFilter(VectorXf pResidue_);
    void     optimFunctions (MatrixXf &JtJ_, VectorXf &JtE_, float &errNorm_, VectorXf &E);
    void     optimFunctionsH(MatrixXf &JtJ_, VectorXf &JtE_, float &errNorm_, VectorXf &E);
    void     estimateTStudentWeights(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_);
    void     estimateTStudentWeights2(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_);
    void     estimateGammaWeights(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_);
    void     estimateGammaWeights2(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_);
    void     estimateGammaWeights3(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_);

    void     estimateGammaWeights_edo(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_);
    void     estimateGammaWeights2_edo(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_);
    void     estimateGammaWeights3_edo(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_);
    void     estimateGammaWeights4_edo(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_);

    void     optimFunctions_new(MatrixXf &JtJ_, VectorXf &JtE_, float &errNorm_, VectorXf &E);
    void     optimFunctionsRobust_new(MatrixXf &JtJ_, VectorXf &JtE_, float &errNorm_, VectorXf &E_, VectorXf &Eh);
    void     estimateGaussianWeights(VectorXf &weights, MatrixXf &JtW_, float &errNorm_);
    void     estimateGaussianWeights2(VectorXf &residues, VectorXf &weights, MatrixXf &JtW_, float &errNorm_);

    void     svoInliers(Matrix4f pose_inc, string &inliers_ratio);

private:

    // Variables
    BFMatcher         bfm;
    Mat               imgLeft, imgRight, imgFirst, imgSecond;
    Mat               pdescFirst, pdescSecond, pdescThird, pdescFourth,  pd1aux, pd2aux, pdesc1L, pdesc1R, pdesc2L, pdesc3L, pdesc4L, pdesc2R, pdesc3R, pdesc4R;
    Mat               ldescFirst, ldescSecond, ldescThird, ldescFourth,  ld1aux, ld2aux, ldesc1L, ldesc2L, ldesc3L, ldesc4L, ldesc1R, ldesc2R, ldesc3R, ldesc4R;
    vector<KeyPoint>  pointsFirst,pointsSecond,pointsThird,pointsFourth, p1aux,  p2aux,  points1L, points1R, points2L, points2R, points3L, points3R, points4L, points4R;
    vector<KeyLine>   linesFirst, linesSecond, linesThird, linesFourth,  l1aux,  l2aux,  lines1L, lines2L, lines3L, lines4L, lines1R, lines2R, lines3R, lines4R;
    MatrixXf          pData, lData, pData_full, lData_full;
    vector<bool>      inliers;
    VectorXf          JtErr, epsInc, W, E;
    MatrixXf          cov, JtJ, K_amp;
    Vector3f          e1, e2, e3;
    Matrix4f          param, prevParam, T, x_0;

    VectorXf          prevPoseInc;
    MatrixXf          prevCov;

    // Parameters
    unsigned int      nFeatures, nFeaturesB, nLevels, scale, nOctaves, c, r, C, R;
    float             minDist12, maxStDist, maxF2FDist, maxF2FDisp, scaleFactor, dispMax, alphaMax, horizTh, alphaTh, flowTh, matchPr, dispMin;
    unsigned int      state = 1, iters = 0, maxIters, maxItersH, nPoints, nPointsH, nLines, nLinesH, n, Np, Nl, Nmin;
    float             prevErrNorm, errNorm, paramChange, errorChange, huberTh, huberK, huberKL, huberKP, minErrorH, minErrorChangeH, minChange = numeric_limits<double>::epsilon(), homogTh, minError, minErrorChange;
    float             b, b2, sigmaP, sigmaL, bsigmaL, bsigmaL_inv, bsigmaP, bsigmaP_inv, sigmaP2, sigmaP2_inv, baseline, f, cx, cy;
    Matrix3f          K;

    // Flags
    bool              parallel, bucketing, points, lines, uncertainty, overlap, erode, gamma, motionprior;
    unsigned int      optimMode;

};

