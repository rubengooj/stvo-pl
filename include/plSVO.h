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

class plSVO{

public:

    // Constructors and destructor
    plSVO();
    ~plSVO();

    // Stereo Visual Odometry
    Matrix4f SVO(Mat imgL, Mat imgR, Matrix4f x_ini);

    // Tracking Functions
    void setStereo(Matrix3f K_, float baseline_);
    void initialize(Mat imgL, Mat imgR);
    void readImages(Mat imgL, Mat imgR);
    void detectLR();
    void detectL();
    void detectR();
    void stereoMatching();
    void f2fTracking();
    void updateTracking();
    void discardTracking();
    Mat  imageInliersWeights(Matrix4f x_optim);

    // Optimization Functions
    void setInitialValue(Matrix4f X);
    void getErrorNorm(float &errNorm_);
    void getItersNErrors(unsigned int &iters_, float &errNorm_);
    void getMatches(unsigned int &pMatches, unsigned int &pMatchesH, unsigned int &lMatches, unsigned int &lMatchesH);
    void getCovariance(MatrixXf &cov_);
    vector<bool> getInliers();
    MatrixXf getPOptimData();
    MatrixXf getLOptimData();
    void     svoOptim(Matrix4f &pose_);
    MatrixXf optimizeGN();
    MatrixXf optimizeRobustGN(VectorXf &residue_);
    MatrixXf erodeOptimizationGN();
    void     huberFunction(float &d, float &eNorm, float &res );
    void     removeERODE(VectorXf pResidue_);
    void     optimFunctions (MatrixXf &JtJ_, VectorXf &JtE_, float &errNorm_, VectorXf &E);
    void     optimFunctionsH(MatrixXf &JtJ_, VectorXf &JtE_, float &errNorm_, VectorXf &E);
    void     calculateWeights(Matrix4f x_optim);

private:

    // Variables
    BFMatcher         bfm;
    Mat               imgLeft, imgRight, imgFirst, imgSecond;
    Mat               pdescFirst, pdescSecond, pdescThird, pdescFourth,  pd1aux, pd2aux;
    Mat               ldescFirst, ldescSecond, ldescThird, ldescFourth,  ld1aux, ld2aux;
    vector<KeyPoint>  pointsFirst,pointsSecond,pointsThird,pointsFourth, p1aux,  p2aux;
    vector<KeyLine>   linesFirst, linesSecond, linesThird, linesFourth,  l1aux,  l2aux;
    MatrixXf          pData, lData;
    vector<bool>      inliers;
    VectorXf          JtErr, epsInc;
    MatrixXf          cov, JtJ, K_amp, W;
    Vector3f          e1, e2, e3;
    Matrix4f          param, prevParam, T, x_0;

    // Parameters
    unsigned int      nFeatures, nLevels, scale, nOctaves, matchTh, matchDist;
    float             minDist12, maxStDist, maxF2FDist, maxF2FDisp, scaleFactor, dispMax, alphaMax, horizTh, alphaTh, flowTh, matchPr, dispMin;
    unsigned int      state = 1, iters = 0, maxIters, maxItersH, nPoints, nPointsH, nLines, nLinesH, n, Np, Nl, Nmin;
    float             prevErrNorm, errNorm, paramChange, errorChange, huberTh, huberK, huberKL, huberKP, minErrorH, minErrorChangeH, minChange = numeric_limits<double>::epsilon(), homogTh, minError, minErrorChange;
    float             b, b2, sigmaP, sigmaL, bsigmaL, bsigmaL_inv, bsigmaP, bsigmaP_inv, sigmaP2, sigmaP2_inv, baseline, f, cx, cy;
    Matrix3f          K;

    // Flags
    bool              parallel, points, lines, uncertainty, overlap, erode;

};

