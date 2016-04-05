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

//using namespace std;
//#include <mrpt/opengl.h>
//#include <mrpt/gui.h>
//#include <mrpt/utils/CConfigFile.h>
//#include <mrpt/utils/CConfigFileBase.h>
//using namespace mrpt;
//using namespace mrpt::gui;
//using namespace mrpt::poses;
//using namespace mrpt::utils;
//using namespace mrpt::math;
//using namespace mrpt::opengl;
//#include <opencv/cv.h>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/line_descriptor.hpp>
//using namespace cv;
//using namespace cv::line_descriptor;
//#include <eigen3/Eigen/Core>
//using namespace Eigen;
//#include <stereoFrame.h>
//#include <pinholeStereoCamera.h>

#pragma once
#include <stereoFrame.h>
#include <stereoFeatures.h>

typedef Matrix<double,6,6> Matrix6d;
typedef Matrix<double,6,1> Vector6d;

class StereoFrame;

class StereoFrameHandler
{

public:

    StereoFrameHandler( PinholeStereoCamera* cam_ );
    ~StereoFrameHandler();

    void initialize( const Mat& img_l_, const Mat& img_r_, const int idx_);
    void insertStereoPair(const Mat& img_l_, const Mat& img_r_, const int idx_);
    void insertStereoPair(const Mat& img_l_, const Mat& img_r_, const int idx_, Matrix4d DT_ini);
    void processOdometry();
    void f2fTracking();
    void optimizePose();
    void updateFrame();

    int  n_inliers, n_inliers_pt, n_inliers_ls;

    list<PointFeature*> matched_pt;
    list<LineFeature*>  matched_ls;

    StereoFrame* prev_frame;
    StereoFrame* curr_frame;
    PinholeStereoCamera* cam;

private:

    void matchPointFeatures(BFMatcher* bfm, Mat pdesc_1, Mat pdesc_2, vector<vector<DMatch>> &pmatches_12  );   // TODO: remove (already implemented in stereoFrame.cpp)
    void matchLineFeatures(Ptr<BinaryDescriptorMatcher> bdm, Mat ldesc_1, Mat ldesc_2, vector<vector<DMatch>> &lmatches_12  ); // TODO: remove (already implemented in stereoFrame.cpp)

    void removeOutliers( Matrix4d DT );
    void gaussNewtonOptimization(Matrix4d &DT, Matrix6d &DT_cov);
    void optimizeFunctions_nonweighted(Matrix4d DT, Matrix6d &H, Vector6d &g, double &err);
    void optimizeFunctions_uncweighted(Matrix4d DT, Matrix6d &H, Vector6d &g, double &err);

};


