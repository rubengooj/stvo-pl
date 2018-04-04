/*****************************************************************************
**      Stereo VO and SLAM by combining point and line segment features     **
******************************************************************************
**                                                                          **
**  Copyright(c) 2016-2018, Ruben Gomez-Ojeda, University of Malaga         **
**  Copyright(c) 2016-2018, David Zuñiga-Noël, University of Malaga         **
**  Copyright(c) 2016-2018, MAPIR group, University of Malaga               **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify    **
**  it under the terms of the GNU General Public License (version 3) as     **
**  published by the Free Software Foundation.                              **
**                                                                          **
**  This program is distributed in the hope that it will be useful, but     **
**  WITHOUT ANY WARRANTY; without even the implied warranty of              **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            **
**  GNU General Public License for more details.                            **
**                                                                          **
**  You should have received a copy of the GNU General Public License       **
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.   **
**                                                                          **
*****************************************************************************/

#pragma once

#include <future>
#include <thread>
#include <time.h>
#include <set>
#include <utility>
using namespace std;

#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/fast_line_detector.hpp>

#include <line_descriptor_custom.hpp>
#include <line_descriptor/descriptor_custom.hpp>
using namespace cv;
using namespace line_descriptor;

#include <eigen3/Eigen/Core>
using namespace Eigen;

#include <config.h>
#include <stereoFeatures.h>
#include <pinholeStereoCamera.h>
#include <auxiliar.h>

#define GRID_ROWS 48
#define GRID_COLS 64

typedef Matrix<double,6,6> Matrix6d;
typedef Matrix<double,6,1> Vector6d;

namespace StVO{

class StereoFrame
{

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StereoFrame();
    StereoFrame(const Mat img_l_, const Mat img_r_, const int idx_, PinholeStereoCamera *cam_ );
    ~StereoFrame();

    void extractStereoFeatures( double llength_th, int fast_th = 20 );
    void extractRGBDFeatures(   double llength_th, int fast_th = 20 );

    void detectStereoPoints(int fast_th = 20);
    void detectPointFeatures( Mat img, vector<KeyPoint> &points, Mat &pdesc, int fast_th = 20 );
    void matchStereoPoints(vector<KeyPoint> points_l, vector<KeyPoint> points_r, Mat &pdesc_l_, Mat pdesc_r, bool initial = false );
    void matchPointFeatures(BFMatcher* bfm, Mat pdesc_1, Mat pdesc_2, vector<vector<DMatch>> &pmatches_12);

    void detectStereoLineSegments(double llength_th);
    void detectLineFeatures( Mat img, vector<KeyLine> &lines, Mat &ldesc, double min_line_length );
    void matchStereoLines(vector<KeyLine> lines_l, vector<KeyLine> lines_r, Mat &ldesc_l_, Mat ldesc_r, bool initial = false );
    void matchLineFeatures(BFMatcher* bfm, Mat ldesc_1, Mat ldesc_2, vector<vector<DMatch>> &lmatches_12 );
    void filterLineSegmentDisparity(Vector2d spl, Vector2d epl, Vector2d spr, Vector2d epr , double &disp_s, double &disp_e);
    void filterLineSegmentDisparity(double &disp_s, double &disp_e);

    double lineSegmentOverlapStereo( double spl_obs, double epl_obs, double spl_proj, double epl_proj  );
    double lineSegmentOverlap( Vector2d spl_obs, Vector2d epl_obs, Vector2d spl_proj, Vector2d epl_proj  );
    void pointDescriptorMAD( const vector<vector<DMatch>> matches, double &nn_mad, double &nn12_mad );
    void lineDescriptorMAD( const vector<vector<DMatch>> matches, double &nn_mad, double &nn12_mad );

    Mat  plotStereoFrame();

    int frame_idx;
    Mat img_l, img_r;
    Matrix4d Tfw;
    Matrix4d DT;

    Matrix6d Tfw_cov;
    Vector6d Tfw_cov_eig;
    double   entropy_first;

    Matrix6d DT_cov;
    Vector6d DT_cov_eig;
    double   err_norm;

    vector< PointFeature* > stereo_pt;
    vector< LineFeature*  > stereo_ls;

    vector<KeyPoint> points_l, points_r;
    vector<KeyLine>  lines_l,  lines_r;
    Mat pdesc_l, pdesc_r, ldesc_l, ldesc_r;

    PinholeStereoCamera *cam;

    double inv_width, inv_height; // grid cell
};

}
