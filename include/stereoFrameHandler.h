/*****************************************************************************
**   Stereo Visual Odometry by combining point and line segment features	**
******************************************************************************
**																			**
**	Copyright(c) 2016, Ruben Gomez-Ojeda, University of Malaga              **
**	Copyright(c) 2016, MAPIR group, University of Malaga					**
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

#pragma once
#include <stereoFrame.h>
#include <stereoFeatures.h>

typedef Matrix<double,6,6> Matrix6d;
typedef Matrix<double,6,1> Vector6d;

class StereoFrame;

namespace StVO{

class StereoFrameHandler
{

public:

    StereoFrameHandler( PinholeStereoCamera* cam_ );
    ~StereoFrameHandler();

    void initialize( const Mat& img_l_, const Mat& img_r_, const int idx_);
    void insertStereoPair(const Mat& img_l_, const Mat& img_r_, const int idx_);
    void insertStereoPair(const Mat& img_l_, const Mat& img_r_, const int idx_, Matrix4d DT_ini);
    void f2fTracking();
    void optimizePose();
    void optimizePose(Matrix4d DT_ini);
    void updateFrame();
    void setMotionPrior(Vector6d prior_inc_, Matrix6d prior_cov_);

    int  n_inliers, n_inliers_pt, n_inliers_ls, max_idx_pt, max_idx_ls, max_idx_pt_prev_kf, max_idx_ls_prev_kf;

    list<PointFeature*> matched_pt;
    list<LineFeature*>  matched_ls;

    StereoFrame* prev_keyframe;
    StereoFrame* prev_frame;
    StereoFrame* curr_frame;
    PinholeStereoCamera* cam;

    Vector6d prior_inc;
    Matrix6d prior_cov;

private:

    void matchPointFeatures(BFMatcher* bfm, Mat pdesc_1, Mat pdesc_2, vector<vector<DMatch>> &pmatches_12  );
    void matchLineFeatures(Ptr<BinaryDescriptorMatcher> bdm, Mat ldesc_1, Mat ldesc_2, vector<vector<DMatch>> &lmatches_12  );
    void removeOutliers( Matrix4d DT );
    void gaussNewtonOptimization(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters);
    void levMarquardtOptimization(Matrix4d &DT, Matrix6d &DT_cov, double &err_, int max_iters);
    void optimizeFunctions_nonweighted(Matrix4d DT, Matrix6d &H, Vector6d &g, double &e);
    void optimizeFunctions_uncweighted(Matrix4d DT, Matrix6d &H, Vector6d &g, double &e);

};

}
