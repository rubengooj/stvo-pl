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

#include <config.h>

#define PI std::acos(-1.0)

Config::Config()
{

    // StVO-PL options
    // -----------------------------------------------------------------------------------------------------
    has_points         = true;      // true if using points
    has_lines          = true;      // true if using line segments
    use_edlines        = true;      // true if using EDLines false if using LSD
    lr_in_parallel     = true;      // true if detecting and matching features in parallel
    best_lr_matches    = true;      // true if double-checking the matches between the two images
    use_lev_marquardt  = false;     // true if Levenberg-Marquardt false if Gauss-Newton
    robust_cost        = true;      // true if employing the Cauchy loss function in the optimization
    scale_points_lines = true;      // true if scaling the influence of P and LS in the optimization
    use_uncertainty    = false;     // true if employing Gaussian uncertainty propagation
    motion_prior       = false;     // true if optimizing with prior information about the motion (i.e. IMU)

    // Tracking parameters
    // -----------------------------------------------------------------------------------------------------

    // Point features
    max_dist_epip    = 1.0;         // max. epipolar distance in pixels
    min_disp         = 1.0;         // min. disparity
    min_ratio_12_p   = 0.1;         // min. ratio between the first and second best matches
    max_f2f_disp     = 0.2;         // max. frame-to-frame disparity (relative to img size)

    // Line segment features
    min_line_length  = 0.015;       // min. line length (relative to img size)
    min_horiz_angle  = 5.0;         // min. angle to avoid horizontal lines
    max_angle_diff   = 10.0;        // max. angular difference between two stereo matches
    max_f2f_ang_diff = 10.0;        // max. angular difference between two f2f matches
    f2f_flow_th      = 100.0;       // max. distance between two f2f matches (pixels)
    line_horiz_th    = 0.1;         // parameter to avoid horizontal lines
    desc_th_l        = 0.5;         // parameter to avoid outliers in line matching

    // Optimization parameters
    // -----------------------------------------------------------------------------------------------------
    lambda_lm        = 0.001;       // initial lambda for Levenberg-Marquardt optimization
    lambda_k         = 4.0;         // factor to increment or decrement the lambda of LM
    homog_th         = 0.0000001;   // avoid points in the infinite
    min_features     = 10;          // min. number of features to perform StVO
    max_iters        = 5;           // max. number of iterations in the first stage of the optimization
    max_iters_ref    = 10;          // max. number of iterations in the refinement stage
    min_error        = 0.0000001;   // min. error to stop the optimization
    min_error_change = 0.0000001;   // min. error change to stop the optimization
    inlier_k         = 2.0;         // factor to discard outliers before the refinement stage
    sigma_px         = 1.0;         // expected standard deviation of features (if use_uncertainty)
    max_optim_error  = 100000.0;    // max. optimization error to consider a solution as good (disabled)


    // Feature detection parameters
    // -----------------------------------------------------------------------------------------------------

    // ORB detector
    orb_nfeatures    = 1200;
    orb_scale_factor = 1.2;
    orb_nlevels      = 1;

    // LSD parameters
    lsd_refine       = 2;
    lsd_scale        = 0.8;
    lsd_sigma_scale  = 0.6;
    lsd_quant        = 2.0;
    lsd_ang_th       = 22.5;
    lsd_log_eps      = 1.0;
    lsd_density_th   = 0.6;
    lsd_n_bins       = 1024;

    // EDLine parameters
    edl_ksize        = 15;
    edl_sigma        = 30.0;
    edl_gradient_th  = 80;
    edl_anchor_th    = 8;
    edl_scan_interv  = 2;
    edl_min_line_len = 15;
    edl_fit_err_th   = 1.6;



    // transform to radians some variables
    min_horiz_angle *= PI / 180.0;
    max_angle_diff  *= PI / 180.0;



}

Config::~Config(){}

Config& Config::getInstance()
{
  static Config instance; // Instantiated on first use and guaranteed to be destroyed
  return instance;
}
