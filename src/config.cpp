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

    // flags
    has_points         = true;
    has_lines          = true;
    lr_in_parallel     = true;
    robust_cost        = true;
    best_lr_matches    = true;
    use_edlines        = true;
    scale_points_lines = true;
    use_lev_marquardt  = true;
    use_uncertainty    = false;
    motion_prior       = false;

    // points detection and matching
    orb_nfeatures    = 1200;
    orb_scale_factor = 1.2;
    orb_nlevels      = 1;
    max_dist_epip    = 1.0;
    min_disp         = 1.0;
    min_ratio_12_p   = 0.1;
    max_f2f_disp     = 0.2;   // relative to img size

    // lines detection and matching
    lsd_refine       = 2;   // 0 NONE - 1 STD - 2 ADV
    lsd_scale        = 0.8;
    lsd_sigma_scale  = 0.6;
    lsd_quant        = 2.0;
    lsd_ang_th       = 22.5;
    lsd_log_eps      = 1.0;
    lsd_density_th   = 0.6;
    lsd_n_bins       = 1024;

    edl_ksize        = 15;
    edl_sigma        = 30.0;
    edl_gradient_th  = 80;
    edl_anchor_th    = 8;
    edl_scan_interv  = 2;
    edl_min_line_len = 15;
    edl_fit_err_th   = 1.6;

    min_line_length  = 0.015; // relative to img size
    min_horiz_angle  = 5.0;
    max_angle_diff   = 10.0;
    max_f2f_ang_diff = 10.0;
    f2f_flow_th      = 100.0;
    line_horiz_th    = 0.1;
    desc_th_l        = 0.5;

    // transform to radians
    min_horiz_angle *= PI / 180.0;
    max_angle_diff  *= PI / 180.0;

    // optimization
    lambda_lm        = 0.001;
    lambda_k         = 5.0;
    homog_th         = 0.0000001;
    min_features     = 10;
    max_iters        = 5;
    max_iters_ref    = 20;
    min_error        = 0.0000001;
    min_error_change = 0.0000001;
    inlier_k         = 2.0;
    sigma_px         = 1.0;
    max_optim_error  = 10.0;

}

#pragma message("TODO: include definition with config file and yaml")

Config::~Config(){}

Config& Config::getInstance()
{
  static Config instance; // Instantiated on first use and guaranteed to be destroyed
  return instance;
}
