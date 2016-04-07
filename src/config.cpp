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
    //motion_prior       = false;
    //fund_matrix_filter = false;

    // points detection and matching
    orb_nfeatures    = 1200;
    orb_scale_factor = 1.2;
    orb_nlevels      = 1;
    max_dist_epip    = 1.0;
    min_disp         = 1.0;
    min_ratio_12_p   = 0.1;
    max_f2f_disp     = 0.2;   // relative to img size
    //desc_th_p        = 2.0;

    // lines detection and matching
    lsd_refine       = 2;   // 0 NONE - 1 STD - 2 ADV
    lsd_scale        = 0.8;
    lsd_sigma_scale  = 0.6;
    lsd_quant        = 2.0;
    lsd_ang_th       = 22.5;
    lsd_log_eps      = 1.0;
    lsd_density_th   = 0.6;
    lsd_n_bins       = 1024;

    min_line_length  = 0.015; // relative to img size
    min_horiz_angle  = 5.0;
    max_angle_diff   = 10.0;
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
    max_optim_error  = 999999.0;

}

#pragma message("TODO: include definition with config file and yaml")

Config::~Config(){}

Config& Config::getInstance()
{
  static Config instance; // Instantiated on first use and guaranteed to be destroyed
  return instance;
}
