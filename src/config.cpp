#include <config.h>

#define PI std::acos(-1.0)

Config::Config()
{
    // flags
    has_points      = true;
    has_lines       = false;
    lr_in_parallel  = true;
    robust_cost     = true;
    motion_prior    = false;

    // points detection and matching
    orb_nfeatures    = 2000;
    orb_scale_factor = 1.0;
    orb_nlevels      = 1;
    max_dist_epip    = 1.0;
    min_disp         = 10.0;
    desc_th_p        = 1.5;

    // lines detection and matching
    lsd_refine       = 2;   // 0 NONE - 1 STD - 2 ADV
    lsd_scale        = 1.0;
    lsd_sigma_scale  = 0.8;
    lsd_quant        = 2.0;
    lsd_ang_th       = 22.5;
    lsd_log_eps      = 1.0;
    lsd_density_th   = 0.6;
    lsd_n_bins       = 1024;

    min_horiz_angle  = 10.0;
    max_angle_diff   = 10.0;
    line_horiz_th    = 0.1;
    min_line_length  = 0.03;
    desc_th_l        = 1.5;

    // transform to radians
    min_horiz_angle *= PI / 180.0;
    max_angle_diff  *= PI / 180.0;

    // transform with the relative size of image

    // optimization
    homog_th         = 0.0000001;
    min_features     = 20;
    max_iters        = 3;
    max_iters_ref    = 10;
    min_error        = 0.0000001;
    min_error_change = 0.0000001;
    inlier_k         = 1.5;
    sigma_px         = 0.1;

}

#pragma message("TODO: include definition with config file and yaml")

Config::~Config(){}

Config& Config::getInstance()
{
  static Config instance; // Instantiated on first use and guaranteed to be destroyed
  return instance;
}
