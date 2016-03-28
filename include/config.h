#pragma once
#include <cmath>

class Config
{

public:

    Config();
    ~Config();

    static Config& getInstance();

    // flags
    static bool&    hasPoints()         { return getInstance().has_points; }
    static bool&    hasLines()          { return getInstance().has_lines; }
    static bool&    lrInParallel()      { return getInstance().lr_in_parallel; }
    static bool&    robustCost()        { return getInstance().robust_cost; }
    static bool&    motionPrior()       { return getInstance().motion_prior; }
    static bool&    bestLRMatches()     { return getInstance().best_lr_matches; }
    static bool&    fundMatrixFilter()  { return getInstance().fund_matrix_filter; }

    // points detection and matching
    static int&     orbNFeatures()      { return getInstance().orb_nfeatures; }
    static double&  orbScaleFactor()    { return getInstance().orb_scale_factor; }
    static int&     orbNLevels()        { return getInstance().orb_nlevels; }
    static double&  maxDistEpip()       { return getInstance().max_dist_epip; }
    static double&  minDisp()           { return getInstance().min_disp; }
    static double&  descThP()           { return getInstance().desc_th_p; }

    // lines detection and matching
    static int&     lsdRefine()         { return getInstance().lsd_refine; }
    static double&  lsdScale()          { return getInstance().lsd_scale; }
    static double&  lsdSigmaScale()     { return getInstance().lsd_sigma_scale; }
    static double&  lsdQuant()          { return getInstance().lsd_quant; }
    static double&  lsdAngTh()          { return getInstance().lsd_ang_th; }
    static double&  lsdLogEps()         { return getInstance().lsd_log_eps; }
    static double&  lsdDensityTh()      { return getInstance().lsd_density_th; }
    static int&     lsdNBins()          { return getInstance().lsd_n_bins; }

    static double&  minHorizAngle()     { return getInstance().min_horiz_angle; }
    static double&  maxAngleDiff()      { return getInstance().max_angle_diff; }
    static double&  lineHorizTh()       { return getInstance().line_horiz_th; }
    static double&  minLineLength()     { return getInstance().min_line_length; }
    static double&  descThL()           { return getInstance().desc_th_l; }

    // optimization
    static double&  homogTh()           { return getInstance().homog_th; }
    static int&     minFeatures()       { return getInstance().min_features; }
    static int&     maxIters()          { return getInstance().max_iters; }
    static int&     maxItersRef()       { return getInstance().max_iters_ref; }
    static double&  minError()          { return getInstance().min_error; }
    static double&  minErrorChange()    { return getInstance().min_error_change; }
    static double&  inlierK()           { return getInstance().inlier_k; }
    static double&  sigmaPx()           { return getInstance().sigma_px; }

private:

    // flags
    bool has_points;
    bool has_lines;
    bool lr_in_parallel;
    bool robust_cost;
    bool motion_prior;
    bool best_lr_matches;
    bool fund_matrix_filter;

    // points detection and matching
    int    orb_nfeatures;
    double orb_scale_factor;
    int    orb_nlevels;

    double max_dist_epip;
    double min_disp;
    double desc_th_p;

    // lines detection and matching
    int    lsd_refine;
    double lsd_scale;
    double lsd_sigma_scale;
    double lsd_quant;
    double lsd_ang_th;
    double lsd_log_eps;
    double lsd_density_th;
    int    lsd_n_bins;

    double min_horiz_angle;
    double max_angle_diff;
    double line_horiz_th;
    double min_line_length;
    double desc_th_l;

    // optimization
    double homog_th;
    int    min_features;
    int    max_iters;
    int    max_iters_ref;
    double min_error;
    double min_error_change;
    double inlier_k;
    double sigma_px;

};

