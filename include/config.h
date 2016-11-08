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
#include <cmath>
#include <string>

using namespace std;

class Config
{

public:

    Config();
    ~Config();

    static Config& getInstance();

    // flags
    static bool&    isOutdoor()         { return getInstance().is_outdoor; }
    static bool&    hasPoints()         { return getInstance().has_points; }
    static bool&    hasLines()          { return getInstance().has_lines; }
    static bool&    lrInParallel()      { return getInstance().lr_in_parallel; }
    static bool&    robustCost()        { return getInstance().robust_cost; }
    static bool&    motionPrior()       { return getInstance().motion_prior; }
    static bool&    bestLRMatches()     { return getInstance().best_lr_matches; }
    static bool&    useBRISK()          { return getInstance().use_brisk; }
    static bool&    useEDLines()        { return getInstance().use_edlines; }
    static bool&    scalePointsLines()  { return getInstance().scale_points_lines; }
    static bool&    useUncertainty()    { return getInstance().use_uncertainty; }
    static bool&    useBFMLines()       { return getInstance().use_bfm_lines; }

    // points detection and matching
    static int&     orbNFeatures()      { return getInstance().orb_nfeatures; }
    static double&  orbScaleFactor()    { return getInstance().orb_scale_factor; }
    static int&     orbNLevels()        { return getInstance().orb_nlevels; }
    static int&     brsThreshold()      { return getInstance().brs_threshold; }
    static double&  brsScaleFactor()    { return getInstance().brs_scale_factor; }
    static int&     brsNLevels()        { return getInstance().brs_nlevels; }
    static double&  maxDistEpip()       { return getInstance().max_dist_epip; }
    static double&  minDisp()           { return getInstance().min_disp; }
    static double&  minRatio12P()       { return getInstance().min_ratio_12_p; }
    static double&  maxF2FDisp()        { return getInstance().max_f2f_disp; }

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
    static double&  maxF2FAngDiff()     { return getInstance().max_f2f_ang_diff; }
    static double&  f2fFlowTh()         { return getInstance().f2f_flow_th; }
    static double&  lineHorizTh()       { return getInstance().line_horiz_th; }
    static double&  minLineLength()     { return getInstance().min_line_length; }
    static double&  descThL()           { return getInstance().desc_th_l; }
    static double&  minRatio12L()       { return getInstance().min_ratio_12_l; }
    static int&     edlKsize()          { return getInstance().edl_ksize; }
    static double&  edlSigma()          { return getInstance().edl_sigma; }
    static int&     edlGradientTh()     { return getInstance().edl_gradient_th; }
    static int&     edlAnchorTh()       { return getInstance().edl_anchor_th; }
    static int&     edlScanInterv()     { return getInstance().edl_scan_interv; }
    static int&     edlMinLineLen()     { return getInstance().edl_min_line_len; }
    static double&  edlFitErrTh()       { return getInstance().edl_fit_err_th; }
    static double&  lineCovTh()         { return getInstance().line_cov_th; }

    // optimization
    static double&  lambdaLM()          { return getInstance().lambda_lm; }
    static double&  lambdaK()           { return getInstance().lambda_k; }
    static double&  homogTh()           { return getInstance().homog_th; }
    static int&     minFeatures()       { return getInstance().min_features; }
    static int&     maxIters()          { return getInstance().max_iters; }
    static int&     maxItersRef()       { return getInstance().max_iters_ref; }
    static double&  minError()          { return getInstance().min_error; }
    static double&  minErrorChange()    { return getInstance().min_error_change; }
    static double&  inlierK()           { return getInstance().inlier_k; }
    static double&  sigmaPx()           { return getInstance().sigma_px; }
    static double&  maxOptimError()     { return getInstance().max_optim_error; }
    static double&  maxCovEigval()      { return getInstance().max_cov_eigval; }

private:

    // SLAM parameters
    double min_entropy_ratio;
    int    max_kf_num_frames;
    double min_kf_t_dist;
    double min_kf_r_dist;
    double max_kf_t_dist;
    double max_kf_r_dist;
    int    min_kf_n_feats;
    double max_kf_epip_p;
    double max_kf_epip_l;
    int    min_lm_cov_graph;
    int    min_lm_ess_graph;
    double max_lm_3d_err;
    double max_lm_dir_err;

    double lambda_lba_lm;
    double lambda_lba_k;
    int    max_iters_lba;
    int    min_lm_obs;
    double max_common_fts_kf;

    string vocabulary_p, vocabulary_l;
    double lc_res;
    double lc_unc;
    double lc_inl;
    double lc_trs;
    double lc_rot;
    double lc_mat;
    int    max_iters_pgo;

    int    lc_kf_dist;
    int    lc_kf_max_dist;
    int    lc_nkf_closest;
    double lc_dbow_score_max;
    double lc_dbow_score_min;
    double lc_inlier_ratio;

    // flags
    bool has_points;
    bool has_lines;
    bool lr_in_parallel;
    bool robust_cost;
    bool motion_prior;
    bool best_lr_matches;
    bool use_brisk;
    bool use_edlines;
    bool scale_points_lines;
    bool use_uncertainty;
    bool is_outdoor;
    bool use_bfm_lines;

    // points detection and matching
    int    orb_nfeatures;
    double orb_scale_factor;
    int    orb_nlevels;
    int    brs_threshold;
    double brs_scale_factor;
    int    brs_nlevels;
    double max_dist_epip;
    double min_disp;
    double min_ratio_12_p;
    double max_f2f_disp;

    // lines detection and matching
    int    lsd_refine;
    double lsd_scale;
    double lsd_sigma_scale;
    double lsd_quant;
    double lsd_ang_th;
    double lsd_log_eps;
    double lsd_density_th;
    int    lsd_n_bins;
    int    edl_ksize;
    double edl_sigma;
    int    edl_gradient_th;
    int    edl_anchor_th;
    int    edl_scan_interv;
    int    edl_min_line_len;
    double edl_fit_err_th;
    double min_horiz_angle;
    double max_angle_diff;
    double max_f2f_ang_diff;
    double f2f_flow_th;
    double line_horiz_th;
    double min_line_length;
    double desc_th_l;
    double min_ratio_12_l;
    double line_cov_th;

    // optimization
    double lambda_lm;
    double lambda_k;
    double homog_th;
    int    min_features;
    int    max_iters;
    int    max_iters_ref;
    double min_error;
    double min_error_change;
    double inlier_k;
    double sigma_px;
    double max_optim_error;
    double max_cov_eigval;

};

