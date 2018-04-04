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

#include <stereoFrame.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

#include "lineIterator.h"
#include "matching.h"

namespace StVO{

/* Constructor and main method */

StereoFrame::StereoFrame(){}

StereoFrame::StereoFrame(const Mat img_l_, const Mat img_r_ , const int idx_, PinholeStereoCamera *cam_) :
    img_l(img_l_), img_r(img_r_), frame_idx(idx_), cam(cam_) {

    if (img_l_.size != img_r_.size)
        throw std::runtime_error("[StereoFrame] Left and right images have different sizes");

    inv_width  = GRID_COLS / static_cast<double>(img_l.cols);
    inv_height = GRID_ROWS / static_cast<double>(img_l.rows);
}

StereoFrame::~StereoFrame()
{
    for( auto pt: stereo_pt )
        delete pt;
    for( auto ls: stereo_ls )
        delete ls;
}

void StereoFrame::extractStereoFeatures( double llength_th, int fast_th )
{

    if( Config::plInParallel() )
    {
        auto detect_p = async(launch::async, &StereoFrame::detectStereoPoints,        this, fast_th );
        auto detect_l = async(launch::async, &StereoFrame::detectStereoLineSegments,  this, llength_th );
        detect_p.wait();
        detect_l.wait();
    }
    else
    {
        detectStereoPoints(fast_th);
        detectStereoLineSegments(llength_th);
    }

}

/* Stereo point features extraction */

void StereoFrame::detectStereoPoints( int fast_th )
{

    if( !Config::hasPoints() )
        return;

    // detect and estimate each descriptor for both the left and right image
    if( Config::lrInParallel() )
    {
        auto detect_l = async(launch::async, &StereoFrame::detectPointFeatures, this, img_l, ref(points_l), ref(pdesc_l), fast_th );
        auto detect_r = async(launch::async, &StereoFrame::detectPointFeatures, this, img_r, ref(points_r), ref(pdesc_r), fast_th );
        detect_l.wait();
        detect_r.wait();
    }
    else
    {
        detectPointFeatures( img_l, points_l, pdesc_l, fast_th );
        detectPointFeatures( img_r, points_r, pdesc_r, fast_th );
    }

    // perform the stereo matching
    matchStereoPoints(points_l, points_r, pdesc_l, pdesc_r, (frame_idx==0) );

}

void StereoFrame::detectPointFeatures( Mat img, vector<KeyPoint> &points, Mat &pdesc, int fast_th )
{
    // Detect point features
    if( Config::hasPoints() )
    {
        int fast_th_ = Config::orbFastTh();
        if( fast_th != 0 )
            fast_th_ = fast_th;
        Ptr<ORB> orb = ORB::create( Config::orbNFeatures(), Config::orbScaleFactor(), Config::orbNLevels(),
                                    Config::orbEdgeTh(), 0, Config::orbWtaK(), Config::orbScore(),
                                    Config::orbPatchSize(), fast_th_ );
        orb->detectAndCompute( img, Mat(), points, pdesc, false);
    }

}

void StereoFrame::matchStereoPoints( vector<KeyPoint> points_l, vector<KeyPoint> points_r, Mat &pdesc_l_, Mat pdesc_r, bool initial )
{

    // Points stereo matching
    // --------------------------------------------------------------------------------------------------------------------
    stereo_pt.clear();
    if (!Config::hasPoints() || points_l.empty() || points_r.empty())
        return;

    std::vector<point_2d> coords;
    coords.reserve(points_l.size());
    for (const KeyPoint &kp : points_l)
        coords.push_back(std::make_pair(kp.pt.x * inv_width, kp.pt.y * inv_height));

    //Fill in grid
    GridStructure grid(GRID_ROWS, GRID_COLS);
    for (int idx = 0; idx < points_r.size(); ++idx) {
        const KeyPoint &kp = points_r[idx];
        grid.at(kp.pt.x * inv_width, kp.pt.y * inv_height).push_back(idx);
    }

    GridWindow w;
    w.width = std::make_pair(Config::matchingSWs(), 0);
    w.height = std::make_pair(0, 0);

    std::vector<int> matches_12;
    matchGrid(coords, pdesc_l, grid, pdesc_r, w, matches_12);
//    match(pdesc_l, pdesc_r, Config::minRatio12P(), matches_12);

    // bucle around pmatches
    Mat pdesc_l_aux;
    int pt_idx = 0;
    for (int i1 = 0; i1 < matches_12.size(); ++i1) {
        const int i2 = matches_12[i1];
        if (i2 < 0) continue;

        // check stereo epipolar constraint
        if (std::abs(points_l[i1].pt.y - points_r[i2].pt.y) <= Config::maxDistEpip()) {
            // check minimal disparity
            double disp_ = points_l[i1].pt.x - points_r[i2].pt.x;
            if (disp_ >= Config::minDisp()){
                pdesc_l_aux.push_back(pdesc_l_.row(i1));
                Vector2d pl_(points_l[i1].pt.x, points_l[i1].pt.y);
                Vector3d P_ = cam->backProjection(pl_(0), pl_(1), disp_);
                if (initial)
                    stereo_pt.push_back(new PointFeature(pl_, disp_, P_, pt_idx++, points_l[i1].octave));
                else
                    stereo_pt.push_back(new PointFeature(pl_, disp_, P_, -1, points_l[i1].octave));
            }
        }
    }

    pdesc_l_ = pdesc_l_aux;
}

void StereoFrame::matchPointFeatures(BFMatcher* bfm, Mat pdesc_1, Mat pdesc_2, vector<vector<DMatch>> &pmatches_12  )
{
    bfm->knnMatch( pdesc_1, pdesc_2, pmatches_12, 2);
}

/* Stereo line segment features extraction */

void StereoFrame::detectStereoLineSegments(double llength_th)
{

    if( !Config::hasLines() )
        return;

    // detect and estimate each descriptor for both the left and right image
    if( Config::lrInParallel() )
    {
        auto detect_l = async(launch::async, &StereoFrame::detectLineFeatures, this, img_l, ref(lines_l), ref(ldesc_l), llength_th );
        auto detect_r = async(launch::async, &StereoFrame::detectLineFeatures, this, img_r, ref(lines_r), ref(ldesc_r), llength_th );
        detect_l.wait();
        detect_r.wait();
    }
    else
    {
        detectLineFeatures( img_l, lines_l, ldesc_l, llength_th );
        detectLineFeatures( img_r, lines_r, ldesc_r, llength_th );
    }

    // perform the stereo matching
    matchStereoLines(lines_l,  lines_r,  ldesc_l, ldesc_r, (frame_idx==0));

}

void StereoFrame::detectLineFeatures( Mat img, vector<KeyLine> &lines, Mat &ldesc, double min_line_length )
{

    // Detect line features
    lines.clear();
    Ptr<BinaryDescriptor>   lbd = BinaryDescriptor::createBinaryDescriptor();
    if( Config::hasLines() )
    {

        if( !Config::useFLDLines() )
        {
            Ptr<line_descriptor::LSDDetectorC> lsd = line_descriptor::LSDDetectorC::createLSDDetectorC();
            // lsd parameters
            line_descriptor::LSDDetectorC::LSDOptions opts;
            opts.refine       = Config::lsdRefine();
            opts.scale        = Config::lsdScale();
            opts.sigma_scale  = Config::lsdSigmaScale();
            opts.quant        = Config::lsdQuant();
            opts.ang_th       = Config::lsdAngTh();
            opts.log_eps      = Config::lsdLogEps();
            opts.density_th   = Config::lsdDensityTh();
            opts.n_bins       = Config::lsdNBins();
            opts.min_length   = min_line_length;
            lsd->detect( img, lines, Config::lsdScale(), 1, opts);
            // filter lines
            if( lines.size()>Config::lsdNFeatures() && Config::lsdNFeatures()!=0  )
            {
                // sort lines by their response
                sort( lines.begin(), lines.end(), sort_lines_by_response() );
                //sort( lines.begin(), lines.end(), sort_lines_by_length() );
                lines.resize(Config::lsdNFeatures());
                // reassign index
                for( int i = 0; i < Config::lsdNFeatures(); i++  )
                    lines[i].class_id = i;
            }
            lbd->compute( img, lines, ldesc);
        }
        else
        {
            Mat fld_img, img_gray;
            vector<Vec4f> fld_lines;

            if( img.channels() != 1 )
            {
                cv::cvtColor( img, img_gray, CV_RGB2GRAY );
                img_gray.convertTo( fld_img, CV_8UC1 );
            }
            else
                img.convertTo( fld_img, CV_8UC1 );

            Ptr<cv::ximgproc::FastLineDetector> fld = cv::ximgproc::createFastLineDetector(min_line_length);
            fld->detect( fld_img, fld_lines );

            // filter lines
            if( fld_lines.size()>Config::lsdNFeatures() && Config::lsdNFeatures()!=0  )
            {
                // sort lines by their response
                sort( fld_lines.begin(), fld_lines.end(), sort_flines_by_length() );
                fld_lines.resize(Config::lsdNFeatures());
            }

            // loop over lines object transforming into a vector<KeyLine>
            lines.reserve(fld_lines.size());
            for( int i = 0; i < fld_lines.size(); i++ )
            {
                KeyLine kl;
                double octaveScale = 1.f;
                int    octaveIdx   = 0;

                kl.startPointX     = fld_lines[i][0] * octaveScale;
                kl.startPointY     = fld_lines[i][1] * octaveScale;
                kl.endPointX       = fld_lines[i][2] * octaveScale;
                kl.endPointY       = fld_lines[i][3] * octaveScale;

                kl.sPointInOctaveX = fld_lines[i][0];
                kl.sPointInOctaveY = fld_lines[i][1];
                kl.ePointInOctaveX = fld_lines[i][2];
                kl.ePointInOctaveY = fld_lines[i][3];

                kl.lineLength = (float) sqrt( pow( fld_lines[i][0] - fld_lines[i][2], 2 ) + pow( fld_lines[i][1] - fld_lines[i][3], 2 ) );

                kl.angle    = atan2( ( kl.endPointY - kl.startPointY ), ( kl.endPointX - kl.startPointX ) );
                kl.class_id = i;
                kl.octave   = octaveIdx;
                kl.size     = ( kl.endPointX - kl.startPointX ) * ( kl.endPointY - kl.startPointY );
                kl.pt       = Point2f( ( kl.endPointX + kl.startPointX ) / 2, ( kl.endPointY + kl.startPointY ) / 2 );

                kl.response = kl.lineLength / max( fld_img.cols, fld_img.rows );
                cv::LineIterator li( fld_img, Point2f( fld_lines[i][0], fld_lines[i][1] ), Point2f( fld_lines[i][2], fld_lines[i][3] ) );
                kl.numOfPixels = li.count;

                lines.push_back( kl );

            }

            // compute lbd descriptor
            lbd->compute( fld_img, lines, ldesc);
        }

    }
}

void StereoFrame::matchStereoLines( vector<KeyLine> lines_l, vector<KeyLine> lines_r, Mat &ldesc_l_, Mat ldesc_r, bool initial )
{

    // Line segments stereo matching
    // --------------------------------------------------------------------------------------------------------------------
    stereo_ls.clear();
    if (!Config::hasLines() || lines_l.empty() || lines_r.empty())
        return;

    std::vector<line_2d> coords;
    coords.reserve(lines_l.size());
    for (const KeyLine &kl : lines_l)
        coords.push_back(std::make_pair(std::make_pair(kl.startPointX * inv_width, kl.startPointY * inv_height),
                                        std::make_pair(kl.endPointX * inv_width, kl.endPointY * inv_height)));

    //Fill in grid & directions
    list<pair<int, int>> line_coords;
    GridStructure grid(GRID_ROWS, GRID_COLS);
    std::vector<std::pair<double, double>> directions(lines_r.size());
    for (int idx = 0; idx < lines_r.size(); ++idx) {
        const KeyLine &kl = lines_r[idx];

        std::pair<double, double> &v = directions[idx];
        v = std::make_pair((kl.endPointX - kl.startPointX) * inv_width, (kl.endPointY - kl.startPointY) * inv_height);
        normalize(v);

        getLineCoords(kl.startPointX * inv_width, kl.startPointY * inv_height, kl.endPointX * inv_width, kl.endPointY * inv_height, line_coords);
        for (const std::pair<int, int> &p : line_coords)
            grid.at(p.first, p.second).push_back(idx);
    }

    GridWindow w;
    w.width = std::make_pair(Config::matchingSWs(), 0);
    w.height = std::make_pair(0, 0);

    std::vector<int> matches_12;
    matchGrid(coords, ldesc_l, grid, ldesc_r, directions, w, matches_12);
//    match(ldesc_l, ldesc_r, Config::minRatio12P(), matches_12);

    // bucle around lmatches
    Mat ldesc_l_aux;
    int ls_idx = 0;
    for (int i1 = 0; i1 < matches_12.size(); ++i1) {
        const int i2 = matches_12[i1];
        if (i2 < 0) continue;

        // estimate the disparity of the endpoints
        Vector3d sp_l; sp_l << lines_l[i1].startPointX, lines_l[i1].startPointY, 1.0;
        Vector3d ep_l; ep_l << lines_l[i1].endPointX,   lines_l[i1].endPointY,   1.0;
        Vector3d le_l; le_l << sp_l.cross(ep_l); le_l = le_l / std::sqrt( le_l(0)*le_l(0) + le_l(1)*le_l(1) );
        Vector3d sp_r; sp_r << lines_r[i2].startPointX, lines_r[i2].startPointY, 1.0;
        Vector3d ep_r; ep_r << lines_r[i2].endPointX,   lines_r[i2].endPointY,   1.0;
        Vector3d le_r; le_r << sp_r.cross(ep_r);

        double overlap = lineSegmentOverlapStereo( sp_l(1), ep_l(1), sp_r(1), ep_r(1) );

        double disp_s, disp_e;
        sp_r << ( sp_r(0)*( sp_l(1) - ep_r(1) ) + ep_r(0)*( sp_r(1) - sp_l(1) ) ) / ( sp_r(1)-ep_r(1) ) , sp_l(1) ,  1.0;
        ep_r << ( sp_r(0)*( ep_l(1) - ep_r(1) ) + ep_r(0)*( sp_r(1) - ep_l(1) ) ) / ( sp_r(1)-ep_r(1) ) , ep_l(1) ,  1.0;
        filterLineSegmentDisparity( sp_l.head(2), ep_l.head(2), sp_r.head(2), ep_r.head(2), disp_s, disp_e );

        // check minimal disparity
        if( disp_s >= Config::minDisp() && disp_e >= Config::minDisp()
            && std::abs( sp_l(1)-ep_l(1) ) > Config::lineHorizTh()
            && std::abs( sp_r(1)-ep_r(1) ) > Config::lineHorizTh()
            && overlap > Config::stereoOverlapTh() )
        {
            Vector3d sP_; sP_ = cam->backProjection( sp_l(0), sp_l(1), disp_s);
            Vector3d eP_; eP_ = cam->backProjection( ep_l(0), ep_l(1), disp_e);
            double angle_l = lines_l[i1].angle;
            if( initial )
            {
                ldesc_l_aux.push_back( ldesc_l_.row(i1) );
                stereo_ls.push_back( new LineFeature(Vector2d(sp_l(0),sp_l(1)),disp_s,sP_,
                                                     Vector2d(ep_l(0),ep_l(1)),disp_e,eP_,
                                                     le_l,angle_l,ls_idx,lines_l[i1].octave) );
                ls_idx++;
            }
            else
            {
                ldesc_l_aux.push_back( ldesc_l_.row(i1) );
                stereo_ls.push_back( new LineFeature(Vector2d(sp_l(0),sp_l(1)),disp_s,sP_,
                                                     Vector2d(ep_l(0),ep_l(1)),disp_e,eP_,
                                                     le_l,angle_l,-1,lines_l[i1].octave) );
            }
        }
    }

    ldesc_l_aux.copyTo(ldesc_l_);
}

void StereoFrame::matchLineFeatures(BFMatcher* bfm, Mat ldesc_1, Mat ldesc_2, vector<vector<DMatch>> &lmatches_12  )
{
    bfm->knnMatch( ldesc_1, ldesc_2, lmatches_12, 2);
}

void StereoFrame::filterLineSegmentDisparity( Vector2d spl, Vector2d epl, Vector2d spr, Vector2d epr, double &disp_s, double &disp_e )
{
    disp_s = spl(0) - spr(0);
    disp_e = epl(0) - epr(0);
    // if they are too different, ignore them
    if(  min( disp_s, disp_e ) / max( disp_s, disp_e ) < Config::lsMinDispRatio() )
    {
        disp_s = -1.0;
        disp_e = -1.0;
    }
}

/* Auxiliar methods */

void StereoFrame::pointDescriptorMAD( const vector<vector<DMatch>> matches, double &nn_mad, double &nn12_mad )
{

    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = matches;
    matches_12 = matches;

    // estimate the NN's distance standard deviation
    double nn_dist_median;
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist() );
    nn_mad = matches_nn[int(matches_nn.size()/2)][0].distance;
    for( int j = 0; j < matches_nn.size(); j++)
        matches_nn[j][0].distance = fabsf( matches_nn[j][0].distance - nn_dist_median );
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist() );
    nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

    // estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort( matches_12.begin(), matches_12.end(), compare_descriptor_by_NN12_ratio() );
    nn_dist_median = matches_12[int(matches_12.size()/2)][0].distance / matches_12[int(matches_12.size()/2)][1].distance;
    for( int j = 0; j < matches_12.size(); j++)
        matches_12[j][0].distance = fabsf( matches_12[j][0].distance / matches_12[j][1].distance - nn_dist_median );
    sort( matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist() );
    nn12_mad =  1.4826 * matches_12[int(matches_12.size()/2)][0].distance;

}

void StereoFrame::lineDescriptorMAD( const vector<vector<DMatch>> matches, double &nn_mad, double &nn12_mad )
{

    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = matches;
    matches_12 = matches;

    // estimate the NN's distance standard deviation
    double nn_dist_median;
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist() );
    nn_mad = matches_nn[int(matches_nn.size()/2)][0].distance;
    for( int j = 0; j < matches_nn.size(); j++)
        matches_nn[j][0].distance = fabsf( matches_nn[j][0].distance - nn_dist_median );
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist() );
    nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

    // estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort( matches_12.begin(), matches_12.end(), compare_descriptor_by_NN12_dist() );
    nn12_mad = matches_12[int(matches_12.size()/2)][1].distance - matches_12[int(matches_12.size()/2)][0].distance;
    for( int j = 0; j < matches_12.size(); j++)
        matches_12[j][0].distance = fabsf( matches_12[j][1].distance - matches_12[j][0].distance - nn_dist_median );
    sort( matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist() );
    nn12_mad =  1.4826 * matches_12[int(matches_12.size()/2)][0].distance;

}

double StereoFrame::lineSegmentOverlapStereo( double spl_obs, double epl_obs, double spl_proj, double epl_proj  )
{

    double overlap = 1.f;

    if( fabs( epl_obs - spl_obs ) > Config::lineHorizTh() ) // normal lines (verticals included)
    {
        double sln    = min(spl_obs,  epl_obs);
        double eln    = max(spl_obs,  epl_obs);
        double spn    = min(spl_proj, epl_proj);
        double epn    = max(spl_proj, epl_proj);

        double length = eln-spn;

        if ( (epn < sln) || (spn > eln) )
            overlap = 0.f;
        else{
            if ( (epn>eln) && (spn<sln) )
                overlap = eln-sln;
            else
                overlap = min(eln,epn) - max(sln,spn);
        }

        if(length>0.01f)
            overlap = overlap / length;
        else
            overlap = 0.f;

        if( overlap > 1.f )
            overlap = 1.f;

    }

    return overlap;

}

double StereoFrame::lineSegmentOverlap( Vector2d spl_obs, Vector2d epl_obs, Vector2d spl_proj, Vector2d epl_proj  )
{

    double overlap = 1.f;

    if( fabs(spl_obs(0)-epl_obs(0)) < 1.0 )         // vertical lines
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        spl_proj_line << spl_obs(0), spl_proj(1);
        epl_proj_line << epl_obs(0), epl_proj(1);

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(1)-spl_obs(1)) / l(1);
        double lambda_e = (epl_proj_line(1)-spl_obs(1)) / l(1);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }
    else if( fabs(spl_obs(1)-epl_obs(1)) < 1.0 )    // horizontal lines (previously removed)
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        spl_proj_line << spl_proj(0), spl_obs(1);
        epl_proj_line << epl_proj(0), epl_obs(1);

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(0)-spl_obs(0)) / l(0);
        double lambda_e = (epl_proj_line(0)-spl_obs(0)) / l(0);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }
    else                                            // non-degenerate cases
    {

        // line equations
        Vector2d l = epl_obs - spl_obs;
        double a = spl_obs(1)-epl_obs(1);
        double b = epl_obs(0)-spl_obs(0);
        double c = spl_obs(0)*epl_obs(1) - epl_obs(0)*spl_obs(1);

        // intersection points
        Vector2d spl_proj_line, epl_proj_line;
        double lxy = 1.f / (a*a+b*b);

        spl_proj_line << ( b*( b*spl_proj(0)-a*spl_proj(1))-a*c ) * lxy,
                         ( a*(-b*spl_proj(0)+a*spl_proj(1))-b*c ) * lxy;

        epl_proj_line << ( b*( b*epl_proj(0)-a*epl_proj(1))-a*c ) * lxy,
                         ( a*(-b*epl_proj(0)+a*epl_proj(1))-b*c ) * lxy;

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(0)-spl_obs(0)) / l(0);
        double lambda_e = (epl_proj_line(0)-spl_obs(0)) / l(0);

        double lambda_min = min(lambda_s,lambda_e);
        double lambda_max = max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }

    return overlap;

}

Mat StereoFrame::plotStereoFrame()
{

    // create new image to modify it
    Mat img_l_aux;
    img_l.copyTo( img_l_aux );
    if( img_l_aux.channels() == 1 )
        cvtColor(img_l_aux, img_l_aux, CV_GRAY2BGR, 3);
    else if (img_l_aux.channels() == 4)
        cvtColor(img_l_aux, img_l_aux, CV_BGRA2BGR, 3);
    else if (img_l_aux.channels() != 3)
        throw std::runtime_error(std::string("[StereoFrame->plotStereoFrame] unsupported image format: ") +
                                 std::to_string(img_l_aux.channels()));
    img_l_aux.convertTo(img_l_aux, CV_8UC3);

    // Variables
    unsigned int    r = 0, g, b = 0;
    Point2f         p,q;
    double          thick = 1.5;
    int             k = 0, radius  = 3;

    // plot point features
    for( auto pt_it = stereo_pt.begin(); pt_it != stereo_pt.end(); pt_it++)
    {
        if( (*pt_it)->inlier )
        {
            g = 200;
            p = cv::Point( int((*pt_it)->pl(0)), int((*pt_it)->pl(1)) );
            circle( img_l_aux, p, radius, Scalar(b,g,r), thick);
        }
    }

    // plot line segment features
    for( auto ls_it = stereo_ls.begin(); ls_it != stereo_ls.end(); ls_it++)
    {
        if( (*ls_it)->inlier )
        {
            g = 200;
            p = cv::Point( int((*ls_it)->spl(0)), int((*ls_it)->spl(1)) );
            q = cv::Point( int((*ls_it)->epl(0)), int((*ls_it)->epl(1)) );
            line( img_l_aux, p, q, Scalar(b,g,r), thick);
        }
    }

    return img_l_aux;
}

/* RGB-D Functions */

void StereoFrame::extractRGBDFeatures( double llength_th, int fast_th )
{

    bool initial = (frame_idx == 0);

    // Feature detection and description
    vector<KeyPoint> points_l;
    vector<KeyLine>  lines_l;
    if(  Config::hasPoints() && Config::hasLines() )
    {
        if( Config::plInParallel() )
        {
            auto detect_p = async(launch::async, &StereoFrame::detectPointFeatures, this, img_l, ref(points_l), ref(pdesc_l), fast_th );
            auto detect_l = async(launch::async, &StereoFrame::detectLineFeatures,  this, img_l, ref(lines_l), ref(ldesc_l), llength_th );
            detect_p.wait();
            detect_l.wait();
        }
        else
        {
            detectPointFeatures( img_l, points_l, pdesc_l, fast_th );
            detectLineFeatures( img_l, lines_l, ldesc_l, llength_th );
        }
    }
    else
    {
        if( Config::hasPoints() )
            detectPointFeatures( img_l, points_l, pdesc_l, fast_th );
        else
            detectLineFeatures( img_l, lines_l, ldesc_l, llength_th );
    }

    // Points stereo matching
    if( Config::hasPoints() && !(points_l.size()==0) )
    {
        // bucle around pmatches
        stereo_pt.clear();
        int pt_idx = 0;
        Mat pdesc_l_;
        for( int i = 0; i < points_l.size(); i++ )
        {
            if( img_r.type() == CV_32FC1 )
            {
                // read the depth for each point and estimate disparity
                float depth  = img_r.at<float>(points_l[i].pt.y,points_l[i].pt.x);
                // check correct depth values
                if( depth > Config::rgbdMinDepth() && depth < Config::rgbdMaxDepth() )
                {
                    // check minimal disparity
                    double disp   = cam->getFx() * cam->getB() / depth; // TUM factor (read also if in the ICL-NUIM is different)
                    if( disp >= Config::minDisp() ){
                        pdesc_l_.push_back( pdesc_l.row(i) );
                        Vector2d pl_; pl_ << points_l[i].pt.x, points_l[i].pt.y;
                        Vector3d P_;  P_ = cam->backProjection( pl_(0), pl_(1), disp);
                        stereo_pt.push_back( new PointFeature(pl_,disp,P_,-1) );
                        pt_idx++;
                    }
                }

            }
            else if( img_r.type() == CV_16UC1 )
            {
                // read the depth for each point and estimate disparity
                ushort depth  = img_r.at<ushort>(points_l[i].pt.y,points_l[i].pt.x);
                double depthd = double(depth/5000.0) ;
                // check correct depth values
                if( depthd > Config::rgbdMinDepth() && depthd < Config::rgbdMaxDepth() )
                {
                    // check minimal disparity
                    double disp   = cam->getFx() * cam->getB() / depthd; // TUM factor (read also if in the ICL-NUIM is different)
                    if( disp >= Config::minDisp() ){
                        pdesc_l_.push_back( pdesc_l.row(i) );
                        Vector2d pl_; pl_ << points_l[i].pt.x, points_l[i].pt.y;
                        Vector3d P_;  P_ = cam->backProjection( pl_(0), pl_(1), disp);
                        stereo_pt.push_back( new PointFeature(pl_,disp,P_,-1) );
                        pt_idx++;
                    }
                }
            }
        }
        pdesc_l_.copyTo(pdesc_l);
    }

    // Line segments stereo matching
    if( Config::hasLines() && !lines_l.empty() )
    {
        stereo_ls.clear();
        Mat ldesc_l_;
        int ls_idx = 0;
        for( int i = 0; i < lines_l.size(); i++ )
        {
            if( img_r.type() == CV_32FC1 )
            {
                // read the depth for each point and estimate disparity
                float depth_s  = img_r.at<float>(lines_l[i].pt.y,lines_l[i].pt.x);
                float depth_e  = img_r.at<float>(lines_l[i].pt.y,lines_l[i].pt.x);
                // discard points with bad depth estimation
                if( depth_s > Config::rgbdMinDepth() && depth_s < Config::rgbdMaxDepth() && depth_e > Config::rgbdMinDepth() && depth_e < Config::rgbdMaxDepth() )
                {
                    // estimate the disparity of the endpoints
                    double disp_s = cam->getFx() * cam->getB() / depth_s;
                    double disp_e = cam->getFx() * cam->getB() / depth_e;
                    filterLineSegmentDisparity(disp_s,disp_e);
                    Vector3d sp_l; sp_l << lines_l[i].startPointX, lines_l[i].startPointY, 1.0;
                    Vector3d ep_l; ep_l << lines_l[i].endPointX,   lines_l[i].endPointY,   1.0;
                    Vector3d le_l; le_l << sp_l.cross(ep_l); le_l = le_l / sqrt( le_l(0)*le_l(0) + le_l(1)*le_l(1) );
                    // check minimal disparity
                    if( disp_s >= Config::minDisp() && disp_e >= Config::minDisp() )
                    {
                        ldesc_l_.push_back( ldesc_l.row(i) );
                        Vector3d sP_; sP_ = cam->backProjection( sp_l(0), sp_l(1), disp_s);
                        Vector3d eP_; eP_ = cam->backProjection( ep_l(0), ep_l(1), disp_e);
                        double angle_l = lines_l[i].angle;
                        stereo_ls.push_back( new LineFeature(Vector2d(sp_l(0),sp_l(1)),disp_s,sP_,Vector2d(ep_l(0),ep_l(1)),disp_e,eP_,le_l,angle_l,-1) );
                        ls_idx++;
                    }
                }
            }
            else //if( img_r.type() == CV_16UC1 )
            {
                // read the depth for each point and estimate disparity
                ushort depth_s  = img_r.at<ushort>(lines_l[i].pt.y,lines_l[i].pt.x);
                ushort depth_e  = img_r.at<ushort>(lines_l[i].pt.y,lines_l[i].pt.x);
                double depthd_s = double(depth_s/5000.0) ;
                double depthd_e = double(depth_e/5000.0) ;
                // discard points with bad depth estimation
                if( depthd_s > Config::rgbdMinDepth() && depthd_s < Config::rgbdMaxDepth() &&  depthd_e > Config::rgbdMinDepth() && depthd_e < Config::rgbdMaxDepth() )
                {
                    // estimate the disparity of the endpoints
                    double disp_s = cam->getFx() * cam->getB() / depthd_s;
                    double disp_e = cam->getFx() * cam->getB() / depthd_e;
                    filterLineSegmentDisparity(disp_s,disp_e);
                    Vector3d sp_l; sp_l << lines_l[i].startPointX, lines_l[i].startPointY, 1.0;
                    Vector3d ep_l; ep_l << lines_l[i].endPointX,   lines_l[i].endPointY,   1.0;
                    Vector3d le_l; le_l << sp_l.cross(ep_l); le_l = le_l / sqrt( le_l(0)*le_l(0) + le_l(1)*le_l(1) );
                    // check minimal disparity
                    if( disp_s >= Config::minDisp() && disp_e >= Config::minDisp() )
                    {
                        ldesc_l_.push_back( ldesc_l.row(i) );
                        Vector3d sP_; sP_ = cam->backProjection( sp_l(0), sp_l(1), disp_s);
                        Vector3d eP_; eP_ = cam->backProjection( ep_l(0), ep_l(1), disp_e);
                        double angle_l = lines_l[i].angle;
                        stereo_ls.push_back( new LineFeature(Vector2d(sp_l(0),sp_l(1)),disp_s,sP_,Vector2d(ep_l(0),ep_l(1)),disp_e,eP_,le_l,angle_l,-1) );
                        ls_idx++;
                    }
                }
            }

        }
        ldesc_l_.copyTo(ldesc_l);
    }

}

void StereoFrame::filterLineSegmentDisparity( double &disp_s, double &disp_e )
{

    // TODO: ask David for bresenham to filter the line with the depth along it!!!!

    // if they are too different, ignore them
    if(  min( disp_s, disp_e ) / max( disp_s, disp_e ) < Config::lsMinDispRatio() )
    {
        disp_s = -1.0;
        disp_e = -1.0;
    }
}

}
