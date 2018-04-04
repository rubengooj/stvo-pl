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

#include <stereoFeatures.h>

namespace StVO{

// Point feature

PointFeature::PointFeature( Vector3d P_, Vector2d pl_obs_) :
    P(P_), pl_obs(pl_obs_), level(0)
{}

PointFeature::PointFeature( Vector2d pl_, double disp_, Vector3d P_ ) :
    pl(pl_), disp(disp_), P(P_), inlier(true), level(0)
{}

PointFeature::PointFeature( Vector2d pl_, double disp_, Vector3d P_, int idx_ ) :
    pl(pl_), disp(disp_), P(P_), inlier(true), idx(idx_), level(0)
{}

PointFeature::PointFeature( Vector2d pl_, double disp_, Vector3d P_, int idx_, int level_ ) :
    pl(pl_), disp(disp_), P(P_), inlier(true), idx(idx_), level(level_)
{
    for( int i = 0; i < level; i++ )
        sigma2 *= Config::orbScaleFactor();
    sigma2 = 1.f / (sigma2*sigma2);
}

PointFeature::PointFeature( Vector2d pl_, double disp_, Vector3d P_, int idx_, int level_, Matrix3d covP_an_ ) :
    pl(pl_), disp(disp_), P(P_), inlier(true), idx(idx_), level(level_), covP_an(covP_an_)
{
    for( int i = 0; i < level; i++ )
        sigma2 *= Config::orbScaleFactor();
    sigma2 = 1.f / (sigma2*sigma2);
}

PointFeature::PointFeature( Vector2d pl_, double disp_, Vector3d P_, Vector2d pl_obs_ ) :
    pl(pl_), disp(disp_), P(P_), pl_obs(pl_obs_), inlier(true), level(0)
{}

PointFeature::PointFeature( Vector2d pl_, double disp_, Vector3d P_, Vector2d pl_obs_,
              int idx_, int level_, double sigma2_, Matrix3d covP_an_, bool inlier_ ) :
    pl(pl_), disp(disp_), P(P_), pl_obs(pl_obs_), inlier(inlier_), level(level_), sigma2(sigma2_), covP_an(covP_an_)
{}

PointFeature* PointFeature::safeCopy(){
    return new PointFeature( pl, disp, P, pl_obs, idx, level, sigma2, covP_an, inlier );
}



// Line segment feature

LineFeature::LineFeature( Vector3d sP_, Vector3d eP_, Vector3d le_obs_) :
    sP(sP_), eP(eP_), le_obs(le_obs_), level(0)
{}


LineFeature::LineFeature( Vector3d sP_, Vector3d eP_, Vector3d le_obs_, Vector2d spl_obs_, Vector2d epl_obs_) :
    sP(sP_), eP(eP_), le_obs(le_obs_), spl_obs(spl_obs_), epl_obs(epl_obs_), level(0)
{}


LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                          Vector2d epl_, double edisp_, Vector3d eP_, Vector3d le_) :
    spl(spl_), sdisp(sdisp_), sP(sP_), epl(epl_), edisp(edisp_), eP(eP_), le(le_), inlier(true), level(0)
{}

LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                          Vector2d epl_, double edisp_, Vector3d eP_,
                          Vector3d le_, Vector3d le_obs_ ) :
    spl(spl_), sdisp(sdisp_), sP(sP_), epl(epl_), edisp(edisp_), eP(eP_), le(le_), le_obs(le_obs_), inlier(true), level(0)
{}

LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                          Vector2d epl_, double edisp_, Vector3d eP_,
                          Vector3d le_,  int    idx_) :
    spl(spl_), sdisp(sdisp_), sP(sP_), epl(epl_), edisp(edisp_), eP(eP_), le(le_), inlier(true), idx(idx_), level(0)
{}

LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                          Vector2d epl_, double edisp_, Vector3d eP_,
                          Vector3d le_,  double angle_, int    idx_) :
    spl(spl_), sdisp(sdisp_), sP(sP_), epl(epl_), edisp(edisp_), eP(eP_), le(le_), inlier(true), idx(idx_), angle(angle_), level(0)
{}

LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                          Vector2d epl_, double edisp_, Vector3d eP_,
                          Vector3d le_,  double angle_, int idx_, int level_) :
    spl(spl_), sdisp(sdisp_), sP(sP_), epl(epl_), edisp(edisp_), eP(eP_), le(le_), inlier(true), idx(idx_), angle(angle_), level(level_)
{
    for( int i = 0; i < level; i++ )
        sigma2 *= Config::lsdScale();
    sigma2 = 1.f / (sigma2*sigma2);
}

LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_, Vector2d spl_obs_, double sdisp_obs_,
                          Vector2d epl_, double edisp_, Vector3d eP_, Vector2d epl_obs_, double edisp_obs_,
                          Vector3d le_, Vector3d le_obs_, double angle_, int idx_, int level_, bool inlier_, double sigma2_,
                          Matrix3d covE_an_, Matrix3d covS_an_) :

    spl(spl_), sdisp(sdisp_), sP(sP_), spl_obs(spl_obs_), sdisp_obs(sdisp_obs_),
    epl(epl_), edisp(edisp_), eP(eP_), epl_obs(epl_obs_), edisp_obs(edisp_obs_),
    le(le_), le_obs(le_obs_), angle(angle_), idx(idx_), level(level_), inlier(inlier_), sigma2(sigma2_), covE_an(covE_an_), covS_an(covS_an_)
{
    for( int i = 0; i < level; i++ )
        sigma2 *= Config::lsdScale();
    sigma2 = 1.f / (sigma2*sigma2);
}

LineFeature* LineFeature::safeCopy(){
    return new LineFeature( spl, sdisp, sP, spl_obs, sdisp_obs,
                            epl, edisp, eP, epl_obs, edisp_obs,
                            le, le_obs, angle, idx, level, inlier, sigma2, covE_an, covS_an );
}

/*LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                          Vector2d epl_, double edisp_, Vector3d eP_,
                          Vector3d le_,  double angle_, int idx_, int level_, Vector2d spr_, Vector2d epr_ ) :
    spl(spl_), sdisp(sdisp_), sP(sP_), epl(epl_), edisp(edisp_), eP(eP_), le(le_), inlier(true), idx(idx_), angle(angle_), level(level_), spr(spr_), epr(epr_)
{
    for( int i = 0; i < level+1; i++ )
        sigma2 *= Config::lsdScale();
    sigma2 = 1.f / (sigma2*sigma2);
}*/

/*LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                          Vector2d epl_, double edisp_, Vector3d eP_,
                          Vector3d le_,  double angle_, int idx_, int level_, Matrix3d covS_an_, Matrix3d covE_an_) :
    spl(spl_), sdisp(sdisp_), sP(sP_), epl(epl_), edisp(edisp_), eP(eP_), le(le_), inlier(true), idx(idx_), angle(angle_), level(level_), covS_an(covS_an_), covE_an(covS_an_)
{
    for( int i = 0; i < level; i++ )
        sigma2 *= Config::lsdScale();
    sigma2 = 1.f / (sigma2*sigma2);
}*/


}
