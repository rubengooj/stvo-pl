#include <stereoFeatures.h>

namespace StVO{

PointFeature::PointFeature( Vector2d pl_, double disp_, Vector3d P_ ) :
    pl(pl_), disp(disp_), P(P_), inlier(true)
{}

PointFeature::PointFeature( Vector2d pl_, double disp_, Vector3d P_, int idx_ ) :
    pl(pl_), disp(disp_), P(P_), inlier(true), idx(idx_)
{}

PointFeature::PointFeature( Vector2d pl_, double disp_, Vector3d P_, Vector2d pl_obs_ ) :
    pl(pl_), disp(disp_), P(P_), pl_obs(pl_obs_), inlier(true)
{}

LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                          Vector2d epl_, double edisp_, Vector3d eP_, Vector3d le_) :
    spl(spl_), sdisp(sdisp_), sP(sP_), epl(epl_), edisp(edisp_), eP(eP_), le(le_), inlier(true)
{}

LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                          Vector2d epl_, double edisp_, Vector3d eP_,
                          Vector3d le_, Vector3d le_obs_) :
    spl(spl_), sdisp(sdisp_), sP(sP_), epl(epl_), edisp(edisp_), eP(eP_), le(le_), le_obs(le_obs_), inlier(true)
{}

LineFeature::LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                          Vector2d epl_, double edisp_, Vector3d eP_,
                          Vector3d le_,  int    idx_) :
    spl(spl_), sdisp(sdisp_), sP(sP_), epl(epl_), edisp(edisp_), eP(eP_), le(le_), inlier(true), idx(idx_)
{}

}
