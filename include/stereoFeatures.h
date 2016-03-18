#pragma once
#include <eigen3/Eigen/Core>
using namespace Eigen;

class PointFeature
{
public:

    PointFeature( Vector2d pl_, double disp_, Vector3d P_ );
    PointFeature( Vector2d pl_, double disp_, Vector3d P_, Vector2d pl_obs_ );
    ~PointFeature(){};

    Vector2d pl, pl_obs;
    double   disp;
    Vector3d P;
    bool inlier;

private:
    // include jacobians & uncertainty


};

class LineFeature
{
public:

    LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                 Vector2d epl_, double edisp_, Vector3d eP_, Vector3d le_);
    LineFeature( Vector2d spl_, double sdisp_, Vector3d sP_,
                 Vector2d epl_, double edisp_, Vector3d eP_,
                 Vector3d le_, Vector3d le_obs_);
    ~LineFeature(){};

    Vector2d spl,epl, spl_obs, epl_obs;
    double   sdisp, edisp;
    Vector3d sP,eP;
    Vector3d le, le_obs;
    bool inlier;

private:
    // include jacobians & uncertainty

};
