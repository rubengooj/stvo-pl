/*****************************************************************************
**   Stereo Visual Odometry by combining point and line segment features	**
******************************************************************************
**																			**
**	Copyright(c) 2015, Ruben Gomez-Ojeda, University of Malaga              **
**	Copyright(c) 2015, MAPIR group, University of Malaga					**
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

#include <pinholeStereoCamera.h>

PinholeStereoCamera::PinholeStereoCamera( int width_, int height_, double fx_, double fy_, double cx_, double cy_, double b_,
                                          double d0, double d1, double d2, double d3, double d4) :
    width(width_), height(height_), fx(fx_), fy(fy_), cx(cx_), cy(cy_), b(b_), dist( d0 == 0.0 )
{
    d   << d0, d1, d2, d3, d4;
    Kcv  = ( Mat_<float>(3,3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0 );
    Dcv  = ( Mat_<float>(1,5) << d(0), d(1), d(2), d(3), d(4) );
    K   << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    Kinv = K.inverse();
    // initialize undistort rectify map OpenCV
    initUndistortRectifyMap( Kcv, Dcv, cv::Mat_<double>::eye(3,3), Kcv, cv::Size(width,height), CV_16SC2, undistmap1, undistmap2 );
}

PinholeStereoCamera::~PinholeStereoCamera() {};

void PinholeStereoCamera::rectifyImage( const Mat& img_src, Mat& img_rec)
{
    if(dist)
      remap( img_src, img_rec, undistmap1, undistmap2, cv::INTER_LINEAR);
    else
      img_rec = img_src.clone();
}

// Proyection and Back-projection (internally we are supposed to work with rectified images because of the line segments)
Vector3d PinholeStereoCamera::backProjection_unit( const double &u, const double &v, const double &disp, double &depth )
{
    Vector3d P_unit;
    P_unit(0) = (u-cx)/fx;
    P_unit(1) = (v-cy)/fy;
    P_unit(2) = 1.0;
    depth     = b*fx/disp;
    return P_unit;
}

Vector3d PinholeStereoCamera::backProjection( const double &u, const double &v, const double &disp )
{
    Vector3d P;
    double bd = b/disp;
    P(0) = bd*(u-cx);
    P(1) = bd*(v-cy);
    P(2) = bd*fx;
    return P;
}

Vector2d PinholeStereoCamera::projection( Vector3d P )
{
    Vector2d uv_unit;
    uv_unit(0) = cx + fx * P(0) / P(2);
    uv_unit(1) = cy + fy * P(1) / P(2);
    return uv_unit;
}

Vector2d PinholeStereoCamera::nonHomogeneous( Vector3d x)
{
    Vector2d x_; x_ << x(0) / x(2), x(1) / x(2);
    return x_;
}


