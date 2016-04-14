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

#include <bumblebeeGrabber.h>

bumblebeeGrabber::bumblebeeGrabber(){
    bbOptions.stereo_mode   = true;
    bbOptions.get_rectified = true;
    bbOptions.rect_width    = 320;
    bbOptions.rect_height   = 240;
    bbOptions.videomode     = "VIDEOMODE_1024x768RGB";
    bbOptions.framerate     = "FRAMERATE_20";
    bb = new CImageGrabber_FlyCapture2(bbOptions);
}

bumblebeeGrabber::bumblebeeGrabber(int img_width, int img_height, string frame_rate){

    CConfigFile config(frame_rate);
    bbOptions.stereo_mode   = true;
    bbOptions.get_rectified = true;
    bbOptions.rect_width    = img_width;
    bbOptions.rect_height   = img_height;
    bbOptions.videomode     = "VIDEOMODE_1024x768RGB";
    bbOptions.framerate     = frame_rate;
    bb = new CImageGrabber_FlyCapture2(bbOptions);

}

bumblebeeGrabber::~bumblebeeGrabber(){

}

void bumblebeeGrabber::grabStereo(Mat &imgLeft, Mat &imgRight){
    bb->getObservation(stereoObservation);
    imgLeft  = cvarrToMat( stereoObservation.imageLeft.getAs<IplImage>()  );
    imgRight = cvarrToMat( stereoObservation.imageRight.getAs<IplImage>() );
}

void bumblebeeGrabber::getCalib(Matrix3f &K, float &baseline){
    bb->getObservation(stereoObservation);
    fx  = stereoObservation.leftCamera.intrinsicParams(0,0);
    fy  = stereoObservation.leftCamera.intrinsicParams(1,1);
    cx  = stereoObservation.leftCamera.intrinsicParams(0,2);
    cy  = stereoObservation.leftCamera.intrinsicParams(1,2);
    baseline = stereoObservation.rightCameraPose.x();
    K  << fx, 0, cx, 0, fy, cy, 0, 0, 1;
}
