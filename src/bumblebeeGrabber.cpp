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

bumblebeeGrabber::bumblebeeGrabber(string configFile){

    CConfigFile config(configFile);
    bbOptions.stereo_mode   = true;
    bbOptions.get_rectified = true;
    bbOptions.rect_width    = config.read_uint64_t("BumblebeeGrabber","width",320);
    bbOptions.rect_height   = config.read_uint64_t("BumblebeeGrabber","height",240);
    bbOptions.videomode     = config.read_string("BumblebeeGrabber","videomode","VIDEOMODE_1024x768RGB");
    bbOptions.framerate     = config.read_string("BumblebeeGrabber","framerate","FRAMERATE_20");
    bb = new CImageGrabber_FlyCapture2(bbOptions);

}

bumblebeeGrabber::~bumblebeeGrabber(){

}

void bumblebeeGrabber::grabStereo(Mat &imgLeft, Mat &imgRight){
    bb->getObservation(stereoObservation);
    imgLeft  = cvarrToMat( stereoObservation.imageLeft.getAs<IplImage>() );
    imgRight = cvarrToMat( stereoObservation.imageRight.getAs<IplImage>() );
}

void bumblebeeGrabber::getCalib(Matrix3f &K, float &baseline){
    bb->getObservation(stereoObservation);
    f   = stereoObservation.leftCamera.intrinsicParams(0,0);
    cx  = stereoObservation.leftCamera.intrinsicParams(0,2);
    cy  = stereoObservation.leftCamera.intrinsicParams(1,2);
    baseline = stereoObservation.rightCameraPose.x();
    K  << f, 0, cx, 0, f, cy, 0, 0, 1;
}
