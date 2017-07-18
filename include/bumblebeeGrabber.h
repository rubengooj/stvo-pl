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

#include <opencv/cv.h>
#include <string>
#include <mrpt/opengl.h>
#include <mrpt/gui.h>
#include <mrpt/utils/CConfigFile.h>
#include <mrpt/utils/CConfigFileBase.h>
#include <mrpt/utils/CTicTac.h>
#include <mrpt/hwdrivers/CImageGrabber_FlyCapture2.h>
#include <mrpt/obs/CObservationStereoImages.h>
#include <eigen3/Eigen/Core>

using namespace mrpt;
using namespace mrpt::hwdrivers;
using namespace mrpt::obs;
using namespace mrpt::utils;
using namespace Eigen;
using namespace cv;

class bumblebeeGrabber{

public:

    bumblebeeGrabber();
    bumblebeeGrabber(int img_width, int img_height, string frame_rate);
    ~bumblebeeGrabber();
    void grabStereo(Mat &imgLeft, Mat &imgRight);
    void getCalib(Matrix3f &K, float &baseline);
    IplImage *grabIplImage();

private:

    float fx, fy, cx, cy;
    Matrix3f K;

    TCaptureOptions_FlyCapture2 bbOptions;
    CImageGrabber_FlyCapture2*  bb;
    CObservationStereoImages    stereoObservation;
    CImage                      imgLeft_, imgRight_;

};

