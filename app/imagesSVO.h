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

#include <mutex>
#include <thread>
#include <iomanip>
using namespace std;

#include <mrpt/opengl.h>
#include <mrpt/gui.h>
#include <mrpt/utils/CConfigFile.h>
#include <mrpt/utils/CConfigFileBase.h>
#include <mrpt/utils/CTicTac.h>
//#include <mrpt/hwdrivers/CImageGrabber_FlyCapture2.h>
//#include <mrpt/slam/CObservationStereoImages.h>
using namespace mrpt;
using namespace mrpt::gui;
//using namespace mrpt::hwdrivers;
using namespace mrpt::poses;
//using namespace mrpt::obs;
using namespace mrpt::utils;
using namespace mrpt::math;
using namespace mrpt::opengl;

#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/line_descriptor.hpp>
using namespace cv;
using namespace cv::line_descriptor;

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
using namespace Eigen;

#include <include/imageGrabber.h>
#include <include/plSVO_mrpt.h>
#include <include/sceneRepresentation.h>
