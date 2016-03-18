// ------------------------------------------
// includes
// ------------------------------------------
#pragma once

#include <thread>
#include <time.h>
using namespace std;

#include <mrpt/opengl.h>
#include <mrpt/gui.h>
#include <mrpt/utils/CConfigFile.h>
#include <mrpt/utils/CConfigFileBase.h>
using namespace mrpt;
using namespace mrpt::gui;
using namespace mrpt::poses;
using namespace mrpt::utils;
using namespace mrpt::math;
using namespace mrpt::opengl;

#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/line_descriptor.hpp>
using namespace cv;
using namespace cv::line_descriptor;

#include <eigen3/Eigen/Core>
using namespace Eigen;
// ------------------------------------------
// end includes
// ------------------------------------------

#include <pinholeStereoCamera.h>
#include <stereoFeatures.h>
#include <stereoFrame.h>
#include <config.h>
#include <kinematics.h>
#include <stereoFrameHandler.h>

class StereoFrameHandler;

class StVO{

public:

    StVO();
    ~StVO();

    StereoFrameHandler*  stvo;
    PinholeStereoCamera* cam;
    //Config*              cfg;



};
