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
#include <stereoFrame.h>
#include <stereoFrameHandler.h>
#include <sceneRepresentation.h>
#include "yaml-cpp/yaml.h"

using namespace StVO;

int main(int argc, char **argv)
{

    // Initialize the camera
    Mat         img_l, img_r;
    Matrix3f    K;
    float       b;
    int         img_width  = 320,
                img_height = 240;
    string      frame_rate = "FRAMERATE_15";
    bumblebeeGrabber* bbGrabber = new bumblebeeGrabber(img_width,img_height,frame_rate);
    bbGrabber->getCalib(K,b);
    bbGrabber->grabStereo(img_l,img_r);

    // create scene
    sceneRepresentation scene("../config/bb_scene_config.ini");
    Matrix4d Tcw, Tfw = Matrix4d::Identity(), Tfw_prev = Matrix4d::Identity(), T_inc;
    Vector6d cov_eig;
    Matrix6d cov;
    Tcw = Matrix4d::Identity();
    Tcw << 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1;
    scene.initializeScene(Tcw,false);

    // initialize
    PinholeStereoCamera* cam_pin = new PinholeStereoCamera(img_height,img_width,K(0,0),K(1,1),K(0,2),K(1,2),b);
    StereoFrameHandler* StVO     = new StereoFrameHandler(cam_pin);
    bbGrabber->grabStereo(img_l,img_r);
    StVO->initialize(img_l,img_r,0);

    // run PL-StVO
    mrpt::utils::CTicTac clock;
    int frame_counter = 1;
    while(true)
    {
        // Point-Line Tracking
        clock.Tic();
        bbGrabber->grabStereo(img_l,img_r);
        double t0 = 1000 * clock.Tac(); //ms
        StVO->insertStereoPair( img_l, img_r, frame_counter );
        StVO->optimizePose();
        double t1 = 1000 * clock.Tac(); //ms

        // update scene
        scene.setText(frame_counter,t1,StVO->n_inliers_pt,StVO->matched_pt.size(),StVO->n_inliers_ls,StVO->matched_ls.size());
        scene.setCov( cov );
        scene.setPose( StVO->curr_frame->DT );
        imwrite("../config/aux/img_aux.png",StVO->curr_frame->plotStereoFrame());
        scene.setImage( "../config/aux/img_aux.png" );
        //scene.setImage( img_l );
        scene.updateScene();

        // console output
        cout.setf(ios::fixed,ios::floatfield); cout.precision(8);
        cout << "Frame: " << frame_counter << " \t Residual error: " << StVO->curr_frame->err_norm;
        cout.setf(ios::fixed,ios::floatfield); cout.precision(3);
        cout << " \t BB grabber time: " << t0 << " ms ";
        cout << " \t Proc. time: " << t1-t0 << " ms\t ";
        cout << "\t Points: " << StVO->matched_pt.size() << " (" << StVO->n_inliers_pt << ") " <<
                "\t Lines:  " << StVO->matched_ls.size() << " (" << StVO->n_inliers_ls << ") " << endl;

        // update StVO
        StVO->updateFrame();
        frame_counter++;

    }

    return 0;

}
