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

#ifdef HAS_MRPT
#include <sceneRepresentation.h>
#include <mrpt/utils/CTicTac.h>
#endif

#include <stereoFrame.h>
#include <stereoFrameHandler.h>
#include <ctime>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>


using namespace StVO;

int main(int argc, char **argv)
{

    // read dataset name
    if( argc < 2 )
    {
        cout << endl << "Usage: ./imagesStVO <dataset_name>" << endl;
        return -1;
    }
    string dataset_name = argv[1];

    // read dataset root dir fron environment variable
    string dataset_dir( string( getenv("DATASETS_DIR") ) + "/" + dataset_name );

    // read content of the .yaml dataset configuration file
    YAML::Node dset_config = YAML::LoadFile(dataset_dir+"/dataset_params.yaml");

    // setup camera
    YAML::Node cam_config = dset_config["cam0"];
    string camera_model = cam_config["cam_model"].as<string>();
    PinholeStereoCamera*  cam_pin;
    if( camera_model == "Pinhole" )
    {
        cam_pin = new PinholeStereoCamera(
            cam_config["cam_width"].as<double>(),
            cam_config["cam_height"].as<double>(),
            fabs(cam_config["cam_fx"].as<double>()),
            fabs(cam_config["cam_fy"].as<double>()),
            cam_config["cam_cx"].as<double>(),
            cam_config["cam_cy"].as<double>(),
            cam_config["cam_bl"].as<double>(),
            cam_config["cam_d0"].as<double>(),
            cam_config["cam_d1"].as<double>(),
            cam_config["cam_d2"].as<double>(),
            cam_config["cam_d3"].as<double>()  );
    }
    else
    {
        cout << endl << "Not implemented yet." << endl;
        return -1;
    }

    // setup image directories
    string img_dir_l = dataset_dir + "/" + dset_config["images_subfolder_l"].as<string>();
    string img_dir_r = dataset_dir + "/" + dset_config["images_subfolder_r"].as<string>();

    // get a sorted list of files in the img directories
    boost::filesystem::path img_dir_path_l(img_dir_l.c_str());
    if (!boost::filesystem::exists(img_dir_path_l))
    {
        cout << endl << "Left image directory does not exist: \t" << img_dir_l << endl;
        return -1;
    }
    boost::filesystem::path img_dir_path_r(img_dir_r.c_str());
    if (!boost::filesystem::exists(img_dir_path_r))
    {
        cout << endl << "Right image directory does not exist: \t" << img_dir_r << endl;
        return -1;
    }

    // get all files in the img directories
    size_t max_len_l = 0;
    std::list<std::string> imgs_l;
    boost::filesystem::directory_iterator end_itr;
    for (boost::filesystem::directory_iterator file(img_dir_path_l); file != end_itr; ++file)
    {
        boost::filesystem::path filename_path = file->path().filename();
        if (boost::filesystem::is_regular_file(file->status()) &&
                (filename_path.extension() == ".png"  ||
                 filename_path.extension() == ".jpg"  ||
                 filename_path.extension() == ".jpeg" ||
                 filename_path.extension() == ".tiff") )
        {
            std::string filename(filename_path.string());
            imgs_l.push_back(filename);
            max_len_l = max(max_len_l, filename.length());
        }
    }
    size_t max_len_r = 0;
    std::list<std::string> imgs_r;
    for (boost::filesystem::directory_iterator file(img_dir_path_r); file != end_itr; ++file)
    {
        boost::filesystem::path filename_path = file->path().filename();
        if (boost::filesystem::is_regular_file(file->status()) &&
                (filename_path.extension() == ".png"  ||
                 filename_path.extension() == ".jpg"  ||
                 filename_path.extension() == ".jpeg" ||
                 filename_path.extension() == ".tiff") )
        {
            std::string filename(filename_path.string());
            imgs_r.push_back(filename);
            max_len_r = max(max_len_r, filename.length());
        }
    }

    // sort them by filename; add leading zeros to make filename-lengths equal if needed
    std::map<std::string, std::string> sorted_imgs_l, sorted_imgs_r;
    int n_imgs_l = 0, n_imgs_r = 0;
    for (std::list<std::string>::iterator img = imgs_l.begin(); img != imgs_l.end(); ++img)
    {
        sorted_imgs_l[std::string(max_len_l - img->length(), '0') + (*img)] = *img;
        n_imgs_l++;
    }
    for (std::list<std::string>::iterator img = imgs_r.begin(); img != imgs_r.end(); ++img)
    {
        sorted_imgs_r[std::string(max_len_r - img->length(), '0') + (*img)] = *img;
        n_imgs_r++;
    }
    if( n_imgs_l != n_imgs_r)
    {
        cout << endl << "Different number of left and right images." << endl;
        return -1;
    }

    // create scene
    Matrix4d Tcw, Tfw = Matrix4d::Identity(), Tfw_prev = Matrix4d::Identity(), T_inc;
    Vector6d cov_eig;
    Matrix6d cov;
    Tcw = Matrix4d::Identity();
    #ifdef HAS_MRPT
    sceneRepresentation* scene = new sceneRepresentation("scene_config.ini");
    scene->initializeScene(Tfw);
    mrpt::utils::CTicTac clock;
    #endif

    // initialize and run PL-StVO
    int frame_counter = 0;
    double t1;
    StereoFrameHandler* StVO = new StereoFrameHandler(cam_pin);
    for (std::map<std::string, std::string>::iterator it_l = sorted_imgs_l.begin(), it_r = sorted_imgs_r.begin();
         it_l != sorted_imgs_l.end(), it_r != sorted_imgs_r.end(); ++it_l, ++it_r, frame_counter++)
    {
        // load images
        boost::filesystem::path img_path_l = img_dir_path_l / boost::filesystem::path(it_l->second.c_str());
        boost::filesystem::path img_path_r = img_dir_path_r / boost::filesystem::path(it_r->second.c_str());
        Mat img_l( imread(img_path_l.string(), CV_8UC3) );  assert(!img_l.empty());
        Mat img_r( imread(img_path_r.string(), CV_8UC3) );  assert(!img_r.empty());

        // initialize (TODO: out of the for loop)
        if( frame_counter == 0 )
            StVO->initialize(img_l,img_r,0);
        // run
        else
        {
            // PL-StVO
            #ifdef HAS_MRPT
            clock.Tic();
            #endif
            StVO->insertStereoPair( img_l, img_r, frame_counter, T_inc );
            if(Config::motionPrior())
                StVO->setMotionPrior( logarithm_map(T_inc) , cov );
            StVO->optimizePose();
            #ifdef HAS_MRPT
            t1 = 1000 * clock.Tac(); //ms
            #endif

            // acces the pose
            T_inc   = StVO->curr_frame->DT;
            cov     = StVO->curr_frame->DT_cov;
            cov_eig = StVO->curr_frame->DT_cov_eig;

            // update scene
            #ifdef HAS_MRPT
            scene->setText(frame_counter,t1,StVO->n_inliers_pt,StVO->matched_pt.size(),StVO->n_inliers_ls,StVO->matched_ls.size());
            scene->setCov( cov );
            scene->setPose( T_inc );
            scene->setImage( img_path_l.string() );
            scene->updateScene();
            #endif

            // console output
            cout.setf(ios::fixed,ios::floatfield); cout.precision(8);
            cout << "Frame: " << frame_counter << " \t Residual error: " << StVO->curr_frame->err_norm;
            cout.setf(ios::fixed,ios::floatfield); cout.precision(3);
            cout << " \t Proc. time: " << t1 << " ms\t ";
            cout << "\t Points: " << StVO->matched_pt.size() << " (" << StVO->n_inliers_pt << ") " <<
                    "\t Lines:  " << StVO->matched_ls.size() << " (" << StVO->n_inliers_ls << ") " << endl;

            // update StVO
            StVO->updateFrame();

        }
    }

    return 0;
}

