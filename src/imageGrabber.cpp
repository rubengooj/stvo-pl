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

#include <imageGrabber.h>

/* Constructors and destructor */

imageGrabber::imageGrabber(){
    path_images  = "~/datasets_bigres/"	;
    listLname	 = "./lists/k00L";
    listLname    = "./lists/k00R";
    path_output  = "./results/k00LR_odometry";
    readConfig();
}

imageGrabber::imageGrabber(string configFile){
    CConfigFile config(configFile);
    path_images  = config.read_string("ImageGrabber","path_images","~/datasets_bigres/");
    listLname    = config.read_string("ImageGrabber","listL","./lists/k00L");
    listRname    = config.read_string("ImageGrabber","listR","./lists/k00R");
    path_output  = config.read_string("ImageGrabber","path_output","./results/k00LR_odometry");
    readConfig();
}

imageGrabber::~imageGrabber(){

}

/* Setters and getters */

void imageGrabber::setSampling(float sample){
    F = sample;
    K = K / sample;
}

void imageGrabber::setListL(string listL_){
    listLname = listL_;
}

void imageGrabber::setListR(string listR_){
    listRname = listR_;
}

void imageGrabber::setPath(string path_){
    path_images = path_;
}

void imageGrabber::setOutput(string output_){
    path_output = output_;
}

void imageGrabber::getCalib(Matrix3f &K_, float &b_){
    K_ = K;
    b_ = b;
}

void imageGrabber::getOutpath(string &out_){
    out_ = path_output;
}

void imageGrabber::getLastIndex(int &idx){
    idx = lastIdx;
}

/* Grabbing methods */

void imageGrabber::readConfig(){
    string configFile = path_images + "acalib.txt";
    CConfigFile config(configFile);
    b      = config.read_double("","b",0.1);
    f      = config.read_double("","f",0.1);
    cx     = config.read_double("","cx",0.1);
    cy     = config.read_double("","cy",0.1);
    K << f, 0, cx, 0, f, cy, 0, 0, 1;
    ifstream        flistL(listLname);
    ifstream        flistR(listRname);
    string          imageL, imageR;
    imagesL.clear();
    imagesR.clear();
    lastIdx = 0;
    idx = 0;
    while( getline(flistL,imageL) && getline(flistR,imageR) ){
        imagesL.push_back(imageL);
        imagesR.push_back(imageR);
        lastIdx++;
    }
}

bool imageGrabber::grabStereo(Mat &imgLeft, Mat &imgRight){    
    string imageL, imageR;
    if( idx < lastIdx ){
        imageL = path_images + imagesL[idx];
        imageR = path_images + imagesR[idx];
        imgLeft  = imread(imageL,IMREAD_COLOR);
        imgRight = imread(imageR,IMREAD_COLOR);
        idx++;
        return true;
    }
    else
        return false;
}

bool imageGrabber::grabStereoSampled(Mat &imgLeft, Mat &imgRight){
    Mat imgLeft_, imgRight_;
    string imageL, imageR;
    if( idx < lastIdx ){
        imageL    = path_images + imagesL[idx];
        imageR    = path_images + imagesR[idx];
        imgLeft_  = imread(imageL,IMREAD_COLOR);
        imgRight_ = imread(imageR,IMREAD_COLOR);
        resize(imgLeft_,imgLeft,Size(imgLeft_.cols/F,imgLeft_.rows/F));
        resize(imgRight_,imgRight,Size(imgRight_.cols/F,imgRight_.rows/F));
        idx++;
        return true;
    }
    else
        return false;
}


