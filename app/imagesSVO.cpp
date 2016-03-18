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

#include <stereoFrame.h>
#include <stereoFrameHandler.h>

int main(int argc, char **argv){

    PinholeStereoCamera* cam = new PinholeStereoCamera(1241,376,707.0912,707.0912,601.8873,183.1104,0.54);
    StereoFrameHandler* StVO = new StereoFrameHandler(cam);

    Mat img_l1, img_r1, img_l1_, img_l2, img_r2, img_l2_;
    img_l1 = imread("/home/ruben/datasets_bigres/Images_kittiC00/image_2/000000.png",IMREAD_GRAYSCALE);
    img_r1 = imread("/home/ruben/datasets_bigres/Images_kittiC00/image_3/000000.png",IMREAD_GRAYSCALE);
    StVO->initialize(img_l1,img_r1,0);

    img_l2 = imread("/home/ruben/datasets_bigres/Images_kittiC00/image_2/000001.png",IMREAD_GRAYSCALE);
    img_r2 = imread("/home/ruben/datasets_bigres/Images_kittiC00/image_3/000001.png",IMREAD_GRAYSCALE);
    StVO->insertStereoPair(img_l2,img_r2,0,Matrix4d::Identity());

    StVO->optimizePose();

    cout << endl << StVO->curr_frame->DT << endl << endl;

    // plot features
    list<PointFeature*> matched_pt;
    list<LineFeature*>  matched_ls;
    matched_pt = StVO->matched_pt;
    matched_ls = StVO->matched_ls;

    int lowest=100, highest=255;
    int range=(highest-lowest)+1;
    int radius  = 3;
    float thick = 1.5f;
    unsigned int r, g, b; //the color of lines
    Point2f P,Q;
    cvtColor(img_l1,img_l1_,CV_GRAY2BGR);
    cvtColor(img_l2,img_l2_,CV_GRAY2BGR);

    for( list<PointFeature*>::iterator it = matched_pt.begin(); it!=matched_pt.end(); it++)
    {
        r = lowest+int(rand()%range);
        g = lowest+int(rand()%range);
        b = lowest+int(rand()%range);
        P = Point2f( (*it)->pl(0) , (*it)->pl(1) );
        circle(img_l1_, P, radius, Scalar(b,g,r), thick);
        P = Point2f( (*it)->pl_obs(0), (*it)->pl_obs(1) );
        circle(img_l2_, P, radius, Scalar(b,g,r), thick);
    }

    for( list<LineFeature*>::iterator it = matched_ls.begin(); it!=matched_ls.end(); it++)
    {
        r = lowest+int(rand()%range);
        g = lowest+int(rand()%range);
        b = lowest+int(rand()%range);
        P = Point2f( (*it)->spl(0) , (*it)->spl(1) );
        Q = Point2f( (*it)->epl(0) , (*it)->epl(1) );
        line(img_l1_,P,Q,Scalar(b,g,r), thick);
        P = Point2f( (*it)->spl_obs(0), (*it)->spl_obs(1) );
        Q = Point2f( (*it)->epl_obs(0), (*it)->epl_obs(1) );
        line(img_l2_,P,Q,Scalar(b,g,r), thick);
    }

    imshow("first left",img_l1_);
    imshow("second left",img_l2_);
    waitKey(0);


    return 0;
}

