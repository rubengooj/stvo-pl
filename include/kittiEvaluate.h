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

struct errors {
    int32_t first_frame;
    float   r_err;
    float   t_err;
    float   len;
    float   speed;
    errors (int32_t first_frame,float r_err,float t_err,float len,float speed) :
    first_frame(first_frame),r_err(r_err),t_err(t_err),len(len),speed(speed) {}
};

class kittiEval{

public:

    kittiEval();
    ~kittiEval();

    vector<MatrixKitti> loadPoses(string file_name);
    void writePoses(string file_name, vector<MatrixKitti> poses);
    vector<float>       trajectoryDistances (vector<MatrixKitti> &poses);
    int32_t             lastFrameFromSegmentLength(vector<float> &dist,int32_t first_frame,float len);
    inline float        rotationError(MatrixKitti &pose_error);
    inline float        translationError(MatrixKitti &pose_error);
    vector<errors>      calcSequenceErrors (vector<MatrixKitti> &poses_gt,vector<MatrixKitti> &poses_result);
    void                saveSequenceErrors (vector<errors> &err,string file_name);
    void                savePathPlot (vector<MatrixKitti> &poses_gt,vector<MatrixKitti> &poses_result,string file_name);
    vector<int32_t>     computeRoi (vector<MatrixKitti> &poses_gt,vector<MatrixKitti> &poses_result);
    string saveStats(vector<errors> err,string dir);
    void                saveErrorPlots(vector<errors> &seq_err,string plot_error_dir,char* prefix);
    bool                eval (vector<vector<MatrixKitti> > poses_result_, vector<vector<MatrixKitti> > poses_gt_, vector<string> v_kitti, string result_dir, string &stats);
    void                plotErrorPlots (string dir,char* prefix);
    void                plotPathPlot (string dir,vector<int32_t> &roi,int32_t idx);

private:

    int             idx = 0, lastIdx = 0, F;
    float           f, cx, cy, b;
    Matrix3f        K;
    ifstream        flistL, flistR;
    string          listLname, listRname, path_images, path_output;
    vector<string>  imagesL, imagesR, odometry;
    Mat             imageLeft, imageRight;

};

