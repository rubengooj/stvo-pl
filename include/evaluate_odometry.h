#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <limits>

// static parameter
float lengths[] = {100,200,300,400,500,600,700,800};
int32_t num_lengths = 8;

struct errors {
  int32_t first_frame;
  float   r_err;
  float   t_err;
  float   len;
  float   speed;
  errors (int32_t first_frame,float r_err,float t_err,float len,float speed) :
    first_frame(first_frame),r_err(r_err),t_err(t_err),len(len),speed(speed) {}
};

vector<Matrix> loadPoses(string file_name);

vector<float> trajectoryDistances (vector<Matrix> &poses);

int32_t lastFrameFromSegmentLength(vector<float> &dist,int32_t first_frame,float len);

inline float rotationError(Matrix &pose_error);

inline float translationError(Matrix &pose_error);

vector<errors> calcSequenceErrors (vector<Matrix> &poses_gt,vector<Matrix> &poses_result);

void saveSequenceErrors (vector<errors> &err,string file_name);

void savePathPlot (vector<Matrix> &poses_gt,vector<Matrix> &poses_result,string file_name);

vector<int32_t> computeRoi (vector<Matrix> &poses_gt,vector<Matrix> &poses_result) ;

void plotPathPlot (string dir,vector<int32_t> &roi,int32_t idx) ;

void saveErrorPlots(vector<errors> &seq_err,string plot_error_dir,char* prefix) ;

void plotErrorPlots (string dir,char* prefix);

void saveStats (vector<errors> err,string dir);

bool eval (string result_sha);

//int32_t main (int32_t argc,char *argv[]) {

//  // we need 2 arguments!
//  if (argc!=2) {
//    cout << "Usage: ./eval_odometry result_sha" << endl;
//    return 1;
//  }

//  // read arguments
//  string result_sha = argv[1];

//  // run evaluation
//  bool success = eval(result_sha);

//  return 0;
//}

