#pragma once

#include <cv.h>
using namespace cv;

#include <vector>
using namespace std;

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
using namespace Eigen;

// Kinematics functions
Matrix4d inverse_transformation(Matrix4d T);
Matrix3d skew(Vector3d v);
Matrix3d fast_skewexp(Vector3d v);
Vector3d skewcoords(Matrix3d M);
Matrix3d skewlog(Matrix3d M);
MatrixXd kroen_product(MatrixXd A, MatrixXd B);
Matrix3d v_logmap(VectorXd x);
MatrixXd diagonalMatrix(MatrixXd M, unsigned int N);
Matrix4d transformation_expmap(VectorXd x);
VectorXd logarithm_map(Matrix4d T);
Matrix4d transformation_expmap_approximate(VectorXd x);
VectorXd logarithm_map_approximate(Matrix4d T);
double diffManifoldError(Matrix4d T1, Matrix4d T2);
bool is_finite(const MatrixXd x);
bool is_nan(const MatrixXd x);
double angDiff(double alpha, double beta);
double angDiff_d(double alpha, double beta);

// Auxiliar functions and structs for vectors
double vector_stdv_mad( VectorXf residues);
double vector_stdv_mad( vector<double> residues);

struct compare_descriptor_by_NN_dist
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b){
        return ( a[0].distance < b[0].distance );
    }
};

struct compare_descriptor_by_NN12_dist
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b){
        return ( a[1].distance - a[0].distance > b[1].distance-b[0].distance );
    }
};

struct compare_descriptor_by_NN12_ratio
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b){
        return ( a[0].distance / a[1].distance > b[0].distance / b[1].distance );
    }
};

struct sort_descriptor_by_queryIdx
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b){
        return ( a[0].queryIdx < b[0].queryIdx );
    }
};

struct sort_descriptor_by_trainIdx
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b){
        return ( a[0].trainIdx < b[0].trainIdx );
    }
};

