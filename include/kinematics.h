#include <cv.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
using namespace Eigen;

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






