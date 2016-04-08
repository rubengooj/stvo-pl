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

#include <auxiliar.h>

#define PI std::acos(-1.0)

/* Kinematics functions */

Matrix4d inverse_transformation(Matrix4d T){
    Matrix4d T_inv = Matrix4d::Identity();
    Matrix3d R;
    Vector3d t;
    t << T(0,3), T(1,3), T(2,3);
    R << T.block(0,0,3,3);
    T_inv.block(0,0,3,4) << R.transpose(), -R.transpose() * t;
    return T_inv;
}

Matrix3d skew(Vector3d v){

    Matrix3d skew;

    skew(0,0) = 0; skew(1,1) = 0; skew(2,2) = 0;

    skew(0,1) = -v(2);
    skew(0,2) =  v(1);
    skew(1,2) = -v(0);

    skew(1,0) =  v(2);
    skew(2,0) = -v(1);
    skew(2,1) =  v(0);

    return skew;
}

Matrix3d fast_skewexp(Vector3d v){
    Matrix3d M, s, I = Matrix3d::Identity();
    double theta = v.norm();
    if(theta==0.f)
        M = I;
    else{
        s = skew(v)/theta;
        M << I + s * sin(theta) + s * s * (1.f-cos(theta));
    }
    return M;
}

Vector3d skewcoords(Matrix3d M){
    Vector3d skew;
    skew << M(2,1), M(0,2), M(1,0);
    return skew;
}

Matrix3d skewlog(Matrix3d M){
    Matrix3d skew;
    double val = (M.trace() - 1.f)/2.f;
    if(val > 1.f)
        val = 1.f;
    else if (val < -1.f)
        val = -1.f;
    double theta = acos(val);
    if(theta == 0.f)
        skew << 0,0,0,0,0,0,0,0,0;
    else
        skew << (M-M.transpose())/(2.f*sin(theta))*theta;
    return skew;
}

MatrixXd kroen_product(MatrixXd A, MatrixXd B){
    unsigned int Ar = A.rows(), Ac = A.cols(), Br = B.rows(), Bc = B.cols();
    MatrixXd AB(Ar*Br,Ac*Bc);
    for (unsigned int i=0; i<Ar; ++i)
        for (unsigned int j=0; j<Ac; ++j)
            AB.block(i*Br,j*Bc,Br,Bc) = A(i,j)*B;
    return AB;
}

Matrix3d v_logmap(VectorXd x){
    Vector3d w;
    double theta, theta2, theta3;
    Matrix3d W, I, V;
    w << x(0), x(1), x(2);
    theta = w.norm();   theta2 = theta*theta; theta3 = theta2*theta;
    W = skew(w);
    I << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    if(theta>0.00001)
        V << I + ((1-cos(theta))/theta2)*W + ((theta-sin(theta))/theta3)*W*W;
    else
        V << I;
    return V;
}

MatrixXd diagonalMatrix(MatrixXd M, unsigned int N){
    MatrixXd A = MatrixXd::Zero(N,N);
    for(unsigned int i = 0; i < N; i++ ){
        A(i,i) = M(i,i);
    }
    return A;
}

Matrix4d transformation_expmap(VectorXd x){
    Matrix3d R, V, s, I = Matrix3d::Identity();
    Vector3d t, w;
    Matrix4d T = Matrix4d::Identity();
    t << x(0), x(1), x(2);
    w << x(3), x(4), x(5);
    double theta = w.norm();
    if(theta==0.0f)
        R = I;
    else{
        s = skew(w)/theta;
        R << I + s * sin(theta) + s * s * (1.0f-cos(theta));
        V << I + s * (1.0f - cos(theta)) / theta + s * s * (theta - sin(theta)) / theta;
        t << V * t;
    }
    T.block(0,0,3,4) << R, t;
    return T;
}

VectorXd logarithm_map(Matrix4d T){
    Matrix3d R, Id3 = Matrix3d::Identity();
    Vector3d Vt, t, w;
    Matrix3d V = Matrix3d::Identity(), w_hat = Matrix3d::Zero();
    VectorXd x(6);
    Vt << T(0,3), T(1,3), T(2,3);
    w << 0.f, 0.f, 0.f;
    R = T.block(0,0,3,3);
    double val = (R.trace() - 1.f)/2.f;
    if(val > 1.f)
        val = 1.f;
    else if (val < -1.f)
        val = -1.f;
    double theta  = acos(val);
    double seno   = sin(theta);
    double coseno = cos(theta);
    if(theta != 0.f){
        w_hat << (R-R.transpose())/(2.f*seno)*theta;
        w << -w_hat(1,2), w_hat(0,2), -w_hat(0,1);
        Matrix3d s;
        s << skew(w) / theta;
        V << Id3 + s * (1.f-coseno) / theta + s * s * (theta - seno) / theta;
    }
    t = V.inverse() * Vt;
    x << t, w;
    return x;
}

Matrix4d transformation_expmap_approximate(VectorXd x){
    Matrix4d T = Matrix4d::Identity();
    T << 1.f, -x(5), x(4), x(0), x(5), 1.f, -x(3), x(1), -x(4), x(3), 1.f, x(2), 0.f, 0.f, 0.f, 1.f;
    return T;
}

VectorXd logarithm_map_approximate(Matrix4d T){
    VectorXd x(6);
    x(0) = T(0,3);
    x(1) = T(1,3);
    x(2) = T(2,3);
    x(3) = T(2,1);
    x(4) = T(0,2);
    x(5) = T(1,0);
    return x;
}

double diffManifoldError(Matrix4d T1, Matrix4d T2){
    return ( logarithm_map(T1)-logarithm_map(T2) ).norm();
}

bool is_finite(const MatrixXd x){
    return ((x - x).array() == (x - x).array()).all();
}

bool is_nan(const MatrixXd x){
    //bool aux = false;   // why it does not work with returns?
    for(unsigned int i = 0; i < x.rows(); i++){
        for(unsigned int j = 0; j < x.cols(); j++){
            if(std::isnan(x(i,j)))
                return true;
        }
    }
    return false;
}

double angDiff(double alpha, double beta){
    double theta = alpha - beta;
    if(theta>PI)
        theta -= 2.f * PI;
    if(theta<-PI)
        theta += 2.f * PI;
    return theta;
}

double angDiff_d(double alpha, double beta){
    double theta = alpha - beta;
    if(theta > 180.0)
        theta -= 360.0;
    if(theta<-PI)
        theta += 360.0;
    return theta;
}

/* Auxiliar functions and structs for vectors */

double vector_stdv_mad( VectorXf residues)
{
    // Return the standard deviation of vector with MAD estimation
    int n_samples = residues.size();
    sort( residues.derived().data(),residues.derived().data()+residues.size());
    double median = residues( n_samples/2 );
    residues << ( residues - VectorXf::Constant(n_samples,median) ).cwiseAbs();
    sort(residues.derived().data(),residues.derived().data()+residues.size());
    double MAD = residues( n_samples/2 );
    return 1.4826 * MAD;
}


double vector_stdv_mad( vector<double> residues)
{
    if( residues.size() != 0 )
    {
        // Return the standard deviation of vector with MAD estimation
        int n_samples = residues.size();
        sort( residues.begin(),residues.end() );
        double median = residues[ n_samples/2 ];
        for( int i = 0; i < n_samples; i++)
            residues[i] = fabsf( residues[i] - median );
        sort( residues.begin(),residues.end() );
        double MAD = residues[ n_samples/2 ];
        return 1.4826 * MAD;
    }
    else
        return 0.0;
}
