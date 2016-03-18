# -------------------------------------------------------------------------------------------- #
#  Stereo Visual Odometry algorithm through the combination of point and line segment features #
# -------------------------------------------------------------------------------------------- #
This code contains an algorithm to compute stereo visual odometry.
It has been tested on Ubuntu 14.04.

# Configuration and generation
# -------------------------------------------------------------------------------------------- #
A CMakeLists.txt file is included to detect external dependencies and generate the project automatically. OpenCV and MRPT are used, but only OpenCV is required.

The project builds 2 different applications to evaluate and visualize it.

The first one is a "testSVO" customizable application where the user must introduce the inputs to the SVO algorithm, and then process the provided output. 

The second one, called "bumblebeeSVO", is an application that computes stereo visual odometry between the successive frames readed by a PointGrey Bumblebee2 stereo camera, and shows a 3D visualization of the camera motion. It is built or not depending on the CMake variable "HAS_MRPT".

In case you don't know MRPT I encourage you to have a look at its website here: http://www.mrpt.org/
Detailed instructions about how to install it (or some of its modules) can be found here: http://www.mrpt.org/download-mrpt/

# Compiling
# -------------------------------------------------------------------------------------------- #
The compiling process should be straightforward (OpenCV and MRPT should be easily found by CMake).

The only problem might arise from headers that are included but the compiler cannot find. In this case you should find those files on your computer and include them with the correct path (for your machine).

# Usage
# -------------------------------------------------------------------------------------------- #
The "bumblebeeSVO" application reads a configuration file named "svoConfig.ini" where the user can change the input parameters if desired. 

A complete configuration file "svoConfig_full.ini" is also provided, which includes all the parameters employed internally by the algorithm, but we recommend not to change the parameters.

The provided code is published under the General Public License Version 3 (GPL v3). More information can be found in the "GPU_LICENSE.txt" also included in the repository.

Please do not hesitate to contact the authors if you have any further questions.
