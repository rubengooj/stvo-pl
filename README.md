# PL-StVO #

This code contains an algorithm to compute stereo visual odometry by using both point and line segment features.

**Authors:** [Ruben Gomez-Ojeda](http://mapir.isa.uma.es/mapirwebsite/index.php/people/164-ruben-gomez) and [Javier Gonzalez-Jimenez](http://mapir.isa.uma.es/mapirwebsite/index.php/people/95-javier-gonzalez-jimenez)

**Related publication:** [*Robust Stereo Visual Odometry through a Probabilistic Combination of Points and Line Segments*](http://mapir.isa.uma.es/mapirwebsite/index.php/people/164-ruben-gomez)

If you use PL-StVO in your research work, please cite:

    @InProceedings{Gomez2015,
      Title                    = {Robust Stereo Visual Odometry through a Probabilistic Combination of Points and Line Segments},
      Author                   = {Gomez-Ojeda, Ruben and Gonzalez-Jimenez, Javier},
      Booktitle                = {Robotics and Automation (ICRA), 2016 IEEE International Conference on},
      Year                     = {2016},
      Publisher                = {IEEE}
    }

The provided code is published under the General Public License Version 3 (GPL v3). More information can be found in the "GPU_LICENSE.txt" also included in the repository.

Please do not hesitate to contact the authors if you have any further questions.

## 1. Prerequisites and dependencies



## 2. Configuration and generation

### OpenCV 3.0.0
We use OpenCV for extracting and matching features from stereo images, which can be easily found at http://opencv.org. 
In the case of line segments, we have modified the *line_descriptor* from the *opencv_contrib* 
[repository](https://github.com/Itseez/opencv_contrib), included in the *3rdparty* folder.

### Eigen3
We have employed 

### Boost

### YAML

### MRPT






# Configuration and generation
# -------------------------------------------------------------------------------------------- #
A CMakeLists.txt file is included to detect external dependencies and generate the project automatically. OpenCV and MRPT are used, but only OpenCV is required.

The project builds 2 different applications to evaluate and visualize it.

The first one is "imagesStVO", a customizable application where the user must introduce the inputs to the SVO algorithm, and then process the provided output. 

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

