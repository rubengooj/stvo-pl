# StVO-PL #

This code contains an algorithm to compute stereo visual odometry by using both point and line segment features.

**Authors:** [Ruben Gomez-Ojeda](http://mapir.isa.uma.es/mapirwebsite/index.php/people/164-ruben-gomez) and [Javier Gonzalez-Jimenez](http://mapir.isa.uma.es/mapirwebsite/index.php/people/95-javier-gonzalez-jimenez)

**Related publication:** [*Robust Stereo Visual Odometry through a Probabilistic Combination of Points and Line Segments*](http://mapir.isa.uma.es/mapirwebsite/index.php/people/164-ruben-gomez)

If you use StVO-PL in your research work, please cite:

	@inproceedings{gomez2016robust,
  	  title={Robust stereo visual odometry through a probabilistic combination of points and line segments},
  	  author={Gomez-Ojeda, Ruben and Gonzalez-Jimenez, Javier},
      booktitle={Robotics and Automation (ICRA), 2016 IEEE International Conference on},
      pages={2521--2526},
      year={2016},
      organization={IEEE}
	}

The provided code is published under the General Public License Version 3 (GPL v3). More information can be found in the "GPU_LICENSE.txt" also included in the repository.

Please do not hesitate to contact the authors if you have any further questions.


## 1. Prerequisites and dependencies

We have tested StVO-PL with Ubuntu 12.04, 14.04 and 16.04, but it should be straightforward to compile along other platforms. Please notice that several internal processes, such as feature detection and matching can work in parallel, for which a powerful computer would be useful if working with the parallel configuration (change flags in the config file).

### OpenCV 3.x
It can be easily found at http://opencv.org. 

### Eigen3
http://eigen.tuxfamily.org

### Boost
Installation on Ubuntu:
```
sudo apt-get install libboost-dev
```

### YAML
Installation on Ubuntu:
```
sudo apt-get install libyaml-dev
```

### MRPT
In case of using the provided representation class or the PointGrey Bumblebee2 Stereo Camera application. 
```
sudo apt-get install libmrpt-dev
```

Download and install instructions can be also found at: http://www.mrpt.org/ .

### Line descriptor (in 3rdparty folder)
We have modified the [*line_descriptor*](https://github.com/opencv/opencv_contrib/tree/master/modules/line_descriptor) module from the [OpenCV/contrib](https://github.com/opencv/opencv_contrib) library (both BSD) which is included in the *3rdparty* folder.


## 2. Configuration and generation

Executing the file *build.sh* will configure and generate the *line_descriptor* module, and then will configure and generate the *StVO-PL* library for which we generate: **libstvo.so** in the lib folder, and the applications **bbStVO** that works with the PointGrey Bumblebee2 stereo camera, and **imagesStVO** that works with our dataset format (explained in the next section).


## 3. Dataset format and usage

The **imagesStVO** usage is: 
```
./imagesStVO  <dataset_path>  
```

with the following options:

-o,	Offset (number of frames to skip in the dataset directory

-n,	Number of frames to process the sequence

-s,	Parameter to skip s-1 frames (default 1)

and where *<dataset_path>* refers to the sequence folder relative to the environment variable *${DATASETS_DIR}* that must be previously set. That sequence folder must contain the dataset configuration file named **dataset_params.yaml** following the examples in **StVO-PL/config**, where **images_subfolder_x** refers to the left and right subfolders where the stereo images can be found.

The configuration files for the [EuRoC](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) and [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) datasets can be found in  **StVO-PL/config/euroc** and **StVO-PL/config/kitti** respectively.

