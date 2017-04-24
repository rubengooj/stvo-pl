project( StVO-PL )

cmake_minimum_required(VERSION 2.7)
find_package( OpenCV 3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system filesystem)	

if(COMMAND cmake_policy)
  	cmake_policy(SET CMP0003 NEW)	
endif(COMMAND cmake_policy)
link_directories(${OpenCV_LIBS_DIR})
include_directories(${OpenCV2_INCLUDE_DIRS})

set(DEFAULT_HAS_MRPT ON)
set(HAS_MRPT ${DEFAULT_HAS_MRPT} CACHE BOOL "Build the PointGrey Bumblebee2 SVO application that employs the MRPT library")

SET(BUILD_SHARED_LIBS ON)
SET(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O3 -mtune=native -march=native")

# MRPT library (optional, only with representation purposes)
if(HAS_MRPT)
find_package(MRPT REQUIRED base opengl gui hwdrivers)
set(MRPT_DONT_USE_DBG_LIBS 1) #use release libraries for linking even if "Debug" CMake build
add_definitions(-DHAS_MRPT)
endif(HAS_MRPT)

# YAML library
find_library(YAML_CPP_LIBRARIES yaml-cpp)
if(NOT YAML_CPP_LIBRARIES)
  # If yaml-cpp not found in the system, try finding it as a user CMake-generated project
  find_package(yaml-cpp REQUIRED)
  include_directories(${YAML_CPP_INCLUDE_DIRS})
endif(NOT YAML_CPP_LIBRARIES)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Include dirs
include_directories(
  include
  ${Eigen3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/3rdparty/line_descriptor/include/
)

# Set link libraries
list(APPEND LINK_LIBS 
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/3rdparty/line_descriptor/lib/liblinedesc.so
)

# Set source files 
if(HAS_MRPT)
list(APPEND SOURCEFILES
  src/sceneRepresentation.cpp
  src/auxiliar.cpp
  src/bumblebeeGrabber.cpp
  src/config.cpp
  src/pinholeStereoCamera.cpp
  src/stereoFeatures.cpp
  src/stereoFrame.cpp
  src/stereoFrameHandler.cpp
)
else()
list(APPEND SOURCEFILES
  src/auxiliar.cpp
  src/config.cpp
  src/pinholeStereoCamera.cpp
  src/stereoFeatures.cpp
  src/stereoFrame.cpp
  src/stereoFrameHandler.cpp
)
endif()

# List all files (headers) contained by StVO-PL library
file(GLOB_RECURSE all_include_files RELATIVE "${CMAKE_SOURCE_DIR}" *.h *.hpp)

# Visualize the files of this directory in IDE creating an custom empty target
add_custom_target( stvo_includes DEPENDS ${all_include_files} SOURCES ${all_include_files} )

# Create StVO-PL library
add_library(stvo SHARED ${SOURCEFILES})

if(HAS_MRPT)
target_link_libraries(stvo ${LINK_LIBS} ${MRPT_LIBS})
else()
target_link_libraries(stvo ${LINK_LIBS})
endif()

# Applications
if(HAS_MRPT)
add_executable       ( bbStVO app/bbStVO.cpp )
target_link_libraries( bbStVO stvo )
add_executable       ( imagesStVO app/imagesStVO.cpp )
target_link_libraries( imagesStVO stvo )
endif(HAS_MRPT)


