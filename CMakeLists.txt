cmake_minimum_required(VERSION 3.24)
project(png2pc)

set(CMAKE_CXX_STANDARD 17)
add_executable(png2pc main.cpp)

# include opencv
set(OpenCV_DIR /usr/local/share/OpenCV)
find_package(OpenCV 3.4.5 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(png2pc ${OpenCV_LIBS})

find_package(PCL 1.13 REQUIRED)
find_package(VTK 7.1 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${VTK_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${VTK_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

target_link_libraries(png2pc ${PCL_LIBRARIES})
target_link_libraries(png2pc ${VTK_LIBRARIES})
target_link_libraries(png2pc pcl_common pcl_io pcl_visualization)

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})