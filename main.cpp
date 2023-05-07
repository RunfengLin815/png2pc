//
// Created by root on 23-4-13.
//
#include <iostream>
#include <pcl-1.13/pcl/point_types.h>
#include <pcl-1.13/pcl/io/pcd_io.h>
#include <pcl-1.13/pcl/io/ply_io.h>
#include <pcl-1.13/pcl/visualization/pcl_visualizer.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// declaration
PointCloud::Ptr cloudGenerator(string path_rgb, string path_depth);
void saveAsPly(PointCloud::Ptr cloud, string path_cloud);

int main(int argc, char **argv) {
    // Load RGB image and depth image
    string path_rgb = "/home/linrunfeng/Lab/data/shelter-demo/rgb/900.png";
    string path_depth = "/home/linrunfeng/Lab/data/shelter-demo/depth/900.png";

    // save to where
    string path_save = "/home/linrunfeng/Lab/data/shelter-demo/cloud_test.ply";

    // test for on image
    PointCloud::Ptr pointCloud = cloudGenerator(path_rgb, path_depth);
    saveAsPly(pointCloud, path_save);


    // Visualize point cloud
//    pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
}

PointCloud::Ptr cloudGenerator(string path_rgb, string path_depth) {
    // Camera parameters
    float fx = 606.26f; // focal length x
    float fy = 606.691f; // focal length y
    float cx = 315.016f; // optical center x
    float cy = 234.613f; // optical center y

    // Load RGB image and depth image
    cv::Mat img_rgb = cv::imread(path_rgb);
    cv::Mat img_depth = cv::imread(path_depth, CV_LOAD_IMAGE_ANYDEPTH);

    // Point cloud data
    PointCloud::Ptr cloud(new PointCloud);
    cloud->width = img_rgb.cols;
    cloud->height = img_rgb.rows;
    cloud->points.resize(cloud->width * cloud->height);

    // Generate point cloud
    for (int i = 0; i < cloud->points.size(); ++i) {
        PointT &point = cloud->points[i];

        // Get RGB color
        point.r = img_rgb.at<cv::Vec3b>(i)[2];
        point.g = img_rgb.at<cv::Vec3b>(i)[1];
        point.b = img_rgb.at<cv::Vec3b>(i)[0];

        // Get depth value
        uint16_t depth_value = img_depth.at<uint16_t>(i);
        if (depth_value == 0) continue;

        // Convert depth value to 3D coordinates
        point.z = depth_value / 1000.0f;
        point.x = (i % cloud->width - cx) * point.z / fx;
        point.y = (i / cloud->width - cy) * point.z / fy;
    }

    return cloud;
}

void saveAsPly(PointCloud::Ptr cloud, string path_cloud) {
    // save as ply file
    pcl::PLYWriter writer;
//    writer.write<PointT>("point_cloud.ply", *cloud);
    writer.write<PointT>(path_cloud, *cloud);

    // print
    std::cout << "Save point cloud to " << path_cloud << " ." << std::endl;
}