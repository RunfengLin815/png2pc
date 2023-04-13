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


int main (int argc, char** argv) {
    // Load RGB image and depth image
    cv::Mat rgb_image = cv::imread("/home/linrunfeng/Lab/data/shelter-demo/rgb/232.png");
    cv::Mat depth_image = cv::imread("/home/linrunfeng/Lab/data/shelter-demo/depth/232.png", CV_LOAD_IMAGE_ANYDEPTH);

    // Camera parameters
    float fx = 606.26f; // focal length x
    float fy = 606.691f; // focal length y
    float cx = 315.016f; // optical center x
    float cy = 234.613f; // optical center y

    // Point cloud data
    PointCloud::Ptr cloud(new PointCloud);
    cloud->width = rgb_image.cols;
    cloud->height = rgb_image.rows;
    cloud->points.resize(cloud->width * cloud->height);

    // Generate point cloud
    for (int i = 0; i < cloud->points.size(); ++i) {
        PointT &point = cloud->points[i];

        // Get RGB color
        point.r = rgb_image.at<cv::Vec3b>(i)[2];
        point.g = rgb_image.at<cv::Vec3b>(i)[1];
        point.b = rgb_image.at<cv::Vec3b>(i)[0];

        // Get depth value
        uint16_t depth_value = depth_image.at<uint16_t>(i);
        if (depth_value == 0) continue;

        // Convert depth value to 3D coordinates
        point.z = depth_value / 1000.0f;
        point.x = (i % cloud->width - cx) * point.z / fx;
        point.y = (i / cloud->width - cy) * point.z / fy;
    }

    // Save point cloud to PCD file
//    pcl::PCDWriter writer;
//    writer.write<PointT>("point_cloud.pcd", *cloud);
    pcl::PLYWriter writer;
    writer.write<PointT>("point_cloud.ply", *cloud);

    // Visualize point cloud
    pcl::visualization::PCLVisualizer viewer("Point Cloud Viewer");
}