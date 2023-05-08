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
#include <eigen3/Eigen/Dense>

using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// declaration
PointCloud::Ptr cloudGenerator(string path_rgb, string path_depth);

void saveAsPly(PointCloud::Ptr cloud, string path_cloud);

vector<Eigen::Matrix4d> readTrajectory(const string &filename);

int main(int argc, char **argv) {
    // Load RGB image and depth image
    string path_rgb = "/home/linrunfeng/Lab/data/shelter-demo/rgb/900.png";
    string path_depth = "/home/linrunfeng/Lab/data/shelter-demo/depth/900.png";

    // directory
    string dir_rgb = "/home/linrunfeng/Lab/data/shelter-demo/rgb/";
    string dir_depth = "/home/linrunfeng/Lab/data/shelter-demo/depth/";

    // read pose
    string file_camera_trajectory = "/home/linrunfeng/Lab/shelter-reconstrustion/ManhattanSLAM/CameraTrajectory.txt";
    string file_keyframe_trajectory = "/home/linrunfeng/Lab/shelter-reconstrustion/ManhattanSLAM/KeyFrameTrajectory.txt";

    // test camera traj
    vector<Eigen::Matrix4d> v_trajectory_camera;
    vector<Eigen::Matrix4d> v_trajectory_keyframe;
    v_trajectory_camera = readTrajectory(file_camera_trajectory);
    v_trajectory_keyframe = readTrajectory(file_keyframe_trajectory);



    // save to where
    // string path_save = "/home/linrunfeng/Lab/data/shelter-demo/cloud_test.ply";

    // test for on image
    // PointCloud::Ptr pointCloud = cloudGenerator(path_rgb, path_depth);
    // saveAsPly(pointCloud, path_save);


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

vector<Eigen::Matrix4d> readTrajectory(const string &filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    Eigen::Matrix4d trajectory = Eigen::Matrix4d::Identity();
    vector<Eigen::Matrix4d> v_trajectory;

    std::string line;
    double time, tx, ty, tz, qx, qy, qz, qw;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ss >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        // Convert quaternion to rotation matrix
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Matrix3d R = q.toRotationMatrix();

        // Construct 4x4 transformation matrix
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) << tx, ty, tz;

        // Append to trajectory matrix
        trajectory *= T;
        v_trajectory.push_back(trajectory);
    }
    file.close();

//    for (int i = 0; i < 10; i++) {
//        cout << v_trajectory[i] << endl;
//    }

    return v_trajectory;

}