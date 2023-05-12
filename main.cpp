//
// Created by root on 23-4-13.
//
#include <iostream>
#include <filesystem> // c++17后的标准库
#include <pcl-1.13/pcl/point_types.h>
#include <pcl-1.13/pcl/io/pcd_io.h>
#include <pcl-1.13/pcl/io/ply_io.h>
#include <pcl-1.13/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.13/pcl/registration/ndt.h>
#include <pcl-1.13/pcl/filters/voxel_grid.h>
//#include <pcl-1.13/pcl/filters/radius_outlier_removal.h>
#include <pcl-1.13/pcl/filters/passthrough.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// declaration
PointCloud::Ptr cloudGenerator(string path_rgb, string path_depth);

void saveAsPly(PointCloud::Ptr cloud, string path_cloud);

void readTrajectory(const string &filename, vector<Eigen::Matrix4d> *v_trajectory, vector<int> *v_timestamp);
void readTrajectory_1(const string &filename, vector<Eigen::Matrix4d> *v_trajectory, vector<int> *v_timestamp);

int countFiles(string dir_path);

int main(int argc, char **argv) {
    // directory
    string dir_rgb = "/home/linrunfeng/Lab/data/shelter-demo/rgb/";
    string dir_depth = "/home/linrunfeng/Lab/data/shelter-demo/depth/";

    // read pose (only keyframe)
    // string file_camera_trajectory = "/home/linrunfeng/Lab/shelter-reconstrustion/ManhattanSLAM/CameraTrajectory.txt";
    string file_keyframe_trajectory = "/home/linrunfeng/Lab/shelter-reconstrustion/ManhattanSLAM/KeyFrameTrajectory.txt";
    vector<Eigen::Matrix4d> v_trajectory_keyframe;
    vector<int> timestamp;
    readTrajectory(file_keyframe_trajectory, &v_trajectory_keyframe, &timestamp);

    // point cloud
    PointCloud::Ptr cloud_final(new PointCloud);
    PointCloud::Ptr cloud(new PointCloud);
    PointCloud::Ptr cloud_tf(new PointCloud);
    PointCloud::Ptr cloud_filtered(new PointCloud);

//    pcl::RadiusOutlierRemoval<PointT> ror;
//    ror.setRadiusSearch(0.5);
//    ror.setMinNeighborsInRadius(100);
    // passthrough filter
    pcl::PassThrough<PointT> pass;
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.0, 1.5);


    // read all png in timestamp from keyframes
    for (int i = 0; i < timestamp.size(); i++) {

        cout << "No.[" << i + 1 << "/" << timestamp.size() << "] point cloud started..." << endl;

        // set file name
        string filename_rgb = dir_rgb + to_string(timestamp[i]) + ".png";
        string filename_depth = dir_depth + to_string(timestamp[i]) + ".png";

        // generate point cloud
        cout << "Start generating..." << endl;
        cloud = cloudGenerator(filename_rgb, filename_depth);

        //
        if (i % 50 == 0) {
            string path_save = "/home/linrunfeng/Lab/data/shelter-demo/cloud-single/cloud_" + to_string(i) + ".ply";
            saveAsPly(cloud,path_save);
        }

        // radius outlier
        cout << "Start filtering..." << " cloud size: " << cloud->size() << endl;
        pass.setInputCloud(cloud);
        pass.filter(*cloud_filtered);

        // transfer by trajectory
        cout << "Start transfer..." << " cloud_filtered size: " << cloud_filtered->size() << endl;
        pcl::transformPointCloud(*cloud_filtered, *cloud_tf, v_trajectory_keyframe[i]);

        // add to final scene
        cout << "Start adding..." << " cloud_transfer size: " << cloud_tf->size() << endl;
        *cloud_final += *cloud_tf;

        // print
        cout << "No.[" << i + 1 << "/" << timestamp.size() << "] point cloud completed." << endl;

        // clear
        cloud->clear();
        cloud_tf->clear();
        cloud_filtered->clear();
    }

    cout << "before ds: " << cloud_final->size() << endl;

    // down sampling
    // Create a VoxelGrid object
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(cloud_final);
    const float leafSize = 0.005f;
    sor.setLeafSize(leafSize, leafSize, leafSize);

    // do
    PointCloud::Ptr cloud_ds(new PointCloud);
    sor.filter(*cloud_ds);

    cout << "after ds: " << cloud_ds->size() << endl;

    // save
    string path_save = "/home/linrunfeng/Lab/data/shelter-demo/cloud_final.ply";
    saveAsPly(cloud_ds, path_save);

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
    writer.write<PointT>(path_cloud, *cloud);

    // print
    std::cout << "Save point cloud to " << path_cloud << " ." << std::endl;
}

void readTrajectory_1(const string &filename, vector<Eigen::Matrix4d> *v_trajectory, vector<int> *v_timestamp) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    Eigen::Matrix4d trajectory = Eigen::Matrix4d::Identity();
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
        v_trajectory->push_back(trajectory);

        // update
        v_timestamp->push_back(time);
    }
    file.close();

}

int countFiles(string dir_path) {
    int count = 0;
    for (const auto &entry: std::filesystem::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) { // 只计算文件，不包括文件夹
            count++;
        }
    }
    return count;
}


void readTrajectory(const string &filename, vector<Eigen::Matrix4d> *v_trajectory, vector<int> *v_timestamp) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    Eigen::Matrix4d trajectory = Eigen::Matrix4d::Identity();
    std::string line;
    double time, tx, ty, tz, qx, qy, qz, qw;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        ss >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

        // Convert quaternion to rotation matrix
        Eigen::Vector3d t(tx, ty, tz);
        Eigen::Quaterniond q(qw, qx, qy, qz);
        Eigen::Matrix3d R = q.normalized().toRotationMatrix();

        // Construct 4x4 transformation matrix
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
//        T.block<3, 3>(0, 0) = R;
//        T.block<3, 1>(0, 3) << tx, ty, tz;
        T.topLeftCorner(3, 3) = R;
        T.topRightCorner(3, 1) = t;


        // Append to trajectory matrix
        v_trajectory->push_back(T);

        // update
        v_timestamp->push_back(time);
    }
    file.close();

}
