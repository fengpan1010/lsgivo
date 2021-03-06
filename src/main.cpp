//ROS
#include "ros/ros.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

//STDLIB
#include <sstream>

//OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

//PCL
#include <pcl/filters/voxel_grid.h>

//UTILS
#include <utils/include/all_utils.h>

//VO
#include "include/triangulation.h"
#include "capture.h"

using namespace sensor_msgs;
using namespace message_filters;
using namespace std;


unique_ptr<Capture> curr_capture;
unique_ptr<Capture> prev_capture;
unique_ptr<Capture> keyframe_capture;
cv::Mat K0;
cv::Mat c0_RM[2],c1_RM[2];
Mat3x4 P0,P1;

ros::Publisher triangulation_result_pub;
image_transport::Publisher vo_img0_pub,vo_img1_pub,vo_matching_pub;
ros::Publisher odom_pub;
ros::Publisher path_pub;
nav_msgs::Path path;

bool is_first_capture;
PointCloudP_ptr pc_map;

void publish_tf(SE3 T_w_c0_in, ros::Time stamp){
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.stamp = stamp;
  transformStamped.header.frame_id = "world";
  transformStamped.child_frame_id =  "camera0";
  transformStamped.transform.translation.x = T_w_c0_in.translation().x();
  transformStamped.transform.translation.y = T_w_c0_in.translation().y();
  transformStamped.transform.translation.z = T_w_c0_in.translation().z();
  transformStamped.transform.rotation.x = T_w_c0_in.so3().unit_quaternion().x();
  transformStamped.transform.rotation.y = T_w_c0_in.so3().unit_quaternion().y();
  transformStamped.transform.rotation.z = T_w_c0_in.so3().unit_quaternion().z();
  transformStamped.transform.rotation.w = T_w_c0_in.so3().unit_quaternion().w();

  br.sendTransform(transformStamped);

  //next, we'll publish the odometry message over ROS
  nav_msgs::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = "world";
  //set the position
  odom.pose.pose.position.x = T_w_c0_in.translation().x();
  odom.pose.pose.position.y = T_w_c0_in.translation().y();
  odom.pose.pose.position.z = T_w_c0_in.translation().z();
  odom.pose.pose.orientation.x= T_w_c0_in.so3().unit_quaternion().x();
  odom.pose.pose.orientation.y= T_w_c0_in.so3().unit_quaternion().y();
  odom.pose.pose.orientation.z= T_w_c0_in.so3().unit_quaternion().z();
  odom.pose.pose.orientation.w= T_w_c0_in.so3().unit_quaternion().w();

  //TODO::set the velocity

  odom.child_frame_id = "camera0";
  //publish the message
  odom_pub.publish(odom);

  //publish the path message over ROS
  geometry_msgs::PoseStamped pose;
  pose.header.stamp = stamp;
  pose.header.frame_id = "world";
  //set the position
  pose.pose.position.x = T_w_c0_in.translation().x();
  pose.pose.position.y = T_w_c0_in.translation().y();
  pose.pose.position.z = T_w_c0_in.translation().z();
  pose.pose.orientation.x= T_w_c0_in.so3().unit_quaternion().x();
  pose.pose.orientation.y= T_w_c0_in.so3().unit_quaternion().y();
  pose.pose.orientation.z= T_w_c0_in.so3().unit_quaternion().z();
  pose.pose.orientation.w= T_w_c0_in.so3().unit_quaternion().w();
  path.header.stamp = stamp;
  path.header.frame_id = "world";
  path.poses.push_back(pose);
  //publish the message
  path_pub.publish(path);
}


void callback(const ImageConstPtr& img0_msg, const ImageConstPtr& img1_msg)
{
  cv_bridge::CvImagePtr cvbridge_img0  = cv_bridge::toCvCopy(img0_msg, img0_msg->encoding);
  cv_bridge::CvImagePtr cvbridge_img1  = cv_bridge::toCvCopy(img1_msg, img1_msg->encoding);

  cv::Mat img0_distort,img1_distort;
  img0_distort = cvbridge_img0->image;
  img1_distort = cvbridge_img1->image;
  cv::remap(img0_distort, curr_capture->img_0, c0_RM[0], c0_RM[1],cv::INTER_LINEAR);
  cv::remap(img1_distort, curr_capture->img_1, c1_RM[0], c1_RM[1],cv::INTER_LINEAR);
  cv::cvtColor(curr_capture->img_1,curr_capture->img_1_out,cv::COLOR_GRAY2RGB);

  //Matching
  if(is_first_capture)
  {
    //init
    curr_capture->detection();
    curr_capture->depth_recovery(P0,P1);
    Mat3x3 R_w_c0;
    // 0  0  1
    //-1  0  0
    // 0 -1  0
    R_w_c0 << 0, 0, 1, -1, 0, 0, 0,-1, 0;
    Vec3   t_w_c=Vec3(0,0,0);
    SE3    T_w_c(R_w_c0,t_w_c);
    curr_capture->T_w_c=SE3(R_w_c0,t_w_c);
    *keyframe_capture = *curr_capture;
    is_first_capture = false;
  }else
  {
    // match between curr and keyframe
    cout << endl << "match between curr and keyframe" << endl;

    curr_capture->detection();
    cv::Ptr<cv::DescriptorMatcher>   matcher    = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    vector<cv::DMatch> matches;
    std::vector< cv::DMatch > good_matches;
    cv::Mat d_curr, d_keyframe;

    cout << "keyframe_capture->kp_2d.size()" << keyframe_capture->kp_2d.size() << endl;
    cout << "keyframe_capture->kp_3d.size()" << keyframe_capture->kp_3d.size() << endl;
    cout << "keyframe_capture->kp_descriptor.size()" << keyframe_capture->kp_descriptor.size() << endl;
    cout << "curr_capture->kp_2d.size()" << curr_capture->kp_2d.size() << endl;
    cout << "curr_capture->kp_3d.size()" << curr_capture->kp_3d.size() << endl;
    cout << "curr_capture->kp_descriptor.size()" << curr_capture->kp_descriptor.size() << endl;
    vMat_to_descriptors(keyframe_capture->kp_descriptor, d_keyframe);
    vMat_to_descriptors(curr_capture->kp_descriptor, d_curr);
    matcher->match (d_keyframe, d_curr, matches);
    double min_dist=10000, max_dist=0;
    for ( int i = 0; i < d_keyframe.rows; i++ ){
      double dist = matches[i].distance;
      if ( dist < min_dist ) min_dist = dist;
      if ( dist > max_dist ) max_dist = dist;
    }
    good_matches.clear();
    for ( int i = 0; i < d_keyframe.rows; i++ ){
      if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) ){
        good_matches.push_back ( matches[i] );
      }
    }
    cout << "goodmatch " << good_matches.size() << endl;

    //match between curr and prev
    if (good_matches.size() <= 50)
    {
      cout << "use last frame as keyframe ----" << endl;
      // update keyframe
      *keyframe_capture = *prev_capture;
      vMat_to_descriptors(keyframe_capture->kp_descriptor, d_keyframe);
      vMat_to_descriptors(curr_capture->kp_descriptor, d_curr);
      matcher->match (d_keyframe, d_curr, matches);
      double min_dist=10000, max_dist=0;
      for ( int i = 0; i < d_keyframe.rows; i++ ){
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
      }
      good_matches.clear();
      for ( int i = 0; i < d_keyframe.rows; i++ ){
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) ){
          good_matches.push_back ( matches[i] );
        }
      }
      cout << "goodmatch " << good_matches.size() << endl;
    }
    cout << "2d-2d matching finished "<< endl;
    cv::Mat img_match;
    drawMatches ( keyframe_capture->img_0, keyframe_capture->get_kp(),
                  curr_capture->img_0, curr_capture->get_kp(), good_matches, img_match);
    cout << "draw match finished "<< endl;
    //Pose Estimation
    //Creat 3D-2D Pairs
    vector<Vec3> keyframe_3d;
    vector<cv::KeyPoint> curr_2d;
    for(auto a_match:good_matches)
    {
      keyframe_3d.push_back(keyframe_capture->kp_3d.at(a_match.queryIdx));
      curr_2d.push_back(curr_capture->kp_2d.at(a_match.trainIdx));
    }
    cout << "extract pt-pairs finished "<< endl;
    vector<cv::Point3f> keyframe_3d_cv;
    vector<cv::Point2f> curr_2d_cv;
    for(size_t i=0; i<keyframe_3d.size(); i++)
    {
      keyframe_3d_cv.push_back(cv::Point3f(keyframe_3d.at(i).x(),keyframe_3d.at(i).y(),keyframe_3d.at(i).z()));
      curr_2d_cv.push_back(curr_2d.at(i).pt);
    }

    cv::Mat no_dist = (cv::Mat1d(4, 1) << 0,0,0,0);
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers;
    cv::solvePnPRansac(keyframe_3d_cv,curr_2d_cv, K0, no_dist, rvec, tvec, false, 100,3.0,0.99,inliers,cv::SOLVEPNP_P3P);
    SE3 T_c0curr_c0keyframe = SE3_from_rvec_tvec(rvec,tvec);
    cout << T_c0curr_c0keyframe.translation().transpose() << endl;

    //Calculate T_w_c0curr with given T_w_c0keyframe and T_c0curr_c0keyframe
    SE3 T_w_c0keyframe =keyframe_capture->T_w_c;
    SE3 T_w_c0curr = T_w_c0keyframe * T_c0curr_c0keyframe.inverse();
    curr_capture->T_w_c=T_w_c0curr;

    publish_tf(T_w_c0curr,img0_msg->header.stamp);
    //Depth recovery
    curr_capture->depth_recovery(P0,P1);

    //visualization pc_map
    for(size_t i=0; i<curr_capture->kp_3d.size(); i++)
    {
      //convert from camera to world
      Vec3 kp_3d_w= T_w_c0curr*curr_capture->kp_3d.at(i);

      pc_map->points.push_back (PointP(kp_3d_w[0],kp_3d_w[1],kp_3d_w[2]));
    }

    // Create the filtering object
    PointCloudP_ptr pc_map_filtered = PointCloudP_ptr(new PointCloudP());
    pcl::VoxelGrid<PointP> sor;
    sor.setInputCloud (pc_map);
    sor.setLeafSize (0.05f, 0.05f, 0.05f);
    sor.filter (*pc_map_filtered);
    pc_map->points.clear();
    pc_map->points = pc_map_filtered->points;

    pc_map->header.frame_id = "world";
    pc_map->height = 1;
    pc_map->width = pc_map->points.size();
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*pc_map , output);
    output.header.stamp = img0_msg->header.stamp;
    triangulation_result_pub.publish(output);
    //visualization triangulation
    sensor_msgs::ImagePtr vis_img0_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", curr_capture->img_0_out).toImageMsg();
    sensor_msgs::ImagePtr vis_img1_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", curr_capture->img_1_out).toImageMsg();
    vo_img0_pub.publish(vis_img0_msg);
    vo_img1_pub.publish(vis_img1_msg);
    //visualization between captures

    sensor_msgs::ImagePtr vis_matching_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_match).toImageMsg();
    vo_matching_pub.publish(vis_matching_msg);
  }

  curr_capture.swap(prev_capture);
  curr_capture->kp_2d.clear();
  curr_capture->kp_3d.clear();
  curr_capture->kp_descriptor.clear();

}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vision_node");
  ros::NodeHandle nh;

  string config_file_path;
  nh.getParam("/para_lsgivo_config_file_path",   config_file_path);

  cv::Mat K0_distorted,K1_distorted;
  cv::Mat distCoeffs_0,distCoeffs_1;
  K0_distorted = cameraMatrixFromYamlIntrinsics(config_file_path,"cam0_K");
  K1_distorted = cameraMatrixFromYamlIntrinsics(config_file_path,"cam1_K");
  distCoeffs_0 = distCoeffsFromYaml(config_file_path,"cam0_dc");
  distCoeffs_1 = distCoeffsFromYaml(config_file_path,"cam1_dc");
  Mat4x4 T_i_c0 = Mat44FromYaml(config_file_path,"T_imu_cam0");
  Mat4x4 T_i_c1 = Mat44FromYaml(config_file_path,"T_imu_cam1");

  Mat4x4 T_mat_c0_c1 = (T_i_c0.inverse())*T_i_c1;
  cout << T_mat_c0_c1.matrix() << endl;
  SE3 T_c0_c1(T_mat_c0_c1.topLeftCorner(3,3),T_mat_c0_c1.topRightCorner(3,1));


  SE3 T_c1_c0 = T_c0_c1.inverse();
  cv::Mat cvR0,cvR1,cvP0,cvP1,cvQ;
  Mat3x3 R_ = T_c1_c0.rotation_matrix();
  Vec3   T_ = T_c1_c0.translation();
  cv::Mat R__ = (cv::Mat1d(3, 3) << R_(0,0), R_(0,1), R_(0,2),
                 R_(1,0), R_(1,1), R_(1,2),
                 R_(2,0), R_(2,1), R_(2,2));
  cv::Mat T__ = (cv::Mat1d(3, 1) << T_(0), T_(1), T_(2));

  cv::stereoRectify(K0_distorted,distCoeffs_0,K1_distorted,distCoeffs_1,cv::Size(752,480),R__,T__,
                    cvR0,cvR1,cvP0,cvP1,cvQ,
                    cv::CALIB_ZERO_DISPARITY,0,cv::Size(752,480));//0 or 1
  cv::initUndistortRectifyMap(K0_distorted,distCoeffs_0,cvR0,cvP0,cv::Size(752,480),CV_32F,
                              c0_RM[0],c0_RM[1]);
  cv::initUndistortRectifyMap(K1_distorted,distCoeffs_1,cvR1,cvP1,cv::Size(752,480),CV_32F,
                              c1_RM[0],c1_RM[1]);
  P0(0,0) = cvP0.at<double>(0,0);
  P0(1,1) = cvP0.at<double>(1,1);
  P0(0,2) = cvP0.at<double>(0,2);
  P0(1,2) = cvP0.at<double>(1,2);
  P0(2,2) = cvP0.at<double>(2,2);
  P0(0,3) = cvP0.at<double>(0,3);
  P0(1,3) = cvP0.at<double>(1,3);
  P0(2,3) = cvP0.at<double>(2,3);

  P1(0,0) = cvP1.at<double>(0,0);
  P1(1,1) = cvP1.at<double>(1,1);
  P1(0,2) = cvP1.at<double>(0,2);
  P1(1,2) = cvP1.at<double>(1,2);
  P1(2,2) = cvP1.at<double>(2,2);
  P1(0,3) = cvP1.at<double>(0,3);
  P1(1,3) = cvP1.at<double>(1,3);
  P1(2,3) = cvP1.at<double>(2,3);

  K0 = (cv::Mat1d(3, 3) << P0(0,0), 0, P0(0,2), 0, P0(1,1), P0(1,2), 0, 0, 1);
  pc_map = PointCloudP_ptr(new PointCloudP());
  //  publish ros topics
  image_transport::ImageTransport it(nh);
  vo_img0_pub     = it.advertise("/vo_img0", 1);
  vo_img1_pub     = it.advertise("/vo_img1", 1);
  vo_matching_pub = it.advertise("/vo_matching",1);
  triangulation_result_pub = nh.advertise<sensor_msgs::PointCloud2>("/triangulation_result", 1);

  odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 50);
  path_pub = nh.advertise<nav_msgs::Path>("path", 50);

  is_first_capture = true;
  curr_capture = unique_ptr<Capture>(new Capture());
  prev_capture = unique_ptr<Capture>(new Capture());
  keyframe_capture = unique_ptr<Capture>(new Capture());

  message_filters::Subscriber<Image> image0_sub(nh, "/cam0/image_raw" , 1);
  message_filters::Subscriber<Image> image1_sub(nh, "/cam1/image_raw", 1);
  TimeSynchronizer<Image, Image> sync(image0_sub, image1_sub, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  ros::spin();

  return 0;
}
