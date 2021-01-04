#ifndef CAPTURE_H
#define CAPTURE_H

//UTILS
#include <utils/include/all_utils.h>

//STDLIB
#include <sstream>

//OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

//VO
#include "include/triangulation.h"

class Capture
{
public:
  cv::Mat img_0, img_1;
  cv::Mat img_0_out, img_1_out;
  vector<Vec2>     kp_2d;
  vector<Vec3>     kp_3d;
  vector<cv::Mat>  kp_descriptor;

  Capture();
  void detection();
  void depth_recovery(Mat3x4 P0, Mat3x4 P1);
  vector<cv::KeyPoint> get_kp(void);
};

#endif // CAPTURE_H
