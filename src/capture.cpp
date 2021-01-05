#include "capture.h"

Capture::Capture()
{

}

void Capture::detection()
{
  cv::Mat descriptorsMat;
  cv::Ptr<cv::FeatureDetector>     detector   = cv::ORB::create(3000,1.2f, 2, 31, 0,2, cv::ORB::HARRIS_SCORE, 31,10);
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create(3000,1.2f, 2, 31, 0,2, cv::ORB::HARRIS_SCORE, 31,10);
  detector->detect(img_0, kp_2d);

  vector<cv::KeyPoint>  voxel[25];

  int x_step=img_0.cols/5;
  int y_step=img_0.rows/5;
  for (auto item:kp_2d){
    int idx_x = item.pt.x/x_step;
    int idx_y = item.pt.y/y_step;
    int dims  = idx_x + idx_y*5;
    if ( voxel[dims].size()<20){
      voxel[dims].push_back(item);
    }
  }

  kp_2d.clear();
  for(size_t i=0; i<25; i++){
    kp_2d.insert( kp_2d.end(), voxel[i].begin(), voxel[i].end() );
  }

  descriptor->compute ( img_0, kp_2d, descriptorsMat );
  descriptors_to_vMat(descriptorsMat, kp_descriptor);
  //vis
  cv::drawKeypoints(img_0, kp_2d, img_0_out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
}

void Capture::depth_recovery(Mat3x4 P0, Mat3x4 P1)
{
  std::vector<cv::Point2f> p0,p1;
  for (auto item:kp_2d)
  {
    p0.push_back(item.pt);
  }
  vector<unsigned char> status;
  vector<float> err;
  cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.01);
  cv::calcOpticalFlowPyrLK(img_0, img_1, p0, p1, status, err, cv::Size(21,21), 2, criteria);
  kp_3d.clear();
  vector<unsigned char> mask;
  for(size_t i=0; i<p0.size(); i++)
  {
    if(status.at(i)==1)
    {
      cv::circle( img_1_out, p0.at(i), 2, cv::Scalar( 0, 0, 255 ), 1, 8 );
      cv::circle( img_1_out, p1.at(i), 2, cv::Scalar( 0, 255, 0 ), 1, 8 );
      cv::line(   img_1_out, p0.at(i), p1.at(i),cv::Scalar( 255, 0, 0 ),2);
      Vec3 pt3d_c;
      if(Triangulation::trignaulationPtFromStereo(Vec2(p0.at(i).x,p0.at(i).y),
                                                  Vec2(p1.at(i).x,p1.at(i).y),
                                                  P0,
                                                  P1,
                                                  pt3d_c))
      {
        mask.push_back(true);
        kp_3d.push_back(pt3d_c);
      }
      else
      {
        mask.push_back(false);
      }
    }
    else
    {
      mask.push_back(false);
    }

  }

  reduceVector_cvKP(kp_2d,mask);
  reduceVector_cvMat(kp_descriptor,mask);

}

vector<cv::KeyPoint> Capture::get_kp(void)
{
  return kp_2d;
}
