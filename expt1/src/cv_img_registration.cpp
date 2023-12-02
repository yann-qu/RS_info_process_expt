/**
 * @Brief: Image registration based on opencv.
 * @Date: 2023-11-11
 * @Author: yann
 * @Email: yannqu@qq.com
 * @Attention: opencv_contrib needed
 */

#include "cv_img_registration.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

int main(const int argc, const char **argv) {
  std::vector<std::string> img_path;
  parse_param(argc, argv, img_path);

  // load image
  auto img1 = cv::imread(img_path[0], cv::IMREAD_GRAYSCALE);
  auto img2 = cv::imread(img_path[1], cv::IMREAD_GRAYSCALE);
  cv::imshow("img1", img1);
  cv::imshow("img2", img2);

  // preprocess
  cv::equalizeHist(img1, img1);
  cv::equalizeHist(img2, img2);
  cv::imshow("img1_equalize", img1);
  cv::imshow("img2_equalize", img2);

  // detect keypoint and compute descriptor
  cv::Ptr<cv::SIFT> SIFT_detector = cv::SIFT::create(nfeatures);
  std::vector<cv::KeyPoint> KeyPoints1, KeyPoints2;
  cv::Mat descriptor1, descriptor2;

  SIFT_detector->detectAndCompute(img1, cv::noArray(), KeyPoints1, descriptor1);
  SIFT_detector->detectAndCompute(img2, cv::noArray(), KeyPoints2, descriptor2);

  // draw image with keypoint
  auto img1_point = img1.clone();
  auto img2_point = img2.clone();

  cv::drawKeypoints(img1_point, KeyPoints1, img1_point);
  cv::drawKeypoints(img2_point, KeyPoints2, img2_point);
  cv::imshow("img1_point", img1_point);
  cv::imshow("img2_point", img2_point);

  // match the descriptor
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);
  std::cout << "All knn matches num: " << knn_matches.size() << "\n";

  // filter match
  // The distance ratio between the two nearest matches of a considered keypoint
  // is computed and it is a good match when this value is below a threshold.
  const float ratio_thresh = 0.7f;
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance <
        ratio_thresh * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  std::cout << "Good matches num: " << good_matches.size() << "\n";
  std::cout << "Good matches ratio = " << 1.0 * good_matches.size() / knn_matches.size() * 100 << "%\n";

  // draw matches
  cv::Mat img_match;
  cv::drawMatches(img1, KeyPoints1, img2, KeyPoints2, good_matches, img_match);
  cv::imshow("img_match", img_match);

  // calculate transform matrix
  std::vector<cv::Point2f> points1, points2;
  for (auto &&i : good_matches) {
    points1.push_back(KeyPoints1[i.queryIdx].pt);
    points2.push_back(KeyPoints2[i.trainIdx].pt);
  }
  if (good_matches.size() < 4) {
    std::cout << "At least 4 pair of points needed to calculate homography mat\n";
    exit(-1);
  }
  auto H = cv::findHomography(points2, points1, cv::RANSAC);

  // perspective transform img2 to img1
  cv::Mat img_warp_perspective;
  cv::warpPerspective(img2, img_warp_perspective, H, img1.size());
  cv::imshow("img_warp_perspective", img_warp_perspective);

  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}

void disp_info() {
  std::cout << "Please input the proper param!\n";
  std::cout << "  eg1. $ ./expt1_cv_img_registration FILE1 FILE2\n";
  std::cout << "  eg2. $ ./expt1_cv_img_registration < INPUT_FILE\n";
  std::cout << "  eg3. $ ./expt1_cv_img_registration\n";
}

void parse_param(const int argc, const char **argv,
                 std::vector<std::string> &img_path) {
  if (argc == 3) {
    // Get param directly.
    img_path.push_back(argv[1]);
    img_path.push_back(argv[2]);
  } else if (argc == 1) {
    // Try to read param from stdin.
    for (std::string line; std::getline(std::cin, line) && !line.empty();) {
      img_path.push_back(line);
    }
    if (img_path.size() != 2) {
      disp_info();
      exit(-1);
    }
  } else {
    disp_info();
    exit(-1);
  }

  std::cout << "Image path:\n";
  for (auto &&i : img_path)
    std::cout << "  " << i << "\n";
}
