/**
 * @Brief: Image registration based on opencv.
 * @Date: 2023-11-11
 * @Author: yann
 * @Email: yannqu@qq.com
 * @Attention: opencv_contrib needed
 */

#ifndef CV_IMG_REGISTRATION_SRC_CV_IMG_REGISTRATION_H
#define CV_IMG_REGISTRATION_SRC_CV_IMG_REGISTRATION_H

#include <string>
#include <vector>

const int nfeatures = 200;

void disp_info();
void parse_param(const int argc, const char **argv,
                 std::vector<std::string> &img_path);

#endif // CV_IMG_REGISTRATION_SRC_CV_IMG_REGISTRATION_H
