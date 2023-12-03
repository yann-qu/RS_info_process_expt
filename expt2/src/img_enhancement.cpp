#include <string>

#include <opencv2/opencv.hpp>
#include <gdal.h>
#include <gdal_priv.h>

// TODO: Refactor
const std::string imgPath = "../../expt2/img/Exp2_TM.tif";

/**
 * Transform multi-spectral image to cv::Mat using gdal.
 * @param fileName
 * @return
 */
std::vector<cv::Mat> GDAL2Mat(const std::string &fileName) {
  GDALAllRegister();
  auto poDataset = GDALDatasetUniquePtr(GDALDataset::FromHandle(GDALOpen(fileName.c_str(), GA_ReadOnly)));
  if (!poDataset) {
    exit(-1);
  }
  int Cols = poDataset->GetRasterXSize(); // column
  int Rows = poDataset->GetRasterYSize(); // row
  int BandSize = poDataset->GetRasterCount(); // num of bands
  auto adfGeoTransform = new double[6];
  poDataset->GetGeoTransform(adfGeoTransform);

  std::vector<cv::Mat> imgVec;
  auto pafScan = new float[Cols * Rows];

  for (int i = 0; i < BandSize; i++) {
    GDALRasterBand *pBand = poDataset->GetRasterBand(i + 1); // read data of i+1 band
    CPLErr ret = pBand->RasterIO(GF_Read, 0, 0, Cols, Rows, pafScan,
                                 Cols, Rows, GDT_Byte, 0, 0); // put data of i+1 band to pafScan
    if (ret != CE_None) {
      exit(-1);
    }
    cv::Mat A = cv::Mat(Rows, Cols, CV_8U, pafScan); // put data of i+1 band to Mat A
    imgVec.push_back(A.clone()); // Have to clone, pafScan is an array.
  }
  delete[] pafScan;
  cv::Mat img;
  img.create(Rows, Cols, CV_8UC(BandSize));
  return imgVec;
}

/**
 * Histogram equalization.
 * @param src
 * @param dst
 */
void equalize_hist(cv::Mat &src, cv::Mat &dst) {
  dst = src.clone();
  std::vector<int> hist(256), value_equal(256);
  std::vector<double> hist_prob(256); // probability density, [0, 1]
  std::vector<double> hist_prob_dist(256); // probability distribution, [0, 1]
  int pixel_sum = src.cols * src.rows;

  //统计每个灰度下的像素个数
  for (int i = 0; i < src.rows; i++) {
    uchar *p = src.ptr<uchar>(i);
    for (int j = 0; j < src.cols; j++) {
      int vaule = p[j];
      hist[vaule]++;
    }
  }
  //统计灰度频率
  for (int i = 0; i < 256; i++) {
    hist_prob[i] = ((double) hist[i] / pixel_sum);
  }
  //计算累计密度
  hist_prob_dist[0] = hist_prob[0];
  for (int i = 1; i < 256; i++) {
    hist_prob_dist[i] = hist_prob_dist[i - 1] + hist_prob[i];
  }
  //计算均衡化后灰度值
  for (int i = 0; i < 256; i++) {
    value_equal[i] = (uchar) (255 * hist_prob_dist[i] + 0.5);
  }
  //直方图均衡化,更新原图每个点的像素值
  for (int i = 0; i < dst.rows; i++) {
    uchar *p = dst.ptr<uchar>(i);
    for (int j = 0; j < dst.cols; j++) {
      p[j] = value_equal[p[j]];
    }
  }
}

/**
 * Linear transformation.
 * [a, b] -> [a', b']
 * y = a' + (b' - a')/(b - a) * (x - a)
 * @param src
 * @param dst
 */
void linear_transform(cv::Mat &src, cv::Mat &dst) {
  dst = src.clone();
  std::vector<int> hist(256), value_equal(256);
  int le_src, ri_src, le_dst = 0, ri_dst = 255;
  double minVal, maxVal;
  cv::minMaxLoc(src, &minVal, &maxVal, nullptr, nullptr);
  le_src = (uchar) minVal;
  ri_src = (uchar) maxVal;

  //计算线性变换后灰度值
  for (int i = 0; i < 256; i++) {
    value_equal[i] = (uchar) (le_dst + 1.0 * (ri_dst - le_dst) / (ri_src - le_src) * (i - ri_src) + 0.5);
  }

  for (int i = 0; i < dst.rows; i++) {
    uchar *p = dst.ptr<uchar>(i);
    for (int j = 0; j < dst.cols; j++) {
      p[j] = value_equal[p[j]];
    }
  }
}

/**
 * Draw histogram.
 * @param hist
 * @param dst
 */
void draw_hist(cv::Mat &src, cv::Mat &dst) {
  int channels = 0;
  cv::MatND dstHist;
  int histSize[] = {256};
  float midRanges[] = {0, 256};
  const float *ranges[] = {midRanges};
  calcHist(&src, 1, &channels, cv::Mat(), dstHist, 1, histSize, ranges, true, false);
  dst = cv::Mat::zeros(cv::Size(256, 256), CV_8UC3);
  double g_dHistMaxValue;
  minMaxLoc(dstHist, 0, &g_dHistMaxValue, 0, 0);
  for (int i = 0; i < 256; i++) {
    int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / g_dHistMaxValue);
    line(dst, cv::Point(i, dst.rows - 1), cv::Point(i, dst.rows - 1 - value),
         cv::Scalar(255, 255, 255));
  }
}

int main() {
  std::vector<cv::Mat> imgVec, imgVec_enhanced, imgHistVec, imgHistVec_enhanced;
  cv::Mat imgConcat, imgConcat_enhanced, imgHistConcat, imgHistConcat_enhanced;
  imgVec = GDAL2Mat(imgPath); // img中含有所有波段的数据
  for (auto &&i: imgVec) {
    cv::Mat img_equalized, img_temp;
    equalize_hist(i, img_equalized);  // Histogram equalization.
    // linear_transform(i, img_equalized); // Linear transformation.
    imgVec_enhanced.push_back(img_equalized.clone());
    draw_hist(i, img_temp);
    imgHistVec.push_back(img_temp.clone());
    draw_hist(img_equalized, img_temp);
    imgHistVec_enhanced.push_back(img_temp.clone());
  }

  // display
  cv::hconcat(imgVec, imgConcat);
  cv::hconcat(imgVec_enhanced, imgConcat_enhanced);
  cv::hconcat(imgHistVec, imgHistConcat);
  cv::hconcat(imgHistVec_enhanced, imgHistConcat_enhanced);
  cv::imshow("img bands 1-6", imgConcat);
  cv::imshow("img bands equalized 1-6", imgConcat_enhanced);
  cv::imshow("img bands hist 1-6", imgHistConcat);
  cv::imshow("img bands equalized hist 1-6", imgHistConcat_enhanced);

  cv::waitKey(0);
  return 0;
}

