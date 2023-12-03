#include <string>
#include <cmath>

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

// Normalizes a given image into a value range between 0 and 255.
cv::Mat norm_0_255(const cv::Mat &src) {
  // Create and return normalized image:
  cv::Mat dst;
  switch (src.channels()) {
    case 1:
      cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
      break;
    case 3:
      cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
      break;
    default:
      src.copyTo(dst);
      break;
  }
  return dst;
}

static cv::Mat formatImagesForPCA(const std::vector<cv::Mat> &data) {
  cv::Mat dst(static_cast<int>(data.size()), data[0].rows * data[0].cols, CV_32FC1);
  for (int i = 0; i < data.size(); i++) {
    cv::Mat image_row = data[i].clone().reshape(1, 1);
    cv::Mat row_i = dst.row(i);
    image_row.convertTo(row_i, CV_32FC1);
  }
  return dst;
}

template<class T>
void printMatrix(const cv::Mat &matrix) {
  std::cout << "\n[\n";
  for (int i = 0; i < matrix.rows; i++) {
    for (int j = 0; j < matrix.cols; j++) {
      std::cout << std::setw(10) << matrix.at<T>(i, j) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << "]\n";
}


// 进行PCA变换
void PCA_project(std::vector<cv::Mat> &srcVec, std::vector<cv::Mat> &dstVec, std::vector<float> &var,
                 int num_components = 6) {
  // Build a matrix with the observations in row:
  cv::Mat data = formatImagesForPCA(srcVec);

  // Number of components to keep for the PCA:
  int maxComponents = num_components;

  // Perform a PCA:
  // cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, maxComponents);
  cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 6);

  // And copy the PCA results:
  cv::Mat mean = pca.mean.clone();
  cv::Mat eigenvalues = pca.eigenvalues.clone();
  cv::Mat eigenvectors = pca.eigenvectors.clone();

  std::cout << "EigenValues:\n";
  printMatrix<float>(eigenvalues);

  for (int i = 0; i < eigenvectors.rows; i++) {
    dstVec.push_back(norm_0_255(pca.eigenvectors.row(i)).reshape(1, srcVec[0].rows).clone());
  }

  // PCA project and back project
  for (int i = 0; i < srcVec.size(); i++) {
    // Demonstration of the effect of retainedVariance on the first image
    cv::Mat point = pca.project(data.row(i)); // project into the eigenspace, thus the image becomes a "point"
    cv::Mat reconstruction = pca.backProject(point); // re-create the image from the "point"
    reconstruction = norm_0_255(reconstruction.reshape(srcVec[0].channels(), srcVec[0].rows)); // reshape from a row vector into image shape
    cv::Mat diff;
    cv::Scalar mean_val, dev_val;
    cv::subtract(srcVec[i], reconstruction, diff);
    cv::meanStdDev(diff, mean_val, dev_val);
    var.push_back((float) (dev_val.val[0] * dev_val.val[0]));
  }
}

// PCA变换验证相关性
// 计算协方差矩阵
void PCA_project_verify(std::vector<cv::Mat> &imgVec) {
  // 120000 points
  int dim = imgVec.size();
  int row = imgVec[0].rows;
  int col = imgVec[0].cols;

  cv::Mat m(dim, 1, CV_32FC1);

  for (int i = 0; i < dim; i++) {
    auto mean_temp = cv::mean(imgVec[i]);
    m.at<float>(i) = (float) mean_temp.val[0];
  }

  cv::Mat sum = cv::Mat::zeros(dim, dim, CV_32FC1);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      cv::Mat x_k(dim, 1, CV_32FC1);
      for (int k = 0; k < dim; k++) {
        x_k.at<float>(k) = (float) imgVec[k].at<uchar>(i, j);
      }
      sum = sum + (x_k - m) * (x_k - m).t();
    }
  }

  cv::Mat C_x = sum / (row * col - 1);

  printMatrix<float>(C_x);

  cv::Mat R_x = C_x.clone();
  for (int i = 0; i < R_x.rows; i++) {
    for (int j = 0; j < R_x.cols; j++) {
      R_x.at<float>(i, j) = C_x.at<float>(i, j) / (sqrt(C_x.at<float>(i, i) * C_x.at<float>(j, j)));
    }
  }

  printMatrix<float>(R_x);
}


int main() {
  std::vector<cv::Mat> imgVec, imgVec_PCA, imgVec_PCA_back;
  cv::Mat imgConcat, imgConcat_PCA, imgConcat_PCA_back;
  imgVec = GDAL2Mat(imgPath); // img中含有所有波段的数据

  std::vector<float> var;
  PCA_project(imgVec, imgVec_PCA, var, 6);
  std::cout << "variance: ";
  for (auto i: var) std::cout << i << " ";

  cv::hconcat(imgVec, imgConcat);
  cv::hconcat(imgVec_PCA, imgConcat_PCA);
  cv::imshow("img bands 1-6", imgConcat);
  cv::imshow("img bands PCA 1-6", imgConcat_PCA);

  PCA_project_verify(imgVec);
  PCA_project_verify(imgVec_PCA);

  cv::waitKey(0);
  return 0;
}