#pragma once
#include <opencv2/core.hpp>

enum class NeighborhoodPCA
{
	MEAN_SUB_32F,
	OPENCV_PCA,
	OPENCV_COV,

	SIZE
};

void patchPCA(const cv::Mat& src, cv::Mat& dst, const int neighborhood_r, const int dest_channels, const int border = cv::BORDER_DEFAULT, const int method = 0, const bool isParallel = false);
void patchPCA(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, const int neighborhood_r, const int dest_channels, const int border, const int method, const bool isParallel, cv::Mat& projectionMatrix, cv::Mat& eigenValue);
void patchPCA(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dst, const int neighborhood_r, const int dest_channels, const int border = cv::BORDER_DEFAULT, const int method = 0, const bool isParallel = false);
void patchPCATile(const cv::Mat& src, cv::Mat& dest, const int neighborhood_r, const int dest_channels, const int border, const int method, const cv::Size div);
