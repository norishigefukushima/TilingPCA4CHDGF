#include "HDGF.hpp"
#include <stdio.h>
#include <string>

void quantization(const cv::Mat& input_image, int K, cv::Mat& centers, cv::Mat& labels)
{
	// ppmに変換
	cv::Mat input_ppm;
	cv::imwrite("Fourier/input.ppm", input_image);

	// quantization
	FILE* fp = NULL;
	// 絶対パスじゃないと駄目っぽい
//	std::string cmd = "C:/Users/toris/source/repos/bilateral_clustering/bilateral_clustering/Fourier/Fourier.exe Fourier/input.ppm ";
	//相対パスで動いた
	std::string cmd = "Fourier\\Fourier.exe Fourier/input.ppm ";

	//std::string cmd = "Fourier/Fourier.exe Fourier/input.ppm ";

	cmd = cmd + std::to_string(K);
	fp = _popen(cmd.c_str(), "w");

	if (NULL == fp)
	{
		printf("file open error ! \n");
		return;
	}
	_pclose(fp);

}

void nQunat(const cv::Mat& input_image, int K, ClusterMethod cm)
{
	FILE* fp = NULL;
	cv::imwrite("nQuantCpp/input.png", input_image);
	std::string cmd = "nQuantCpp\\nQuantCpp.exe nQuantCpp/input.png ";
	
	switch (cm)
	{
	case ClusterMethod::quantize_DIV:
	case ClusterMethod::kmeans_DIV:
		cmd = cmd + "/a DIV /m ";
		break;
	case ClusterMethod::quantize_PNN:
	case ClusterMethod::kmeans_PNN:
		cmd = cmd + "/a PNN /m ";
		break;
	case ClusterMethod::quantize_EAS:
	case ClusterMethod::kmeans_EAS:
		cmd = cmd + "/a EAS /m ";
		break;
	case ClusterMethod::quantize_SPA:
	case ClusterMethod::kmeans_SPA:
		cmd = cmd + "/a SPA /m ";
		break;
	default: 
		break;
	}

	cmd = cmd + std::to_string(K);
	fp = _popen(cmd.c_str(), "w");

	if (NULL == fp)
	{
		printf("file open error ! \n");
		return;
	}
	_pclose(fp);

}
