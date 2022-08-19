#pragma once
#include <opencv2/opencv.hpp>

class PCATransform
{
private:
	int area;
	std::vector<cv::Mat> mean;
	std::vector<cv::Mat> values;
	std::vector<cv::Mat> vectors;
	std::vector<cv::Mat> vectemp;
	std::vector<cv::Mat> cov;
public:
	PCATransform();
	void Project(const cv::Mat& src, cv::Mat& dest);
	void Project(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dest);
	void BackProject(const cv::Mat& src, cv::Mat& dest);
	void BackProject(const std::vector<cv::Mat>& src, std::vector<cv::Mat>& dest);
};

struct ComputeMean : cv::ParallelLoopBody
{
	cv::Mat& Input;
	cv::Mat& Mean;

	ComputeMean(cv::Mat& _Input, cv::Mat& _Mean)
		: Input(_Input), Mean(_Mean){};
	void operator()(const cv::Range& range) const;
};

struct ComputeCov : cv::ParallelLoopBody
{
	cv::Mat& Input;
	cv::Mat& Mean;
	cv::Mat& Cov;
	cv::Mat& Output;

	ComputeCov(cv::Mat& _Input, cv::Mat& _Mean, cv::Mat& _Cov, cv::Mat& _Output)
		: Input(_Input), Mean(_Mean), Cov(_Cov), Output(_Output) {};
	void operator()(const cv::Range& range) const;
};

struct pcaProject : cv::ParallelLoopBody
{
	cv::Mat& Input;
	cv::Mat& Vec;

	pcaProject(cv::Mat& _Input, cv::Mat& _Vec)
		: Input(_Input), Vec(_Vec){};
	void operator()(const cv::Range& range) const;
};

struct pcaBackProject : cv::ParallelLoopBody
{
	cv::Mat& Input;
	cv::Mat& Mean;
	cv::Mat& Vec;
	cv::Mat& Output;

	pcaBackProject(cv::Mat& _Input, cv::Mat& _Mean, cv::Mat& _Vec, cv::Mat& _Output)
		: Input(_Input), Mean(_Mean), Vec(_Vec), Output(_Output) {};
	void operator()(const cv::Range& range) const;
};

struct BackProject_sp : cv::ParallelLoopBody
{
	const std::vector<cv::Mat>& Input;
	cv::Mat& Mean;
	cv::Mat& Vec;
	std::vector<cv::Mat>& Output;

	BackProject_sp(const std::vector<cv::Mat>& _Input, cv::Mat& _Mean, cv::Mat& _Vec, std::vector<cv::Mat>& _Output)
		: Input(_Input), Mean(_Mean), Vec(_Vec), Output(_Output) {};
	void operator()(const cv::Range& range) const;
};

struct Project_sp : cv::ParallelLoopBody
{
	const std::vector<cv::Mat>& Input;
	cv::Mat& Mean;
	cv::Mat& Vec;
	std::vector<cv::Mat>& Output;

	Project_sp(const std::vector<cv::Mat>& _Input, cv::Mat& _Mean, cv::Mat& _Vec, std::vector<cv::Mat>& _Output)
		: Input(_Input), Mean(_Mean), Vec(_Vec), Output(_Output) {};
	void operator()(const cv::Range& range) const;
};
