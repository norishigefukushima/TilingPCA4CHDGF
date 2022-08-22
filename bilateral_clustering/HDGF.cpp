#include "HDGF.hpp"
#include "patchPCA.hpp"

using namespace std;
using namespace cv;

void randomSet(const int K, const int destChannels, cv::Mat& dest_center, float minv, float maxv)
{
	dest_center.create(K, destChannels, CV_32F);
	cv::RNG rng(cv::getTickCount());
	for (int i = 0; i < K; i++)
	{
		for (int c = 0; c < dest_center.cols; c++)
		{
			dest_center.at<float>(i, c) = rng.uniform(minv, maxv);
		}
	}
}

void randomSample(std::vector<cv::Mat>& vsrc32f, const int K, cv::Mat& dest_center)
{
	//print_debug3(src32f.cols, src32f.rows, dest_center.cols);

	cv::RNG rng(cv::getTickCount());

	dest_center.create(K, 1, CV_32FC3);
	int size = vsrc32f[0].size().area();
	for (int i = 0; i < K; i++)
	{
		const int idx = rng.uniform(0, size);
		for (int c = 0; c < dest_center.cols; c++)
		{
			dest_center.at<float>(i, c) = vsrc32f[c].at<float>(idx);
		}
	}
}

void randomSample(cv::Mat& src32f, const int K, cv::Mat& dest_center)
{
	//randomSet(K, src32f.rows, dest_center, 0.f, 255.f);
	dest_center.create(K, src32f.rows, CV_32F);
	//print_debug3(src32f.cols, src32f.rows, dest_center.cols);

	cv::Mat a(1, src32f.cols, CV_32F);
	int* aptr = a.ptr<int>();
	for (int i = 0; i < src32f.cols; i++)
	{
		aptr[i] = i;
	}
	cv::RNG rng(cv::getTickCount());
	cv::randShuffle(a, 2, &rng);

	const float* s = src32f.ptr<float>();
	for (int k = 0; k < K; k++)
	{
		const int idx = a.at<int>(k);
		for (int c = 0; c < src32f.rows; c++)
		{
			dest_center.at<float>(k, c) = s[src32f.cols * c + idx];
		}
	}
}

void get_label_center(cv::Mat src, const int K, cv::Mat& labels, cv::Mat& centers)
{
	// centerÇ©ÇÁlabelÇçÏê¨Ç∑ÇÈ
	labels = cv::Mat(src.rows * src.cols, 1, CV_32SC1);
	centers = cv::Mat(K, 1, CV_32FC3);
	uchar b, g, r;
	int k = 0;

	centers.ptr<cv::Vec3f>(0)[0][0] = (float)src.ptr<cv::Vec3b>(0)[0][0];
	centers.ptr<cv::Vec3f>(0)[0][1] = (float)src.ptr<cv::Vec3b>(0)[0][1];
	centers.ptr<cv::Vec3f>(0)[0][2] = (float)src.ptr<cv::Vec3b>(0)[0][2];
	k += 1;

	for (int y = 0; y < src.rows; y++)
	{
		cv::Vec3b* bgr = src.ptr<cv::Vec3b>(y);
		for (int x = 0; x < src.cols; x++)
		{
			b = bgr[x][0];
			g = bgr[x][1];
			r = bgr[x][2];
			//bool flag = false;


			for (int i = 0; i <= k; i++)
			{
				if (i == k)
				{
					centers.ptr<cv::Vec3f>(k)[0][0] = (float)b;
					centers.ptr<cv::Vec3f>(k)[0][1] = (float)g;
					centers.ptr<cv::Vec3f>(k)[0][2] = (float)r;
					labels.ptr<int>(y * src.cols + x)[0] = i;
					k += 1;
					break;
				}

				if (centers.ptr<cv::Vec3f>(i)[0][0] == (float)b &&
					centers.ptr<cv::Vec3f>(i)[0][1] == (float)g &&
					centers.ptr<cv::Vec3f>(i)[0][2] == (float)r)
				{
					labels.ptr<int>(y * src.cols + x)[0] = i;
					//flag = true;
					break;
				}
			}
		}
	}

}

#pragma region ConstantTimeCBFBase

cv::Ptr<ConstantTimeHDGFBase> createConstantTimeHDGF(ConstantTimeHDGF method)
{
	switch (method)
	{
	case ConstantTimeHDGF::Interpolation:
		//return cv::Ptr<ConstantTimeHDGFBase>(new ConstantTimeHDGF_Interpolation); break;
		std::cout << "not implemented" << std::endl;
		break;
	case ConstantTimeHDGF::Nystrom:
		return cv::Ptr<ConstantTimeHDGFBase>(new ConstantTimeHDGF_Nystrom); break;
	case ConstantTimeHDGF::SoftAssignment:
		return cv::Ptr<ConstantTimeHDGFBase>(new ConstantTimeHDGF_SoftAssignment); break;
	default:
		std::cout << "do not support this method in createConstantTimeHDGFSingle" << std::endl;
		std::cout << "retun interpolation, instead" << std::endl;

		break;
	}
	return cv::Ptr<ConstantTimeHDGFBase>(new ConstantTimeHDGF_Nystrom);
}

void ConstantTimeHDGFBase::clustering()
{
	//int64 start, end;
	if (downsampleMethod == DownsampleMethod::IMPORTANCE_MAP)
		input_image32f.convertTo(input_image8u, CV_8U);

	//start = cv::getTickCount();
	if (isDownsampleClustering)
	{
		downsampleForClustering();
	}
	else
	{
		reshaped_image32f = input_image32f.reshape(1, img_size.width * img_size.height);
	}

	//end = cv::getTickCount();
	//std::cout << "Downsample time:" << (end - start)*1000/ (cv::getTickFrequency()) << std::endl;

	if (cm == ClusterMethod::K_means_fast || cm == ClusterMethod::K_means_pp_fast)
	{
		const int vecsize = sizeof(__m256) / sizeof(float);//8
		int remsize = reshaped_image32f.rows % vecsize;
		if (remsize != 0)
		{
			cv::Rect roi(cv::Point(0, 0), cv::Size(reshaped_image32f.cols, reshaped_image32f.rows - remsize));
			reshaped_image32f = reshaped_image32f(roi);
		}
	}

	if (reshaped_image32f.depth() != CV_32F)
		reshaped_image32f.convertTo(reshaped_image32f, CV_32F);
	assert(reshaped_image32f.type() == CV_32F);
	cv::TermCriteria criteria(cv::TermCriteria::COUNT, iterations, 1);
	//cv::setNumThreads(1);

#pragma region clustering
	switch (cm)
	{
	case ClusterMethod::random_sample:
		randomSample(reshaped_image32f, K, centers);
		break;
	case ClusterMethod::K_means:
		//start = cv::getTickCount();
		// K-means Clustering
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);
		//end = cv::getTickCount();
		//diff = end - start;
		//time = (diff) * 1000 / (cv::getTickFrequency());
		//cout << time << endl;
		//cout << labels.type() << endl;
		//cout << labels.size() << endl;
		break;
	case ClusterMethod::K_means_pp:
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);
		//kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
		//kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
		//kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
		//kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);

		break;
	case ClusterMethod::K_means_fast:
		kmcluster.clustering(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);
		break;
	case ClusterMethod::K_means_pp_fast:
		kmcluster.clustering(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers);
		break;
	case ClusterMethod::KGaussInvMeansPPFast:
		kmcluster.setSigma(30);
		kmcluster.clustering(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers, cp::KMeans::MeanFunction::GaussInv);
		break;

	case ClusterMethod::mediancut_median:
		mediancut(input_image8u, K, labels, centers, cp::MedianCutMethod::MEDIAN);
		centers.convertTo(centers, CV_32FC3, 1.0 / 255.0);
		break;
	case ClusterMethod::mediancut_max:
		mediancut(input_image8u, K, labels, centers, cp::MedianCutMethod::MAX);
		centers.convertTo(centers, CV_32FC3, 1.0 / 255.0);
		break;
	case ClusterMethod::mediancut_min:
		mediancut(input_image8u, K, labels, centers, cp::MedianCutMethod::MIN);
		centers.convertTo(centers, CV_32FC3, 1.0 / 255.0);
		break;

	case ClusterMethod::X_means:
		setK(cp::xmeans(reshaped_image32f, labels, criteria, 1, cv::KMEANS_PP_CENTERS, centers));
		break;

		//	quantize_wu,
		//	kmeans_wu,
		//	quantize_neural,
		//	kmeans_neural,
	case ClusterMethod::quantize_wan:
		quantization(input_image8u, K, centers, labels);
		input_image8u = cv::imread("out_wan.ppm", 1);
		get_label_center(input_image8u, K, labels, centers);
		break;
	case ClusterMethod::kmeans_wan:
		quantization(input_image8u, K, centers, labels);
		input_image8u = cv::imread("out_wan.ppm", 1);
		get_label_center(input_image8u, K, labels, centers);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
		break;
	case ClusterMethod::quantize_wu:
		quantization(input_image8u, K, centers, labels);
		input_image8u = cv::imread("out_wu.ppm", 1);
		get_label_center(input_image8u, K, labels, centers);
		break;
	case ClusterMethod::kmeans_wu:
		quantization(input_image8u, K, centers, labels);
		input_image8u = cv::imread("out_wu.ppm", 1);
		get_label_center(input_image8u, K, labels, centers);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
		break;
	case ClusterMethod::quantize_neural:
		quantization(input_image8u, K, centers, labels);
		input_image8u = cv::imread("out_neural.ppm", 1);
		get_label_center(input_image8u, K, labels, centers);
		break;
	case ClusterMethod::kmeans_neural:
		quantization(input_image8u, K, centers, labels);
		input_image8u = cv::imread("out_neural.ppm", 1);
		get_label_center(input_image8u, K, labels, centers);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
		break;
	case ClusterMethod::quantize_DIV:
		nQunat(input_image8u, K, cm);
		input_image8u = cv::imread("nQuantCpp/input-DIVquant" + std::to_string(K) + ".png", 1);
		get_label_center(input_image8u, K, labels, centers);
		break;
	case ClusterMethod::kmeans_DIV:
		nQunat(input_image8u, K, cm);
		input_image8u = cv::imread("nQuantCpp/input-DIVquant" + std::to_string(K) + ".png", 1);
		get_label_center(input_image8u, K, labels, centers);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
		break;
	case ClusterMethod::quantize_PNN:
		nQunat(input_image8u, K, cm);
		input_image8u = cv::imread("nQuantCpp/input-PNNquant" + std::to_string(K) + ".png", 1);
		get_label_center(input_image8u, K, labels, centers);
		break;
	case ClusterMethod::kmeans_PNN:
		nQunat(input_image8u, K, cm);
		input_image8u = cv::imread("nQuantCpp/input-PNNquant" + std::to_string(K) + ".png", 1);
		get_label_center(input_image8u, K, labels, centers);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
		break;
	case ClusterMethod::quantize_SPA:
		nQunat(input_image8u, K, cm);
		input_image8u = cv::imread("nQuantCpp/input-SPAquant" + std::to_string(K) + ".png", 1);
		get_label_center(input_image8u, K, labels, centers);
		break;
	case ClusterMethod::kmeans_SPA:
		nQunat(input_image8u, K, cm);
		input_image8u = cv::imread("nQuantCpp/input-SPAquant" + std::to_string(K) + ".png", 1);
		get_label_center(input_image8u, K, labels, centers);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
		break;
	case ClusterMethod::quantize_EAS:
		nQunat(input_image8u, K, cm);
		input_image8u = cv::imread("nQuantCpp/input-EASquant" + std::to_string(K) + ".png", 1);
		get_label_center(input_image8u, K, labels, centers);
		break;
	case ClusterMethod::kmeans_EAS:
		nQunat(input_image8u, K, cm);
		input_image8u = cv::imread("nQuantCpp/input-EASquant" + std::to_string(K) + ".png", 1);
		get_label_center(input_image8u, K, labels, centers);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers);
		break;
	default:
		break;
	}
#pragma endregion
	if (centers.rows != K)
	{
		std::cout << "Clustering is not working" << std::endl;
	}
	assert(centers.rows == K);
}

void ConstantTimeHDGFBase::downsampleForClustering()
{
	switch (downsampleMethod)
	{
	case NEAREST:
	case LINEAR:
	case CUBIC:
	case AREA:
	case LANCZOS:
		cv::resize(input_image32f, reshaped_image32f,
			cv::Size(img_size.width / downsampleRate, img_size.height / downsampleRate),
			0, 0, downsampleMethod);
		reshaped_image32f = reshaped_image32f.reshape(1, (img_size.width / downsampleRate) * (img_size.height / downsampleRate));
		break;

	case IMPORTANCE_MAP:
		cp::generateSamplingMaskRemappedDitherTexturenessPackedAoS(input_image8u, reshaped_image32f, 1.f / (downsampleRate * downsampleRate));
		//reshaped_image32f.convertTo(reshaped_image32f, CV_32FC3, 1.0 / 255);
		reshaped_image32f = reshaped_image32f.reshape(1, reshaped_image32f.rows);
		break;

	default:
		break;
	}
}

ConstantTimeHDGFBase::ConstantTimeHDGFBase()
{
	GF.resize(threadMax);
}

//Base class of constant time bilateral filtering for multi thread version
ConstantTimeHDGFBase::~ConstantTimeHDGFBase()
{
	;
}

void ConstantTimeHDGFBase::setGaussianFilterRadius(const int r)
{
	this->radius = r;
}

void ConstantTimeHDGFBase::setGaussianFilter(const double sigma_space,
	const cp::SpatialFilterAlgorithm method, const int gf_order)
{
	bool isCompute = false;
	if (GF[0].empty())
	{
		isCompute = true;
	}
	else
	{
		if (GF[0]->getSigma() != sigma_space ||
			GF[0]->getAlgorithmType() != method ||
			GF[0]->getOrder() != gf_order ||
			GF[0]->getSize() != img_size)
		{
			isCompute = true;
		}
	}

	if (isCompute)
	{
		//cout << "alloc GF" << endl;
		//this->sigma_space = sigma_space;


		for (int i = 0; i < threadMax; ++i)
		{
			GF[i] = cp::createSpatialFilter(method, CV_32F, cp::SpatialKernel::GAUSSIAN);
		}

		if (radius != 0)
		{
			if (radius != 0)
			{
				for (int i = 0; i < threadMax; ++i)
				{
					GF[i]->setFixRadius(radius);
				}
			}
		}
	}
}


void ConstantTimeHDGFBase::setParameter(cv::Size img_size, double sigma_space, double sigma_range, ClusterMethod cm,
	int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth,
	bool isDownsampleClustering, int downsampleRate, int downsampleMethod)
{
	this->depth = depth;
	this->K = K;

	this->img_size = img_size;
	this->sigma_space = sigma_space;
	this->sigma_range = sigma_range;

	this->cm = cm;
	this->setGaussianFilterRadius((int)ceil(this->sigma_space * 3));
	this->setGaussianFilter(sigma_space, gf_method, gf_order);

	this->isDownsampleClustering = isDownsampleClustering;
	this->downsampleRate = downsampleRate;
	this->downsampleMethod = downsampleMethod;
}

void ConstantTimeHDGFBase::setK(int k)
{
	this->K = k;
}

int ConstantTimeHDGFBase::getK()
{
	return this->K;
}

void ConstantTimeHDGFBase::setConcat_offset(int concat_offset)
{
	this->concat_offset = concat_offset;
}

void ConstantTimeHDGFBase::setPca_r(int pca_r)
{
	this->pca_r = pca_r;
}

void ConstantTimeHDGFBase::setKmeans_ratio(float kmeans_ratio)
{
	this->kmeans_ratio = kmeans_ratio;
}
#pragma endregion

#pragma region ConstantTimeHDGFSingleBase

cv::Ptr<ConstantTimeHDGFSingleBase> createConstantTimeHDGFSingle(ConstantTimeHDGF method)
{
	switch (method)
	{
	case ConstantTimeHDGF::Interpolation:
		return cv::Ptr<ConstantTimeHDGFSingleBase>(new ConstantTimeHDGF_InterpolationSingle); break;
		break;
	case ConstantTimeHDGF::Nystrom:
		return cv::Ptr<ConstantTimeHDGFSingleBase>(new ConstantTimeHDGF_NystromSingle); break;
		break;
	case ConstantTimeHDGF::SoftAssignment:
		return cv::Ptr<ConstantTimeHDGFSingleBase>(new ConstantTimeHDGF_SoftAssignmentSingle); break;
		break;
	default:
		std::cout << "do not support this method in createConstantTimeHDGFSingle" << std::endl;
		std::cout << "retun interpolation, instead" << std::endl;

		break;
	}
	return cv::Ptr<ConstantTimeHDGFSingleBase>(new ConstantTimeHDGF_InterpolationSingle);
}

ConstantTimeHDGFSingleBase::ConstantTimeHDGFSingleBase()
{
	timer.resize(num_timerblock_max);
	for (int i = 0; i < num_timerblock_max; i++)timer[i].setIsShow(false);
}

ConstantTimeHDGFSingleBase::~ConstantTimeHDGFSingleBase()
{
	;
}

void reshapeDownSample(std::vector<cv::Mat> vsrc, cv::Mat& dest, const int downsampleRate, const int boundaryLength)
{
	//const int size = ((vsrc[0].cols - boundaryLength * 2) / downsampleRate) * ((vsrc[0].rows - boundaryLength * 2) / downsampleRate);
	const int size = ((vsrc[0].cols) / downsampleRate) * ((vsrc[0].rows) / downsampleRate);
	dest.create(cv::Size(size, (int)vsrc.size()), CV_32F);

	int index = 0;
	float* dptr0 = dest.ptr<float>(0);
	float* dptr1 = dest.ptr<float>(1);
	float* dptr2 = dest.ptr<float>(2);

#if 0//random sampling
	cv::RNG rng(cv::getTickCount());
	float* bptr = vsrc[0].ptr<float>(0);
	float* gptr = vsrc[1].ptr<float>(0);
	float* rptr = vsrc[2].ptr<float>(0);
	cv::Mat buff = cv::Mat::zeros(vsrc[0].size().area(), 1, CV_8U);
	for (int i = 0; i < size; i++)
	{
		int idx = rng.uniform(0, (int)vsrc[0].size().area());
		for (;;)
		{
			if (buff.at<uchar>(idx) == 0)
			{
				buff.at<uchar>(idx) = 255;
				break;
			}
			idx = rng.uniform(0, (int)vsrc[0].size().area());
		}

		dptr0[index] = bptr[idx];
		dptr1[index] = gptr[idx];
		dptr2[index++] = rptr[idx];
	}
#elif 1
	for (int j = 0; j < vsrc[0].rows; j += downsampleRate)
	{
		float* bptr = vsrc[0].ptr<float>(j);
		float* gptr = vsrc[1].ptr<float>(j);
		float* rptr = vsrc[2].ptr<float>(j);
		for (int i = 0; i < vsrc[0].cols; i += downsampleRate)
		{
			dptr0[index] = bptr[i];
			dptr1[index] = gptr[i];
			dptr2[index++] = rptr[i];
		}
	}
#else
	for (int j = boundaryLength; j < vsrc[0].rows - boundaryLength; j += downsampleRate)
	{
		float* bptr = vsrc[0].ptr<float>(j);
		float* gptr = vsrc[1].ptr<float>(j);
		float* rptr = vsrc[2].ptr<float>(j);
		for (int i = boundaryLength; i < vsrc[0].cols - boundaryLength; i += downsampleRate)
		{
			dptr0[index] = bptr[i];
			dptr1[index] = gptr[i];
			dptr2[index++] = rptr[i];
		}
	}
#endif

}

double ConstantTimeHDGFSingleBase::testClustering(const std::vector<cv::Mat>& src)
{
	clusteringErrorMap.create(src[0].size(), CV_32F);
	//mu;
	const int size = src[0].size().area();
	const int ch = (int)src.size();
	double error = 0.0;
	for (int i = 0; i < size; i++)
	{
		float mindiff = FLT_MAX;
		for (int k = 0; k < K; k++)
		{
			float diff = 0.f;
			for (int c = 0; c < ch; c++)
			{
				const float v = src[c].at<float>(i) - mu.at<float>(k, c);
				diff += v * v;
			}
			mindiff = std::min(mindiff, diff);
		}
		clusteringErrorMap.at<float>(i) = mindiff / ch;
		error += mindiff;
	}
	return error / size;
}

void ConstantTimeHDGFSingleBase::getClusteringErrorMap(cv::Mat& dest)
{
	this->clusteringErrorMap.copyTo(dest);
}

void ConstantTimeHDGFSingleBase::clustering()
{
	switch (cm)
	{
	case ClusterMethod::quantize_wan:
	case ClusterMethod::kmeans_wan:
	case ClusterMethod::quantize_wu:
	case ClusterMethod::kmeans_wu:
	case ClusterMethod::quantize_neural:
	case ClusterMethod::kmeans_neural:
	case ClusterMethod::quantize_DIV:
	case ClusterMethod::kmeans_DIV:
	case ClusterMethod::quantize_PNN:
	case ClusterMethod::kmeans_PNN:
	case ClusterMethod::quantize_SPA:
	case ClusterMethod::kmeans_SPA:
	case ClusterMethod::quantize_EAS:
	case ClusterMethod::kmeans_EAS:
		if (isJoint) cp::mergeConvert(vguide, guide_image8u, CV_8U);
		else cp::mergeConvert(vsrc, guide_image8u, CV_8U);
		break;

	case ClusterMethod::random_sample:
	case ClusterMethod::K_means:
	case ClusterMethod::K_means_pp:
	case ClusterMethod::K_means_fast:
	case ClusterMethod::K_means_pp_fast:
	case ClusterMethod::KGaussInvMeansPPFast:

	case ClusterMethod::mediancut_median:
	case ClusterMethod::mediancut_max:
	case ClusterMethod::mediancut_min:

	case ClusterMethod::X_means:

	default:
		break;
	}

	if (isDownsampleClustering)
	{
		if (isJoint) downsampleForClustering(vguide, reshaped_image32f, isCropBoundaryClustering);
		else downsampleForClustering(vsrc, reshaped_image32f, isCropBoundaryClustering);
		//print_matinfo(reshaped_image32f);
	}
	else
	{
		if (isJoint) mergeForClustering(vguide, reshaped_image32f, isCropBoundaryClustering);
		else mergeForClustering(vsrc, reshaped_image32f, isCropBoundaryClustering);
	}

	if (reshaped_image32f.depth() != CV_32F || reshaped_image32f.channels() != 1)
	{
#pragma omp critical
		{
			std::cout << "depth(vsrc) " << vsrc[0].depth() << std::endl;
			std::cout << "depth " << reshaped_image32f.depth() << std::endl;
			std::cout << "channel " << reshaped_image32f.channels() << std::endl;
		}
		CV_Assert(reshaped_image32f.type() == CV_32FC1);
	}

	if (cm == ClusterMethod::K_means_fast || cm == ClusterMethod::K_means_pp_fast || cm == ClusterMethod::KGaussInvMeansPPFast)
	{
		const int vecsize = sizeof(__m256) / sizeof(float);//8
		if (reshaped_image32f.cols < reshaped_image32f.rows)
		{
			int remsize = reshaped_image32f.rows % vecsize;
			if (remsize != 0)
			{
				cv::Rect roi(cv::Point(0, 0), cv::Size(reshaped_image32f.cols, reshaped_image32f.rows - remsize));
				reshaped_image32f = reshaped_image32f(roi);
			}
		}
		else
		{
			int remsize = reshaped_image32f.cols % vecsize;
			//print_debug(remsize);
			if (remsize != 0)
			{
				//std::cout << "KMEANSFAST" << std::endl;
				cv::Rect roi(cv::Point(0, 0), cv::Size(reshaped_image32f.cols - remsize, reshaped_image32f.rows));
				reshaped_image32f = reshaped_image32f(roi);
			}
		}
	}

	//print_matinfo(reshaped_image32f);
	if (reshaped_image32f.cols < K)
	{
		std::cout << "K is large for K-means: reshaped_image32f.cols < K" << std::endl;
	}

	cv::TermCriteria criteria(cv::TermCriteria::COUNT, iterations, 1);
	cv::setNumThreads(1);

	switch (cm)
	{
	case ClusterMethod::random_sample:
		//if (isJoint) randomSample(vguide, K, mu);
		//else randomSample(vsrc, K, mu);
		randomSample(reshaped_image32f, K, mu);
		//randomSample(K, mu);
		break;

	case ClusterMethod::K_means:
	{
		//start = cv::getTickCount();
		// K-means Clustering
		//kmeans(reshaped_image32f.t(), K, labels, criteria, attempts, cv::KMEANS_RANDOM_CENTERS, mu);
		kmeans(reshaped_image32f.t(), K, labels, criteria, attempts, cv::KMEANS_RANDOM_CENTERS, mu);
		//print_matinfo(reshaped_image32f);
		//print_debug(K);
		//print_matinfo(labels);
		//cv::Mat mu1 = mu;
		//print_matinfo(mu1);
		//end = cv::getTickCount();
		//diff = end - start;
		//time = (diff) * 1000 / (cv::getTickFrequency());
		//cout << time << endl;
		//cout << labels.type() << endl;
		//cout << labels.size() << endl;
		break;
	}
	case ClusterMethod::K_means_pp:
		kmeans(reshaped_image32f.t(), K, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, mu);
		break;
	case ClusterMethod::K_means_fast:
		kmcluster.clustering(reshaped_image32f, K, labels, criteria, attempts, cv::KMEANS_RANDOM_CENTERS, mu, cp::KMeans::MeanFunction::Mean, cp::KMeans::Schedule::SoA_KND);
		break;
	case ClusterMethod::K_means_pp_fast:
	{
		//print_matinfo(reshaped_image32f);
		//cv::Mat mu1 = mu;
		//print_matinfo(mu1);
		kmcluster.clustering(reshaped_image32f, K, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, mu, cp::KMeans::MeanFunction::Mean, cp::KMeans::Schedule::SoA_KND);
	}
	break;
	case ClusterMethod::KGaussInvMeansPPFast:
		kmcluster.setSigma((float)kmeans_sigma);
		kmcluster.clustering(reshaped_image32f, K, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, mu, cp::KMeans::MeanFunction::GaussInv, cp::KMeans::Schedule::SoA_KND);
		break;

	case ClusterMethod::mediancut_median:
	case ClusterMethod::mediancut_max:
	case ClusterMethod::mediancut_min:
		reshaped_image32f.convertTo(reshaped_image8u, CV_8U);
		{
			if (cm == ClusterMethod::mediancut_median) mediancut(reshaped_image8u, K, labels, mu, cp::MedianCutMethod::MEDIAN);
			if (cm == ClusterMethod::mediancut_max) mediancut(reshaped_image8u, K, labels, mu, cp::MedianCutMethod::MAX);
			if (cm == ClusterMethod::mediancut_min) mediancut(reshaped_image8u, K, labels, mu, cp::MedianCutMethod::MIN);
		}
		mu.convertTo(mu, CV_32F);
		break;
		break;

	case ClusterMethod::quantize_wan:
		quantization(guide_image8u, K, mu, labels);
		guide_image8u = cv::imread("out_wan.ppm", 1);
		get_label_center(guide_image8u, K, labels, mu);
		break;
	case ClusterMethod::kmeans_wan:
		quantization(guide_image8u, K, mu, labels);
		guide_image8u = cv::imread("out_wan.ppm", 1);
		get_label_center(guide_image8u, K, labels, mu);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
		break;
	case ClusterMethod::quantize_wu:
		quantization(guide_image8u, K, mu, labels);
		guide_image8u = cv::imread("out_wu.ppm", 1);
		get_label_center(guide_image8u, K, labels, mu);
		break;
	case ClusterMethod::kmeans_wu:
		quantization(guide_image8u, K, mu, labels);
		guide_image8u = cv::imread("out_wu.ppm", 1);
		get_label_center(guide_image8u, K, labels, mu);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
		break;
	case ClusterMethod::quantize_neural:
		quantization(guide_image8u, K, mu, labels);
		guide_image8u = cv::imread("out_neural.ppm", 1);
		get_label_center(guide_image8u, K, labels, mu);
		break;
	case ClusterMethod::kmeans_neural:
		quantization(guide_image8u, K, mu, labels);
		guide_image8u = cv::imread("out_neural.ppm", 1);
		get_label_center(guide_image8u, K, labels, mu);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
		break;
	case ClusterMethod::quantize_DIV:
		nQunat(guide_image8u, K, cm);
		guide_image8u = cv::imread("nQuantCpp/input-DIVquant" + std::to_string(K) + ".png", 1);
		get_label_center(guide_image8u, K, labels, mu);
		break;
	case ClusterMethod::kmeans_DIV:
		nQunat(guide_image8u, K, cm);
		guide_image8u = cv::imread("nQuantCpp/input-DIVquant" + std::to_string(K) + ".png", 1);
		get_label_center(guide_image8u, K, labels, mu);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
		break;
	case ClusterMethod::quantize_PNN:
		nQunat(guide_image8u, K, cm);
		guide_image8u = cv::imread("nQuantCpp/input-PNNquant" + std::to_string(K) + ".png", 1);
		get_label_center(guide_image8u, K, labels, mu);
		break;
	case ClusterMethod::kmeans_PNN:
		nQunat(guide_image8u, K, cm);
		guide_image8u = cv::imread("nQuantCpp/input-PNNquant" + std::to_string(K) + ".png", 1);
		get_label_center(guide_image8u, K, labels, mu);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
		break;
	case ClusterMethod::quantize_SPA:
		nQunat(guide_image8u, K, cm);
		guide_image8u = cv::imread("nQuantCpp/input-SPAquant" + std::to_string(K) + ".png", 1);
		get_label_center(guide_image8u, K, labels, mu);
		break;
	case ClusterMethod::kmeans_SPA:
		nQunat(guide_image8u, K, cm);
		guide_image8u = cv::imread("nQuantCpp/input-SPAquant" + std::to_string(K) + ".png", 1);
		get_label_center(guide_image8u, K, labels, mu);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
		break;
	case ClusterMethod::quantize_EAS:
		nQunat(guide_image8u, K, cm);
		guide_image8u = cv::imread("nQuantCpp/input-EASquant" + std::to_string(K) + ".png", 1);
		get_label_center(guide_image8u, K, labels, mu);
		break;
	case ClusterMethod::kmeans_EAS:
		nQunat(guide_image8u, K, cm);
		guide_image8u = cv::imread("nQuantCpp/input-EASquant" + std::to_string(K) + ".png", 1);
		get_label_center(guide_image8u, K, labels, mu);
		kmeans(reshaped_image32f, K, labels, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, mu);
		break;

	default:
		break;
	}

	//print_matinfo(mu);
	if (mu.rows != K)
	{
		std::cout << "Clustering is not working (mu.rows != K)" << std::endl;
	}
	CV_Assert(mu.rows == K);

	if (isTestClustering)testClustering(vguide);
}

void ConstantTimeHDGFSingleBase::downsampleForClustering(cv::Mat& src, cv::Mat& dest)
{
	switch (downsampleClusteringMethod)
	{
	case NEAREST:
	case LINEAR:
	case CUBIC:
	case AREA:
	case LANCZOS:
		//std::cout << "DOWNSAMPLE" << std::endl;
		cv::resize(src, dest,
			cv::Size(img_size.width / downsampleRate, img_size.height / downsampleRate),
			0, 0, downsampleClusteringMethod);
		//reshaped_image32f.convertTo(input_image8u, CV_8UC3);
		//cv::imshow("test", input_image8u);
		//cv::waitKey();
		dest = dest.reshape(1, (img_size.width / downsampleRate) * (img_size.height / downsampleRate));
		break;

	case IMPORTANCE_MAP:
		//std::cout << "IMPORTANCE MAP" << std::endl;
		cp::generateSamplingMaskRemappedDitherTexturenessPackedAoS(src, dest, 1.f / (downsampleRate * downsampleRate));
		dest = dest.reshape(1, reshaped_image32f.rows);
		break;

	default:
		break;
	}
}

void ConstantTimeHDGFSingleBase::downsampleImage(const std::vector<cv::Mat>& vsrc, std::vector<cv::Mat>& vsrcRes, const std::vector<cv::Mat>& vguide, std::vector<cv::Mat>& vguideRes, const int downsampleImageMethod)
{
	const double res = 1.0 / downSampleImage;
	if (downSampleImage != 1)
	{
		for (int c = 0; c < channels; c++)
		{
			resize(vsrc[c], vsrcRes[c], cv::Size(), res, res, downsampleImageMethod);
		}

		if (isJoint)
		{
			for (int c = 0; c < guide_channels; c++)
			{
				resize(vguide[c], vguideRes[c], cv::Size(), res, res, downsampleImageMethod);
			}
		}
	}
}

void sampling_imgproc(Mat& src_, Mat& dest)
{
	Mat src = src_.clone();

	
	double ss1 = 3.0;
	Mat temp;
	GaussianBlur(src, temp, Size((int)ceil(ss1 * 3) * 2 + 1, (int)ceil(ss1 * 3) * 2 + 1), ss1);
	absdiff(temp, src, dest);

	Size ksize = Size(5, 5);
	GaussianBlur(dest, dest, ksize, 1);
	
	/*Mat temp;
	cv::pyrDown(src, temp);
	cv::pyrUp(temp, dest);
	absdiff(dest, src, dest);
	Size ksize = Size(5, 5);
	GaussianBlur(dest, dest, ksize, 2);
	normalize(dest, dest, 0.f, 1.f, NORM_MINMAX);*/
}

void generateSamplingMaskRemappedDitherTest(vector<cv::Mat>& guide, cv::Mat& dest, const float sampling_ratio, const bool isUseAverage = false, int ditheringMethod = 0)
{
	CV_Assert(guide[0].depth() == CV_32F);

	const int channels = (int)guide.size();

	int sample_num = 0;
	cv::Mat mask(guide[0].size(), CV_8U);

	Mat v = guide[0].clone();
	/*for (int i = 1; i < guide.size(); i++)
	{
		add(v, guide[i], v);
	}*/
	v.convertTo(v, CV_32F, 1.0 / (1 * 255));
	sampling_imgproc(v, v);

	sample_num = cp::generateSamplingMaskRemappedDitherWeight(v, mask, sampling_ratio, ditheringMethod, cp::DITHER_SCANORDER::MEANDERING, 0.1, cp::DITHER_POSTPROCESS::NO_POSTPROCESS);
	sample_num = get_simd_floor(sample_num, 8);
	//print_debug(sample_num);
	dest.create(Size(sample_num, channels), CV_32F);

	AutoBuffer<float*> s(channels);
	AutoBuffer<float*> d(channels);
	for (int c = 0; c < channels; c++)
	{
		d[c] = dest.ptr<float>(c);
	}

	for (int y = 0, count = 0; y < mask.rows; y++)
	{
		uchar* mask_ptr = mask.ptr<uchar>(y);
		for (int c = 0; c < channels; c++)
		{
			s[c] = guide[c].ptr<float>(y);
		}

		for (int x = 0; x < mask.cols; x++)
		{
			if (mask_ptr[x] == 255)
			{
				for (int c = 0; c < channels; c++)
				{
					d[c][count] = s[c][x];
				}
				count++;
				if (count == sample_num)return;
			}
		}
	}
}
void ConstantTimeHDGFSingleBase::downsampleForClustering(std::vector<cv::Mat>& src, cv::Mat& dest, const bool isCropBoundary)
{
	const int channels = (int)src.size();
	//std::vector<cv::Mat> cropBuffer;
	cropBufferForClustering.resize(channels);
	if (isCropBoundary)
	{
		//std::cout << "Crop" << std::endl;
		const cv::Rect roi(cv::Point(boundaryLength, boundaryLength), cv::Size(img_size.width - 2 * boundaryLength, img_size.height - 2 * boundaryLength));
		for (int c = 0; c < channels; c++)
		{
			cropBufferForClustering[c] = src[c](roi).clone();
		}
	}
	else
	{
		for (int c = 0; c < channels; c++)
		{
			cropBufferForClustering[c] = src[c];
		}
	}

	switch (downsampleClusteringMethod)
	{
	case NEAREST:
	case LINEAR:
	case CUBIC:
	case AREA:
	case LANCZOS:
	{
		//std::cout << "DOWNSAMPLE" << std::endl;
		const cv::Size size = cv::Size(img_size.width / downsampleRate, img_size.height / downsampleRate);
		dest.create(cv::Size(size.area(), channels), CV_32F);
		for (int c = 0; c < channels; c++)
		{
			cv::Mat d(size, CV_32F, dest.ptr<float>(c));
			cv::resize(cropBufferForClustering[c], d, size, 0, 0, downsampleClusteringMethod);
		}
		break;
	}
	case IMPORTANCE_MAP:
		//std::cout << "IMPORTANCE MAP" << std::endl;
		cp::generateSamplingMaskRemappedDitherTexturenessPackedSoA(cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate), false);
		break;
	case IMPORTANCE_MAP2:
	{
		generateSamplingMaskRemappedDitherTest(cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate), false);
		/*
		const cv::Rect roi(cv::Point(boundaryLength, boundaryLength), cv::Size(img_size.width - 2 * boundaryLength, img_size.height - 2 * boundaryLength));
		//std::cout << "IMPORTANCE MAP" << std::endl;
		cropBufferForClustering2.resize(this->vsrc.size());
		for (int c = 0; c < this->vsrc.size(); c++)
		{
			cropBufferForClustering2[c] = this->vsrc[c](roi).clone();
		}
		cp::generateSamplingMaskRemappedDitherTexturenessPackedSoA(cropBufferForClustering2, cropBufferForClustering, dest, 1.f / (downsampleRate * downsampleRate));
		*/
	}
	break;

	default:
		break;
	}
}

void ConstantTimeHDGFSingleBase::mergeForClustering(std::vector<cv::Mat>& src, cv::Mat& dest, const bool isCropBoundary)
{
	const int channels = (int)src.size();
	//std::vector<cv::Mat> cropBuffer;
	cropBufferForClustering.resize(channels);
	if (isCropBoundary)
	{
		//std::cout << "Crop" << std::endl;
		const cv::Rect roi(cv::Point(boundaryLength, boundaryLength), cv::Size(img_size.width - 2 * boundaryLength, img_size.height - 2 * boundaryLength));
		for (int c = 0; c < channels; c++)
		{
			cropBufferForClustering[c] = src[c](roi).clone();
		}
	}
	else
	{
		for (int c = 0; c < channels; c++)
		{
			cropBufferForClustering[c] = src[c];
		}
	}

	dest.create(cv::Size(src[0].size().area(), channels), CV_32F);
	for (int i = 0; i < channels; i++)
	{
		src[i].copyTo(dest.row(i));
	}
}


void ConstantTimeHDGFSingleBase::filter(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod, int border)
{
	isJoint = false;
	statePCA = 0;

	if (src.channels() != 3)
	{
		std::cout << "channels is not 3" << std::endl;
		assert(src.channels() == 3);
	}

	setParameter(src.size(), sigma_space, sigma_range, cm,
		K, gf_method, gf_order, depth,
		isDownsampleClustering, downsampleRate, downsampleMethod);

	if (src.depth() == CV_32F)guide_image32f = src;
	else src.convertTo(guide_image32f, CV_32FC3);
	cv::split(guide_image32f, vsrc);

	body(vsrc, dst, std::vector<cv::Mat>());
}

void ConstantTimeHDGFSingleBase::filter(const std::vector<cv::Mat>& src, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod, int boundaryLength, int border)
{
	isJoint = false;
	statePCA = 0;

	setParameter(src[0].size(), sigma_space, sigma_range, cm,
		K, gf_method, gf_order, depth,
		isDownsampleClustering, downsampleRate, downsampleMethod, boundaryLength, border);

	guide_channels = channels = (int)src.size();
	vsrc.resize(channels);
	for (int c = 0; c < channels; c++)
	{
		vsrc[c] = src[c];
	}

	body(vsrc, dst, std::vector<cv::Mat>());
}

void ConstantTimeHDGFSingleBase::PCAfilter(const std::vector<cv::Mat>& src, const int pca_channels, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod, int boundaryLength, int border)
{
	isJoint = true;
	statePCA = 1;

	setParameter(src[0].size(), sigma_space, sigma_range, cm,
		K, gf_method, gf_order, depth,
		isDownsampleClustering, downsampleRate, downsampleMethod, boundaryLength, border);

	channels = (int)src.size();
	vsrc.resize(channels);
	for (int c = 0; c < channels; c++)
	{
		vsrc[c] = src[c];
	}

	guide_channels = pca_channels;
	cp::cvtColorPCA(vsrc, vguide, pca_channels, projectionMatrix);

	body(vsrc, dst, vguide);
}

void ConstantTimeHDGFSingleBase::jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, const double sigma_space, const double sigma_range, const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth, const bool isDownsampleClustering, const int downsampleRate, const int downsampleMethod, int border)
{
	isJoint = true;
	statePCA = 0;

	if (src.channels() != 3)
	{
		std::cout << "channels is not 3" << std::endl;
		assert(src.channels() == 3);
	}

	setParameter(src.size(), sigma_space, sigma_range, cm,
		K, gf_method, gf_order, depth,
		isDownsampleClustering, downsampleRate, downsampleMethod, border);
	//src.convertTo(input_image32f, CV_32FC3, 1.0 / 255.0);
	src.convertTo(guide_image32f, CV_32F);
	cv::split(guide_image32f, vsrc);

	guide.convertTo(guide_image32f, CV_32F);
	cv::split(guide_image32f, vguide);
	body(vsrc, dst, vguide);
}

void ConstantTimeHDGFSingleBase::jointfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, cv::Mat& dst, const double sigma_space, const double sigma_range, const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth, const bool isDownsampleClustering, const int downsampleRate, const int downsampleMethod, const int boundaryLength, int border)
{
	isJoint = true;
	statePCA = 0;

	setParameter(src[0].size(), sigma_space, sigma_range, cm,
		K, gf_method, gf_order, depth,
		isDownsampleClustering, downsampleRate, downsampleMethod, boundaryLength, border);

	//print_debug(src[0].size());
	//print_debug(GF->getRadius());

	channels = (int)src.size();
	vsrc.resize(channels);
	for (int c = 0; c < channels; c++)
	{
		vsrc[c] = src[c];
	}

	guide_channels = (int)guide.size();
	vguide.resize(guide_channels);
	for (int i = 0; i < guide_channels; i++)
	{
		vguide[i] = guide[i];
	}

	body(vsrc, dst, vguide);
}

void ConstantTimeHDGFSingleBase::jointPCAfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, const int pca_channels, cv::Mat& dst, const double sigma_space, const double sigma_range, const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth, const bool isDownsampleClustering, const int downsampleRate, const int downsampleMethod, const int boundaryLength, int border)
{
	isJoint = true;
	statePCA = 2;

	setParameter(src[0].size(), sigma_space, sigma_range, cm,
		K, gf_method, gf_order, depth,
		isDownsampleClustering, downsampleRate, downsampleMethod, boundaryLength, border);

	channels = (int)src.size();
	vsrc.resize(channels);
	for (int c = 0; c < channels; c++)
	{
		vsrc[c] = src[c];
	}

	guide_channels = pca_channels;
	vguide.resize(guide_channels);
	//std::cout << "jointPCA" << std::endl;
	cp::cvtColorPCA(guide, vguide, pca_channels, projectionMatrix, eigenValue);
	body(vsrc, dst, vguide);
}

void ConstantTimeHDGFSingleBase::nlmfilter(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& guide, cv::Mat& dst, const double sigma_space, const double sigma_range, const int parch_r, const int reduced_dim, const ClusterMethod cm, const int K, const cp::SpatialFilterAlgorithm gf_method, const int gf_order, const int depth, const bool isDownsampleClustering, const int downsampleRate, const int downsampleMethod, const int boundaryLength, int border)
{
	isJoint = true;

	setParameter(src[0].size(), sigma_space, sigma_range, cm,
		K, gf_method, gf_order, depth,
		isDownsampleClustering, downsampleRate, downsampleMethod, boundaryLength, border);

	channels = (int)src.size();
	vsrc.resize(channels);
	for (int c = 0; c < src.size(); c++)
	{
		vsrc[c] = src[c];
	}

	patchPCA(guide, vguide, parch_r, reduced_dim, border, patchPCAMethod, false, projectionMatrix, eigenValue);
	guide_channels = (int)vguide.size();
	/*for (int c = 0; c < guide_channels; c++)
	{
		double minv, maxv;
		minMaxLoc(vguide[c], &minv, &maxv);
		//print_debug(patchPCAMethod);
		if (maxv > 20000.0|| minv <-20000.0)
		{
			std::cout << c << " ";
			cp::printMinMax(vguide[c]);
			cv::Mat temp;
			cv::merge(guide, temp);
			cp::imshowScale("a", temp);
			cv::waitKey();
		}
	}*/
	body(vsrc, dst, vguide);
}

#pragma endregion

#pragma region setter
void ConstantTimeHDGFSingleBase::setGaussianFilterRadius(const int r)
{
	this->radius = r;
}

void ConstantTimeHDGFSingleBase::setGaussianFilter(const double sigma_space, const cp::SpatialFilterAlgorithm method, const int gf_order)
{
	bool isCompute = false;
	if (GF.empty())
	{
		isCompute = true;
	}
	else
	{
		if (GF->getSigma() != sigma_space ||
			GF->getAlgorithmType() != method ||
			GF->getOrder() != gf_order ||
			GF->getSize() != img_size / downSampleImage)
		{
			isCompute = true;
		}
	}

	if (isCompute)
	{
		//std::cout << "createcreateSpatialFilter" << std::endl;
		GF = cp::createSpatialFilter(method, CV_32F, cp::SpatialKernel::GAUSSIAN);
		const int boundaryLength = 0;//should be fixed
		GF->setIsInner(boundaryLength, boundaryLength, boundaryLength, boundaryLength);
		//if (radius != 0)
		{
			//GF->setFixRadius(radius);

		}
	}
}


void ConstantTimeHDGFSingleBase::setBoundaryLength(const int length)
{
	boundaryLength = length;
}

void ConstantTimeHDGFSingleBase::setParameter(cv::Size img_size, double sigma_space, double sigma_range, ClusterMethod cm, int K, cp::SpatialFilterAlgorithm method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod, int boundarylength, int borderType)
{
	this->num_sample_max = cv::Size(img_size.width / downsampleRate, img_size.height / downsampleRate).area();
	this->depth = depth;

	this->K = std::min(K, num_sample_max);
	if (this->K == num_sample_max) 	std::cout << "full sample (debug message)" << K << "/" << num_sample_max << std::endl;
	//std::cout << (double)K / num_sample_max << std::endl;
	this->img_size = img_size;
	this->sigma_space = sigma_space;
	this->spatial_order = gf_order;
	this->sigma_range = sigma_range;

	this->cm = cm;
	this->setBoundaryLength(boundarylength);
	this->borderType = borderType;
	//this->setGaussianFilterRadius((int)ceil(this->sigma_space * 3));
	this->setGaussianFilter(sigma_space / downSampleImage, method, gf_order);

	this->isDownsampleClustering = isDownsampleClustering;
	this->downsampleRate = downsampleRate;
	this->downsampleClusteringMethod = downsampleMethod;
}

void ConstantTimeHDGFSingleBase::setConcat_offset(int concat_offset)
{
	this->concat_offset = concat_offset;
}

void ConstantTimeHDGFSingleBase::setPca_r(int pca_r)
{
	this->pca_r = pca_r;
}

void ConstantTimeHDGFSingleBase::setKmeans_ratio(float kmeans_ratio)
{
	this->kmeans_ratio = kmeans_ratio;
}

void ConstantTimeHDGFSingleBase::setCropClustering(bool isCropClustering)
{
	this->isCropBoundaryClustering = isCropClustering;
}

void ConstantTimeHDGFSingleBase::setPatchPCAMethod(int method)
{
	this->patchPCAMethod = method;
}

cv::Mat ConstantTimeHDGFSingleBase::getSamplingPoints()
{
	return this->mu;
}

cv::Mat ConstantTimeHDGFSingleBase::cloneEigenValue()
{
	return this->eigenValue.clone();
}

void ConstantTimeHDGFSingleBase::printRapTime()
{
	for (int i = 0; i < num_timerblock_max; i++)
	{
		timer[i].getLapTimeMedian(true, cv::format("%d", i));
	}
}
#pragma endregion

#pragma region tileHDGF
using namespace cv;
TileHDGF::TileHDGF(cv::Size div_) :
	thread_max(omp_get_max_threads()), div(div_)
{
	;
}

TileHDGF::~TileHDGF()
{
	;
}

void TileHDGF::nlmFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, const int patch_r, const int reduced_dim, const int pca_method, double truncateBoundary, const int borderType)
{
	channels = src.channels();
	guide_channels = guide.channels();

	if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

	const int vecsize = sizeof(__m256) / sizeof(float);//8

	int d = 2 * (int)(ceil(sigma_space * 3.0)) + 1;
	Size ksize = Size(d, d);
	if (div.area() == 1)
	{
		cp::highDimensionalGaussianFilter(src, guide, dst, ksize, (float)sigma_range, (float)sigma_space, borderType);
		tileSize = src.size();
	}
	else
	{
		int r = (int)ceil(truncateBoundary * sigma_space);
		const int R = get_simd_ceil(r, 8);
		tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
		divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

		if (split_dst.size() != channels) split_dst.resize(channels);

		for (int c = 0; c < channels; c++)
		{
			split_dst[c].create(tileSize, CV_32FC1);
		}

		if (subImageInput.empty())
		{
			subImageInput.resize(thread_max);
			subImageGuide.resize(thread_max);
			subImageOutput.resize(thread_max);
			for (int n = 0; n < thread_max; n++)
			{
				subImageInput[n].resize(channels);
				subImageGuide[n].resize(guide_channels);
				subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
			}
		}
		else
		{
			if (subImageGuide.empty()) subImageGuide.resize(thread_max);
			if (subImageGuide[0].size() != guide_channels)
			{
				for (int n = 0; n < thread_max; n++)
				{
					subImageGuide[n].resize(guide_channels);
				}
			}
		}


		if (src.channels() != 3)split(src, srcSplit);
		if (guide.channels() != 3)split(guide, guideSplit);

#pragma omp parallel for schedule(static)
		for (int n = 0; n < div.area(); n++)
		{
			const int thread_num = omp_get_thread_num();
			const cv::Point idx = cv::Point(n % div.width, n / div.width);


			if (src.channels() == 3)
			{
				cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
			}
			else
			{
				for (int c = 0; c < srcSplit.size(); c++)
				{
					cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
			}
			if (guide.channels() == 3)
			{
				cp::cropSplitTileAlign(guide, subImageGuide[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
			}
			else
			{
				for (int c = 0; c < guideSplit.size(); c++)
				{
					cp::cropTileAlign(guideSplit[c], subImageGuide[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
			}
			Mat s, g;
			merge(subImageInput[thread_num], s);
			std::vector<Mat> buff;
			patchPCA(subImageGuide[thread_num], buff, patch_r, reduced_dim, borderType, pca_method);

			merge(buff, g);
			cp::highDimensionalGaussianFilter(s, g, subImageOutput[thread_num], ksize, (float)sigma_range, (float)sigma_space, borderType);
			cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
		}
	}
}

double getError(Mat& rgb, Mat& gray)
{
	double ret = 0.0;
	float* s = rgb.ptr<float>();
	float* g = gray.ptr<float>();
	for (int i = 0; i < rgb.size().area(); i++)
	{
		double dist
			= (s[3 * i + 0] - g[i]) * (s[3 * i + 0] - g[i])
			+ (s[3 * i + 1] - g[i]) * (s[3 * i + 1] - g[i])
			+ (s[3 * i + 2] - g[i]) * (s[3 * i + 2] - g[i]);
		ret += sqrt(dist);
	}
	return ret / rgb.size().area();
}

cp::RGBHistogram rgbh;
void TileHDGF::cvtgrayFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, const int method, double truncateBoundary)
{
	static int bb = 50; createTrackbar("b", "", &bb, 300);
	static int gg = 50; createTrackbar("g", "", &gg, 300);
	static int rr = 50; createTrackbar("r", "", &rr, 300);
	static int index = 0; createTrackbar("index", "", &index, div.area() - 1);
	channels = src.channels();
	guide_channels = guide.channels();

	if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

	const int borderType = cv::BORDER_REFLECT;
	const int vecsize = sizeof(__m256) / sizeof(float);//8

	int d = 2 * (int)(ceil(sigma_space * 3.0)) + 1;
	Size ksize = Size(d, d);
	if (div.area() == 1)
	{
		cp::highDimensionalGaussianFilter(src, guide, dst, ksize, (float)sigma_range, (float)sigma_space, borderType);
		tileSize = src.size();
	}
	else
	{
		int r = (int)ceil(truncateBoundary * sigma_space);
		const int R = get_simd_ceil(r, 8);
		tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
		divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

		if (split_dst.size() != channels) split_dst.resize(channels);

		for (int c = 0; c < channels; c++)
		{
			split_dst[c].create(tileSize, CV_32FC1);
		}

		if (subImageInput.empty())
		{
			subImageInput.resize(thread_max);
			subImageGuide.resize(thread_max);
			subImageOutput.resize(thread_max);
			for (int n = 0; n < thread_max; n++)
			{
				subImageInput[n].resize(channels);
				subImageGuide[n].resize(guide_channels);
				subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
			}
		}
		else
		{
			if (subImageGuide.empty()) subImageGuide.resize(thread_max);
			if (subImageGuide[0].size() != guide_channels)
			{
				for (int n = 0; n < thread_max; n++)
				{
					subImageGuide[n].resize(guide_channels);
				}
			}
		}

		std::vector<cv::Mat> srcSplit;
		std::vector<cv::Mat> guideSplit;
		if (src.channels() != 3)split(src, srcSplit);
		if (guide.channels() != 3)split(guide, guideSplit);

		//#pragma omp parallel for schedule(static)
		for (int n = 0; n < div.area(); n++)
		{
			const int thread_num = omp_get_thread_num();
			const cv::Point idx = cv::Point(n % div.width, n / div.width);

			if (src.channels() == 3)
			{
				cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
			}
			else
			{
				for (int c = 0; c < srcSplit.size(); c++)
				{
					cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
			}
			if (guide.channels() == 3)
			{
				cp::cropSplitTileAlign(guide, subImageGuide[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
			}
			else
			{
				for (int c = 0; c < guideSplit.size(); c++)
				{
					cp::cropTileAlign(guideSplit[c], subImageGuide[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
			}
			Mat s, mergedGuide, mergedGuideReduction;
			merge(subImageGuide[thread_num], mergedGuide);
			merge(subImageInput[thread_num], s);
			if (method == 0)
			{
				cp::cvtColorAverageGray(mergedGuide, mergedGuideReduction, true);
			}
			else
			{
				namedWindow("3DPlot");
				moveWindow("3DPlot", 800, 200);
				Scalar a = Scalar(bb, gg, rr, 0);//mean(g);
				Mat evec, eval, mean, temp;
				cp::cvtColorPCA(mergedGuide, temp, 1, evec, eval, mean);
				Scalar ev0 = Scalar(evec.at<double>(0, 0), evec.at<double>(0, 1), evec.at<double>(0, 2), 0);
				Scalar ev1 = Scalar(evec.at<double>(1, 0), evec.at<double>(1, 1), evec.at<double>(1, 2), 0);
				a = eval.at<double>(0) * ev0 + eval.at<double>(1) * ev1;
				Scalar b = Scalar(255.0, 255.0, 255.0, 255.0);
				Scalar m, v;
				meanStdDev(mergedGuide, m, v);

				//double al = 0.95;
				//a = (al * a + (1.0 - al) * b);


				Mat mat(1, 3, CV_32F);
				double norm = 1.f / sqrt(a.val[0] * a.val[0] + a.val[1] * a.val[1] + a.val[2] * a.val[2]);
				mat.at<float>(0) = float(a.val[0] * norm);
				mat.at<float>(1) = float(a.val[1] * norm);
				mat.at<float>(2) = float(a.val[2] * norm);
				transform(mergedGuide, mergedGuideReduction, mat);
				if (n == index)
				{

					Mat evec, eval, mean, temp;
					cp::cvtColorPCA(mergedGuide, temp, 1, evec, eval, mean);
					std::cout << evec << std::endl;
					std::cout << eval << std::endl;
					//std::system("cls");
					//std::cout << "index:" << idx.y * div.width + idx.x << std::endl;
					//std::cout << getError(g, gr) << std::endl;
					//std::cout << mat << std::endl;
					//std::cout << "cave" << m << std::endl;
					//std::cout << "cvar" << v << std::endl;
					//std::cout << "xxxx" << v * sqrt(3) / (v.val[0] + v.val[1] + v.val[2]) << std::endl;
					//meanStdDev(mergedGuideReduction, m, v);
					//std::cout << "gave" << m << std::endl;
					//std::cout << "gvar" << m << std::endl << std::endl;
					//rgbh.push_back(g);

#pragma omp critical
					{
						//rgbh.push_back_line(0, 0, 0, mat.at<float>(0) * 255*sqrt(3.0), mat.at<float>(1) * 255 * sqrt(3.0), mat.at<float>(2) * 255 * sqrt(3.0));
						Point3f s = Point3f((float)mean.at<double>(0), (float)mean.at<double>(1), (float)mean.at<double>(2));
						Point3f d = Point3f(mat.at<float>(0), mat.at<float>(1), mat.at<float>(2)) * 100.f * sqrt(3.f);
						rgbh.push_back_line(s - d, s + d);
						cp::imshowScale("a", mergedGuide);
						rgbh.plot(mergedGuide, false, "3DPlot");
						rgbh.clear();
					}
				}

			}
			cp::highDimensionalGaussianFilter(s, mergedGuideReduction, subImageOutput[thread_num], ksize, (float)sigma_range, (float)sigma_space, borderType);
			cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
		}
	}
}

void TileHDGF::pcaFilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, const int reduced_dim, double truncateBoundary)
{
	channels = src.channels();
	guide_channels = guide.channels();

	if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

	const int borderType = cv::BORDER_REFLECT;
	const int vecsize = sizeof(__m256) / sizeof(float);//8

	int d = 2 * (int)(ceil(sigma_space * 3.0)) + 1;
	Size ksize = Size(d, d);
	if (div.area() == 1)
	{
		cp::highDimensionalGaussianFilter(src, guide, dst, ksize, (float)sigma_range, (float)sigma_space, borderType);
		tileSize = src.size();
	}
	else
	{
		int r = (int)ceil(truncateBoundary * sigma_space);
		const int R = get_simd_ceil(r, 8);
		tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
		divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

		if (split_dst.size() != channels) split_dst.resize(channels);

		for (int c = 0; c < channels; c++)
		{
			split_dst[c].create(tileSize, CV_32FC1);
		}

		if (subImageInput.empty())
		{
			subImageInput.resize(thread_max);
			subImageGuide.resize(thread_max);
			subImageOutput.resize(thread_max);
			for (int n = 0; n < thread_max; n++)
			{
				subImageInput[n].resize(channels);
				subImageGuide[n].resize(guide_channels);
				subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
			}
		}
		else
		{
			if (subImageGuide.empty()) subImageGuide.resize(thread_max);
			if (subImageGuide[0].size() != guide_channels)
			{
				for (int n = 0; n < thread_max; n++)
				{
					subImageGuide[n].resize(guide_channels);
				}
			}
		}

		std::vector<cv::Mat> srcSplit;
		std::vector<cv::Mat> guideSplit;
		if (src.channels() != 3)split(src, srcSplit);
		if (guide.channels() != 3)split(guide, guideSplit);

#pragma omp parallel for schedule(static)
		for (int n = 0; n < div.area(); n++)
		{
			const int thread_num = omp_get_thread_num();
			const cv::Point idx = cv::Point(n % div.width, n / div.width);


			if (src.channels() == 3)
			{
				cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
			}
			else
			{
				for (int c = 0; c < srcSplit.size(); c++)
				{
					cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
			}
			if (guide.channels() == 3)
			{
				cp::cropSplitTileAlign(guide, subImageGuide[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
			}
			else
			{
				for (int c = 0; c < guideSplit.size(); c++)
				{
					cp::cropTileAlign(guideSplit[c], subImageGuide[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
			}
			Mat s, g;
			std::vector<Mat> buff;
			cp::cvtColorPCA(subImageGuide[thread_num], buff, reduced_dim);
			merge(subImageInput[thread_num], s);
			merge(buff, g);
			cp::highDimensionalGaussianFilter(s, g, subImageOutput[thread_num], ksize, (float)sigma_range, (float)sigma_space, borderType);
			cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
		}
	}
}

void TileHDGF::filter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, double truncateBoundary)
{
	channels = src.channels();
	guide_channels = guide.channels();

	if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));

	const int borderType = cv::BORDER_REFLECT;
	const int vecsize = sizeof(__m256) / sizeof(float);//8

	int d = 2 * (int)(ceil(sigma_space * 3.0)) + 1;
	Size ksize = Size(d, d);
	if (div.area() == 1)
	{
		cp::highDimensionalGaussianFilter(src, guide, dst, ksize, (float)sigma_range, (float)sigma_space, borderType);
		tileSize = src.size();
	}
	else
	{
		int r = (int)ceil(truncateBoundary * sigma_space);
		const int R = get_simd_ceil(r, 8);
		tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
		divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

		if (split_dst.size() != channels) split_dst.resize(channels);

		for (int c = 0; c < channels; c++)
		{
			split_dst[c].create(tileSize, CV_32FC1);
		}

		if (subImageInput.empty())
		{
			subImageInput.resize(thread_max);
			subImageGuide.resize(thread_max);
			subImageOutput.resize(thread_max);
			for (int n = 0; n < thread_max; n++)
			{
				subImageInput[n].resize(channels);
				subImageGuide[n].resize(guide_channels);
				subImageOutput[n].create(tileSize, CV_MAKETYPE(CV_32F, channels));
			}
		}
		else
		{
			if (subImageGuide.empty()) subImageGuide.resize(thread_max);
			if (subImageGuide[0].size() != guide_channels)
			{
				for (int n = 0; n < thread_max; n++)
				{
					subImageGuide[n].resize(guide_channels);
				}
			}
		}

		std::vector<cv::Mat> srcSplit;
		std::vector<cv::Mat> guideSplit;
		if (src.channels() != 3)split(src, srcSplit);
		if (guide.channels() != 3)split(guide, guideSplit);

#pragma omp parallel for schedule(static)
		for (int n = 0; n < div.area(); n++)
		{
			const int thread_num = omp_get_thread_num();
			const cv::Point idx = cv::Point(n % div.width, n / div.width);


			if (src.channels() == 3)
			{
				cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
			}
			else
			{
				for (int c = 0; c < srcSplit.size(); c++)
				{
					cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
			}
			if (guide.channels() == 3)
			{
				cp::cropSplitTileAlign(guide, subImageGuide[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
			}
			else
			{
				for (int c = 0; c < guideSplit.size(); c++)
				{
					cp::cropTileAlign(guideSplit[c], subImageGuide[thread_num][c], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
				}
			}
			Mat s, g;
			merge(subImageInput[thread_num], s);
			merge(subImageGuide[thread_num], g);
			cp::highDimensionalGaussianFilter(s, g, subImageOutput[thread_num], ksize, (float)sigma_range, (float)sigma_space, borderType);
			cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
		}
	}
}

cv::Size TileHDGF::getTileSize()
{
	return tileSize;
}

void TileHDGF::getTileInfo()
{
	print_debug(div);
	print_debug(divImageSize);
	print_debug(tileSize);
	int borderLength = (tileSize.width - divImageSize.width) / 2;
	print_debug(borderLength);

}
#pragma endregion
