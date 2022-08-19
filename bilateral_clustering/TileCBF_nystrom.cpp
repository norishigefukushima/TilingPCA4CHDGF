#include "CBF.hpp"
#include "init_center.hpp"
#include <opencp.hpp>

TileConstantTimeHDGF_Nystrom::TileConstantTimeHDGF_Nystrom(cv::Size div_) :
	thread_max(omp_get_max_threads()), div(div_)
{
	if (div.area() == 1)
	{
		scbf = new ConstantTimeHDGF_NystromSingle * [1];
		scbf[0] = new ConstantTimeHDGF_NystromSingle;
	}
	else
	{
		scbf = new ConstantTimeHDGF_NystromSingle * [thread_max];
		for (int i = 0; i < thread_max; i++)
		{
			scbf[i] = new ConstantTimeHDGF_NystromSingle;
		}
	}
}

TileConstantTimeHDGF_Nystrom::~TileConstantTimeHDGF_Nystrom()
{
	if (div.area() == 1)
	{
		for (int i = 0; i < 1; i++)
		{
			delete scbf[i];
		}
		delete[] scbf;
	}
	else
	{
		for (int i = 0; i < thread_max; i++)
		{
			delete scbf[i];
		}
		delete[] scbf;
	}
}

void TileConstantTimeHDGF_Nystrom::setBoundaryLength(const int length)
{
	if (div.area() == 1)
	{
		scbf[0]->setBoundaryLength(length);
	}
	else
	{
		for (int i = 0; i < thread_max; i++)
		{
			scbf[i]->setBoundaryLength(length);
		}
	}
}

void TileConstantTimeHDGF_Nystrom::setKMeansAttempts(const int attempts)
{
	if (div.area() == 1)
	{
		scbf[0]->setKMeansAttempts(attempts);
	}
	else
	{
		for (int i = 0; i < thread_max; i++)
		{
			scbf[i]->setKMeansAttempts(attempts);
		}
	}
}

void TileConstantTimeHDGF_Nystrom::setKMeansSigma(const double sigma)
{
	if (div.area() == 1)
	{
		scbf[0]->setKMeansSigma(sigma);
	}
	else
	{
		for (int i = 0; i < thread_max; i++)
		{
			//std::cout << "set sigma" << sigma << std::endl;
			scbf[i]->setKMeansSigma(sigma);
		}
	}
}

void TileConstantTimeHDGF_Nystrom::setNumIterations(const int iterations)
{
	if (div.area() == 1)
	{
		scbf[0]->setNumIterations(iterations);
	}
	else
	{
		for (int i = 0; i < thread_max; i++)
		{
			scbf[i]->setNumIterations(iterations);
		}
	}
}

void TileConstantTimeHDGF_Nystrom::setConcat_offset(int concat_offset)
{
	if (div.area() == 1)
	{
		scbf[0]->setConcat_offset(concat_offset);
	}
	else
	{
		for (int i = 0; i < thread_max; i++)
		{
			scbf[i]->setConcat_offset(concat_offset);
		}
	}
}

void TileConstantTimeHDGF_Nystrom::setPca_r(int pca_r)
{
	if (div.area() == 1)
	{
		scbf[0]->setPca_r(pca_r);
	}
	else
	{
		for (int i = 0; i < thread_max; i++)
		{
			scbf[i]->setPca_r(pca_r);
		}
	}
}

void TileConstantTimeHDGF_Nystrom::setKmeans_ratio(float kmeans_ratio)
{
	if (div.area() == 1)
	{
		scbf[0]->setKmeans_ratio(kmeans_ratio);
	}
	else
	{
		for (int i = 0; i < thread_max; i++)
		{
			scbf[i]->setKmeans_ratio(kmeans_ratio);
		}
	}
}

void TileConstantTimeHDGF_Nystrom::setCropClustering(bool isCropClustering)
{
	if (div.area() > 1)
	{		
		for (int i = 0; i < thread_max; i++)
		{
			scbf[i]->setCropClustering(isCropClustering);
		}
	}
}

using namespace cv;
void createSubImageCV(const Mat& src, Mat& dest, const Size div, const Point index, const int top, const int bottom, const int left, const int right, const int borderType)
{
	const int tile_width = src.cols / div.width;
	const int tile_height = src.rows / div.height;

	Mat im; copyMakeBorder(src, im, top, bottom, left, right, borderType);
	dest.create(Size(tile_width + left + right, tile_height + left + right), src.type());
	Rect roi = Rect(tile_width * index.x, tile_height * index.y, dest.cols, dest.rows);
	im(roi).copyTo(dest);
}

void createSubImageCVAlign(const cv::Mat& src, cv::Mat& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType, const int align_x, const int align_y, const int left_multiple = 1, const int top_multiple = 1)
{
	const int tilex = src.cols / div_size.width;
	const int tiley = src.rows / div_size.height;

	const int L = get_simd_ceil(r, left_multiple);
	const int T = get_simd_ceil(r, top_multiple);

	const int align_width = get_simd_ceil(tilex + L + r, align_x);
	const int padx = align_width - (tilex + L + r);
	const int align_height = get_simd_ceil(tiley + T + r, align_y);
	const int pady = align_height - (tiley + T + r);
	const int R = r + padx;
	const int B = r + pady;

	createSubImageCV(src, dest, div_size, idx, T, B, L, R, borderType);
}

void createSubImageCVAlign(const cv::Mat& src, std::vector<cv::Mat>& dest, const cv::Size div_size, const cv::Point idx, const int r, const int borderType, const int align_x, const int align_y, const int left_multiple = 1, const int top_multiple = 1)
{
	Mat temp;
	createSubImageCVAlign(src, temp, div_size, idx, r, borderType, align_x, align_y, left_multiple, top_multiple);
	split(temp, dest);
}

void TileConstantTimeHDGF_Nystrom::filtering(const cv::Mat& src, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, gf::GFMethod gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod, double truncateBoundary)
{
	if (src.channels() != 3)
	{
		std::cout << "channels is not 3" << std::endl;
		assert(src.channels() == 3);
	}

	if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_32FC3);

	const int borderType = cv::BORDER_REFLECT;
	const int vecsize = sizeof(__m256) / sizeof(float);//8

	if (div.area() == 1)
	{
		scbf[0]->filtering(src, dst, sigma_space, sigma_range, cm, K, gf_method
			, gf_order, depth, isDownsampleClustering, downsampleRate, downsampleMethod);
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
			subImageOutput.resize(thread_max);
			for (int n = 0; n < thread_max; n++)
			{
				subImageInput[n].resize(channels);
				subImageOutput[n].create(tileSize, CV_32FC3);
			}
		}

#pragma omp parallel for schedule(static)
		for (int n = 0; n < div.area(); n++)
		{
			const int thread_num = omp_get_thread_num();
			const cv::Point idx = cv::Point(n % div.width, n / div.width);

			cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
			//createSubImageCVAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);

			scbf[thread_num]->filtering(subImageInput[thread_num], subImageOutput[thread_num], sigma_space, sigma_range, cm, K, gf_method, gf_order, depth, isDownsampleClustering, downsampleRate, downsampleMethod, R);
			//merge(subImageInput[thread_num], subImageOutput[thread_num]);

			cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
		}
	} 
}


void TileConstantTimeHDGF_Nystrom::jointfilter(const cv::Mat& src, const cv::Mat& guide, cv::Mat& dst, double sigma_space, double sigma_range, ClusterMethod cm, int K, gf::GFMethod gf_method, int gf_order, int depth, bool isDownsampleClustering, int downsampleRate, int downsampleMethod, double truncateBoundary)
{
	guide_channels = guide.channels();
	if (src.channels() != 3)
	{
		std::cout << "channels is not 3" << std::endl;
		assert(src.channels() == 3);
	}

	if (dst.empty() || dst.size() != src.size()) dst.create(src.size(), CV_32FC3);

	const int borderType = cv::BORDER_REFLECT;
	const int vecsize = sizeof(__m256) / sizeof(float);//8

	if (div.area() == 1)
	{
		scbf[0]->jointfilter(src, guide, dst, sigma_space, sigma_range, cm, K, gf_method
			, gf_order, depth, isDownsampleClustering, downsampleRate, downsampleMethod);
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
				subImageOutput[n].create(tileSize, CV_32FC3);
			}
		}
		else
		{
			if (subImageGuide[0].size() != guide_channels)
			{
				for (int n = 0; n < thread_max; n++)
				{
					subImageGuide[n].resize(guide_channels);
				}
			}
		}

		std::vector<cv::Mat> guideSplit;
		if (guide.channels() != 3)split(guide, guideSplit);

#pragma omp parallel for schedule(static)
		for (int n = 0; n < div.area(); n++)
		{
			const int thread_num = omp_get_thread_num();
			const cv::Point idx = cv::Point(n % div.width, n / div.width);

			cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);
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
			//createSubImageCVAlign(src, subImageInput[thread_num], div, idx, r, borderType, vecsize, vecsize, vecsize, vecsize);

			scbf[thread_num]->jointfilter(subImageInput[thread_num], subImageGuide[thread_num], subImageOutput[thread_num], sigma_space, sigma_range, cm, K, gf_method, gf_order, depth, isDownsampleClustering, downsampleRate, downsampleMethod, R);
			//merge(subImageInput[thread_num], subImageOutput[thread_num]);

			cp::pasteTileAlign(subImageOutput[thread_num], dst, div, idx, r, 8, 8);
		}
	}
}


cv::Size TileConstantTimeHDGF_Nystrom::getTileSize()
{
	return tileSize;
}

void TileConstantTimeHDGF_Nystrom::getTileInfo()
{
	print_debug(div);
	print_debug(divImageSize);
	print_debug(tileSize);
	int borderLength = (tileSize.width - divImageSize.width) / 2;
	print_debug(borderLength);

}
