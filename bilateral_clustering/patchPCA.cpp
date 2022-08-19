#include "patchPCA.hpp"
#include <opencp.hpp>

using namespace cv;
using namespace std;

//#define SCHEDULE schedule(dynamic)
#define SCHEDULE schedule(static)

class CalcPatchCovarMatrix
{
public:
	void computeCov(const std::vector<cv::Mat>& src, const int patch_rad, cv::Mat& cov, const NeighborhoodPCA method = NeighborhoodPCA::MEAN_SUB_32F, const int skip = 1, const bool isParallel = false);
	void computeCov(const cv::Mat& src, const int patch_rad, cv::Mat& cov, const NeighborhoodPCA method = NeighborhoodPCA::MEAN_SUB_32F, const int skip = 1, const bool isParallel = false);

private:
	int patch_rad;
	int D;
	int color_channels;
	int dim;
	std::vector<cv::Mat> data;
	void getScanorder(int* scan, const int step, const int channels);

	void setCoVar(const std::vector<double>& meanv, const std::vector<double>& mulSum, cv::Mat& covar, double* sum, const double normalSize);

	void naive(const std::vector<cv::Mat>& src, cv::Mat& cov, const int skip);
	void naive(const std::vector<cv::Mat>& src, cv::Mat& cov, cv::Mat& mask);

	enum class CovMethod
	{
		MEAN_SUB,
		CONSTANT_SUB,
		NO_SUB,
		FULL_SUB
	};
	CovMethod getCovMethod(const NeighborhoodPCA method);

	template<int color_channels, int dim>
	void simd_32F(const std::vector<cv::Mat>& src_, cv::Mat& cov, const CovMethod method, const float constant_sub = 127.5f);
	void simd_32FCn(const std::vector<cv::Mat>& src_, cv::Mat& cov, const CovMethod method, const float constant_sub = 127.5f);

	template<int color_channels, int dim>
	void simdOMP_32F(const std::vector<cv::Mat>& src_, cv::Mat& cov, const CovMethod method, const float constant_sub = 127.5f);
	void simdOMP_32FCn(const std::vector<cv::Mat>& src_, cv::Mat& cov, const CovMethod method, const float constant_sub = 127.5f);
};

inline bool is32F(const NeighborhoodPCA method)
{
	bool ret = false;
	switch (method)
	{
	case NeighborhoodPCA::MEAN_SUB_32F:
		ret = true; break;
	default:
		break;
	}
	return ret;
}

CalcPatchCovarMatrix::CovMethod CalcPatchCovarMatrix::getCovMethod(const NeighborhoodPCA method)
{
	CalcPatchCovarMatrix::CovMethod cm = CalcPatchCovarMatrix::CovMethod::MEAN_SUB;
	switch (method)
	{
	case NeighborhoodPCA::MEAN_SUB_32F:
		cm = CalcPatchCovarMatrix::CovMethod::MEAN_SUB; break;

	default:
		break;
	}
	return cm;
}

#pragma region CalcPatchCovarMatrix

void CalcPatchCovarMatrix::computeCov(const vector<Mat>& src, const int patch_rad, Mat& cov, const NeighborhoodPCA method, const int skip, const bool isParallel)
{
	this->patch_rad = patch_rad;
	D = 2 * patch_rad + 1;
	color_channels = (int)src.size();
	dim = color_channels * D * D;

	cov.create(dim, dim, CV_64F);

	//cout << method << endl;
	//cp::Timer t;
	if (method == NeighborhoodPCA::OPENCV_PCA || method == NeighborhoodPCA::OPENCV_COV)
	{
		Mat highDimGuide(src[0].size(), CV_MAKE_TYPE(CV_32F, dim));
		cp::IM2COL(src, highDimGuide, patch_rad, BORDER_REFLECT101);
		Mat x = highDimGuide.reshape(1, src[0].size().area());
		Mat mean = Mat::zeros(dim, 1, CV_64F);
		cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
	}
	else
	{
		if (skip == 1)
		{
			CovMethod cm = getCovMethod(method);
			if (isParallel)
			{
				if (is32F(method))
				{
					if (color_channels == 1 && dim == 1)       simdOMP_32F<1, 1>(src, cov, cm);
					else if (color_channels == 3 && dim == 3)  simdOMP_32F<3, 3>(src, cov, cm);
					else if (color_channels == 1 && dim == 9)  simdOMP_32F<1, 9>(src, cov, cm);
					else if (color_channels == 3 && dim == 27) simdOMP_32F<3, 27>(src, cov, cm);
					else if (color_channels == 1 && dim == 25) simdOMP_32F<1, 25>(src, cov, cm);
					else if (color_channels == 3 && dim == 75) simdOMP_32F<3, 75>(src, cov, cm);
					else if (color_channels == 1 && dim == 49) simdOMP_32F<1, 49>(src, cov, cm);
					else if (color_channels == 3 && dim == 147)simdOMP_32F<3, 147>(src, cov, cm);
					else if (color_channels == 1 && dim == 81) simdOMP_32F<1, 81>(src, cov, cm);
					else if (color_channels == 3 && dim == 243)simdOMP_32F<3, 243>(src, cov, cm);
					else simdOMP_32FCn(src, cov, cm);
				}
			}
			else
			{
				if (color_channels == 1 && dim == 9)simd_32F<1, 9>(src, cov, cm);
				else if (color_channels == 3 && dim == 27)simd_32F<3, 27>(src, cov, cm);
				else if (color_channels == 1 && dim == 25)simd_32F<1, 25>(src, cov, cm);
				else if (color_channels == 3 && dim == 75)simd_32F<3, 75>(src, cov, cm);
				else if (color_channels == 1 && dim == 49) simd_32F<1, 49>(src, cov, cm);
				else if (color_channels == 3 && dim == 147)simd_32F<3, 147>(src, cov, cm);
				else if (color_channels == 1 && dim == 81) simd_32F<1, 81>(src, cov, cm);
				else if (color_channels == 3 && dim == 243)simd_32F<3, 243>(src, cov, cm);
				else simd_32FCn(src, cov, cm);
			}
		}
	}
}

//call vector<Mat> version
void CalcPatchCovarMatrix::computeCov(const Mat& src, const int patch_rad, Mat& cov, const NeighborhoodPCA method, const int skip, const bool isParallel)
{
	vector<Mat> vsrc(src.channels());
	if (src.channels() == 1) vsrc[0] = src;
	else split(src, vsrc);

	computeCov(vsrc, patch_rad, cov, method, skip, isParallel);
}

void CalcPatchCovarMatrix::getScanorder(int* scan, const int step, const int channels)
{
	int idx = 0;
	for (int j = 0; j < D; j++)
	{
		for (int i = 0; i < D; i++)
		{
			for (int k = 0; k < channels; k++)
			{
				//scan[idx++] = channels * src.cols * j + channels * i + k;
				scan[idx++] = channels * step * (j - patch_rad) + channels * (i - patch_rad) + k;
			}
		}
	}
}

void CalcPatchCovarMatrix::setCoVar(const vector<double>& meanv, const vector<double>& mulSum, Mat& covar, double* sum, const double normalSize)
{
	for (int k = 0, idx = 0; k < dim; k++)
	{
		const int c1 = k % color_channels;
		covar.at<double>(k, k) = mulSum[c1] * normalSize - meanv[c1] * meanv[c1];

		for (int l = k + 1; l < dim; l++)
		{
			const int c2 = l % color_channels;
			covar.at<double>(k, l) = covar.at<double>(l, k) = sum[idx++] * normalSize - meanv[c1] * meanv[c2];
		}
	}
}

void CalcPatchCovarMatrix::naive(const vector<Mat>& src, Mat& cov, const int skip)
{
	vector<Mat> sub(color_channels);
	vector<double> meanv(color_channels);
	vector<double> var(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		double aa, vv;
		cp::average_variance(src[c], aa, vv);
		subtract(src[c], aa, sub[c]);
		meanv[c] = 0.0;
		var[c] = vv;
	}

	const int DD = D * D;
	AutoBuffer<int> scan(DD);
	getScanorder(scan, src[0].cols, 1);

	AutoBuffer<double> sum(dim * dim);
	for (int i = 0; i < dim * dim; i++) sum[i] = 0.0;

	AutoBuffer<float> patch(dim);

	int count = 0;
	for (int j = patch_rad; j < src[0].rows - patch_rad; j += skip)
	{
		for (int i = patch_rad; i < src[0].rows - patch_rad; i += skip)
		{
			AutoBuffer<float*> sptr(color_channels);
			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] = sub[c].ptr<float>(j, i);
			}

			for (int k = 0, idx = 0; k < DD; k++)
			{
				for (int c = 0; c < color_channels; c++)
				{
					patch[idx++] = sptr[c][scan[k]];
				}
			}
			for (int k = 0, idx = 0; k < dim; k++)
			{
				for (int l = k + 1; l < dim; l++)
				{
					sum[idx++] += patch[k] * patch[l];
				}
			}
			count++;
		}
	}

	setCoVar(meanv, var, cov, sum, 1.0 / count);
}

void CalcPatchCovarMatrix::naive(const vector<Mat>& src, Mat& cov, Mat& mask)
{
	vector<Mat> sub(color_channels);
	vector<double> meanv(color_channels);
	vector<double> var(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		double aa, vv;
		cp::average_variance(src[c], aa, vv);
		subtract(src[c], aa, sub[c]);
		meanv[c] = 0.0;
		var[c] = vv;
	}

	const int DD = D * D;
	AutoBuffer<int> scan(DD);
	getScanorder(scan, src[0].cols, 1);

	AutoBuffer<double> sum(dim * dim);
	for (int i = 0; i < dim * dim; i++) sum[i] = 0.0;

	AutoBuffer<float> patch(dim);

	int count = 0;
	for (int j = patch_rad; j < src[0].rows - patch_rad; j++)
	{
		for (int i = patch_rad; i < src[0].rows - patch_rad; i++)
		{
			if (mask.at<uchar>(j, i) == 0)continue;

			AutoBuffer<float*> sptr(color_channels);
			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] = sub[c].ptr<float>(j, i);
			}

			for (int k = 0, idx = 0; k < DD; k++)
			{
				for (int c = 0; c < color_channels; c++)
				{
					patch[idx++] = sptr[c][scan[k]];
				}
			}
			for (int k = 0, idx = 0; k < dim; k++)
			{
				for (int l = k + 1; l < dim; l++)
				{
					sum[idx++] += patch[k] * patch[l];
				}
			}
			count++;
		}
	}

	setCoVar(meanv, var, cov, sum, 1.0 / count);
}

inline int getIndex(int j, int i, int step, int patch_rad)
{
	return step * (j - patch_rad) + (i - patch_rad);
}

inline int getComputeCovSize(const int dim)
{
	int ret = 0;
	for (int k = 0; k < dim; k++)
	{
		for (int l = k + 1; l < dim; l++)
		{
			ret++;
		}
	}
	return ret;
}

void sub_const(InputArray src_, const float val, Mat& dest)
{
	Mat src = src_.getMat();
	CV_Assert(src.type() == CV_32FC1);
	dest.create(src.size(), CV_32F);
	const int simdsize = get_simd_floor(src.size().area(), 32);
	const int S = simdsize / 32;
	float* sptr = src.ptr<float>();
	float* dptr = dest.ptr<float>();
	const __m256 msub = _mm256_set1_ps(val);
	for (int i = 0; i < S; i++)
	{
		_mm256_storeu_ps(dptr, _mm256_sub_ps(_mm256_loadu_ps(sptr), msub));
		_mm256_storeu_ps(dptr + 8, _mm256_sub_ps(_mm256_loadu_ps(sptr + 8), msub));
		_mm256_storeu_ps(dptr + 16, _mm256_sub_ps(_mm256_loadu_ps(sptr + 16), msub));
		_mm256_storeu_ps(dptr + 24, _mm256_sub_ps(_mm256_loadu_ps(sptr + 24), msub));
		sptr += 32;
		dptr += 32;
	}
	sptr = src.ptr<float>();
	dptr = dest.ptr<float>();
	for (int i = simdsize; i < src.size().area(); i++)
	{
		dptr[i] = sptr[i] - val;
	}
}

#pragma region main process
template<int color_channels, int dim>
void CalcPatchCovarMatrix::simd_32F(const vector<Mat>& src_, Mat& cov, const CovMethod method, const float constant_sub)
{
	//cout << "simd_32F" << endl;
	const int simd_step = 8;
	data.resize(color_channels);

	vector<double> meanForCov(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CovMethod::MEAN_SUB)
		{
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
			meanForCov[c] = 0.0;
		}
		else if (method == CovMethod::CONSTANT_SUB)
		{
			//subtract(src_[c], constant_sub, data[c]);
			sub_const(src_[c], constant_sub, data[c]);
			meanForCov[c] = 0.0;
		}
		else if (method == CovMethod::NO_SUB)
		{
			if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			else meanForCov[0] = 0.0;
			*&data[c] = *&src_[c];
		}
		else
		{
			cout << "No support method (simd_32F)" << endl;
		}
	}

	const int DD = dim / color_channels;//d*d
	const int center = (DD / 2) * color_channels;

	int* scan = (int*)_mm_malloc(sizeof(int) * DD, AVX_ALIGN);
	getScanorder(scan, data[0].cols, 1);

	const int simd_end_x = get_simd_floor(data[0].cols - 2 * patch_rad, simd_step) + patch_rad;
	const double normalSize = 1.0 / ((data[0].rows - 2 * patch_rad) * (simd_end_x - patch_rad));
	const int covsize = getComputeCovSize(dim);
	//__m256* msrc = (__m256*)_mm_malloc(sizeof(__m256) * dim, AVX_ALIGN);
	//AutoBuffer<__m256> msrc(dim);	

	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * (covsize + color_channels), AVX_ALIGN);
	for (int i = 0; i < covsize + color_channels; i++)mbuff[i] = _mm256_setzero_ps();
	__m256* msum = &mbuff[0];
	__m256* mvar = &mbuff[covsize];

	for (int j = patch_rad; j < data[0].rows - patch_rad; j++)
	{
		AutoBuffer<__m256> msrc(dim);
		AutoBuffer<const float*> sptr(color_channels);
		for (int c = 0; c < color_channels; c++)
		{
			sptr[c] = data[c].ptr<float>(j, patch_rad);
		}

		for (int i = patch_rad; i < simd_end_x; i += simd_step)
		{
			__m256* msrcptr = &msrc[0];
			int* scptr = &scan[0];
			for (int k = 0; k < DD; k++)
			{
				for (int c = 0; c < color_channels; c++)
				{
					*msrcptr++ = _mm256_loadu_ps(sptr[c] + *scptr);
				}
				scptr++;
			}

			for (int k = 0, idx = 0; k < dim; k++)
			{
				for (int l = k + 1; l < dim; l++)
				{
					msum[idx] = _mm256_fmadd_ps(msrc[k], msrc[l], mbuff[idx]);
					idx++;
				}
			}

			for (int c = 0; c < color_channels; c++)
			{
				mvar[c] = _mm256_fmadd_ps(msrc[c + center], msrc[c + center], mvar[c]);
			}

			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] += simd_step;
			}
		}
	}

	//reduction
	vector<double> var(color_channels);
	AutoBuffer<double> sum(covsize);
	for (int i = 0; i < covsize; i++)sum[i] = 0.0;
	for (int i = 0; i < color_channels; i++) var[i] = 0.0;

	for (int idx = 0; idx < covsize; idx++)
	{
		//sum[idx] += _mm256_reduceadd_ps(msum[idx]);
		sum[idx] += _mm256_reduceadd_pspd(msum[idx]);
	}

	for (int c = 0; c < color_channels; c++)
	{
		//var[c] += _mm256_reduceadd_ps(mvar[c]);
		var[c] += _mm256_reduceadd_pspd(mvar[c]);
	}

	setCoVar(meanForCov, var, cov, sum, normalSize);
	_mm_free(scan);
	//_mm_free(msrc);
	_mm_free(mbuff);
}

void CalcPatchCovarMatrix::simd_32FCn(const vector<Mat>& src_, Mat& cov, const CovMethod method, const float constant_sub)
{
	//cout << "simd_32F" << endl;
	const int simd_step = 8;
	data.resize(color_channels);

	vector<double> meanForCov(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CovMethod::MEAN_SUB)
		{
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
			meanForCov[c] = 0.0;
		}
		else
		{
			cout << "No support method (simd_32FCn)" << endl;
		}
	}

	const int DD = dim / color_channels;//d*d
	const int center = (DD / 2) * color_channels;

	int* scan = (int*)_mm_malloc(sizeof(int) * DD, AVX_ALIGN);
	getScanorder(scan, data[0].cols, 1);

	const int simd_end_x = get_simd_floor(data[0].cols - 2 * patch_rad, simd_step) + patch_rad;
	const double normalSize = 1.0 / ((data[0].rows - 2 * patch_rad) * (simd_end_x - patch_rad));
	const int covsize = getComputeCovSize(dim);
	//__m256* msrc = (__m256*)_mm_malloc(sizeof(__m256) * dim, AVX_ALIGN);
	//AutoBuffer<__m256> msrc(dim);	

	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * (covsize + color_channels), AVX_ALIGN);
	for (int i = 0; i < covsize + color_channels; i++)mbuff[i] = _mm256_setzero_ps();
	__m256* msum = &mbuff[0];
	__m256* mvar = &mbuff[covsize];

	for (int j = patch_rad; j < data[0].rows - patch_rad; j++)
	{
		AutoBuffer<__m256> msrc(dim);
		AutoBuffer<const float*> sptr(color_channels);
		for (int c = 0; c < color_channels; c++)
		{
			sptr[c] = data[c].ptr<float>(j, patch_rad);
		}

		for (int i = patch_rad; i < simd_end_x; i += simd_step)
		{
			__m256* msrcptr = &msrc[0];
			int* scptr = &scan[0];
			for (int k = 0; k < DD; k++)
			{
				for (int c = 0; c < color_channels; c++)
				{
					*msrcptr++ = _mm256_loadu_ps(sptr[c] + *scptr);
				}
				scptr++;
			}

			for (int k = 0, idx = 0; k < dim; k++)
			{
				for (int l = k + 1; l < dim; l++)
				{
					msum[idx] = _mm256_fmadd_ps(msrc[k], msrc[l], mbuff[idx]);
					idx++;
				}
			}

			for (int c = 0; c < color_channels; c++)
			{
				mvar[c] = _mm256_fmadd_ps(msrc[c + center], msrc[c + center], mvar[c]);
			}

			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] += simd_step;
			}
		}
	}

	//reduction
	vector<double> var(color_channels);
	AutoBuffer<double> sum(covsize);
	for (int i = 0; i < covsize; i++)sum[i] = 0.0;
	for (int i = 0; i < color_channels; i++) var[i] = 0.0;

	for (int idx = 0; idx < covsize; idx++)
	{
		//sum[idx] += _mm256_reduceadd_ps(msum[idx]);
		sum[idx] += _mm256_reduceadd_pspd(msum[idx]);
	}

	for (int c = 0; c < color_channels; c++)
	{
		//var[c] += _mm256_reduceadd_ps(mvar[c]);
		var[c] += _mm256_reduceadd_pspd(mvar[c]);
	}

	setCoVar(meanForCov, var, cov, sum, normalSize);
	_mm_free(scan);
	//_mm_free(msrc);
	_mm_free(mbuff);
}

template<int color_channels, int dim>
void CalcPatchCovarMatrix::simdOMP_32F(const vector<Mat>& src_, Mat& cov, const CovMethod method, const float constant_sub)
{
	//cout << "simdOMP_32F" << endl;
	const int simd_step = 8;
	data.resize(color_channels);

	vector<double> meanForCov(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CovMethod::MEAN_SUB)
		{
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
			meanForCov[c] = 0.0;
		}
		else if (method == CovMethod::CONSTANT_SUB)
		{
			//subtract(src_[c], constant_sub, data[c]);
			sub_const(src_[c], constant_sub, data[c]);
			meanForCov[c] = 0.0;
		}
		else if (method == CovMethod::NO_SUB)
		{
			if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			else meanForCov[0] = 0.0;
			data[c] = src_[c];
		}
		else
		{
			cout << "No support method (simdOMP_32F)" << endl;
		}
	}

	const int DD = dim / color_channels;
	const int center = (DD / 2) * color_channels;

	int* scan = (int*)_mm_malloc(sizeof(int) * DD, AVX_ALIGN);
	getScanorder(scan, data[0].cols, 1);

	const int thread_max = omp_get_max_threads();
	const int simd_end_x = get_simd_floor(data[0].cols - 2 * patch_rad, simd_step) + patch_rad;
	const double normalSize = 1.0 / ((data[0].rows - 2 * patch_rad) * (simd_end_x - patch_rad));
	const int covsize = getComputeCovSize(dim);

	//__m256* msrc = (__m256*)_mm_malloc(sizeof(__m256) * dim * thread_max, AVX_ALIGN);

	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * (covsize + color_channels) * thread_max, AVX_ALIGN);
	for (int i = 0; i < (covsize + color_channels) * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

	const int yend = data[0].rows - 2 * patch_rad;
#pragma omp parallel for SCHEDULE
	for (int y = 0; y < yend; y++)
	{
		const int j = y + patch_rad;
		const int tindex = omp_get_thread_num();

		__m256* msum_local = &mbuff[(covsize + color_channels) * tindex];
		__m256* mvar_local = &mbuff[(covsize + color_channels) * tindex + covsize];

		AutoBuffer<__m256> msrc_local(dim);
		//__m256* msrc_local = &msrc[dim * tindex];
		AutoBuffer<const float*> sptr(color_channels);
		for (int c = 0; c < color_channels; c++)
		{
			sptr[c] = data[c].ptr<float>(j, patch_rad);
		}

		for (int i = patch_rad; i < simd_end_x; i += simd_step)
		{
			__m256* msrcptr = &msrc_local[0];
			int* scptr = &scan[0];
			for (int k = 0; k < DD; k++)
			{
				for (int c = 0; c < color_channels; c++)
				{
					*msrcptr++ = _mm256_loadu_ps(sptr[c] + *scptr);
				}
				scptr++;
			}

			for (int c = 0; c < color_channels; c++)
			{
				mvar_local[c] = _mm256_fmadd_ps(msrc_local[c + center], msrc_local[c + center], mvar_local[c]);
			}

			for (int k = 0, idx = 0; k < dim; k++)
			{
				const __m256 mk = msrc_local[k];
				for (int l = k + 1; l < dim; l++)
				{
					msum_local[idx] = _mm256_fmadd_ps(mk, msrc_local[l], msum_local[idx]);
					idx++;
				}
			}

			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] += simd_step;
			}
		}
	}

	//reduction
	vector<double> var(color_channels);
	AutoBuffer<double> sum(covsize);
	for (int i = 0; i < covsize; i++) sum[i] = 0.0;
	for (int i = 0; i < color_channels; i++) var[i] = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		int sstep = (covsize + color_channels) * t;
		__m256* mvar = &mbuff[(covsize + color_channels) * t + covsize];
		for (int k = 0, idx = 0; k < dim; k++)
		{
			for (int l = k + 1; l < dim; l++)
			{
				//sum[idx] += _mm256_reduceadd_ps(mbuff[idx+sstep]);
				sum[idx] += _mm256_reduceadd_pspd(mbuff[idx + sstep]);
				idx++;
			}
		}

		for (int c = 0; c < color_channels; c++)
		{
			//var[c] += _mm256_reduceadd_ps(mvar[c]);
			var[c] += _mm256_reduceadd_pspd(mvar[c]);
		}
	}

	setCoVar(meanForCov, var, cov, sum, normalSize);
	_mm_free(scan);
	//_mm_free(msrc);
	_mm_free(mbuff);
}

void CalcPatchCovarMatrix::simdOMP_32FCn(const vector<Mat>& src_, Mat& cov, const CovMethod method, const float constant_sub)
{
	//cout << "simdOMP_32F" << endl;
	const int simd_step = 8;
	data.resize(color_channels);

	vector<double> meanForCov(color_channels);
	for (int c = 0; c < color_channels; c++)
	{
		if (method == CovMethod::MEAN_SUB)
		{
			double ave = cp::average(src_[c]);
			//subtract(src_[c], float(ave), data[c]);
			sub_const(src_[c], float(ave), data[c]);
			meanForCov[c] = 0.0;
		}
		else if (method == CovMethod::CONSTANT_SUB)
		{
			//subtract(src_[c], constant_sub, data[c]);
			sub_const(src_[c], constant_sub, data[c]);
			meanForCov[c] = 0.0;
		}
		else if (method == CovMethod::NO_SUB)
		{
			if (color_channels != 1) meanForCov[c] = cp::average(src_[c]);
			else meanForCov[0] = 0.0;
			data[c] = src_[c];
		}
		else
		{
			cout << "No support method (simdOMP_32FCn)" << endl;
		}
	}

	const int DD = dim / color_channels;
	const int center = (DD / 2) * color_channels;

	int* scan = (int*)_mm_malloc(sizeof(int) * DD, AVX_ALIGN);
	getScanorder(scan, data[0].cols, 1);

	const int thread_max = omp_get_max_threads();
	const int simd_end_x = get_simd_floor(data[0].cols - 2 * patch_rad, simd_step) + patch_rad;
	const double normalSize = 1.0 / ((data[0].rows - 2 * patch_rad) * (simd_end_x - patch_rad));
	const int covsize = getComputeCovSize(dim);

	//__m256* msrc = (__m256*)_mm_malloc(sizeof(__m256) * dim * thread_max, AVX_ALIGN);

	__m256* mbuff = (__m256*)_mm_malloc(sizeof(__m256) * (covsize + color_channels) * thread_max, AVX_ALIGN);
	for (int i = 0; i < (covsize + color_channels) * thread_max; i++) mbuff[i] = _mm256_setzero_ps();

	const int yend = data[0].rows - 2 * patch_rad;
#pragma omp parallel for SCHEDULE
	for (int y = 0; y < yend; y++)
	{
		const int j = y + patch_rad;
		const int tindex = omp_get_thread_num();

		__m256* msum_local = &mbuff[(covsize + color_channels) * tindex];
		__m256* mvar_local = &mbuff[(covsize + color_channels) * tindex + covsize];

		AutoBuffer<__m256> msrc_local(dim);
		//__m256* msrc_local = &msrc[dim * tindex];
		AutoBuffer<const float*> sptr(color_channels);
		for (int c = 0; c < color_channels; c++)
		{
			sptr[c] = data[c].ptr<float>(j, patch_rad);
		}

		for (int i = patch_rad; i < simd_end_x; i += simd_step)
		{
			__m256* msrcptr = &msrc_local[0];
			int* scptr = &scan[0];
			for (int k = 0; k < DD; k++)
			{
				for (int c = 0; c < color_channels; c++)
				{
					*msrcptr++ = _mm256_loadu_ps(sptr[c] + *scptr);
				}
				scptr++;
			}

			for (int c = 0; c < color_channels; c++)
			{
				mvar_local[c] = _mm256_fmadd_ps(msrc_local[c + center], msrc_local[c + center], mvar_local[c]);
			}

			for (int k = 0, idx = 0; k < dim; k++)
			{
				const __m256 mk = msrc_local[k];
				for (int l = k + 1; l < dim; l++)
				{
					msum_local[idx] = _mm256_fmadd_ps(mk, msrc_local[l], msum_local[idx]);
					idx++;
				}
			}

			for (int c = 0; c < color_channels; c++)
			{
				sptr[c] += simd_step;
			}
		}
	}

	//reduction
	vector<double> var(color_channels);
	AutoBuffer<double> sum(covsize);
	for (int i = 0; i < covsize; i++) sum[i] = 0.0;
	for (int i = 0; i < color_channels; i++) var[i] = 0.0;

	for (int t = 0; t < thread_max; t++)
	{
		int sstep = (covsize + color_channels) * t;
		__m256* mvar = &mbuff[(covsize + color_channels) * t + covsize];
		for (int k = 0, idx = 0; k < dim; k++)
		{
			for (int l = k + 1; l < dim; l++)
			{
				//sum[idx] += _mm256_reduceadd_ps(mbuff[idx+sstep]);
				sum[idx] += _mm256_reduceadd_pspd(mbuff[idx + sstep]);
				idx++;
			}
		}

		for (int c = 0; c < color_channels; c++)
		{
			//var[c] += _mm256_reduceadd_ps(mvar[c]);
			var[c] += _mm256_reduceadd_pspd(mvar[c]);
		}
	}

	setCoVar(meanForCov, var, cov, sum, normalSize);
	_mm_free(scan);
	//_mm_free(msrc);
	_mm_free(mbuff);
}

#pragma endregion
#pragma endregion

#pragma region normal
void projectNeighborhoodEigenVec(const Mat& src, const Mat& evec, Mat& dest, const int r, const int channels, const int border, const bool isParallel)
{
	//cout << evec.cols << endl;
	//print_matinfo(evec);
	dest.create(src.size(), CV_MAKE_TYPE(CV_32F, evec.rows));

	const int D = 2 * r + 1;
	const int dim = D * D * src.channels();//D * D * src.channels()
	bool isSIMD = true;
	AutoBuffer<const float*> eptr(evec.rows);
	for (int m = 0; m < evec.rows; m++)
	{
		eptr[m] = evec.ptr<float>(m);
	}

	if (isParallel)
	{
		if (src.channels() == 1)
		{
			Mat srcborder;
			copyMakeBorder(src, srcborder, r, r, r, r, border);
			if (isSIMD)
			{
#pragma omp parallel for
				for (int j = 0; j < src.rows; j++)
				{
					AutoBuffer<__m256> patch(dim);
					for (int i = 0; i < src.cols; i += 8)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								patch[idx++] = _mm256_loadu_ps(srcborder.ptr<float>(j + l, i + m));
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							__m256 mval = _mm256_setzero_ps();
							for (int l = 0; l < dim; l++)
							{
								mval = _mm256_fmadd_ps(patch[l], _mm256_set1_ps(eptr[m][l]), mval);
							}
							for (int mm = 0; mm < 8; mm++)
							{
								//d[evec.rows * mm + m] = mval.m256_f32[mm];
								d[evec.rows * mm + m] = ((float*)&mval)[mm];
							}
						}
					}
				}
			}
			else
			{
#pragma omp parallel for
				for (int j = 0; j < src.rows; j++)
				{
					AutoBuffer<float> patch(dim);
					for (int i = 0; i < src.cols; i++)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								patch[idx++] = srcborder.at<float>(j + l, i + m);
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							float val = 0.f;
							for (int l = 0; l < dim; l++)
							{
								val += patch[l] * eptr[m][l];
							}
							d[m] = val;
						}
					}
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> srcborder;
			cp::splitCopyMakeBorder(src, srcborder, r, r, r, r, border);
			bool isSIMD = true;

			if (isSIMD)
			{
#pragma omp parallel for
				for (int j = 0; j < src.rows; j++)
				{
					AutoBuffer<__m256> patch(dim);
					for (int i = 0; i < src.cols; i += 8)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								for (int c = 0; c < 3; c++)
								{
									patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + l, i + m));
								}
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							__m256 mval = _mm256_setzero_ps();
							for (int l = 0; l < dim; l++)
							{
								mval = _mm256_fmadd_ps(patch[l], _mm256_set1_ps(eptr[m][l]), mval);
							}
							for (int mm = 0; mm < 8; mm++)
							{
								//d[evec.rows * mm + m] = mval.m256_f32[mm];
								d[evec.rows * mm + m] = ((float*)&mval)[mm];
							}
						}
					}
				}
			}
			else
			{
#pragma omp parallel for
				for (int j = 0; j < src.rows; j++)
				{
					AutoBuffer<float> patch(dim);
					for (int i = 0; i < src.cols; i++)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								for (int c = 0; c < 3; c++)
								{
									patch[idx++] = srcborder[c].at<float>(j + l, i + m);
								}
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							float val = 0.f;
							for (int l = 0; l < dim; l++)
							{
								val += patch[l] * eptr[m][l];
							}
							d[m] = val;
						}
					}
				}
			}
		}
	}
	else
	{
		if (src.channels() == 1)
		{
			Mat srcborder;
			copyMakeBorder(src, srcborder, r, r, r, r, border);
			if (isSIMD)
			{
				AutoBuffer<__m256> patch(dim);
				for (int j = 0; j < src.rows; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								patch[idx++] = _mm256_loadu_ps(srcborder.ptr<float>(j + l, i + m));
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							__m256 mval = _mm256_setzero_ps();
							for (int l = 0; l < dim; l++)
							{
								mval = _mm256_fmadd_ps(patch[l], _mm256_set1_ps(eptr[m][l]), mval);
							}
							for (int mm = 0; mm < 8; mm++)
							{
								//d[evec.rows * mm + m] = mval.m256_f32[mm];
								d[evec.rows * mm + m] = ((float*)&mval)[mm];
							}
						}
					}
				}
			}
			else
			{
				AutoBuffer<float> patch(dim);
				for (int j = 0; j < src.rows; j++)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								patch[idx++] = srcborder.at<float>(j + l, i + m);
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							float val = 0.f;
							for (int l = 0; l < dim; l++)
							{
								val += patch[l] * eptr[m][l];
							}
							d[m] = val;
						}
					}
				}
			}
		}
		else if (src.channels() == 3)
		{
			vector<Mat> srcborder;
			cp::splitCopyMakeBorder(src, srcborder, r, r, r, r, border);
			bool isSIMD = true;
			if (isSIMD)
			{
				AutoBuffer<__m256> patch(dim);
				for (int j = 0; j < src.rows; j++)
				{
					for (int i = 0; i < src.cols; i += 8)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								for (int c = 0; c < 3; c++)
								{
									patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + l, i + m));
								}
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							__m256 mval = _mm256_setzero_ps();
							for (int l = 0; l < dim; l++)
							{
								mval = _mm256_fmadd_ps(patch[l], _mm256_set1_ps(eptr[m][l]), mval);
							}
							for (int mm = 0; mm < 8; mm++)
							{
								//d[evec.rows * mm + m] = mval.m256_f32[mm];
								d[evec.rows * mm + m] = ((float*)&mval)[mm];
							}
						}
					}
				}
			}
			else
			{
				AutoBuffer<float> patch(dim);
				for (int j = 0; j < src.rows; j++)
				{
					for (int i = 0; i < src.cols; i++)
					{
						float* d = dest.ptr<float>(j, i);

						int idx = 0;
						for (int l = 0; l < D; l++)
						{
							for (int m = 0; m < D; m++)
							{
								for (int c = 0; c < 3; c++)
								{
									patch[idx++] = srcborder[c].at<float>(j + l, i + m);
								}
							}
						}

						for (int m = 0; m < evec.rows; m++)
						{
							float val = 0.f;
							for (int l = 0; l < dim; l++)
							{
								val += patch[l] * eptr[m][l];
							}
							d[m] = val;
						}
					}
				}
			}
		}
	}
}

void imshowNeighboorhoodEigenVectors(string wname, Mat& evec, const int channels)
{
	const int w = (int)sqrt(evec.cols / channels);
	vector<Mat> v;
	for (int i = 0; i < evec.rows; i++)
	{
		Mat a = evec.row(i).reshape(1, w).clone();
		//if (i == 0)print_mat(a);
		Mat b;
		copyMakeBorder(a, b, 0, 1, 0, 1, BORDER_CONSTANT, Scalar::all(1));
		v.push_back(b);
	}
	Mat dest;
	cp::concat(v, dest, w, w);
	copyMakeBorder(dest, dest, 1, 0, 1, 0, BORDER_CONSTANT, Scalar::all(1));
	resize(dest, dest, Size(1024, 1024), 1, 1, INTER_NEAREST);

	cp::imshowScale(wname, dest, 255, 128);
}

void patchPCA(const Mat& src, Mat& dst, const int neighborhood_r, const int dest_channels, const int border, const int method, const bool isParallel)
{
	CV_Assert(src.depth() == CV_8U || src.depth() == CV_32F);

	const int width = src.cols;
	const int height = src.rows;
	const int ch = src.channels();
	const int d = 2 * neighborhood_r + 1;
	const int patch_area = d * d;
	const int dim = ch * patch_area;
	const Size imsize = src.size();
	const int imarea = imsize.area();
	const int num_points = cvRound((float)imarea * 0.1);

	if (double(src.size().area() * 255 * 255) > FLT_MAX)
	{
		cout << "overflow in float" << endl;
	}

	if (method < (int)NeighborhoodPCA::OPENCV_PCA)
	{
		Mat cov, eval, evec;
		CalcPatchCovarMatrix pcov; pcov.computeCov(src, neighborhood_r, cov, (NeighborhoodPCA)method, 1, isParallel);

		eigen(cov, eval, evec);
		//if (isParallel) imshowNeighboorhoodEigenVectors("evec", evec, 1);

		Mat transmat;
		evec(Rect(0, 0, evec.cols, dest_channels)).convertTo(transmat, CV_32F);
		projectNeighborhoodEigenVec(src, transmat, dst, neighborhood_r, 1, border, isParallel);
	}
	else
	{
		Mat highDimGuide(src.size(), CV_MAKE_TYPE(CV_32F, dim));
		{
			//cp::Timer t("cvt HDI");
			cp::IM2COL(src, highDimGuide, neighborhood_r, border);
		}

		if (method == (int)NeighborhoodPCA::OPENCV_PCA)
		{
			PCA pca(highDimGuide.reshape(1, imsize.area()), cv::Mat(), cv::PCA::DATA_AS_ROW, dest_channels);
			dst = pca.project(highDimGuide.reshape(1, imsize.area())).reshape(dest_channels, src.rows);
		}
		else if (method == (int)NeighborhoodPCA::OPENCV_COV)
		{
			Mat x = highDimGuide.reshape(1, imsize.area());
			Mat cov, mean;
			cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
			//print_mat(cov);
			Mat eval, evec;
			eigen(cov, eval, evec);

			Mat transmat;
			evec(Rect(0, 0, evec.cols, dest_channels)).convertTo(transmat, CV_32F);
			//transmat = Mat::eye(transmat.size(), CV_32F);
			cv::transform(highDimGuide, dst, transmat);
		}
	}
}
#pragma endregion

#pragma region vector in/out
void projectNeighborhoodEigenVecCn(const vector<Mat>& src, const Mat& evec, vector<Mat>& dest, const int r, const int border)
{
	//cout << evec.cols << endl;
	//print_matinfo(evec);
	const int src_channels = (int)src.size();
	const int dest_channels = evec.rows;

	dest.resize(dest_channels);
	for (int c = 0; c < dest_channels; c++)
	{
		dest[c].create(src[0].size(), CV_32F);
	}

	const int width = src[0].cols;
	const int height = src[0].rows;

	const int D = 2 * r + 1;
	const int DD = D * D;
	const int dim = DD * src_channels;//D * D * src_channels()

	AutoBuffer<const float*> eptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eptr[m] = evec.ptr<float>(m);
	}

	vector<Mat> srcborder(src_channels);
	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(src[c], srcborder[c], r, r, r, r, border);
	}

	bool isSIMD = true;
	if (isSIMD)
	{
		AutoBuffer<__m256> patch(dim);
		AutoBuffer<float*> dptr(dest_channels);

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i += 8)
			{
				for (int c = 0; c < dest_channels; c++)
				{
					dptr[c] = dest[c].ptr<float>(j, i);
				}

				for (int l = 0, idx = 0; l < D; l++)
				{
					for (int m = 0; m < D; m++)
					{
						for (int c = 0; c < src_channels; c++)
						{
							patch[idx++] = _mm256_loadu_ps(srcborder[c].ptr<float>(j + l, i + m));
						}
					}
				}

				for (int c = 0; c < dest_channels; c++)
				{
					__m256 mval = _mm256_setzero_ps();
					for (int d = 0; d < dim; d++)
					{
						mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
					}
					_mm256_storeu_ps(dptr[c], mval);
				}
			}
		}
	}
	else
	{
		AutoBuffer<float> patch(dim);
		AutoBuffer<float*> dptr(dest_channels);

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				for (int c = 0; c < dest_channels; c++)
				{
					dptr[c] = dest[c].ptr<float>(j, i);
				}

				for (int l = 0, idx = 0; l < D; l++)
				{
					for (int m = 0; m < D; m++)
					{
						for (int c = 0; c < src_channels; c++)
						{
							patch[idx++] = srcborder[c].at<float>(j + l, i + m);
						}
					}
				}

				for (int c = 0; c < dest_channels; c++)
				{
					float val = 0.f;
					for (int d = 0; d < dim; d++)
					{
						val += patch[d] * eptr[c][d];
					}
					*dptr[c] = val;
				}
			}
		}
	}
}

template<int dest_channels>
void projectNeighborhoodEigenVec(const vector<Mat>& src, const Mat& evec, vector<Mat>& dest, const int r, const int border)
{
	//cout << evec.cols << endl;
	//print_matinfo(evec);
	const int src_channels = (int)src.size();

	if (dest.size() != dest_channels) dest.resize(dest_channels);
	for (int c = 0; c < dest_channels; c++)
	{
		dest[c].create(src[0].size(), CV_32F);
	}


	const int D = 2 * r + 1;
	const int DD = D * D;
	const int dim = DD * src_channels;//D * D * src_channels()

	AutoBuffer<const float*> eptr(dest_channels);
	for (int m = 0; m < dest_channels; m++)
	{
		eptr[m] = evec.ptr<float>(m);
	}

	vector<Mat> srcborder(src_channels);
	for (int c = 0; c < src_channels; c++)
	{
		copyMakeBorder(src[c], srcborder[c], r, r, r, r, border);
	}

	bool isSIMD = true;
	if (isSIMD)
	{
		const int unroll = 8;
		const int width = get_simd_floor(src[0].cols, unroll);
		const int height = src[0].rows;

		AutoBuffer<float*> sptr(src_channels);

		const int step = srcborder[0].cols;
		for (int c = 0; c < src_channels; c++)
		{
			sptr[c] = srcborder[c].ptr<float>();
		}

		if (unroll == 8)
		{
			AutoBuffer<__m256> patch(dim);
			AutoBuffer<float*> dptr(dest_channels);
			for (int j = 0; j < height; j++)
			{
				for (int i = 0; i < width; i += 8)
				{
					for (int c = 0; c < dest_channels; c++)
					{
						dptr[c] = dest[c].ptr<float>(j, i);
					}

					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + i;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								patch[idx++] = _mm256_loadu_ps(sptr[c] + index);
							}
						}
					}

					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval = _mm256_setzero_ps();
						for (int d = 0; d < dim; d++)
						{
							mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
						}
						_mm256_storeu_ps(dptr[c], mval);
					}
				}
			}
		}
		else if (unroll == 16)
		{
			AutoBuffer<__m256> me(dim * dest_channels);
			for (int c = 0, idx = 0; c < dest_channels; c++)
			{
				for (int d = 0; d < dim; d++)
				{
					me[idx++] = _mm256_set1_ps(eptr[c][d]);
				}
			}
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < height; j++)
			{
				AutoBuffer<float*> dptr(dest_channels);
				AutoBuffer<__m256> patch(dim);
				AutoBuffer<__m256> patch2(dim);
				for (int c = 0; c < dest_channels; c++)
				{
					dptr[c] = dest[c].ptr<float>(j);
				}
				for (int i = 0; i < width; i += 16)
				{
					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + i;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								patch[idx] = _mm256_loadu_ps(sptr[c] + index);
								patch2[idx++] = _mm256_loadu_ps(sptr[c] + index + 8);
							}
						}
					}

					int idx = 0;
					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval = _mm256_setzero_ps();
						__m256 mval2 = _mm256_setzero_ps();
						for (int d = 0; d < dim; d++)
						{
							//mval = _mm256_fmadd_ps(patch[d], _mm256_set1_ps(eptr[c][d]), mval);
							//mval2 = _mm256_fmadd_ps(patch2[d], _mm256_set1_ps(eptr[c][d]), mval2);
							mval = _mm256_fmadd_ps(patch[d], me[idx], mval);
							mval2 = _mm256_fmadd_ps(patch2[d], me[idx], mval2);
							idx++;
						}
						_mm256_storeu_ps(dptr[c] + i, mval);
						_mm256_storeu_ps(dptr[c] + i + 8, mval2);
					}
				}
			}
		}
		else if (unroll == 32)
		{
			AutoBuffer<__m256> me(dim * dest_channels);
			for (int c = 0, idx = 0; c < dest_channels; c++)
			{
				for (int d = 0; d < dim; d++)
				{
					me[idx++] = _mm256_set1_ps(eptr[c][d]);
				}
			}
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < height; j++)
			{
				AutoBuffer<float*> dptr(dest_channels);
				AutoBuffer<__m256> patch1(dim);
				AutoBuffer<__m256> patch2(dim);
				AutoBuffer<__m256> patch3(dim);
				AutoBuffer<__m256> patch4(dim);
				for (int c = 0; c < dest_channels; c++)
				{
					dptr[c] = dest[c].ptr<float>(j);
				}
				for (int i = 0; i < width; i += 32)
				{
					for (int l = 0, idx = 0; l < D; l++)
					{
						const int vstep = step * (j + l) + i;
						for (int m = 0; m < D; m++)
						{
							const int index = vstep + m;
							for (int c = 0; c < src_channels; c++)
							{
								float* s = sptr[c] + index;
								patch1[idx] = _mm256_loadu_ps(s);
								patch2[idx] = _mm256_loadu_ps(s + 8);
								patch3[idx] = _mm256_loadu_ps(s + 16);
								patch4[idx] = _mm256_loadu_ps(s + 24);
								idx++;
							}
						}
					}

					int idx = 0;
					for (int c = 0; c < dest_channels; c++)
					{
						__m256 mval1 = _mm256_mul_ps(patch1[0], me[idx]);
						__m256 mval2 = _mm256_mul_ps(patch2[0], me[idx]);
						__m256 mval3 = _mm256_mul_ps(patch3[0], me[idx]);
						__m256 mval4 = _mm256_mul_ps(patch4[0], me[idx]);
						__m256 m1 = patch1[1];
						__m256 m2 = patch2[1];
						__m256 m3 = patch3[1];
						__m256 m4 = patch4[1];
						__m256 e = me[1];
						for (int d = 1; d < dim - 1; d++)
						{
							mval1 = _mm256_fmadd_ps(m1, e, mval1);
							mval2 = _mm256_fmadd_ps(m2, e, mval2);
							mval3 = _mm256_fmadd_ps(m3, e, mval3);
							mval4 = _mm256_fmadd_ps(m4, e, mval4);
							idx++;
							m1 = patch1[d + 1];
							m2 = patch2[d + 1];
							m3 = patch3[d + 1];
							m4 = patch4[d + 1];
							e = me[idx];
						}
						{
							mval1 = _mm256_fmadd_ps(m1, me[idx], mval1);
							mval2 = _mm256_fmadd_ps(m2, me[idx], mval2);
							mval3 = _mm256_fmadd_ps(m3, me[idx], mval3);
							mval4 = _mm256_fmadd_ps(m4, me[idx], mval4);
						}
						_mm256_storeu_ps(dptr[c] + i, mval1);
						_mm256_storeu_ps(dptr[c] + i + 8, mval2);
						_mm256_storeu_ps(dptr[c] + i + 16, mval3);
						_mm256_storeu_ps(dptr[c] + i + 24, mval4);
					}
				}
			}
		}
	}
	else
	{
		const int width = src[0].cols;
		const int height = src[0].rows;
		AutoBuffer<float> patch(dim);
		AutoBuffer<float*> dptr(dest_channels);

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i++)
			{
				for (int c = 0; c < dest_channels; c++)
				{
					dptr[c] = dest[c].ptr<float>(j, i);
				}

				for (int l = 0, idx = 0; l < D; l++)
				{
					for (int m = 0; m < D; m++)
					{
						for (int c = 0; c < src_channels; c++)
						{
							patch[idx++] = srcborder[c].at<float>(j + l, i + m);
						}
					}
				}

				for (int c = 0; c < dest_channels; c++)
				{
					float val = 0.f;
					for (int d = 0; d < dim; d++)
					{
						val += patch[d] * eptr[c][d];
					}
					*dptr[c] = val;
				}
			}
		}
	}
}

void patchPCA(const vector<Mat>& src, vector<Mat>& dst, const int neighborhood_r, const int dest_channels, const int border, const int method, const bool isParallel)
{
	CV_Assert(src[0].depth() == CV_8U || src[0].depth() == CV_32F);

	const int width = src[0].cols;
	const int height = src[0].rows;
	const int ch = (int)src.size();
	const int d = 2 * neighborhood_r + 1;
	const int patch_area = d * d;
	const int dim = ch * patch_area;
	const Size imsize = src[0].size();
	const int imarea = imsize.area();
	const int num_points = cvRound((float)imarea * 0.1);

	if (double(src[0].size().area() * 255 * 255) > FLT_MAX)
	{
		cout << "overflow in float" << endl;
	}

	if (method < (int)NeighborhoodPCA::OPENCV_PCA)
	{
		Mat cov, eval, evec;
		CalcPatchCovarMatrix pcov; pcov.computeCov(src, neighborhood_r, cov, (NeighborhoodPCA)method, 1, isParallel);

		eigen(cov, eval, evec);

		Mat transmat;
		evec(Rect(0, 0, evec.cols, dest_channels)).convertTo(transmat, CV_32F);

		switch (dest_channels)
		{
		case 1: projectNeighborhoodEigenVec<1>(src, transmat, dst, neighborhood_r, border); break;
		case 2: projectNeighborhoodEigenVec<2>(src, transmat, dst, neighborhood_r, border); break;
		case 3: projectNeighborhoodEigenVec<3>(src, transmat, dst, neighborhood_r, border); break;
		case 4: projectNeighborhoodEigenVec<4>(src, transmat, dst, neighborhood_r, border); break;
		case 5: projectNeighborhoodEigenVec<5>(src, transmat, dst, neighborhood_r, border); break;
		case 6: projectNeighborhoodEigenVec<6>(src, transmat, dst, neighborhood_r, border); break;
		case 7: projectNeighborhoodEigenVec<7>(src, transmat, dst, neighborhood_r, border); break;
		case 8: projectNeighborhoodEigenVec<8>(src, transmat, dst, neighborhood_r, border); break;
		case 9: projectNeighborhoodEigenVec<9>(src, transmat, dst, neighborhood_r, border); break;
		default:
			projectNeighborhoodEigenVecCn(src, transmat, dst, neighborhood_r, border);
			break;
		}
	}
	else
	{
		Mat highDimGuide(src[0].size(), CV_MAKE_TYPE(CV_32F, dim));
		{
			//cp::Timer t("cvt HDI");
			cp::IM2COL(src, highDimGuide, neighborhood_r, border);
		}
		if (method == (int)NeighborhoodPCA::OPENCV_PCA)
		{
			PCA pca(highDimGuide.reshape(1, imsize.area()), cv::Mat(), cv::PCA::DATA_AS_ROW, dest_channels);
			Mat temp = pca.project(highDimGuide.reshape(1, imsize.area())).reshape(dest_channels, src[0].rows);
			split(temp, dst);
		}
		else if (method == (int)NeighborhoodPCA::OPENCV_COV)
		{
			Mat x = highDimGuide.reshape(1, imsize.area());
			Mat cov, mean;
			cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL | cv::COVAR_SCALE | cv::COVAR_ROWS);
			//cv::calcCovarMatrix(x, cov, mean, cv::COVAR_NORMAL |  cv::COVAR_ROWS);
			//print_mat(cov);
			Mat eval, evec;
			eigen(cov, eval, evec);


			Mat transmat;
			//print_matinfo(evec);
			evec(Rect(0, 0, evec.cols, dest_channels)).convertTo(transmat, CV_32F);
			//transmat = Mat::eye(transmat.size(), CV_32F);

			switch (dest_channels)
			{
			case 1: projectNeighborhoodEigenVec<1>(src, transmat, dst, neighborhood_r, border); break;
			case 2: projectNeighborhoodEigenVec<2>(src, transmat, dst, neighborhood_r, border); break;
			case 3: projectNeighborhoodEigenVec<3>(src, transmat, dst, neighborhood_r, border); break;
			case 4: projectNeighborhoodEigenVec<4>(src, transmat, dst, neighborhood_r, border); break;
			case 5: projectNeighborhoodEigenVec<5>(src, transmat, dst, neighborhood_r, border); break;
			case 6: projectNeighborhoodEigenVec<6>(src, transmat, dst, neighborhood_r, border); break;
			case 7: projectNeighborhoodEigenVec<7>(src, transmat, dst, neighborhood_r, border); break;
			case 8: projectNeighborhoodEigenVec<8>(src, transmat, dst, neighborhood_r, border); break;
			case 9: projectNeighborhoodEigenVec<9>(src, transmat, dst, neighborhood_r, border); break;
			default:
				projectNeighborhoodEigenVecCn(src, transmat, dst, neighborhood_r, border);
				break;
			}
			/*
			Mat temp;
			cv::transform(highDimGuide, temp, transmat);
			split(temp, dst);
			*/
		}
	}
}
#pragma endregion

void patchPCATile(const Mat& src, Mat& dest, const int neighborhood_r, const int dest_channels, const int border, const int method, const Size div)
{
	dest.create(src.size(), CV_MAKE_TYPE(CV_32F, dest_channels));
	const int channels = src.channels();

	const int vecsize = sizeof(__m256) / sizeof(float);//8

	if (div.area() == 1)
	{
		patchPCA(src, dest, neighborhood_r, dest_channels, border, method);
	}
	else
	{
		int r = neighborhood_r;
		const int R = get_simd_ceil(r, 8);
		Size tileSize = cp::getTileAlignSize(src.size(), div, r, vecsize, vecsize);
		Size divImageSize = cv::Size(src.cols / div.width, src.rows / div.height);

		vector<Mat> split_dst(channels);

		for (int c = 0; c < channels; c++)
		{
			split_dst[c].create(tileSize, CV_32FC1);
		}

		const int thread_max = omp_get_max_threads();
		vector<vector<Mat>>	subImageInput(thread_max);
		vector<vector<Mat>>	subImageOutput(thread_max);
		vector<Mat>	subImageOutput2(thread_max);
		for (int n = 0; n < thread_max; n++)
		{
			subImageInput[n].resize(channels);
			subImageOutput[n].resize(channels);
		}

		std::vector<cv::Mat> srcSplit;
		if (src.channels() != 3)split(src, srcSplit);

#pragma omp parallel for schedule(static)
		for (int n = 0; n < div.area(); n++)
		{
			const int thread_num = omp_get_thread_num();
			const cv::Point idx = cv::Point(n % div.width, n / div.width);

			if (src.channels() == 3)
			{
				cp::cropSplitTileAlign(src, subImageInput[thread_num], div, idx, r, border, vecsize, vecsize, vecsize, vecsize);
			}
			else
			{
				for (int c = 0; c < srcSplit.size(); c++)
				{
					cp::cropTileAlign(srcSplit[c], subImageInput[thread_num][c], div, idx, r, border, vecsize, vecsize, vecsize, vecsize);
				}
			}
			patchPCA(subImageInput[thread_num], subImageOutput[thread_num], neighborhood_r, dest_channels, border, method);
			merge(subImageOutput[thread_num], subImageOutput2[thread_num]);
			cp::pasteTileAlign(subImageOutput2[thread_num], dest, div, idx, r, 8, 8);
		}
	}
}
