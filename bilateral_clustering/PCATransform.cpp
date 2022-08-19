#include "PCATransform.hpp"

#define useOpenMP

using namespace cv;
using namespace std;


PCATransform::PCATransform()
{
	mean.resize(1);
	mean[0].create(Size(3, 1), CV_32F);
	values.resize(1);
	values[0].create(Size(3, 1), CV_32F);
	vectors.resize(1);
	vectors[0].create(Size(3, 3), CV_32F);
	cov.resize(1);
	cov[0].create(Size(3, 3), CV_32F);
}

void PCATransform::Project(const Mat& _src, Mat& _dest)
{
#ifdef useOpenMP
	_dest = Mat(_src.size(), _src.type());

	area = _src.size().area();
	int simdarea = area / 8;
	float sum_1 = 0;
	float sum_2 = 0;
	float sum_3 = 0;
	const float* src_ptr = _src.ptr<float>(0);

	//mean
	for (int i = 0; i < area; i++)
	{
		const float* ptr = src_ptr + (i * 3);
		sum_1 += *ptr;
		sum_2 += *(ptr + 1);
		sum_3 += *(ptr + 2);
	}
	float* mean_ptr = mean[0].ptr<float>(0);
	*(mean_ptr) = sum_1 / (float)area;
	*(mean_ptr + 1) = sum_2 / (float)area;
	*(mean_ptr + 2) = sum_3 / (float)area;

	//cov
	//èâä˙âª
	sum_1 = 0;
	sum_2 = 0;
	sum_3 = 0;
	float* dest_ptr = _dest.ptr<float>(0);
	float sum_4 = 0;
	float sum_5 = 0;
	float sum_6 = 0;

	for (int i = 0; i < area; i++) 
	{
		const float* s_ptr = src_ptr + i * 3;
		float* d_ptr = dest_ptr + i * 3;

		*d_ptr = *s_ptr - *mean_ptr;
		*(d_ptr + 1) = *(s_ptr + 1) - *(mean_ptr + 1);
		*(d_ptr + 2) = *(s_ptr + 2) - *(mean_ptr + 2);

		sum_1 += *d_ptr * *d_ptr;
		sum_2 += *(d_ptr + 1) * *d_ptr;
		sum_3 += *(d_ptr + 1) * *(d_ptr + 1);
		sum_4 += *(d_ptr + 2) * *d_ptr;
		sum_5 += *(d_ptr + 2) * *(d_ptr + 1);
		sum_6 += *(d_ptr + 2) * *(d_ptr + 2);
	}

	float* cov_ptr = cov[0].ptr<float>(0);

	*cov_ptr = sum_1 / area;
	*(cov_ptr + 1) = sum_2 / area;
	*(cov_ptr + 2) = sum_4 / area;
	*(cov_ptr + 3) = *(cov_ptr + 1);
	*(cov_ptr + 4) = sum_3 / area;
	*(cov_ptr + 5) = sum_5 / area;
	*(cov_ptr + 6) = *(cov_ptr + 2);
	*(cov_ptr + 7) = *(cov_ptr + 5);
	*(cov_ptr + 8) = sum_6 / area;

	cv::eigen(cov[0], values[0], vectors[0]);
	float* vec_ptr = vectors[0].ptr<float>(0);


#pragma omp parallel for
	for (int i = 0; i < area; i++) {
		float cn1, cn2, cn3;
		float* d_ptr = dest_ptr + i * 3;
		cn1 = *d_ptr;
		cn2 = *(d_ptr + 1);
		cn3 = *(d_ptr + 2);
		*d_ptr = cn1 * *vec_ptr + cn2 * *(vec_ptr + 1) + cn3 * *(vec_ptr + 2);
		*(d_ptr + 1) = cn1 * *(vec_ptr + 3) + cn2 * *(vec_ptr + 4) + cn3 * *(vec_ptr + 5);
		*(d_ptr + 2) = cn1 * *(vec_ptr + 6) + cn2 * *(vec_ptr + 7) + cn3 * *(vec_ptr + 8);
	}
	//parallel_for_(Range(0, area / 8), pcaProject(_dest, vectors[0]));
#else
	parallel_for_(Range(0, 1), ComputeMean(_src, mean[0]));
	parallel_for_(Range(0, 1), ComputeCov(_src, mean[0], cov[0], _dest));
	eigen(cov[0], values[0], vectors[0]);
	parallel_for_(Range(0, area / 8), pcaProject(_dest, vectors[0]));
#endif
}

void PCATransform::BackProject(const Mat& _src, Mat& _dest) {
#ifdef useOpenMP
	_dest = Mat(_src.size(), _src.type());
	area = _src.size().area();
	const float* src_ptr = _src.ptr<float>(0);
	float* dest_ptr = _dest.ptr<float>(0);
	float* vec_ptr = vectors[0].ptr<float>(0);
	float* mean_ptr = mean[0].ptr<float>(0);

#pragma omp parallel for
	for (int i = 0; i < area; i++) {
		float cn1, cn2, cn3;
		float* d_ptr = dest_ptr + i * 3;
		const float* s_ptr = src_ptr + i * 3;
		cn1 = *s_ptr;
		cn2 = *(s_ptr + 1);
		cn3 = *(s_ptr + 2);
		*d_ptr = cn1 * *vec_ptr + cn2 * *(vec_ptr + 3) + cn3 * *(vec_ptr + 6) + *mean_ptr;
		*(d_ptr + 1) = cn1 * *(vec_ptr + 1) + cn2 * *(vec_ptr + 4) + cn3 * *(vec_ptr + 7) + *(mean_ptr + 1);
		*(d_ptr + 2) = cn1 * *(vec_ptr + 2) + cn2 * *(vec_ptr + 5) + cn3 * *(vec_ptr + 8) + *(mean_ptr + 2);
	}

#else
	parallel_for_(Range(0, 1), pcaBackProject(_src, mean[0], vectors[0], _dest));
#endif
}

void PCATransform::Project(const vector<Mat>& _src, vector<Mat>& _dest) 
{

	area = _src[0].size().area();
	int simdarea = area / 8;

	const float* src1_ptr = _src[0].ptr<float>();
	const float* src2_ptr = _src[1].ptr<float>();
	const float* src3_ptr = _src[2].ptr<float>();

	//mean

	__m256 mSum01, mSum02, mSum03;
	mSum01 = _mm256_setzero_ps();
	mSum02 = _mm256_setzero_ps();
	mSum03 = _mm256_setzero_ps();
	for (int i = 0; i < simdarea; i++) 
	{
		const float* ptr1 = src1_ptr + (i * 8);
		const float* ptr2 = src2_ptr + (i * 8);
		const float* ptr3 = src3_ptr + (i * 8);
		mSum01 = _mm256_add_ps(mSum01, _mm256_loadu_ps(ptr1));
		mSum02 = _mm256_add_ps(mSum02, _mm256_loadu_ps(ptr2));
		mSum03 = _mm256_add_ps(mSum03, _mm256_loadu_ps(ptr3));
	}
	float* mean_ptr = mean[0].ptr<float>(0);

	mSum01 = _mm256_hadd_ps(mSum01, mSum01);
	mSum01 = _mm256_hadd_ps(mSum01, mSum01);
	__m256 rmSum = _mm256_permute2f128_ps(mSum01, mSum01, 0 << 4 | 1);
	mSum01 = _mm256_unpacklo_ps(mSum01, rmSum);
	mSum01 = _mm256_hadd_ps(mSum01, mSum01);

	mSum02 = _mm256_hadd_ps(mSum02, mSum02);
	mSum02 = _mm256_hadd_ps(mSum02, mSum02);
	rmSum = _mm256_permute2f128_ps(mSum02, mSum02, 0 << 4 | 1);
	mSum02 = _mm256_unpacklo_ps(mSum02, rmSum);
	mSum02 = _mm256_hadd_ps(mSum02, mSum02);

	mSum03 = _mm256_hadd_ps(mSum03, mSum03);
	mSum03 = _mm256_hadd_ps(mSum03, mSum03);
	rmSum = _mm256_permute2f128_ps(mSum03, mSum03, 0 << 4 | 1);
	mSum03 = _mm256_unpacklo_ps(mSum03, rmSum);
	mSum03 = _mm256_hadd_ps(mSum03, mSum03);

	*(mean_ptr) = mSum01.m256_f32[0] / (float)area;
	*(mean_ptr + 1) = mSum02.m256_f32[0] / (float)area;
	*(mean_ptr + 2) = mSum03.m256_f32[0] / (float)area;

	//cov
	__m256 mSum04, mSum05, mSum06;
	mSum01 = _mm256_setzero_ps();
	mSum02 = _mm256_setzero_ps();
	mSum03 = _mm256_setzero_ps();
	mSum04 = _mm256_setzero_ps();
	mSum05 = _mm256_setzero_ps();
	mSum06 = _mm256_setzero_ps();
	__m256 mMean00, mMean01, mMean02;
	mMean00 = _mm256_set1_ps(*mean_ptr);
	mean_ptr++;
	mMean01 = _mm256_set1_ps(*mean_ptr);
	mean_ptr++;
	mMean02 = _mm256_set1_ps(*mean_ptr);
	for (int i = 0; i < simdarea; i++) 
	{
		__m256 mCn01, mCn02, mCn03;
		const float* ptr1 = src1_ptr + (i * 8);
		const float* ptr2 = src2_ptr + (i * 8);
		const float* ptr3 = src3_ptr + (i * 8);
		mCn01 = _mm256_sub_ps(_mm256_loadu_ps(ptr1), mMean00);
		mCn02 = _mm256_sub_ps(_mm256_loadu_ps(ptr2), mMean01);
		mCn03 = _mm256_sub_ps(_mm256_loadu_ps(ptr3), mMean02);

		mSum01 = _mm256_add_ps(mSum01, _mm256_mul_ps(mCn01, mCn01));
		mSum02 = _mm256_add_ps(mSum02, _mm256_mul_ps(mCn02, mCn01));
		mSum03 = _mm256_add_ps(mSum03, _mm256_mul_ps(mCn02, mCn02));
		mSum04 = _mm256_add_ps(mSum04, _mm256_mul_ps(mCn03, mCn01));
		mSum05 = _mm256_add_ps(mSum05, _mm256_mul_ps(mCn03, mCn02));
		mSum06 = _mm256_add_ps(mSum06, _mm256_mul_ps(mCn03, mCn03));
	}
	float* cov_ptr = cov[0].ptr<float>(0);

	mSum01 = _mm256_hadd_ps(mSum01, mSum01);
	mSum01 = _mm256_hadd_ps(mSum01, mSum01);
	rmSum = _mm256_permute2f128_ps(mSum01, mSum01, 0 << 4 | 1);
	mSum01 = _mm256_unpacklo_ps(mSum01, rmSum);
	mSum01 = _mm256_hadd_ps(mSum01, mSum01);

	mSum02 = _mm256_hadd_ps(mSum02, mSum02);
	mSum02 = _mm256_hadd_ps(mSum02, mSum02);
	rmSum = _mm256_permute2f128_ps(mSum02, mSum02, 0 << 4 | 1);
	mSum02 = _mm256_unpacklo_ps(mSum02, rmSum);
	mSum02 = _mm256_hadd_ps(mSum02, mSum02);

	mSum03 = _mm256_hadd_ps(mSum03, mSum03);
	mSum03 = _mm256_hadd_ps(mSum03, mSum03);
	rmSum = _mm256_permute2f128_ps(mSum03, mSum03, 0 << 4 | 1);
	mSum03 = _mm256_unpacklo_ps(mSum03, rmSum);
	mSum03 = _mm256_hadd_ps(mSum03, mSum03);

	mSum04 = _mm256_hadd_ps(mSum04, mSum04);
	mSum04 = _mm256_hadd_ps(mSum04, mSum04);
	rmSum = _mm256_permute2f128_ps(mSum04, mSum04, 0 << 4 | 1);
	mSum04 = _mm256_unpacklo_ps(mSum04, rmSum);
	mSum04 = _mm256_hadd_ps(mSum04, mSum04);

	mSum05 = _mm256_hadd_ps(mSum05, mSum05);
	mSum05 = _mm256_hadd_ps(mSum05, mSum05);
	rmSum = _mm256_permute2f128_ps(mSum05, mSum05, 0 << 4 | 1);
	mSum05 = _mm256_unpacklo_ps(mSum05, rmSum);
	mSum05 = _mm256_hadd_ps(mSum05, mSum05);

	mSum06 = _mm256_hadd_ps(mSum06, mSum06);
	mSum06 = _mm256_hadd_ps(mSum06, mSum06);
	rmSum = _mm256_permute2f128_ps(mSum06, mSum06, 0 << 4 | 1);
	mSum06 = _mm256_unpacklo_ps(mSum06, rmSum);
	mSum06 = _mm256_hadd_ps(mSum06, mSum06);

	*(cov_ptr) = mSum01.m256_f32[0] / (float)area;
	*(cov_ptr + 1) = mSum02.m256_f32[0] / (float)area;
	*(cov_ptr + 2) = mSum04.m256_f32[0] / (float)area;
	*(cov_ptr + 3) = *(cov_ptr + 1);
	*(cov_ptr + 4) = mSum03.m256_f32[0] / (float)area;
	*(cov_ptr + 5) = mSum05.m256_f32[0] / (float)area;
	*(cov_ptr + 6) = *(cov_ptr + 2);
	*(cov_ptr + 7) = *(cov_ptr + 5);
	*(cov_ptr + 8) = mSum06.m256_f32[0] / (float)area;

	cv::eigen(cov[0], values[0], vectors[0]);

	parallel_for_(Range(0, area / 24), Project_sp(_src, mean[0], vectors[0], _dest));

}

void PCATransform::BackProject(const vector<Mat>& _src, vector<Mat>& _dest)
{
	area = _src[0].size().area();
	parallel_for_(Range(0, area / 24), BackProject_sp(_src, mean[0], vectors[0], _dest));
}

void ComputeMean::operator() (const Range& range) const
{
	for (int i = range.start; i < range.end; i++)
	{
		__m256 mTmp00, mTmp01, mTmp02;
		float* src0_ptr = Input.ptr<float>(0) + 0 + (i * 24);
		float* src1_ptr = Input.ptr<float>(0) + 8 + (i * 24);
		float* src2_ptr = Input.ptr<float>(0) + 16 + (i * 24);
	}
}

void Project_sp::operator() (const Range& range) const
{
	float* vec_ptr = Vec.ptr<float>(0);
	float* mean_ptr = Mean.ptr<float>(0);
	__m256 mVec00, mVec01, mVec02, mVec03, mVec04, mVec05, mVec06, mVec07, mVec08;
	__m256 mMean00, mMean01, mMean02;

	mVec00 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec01 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec02 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;

	mVec03 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec04 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec05 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;

	mVec06 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec07 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec08 = _mm256_set1_ps(*vec_ptr);

	mMean00 = _mm256_set1_ps(*mean_ptr);
	mean_ptr++;
	mMean01 = _mm256_set1_ps(*mean_ptr);
	mean_ptr++;
	mMean02 = _mm256_set1_ps(*mean_ptr);

	for (int i = range.start; i < range.end; i++)
	{
		__m256 mCn00, mCn01, mCn02;
		__m256 mTmp00, mTmp01, mTmp02;
		const float* src1_ptr = Input[0].ptr<float>(0) + (i * 24);
		const float* src2_ptr = Input[1].ptr<float>(0) + (i * 24);
		const float* src3_ptr = Input[2].ptr<float>(0) + (i * 24);
		float* dest1_ptr = Output[0].ptr<float>(0) + (i * 24);
		float* dest2_ptr = Output[1].ptr<float>(0) + (i * 24);
		float* dest3_ptr = Output[2].ptr<float>(0) + (i * 24);

		mCn00 = _mm256_sub_ps(_mm256_loadu_ps(src1_ptr), mMean00);
		mCn01 = _mm256_sub_ps(_mm256_loadu_ps(src2_ptr), mMean01);
		mCn02 = _mm256_sub_ps(_mm256_loadu_ps(src3_ptr), mMean02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec00);
		mTmp01 = _mm256_mul_ps(mCn01, mVec01);
		mTmp02 = _mm256_mul_ps(mCn02, mVec02);
		mTmp02 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(dest1_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec03);
		mTmp01 = _mm256_mul_ps(mCn01, mVec04);
		mTmp02 = _mm256_mul_ps(mCn02, mVec05);
		mTmp02 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(dest2_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec06);
		mTmp01 = _mm256_mul_ps(mCn01, mVec07);
		mTmp02 = _mm256_mul_ps(mCn02, mVec08);
		mTmp02 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(dest3_ptr, mTmp02);

		//8
		src1_ptr += 8;
		src2_ptr += 8;
		src3_ptr += 8;
		dest1_ptr += 8;
		dest2_ptr += 8;
		dest3_ptr += 8;

		mCn00 = _mm256_sub_ps(_mm256_loadu_ps(src1_ptr), mMean00);
		mCn01 = _mm256_sub_ps(_mm256_loadu_ps(src2_ptr), mMean01);
		mCn02 = _mm256_sub_ps(_mm256_loadu_ps(src3_ptr), mMean02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec00);
		mTmp01 = _mm256_mul_ps(mCn01, mVec01);
		mTmp02 = _mm256_mul_ps(mCn02, mVec02);
		mTmp02 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(dest1_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec03);
		mTmp01 = _mm256_mul_ps(mCn01, mVec04);
		mTmp02 = _mm256_mul_ps(mCn02, mVec05);
		mTmp02 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(dest2_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec06);
		mTmp01 = _mm256_mul_ps(mCn01, mVec07);
		mTmp02 = _mm256_mul_ps(mCn02, mVec08);
		mTmp02 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(dest3_ptr, mTmp02);


		//16
		src1_ptr += 8;
		src2_ptr += 8;
		src3_ptr += 8;
		dest1_ptr += 8;
		dest2_ptr += 8;
		dest3_ptr += 8;

		mCn00 = _mm256_sub_ps(_mm256_loadu_ps(src1_ptr), mMean00);
		mCn01 = _mm256_sub_ps(_mm256_loadu_ps(src2_ptr), mMean01);
		mCn02 = _mm256_sub_ps(_mm256_loadu_ps(src3_ptr), mMean02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec00);
		mTmp01 = _mm256_mul_ps(mCn01, mVec01);
		mTmp02 = _mm256_mul_ps(mCn02, mVec02);
		mTmp02 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(dest1_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec03);
		mTmp01 = _mm256_mul_ps(mCn01, mVec04);
		mTmp02 = _mm256_mul_ps(mCn02, mVec05);
		mTmp02 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(dest2_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec06);
		mTmp01 = _mm256_mul_ps(mCn01, mVec07);
		mTmp02 = _mm256_mul_ps(mCn02, mVec08);
		mTmp02 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(dest3_ptr, mTmp02);
	}
}

void pcaProject::operator() (const Range& range) const
{
	float* vec_ptr = Vec.ptr<float>(0);
	__m256 mVec00, mVec01, mVec02, mVec03, mVec04, mVec05, mVec06, mVec07, mVec08;
	__m256i mIdx;
	mIdx = _mm256_set_epi32(0, 1, 2, 0, 1, 2, 0, 1);
	mVec00 = _mm256_i32gather_ps(vec_ptr, mIdx, 4);
	mIdx = _mm256_set_epi32(2, 0, 1, 2, 0, 1, 2, 0);
	mVec01 = _mm256_i32gather_ps(vec_ptr, mIdx, 4);
	mIdx = _mm256_set_epi32(1, 2, 0, 1, 2, 0, 1, 2);
	mVec02 = _mm256_i32gather_ps(vec_ptr, mIdx, 4);

	mIdx = _mm256_set_epi32(3, 4, 5, 3, 4, 5, 3, 4);
	mVec03 = _mm256_i32gather_ps(vec_ptr, mIdx, 4);
	mIdx = _mm256_set_epi32(5, 3, 4, 5, 3, 4, 5, 3);
	mVec04 = _mm256_i32gather_ps(vec_ptr, mIdx, 4);
	mIdx = _mm256_set_epi32(4, 5, 3, 4, 5, 3, 4, 5);
	mVec05 = _mm256_i32gather_ps(vec_ptr, mIdx, 4);

	mIdx = _mm256_set_epi32(6, 7, 8, 6, 7, 8, 6, 7);
	mVec06 = _mm256_i32gather_ps(vec_ptr, mIdx, 4);
	mIdx = _mm256_set_epi32(8, 6, 7, 8, 6, 7, 8, 6);
	mVec07 = _mm256_i32gather_ps(vec_ptr, mIdx, 4);
	mIdx = _mm256_set_epi32(7, 8, 6, 7, 8, 6, 7, 8);
	mVec08 = _mm256_i32gather_ps(vec_ptr, mIdx, 4);

	for (int i = range.start; i < range.end; i++)
	{
		__m256 mCn00, mCn01, mCn02;
		__m256 mTmp00, mTmp01, mTmp02, mTmp03, mTmp04, mTmp05, mTmp06, mTmp07, mTmp08;
		float* src_ptr = Input.ptr<float>(0) + (i * 24);

		mCn00 = _mm256_loadu_ps(src_ptr);
		mCn01 = _mm256_loadu_ps(src_ptr + 8);
		mCn02 = _mm256_loadu_ps(src_ptr + 16);

		mTmp00 = _mm256_mul_ps(mCn00, mVec00);
		mTmp01 = _mm256_mul_ps(mCn00, mVec03);
		mTmp02 = _mm256_mul_ps(mCn00, mVec06);
		mTmp03 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(src_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn01, mVec01);
		mTmp01 = _mm256_mul_ps(mCn01, mVec04);
		mTmp02 = _mm256_mul_ps(mCn01, mVec07);
		mTmp04 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(src_ptr + 8, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn02, mVec02);
		mTmp01 = _mm256_mul_ps(mCn02, mVec05);
		mTmp02 = _mm256_mul_ps(mCn02, mVec08);
		mTmp02 = _mm256_add_ps(mTmp02, _mm256_add_ps(mTmp00, mTmp01));
		_mm256_storeu_ps(src_ptr + 16, mTmp02);
	}
}

void BackProject_sp::operator() (const Range& range) const
{
	float* vec_ptr = Vec.ptr<float>(0);
	float* mean_ptr = Mean.ptr<float>(0);
	__m256 mVec00, mVec01, mVec02, mVec03, mVec04, mVec05, mVec06, mVec07, mVec08;
	__m256 mMean00, mMean01, mMean02;

	mVec00 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec01 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec02 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;

	mVec03 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec04 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec05 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;

	mVec06 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec07 = _mm256_set1_ps(*vec_ptr);
	vec_ptr++;
	mVec08 = _mm256_set1_ps(*vec_ptr);

	mMean00 = _mm256_set1_ps(*mean_ptr);
	mean_ptr++;
	mMean01 = _mm256_set1_ps(*mean_ptr);
	mean_ptr++;
	mMean02 = _mm256_set1_ps(*mean_ptr);


	for (int i = range.start; i < range.end; i++)
	{
		__m256 mCn00, mCn01, mCn02;
		__m256 mTmp00, mTmp01, mTmp02;
		const float* src1_ptr = Input[0].ptr<float>(0) + (i * 24);
		const float* src2_ptr = Input[1].ptr<float>(0) + (i * 24);
		const float* src3_ptr = Input[2].ptr<float>(0) + (i * 24);
		float* dest1_ptr = Output[0].ptr<float>(0) + (i * 24);
		float* dest2_ptr = Output[1].ptr<float>(0) + (i * 24);
		float* dest3_ptr = Output[2].ptr<float>(0) + (i * 24);

		mCn00 = _mm256_loadu_ps(src1_ptr);
		mCn01 = _mm256_loadu_ps(src2_ptr);
		mCn02 = _mm256_loadu_ps(src3_ptr);

		mTmp00 = _mm256_mul_ps(mCn00, mVec00);
		mTmp01 = _mm256_mul_ps(mCn01, mVec03);
		mTmp02 = _mm256_mul_ps(mCn02, mVec06);
		mTmp02 = _mm256_add_ps(_mm256_add_ps(mTmp00, mTmp01), _mm256_add_ps(mTmp02, mMean00));
		_mm256_storeu_ps(dest1_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec01);
		mTmp01 = _mm256_mul_ps(mCn01, mVec04);
		mTmp02 = _mm256_mul_ps(mCn02, mVec07);
		mTmp02 = _mm256_add_ps(_mm256_add_ps(mTmp00, mTmp01), _mm256_add_ps(mTmp02, mMean01));
		_mm256_storeu_ps(dest2_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec02);
		mTmp01 = _mm256_mul_ps(mCn01, mVec05);
		mTmp02 = _mm256_mul_ps(mCn02, mVec08);
		mTmp02 = _mm256_add_ps(_mm256_add_ps(mTmp00, mTmp01), _mm256_add_ps(mTmp02, mMean02));
		_mm256_storeu_ps(dest3_ptr, mTmp02);

		//8
		src1_ptr += 8;
		src2_ptr += 8;
		src3_ptr += 8;
		dest1_ptr += 8;
		dest2_ptr += 8;
		dest3_ptr += 8;

		mCn00 = _mm256_loadu_ps(src1_ptr);
		mCn01 = _mm256_loadu_ps(src2_ptr);
		mCn02 = _mm256_loadu_ps(src3_ptr);

		mTmp00 = _mm256_mul_ps(mCn00, mVec00);
		mTmp01 = _mm256_mul_ps(mCn01, mVec03);
		mTmp02 = _mm256_mul_ps(mCn02, mVec06);
		mTmp02 = _mm256_add_ps(_mm256_add_ps(mTmp00, mTmp01), _mm256_add_ps(mTmp02, mMean00));
		_mm256_storeu_ps(dest1_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec01);
		mTmp01 = _mm256_mul_ps(mCn01, mVec04);
		mTmp02 = _mm256_mul_ps(mCn02, mVec07);
		mTmp02 = _mm256_add_ps(_mm256_add_ps(mTmp00, mTmp01), _mm256_add_ps(mTmp02, mMean01));
		_mm256_storeu_ps(dest2_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec02);
		mTmp01 = _mm256_mul_ps(mCn01, mVec05);
		mTmp02 = _mm256_mul_ps(mCn02, mVec08);
		mTmp02 = _mm256_add_ps(_mm256_add_ps(mTmp00, mTmp01), _mm256_add_ps(mTmp02, mMean02));
		_mm256_storeu_ps(dest3_ptr, mTmp02);

		//16
		src1_ptr += 8;
		src2_ptr += 8;
		src3_ptr += 8;
		dest1_ptr += 8;
		dest2_ptr += 8;
		dest3_ptr += 8;

		mCn00 = _mm256_loadu_ps(src1_ptr);
		mCn01 = _mm256_loadu_ps(src2_ptr);
		mCn02 = _mm256_loadu_ps(src3_ptr);

		mTmp00 = _mm256_mul_ps(mCn00, mVec00);
		mTmp01 = _mm256_mul_ps(mCn01, mVec03);
		mTmp02 = _mm256_mul_ps(mCn02, mVec06);
		mTmp02 = _mm256_add_ps(_mm256_add_ps(mTmp00, mTmp01), _mm256_add_ps(mTmp02, mMean00));
		_mm256_storeu_ps(dest1_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec01);
		mTmp01 = _mm256_mul_ps(mCn01, mVec04);
		mTmp02 = _mm256_mul_ps(mCn02, mVec07);
		mTmp02 = _mm256_add_ps(_mm256_add_ps(mTmp00, mTmp01), _mm256_add_ps(mTmp02, mMean01));
		_mm256_storeu_ps(dest2_ptr, mTmp02);

		mTmp00 = _mm256_mul_ps(mCn00, mVec02);
		mTmp01 = _mm256_mul_ps(mCn01, mVec05);
		mTmp02 = _mm256_mul_ps(mCn02, mVec08);
		mTmp02 = _mm256_add_ps(_mm256_add_ps(mTmp00, mTmp01), _mm256_add_ps(mTmp02, mMean02));
		_mm256_storeu_ps(dest3_ptr, mTmp02);
	}
}
