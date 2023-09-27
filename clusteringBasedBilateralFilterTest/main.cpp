#include <opencp.hpp>

#include "common.hpp"
#include "HDGF.hpp"
#include "patchPCA.hpp"
#include <stdio.h>
#include <fstream>

using namespace std;
using namespace cv;

#if _DEBUG
#pragma comment(lib,"../x64/Debug/clusteringHDGF.lib")
#pragma comment(lib,"opencpd.lib")
#pragma comment(lib,"SpatialFilter.lib")
#else
#pragma comment(lib,"../x64/Release/clusteringHDGF.lib")
#pragma comment(lib,"opencp.lib")
#pragma comment(lib,"SpatialFilter.lib")
#endif


using namespace std;
using namespace cv;

void highDimensionalGaussianFilterTilePCA(Mat& src, Mat& guide, Mat& dest, Size ksize, const float sigma_range, const float sigma_space, const int destChannels, Size div, int border)
{
	dest.create(src.size(), CV_MAKETYPE(CV_32F, src.channels()));
	CV_Assert(src.cols % div.width == 0);
	CV_Assert(src.rows % div.height == 0);
	const int rectx = src.cols / div.width;
	const int recty = src.rows / div.height;

	const int r = get_simd_ceil(ksize.width >> 1, 8);
	Mat sbb; copyMakeBorder(src, sbb, r, r, r, r, border);
	Mat gbb; copyMakeBorder(guide, gbb, r, r, r, r, border);

#pragma omp parallel for schedule (static)
	for (int n = 0; n < div.area(); n++)
	{
		const int j = n / div.width;
		const int i = n % div.width;

		const Rect roi = Rect(i * rectx, j * recty, rectx + 2 * r, recty + 2 * r);
		const Rect roi2 = Rect(i * rectx, j * recty, rectx, recty);
		Mat a = sbb(roi).clone();
		Mat b = gbb(roi).clone();
		Mat t, g;
		cp::cvtColorPCA(b, g, destChannels);
		cp::highDimensionalGaussianFilter(a, g, t, ksize, sigma_range, sigma_space, border);
		t(Rect(r, r, rectx, recty)).copyTo(dest(roi2));
	}
	//cp::imshowScale("test", dest);
}

template<typename T>
void mergeVector(vector<Mat>& src1, vector<Mat>& src2, Mat& dest, const double lambda_for_src2)
{
	const int c1 = (int)src1.size();
	const int c2 = (int)src2.size();
	const int ch = c1 + c2;
	const Size size = src1[0].size();
	if (typeid(T) == typeid(uchar)) dest.create(size, CV_MAKETYPE(CV_8U, ch));
	if (typeid(T) == typeid(float)) dest.create(size, CV_MAKETYPE(CV_32F, ch));
	if (typeid(T) == typeid(double)) dest.create(size, CV_MAKETYPE(CV_64F, ch));
	const int s = size.area();
	AutoBuffer<T*> sptr1(c1);
	AutoBuffer<T*> sptr2(c2);

	for (int c = 0; c < c1; c++)
	{
		sptr1[c] = src1[c].ptr<T>();
	}
	for (int c = 0; c < c2; c++)
	{
		sptr2[c] = src2[c].ptr<T>();
	}

	T* dptr = dest.ptr<T>();
	for (int i = 0; i < s; i++)
	{
		for (int c = 0; c < c1; c++)
		{
			dptr[ch * i + c] = sptr1[c][i];
		}
		for (int c = 0; c < c2; c++)
		{
			dptr[ch * i + c + c1] = T(sptr2[c][i] * lambda_for_src2);
		}
	}
}

template<typename T, typename destT>
void mergeConvertVector(vector<Mat>& src1, vector<Mat>& src2, Mat& dest, const destT lambda_for_src2)
{
	const int c1 = (int)src1.size();
	const int c2 = (int)src2.size();
	const int ch = c1 + c2;
	const Size size = src1[0].size();
	if (typeid(destT) == typeid(uchar)) dest.create(size, CV_MAKETYPE(CV_8U, ch));
	if (typeid(destT) == typeid(float)) dest.create(size, CV_MAKETYPE(CV_32F, ch));
	if (typeid(destT) == typeid(double)) dest.create(size, CV_MAKETYPE(CV_64F, ch));
	const int s = size.area();
	AutoBuffer<T*> sptr1(c1);
	AutoBuffer<T*> sptr2(c2);

	for (int c = 0; c < c1; c++)
	{
		sptr1[c] = src1[c].ptr<T>();
	}
	for (int c = 0; c < c2; c++)
	{
		sptr2[c] = src2[c].ptr<T>();
	}

	destT* dptr = dest.ptr<destT>();
	for (int i = 0; i < s; i++)
	{
		for (int c = 0; c < c1; c++)
		{
			dptr[ch * i + c] = (destT)sptr1[c][i];
		}

		for (int c = 0; c < c2; c++)
		{
			dptr[ch * i + c + c1] = (destT)(sptr2[c][i] * lambda_for_src2);
		}
	}
}

void mergeImage(Mat& src1, Mat& src2, Mat& dest, const double lambda_for_src2)
{
	vector<Mat> v1; split(src1, v1);
	vector<Mat> v2; split(src2, v2);
	if (src1.depth() == CV_8U) mergeVector<uchar>(v1, v2, dest, lambda_for_src2);
	if (src1.depth() == CV_32F) mergeVector<uchar>(v1, v2, dest, lambda_for_src2);
	if (src1.depth() == CV_64F) mergeVector<uchar>(v1, v2, dest, lambda_for_src2);
}

void mergeConvertImage(Mat& src1, Mat& src2, Mat& dest, const double lambda_for_src2, const int depth)
{
	vector<Mat> v1; split(src1, v1);
	vector<Mat> v2; split(src2, v2);
	if (src1.depth() == CV_8U)
	{
		if (depth == CV_8U)  mergeConvertVector<uchar, uchar>(v1, v2, dest, lambda_for_src2);
		if (depth == CV_32F) mergeConvertVector<uchar, float>(v1, v2, dest, lambda_for_src2);
		if (depth == CV_64F) mergeConvertVector<uchar, double>(v1, v2, dest, lambda_for_src2);
	}
	if (src1.depth() == CV_32F)
	{
		if (depth == CV_8U)  mergeConvertVector<float, uchar>(v1, v2, dest, lambda_for_src2);
		if (depth == CV_32F) mergeConvertVector<float, float>(v1, v2, dest, lambda_for_src2);
		if (depth == CV_64F) mergeConvertVector<float, double>(v1, v2, dest, lambda_for_src2);
	}
	if (src1.depth() == CV_32F)
	{
		if (depth == CV_8U)  mergeConvertVector<double, uchar>(v1, v2, dest, lambda_for_src2);
		if (depth == CV_32F) mergeConvertVector<double, float>(v1, v2, dest, lambda_for_src2);
		if (depth == CV_64F) mergeConvertVector<double, double>(v1, v2, dest, lambda_for_src2);
	}
}

enum
{
	//GIR, //1+1=2
	RGB, //3
	RGBIR, //3+1=4
	RGBD,  //1+3=4,
	FNF,//3+3=6
	HSI,//33
	NLM,//3x3x3=27
};

string getHDGFTypeName(int type)
{
	string ret = "";
	switch (type)
	{
		//case GIR:   ret = "Gray-IR 2:1+1"; break;
	case RGB:   ret = "RGB     3"; break;
	case RGBIR: ret = "RGB-IR  4:3+1"; break;
	case RGBD:	ret = "RGB-D   4:1+3"; break;
	case FNF:	ret = "FL-NoFL 6:3+3"; break;
	case HSI:	ret = "HSI     33"; break;
	case NLM:	ret = "NLM     27:3x3x"; break;
	default:
		break;
	}
	return ret;
}

string getclusteringHDGFMethodName(int method)
{
	string ret = "";
	if (method == 0) ret = "Interpolation";
	if (method == 1) ret = "Nystrom";
	if (method == 2) ret = "Soft";
	return ret;
}
void generateRGBIR()
{
	for (int i = 0; i < 10; i++)
	{
		string s;
		if (i == 0)s = "0050";
		if (i == 1)s = "0012";
		if (i == 2)s = "0005";
		if (i == 3)s = "0008";
		if (i == 4)s = "0030";
		if (i == 5)s = "0061";
		if (i == 6)s = "0076";
		if (i == 7)s = "0088";
		if (i == 8)s = "0020";
		if (i == 9)s = "0016";

		cout << "img/RGBIR/oldbuilding/" + s + "_nir.png" << endl;
		Mat ir = imread("img/RGBIR/oldbuilding/" + s + "_nir.png", 0);
		if (ir.empty())cout << "empty" << endl;
		resize(ir, ir, Size(512, 512), 0.0, 0.0, INTER_AREA);
		bool ret = imwrite(format("img/RGBIR/%d_ir.png", i), ir);
		if (!ret)cout << "NG:" << format("RGBIR/%d_ir.png", i) << endl;

		Mat rgb = imread("img/RGBIR/oldbuilding/" + s + "_rgb.png");
		resize(rgb, rgb, Size(512, 512), 0.0, 0.0, INTER_AREA);
		imwrite(format("img/RGBIR/%d_rgb.png", i), rgb);
	}
	getchar();
}

void generateRGBD()
{
	for (int i = 0; i < 10; i++)
	{
		string s;
		if (i == 0)s = "Aloe";
		if (i == 1)s = "Baby1";
		if (i == 2)s = "Books";
		if (i == 3)s = "Bowling1";
		if (i == 4)s = "Cloth1";
		if (i == 5)s = "ConesH";
		if (i == 6)s = "Dolls";
		if (i == 7)s = "Laundry";
		if (i == 8)s = "Reindeer";
		if (i == 9)s = "TeddyH";
		cout << "img/stereo/" + s + "/disp1.png" << endl;
		Mat disp = imread("img/stereo/" + s + "/disp1.png", 0);
		if (i == 5 || i == 9)
		{
			cp::fillOcclusion(disp);
			resize(disp, disp, Size(512, 512), 0, 0, INTER_NEAREST);

		}
		else
		{
			disp = disp(Rect(0, 0, 512, 512)).clone();
			cp::fillOcclusion(disp);
		}

		imwrite(s + "_disp.png", disp);

		Mat left = imread("img/stereo/" + s + "/view1.png");
		if (i == 5 || i == 9)
		{
			resize(left, left, Size(512, 512), 0, 0, INTER_AREA);
		}
		else
		{
			left = left(Rect(0, 0, 512, 512)).clone();
		}
		imwrite(s + "_rgb.png", left);
	}
	//getchar();
}

void generateHSI()
{
	Mat a, b;
	for (int n = 0; n < 10; n++)
	{
		vector<Mat> hsi;
		cout << n << " ";
		for (int i = 1; i < 34; i++)
		{
			if (n == 0)a = imread(format("img/hsi/crown/%02d.png", i), 0);
			if (n == 1)a = imread(format("img/hsi/ruivaes/%02d.png", i), 0);
			if (n == 2)a = imread(format("img/hsi/mosteiro/%02d.png", i), 0);
			if (n == 3)a = imread(format("img/hsi/cyflower/%02d.png", i), 0);
			if (n == 4)a = imread(format("img/hsi/cbrufefields/%02d.png", i), 0);
			if (n == 5)a = imread(format("img/hsi/braga/%02d.png", i), 0);
			if (n == 6)a = imread(format("img/hsi/ribeira/%02d.png", i), 0);
			if (n == 7)a = imread(format("img/hsi/farme/%02d.png", i), 0);

			//if (n == 8)a = imread(format("img/hsi/apartment/%02d.png", min(i, 31)), 0);
			if (n == 8)a = imread(format("img/hsi/city/%02d.png", min(i, 31)), 0);
			//if (n == 10)a = imread(format("img/hsi/flower/%02d.png", min(i, 31)), 0);
			//if (n == 11)a = imread(format("img/hsi/forest/%02d.png", min(i, 31)), 0);
			//if (n == 12)a = imread(format("img/hsi/forest2/%02d.png", min(i, 31)), 0);
			//if (n == 13)a = imread(format("img/hsi/forest3/%02d.png", min(i, 31)), 0);
			//if (n == 14)a = imread(format("img/hsi/house/%02d.png", min(i, 31)), 0);
			if (n == 9)a = imread(format("img/hsi/toys/%02d.png", min(i, 31)), 0);

			const int w = get_simd_floor(a.cols, 8 * 4);
			const int h = get_simd_floor(a.rows, 8 * 4);
			cv::resize(a(Rect(0, 0, w, h)), b, Size(512, 512), 0, 0, INTER_AREA);
			imwrite(format("img/hsi/%d_%03d.png", n, i), b);
			hsi.push_back(b.clone());
		}
		Mat h, rgb;
		merge(hsi, h);
		cp::cvtColorHSI2BGR(h, rgb);
		imwrite(format("rgb%d.png", n), rgb);
	}
}

void generateFnF()
{
	for (int i = 1; i < 10; i++)
	{
		Mat f = imread(format("img/flash/%df.png", i));
		Mat n = imread(format("img/flash/%dn.png", i));

		int l = min(f.cols, f.rows);
		resize(f(Rect(0, 0, l, l)), f, Size(512, 512), 0, 0, INTER_AREA);
		resize(n(Rect(0, 0, l, l)), n, Size(512, 512), 0, 0, INTER_AREA);
		imwrite(format("img/flash/%dflash.png", i), f);
		imwrite(format("img/flash/%dnoflash.png", i), n);
		//cp::guiAlphaBlend(f, n);
	}
}

void testClusteringHDGF_SpringerNature(string wname)
{
	const int color = 1;//color
	int ColorOptionRGB = IMREAD_COLOR;
	int ColorOptionGRAY = IMREAD_GRAYSCALE;
	//int ColorOptionRGB = IMREAD_REDUCED_COLOR_2;
	//int ColorOptionGRAY = IMREAD_REDUCED_GRAYSCALE_2;
	const int readw = 512 / 1;
	const int readh = 512 / 1;
	//#define VIS_TILING_PCA // for visualization. not using for experiment.
	//generateFnF();
	//generateRGBD();
	//generateRGBIR();
	//generateHSI();
	bool isReadAll = true;
	//bool isReadAll = false;

	//full PCA
	const int iterationRef = 1;
	const bool isRefFullPCA = true;
	const bool isRefTilePCA = true;
	const bool isRefTilePCA2 = false;
	const bool isRefTileDCT = false;


	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	//patchPCATest();

	namedWindow(wname);
	moveWindow(wname, 100, 100);
	string wname2 = wname;
	//string wname2 = "";
	namedWindow(wname2);

	//const int color = 0;//gray
	//src[0] = imread("img/lenna.png");

	cv::Size maxdivision(32, 32);
	//cv::Size division(4, 4);
	//cv::Size division(8, 4);
	//cv::Size division(4, 4);
	//cv::Size division(8, 4);
	//cv::Size division(2, 2);
	//cv::Size division(1, 1);

	int typeHDGF = RGB;
	//int typeHDGF = RGBD;
	//int typeHDGF = RGBIR;
	//int typeHDGF = FNF;
	//int typeHDGF = HSI;
	//int typeHDGF = NLM;
	createTrackbar("HDGF method", wname2, &typeHDGF, 5);
	//int clusteringHDGFMethod = 0; //interpolation
	int clusteringHDGFMethod = 1; //Nystrom
	//int clusteringHDGFMethod = 2; //soft
	createTrackbar("clusteringHDGF", wname2, &clusteringHDGFMethod, 2);

#pragma region readimage
	const int imgNum = 10;
	vector<Mat> src(imgNum);
	for (int i = 0; i < imgNum; ++i)
	{
		const int idx = (i == 9) ? 13 : i + 1;
		string path = format("img/Kodak/kodim%02d.png", idx);
		Mat a = imread(path, (color == 1) ? ColorOptionRGB : ColorOptionGRAY);
		if (a.empty())cout << "empty:" << path << endl;
		const int bb = 10;
		Mat d;
		copyMakeBorder(a(Rect(bb, bb, readw - 2 * bb, readh - 2 * bb)).clone(), d, bb, bb, bb, bb, BORDER_REFLECT101);
		src[i] = d.clone();
	}
	//Mat v; cp::concat(src, v, 5, 2); imshow("dst", v); cv::waitKey();
	vector<Mat> src_rgbir_rgb(10);
	vector<Mat> src_rgbir_ir(10);
	if (typeHDGF == RGBIR || isReadAll)
	{
		cout << "load RGB-NIR" << endl;
		//E:\Github\bilateral_clustering\bilateral_clustering\
		Mat temp = imread("img/ir/books.png");
		for (int idx = 0; idx < imgNum; idx++)
		{
			src_rgbir_rgb[idx] = imread(format("img/RGBIR/%d_rgb.png", idx), (color == 1) ? ColorOptionRGB : ColorOptionGRAY);
			src_rgbir_ir[idx] = imread(format("img/RGBIR/%d_ir.png", idx), ColorOptionGRAY);
		}
	}
	bool isShowAllImage = true;
	/*if (isShowAllImage)
	{
		Mat v;
		vector<Mat> s(imgNum);

		for (int idx = 0; idx < imgNum; idx++)
		{
			Mat c; cvtColor(src_rgbir_ir[idx], c, COLOR_GRAY2BGR);
			cp::dissolveSlideBlend(src_rgbir_rgb[idx], c, s[idx], 0.7,0.3);
		}
		cp::concat(s, v, 5, 2); imshow("dst", v); cv::waitKey();
	}*/

	vector<Mat> src_fnf_flash(imgNum);
	vector<Mat> src_fnf_noflash(imgNum);
	if (typeHDGF == FNF || isReadAll)
	{
		cout << "read flash noflash" << endl;
		for (int i = 0; i < imgNum; i++)
		{
			src_fnf_flash[i] = imread(format("img/flash/%dflash.png", i), (color == 1) ? ColorOptionRGB : ColorOptionGRAY);
			src_fnf_noflash[i] = imread(format("img/flash/%dnoflash.png", i), (color == 1) ? ColorOptionRGB : ColorOptionGRAY);
		}
	}
	/*if (isShowAllImage)
	{
		Mat v;
		vector<Mat> s(imgNum);

		for (int idx = 0; idx < imgNum; idx++)
		{
			cp::dissolveSlideBlend(src_fnf_noflash[idx], src_fnf_flash[idx], s[idx], 0.5,0.3);
		}
		cp::concat(s, v, 5, 2); imshow("dst", v); cv::waitKey();
	}*/

	vector<Mat> src_rgbd_rgb(imgNum);
	vector<Mat> src_rgbd_disp(imgNum);
	if (typeHDGF == RGBD || isReadAll)
	{
		cout << "read RGBD" << endl;
		for (int i = 0; i < imgNum; i++)
		{
			string s;
			if (i == 0)s = "Aloe";
			if (i == 1)s = "Baby1";
			if (i == 2)s = "Books";
			if (i == 3)s = "Bowling1";
			if (i == 4)s = "Cloth1";
			if (i == 5)s = "ConesH";
			if (i == 6)s = "Dolls";
			if (i == 7)s = "Laundry";
			if (i == 8)s = "Reindeer";
			if (i == 9)s = "TeddyH";

			src_rgbd_disp[i] = imread("img/RGBD/" + s + "_disp.png", ColorOptionGRAY);
			src_rgbd_rgb[i] = imread("img/RGBD/" + s + "_rgb.png", (color == 1) ? ColorOptionRGB : ColorOptionGRAY);
		}
	}
	/*if (isShowAllImage)
	{
		Mat v;
		vector<Mat> s(imgNum);

		for (int idx = 0; idx < imgNum; idx++)
		{
			Mat c; cvtColor(src_rgbd_disp[idx], c, COLOR_GRAY2BGR);
			cp::dissolveSlideBlend(src_rgbd_rgb[idx], c, s[idx], 0.7, 0.3);
		}
		cp::concat(s, v, 5, 2); imshow("dst", v); cv::waitKey();
	}*/
	vector<Mat> hsisrc(imgNum);
	vector<Mat> hsi32f;
	if (typeHDGF == HSI || isReadAll)
	{
		cout << "read HSI" << endl;
		{
			for (int n = 0; n < imgNum; n++)
			{
				vector<Mat> hsi;
				cout << n << " ";
				Mat a, b;
				int res = 2;
				double amp = 1.0 / res;
				for (int i = 1; i < 34; i++)
				{
					//\\fukushima-nas\share\2.�\�t�g�E�F�A�E�f�[�^�Z�b�g\3.�f�[�^�Z�b�g\hyperspectral\2002
					//\\fukushima - nas\share\2.�\�t�g�E�F�A�E�f�[�^�Z�b�g\3.�f�[�^�Z�b�g\hyperspectral\2004
					a = imread(format("img/hsi/%d_%03d.png", n, i), ColorOptionGRAY);
					hsi.push_back(a.clone());
				}
				merge(hsi, hsisrc[n]);
			}
			cout << endl;
		}
	}
	/*if (isShowAllImage)
	{
		Mat v;
		vector<Mat> s(imgNum);

		for (int idx = 0; idx < imgNum; idx++)
		{
			cp::cvtColorHSI2BGR(hsisrc[idx], s[idx]);
		}
		cp::concat(s, v, 5, 2); imshow("dst", v); cv::waitKey();
	}*/
	/*if (isShowAllImage)
	{
		Mat v;
		vector<Mat> vv;
		split(hsisrc[2], vv);
		cp::concat(vv, v, 11, 3); imshow("dst", v); cv::waitKey();
	}*/
#pragma endregion

#pragma region setup
#ifdef VIS_TILING_PCA
	int test = 0; createTrackbar("test", "", &test, 1);
#endif
	int src_num = 2; createTrackbar("src_num", wname2, &src_num, 9);
	int lambda = 100; createTrackbar("lambda", wname2, &lambda, 100);
	//convmethod = "interpolation NLM" sw=11;
	int sw1 = 2; createTrackbar("sw1", wname2, &sw1, 8); setTrackbarMin("sw1", wname2, -1);
	int sw2 = -1; createTrackbar("sw2", wname2, &sw2, 8); setTrackbarMin("sw2", wname2, -1);
	int sw3 = -1; createTrackbar("sw3", wname2, &sw3, 8); setTrackbarMin("sw3", wname2, -1);

	int showIndex = 0; createTrackbar("showIndex", wname2, &showIndex, 2);
	int alpha = 0; createTrackbar("a", wname2, &alpha, 100);
	int dboost = 5; createTrackbar("diff: 2^n", wname2, &dboost, 10);

	int sigma_space = 3; createTrackbar("ss", wname2, &sigma_space, 32);
	int sigma_range = 40; createTrackbar("sr", wname2, &sigma_range, 255);
	int tile_truncate_r = 10; createTrackbar("tile_truncate_r", wname2, &tile_truncate_r, 60);
	int gf_order = 2; createTrackbar("gf_order", wname2, &gf_order, 9); setTrackbarMin("gf_order", wname2, 1);
	int srcdownsample = 0; createTrackbar("src_downsample", wname2, &srcdownsample, 3);

	//int K_ = 5; createTrackbar("K", wname, &K_, 5000);
	int K_ = 10; createTrackbar("K", wname2, &K_, 1024); setTrackbarMin("K", wname2, 2);
	int km_iter = 5; createTrackbar("km iter", wname2, &km_iter, 100);
	setTrackbarMin("km iter", wname2, 1);
	int km_attempts = 1; createTrackbar("km attempts", wname2, &km_attempts, 5);
	setTrackbarMin("km attempts", wname2, 1);
	int km_sigma = 250; createTrackbar("km sigma", wname2, &km_sigma, 5000);

	//int cm = (int)ClusterMethod::random_sample;
	//int cm = (int)ClusterMethod::K_means; 
	//int cm = (int)ClusterMethod::K_means_pp;
	int cm = (int)ClusterMethod::K_means_pp_fast;
	//int cm = (int)ClusterMethod::KGaussInvMeansPPFast;
	//int cm = (int)ClusterMethod::mediancut_median;
	//int cm = (int)ClusterMethod::K_means_mediancut;
	//int cm = ClusterMethod::X_means; 
	createTrackbar("ClusterMethod", wname2, &cm, (int)ClusterMethod::Size - 1);
	int ds = 2; createTrackbar("downsample 2^n", wname2, &ds, 5);
	//int ds_method = INTER_NEAREST;
	int ds_method = DownsampleMethod::IMPORTANCE_MAP;
	createTrackbar("downsample method", wname2, &ds_method, DownsampleMethod::DownsampleMethodSize - 1);
	int crop = 1; createTrackbar("isCropClustering", wname2, &crop, 1);
	int tilex = 3; createTrackbar("tilex", wname2, &tilex, 5);
	int tiley = 3; createTrackbar("tiley", wname2, &tiley, 5);
	int softlambda = 50; createTrackbar("soft:lambda*0.001", wname2, &softlambda, 2000);
	int localmu = 1; createTrackbar("isLocalMu", wname2, &localmu, 1);
	int localsp = 0; createTrackbar("isLocalSP", wname2, &localsp, 1);
	int localsp_delta = 0; createTrackbar("LocalSP delta", wname2, &localsp_delta, 1000);
	int nlm_r = 1;
	int max_dim = (2 * nlm_r + 1) * (2 * nlm_r + 1) * src[0].channels();
	int pca_channel = 10; createTrackbar("pca_ch", wname2, &pca_channel, max_dim); setTrackbarMin("pca_ch", wname2, 1);
	int pca_method1 = 0; createTrackbar("pca_method1", wname2, &pca_method1, (int)NeighborhoodPCA::SIZE - 1);

	int border = cv::BORDER_DEFAULT; createTrackbar("border", wname2, &border, 4);
	cp::SpatialFilterAlgorithm gf_method = cp::SpatialFilterAlgorithm::SlidingDCT5_AVX;
	//cp::SpatialFilterAlgorithm gf_method = cp::SpatialFilterAlgorithm::FIR_OPENCV;

	int depth = CV_32F;
	bool isDownsampleClustering = true;
	//bool isDownsampleClustering = false;

	Mat dst, dst1, dst2, dst3;
	Mat show;
	Mat src32;
	Mat guide32f;
	Mat ref8u;
	Mat ref;

	cp::UpdateCheck ucRecomputeRef(sigma_range, sigma_space, src_num, typeHDGF, pca_channel, pca_method1, lambda, tilex, tiley, clusteringHDGFMethod, border);
	cp::UpdateCheck uc2(sw1, sw2, sw3, cm, ds, K_, ds_method, tile_truncate_r, crop, gf_order, srcdownsample);
	cp::UpdateCheck uc3(km_attempts, km_sigma, km_iter, localmu, localsp, localsp_delta, softlambda);
	//cp::UpdateCheck uc4(lambda, delta, localmu, localsp);
	cp::ConsoleImage ci(Size(640, 800), "Info");

	cp::Timer tpca("time", cp::TIME_MSEC, false);
	cp::Timer t1("time", cp::TIME_MSEC, false);
	cp::Timer t2("time", cp::TIME_MSEC, false);
	cp::Timer t3("time", cp::TIME_MSEC, false);
	cp::Stat psnr1;
	cp::Stat psnr2;
	cp::Stat psnr3;
	cp::Stat psnrsrc;

	vector<cp::Stat> psnr_tile1(maxdivision.area());
	vector<cp::Stat> psnr_tile2(maxdivision.area());
	//omp_set_num_threads(1);
	Mat input32f;
	Mat guide4Filter;
	int key = 0;
	//const int methodNum = 1;
	//const int methodNum = 2;
	const int methodNum = 3;
	double timeref = 0.0;
	double timeref_fullpca = 0.0;
	double timeref_tilepca = 0.0;
	double psnr_fullpca = 0.0;
	double psnr_tilepca = 0.0;
	double timeref_tilepca2 = 0.0;
	double timeref_tiledct = 0.0;
	double psnr_tilepca2 = 0.0;
	double psnr_tiledct = 0.0;
	double psnrpcaonly = 0.0;

	TileHDGF hdgf(maxdivision);

	ConstantTimeHDGF CHDGF = ConstantTimeHDGF::Nystrom;
	Ptr<TileConstantTimeHDGF> tHDGF1 = new TileConstantTimeHDGF(maxdivision, ConstantTimeHDGF(clusteringHDGFMethod));
	Ptr<TileConstantTimeHDGF> tHDGF2 = new TileConstantTimeHDGF(maxdivision, ConstantTimeHDGF(clusteringHDGFMethod));
	Ptr<TileConstantTimeHDGF> tHDGF3 = new TileConstantTimeHDGF(maxdivision, ConstantTimeHDGF(clusteringHDGFMethod));

#pragma endregion
	while (key != 'q')
	{
		bool isClear = false;

		if (ucRecomputeRef.isUpdate(sigma_range, sigma_space, src_num, typeHDGF, pca_channel, pca_method1, lambda, tilex, tiley, clusteringHDGFMethod, border))
		{
			int d_ref = 2 * (ceil(sigma_space * 3.0)) + 1;
			maxdivision = Size(pow(2, tilex), pow(2, tiley));
			tHDGF1 = new TileConstantTimeHDGF(maxdivision, ConstantTimeHDGF(clusteringHDGFMethod));
			tHDGF2 = new TileConstantTimeHDGF(maxdivision, ConstantTimeHDGF(clusteringHDGFMethod));
			tHDGF3 = new TileConstantTimeHDGF(maxdivision, ConstantTimeHDGF(clusteringHDGFMethod));

			dst.release();
			Mat d32;
			string method = (typeHDGF == 0) ? "naive BF" : "naive NLM";
			cp::Timer t(method, cp::TIME_MSEC, false);
			if (typeHDGF == RGB)
			{
				src[src_num].convertTo(src32, CV_32F);
				src[src_num].convertTo(guide32f, CV_32F);
			}
			/*else if (methodHDGF == GIR)
			{
				src_gir_g[src_num].convertTo(src32, CV_32F);
				mergeConvertImage(src_gir_g[src_num], src_gir_ir[src_num], guide32f, lambda * 0.01, CV_32F);
				t.start();
				cp::highDimensionalGaussianFilter(src32, guide32f, d32, Size(d, d), sigma_range, sigma_space, border);
				d32.convertTo(ref, CV_64F);
			}*/
			else if (typeHDGF == RGBIR)
			{
				src_rgbir_rgb[src_num].convertTo(src32, CV_32F);
				mergeConvertImage(src_rgbir_rgb[src_num], src_rgbir_ir[src_num], guide32f, lambda * 0.01, CV_32F);
			}
			else if (typeHDGF == RGBD)
			{
				src_rgbd_disp[src_num].convertTo(src32, CV_32F);
				mergeConvertImage(src_rgbd_rgb[src_num], src_rgbd_disp[src_num], guide32f, lambda * 0.01, CV_32F);
			}
			else if (typeHDGF == FNF)
			{
				src_fnf_noflash[src_num].convertTo(src32, CV_32F);
				mergeConvertImage(src_fnf_flash[src_num], src_fnf_noflash[src_num], guide32f, lambda * 0.01, CV_32F);
			}
			else if (typeHDGF == HSI)
			{
				hsisrc[src_num].convertTo(src32, CV_32F);
				hsisrc[src_num].convertTo(guide32f, CV_32F);
				split(guide32f, hsi32f);
			}
			else if (typeHDGF == NLM)
			{
				src[src_num].convertTo(src32, CV_32F);
				cp::IM2COL(src32, guide32f, nlm_r, border);
			}

			for (int i = 0; i < iterationRef; i++)
			{
				t.start();
				cp::highDimensionalGaussianFilter(src32, guide32f, d32, Size(d_ref, d_ref), sigma_range, sigma_space, border);
				//cp::bilateralFilterL2(src32, d32, d / 2, sigma_range, sigma_space, border);
				t.pushLapTime();
			}
			timeref = t.getLapTimeMedian();
			d32.convertTo(ref, CV_64F);

			Mat hguide;
			Mat pca;
			if (isRefFullPCA)
			{
				for (int i = 0; i < iterationRef; i++)
				{
					t.start();
					if (typeHDGF == NLM)
					{
						patchPCA(src32, hguide, nlm_r, pca_channel, border, pca_method1);
						cp::highDimensionalGaussianFilter(src32, hguide, pca, Size(d_ref, d_ref), sigma_range, sigma_space, border, cp::HDGFSchedule::COMPUTE);
						pca.convertTo(pca, CV_64F);
					}
					else
					{
						cp::cvtColorPCA(guide32f, hguide, pca_channel);
						cp::highDimensionalGaussianFilter(src32, hguide, pca, Size(d_ref, d_ref), sigma_range, sigma_space, border, cp::HDGFSchedule::COMPUTE);
						//cp::highDimensionalGaussianFilter(src32, hguide, pca, Size(d_ref, d_ref), sigma_range, sigma_space, border, cp::HDGFSchedule::LUT_SQRT);
						pca.convertTo(pca, CV_64F);
					}
					t.pushLapTime();
				}
				timeref_fullpca = t.getLapTimeMedian();
				psnr_fullpca = cp::getPSNR(pca, ref);
			}
			if (isRefTilePCA)
			{
				t.start();
				if (typeHDGF == NLM)
				{
					cp::IM2COL(src32, hguide, nlm_r, border);
					highDimensionalGaussianFilterTilePCA(src32, hguide, pca, Size(d_ref, d_ref), sigma_range, sigma_space, pca_channel, maxdivision, border);
					pca.convertTo(pca, CV_64F);
				}
				else
				{
					highDimensionalGaussianFilterTilePCA(src32, guide32f, pca, Size(d_ref, d_ref), sigma_range, sigma_space, pca_channel, maxdivision, border);
					pca.convertTo(pca, CV_64F);
				}
				timeref_tilepca = t.getTime();
				psnr_tilepca = cp::getPSNR(pca, ref);
			}

			if (typeHDGF == NLM)
			{
				cp::IM2COL(src32, hguide, nlm_r, border);
				psnrpcaonly = cp::cvtColorPCAErrorPSNR(hguide, pca_channel);
			}
			else if (typeHDGF == RGB)
			{
				psnrpcaonly = cp::cvtColorPCAErrorPSNR(src32, pca_channel);
			}
			else
			{
				//psnrpcaonly = cp::cvtColorPCAErrorPSNR(guide32f, pca_channel, true, 8);
				psnrpcaonly = cp::cvtColorPCAErrorPSNR(guide32f, pca_channel);
			}

			isClear = true;
		}

		if (uc2.isUpdate(sw1, sw2, sw3, cm, ds, K_, ds_method, tile_truncate_r, crop, gf_order, srcdownsample) ||
			uc3.isUpdate(km_attempts, km_sigma, km_iter, localmu, localsp, localsp_delta, softlambda)// ||
			//uc4.isUpdate(lambda, delta,localsp)
			)
		{
			isClear = true;
		}

		if (key == 'r' || isClear)
		{
			t1.clearStat();
			t2.clearStat();
			t3.clearStat();
			tpca.clearStat();
			psnr1.clear();
			psnr2.clear();
			psnr3.clear();
			psnrsrc.clear();
			for (int i = 0; i < maxdivision.area(); i++)
			{
				psnr_tile1[i].clear();
				psnr_tile2[i].clear();
			}
		}

		const bool isAddNoise = false;
		if (isAddNoise)
		{
			cp::addNoise(src32, input32f, 15);
		}
		else
		{
			src32.copyTo(input32f);
		}

		string convmethod1 = "";
		string convmethod2 = "";
		string convmethod3 = "";
		for (int n = 0; n < methodNum; n++)
		{
			Ptr<TileConstantTimeHDGF> tHDGF = (n == 0) ? tHDGF1 : (n == 1) ? tHDGF2 : tHDGF3;
			tHDGF->setLambdaInterpolation(softlambda * 0.001);
			tHDGF->setIsUseLocalMu(localmu == 1);
			tHDGF->setIsUseLocalStatisticsPrior(localsp == 1);
			tHDGF->setDeltaLocalStatisticsPrior(localsp_delta * 0.1);
			tHDGF->setKMeansAttempts(km_attempts);
			tHDGF->setNumIterations(km_iter);
			//std::cout << "km sigma" << km_sigma << std::endl;
			tHDGF->setDownsampleImageSize(pow(2, srcdownsample));
			tHDGF->setKMeansSigma(km_sigma * 0.1);
			tHDGF->setKMeansSignalMax(255.0 * sqrt(3.0));
			tHDGF->setCropClustering((bool)crop);
			tHDGF->setPatchPCAMethod(pca_method1);

			string convmethod = "";
			dst.setTo(0);
			if (n == 0) t1.start();
			if (n == 1) t2.start();
			if (n == 2) t3.start();

			int sw = (n == 0) ? sw1 : (n == 1) ? sw2 : sw3;
			int pca_method = pca_method1;

			if (sw == -1)
			{
				convmethod = "copy";
				input32f.copyTo(dst);
			}
			else if (sw == 0)
			{
				if (typeHDGF == NLM)
				{
					convmethod = "Full     NLM";

					//add(input32f, 1.f, input32f);				 
					cp::IM2COL(input32f, guide4Filter, nlm_r, border);
					double minv, maxv;
					//cv::minMaxLoc(input32f, &minv, &maxv);
					//print_debug2(minv, maxv);
					//cv::minMaxLoc(guide4Filter, &minv, &maxv);
					//print_debug2(minv, maxv);
					tHDGF->jointfilter(input32f, guide4Filter, dst, sigma_space, sigma_range, (ClusterMethod)cm, K_, gf_method, gf_order, depth, isDownsampleClustering, pow(2, ds), ds_method, tile_truncate_r * 0.1, border);
				}
				else
				{
					convmethod = "Full     JBF";
					tHDGF->jointfilter(input32f, guide32f, dst, sigma_space, sigma_range, (ClusterMethod)cm, K_, gf_method, gf_order, depth, isDownsampleClustering, pow(2, ds), ds_method, tile_truncate_r * 0.1, border);
					//tHDGF->filter(input32f,  dst, sigma_space, sigma_range, (ClusterMethod)cm, K_, gf_method, gf_order, depth, isDownsampleClustering, pow(2, ds), ds_method, truncate_r * 0.1);
				}
			}
			else if (sw == 1)
			{
				if (typeHDGF == NLM)
				{
					convmethod = "Full PCA NLM";
					{
						//cp::Timer t("PCA");
						Mat b;
						//GaussianBlur(input32f, b, Size(2 * br + 1, 2 * br + 1), br / 3.0);
						//convertNeighborhoodToChannelsPCA(b, guide32f, nlm_r, pca_channel, border, pca_method);
						//cp::cvtColorAverageGray(input32f, b, true);
						//convertNeighborhoodToChannelsPCA(b, guide32f, nlm_r, pca_channel, border, pca_method);
						patchPCA(input32f, guide4Filter, nlm_r, pca_channel, border, pca_method);
					}
					//convertNeighborhoodToChannelsPCA(input32f, guide4Filter, nlm_r, pca_channel, border, pca_method);
					//cp::imshowSplitScale("tilePCA", guide4Filter, 0.1);
					tHDGF->jointfilter(input32f, guide4Filter, dst, sigma_space, sigma_range, (ClusterMethod)cm, K_, gf_method, gf_order, depth, isDownsampleClustering, pow(2, ds), ds_method, tile_truncate_r * 0.1, border);
				}
				else
				{
					convmethod = "Full PCA JBF";
#ifdef VIS_TILING_PCA
					if (test == 0)
					{
						cp::cvtColorPCA(guide32f, guide4Filter, pca_channel);
						cp::imshowSplitScale("fullPCA", guide4Filter);
					}
					else
					{
						cvtColorPCATile(guide32f, guide4Filter, pca_channel, division);
						cp::imshowSplitScale("tilePCA", guide4Filter);
					}
#else
					cp::cvtColorPCA(guide32f, guide4Filter, pca_channel);
#endif
					tHDGF->jointfilter(input32f, guide4Filter, dst, sigma_space, sigma_range, (ClusterMethod)cm, K_, gf_method, gf_order, depth, isDownsampleClustering, pow(2, ds), ds_method, tile_truncate_r * 0.1, border);
				}
			}
			else if (sw == 2)
			{
				if (typeHDGF == NLM)
				{
					convmethod = "Tile PCA NLM";
					tHDGF->nlmfilter(input32f, input32f, dst, sigma_space, sigma_range, nlm_r, pca_channel, (ClusterMethod)cm, K_, gf_method, gf_order, depth, isDownsampleClustering, pow(2, ds), ds_method, tile_truncate_r * 0.1, border);
				}
				else if (typeHDGF == HSI)
				{
					convmethod = "Tile PCA JBF";
					//tnystrom.jointPCAfilter(input32f, input32f, dst, sigma_space, sigma_range, min(pca_channel, 33), (ClusterMethod)cm, K_, gf_method, gf_order, depth, isDownsampleClustering, pow(2, ds), ds_method, truncate_r * 0.1);
					tHDGF->jointPCAfilter(hsi32f, hsi32f, min(pca_channel, 33), dst, sigma_space, sigma_range, (ClusterMethod)cm, K_, gf_method, gf_order, depth, isDownsampleClustering, pow(2, ds), ds_method, tile_truncate_r * 0.1, border);
					//tnystrom.jointPCAfilter(input32f, guide32f, dst, sigma_space, sigma_range, pca_channel, (ClusterMethod)cm, K_, gf_method, gf_order, depth, isDownsampleClustering, pow(2, ds), ds_method, truncate_r * 0.1);
				}
				else
				{
					convmethod = "Tile PCA JBF";
					//print_matinfo(input32f);
					tHDGF->jointPCAfilter(input32f, guide32f, pca_channel, dst, sigma_space, sigma_range, (ClusterMethod)cm, K_, gf_method, gf_order, depth, isDownsampleClustering, pow(2, ds), ds_method, tile_truncate_r * 0.1, border);
				}
			}
			else if (sw == 3)
			{
				if (typeHDGF == NLM)
				{
					convmethod = "Full PermutohedralLattice NLM";
					cp::IM2COL(input32f, guide4Filter, nlm_r, border);
					cp::highDimensionalGaussianFilterPermutohedralLatticeTile(input32f, guide4Filter, dst, sigma_range, sigma_space, maxdivision, tile_truncate_r * 0.1);
				}
				else
				{
					convmethod = "Full PermutohedralLattice";
					cp::highDimensionalGaussianFilterPermutohedralLatticeTile(input32f, guide32f, dst, sigma_range, sigma_space, maxdivision, tile_truncate_r * 0.1);
				}
			}
			else if (sw == 4)
			{
				if (typeHDGF == NLM)
				{
					convmethod = "Full PCA PermutohedralLattice NLM";
					patchPCA(input32f, guide4Filter, nlm_r, pca_channel, border, pca_method);
					cp::highDimensionalGaussianFilterPermutohedralLatticeTile(input32f, guide4Filter, dst, sigma_range, sigma_space, maxdivision, tile_truncate_r * 0.1);
				}
				else
				{
					convmethod = "Full PCA PermutohedralLattice";
					cp::cvtColorPCA(guide32f, guide4Filter, pca_channel);
					cp::highDimensionalGaussianFilterPermutohedralLatticeTile(input32f, guide4Filter, dst, sigma_range, sigma_space, maxdivision, tile_truncate_r * 0.1);
				}
			}
			else if (sw == 5)
			{
				if (typeHDGF == NLM)
				{
					convmethod = "Tile PCA PermutohedralLattice NLM";
					patchPCATile(input32f, guide4Filter, nlm_r, pca_channel, border, pca_method, maxdivision);
					cp::highDimensionalGaussianFilterPermutohedralLatticeTile(input32f, guide4Filter, dst, sigma_range, sigma_space, maxdivision, tile_truncate_r * 0.1);
				}
				else
				{
					convmethod = "Tile PCA PermutohedralLattice";
					//print_matinfo(guide32f);
					cp::highDimensionalGaussianFilterPermutohedralLatticePCATile(input32f, guide32f, dst, sigma_range, sigma_space, pca_channel, maxdivision, tile_truncate_r * 0.1);
				}
			}
			else if (sw == 6)
			{
				if (typeHDGF == NLM)
				{
					convmethod = "Full GaussianKD-Tree NLM";
					cp::IM2COL(input32f, guide4Filter, nlm_r, border);
					cp::highDimensionalGaussianFilterGaussianKDTreeTile(input32f, guide4Filter, dst, sigma_range, sigma_space, maxdivision, tile_truncate_r * 0.1);
				}
				else
				{
					convmethod = "Full GaussianKD-Tree";
					cp::highDimensionalGaussianFilterGaussianKDTreeTile(input32f, guide32f, dst, sigma_range, sigma_space, maxdivision, tile_truncate_r * 0.1);
				}
			}
			else if (sw == 7)
			{
				if (typeHDGF == NLM)
				{
					convmethod = "Full PCA GaussianKD-Tree NLM";
					patchPCA(input32f, guide4Filter, nlm_r, pca_channel, border, pca_method);
					cp::highDimensionalGaussianFilterGaussianKDTreeTile(input32f, guide4Filter, dst, sigma_range, sigma_space, maxdivision, tile_truncate_r * 0.1);
				}
				else
				{
					convmethod = "Full PCA GaussianKD-Tree";
					cp::cvtColorPCA(guide32f, guide4Filter, pca_channel);
					cp::highDimensionalGaussianFilterGaussianKDTreeTile(input32f, guide4Filter, dst, sigma_range, sigma_space, maxdivision, tile_truncate_r * 0.1);
				}
			}
			else if (sw == 8)
			{
				if (typeHDGF == NLM)
				{
					convmethod = "Tile PCA GaussianKD-Tree NLM";
					patchPCATile(input32f, guide4Filter, nlm_r, pca_channel, border, pca_method, maxdivision);
					cp::highDimensionalGaussianFilterGaussianKDTreeTile(input32f, guide4Filter, dst, sigma_range, sigma_space, maxdivision, tile_truncate_r * 0.1);
				}
				else
				{
					convmethod = "Tile PCA GaussianKD-Tree";
					cp::highDimensionalGaussianFilterGaussianKDTreePCATile(input32f, guide32f, dst, sigma_range, sigma_space, pca_channel, maxdivision, tile_truncate_r * 0.1);
				}
			}

			convmethod += to_string(sw);
			if (n == 0)
			{
				t1.getpushLapTime();
				convmethod1 = convmethod;
				dst.copyTo(dst1);
			}
			else if (n == 1)
			{
				t2.getpushLapTime();
				convmethod2 = convmethod;
				dst.copyTo(dst2);
			}
			else if (n == 2)
			{
				t3.getpushLapTime();
				convmethod3 = convmethod;
				dst.copyTo(dst3);
			}

			//tHDGF->getEigenValueInfo();
		}

		Mat v = dst1;
		if (showIndex == 1) v = dst2;
		if (showIndex == 2) v = dst3;
#pragma region imshow
		if (typeHDGF == HSI)
		{
			Mat ss, ds;
			cp::cvtColorHSI2BGR(ref, ss);
			cp::cvtColorHSI2BGR(v, ds);
			imshow(wname, ss);

			cp::alphaBlend(ss, ds, alpha * 0.01, show);
			if (key == 'a')cp::guiAlphaBlend(ss, ds);
			imshow(wname, show);
			if (key == 'c')cp::guiCropZoom(show);
		}
		else
		{
			if (dboost != 0)
			{
				cp::diffshow(wname, ref, v, pow(2, dboost));
			}
			else
			{
				ref.convertTo(ref8u, CV_8U);
				v.convertTo(show, CV_8U);
				cp::alphaBlend(ref8u, show, alpha * 0.01, show);
				imshow(wname, show);
				if (key == 'c')
				{
					if (show.channels() == 1)
					{
						Mat cc; cvtColor(show, cc, COLOR_GRAY2BGR);
						cp::guiCropZoom(cc);
					}
					else cp::guiCropZoom(show);
				}
			}
		}

		key = waitKey(1);
#pragma endregion

		if (methodNum > 0)psnr1.push_back(cp::getPSNR(dst1, ref, 16));
		if (methodNum > 1)psnr2.push_back(cp::getPSNR(dst2, ref, 16));
		if (methodNum > 2)psnr3.push_back(cp::getPSNR(dst3, ref, 16));
		if (key == 'a')cp::guiAlphaBlend(dst1, ref);
		ci(getHDGFTypeName(typeHDGF));
		ci(getclusteringHDGFMethodName(clusteringHDGFMethod));
		ci("Size %dx%d: Tile %dx%d ,div (%d,%d)", src32.cols, src32.rows, src32.cols / maxdivision.width, src32.rows / maxdivision.height, maxdivision.width, maxdivision.height);
		ci("TileB %dx%d", tHDGF1->getTileSize().width, tHDGF1->getTileSize().height);
		ci("%d| space %d range %d", t1.getStatSize(), sigma_space, sigma_range);
		ci("clustering: " + getClusterMethodName(ClusterMethod(cm)) + format(": K= %d (%3.1f), iter=%d", K_, pow(double(K_), 1.0 / 3.0), km_iter));
		ci("downsample: " + getDownsampleMethodName(ds_method) + format(" size %d", (int)pow(2, ds)));
		ci("1: " + convmethod1);
		ci("2: " + convmethod2);
		ci("3: " + convmethod3);

		//ci("tpca  %f ms", tpca.getLapTimeMean());
		ci("naive full  %7.2f ms - dB", timeref);
		if (isRefFullPCA)ci("naive f-PCA %7.2f ms %5.2f dB", timeref_fullpca, psnr_fullpca);
		if (isRefTilePCA)ci("naive t-PCA %7.2f ms %5.2f dB", timeref_tilepca, psnr_tilepca);
		if (isRefTilePCA2)ci("naive t-PCA2%7.2f ms %5.2f dB", timeref_tilepca2, psnr_tilepca2);
		if (isRefTileDCT)ci("naive t-DCT %7.2f ms %5.2f dB", timeref_tiledct, psnr_tiledct);
		ci("PCA PSNR    %5.2f dB ", psnrpcaonly);
		ci("time   1    Mean %7.2f MED %7.2f ms", t1.getLapTimeMean(), t1.getLapTimeMedian());
		ci("time   2    Mean %7.2f MED %7.2f ms", t2.getLapTimeMean(), t2.getLapTimeMedian());
		ci("time   3    Mean %7.2f MED %7.2f ms", t3.getLapTimeMean(), t3.getLapTimeMedian());
		ci(format("PSNR %5.2f dB %5.2f dB %5.2f-%5.2f (%5.2f)", psnr1.getMedian(), psnr1.getMean(), psnr1.getMin(), psnr1.getMax(), psnr1.getStd()));
		ci(format("PSNR %5.2f dB %5.2f dB %5.2f-%5.2f (%5.2f)", psnr2.getMedian(), psnr2.getMean(), psnr2.getMin(), psnr2.getMax(), psnr2.getStd()));
		ci(format("PSNR %5.2f dB %5.2f dB %5.2f-%5.2f (%5.2f)", psnr3.getMedian(), psnr3.getMean(), psnr3.getMin(), psnr3.getMax(), psnr3.getStd()));
		ci(format("PSNR-src %5.3f dB %5.3f-%5.3f (%5.3f)", psnrsrc.getMean(), psnrsrc.getMin(), psnrsrc.getMax(), psnrsrc.getStd()));
		/*for (int i = 0; i < division.area(); i++)
		{
			double psnr1 = psnr_tile1[i].getMedian();
			double psnr2 = psnr_tile2[i].getMedian();
			ci(format("%2d| %5.3f, %5.3f (%6.3f)", i, psnr1, psnr2, psnr2 - psnr1));
		}*/
		if (key == 't') ci.push();

		ci.show();
		//psnr.drawDistribution("psnr");
		//t.drawDistribution("time");
	}
}

int main()
{
	testClusteringHDGF_SpringerNature("Test ClusteringHDGF NSCS");
}
