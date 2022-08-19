#pragma once
#include <opencv2/opencv.hpp>
#include <intrin.h>
//ã´äEèàóù
//#define USE_BORDER_REPLICATE 1 //BORDER_REPLICATE   = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
#define USE_BORDER_REFLECT 1 //BORDER_REFLECT     = 2, //!< `fedcba|abcdefgh|hgfedcb`
//#define USE_BORDER_REFLECT_101 1 //BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`

		//extrapolation functions
		//(atE() and atS() require variables w and h in their scope respectively)
#ifdef USE_BORDER_REPLICATE 
#define LREF(n) (std::max(n,0))
#define RREF(n) (std::min(n,width-1))
#define UREF(n) (std::max(n,0) * width)
#define DREF(n) (std::min(n,height-1) * width)
#elif USE_BORDER_REFLECT 
//#define REFLECT(x,y) ((y < 0 ? abs(y + 1) : (height <= y ? 2*width - (y) - 1: y)) * width + (x < 0 ? abs(x + 1) : (width <= x ? 2*width - (x) - 1: x)))
#define LREF(n) (n < 0 ? - (n) - 1: n)
#define RREF(n) (n < width ? n: 2*width - (n) - 1)
#define UREF(n) ((n < 0 ? - (n) - 1: n) * width)
#define DREF(n) ((n < height ? n : 2*height - (n) - 1) * width)
//#define RREF(n) ((width - 1) - std::abs(int(int(std::abs(n+0.5f))%(2*width) - ((2*width - 1)/ 2.f))))
//#define LREF(n) ((width - 1) - std::abs(int(int(std::abs(n+0.5f))%(2*width) - ((2*width - 1)/ 2.f))))
//#define UREF(n) (((height - 1) - std::abs(int(int(std::abs(n+0.5f))%(2*height) - ((2*height - 1)/ 2.f))))*width)
//#define DREF(n) (((height - 1) - std::abs(int(int(std::abs(n+0.5f))%(2*height) - ((2*height - 1)/ 2.f))))*width)
#elif USE_BORDER_REFLECT_101
#define LREF(n) (std::abs(n))
#define RREF(n) (width-1-std::abs(width-1-(n)))
#define UREF(n) (std::abs(n) * width)
#define DREF(n) ((height-1-std::abs(height-1-(n))) * width)
#endif

////FFTW
//#pragma comment(lib, "libfftw3-3.lib")
//#pragma comment(lib, "libfftw3f-3.lib")
//#pragma comment(lib, "libfftw3l-3.lib")


//VYV
#define VYV_NUM_NEWTON_ITERATIONS       6
#define VYV_ORDER_MAX 5
#define VYV_ORDER_MIN 3
#define VYV_VALID_ORDER(K)  (VYV_ORDER_MIN <= (K) && (K) <= VYV_ORDER_MAX)

//Deriche
#define DERICHE_ORDER_MIN       2
#define DERICHE_ORDER_MAX       4
#define DERICHE_VALID_ORDER(K)  (DERICHE_ORDER_MIN <= (K) && (K) <= DERICHE_ORDER_MAX)

#define COLOR_WHITE cv::Scalar(255,255,255)
#define COLOR_GRAY10 cv::Scalar(10,10,10)
#define COLOR_GRAY20 cv::Scalar(20,20,20)
#define COLOR_GRAY30 cv::Scalar(10,30,30)
#define COLOR_GRAY40 cv::Scalar(40,40,40)
#define COLOR_GRAY50 cv::Scalar(50,50,50)
#define COLOR_GRAY60 cv::Scalar(60,60,60)
#define COLOR_GRAY70 cv::Scalar(70,70,70)
#define COLOR_GRAY80 cv::Scalar(80,80,80)
#define COLOR_GRAY90 cv::Scalar(90,90,90)
#define COLOR_GRAY100 cv::Scalar(100,100,100)
#define COLOR_GRAY110 cv::Scalar(101,110,110)
#define COLOR_GRAY120 cv::Scalar(120,120,120)
#define COLOR_GRAY130 cv::Scalar(130,130,140)
#define COLOR_GRAY140 cv::Scalar(140,140,140)
#define COLOR_GRAY150 cv::Scalar(150,150,150)
#define COLOR_GRAY160 cv::Scalar(160,160,160)
#define COLOR_GRAY170 cv::Scalar(170,170,170)
#define COLOR_GRAY180 cv::Scalar(180,180,180)
#define COLOR_GRAY190 cv::Scalar(190,190,190)
#define COLOR_GRAY200 cv::Scalar(200,200,200)
#define COLOR_GRAY210 cv::Scalar(210,210,210)
#define COLOR_GRAY220 cv::Scalar(220,220,220)
#define COLOR_GRAY230 cv::Scalar(230,230,230)
#define COLOR_GRAY240 cv::Scalar(240,240,240)
#define COLOR_GRAY250 cv::Scalar(250,250,250)
#define COLOR_BLACK cv::Scalar(0,0,0)

#define COLOR_RED cv::Scalar(0,0,255)
#define COLOR_GREEN cv::Scalar(0,255,0)
#define COLOR_BLUE cv::Scalar(255,0,0)
#define COLOR_ORANGE cv::Scalar(0,100,255)
#define COLOR_YELLOW cv::Scalar(0,255,255)
#define COLOR_MAGENDA cv::Scalar(255,0,255)
#define COLOR_CYAN cv::Scalar(255,255,0)

//utility function
namespace util
{
	void cvt32F8U(const cv::Mat& src, cv::Mat& dest);
	void cvt32F16F(cv::Mat& srcdst);
	double Combination(const int N, const int n);

	void bilateralFilter32f(const cv::Mat& src, cv::Mat& dst, const cv::Size kernelSize, float sigma_range, float sigma_space, const int borderType, const bool isRectangle, const bool isKahan);
	void bilateralFilter64f(const cv::Mat& src, cv::Mat& dst, const cv::Size kernelSize, double sigma_range, double sigma_space, const int borderType, const bool isRectangle, const bool isKahan);
	void bilateralFilter64f_Laplacian(const cv::Mat& src, cv::Mat& dst, const cv::Size kernelSize, double sigma_range, double sigma_space, const int borderType, const bool isRectangle);
	void bilateralFilter64f_Hat(const cv::Mat& src, cv::Mat& dst, const cv::Size kernelSize, double sigma_range, double sigma_space, const int borderType, const bool isRectangle);

	template<typename T>
	inline int typeToDepth();

	void calcMaxDiffParallel(const cv::Mat& src, cv::Mat& buff, const int r, int& T);
	int calcMaxDiff(const cv::Mat& src, cv::Mat& buff, const int r);
	int calcMaxDiffV(const cv::Mat& src, const int rad);

	void blockAnalysis(cv::Mat& reference, cv::Mat& ideal, cv::Mat& filtered, cv::Size div, const int r);
	void copyMakeBorderInteral(cv::Mat& src, cv::Mat& dest, int top, int bottom, int left, int right, int borderType);
	void splitBGRLineInterleaveAVX(cv::InputArray src, cv::OutputArray dest);
}