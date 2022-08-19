#pragma once

//please change this options if you want.
#define USE_SSE			//providing SSE computation
#define USE_OPENCV2		//providing OpenCV2 interface

const double PI=3.1415926535897932384626433832795;

//#define USE_BORDER_REPLICATE 1	 //BORDER_REPLICATE   = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
//#define USE_BORDER_REFLECT 1	 //BORDER_REFLECT     = 2, //!< `fedcba|abcdefgh|hgfedcb`
#define USE_BORDER_REFLECT_101 1//BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`

//extrapolation functions
//(atE() and atS() require variables w and h in their scope respectively)
#ifdef USE_BORDER_REPLICATE 
#define atW(x) (std::max(x,0))
#define atN(y) (std::max(y,0))
#define atE(x) (std::min(x,w-1))
#define atS(y) (std::min(y,h-1))
#elif USE_BORDER_REFLECT 
#define atW(x) ( x < 0 ? std::abs(x+1) : std::abs(x))
#define atN(y) ( y < 0 ? std::abs(y+1) : std::abs(y))
#define atE(x) ( x < w ? w-1-std::abs(w-1-(x)) : w-std::abs(w-1-(x)))
#define atS(y) ( y < h ? h-1-std::abs(h-1-(y)) : h-std::abs(h-1-(y)))
#elif USE_BORDER_REFLECT_101
#define atW(x) (std::abs(x))
#define atN(y) (std::abs(y))
#define atE(x) (w-1-std::abs(w-1-(x)))
#define atS(y) (h-1-std::abs(h-1-(y)))
#endif


//boundary processing
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
#define VYV_MAX_ORDER 5
#define VYV_MIN_ORDER 3
#define VYV_VALID_ORDER(K)  (VYV_MIN_ORDER <= (K) && (K) <= VYV_MAX_ORDER)

//Deriche
#define DERICHE_MIN_ORDER       2
#define DERICHE_MAX_ORDER       4
#define DERICHE_VALID_ORDER(K)  (DERICHE_MIN_ORDER <= (K) && (K) <= DERICHE_MAX_ORDER)

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