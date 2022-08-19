#pragma once

#include <opencv2/opencv.hpp>
#define __AVX2__ 1
#define __FMA__ 1
#define __AVX__ 1

#include "inlineSIMDFunctions.hpp"

#include "common.hpp"
#include "avx_util.hpp"
#include "GaussianFilter.hpp"

//#include <opencp.hpp>
//#include "ConstantTimeBFColor_NormalForm_template.hpp"
