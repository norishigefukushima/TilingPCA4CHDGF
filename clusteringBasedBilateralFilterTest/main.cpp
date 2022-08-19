#include <opencp.hpp>

#include "common.hpp"
#include "HDGF.hpp"
#include "patchPCA.hpp"
#include <stdio.h>
#include <fstream>

using namespace std;
using namespace cv;
#pragma comment(lib,"../x64/Release/clusteringHDGF.lib")
#if _DEBUG
#pragma comment(lib,"opencpd.lib")
#else
#pragma comment(lib,"opencp.lib")
#pragma comment(lib,"SpatialFilter.lib")
#endif


void testClusteringCBF_SpringerNature(string wname = "ClusteringCBF SN");
int main()
{
	testClusteringCBF_SpringerNature();
}
