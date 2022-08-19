#include "common.hpp"
#include "bilateral.hpp"

using namespace std;
using namespace cv;

int main()
{
	
	Mat input_image = imread("lenna.png");
	Mat output_image(input_image.size(), CV_32F);

	// parameter
	double ss = 5;
	double sr = 60;
	int r = int(ceil(ss*3.0));
	int K = 15;   // Cluster number
	
	// bilateral filter by clustering
	bilateral_clustering(input_image, output_image, ss, sr, r, K);

	//imshow("test", input_image);
	//waitKey();


	return 0;
}