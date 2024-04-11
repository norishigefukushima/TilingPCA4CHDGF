#include <opencp.hpp>

#include "highdimensionalkernelfilter/ClusteringHDKF.hpp"

using namespace std;
using namespace cv;
using namespace cp;

int command(int argc, const char* const argv[])
{
	std::string key =
		"{h help ?      |         | show help command}"
		"{@src_img      |         | source image}"
		"{@dest_img     | out.png | dest image}"
		"{@guide_img    |         | guide image}"
		"{r radius      | 0       | radius of filtering}"
		"{ss            | 3.0     | sigma space}"
		"{sr            | 30.0    | sigma range }"
		"{f             | 1       | filtering method (0: naive, 1: nystrom, 2: PL, 3: GKT)}"
		"{k             | 10      | clustering numbers}"
		"{ki            | 10      | k-means clustering iterations}"
		"{p             | 2       | PCA method (0: no-PCA, 1: full-PCA, 2: tile-PCA) }"
		"{d             | 3       | PCA dimensions}"
		"{nlm_r         | 0       | patch radius of non-local means (0: usual, else non-local means}"
		"{t             | 8       | tile width and height}"
		"{tw            | 0       | tile width}"
		"{th            | 0       | tile height}"
		"{d debug       |         | GUI of viewing local LUT ('q' is quit key for GUI) }"
		;
	cv::CommandLineParser parser(argc, argv, key);

	if (parser.has("h"))
	{
		parser.about("high dimensional Gaussian filtering");

		parser.printMessage();
		cout << "Example: " << endl;
		cout << "hdgf source.png processed.png out.png -r=2 -n=256 -R=2 -L=0 -U=6 -B=3 -o -d" << endl;
		return 0;
	}


	Mat src32f, guide32f;
	if (parser.has("@src_img"))
	{
		src32f = cp::convert(imread(parser.get<string>(0)), CV_32F);
		if (src32f.empty())
		{
			cerr << "source file open error: " << parser.get<string>(0) << endl;
		}
	}
	else
	{
		cout << "src_img is not set" << endl;
		return -1;
	}

	if (parser.has("@guide_img"))
	{
		guide32f = cp::convert(imread(parser.get<string>(2)), CV_32F);
		if (guide32f.empty())
		{
			cerr << "guide image file open error: " << parser.get<string>(2) << endl;
		}
	}
	else
	{
		guide32f = src32f.clone();
	}

	const int r_ = parser.get<int>("r");
	const double ss = parser.get<double>("ss");
	const double sr = parser.get<double>("sr");
	const int r = r_ == 0 ? int(ceil(ss * 3.0)) : r_;
	const int k = parser.get<int>("k");
	const int ki = parser.get<int>("ki");
	const int d = parser.get<int>("d");
	const int pcaMethod = parser.get<int>("p");
	const int nlm_r = parser.get<int>("nlm_r");
	const int tw = parser.get<int>("tw") == 0 ? parser.get<int>("t") : parser.get<int>("tw");
	const int th = parser.get<int>("th") == 0 ? parser.get<int>("t") : parser.get<int>("th");
	cv::Size maxdivision(tw, th);
	const int filteringMethod = parser.get<int>("f");

	Ptr<TileClusteringHDKF> tHDGF = new TileClusteringHDKF(maxdivision, ConstantTimeHDGF::Nystrom);

	Mat dst32f;
	const int border = cv::BORDER_DEFAULT;
	cp::Timer t("", TIME_MSEC, false);
	const bool isInfo = parser.has("d");
	const int iter = isInfo ? 5 : 1;

	cout << "filter method: ";
	if (nlm_r != 0)
	{
		cout << "(NLM) ";
	}

	if (filteringMethod == 0)
	{
		cout << "Naive" << endl;
	}
	if (filteringMethod == 1)
	{
		cout << "HDKF (nystrom)" << endl;
	}
	if (filteringMethod == 2)
	{
		cout << "Permutohedral Lattice" << endl;
	}
	if (filteringMethod == 3)
	{
		cout << "Gaussian KD-Tree" << endl;
	}

	cout << "PCA method:    ";
	if (pcaMethod == 0)
	{
		cout << "no-PCA" << endl;
	}
	if (pcaMethod == 1)
	{
		cout << "full-PCA" << endl;
	}
	if (pcaMethod == 2)
	{
		cout << "tile-PCA" << endl;
	}

	Mat guide4Filter;
	for (int i = 0; i < iter; i++)
	{
		t.start();
		if (nlm_r == 0)
		{
			if (filteringMethod == 0)
			{
				const int d = 2 * r + 1;
				cp::highDimensionalGaussianFilter(src32f, guide32f, dst32f, Size(d, d), sr, ss, border);
			}
			else if (filteringMethod == 1)
			{
				tHDGF->setKMeansAttempts(0);
				tHDGF->setNumIterations(ki);
				tHDGF->setKMeansSigma(25.0);
				//tHDGF->setCropClustering(true);

				if (pcaMethod == 0)
				{
					tHDGF->jointfilter(src32f, guide32f, dst32f, ss, sr, ClusterMethod::KGaussInvMeansPPFast, k, cp::SpatialFilterAlgorithm::SlidingDCT5_AVX, 2, CV_32F, 0.25, DownsampleMethod::DITHER_GRADIENT_MAX, 3.0, border);
				}
				if (pcaMethod == 1)
				{
					cp::cvtColorPCA(guide32f, guide4Filter, d);
					tHDGF->jointfilter(src32f, guide32f, dst32f, ss, sr, ClusterMethod::KGaussInvMeansPPFast, k, cp::SpatialFilterAlgorithm::SlidingDCT5_AVX, 2, CV_32F, 0.25, DownsampleMethod::DITHER_GRADIENT_MAX, 3.0, border);
				}
				if (pcaMethod == 2)
				{
					tHDGF->jointPCAfilter(src32f, guide32f, d, dst32f, ss, sr, ClusterMethod::KGaussInvMeansPPFast, k, cp::SpatialFilterAlgorithm::SlidingDCT5_AVX, 2, CV_32F, 0.25, DownsampleMethod::DITHER_GRADIENT_MAX, 3.0, border);
				}
			}
			else if (filteringMethod == 2)
			{
				if (pcaMethod == 0)
				{
					cp::highDimensionalGaussianFilterPermutohedralLatticeTile(src32f, guide32f, dst32f, sr, ss, maxdivision);
				}
				if (pcaMethod == 1)
				{
					cp::cvtColorPCA(guide32f, guide4Filter, d);
					cp::highDimensionalGaussianFilterPermutohedralLatticeTile(src32f, guide4Filter, dst32f, sr, ss, maxdivision);
				}
				if (pcaMethod == 2)
				{
					cp::highDimensionalGaussianFilterPermutohedralLatticePCATile(src32f, guide32f, dst32f, sr, ss, d, maxdivision);
				}
			}
			else if (filteringMethod == 3)
			{
				if (pcaMethod == 0)
				{
					cp::highDimensionalGaussianFilterGaussianKDTreeTile(src32f, guide32f, dst32f, sr, ss, maxdivision);
				}
				if (pcaMethod == 1)
				{
					cp::cvtColorPCA(guide32f, guide4Filter, d);
					cp::highDimensionalGaussianFilterGaussianKDTreeTile(src32f, guide4Filter, dst32f, sr, ss, maxdivision);
				}
				if (pcaMethod == 2)
				{
					cp::highDimensionalGaussianFilterGaussianKDTreePCATile(src32f, guide32f, dst32f, sr, ss, d, maxdivision);
				}
			}
		}
		else //non-local means
		{
			if (filteringMethod == 0)
			{
				const int d = 2 * r + 1;
				cp::IM2COL(src32f, guide4Filter, nlm_r, border);
				cp::highDimensionalGaussianFilter(src32f, guide4Filter, dst32f, Size(d, d), sr, ss, border);
			}
			else if (filteringMethod == 1)
			{
				tHDGF->setKMeansAttempts(0);
				tHDGF->setNumIterations(ki);
				tHDGF->setKMeansSigma(25.0);
				//tHDGF->setCropClustering(true);

				if (pcaMethod == 0)
				{
					cp::IM2COL(guide32f, guide4Filter, nlm_r, border);
					tHDGF->jointfilter(src32f, guide4Filter, dst32f, ss, sr,
						ClusterMethod::KGaussInvMeansPPFast, k, cp::SpatialFilterAlgorithm::SlidingDCT5_AVX, 2, CV_32F, 0.25, DownsampleMethod::DITHER_GRADIENT_MAX, 3.0, border);
				}
				if (pcaMethod == 1)
				{
					DRIM2COL(guide32f, guide4Filter, nlm_r, d, border, 0);
					tHDGF->jointfilter(src32f, guide32f, dst32f, ss, sr,
						ClusterMethod::KGaussInvMeansPPFast, k, cp::SpatialFilterAlgorithm::SlidingDCT5_AVX, 2, CV_32F, 0.25, DownsampleMethod::DITHER_GRADIENT_MAX, 3.0, border);
				}
				if (pcaMethod == 2)
				{
					tHDGF->nlmfilter(src32f, guide32f, dst32f, ss, sr, nlm_r, d,
						ClusterMethod::KGaussInvMeansPPFast, k, cp::SpatialFilterAlgorithm::SlidingDCT5_AVX, 2, CV_32F, 0.25, DownsampleMethod::DITHER_GRADIENT_MAX, 3.0, border);
				}
			}
			else if (filteringMethod == 2)
			{
				if (pcaMethod == 0)
				{
					cp::IM2COL(src32f, guide4Filter, nlm_r, border);
					cp::highDimensionalGaussianFilterPermutohedralLatticeTile(src32f, guide4Filter, dst32f, sr, ss, maxdivision);
				}
				if (pcaMethod == 1)
				{
					DRIM2COL(guide32f, guide4Filter, nlm_r, d, border, 0);
					cp::highDimensionalGaussianFilterPermutohedralLatticeTile(src32f, guide4Filter, dst32f, sr, ss, maxdivision);
				}
				if (pcaMethod == 2)
				{
					DRIM2COLTile(guide32f, guide4Filter, nlm_r, d, border, 0, maxdivision);
					cp::highDimensionalGaussianFilterPermutohedralLatticeTile(src32f, guide4Filter, dst32f, sr, ss, maxdivision);
				}
			}
			else if (filteringMethod == 3)
			{
				if (pcaMethod == 0)
				{
					cp::IM2COL(src32f, guide4Filter, nlm_r, border);
					cp::highDimensionalGaussianFilterGaussianKDTreeTile(src32f, guide4Filter, dst32f, sr, ss, maxdivision);
				}
				if (pcaMethod == 1)
				{
					DRIM2COL(guide32f, guide4Filter, nlm_r, d, border, 0);
					cp::highDimensionalGaussianFilterGaussianKDTreeTile(src32f, guide4Filter, dst32f, sr, ss, maxdivision);
				}
				if (pcaMethod == 2)
				{
					DRIM2COLTile(guide32f, guide4Filter, nlm_r, d, border, 0, maxdivision);
					cp::highDimensionalGaussianFilterGaussianKDTreePCATile(src32f, guide4Filter, dst32f, sr, ss, d, maxdivision);
				}
			}
		}
		t.pushLapTime();

		if (isInfo)
		{
			const int d = 2 * int(ss * 6.0) + 1;
			Mat ref;
			if (nlm_r != 0)
			{
				cp::IM2COL(src32f, guide4Filter, nlm_r, border);
				cp::highDimensionalGaussianFilter(src32f, guide4Filter, ref, Size(d, d), sr, ss, border);
			}
			else
			{
				cp::highDimensionalGaussianFilter(src32f, guide32f, ref, Size(d, d), sr, ss, border);
			}

			cout << "PSNR [dB], time [ms] (iteration " << iter << ")" << endl;
			cout << format("%9.2f, %8.2f", cp::getPSNR(ref, dst32f), t.getLapTimeMin()) << endl;
		}

		Mat dest = cp::convert(dst32f, CV_8U);
		cv::imwrite(parser.get<string>(1), dest);
		return 0;
	}
}