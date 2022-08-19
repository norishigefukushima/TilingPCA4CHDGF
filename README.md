# パッチPCAのメソッド説明
実装している関数は以下．
```cpp
enum class NeighborhoodPCA
{
	//事前にデータを差分するor差分しない実装
	MEAN_SUB_32F,//代表平均値差分32bit実装（Eq. 13で計算するが，x_m,x_nは事前に平均値を引き算したもの）
	NO_SUB_32F,//差分無し32bit実装 （Eq. 13で計算）
	CONSTANT_SUB_32F,//固定値差分32bit実装（Eq. 13で計算するがx_m,x_nは事前に定数を引き算したもの）
	MEAN_SUB_64F,//代表平均値差分64bit実装（Eq. 13で計算するが，x_m,x_nは事前に平均値を引き算）
	NO_SUB_64F,//差分無し64bit実装（Eq. 13で計算）
	CONSTANT_SUB_64F,//固定値差分64bit実装（Eq. 13で計算するが，x_m,x_nは事前に定数を引き算）

	//元の式通り要素ごとに差分する
	MEAN_SUB_REDUNDANT_32F, 　 //代表平均値を差分するEq. 5相当32bit実装．ただし対角要素の計算はカット, e.g. σ12とσ21は同じ)
	CONSTANT_SUB_REDUNDANT_32F,//固定定数値を差分するEq. 5相当32bit実装．ただし対角要素の計算はカット, e.g. σ12とσ21は同じ)
	MEAN_SUB_REDUNDANT_64F,　　//代表平均値を差分するEq. 5相当32bit実装．ただし対角要素の計算はカット, e.g. σ12とσ21は同じ)
	CONSTANT_SUB_REDUNDANT_64F,//固定定数値を差分するEq. 5相当64bit実装．ただし対角要素の計算はカット, e.g. σ12とσ21は同じ)
	FULL_SUB_REDUNDANT_32F,    //完全に元の式通り要素ごとに要素ごとの平均値を差分するEq. 5相当．ただし対角要素の冗長計算はカット, e.g. σ12とσ21は同じ）
	FULL_SUB_REDUNDANT_64F,    //完全に元の式通り要素ごとに要素ごとの平均値を差分するEq. 5相当．ただし対角要素の冗長計算はカット, e.g. σ12とσ21は同じ）

	//OpenCVの関数を使うもの．でっかいバッファを作ってPCA．対角計算も冗長．非常に遅い．
	OPENCV_PCA,//OpenCVでcv::calcCovarMatrixやPCA
	OPENCV_COV,//OpenCVでcv::calcCovarMatrix

	SIZE
};
```

テスト関数は以下．

* patchPCATest();//次元圧縮の行列計算だけを計測．フロベニウスノルムを計算しているが，行列の絶対的な差には意味ない？
* patchPCAHDGF();//高次元ガウシアンフィルタ（ほぼNLM）でPCA無しのもととのPSNRを計測

# SIGMAPからの発展
関数リスト

ガイドはnチャネル対応・現在フィルタ対象は3チャネル
Naive: cp::highDimensionalGaussianFilter 
TileConstantTimeHDGF tnystrom(division, ConstantTimeHDGF::Nystrom);
filtering
jointfilter
jointPCAfilter（タイル内PCA） vs cvtColorPCA then jointfilter（全体PCA）
tnystrom.nlmfilter（タイル内NLM用PCA） vs convertNeighborhoodToChannelsPCA then jointfilter（全体PCA）;

PHL
https://link.springer.com/content/pdf/10.1007/s10851-012-0379-2.pdf
https://graphics.stanford.edu/papers/permutohedral/permutohedral_techreport.pdf


## SIGMAP投稿時点
前提
* クラスタリングベースの方法（ナイシュトローム）がカラーバイラテラルフィルタに対して提案されている．

提案
ナイシュトローム法に対して，HDGFに適用する形で定式化する．
そして，下記の拡張を行う．
* HDGFの定義をクラスタリングベースの方法に使いやすいように一般化
	* カラーバイラテラル
	* マルチラテラル（フラッシュノーフラッシュ）
	* トリラテラル（デプスリファイン）
	* ノンローカルミーン
* 高次元空間をPCAで次元圧縮
* タイリングで高速化．タイルでPCAする．全体ではないので高速
* Kmeansについては特に触れられてない

パミュードヘドラルラティス：発表だけ．原稿にはなし

## 発展予定
### 手法
* 一般化にハイパースペクトルを追加
* 高速化に空間ダウンサンプルを追加（比較となるKD木，ラティスは空間ダウンサンプルを含んでいるため）
* K-meansを，PCAのドメインで実行

### 実験
* 実験の比較にガウシアンKD木とパムードヘドラルラティスを追加
* PCAのあとで，フルサンプルフィルタを実験に追加
* タイリングPCAvs画像全体PCAを実験に追加
* どんな信号が次元圧縮に聞きやすいかをまとめる

# bilateral_clustering
kを動的に変えるXmeansなど．→畳み込み回数の削減  
nを動的に変える→サブサンプル，ランダムサンプル，ディザリング→クラスタリングの高速化  
初期クラスタの選択  
イタレーション回数とPSNRの安定性  
クラスタリング前の色変換を挟むと精度上がるかも？

サンプリングポイントは外周（copymakeborderした領域）は取らないほうがいい．取りすぎるとPSNRが劣化している．

乱択はボックスフィルタの空間カーネルに弱いはず．
CBFはボックスだと速くなる

ダウンサンプルの導入

constant-time clerstering color BF
* 初期クラスタの割り当て方法
* k-meansの繰り返し回数
* クラスタリング高速化のためのサンプル点削減

* クラスタリング方法とPSNRの安定性
* タイリングによるkの削減もしくは精度向上

# K-means
今の実装はk-d-nループ
これは，たぶんkを最内ループにしたほうが速い．
## メモ
https://www.slideshare.net/HirotakaHachiya/12-k-122960307
https://www.slideshare.net/y-uti/kmeans-49981384

https://github.com/ghamerly/fast-kmeans

https://github.com/src-d/kmcuda
https://github.com/siddheshk/Faster-Kmeans


Clustering is considered as one of the most important unsupervised learning problem. 

KMeans++ [5] obtained an algorithm that is Oሺlog kሻ competitive with the optimal
clustering. K-Means|| [6] proposed an efficient parallel version of the sequential KMeans++. Both of them focus on the initialization of K-Means, which is crucial for
obtaining good final solutions. K-Means# [7] provided a clustering algorithm that
approximately optimizes the k-means objective in the one-pass streaming setting
based on K-Means++

Dayan, P.: Unsupervised Learning. The MIT Encyclopedia of the Cognitive Sciences
Google Scholar
2.
MacQueen, J.B.: Some Methods for classification and Analysis of Multivariate Observations. In: Proceedings of 5th Berkeley Symposium on Mathematical Statistics and Probability, vol. 1, pp. 281–297. University of California Press, Berkeley (1967)
Google Scholar
3.
Wu, X., Kumar, V., Ross Quinlan, J., Ghosh, J., Yang, Q., Motoda, H., McLachlan, G.J., Ng, A., Liu, B., Yu, P.S., Zhou, Z.-H., Steinbach, M., Hand, D.J., Steinberg, D.: Top 10 algorithms in data mining. Knowl. Inf. Syst. 14, 1–37 (2007)
CrossRefGoogle Scholar
4.
Lloyd, S.: Least Squares Quantization in PCM. IEEE Transactions on Information Theory 28(2), 129–136 (1982)
MathSciNetCrossRefzbMATHGoogle Scholar
5.
Arthur, D., Vassilvitskii, S.: k-means++: the advantages of careful seeding. In: SODA (2007)
Google Scholar
6.
Bahman, B., Moseley, B., Vattani, A.: Scalable K-Means++. Proceedings of the VLDB Endowment 5(7)
Google Scholar
7.
Ailon, N., Jaiswal, R., Monteleoni, C.: Streaming K-Means approximation. In: NIPS (2009)
Google Scholar
8.
Che, S., Boyer, M., Meng, J., Tarjan, D.: A performance study of general-purpose applications on graphics processors using CUDA. Journal of Parallel and Distributed Computing 68(10), 1370–1380 (2008)
CrossRefGoogle Scholar
9.
Bai, H.-T., He, L.-L., Ouyang, D.-T., Li, Z.-S., Li, H.: K-Means on commodity GPUs with CUDA. In: World Congress on Computer Science and Information Engineering (2009)
Google Scholar
10.
Farivar, R., Rebolledo, D., Chan, E., Campbell, R.: A parallel implementation of K-Means clustering on GPUs. In: Proceeding of International Conference on Parallel and Distributed Processing Techniques and Applications (2008)
Google Scholar
11.
Mahout, http://mahout.apache.org/
12.
Hadoop, http://hadoop.apache.org/
13.
Dean, J., Ghemawat, S.: MapReduce: Simplified data processing on large clusters. In: OSDI, pp. 137–150 (2004)
Google Scholar
14.
Feldman, M.: TACC Steps Up to the MIC. HPCwire (April 21, 2011), http://www.hpcwire.com/hpcwire/2011-04-21/tacc_steps_up_to_the_mic.html
15.
Nvidia. Oak Ridge National Lab Turns to NVIDIA Tesla GPUs to Deploy World’s Leading Supercomputer. HPCwire (October 11, 2011), http://www.hpcwire.com/hpcwire/2011-10-11/oak_ridge_national_lab_turns_to_Nvidia_tesla_gpus_to_deploy_world_s_leading_supercomputer.html
16.
Intel. Introducing Intel Many Integrated Core Architecture. Press release (2011), http://www.intel.com/technology/architecture-silicon/mic/index.htm
17.
Seiler, L., et al.: Larrabee: A Many-Core x86 Architecture for visual Computing. ACM Trans. Graphics 27(3), 18:1–18:15 (2008)
MathSciNetCrossRefGoogle Scholar
18.
Saule, E., Catalyurek, U.V.: An early evaluation of the scalability of graph algorithms on the Intel MIC Architecture. In: IEEE IPDPSW (2012), doi:10.1109
Google Scholar
19.
McFarlin, D.S., Arbatov, V., Franchetti, F., Puschel, M., Zurich, E.: Automatic SIMD vectorization of fast fourier transforms for the Larrabee and AVX Instruction sets. In: ACM ICS (2011)
Google Scholar
20.
Owens, J.D., Houston, M., Luebke, D., Green, S., Stone, J.E., Phillips, J.C.: GPU Computing. Proceedings of the IEEE 96(5) (May 2008)
Google Scholar
21.
Aloise, D., Deshpande, A., Hansen, P., Popat, P.: NP-hardness of Euclidean sum-of-squares clustering. Machine Learning 75(2), 245–248 (2009)
CrossRefGoogle Scholar
22.
Henretty, T., Stock, K., Pouchet, L.-N., Franchetti, F., Ramanujam, J., Sadayappan, P.: Data Layout Transformation for Stencil Computations on Short-Vector SIMD Architectures. In: Knoop, J. (ed.) CC 2011. LNCS, vol. 6601, pp. 225–245. Springer, Heidelberg (2011)
CrossRefGoogle Scholar
23.
He, B.S., Govindaraju, N.K., Luo, Q., Smith, B.: Efficient Gather and Scatter operations on graphics processors. In: Proceeding of the 2007 ACM/IEEE Conference on Supercomputing, p. 46. ACM (2007)
Google Scholar


Since the k-means++ initialization needs k passes over the data, it does not scale very well to large data sets. Bahman Bahmani et al. have proposed a scalable variant of k-means++ called k-means|| which provides the same theoretical guarantees and yet is highly scalable.[12]


A vectorized k-means algorithm for compressed datasets: design and experimental analysis
データ量の増加に伴い、クラスタリングアルゴリズムの性能はメモリサブシステムに大きく影響される。本論文では、メモリ階層に沿ったデータの移動を大幅に削減するために、Lloydのk-meansクラスタリングアルゴリズムの新規かつ効率的な実装を提案する。我々の貢献は、大多数のプロセッサには強力なSIMD（Single Instruction Multiple Data）命令が搭載されているが、ほとんどの場合、十分に利用されていないという事実に基づいている。SIMDはCPUの計算能力を向上させ、賢く使えば、特にメモリに依存するアプリケーションのために、データを圧縮/解凍することによって、アプリケーションのデータ転送を改善する機会とみなすことができます。我々の貢献には、SIMDフレンドリーなデータレイアウト構成、キー関数のインレジスタ実装、SIMDベースの圧縮が含まれます。我々の最適化されたSIMDベースの圧縮方法を用いて、i7 Haswellマシンではk-meansの性能とエネルギーをそれぞれ4.5倍と8.7倍、Xeon Phi: KNLではシングルスレッドで22倍と22.2倍向上させることが可能であることを実証した。


最初の N: いくつかの初期データ ポイント数がデータセットから選択され、初期の手段として使用されます。
この方法は "Forgy 法" とも呼ばれます。

Lloyd（Forgy）クラスタの更新：
全データをクラスタに割り当てて，その後に全部のセントロイドを更新
MacQueen
一つのデータをクラスタを割り当てるごとにセントロイドを更新


ランダム: アルゴリズムによって、クラスターにデータ ポイントがランダムに配置され、クラスターのランダムに割り当てられたポイントの重心になる初期平均値が計算されます。
この方法は "ランダム パーティション" 法とも呼ばれます。

疑似コード
```cpp
inline float floorTo(float, int) {}
void func() {

	//AoS
	const int d = 3;//number of channels/dimensions
	const int N = 1024;//data size
	const int K = 8;//number of clusters
	float centroid[K][d];
	float data[N][d];
	float label[N];
	for (int n = 0; n < N; n++)
	{
		float Dxn = FLT_MAX;
		for (int k = 0; k < K; k++)
		{
			float Dk = 0.f;
			for (int i = 0; i < d; i++)
			{
				float diff = (data[n][i] - centroid[k][i]);
				Dk += diff * diff;
			}
			if (Dk < Dxn)
			{
				Dxn = Dk;
				label[n] = k;
			}
		}
	}

	//AoS SIMD
	for (int n = 0; n < N; n++)
	{
		float Dxn = FLT_MAX;
		for (int k = 0; k < K; k++)
		{
			float Dk = 0.f;
			int dsimd = floorTo(d, 4);//floor function for 2nd argment
			for (int i = 0; i < dsimd; i += 4)
			{
				//__m128 diff = _mm_sub_ps(&data[n],&centroid[k]);
				float diff0 = (data[n][i + 0] - centroid[k][i + 0]);
				float diff1 = (data[n][i + 1] - centroid[k][i + 1]);
				float diff2 = (data[n][i + 2] - centroid[k][i + 2]);
				float diff3 = (data[n][i + 3] - centroid[k][i + 3]);
				// diff = _mm_mul_ps(diff,diff);
				diff0 = diff0 * diff0;
				diff1 = diff1 * diff1;
				diff2 = diff2 * diff2;
				diff3 = diff3 * diff3;
				// horizontal add: 
				//diff = _mm_hadd_ps(diff,diff);
				//diff = _mm_hadd_ps(diff,diff);
				//Dk = diff[0];
				Dk = diff0 + diff1 + diff2 + diff3;
			}
			for (int i = dsimd; i < d; i++)//radidual processing
			{
				float diff = (data[n][i] - centroid[k][i]);
				Dk += diff * diff;
			}
			if (Dk < Dxn)
			{
				Dxn = Dk;
				label[n] = k;
			}
		}
	}

	//SoA
	const int d = 3;//number of channels/dimensions
	const int N = 1024;//data size
	const int K = 8;//number of clusters
	float centroid[d][K];
	float data[d][N];
	float label[N];
	for (int n = 0; n < N; n++)
	{
		float Dxn = FLT_MAX;
		for (int k = 0; k < K; k++)
		{
			float Dk = 0.f;
			for (int i = 0; i < d; i++)
			{
				float diff = (data[i][n] - centroid[i][k]);
				Dk += diff * diff;
			}
			if (Dk < Dxn)
			{
				Dxn = Dk;
				label[n] = k;
			}
		}
	}

	//AoS SIMD
	int Nsimd = floorTo(N, 4);//floor function for 2nd argment
	for (int n = 0; n < Nsimd; n += 4)
	{
		float Dxn0 = FLT_MAX;
		float Dxn1 = FLT_MAX;
		float Dxn2 = FLT_MAX;
		float Dxn3 = FLT_MAX;
		for (int k = 0; k < K; k++)
		{
			float Dk0 = 0.f;
			float Dk1 = 0.f;
			float Dk2 = 0.f;
			float Dk3 = 0.f;
			for (int i = 0; i < d; i++)
			{
				//__m128 diff = _mm_sub_ps(&data[i],&centroid[i]);
				float diff0 = (data[i][n + 0] - centroid[i][k]);
				float diff1 = (data[i][n + 1] - centroid[i][k]);
				float diff2 = (data[i][n + 2] - centroid[i][k]);
				float diff3 = (data[i][n + 3] - centroid[i][k]);
				// diff = _mm_mul_ps(diff,diff);
				Dk0 += diff0 * diff0;
				Dk1 += diff1 * diff1;
				Dk2 += diff2 * diff2;
				Dk3 += diff3 * diff3;
			}

			if (Dk0 < Dxn0) { Dxn0 = Dk0; label[n + 0] = k; }
			if (Dk1 < Dxn1) { Dxn1 = Dk1; label[n + 1] = k; }
			if (Dk2 < Dxn2) { Dxn2 = Dk2; label[n + 2] = k; }
			if (Dk3 < Dxn3) { Dxn3 = Dk3; label[n + 3] = k; }
			
		}
	}
	//radidual processig...
}
```
