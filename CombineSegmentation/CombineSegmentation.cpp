// CombineSegmentation.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include "basic_processing.h"
#include "watershed.h"
#include <iostream>
#include <opencv2/opencv.hpp>  
#include <cmath>

using namespace std;
using namespace cv;

int main()
{
	std::cout << "Please enter image path : ";
	string infile;
	std::cin >> infile;

	std::cout << "Please enter blur square size for Area : ";
	int blurAreaSize = 0;
	std::cin >> blurAreaSize;

	if (blurAreaSize % 2 == 0) { --blurAreaSize; }

	std::cout << "Please enter blur square size for Line : ";
	int blurLineSize = 0;
	std::cin >> blurLineSize;

	if (blurLineSize % 2 == 0) { --blurLineSize; }

	std::cout << "Please enter segmentational size for Seed : ";
	int seedSize = 0;
	std::cin >> seedSize;

	/*設定輸出文件名*/

	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));							//檔案路徑
	string infilename(infile.substr(pos1 + 1, pos2 - pos1 - 1));		//檔案名稱

	/****基礎影像設定****/

	/*載入原始影像*/

	Mat image = imread(infile);			//原始影像(8UC1 || 8UC3 )
	if (!image.data) { printf("Oh，no，讀取image錯誤~！ \n"); return false; }

	/*轉換原始影像為灰階影像*/

	Mat gray;			//灰階影像(8UC1)

	cvtColor(image, gray, CV_BGR2GRAY);

	Mat gray_R;			//輸出用(8UC3)
	DrawColorBar(gray, gray_R);

	string gray_G_file = filepath + "\\" + infilename + "_0.0_GRAY(G).png";			//轉換原始影像為灰階影像(灰階)
	imwrite(gray_G_file, gray);
	string gray_R_file = filepath + "\\" + infilename + "_0.1_GRAY(R).png";			//轉換原始影像為灰階影像(紅藍)
	imwrite(gray_R_file, gray_R);


	/****基於面的影像萃取****/

	/*模糊灰階影像*/

	Mat grayBlur;			//模糊灰階影像(8UC1)	
	GaussianBlur(gray, grayBlur, Size(blurAreaSize, blurAreaSize), 0, 0);

	Mat grayBlur_R;			//輸出用(8UC3)
	DrawColorBar(grayBlur, grayBlur_R);

	string grayBlur_G_file = filepath + "\\" + infilename + "_1.0_BLUR_I(G).png";			//模糊灰階影像(灰階)
	imwrite(grayBlur_G_file, grayBlur);
	string grayBlur_R_file = filepath + "\\" + infilename + "_1.1_BLUR_I(R).png";			//模糊灰階影像(紅藍)
	imwrite(grayBlur_R_file, grayBlur_R);

	/*消除灰階影像區域亮度*/

	Mat grayDIV;			//消除灰階影像區域亮度(8UC1)
	DivideArea(gray, grayBlur, grayDIV);

	Mat grayDIV_R;			//輸出用(8UC3)
	DrawColorBar(grayDIV, grayDIV_R);

	string grayDIV_G_file = filepath + "\\" + infilename + "_2.0_DIV_I(G).png";			//消除灰階影像區域亮度(灰階)
	imwrite(grayDIV_G_file, grayDIV);
	string grayDIV_R_file = filepath + "\\" + infilename + "_2.1_DIV_I(R).png";			//消除灰階影像區域亮度(紅藍)
	imwrite(grayDIV_R_file, grayDIV_R);

	/*二值化灰階影像*/

	Mat grayTH;			//二值化灰階影像(8UC1(BW))
	threshold(grayDIV, grayTH, 150, 255, THRESH_BINARY);

	string grayTH_B_file = filepath + "\\" + infilename + "_3.0_TH_I(B).png";			//二值化灰階影像(二值)
	imwrite(grayTH_B_file, grayTH);

	/*去除面白色雜訊*/

	Mat areaCW;			//去除面白色雜訊(8UC1(BW))
	ClearNoise(grayTH, areaCW, 5, 4, 1);

	string areaCW_B_file = filepath + "\\" + infilename + "_4.0_CW_A(B).png";			//去除面白色雜訊(二值)
	imwrite(areaCW_B_file, areaCW);

	/*去除面黑色雜訊*/

	Mat areaCB;			//去除面黑色雜訊(8UC1(BW))
	ClearNoise(areaCW, areaCB, 5, 4, 0);

	string areaCB_B_file = filepath + "\\" + infilename + "_5.0_CB_A(B).png";			//去除面黑色雜訊(二值)
	imwrite(areaCB_B_file, areaCB);

	/*基於面的切割結果*/

	Mat area = grayTH;			//基於面的切割結果(8UC1(BW))

	Mat area_L, area_I;			//輸出用(8UC3、8UC3)
	DrawLabel(area, area_L);
	DrawEdge(area, image, area_I);

	string area_B_file = filepath + "\\" + infilename + "_6.0_AREA(B).png";			//基於面的切割結果(二值)
	imwrite(area_B_file, area);
	string area_L_file = filepath + "\\" + infilename + "_6.1_AREA(L).png";			//基於面的切割結果(標籤)
	imwrite(area_L_file, area_L);
	string area_I_file = filepath + "\\" + infilename + "_6.2_AREA(I).png";			//基於面的切割結果(疊圖)
	imwrite(area_I_file, area_I);


	/****基於線的影像萃取****/

	/*計算影像梯度*/

	Mat gradx, grady;			//水平及垂直梯度(16SC1)
	Differential(gray, gradx, grady);

	Mat gradf;			//梯度場域(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;			//梯度幅值及梯度方向(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_G, grady_G, gradm_G, gradm_R, gradd_C, gradf_C;			//輸出用(8UC1、8UC1、8UC1、8UC3、8UC3、8UC3)
	DrawGrayBar(gradx, gradx_G);
	DrawGrayBar(grady, grady_G);
	DrawGrayBar(gradm, gradm_G);
	DrawColorBar(gradm, gradm_R);
	DrawColorRing(gradd, gradd_C);
	DrawColorRing(gradf, gradf_C);

	string gradx_G_file = filepath + "\\" + infilename + "_7.0_GRAD_X(G).png";			//水平梯度(灰階)
	imwrite(gradx_G_file, gradx_G);
	string grady_G_file = filepath + "\\" + infilename + "_7.1_GRAD_Y(G).png";			//垂直梯度(灰階)
	imwrite(grady_G_file, grady_G);
	string gradm_G_file = filepath + "\\" + infilename + "_7.2_GRAD_M(G).png";			//梯度幅值(灰階)
	imwrite(gradm_G_file, gradm_G);
	string gradm_R_file = filepath + "\\" + infilename + "_7.3_GRAD_M(R).png";			//梯度幅值(紅藍)
	imwrite(gradm_R_file, gradm_R);
	string gradd_C_file = filepath + "\\" + infilename + "_7.4_GRAD_D(C).png";			//梯度方向(色環)
	imwrite(gradd_C_file, gradd_C);
	string gradf_C_file = filepath + "\\" + infilename + "_7.5_GRAD_F(C).png";			//梯度場域(色環)
	imwrite(gradf_C_file, gradf_C);

	/*模糊梯度幅值*/

	Mat gradmBlur;			//模糊梯度幅值(8UC1)	
	GaussianBlur(gradm, gradmBlur, Size(blurLineSize, blurLineSize), 0, 0);

	Mat gradmBlur_R;			//輸出用(8UC3)
	DrawColorBar(gradmBlur, gradmBlur_R);

	string gradmBlur_G_file = filepath + "\\" + infilename + "_8.0_BLUR_M(G).png";			//模糊梯度幅值(灰階)
	imwrite(gradmBlur_G_file, gradmBlur);
	string gradmBlur_R_file = filepath + "\\" + infilename + "_8.1_BLUR_R(G).png";			//模糊梯度幅值(紅藍)
	imwrite(gradmBlur_R_file, gradmBlur_R);

	/*消除梯度幅值區域亮度*/

	Mat gradmDIV;			//消除梯度幅值區域亮度(8UC1)
	DivideLine(gradm, gradmBlur, gradmDIV);

	Mat gradmDIV_G, gradfDIV_C;			//輸出用(8UC1、8UC3)
	DrawGrayBar(gradmDIV, gradmDIV_G);
	DrawColorRing(gradmDIV, gradd, gradfDIV_C);

	string gradmDIV_file = filepath + "\\" + infilename + "_9.0_DIV_M(G).png";			//消除梯度幅值區域亮度(灰階)
	imwrite(gradmDIV_file, gradmDIV_G);
	string gradfDIV_C_file = filepath + "\\" + infilename + "_9.1_DIV_F(C).png";		//消除梯度幅值區域亮度(色環)
	imwrite(gradfDIV_C_file, gradfDIV_C);

	/*二值化梯度幅值*/

	Mat gradmHT;			//二值化梯度幅值(8UC1(BW))
	threshold(gradmDIV, gradmHT, 1, 255, THRESH_BINARY);

	string gradmHT_B_file = filepath + "\\" + infilename + "_10.0_HT_M(B).png";			//二值化梯度幅值(二值)
	imwrite(gradmHT_B_file, gradmHT);

	/*滯後切割線*/

	Mat lineHC;			//滯後切割線(8UC1(BW))
	HysteresisCut(gradmHT, area, lineHC);

	string lineHC_B_file = filepath + "\\" + infilename + "_11.0_HC_L(B).png";			//滯後切割線(二值)
	imwrite(lineHC_B_file, lineHC);

	/*去除線雜訊*/

	Mat lineCN;			//去除線雜訊(8UC1(BW))
	ClearNoise(lineHC, lineCN, 5, 4, 1);

	string lineCN_B_file = filepath + "\\" + infilename + "_12.0_CN_L(B).png";			//去除線雜訊(二值)
	imwrite(lineCN_B_file, lineCN);

	/*基於線的切割結果*/

	Mat line;			//基於線的切割結果(8UC1(BW))
	BWReverse(lineHC, line);

	Mat line_L, line_I;			//輸出用(8UC3、8UC3)
	DrawLabel(line, line_L);
	DrawEdge(line, image, line_I);

	string line_B_file = filepath + "\\" + infilename + "_13.0_LINE(B).png";			//基於線的切割結果(二值)
	imwrite(line_B_file, line);
	string line_L_file = filepath + "\\" + infilename + "_13.1_LINE(L).png";			//基於線的切割結果(標籤)
	imwrite(line_L_file, line_L);
	string line_I_file = filepath + "\\" + infilename + "_13.2_LINE(I).png";			//基於線的切割結果(疊圖)
	imwrite(line_I_file, line_I);

	/****結合面與線的萃取結果****/

	/*結合面與線*/

	Mat objectCOM;			//結合面與線(8UC1(BW))
	BWCombine(area, line, objectCOM);

	Mat objectCOM_L, objectCOM_I;			//輸出用(8UC3、8UC3)
	DrawLabel(objectCOM, objectCOM_L);
	DrawEdge(objectCOM, image, objectCOM_I);

	string  objectCOM_B_file = filepath + "\\" + infilename + "_14.0_COM_O(B).png";			//結合面與線(二值)
	imwrite(objectCOM_B_file, objectCOM);
	string  objectCOM_L_file = filepath + "\\" + infilename + "_14.1_COM_O(L).png";			//結合面與線(標籤)
	imwrite(objectCOM_L_file, objectCOM_L);
	string  objectCOM_I_file = filepath + "\\" + infilename + "_14.2_COM_O(I).png";			//結合面與線(疊圖)
	imwrite(objectCOM_I_file, objectCOM_I);

	/*開運算*/

	Mat objectOpen;			//開運算(8UC1(BW))
	Mat element = (Mat_<uchar>(5, 5) << 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0);
	morphologyEx(objectCOM, objectOpen, MORPH_OPEN, element);

	Mat objectOpen_L, objectOpen_I;			//輸出用(8UC3、8UC3)
	DrawLabel(objectOpen, objectOpen_L);
	DrawEdge(objectOpen, image, objectOpen_I);

	string  objectOpen_B_file = filepath + "\\" + infilename + "_15.0_OPEN_O(B).png";			//開運算(二值)
	imwrite(objectOpen_B_file, objectOpen);
	string  objectOpen_L_file = filepath + "\\" + infilename + "_15.1_OPEN_O(L).png";			//開運算(標籤)
	imwrite(objectOpen_L_file, objectOpen_L);
	string  objectOpen_I_file = filepath + "\\" + infilename + "_15.2_OPEN_O(I).png";			//開運算(疊圖)
	imwrite(objectOpen_I_file, objectOpen_I);

	/*距離轉換*/

	Mat objectDT;		//距離轉換(32FC1(BW))
	distanceTransform(objectOpen, objectDT, CV_DIST_L2, 3);

	Mat objectDT_G, objectDT_R;		//輸出用(8UC1、8UC3)
	DrawGrayBar(objectDT, objectDT_G);
	DrawColorBar(objectDT, objectDT_R);

	string objectDT_G_file = filepath + "\\" + infilename + "_16.0_DT_O(G).png";			//距離轉換(灰階)
	imwrite(objectDT_G_file, objectDT_G);
	string objectDT_R_file = filepath + "\\" + infilename + "_16.1_DT_O(R).png";			//距離轉換(藍紅)
	imwrite(objectDT_R_file, objectDT_R);

	/*填補空洞*/

	Mat objectFH;			//填補空洞(8UC1(BW))
	BWFillhole(objectOpen, objectFH);

	Mat objectFH_L, objectFH_I;			//輸出用(8UC3、8UC3)
	DrawLabel(objectFH, objectFH_L);
	DrawEdge(objectFH, image, objectFH_I);

	string  objectFH_B_file = filepath + "\\" + infilename + "_17.0_FH_O(B).png";			//填補空洞(二值)
	imwrite(objectFH_B_file, objectFH);
	string  objectFH_L_file = filepath + "\\" + infilename + "_17.1_FH_O(L).png";			//填補空洞(標籤)
	imwrite(objectFH_L_file, objectFH_L);
	string  objectFH_I_file = filepath + "\\" + infilename + "_17.2_FH_O(I).png";			//填補空洞(疊圖)
	imwrite(objectFH_I_file, objectFH_I);

	/*求取擴展區域最小值*/

	Mat objectEM;		//求取擴展區域最小值(8UC1(BW))
	ExtendLocalMinimaDetection(objectFH, objectEM, 25);

	Mat objectEM_S;		//輸出用(8UC1)
	DrawSeed(objectOpen, objectEM, objectEM_S);

	string  objectEM_B_file = filepath + "\\" + infilename + "_18.0_EM_O(B).png";			//求取擴展區域最小值(二值)
	imwrite(objectEM_B_file, objectEM);
	string  objectEM_S_file = filepath + "\\" + infilename + "_18.1_EM_O(S).png";			//求取擴展區域最小值(種子)
	imwrite(objectEM_S_file, objectEM_S);

	/*增加未標記之標籤*/

	Mat objectAL;		//增加未標記之標籤(8UC1(BW))
	AddLabel(objectOpen, objectEM, objectAL);

	Mat objectAL_S;		//輸出用(8UC1)
	DrawSeed(objectOpen, objectAL, objectAL_S);

	string  objectAL_B_file = filepath + "\\" + infilename + "_19.0_AL_O(B).png";			//增加未標記之標籤(二值)
	imwrite(objectAL_B_file, objectAL);
	string  objectAL_S_file = filepath + "\\" + infilename + "_19.1_AL_O(S).png";			//增加未標記之標籤(種子)
	imwrite(objectAL_S_file, objectAL_S);

	/*加深低窪區*/

	Mat objectIM;		//加深低窪區(32FC1(BW))
	ImposeMinima(objectDT, objectAL, objectIM);

	Mat objectIM_G, objectIM_R;		//輸出用(8UC1、8UC3)
	DrawGrayBar(objectIM, objectIM_G);
	DrawColorBar(objectIM, objectIM_R);

	string objectIM_G_file = filepath + "\\" + infilename + "_20.0_IM_O(G).png";			//加深低窪區(灰階)
	imwrite(objectIM_G_file, objectIM_G);
	string objectIM_R_file = filepath + "\\" + infilename + "_20.1_IM_O(R).png";			//加深低窪區(藍紅)
	imwrite(objectIM_R_file, objectIM_R);

	/*分水嶺演算法*/

	Mat objectWT;		//分水嶺演算法(32SC1(BW))
	WatershedTransform(objectOpen, objectIM, objectWT);

	Mat objectWT_L, objectWT_I;		//輸出用(8UC3、8UC3)
	DrawLabel(objectWT, objectWT_L);
	DrawEdge(objectWT, image, objectWT_I);

	string  objectWT_B_file = filepath + "\\" + infilename + "_21.0_WT_O(B).png";			//分水嶺演算法(二值)
	imwrite(objectWT_B_file, objectWT);
	string  objectWT_L_file = filepath + "\\" + infilename + "_21.1_WT_O(L).png";			//分水嶺演算法(標籤)
	imwrite(objectWT_L_file, objectWT_L);
	string  objectWT_I_file = filepath + "\\" + infilename + "_21.2_WT_O(I).png";			//分水嶺演算法(疊圖)
	imwrite(objectWT_I_file, objectWT_I);

	/*距離轉換2*/

	Mat objectDT2;		//距離轉換2(32FC1(BW))
	distanceTransform(objectWT, objectDT2, CV_DIST_L2, 3);

	Mat objectDT2_G, objectDT2_R;		//輸出用(8UC1、8UC3)
	DrawGrayBar(objectDT2, objectDT2_G);
	DrawColorBar(objectDT2, objectDT2_R);

	string objectDT2_G_file = filepath + "\\" + infilename + "_16.0_DT2_O(G).png";			//距離轉換2(灰階)
	imwrite(objectDT2_G_file, objectDT2_G);
	string objectDT2_R_file = filepath + "\\" + infilename + "_16.1_DT2_O(R).png";			//距離轉換2(藍紅)
	imwrite(objectDT2_R_file, objectDT2_R);

	/*填補空洞2*/

	Mat objectFH2;			//填補空洞2(8UC1(BW))
	BWFillhole(objectWT, objectFH2);

	Mat objectFH2_L, objectFH2_I;			//輸出用2(8UC3、8UC3)
	DrawLabel(objectFH2, objectFH2_L);
	DrawEdge(objectFH2, image, objectFH2_I);

	string  objectFH2_B_file = filepath + "\\" + infilename + "_22.0_FH2_O(B).png";			//填補空洞2(二值)
	imwrite(objectFH2_B_file, objectFH2);
	string  objectFH2_L_file = filepath + "\\" + infilename + "_22.1_FH2_O(L).png";			//填補空洞2(標籤)
	imwrite(objectFH2_L_file, objectFH2_L);
	string  objectFH2_I_file = filepath + "\\" + infilename + "_22.2_FH2_O(I).png";			//填補空洞2(疊圖)
	imwrite(objectFH2_I_file, objectFH2_I);

	/*求取擴展區域最小值2*/

	Mat objectEM2;		//求取擴展區域最小值2(8UC1(BW))
	ExtendLocalMinimaDetection(objectFH2, objectEM2, 5);

	Mat objectEM2_S;		//輸出用(8UC1)
	DrawSeed(objectWT, objectEM2, objectEM2_S);

	string  objectEM2_B_file = filepath + "\\" + infilename + "_23.0_EM2_O(B).png";			//求取擴展區域最小值2(二值)
	imwrite(objectEM2_B_file, objectEM2);
	string  objectEM2_S_file = filepath + "\\" + infilename + "_23.1_EM2_O(S).png";			//求取擴展區域最小值2(種子)
	imwrite(objectEM2_S_file, objectEM2_S);

	/*增加未標記之標籤2*/

	Mat objectAL2;		//增加未標記之標籤2(8UC1(BW))
	AddLabel(objectWT, objectEM2, objectAL2);

	Mat objectAL2_S;		//輸出用(8UC1)
	DrawSeed(objectWT, objectAL2, objectAL2_S);

	string  objectAL2_B_file = filepath + "\\" + infilename + "_24.0_AL2_O(B).png";			//增加未標記之標籤2(二值)
	imwrite(objectAL2_B_file, objectAL2);
	string  objectAL2_S_file = filepath + "\\" + infilename + "_24.1_AL2_O(S).png";			//增加未標記之標籤2(種子)
	imwrite(objectAL2_S_file, objectAL2_S);

	/*加深低窪區2*/

	Mat objectIM2;		//加深低窪區(32FC1(BW))
	ImposeMinima(objectDT2, objectAL2, objectIM2);

	Mat objectIM2_G, objectIM2_R;		//輸出用(8UC1、8UC3)
	DrawGrayBar(objectIM2, objectIM2_G);
	DrawColorBar(objectIM2, objectIM2_R);

	string objectIM2_G_file = filepath + "\\" + infilename + "_25.0_IM2_O(G).png";			//加深低窪區2(灰階)
	imwrite(objectIM2_G_file, objectIM2_G);
	string objectIM2_R_file = filepath + "\\" + infilename + "_25.1_IM2_O(R).png";			//加深低窪區2(藍紅)
	imwrite(objectIM2_R_file, objectIM2_R);

	/*分水嶺演算法2*/

	Mat objectWT2;		//分水嶺演算法2(32SC1(BW))
	WatershedTransform(objectWT, objectIM2, objectWT2);

	Mat objectWT2_L, objectWT2_I;		//輸出用(8UC3、8UC3)
	DrawLabel(objectWT2, objectWT2_L);
	DrawEdge(objectWT2, image, objectWT2_I);

	string  objectWT2_B_file = filepath + "\\" + infilename + "_26.0_WT2_O(B).png";			//分水嶺演算法2(二值)
	imwrite(objectWT2_B_file, objectWT2);
	string  objectWT2_L_file = filepath + "\\" + infilename + "_26.1_WT2_O(L).png";			//分水嶺演算法2(標籤)
	imwrite(objectWT2_L_file, objectWT2_L);
	string  objectWT2_I_file = filepath + "\\" + infilename + "_26.2_WT2_O(I).png";			//分水嶺演算法2(疊圖)
	imwrite(objectWT2_I_file, objectWT2_I);

	return 0;
}



