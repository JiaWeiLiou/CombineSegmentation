// CombineSegmentation.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include "basic_processing.h"
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

	Mat area = areaCB;			//基於面的切割結果(8UC1(BW))

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
	BWReverse(lineCN, line);

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

	Mat objectOpen;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
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

	///*分水嶺演算法*/

	//Mat objectWS;		//分水嶺演算法(32SC1(BW))
	//BWWatershed(image, objectOpen, area, objectWS);

	//Mat objectWS_L, objectWS_I;		//輸出用(8UC3、8UC3)
	//DrawLabel(objectWS, objectWS_L);
	//DrawEdge(objectWS, image, objectWS_I);

	//string  objectWS_B_file = filepath + "\\" + infilename + "_16.0_WS_O(B).png";			//分水嶺演算法(二值)
	//imwrite(objectWS_B_file, objectWS);
	//string  objectWS_L_file = filepath + "\\" + infilename + "_16.1_WS_O(L).png";			//分水嶺演算法(標籤)
	//imwrite(objectWS_L_file, objectWS_L);
	//string  objectWS_I_file = filepath + "\\" + infilename + "_16.2_WS_O(I).png";			//分水嶺演算法(疊圖)
	//imwrite(objectWS_I_file, objectWS_I);

	return 0;
}



