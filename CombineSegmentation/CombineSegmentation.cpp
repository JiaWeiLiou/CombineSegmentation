// CombineSegmentation.cpp : �w�q�D���x���ε{�����i�J�I�C
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

	/*�]�w��X���W*/

	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));							//�ɮ׸��|
	string infilename(infile.substr(pos1 + 1, pos2 - pos1 - 1));		//�ɮצW��

	/****��¦�v���]�w****/

	/*���J��l�v��*/

	Mat image = imread(infile);			//��l�v��(8UC1 || 8UC3 )
	if (!image.data) { printf("Oh�Ano�AŪ��image���~~�I \n"); return false; }

	/*�ഫ��l�v�����Ƕ��v��*/

	Mat gray;			//�Ƕ��v��(8UC1)

	cvtColor(image, gray, CV_BGR2GRAY);

	Mat gray_R;			//��X��(8UC3)
	DrawColorBar(gray, gray_R);

	string gray_G_file = filepath + "\\" + infilename + "_0.0_GRAY(G).png";			//�ഫ��l�v�����Ƕ��v��(�Ƕ�)
	imwrite(gray_G_file, gray);
	string gray_R_file = filepath + "\\" + infilename + "_0.1_GRAY(R).png";			//�ഫ��l�v�����Ƕ��v��(����)
	imwrite(gray_R_file, gray_R);


	/****��󭱪��v���Ѩ�****/

	/*�ҽk�Ƕ��v��*/

	Mat grayBlur;			//�ҽk�Ƕ��v��(8UC1)	
	GaussianBlur(gray, grayBlur, Size(blurAreaSize, blurAreaSize), 0, 0);

	Mat grayBlur_R;			//��X��(8UC3)
	DrawColorBar(grayBlur, grayBlur_R);

	string grayBlur_G_file = filepath + "\\" + infilename + "_1.0_BLUR_I(G).png";			//�ҽk�Ƕ��v��(�Ƕ�)
	imwrite(grayBlur_G_file, grayBlur);
	string grayBlur_R_file = filepath + "\\" + infilename + "_1.1_BLUR_I(R).png";			//�ҽk�Ƕ��v��(����)
	imwrite(grayBlur_R_file, grayBlur_R);

	/*�����Ƕ��v���ϰ�G��*/

	Mat grayDIV;			//�����Ƕ��v���ϰ�G��(8UC1)
	DivideArea(gray, grayBlur, grayDIV);

	Mat grayDIV_R;			//��X��(8UC3)
	DrawColorBar(grayDIV, grayDIV_R);

	string grayDIV_G_file = filepath + "\\" + infilename + "_2.0_DIV_I(G).png";			//�����Ƕ��v���ϰ�G��(�Ƕ�)
	imwrite(grayDIV_G_file, grayDIV);
	string grayDIV_R_file = filepath + "\\" + infilename + "_2.1_DIV_I(R).png";			//�����Ƕ��v���ϰ�G��(����)
	imwrite(grayDIV_R_file, grayDIV_R);

	/*�G�ȤƦǶ��v��*/

	Mat grayTH;			//�G�ȤƦǶ��v��(8UC1(BW))
	threshold(grayDIV, grayTH, 150, 255, THRESH_BINARY);

	string grayTH_B_file = filepath + "\\" + infilename + "_3.0_TH_I(B).png";			//�G�ȤƦǶ��v��(�G��)
	imwrite(grayTH_B_file, grayTH);

	/*�h�����զ����T*/

	Mat areaCW;			//�h�����զ����T(8UC1(BW))
	ClearNoise(grayTH, areaCW, 5, 4, 1);

	string areaCW_B_file = filepath + "\\" + infilename + "_4.0_CW_A(B).png";			//�h�����զ����T(�G��)
	imwrite(areaCW_B_file, areaCW);

	/*�h�����¦����T*/

	Mat areaCB;			//�h�����¦����T(8UC1(BW))
	ClearNoise(areaCW, areaCB, 5, 4, 0);

	string areaCB_B_file = filepath + "\\" + infilename + "_5.0_CB_A(B).png";			//�h�����¦����T(�G��)
	imwrite(areaCB_B_file, areaCB);

	/*��󭱪����ε��G*/

	Mat area = areaCB;			//��󭱪����ε��G(8UC1(BW))

	Mat area_L, area_I;			//��X��(8UC3�B8UC3)
	DrawLabel(area, area_L);
	DrawEdge(area, image, area_I);

	string area_B_file = filepath + "\\" + infilename + "_6.0_AREA(B).png";			//��󭱪����ε��G(�G��)
	imwrite(area_B_file, area);
	string area_L_file = filepath + "\\" + infilename + "_6.1_AREA(L).png";			//��󭱪����ε��G(����)
	imwrite(area_L_file, area_L);
	string area_I_file = filepath + "\\" + infilename + "_6.2_AREA(I).png";			//��󭱪����ε��G(�|��)
	imwrite(area_I_file, area_I);


	/****���u���v���Ѩ�****/

	/*�p��v�����*/

	Mat gradx, grady;			//�����Ϋ������(16SC1)
	Differential(gray, gradx, grady);

	Mat gradf;			//��׳���(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;			//��״T�Ȥα�פ�V(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_G, grady_G, gradm_G, gradm_R, gradd_C, gradf_C;			//��X��(8UC1�B8UC1�B8UC1�B8UC3�B8UC3�B8UC3)
	DrawGrayBar(gradx, gradx_G);
	DrawGrayBar(grady, grady_G);
	DrawGrayBar(gradm, gradm_G);
	DrawColorBar(gradm, gradm_R);
	DrawColorRing(gradd, gradd_C);
	DrawColorRing(gradf, gradf_C);

	string gradx_G_file = filepath + "\\" + infilename + "_7.0_GRAD_X(G).png";			//�������(�Ƕ�)
	imwrite(gradx_G_file, gradx_G);
	string grady_G_file = filepath + "\\" + infilename + "_7.1_GRAD_Y(G).png";			//�������(�Ƕ�)
	imwrite(grady_G_file, grady_G);
	string gradm_G_file = filepath + "\\" + infilename + "_7.2_GRAD_M(G).png";			//��״T��(�Ƕ�)
	imwrite(gradm_G_file, gradm_G);
	string gradm_R_file = filepath + "\\" + infilename + "_7.3_GRAD_M(R).png";			//��״T��(����)
	imwrite(gradm_R_file, gradm_R);
	string gradd_C_file = filepath + "\\" + infilename + "_7.4_GRAD_D(C).png";			//��פ�V(����)
	imwrite(gradd_C_file, gradd_C);
	string gradf_C_file = filepath + "\\" + infilename + "_7.5_GRAD_F(C).png";			//��׳���(����)
	imwrite(gradf_C_file, gradf_C);

	/*�ҽk��״T��*/

	Mat gradmBlur;			//�ҽk��״T��(8UC1)	
	GaussianBlur(gradm, gradmBlur, Size(blurLineSize, blurLineSize), 0, 0);

	Mat gradmBlur_R;			//��X��(8UC3)
	DrawColorBar(gradmBlur, gradmBlur_R);

	string gradmBlur_G_file = filepath + "\\" + infilename + "_8.0_BLUR_M(G).png";			//�ҽk��״T��(�Ƕ�)
	imwrite(gradmBlur_G_file, gradmBlur);
	string gradmBlur_R_file = filepath + "\\" + infilename + "_8.1_BLUR_R(G).png";			//�ҽk��״T��(����)
	imwrite(gradmBlur_R_file, gradmBlur_R);

	/*������״T�Ȱϰ�G��*/

	Mat gradmDIV;			//������״T�Ȱϰ�G��(8UC1)
	DivideLine(gradm, gradmBlur, gradmDIV);

	Mat gradmDIV_G, gradfDIV_C;			//��X��(8UC1�B8UC3)
	DrawGrayBar(gradmDIV, gradmDIV_G);
	DrawColorRing(gradmDIV, gradd, gradfDIV_C);

	string gradmDIV_file = filepath + "\\" + infilename + "_9.0_DIV_M(G).png";			//������״T�Ȱϰ�G��(�Ƕ�)
	imwrite(gradmDIV_file, gradmDIV_G);
	string gradfDIV_C_file = filepath + "\\" + infilename + "_9.1_DIV_F(C).png";		//������״T�Ȱϰ�G��(����)
	imwrite(gradfDIV_C_file, gradfDIV_C);

	/*�G�ȤƱ�״T��*/

	Mat gradmHT;			//�G�ȤƱ�״T��(8UC1(BW))
	threshold(gradmDIV, gradmHT, 1, 255, THRESH_BINARY);

	string gradmHT_B_file = filepath + "\\" + infilename + "_10.0_HT_M(B).png";			//�G�ȤƱ�״T��(�G��)
	imwrite(gradmHT_B_file, gradmHT);

	/*������νu*/

	Mat lineHC;			//������νu(8UC1(BW))
	HysteresisCut(gradmHT, area, lineHC);

	string lineHC_B_file = filepath + "\\" + infilename + "_11.0_HC_L(B).png";			//������νu(�G��)
	imwrite(lineHC_B_file, lineHC);

	/*�h���u���T*/

	Mat lineCN;			//�h���u���T(8UC1(BW))
	ClearNoise(lineHC, lineCN, 5, 4, 1);

	string lineCN_B_file = filepath + "\\" + infilename + "_12.0_CN_L(B).png";			//�h���u���T(�G��)
	imwrite(lineCN_B_file, lineCN);

	/*���u�����ε��G*/

	Mat line;			//���u�����ε��G(8UC1(BW))
	BWReverse(lineCN, line);

	Mat line_L, line_I;			//��X��(8UC3�B8UC3)
	DrawLabel(line, line_L);
	DrawEdge(line, image, line_I);

	string line_B_file = filepath + "\\" + infilename + "_13.0_LINE(B).png";			//���u�����ε��G(�G��)
	imwrite(line_B_file, line);
	string line_L_file = filepath + "\\" + infilename + "_13.1_LINE(L).png";			//���u�����ε��G(����)
	imwrite(line_L_file, line_L);
	string line_I_file = filepath + "\\" + infilename + "_13.2_LINE(I).png";			//���u�����ε��G(�|��)
	imwrite(line_I_file, line_I);

	/****���X���P�u���Ѩ����G****/

	/*���X���P�u*/

	Mat objectCOM;			//���X���P�u(8UC1(BW))
	BWCombine(area, line, objectCOM);

	Mat objectCOM_L, objectCOM_I;			//��X��(8UC3�B8UC3)
	DrawLabel(objectCOM, objectCOM_L);
	DrawEdge(objectCOM, image, objectCOM_I);

	string  objectCOM_B_file = filepath + "\\" + infilename + "_14.0_COM_O(B).png";			//���X���P�u(�G��)
	imwrite(objectCOM_B_file, objectCOM);
	string  objectCOM_L_file = filepath + "\\" + infilename + "_14.1_COM_O(L).png";			//���X���P�u(����)
	imwrite(objectCOM_L_file, objectCOM_L);
	string  objectCOM_I_file = filepath + "\\" + infilename + "_14.2_COM_O(I).png";			//���X���P�u(�|��)
	imwrite(objectCOM_I_file, objectCOM_I);

	/*�}�B��*/

	Mat objectOpen;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	morphologyEx(objectCOM, objectOpen, MORPH_OPEN, element);

	Mat objectOpen_L, objectOpen_I;			//��X��(8UC3�B8UC3)
	DrawLabel(objectOpen, objectOpen_L);
	DrawEdge(objectOpen, image, objectOpen_I);

	string  objectOpen_B_file = filepath + "\\" + infilename + "_15.0_OPEN_O(B).png";			//�}�B��(�G��)
	imwrite(objectOpen_B_file, objectOpen);
	string  objectOpen_L_file = filepath + "\\" + infilename + "_15.1_OPEN_O(L).png";			//�}�B��(����)
	imwrite(objectOpen_L_file, objectOpen_L);
	string  objectOpen_I_file = filepath + "\\" + infilename + "_15.2_OPEN_O(I).png";			//�}�B��(�|��)
	imwrite(objectOpen_I_file, objectOpen_I);

	///*�������t��k*/

	//Mat objectWS;		//�������t��k(32SC1(BW))
	//BWWatershed(image, objectOpen, area, objectWS);

	//Mat objectWS_L, objectWS_I;		//��X��(8UC3�B8UC3)
	//DrawLabel(objectWS, objectWS_L);
	//DrawEdge(objectWS, image, objectWS_I);

	//string  objectWS_B_file = filepath + "\\" + infilename + "_16.0_WS_O(B).png";			//�������t��k(�G��)
	//imwrite(objectWS_B_file, objectWS);
	//string  objectWS_L_file = filepath + "\\" + infilename + "_16.1_WS_O(L).png";			//�������t��k(����)
	//imwrite(objectWS_L_file, objectWS_L);
	//string  objectWS_I_file = filepath + "\\" + infilename + "_16.2_WS_O(I).png";			//�������t��k(�|��)
	//imwrite(objectWS_I_file, objectWS_I);

	return 0;
}



