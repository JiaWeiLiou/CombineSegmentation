#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

/*尋找根結點*/
int findroot(int labeltable[], int label);

/*尋找連通物*/
int bwlabel(InputArray _binaryImg, OutputArray _labels, int nears);

/*創建色環*/
void makecolorwheel(vector<Scalar> &colorwheel);

/*將灰階圖片轉以色條顯示*/
void DrawColorBar(InputArray _grayImage, OutputArray _colorbarImage, int upperbound = 255, int lowerbound = 0);

/*將圖片轉以色環方向場顯示(輸入梯度場或梯度方向)*/
void DrawColorRing(InputArray _field, OutputArray _colorField);

/*將圖片轉以色環方向場顯示(輸入梯度幅值及梯度方向)*/
void DrawColorRing(InputArray _gradm, InputArray _gradd, OutputArray _colorField);

/*將圖片轉線性拉伸並以灰階值顯示*/
void DrawGrayBar(InputArray _field, OutputArray _grayField);

/*將結果以標籤顯示*/
void DrawLabel(InputArray _bwImage, OutputArray _combineLabel);

/*將結果顯示在彩色圖像上*/
void DrawEdge(InputArray _bwImage, InputArray _realImage, OutputArray _combineImage);

/*將種子點顯示在原物件上*/
void DrawSeed(InputArray _object, InputArray _objectSeed, OutputArray _combineSeed);

/*基於面的分割混合模式*/
void DivideArea(InputArray _grayImage, InputArray _mixImage, OutputArray _divideImage);

/*去除基於面的雜訊*/
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears = 4, bool BW = 0);

/*中央差分*/
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady);

/*結合水平及垂直方向梯度為梯度場*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*計算梯度幅值及方向*/
void CalculateGradient(InputArray _gradientField, OutputArray _gradm, OutputArray _gradd);

/*基於線的分割混合模式*/
void DivideLine(InputArray _gradm, InputArray _gradmblur, OutputArray _gradmDivide);

/*滯後切割*/
void HysteresisCut(InputArray _lineHT, InputArray _area, OutputArray _lineHC);

/*滯後閥值*/
void HysteresisThreshold(InputArray _gradm, OutputArray _bwLine, int upperThreshold = 150, int lowerThreshold = 50);

/*結合線與面的二值邊緣*/
// flag = 0  -> 物體為白色
// flag = 1  -> 背景為白色
void BWCombine(InputArray _area, InputArray _line, OutputArray _object);

/*反轉二值圖*/
void BWReverse(InputArray _bwImage, OutputArray _bwImageR);

/*空洞填補*/
void BWFillhole(InputArray _bwImage, OutputArray _bwFillhole);