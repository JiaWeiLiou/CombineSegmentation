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

/*判斷點的類型*/
// channel 1 put the type of points
//  0 -- None Point
//  1 -- Isolated Point
//  2 -- EndPoint 
//  3 -- Line Point
//  4 -- Bifurcation point
//  5 -- Out of Endpoint 

// channel 2 put the neighborhood direction at end of Line Point
//  0 -- None
//  1 - 8 -- Location of neighborhood point
//	+ - + - + - +
//	| 2 | 3 | 4 |
//	+ - + - + - +
//	| 1 | 0 | 5 |
//	+ - + - + - +
//	| 8 | 7 | 6 |
//	+ - + - + - +
void pointlabel(InputArray _gradm, OutputArray _labels);

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

/*圖層混合模式*/
void LayerMix(InputArray _grayImage, InputArray _blurImage, OutputArray _mixImage);

/*基於面的分割混合模式*/
void DivideArea(InputArray _grayImage, InputArray _mixImage, OutputArray _divideImage);

/*實色印疊合混合模式*/
void HardMix(InputArray _grayImage, InputArray _mixImage, OutputArray _hardmixImage);

/*去除基於面的雜訊*/
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears = 4, bool BW = 0);

/*中央差分*/
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady);

/*結合水平及垂直方向梯度為梯度場*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*計算梯度幅值及方向*/
void CalculateGradient(InputArray _gradientField, OutputArray _gradm, OutputArray _gradd);

/*梯度方向模糊*/
void BlurDirection(InputArray _gradd, OutputArray _graddblur, int blurLineSize);

/*基於線的分割混合模式*/
void DivideLine(InputArray _gradm, InputArray _gradmblur, OutputArray _gradmDivide);

/*滯後切割*/
void HysteresisCut(InputArray _lineHT, InputArray _area, OutputArray _lineHC);

/*非極大值抑制*/
void NonMaximumSuppression(InputArray _gradm, InputArray _gradd, OutputArray _gradmNMS, OutputArray _graddNMS);

/*清除分岔點點*/
void ClearBifPoint(InputArray _gradm, InputArray _gradd, OutputArray _gradmCBP, OutputArray _graddCBP);

/*清除異方向點*/
// flagD = 0  -> 無預測方向
// flagD = 1  -> 有預測方向
void ClearDifferentDirection(InputArray _gradm, InputArray _gradd, OutputArray _gradmCDD, OutputArray _graddCDD, bool flag = 0);

/*清除孤立點*/
void ClearIsoPoint(InputArray _gradm, InputArray _gradd, OutputArray _gradmCIP, OutputArray _graddCIP);

/*梯度場斷線連通*/
// startSpace -> 起始搜尋的間距
// endSpace   -> 結束搜尋的間距
// degree     -> 容忍誤差角度
// flagT = 0  -> 只搜尋端點
// flagT = 1  -> 可搜尋端點、線段點、分岔點
// flagD = 0  -> 只搜尋90度範圍
// flagD = 1  -> 可搜尋180度範圍
void ConnectLine(InputArray _gradm, InputArray _gradd, OutputArray _gradmCL, OutputArray _graddCL, int startSpace = 2, int endSpace = 5, int degree = 60, int flagT = 0, bool flagD = 0);

/*滯後閥值*/
void HysteresisThreshold(InputArray _gradm, OutputArray _bwLine, int upperThreshold = 150, int lowerThreshold = 50);

/*影像處理*/
void ImageProcess(InputArray _image, OutputArray _imageIP);

/*二值清除孤立點*/
//  iter -- 刪除次數
//  flagT = 0 -> 清除孤立點
//  flagT = 1 -> 清除端點
void BWClearIsoPoint(InputArray _bwLine, OutputArray _bwLineCP, int border = 0, int iter = 1, bool flagT = 0);

/*二值斷線連通*/
// startSpace -> 起始搜尋的間距
// endSpace   -> 結束搜尋的間距
// degree     -> 容忍誤差角度
// flagT = 0  -> 只搜尋端點
// flagT = 1  -> 可搜尋端點、線段點、分岔點
// flagD = 0  -> 只搜尋90度範圍
// flagD = 1  -> 可搜尋180度範圍
void BWConnectLine(InputArray _gradm, InputArray _gradd, InputArray _bwLine, OutputArray _gradmCL, OutputArray _graddCL, OutputArray _bwLineCL, int startSpace, int endSpace, int degree, int flagT, bool flagD);

/*結合線與面的二值邊緣*/
// flag = 0  -> 物體為白色
// flag = 1  -> 背景為白色
void BWCombine(InputArray _area, InputArray _line, OutputArray _object);

/*反轉二值圖*/
void BWReverse(InputArray _bwImage, OutputArray _bwImageR);

/*分水嶺演算法*/
void BWWatershed(InputArray _srcImage, InputArray _bwSeed, InputArray _bwObject, OutputArray _bwWatershed);

/*分水嶺轉換*/
void WatershedTransform(InputArray _srcImage, OutputArray _bwWatershed);