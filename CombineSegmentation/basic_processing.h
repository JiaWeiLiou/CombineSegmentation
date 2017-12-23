#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

/*�M��ڵ��I*/
int findroot(int labeltable[], int label);

/*�M��s�q��*/
int bwlabel(InputArray _binaryImg, OutputArray _labels, int nears);

/*�Ыئ���*/
void makecolorwheel(vector<Scalar> &colorwheel);

/*�N�Ƕ��Ϥ���H������*/
void DrawColorBar(InputArray _grayImage, OutputArray _colorbarImage, int upperbound = 255, int lowerbound = 0);

/*�N�Ϥ���H������V�����(��J��׳��α�פ�V)*/
void DrawColorRing(InputArray _field, OutputArray _colorField);

/*�N�Ϥ���H������V�����(��J��״T�Ȥα�פ�V)*/
void DrawColorRing(InputArray _gradm, InputArray _gradd, OutputArray _colorField);

/*�N�Ϥ���u�ʩԦ��åH�Ƕ������*/
void DrawGrayBar(InputArray _field, OutputArray _grayField);

/*�N���G�H�������*/
void DrawLabel(InputArray _bwImage, OutputArray _combineLabel);

/*�N���G��ܦb�m��Ϲ��W*/
void DrawEdge(InputArray _bwImage, InputArray _realImage, OutputArray _combineImage);

/*�N�ؤl�I��ܦb�쪫��W*/
void DrawSeed(InputArray _object, InputArray _objectSeed, OutputArray _combineSeed);

/*��󭱪����βV�X�Ҧ�*/
void DivideArea(InputArray _grayImage, InputArray _mixImage, OutputArray _divideImage);

/*�h����󭱪����T*/
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears = 4, bool BW = 0);

/*�����t��*/
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady);

/*���X�����Ϋ�����V��׬���׳�*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*�p���״T�ȤΤ�V*/
void CalculateGradient(InputArray _gradientField, OutputArray _gradm, OutputArray _gradd);

/*���u�����βV�X�Ҧ�*/
void DivideLine(InputArray _gradm, InputArray _gradmblur, OutputArray _gradmDivide);

/*�������*/
void HysteresisCut(InputArray _lineHT, InputArray _area, OutputArray _lineHC);

/*����֭�*/
void HysteresisThreshold(InputArray _gradm, OutputArray _bwLine, int upperThreshold = 150, int lowerThreshold = 50);

/*���X�u�P�����G����t*/
// flag = 0  -> ���鬰�զ�
// flag = 1  -> �I�����զ�
void BWCombine(InputArray _area, InputArray _line, OutputArray _object);

/*����G�ȹ�*/
void BWReverse(InputArray _bwImage, OutputArray _bwImageR);

/*�Ŭ}���*/
void BWFillhole(InputArray _bwImage, OutputArray _bwFillhole);