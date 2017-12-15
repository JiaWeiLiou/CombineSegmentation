#include "stdafx.h"
#include "basic_processing.h"

/*�ͦ��G��������*/
Mat gaussian_kernal(int size, int sigma)
{
	if (size % 2 == 0) { --size; }
	int x = (size - 1) / 2;
	Mat kernal(size, size, CV_32FC1);

	if (sigma = -1) { sigma = 0.3*((size - 1)*0.5 - 1) + 0.8; }

	float s2 = 2.0 * sigma * sigma;
	for (int i = -x; i <= x; i++)
	{
		int m = i + x;
		for (int j = -x; j <= x; j++)
		{
			int n = j + x;
			float v = exp(-(1.0*i*i + 1.0*j*j) / s2);
			kernal.ptr<float>(m)[n] = v;
		}
	}
	Scalar all = sum(kernal);
	Mat gaussK;
	kernal.convertTo(gaussK, CV_32FC1, (1 / all[0]));
	return gaussK;
}

/*�M��ڵ��I*/
int findroot(int labeltable[], int label)
{
	int x = label;
	while (x != labeltable[x])
		x = labeltable[x];
	return x;
}

/*�M��s�q��*/
int bwlabel(InputArray _binaryImg, OutputArray _labels, int nears)
{
	Mat binaryImg = _binaryImg.getMat();
	CV_Assert(binaryImg.type() == CV_8UC1);

	_labels.create(binaryImg.size(), CV_32SC1);
	Mat labels = _labels.getMat();
	labels = Scalar(0);

	if (nears != 4 && nears != 6 && nears != 8)
		nears = 8;

	int nobj = 0;    // number of objects found in image  
	int *labeltable = new int[binaryImg.rows*binaryImg.cols];		// initialize label table with zero  
	memset(labeltable, 0, binaryImg.rows*binaryImg.cols * sizeof(int));
	int ntable = 0;

	//	labeling scheme
	//	+ - + - + - +
	//	| D | C | E |
	//	+ - + - + - +
	//	| B | A |   |
	//	+ - + - + - +
	//	A is the center pixel of a neighborhood.In the 3 versions of connectedness :
	//	4 : A connects to B and C
	//	6 : A connects to B, C, and D
	//	8 : A connects to B, C, D, and E


	for (int i = 0; i < binaryImg.rows; i++)
		for (int j = 0; j < binaryImg.cols; j++)
			if (binaryImg.at<uchar>(i, j) == 255)   // if A is an object  
			{
				// get the neighboring labels B, C, D, and E
				int B, C, D, E;

				if (j == 0) { B = 0; }
				else { B = findroot(labeltable, labels.at<int>(i, j - 1)); }

				if (i == 0) { C = 0; }
				else { C = findroot(labeltable, labels.at<int>(i - 1, j)); }

				if (i == 0 || j == 0) { D = 0; }
				else { D = findroot(labeltable, labels.at<int>(i - 1, j - 1)); }

				if (i == 0 || j == binaryImg.cols - 1) { E = 0; }
				else { E = findroot(labeltable, labels.at<int>(i - 1, j + 1)); }

				if (nears == 4)		// apply 4 connectedness  
				{
					if (B && C)	// B and C are labeled  
					{
						if (B == C) { labels.at<int>(i, j) = B; }
						else
						{
							labeltable[C] = B;
							labels.at<int>(i, j) = B;
						}
					}
					else if (B) { labels.at<int>(i, j) = B; }            // B is object but C is not  
					else if (C) { labels.at<int>(i, j) = C; }            // C is object but B is not  
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; }	// B, C, D not object - new object label and put into table  
				}
				else if (nears == 6)	// apply 6 connected ness  
				{
					if (D) { labels.at<int>(i, j) = D; }              // D object, copy label and move on  
					else if (B && C)		// B and C are labeled  
					{
						if (B == C) { labels.at<int>(i, j) = B; }
						else
						{
							int tlabel = B < C ? B : C;
							labeltable[B] = tlabel;
							labeltable[C] = tlabel;
							labels.at<int>(i, j) = tlabel;
						}
					}
					else if (B) { labels.at<int>(i, j) = B; }        // B is object but C is not  	
					else if (C) { labels.at<int>(i, j) = C; }        // C is object but B is not  
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; } 	// B, C, D not object - new object label and put into table
				}
				else if (nears == 8)	// apply 8 connectedness  
				{
					if (B || C || D || E)
					{
						int tlabel;
						if (B) { tlabel = B; }
						else if (C) { tlabel = C; }
						else if (D) { tlabel = D; }
						else if (E) { tlabel = E; }

						labels.at<int>(i, j) = tlabel;

						if (B && B != tlabel) { labeltable[B] = tlabel; }
						if (C && C != tlabel) { labeltable[C] = tlabel; }
						if (D && D != tlabel) { labeltable[D] = tlabel; }
						if (E && E != tlabel) { labeltable[E] = tlabel; }
					}
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; } // label and put into table
				}
			}
			else { labels.at<int>(i, j) = 0; }	// A is not an object so leave it

			for (int i = 0; i <= ntable; i++)
				labeltable[i] = findroot(labeltable, i);	// consolidate component table  

			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					labels.at<int>(i, j) = labeltable[labels.at<int>(i, j)];	// run image through the look-up table  

			// count up the objects in the image  
			for (int i = 0; i <= ntable; i++)
				labeltable[i] = 0;		//clear all table label
			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					++labeltable[labels.at<int>(i, j)];		//calculate all label numbers

			labeltable[0] = 0;		//clear 0 label
			for (int i = 1; i <= ntable; i++)
				if (labeltable[i] > 0)
					labeltable[i] = ++nobj;	// number the objects from 1 through n objects  and reset label table

			// run through the look-up table again  
			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					labels.at<int>(i, j) = labeltable[labels.at<int>(i, j)];

			delete[] labeltable;
			labeltable = nullptr;
			return nobj;
}

/*�P�_�I������*/
void pointlabel(InputArray _gradm, OutputArray _labels)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	_labels.create(gradm.size(), CV_8UC2);
	Mat labels = _labels.getMat();

	Mat gradmRef;
	copyMakeBorder(gradm, gradmRef, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			vector<char> nearPoint;
			int num = 0;

			if (gradmRef.at<uchar>(i + 1, j + 1) != 0)
			{
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j + 1));		//0
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j));			//1
				nearPoint.push_back(gradmRef.at<uchar>(i, j));				//2
				nearPoint.push_back(gradmRef.at<uchar>(i, j + 1));			//3
				nearPoint.push_back(gradmRef.at<uchar>(i, j + 2));			//4
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j + 2));		//5
				nearPoint.push_back(gradmRef.at<uchar>(i + 2, j + 2));		//6
				nearPoint.push_back(gradmRef.at<uchar>(i + 2, j + 1));		//7
				nearPoint.push_back(gradmRef.at<uchar>(i + 2, j));			//8
				num = 9 - count(nearPoint.begin(), nearPoint.end(), 0);

				if (num == 1)		//Isolated Point
				{
					labels.at<Vec2b>(i, j)[0] = 1;
					labels.at<Vec2b>(i, j)[1] = 0;
				}
				else if (num == 2)		//End of Line Point 
				{
					labels.at<Vec2b>(i, j)[0] = 2;
					for (int k = 1; k <= 8; ++k)
						if (nearPoint[k] != 0)
						{
							labels.at<Vec2b>(i, j)[1] = k;
							break;
						}
				}
				else if (num == 3)
				{
					for (int k = 1; k <= 8; ++k)
						if (nearPoint[(k % 9)] != 0 && nearPoint[(k % 8) + 1] != 0) //1,2�B2,3�B...�B8,1   End of Line Point 
						{
							labels.at<Vec2b>(i, j)[0] = 2;
							if (k % 2 == 1)		//�u�s���﨤�u����V
								labels.at<Vec2b>(i, j)[1] = k + 1;
							else
								labels.at<Vec2b>(i, j)[1] = k;
							break;
						}
						else	//Line Point
						{
							labels.at<Vec2b>(i, j)[0] = 3;
							labels.at<Vec2b>(i, j)[1] = 0;
						}
				}
				else		//Line Point
				{
					labels.at<Vec2b>(i, j)[0] = 3;
					labels.at<Vec2b>(i, j)[1] = 0;
				}
			}
			else
			{
				labels.at<Vec2b>(i, j)[0] = 0;
				labels.at<Vec2b>(i, j)[1] = 0;
			}
		}
}

/*�Ыئ���*/
void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;	//����(Red)     �ܶ���(Yellow)
	int YG = 15;	//����(Yellow)  �ܺ��(Green)
	int GC = 15;	//���(Green)   �ܫC��(Cyan)
	int CB = 15;	//�C��(Cyan)    ���Ŧ�(Blue)
	int BM = 15;	//�Ŧ�(Blue)    �ܬv��(Magenta)
	int MR = 15;	//�v��(Magenta) �ܬ���(Red)

	for (int i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (int i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (int i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (int i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (int i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (int i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

/*�Ыئ��*/
void makecolorbar(vector<Scalar> &colorbar)
{
	vector<Scalar> maincolor;

	maincolor.push_back(Scalar(127.5, 0, 0));      //�`����
	maincolor.push_back(Scalar(255, 0, 0));		   //����
	maincolor.push_back(Scalar(255, 127.5, 0));	   //����ܶ���
	maincolor.push_back(Scalar(255, 255, 0));	   //����
	maincolor.push_back(Scalar(127.5, 255, 0));	   //����ܺ��
	maincolor.push_back(Scalar(0, 255, 0));		   //���
	maincolor.push_back(Scalar(0, 255, 127.5));	   //���ܫC��
	maincolor.push_back(Scalar(0, 255, 255));	   //�C��
	maincolor.push_back(Scalar(0, 127.5, 255));	   //�C����Ŧ�
	maincolor.push_back(Scalar(0, 0, 255));		   //�Ŧ�
	maincolor.push_back(Scalar(0, 0, 127.5));      //�`�Ŧ�

	int layer = 15;		//�U���h���ܶ��h��

	for (int i = 0; i < maincolor.size() - 1; i++)
	{
		for (int j = 0; j < layer; j++)
		{
			double r = maincolor[i][0] + (maincolor[i + 1][0] - maincolor[i][0]) / layer*j;
			double g = maincolor[i][1] + (maincolor[i + 1][1] - maincolor[i][1]) / layer*j;
			double b = maincolor[i][2] + (maincolor[i + 1][2] - maincolor[i][2]) / layer*j;
			colorbar.push_back(Scalar(r, g, b));
		}
	}
}

/*�N�Ƕ��Ϥ���H������*/
void DrawColorBar(InputArray _grayImage, OutputArray _colorbarImage, int upperbound, int lowerbound)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	_colorbarImage.create(grayImage.size(), CV_8UC3);
	Mat colorbarImage = _colorbarImage.getMat();

	static vector<Scalar> colorbar; //Scalar i,g,b  
	if (colorbar.empty()) { makecolorbar(colorbar); }

	int maxrad = upperbound - lowerbound + 1;

	for (int i = 0; i < colorbarImage.rows; ++i)
		for (int j = 0; j < colorbarImage.cols; ++j)
		{
			uchar *data = colorbarImage.data + colorbarImage.step[0] * i + colorbarImage.step[1] * j;

			float fk = (1 - (float)grayImage.at<uchar>(i, j) / (float)maxrad) * (colorbar.size() - 1);  //�p��ǫ׭ȹ��������ަ�m
			int k0 = floor(fk);
			int k1 = ceil(fk);
			float f = fk - k0;

			float col0 = 0.0f;
			float col1 = 0.0f;
			float col = 0.0f;
			for (int b = 0; b < 3; b++)
			{
				col0 = colorbar[k0][b] / 255.0f;
				col1 = colorbar[k1][b] / 255.0f;
				col = (1 - f) * col0 + f * col1;
				data[2 - b] = (int)(255.0f * col);
			}
		}
}

/*�N�Ϥ���H������V�����(��J��׳��α�פ�V)*/
void DrawColorRing(InputArray _field, OutputArray _colorField)
{
	Mat field;
	Mat temp = _field.getMat();
	if (temp.type() == CV_16SC2) {
		temp.convertTo(field, CV_32FC2);
	}
	else {
		field = _field.getMat();
	}

	_colorField.create(field.size(), CV_8UC3);
	Mat colorField = _colorField.getMat();

	static vector<Scalar> colorwheel; //Scalar i,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:  
	int maxrad = -1;

	if (field.type() == CV_32FC1)
	{
		maxrad = 255;		//�u����פ�V�L��״T��

		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;

				if (field.at<float>(i, j) == -1000.0f)		//�ΥH��ܵL��פ�V
				{
					for (int b = 0; b < 3; b++)
					{
						data[2 - b] = 255;
					}
				}
				else
				{
					float rad = maxrad;

					float angle = field.at<float>(i, j) / CV_PI;    //��쬰-1��+1
					float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //�p�⨤�׹��������ަ�m
					int k0 = (int)fk;
					int k1 = (k0 + 1) % colorwheel.size();
					float f = fk - k0;

					float col0 = 0.0f;
					float col1 = 0.0f;
					float col = 0.0f;
					for (int b = 0; b < 3; b++)
					{
						col0 = colorwheel[k0][b] / 255.0f;
						col1 = colorwheel[k1][b] / 255.0f;
						col = (1 - f) * col0 + f * col1;
						if (rad <= 1)
							col = 1 - rad * (1 - col); // increase saturation with radius  
						else
							col = col;  //out of range
						data[2 - b] = (int)(255.0f * col);
					}
				}
			}
	}
	else if (field.type() == CV_32FC2)
	{
		// Find max flow to normalize fx and fy  
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				Vec2f field_at_point = field.at<Vec2f>(i, j);
				float fx = field_at_point[0];
				float fy = field_at_point[1];
				float rad = sqrt(fx * fx + fy * fy);
				maxrad = maxrad > rad ? maxrad : rad;
			}

		maxrad = maxrad / 2;		//�[�`��ܵ��G(�i��������)

		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;
				Vec2f field_at_point = field.at<Vec2f>(i, j);

				float fx = field_at_point[0];
				float fy = field_at_point[1];

				float rad = sqrt(fx * fx + fy * fy) / maxrad;

				float angle = atan2(fy, fx) / CV_PI;    //��쬰-1��+1
				float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //�p�⨤�׹��������ަ�m
				int k0 = (int)fk;
				int k1 = (k0 + 1) % colorwheel.size();
				float f = fk - k0;

				float col0 = 0.0f;
				float col1 = 0.0f;
				float col = 0.0f;
				for (int b = 0; b < 3; b++)
				{
					col0 = colorwheel[k0][b] / 255.0f;
					col1 = colorwheel[k1][b] / 255.0f;
					col = (1 - f) * col0 + f * col1;
					if (rad <= 1)
						col = 1 - rad * (1 - col); // increase saturation with radius  
					else
						col = col;  //out of range
					data[2 - b] = (int)(255.0f * col);
				}
			}
	}
}

/*�N�Ϥ���H������V�����(��J��״T�Ȥα�פ�V)*/
void DrawColorRing(InputArray _gradm, InputArray _gradd, OutputArray _colorField)
{
	Mat gradm;
	Mat temp1 = _gradm.getMat();
	if (temp1.type() == CV_8UC1) { temp1.convertTo(gradm, CV_32FC1); }
	else { gradm = _gradm.getMat(); }

	Mat gradd;
	Mat temp2 = _gradd.getMat();
	if (temp2.type() == CV_8UC1) { temp2.convertTo(gradd, CV_32FC1); }
	else { gradd = _gradd.getMat(); }

	_colorField.create(gradm.size(), CV_8UC3);
	Mat colorField = _colorField.getMat();

	static vector<Scalar> colorwheel; //Scalar i,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:  
	int maxrad = -1;

	// Find max flow to normalize fx and fy  
	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			float rad = gradm.at<float>(i, j);
			maxrad = maxrad > rad ? maxrad : rad;
		}

	maxrad = maxrad / 2;		//�[�`��ܵ��G(�i��������)

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;

			if (gradd.at<float>(i, j) == -1000.0f)		//�ΥH��ܵL��פ�V
			{
				for (int b = 0; b < 3; b++)
				{
					data[2 - b] = 255;
				}
			}
			else
			{
				float rad = gradm.at<float>(i, j) / maxrad;

				float angle = gradd.at<float>(i, j) / CV_PI;    //��쬰-1��+1
				float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //�p�⨤�׹��������ަ�m
				int k0 = (int)fk;
				int k1 = (k0 + 1) % colorwheel.size();
				float f = fk - k0;

				float col0 = 0.0f;
				float col1 = 0.0f;
				float col = 0.0f;
				for (int b = 0; b < 3; b++)
				{
					col0 = colorwheel[k0][b] / 255.0f;
					col1 = colorwheel[k1][b] / 255.0f;
					col = (1 - f) * col0 + f * col1;
					if (rad <= 1)
						col = 1 - rad * (1 - col); // increase saturation with radius  
					else
						col = col;  //out of range
					data[2 - b] = (int)(255.0f * col);
				}
			}
		}
}

/*�N�Ϥ���u�ʩԦ��åH�Ƕ������*/
void DrawGrayBar(InputArray _field, OutputArray _grayField)
{
	Mat field;
	Mat temp = _field.getMat();
	if (temp.type() == CV_16SC2) { temp.convertTo(field, CV_32FC2); }
	else if (temp.type() == CV_16SC1) { temp.convertTo(field, CV_32FC1); }
	else { field = _field.getMat(); }

	_grayField.create(field.size(), CV_8UC1);
	Mat grayField = _grayField.getMat();



	// determine motion range:  
	float maxvalue = 0;

	if (field.type() == CV_8UC1)
	{
		// Find max value
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
				maxvalue = maxvalue > field.at<uchar>(i, j) ? maxvalue : field.at<uchar>(i, j);

		//linear stretch to 255
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
				grayField.at<uchar>(i, j) = ((float)field.at<uchar>(i, j) / maxvalue) * 255;
	}
	else if (field.type() == CV_32FC1)
	{
		// Find max value
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
				maxvalue = maxvalue > abs(field.at<float>(i, j)) ? maxvalue : abs(field.at<float>(i, j));

		//linear stretch to 255
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
				grayField.at<uchar>(i, j) = ((float)abs(field.at<float>(i, j)) / maxvalue) * 255;
	}
	else if (field.type() == CV_32FC2)
	{
		// Find max value
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				float fx = field.at<Vec2f>(i, j)[0];
				float fy = field.at<Vec2f>(i, j)[1];
				float absvalue = sqrt(fx * fx + fy * fy);
				maxvalue = maxvalue > absvalue ? maxvalue : absvalue;
			}

		//linear stretch to 255
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				float fx = field.at<Vec2f>(i, j)[0];
				float fy = field.at<Vec2f>(i, j)[1];
				float absvalue = sqrt(fx * fx + fy * fy);
				grayField.at<uchar>(i, j) = (absvalue / maxvalue) * 255;
			}
	}
}

/*�N���G�H�������*/
void DrawLabel(InputArray _bwImage, OutputArray _combineLabel)
{
	Mat bwImage = _bwImage.getMat();
	CV_Assert(bwImage.type() == CV_8UC1);

	_combineLabel.create(bwImage.size(), CV_8UC3);
	Mat combineLabel = _combineLabel.getMat();

	Mat object(bwImage.size(), CV_8UC1);
	for (int i = 0; i < bwImage.rows; i++)
		for (int j = 0; j < bwImage.cols; j++)
			if (bwImage.at<uchar>(i, j) == 255) { object.at<uchar>(i, j) = 255; }
			else { object.at<uchar>(i, j) = 0; }

			Mat labels;
			int objectNum = bwlabel(object, labels, 4);

			RNG rng(12345);
			vector<Scalar> color;

			color.push_back(Scalar(0, 0, 0));
			for (int i = 1; i <= objectNum; i++) { color.push_back(Scalar(rng.uniform(20, 255), rng.uniform(20, 255), rng.uniform(20, 255))); }

			for (int i = 0; i < bwImage.rows; i++)
				for (int j = 0; j < bwImage.cols; j++)
				{
					combineLabel.at<Vec3b>(i, j)[0] = color[labels.at<int>(i, j)][0];
					combineLabel.at<Vec3b>(i, j)[1] = color[labels.at<int>(i, j)][1];
					combineLabel.at<Vec3b>(i, j)[2] = color[labels.at<int>(i, j)][2];
				}
}

/*�N��t�������G��ܦb�m��Ϲ��W*/
void DrawEdge(InputArray _bwImage, InputArray _realImage, OutputArray _combineImage)
{
	Mat bwImage = _bwImage.getMat();
	CV_Assert(bwImage.type() == CV_8UC1);

	Mat realImage = _realImage.getMat();

	if (realImage.type() == CV_8UC1)
	{
		_combineImage.create(realImage.size(), CV_8UC3);
		Mat combineImage = _combineImage.getMat();

		for (int i = 0; i < bwImage.rows; i++)
			for (int j = 0; j < bwImage.cols; j++)
			{
				if (bwImage.at<uchar>(i, j) == 0)
				{
					combineImage.at<Vec3b>(i, j)[0] = 255;
					combineImage.at<Vec3b>(i, j)[1] = 0;
					combineImage.at<Vec3b>(i, j)[2] = 0;
				}
				else
				{
					combineImage.at<Vec3b>(i, j)[0] = realImage.at<uchar>(i, j);
					combineImage.at<Vec3b>(i, j)[1] = realImage.at<uchar>(i, j);
					combineImage.at<Vec3b>(i, j)[2] = realImage.at<uchar>(i, j);
				}
			}
	}
	else if (realImage.type() == CV_8UC3)
	{
		_combineImage.create(realImage.size(), CV_8UC3);
		Mat combineImage = _combineImage.getMat();

		for (int i = 0; i < bwImage.rows; i++)
			for (int j = 0; j < bwImage.cols; j++)
			{
				if (bwImage.at<uchar>(i, j) == 0)
				{
					combineImage.at<Vec3b>(i, j)[0] = 255;
					combineImage.at<Vec3b>(i, j)[1] = 0;
					combineImage.at<Vec3b>(i, j)[2] = 0;
				}
				else
				{
					combineImage.at<Vec3b>(i, j)[0] = realImage.at<Vec3b>(i, j)[0];
					combineImage.at<Vec3b>(i, j)[1] = realImage.at<Vec3b>(i, j)[1];
					combineImage.at<Vec3b>(i, j)[2] = realImage.at<Vec3b>(i, j)[2];
				}
			}
	}
}

/*�ϼh�V�X�Ҧ�*/
void LayerMix(InputArray _grayImage, InputArray _blurImage, OutputArray _mixImage)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	Mat blurImage = _blurImage.getMat();
	CV_Assert(blurImage.type() == CV_8UC1);

	_mixImage.create(grayImage.size(), CV_8UC1);
	Mat mixImage = _mixImage.getMat();

	double divide = 0;
	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j)
		{
			divide = (double)grayImage.at<uchar>(i, j) / (double)blurImage.at<uchar>(i, j) > 1 ? 255 : ((double)grayImage.at<uchar>(i, j) / (double)blurImage.at<uchar>(i, j)) * 255.0;
			mixImage.at<uchar>(i, j) = divide + (double)blurImage.at<uchar>(i, j) < 255.0 ? 0 : 255;
		}
}

/*��󭱪����βV�X�Ҧ�*/
void DivideArea(InputArray _grayImage, InputArray _mixImage, OutputArray _divideImage)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	Mat mixImage = _mixImage.getMat();
	CV_Assert(mixImage.type() == CV_8UC1);

	_divideImage.create(grayImage.size(), CV_8UC1);
	Mat divideImage = _divideImage.getMat();

	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j)
			divideImage.at<uchar>(i, j) = (double)grayImage.at<uchar>(i, j) / (double)mixImage.at<uchar>(i, j) > 1 ? 255 : ((double)grayImage.at<uchar>(i, j) / (double)mixImage.at<uchar>(i, j)) * 255;
}

/*���L�|�X�V�X�Ҧ�*/
void HardMix(InputArray _grayImage, InputArray _mixImage, OutputArray _hardmixImage)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	Mat mixImage = _mixImage.getMat();
	CV_Assert(mixImage.type() == CV_8UC1);

	_hardmixImage.create(grayImage.size(), CV_8UC1);
	Mat hardmixImage = _hardmixImage.getMat();

	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j)
			hardmixImage.at<uchar>(i, j) = grayImage.at<uchar>(i, j) + mixImage.at<uchar>(i, j) < 255 ? 0 : 255;
}

/*�h�����T*/
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears, bool BW)
{
	Mat binaryImg = _binaryImg.getMat();
	CV_Assert(binaryImg.type() == CV_8UC1);

	_clearAreaImage.create(binaryImg.size(), CV_8UC1);
	Mat clearAreaImage = _clearAreaImage.getMat();

	Mat labels(binaryImg.size(), CV_32SC1, Scalar(0));

	// 0 claer black noise
	// 1 clear wite noise
	if (BW == 0)
	{
		for (int i = 0; i < binaryImg.rows; i++)
			for (int j = 0; j < binaryImg.cols; j++)
				if (binaryImg.at<uchar>(i, j) == 255) { binaryImg.at<uchar>(i, j) = 0; }
				else { binaryImg.at<uchar>(i, j) = 255; }
	}

	if (nears != 4 && nears != 6 && nears != 8)
		nears = 8;

	int nobj = 0;    // number of objects found in image  

	int *labeltable = new int[binaryImg.rows*binaryImg.cols];		// initialize label table with zero  
	memset(labeltable, 0, binaryImg.rows*binaryImg.cols * sizeof(int));
	int ntable = 0;

	//	labeling scheme
	//	+ - + - + - +
	//	| D | C | E |
	//	+ - + - + - +
	//	| B | A |   |
	//	+ - + - + - +
	//	A is the center pixel of a neighborhood.In the 3 versions of connectedness :
	//	4 : A connects to B and C
	//	6 : A connects to B, C, and D
	//	8 : A connects to B, C, D, and E


	for (int i = 0; i < binaryImg.rows; i++)
		for (int j = 0; j < binaryImg.cols; j++)
			if (binaryImg.at<uchar>(i, j) == 255)   // if A is an object  
			{
				// get the neighboring labels B, C, D, and E
				int B, C, D, E;

				if (j == 0) { B = 0; }
				else { B = findroot(labeltable, labels.at<int>(i, j - 1)); }

				if (i == 0) { C = 0; }
				else { C = findroot(labeltable, labels.at<int>(i - 1, j)); }

				if (i == 0 || j == 0) { D = 0; }
				else { D = findroot(labeltable, labels.at<int>(i - 1, j - 1)); }

				if (i == 0 || j == binaryImg.cols - 1) { E = 0; }
				else { E = findroot(labeltable, labels.at<int>(i - 1, j + 1)); }

				if (nears == 4)		// apply 4 connectedness  
				{
					if (B && C)	// B and C are labeled  
					{
						if (B == C) { labels.at<int>(i, j) = B; }
						else
						{
							labeltable[C] = B;
							labels.at<int>(i, j) = B;
						}
					}
					else if (B) { labels.at<int>(i, j) = B; }            // B is object but C is not  
					else if (C) { labels.at<int>(i, j) = C; }            // C is object but B is not  
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; }	// B, C, D not object - new object label and put into table  
				}
				else if (nears == 6)	// apply 6 connected ness  
				{
					if (D) { labels.at<int>(i, j) = D; }              // D object, copy label and move on  
					else if (B && C)		// B and C are labeled  
					{
						if (B == C) { labels.at<int>(i, j) = B; }
						else
						{
							int tlabel = B < C ? B : C;
							labeltable[B] = tlabel;
							labeltable[C] = tlabel;
							labels.at<int>(i, j) = tlabel;
						}
					}
					else if (B) { labels.at<int>(i, j) = B; }        // B is object but C is not  	
					else if (C) { labels.at<int>(i, j) = C; }        // C is object but B is not  
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; } 	// B, C, D not object - new object label and put into table
				}
				else if (nears == 8)	// apply 8 connectedness  
				{
					if (B || C || D || E)
					{
						int tlabel;
						if (B) { tlabel = B; }
						else if (C) { tlabel = C; }
						else if (D) { tlabel = D; }
						else if (E) { tlabel = E; }

						labels.at<int>(i, j) = tlabel;

						if (B && B != tlabel) { labeltable[B] = tlabel; }
						if (C && C != tlabel) { labeltable[C] = tlabel; }
						if (D && D != tlabel) { labeltable[D] = tlabel; }
						if (E && E != tlabel) { labeltable[E] = tlabel; }
					}
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; } // label and put into table
				}
			}
			else { labels.at<int>(i, j) = 0; }	// A is not an object so leave it

			for (int i = 0; i <= ntable; i++)
				labeltable[i] = findroot(labeltable, i);	// consolidate component table  

			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					labels.at<int>(i, j) = labeltable[labels.at<int>(i, j)];	// run image through the look-up table  

			// count up the objects in the image  
			for (int i = 0; i <= ntable; i++)
				labeltable[i] = 0;		//clear all table label
			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					++labeltable[labels.at<int>(i, j)];		//calculate all label numbers

			labeltable[0] = 0;		//clear 0 label
			for (int i = 1; i <= ntable; i++)
				if (labeltable[i] > noise) { labeltable[i] = 255; }	// number the objects from 1 through n objects  and reset label table
				else { labeltable[i] = 0; }

				// run through the look-up table again  
				for (int i = 0; i < binaryImg.rows; i++)
					for (int j = 0; j < binaryImg.cols; j++)
						clearAreaImage.at<uchar>(i, j) = labeltable[labels.at<int>(i, j)];

				delete[] labeltable;
				labeltable = nullptr;

				if (BW == 0)
				{
					for (int i = 0; i < clearAreaImage.rows; i++)
						for (int j = 0; j < clearAreaImage.cols; j++)
						{
							if (binaryImg.at<uchar>(i, j) == 255) { binaryImg.at<uchar>(i, j) = 0; }
							else { binaryImg.at<uchar>(i, j) = 255; }

							if (clearAreaImage.at<uchar>(i, j) == 255) { clearAreaImage.at<uchar>(i, j) = 0; }
							else { clearAreaImage.at<uchar>(i, j) = 255; }
						}
				}
}

/*�����t��*/
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady) {

	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	_gradx.create(grayImage.size(), CV_16SC1);
	Mat gradx = _gradx.getMat();

	_grady.create(grayImage.size(), CV_16SC1);
	Mat grady = _grady.getMat();

	Mat grayImageRef;
	copyMakeBorder(grayImage, grayImageRef, 1, 1, 1, 1, BORDER_REPLICATE);
	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j) {
			gradx.at<short>(i, j) = ((float)-grayImageRef.at<uchar>(i + 1, j) + (float)grayImageRef.at<uchar>(i + 1, j + 2))*0.5;
			grady.at<short>(i, j) = ((float)-grayImageRef.at<uchar>(i, j + 1) + (float)grayImageRef.at<uchar>(i + 2, j + 1))*0.5;
		}
}

/*���X�����Ϋ�����V��׬���׳�*/
void GradientField(InputArray _gradx, InputArray _grady, OutputArray _gradf) {

	Mat gradx = _gradx.getMat();
	CV_Assert(gradx.type() == CV_16SC1);

	Mat grady = _grady.getMat();
	CV_Assert(grady.type() == CV_16SC1);

	_gradf.create(grady.rows, gradx.cols, CV_16SC2);
	Mat gradf = _gradf.getMat();

	for (int i = 0; i < grady.rows; ++i)
		for (int j = 0; j < gradx.cols; ++j)
		{
			gradf.at<Vec2s>(i, j)[0] = gradx.at<short>(i, j);
			gradf.at<Vec2s>(i, j)[1] = grady.at<short>(i, j);
		}

}

/*�p���״T�ȤΤ�V*/
void CalculateGradient(InputArray _gradf, OutputArray _gradm, OutputArray _gradd)
{
	Mat gradf = _gradf.getMat();
	CV_Assert(gradf.type() == CV_16SC2);

	_gradm.create(gradf.size(), CV_8UC1);
	Mat gradm = _gradm.getMat();

	_gradd.create(gradf.size(), CV_32FC1);
	Mat gradd = _gradd.getMat();

	for (int i = 0; i < gradf.rows; ++i)
		for (int j = 0; j < gradf.cols; ++j)
		{
			short x = gradf.at<Vec2s>(i, j)[0];
			short y = gradf.at<Vec2s>(i, j)[1];
			gradm.at<uchar>(i, j) = sqrt(x*x + y*y);

			if (x == 0 && y == 0) { gradd.at<float>(i, j) = -1000.0f; }	//�ΥH��ܵL��פ�V
			else { gradd.at<float>(i, j) = atan2(y, x); }
		}
}

/*��פ�V�ҽk*/
void BlurDirection(InputArray _gradd, OutputArray _graddblur, int blurLineSize)
{
	Mat gradd = _gradd.getMat();
	CV_Assert(gradd.type() == CV_32FC1);

	_graddblur.create(gradd.size(), CV_32FC1);
	Mat graddblur = _graddblur.getMat();

	if (blurLineSize % 2 == 0) { --blurLineSize; }

	int x = (blurLineSize - 1) / 2;

	Mat gaussianKernel = gaussian_kernal(blurLineSize, -1);

	Mat graddRef;
	copyMakeBorder(gradd, graddRef, x, x, x, x, BORDER_CONSTANT, Scalar(-1000.0f));

	for (int i = 0; i < gradd.rows; ++i)
		for (int j = 0; j < gradd.cols; ++j)
			if (gradd.at<float>(i, j) != -1000.0f)
			{
				float sinsum = 0.0f;
				float cossum = 0.0f;
				int num = 0;

				for (int ic = i, ig = 0; ic <= i + 2 * x; ++ic, ++ig)
					for (int jc = j, jg = 0; jc <= j + 2 * x; ++jc, ++jg)
						if (graddRef.at<float>(ic, jc) != -1000.0f)
						{
							sinsum += sin(graddRef.at<float>(ic, jc))*gaussianKernel.at<float>(ig, jg);
							cossum += cos(graddRef.at<float>(ic, jc))*gaussianKernel.at<float>(ig, jg);
							++num;
						}
				graddblur.at<float>(i, j) = atan2(sinsum, cossum);
			}
			else { graddblur.at<float>(i, j) = -1000.0f; }
}

/*���u�����βV�X�Ҧ�*/
void DivideLine(InputArray _gradm, InputArray _gradmblur, OutputArray _gradmDivide)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradmblur = _gradmblur.getMat();
	CV_Assert(gradmblur.type() == CV_8UC1);

	_gradmDivide.create(gradm.size(), CV_8UC1);
	Mat gradmDivide = _gradmDivide.getMat();

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
			gradmDivide.at<uchar>(i, j) = ((double)gradmblur.at<uchar>(i, j) / (double)gradm.at<uchar>(i, j) >= 1 || gradm.at<uchar>(i, j) == 0 || gradmblur.at<uchar>(i, j) == 0) ? 0 : (1 - (double)gradmblur.at<uchar>(i, j) / (double)gradm.at<uchar>(i, j)) * 255;
}

/*�������*/
void HysteresisCut(InputArray _lineHT, InputArray _area, OutputArray _lineHC)
{
	Mat lineHT = _lineHT.getMat();
	CV_Assert(lineHT.type() == CV_8UC1);

	Mat area = _area.getMat();
	CV_Assert(area.type() == CV_8UC1);

	_lineHC.create(lineHT.size(), CV_8UC1);
	Mat lineHC = _lineHC.getMat();

	Mat UT(lineHT.size(), CV_8UC1, Scalar(0));		//�W�֭�

	for (int i = 0; i < lineHT.rows; ++i)
		for (int j = 0; j < lineHT.cols; ++j)
			if (area.at<uchar>(i, j) == 0 && lineHT.at<uchar>(i, j) == 255)
				UT.at<uchar>(i, j) = 255;

	Mat MT(lineHT.size(), CV_8UC1, Scalar(0));	//���֭�
	for (int i = 0; i < lineHT.rows; ++i)
		for (int j = 0; j < lineHT.cols; ++j)
			if (area.at<uchar>(i, j) == 255 && lineHT.at<uchar>(i, j) == 255)
				MT.at<uchar>(i, j) = 255;

	Mat labelImg;
	int labelNum = bwlabel(MT, labelImg, 4);
	labelNum = labelNum + 1;	// include label 0
	int* labeltable = new int[labelNum];		// initialize label table with zero  
	memset(labeltable, 0, labelNum * sizeof(int));

	for (int i = 0; i < lineHT.rows; ++i)
		for (int j = 0; j < lineHT.cols; ++j)
		{
			//+ - + - + - +
			//| B | C | D |
			//+ - + - + - +
			//| E | A | F |
			//+ - + - + - +
			//| G | H | I |
			//+ - + - + - +

			int B, C, D, E, F, G, H, I;

			if (i == 0 || j == 0) { B = 0; }
			else { B = UT.at<uchar>(i - 1, j - 1); }

			if (i == 0) { C = 0; }
			else { C = UT.at<uchar>(i - 1, j); }

			if (i == 0 || j == lineHT.cols - 1) { D = 0; }
			else { D = UT.at<uchar>(i - 1, j + 1); }

			if (j == 0) { E = 0; }
			else { E = UT.at<uchar>(i, j - 1); }

			if (j == lineHT.cols - 1) { F = 0; }
			else { F = UT.at<uchar>(i, j + 1); }

			if (i == lineHT.rows - 1 || j == 0) { G = 0; }
			else { G = UT.at<uchar>(i + 1, j - 1); }

			if (i == lineHT.rows - 1) { H = 0; }
			else { H = UT.at<uchar>(i + 1, j); }

			if (i == lineHT.rows - 1 || j == lineHT.cols - 1) { I = 0; }
			else { I = UT.at<uchar>(i + 1, j + 1); }

			// apply 8 connectedness  
			if (B || C || D || E || F || G || H || I)
			{
				++labeltable[labelImg.at<int>(i, j)];
			}
		}

	labeltable[0] = 0;		//clear 0 label

	Mat mask(lineHT.size(), CV_8UC1, Scalar(0));
	for (int i = 0; i < labelImg.rows; i++)
		for (int j = 0; j < labelImg.cols; j++)
			if (labeltable[labelImg.at<int>(i, j)] > 0 || UT.at<uchar>(i, j) == 255) { mask.at<uchar>(i, j) = 255; }
	delete[] labeltable;
	labeltable = nullptr;

	for (int i = 0; i < lineHT.rows; ++i)
		for (int j = 0; j < lineHT.cols; ++j)
			if (mask.at<uchar>(i, j) == 255) { lineHC.at<uchar>(i, j) = 255; }
			else { lineHC.at<uchar>(i, j) = 0; }
}

/*�D���j�ȧ���*/
void NonMaximumSuppression(InputArray _gradm, InputArray _gradd, OutputArray _gradmNMS, OutputArray _graddNMS)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradd = _gradd.getMat();
	CV_Assert(gradd.type() == CV_32FC1);

	_gradmNMS.create(gradm.size(), CV_8UC1);
	Mat gradmNMS = _gradmNMS.getMat();

	_graddNMS.create(gradd.size(), CV_32FC1);
	Mat graddNMS = _graddNMS.getMat();

	Mat gradmRef;
	copyMakeBorder(gradm, gradmRef, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

	float theta = 0.0f;			//�ثe��������V
	int amplitude = 0;			//�ثe�������T��
	int amplitude1 = 0;			//�F�칳��1���T��
	int amplitude2 = 0;			//�F�칳��2���T��
	float A1 = 0.0f;			//�W�{��1�T��
	float A2 = 0.0f;			//�W�{��2�T��
	float B1 = 0.0f;			//�U�{��1�T��
	float B2 = 0.0f;			//�U�{��2�T��
	float alpha = 0.0f;			//��ҫY��

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			amplitude = gradm.at<uchar>(i, j);
			theta = ((gradd.at<float>(i, j) + CV_PI) / CV_PI)*180.0f;

			if (gradd.at<float>(i, j) == -1000.0f)
			{
				gradmNMS.at<uchar>(i, j) = 0;
				graddNMS.at<float>(i, j) = -1000.0f;
				continue;
			}


			if ((theta >= 0.0f && theta < 45.0f) || (theta >= 180.0f && theta < 225.0f))
			{
				alpha = tan(theta* CV_PI / 180.0);
				A1 = gradmRef.at<uchar>(i + 1, j);
				A2 = gradmRef.at<uchar>(i, j);
				B1 = gradmRef.at<uchar>(i + 1, j + 2);
				B2 = gradmRef.at<uchar>(i + 2, j + 2);

			}
			else if ((theta >= 45.0f && theta < 90.0f) || (theta >= 225.0f && theta < 270.0f))
			{
				alpha = tan((90.0f - theta)* CV_PI / 180.0);
				A1 = gradmRef.at<uchar>(i, j + 1);
				A2 = gradmRef.at<uchar>(i, j);
				B1 = gradmRef.at<uchar>(i + 2, j + 1);
				B2 = gradmRef.at<uchar>(i + 2, j + 2);
			}
			else if ((theta >= 90.0f && theta < 135.0f) || (theta >= 270.0f && theta < 315.0f))
			{
				alpha = tan((theta - 90.0f)* CV_PI / 180.0);
				A1 = gradmRef.at<uchar>(i, j + 1);
				A2 = gradmRef.at<uchar>(i, j + 2);
				B1 = gradmRef.at<uchar>(i + 2, j + 1);
				B2 = gradmRef.at<uchar>(i + 2, j);
			}
			else if ((theta >= 135.0f && theta < 180.0f) || (theta >= 315.0f && theta <= 360.0f))
			{
				alpha = tan((180.0f - theta)* CV_PI / 180.0);
				A1 = gradmRef.at<uchar>(i + 1, j + 2);
				A2 = gradmRef.at<uchar>(i, j + 2);
				B1 = gradmRef.at<uchar>(i + 1, j);
				B2 = gradmRef.at<uchar>(i + 2, j);
			}

			amplitude1 = A1*(1 - alpha) + A2*alpha;
			amplitude2 = B1*(1 - alpha) + B2*alpha;

			if (amplitude > amplitude1 && amplitude > amplitude2)
			{
				gradmNMS.at<uchar>(i, j) = gradm.at<uchar>(i, j);
				graddNMS.at<float>(i, j) = gradd.at<float>(i, j);
			}
			else
			{
				gradmNMS.at<uchar>(i, j) = 0;
				graddNMS.at<float>(i, j) = -1000.0f;		//�ΥH�Ϥ��L����
			}
		}
}

/*�M�������I�I*/
void ClearBifPoint(InputArray _gradm, InputArray _gradd, OutputArray _gradmCBP, OutputArray _graddCBP)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradd = _gradd.getMat();
	CV_Assert(gradd.type() == CV_32FC1);

	_gradmCBP.create(gradm.size(), CV_8UC1);
	Mat gradmCBP = _gradmCBP.getMat();

	_graddCBP.create(gradd.size(), CV_32FC1);
	Mat graddCBP = _graddCBP.getMat();

	gradm.copyTo(gradmCBP);
	gradd.copyTo(graddCBP);

	Mat gradmRef;
	copyMakeBorder(gradm, gradmRef, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

	Mat mask(gradmCBP.rows + 2, gradmCBP.cols + 2, CV_8UC1, Scalar(1));

	for (int i = 1; i <= gradm.rows; ++i)
		for (int j = 1; j <= gradm.cols; ++j)
		{
			vector<char> nearPoint;
			int num = 0;

			if (gradmCBP.at<uchar>(i - 1, j - 1) != 0 && mask.at<uchar>(i, j) != 0)
			{
				nearPoint.push_back(gradmRef.at<uchar>(i, j - 1));			//1
				nearPoint.push_back(gradmRef.at<uchar>(i - 1, j - 1));				//2
				nearPoint.push_back(gradmRef.at<uchar>(i - 1, j));			//3
				nearPoint.push_back(gradmRef.at<uchar>(i - 1, j + 1));			//4
				nearPoint.push_back(gradmRef.at<uchar>(i, j + 1));		//5
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j + 1));		//6
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j));		//7
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j - 1));			//8
				num = 8 - count(nearPoint.begin(), nearPoint.end(), 0);

				bool flag = 0;
				if (num == 3 || num == 4)
				{
					int realnum = 0;
					if (gradmRef.at<uchar>(i, j - 1) != 0 && gradmRef.at<uchar>(i - 1, j - 1) != 0) { ++realnum; }
					if (gradmRef.at<uchar>(i - 1, j - 1) != 0 && gradmRef.at<uchar>(i - 1, j) != 0) { ++realnum; }
					if (gradmRef.at<uchar>(i - 1, j) != 0 && gradmRef.at<uchar>(i - 1, j + 1) != 0) { ++realnum; }
					if (gradmRef.at<uchar>(i - 1, j + 1) != 0 && gradmRef.at<uchar>(i, j + 1) != 0) { ++realnum; }
					if (gradmRef.at<uchar>(i, j + 1) != 0 && gradmRef.at<uchar>(i + 1, j + 1) != 0) { ++realnum; }
					if (gradmRef.at<uchar>(i + 1, j + 1) != 0 && gradmRef.at<uchar>(i + 1, j) != 0) { ++realnum; }
					if (gradmRef.at<uchar>(i + 1, j) != 0 && gradmRef.at<uchar>(i + 1, j - 1) != 0) { ++realnum; }
					if (gradmRef.at<uchar>(i + 1, j - 1) != 0 && gradmRef.at<uchar>(i, j - 1) != 0) { ++realnum; }
					if ((num == 3 && realnum == 1) || (num == 4 && realnum == 2)){ flag = 1; }
				}

				if (num != 0 && num != 1 && num != 2 && !flag)	//Bifurcation point
				{
					mask.at<uchar>(i, j) = mask.at<uchar>(i, j - 1) = mask.at<uchar>(i - 1, j - 1) = mask.at<uchar>(i - 1, j) = mask.at<uchar>(i - 1, j + 1) = 0;
					mask.at<uchar>(i, j + 1) = mask.at<uchar>(i + 1, j + 1) = mask.at<uchar>(i + 1, j) = mask.at<uchar>(i + 1, j - 1) = 0;
				}
			}
		}

	for (int i = 0; i < gradmCBP.rows; ++i)
		for (int j = 0; j < gradmCBP.cols; ++j)
			if (!mask.at<uchar>(i + 1, j + 1))
			{
				gradmCBP.at<uchar>(i, j) = 0;
				graddCBP.at<float>(i, j) = -1000.0f;
			}
			else
			{
				gradmCBP.at<uchar>(i, j) = gradm.at<uchar>(i, j);
				graddCBP.at<float>(i, j) = gradd.at<float>(i, j);
			}
}

/*�M������V�I*/
void ClearDifferentDirection(InputArray _gradm, InputArray _gradd, OutputArray _gradmCDD, OutputArray _graddCDD, bool flag)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradd = _gradd.getMat();
	CV_Assert(gradd.type() == CV_32FC1);

	_gradmCDD.create(gradm.size(), CV_8UC1);
	Mat gradmCDD = _gradmCDD.getMat();

	_graddCDD.create(gradd.size(), CV_32FC1);
	Mat graddCDD = _graddCDD.getMat();

	gradm.copyTo(gradmCDD);
	gradd.copyTo(graddCDD);

	float theta = 0.0f;			//�ثe��������V

	Mat mask(gradmCDD.rows + 2, gradmCDD.cols + 2, CV_8UC1, Scalar(1));

	for (int i = 1; i <= gradmCDD.rows; ++i)
		for (int j = 1; j <= gradmCDD.cols; ++j)
			/*if (gradmCDD.at<uchar>(i - 1, j - 1) != 0 && mask.at<uchar>(i, j) != 0)*/
			if (gradmCDD.at<uchar>(i - 1, j - 1) != 0)
			{
				float theta = ((gradd.at<float>(i - 1, j - 1) + CV_PI) / CV_PI)*180.0f;

				if (!flag)
				{
					if ((theta >= 22.5f && theta < 67.5f) || (theta >= 202.5f && theta < 247.5f))
						mask.at<uchar>(i - 1, j - 1) = mask.at<uchar>(i + 1, j + 1) = 0;
					else if ((theta >= 67.5f && theta < 112.5f) || (theta >= 247.5f && theta < 292.5f))
						mask.at<uchar>(i - 1, j) = mask.at<uchar>(i + 1, j) = 0;
					else if ((theta >= 112.5f && theta < 157.5f) || (theta >= 292.5f && theta < 337.5f))
						mask.at<uchar>(i - 1, j + 1) = mask.at<uchar>(i + 1, j - 1) = 0;
					else
						mask.at<uchar>(i, j - 1) = mask.at<uchar>(i, j + 1) = 0;
				}
				else
				{
					if ((theta >= 22.5f && theta < 67.5f) || (theta >= 202.5f && theta < 247.5f))
						mask.at<uchar>(i - 1, j - 1) = mask.at<uchar>(i + 1, j + 1) = mask.at<uchar>(i - 1, j) = mask.at<uchar>(i + 1, j) = 0;
					else if ((theta >= 67.5f && theta < 112.5f) || (theta >= 247.5f && theta < 292.5f))
						mask.at<uchar>(i - 1, j) = mask.at<uchar>(i + 1, j) = mask.at<uchar>(i - 1, j + 1) = mask.at<uchar>(i + 1, j - 1) = 0;
					else if ((theta >= 112.5f && theta < 157.5f) || (theta >= 292.5f && theta < 337.5f))
						mask.at<uchar>(i - 1, j + 1) = mask.at<uchar>(i + 1, j - 1) = mask.at<uchar>(i, j + 1) = mask.at<uchar>(i, j - 1) = 0;
					else
						mask.at<uchar>(i, j - 1) = mask.at<uchar>(i, j + 1) = mask.at<uchar>(i, j - 1) = mask.at<uchar>(i, j + 1) = 0;
				}
			}

	for (int i = 0; i < gradmCDD.rows; ++i)
		for (int j = 0; j < gradmCDD.cols; ++j)
			if (!mask.at<uchar>(i + 1, j + 1))
			{
				gradmCDD.at<uchar>(i, j) = 0;
				graddCDD.at<float>(i, j) = -1000.0f;
			}
			else
			{
				gradmCDD.at<uchar>(i, j) = gradm.at<uchar>(i, j);
				graddCDD.at<float>(i, j) = gradd.at<float>(i, j);
			}
}

/*�M���t���I*/
void ClearIsoPoint(InputArray _gradm, InputArray _gradd, OutputArray _gradmCIP, OutputArray _graddCIP)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradd = _gradd.getMat();
	CV_Assert(gradd.type() == CV_32FC1);

	_gradmCIP.create(gradm.size(), CV_8UC1);
	Mat gradmCIP = _gradmCIP.getMat();

	_graddCIP.create(gradd.size(), CV_32FC1);
	Mat graddCIP = _graddCIP.getMat();

	Mat gradmRef;
	copyMakeBorder(gradm, gradmRef, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			vector<char> nearPoint;
			int num = 0;

			if (gradmRef.at<uchar>(i + 1, j + 1) != 0)
			{
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j));			//1
				nearPoint.push_back(gradmRef.at<uchar>(i, j));				//2
				nearPoint.push_back(gradmRef.at<uchar>(i, j + 1));			//3
				nearPoint.push_back(gradmRef.at<uchar>(i, j + 2));			//4
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j + 2));		//5
				nearPoint.push_back(gradmRef.at<uchar>(i + 2, j + 2));		//6
				nearPoint.push_back(gradmRef.at<uchar>(i + 2, j + 1));		//7
				nearPoint.push_back(gradmRef.at<uchar>(i + 2, j));			//8
				num = 8 - count(nearPoint.begin(), nearPoint.end(), 0);

				if (num == 0)		//Isolated Point
				{
					gradmCIP.at<uchar>(i, j) = 0;
					graddCIP.at<float>(i, j) = -1000.0f;
				}
				else
				{
					gradmCIP.at<uchar>(i, j) = gradm.at<uchar>(i, j);
					graddCIP.at<float>(i, j) = gradd.at<float>(i, j);
				}
			}
			else
			{
				gradmCIP.at<uchar>(i, j) = 0;
				graddCIP.at<float>(i, j) = -1000.0f;
			}
		}
}

/*��׳��_�u�s�q*/
void ConnectLine(InputArray _gradm, InputArray _gradd, OutputArray _gradmCL, OutputArray _graddCL, int startSpace, int endSpace, int degree, int flagT, bool flagD)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradd = _gradd.getMat();
	CV_Assert(gradd.type() == CV_32FC1);

	_gradmCL.create(gradm.size(), CV_8UC1);
	Mat gradmCL = _gradmCL.getMat();

	_graddCL.create(gradd.size(), CV_32FC1);
	Mat graddCL = _graddCL.getMat();

	if (flagT != 0 && flagT != 1) { flagT = 0; }

	gradm.copyTo(gradmCL);
	gradd.copyTo(graddCL);

	Mat endPointMap;	//�I������
	pointlabel(gradm, endPointMap);

	Mat nearPoint1(gradm.size(), CV_8UC1, Scalar(1));		//�j�M�B�װ϶�1
	Mat nearPoint2(gradm.size(), CV_8UC1, Scalar(1));		//�j�M�B�װ϶�2
	Mat nearPoint3(gradm.size(), CV_8UC1, Scalar(1));		//�j�M�B�װ϶�3

	/*�d�߬O�_�B��*/
	for (int x = 2; x <= startSpace - 1 && startSpace != 2; ++x)
	{
		Mat graddRef;
		copyMakeBorder(gradd, graddRef, x - 1, x - 1, x - 1, x - 1, BORDER_CONSTANT, Scalar(-1000.0f));

		for (int i = 1; i < gradd.rows - 1; ++i)		//���j�M�v�����
			for (int j = 1; j < gradd.cols - 1; ++j)		//���j�M�v�����
			{
				int ir = i + x - 1, jr = j + x - 1;		//reference index i,j for graddRef

				if (endPointMap.at<Vec2b>(i, j)[0] == 2)	//�P�_�O�_�����I
				{

					bool flag1 = 1, flag2 = 1, flag3 = 1;	//�j�M�϶�flag

					/*�d�߬O�_�B��(�K�ϰ�j�M)*/
					if (endPointMap.at<Vec2b>(i, j)[1] == 1)		//8�ϰ�j�M - 1��(W��)
					{
						//�϶�1(N->NE)
						for (int is = ir - x, js = jr; js <= jr + x && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//�϶�2(NE->SE)
						for (int is = ir - x + 1, js = jr + x; is <= ir + x - 1 && nearPoint2.at<uchar>(i, j) == 1; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�3(SE->S)
						for (int is = ir + x, js = jr + x; js >= jr && nearPoint3.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 2)		//8�ϰ�j�M - 2��(NW��)
					{
						//�϶�1(NE->E)
						for (int is = ir - x, js = jr + x; is <= ir && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//�϶�2(E->SE)
						for (int is = ir + 1, js = jr + x; is >= ir + x && nearPoint2.at<uchar>(i, j) == 1; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�2(SE->S)
						for (int is = ir + x, js = jr + x - 1; js >= jr + 1 && nearPoint2.at<uchar>(i, j) == 1; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�3(S->SW)
						for (int is = ir + x, js = jr; js >= jr - x && nearPoint3.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 3)		//8�ϰ�j�M - 3��(N��)
					{
						//�϶�1(E->SE)
						for (int is = ir, js = jr + x; is <= ir + x && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//�϶�2(SE->SW)
						for (int is = ir + x, js = jr + x - 1; js >= jr - x + 1 && nearPoint2.at<uchar>(i, j) == 1; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�3(SW->W)
						for (int is = ir + x, js = jr - x; is >= ir && nearPoint3.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 4)		//8�ϰ�j�M - 4��(NE��)
					{
						//�϶�1(SE->S)
						for (int is = ir + x, js = jr + x; js >= jr && nearPoint1.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//�϶�2(S->SW)
						for (int is = ir + x, js = jr - 1; js >= jr - x && nearPoint2.at<uchar>(i, j) == 1; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�2(SW->W)
						for (int is = ir + x - 1, js = jr - x; is >= ir + 1 && nearPoint2.at<uchar>(i, j) == 1; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�3(W->NW)
						for (int is = ir, js = jr - x; is >= ir - x && nearPoint3.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 5)		//8�ϰ�j�M - 5��(E��)
					{
						//�϶�1(S->SW)
						for (int is = ir + x, js = jr; js >= jr - x && nearPoint1.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//�϶�2(SW->NW)
						for (int is = ir + x - 1, js = jr - x; is >= ir - x + 1 && nearPoint2.at<uchar>(i, j) == 1; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�3(NW->N)
						for (int is = ir - x, js = jr - x; js <= jr && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 6)		//8�ϰ�j�M - 6��(SE��)
					{
						//�϶�1(SW->W)
						for (int is = ir + x, js = jr - x; is >= ir && nearPoint1.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//�϶�2(W->NW)
						for (int is = ir - 1, js = jr - x; is <= ir - x && nearPoint2.at<uchar>(i, j) == 1; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�2(NW->N)
						for (int is = ir - x, js = jr - x + 1; js <= jr - 1 && nearPoint2.at<uchar>(i, j) == 1; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�3(N->NE)
						for (int is = ir - x, js = jr; js <= jr + x && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 7)		//8�ϰ�j�M - 7��(S��)
					{
						//�϶�1(W->NW)
						for (int is = ir, js = jr - x; is >= ir - x && nearPoint1.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//�϶�2(NW->NE)
						for (int is = ir - x, js = jr - x + 1; js <= jr + x - 1 && nearPoint2.at<uchar>(i, j) == 1; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�3(NE->E)
						for (int is = ir - x, js = jr + x; is <= ir && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 8)		//8�ϰ�j�M - 8��(SW��)
					{
						//�϶�1(NW->N)
						for (int is = ir - x, js = jr - x; js <= jr && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//�϶�2(N->NE)
						for (int is = ir - x, js = jr + 1; js <= jr + x && nearPoint2.at<uchar>(i, j) == 1; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�2(NE->E)
						for (int is = ir - x + 1, js = jr + x; is <= ir - 1 && nearPoint2.at<uchar>(i, j) == 1; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//�϶�3(E->SE)
						for (int is = ir, js = jr + x; is <= ir + x && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}

					/*�ק���I����*/
					if (!flag1) { nearPoint1.at<uchar>(i, j) = 0; }
					if (!flag2) { nearPoint2.at<uchar>(i, j) = 0; }
					if (!flag3) { nearPoint3.at<uchar>(i, j) = 0; }
					if (nearPoint1.at<uchar>(i, j) == 0 && nearPoint2.at<uchar>(i, j) == 0 && nearPoint3.at<uchar>(i, j) == 0)
						endPointMap.at<Vec2b>(i, j)[0] = 5;		//�^�O���I
				}
			}
	}

	/*�_�u�s�q*/
	for (int x = startSpace; x <= endSpace; ++x)
	{
		Mat graddRef;
		copyMakeBorder(gradd, graddRef, x - 1, x - 1, x - 1, x - 1, BORDER_CONSTANT, Scalar(-1000.0f));

		Mat endPointMapRef;
		copyMakeBorder(endPointMap, endPointMapRef, x - 1, x - 1, x - 1, x - 1, BORDER_CONSTANT, Scalar(0, 0));

		/*�j�M�ós�q�u*/
		for (int i = 1; i < gradd.rows - 1; ++i)		//���j�M�v�����
			for (int j = 1; j < gradd.cols - 1; ++j)		//���j�M�v�����
			{
				int ir = i + x - 1, jr = j + x - 1;		//reference index i,j for graddRef

				if (endPointMap.at<Vec2b>(i, j)[0] == 2)	//�P�_�O�_�����I
				{
					float theta0 = (gradd.at<float>(i, j) / CV_PI + 1)*180.0f;	//�ثe���I����
					float divtheta = 0.0f;		//�j���I�ۮt����
					float mintheta = 180.0f;	//�̤p�ۮt����

					char searchLocation = 0;	//�|�ϰ����(1(W),2(N),3(E),4(S))
					int ik = 0;					//�̨ηj�M�I��mik
					int jk = 0;					//�̨ηj�M�I��mjk

					float connectgradm = 0;		//�s�q�ؼдT��
					float connectgradd = 0.0f;		//�s�q�ؼФ�V

					bool flag1 = 1, flag2 = 1, flag3 = 1;	//�۾FflagT

					/*�j�M�̨��I(�K�ϰ�j�M)*/
					if (endPointMap.at<Vec2b>(i, j)[1] == 1)		//8�ϰ�j�M - 1��(W��)
					{
						//�϶�1(N->NE)
						for (int is = ir - x, js = jr; js <= jr + x && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag1 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 2; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(NE->SE)
						for (int is = ir - x + 1, js = jr + x; is <= ir + x - 1 && nearPoint2.at<uchar>(i, j) == 1; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 3; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�3(SE->S)
						for (int is = ir + x, js = jr + x; js >= jr && nearPoint3.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag3 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 4; ik = is - x + 1; jk = js - x + 1; }
								}
							}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 2)		//8�ϰ�j�M - 2��(NW��)
					{
						//�϶�1(NE->E)
						for (int is = ir - x, js = jr + x; is <= ir && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag1 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 3; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(E->SE)
						for (int is = ir + 1, js = jr + x; is >= ir + x && nearPoint2.at<uchar>(i, j) == 1; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 3; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(SE->S)
						for (int is = ir + x, js = jr + x - 1; js >= jr + 1 && nearPoint2.at<uchar>(i, j) == 1; --js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 4; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�3(S->SW)
						for (int is = ir + x, js = jr; js >= jr - x && nearPoint3.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag3 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 4; ik = is - x + 1; jk = js - x + 1; }
								}
							}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 3)		//8�ϰ�j�M - 3��(N��)
					{
						//�϶�1(E->SE)
						for (int is = ir, js = jr + x; is <= ir + x && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag1 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 3; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(SE->SW)
						for (int is = ir + x, js = jr + x - 1; js >= jr - x + 1 && nearPoint2.at<uchar>(i, j) == 1; --js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 4; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�3(SW->W)
						for (int is = ir + x, js = jr - x; is >= ir && nearPoint3.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag3 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 1; ik = is - x + 1; jk = js - x + 1; }
								}
							}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 4)		//8�ϰ�j�M - 4��(NE��)
					{
						//�϶�1(SE->S)
						for (int is = ir + x, js = jr + x; js >= jr && nearPoint1.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag1 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 4; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(S->SW)
						for (int is = ir + x, js = jr - 1; js >= jr - x && nearPoint2.at<uchar>(i, j) == 1; --js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 4; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(SW->W)
						for (int is = ir + x - 1, js = jr - x; is >= ir + 1 && nearPoint2.at<uchar>(i, j) == 1; --is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 1; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�3(W->NW)
						for (int is = ir, js = jr - x; is >= ir - x && nearPoint3.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag3 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 1; ik = is - x + 1; jk = js - x + 1; }
								}
							}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 5)		//8�ϰ�j�M - 5��(E��)
					{
						//�϶�1(S->SW)
						for (int is = ir + x, js = jr; js >= jr - x && nearPoint1.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag1 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 4; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(SW->NW)
						for (int is = ir + x - 1, js = jr - x; is >= ir - x + 1 && nearPoint2.at<uchar>(i, j) == 1; --is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 1; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�3(NW->N)
						for (int is = ir - x, js = jr - x; js <= jr && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag3 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 2; ik = is - x + 1; jk = js - x + 1; }
								}
							}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 6)		//8�ϰ�j�M - 6��(SE��)
					{
						//�϶�1(SW->W)
						for (int is = ir + x, js = jr - x; is >= ir + 1 && nearPoint1.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag1 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 1; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(W->NW)
						for (int is = ir - 1, js = jr - x; is <= ir - x && nearPoint2.at<uchar>(i, j) == 1; --is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 1; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(NW->N)
						for (int is = ir - x, js = jr - x + 1; js <= jr - 1 && nearPoint2.at<uchar>(i, j) == 1; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 2; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�3(N->NE)
						for (int is = ir - x, js = jr; js <= jr + x && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag3 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 2; ik = is - x + 1; jk = js - x + 1; }
								}
							}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 7)		//8�ϰ�j�M - 7��(S��)
					{
						//�϶�1(W->NW)
						for (int is = ir, js = jr - x; is >= ir - x && nearPoint1.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag1 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 1; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(NW->NE)
						for (int is = ir - x, js = jr - x + 1; js <= jr + x - 1 && nearPoint2.at<uchar>(i, j) == 1; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 2; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�3(NE->E)
						for (int is = ir - x, js = jr + x; is <= ir && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag3 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 3; ik = is - x + 1; jk = js - x + 1; }
								}
							}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 8)		//8�ϰ�j�M - 8��(SW��)
					{
						//�϶�1(NW->N)
						for (int is = ir - x, js = jr - x; js <= jr && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag1 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 2; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(N->NE)
						for (int is = ir - x, js = jr + 1; js <= jr + x && nearPoint2.at<uchar>(i, j) == 1; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 2; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�2(NE->E)
						for (int is = ir - x + 1, js = jr + x; is <= ir - 1 && nearPoint2.at<uchar>(i, j) == 1; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag2 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 3; ik = is - x + 1; jk = js - x + 1; }
								}
							}
						//�϶�3(E->SE)
						for (int is = ir, js = jr + x; is <= ir + x && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f)
							{
								flag3 = 0;
								if (endPointMapRef.at<Vec2b>(is, js)[0] == 2 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 3 || flagT*endPointMapRef.at<Vec2b>(is, js)[0] == 4)
								{
									divtheta = abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0) > 180 ? (360 - abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0)) : abs((graddRef.at<float>(is, js) / CV_PI + 1)*180.0f - theta0);
									if (divtheta < mintheta) { mintheta = divtheta; searchLocation = 3; ik = is - x + 1; jk = js - x + 1; }
								}
							}
					}

					/*�ק���I����*/
					if (!flag1) { nearPoint1.at<uchar>(i, j) = 0; }
					if (!flag2) { nearPoint2.at<uchar>(i, j) = 0; }
					if (!flag3) { nearPoint3.at<uchar>(i, j) = 0; }
					if (nearPoint1.at<uchar>(i, j) == 0 && nearPoint2.at<uchar>(i, j) == 0 && nearPoint3.at<uchar>(i, j) == 0) { endPointMap.at<Vec2b>(i, j)[0] = 5; }	//�^�O���I

					/*�s�q�̨��I(�|�ϰ�s�q)*/
					if (searchLocation == 1 && mintheta <= degree)		//4�ϰ�s�q - 1��(W��)
					{
						double a = (double)(j - jk) / (double)(i - ik);			//�p��s�q�u�ײv(j = a * i + b)
						double b = (double)(j*ik - jk*i) / (double)(ik - i);	//�p��s�q�u�I�Z(j = a * i + b)
						int step = 0;	//�B�Ƥ��v��(from 0 to x)

						for (int jc = j; jc >= jk; --jc, ++step)		//�s�q���u
						{
							int ic = (i - ik == 0) ? i : round((double)(jc - b) / a);		//����ƭ�Inf
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + gradm.at<uchar>(ik, jk)*step / x;
							graddCL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(gradd.at<float>(ik, jk))*step, cos(gradd.at<float>(i, j))*(x - step) + cos(gradd.at<float>(ik, jk))*step);
						}
					}
					else if (searchLocation == 2 && mintheta <= degree)		//4�ϰ�s�q - 2��(N��)
					{
						double a = (double)(j - jk) / (double)(i - ik);			//�p��s�q�u�ײv(j = a * i + b)
						double b = (double)(j*ik - jk*i) / (double)(ik - i);	//�p��s�q�u�I�Z(j = a * i + b)
						int step = 0;	//�B�Ƥ��v��(from 0 to x)

						for (int ic = i; ic >= ik; --ic, ++step)		//�s�q���u
						{
							int jc = round((double)ic * a + b);
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + gradm.at<uchar>(ik, jk)*step / x;
							graddCL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(gradd.at<float>(ik, jk))*step, cos(gradd.at<float>(i, j))*(x - step) + cos(gradd.at<float>(ik, jk))*step);
						}
					}
					else if (searchLocation == 3 && mintheta <= degree)		//4�ϰ�s�q - 3��(E��)
					{
						double a = (double)(j - jk) / (double)(i - ik);			//�p��s�q�u�ײv(j = a * i + b)
						double b = (double)(j*ik - jk*i) / (double)(ik - i);	//�p��s�q�u�I�Z(j = a * i + b)
						int step = 0;	//�B�Ƥ��v��(from 0 to x)

						for (int jc = j; jc <= jk; ++jc, ++step)		//�s�q���u
						{
							int ic = (i - ik == 0) ? i : round((double)(jc - b) / a);		//����ƭ�Inf
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + gradm.at<uchar>(ik, jk)*step / x;
							graddCL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(gradd.at<float>(ik, jk))*step, cos(gradd.at<float>(i, j))*(x - step) + cos(gradd.at<float>(ik, jk))*step);
						}
					}
					else if (searchLocation == 4 && mintheta <= degree)		//4�ϰ�s�q - 4��(S��)
					{
						double a = (double)(j - jk) / (double)(i - ik);			//�p��s�q�u�ײv(j = a * i + b)
						double b = (double)(j*ik - jk*i) / (double)(ik - i);	//�p��s�q�u�I�Z(j = a * i + b)
						int step = 0;	//�B�Ƥ��v��(from 0 to x)

						for (int ic = i; ic <= ik; ++ic, ++step)		//�s�q���u
						{
							int jc = round((double)ic * a + b);
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + gradm.at<uchar>(ik, jk)*step / x;
							graddCL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(gradd.at<float>(ik, jk))*step, cos(gradd.at<float>(i, j))*(x - step) + cos(gradd.at<float>(ik, jk))*step);
						}
					}
				}
			}
	}
}

/*����֭�*/
void HysteresisThreshold(InputArray _gradm, OutputArray _bwLine, int upperThreshold, int lowerThreshold)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	_bwLine.create(gradm.size(), CV_8UC1);
	Mat bwLine = _bwLine.getMat();

	Mat UT;		//�W�֭ȤG�Ȥ�
	threshold(gradm, UT, upperThreshold, 255, THRESH_BINARY);
	Mat LT;		//�U�֭ȤG�Ȥ�
	threshold(gradm, LT, lowerThreshold, 255, THRESH_BINARY);
	Mat MT;		//�z��t
	MT.create(gradm.size(), CV_8UC1);
	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			if (LT.at<uchar>(i, j) == 255 && UT.at<uchar>(i, j) == 0)
				MT.at<uchar>(i, j) = 255;
			else
				MT.at<uchar>(i, j) = 0;

			if (UT.at<uchar>(i, j) == 255)
				bwLine.at<uchar>(i, j) = 255;
			else
				bwLine.at<uchar>(i, j) = 0;
		}

	Mat labelImg;
	int labelNum = bwlabel(MT, labelImg, 8);
	labelNum = labelNum + 1;	// include label 0
	int* labeltable = new int[labelNum];		// initialize label table with zero  
	memset(labeltable, 0, labelNum * sizeof(int));

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			//+ - + - + - +
			//| B | C | D |
			//+ - + - + - +
			//| E | A | F |
			//+ - + - + - +
			//| G | H | I |
			//+ - + - + - +

			int B, C, D, E, F, G, H, I;

			if (i == 0 || j == 0) { B = 0; }
			else { B = UT.at<uchar>(i - 1, j - 1); }

			if (i == 0) { C = 0; }
			else { C = UT.at<uchar>(i - 1, j); }

			if (i == 0 || j == gradm.cols - 1) { D = 0; }
			else { D = UT.at<uchar>(i - 1, j + 1); }

			if (j == 0) { E = 0; }
			else { E = UT.at<uchar>(i, j - 1); }

			if (j == gradm.cols - 1) { F = 0; }
			else { F = UT.at<uchar>(i, j + 1); }

			if (i == gradm.rows - 1 || j == 0) { G = 0; }
			else { G = UT.at<uchar>(i + 1, j - 1); }

			if (i == gradm.rows - 1) { H = 0; }
			else { H = UT.at<uchar>(i + 1, j); }

			if (i == gradm.rows - 1 || j == gradm.cols - 1) { I = 0; }
			else { I = UT.at<uchar>(i + 1, j + 1); }

			// apply 8 connectedness  
			if (B || C || D || E || F || G || H || I)
			{
				++labeltable[labelImg.at<int>(i, j)];
			}
		}

	labeltable[0] = 0;		//clear 0 label

	for (int i = 0; i < labelImg.rows; i++)
		for (int j = 0; j < labelImg.cols; j++)
		{
			if (labeltable[labelImg.at<int>(i, j)] > 0)
			{
				bwLine.at<uchar>(i, j) = 255;
			}
		}
	delete[] labeltable;
	labeltable = nullptr;
}

/*�M���t���I*/
void BWClearIsoPoint(InputArray _bwLine, OutputArray _bwLineCP, int border, int iter, bool flagT)
{
	Mat bwLine = _bwLine.getMat();
	CV_Assert(bwLine.type() == CV_8UC1);

	_bwLineCP.create(bwLine.size(), CV_8UC1);
	Mat bwLineCP = _bwLineCP.getMat();

	bwLine.copyTo(bwLineCP);

	int type;
	if (!flagT) { type = 1; }		//Isolated Point
	else { type = 2; }				//EndPoint 

	bool stopflag = 1;

	for (int step = 1; step <= iter && stopflag; ++step)
	{
		Mat pointMap;	//�I������
		pointlabel(bwLineCP, pointMap);
		int exenums = 0;
		for (int i = border; i < pointMap.rows - border; ++i)
			for (int j = border; j < pointMap.cols - border; ++j)
				if (pointMap.at<Vec2b>(i, j)[0] == 1 || pointMap.at<Vec2b>(i, j)[0] == type)
				{
					bwLineCP.at<uchar>(i, j) = 0;
					++exenums;
				}
		if (exenums == 0) { stopflag = 0; }
	}
}

/*�G�ȹ��_�u�s�q*/
void BWConnectLine(InputArray _gradm, InputArray _gradd, InputArray _bwLine, OutputArray _gradmCL, OutputArray _graddCL, OutputArray _bwLineCL, int startSpace, int endSpace, int degree, int flagT, bool flagD)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradd = _gradd.getMat();
	CV_Assert(gradd.type() == CV_32FC1);

	Mat bwLine = _bwLine.getMat();
	CV_Assert(bwLine.type() == CV_8UC1);

	_gradmCL.create(gradm.size(), CV_8UC1);
	Mat gradmCL = _gradmCL.getMat();

	_graddCL.create(gradd.size(), CV_32FC1);
	Mat graddCL = _graddCL.getMat();

	_bwLineCL.create(bwLine.size(), CV_8UC1);
	Mat bwLineCL = _bwLineCL.getMat();

	if (flagT != 0 && flagT != 1) { flagT = 0; }

	Mat gradmRef(gradm.size(), CV_8UC1);
	Mat graddRef(gradd.size(), CV_32FC1);

	for (int i = 0; i < bwLine.rows; ++i)
		for (int j = 0; j < bwLine.cols; ++j)
		{
			if (bwLine.at<uchar>(i, j) == 255)
			{
				gradmRef.at<uchar>(i, j) = gradm.at<uchar>(i, j);
				graddRef.at<float>(i, j) = gradd.at<float>(i, j);
			}
			else
			{
				gradmRef.at<uchar>(i, j) = 0;
				graddRef.at<float>(i, j) = -1000.0f;
			}
		}

	ConnectLine(gradmRef, graddRef, gradmCL, graddCL, startSpace, endSpace, degree, flagT, flagD);

	for (int i = 0; i < graddCL.rows; ++i)
		for (int j = 0; j < graddCL.cols; ++j)
		{
			if (graddCL.at<float>(i, j) != -1000.0f) { bwLineCL.at<uchar>(i, j) = 255; }
			else { bwLineCL.at<uchar>(i, j) = 0; }
		}
}

/*���X�u�P�����G����t*/
void BWCombine(InputArray _area, InputArray _line, OutputArray _object)
{
	Mat area = _area.getMat();
	CV_Assert(area.type() == CV_8UC1);

	Mat line = _line.getMat();
	CV_Assert(line.type() == CV_8UC1);

	_object.create(area.size(), CV_8UC1);
	Mat object = _object.getMat();

	for (int i = 0; i < object.rows; ++i)
		for (int j = 0; j < object.cols; ++j)
			if (area.at<uchar>(i, j) == 0 || line.at<uchar>(i, j) == 0) { object.at<uchar>(i, j) = 0; }
			else { object.at<uchar>(i, j) = 255; }
}

/*����G�ȹ�*/
void BWReverse(InputArray _bwImage, OutputArray _bwImageR)
{
	Mat bwImage = _bwImage.getMat();
	CV_Assert(bwImage.type() == CV_8UC1);

	_bwImageR.create(bwImage.size(), CV_8UC1);
	Mat bwImageR = _bwImageR.getMat();

	for (int i = 0; i < bwImage.rows; ++i)
		for (int j = 0; j < bwImage.cols; ++j)
			if (bwImage.at<uchar>(i, j) == 0) { bwImageR.at<uchar>(i, j) = 255; }
			else { bwImage.at<uchar>(i, j) = 0; }
}

/*�������t��k*/
void BWWatershed(InputArray _srcImage, InputArray _bwSeed, InputArray _bwObject, OutputArray _bwWatershed)
{
	Mat srcImage = _srcImage.getMat();
	CV_Assert(srcImage.type() == CV_8UC1 || srcImage.type() == CV_8UC3);
	if (srcImage.type() == CV_8UC1) { cvtColor(srcImage, srcImage, CV_GRAY2BGR); }

	Mat bwSeed = _bwSeed.getMat();
	CV_Assert(bwSeed.type() == CV_8UC1);

	Mat bwObject = _bwObject.getMat();
	CV_Assert(bwObject.type() == CV_8UC1);

	_bwWatershed.create(srcImage.size(), CV_8UC1);
	Mat bwWatershed = _bwWatershed.getMat();
	bwObject.copyTo(bwWatershed);

	Mat seedLabels;
	int num = bwlabel(bwSeed, seedLabels, 4);

	Mat label(seedLabels.rows, seedLabels.cols, CV_32SC1);

	seedLabels.copyTo(label);

	for (int i = 0; i < bwObject.rows; ++i)
		for (int j = 0; j < bwObject.cols; ++j)
			if (bwObject.at<uchar>(i, j) == 0) { label.at<int>(i, j) = -1; }

	watershed(srcImage, label);

	for (int i = 0; i < label.rows; ++i)
		for (int j = 0; j < label.cols; ++j)
			if (label.at<int>(i, j) == -1) { bwWatershed.at<uchar>(i, j) = 0; }
}