#include "stdafx.h"
#include "basic_processing.h"

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
	Mat grayImage;
	Mat temp = _grayImage.getMat();
	if (temp.type() == CV_16SC1) { temp.convertTo(grayImage, CV_32FC1); }
	else { grayImage = _grayImage.getMat(); }

	_colorbarImage.create(grayImage.size(), CV_8UC3);
	Mat colorbarImage = _colorbarImage.getMat();

	static vector<Scalar> colorbar; //Scalar i,g,b  
	if (colorbar.empty()) { makecolorbar(colorbar); }

	int maxrad = upperbound - lowerbound + 1;

	if (grayImage.type() == CV_8UC1)
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
	else
		for (int i = 0; i < colorbarImage.rows; ++i)
			for (int j = 0; j < colorbarImage.cols; ++j)
			{
				uchar *data = colorbarImage.data + colorbarImage.step[0] * i + colorbarImage.step[1] * j;

				float fk = (1 - grayImage.at<float>(i, j) / (float)maxrad) * (colorbar.size() - 1);  //�p��ǫ׭ȹ��������ަ�m
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
	Mat bwImage;
	Mat temp = _bwImage.getMat();
	CV_Assert(temp.type() == CV_8UC1 || temp.type() == CV_32FC1);
	if (temp.type() == CV_32FC1) { temp.convertTo(bwImage, CV_8UC1); }
	else { bwImage = temp; }

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

/*�N���G��ܦb�m��Ϲ��W*/
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
					combineImage.at<Vec3b>(i, j)[1] = realImage.at<Vec3b>(i, j)[1];
					combineImage.at<Vec3b>(i, j)[2] = realImage.at<Vec3b>(i, j)[2];
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

/*�N�ؤl�I��ܦb�쪫��W*/
void DrawSeed(InputArray _object, InputArray _objectSeed, OutputArray _combineSeed)
{
	Mat object = _object.getMat();
	CV_Assert(object.type() == CV_8UC1);

	Mat objectSeed = _objectSeed.getMat();
	CV_Assert(objectSeed.type() == CV_8UC1);

	_combineSeed.create(object.size(), CV_8UC1);
	Mat combineSeed = _combineSeed.getMat();

	for (int i = 0; i < combineSeed.rows; ++i)
		for (int j = 0; j < combineSeed.cols; ++j)
		{
			if (object.at<uchar>(i, j) == 255) { combineSeed.at<uchar>(i, j) = 128; }
			if (objectSeed.at<uchar>(i, j) == 255) { combineSeed.at<uchar>(i, j) = 255; }
			if (object.at<uchar>(i, j) == 0) { combineSeed.at<uchar>(i, j) = 0; }
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

/*��ɪŬ}*/
void BWFillhole(InputArray _bwImage, OutputArray _bwFillhole)
{
	Mat bwImage = _bwImage.getMat();
	CV_Assert(bwImage.type() == CV_8UC1);
	
	_bwFillhole.create(bwImage.size(), CV_8UC1);
	Mat bwFillhole = _bwFillhole.getMat();

	Mat mask;
	bwImage.copyTo(mask);
	for (int i = 0; i < mask.cols; ++i) 
	{
		if (mask.at<uchar>(0, i) == 0) {	floodFill(mask, Point(i, 0), 255, 0, 10, 10, 8); }
		if (mask.at<uchar>(mask.rows - 1, i) == 0) {	floodFill(mask, Point(i, mask.rows - 1), 255, 0, 10, 10, 8); }
	}
	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i, 0) == 0) { floodFill(mask, Point(0, i), 255, 0, 10, 10, 8); }
		if (mask.at<uchar>(i, mask.cols - 1) == 0) { floodFill(mask, Point(mask.cols - 1, i), 255, 0, 10, 10, 8); }
	}


	// Compare mask with original.
	bwImage.copyTo(bwFillhole);
	for (int i = 0; i < mask.rows; ++i) 
		for (int j = 0; j < mask.cols; ++j)
			if (mask.at<uchar>(i, j) == 0)
				bwFillhole.at<uchar>(i, j) = 255;
}

