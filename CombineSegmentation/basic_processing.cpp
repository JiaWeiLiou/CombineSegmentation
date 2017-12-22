#include "stdafx.h"
#include "basic_processing.h"

/*生成二維高斯核*/
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

/*尋找根結點*/
int findroot(int labeltable[], int label)
{
	int x = label;
	while (x != labeltable[x])
		x = labeltable[x];
	return x;
}

/*尋找連通物*/
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

/*判斷點的類型*/
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
						if (nearPoint[(k % 9)] != 0 && nearPoint[(k % 8) + 1] != 0) //1,2、2,3、...、8,1   End of Line Point 
						{
							labels.at<Vec2b>(i, j)[0] = 2;
							if (k % 2 == 1)		//只存取對角線的方向
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

/*創建色環*/
void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;	//紅色(Red)     至黃色(Yellow)
	int YG = 15;	//黃色(Yellow)  至綠色(Green)
	int GC = 15;	//綠色(Green)   至青色(Cyan)
	int CB = 15;	//青澀(Cyan)    至藍色(Blue)
	int BM = 15;	//藍色(Blue)    至洋紅(Magenta)
	int MR = 15;	//洋紅(Magenta) 至紅色(Red)

	for (int i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (int i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (int i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (int i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (int i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (int i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

/*創建色條*/
void makecolorbar(vector<Scalar> &colorbar)
{
	vector<Scalar> maincolor;

	maincolor.push_back(Scalar(127.5, 0, 0));      //深紅色
	maincolor.push_back(Scalar(255, 0, 0));		   //紅色
	maincolor.push_back(Scalar(255, 127.5, 0));	   //紅色至黃色
	maincolor.push_back(Scalar(255, 255, 0));	   //黃色
	maincolor.push_back(Scalar(127.5, 255, 0));	   //黃色至綠色
	maincolor.push_back(Scalar(0, 255, 0));		   //綠色
	maincolor.push_back(Scalar(0, 255, 127.5));	   //綠色至青色
	maincolor.push_back(Scalar(0, 255, 255));	   //青色
	maincolor.push_back(Scalar(0, 127.5, 255));	   //青色至藍色
	maincolor.push_back(Scalar(0, 0, 255));		   //藍色
	maincolor.push_back(Scalar(0, 0, 127.5));      //深藍色

	int layer = 15;		//各漸層漸變階層數

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

/*將灰階圖片轉以色條顯示*/
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

				float fk = (1 - (float)grayImage.at<uchar>(i, j) / (float)maxrad) * (colorbar.size() - 1);  //計算灰度值對應之索引位置
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

				float fk = (1 - grayImage.at<float>(i, j) / (float)maxrad) * (colorbar.size() - 1);  //計算灰度值對應之索引位置
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

/*將圖片轉以色環方向場顯示(輸入梯度場或梯度方向)*/
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
		maxrad = 255;		//只有梯度方向無梯度幅值

		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;

				if (field.at<float>(i, j) == -1000.0f)		//用以顯示無梯度方向
				{
					for (int b = 0; b < 3; b++)
					{
						data[2 - b] = 255;
					}
				}
				else
				{
					float rad = maxrad;

					float angle = field.at<float>(i, j) / CV_PI;    //單位為-1至+1
					float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
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

		maxrad = maxrad / 2;		//加深顯示結果(可取消此行)

		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;
				Vec2f field_at_point = field.at<Vec2f>(i, j);

				float fx = field_at_point[0];
				float fy = field_at_point[1];

				float rad = sqrt(fx * fx + fy * fy) / maxrad;

				float angle = atan2(fy, fx) / CV_PI;    //單位為-1至+1
				float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
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

/*將圖片轉以色環方向場顯示(輸入梯度幅值及梯度方向)*/
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

	maxrad = maxrad / 2;		//加深顯示結果(可取消此行)

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;

			if (gradd.at<float>(i, j) == -1000.0f)		//用以顯示無梯度方向
			{
				for (int b = 0; b < 3; b++)
				{
					data[2 - b] = 255;
				}
			}
			else
			{
				float rad = gradm.at<float>(i, j) / maxrad;

				float angle = gradd.at<float>(i, j) / CV_PI;    //單位為-1至+1
				float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
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

/*將圖片轉線性拉伸並以灰階值顯示*/
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

/*將結果以標籤顯示*/
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

/*將結果顯示在彩色圖像上*/
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

/*圖層混合模式*/
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

/*基於面的分割混合模式*/
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

/*實色印疊合混合模式*/
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

/*去除雜訊*/
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

/*中央差分*/
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

/*結合水平及垂直方向梯度為梯度場*/
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

/*計算梯度幅值及方向*/
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

			if (x == 0 && y == 0) { gradd.at<float>(i, j) = -1000.0f; }	//用以顯示無梯度方向
			else { gradd.at<float>(i, j) = atan2(y, x); }
		}
}

/*梯度方向模糊*/
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

/*基於線的分割混合模式*/
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

/*滯後切割*/
void HysteresisCut(InputArray _lineHT, InputArray _area, OutputArray _lineHC)
{
	Mat lineHT = _lineHT.getMat();
	CV_Assert(lineHT.type() == CV_8UC1);

	Mat area = _area.getMat();
	CV_Assert(area.type() == CV_8UC1);

	_lineHC.create(lineHT.size(), CV_8UC1);
	Mat lineHC = _lineHC.getMat();

	Mat UT(lineHT.size(), CV_8UC1, Scalar(0));		//上閥值

	for (int i = 0; i < lineHT.rows; ++i)
		for (int j = 0; j < lineHT.cols; ++j)
			if (area.at<uchar>(i, j) == 0 && lineHT.at<uchar>(i, j) == 255)
				UT.at<uchar>(i, j) = 255;

	Mat MT(lineHT.size(), CV_8UC1, Scalar(0));	//中閥值
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

/*非極大值抑制*/
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

	float theta = 0.0f;			//目前像素的方向
	int amplitude = 0;			//目前像素的幅值
	int amplitude1 = 0;			//鄰域像素1的幅值
	int amplitude2 = 0;			//鄰域像素2的幅值
	float A1 = 0.0f;			//上臨域1幅值
	float A2 = 0.0f;			//上臨域2幅值
	float B1 = 0.0f;			//下臨域1幅值
	float B2 = 0.0f;			//下臨域2幅值
	float alpha = 0.0f;			//比例係數

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
				graddNMS.at<float>(i, j) = -1000.0f;		//用以區分無角度
			}
		}
}

/*清除分岔點點*/
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

/*清除異方向點*/
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

	float theta = 0.0f;			//目前像素的方向

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

/*清除孤立點*/
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

/*梯度場斷線連通*/
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

	Mat endPointMap;	//點的種類
	pointlabel(gradm, endPointMap);

	Mat nearPoint1(gradm.size(), CV_8UC1, Scalar(1));		//搜尋遮擋區塊1
	Mat nearPoint2(gradm.size(), CV_8UC1, Scalar(1));		//搜尋遮擋區塊2
	Mat nearPoint3(gradm.size(), CV_8UC1, Scalar(1));		//搜尋遮擋區塊3

	/*查詢是否遮擋*/
	for (int x = 2; x <= startSpace - 1 && startSpace != 2; ++x)
	{
		Mat graddRef;
		copyMakeBorder(gradd, graddRef, x - 1, x - 1, x - 1, x - 1, BORDER_CONSTANT, Scalar(-1000.0f));

		for (int i = 1; i < gradd.rows - 1; ++i)		//不搜尋影像邊界
			for (int j = 1; j < gradd.cols - 1; ++j)		//不搜尋影像邊界
			{
				int ir = i + x - 1, jr = j + x - 1;		//reference index i,j for graddRef

				if (endPointMap.at<Vec2b>(i, j)[0] == 2)	//判斷是否為端點
				{

					bool flag1 = 1, flag2 = 1, flag3 = 1;	//搜尋區塊flag

					/*查詢是否遮擋(八區域搜尋)*/
					if (endPointMap.at<Vec2b>(i, j)[1] == 1)		//8區域搜尋 - 1區(W區)
					{
						//區塊1(N->NE)
						for (int is = ir - x, js = jr; js <= jr + x && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//區塊2(NE->SE)
						for (int is = ir - x + 1, js = jr + x; is <= ir + x - 1 && nearPoint2.at<uchar>(i, j) == 1; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊3(SE->S)
						for (int is = ir + x, js = jr + x; js >= jr && nearPoint3.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 2)		//8區域搜尋 - 2區(NW區)
					{
						//區塊1(NE->E)
						for (int is = ir - x, js = jr + x; is <= ir && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//區塊2(E->SE)
						for (int is = ir + 1, js = jr + x; is >= ir + x && nearPoint2.at<uchar>(i, j) == 1; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊2(SE->S)
						for (int is = ir + x, js = jr + x - 1; js >= jr + 1 && nearPoint2.at<uchar>(i, j) == 1; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊3(S->SW)
						for (int is = ir + x, js = jr; js >= jr - x && nearPoint3.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 3)		//8區域搜尋 - 3區(N區)
					{
						//區塊1(E->SE)
						for (int is = ir, js = jr + x; is <= ir + x && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//區塊2(SE->SW)
						for (int is = ir + x, js = jr + x - 1; js >= jr - x + 1 && nearPoint2.at<uchar>(i, j) == 1; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊3(SW->W)
						for (int is = ir + x, js = jr - x; is >= ir && nearPoint3.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 4)		//8區域搜尋 - 4區(NE區)
					{
						//區塊1(SE->S)
						for (int is = ir + x, js = jr + x; js >= jr && nearPoint1.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//區塊2(S->SW)
						for (int is = ir + x, js = jr - 1; js >= jr - x && nearPoint2.at<uchar>(i, j) == 1; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊2(SW->W)
						for (int is = ir + x - 1, js = jr - x; is >= ir + 1 && nearPoint2.at<uchar>(i, j) == 1; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊3(W->NW)
						for (int is = ir, js = jr - x; is >= ir - x && nearPoint3.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 5)		//8區域搜尋 - 5區(E區)
					{
						//區塊1(S->SW)
						for (int is = ir + x, js = jr; js >= jr - x && nearPoint1.at<uchar>(i, j) == 1 && flagD; --js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//區塊2(SW->NW)
						for (int is = ir + x - 1, js = jr - x; is >= ir - x + 1 && nearPoint2.at<uchar>(i, j) == 1; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊3(NW->N)
						for (int is = ir - x, js = jr - x; js <= jr && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 6)		//8區域搜尋 - 6區(SE區)
					{
						//區塊1(SW->W)
						for (int is = ir + x, js = jr - x; is >= ir && nearPoint1.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//區塊2(W->NW)
						for (int is = ir - 1, js = jr - x; is <= ir - x && nearPoint2.at<uchar>(i, j) == 1; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊2(NW->N)
						for (int is = ir - x, js = jr - x + 1; js <= jr - 1 && nearPoint2.at<uchar>(i, j) == 1; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊3(N->NE)
						for (int is = ir - x, js = jr; js <= jr + x && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 7)		//8區域搜尋 - 7區(S區)
					{
						//區塊1(W->NW)
						for (int is = ir, js = jr - x; is >= ir - x && nearPoint1.at<uchar>(i, j) == 1 && flagD; --is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//區塊2(NW->NE)
						for (int is = ir - x, js = jr - x + 1; js <= jr + x - 1 && nearPoint2.at<uchar>(i, j) == 1; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊3(NE->E)
						for (int is = ir - x, js = jr + x; is <= ir && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 8)		//8區域搜尋 - 8區(SW區)
					{
						//區塊1(NW->N)
						for (int is = ir - x, js = jr - x; js <= jr && nearPoint1.at<uchar>(i, j) == 1 && flagD; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag1 = 0; }
						//區塊2(N->NE)
						for (int is = ir - x, js = jr + 1; js <= jr + x && nearPoint2.at<uchar>(i, j) == 1; ++js)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊2(NE->E)
						for (int is = ir - x + 1, js = jr + x; is <= ir - 1 && nearPoint2.at<uchar>(i, j) == 1; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag2 = 0; }
						//區塊3(E->SE)
						for (int is = ir, js = jr + x; is <= ir + x && nearPoint3.at<uchar>(i, j) == 1 && flagD; ++is)
							if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
					}

					/*修改端點類型*/
					if (!flag1) { nearPoint1.at<uchar>(i, j) = 0; }
					if (!flag2) { nearPoint2.at<uchar>(i, j) = 0; }
					if (!flag3) { nearPoint3.at<uchar>(i, j) = 0; }
					if (nearPoint1.at<uchar>(i, j) == 0 && nearPoint2.at<uchar>(i, j) == 0 && nearPoint3.at<uchar>(i, j) == 0)
						endPointMap.at<Vec2b>(i, j)[0] = 5;		//淘汰端點
				}
			}
	}

	/*斷線連通*/
	for (int x = startSpace; x <= endSpace; ++x)
	{
		Mat graddRef;
		copyMakeBorder(gradd, graddRef, x - 1, x - 1, x - 1, x - 1, BORDER_CONSTANT, Scalar(-1000.0f));

		Mat endPointMapRef;
		copyMakeBorder(endPointMap, endPointMapRef, x - 1, x - 1, x - 1, x - 1, BORDER_CONSTANT, Scalar(0, 0));

		/*搜尋並連通線*/
		for (int i = 1; i < gradd.rows - 1; ++i)		//不搜尋影像邊界
			for (int j = 1; j < gradd.cols - 1; ++j)		//不搜尋影像邊界
			{
				int ir = i + x - 1, jr = j + x - 1;		//reference index i,j for graddRef

				if (endPointMap.at<Vec2b>(i, j)[0] == 2)	//判斷是否為端點
				{
					float theta0 = (gradd.at<float>(i, j) / CV_PI + 1)*180.0f;	//目前端點角度
					float divtheta = 0.0f;		//搜索點相差角度
					float mintheta = 180.0f;	//最小相差角度

					char searchLocation = 0;	//四區域分類(1(W),2(N),3(E),4(S))
					int ik = 0;					//最佳搜尋點位置ik
					int jk = 0;					//最佳搜尋點位置jk

					float connectgradm = 0;		//連通目標幅值
					float connectgradd = 0.0f;		//連通目標方向

					bool flag1 = 1, flag2 = 1, flag3 = 1;	//相鄰flagT

					/*搜尋最佳點(八區域搜尋)*/
					if (endPointMap.at<Vec2b>(i, j)[1] == 1)		//8區域搜尋 - 1區(W區)
					{
						//區塊1(N->NE)
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
						//區塊2(NE->SE)
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
						//區塊3(SE->S)
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 2)		//8區域搜尋 - 2區(NW區)
					{
						//區塊1(NE->E)
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
						//區塊2(E->SE)
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
						//區塊2(SE->S)
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
						//區塊3(S->SW)
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 3)		//8區域搜尋 - 3區(N區)
					{
						//區塊1(E->SE)
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
						//區塊2(SE->SW)
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
						//區塊3(SW->W)
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 4)		//8區域搜尋 - 4區(NE區)
					{
						//區塊1(SE->S)
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
						//區塊2(S->SW)
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
						//區塊2(SW->W)
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
						//區塊3(W->NW)
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 5)		//8區域搜尋 - 5區(E區)
					{
						//區塊1(S->SW)
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
						//區塊2(SW->NW)
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
						//區塊3(NW->N)
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 6)		//8區域搜尋 - 6區(SE區)
					{
						//區塊1(SW->W)
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
						//區塊2(W->NW)
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
						//區塊2(NW->N)
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
						//區塊3(N->NE)
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 7)		//8區域搜尋 - 7區(S區)
					{
						//區塊1(W->NW)
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
						//區塊2(NW->NE)
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
						//區塊3(NE->E)
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 8)		//8區域搜尋 - 8區(SW區)
					{
						//區塊1(NW->N)
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
						//區塊2(N->NE)
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
						//區塊2(NE->E)
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
						//區塊3(E->SE)
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

					/*修改端點類型*/
					if (!flag1) { nearPoint1.at<uchar>(i, j) = 0; }
					if (!flag2) { nearPoint2.at<uchar>(i, j) = 0; }
					if (!flag3) { nearPoint3.at<uchar>(i, j) = 0; }
					if (nearPoint1.at<uchar>(i, j) == 0 && nearPoint2.at<uchar>(i, j) == 0 && nearPoint3.at<uchar>(i, j) == 0) { endPointMap.at<Vec2b>(i, j)[0] = 5; }	//淘汰端點

					/*連通最佳點(四區域連通)*/
					if (searchLocation == 1 && mintheta <= degree)		//4區域連通 - 1區(W區)
					{
						double a = (double)(j - jk) / (double)(i - ik);			//計算連通線斜率(j = a * i + b)
						double b = (double)(j*ik - jk*i) / (double)(ik - i);	//計算連通線截距(j = a * i + b)
						int step = 0;	//步數及權重(from 0 to x)

						for (int jc = j; jc >= jk; --jc, ++step)		//連通直線
						{
							int ic = (i - ik == 0) ? i : round((double)(jc - b) / a);		//防止數值Inf
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + gradm.at<uchar>(ik, jk)*step / x;
							graddCL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(gradd.at<float>(ik, jk))*step, cos(gradd.at<float>(i, j))*(x - step) + cos(gradd.at<float>(ik, jk))*step);
						}
					}
					else if (searchLocation == 2 && mintheta <= degree)		//4區域連通 - 2區(N區)
					{
						double a = (double)(j - jk) / (double)(i - ik);			//計算連通線斜率(j = a * i + b)
						double b = (double)(j*ik - jk*i) / (double)(ik - i);	//計算連通線截距(j = a * i + b)
						int step = 0;	//步數及權重(from 0 to x)

						for (int ic = i; ic >= ik; --ic, ++step)		//連通直線
						{
							int jc = round((double)ic * a + b);
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + gradm.at<uchar>(ik, jk)*step / x;
							graddCL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(gradd.at<float>(ik, jk))*step, cos(gradd.at<float>(i, j))*(x - step) + cos(gradd.at<float>(ik, jk))*step);
						}
					}
					else if (searchLocation == 3 && mintheta <= degree)		//4區域連通 - 3區(E區)
					{
						double a = (double)(j - jk) / (double)(i - ik);			//計算連通線斜率(j = a * i + b)
						double b = (double)(j*ik - jk*i) / (double)(ik - i);	//計算連通線截距(j = a * i + b)
						int step = 0;	//步數及權重(from 0 to x)

						for (int jc = j; jc <= jk; ++jc, ++step)		//連通直線
						{
							int ic = (i - ik == 0) ? i : round((double)(jc - b) / a);		//防止數值Inf
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + gradm.at<uchar>(ik, jk)*step / x;
							graddCL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(gradd.at<float>(ik, jk))*step, cos(gradd.at<float>(i, j))*(x - step) + cos(gradd.at<float>(ik, jk))*step);
						}
					}
					else if (searchLocation == 4 && mintheta <= degree)		//4區域連通 - 4區(S區)
					{
						double a = (double)(j - jk) / (double)(i - ik);			//計算連通線斜率(j = a * i + b)
						double b = (double)(j*ik - jk*i) / (double)(ik - i);	//計算連通線截距(j = a * i + b)
						int step = 0;	//步數及權重(from 0 to x)

						for (int ic = i; ic <= ik; ++ic, ++step)		//連通直線
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

/*滯後閥值*/
void HysteresisThreshold(InputArray _gradm, OutputArray _bwLine, int upperThreshold, int lowerThreshold)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	_bwLine.create(gradm.size(), CV_8UC1);
	Mat bwLine = _bwLine.getMat();

	Mat UT;		//上閥值二值化
	threshold(gradm, UT, upperThreshold, 255, THRESH_BINARY);
	Mat LT;		//下閥值二值化
	threshold(gradm, LT, lowerThreshold, 255, THRESH_BINARY);
	Mat MT;		//弱邊緣
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

/*影像處理*/
void ImageProcess(InputArray _image, OutputArray _imageIP)
{
	Mat image = _image.getMat();
	CV_Assert(image.type() == CV_8UC3);

	_imageIP.create(image.size(), CV_8UC3);
	Mat imageIP = _imageIP.getMat();

	for (int i = 0; i < imageIP.rows; ++i)
		for (int j = 0; j < imageIP.cols; ++j)
			for (int k = 0; k < 3; ++k)
			{
				imageIP.at<Vec3b>(i, j)[k] = 255 - image.at<Vec3b>(i, j)[k];
			}
}

/*清除孤立點*/
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
		Mat pointMap;	//點的種類
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

/*二值圖斷線連通*/
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

/*結合線與面的二值邊緣*/
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

/*反轉二值圖*/
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

/*分水嶺演算法*/
void BWWatershed(InputArray _srcImage, InputArray _bwSeed, InputArray _bwObject, OutputArray _bwWatershed)
{
	Mat srcImage = _srcImage.getMat();
	CV_Assert(srcImage.type() == CV_8UC1 || srcImage.type() == CV_8UC3);
	if (srcImage.type() == CV_8UC1) { cvtColor(srcImage, srcImage, CV_GRAY2BGR); }

	Mat bwSeed;
	Mat temp = _bwSeed.getMat();
	CV_Assert(temp.type() == CV_8UC1 || temp.type() == CV_32FC1);
	if (temp.type() == CV_32FC1) { temp.convertTo(bwSeed, CV_8UC1); }
	else { bwSeed = temp; }

	Mat bwObject = _bwObject.getMat();
	CV_Assert(bwObject.type() == CV_8UC1);

	_bwWatershed.create(srcImage.size(), CV_8UC1);
	Mat bwWatershed = _bwWatershed.getMat();
	bwObject.copyTo(bwWatershed);

	Mat seedLabels;
	int num = bwlabel(bwSeed, seedLabels, 8);

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

/*填補空洞*/
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

