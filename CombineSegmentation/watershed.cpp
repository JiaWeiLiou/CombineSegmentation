#include "stdafx.h"
#include "watershed.h"
#include "basic_processing.h"

PixelElement::PixelElement(float i, int j, int k)
{
	value = i;
	x = j;
	y = k;
}

/*影像形態學重建*/
void Reconstruct(InputArray _marker, InputArray _mask, OutputArray _objectR)
{
	Mat marker = _marker.getMat();
	CV_Assert(marker.type() == CV_32FC1);

	Mat mask = _mask.getMat();
	CV_Assert(mask.type() == CV_32FC1);

	_objectR.create(mask.size(), CV_32FC1);
	Mat objectR = _objectR.getMat();

	min(marker, mask, objectR);
	dilate(objectR, objectR, Mat());
	min(objectR, mask, objectR);
	Mat temp1 = Mat(marker.size(), CV_8UC1);
	Mat temp2 = Mat(marker.size(), CV_8UC1);
	do
	{
		objectR.copyTo(temp1);
		dilate(objectR, objectR, Mat());
		min(objectR, mask, objectR);
		compare(temp1, objectR, temp2, CV_CMP_NE);
	} while (sum(temp2).val[0] != 0);
}

/*H-min 轉換*/
void HMinimaTransform(InputArray _objectDT, OutputArray _objectHMT, float H)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	_objectHMT.create(objectDT.size(), CV_32FC1);
	Mat objectHMT = _objectHMT.getMat();

	Mat mask = objectDT;

	Mat marker(objectDT.size(), CV_32FC1);
	for (int i = 0; i < mask.rows; ++i)
		for (int j = 0; j < mask.cols; ++j)
			marker.at<float>(i, j) = mask.at<float>(i, j) - H;

	Reconstruct(marker, mask, objectHMT);
}

/*區域最小值*/
void LocalMinimaDetection(InputArray _objectDT, OutputArray _objectLM)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	_objectLM.create(objectDT.size(), CV_8UC1);
	Mat objectLM = _objectLM.getMat();

	for (int i = 0; i < objectLM.rows; ++i)
		for (int j = 0; j < objectLM.cols; ++j)
			objectLM.at<uchar>(i, j) = 255;

	queue<Point2i> *mpFifo = new queue<Point2i>();

	for (int y = 0; y < objectLM.rows; ++y)
		for (int x = 0; x < objectLM.cols; ++x)
			if (objectLM.at<uchar>(y, x) == 255)
			{
				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx)
						if ((x + dx >= 0) && (x + dx < objectDT.cols) && (y + dy >= 0) && (y + dy < objectDT.rows))
						{
							if (floor(objectDT.at<float>(y, x)) > floor(objectDT.at<float>(y + dy, x + dx)))  // If pe2.value < pe1.value, pe1 is not a local minimum
							{
								objectLM.at<uchar>(y, x) = 0;
								mpFifo->push(Point2i(x, y));

								while (!(mpFifo->empty()))
								{
									Point2i pe3 = mpFifo->front();
									mpFifo->pop();

									int xh = pe3.x;
									int yh = pe3.y;

									for (int dyh = -1; dyh <= 1; ++dyh)
										for (int dxh = -1; dxh <= 1; ++dxh)
											if ((xh + dxh >= 0) && (xh + dxh < objectDT.cols) && (yh + dyh >= 0) && (yh + dyh < objectDT.rows))
												if (objectLM.at<uchar>(yh + dyh, xh + dxh) == 255)
												{
													if (floor(objectDT.at<float>(yh + dyh, xh + dxh)) == floor(objectDT.at<float>(y, x)))
													{
														objectLM.at<uchar>(yh + dyh, xh + dxh) = 0;
														mpFifo->push(Point2i(xh + dxh, yh + dyh));
													}
												}
								}
							}
						}

			}
}

/*區域最小值*/
void LocalMinimaDetection(InputArray _objectDT, OutputArray _label, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue, float precision)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	_label.create(objectDT.size(), CV_32SC1);
	Mat label = _label.getMat();

	for (int i = 0; i < label.rows; ++i)
		for (int j = 0; j < label.cols; ++j)
			label.at<int>(i, j) = LABEL_UNPROCESSED;

	queue<Point2i> *mpFifo = new queue<Point2i>();

	for (int y = 0; y < objectDT.rows; ++y)
		for (int x = 0; x < objectDT.cols; ++x)
			if (label.at<int>(y, x) == LABEL_UNPROCESSED)
			{
				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx)
						if ((x + dx >= 0) && (x + dx < objectDT.cols) && (y + dy >= 0) && (y + dy < objectDT.rows))
						{
							if (floor(objectDT.at<float>(y, x) / precision) > floor(objectDT.at<float>(y + dy, x + dx) / precision))  // If pe2.value < pe1.value, pe1 is not a local minimum
							{
								label.at<int>(y, x) = LABEL_NOLOCALMINIMUM;
								mpFifo->push(Point2i(x, y));

								while (!(mpFifo->empty()))
								{
									Point2i pe3 = mpFifo->front();
									mpFifo->pop();

									int xh = pe3.x;
									int yh = pe3.y;

									for (int dyh = -1; dyh <= 1; ++dyh)
										for (int dxh = -1; dxh <= 1; ++dxh)
											if ((xh + dxh >= 0) && (xh + dxh < objectDT.cols) && (yh + dyh >= 0) && (yh + dyh < objectDT.rows))
												if (label.at<int>(yh + dyh, xh + dxh) == LABEL_UNPROCESSED)
												{
													if (floor(objectDT.at<float>(yh + dyh, xh + dxh) / precision) == floor(objectDT.at<float>(y, x) / precision))
													{
														label.at<int>(yh + dyh, xh + dxh) = LABEL_NOLOCALMINIMUM;
														mpFifo->push(Point2i(xh + dxh, yh + dyh));
													}
												}
								}
							}
						}

			}
	delete mpFifo;

	mpFifo = new queue<Point2i>();
	int mNumberOfLabels = LABEL_MINIMUM;

	for (int y = 0; y < objectDT.rows; ++y)
		for (int x = 0; x < objectDT.cols; ++x)
			if (label.at<int>(y, x) == LABEL_UNPROCESSED)
			{
				label.at<int>(y, x) = mNumberOfLabels;
				mpFifo->push(Point2i(x, y));

				while (!(mpFifo->empty()))
				{
					Point2i fifoElement = mpFifo->front();
					int xf = fifoElement.x;
					int yf = fifoElement.y;

					for (int dyf = -1; dyf <= 1; ++dyf)
						for (int dxf = -1; dxf <= 1; ++dxf)
							if ((xf + dxf >= 0) && (xf + dxf < objectDT.cols) && (yf + dyf >= 0) && (yf + dyf < objectDT.rows))
							{
								if (label.at<int>(yf + dyf, xf + dxf) == LABEL_UNPROCESSED)
								{
									label.at<int>(yf + dyf, xf + dxf) = mNumberOfLabels;
									mpFifo->push(Point2i(xf + dxf, yf + dyf));
								}
								else if (label.at<int>(yf + dyf, xf + dxf) == LABEL_NOLOCALMINIMUM)
								{
									label.at<int>(yf + dyf, xf + dxf) = LABEL_PROCESSING;
									mvSortedQueue.push(PixelElement(objectDT.at<float>(yf + dyf, xf + dxf), xf + dxf, yf + dyf));
								}
							}

					mpFifo->pop();
				}
				++mNumberOfLabels;
			}
	delete mpFifo;
}

/*最小值*/
void MinimaDetection(InputArray _objectIM, OutputArray _label, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue)
{
	Mat objectIM = _objectIM.getMat();
	CV_Assert(objectIM.type() == CV_32FC1);

	_label.create(objectIM.size(), CV_32SC1);
	Mat label = _label.getMat();

	float min = 0.0f;
	for (int i = 0; i < objectIM.rows; ++i)
		for (int j = 0; j < objectIM.cols; ++j)
			if (objectIM.at<float>(i, j) < min) { min = objectIM.at<float>(i, j); }

	for (int i = 0; i < label.rows; ++i)
		for (int j = 0; j < label.cols; ++j)
			label.at<int>(i, j) = LABEL_UNPROCESSED;

	queue<Point2i> *mpFifo = new queue<Point2i>();

	for (int y = 0; y < objectIM.rows; ++y)
		for (int x = 0; x < objectIM.cols; ++x)
			if (label.at<int>(y, x) == LABEL_UNPROCESSED)
			{
				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx)
						if ((x + dx >= 0) && (x + dx < objectIM.cols) && (y + dy >= 0) && (y + dy < objectIM.rows))
						{
							if (objectIM.at<float>(y, x) != min)
							{
								label.at<int>(y, x) = LABEL_NOLOCALMINIMUM;
								mpFifo->push(Point2i(x, y));

								while (!(mpFifo->empty()))
								{
									Point2i pe3 = mpFifo->front();
									mpFifo->pop();

									int xh = pe3.x;
									int yh = pe3.y;

									for (int dyh = -1; dyh <= 1; ++dyh)
										for (int dxh = -1; dxh <= 1; ++dxh)
											if ((xh + dxh >= 0) && (xh + dxh < objectIM.cols) && (yh + dyh >= 0) && (yh + dyh < objectIM.rows))
												if (label.at<int>(yh + dyh, xh + dxh) == LABEL_UNPROCESSED)
												{
													if (objectIM.at<float>(yh + dyh, xh + dxh) == objectIM.at<float>(y, x))
													{
														label.at<int>(yh + dyh, xh + dxh) = LABEL_NOLOCALMINIMUM;
														mpFifo->push(Point2i(xh + dxh, yh + dyh));
													}
												}
								}
							}
						}

			}
	delete mpFifo;

	mpFifo = new queue<Point2i>();
	int mNumberOfLabels = LABEL_MINIMUM;

	for (int y = 0; y < objectIM.rows; ++y)
		for (int x = 0; x < objectIM.cols; ++x)
			if (label.at<int>(y, x) == LABEL_UNPROCESSED)
			{
				label.at<int>(y, x) = mNumberOfLabels;
				mpFifo->push(Point2i(x, y));

				while (!(mpFifo->empty()))
				{
					Point2i fifoElement = mpFifo->front();
					int xf = fifoElement.x;
					int yf = fifoElement.y;

					for (int dyf = -1; dyf <= 1; ++dyf)
						for (int dxf = -1; dxf <= 1; ++dxf)
							if ((xf + dxf >= 0) && (xf + dxf < objectIM.cols) && (yf + dyf >= 0) && (yf + dyf < objectIM.rows))
							{
								if (label.at<int>(yf + dyf, xf + dxf) == LABEL_UNPROCESSED)
								{
									label.at<int>(yf + dyf, xf + dxf) = mNumberOfLabels;
									mpFifo->push(Point2i(xf + dxf, yf + dyf));
								}
								else if (label.at<int>(yf + dyf, xf + dxf) == LABEL_NOLOCALMINIMUM)
								{
									label.at<int>(yf + dyf, xf + dxf) = LABEL_PROCESSING;
									mvSortedQueue.push(PixelElement(objectIM.at<float>(yf + dyf, xf + dxf), xf + dxf, yf + dyf));
								}
							}

					mpFifo->pop();
				}
				++mNumberOfLabels;
			}
	delete mpFifo;
}

/*檢查相鄰區域是否存在標籤*/
bool CheckForAlreadyLabeledNeighbours(int x, int y, Mat &label, Point2i &outLabeledNeighbour, int &outLabel)
{
	for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++)
			if ((x + dx >= 0) && (x + dx < label.cols) && (y + dy >= 0) && (y + dy < label.rows))
				if (label.at<int>(y + dy, x + dx) > LABEL_RIDGE)
				{
					outLabeledNeighbour = Point2i(x + dx, y + dy);
					outLabel = label.at<int>(y + dy, x + dx);
					return true;
				}
	return false;
}

/*檢查是否為分水嶺*/
bool CheckIfPixelIsWatershed(int x, int y, Mat &label, Point2i &inLabeledNeighbour, int &inLabelOfNeighbour)
{
	for (int dy = -1; dy <= 1; dy++)
		for (int dx = -1; dx <= 1; dx++)
			if ((x + dx >= 0) && (x + dx < label.cols) && (y + dy >= 0) && (y + dy < label.rows))
				if ((label.at<int>(y + dy, x + dx) >= LABEL_MINIMUM) && (label.at<int>(y + dy, x + dx) != label.at<int>(inLabeledNeighbour.y, inLabeledNeighbour.x)) && ((inLabeledNeighbour.x != x + dx) || (inLabeledNeighbour.y != y + dy)))
				{
					label.at<int>(y, x) = LABEL_RIDGE;
					return true;
				}
	return false;
}

/*擴展區域最小值*/
void ExtendLocalMinimaDetection(InputArray _objectFH, OutputArray _objectEM, float H)
{
	Mat objectFH = _objectFH.getMat();
	CV_Assert(objectFH.type() == CV_8UC1);

	_objectEM.create(objectFH.size(), CV_8UC1);
	Mat objectEM = _objectEM.getMat();

	Mat objectDT;		//距離轉換(32FC1(BW))
	distanceTransform(objectFH, objectDT, CV_DIST_L2, 3);

	Mat objectHMT;
	HMinimaTransform(objectDT, objectHMT, H);

	for (int i = 0; i < objectHMT.rows; ++i)
		for (int j = 0; j < objectHMT.cols; ++j)
			objectHMT.at<float>(i, j) = -objectHMT.at<float>(i, j);

	LocalMinimaDetection(objectHMT, objectEM);
}

/*增加未標記之標籤*/
void AddLabel(InputArray _object, InputArray _objectSeed, OutputArray _objectAL)
{
	Mat object = _object.getMat();
	CV_Assert(object.type() == CV_8UC1);

	Mat objectSeed = _objectSeed.getMat();
	CV_Assert(objectSeed.type() == CV_8UC1);

	_objectAL.create(object.size(), CV_8UC1);
	Mat objectAL = _objectAL.getMat();

	for (int i = 0; i < object.rows; ++i)
		for (int j = 0; j < object.cols; ++j)
			if (object.at<uchar>(i, j) == 0) { objectSeed.at<uchar>(i, j) = 0; }
	
	objectSeed.copyTo(objectAL);

	Mat labels;
	int num = bwlabel(object, labels, 8) + 1;	//include label 0

	int *labeltable = new int[num];  
	memset(labeltable, 0, num * sizeof(int));

	for (int i = 0; i < object.rows; ++i)
		for (int j = 0; j < object.cols; ++j)
			if (objectSeed.at<uchar>(i, j) != 0) { labeltable[labels.at<int>(i, j)] = 1; }

	labeltable[0] = 1;

	for (int i = 0; i < object.rows; ++i)
		for (int j = 0; j < object.cols; ++j)
			if (labeltable[labels.at<int>(i, j)] == 0) 
			{ 
				objectAL.at<uchar>(i, j) = 255; 
			}
}

/*加深低窪區*/
void ImposeMinima(InputArray _objectDT, InputArray _objectAL, OutputArray _objectIM)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	Mat objectAL = _objectAL.getMat();
	CV_Assert(objectAL.type() == CV_8UC1);

	_objectIM.create(objectDT.size(), CV_32FC1);
	Mat objectIM = _objectIM.getMat();

	float min = 0;
	for (int i = 0; i < objectDT.rows; ++i)
		for (int j = 0; j < objectDT.cols; ++j)
		{
			objectDT.at<float>(i, j) = -objectDT.at<float>(i, j);
			if (objectDT.at<float>(i, j) < min) { min = objectDT.at<float>(i, j); }
		}

	objectDT.copyTo(objectIM);
			
	for (int i = 0; i < objectDT.rows; ++i)
		for (int j = 0; j < objectDT.cols; ++j)
			if (objectAL.at<uchar>(i, j) != 0) { objectIM.at<float>(i, j) = min; }
}

/*分水嶺轉換*/
void WatershedTransform(InputArray _object, InputArray _objectIM, OutputArray _objectWT)
{
	Mat object = _object.getMat();
	CV_Assert(object.type() == CV_8UC1);

	Mat objectIM = _objectIM.getMat();
	CV_Assert(objectIM.type() == CV_32FC1);

	_objectWT.create(objectIM.size(), CV_8UC1);
	Mat objectWT = _objectWT.getMat();

	Mat label;
	priority_queue<PixelElement, vector<PixelElement>, mycomparison> mvSortedQueue;
	MinimaDetection(objectIM, label, mvSortedQueue);

	while (!mvSortedQueue.empty())
	{
		PixelElement lItemA = mvSortedQueue.top();
		mvSortedQueue.pop();
		int labelOfNeighbour = 0;

		// (a) Pop element and find positive labeled neighbour
		Point2i alreadyLabeledNeighbour;
		int x = lItemA.x;
		int y = lItemA.y;

		// (b) Check if current pixel is watershed pixel by checking if there are different finally labeled neighbours
		if (CheckForAlreadyLabeledNeighbours(x, y, label, alreadyLabeledNeighbour, labelOfNeighbour))
			if (!(CheckIfPixelIsWatershed(x, y, label, alreadyLabeledNeighbour, labelOfNeighbour)))
			{
				// c) if not watershed pixel, assign label of neighbour and add the LABEL_NOLOCALMINIMUM neighbours to priority_queue for processing
				/*UpdateLabel(x, y, objectIM, label, labelOfNeighbour, mvSortedQueue);*/

				label.at<int>(y, x) = labelOfNeighbour;

				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx)
						if ((x + dx >= 0) && (x + dx < label.cols) && (y + dy >= 0) && (y + dy < label.rows))
							if (label.at<int>(y + dy, x + dx) == LABEL_NOLOCALMINIMUM)
							{
								label.at<int>(y + dy, x + dx) = LABEL_PROCESSING;
								mvSortedQueue.push(PixelElement(objectIM.at<float>(y + dy, x + dx), x + dx, y + dy));
							}
			}
	}

	object.copyTo(objectWT);

	// d) finalize the labelImage
	for (int y = 0; y < objectWT.rows; ++y)
		for (int x = 0; x < objectWT.cols; ++x)
			if (label.at<int>(y, x) == LABEL_RIDGE) { objectWT.at<uchar>(y, x) = 0; }
}