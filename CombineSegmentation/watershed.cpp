#include "stdafx.h"
#include "watershed.h"

PixelElement::PixelElement(float i, int j, int k)
{
	value = i;
	x = j;
	y = k;
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

/*分水嶺轉換*/
void WatershedTransform(InputArray _objectDT, OutputArray _objectWT, OutputArray _label, float precision)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	_objectWT.create(objectDT.size(), CV_8UC1);
	Mat objectWT = _objectWT.getMat();

	_label.create(objectDT.size(), CV_32SC1);
	Mat label = _label.getMat();

	//Mat label;
	priority_queue<PixelElement, vector<PixelElement>, mycomparison> mvSortedQueue;
	LocalMinimaDetection(objectDT, label, mvSortedQueue, precision);

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
				/*UpdateLabel(x, y, objectDT, label, labelOfNeighbour, mvSortedQueue);*/

				label.at<int>(y, x) = labelOfNeighbour;

				for (int dy = -1; dy <= 1; ++dy)
					for (int dx = -1; dx <= 1; ++dx)
						if ((x + dx >= 0) && (x + dx < label.cols) && (y + dy >= 0) && (y + dy < label.rows))
							if (label.at<int>(y + dy, x + dx) == LABEL_NOLOCALMINIMUM)
							{
								label.at<int>(y + dy, x + dx) = LABEL_PROCESSING;
								mvSortedQueue.push(PixelElement(objectDT.at<float>(y + dy, x + dx), x + dx, y + dy));
							}
			}
	}

	// d) finalize the labelImage
	for (int y = 0; y < objectWT.rows; ++y)
		for (int x = 0; x < objectWT.cols; ++x)
			if (label.at<int>(y, x) == LABEL_RIDGE) { objectWT.at<uchar>(y, x) = 0; }
			else { objectWT.at<uchar>(y, x) = 255; }
}