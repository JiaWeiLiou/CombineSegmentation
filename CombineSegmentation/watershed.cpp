#include "stdafx.h"
#include "watershed.h"

PixelElement::PixelElement(float i, int j, int k)
{
	value = i;
	x = j;
	y = k;
}

/*跋办程*/
void LocalMinimaDetection(InputArray _objectDT, OutputArray _label, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue)
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
							if (objectDT.at<float>(y, x) > objectDT.at<float>(y + dy, x + dx))  // If pe2.value < pe1.value, pe1 is not a local minimum
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
													if (objectDT.at<float>(yh + dyh, xh + dxh) == objectDT.at<float>(y, x))
													{
														label.at<int>(yh + dyh, xh + dxh) = LABEL_NOLOCALMINIMUM;
														mpFifo->push(Point2i(x + dx, y + dy));
													}
												}
								}
							}
						}

			}
	delete mpFifo;

	for (int y = 0; y < objectDT.rows; ++y)
		for (int x = 0; x < objectDT.cols; ++x)
			if (objectDT.at<float>(y, x) == 0) { label.at<int>(y, x) = LABEL_NOLOCALMINIMUM; }

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

//void UpdateLabel(int x, int y, Mat &srcImage, Mat &label, int &inLabelOfNeighbour, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue)
//{
//	// from c) assign label of neighbour
//	label.at<int>(y, x) = inLabelOfNeighbour;
//
//	for (int dy = -1; dy <= 1; dy++)
//		for (int dx = -1; dx <= 1; dx++)
//			if ((x + dx >= 0) && (x + dx < label.cols) && (y + dy >= 0) && (y + dy < label.rows))
//				if (label.at<int>(y + dy, x + dx) == LABEL_NOLOCALMINIMUM)
//				{
//					label.at<int>(y + dy, x + dx) = LABEL_PROCESSING;
//					mvSortedQueue.push(PixelElement(srcImage.at<float>(y + dy, x + dx), x + dx, y + dy));
//				}
//}

/*だ拉锣传*/
void WatershedTransform(InputArray _objectDT, OutputArray _objectWT, OutputArray _label)
{
	Mat objectDT = _objectDT.getMat();
	CV_Assert(objectDT.type() == CV_32FC1);

	_objectWT.create(objectDT.size(), CV_8UC1);
	Mat objectWT = _objectWT.getMat();

	_label.create(objectDT.size(), CV_32SC1);
	Mat label = _label.getMat();

	//Mat label;
	priority_queue<PixelElement, vector<PixelElement>, mycomparison> mvSortedQueue;
	LocalMinimaDetection(objectDT, label, mvSortedQueue);

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