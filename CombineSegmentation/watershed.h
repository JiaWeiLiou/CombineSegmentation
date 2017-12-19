#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <queue>

// Minimum watershed segment label
#define LABEL_MINIMUM 0

// Label for the watersheds pixels (local minima)
#define LABEL_RIDGE -1

// Label for pixels that are not local minima
#define LABEL_NOLOCALMINIMUM -2

// Label for pixels that are or have been in the OrderedQueques
#define LABEL_PROCESSING -3

// Label for unprocessed pixels
#define LABEL_UNPROCESSED -4

using namespace std;
using namespace cv;

class PixelElement
{
public:
	PixelElement(float i, int j, int k);
	float value;  ///< pixel intensity
	int x;      ///< x coordinate of pixel
	int y;      ///< y coordinate of pixel
};

class mycomparison
{
	bool reverse;
public:
	/** \brief constructor
	* \param revparam true if we want to sort from large to small*/
	mycomparison(const bool& revparam = true)
	{
		reverse = revparam;
	}

	/** \brief comparison operator: compares the values of two pixel elements
	* \param lhs first pixel element
	* \param rhs second pixel element*/
	bool operator() (PixelElement& lhs, PixelElement& rhs) const
	{
		if (reverse) return (lhs.value > rhs.value);
		else return (lhs.value < rhs.value);
	}
};

/*區域最小值*/
void LocalMinimaDetection(InputArray _objectDT, OutputArray _label, priority_queue<PixelElement, vector<PixelElement>, mycomparison> &mvSortedQueue);

/*檢查相鄰區域是否存在標籤*/
bool CheckForAlreadyLabeledNeighbours(int x, int y, Mat &label, Point2i &outLabeledNeighbour, int &outLabel);

/*檢查是否為分水嶺*/
bool CheckIfPixelIsWatershed(int x, int y, Mat &label, Point2i &inLabeledNeighbour, int &inLabelOfNeighbour);

/*分水嶺轉換*/
void WatershedTransform(InputArray _objectDT, OutputArray _objectWT, OutputArray _label);
