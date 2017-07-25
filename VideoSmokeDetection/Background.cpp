#include "stdafx.h"
#include "Background.h"

Background::Background()
{

}


Background::~Background()
{

}


bool Background::Stable(const Mat &mask)
{
	CHECK(!mask.empty()) << " Mask for stable check is empty" << endl;
	//CHECK(bg_parameter_.has_stable_threshold()) << " Frame stable threshold is not set" << endl;
	Mat resizeMask;
	resize(mask, resizeMask, Size(mask.cols / 2, mask.rows / 2));
	int threshold_count = (bg_parameter_.stable_threshold() * mask.rows*mask.cols) / 4;
	int non_zero_count= countNonZero(resizeMask);
	if (non_zero_count>threshold_count)
	{
		return false;
	}
	return true;
}

void Background::MaskOperation(Mat &mask)
{
	CHECK(!mask.empty()) << " Mask for image operation is empty" << endl;
	if (bg_parameter_.has_bg_operation())
	{
		for (size_t ii = 0; ii < bg_parameter_.bg_operation().morphology_open_times(); ii++)
		{
			morphologyEx(mask, mask, MORPH_OPEN, Mat());
		}
		for (size_t ii = 0; ii < bg_parameter_.bg_operation().dilate_times(); ii++)
		{
			dilate(mask,mask, NULL, cvPoint(-1, -1), 2);
		}
	}
}