#include "stdafx.h"
#include "BgGussian.h"


BgGussian::BgGussian()
{
}


BgGussian::~BgGussian()
{
}

BgGussian::BgGussian(const char *file_name, const char *section_name)
{
	fstream bgFile;
	bgFile.open(file_name, ios::in);
	if (!bgFile)
	{
		cout << "[SmokeDetecterPro.cpp] MOG: Background Model File " << file_name << " Does not exist " << endl;
		return;
	}
	num_history_ = ZIni::readInt(section_name, "num_history", 10, file_name);
	var_threahold_ = ZIni::readInt(section_name, "var_threahold", 16, file_name);
	use_shadow_detection_ = ZIni::readInt(section_name, "use_shadow_detection", 1, file_name);
	bg_size_.width = ZIni::readInt(section_name, "bg_width", 800, file_name);
	bg_size_.height = ZIni::readInt(section_name, "bg_height", 576, file_name);
	bg_count_threahold_ = ZIni::readInt(section_name, "bg_count_thredhold", 20, file_name);
	learn_rate_ = ZIni::readDouble(section_name, "learn_rate", 0.001, file_name);
	morphologyEx_times_ = ZIni::readInt(section_name, "morphologyEx_times", 1, file_name);
	dilate_times_ = ZIni::readInt(section_name, "dilate_times", 1, file_name);
	mog_ = BackgroundSubtractorMOG2(num_history_, var_threahold_, use_shadow_detection_);
	bgFile.close();
}

void BgGussian::Update(const Mat& frame)
{
	if (frame.empty())
	{
		cout << "[BgGussian] Update:: frame is empty";
		return;
	}
	Mat gray_frame;
	if (frame.channels() == 3)
	{
		cvtColor(frame, gray_frame, CV_RGB2GRAY);
	}
	else gray_frame = frame;
	Mat resize_frame;

	if (gray_frame.cols != bg_size_.width || gray_frame.rows != bg_size_.height)
	{
		resize(gray_frame, resize_frame, bg_size_);
	}
	else resize_frame = gray_frame;
	mog_(resize_frame, mask_, learn_rate_);

	for (size_t ii = 0; ii < morphologyEx_times_; ii++)
	{
		morphologyEx(mask_, mask_, MORPH_OPEN, Mat());
	}
	for (size_t ii = 0; ii < dilate_times_; ii++)
	{
		dilate(mask_, mask_, NULL, cvPoint(-1, -1), 2);
	}

	if (mask_.empty())bg_status_ = SETUP;
	else bg_status_ = UPDATING;

}

bool BgGussian::BeImageStable()
{
	try
	{
		int count(0);
		Mat resizeMask;
		resize(mask_, resizeMask, Size(mask_.cols / 2, mask_.rows / 2));

		morphologyEx(resizeMask, resizeMask, MORPH_OPEN, Mat());
		dilate(resizeMask, resizeMask, NULL, cvPoint(-1, -1), 2);

		const uchar *pdata = resizeMask.data;
		int threadhold = (float)(bg_count_threahold_ * mask_.rows*mask_.cols) / 400;
		for (int i = 0; i<resizeMask.rows; i++)
		{
			for (int j = 0; j<resizeMask.cols; j++)
			{
				if (*(pdata + i*resizeMask.cols + j)>0)
				{
					count++;
					if (count > threadhold) return false;
				}
			}
		}
		return true;
	}
	catch (...)
	{

	}
}