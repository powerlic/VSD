#include "stdafx.h"
#include "SmokeFilter.h"


SmokeFilter::SmokeFilter()
{
}


SmokeFilter::~SmokeFilter()
{
	gray_frame_list_.clear();
	gray_diff_frame_list_.clear();
	//cout << "smoke filter destroy" << endl;

	/*filter_frame_.release();
	filter_fore_mask_.release();
	gray_filter_frame_.release();

	diff_acc_mask_.release();
	diff_acc_frame_.release();
	color_mask_.release();*/
}


bool SmokeFilter::UpdateFilter(const Mat &frame, const Mat &fore_mask)
{
	if (!lio::CheckFrame(frame, 1)) return false;
	if (!lio::CheckFrame(fore_mask, 0)) return false;

	if (!lio::SizeEqual(frame.size(), filter_size_))
	{
		resize(frame, filter_frame_, filter_size_);
	}
	else frame.copyTo(filter_frame_);

	cvtColor(filter_frame_, gray_filter_frame_, CV_RGB2GRAY);

	if (!lio::SizeEqual(fore_mask.size(), filter_size_))
	{
		resize(fore_mask, filter_fore_mask_, filter_size_);
	}

	if (smoke_filter_mode_ == FRAME_COLOR_CPU_MODE||smoke_filter_mode_ == FRAME_COLOR_GPU_MODE)
	{
		AccumulateDiffImage(gray_filter_frame_);
	}
	return true;

}

void SmokeFilter::Filtrate(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	if (smoke_filter_mode_ == FRAME_COLOR_CPU_MODE || smoke_filter_mode_ == FRAME_COLOR_GPU_MODE)
	{
		FrameFiltrate(frame, fore_mask,out_mask, contours, rects);
	}
	else if (smoke_filter_mode_ == CONTOUR_COLOR_ACC_MODE || smoke_filter_mode_ == CONTOUR_COLOR_MODE || smoke_filter_mode_ == CONTOUR_ACC_MODE)
	{
		ContourFiltrate(frame, fore_mask, out_mask, contours, rects);
	}
	else if (smoke_filter_mode_ == PURE_CONTOUR_AREA_MODE)
	{
		PureContourFiltrate(frame, fore_mask, out_mask, contours, rects);
	}

}
void SmokeFilter::FrameFiltrate(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	if (!UpdateFilter(frame, fore_mask)) return;

	/*--------------ACC DIFF-----------------*/
	Mat acc_filtered_mask;
	if (!diff_acc_mask_.empty())
	{
		Mat revers_mat = diff_acc_mask_ < 10;
		bitwise_and(filter_fore_mask_, revers_mat, acc_filtered_mask);
	}
	else  acc_filtered_mask = filter_fore_mask_;

	//imshow("acc", diff_acc_mask_);

	/*--------------Color Filter--------------*/
	out_mask = Mat::zeros(filter_size_, CV_8UC1);
	if (smoke_filter_mode_ == FRAME_COLOR_GPU_MODE)
	{
		ColorFilter_Caller(filter_frame_, acc_filtered_mask, &th_ycbcr_, &th_rgb_, filter_frame_.cols, filter_frame_.rows, out_mask, color_mask_);
	}
	else if (smoke_filter_mode_ == FRAME_COLOR_CPU_MODE)
	{
		ColorPixelFilter(filter_frame_, acc_filtered_mask, out_mask);
	}

	//imshow("color", out_mask);

	
	area_perimeter_ratio_threahold_ = 0.001;
	lio::FindContoursRects(out_mask, contour_area_threahold_, contour_perimeter_threahold_, area_perimeter_ratio_threahold_, rects, contours);
}

void SmokeFilter::PureContourFiltrate(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	if (!UpdateFilter(frame, fore_mask)) return;
	out_mask = Mat::zeros(filter_size_, CV_8UC1);

	
	lio::FindContoursRects(fore_mask, contour_area_threahold_, contour_perimeter_threahold_, area_perimeter_ratio_threahold_, rects, contours);
	
	drawContours(out_mask, contours, -1, Scalar(255),CV_FILLED);
	
}

void SmokeFilter::ContourFiltrate(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	if (!UpdateFilter(frame, fore_mask)) return;

	if (smoke_filter_mode_ == CONTOUR_COLOR_ACC_MODE)
	{
		FiltrateContourCombined(frame, fore_mask, out_mask, contours, rects);
	}
	else if (smoke_filter_mode_ == CONTOUR_ACC_MODE)
	{
		FiltrateContourAcc(frame, fore_mask, out_mask, contours, rects);
	}
	else if (smoke_filter_mode_ == CONTOUR_COLOR_MODE)
	{
		FiltrateContourColor(frame, fore_mask, out_mask, contours, rects);
	}
}
void SmokeFilter::FiltrateContourCombined(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	contours.clear();
	rects.clear();
	vector<CvRect> f_rects;
	vector<vector<Point>> t_contours;
	vector<CvRect> t_rects;

	
	lio::FindContoursRects(filter_fore_mask_, contour_area_threahold_, 
			                   contour_perimeter_threahold_, area_perimeter_ratio_threahold_, 
							   t_rects, t_contours);
	

	Mat t_contours_filled_mask = Mat::zeros(filter_size_, CV_8UC1);
	if (out_mask.empty())
	{
		out_mask = Mat::zeros(filter_size_, CV_8UC1);
	}
	
	for (size_t i = 0; i < t_contours.size() && i < t_rects.size(); i++)
	{
		vector<Point> contour = t_contours[i];
		CvRect t_rect = t_rects[i];

		Mat t_rect_mat = filter_frame_(t_rect);
		drawContours(t_contours_filled_mask, t_contours, i, Scalar(255), CV_FILLED);
		Mat t_rect_contours_mat = t_contours_filled_mask(t_rect);
		Mat t_rect_acc_mat = diff_acc_frame_(t_rect);

		int move_count = 0;
		int rgb_fit_count = 0;
		int acc_fit_count = 0;

		for (size_t ii = 0; ii < t_rect.height; ii++)
		for (size_t jj = 0; jj < t_rect.height; jj++)
		{
			uchar r, g, b, p, q;
			r = *(t_rect_mat.data + ii*t_rect.width * 3 + jj * 3);
			g = *(t_rect_mat.data + ii*t_rect.width * 3 + jj * 3 + 1);
			b = *(t_rect_mat.data + ii*t_rect.width * 3 + jj * 3 + 2);

			p = *(t_rect_contours_mat.data + ii*t_rect.width + jj);
			q = *(t_rect_acc_mat.data + ii*t_rect.width + jj);

			if (p>0)
			{
				move_count++;
				if (PixelColorFit(r,g,b))
				{
					rgb_fit_count++;
				}
				if (q>0)
				{
					acc_fit_count++;
				}
			}
		}
		float acc_move_ratio = (float)acc_fit_count / move_count;
		float rgb_move_ratio = (float)rgb_fit_count / move_count;
		
		if (acc_move_ratio<acc_move_ratio_threahold_&&rgb_move_ratio>rbg_fit_move_ratio_)
		{
			//cout << "acc_move_ratio " << acc_move_ratio << endl;
			//cout << "rgb_move_ratio" << rgb_move_ratio << endl;
			//imshow("acc_rect", t_rect_acc_mat);
			//imshow("contour_rect", t_rect_contours_mat);
			//waitKey(-1);
			drawContours(out_mask, t_contours, i, Scalar(255), CV_FILLED);
			contours.push_back(contour);
			rects.push_back(t_rect);
		}
	}
}
void SmokeFilter::FiltrateContourColor(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	vector<vector<Point>> t_contours;
	vector<CvRect> t_rects;

	
	lio::FindContoursRects(filter_fore_mask_, contour_area_threahold_,contour_perimeter_threahold_, area_perimeter_ratio_threahold_,
			t_rects, t_contours);
	

	Mat t_contours_filled_mask = Mat::zeros(filter_size_, CV_8UC1);
	out_mask = Mat::zeros(filter_size_, CV_8UC1);

	contours.clear();
	rects.clear();

	for (size_t i = 0; i < t_contours.size(); i++)
	{
		vector<Point> contour = t_contours[i];
		CvRect rect = t_rects[i];

		Mat t_rect_mat = filter_frame_(rect);
		drawContours(t_contours_filled_mask, t_contours, i, Scalar(255), CV_FILLED);
		Mat t_rect_contours_mat = t_contours_filled_mask(rect);

		int move_count = 0;
		int rgb_fit_count = 0;

		for (size_t ii = 0; ii < rect.height; ii++)
		for (size_t jj = 0; jj < rect.height; jj++)
		{
			uchar r, g, b, p, q;
			r = *(t_rect_mat.data + ii*rect.width * 3 + jj * 3);
			g = *(t_rect_mat.data + ii*rect.width * 3 + jj * 3 + 1);
			b = *(t_rect_mat.data + ii*rect.width * 3 + jj * 3 + 2);

			p = *(t_rect_contours_mat.data + ii*rect.width + jj);

			if (p>0)
			{
				move_count++;
				if (PixelColorFit(r, g, b))
				{
					rgb_fit_count++;
				}
			}
		}
		float rgb_move_ratio = (float)rgb_fit_count / move_count;
		if (rgb_move_ratio>rbg_fit_move_ratio_)
		{
			contours.push_back(contour);
			rects.push_back(rect);
			drawContours(out_mask, t_contours, i, Scalar(255), CV_FILLED);
		}
	}
}
void SmokeFilter::FiltrateContourAcc(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	vector<vector<Point>> t_contours;
	vector<CvRect> t_rects;

	
		lio::FindContoursRects(filter_fore_mask_, contour_area_threahold_,
			contour_perimeter_threahold_, area_perimeter_ratio_threahold_,
			t_rects, t_contours);
	

	Mat t_contours_filled_mask = Mat::zeros(filter_size_, CV_8UC1);
	out_mask = Mat::zeros(filter_size_, CV_8UC1);

	contours.clear();
	rects.clear();

	for (size_t i = 0; i < t_contours.size(); i++)
	{
		vector<Point> contour = t_contours[i];
		CvRect rect = t_rects[i];

		drawContours(t_contours_filled_mask, t_contours, i, Scalar(255), CV_FILLED);
		Mat t_rect_contours_mat = t_contours_filled_mask(rect);
		Mat t_rect_acc_mat = diff_acc_frame_(rect);

		int move_count = 0;
		int acc_fit_count = 0;

		for (size_t ii = 0; ii < rect.height; ii++)
		for (size_t jj = 0; jj < rect.height; jj++)
		{
			uchar  p, q;
			p = *(t_rect_contours_mat.data + ii*rect.width + jj);
			q = *(t_rect_acc_mat.data + ii*rect.width + jj);
			if (p>0)
			{
				move_count++;
				if (q>0)
				{
					acc_fit_count++;
				}
			}
		}
		float acc_move_ratio = (float)acc_fit_count / move_count;
		if (acc_move_ratio<acc_move_ratio_threahold_)
		{
			contours.push_back(contour);
			rects.push_back(rect);
			drawContours(out_mask, t_contours, i, Scalar(255), CV_FILLED);
		}
	}
}


SmokeFilter::SmokeFilter(const char *file_name, const char *section_name)
{
	
	filter_size_.width = ZIni::readInt(section_name, "filter_width", 0, file_name);
	filter_size_.height = ZIni::readInt(section_name, "filter_height", 0, file_name);

	int filter_mode = ZIni::readInt(section_name, "filter_mode", 6, file_name);
	if (filter_mode == 1) smoke_filter_mode_ = FRAME_COLOR_CPU_MODE;
	else if (filter_mode == 2) smoke_filter_mode_ = FRAME_COLOR_GPU_MODE;
	else if (filter_mode == 3) smoke_filter_mode_ = CONTOUR_COLOR_ACC_MODE;
	else if (filter_mode == 4) smoke_filter_mode_ = CONTOUR_COLOR_MODE;
	else if (filter_mode == 5) smoke_filter_mode_ = CONTOUR_ACC_MODE;
	else if (filter_mode == 6) smoke_filter_mode_ = PURE_CONTOUR_AREA_MODE;


	frame_list_size_ = ZIni::readInt(section_name, "frame_list_size", 10, file_name);
	acc_threahold_ = ZIni::readInt(section_name, "acc_threahold", 50, file_name);

	contour_area_threahold_ = ZIni::readInt(section_name, "contour_area_threahold", 100, file_name);
	contour_perimeter_threahold_ = ZIni::readInt(section_name, "contour_perimeter_threahold", 100, file_name);
	area_perimeter_ratio_threahold_ = ZIni::readDouble(section_name, "area_perimeter_ratio_threahold_", 0.0001, file_name);

	th_rgb_.th1_lower = ZIni::readInt(section_name, "rgb_th1_lower", 0, file_name);
	th_rgb_.th1_upper = ZIni::readInt(section_name, "rgb_th1_upper", 50, file_name);
	th_rgb_.th2_lower = ZIni::readInt(section_name, "rgb_th2_lower", 120, file_name);
	th_rgb_.th2_upper = ZIni::readInt(section_name, "rgb_th2_upper", 230, file_name);
	th_rgb_.th3_lower = ZIni::readInt(section_name, "rgb_th3_lower", 0, file_name);
	th_rgb_.th3_upper = ZIni::readInt(section_name, "rgb_th3_upper", 0, file_name);

	th_ycbcr_.th1_lower = ZIni::readInt(section_name, "ycbcr_th1_lower", 50, file_name);
	th_ycbcr_.th1_upper = ZIni::readInt(section_name, "ycbcr_th1_upper", 255, file_name);
	th_ycbcr_.th2_lower = ZIni::readInt(section_name, "ycbcr_th2_lower", 130, file_name);
	th_ycbcr_.th2_upper = ZIni::readInt(section_name, "ycbcr_th2_upper", 250, file_name);
	th_ycbcr_.th3_lower = ZIni::readInt(section_name, "ycbcr_th3_lower", 0, file_name);
	th_ycbcr_.th3_upper = ZIni::readInt(section_name, "ycbcr_th3_upper", 0, file_name);
	
}
SmokeFilter::SmokeFilter(CvSize filter_size,
	const uint8_t &filter_mode,
	const int &contour_area_threahold,
	const int &contour_perimeter_threahold,
	const float &area_perimeter_ratio_threahold,
	const uint8_t &frame_list_size,
	const uint8_t &acc_threahold,
	const Threshold &th_rgb,
	const Threshold &th_ycbcr)
{

	filter_size_ = filter_size;


	if (filter_mode == 1) smoke_filter_mode_ = FRAME_COLOR_CPU_MODE;
	else if (filter_mode == 2) smoke_filter_mode_ = FRAME_COLOR_GPU_MODE;
	else if (filter_mode == 3) smoke_filter_mode_ = CONTOUR_COLOR_ACC_MODE;
	else if (filter_mode == 4) smoke_filter_mode_ = CONTOUR_COLOR_MODE;
	else if (filter_mode == 5) smoke_filter_mode_ = CONTOUR_ACC_MODE;
	else if (filter_mode == 6) smoke_filter_mode_ = PURE_CONTOUR_AREA_MODE;
	

	contour_area_threahold_ = contour_area_threahold;
	contour_perimeter_threahold_ = contour_perimeter_threahold;
	area_perimeter_ratio_threahold_ = area_perimeter_ratio_threahold;

	frame_list_size_ = frame_list_size;
	acc_threahold_ = acc_threahold;

	th_rgb_ = th_rgb;
	th_ycbcr_ = th_ycbcr;
}




/*-------------------------------------ACC-------------------------------*/
void SmokeFilter::InitAccThreahold(const char *file_name, const char *section_name)
{

	frame_list_size_ = ZIni::readInt(section_name, "frame_list_size", 10, file_name);
	acc_threahold_ = ZIni::readInt(section_name, "acc_threahold", 50, file_name);
}

void SmokeFilter::AccumulateDiffImage(const Mat &gray_frame)
{
	//Mat current_gray;
	//gray_frame.copyTo(current_gray);
	if (diff_acc_frame_.empty())
	{
		diff_acc_frame_ = Mat::zeros(filter_size_, CV_8UC1);
	}
	if (diff_acc_mask_.empty())
	{
		diff_acc_mask_ = Mat::zeros(filter_size_, CV_8UC1);
	}
	if (gray_frame_list_.size() == 0)
	{
		gray_frame_list_.push_back(gray_frame);
		return;
	}

	Mat currentDiff;
	absdiff(gray_frame, gray_frame_list_.back(), currentDiff);

	if (gray_frame_list_.size()<frame_list_size_)
	{
		gray_frame_list_.push_back(gray_frame);
		gray_diff_frame_list_.push_back(currentDiff);
		add(currentDiff, diff_acc_frame_, diff_acc_frame_);
	}
	else
	{
		subtract(diff_acc_frame_, gray_diff_frame_list_.front(), diff_acc_frame_);
		add(diff_acc_frame_, currentDiff, diff_acc_frame_);

		gray_diff_frame_list_.pop_front();
		gray_diff_frame_list_.push_back(currentDiff);

		gray_frame_list_.pop_front();
		gray_frame_list_.push_back(gray_frame);
		threshold(diff_acc_frame_, diff_acc_mask_, acc_threahold_, 255, THRESH_BINARY);
	}
}

/*-------------------------------------ColorFilter--------------------------------*/

void SmokeFilter::InitColorFilter(const char *file_name, const char *section_name)
{

	th_rgb_.th1_lower = ZIni::readInt(section_name, "rgb_th1_lower", 0, file_name);
	th_rgb_.th1_upper = ZIni::readInt(section_name, "rgb_th1_upper", 50, file_name);
	th_rgb_.th2_lower = ZIni::readInt(section_name, "rgb_th2_lower", 120, file_name);
	th_rgb_.th2_upper = ZIni::readInt(section_name, "rgb_th2_upper", 230, file_name);
	th_rgb_.th3_lower = ZIni::readInt(section_name, "rgb_th3_lower", 0, file_name);
	th_rgb_.th3_upper = ZIni::readInt(section_name, "rgb_th3_upper", 0, file_name);

	th_ycbcr_.th1_lower = ZIni::readInt(section_name, "ycbcr_th1_lower", 50, file_name);
	th_ycbcr_.th1_upper = ZIni::readInt(section_name, "ycbcr_th1_upper", 255, file_name);
	th_ycbcr_.th2_lower = ZIni::readInt(section_name, "ycbcr_th2_lower", 130, file_name);
	th_ycbcr_.th2_upper = ZIni::readInt(section_name, "ycbcr_th2_upper", 250, file_name);
	th_ycbcr_.th3_lower = ZIni::readInt(section_name, "ycbcr_th3_lower", 0, file_name);
	th_ycbcr_.th3_upper = ZIni::readInt(section_name, "ycbcr_th3_upper", 0, file_name);

}

inline bool SmokeFilter::PixelColorFit(const uchar &r, const uchar &g, const uchar &b)
{
	int max_rgb, min_rgb;
	max_rgb = r>g ? r : g;
	max_rgb = max_rgb>b ? max_rgb : b;

	min_rgb = r<g ? r : g;
	min_rgb = min_rgb<b ? min_rgb : b;
	int mean = (r + g + b) / 3;

	uchar y = 0.257*(float)r + 0.504*(float)g + 0.098*(float)b + 16;
	uchar cb = -0.148*(float)r - 0.291*(float)g + 0.439*(float)b + 128;
	uchar cr = 0.439*(float)r - 0.368*(float)g - 0.071*(float)b + 128;

	int sum = (int)((cb - 128)*(cb - 128) + (cr - 128)*(cr - 128));

	if (abs(max_rgb - min_rgb) < th_rgb_.th1_upper&&mean <= th_rgb_.th2_upper&&mean >= th_rgb_.th2_lower&&sum < th_ycbcr_.th1_upper*th_ycbcr_.th1_upper&& y >= th_ycbcr_.th2_lower&&y <= th_ycbcr_.th2_upper)
	{
		return true;
	}
	else return false;
}

void SmokeFilter::ColorPixelFilter(const Mat &color_image, const Mat &in_mask, Mat&out_mask)
{
	if (color_image.empty() || color_image.channels() != 3)
	{
		cerr << __FUNCTION__<<": input image error" << endl;
		return;
	}
	for (size_t i = 0; i < in_mask.rows; i++)
	for (size_t j = 0; j < in_mask.cols; j++)
	{
		uchar r = *(color_image.data + i*color_image.cols * 3 + j * 3);
		uchar g = *(color_image.data + i*color_image.cols * 3 + j * 3 + 1);
		uchar b = *(color_image.data + i*color_image.cols * 3 + j * 3 + 2);

		if (*(in_mask.data + i*color_image.cols  + j )>0&&PixelColorFit(r,g,b))
		{
			*(out_mask.data + i*color_image.cols + j) = 255;
		}
		else  *(out_mask.data + i*color_image.cols + j) = 0;
	}

}


/*-------------------------------------Move Contour--------------------------------*/

void SmokeFilter::InitContourThreahold(const char *file_name, const char *section_name)
{
	contour_area_threahold_ = ZIni::readInt(section_name, "contour_area_threahold", 100, file_name);
	contour_perimeter_threahold_ = ZIni::readInt(section_name, "contour_perimeter_threahold", 100, file_name);
	area_perimeter_ratio_threahold_ = ZIni::readDouble(section_name, "area_perimeter_ratio_threahold_", 0.0001, file_name);
}

