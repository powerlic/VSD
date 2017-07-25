#include "stdafx.h"
#include "Filter.h"

Filter::Filter()
{
	
}


Filter::~Filter()
{
	filter_parameter_.Clear();
}
Filter::Filter(const vas::FilterParameter &param)
{
	filter_parameter_.CopyFrom(param);
	CheckParameter(param);
}
Filter::Filter(const char* file_name)
{
	if (vas::ReadProtoFromTextFile(file_name, &filter_parameter_));
	{
		CheckParameter(filter_parameter_);
	}
}

void Filter::CheckParameter(const vas::FilterParameter &param)
{
	CHECK_GE(param.contour_area_threshold(), 200) << "contour_area_threshold must be greater than or equal to 200!" << endl;
	CHECK_GE(param.contour_perimeter_threshold(), 0) << "contour_perimeter_threshold must be greater than or equal to 0!" << endl;
	CHECK_GE(param.area_perimeter_ratio_threshold(), 0) << "area_perimeter_ratio_threshold must be greater than or equal to 0!" << endl;

	CHECK_GE(param.filter_height(), 320) << "filter_height must >= 420!" << endl;
	CHECK_GE(param.filter_width(), 420) << "filter_width must >= than 320!" << endl;

	CHECK_GE(param.color_fit_ratio(), 0.1) << "color_fit_ratio must >= than 0.1" << endl;

	if (param.filter_method() != vas::PURE_CONTOUR_AREA)
	{
		CHECK(param.has_rgb_th_1() && param.has_rgb_th_2() && param.has_ycbcr_th_1() && param.has_ycbcr_th_2()) << " Color Filter must have complete params: rgb_th1,rgb_th2, ycb_th1,ycb_th2!" << endl;
		CHECK_GE(param.rgb_th_1().lower(), 0) << "rgb_th1 lower must be not negative!" << endl;
		CHECK_GE(param.rgb_th_1().upper(), 0) << "rgb_th1 upper must be not negative!" << endl;
		CHECK_GE(param.rgb_th_2().lower(), 0) << "rgb_th2 lower must be not negative!" << endl;
		CHECK_GE(param.rgb_th_2().upper(), 0) << "rgb_th2 upper must be not negative!" << endl;
		CHECK_GE(param.ycbcr_th_1().lower(), 0) << "ycbcr_th_1 lower must be not negative!" << endl;
		CHECK_GE(param.ycbcr_th_1().upper(), 0) << "ycbcr_th_1 upper must be not negative!" << endl;
		CHECK_GE(param.ycbcr_th_2().lower(), 0) << "ycbcr_th_2 lower must be not negative!" << endl;
		CHECK_GE(param.ycbcr_th_2().upper(), 0) << "ycbcr_th_2 upper must be not negative!" << endl;
	}

}



void Filter::Filtrate(const Mat &frame, const Mat &fore_mask, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	Update(frame, fore_mask);
	switch (filter_parameter_.filter_method())
	{			
		case vas::FRAME_COLOR_CPU:
			FrameColorFiltrateCPU(filter_frame_, filter_fore_mask_, frame.size(), out_mask, contours, rects);
			break;
		case vas::FRAME_COLOR_GPU:
			FrameColorFiltrateGPU(filter_frame_, filter_fore_mask_, frame.size(), out_mask, contours, rects);
			break;
		case vas::CONTOUR_COLOR:
			ContourColorFiltrate(filter_frame_, filter_fore_mask_, frame.size(),out_mask, contours, rects);
			break;
		case vas::PURE_CONTOUR_AREA:
			PureContourFiltrate(filter_frame_, filter_fore_mask_, frame.size(), out_mask, contours, rects);
			break;
		default:
			break;
	}
}


void Filter::PureContourFiltrate(const Mat &frame, const Mat &fore_mask, const CvSize &dst_size, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	out_mask = Mat::zeros(dst_size, CV_8UC1);
	Mat resize_mask;
	resize(fore_mask, resize_mask, dst_size);
	lio::FindContoursRects(resize_mask, filter_parameter_.contour_area_threshold(), filter_parameter_.contour_perimeter_threshold(), filter_parameter_.area_perimeter_ratio_threshold(), rects, contours);
	drawContours(out_mask, contours, -1, Scalar(255), CV_FILLED);
}
void Filter::ContourColorFiltrate(const Mat &frame, const Mat fore_mask, const CvSize &dst_size, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	CvSize filter_size = cvSize(filter_parameter_.filter_width(), filter_parameter_.filter_height());

	vector<vector<Point>> t_contours; 
	vector<CvRect> t_rects;
	lio::FindContoursRects(fore_mask, filter_parameter_.contour_area_threshold(), filter_parameter_.contour_perimeter_threshold(), filter_parameter_.area_perimeter_ratio_threshold(), t_rects, t_contours);

	Mat t_contours_filled_mask = Mat::zeros(filter_size, CV_8UC1);
	out_mask = Mat::zeros(filter_size, CV_8UC1);

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
		if (rgb_move_ratio>filter_parameter_.color_fit_ratio())
		{
			drawContours(out_mask, t_contours, i, Scalar(255), CV_FILLED);
		}
	}
	resize(out_mask, out_mask,dst_size);
	lio::FindContoursRects(out_mask, filter_parameter_.contour_area_threshold(), filter_parameter_.contour_perimeter_threshold(), filter_parameter_.area_perimeter_ratio_threshold(), rects, contours);
	drawContours(out_mask, contours, -1, Scalar(255), CV_FILLED);
}

void Filter::FrameColorFiltrateCPU(const Mat &frame, const Mat fore_mask, const CvSize &dst_size, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	if (out_mask.empty())
	{
		out_mask = Mat::zeros(filter_parameter_.filter_height(), filter_parameter_.filter_width(), CV_8UC1);
	}
	for (size_t i = 0; i < frame.rows; i++)
	for (size_t j = 0; j < frame.cols; j++)
	{
		uchar r = *(frame.data + i*frame.cols * 3 + j * 3);
		uchar g = *(frame.data + i*frame.cols * 3 + j * 3 + 1);
		uchar b = *(frame.data + i*frame.cols * 3 + j * 3 + 2);

		if (*(fore_mask.data + i*fore_mask.cols + j)>0 && PixelColorFit(r, g, b))
		{
			*(out_mask.data + i*out_mask.cols + j) = 255;
		}
		else  *(out_mask.data + i*out_mask.cols + j) = 0;
	}
	resize(out_mask, out_mask,dst_size);
	lio::FindContoursRects(out_mask, filter_parameter_.contour_area_threshold(), filter_parameter_.contour_perimeter_threshold(), filter_parameter_.area_perimeter_ratio_threshold(),rects, contours);
	drawContours(out_mask, contours, -1, Scalar(255), CV_FILLED);
}
void Filter::FrameColorFiltrateGPU(const Mat &frame, const Mat fore_mask, const CvSize &dst_size, Mat &out_mask, vector<vector<Point>> &contours, vector<CvRect> &rects)
{
	Mat color_mask;
	filter_mutex.lock();
	ColorFilter_Caller( frame, fore_mask,
					    filter_parameter_.rgb_th_1().upper(),
						filter_parameter_.rgb_th_2().lower(),
						filter_parameter_.rgb_th_2().upper(),
						filter_parameter_.ycbcr_th_1().upper(),
						filter_parameter_.ycbcr_th_2().lower(),
						filter_parameter_.ycbcr_th_2().upper(),
						frame.rows, frame.cols,
						out_mask, color_mask_
					   );
	filter_mutex.unlock();
	resize(out_mask, out_mask, dst_size);
	lio::FindContoursRects(out_mask, filter_parameter_.contour_area_threshold(), filter_parameter_.contour_perimeter_threshold(), filter_parameter_.area_perimeter_ratio_threshold(), rects, contours);
	drawContours(out_mask, contours, -1, Scalar(255), CV_FILLED);
}


bool Filter::PixelColorFit(const uchar &r, const uchar &g, const uchar &b)
{
	int max_rgb, min_rgb;
	max_rgb = r>g ? r : g;
	max_rgb = max_rgb>b ? max_rgb : b;

	min_rgb = r<g ? r : g;
	min_rgb = min_rgb<b ? min_rgb : b;
	int mean = (r + g + b) / 3;

	/*ToDo: use a table to implement this process!*/
	uchar y =   0.257*(float)r + 0.504*(float)g + 0.098*(float)b + 16;
	uchar cb = -0.148*(float)r - 0.291*(float)g + 0.439*(float)b + 128;
	uchar cr =  0.439*(float)r - 0.368*(float)g - 0.071*(float)b + 128;

	int sum = (int)((cb - 128)*(cb - 128) + (cr - 128)*(cr - 128));

	if (abs(max_rgb - min_rgb) < filter_parameter_.rgb_th_1().upper() 
		&& mean <= filter_parameter_.rgb_th_2().upper() 
		&& mean >= filter_parameter_.rgb_th_2().lower() 
		&& sum  <  filter_parameter_.ycbcr_th_1().upper()*filter_parameter_.ycbcr_th_1().upper() 
		&& y >= filter_parameter_.ycbcr_th_2().lower()
		&& y <= filter_parameter_.ycbcr_th_2().upper())
	{
		return true;
	}
	else return false;
}

void Filter::Update(const Mat &frame, const Mat &fore_mask)
{
	CHECK(!frame.empty()) << "input frame is empty!" << endl;
	CHECK(!fore_mask.empty()) << "input fore_mask is empty!" << endl;
	CHECK_EQ(frame.channels(), 3) << "input frame must be colorful " << endl;
	CHECK_EQ(fore_mask.channels(), 1) << "input frame must be binary " << endl;


	CvSize filter_size = cvSize(filter_parameter_.filter_width(), filter_parameter_.filter_height());

	if (!lio::SizeEqual(frame.size(), filter_size))
	{
		resize(frame, filter_frame_, filter_size);
	}
	else frame.copyTo(filter_frame_);

	if (!lio::SizeEqual(fore_mask.size(), filter_size))
	{
		resize(fore_mask, filter_fore_mask_, filter_size);
	}
	else fore_mask.copyTo(filter_fore_mask_);
}