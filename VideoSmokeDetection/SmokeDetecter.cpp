#include "stdafx.h"
#include "SmokeDetecter.h"
#include <math.h>

SmokeDetecter::SmokeDetecter(CvSize set_reg_size,
	                         CvSize set_bg_size,
	                         int set_frame_list_size,
	                         int set_contour_area_threadhold,
	                         int set_contour_perimeter_threadhold,
	                         int set_sensitivity_degree,
							 string set_service_id,
	                         string set_model_version
	                        )
{
	ptr_vibe_extrator = shared_ptr<ViBeExtrator>(new ViBeExtrator);
	reg_size = set_reg_size;
	bg_size = set_bg_size;
	frame_list_size = set_frame_list_size;
	contour_area_threadhold = set_contour_area_threadhold;
	contour_perimeter_threadhold = set_contour_area_threadhold;
	sensitivity_degree = set_sensitivity_degree;
	model_version = set_model_version;
	service_id = set_service_id;
	mat_list.clear();
	diff_mat_list.clear();
	frame_list_size = 10;

	acc_count = 0;
	acc_mask_delay = 10;


	aplha=50;
	ycbcr_th3 = 130;
	ycbcr_th4 = 250; 
	rgb_th1 = 50;
	rgb_th2 = 80;
	rgb_th3 = 250;

	//ptr_vas_classifier = ptr_vas_classifier_pool->GetVasClassifierInstance(model_version);
	//if (!ptr_vas_classifier)
	//{
	//	exit(-1);
	//}

	InitSavePath("setup.ini");
	//mog = BackgroundSubtractorMOG2(5,4,true);
}


SmokeDetecter::~SmokeDetecter()
{
}



void SmokeDetecter::SetFrame(const Mat &set_frame)
{
	//set_frame.copyTo(frame);
	//cvtColor(frame, gray_frame, CV_RGB2GRAY);
	
}
void SmokeDetecter::ProcessFrame(const Mat &frame_mat, vector<CvRect> &rect_list)
{
	Mat gray_mat;
	if (frame_mat.channels()==3)
	{
		cvtColor(frame_mat, gray_mat, CV_RGB2GRAY);
	}
	else gray_mat = Mat(frame_mat);

	frame_mat.copyTo(frame);
	cvtColor(frame, gray_frame, CV_RGB2GRAY);
	AccumulateDiffImage();
	
	Mat corners_mat;
	//ptr_vibe_extrator->moveAreaExtrator(gray_frame, bg_size.width, bg_size.height);

	mog(frame, foreground, 0.001);
	morphologyEx(foreground, mask, MORPH_OPEN, Mat());
	dilate(mask, mask, NULL, cvPoint(-1, -1), 2);
	mog.getBackgroundImage(background);


	if (1)
	{
		//mask = ptr_vibe_extrator->getMask();
		Mat filter_mat;
		if (!acc_threashold.empty() && !mask.empty())
		{
			//subtract(mask, acc_threashold, filter_mat);
			//FindCorners(frame, gray_frame, corners_mat);
			//imshow("filter", filter_mat);
			//
			//imshow("corners", corners_mat);
#ifdef _DEBUG
			
			if (!acc_threashold.empty())
			{
				imshow("acc_threashold", acc_threashold);
			}
			imshow(service_id + "prior_mask", mask);
		
#endif
			

			/*
			vector<CvRect> rect_list_prior;
			vector<vector<Point>> contours_prior;
			lio::FindContoursRects(mask, 100, rect_list_prior, contours_prior);

			for (size_t i = 0; i < contours_prior.size(); i++)
			{
				drawContours(frame, contours_prior, i, CV_RGB(255, 0, 0), 1);
			}
			*/
			time_t t1, t2, t3;
			t1 = clock();
			//CalculateAccMask();
			t2 = clock();

			Threahold ycbcr_th, rgb_th;
			ycbcr_th.th1 = aplha;
			ycbcr_th.th2 = ycbcr_th3;
			ycbcr_th.th3 = ycbcr_th4; 
			rgb_th.th1 = rgb_th1;
			rgb_th.th2 = rgb_th2;
			rgb_th.th3 = rgb_th3;


			//ColorFilter_Caller(frame, mask, &ycbcr_th, &rgb_th, frame.cols, frame.rows, color_mask);
			//Filter_Caller(frame, mask, &ycbcr_th, &rgb_th, frame.cols, frame.rows, color_mask);
			//GPU
			//Filter_Caller(frame, acc_threashold, mask, &ycbcr_th, &rgb_th, frame.cols, frame.rows, color_mask, acc_mask,1,1);
			//ColorCriteriaRGBandYCbCr();
			t3 = clock();
			//imshow("acc", acc_mask);
			//cout << "Acc Mask " << t2 - t1 << endl;
			//cout << "Color Mask " << t3 - t2 << endl;
			morphologyEx(mask, mask, MORPH_OPEN, Mat());
			dilate(mask, mask, NULL, cvPoint(-1, -1), 2);
			vector<CvRect> t_rect_list;
			vector<vector<Point>> contours;
			
			lio::FindContoursRects(mask, contour_area_threadhold, t_rect_list, contours);
			//UpdateSmokeAreaRecord(rect_list, contours);
			/*
			for (size_t i = 0; i < rect_list.size(); i++)
			{
				char text[10];
				float fore_back_diff = CalculateForeAndBackHisDis(gray_mat, rect_list[i], contours[i]);
				sprintf_s(text, 10, "%.3f", fore_back_diff);
				putText(frame, string(text), Point(rect_list[i].x, rect_list[i].y), CV_FONT_ITALIC, 0.8, CV_RGB(255, 0, 0));
			}
			*/
			lio::MergeRects(t_rect_list, mask.cols, mask.rows);
			vector<CvRect> merged_smoke_rects;
			merged_smoke_rects.clear();

			
			UpdateMaskList();
			for (size_t i = 0; i < t_rect_list.size(); i++)
			{
				CvRect t_rect;
				lio::ResizeRect(t_rect_list[i], t_rect, 224, mask.cols, mask.rows);
				//rectangle(frame, rect_list[i], CV_RGB(255, 255, 0), 1);
				//t_rect = t_rect_list[i];
				bool beContinousVar = BeMaskListRectConstantVariation(t_rect_list[i], 0.01);
				//bool beContinousVar = true;
				if (1)
				{
					vector<Prediction> predictions;
					Mat reg_mat = frame(t_rect);
					Mat resized_reg_mat;
					resize(reg_mat, resized_reg_mat, Size(224, 224));
					predictions = caffe_reg_fun(resized_reg_mat, 2);
					char reg_res[100];
				
					if (predictions[0].first.compare("smoke") == 0 && predictions[0].second>0.9)
					{
						rectangle(frame, t_rect, CV_RGB(255, 0, 0), 1);
						merged_smoke_rects.push_back(t_rect);
						if (is_save_image)
						{
							SaveRegImage(resized_reg_mat);
						}
						sprintf_s(reg_res, "%s %.2f", predictions[0].first.data(), predictions[0].second);
						putText(frame, string(reg_res), Point(t_rect.x + t_rect.width / 2, t_rect.y + t_rect.height / 2), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 0, 0));
					}
					else
					{
						rectangle(frame, t_rect, CV_RGB(0, 0, 255), 1);
						sprintf_s(reg_res, "%s %.2f", predictions[0].first.data(), predictions[0].second);
						putText(frame, string(reg_res), Point(t_rect.x + t_rect.width / 2, t_rect.y + t_rect.height / 2), CV_FONT_HERSHEY_PLAIN, 1, CV_RGB(255, 0, 0));
					}
					
				}
			}
			lio::MergeRects(merged_smoke_rects, mask.cols, mask.rows);
			rect_list = std::move(merged_smoke_rects);

#ifdef _DEBUG
			
			//imshow(service_id + "color_mask", color_mask);
			imshow(service_id + "acc_mask", acc_mask);
			
#endif		
			
			imshow(service_id + "_mask", mask);
			
			//imshow(service_id+"_mask", mask);
			imshow("frame", frame);
			
		}
		
	}

}
void SmokeDetecter::UpdateSmokeAreaRecord(vector<CvRect> rect_list, vector<vector<Point>> contours)
{
	vector<Contour> frame_contours_record;
	frame_contours_record.clear();
	if (contours_record.size()<frame_list_size)
	{
		for (size_t i = 0; i < rect_list.size(); i++)
		{
			Contour t_contour;
			t_contour.center.x = rect_list[i].x + rect_list[i].width / 2;
			t_contour.center.y = rect_list[i].y + rect_list[i].height / 2;
			t_contour.area = contourArea(contours[i]);
			t_contour.contour = contours[i];
			frame_contours_record.push_back(t_contour);
		}
		contours_record.push_back(frame_contours_record);
	}
	else
	{
		for (size_t i = 0; i < rect_list.size(); i++)
		{
			Contour t_contour;
			t_contour.center.x = rect_list[i].x + rect_list[i].width / 2;
			t_contour.center.y = rect_list[i].y + rect_list[i].height / 2;
			t_contour.area = contourArea(contours[i]);
			t_contour.contour = contours[i];
			frame_contours_record.push_back(t_contour);
		}
		contours_record.push_back(frame_contours_record);
		contours_record.pop_front();
	}
}
void SmokeDetecter::AccumulateDiffImage()
{
	Mat current_gray;
	gray_frame.copyTo(current_gray);
	if (acc_mat.empty())
	{
		acc_mat = Mat::zeros(cv::Size(gray_frame.cols, gray_frame.rows), CV_8UC1);
	}
	if (mat_list.size() == 0)
	{
		mat_list.push_back(current_gray);
		return;
	}
	Mat currentDiff;

	absdiff(current_gray, mat_list.back(), currentDiff);
	
	if (mat_list.size()<frame_list_size)
	{
		mat_list.push_back(current_gray);
		diff_mat_list.push_back(currentDiff);
		add(currentDiff, acc_mat, acc_mat);
	}
	else
	{
		subtract(acc_mat, diff_mat_list.front(), acc_mat);
		add(acc_mat, currentDiff, acc_mat);

		//updatelist
		diff_mat_list.pop_front();
		diff_mat_list.push_back(currentDiff);

		mat_list.pop_front();
		mat_list.push_back(current_gray);
		threshold(acc_mat, acc_threashold, 100, 255, THRESH_BINARY);

		/*vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(acc_threashold, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		
		int idx = 0;
		for (; idx>=0; idx=hierarchy[idx][0])
		{
			const vector<Point> &c = contours[idx];
			double area = abs(contourArea(c));
			if (area>100)
			{
				drawContours(acc_threashold, contours, idx, Scalar(255), CV_FILLED);
			}
			
		}*/
		//morphologyEx(acc_threashold, acc_threashold, MORPH_OPEN, Mat());
		//dilate(acc_threashold, acc_threashold, NULL, cvPoint(-1, -1), 2);
	}

}
void SmokeDetecter::CalculateAccMask()
{
	
	if (acc_mask.empty() || acc_count==acc_mask_delay)
	{
		acc_mask = Mat::zeros(cv::Size(gray_frame.cols, gray_frame.rows), CV_8UC1);
		acc_count = 0;
	}
	uchar *ptr_acc_data = acc_mask.data;
	uchar *ptr_mask = mask.data;
	int offset;
	for (size_t i = 0; i < gray_frame.cols; i++)
	for (size_t j = 0; j < gray_frame.rows; j++)
	{
		offset = j*gray_frame.cols + i;
		if (*(acc_threashold.data+offset)>0)
		{
			*(ptr_acc_data + offset) = 255;
			*(ptr_mask + offset) = 0;
		}
	}
	acc_count++;
	
}
void SmokeDetecter::UpdateMaskList()
{
	if (mask_list.size()<frame_list_size)
	{
		Mat cur_mat;
		mask.copyTo(cur_mat);
		mask_list.push_back(cur_mat);
	}
	else
	{
		Mat cur_mat;
		mask.copyTo(cur_mat);
		mask_list.pop_front();
		mask_list.push_back(cur_mat);
	}
}
bool SmokeDetecter::BeMaskListRectConstantVariation(const CvRect &rect,const float &ratio)
{	
	if (mask_list.size() == frame_list_size)
	{
		list<Mat>::iterator itor = mask_list.begin();
		Mat pre_mat;
		Mat cur_mat;
		vector<Mat> vec_mat;

		(*itor)(rect).copyTo(pre_mat);
		vec_mat.push_back(pre_mat);
		itor++;
		
		Mat merged_mat = Mat(rect.height*frame_list_size, rect.width, CV_8UC1);
		Mat merged_mat_color = Mat(rect.height*frame_list_size, rect.width, CV_8UC3);
		int rect_size = rect.width*rect.height;
		int i = 1;
		while (itor != mask_list.end())
		{
			(*itor)(rect).copyTo(cur_mat);
			Mat diff_mat;
			absdiff(pre_mat, cur_mat, diff_mat);
			int count = countNonZero(diff_mat);
			float current_ratio = float(count) / rect_size;
			if (current_ratio<ratio)
			{
				return false;
			}
			i++;
			cv::swap(pre_mat, cur_mat);
			itor++;
		}

		return true;
		
	}
	else return false;
	
}
bool SmokeDetecter::BeMaskListForeBackgroundUnique(const CvRect &rect, const float &ratio)
{

	return false;

}



void SmokeDetecter::FilterByAccMask()
{
	uchar *mask_data_ptr = mask.data;
	int offset;
	for (size_t i = 0; i < gray_frame.cols; i++)
	for (size_t j = 0; j < gray_frame.rows; j++)
	{
		offset = j*gray_frame.cols + i;
		if (*(acc_mask.data + offset)>0)
		{
			*(mask_data_ptr + offset) = 0;
		}
	}
}
float SmokeDetecter::CalculateForeAndBackHisDis(const Mat &gray_mat, const CvRect &rect, const vector<Point> &contour)
{
	int fore_count[256] = { 0 };
	int back_count[256] = { 0 };

	uchar *data_ptr = gray_mat.data;
	int offset;
	for (size_t i = rect.x; i < rect.x+rect.width; i++)
	for (size_t j = rect.y; j < rect.y + rect.height; j++)
	{
		offset = j*gray_mat.cols + i;
		
		bool bInside=pointPolygonTest(contour, Point2f(i, j), false);
		if (bInside)
		{
			fore_count[*(data_ptr + offset)]++;
		}
		else
		{
			back_count[*(data_ptr + offset)]++;
		}
	}
	float sum=0;
	for (size_t i = 0; i < 256; i++)
	{
		sum += abs((float)(fore_count[i] - back_count[i])/(float)(rect.width*rect.height));
	}
	return 100*sum;
}
void SmokeDetecter::ColorCriteriaRGBandYCbCr()
{
	Mat ycbcr_mat, ycbcr_mat_list[3];
	//image_mat.copyTo(color_mask);
	color_mask = Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
	cvtColor(frame, ycbcr_mat, CV_RGB2YCrCb);
	split(ycbcr_mat, ycbcr_mat_list);

	Mat rgb_mat_list[3];
	split(frame, rgb_mat_list);
	int offset;
	uchar *mask_data_ptr = mask.data;
	for (int i = 0; i < frame.cols; ++i)
		for (int j = 0; j<frame.rows; ++j)
		{
			offset = j*frame.cols + i;
			int y = *(ycbcr_mat_list[0].data + offset);
			int cb = *(ycbcr_mat_list[1].data + offset);
			int cr = *(ycbcr_mat_list[2].data + offset);

			int r = *(rgb_mat_list[0].data + offset);
			int g = *(rgb_mat_list[1].data + offset);
			int b = *(rgb_mat_list[2].data + offset);

			int max_rgb, min_rgb;
			max_rgb = r>g ? r : g;
			max_rgb = max_rgb>b ? max_rgb : b;

			min_rgb = r<g ? r : g;
			min_rgb = min_rgb<b ? min_rgb : b;
			int mean = (r + g + b) / 3;

			int sum = (int)((cb - 128)*(cb - 128) + (cr - 128)*(cr - 128));

			if (abs(max_rgb - min_rgb)<rgb_th1&&mean <= rgb_th3&&mean >= rgb_th2&&sum<aplha*aplha&& y >= ycbcr_th3&&y <= ycbcr_th4)
			{
				*(color_mask.data + offset) = 255;
			}
			else
			{
				*(mask_data_ptr + offset) = 0;
			}

		}

}
void SmokeDetecter::FilterByColor()
{
	uchar *mask_data_ptr = mask.data;
	int offset;
	
	for (size_t i = 0; i < color_mask.cols; i++)
		for (size_t j = 0; j < color_mask.rows; j++)
	{
		offset = j*color_mask.cols + i;
		if (*(color_mask.data + offset)==0)
		{
			*(mask_data_ptr + offset) = 0;
		}
	}
}
void SmokeDetecter::FindCorners(const Mat &image, const Mat &gray_image, Mat &show_image)
{
	vector<Point2f> prepoint, nextpoint;
	vector<uchar> state;
	vector<float>err;
	image.copyTo(show_image);
	goodFeaturesToTrack(gray_image, prepoint, 1000, 0.001, 10, Mat(), 3, false, 0.04);
	Scalar lineColor = Scalar(0, 0, 255);
	for (int i = 0; i < prepoint.size(); ++i)
	{
		circle(show_image, Point((int)prepoint[i].x, (int)prepoint[i].y), 1, lineColor, 1);
	}

}
void SmokeDetecter::SaveRegImage(const Mat &reg_mat)
{
	SYSTEMTIME sys;
	GetLocalTime(&sys);
	char image_name[200];
	sprintf_s(image_name, 100, "%4d_%02d_%02d_%02d_%02d_%02d", sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond);
	string image_name_str;
	image_name_str = full_save_path_str+"/" +service_id+"_"+ image_name + ".jpg";
	imwrite(image_name_str, reg_mat);


}

void SmokeDetecter::SetCaffeRegFunction(std::function<vector<Prediction>(const Mat&, const int&)> set_fun)
{
	caffe_reg_fun = set_fun;
}

void SmokeDetecter::InitSavePath(const char *file_name)
{
	string save_path_str = ZIni::readString("SaveTrainImage", "SavePath", "null", file_name);
	is_save_image = ZIni::readInt("SaveTrainImage", "is_Save", 0, file_name);

	if (save_path_str.compare("null")==0)
	{
		cout << "[SmokeDetecter]" << " InitSavePath: trained Image save path is not set " << endl;
		exit(1);
	}



	full_save_path_str = save_path_str + "/smoke_yes";
	fstream _file;
	_file.open(full_save_path_str, ios::in);
	if (!_file)
	{
		//cout << full_save_path_str << " not exist" << endl;
		//_mkdir(full_save_path_str.data());
		boost::filesystem::create_directory(full_save_path_str);
		//cout << full_save_path_str << " has been created";
	}
	else
	{
		//cout << full_save_path_str << " has been created";
	}

}