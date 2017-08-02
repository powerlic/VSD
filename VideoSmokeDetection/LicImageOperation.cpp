#include "stdafx.h"
#include "LicImageOperation.h"

namespace lio
{

	CvRect lioRectToCvRect(const lio_rect &value)
	{
		CvRect rect;
		rect.x = value.x;
		rect.y = value.y;
		rect.width = value.width;
		rect.height = value.height;
		return rect;
	}

	void SetRectZero(CvRect &rect)
	{
		rect.x = 0;
		rect.y = 0;
		rect.width = 0;
		rect.height = 0;
	}
	bool SizeEqual(const CvSize &size1, const CvSize &size2)
	{
		if (size1.height==size2.height&&size1.width==size2.height)
		{
			return true;
		}
		else return false;
	}
	void ScaleRect(const CvRect &s_rect, CvRect &d_rect, const CvSize &frame_size, const CvSize &d_size)
	{
		CvPoint center = CenterPoint(s_rect);

		d_rect.x = center.x - d_size.width / 2;
		d_rect.y = center.y - d_size.height / 2;
		d_rect.width = d_size.width;
		d_rect.height = d_size.height;
		
		if (d_rect.x<0)
		{
			d_rect.x = 0;
		}
		else if ((d_rect.x + d_size.width) > frame_size.width)
		{
			d_rect.x = frame_size.width - d_size.width - 1;
		}

		if (d_rect.y<0)
		{
			d_rect.y = 0;
		}
		else if ((d_rect.y + d_size.height) > frame_size.height)
		{
			d_rect.y = frame_size.height - d_size.height - 1;
		}

	}
	bool BeRectReasonable(const CvRect &rect, int width, int height)
	{
		if (rect.x >= 0
			&& rect.x < width
			&&  rect.y >= 0
			&& rect.y < height
			&& (rect.x + rect.width) >= 0
			&& (rect.x + rect.width) < width
			&& (rect.y + rect.height) >= 0
			&& (rect.y + rect.height) < height)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	bool BeTwoRectsMerged(const CvRect &rect1, const CvRect &rect2, CvRect &mergedRect, const int &width, const int &height)
	{
		int a, b, c, d, minX(width), minY(height), maxX(0), maxY(0);
		a = abs(rect1.x + rect1.width / 2 - rect2.x - rect2.width / 2);
		b = abs(rect1.y + rect1.height / 2 - rect2.y - rect2.height / 2);
		c = (rect1.width + rect2.width) / 2;
		d = (rect1.height + rect2.height) / 2;
		if (c > (a - 20) && d > (b - 20))
		{
			if (rect1.x < minX){
				minX = rect1.x;
			}
			if (rect2.x < minX){
				minX = rect2.x;
			}
			if (rect1.y < minY){
				minY = rect1.y;
			}
			if (rect2.y < minY){
				minY = rect2.y;
			}

			if ((rect1.x + rect1.width) > maxX){
				maxX = rect1.x + rect1.width;
			}
			if ((rect2.x + rect2.width) > maxX){
				maxX = rect2.x + rect2.width;
			}
			if ((rect1.y + rect1.height) > maxY){
				maxY = rect1.y + rect1.height;
			}
			if ((rect2.y + rect2.height) > maxY)
			{
				maxY = rect2.y + rect2.height;
			}
			mergedRect.x = minX;
			mergedRect.y = minY;
			mergedRect.width = maxX - minX;
			mergedRect.height = maxY - minY;
			return true;
		}
		else
			return false;
	}

	bool BeRectEqual(const CvRect &rect1, const CvRect &rect2)
	{
		if (  rect1.x == rect2.x
			&&rect1.y == rect2.y
			&&rect1.width == rect2.width
			&&rect2.height == rect1.height)
		{
			return true;
		}
		else return false;
	}
	bool BeExistInRectList(const vector<CvRect> &rectList, const CvRect &rect, int &loc)
	{
		for (int i = 0; i < rectList.size(); i++)
		{
			if (BeRectEqual(rectList[i], rect))
			{
				loc = i;
				return true;
			}
		}
		return false;
	}

	bool CheckFrame(const Mat &frame, bool is_color)
	{
		if (frame.empty())
		{
			return false;
		}
		if (is_color&&frame.channels()!=3)
		{
			return false;
		}
		return true;
	}
	const CvPoint &CenterPoint(const CvRect &rect)
	{
		CvPoint center;
		center.x = rect.x + rect.width / 2;
		center.y = rect.y + rect.height / 2;
		return center;
	}

	bool CorrectRect(CvRect &rect, const int &width, const int &height)
	{
		if (rect.x < 0)
		{
			rect.x = 0;
		}
		else if (rect.x >= width)
		{
			return false;
		}

		if (rect.y < 0)
		{
			rect.y = 0;
		}
		else if (rect.y >= height)
		{
			return false;
		}

		if (rect.x + rect.width >= width)
		{
			rect.width = width - rect.x;
		}
		else if (rect.x + rect.width < 0)
		{
			return false;
		}

		if (rect.y + rect.height >= height)
		{
			rect.height = height - rect.y;
		}
		else if (rect.y + rect.height < 0)
		{
			return false;
		}
		if (BeRectReasonable(rect, width, height))
		{
			return true;
		}
		else
		{
			return false;
		}

	}

	void MapRect(const CvRect &s_rect, CvRect &d_rect, const CvSize &s_frame_size, const CvSize &d_frame_size)
	{
		float w_ratio, h_ratio;
		w_ratio = d_frame_size.width*1.0 / s_frame_size.width;
		h_ratio = d_frame_size.height*1.0 / s_frame_size.height;

		CvPoint s_center, d_center;
		s_center.x = s_rect.x + s_rect.width / 2;
		s_center.y = s_rect.y + s_rect.height / 2;


		d_center.x = s_center.x*w_ratio;
		d_center.y = s_center.y*h_ratio;

		d_rect.width = (int)(s_rect.width*w_ratio);
		d_rect.height = (int)(s_rect.height*h_ratio);

		d_rect.x = d_center.x - d_rect.width / 2;
		d_rect.y = d_center.y - d_rect.height / 2;

	}
	void MapRects(const vector<CvRect>&s_rects, vector<CvRect> &d_rects, const CvSize &s_frame_size, const CvSize &d_frame_size)
	{
		d_rects.clear();
		for (size_t i = 0; i < s_rects.size(); i++)
		{
			CvRect t_rect;
			MapRect(s_rects[i], t_rect, s_frame_size, d_frame_size);
			d_rects.push_back(t_rect);
		}
	}

	void MapRect(const CvRect &s_rect, CvRect &d_rect, const int &s_width, const int &s_height, const int &d_width, const int &d_height, float scale)
	{
		float w_ratio, h_ratio;
		w_ratio = d_width*1.0 / s_width;
		h_ratio = d_height*1.0 / s_height;
	

		if (BeRectReasonable(s_rect, s_width, s_height))
		{
			
			CvPoint s_center,d_center;
			s_center.x = s_rect.x + s_rect.width / 2;
			s_center.y = s_rect.y + s_rect.height / 2;

			CvPoint left_up_p, right_down_p;

			d_center.x = s_center.x*w_ratio;
			d_center.y = s_center.y*h_ratio;


			d_rect.width = (int)(s_rect.width*w_ratio*scale);
			d_rect.height = (int)(s_rect.height*h_ratio*scale);

			d_rect.x = d_center.x - d_rect.width / 2;
			d_rect.y = d_center.y - d_rect.height / 2;

		}
		else
		{
			SetRectZero(d_rect);
		}
	}
	void GetDateTime(int &year, int &month,int &day, int &hour, int &minute, int &second, string &time_str)
	{
		SYSTEMTIME sys;
		GetLocalTime(&sys);
		year = sys.wYear;
		month = sys.wMonth;
		hour = sys.wDay;
		minute = sys.wMinute;
		second = sys.wSecond;

		char time_s[200];
		sprintf_s(time_s, 100, "%4d_%02d_%02d_%02d_%02d_%02d", year, month, day, hour, minute, second);
		time_str = string(time_s);

	}
	string GetDateTimeString()
	{
		SYSTEMTIME sys;
		GetLocalTime(&sys);
		int year = sys.wYear;
		int month = sys.wMonth;
		int day = sys.wDay;
		int hour = sys.wHour;
		int minute = sys.wMinute;
		int second = sys.wSecond;

		char time_s[200];
		sprintf_s(time_s, 100, "%4d_%02d_%02d_%02d_%02d_%02d", year, month, day, hour, minute, second);
		return string(time_s);
	}
	string GetDateString()
	{
		SYSTEMTIME sys;
		GetLocalTime(&sys);
		int year = sys.wYear;
		int month = sys.wMonth;
		int day = sys.wDay;
		char time_s[200];
		sprintf_s(time_s, 100, "%4d_%02d_%02d", year, month, day);
		return string(time_s);
	}
	void MergeRects(vector<CvRect> &rect_list, const int &width, const int &height)
	{
		int i, j;
		CvRect tmpRect1, tmpRect2, tmpMergedRect;
		vector<CvRect> tmpRectList1, tmpRectList2;

		tmpRectList1 = rect_list;
		int mergei(-1), mergej(-1);
		bool bStop(false);
		while (true){
			tmpRectList2.clear();
			mergei = -1;
			mergej = -1;
			bStop = false;
			for (i = 0; i<tmpRectList1.size() && !bStop; i++){
				bStop = false;
				for (j = i + 1; j<tmpRectList1.size(); j++){
					if (i == j){
						continue;
					}
					tmpRect1 = tmpRectList1[i];
					tmpRect2 = tmpRectList1[j];
					if (BeTwoRectsMerged(tmpRect1, tmpRect2, tmpMergedRect, width, height)){
						mergei = i;
						mergej = j;
						tmpRectList2.push_back(tmpMergedRect);
						bStop = true;
						break;
					}
				}
			}
			for (i = 0; i<tmpRectList1.size(); i++){
				if (i == mergei || i == mergej)	{
					continue;
				}
				tmpRectList2.push_back(tmpRectList1[i]);
			}
			if (tmpRectList2.size() == tmpRectList1.size()){
				break;
			}

			tmpRectList1 = tmpRectList2;
		}

		//cout<<"rect numbers "<<rectList.size()<<endl;
		rect_list.clear();
		rect_list = tmpRectList2;
	}
	void MergeRects(const vector<CvRect> &rect_list, vector<CvRect> &dst_list, vector<set<int>> &d_rect_number_set, const int &width, const int &height)
	{
		int i, j;
		CvRect tmpRect1, tmpRect2, tmpMergedRect;
		vector<CvRect> tmpRectList1, tmpRectList2;
		map<lio_rect, int> rect_number_map;
		map<lio_rect, set<int>> merged_rects_number_map;
		for (size_t i = 0; i < rect_list.size(); i++)
		{
			rect_number_map.insert(make_pair(lio_rect(rect_list[i]), i));
		}

		tmpRectList1 = rect_list;
		int mergei(-1), mergej(-1);
		bool bStop(false);

		while (true)
		{
			tmpRectList2.clear();
			mergei = -1;
			mergej = -1;
			bStop = false;

			for (i = 0; i<tmpRectList1.size() && !bStop; i++)
			{
				bStop = false;
				for (j = i + 1; j<tmpRectList1.size(); j++){
					if (i == j){
						continue;
					}
					tmpRect1 = tmpRectList1[i];
					tmpRect2 = tmpRectList1[j];
					if (BeTwoRectsMerged(tmpRect1, tmpRect2, tmpMergedRect, width, height))
					{
						mergei = i;
						mergej = j;
						tmpRectList2.push_back(tmpMergedRect);

						map<lio_rect, int>::iterator iter = rect_number_map.begin();
						iter = rect_number_map.find(lio_rect(tmpRect1));
						if (iter != rect_number_map.end())
						{
							merged_rects_number_map[lio_rect(tmpMergedRect)].insert((*iter).second);
							rect_number_map.erase(iter);
						}

						if (tmpRect1.x != tmpMergedRect.x || tmpRect1.y != tmpMergedRect.y || tmpRect1.width != tmpMergedRect.width || tmpRect1.height != tmpMergedRect.height)
						{
							map<lio_rect, set<int>>::iterator set_iter = merged_rects_number_map.begin();
							while (set_iter != merged_rects_number_map.end())
							{
								//找到有这个标识的set
								if (tmpRect1.x == set_iter->first.x &&
									tmpRect1.y == set_iter->first.y &&
									tmpRect1.width == set_iter->first.width&&
									tmpRect1.height == set_iter->first.height)
								{
									set<int> ::iterator innser_set_iter = set_iter->second.begin();

									while (innser_set_iter != set_iter->second.end())
									{
										merged_rects_number_map[lio_rect(tmpMergedRect)].insert(*innser_set_iter);
										innser_set_iter++;
									}

									merged_rects_number_map.erase(set_iter);
									break;

								}

								set_iter++;
							}
						}
						


						

				iter = rect_number_map.begin();
				iter = rect_number_map.find(lio_rect(tmpRect2));
				if (iter != rect_number_map.end())
				{
					merged_rects_number_map[lio_rect(tmpMergedRect)].insert((*iter).second);
					rect_number_map.erase(iter);
				}
				if (tmpRect1.x != tmpMergedRect.x || tmpRect1.y != tmpMergedRect.y || tmpRect1.width != tmpMergedRect.width || tmpRect1.height != tmpMergedRect.height)
				{
					map<lio_rect, set<int>>::iterator set_iter = merged_rects_number_map.begin();
					while (set_iter != merged_rects_number_map.end())
					{
						//找到有这个标识的set
						if (tmpRect2.x == set_iter->first.x &&
							tmpRect2.y == set_iter->first.y &&
							tmpRect2.width == set_iter->first.width&&
							tmpRect2.height == set_iter->first.height)
						{
							set<int > ::iterator innser_set_iter = set_iter->second.begin();

							while (innser_set_iter != set_iter->second.end())
							{
								merged_rects_number_map[lio_rect(tmpMergedRect)].insert(*innser_set_iter);
								innser_set_iter++;
							}

							merged_rects_number_map.erase(set_iter);
							break;

						}

						set_iter++;
					}
				}

			bStop = true;
			break;
				
					}
				}
			}
			for (i = 0; i<tmpRectList1.size(); i++)
			{
				if (i == mergei || i == mergej)	
				{
					continue;
				}


				tmpRectList2.push_back(tmpRectList1[i]);
			}
			if (tmpRectList2.size() == tmpRectList1.size())
			{
				break;
			}

			tmpRectList1 = tmpRectList2;
		}

		map<lio_rect, int> ::iterator iter = rect_number_map.begin();

		while (iter != rect_number_map.end())
		{
			merged_rects_number_map[iter->first].insert(iter->second);
			iter++;
		}
		map<lio_rect, set<int>>::iterator iter_m = merged_rects_number_map.begin();
		while (iter_m != merged_rects_number_map.end())
		{
			//raw_dst_rect_map.insert(make_pair(lioRectToCvRect(iter_m->first), iter_m->second));
			dst_list.push_back(lioRectToCvRect(iter_m->first));
			d_rect_number_set.push_back(iter_m->second);
			iter_m++;
		}
	}

	
	void ResizeRect(const CvRect &s_rect, CvRect &d_rect, const int &set_length, const int & width, const int &height)
	{
		int srcWidth, srcHeight;
		srcWidth = s_rect.width;
		srcHeight = s_rect.height;

		CvRect interRect;
		CvPoint leftUp, rightBottom;
		CvPoint center;
		center.x = s_rect.x + s_rect.width / 2;
		center.y = s_rect.y + s_rect.height / 2;
		int resizedLength;
		if (srcWidth<set_length&&srcHeight<set_length)
			resizedLength = set_length;
		else
			resizedLength = srcWidth>srcHeight ? srcWidth : srcHeight;

		interRect.x = center.x - resizedLength / 2;
		interRect.y = center.y - resizedLength / 2;
		interRect.height = resizedLength;
		interRect.width = resizedLength;

		if (interRect.x<0){
			leftUp.x = 0;
		}
		else leftUp.x = interRect.x;
		if (interRect.y<0){
			leftUp.y = 0;
		}
		else
			leftUp.y = interRect.y;

		if (interRect.x + resizedLength >= width){
			rightBottom.x = width - 1;
		}
		else rightBottom.x = interRect.x + resizedLength;

		if (interRect.y + resizedLength >= height){
			rightBottom.y = height - 1;
		}
		else rightBottom.y = interRect.y + resizedLength;

		d_rect.x = leftUp.x;
		d_rect.y = leftUp.y;
		d_rect.width = rightBottom.x - leftUp.x;
		d_rect.height = rightBottom.y - leftUp.y;

		CorrectRect(d_rect, width, height);
		
	}

	void RectToSquare(const CvRect &s_rect, CvRect &d_square, const CvSize &frame_size)
	{
		CvPoint center=CenterPoint(s_rect);
		if (s_rect.width==s_rect.height)
		{
			d_square = s_rect;
		}
		else if (s_rect.width>s_rect.height)
		{
			d_square.x = center.x - s_rect.width / 2;
			d_square.y = center.y - s_rect.width / 2;
			d_square.width = s_rect.width;
			d_square.height = s_rect.width;

			if (d_square.y<0)
			{
				d_square.y = 0;
			}
			if ((d_square.y + d_square.height)>frame_size.height)
			{
				d_square.y = frame_size.height - d_square.height - 1;
			}
			
		}
		else if (s_rect.width<s_rect.height)
		{
			d_square.x = center.x - s_rect.height / 2;
			d_square.y = center.y - s_rect.height / 2;
			d_square.width = s_rect.height;
			d_square.height = s_rect.height;
			if (d_square.x<0)
			{
				d_square.x = 0;
			}
			if ((d_square.x + d_square.width)>frame_size.width)
			{
				d_square.x = frame_size.width - d_square.width - 1;
			}
		}
		CorrectRect(d_square, frame_size.width,frame_size.height);

	}
	void MapPoint(const CvPoint &s_point, CvPoint &d_point, const int &s_width, const int &s_height, const int &d_width, const int &d_height)
	{
		float w_ratio, h_ratio;
		w_ratio = d_width*1.0 / s_width;
		h_ratio = d_height*1.0 / s_height;

		d_point.x = s_point.x*w_ratio;
		d_point.y = s_point.y*h_ratio;
	}
	void MapPoints( vector<CvPoint> &s_points, vector<CvPoint> &d_points, const int &s_width, const int &s_height, const int &d_width, const int &d_height)
	{
		CvPoint d_point;
		d_points.clear();
		for (vector<CvPoint>::iterator iter = s_points.begin(); iter != s_points.end(); iter++)
		{
			MapPoint(*iter, d_point, s_width, s_height, d_width, d_height);
			d_points.push_back(d_point);
		}
	}
	void FindContoursRects(const Mat &gray_image, int area_threadhold, vector<CvRect> &rectList, vector<vector<Point>> &contours)
	{
		vector<Vec4i>hierarchy;
		contours.clear();
		//Mat canny_output;
		int thresh = 100;
		//Canny(gray_image, canny_output, thresh, thresh * 2, 3);
		//imshow("gray", gray_image);
		Mat gray_iamge_copy;
		gray_image.copyTo(gray_iamge_copy);
		vector<vector<Point>> t_contours;
		findContours(gray_iamge_copy, t_contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		vector<Rect> boundRect(t_contours.size());
		vector<vector<Point> > contours_poly(t_contours.size());
		contours.clear();
		//test
		//test
		rectList.clear();
		if (!t_contours.empty() && !hierarchy.empty())
		{
			int idx = 0;
			for (; idx >= 0; idx = hierarchy[idx][0])
			{
				//drawContours(canny_output, contours, idx, Scalar(255), CV_FILLED, 1, vector<Vec4i>(), 2, Point());
				float area = contourArea(t_contours[idx]);
				if (area>area_threadhold)
				//if (area > area_threadhold)
				{
					approxPolyDP(Mat(t_contours[idx]), contours_poly[idx], 3, true);
					boundRect[idx] = boundingRect(Mat(contours_poly[idx]));
					//rectangle(canny_output, boundRect[idx], Scalar(255),1);
					rectList.push_back(boundRect[idx]);
					contours.push_back(t_contours[idx]);
					/*
					Point2f vertices[4];
					RotatedRect rect = minAreaRect(contours[idx]);
					rect.points(vertices);
					int top = gray_image.rows;
					int down = 0;
					int left = gray_image.cols;
					int right = 0;
					
					for (int i = 0; i < 4; i++)
					{
						left = vertices[i].x < left ? vertices[i].x : left;
						right = vertices[i].x > right ? vertices[i].x : right;
						top = vertices[i].y < top ? vertices[i].y : top;
						down = vertices[i].y > down ? vertices[i].y : down;
					}
					
					CvRect inRect = cvRect(left, top, right - left, down - top);
					if (lio::CorrectRect(inRect, gray_image.cols, gray_image.rows))
					{
						rectList.push_back(inRect);
					}
					*/
				}
			}

		}
		//imshow("canny", canny_output);
	}
	void FindContoursRects(const Mat &gray_image, int area_threadhold, int perimeter_threahold, float ratio_threahold, vector<CvRect> &rectList, vector<vector<Point>> &contours)
	{
		vector<Vec4i>hierarchy;
		contours.clear();
		//Mat canny_output;
		int thresh = 100;
		//Canny(gray_image, canny_output, thresh, thresh * 2, 3);
		//imshow("gray", gray_image);
		Mat gray_iamge_copy;
		gray_image.copyTo(gray_iamge_copy);
		vector<vector<Point>> t_contours;
		findContours(gray_iamge_copy, t_contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		vector<Rect> boundRect(t_contours.size());
		vector<vector<Point> > contours_poly(t_contours.size());
		contours.clear();
		rectList.clear();
		if (!t_contours.empty() && !hierarchy.empty())
		{
			int idx = 0;
			for (; idx >= 0; idx = hierarchy[idx][0])
			{
				//drawContours(canny_output, contours, idx, Scalar(255), CV_FILLED, 1, vector<Vec4i>(), 2, Point());
				float area = contourArea(t_contours[idx]);
				float perimeter = t_contours[idx].size();
				float area_perimeter_ratio = area / perimeter;
				if (area>area_threadhold&&perimeter>perimeter_threahold&&area_perimeter_ratio>ratio_threahold)
				{
					approxPolyDP(Mat(t_contours[idx]), contours_poly[idx], 3, true);
					boundRect[idx] = boundingRect(Mat(contours_poly[idx]));
					//rectangle(canny_output, boundRect[idx], Scalar(255),1);
					rectList.push_back(boundRect[idx]);
					contours.push_back(t_contours[idx]);
				}
			}

		}
	}
	void FindContoursRects(const Mat &gray_image, int area_threadhold, vector<CvRect> &rectList)
	{
		vector<Vec4i>hierarchy;
		vector<vector<Point>> contours;
		Mat canny_output;
		int thresh = 100;
		Canny(gray_image, canny_output, thresh, thresh * 2, 3);
		findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		rectList.clear();
		if (!contours.empty() && !hierarchy.empty())
		{
			int idx = 0;
			for (; idx >= 0; idx = hierarchy[idx][0])
			{
				float area = contourArea(contours[idx]);
				if (area > area_threadhold)
				{
					RotatedRect rect = minAreaRect(contours[idx]);
					Point2f vertices[4];
					rect.points(vertices);
					int top = gray_image.rows;
					int down = 0;
					int left = gray_image.cols;
					int right = 0;
					for (int i = 0; i < 4; i++)
					{
						left = vertices[i].x < left ? vertices[i].x : left;
						right = vertices[i].x > right ? vertices[i].x : right;
						top = vertices[i].y < top ? vertices[i].y : top;
						down = vertices[i].y > down ? vertices[i].y : down;
					}
					CvRect inRect = cvRect(left, top, right - left, down - top);
					if (lio::CorrectRect(inRect, gray_image.cols, gray_image.rows))
					{
						rectList.push_back(inRect);
					}
				}
			}

		}
	}
	void WriteText(Mat &frame, const CvPoint point,const String text, uchar b=255, uchar g=0, uchar r=0)
	{
		putText(frame, text, Point(point.x, point.y), CV_FONT_HERSHEY_TRIPLEX, 1, Scalar(r, g, b));
	}
	void WtrieLabel(Mat &frame, const CvRect &d_rect, const String text, uchar b, uchar g, uchar r)
	{
		if (frame.empty())
		{
			return;
		}
		Point p;
		p.x = d_rect.x;
		p.y = d_rect.y-5;
		Rect label_rect;
		label_rect.x = d_rect.x;
		label_rect.y = d_rect.y-20;
		label_rect.width = text.length()*16;
		label_rect.height = 20;
		if (label_rect.y<0)
		{
			label_rect.y = d_rect.y+d_rect.height;
			p.y = d_rect.y + d_rect.height+20;
		}
		Rect frame_rect = Rect(0, 0, frame.cols, frame.rows);
		label_rect &= frame_rect;


		Mat raw_roi_frame;
		frame(label_rect).copyTo(raw_roi_frame);
		cv::rectangle(frame, label_rect, Scalar(r, g, b), CV_FILLED);
		Mat label_frame = frame(label_rect);
		addWeighted(label_frame, 0.2, raw_roi_frame, 0.8, 0.0, label_frame);//dst=src1*alpha+src2*beta+gamma;
		putText(frame, text, Point(p.x, p.y), CV_FONT_HERSHEY_TRIPLEX, 0.7, Scalar(255, 255, 255));
	}

	void BoostSleep(int ms)
	{
		boost::this_thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(ms));
	}




	inline void NonMaximumSuppress(const std::vector<CvRect>& srcRects, std::vector<CvRect>& resRects, float thresh, int neighbors)
	{
		resRects.clear();

		const size_t size = srcRects.size();
		if (!size)
		{
			return;
		}

		// Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
		std::multimap<int, size_t> idxs;
		for (size_t i = 0; i < size; ++i)
		{
			idxs.insert(std::pair<int, size_t>(srcRects[i].y + srcRects[i].height, i));
		}

		// keep looping while some indexes still remain in the indexes list
		while (idxs.size() > 0)
		{
			// grab the last rectangle
			auto lastElem = --std::end(idxs);
			const cv::Rect& rect1 = srcRects[lastElem->second];

			int neigborsCount = 0;

			idxs.erase(lastElem);

			for (auto pos = std::begin(idxs); pos != std::end(idxs);)
			{
				// grab the current rectangle
				const cv::Rect& rect2 = srcRects[pos->second];

				float intArea = (rect1 & rect2).area();
				float unionArea = rect1.area() + rect2.area() - intArea;
				float overlap = intArea / unionArea;

				// if there is sufficient overlap, suppress the current bounding box
				if (overlap > thresh)
				{
					pos = idxs.erase(pos);
					++neigborsCount;
				}
				else
				{
					++pos;
				}
			}
			if (neigborsCount >= neighbors)
			{
				resRects.push_back(rect1);
			}
		}
	}
	void NonMaximumSuppress2(const std::vector<CvRect>& srcRects, const std::vector<float>& scores, std::vector<CvRect>& resRects, float thresh, int neighbors)
	{
		resRects.clear();

		const size_t size = srcRects.size();
		if (!size)
		{
			return;
		}

		assert(srcRects.size() == scores.size());

		// Sort the bounding boxes by the detection score
		std::multimap<float, size_t> idxs;
		for (size_t i = 0; i < size; ++i)
		{
			idxs.insert(std::pair<float, size_t>(scores[i], i));
		}

		// keep looping while some indexes still remain in the indexes list
		while (idxs.size() > 0)
		{
			// grab the last rectangle
			auto lastElem = --std::end(idxs);
			const cv::Rect& rect1 = srcRects[lastElem->second];

			int neigborsCount = 0;

			idxs.erase(lastElem);

			for (auto pos = std::begin(idxs); pos != std::end(idxs);)
			{
				// grab the current rectangle
				const cv::Rect& rect2 = srcRects[pos->second];

				float intArea = (rect1 & rect2).area();
				float unionArea = rect1.area() + rect2.area() - intArea;
				float overlap = intArea / unionArea;

				// if there is sufficient overlap, suppress the current bounding box
				if (overlap > thresh)
				{
					pos = idxs.erase(pos);
					++neigborsCount;
				}
				else
				{
					++pos;
				}
			}
			if (neigborsCount >= neighbors)
			{
				resRects.push_back(rect1);
			}
		}
	}
	void SaveRegResult(const char* service_id, const Mat &reg_mat, const string &label)
	{
		string name_str = lio::GetDateTimeString();
		string file_path;
		file_path = "./regres";
		fstream reg_file;
		reg_file.open(file_path, ios::in);
		if (!reg_file)
		{
			_mkdir(file_path.c_str());
		}
		file_path = "./regres/" + label;
		fstream _file;
		_file.open(file_path, ios::in);
		if (!_file)
		{
			_mkdir(file_path.c_str());
		}
		name_str = "./regres/" + label + "/" + String(service_id) + "_" + name_str + ".jpg";
		imwrite(name_str, reg_mat);
	}
	void SaveRegFrame(const char* service_id, const Mat &frame)
	{
		string name_str = lio::GetDateTimeString();
		string file_path="./regframe";
		fstream reg_frame_file;
		reg_frame_file.open(file_path, ios::in);
		if (!reg_frame_file)
		{
			_mkdir(file_path.c_str());
		}

		file_path = "./regframe/" + lio::GetDateString();
		fstream _date_file;
		_date_file.open(file_path, ios::in);
		if (!_date_file)
		{
			_mkdir(file_path.c_str());
		}
		name_str = file_path + "/" + String(service_id) + "_" + name_str + ".jpg";
		imwrite(name_str, frame);
	}
}