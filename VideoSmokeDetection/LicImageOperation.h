#pragma once
#include"stdafx.h"
#include"cvpreload.h"
#include<map>
#include<vector>

using namespace std;
using namespace cv;
namespace lio
{  


		struct lio_rect
		{
			int x;
			int y;
			int width;
			int height;

			lio_rect(int set_x = 0, int set_y = 0, int set_width = 0, int set_heigth = 0)
			{
				x = set_x;
				y = set_y;
				width = set_width;
				height = set_heigth;
			}
			lio_rect(const CvRect &value)
			{
				x = value.x;
				y = value.y;
				width = value.width;
				height = value.height;
			}
			lio_rect& operator=(lio_rect& value)
			{
				x = value.x;
				y = value.y;
				width = value.width;
				height = value.height;
				return *this;
			}

			lio_rect& operator+(lio_rect& value)
			{
				x += value.x;
				y += value.y;
				width += value.width;
				height += value.height;
				return *this;
			}
			bool operator==(lio_rect& value)
			{
				if (x == value.x&&y == value.y&&height == value.height&&width == value.width)
				{
					return true;
				}
				else return false;

			}

			bool operator< (const lio_rect &a) const
			{
				if (x<a.x || (x == a.x&&y<a.y) || (x == a.x&&y == a.y&&width<a.width) || (x == a.x&&y == a.y&&width == a.width&&height<a.height))
				{
					return true;
				}
				else return false;
			}
		};
		CvRect lioRectToCvRect(const lio_rect &value);





	  //--------------------------------------------------Í¼Ïñ²Ù×÷---------------------------------------------------------//
	  bool BeRectReasonable(const CvRect &rect, int width, int height);
	  bool BeRectEqual(const CvRect &rect1, const CvRect &rect2);
	  bool BeExistInRectList(const vector<CvRect> &rectList, const CvRect &rect, int &loc);
	  bool BeTwoRectsMerged(const CvRect &rect1, const CvRect &rect2, CvRect &mergedRect, const int &width, const int &height);

	  bool CheckFrame(const Mat &frame, bool is_color);
	  const CvPoint &CenterPoint(const CvRect &rect);
	  

	  void FindContoursRects(const Mat &gray_image, int area_threadhold, vector<CvRect> &rectList);
	  void FindContoursRects(const Mat &gray_image, int area_threadhold, vector<CvRect> &rectList, vector<vector<Point>> &contours);
	  void FindContoursRects(const Mat &gray_image, int area_threadhold, int perimeter_threahold, float ratio_threahold, vector<CvRect> &rectList, vector<vector<Point>> &contours);

	  void MapPoint(const CvPoint &s_point, CvPoint &d_point, const int &s_width, const int &s_height, const int &d_width, const int &d_height);
	  void MapPoints( vector<CvPoint> &s_points, vector<CvPoint> &d_points, const int &s_width, const int &s_height, const int &d_width, const int &d_height);
	  void MapRect(const CvRect &s_rect, CvRect &d_rect, const int &s_width, const int &s_height, const int &d_width, const int &d_height, float scale = 1.0);
	  void MapRect(const CvRect &s_rect, CvRect &d_rect, const CvSize &s_frame_size, const CvSize &d_frame_size);
	  void MapRects(const vector<CvRect>&s_rects, vector<CvRect> &d_rects, const CvSize &s_frame_size, const CvSize &d_frame_size);




	  void GetDateTime(int &year, int &month, int &day, int &hour, int &minute, int &second, string &time_str);
	  string GetDateTimeString();
	  string GetDateString();

	  void MergeRects(vector<CvRect> &rect_list, const int &width, const int &height);
	  void MergeRects(const vector<CvRect> &rect_list, vector<CvRect> &d_list, vector<set<int>> &d_rect_number_set, const int &width, const int &height);




	  bool CorrectRect(CvRect &rect, const int &width, const int &height);

	  void ResizeRect(const CvRect &s_rect, CvRect &d_rect, const int &set_length, const int &width, const int &height);
	  void RectToSquare(const CvRect &s_rect, CvRect &d_square, const CvSize &frame_size);


	  void SetRectZero(CvRect &rect);
	  bool SizeEqual(const CvSize &size1, const CvSize &size2);
	  void ScaleRect(const CvRect &s_rect, CvRect &d_rect, const CvSize &frame_size, const CvSize &d_size);

	  void WriteText(Mat &frame, const CvPoint point, const String text, uchar b, uchar g, uchar r);
	  void WtrieLabel(Mat &frame, const CvRect &d_rect, const String text,uchar b = 255, uchar g = 0, uchar r = 0);

	  void BoostSleep(int ms);


	  /*
	   Merge Rects through NonMaximumSuppress
	  */
	  inline void NonMaximumSuppress(const std::vector<CvRect>& srcRects,std::vector<CvRect>& resRects,float thresh,int neighbors = 0);
	  void NonMaximumSuppress2(const std::vector<CvRect>& srcRects, const std::vector<float>& scores, std::vector<CvRect>& resRects, float thresh, int neighbors=0);

	  /*
		Save Reg result
	  */
	  void SaveRegResult(const char* service_id, const Mat &reg_mat, const string &label);
	  void SaveRegFrame(const char* service_id, const Mat &frame);

}