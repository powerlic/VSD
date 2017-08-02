#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>


#ifdef _DLL_SSDP
#define DLL_SSDP_API __declspec(dllexport)
#else
#define DLL_SSDP_API __declspec(dllimport)
#endif

using namespace std;
using namespace cv;

struct DetectRect
{
	string label;
	float score;
	float x1;
	float y1;
	float x2;
	float y2;
};


class SsdP
{
	public:
		virtual vector<DetectRect> Detect(const cv::Mat& img) = 0;
};

extern "C" DLL_SSDP_API SsdP* _stdcall CreateSsdP(const string& model_file, const string& weights_file, const string& mean_file, const string& mean_value, const string &label_file);
