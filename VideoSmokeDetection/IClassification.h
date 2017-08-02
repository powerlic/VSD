#pragma once

#include "cvpreload.h"
#include <string>
#include <vector>

#ifdef _DLL_CLASSIFICATION
#define DLL_CLASSIFICATION_API __declspec(dllexport)
#else
#define DLL_CLASSIFICATION_API __declspec(dllimport)
#endif

using std::string;
typedef std::pair<string, float> Prediction;

class Classifier {
public:
	virtual void Release() = 0;
	virtual std::vector<Prediction> Classify(const cv::Mat& img, int N = 5) = 0;
};

extern "C" DLL_CLASSIFICATION_API Classifier* _stdcall CreateClassifier(string model_file, string trained_file, string mean_file, string label_file, bool is_use_GPU);
extern "C" DLL_CLASSIFICATION_API void _stdcall DestroyClassifier(Classifier* pClassifier);