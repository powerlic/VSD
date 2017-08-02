#pragma once
#include"VasDecodeMasterPro.h"
#include"VasDetecterMasterPro.h"

#include"IClassification.h"

#include <crtdbg.h>

#ifdef _DEBUG
#pragma comment(lib,"caffe-classifier_d.lib")
#else 
#pragma comment(lib,"caffe-classifier.lib")
#endif


class VasInstance
{
	string service_id_;
	DecodeParameter decode_parameter_;
	DetectParameter detect_parameter_;
};


class Vas
{
public:
	Vas();
	virtual ~Vas();

	void InitCaffeModel(const char *file_name);
	void test();

private:
	boost::mutex caffe_mutex;
	shared_ptr<Classifier> ptr_classifier;
	vector<Prediction> Predict(const Mat &image_mat, const int &N = 5);

	shared_ptr<VasDecodeMasterPro> ptr_decode_master;
	shared_ptr<VasDetecterMasterPro> ptr_detecter_master;

	void SlotProcessStreamStatus(const string &service_id,const StreamStatus status)
	{
		cout << service_id<<":"<<status << endl;
		for (size_t i = 0; i < 10; i++)
		{
			cout << service_id << ":" << status << endl;
			Sleep(1000);
		}
	}
};

