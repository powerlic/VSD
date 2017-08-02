#pragma once
#include"stdafx.h"
#include"Background.h"
#include"BgGuassian.h"
#include"BgVibe.h"
#include "LicImageOperation.h"
#include "VasProto.prototxt.pb.h"
#include "VasIO.h"
#include "ssdp.h"


#ifdef _DEBUG
#pragma comment(lib, "ssdp_d.lib")
#else
#pragma comment(lib, "ssdp.lib")
#endif




typedef boost::signal<void(const char* service_id, Mat &reg_frame, const vector<CvRect> &reg_rects, const uint32_t reg_width, const uint32_t reg_height, const vector<float> &scores, const vector<string> &labels)>  signal_reg;
typedef boost::signals::connection  connection_t;


class SsdDetecter
{
public:
	SsdDetecter();
	~SsdDetecter();

	SsdDetecter(const vas::SsdDetecterParameter &param);
	SsdDetecter(const char *file_name);



	string service_id_;
	Mat raw_frame_;
	Mat reg_frame_;
	Mat show_frame_;
	int64_t pts_;
	int64_t No_;


	vas::SsdDetecterParameter detect_param_;

	void CheckParameter(const vas::SsdDetecterParameter &param);

	VideoWriter video_writer_;
	string video_name_;

	bool has_window_ = false;
};



class VasSsdMaster
{
public:
	VasSsdMaster();
	virtual ~VasSsdMaster();

	/*
	*Add a Detecter and to start the dectect thread
	*@param proto_file_name[in],the ini file name to initialize a detecter
	*@return whether add success
	*/
	bool AddDetecter(const char *proto_file_name);

	/*Create a detecter only through service_id by default parameters*/
	bool AddDetecter(const string &service_id);

	/*Create a detecter by detect param*/
	bool AddDetecter(const vas::SsdDetecterParameter &param);

	/*
	*Delete a Detecter and stop the detect thread immediately!
	*@param service_id[in]
	*@return whether delete success
	*/
	bool DeleteDetecter(const char *service_id);
	bool DeleteDetecter(const string &service_id);

	void SetSsdDetectFunction(std::function<vector<DetectRect>(const Mat&)> set_fun);
	void SetRegResultProcessFunction(signal_reg::slot_function_type fun);
	void SetGetFrameFunction(std::function<bool(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)> set_fun);


	/*Real time reg result or show frame*/
	//void GetRegResult(const char *service_id, Mat &frame, int64_t &pts, vector<CvRect> &rects, vector<String> &labels, vector<float> &probs);
	//void GetRegResultShowFrame(const char *service_id, Mat &frame);

	bool SetShowRegFrame(const char* service_id, bool is_show);

protected:
	void VideoDetect(shared_ptr<SsdDetecter> ptr_detecter);

	void CreateDetecterContext(shared_ptr<SsdDetecter> ptr_detecter);
	void DestroyDetecterContext(shared_ptr<SsdDetecter> ptr_detecter);

	shared_ptr<SsdDetecter> GetDetecterPtr(const char *service_id);
	shared_ptr<SsdDetecter> GetDetecterPtr(const string &service_id);

	bool UpdateFrame(shared_ptr<SsdDetecter> ptr_detecter, const Mat &frame, const int64_t &pts, const int64_t &No);

	void DetectFrame(shared_ptr<SsdDetecter> ptr_detecter, const Mat &frame, const int64_t &pts, const int64_t &No);

	bool CheckIdUnique(const char* service_id);
	bool CheckIdUnique(const string &service_id);






	void SetupVideoSaveFile(shared_ptr<SsdDetecter> ptr_detecter);
	//caffe
	std::function<vector<DetectRect>(const Mat&)> ssd_detect_fun_;


	std::function<bool(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)> get_frame_fun_;

	vector<shared_ptr<SsdDetecter>> detecter_list_;
	vector<boost::thread> detect_thread_list_;

	bool CheckRegType(const string &label, const vas::SsdDetecterParameter &param);

	boost::mutex mutex_;

	signal_reg reg_signal_;
	connection_t reg_signal_connetct_;
};

