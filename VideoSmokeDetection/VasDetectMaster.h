#pragma once
#include"stdafx.h"
#include"Background.h"
#include"BgGuassian.h"
#include"BgVibe.h"
#include"BgVibeBoost.h"
#include "Filter.h"
#include "LicImageOperation.h"
#include "IClassification.h"
#include "VasProto.prototxt.pb.h"
#include "VasIO.h"

typedef boost::signal<void(const char* service_id, const vector<CvRect> &reg_rects, const uint32_t reg_width, const uint32_t reg_height, const vector<float> &scores, const vector<string> &labels)>  signal_reg;
typedef boost::signal<void(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)>  signal_service_command_reply;
typedef boost::signals::connection  connection_t;


class Detecter
{
public:
	Detecter();
	~Detecter();

	Detecter(const vas::DetectParameter &param);
	Detecter(const char *file_name);

	string service_id_;
	Mat raw_frame_;
	Mat reg_frame_;
	Mat show_frame_;
	
	int64_t pts_;
	int64_t No_;
	shared_ptr<Background> ptr_bg_;
	shared_ptr<Filter> ptr_filter_;

	vas::DetectParameter detect_param_;


	//special for smoke detection
	bool smoke_detect_include_;
	list<Mat> smoke_hot_frame_diff_list_;
	uint32_t smoke_threshold_;
	Mat smoke_hot_frame_;

	//special for fire detection
	bool fire_detect_include_;
	list<Mat> fire_hot_frame_diff_list_;
	uint32_t fire_threshold_;
	Mat fire_hot_frame_;

    void CheckParameter(const vas::DetectParameter &param);


	bool is_send_start_success_msg_ = false;
	bool is_send_start_failed_msg_ = false;
	bool is_send_stop_success_msg_ = false;
	bool is_send_stop_failed_msg_ = false;


	VideoWriter video_writer_;
	string video_name_;
};


class VasDetectMaster
{
public:
	VasDetectMaster();
	VasDetectMaster(std::function<vector<Prediction>(const Mat&, const int&)> caffe_reg_fun,
					std::function<bool(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)> get_frame_fun);
	virtual ~VasDetectMaster();


/*-------------------------Interface------------------------------------------*/

	/*
	*Add a Detecter and to start the dectect thread
	*@param proto_file_name[in],the ini file name to initialize a detecter
	*@return whether add success
	*/
	bool AddDetecter(const char *proto_file_name);

	/*Create a detecter only through service_id by default parameters*/
	bool AddDetecter(const string &service_id);


	/*Create a detecter by detect param*/
	bool AddDetecter(const vas::DetectParameter &param);

	/*
	*Delete a Detecter and stop the detect thread immediately!
	*@param service_id[in]
	*@return whether delete success
	*/
	bool DeleteDetecter(const char *service_id);
	bool DeleteDetecter(const string &service_id);

	/*
	 Reg result process function
	*/
	void SetRegResultProcessFunction(signal_reg::slot_function_type fun);


	/*
	 Service Command Reply Proccess Function
	*/
	void SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun);


	/*SetCaffeRegFunction*/
	void SetCaffeRegFunction(std::function<vector<Prediction>(const Mat&, const int&)> set_fun);
	/*SetGetFrameFunction*/
	void SetGetFrameFunction(std::function<bool(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)> set_fun);

	/*Real time reg result or show frame*/
	void GetRegResult(const char *service_id, Mat &frame, int64_t &pts, vector<CvRect> &rects, vector<String> &labels, vector<float> &probs);
	void GetRegResultShowFrame(const char *service_id, Mat &frame);


	/*Pause all detect threads
	*/
	void Pause();

	/*Pause a detect thread
	*/
	void Pause(const char* service_id);
	void Pause(const string& service_id);


	/*Resume all detect threads after pause
	*/
	void Resume();

	/*Resume a detect thread after pause
	*/
	void Resume(const char* service_id);
	void Resume(const string& service_id);


	/*Config Parameter*/
	void Config(const vas::DetectParameter &detect_parameter);
	void Config(const char *service_id, const vas::DetectParameter &detect_parameter);
	void Config(const string& service_id, const vas::DetectParameter &detect_parameter);

	bool GetRegResult(const char* service_id, Mat &show_frame, int64_t &pts);

protected:
	void VideoDetect(shared_ptr<Detecter> ptr_detecter);


	void CreateDetecterContext(shared_ptr<Detecter> ptr_detecter);
	void DestroyDetecterContext(shared_ptr<Detecter> ptr_detecter);


	shared_ptr<Detecter> GetDetecterPtr(const char *service_id);
	shared_ptr<Detecter> GetDetecterPtr(const string &service_id);


	bool UpdateFrame(shared_ptr<Detecter> ptr_detecter, const Mat &frame, const int64_t &pts, const int64_t &No);
	void UpdateHotFrame(shared_ptr<Detecter> ptr_detecter, const Mat &suspected_frame, const uint32_t detect_threshold, list<Mat> &hot_frame_diff_list_, Mat &hot_frame, Mat &hot_threshold_frame);
	void DetectFrame(shared_ptr<Detecter> ptr_detecter, const Mat &frame, const int64_t &pts, const int64_t &No);
	void DetectFrameMerged(shared_ptr<Detecter> ptr_detecter, const Mat &frame, const int64_t &pts, const int64_t &No);


	bool CheckIdUnique(const char* service_id);
	bool CheckIdUnique(const string &service_id);

	bool GetForeMask(shared_ptr<Detecter> ptr_detecter, Mat &mask);


	void SetupVideoSaveFile(shared_ptr<Detecter> ptr_detecter);

	void GetMoveRectFrame(const Mat &frame, const vector<CvRect> &rects, Mat &out_frame);


//caffe
	std::function<vector<Prediction>(const Mat&, const int&)> caffe_reg_fun_;
//get frame
	std::function<bool(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)> get_frame_fun_;

	vector<shared_ptr<Detecter>> detecter_list_;
	vector<boost::thread> detect_thread_list_;

	bool CheckRegType(const string &label, const vas::DetectParameter &param);
	
	void MergeSmokeRects(vector<CvRect> &rects, int dst_frame_width, int dst_frame_height, int caffe_input_size);

	boost::mutex mutex_;


	signal_reg reg_signal_;
	signal_service_command_reply service_command_reply_signal_;
	connection_t reg_signal_connetct_;
	connection_t service_command_reply_connetct_;
};

