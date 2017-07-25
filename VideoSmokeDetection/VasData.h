#pragma once
#include"stdafx.h"
#include "cvpreload.h"


using namespace std;
using namespace cv;

/*---------------Bcakground-------------------*/
enum BgMethod
{
	VIBE = 0,
	GUSSIAN = 1
};

enum BgStatus
{
	UNINIALIZED,
	SETUP,
	UPDATING
};




/*-------------------Filter---------------*/
struct Threshold
{
	int th1_lower;
	int th1_upper;
	int th2_lower;
	int th2_upper;
	int th3_lower;
	int th3_upper;

	Threshold(int th1_l = 0, int th1_u = 255, int th2_l = 0, int th2_u = 255, int th3_l = 0, int th3_u = 255)
	{
		th1_lower = th1_l;
		th1_upper = th1_u;
		th2_lower = th2_l;
		th2_upper = th2_u;
		th3_lower = th3_l;
		th3_upper = th3_u;
	}
};

enum SmokeFilterMode
{
	FRAME_COLOR_CPU_MODE = 1,
	FRAME_COLOR_GPU_MODE = 2,
	CONTOUR_COLOR_ACC_MODE = 3,
	CONTOUR_COLOR_MODE = 4,
	CONTOUR_ACC_MODE = 5,
	PURE_CONTOUR_AREA_MODE = 6,
};





/*---------------Detect------------------*/
enum DetectStatus
{
	DETECTION = 1,
	PASS = 2,
	STOP_CHECK = 3
};

struct DetectParameter
{
	string service_id;
	CvSize reg_size = cvSize(800, 576);
	bool use_smoke_filter = true;
	uint8_t reg_interval = 20;

	/*-------------BgModel--------------*/
	CvSize bg_size = cvSize(420, 320);
	BgMethod bg_method = GUSSIAN;

	uint8_t morphology_open_times = 1;
	uint8_t dilate_times = 1;
	float stable_threahold = 0.5;

	uint8_t num_samples = 20;
	uint8_t	min_match = 4;
	uint8_t	radius = 10;
	uint8_t subsample_factor = 16;
	uint8_t bg2_delay = 50;
	uint8_t max_mismatch_count = 50;
	bool vibe_use_gpu = 1;

	uint8_t num_history = 20;
	uint8_t	var_threshold = 4;
	bool use_shadow_detection = true;
	float learn_rate = 0.001;


	/*-------Filter----------*/
	uint8_t	acc_threshold = 50;
	uint8_t	frame_list_size = 10;

	CvSize filter_size = cvSize(420, 320);
	SmokeFilterMode filter_mode = PURE_CONTOUR_AREA_MODE;
	int contour_area_threahold;
	int contour_perimeter_threahold;
	float area_perimeter_ratio_threahold;

	Threshold rgb_th;
	Threshold ycbcr_th;


	/*-------Caffe----------*/
	float confience_probability = 0.5;
	int caffe_input_size = 224;
	string model_version;
};


/*-------------------------------------*/
enum VideoSource
{
	VIDEO_FILE,
	RTSP_STREAM
};

enum DecodeMode
{
	USE_CPU = 0,
	USE_GPU = 1
};

enum StreamStatus
{
	PLAY = 1,
	PAUSE = 2,
	CONNECTING = 3,
	NETWORK_FAULT = 4,
	FILE_FAULT = 5,
	CPU_DECODER_ERROR = 6,
	GPU_DECODER_ERROR = 7,
	STOP = 8,
	START = 9,
	SWTICH_DECODE_MODE = 10,
	UNKNOWN
};

struct DecodedFrame
{
	int64 No;
	int64 pts;//present time in ms unit
	Mat frame;

	DecodedFrame(int w, int h)
	{
		No = 0;
		pts = 0;
		frame = Mat::zeros(cvSize(w, h), CV_8UC3);
	}
	~DecodedFrame()
	{
		frame.release();
	}

};

