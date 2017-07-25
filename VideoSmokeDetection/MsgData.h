#pragma once
#include<string>
#include"cvpreload.h"

using namespace std;
using namespace cv;


enum RetCode
{
	success = 1,
	failed = 2,
};

enum StatusCode
{
	vas_starting = 1,
	vas_running = 2,
	vas_paused = 3,
	vas_null = 4,
};

enum ErrorCode
{
	address_error = 1,
	bg_error = 2,
	caffe_error = 3,
	filter_error = 4
};

enum FeedbackCode
{
	service_started = 1,
	service_configed = 2,
	service_stoped = 3
};

struct ResultRect
{
	string reg_type;
	float score;
	CvRect rect;
};


/******************注册服务(vasRegClient)******************/
struct VasRegClient
{
	string psi_name;
	string app_key;
	uint64_t timestamp;
};
struct VasRegClientRet
{
	string psi_name;
	RetCode ret_code;
	string	ret_msg;
	string client_id;
	uint64_t timestamp;
};

/******************启动服务(vasStartService)******************/
struct VasStartService
{
	string psi_name;
	string app_key;
	string client_id;
	string reg_mode;
	vector<String> reg_types;
	String address_type;
	vector<String> address;
	uint64_t timestamp;
};
struct VasStartServiceRet
{
	string psi_name;
	RetCode ret_code;
	string	ret_msg;
	string client_id;
	string service_id;
	uint64_t timestamp;
};

/******************暂停服务(vasPauseService)******************/
struct VasPauseService
{
	string psi_name;
	string app_key;
	string client_id;
	string service_id;
	uint64_t timestamp;
};
struct VasPauseServiceRet
{
	string psi_name;
	RetCode ret_code;
	string	ret_msg;
	string client_id;
	string service_id;
	uint64_t timestamp;
};

/******************停止服务(vasStopService)******************/
struct VasStopService
{
	string psi_name;
	string app_key;
	string client_id;
	string service_id;
	uint64_t timestamp;
};
struct VasStopServiceRet
{
	string psi_name;
	RetCode ret_code;
	string	ret_msg;
	string client_id;
	string service_id;
	uint64_t timestamp;
};

/******************查询当前服务状态(vasQueryServiceStatus)******************/
struct VasQueryServiceStatus
{
	string psi_name;
	string app_key;
	string client_id;
	string service_id;
	uint64_t timestamp;
};
struct VasQueryServiceStatusRet
{
	string psi_name;
	RetCode ret_code;
	string	ret_msg;
	string client_id;
	string service_id;
	StatusCode status_code;
	string	status_info;
	uint64_t timestamp;
};

/******************查询客户端运行服务列表(vasQueryClientServiceList)******************/
struct VasQueryClientServiceList
{
	string psi_name;
	string app_key;
	string client_id;
	uint64_t timestamp;
};
struct VasQueryClientServiceListRet
{
	string psi_name;
	RetCode ret_code;
	string	ret_msg;
	string client_id;
	vector<string> service_list;
	uint64_t timestamp;
};

/******************配置服务(vasConfigService)******************/
struct VasConfigService
{
	string psi_name;
	string app_key;
	string client_id;
	string service_id;
	uint64_t timestamp;

	string reg_mode;
	vector<String> reg_types;
	String address_type;
	vector<String> address;

	vector<vector<Point>> roi_polygons;
	uint16_t reg_width;
	uint16_t reg_height;
	uint8_t bg_method;
	uint16_t bg_width;
	uint16_t bg_height;
	uint8_t morphologyEx_times;
	uint8_t dilate_times;
	float stable_threshold;

	//vibe
	uint8_t num_samples;
	uint8_t min_match;
	uint8_t radius;
	uint8_t subsample_factor;
	uint8_t max_mismatch_count;
	uint8_t bg2_delay;

	//history
	uint8_t num_history;
	uint8_t var_threshold;
	bool use_shadow_detection;
	float learn_rate;

	//filter
	uint8_t filter_mode;
	CvSize filter_size;

	//filter contour
	uint16_t contour_area_threahold;
	uint16_t contour_perimeter_threahold;
	float area_perimeter_ratio_threahold;

	//filter acc
	uint16_t acc_threahold;
	uint8_t frame_list_size;

	//filter color
	uint8_t rgb_th1_lower;
	uint8_t rgb_th1_upper;
	uint8_t rgb_th2_lower;
	uint8_t rgb_th2_upper;
	uint8_t ycbcr_th1_lower;
	uint8_t ycbcr_th1_upper;
	uint8_t ycbcr_th2_lower;
	uint8_t ycbcr_th2_upper;

	//caffe
	float confidence_threshold;
};
struct VasConfigServiceRet
{
	string psi_name;
	RetCode ret_code;
	string	ret_msg;
	string client_id;
	string service_id;
	uint64_t timestamp;
};


/******************查询服务配置(vasQueryServiceConfig)******************/
struct VasQueryServiceConfig
{
	string psi_name;
	string app_key;
	string client_id;
	string service_id;
	uint64_t timestamp;
};
struct VasQueryServiceConfigRet
{
	string psi_name;
	RetCode ret_code;
	string	ret_msg;
	string client_id;
	string service_id;
	uint64_t timestamp;

	string reg_mode;
	vector<String> reg_types;
	String address_type;
	vector<String> address;

	vector<vector<Point>> roi_polygons;
	uint16_t reg_width;
	uint16_t reg_height;
	uint8_t bg_method;
	uint16_t bg_width;
	uint16_t bg_height;
	uint8_t morphologyEx_times;
	uint8_t dilate_times;
	float stable_threshold;

	//vibe
	uint8_t num_samples;
	uint8_t min_match;
	uint8_t radius;
	uint8_t subsample_factor;
	uint8_t max_mismatch_count;
	uint8_t bg2_delay;

	//history
	uint8_t num_history;
	uint8_t var_threshold;
	bool use_shadow_detection;
	float learn_rate;

	//filter
	uint8_t filter_mode;
	uint16_t filter_width;
	uint16_t filter_height;

	//filter contour
	uint16_t contour_area_threahold;
	uint16_t contour_perimeter_threahold;
	float area_perimeter_ratio_threahold;

	//filter acc
	uint8_t acc_threahold;
	uint8_t frame_list_size;

	//filter color
	uint8_t rgb_th1_lower;
	uint8_t rgb_th1_upper;
	uint8_t rgb_th2_lower;
	uint8_t rgb_th2_upper;
	uint8_t ycbcr_th1_lower;
	uint8_t ycbcr_th1_upper;
	uint8_t ycbcr_th2_lower;
	uint8_t ycbcr_th2_upper;

	//caffe
	float confidence_threshold;
};

/******************识别结果上报(Vas)******************/
struct VasResult
{
	string psi_name;
	string app_key;
	string client_id;
	string service_id;
	vector<ResultRect> result_rects;
	int64_t timestamp;
};

/******************状态上报(Vas)******************/
struct VasFeedBack
{
	string client_id;
	string service_id;
	FeedbackCode feedback_code;
	string feedback_info;
	uint64_t timestamp;
};

/******************错误上报(Vas)******************/
struct VasError
{
	string psi_name;
	string app_key;
	string client_id;
	string service_id;
	ErrorCode error_code;
	string error_info;
	string error_detail;
	uint64_t timestamp;
};


class MsgData
{
public:
	MsgData();
	virtual ~MsgData();
};




