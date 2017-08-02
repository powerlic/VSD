#include "stdafx.h"
#include "MsgData.h"


struct ret_code_name
{
	string name;
	RetCode ret_code;
};

const ret_code_name ret_code_name_list[] = 
{
	{ "success", success },
	{ "failed", failed },
};

inline const string& RetCodeName(RetCode ret_code)
{
	return ret_code_name_list[ret_code].name;
}


struct error_code_name
{
	string name;
	ErrorCode error_code;
};

const error_code_name error_code_name_list[] = 
{
	{ "address_error", address_error },
	{ "bg_error", bg_error },
	{ "caffe_error", caffe_error },
	{ "filter_error", filter_error }
};

inline const string& ErrorCodeName(ErrorCode error_code)
{
	return error_code_name_list[error_code].name;
}

/*------------------------feedback_code_name--------------------------------*/
struct feedback_code_name
{
	string name;
	FeedbackCode feedback_code;
};

const feedback_code_name feedback_code_name_list[] =
{
	{ "service_started", service_started },
	{ "service_configed", service_configed },
	{ "service_stoped", service_stoped }
};

inline const string& StatusCodeName()
{

}


/*------------------------status_code_name--------------------------------*/







MsgData::MsgData()
{



}


MsgData::~MsgData()
{
}
