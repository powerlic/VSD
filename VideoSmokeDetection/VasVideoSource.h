#pragma once
#include"stdafx.h"
#include "cvpreload.h"
#include "concurrent_queue.h"
#include "VasProto.prototxt.pb.h"

using namespace std;
using namespace cv;

typedef boost::signal<void(const char* service_id, vas::StreamStatus status)>  signal_t;
typedef boost::signal<void(const char* service_id, vas::CamStatus cam_status)>  cam_signal_t;
typedef boost::signal<void(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)>  signal_service_command_reply;
typedef boost::signals::connection  connection_t;

struct DecodedFrame
{
	Mat frame;
	uint64_t pts;
	uint64_t No;
	DecodedFrame(uint32_t w, uint32_t h)
	{
		frame = Mat::zeros(h, w, CV_8UC3);
		pts = 0;
		No = 0;
	}
};

