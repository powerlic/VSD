#pragma once
#include"stdafx.h"
#include "VasProto.prototxt.pb.h"
#include "json2pb.h"
#include "VasIO.h"

using namespace std;


#define FRAME_HEADER_LEN 26
#define FRAME_TYPE_HEARTBEAT 0x0000
#define FRAME_TYPE_HEARTBEAT_RETURN 0x0000
#define FRAME_TYPE_REG_RESULT 0x0010
#define FRAME_TYPE_REG_RESULT_RETURN 0x0010
#define FRAME_TYPE_FEEDBACK        0x0020
#define FRAME_TYPE_FEEDBACK_RETURN 0x0020


struct FrameHeader
{
	uint8_t version;
	uint8_t	retryCount;
	uint16_t frameId;
	uint32_t totalLength;
	uint32_t nodeId;
	uint32_t centerId;
	uint16_t frameType;
	uint64_t timestamp;
};

class VasMsgTrans
{
public:

	static VasMsgTrans* Get()
	{
		static VasMsgTrans instance_("../proto/compute_node.prototxt");
		return &instance_;
	}
	string TransSendMsg(const vas::Message *msg, uint16_t frame_type);
	void TransReceiveMsg(const string &msg, vas::VasReturn &ret_msg);

private:

	VasMsgTrans(const char *proto_file);
	VasMsgTrans(const vas::VasComputeNode &param);
	virtual ~VasMsgTrans();

	vas::VasComputeNode com_paramter_;
	void CheckParam(const vas::VasComputeNode &param);
	void GenFrameHeader(const uint16_t &frame_type, const int &content_length, BYTE* headerbuf);
	boost::mutex send_mutex_;
	boost::mutex recevice_mutex_;
};

