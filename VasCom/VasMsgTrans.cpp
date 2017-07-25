#include "stdafx.h"
#include "VasMsgTrans.h"


VasMsgTrans::VasMsgTrans(const char *proto_file)
{
	CHECK(vas::ReadProtoFromTextFile(proto_file, &com_paramter_));
}
VasMsgTrans::VasMsgTrans(const vas::VasComputeNode &param)
{
	com_paramter_.CopyFrom(param);

}
void VasMsgTrans::CheckParam(const vas::VasComputeNode &param)
{


}
VasMsgTrans::~VasMsgTrans()
{
	com_paramter_.Clear();
}


string VasMsgTrans::TransSendMsg(const vas::Message *msg, uint16_t frame_type)
{
	string msg_content = pb2json(*msg);
	int msg_content_len = msg_content.length();
	BYTE frame_header[FRAME_HEADER_LEN];
	GenFrameHeader(frame_type, msg_content_len, frame_header);
	string send_msg = string((char*)frame_header) + msg_content;
	return send_msg;
}
void VasMsgTrans::TransReceiveMsg(const string &msg, vas::VasReturn &ret_msg)
{
	if (msg.length()<FRAME_HEADER_LEN)
	{
		return;
	}
	string content = msg.substr(FRAME_HEADER_LEN);
	json2pb(ret_msg, content.data(), content.length() - FRAME_HEADER_LEN);
}

void VasMsgTrans::GenFrameHeader(const uint16_t &frame_type,const int &content_length, BYTE* headerbuf)
{
	FrameHeader header;
	header.version =com_paramter_.com_version();
	header.retryCount = com_paramter_.retry_count();
	header.frameId = 1;
	header.totalLength = (uint32_t)(FRAME_HEADER_LEN + content_length);
	header.nodeId = com_paramter_.node_id();
	header.centerId = com_paramter_.center_id();
	header.frameType = frame_type;

	SYSTEMTIME st;
	GetLocalTime(&st);
	tm temptm = { st.wSecond, st.wMinute, st.wHour, st.wDay, st.wMonth - 1, st.wYear - 1900, st.wDayOfWeek, 0, 0 };
	time_t ts = mktime(&temptm) * 1000 + st.wMilliseconds;
	header.timestamp = ts;

	int offset = 0;
	//version
	headerbuf[offset] = (BYTE)(header.version & (char)0xFF);
	offset += sizeof(header.version);

	//retryCount
	headerbuf[offset] = (BYTE)(header.retryCount & (char)0xFF);
	offset += sizeof(header.retryCount);

	//frameId
	headerbuf[offset] = (BYTE)((header.frameId >> 8) & (char)0xFF);
	headerbuf[offset + 1] = (BYTE)(header.frameId & (char)0xFF);
	offset += sizeof(header.frameId);
	//totalLength
	headerbuf[offset] = (BYTE)((header.totalLength >> 24) & (char)0xFF);
	headerbuf[offset + 1] = (BYTE)((header.totalLength >> 16) & (char)0xFF);
	headerbuf[offset + 2] = (BYTE)((header.totalLength >> 8) & (char)0xFF);
	headerbuf[offset + 3] = (BYTE)(header.totalLength & (char)0xFF);
	offset += sizeof(header.totalLength);

	//nodeId
	headerbuf[offset] = (BYTE)((header.nodeId >> 24) & (char)0xFF);
	headerbuf[offset + 1] = (BYTE)((header.nodeId >> 16) & (char)0xFF);
	headerbuf[offset + 2] = (BYTE)((header.nodeId >> 8) & (char)0xFF);
	headerbuf[offset + 3] = (BYTE)(header.nodeId & (char)0xFF);
	offset += sizeof(header.nodeId);
	//centerId
	headerbuf[offset] = (BYTE)((header.centerId >> 24) & (char)0xFF);
	headerbuf[offset + 1] = (BYTE)((header.centerId >> 16) & (char)0xFF);
	headerbuf[offset + 2] = (BYTE)((header.centerId >> 8) & (char)0xFF);
	headerbuf[offset + 3] = (BYTE)(header.centerId & (char)0xFF);
	offset += sizeof(header.centerId);
	//frameType
	headerbuf[offset] = (BYTE)((header.frameType >> 8) & (char)0xFF);
	headerbuf[offset + 1] = (BYTE)(header.frameType & (char)0xFF);
	offset += sizeof(header.frameType);
	//timestamp
	headerbuf[offset] =     (BYTE)((header.timestamp >> 56) & (char)0xFF);
	headerbuf[offset + 1] = (BYTE)((header.timestamp >> 48) & (char)0xFF);
	headerbuf[offset + 2] = (BYTE)((header.timestamp >> 40) & (char)0xFF);
	headerbuf[offset + 3] = (BYTE)((header.timestamp >> 32) & (char)0xFF);
	headerbuf[offset + 4] = (BYTE)((header.timestamp >> 24) & (char)0xFF);
	headerbuf[offset + 5] = (BYTE)((header.timestamp >> 16) & (char)0xFF);
	headerbuf[offset + 6] = (BYTE)((header.timestamp >> 8) & (char)0xFF);
	headerbuf[offset + 7] = (BYTE)(header.timestamp & (char)0xFF);
}