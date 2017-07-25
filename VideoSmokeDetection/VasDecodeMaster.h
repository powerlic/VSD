#pragma once
#include"stdafx.h"
#include"VasVideoSource.h"
#include"ffmpeg_dxva.h"
#include"LicImageOperation.h"


struct DecodeContext
{
	AVFormatContext* p_format_ctx;
	AVDictionary* p_dict;
	AVCodecContext *p_codec_ctx;
	AVFrame *p_decoded_frame;
	InputStream input_stream;//for GPU decode
	SwsContext *p_sws_ctx;
	int video_stream_index;
	int fps;
	DecodeContext()
	{
		p_format_ctx = NULL;
		p_dict = NULL;
		p_codec_ctx = NULL;
		p_decoded_frame = NULL;
		p_sws_ctx = NULL;
		video_stream_index = -1;
		fps = 0;
	}
	~DecodeContext()
	{
		if (p_format_ctx)
		{
			avformat_close_input(&p_format_ctx);
			p_format_ctx = NULL;
		}
		if (p_dict)
		{
			av_dict_free(&p_dict);
			p_dict = NULL;
		}
		if (p_codec_ctx)
		{
			if (p_codec_ctx->opaque)
			{
				dxva2_uninit(p_codec_ctx);
			}
			avcodec_close(p_codec_ctx);
			p_codec_ctx = NULL;
		}
		if (p_decoded_frame)
		{
			av_frame_free(&p_decoded_frame);
		}
		if (p_sws_ctx)
		{
			sws_freeContext(p_sws_ctx);
			p_sws_ctx = NULL;
		}
	}
};

class VideoStream
{
public:
	VideoStream();
	VideoStream(const char *proto_file_name);
	VideoStream(const vas::DecodeParameter &param);
	~VideoStream();
	string service_id_;
	uint16_t connect_times_ = 0;
	vas::DecodeParameter decode_param_;
	DecodeContext decode_ctx_;
	concurrent_queue<DecodedFrame> frame_queue_;
	uint64_t last_ms = 0;
	uint64_t decode_count = 0;
	void CheckParameter(const vas::DecodeParameter &param);
	bool is_send_start_failed_msg = false;
};



class VasDecodeMaster
{
public:
	VasDecodeMaster();
	virtual ~VasDecodeMaster();



	/*Register a stream status process function for decode process
	  Fun should be as void(const char *, vas::StreamStatus)
	  and use boost::bind(&SomeClass::fun, this, _1,_2) to set
	*/
	void SetStreamStatusErrorFunction(signal_t::slot_function_type fun);





	/*Add a video stream and start immediately
	*@param service_id[in]: unique index for a video stream
	*@param url[in]: the rstp addr or video file path
	*@param url_type[in]: 0, rtsp stream; 1, video file path
	*@param decode_mode[in]: 0, cpu; 1, gpu
	*@warning: GPU decode mode is not support!!!!
	*@param dst_width[in]: assigned width for output video
	*@param dst_height[in]: assigned height for output video
	*@return  0, if init success, -1, if init failed
	*/
	bool AddVideoStream(const char *service_id, const char *url, const int url_type,
		               const int decode_mode, const int dst_width, const int dst_height);


	bool AddVideoStream(const char *proto_file_name);
	bool AddVideoStream(const vas::DecodeParameter &param);


	/*Delete a video stream by the service_id
	*@param service_id[in]: unique service_id for a video stream
	*/
	bool DeleteVideoStream(const char *service_id);
	bool DeleteVideoStream(const string &service_id);


	/*Delete all video streams 
	*@return true if get the frame, else return false
	*/
	void DeleteVideoStream();

	/*Get the front frame from the frame queue
	*@
	*/
	/*bool GetDecodedFrame(const string &service_id, Mat &frame, int64_t &pts, int64_t &No);*/
	bool GetDecodedFrame(const char *service_id, Mat &frame, int64_t &pts, int64_t &No);
	


	/*Pause all decode threads
	*/
	void Pause();
	/*Pause a decode thread
	*/
	bool Pause(const char* service_id);
	bool Pause(const string &service_id);

	/*Resume all decode threads after pause
	*/
	void Resume();

	/*Resume a decode thread after pause
	*/
	bool Resume(const char* service_id);
	bool Resume(const string &service_id);



	/*Skip video to a assigned position in a video file
	*@warning: only useful for a video file
	*return whether success for this skip command
	*/
	bool Seek(const char* service_id, float pos);
	bool Seek(const string &service_id, float pos);

	/*Set decoded dst frame size for model
	@param service_id[in] service_id which need resize
	@param width[in]
	@param height[in]
	@warning: if width<400 or height <300, the size deos not change!
	*/
	bool SetSize(const char *service_id, const int width, const int height);
	bool SetSize(const string &service_id, const int width, const int height);


	/*
	@param service_id[in]
	@return the interval time bewteen two decode process, -1 if service_id does not exist! unit is ms
	*/
	int GetIntervalTime(const char *service_id);
	int GetIntervalTime(const string &service_id);

	/*Set the interval time bewteen two decode process
	@param service_id[in], failed when interval_time<0, unit is ms
	*/
	bool SetIntervalTime(const char *service_id, uint32_t interval_time);
	bool SetIntervalTime(const string &service_id, uint32_t interval_time);
	void SetIntervalTime(uint32_t interval_time);


	/*
		Service Command Reply Proccess Function
	*/
	void SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun);



private:

	signal_t sig_;
	signal_service_command_reply service_command_reply_signal_;
	connection_t sig_connect_;
	connection_t service_command_reply_connetct_;


	void VideoCapture(shared_ptr<VideoStream> ptr_video_stream);

	int InitVideoStream(const char *service_id, const char *url, const int url_type,
		                const int decode_mode, const int dst_width, const int dst_height);


	int CreateDecodeContext(shared_ptr<VideoStream> ptr_video_stream);
	void DestoryDecodeContext(shared_ptr<VideoStream> ptr_video_stream);

	int SetUpVideoFile(shared_ptr<VideoStream> ptr_video_stream);
	int SetUpRtspStream(shared_ptr<VideoStream> ptr_video_stream);
	int FindCpuDecoder(shared_ptr<VideoStream> ptr_video_stream);
	int FindGpuDecoder(shared_ptr<VideoStream> ptr_video_stream);


	inline shared_ptr<VideoStream> GetVideoStreamPtr(const char* service_id);
	inline shared_ptr<VideoStream> GetVideoStreamPtr(const string&service_id);

	vector<boost::thread> video_decode_thread_list_;
	vector<shared_ptr<VideoStream>> video_stream_list_;

	DXVA2Master *p_dxva2_master_;

	char err_buf_[1024];
	boost::mutex decode_mutex_;

	boost::mutex mutex_;

	AVPacket ReadPacket(shared_ptr<VideoStream> ptr_video_stream);

	void Decode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt);
	void CpuDecode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt);
	void GpuDecode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt);

	bool ConvertFrame(shared_ptr<VideoStream> ptr_video_stream, char *out_data, int out_width, int out_height);

	static double r2d(AVRational r);
	int Pts(shared_ptr<VideoStream> ptr_video_stream, const AVPacket *p_pkt);

	bool Seek(shared_ptr<VideoStream> ptr_video_stream, float pos);

	bool CheckIdUnique(const char* service_id);
	bool CheckIdUnique(const string &service_id);
};

