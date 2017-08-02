#pragma once
#include"ffmpeg_dxva.h"
#include"concurrent_queue.h"
#include"LicImageOperation.h"
#include"VasData.h"


using namespace std;
using namespace cv;


//context->ctx
//dictionary->dict
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
//raw->r
//context->ctx
struct VideoStream
{
	string service_id;

	DecodeMode decode_mode=USE_CPU;

	string create_time;

	//解码上下文
	DecodeContext decode_ctx;

	unsigned int dst_width = 0;
	unsigned int dst_height = 0;

	unsigned int interval_time=20;

	int64_t decode_count=0;
	int64_t reg_count=0;

	unsigned int connect_times = 0;//reconnect times after error

	// Rtsp h2664 
	string rtsp_addr;
	VideoSource video_source;

	//VideoFile
	string file_name;
	unsigned int total_ms = 0;
	unsigned int last_ms = 0;


	concurrent_queue<DecodedFrame> frame_queue;

	~VideoStream()
	{
		frame_queue.clear();
		cout << "Destroy " << service_id << " video stream create time: " << create_time << endl;
	}

};

struct DecodeParameter
{
	string service_id;
	DecodeMode decode_mode=USE_CPU;
	unsigned int dst_width = 0;
	unsigned int dst_height = 0;
	StreamStatus stream_status = UNKNOWN;
};



class VasDecodeMasterPro
{
public:

	VasDecodeMasterPro();
	virtual ~VasDecodeMasterPro();

	//typedef boost::signal<void(const string &, const StreamStatus)>  signal_t;
	//typedef boost::signals::connection  connect_t;

	//connect_t ConnectSignal(signal_t::slot_function_type subscriber)
	//{
	//	return stream_status_signal_.connect(subscriber);
	//}

	//void DisConnectSignal(connect_t subscriber)
	//{
	//	subscriber.disconnect();
	//}


	/*Register a video stream, after this should call Run() to start all the decode threads
	 *@param service_id[in]: unique index for a video stream
	 *@param url[in]: the rstp addr or video file path
	 *@param url_type[in]: 0, rtsp stream; 1, video file path
	 *@param decode_mode[in]: 0, cpu; 1, gpu
	 *@param dst_width[in]: assigned width for output video
	 *@param dst_height[in]: assigned height for output video
	 *@return  0, if init success, -1, if init failed
	 */
	int InitVideoStream(const char *service_id, const char *url,const int url_type,
						const int decode_mode,const int dst_width, const int dst_height);


	/*Add a video stream and start immediately
	 *@param service_id[in]: unique index for a video stream
	 *@param url[in]: the rstp addr or video file path
	 *@param url_type[in]: 0, rtsp stream; 1, video file path
	 *@param decode_mode[in]: 0, cpu; 1, gpu
	 *@param dst_width[in]: assigned width for output video
	 *@param dst_height[in]: assigned height for output video
	 *@return  0, if init success, -1, if init failed
	 */
	int AddVideoStream(const char *service_id, const char *url, const int url_type,
		               const int decode_mode, const int dst_width, const int dst_height);


	/*Delete a video stream by the service_id
	 *@param service_id[in]: unique index for a video stream
	 */
	bool DeleteVideoStream(const char *service_id);


	/*Delete all video streams by the service_id
	 *@param service_id[in]: unique index for a video stream
	 *@param frame[out]: the decoded frame
	 *@param pts[out]: the pts of decoded frame
	 *@param No[out]: the No of decode frame
	 *@return true if get the frame, else return false
	 */
	void DeleteVideoStream();


	/*Get the front frame from the frame queue
	 *@
	 */
	bool GetDecodedFrame(const char *service_id, Mat &frame, int64_t &pts, int64_t &No);

	/*Start all decode threads
	 */
	void Start();

	/*Stop all decode threads
	 */
	void Stop();

	/*Pause all decode threads
	*/
	void Pause();
	/*Pause a decode thread
	 */
	void Pause(const char* service_id);

	/*Resume all decode threads after pause
	 */
	void Resume();

	/*Resume a decode thread after pause
	 */
	void Resume(const char* service_id);


	/*Skip video to a assigned position in a video file
	 *@warning: only useful for a video file
	 *return whether success for this skip command
	 */
	bool Seek(const char* service_id, float pos);


	/*Set decoded dst frame size for model
	 @param service_id[in] service_id which need resize
	 @param width[in]
	 @param height[in]
	 @warning: if width<400 or height <300, the size deos not change!
	*/
	void SetSize(const char *service_id, const int width, const int height);

	/*Get decoded dst frame size for model
	@param service_id[in] service_id which need resize
	*/
	const CvSize& GetDstSize(const char *service_id);

	/*
	@param service_id[in]
	@return the interval time bewteen two decode process, -1 if service_id does not exist! unit is ms
	*/
	int GetIntervalTime(const char *service_id);


	/*Set the interval time bewteen two decode process
	@param service_id[in], failed when interval_time<0, unit is ms
	*/
	void SetIntervalTime(const char *service_id, unsigned int interval_time);

	/*Set the interval time bewteen two decode process for all threads, unit is ms
	@param service_id[in], failed when interval_time<0
	*/
	void SetIntervalTime(unsigned int interval_time);

	/*Set decode mode (Use gpu or cpu)
	@param service_id[in]
	@param decode_mode[in]: 0, use cpu; 1, use gpu;
	*/
	void SetDecodeMode(const char *service_id, int decode_mode);
	/*Set decode mode (Use gpu or cpu) for all threads
	*/
	void SetDecodeMode(int decode_mode);

	/*Get Stream Status 
	*/
	StreamStatus GetStreamStatus(const char *service_id)
	{
		shared_ptr<VideoStream> ptr_video_stream = GetVideoStreamPtr(service_id);
		if (ptr_video_stream)
		{
			return ptr_video_stream->stream_status;
		}
		
	}


private:
	void VideoCapture(shared_ptr<VideoStream> ptr_video_stream);

	int CreateDecodeContext(shared_ptr<VideoStream> ptr_video_stream);
	void DestoryDecodeContext(shared_ptr<VideoStream> ptr_video_stream);

	int SetUpVideoFile(shared_ptr<VideoStream> ptr_video_stream);
	int SetUpRtspStream(shared_ptr<VideoStream> ptr_video_stream);
	int FindCpuDecoder(shared_ptr<VideoStream> ptr_video_stream);
	int FindGpuDecoder(shared_ptr<VideoStream> ptr_video_stream);

	inline void BoostSleep(int ms);
	inline shared_ptr<VideoStream> GetVideoStreamPtr(const char* service_id);

	vector<boost::thread> video_decode_thread_list_;

	vector<shared_ptr<VideoStream>> video_stream_list_;

	DXVA2Master *p_dxva2_master_;

	char err_buf_[1024];

	int time_out_=100;//链接失败时间

	boost::mutex decode_mutex_;


	//void SeekVideoFilePoint(shared_ptr<VideoStream> ptr_video_stream, float loc);

	//需要在外部释放AVPacket空间
	AVPacket ReadPacket(shared_ptr<VideoStream> ptr_video_stream);

	void Decode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt);
	void CpuDecode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt);
	void GpuDecode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt);

	bool ConvertFrame(shared_ptr<VideoStream> ptr_video_stream, char *out_data, int out_width, int out_height);

	static double r2d(AVRational r);
	int Pts(shared_ptr<VideoStream> ptr_video_stream, const AVPacket *p_pkt);

	bool Seek(shared_ptr<VideoStream> ptr_video_stream, float pos);

	//signal_t stream_status_signal_;
};

