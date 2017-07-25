#include"stdafx.h"
#include"VasDecodeMaster.h"
#include"VasIO.h"

using namespace std;
using namespace cv;



/*----------------------VideoStream-------------------*/
VideoStream::VideoStream()
{
	decode_param_ = vas::DecodeParameter();
	CheckParameter(decode_param_);
}
VideoStream::VideoStream(const char *proto_file_name)
{
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &decode_param_));
	service_id_ = decode_param_.service_id();
	CheckParameter(decode_param_);
}
VideoStream::VideoStream(const vas::DecodeParameter &param)
{
	decode_param_.CopyFrom(param);
	service_id_ = param.service_id();
	CheckParameter(decode_param_);
}
VideoStream::~VideoStream()
{
	frame_queue_.clear();
	decode_param_.Clear();
}

void VideoStream::CheckParameter(const vas::DecodeParameter &param)
{

	CHECK_EQ(param.decode_method(), vas::DECODE_CPU) << " GPU decode mode is not support!" << endl;
	CHECK(param.has_dst_height() && param.has_dst_width()) << " decode dst with or height not set!" << endl;
	CHECK_GE(param.dst_height(), 320) << " decode dst height must >= 320!" << endl;
	CHECK_GE(param.dst_width(), 420) << " decode dst width must >= 420!" << endl;
	CHECK_GE(param.interval_time(), 10) << "decode time interval must be greater than or equal to 10 " << endl;
	CHECK_GE(param.max_reconnect_times(), 3) << "reconnect times must be greater than or equal to 3 " << endl;
	CHECK_GE(param.connect_timeout(), 10000) << "connect time out must be greater than or equal to 500000 " << endl;
}

/*--------------------------------Interface----------------------------------------*/


void VasDecodeMaster::SetStreamStatusErrorFunction(signal_t::slot_function_type fun)
{
	sig_connect_=sig_.connect(fun);
}



bool VasDecodeMaster::AddVideoStream(const char *service_id, const char *url, const int url_type,
	const int decode_mode, const int dst_width, const int dst_height)
{
	int ret = InitVideoStream(service_id, url, url_type, decode_mode, dst_width, dst_height);
	if (ret == 0)
	{
		video_decode_thread_list_.push_back(boost::thread(&VasDecodeMaster::VideoCapture, this, video_stream_list_.back()));
		lio::BoostSleep(50);
	}
	LOG(INFO) << " Init a video stream from interface, service_id:" << service_id << endl;
	return ret;
}
bool VasDecodeMaster::AddVideoStream(const char *proto_file_name)
{
	boost::mutex::scoped_lock lock(mutex_);
	shared_ptr<VideoStream> ptr_video_stream = shared_ptr<VideoStream>(new VideoStream(proto_file_name));
	video_stream_list_.push_back(ptr_video_stream);
	video_decode_thread_list_.push_back(boost::thread(&VasDecodeMaster::VideoCapture, this, video_stream_list_.back()));
	//lio::BoostSleep(50);
	LOG(INFO) << " Init a video stream from file " << proto_file_name << endl;
	lock.unlock();
	return true;
}
bool VasDecodeMaster::AddVideoStream(const vas::DecodeParameter &param)
{
	boost::mutex::scoped_lock lock(mutex_);
	shared_ptr<VideoStream> ptr_video_stream = shared_ptr<VideoStream>(new VideoStream(param));
	video_stream_list_.push_back(ptr_video_stream);
	video_decode_thread_list_.push_back(boost::thread(&VasDecodeMaster::VideoCapture, this, video_stream_list_.back()));
	lio::BoostSleep(50);
	LOG(INFO) << " Init a video stream from param " << endl;
	lock.unlock();
	return true;
}

bool VasDecodeMaster::DeleteVideoStream(const char *service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			(*iter)->decode_param_.set_stream_status(vas::STREAM_STOP);
			lio::BoostSleep(500);
			(*iter).reset();
			video_stream_list_.erase(iter);
			LOG(INFO) << service_id << " delete!" << endl;
			lock.unlock();
			return true;
		}
	}
	lock.unlock();
	LOG(INFO) << " No service named " << service_id << " exist!" << endl;
	return false;
}

bool VasDecodeMaster::DeleteVideoStream(const string &service_id)
{
	return DeleteVideoStream(service_id.c_str());
}
void VasDecodeMaster::DeleteVideoStream()
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		(*iter)->decode_param_.set_stream_status(vas::STREAM_STOP);
		lio::BoostSleep(500);
		(*iter).reset();
		video_stream_list_.erase(iter);
		LOG(INFO) << t_service_id << " delete!" << endl;
	}
	lock.unlock();
}

bool VasDecodeMaster::Pause(const char* service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			(*iter)->decode_param_.set_stream_status(vas::STREAM_PAUSE);
			lio::BoostSleep(500);
			LOG(INFO) << service_id << " pause!" << endl;
			lock.unlock();
			return true;
		}
	}
	LOG(INFO) << " No service named " << service_id << " exist!" << endl;
	lock.unlock();
	return false;
}

bool VasDecodeMaster::Pause(const string &service_id)
{
	return Pause(service_id.c_str());
}

bool VasDecodeMaster::Resume(const char* service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			if ((*iter)->decode_param_.stream_status() == vas::STREAM_PAUSE)
			{
				(*iter)->decode_param_.set_stream_status(vas::STREAM_NORMAL);
			}
			lio::BoostSleep(500);
			LOG(INFO) << service_id << " resume!" << endl;
			lock.unlock();
			return true;
		}
	}
	LOG(INFO) << " No service named " << service_id << " exist!" << endl;
	lock.unlock();
	return false;
}
bool VasDecodeMaster::Resume(const string &service_id)
{
	return Resume(service_id.c_str());
}
bool VasDecodeMaster::Seek(const char* service_id, float pos)
{
	bool seek_ok = false;
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			seek_ok = Seek(*iter, pos);
			lio::BoostSleep(500);
			int sec, min;
			min = int(pos*(*iter)->decode_param_.total_ms()) / (1000 * 60);
			sec = (int(pos*(*iter)->decode_param_.total_ms()) / 1000) % 60;
			if (seek_ok)
			{
				LOG(INFO) << service_id << " skip to " << min << ":" << sec << " success!" << endl;
			}
			else
			{
				LOG(INFO) << service_id << " skip to " << min << ":" << sec << " failed!" << endl;
			}
			lock.unlock();
			return seek_ok;
		}
	}
	lock.unlock();
	LOG(INFO) << " No service named " << service_id << " exist!" << endl;
	return false;
}

bool VasDecodeMaster::Seek(const string &service_id, float pos)
{
	return Seek(service_id.c_str(), pos);
}

bool VasDecodeMaster::SetSize(const char *service_id, const int width, const int height)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(service_id) == 0)
		{
			if (width>=420 && height>=320)
			{
				(*iter)->decode_param_.set_dst_width(width);
				(*iter)->decode_param_.set_dst_height(height);
				LOG(INFO) << service_id << " size changed to (" << width << "," << height << ")" << endl;
			}
			lock.unlock();
			return true;
		}
	}
	lock.unlock();
	return false;
}
bool VasDecodeMaster::SetSize(const string &service_id, const int width, const int height)
{
	return SetSize(service_id.c_str(), width, height);
}
int VasDecodeMaster::GetIntervalTime(const char *service_id)
{
	shared_ptr<VideoStream> ptr_stream = GetVideoStreamPtr(service_id);
	if (ptr_stream)
	{
		return ptr_stream->decode_param_.interval_time();
	}
	else return -1;
}

int VasDecodeMaster::GetIntervalTime(const string &service_id)
{
	return GetIntervalTime(service_id.c_str());
}
bool VasDecodeMaster::SetIntervalTime(const char *service_id, uint32_t interval_time)
{
	shared_ptr<VideoStream> ptr_stream = GetVideoStreamPtr(service_id);
	if (ptr_stream)
	{
		 ptr_stream->decode_param_.set_interval_time(interval_time);
		 LOG(INFO) << service_id << " interval_time changed to " << interval_time << endl;
		 return true;
	}
	else return false;
}
bool VasDecodeMaster::SetIntervalTime(const string &service_id, uint32_t interval_time)
{
	return SetIntervalTime(service_id.c_str(),interval_time);
}
void VasDecodeMaster::SetIntervalTime(uint32_t interval_time)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		(*iter)->decode_param_.set_interval_time(interval_time);
		LOG(INFO) << (*iter)->service_id_ << " interval_time changed to "<<interval_time << endl;
	}
	lock.unlock();
}

bool VasDecodeMaster::GetDecodedFrame(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)
{
	shared_ptr<VideoStream> ptr_video_stream = GetVideoStreamPtr(service_id);
	if (ptr_video_stream&&ptr_video_stream->frame_queue_.size()>0)
	{
		DecodedFrame t_decoded_frame = ptr_video_stream->frame_queue_.pop();
		if (t_decoded_frame.frame.data)
		{
			if (frame.empty())
			{
				frame = Mat::zeros(cvSize(t_decoded_frame.frame.cols, t_decoded_frame.frame.rows), CV_8UC3);
			}
			memcpy(frame.data, t_decoded_frame.frame.data, t_decoded_frame.frame.cols*t_decoded_frame.frame.rows * 3 * sizeof(uchar));
			pts = t_decoded_frame.pts;
			No = t_decoded_frame.No;
			t_decoded_frame.frame.release();
			return true;
		}
		t_decoded_frame.frame.release();
	}
	pts = -1;
	No = -1;
	return false;
}

//bool VasDecodeMaster::GetDecodedFrame(const string &service_id, Mat &frame, int64_t &pts, int64_t &No)
//{
//	shared_ptr<VideoStream> ptr_video_stream = GetVideoStreamPtr(service_id);
//	if (ptr_video_stream&&ptr_video_stream->frame_queue_.size()>0)
//	{
//		DecodedFrame t_decoded_frame = ptr_video_stream->frame_queue_.pop();
//		if (t_decoded_frame.frame.data)
//		{
//			if (frame.empty())
//			{
//				frame = Mat::zeros(cvSize(t_decoded_frame.frame.cols, t_decoded_frame.frame.rows), CV_8UC3);
//			}
//			memcpy(frame.data, t_decoded_frame.frame.data, t_decoded_frame.frame.cols*t_decoded_frame.frame.rows * 3 * sizeof(uchar));
//			pts = t_decoded_frame.pts;
//			No = t_decoded_frame.No;
//			t_decoded_frame.frame.release();
//			return true;
//		}
//		t_decoded_frame.frame.release();
//	}
//	pts = -1;
//	No = -1;
//	return false;
//}

/*---------------------------------------------------------------------------------------*/
VasDecodeMaster::VasDecodeMaster()
{
	err_buf_[0] = '\0';

	av_register_all();


	int ret = avformat_network_init();
	CHECK_EQ(ret, 0) << " Net enviroment init failed!" << endl;

	p_dxva2_master_ = dxva2_init_master();

	CHECK(p_dxva2_master_) << "GPU dxva2 acceleration init failed" << endl;
#ifdef _DEBUG 
	av_log_set_level(AV_LOG_FATAL);
#else
	av_log_set_level(AV_LOG_FATAL);
#endif
}

VasDecodeMaster::~VasDecodeMaster()
{
	if (p_dxva2_master_)
	{
		dxva2_uninit_master(p_dxva2_master_);
	}
}

/*return -1: file open error or not support this type video, 0 openok
*/

int VasDecodeMaster::SetUpVideoFile(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);

	const char *file_name = ptr_video_stream->decode_param_.url().c_str();

	ptr_video_stream->decode_ctx_.p_format_ctx = NULL;
	int ret = avformat_open_input(&ptr_video_stream->decode_ctx_.p_format_ctx, file_name, NULL, 0);

	AVFormatContext *fc = ptr_video_stream->decode_ctx_.p_format_ctx;

	if (ret != 0)
	{
		lock.unlock();
		av_strerror(ret, err_buf_, sizeof(err_buf_));
		LOG(ERROR) << err_buf_ << endl;
		ptr_video_stream->decode_param_.set_stream_status(vas::FILE_FAULT);
		return -1;
	}

	ptr_video_stream->decode_param_.set_total_ms ((fc->duration / AV_TIME_BASE) * 1000);
	//LOG(INFO) << ptr_video_stream->service_id_ << " total ms:" << ptr_video_stream->decode_param_.total_ms() << endl;

	ret = avformat_find_stream_info(fc, NULL);
	if (ret != 0)
	{
		lock.unlock();
		av_strerror(ret, err_buf_, sizeof(err_buf_));
		LOG(ERROR) << err_buf_ << endl;
		ptr_video_stream->decode_param_.set_stream_status(vas::FILE_FAULT);
		return -1;
	}

	av_dump_format(fc, 0, file_name, 0);
	lock.unlock();

	return 0;
}
/*return -1: file open rtsp stream or not support this type video, 0 openok
*/
int VasDecodeMaster::SetUpRtspStream(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);

	ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_CONNECTING);

	//Init AVDictionary
	av_dict_set(&ptr_video_stream->decode_ctx_.p_dict, "rtsp_transport", "tcp", 0);

	char c_times[10];
	int time_out_ = 100000;
	_itoa_s(time_out_, c_times, 10);
	string time_cout_s(c_times);
	av_dict_set(&ptr_video_stream->decode_ctx_.p_dict, "stimeout", time_cout_s.c_str(), 0);


	//Max wait time
	char c_time[100];
	sprintf_s(c_time, 100, "%d", ptr_video_stream->decode_param_.connect_timeout());
	char option_key2[] = "max_delay";
	av_dict_set(&ptr_video_stream->decode_ctx_.p_dict, option_key2, c_time, 0);

	const char *rtsp_addr = ptr_video_stream->decode_param_.url().c_str();

	int ret = avformat_open_input(&ptr_video_stream->decode_ctx_.p_format_ctx, rtsp_addr, NULL, &ptr_video_stream->decode_ctx_.p_dict);

	AVFormatContext *fc = ptr_video_stream->decode_ctx_.p_format_ctx;

	if (ret<0)
	{
		ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_NETWORK_FAULT);
		av_strerror(ret, err_buf_, sizeof(err_buf_));
		LOG(ERROR) << err_buf_ << endl;
		lock.unlock();
		ptr_video_stream->connect_times_++;
		return -1;
	}

	ret = avformat_find_stream_info(fc, NULL);//取出流信息
	if (ret < 0)
	{
		ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_NETWORK_FAULT);
		av_strerror(ret, err_buf_, sizeof(err_buf_));
		LOG(ERROR) << err_buf_ << endl;
		lock.unlock();
		ptr_video_stream->connect_times_++;
		return -1;
	}

	av_dump_format(fc, 0, rtsp_addr, 0);
	lock.unlock();
	return 0;
}
/*return -1: setup cpu decoder error, 0 ok
*/
int VasDecodeMaster::FindCpuDecoder(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);

	AVFormatContext *fc = ptr_video_stream->decode_ctx_.p_format_ctx;
	int ret = -1;
	for (int i = 0; i < fc->nb_streams; i++)
	{
		AVCodecContext *enc = fc->streams[i]->codec;

		if (enc->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			AVCodec *codec = avcodec_find_decoder(enc->codec_id);
			if (!codec)
			{
				lock.unlock();
				LOG(ERROR) << ptr_video_stream->service_id_<< " avcodec_find_decoder error." << endl;
				ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_CPU_DECODE_FAULT);
				return -1;
			}

			ret = avcodec_open2(enc, codec, NULL);
			if (ret != 0)
			{
				lock.unlock();
				av_strerror(ret, err_buf_, sizeof(err_buf_));
				LOG(ERROR) << ptr_video_stream->service_id_ << err_buf_ << endl;
				ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_CPU_DECODE_FAULT);
				return -1;
			}

			//Setup
			ptr_video_stream->decode_ctx_.video_stream_index = i;
			ptr_video_stream->decode_ctx_.p_codec_ctx = fc->streams[i]->codec;
			ptr_video_stream->decode_ctx_.p_codec_ctx->codec = codec;
			ptr_video_stream->decode_ctx_.fps = r2d(ptr_video_stream->decode_ctx_.p_format_ctx->streams[i]->avg_frame_rate);
			//LOG(INFO) << ptr_video_stream->service_id_ << " set up CPU decode context success." << endl;
			lock.unlock();
			return 0;
		}
	}
	lock.unlock();
	LOG(ERROR) << ptr_video_stream->service_id_ << " find cpu decoder failed." << endl;
	return -1;
}
/*return -1: setup Gpu decoder error, 0 ok
*/
int VasDecodeMaster::FindGpuDecoder(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);
	AVFormatContext *fc = ptr_video_stream->decode_ctx_.p_format_ctx;
	int ret = -1;

	for (int i = 0; i < fc->nb_streams; i++)
	{
		AVCodecContext *enc = fc->streams[i]->codec;

		if (enc->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			AVCodec *codec = avcodec_find_decoder(enc->codec_id);

			if (!codec)
			{
				lock.unlock();
				LOG(ERROR) << ptr_video_stream->service_id_ << " avcodec_find_decoder error." << endl;
				ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_GPU_DECODE_FAULT);
				return -1;
			}

			ret = avcodec_open2(enc, codec, NULL);
			if (ret != 0)
			{
				lock.unlock();
				av_strerror(ret, err_buf_, sizeof(err_buf_));
				LOG(ERROR) << ptr_video_stream->service_id_ << " avcodec_open2 error." << endl;
				ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_GPU_DECODE_FAULT);
				return -1;
			}

			//Setup
			ptr_video_stream->decode_ctx_.video_stream_index = i;
			ptr_video_stream->decode_ctx_.p_codec_ctx = fc->streams[i]->codec;
			ptr_video_stream->decode_ctx_.p_codec_ctx->codec = codec;
			ptr_video_stream->decode_ctx_.fps = r2d(ptr_video_stream->decode_ctx_.p_format_ctx->streams[i]->avg_frame_rate);
			

			//Acc
			ptr_video_stream->decode_ctx_.input_stream.hwaccel_id = HWACCEL_DXVA2;
			ptr_video_stream->decode_ctx_.input_stream.active_hwaccel_id = HWACCEL_DXVA2;
			ptr_video_stream->decode_ctx_.input_stream.hwaccel_device = 0;
			ptr_video_stream->decode_ctx_.input_stream.hwaccel_ctx = NULL;
			ptr_video_stream->decode_ctx_.input_stream.hwaccel_get_buffer = NULL;
			ptr_video_stream->decode_ctx_.input_stream.hwaccel_retrieve_data = NULL;
			ptr_video_stream->decode_ctx_.input_stream.hwaccel_uninit = NULL;

			AVCodecContext *codecctx = ptr_video_stream->decode_ctx_.p_codec_ctx;

			codecctx->opaque = &ptr_video_stream->decode_ctx_.input_stream;
			codecctx->get_format = get_format;
			codecctx->get_buffer2 = get_buffer;
			codecctx->thread_safe_callbacks = 1;

			if (!p_dxva2_master_)
			{
				lock.unlock();
				LOG(ERROR) << ptr_video_stream->service_id_ << " gpu decoder setup failed." << endl;
				ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_GPU_DECODE_FAULT);
				return -1;
			}

			dxva2_init(codecctx, p_dxva2_master_);

			ret = avcodec_open2(codecctx, codec, NULL);

			if (ret<0)
			{
				lock.unlock();
				LOG(ERROR) << ptr_video_stream->service_id_ << " dxva2 decoder open error." << endl;
				ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_GPU_DECODE_FAULT);
				return -1;
			}
			//LOG(INFO) << ptr_video_stream->service_id_ << "set up GPU decode context success. fps:" << ptr_video_stream->decode_ctx_.fps << endl;
			lock.unlock();
			return 0;
		}
	}

	lock.unlock();
	return -1;
}


double VasDecodeMaster::r2d(AVRational r)
{
	return r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den;
}

/*
*Create Decode Context
*/
int VasDecodeMaster::CreateDecodeContext(shared_ptr<VideoStream> ptr_video_stream)
{
	int ret = -1;
	if (ptr_video_stream->decode_param_.video_source() == vas::VIDEO_FILE)
	{
		ret = SetUpVideoFile(ptr_video_stream);
	}
	else if (ptr_video_stream->decode_param_.video_source() == vas::RTSP_STREAM)
	{
		ret = SetUpRtspStream(ptr_video_stream);
	}
	if (ret<0)
	{
		return ret;
	}
	
	if(ptr_video_stream->decode_param_.decode_method()==vas::DECODE_CPU)
	{
		ret = FindCpuDecoder(ptr_video_stream);
	}
	else
	{
		ret = FindGpuDecoder(ptr_video_stream);
		if (ret<0)
		{
			ret = FindCpuDecoder(ptr_video_stream);
			if (ret == 0)
			{
				ptr_video_stream->decode_param_.set_decode_method(vas::DECODE_CPU);
				LOG(INFO) << ptr_video_stream->service_id_ << " gpu decoder setup failed, switch to cpu decoder!" << endl;
			}
		}

	}
	if (ret == 0)
	{
		ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_NORMAL);
		if (ptr_video_stream->decode_param_.decode_method() == vas::DECODE_CPU)
			LOG(INFO) << ptr_video_stream->service_id_ << " create decode thread success! url: " << ptr_video_stream->decode_param_.url() << ", decode_method: CPU."  << endl;
		else 
			LOG(INFO) << ptr_video_stream->service_id_ << " create decode thread success! url: " << ptr_video_stream->decode_param_.url() << ", decode_method: GPU."  << endl;
	}
	return ret;
}
void VasDecodeMaster::DestoryDecodeContext(shared_ptr<VideoStream> ptr_video_stream)
{
	if (ptr_video_stream->decode_ctx_.p_sws_ctx)
	{
		sws_freeContext(ptr_video_stream->decode_ctx_.p_sws_ctx);
		ptr_video_stream->decode_ctx_.p_sws_ctx = NULL;
	}

	if (ptr_video_stream->decode_ctx_.p_codec_ctx)
	{
		if (ptr_video_stream->decode_ctx_.p_codec_ctx->opaque)
		{
			dxva2_uninit(ptr_video_stream->decode_ctx_.p_codec_ctx);
		}
		avcodec_close(ptr_video_stream->decode_ctx_.p_codec_ctx);
		ptr_video_stream->decode_ctx_.p_codec_ctx = NULL;
	}
	if (ptr_video_stream->decode_ctx_.p_format_ctx)
	{
		avformat_close_input(&ptr_video_stream->decode_ctx_.p_format_ctx);
	}
	if (ptr_video_stream->decode_ctx_.p_decoded_frame)
	{
		av_frame_free(&ptr_video_stream->decode_ctx_.p_decoded_frame);
	}
	if (ptr_video_stream->frame_queue_.size() > 0)
	{
		while (ptr_video_stream->frame_queue_.size() > 0)
		{
			DecodedFrame d_frame = ptr_video_stream->frame_queue_.pop();
			d_frame.frame.release();
		}
		ptr_video_stream->frame_queue_.clear();
	}
	LOG(INFO) << ptr_video_stream->service_id_ << " destroy decode context! url:" << ptr_video_stream->decode_param_.url()<< endl;
}

//warning: should release packet after use

AVPacket VasDecodeMaster::ReadPacket(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);
	AVPacket pkt;
	memset(&pkt, 0, sizeof(AVPacket));
	AVFormatContext *fc = ptr_video_stream->decode_ctx_.p_format_ctx;
	if (!fc)
	{
		lock.unlock();
		return pkt;
	}
	int err = av_read_frame(fc, &pkt);

	if (err != 0)
	{
		av_strerror(err, err_buf_, sizeof(err_buf_));
	}
	lock.unlock();
	return pkt;
}

void VasDecodeMaster::CpuDecode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt)
{
	boost::mutex::scoped_lock lock(decode_mutex_);
	AVFormatContext *fc = ptr_video_stream->decode_ctx_.p_format_ctx;
	if (!fc)
	{
		lock.unlock();
		ptr_video_stream->decode_ctx_.p_decoded_frame = NULL;
		return;
	}
	if (ptr_video_stream->decode_ctx_.p_decoded_frame == NULL)
	{
		ptr_video_stream->decode_ctx_.p_decoded_frame = av_frame_alloc();
	}
	int got_picture = 0;
	int btsize = avcodec_decode_video2(ptr_video_stream->decode_ctx_.p_codec_ctx,
		ptr_video_stream->decode_ctx_.p_decoded_frame,
		&got_picture, p_pkt);
	if (btsize <= 0)
	{
		lock.unlock();
		if (ptr_video_stream->decode_ctx_.p_decoded_frame)
		{
			av_frame_free(&ptr_video_stream->decode_ctx_.p_decoded_frame);
			ptr_video_stream->decode_ctx_.p_decoded_frame = NULL;
		}
		return;
	}
	lock.unlock();
}

void VasDecodeMaster::GpuDecode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt)
{

	AVFormatContext *fc = ptr_video_stream->decode_ctx_.p_format_ctx;

	if (!fc)
	{
		ptr_video_stream->decode_ctx_.p_decoded_frame = NULL;
		return;
	}
	if (ptr_video_stream->decode_ctx_.p_decoded_frame == NULL)
	{
		ptr_video_stream->decode_ctx_.p_decoded_frame = av_frame_alloc();
	}
	int got_picture = 0;
	int btsize = avcodec_decode_video2(ptr_video_stream->decode_ctx_.p_codec_ctx,
		ptr_video_stream->decode_ctx_.p_decoded_frame,
		&got_picture, p_pkt);
	if (btsize <= 0)
	{
		if (ptr_video_stream->decode_ctx_.p_decoded_frame)
		{
			av_frame_free(&ptr_video_stream->decode_ctx_.p_decoded_frame);
			ptr_video_stream->decode_ctx_.p_decoded_frame = NULL;
		}
		return;
	}
	if (got_picture)
	{
		if (ptr_video_stream->decode_ctx_.input_stream.hwaccel_retrieve_data
			&& ptr_video_stream->decode_ctx_.p_decoded_frame->format == ptr_video_stream->decode_ctx_.input_stream.hwaccel_pix_fmt)
		{
			boost::mutex::scoped_lock lock(decode_mutex_);
			int err = ptr_video_stream->decode_ctx_.input_stream.hwaccel_retrieve_data(ptr_video_stream->decode_ctx_.p_codec_ctx,
				ptr_video_stream->decode_ctx_.p_decoded_frame);
			lock.unlock();
			if (err < 0)
			{
				if (ptr_video_stream->decode_ctx_.p_decoded_frame)
				{
					av_frame_free(&ptr_video_stream->decode_ctx_.p_decoded_frame);
					ptr_video_stream->decode_ctx_.p_decoded_frame = NULL;
				}
				return;
			}
		}
	}
}

void VasDecodeMaster::Decode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt)
{
	if (ptr_video_stream->decode_param_.decode_method() == vas::DECODE_CPU)
	{
		CpuDecode(ptr_video_stream, p_pkt);
	}
	else if (ptr_video_stream->decode_param_.decode_method() == vas::DECODE_GPU)
	{
		GpuDecode(ptr_video_stream, p_pkt);
	}
}

bool VasDecodeMaster::ConvertFrame(shared_ptr<VideoStream> ptr_video_stream, char *out_data, int out_width, int out_height)
{
	boost::mutex::scoped_lock lock(decode_mutex_);

	if (!ptr_video_stream->decode_ctx_.p_format_ctx)
	{
		lock.unlock();
		return false;
	}
	AVFormatContext *fc = ptr_video_stream->decode_ctx_.p_format_ctx;
	AVCodecContext *video_ctx = fc->streams[ptr_video_stream->decode_ctx_.video_stream_index]->codec;

	ptr_video_stream->decode_ctx_.p_sws_ctx = sws_getCachedContext(ptr_video_stream->decode_ctx_.p_sws_ctx, video_ctx->width, video_ctx->height, video_ctx->pix_fmt, out_width, out_height,
		AV_PIX_FMT_BGR24,
		SWS_BICUBIC,
		NULL, NULL, NULL);

	if (!ptr_video_stream->decode_ctx_.p_sws_ctx)
	{
		lock.unlock();
		return false;
	}

	uint8_t *data[AV_NUM_DATA_POINTERS] = { 0 };
	data[0] = (uint8_t *)out_data;
	int linesize[AV_NUM_DATA_POINTERS] = { 0 };

	linesize[0] = out_width * 3;

	sws_scale(ptr_video_stream->decode_ctx_.p_sws_ctx,
		ptr_video_stream->decode_ctx_.p_decoded_frame->data,
		ptr_video_stream->decode_ctx_.p_decoded_frame->linesize,
		0, video_ctx->height, data, linesize);


	lock.unlock();
	return true;
}

int VasDecodeMaster::Pts(shared_ptr<VideoStream> ptr_video_stream, const AVPacket *p_pkt)
{
	boost::mutex::scoped_lock lock(decode_mutex_);
	if (!ptr_video_stream->decode_ctx_.p_codec_ctx)
	{
		lock.unlock();
		return -1;
	}
	int pts = (p_pkt->pts *r2d(ptr_video_stream->decode_ctx_.p_format_ctx->streams[p_pkt->stream_index]->time_base)) * 1000;
	lock.unlock();
	return pts;
}

bool VasDecodeMaster::Seek(shared_ptr<VideoStream> ptr_video_stream, float pos)
{
	boost::mutex::scoped_lock lock(decode_mutex_);
	if (!ptr_video_stream->decode_ctx_.p_format_ctx)
	{
		lock.unlock();
		return false;
	}
	int64_t stamp = 0;
	stamp = pos * ptr_video_stream->decode_ctx_.p_format_ctx->streams[ptr_video_stream->decode_ctx_.video_stream_index]->duration;
	int re = av_seek_frame(ptr_video_stream->decode_ctx_.p_format_ctx, ptr_video_stream->decode_ctx_.video_stream_index, stamp,
		AVSEEK_FLAG_BACKWARD | AVSEEK_FLAG_FRAME);
	avcodec_flush_buffers(ptr_video_stream->decode_ctx_.p_format_ctx->streams[ptr_video_stream->decode_ctx_.video_stream_index]->codec);
	lock.unlock();

	if (re >= 0)
		return true;
	return false;
}

shared_ptr<VideoStream> VasDecodeMaster::GetVideoStreamPtr(const char* service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			lock.unlock();
			return *iter;
		}
	}
	lock.unlock();
	LOG(INFO) << " No service named " << service_id << endl;
	return NULL;
}

shared_ptr<VideoStream> VasDecodeMaster::GetVideoStreamPtr(const string &service_id)
{
	return GetVideoStreamPtr(service_id.c_str());
}


void VasDecodeMaster::VideoCapture(shared_ptr<VideoStream> ptr_video_stream)
{
	try
	{
		while (true)
		{
			if (ptr_video_stream->decode_param_.stream_status() == vas::STREAM_STOP)
			{
				break;
			}
			ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_CONNECTING);
			sig_(ptr_video_stream->service_id_.c_str(), ptr_video_stream->decode_param_.stream_status());
			bool send_network_normal_sig_ = false;
			int ret = CreateDecodeContext(ptr_video_stream);

			if (ret < 0)
			{
				LOG(INFO) << ptr_video_stream->service_id_ << " create decode thread failed! url:" << ptr_video_stream->decode_param_.url()<< endl;
				//Notice
				DestoryDecodeContext(ptr_video_stream);
				if (ptr_video_stream->connect_times_ > ptr_video_stream->decode_param_.max_reconnect_times())
				{
					sig_(ptr_video_stream->service_id_.c_str(), ptr_video_stream->decode_param_.stream_status());
					//ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_STOP);
				}
				if (ptr_video_stream->decode_param_.video_source() == vas::VIDEO_FILE)
				{
					sig_(ptr_video_stream->service_id_.c_str(), ptr_video_stream->decode_param_.stream_status());
					//ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_STOP);
				}

				if (!ptr_video_stream->is_send_start_failed_msg)
				{
					ptr_video_stream->is_send_start_failed_msg = true;
					service_command_reply_signal_(ptr_video_stream->service_id_.c_str(), vas::SERVICE_START_FALIED, " Service start failed!");
				}
				lio::BoostSleep(ptr_video_stream->decode_param_.reconnect_sleep_times());
				continue;
			}

			//If start success, no need to send falied_msg, send falied_msg just for first time connected
			ptr_video_stream->is_send_start_failed_msg = true;

			ptr_video_stream->connect_times_ = 0;

			AVPacket packet;
			int pts = -1;
			int64_t max_pts;
			if (ptr_video_stream->decode_param_.video_source() == vas::VIDEO_FILE)
			{
				max_pts = ptr_video_stream->decode_param_.total_ms();
			}
			else max_pts = MAX_INT_64;

			if (ptr_video_stream->last_ms>0 && ptr_video_stream->decode_param_.video_source() == vas::VIDEO_FILE)
			{
				float skip_pos = (float)ptr_video_stream->last_ms / max_pts;
				ptr_video_stream->last_ms = 0;
				Seek(ptr_video_stream, skip_pos);

			}

			int frame_no = 0;
			int pts_minus_count = 0;
			uint32_t dst_width = ptr_video_stream->decode_param_.dst_width();
			uint32_t dst_height = ptr_video_stream->decode_param_.dst_height();
			if (pts>(max_pts-10))
			{
				ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_FINISH);
				sig_(ptr_video_stream->service_id_.c_str(), ptr_video_stream->decode_param_.stream_status());
			}
			while (pts<max_pts)
			{
				if (ptr_video_stream->decode_param_.stream_status() == vas::STREAM_PAUSE)
				{
					lio::BoostSleep(500);
					continue;
				}
				if (ptr_video_stream->frame_queue_.size()>10)
				{
					lio::BoostSleep(10);
					continue;
				}
				packet = ReadPacket(ptr_video_stream);
				pts = Pts(ptr_video_stream, &packet);
				if (ptr_video_stream->decode_param_.stream_status() == vas::STREAM_STOP)
				{
					ptr_video_stream->last_ms = pts;
					av_packet_unref(&packet);
					break;
				}

				if (pts == max_pts)
				{
					ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_STOP);
					break;
				}

				if (pts<0)
				{
					pts_minus_count++;
					//break;
					if (pts_minus_count>ptr_video_stream->decode_param_.max_decode_error_frames())
					{
						break;
					}
					else
					{
						lio::BoostSleep(20);
						continue;
					}
				}
				if (packet.data)
				{
					if (!send_network_normal_sig_)
					{
						send_network_normal_sig_ = true;
						ptr_video_stream->decode_param_.set_stream_status(vas::STREAM_NORMAL);
						sig_(ptr_video_stream->service_id_.c_str(), ptr_video_stream->decode_param_.stream_status());

					}

					pts_minus_count = 0;
					Decode(ptr_video_stream, &packet);
					av_packet_unref(&packet);
					if (ptr_video_stream->decode_ctx_.p_decoded_frame)
					{
						DecodedFrame t_decoded_frame(dst_width, dst_height);
						bool convert = ConvertFrame(ptr_video_stream, (char*)t_decoded_frame.frame.data, dst_width, dst_height);
						if (convert)
						{
							ptr_video_stream->decode_count++;
							t_decoded_frame.pts = pts;
							t_decoded_frame.No = ptr_video_stream->decode_count;
							if (ptr_video_stream->frame_queue_.size()<10)
							{
								ptr_video_stream->frame_queue_.push(t_decoded_frame);
							}

							lio::BoostSleep(ptr_video_stream->decode_param_.interval_time());
						}
					}
				}
				av_packet_unref(&packet);

			}

			DestoryDecodeContext(ptr_video_stream);

			lio::BoostSleep(500);
		}
	}
	catch(...)
	{
		LOG(ERROR) << ptr_video_stream->service_id_ << " decode process exit!" << endl;
		sig_(ptr_video_stream->service_id_.c_str(), ptr_video_stream->decode_param_.stream_status());
	}
	
}

int VasDecodeMaster::InitVideoStream(const char *service_id, const char *url, const int url_type,
	const int decode_mode, const int dst_width, const int dst_height)
{
	shared_ptr<VideoStream> ptr_video_stream = shared_ptr<VideoStream>(new VideoStream);

	ptr_video_stream->service_id_ = service_id;


	if (url_type == 0)
	{
		ptr_video_stream->decode_param_.set_video_source(vas::RTSP_STREAM);
	}
	else if (url_type == 1)
	{
		ptr_video_stream->decode_param_.set_video_source(vas::VIDEO_FILE);
	}
	ptr_video_stream->decode_param_.set_url(url);


	if (decode_mode == 0)
	{
		ptr_video_stream->decode_param_.set_decode_method(vas::DECODE_CPU);
	}
	else if (decode_mode == 1)
	{
		ptr_video_stream->decode_param_.set_decode_method(vas::DECODE_GPU);
	}

	ptr_video_stream->decode_param_.set_dst_height(dst_height);
	ptr_video_stream->decode_param_.set_dst_width(dst_width);

	ptr_video_stream->CheckParameter(ptr_video_stream->decode_param_);

	video_stream_list_.push_back(ptr_video_stream);
	return 0;
}

bool VasDecodeMaster::CheckIdUnique(const char* service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			LOG(INFO) << " Already has service named as " << service_id << endl;
			lock.unlock();
			return true;
		}
	}
	lock.unlock();
	return false;
}
bool VasDecodeMaster::CheckIdUnique(const string &service_id)
{
	return CheckIdUnique(service_id.c_str());
}

void VasDecodeMaster::SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun)
{
	service_command_reply_signal_.connect(fun);
}
