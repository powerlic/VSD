#include "stdafx.h"
#include "VasDecodeMasterPro.h"

/*
  1.Register Decoders
  2.Check Network
  3.Load GPU decode Accerlate module
  4.Set Log level

*/
#define MAX_INT_64 9223372036854775807

VasDecodeMasterPro::VasDecodeMasterPro()
{
	err_buf_[0] = '\0';

	av_register_all();

	
	int ret = avformat_network_init();
	if (ret!=0)
	{
		av_strerror(ret, err_buf_, 1024);
		av_log(NULL, AV_LOG_ERROR, "[VasDecodeMasterPro.cpp] VasDecodeMasterPro: Network init failied %s\n",err_buf_);
		return;
	}

	cout << "[VasDecodeMasterPro.cpp] VasDecodeMasterPro:: Connect wait time: " << time_out_ << endl;


	p_dxva2_master_ = dxva2_init_master();

	if (!p_dxva2_master_)
	{
		av_log(NULL, AV_LOG_ERROR, "[VasDecodeMasterPro.cpp] VasDecodeMasterPro: Dxva2 enviroment init failied\n");
	}

#ifdef _DEBUG 
	av_log_set_level(AV_LOG_FATAL);
#else
	av_log_set_level(AV_LOG_FATAL);
#endif


}


VasDecodeMasterPro::~VasDecodeMasterPro()
{

	if (p_dxva2_master_)
	{
		dxva2_uninit_master(p_dxva2_master_);
	}
}


/*-----------------------------------------Interface------------------------------------------------------------------*/

int VasDecodeMasterPro::InitVideoStream(const char *service_id, const char *url, const int url_type,
	const int decode_mode, const int dst_width, const int dst_height)
{
	shared_ptr<VideoStream> ptr_video_stream = shared_ptr<VideoStream>(new VideoStream);
	
	ptr_video_stream->service_id = service_id;
	if (url_type==0)
	{
		ptr_video_stream->video_source = RTSP_STREAM;
		ptr_video_stream->rtsp_addr = url;
	}
	else if (url_type==1)
	{
		ptr_video_stream->video_source = VIDEO_FILE;
		ptr_video_stream->file_name = url;
	}

	if (decode_mode==0)
	{
		ptr_video_stream->decode_mode = USE_CPU;
	}
	else if (decode_mode == 1)
	{
		ptr_video_stream->decode_mode = USE_GPU;
	}

	if (dst_width <= 0 || dst_height<=0)
	{
		cout << __FUNCTION__ << ": " <<" Error dst size for output video " << endl;
		return -1;
	}
	ptr_video_stream->dst_height = dst_height;
	ptr_video_stream->dst_width = dst_width;


	video_stream_list_.push_back(ptr_video_stream);
	return 0;
}

int VasDecodeMasterPro::AddVideoStream(const char *service_id, const char *url, const int url_type,
	               const int decode_mode, const int dst_width, const int dst_height)
{
	int ret = InitVideoStream(service_id, url, url_type,decode_mode, dst_width, dst_height);
	if (ret==0)
	{
		video_decode_thread_list_.push_back(boost::thread(&VasDecodeMasterPro::VideoCapture, this, video_stream_list_.back()));
		boost::this_thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(500));
	}
	return ret;
}


bool VasDecodeMasterPro::DeleteVideoStream(const char *service_id)
{

	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			(*iter)->stream_status = STOP;
			BoostSleep(500);
			(*iter).reset();
			video_stream_list_.erase(iter);
			cout << __FUNCTION__ << ": " << service_id << " delete!" << endl;
			return true;
		}
	}
	cout << __FUNCTION__ << " No service named " << service_id << " exist!" << endl;
	return false;

}
void VasDecodeMasterPro::DeleteVideoStream()
{
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id;
		(*iter)->stream_status = STOP;
		BoostSleep(500);
		video_stream_list_.erase(iter);	
		cout << __FUNCTION__ << ": " << t_service_id << " delete " << endl;
	}
	if (video_stream_list_.size()==0)
	{
		cout << __FUNCTION__ << ": " << " No service exist in list!" << endl;
	}
	
}

void VasDecodeMasterPro::Start()
{
	for (size_t i = 0; i < video_stream_list_.size(); i++)
	{
		video_stream_list_[i]->stream_status = UNKNOWN;
		video_decode_thread_list_.push_back(boost::thread(&VasDecodeMasterPro::VideoCapture, this, video_stream_list_[i]));
		boost::this_thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(500));
	}
}
void VasDecodeMasterPro::Stop()
{
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id;
		(*iter)->stream_status = STOP;
		BoostSleep(500);
		cout << __FUNCTION__ << ": "<< t_service_id << " stop!" << endl;
	}

}

void VasDecodeMasterPro::Pause()
{
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id;
		(*iter)->stream_status = PAUSE;
		BoostSleep(500);
		cout << __FUNCTION__  << ": "<<t_service_id << " pause!" << endl;
	}
}

void VasDecodeMasterPro::Pause(const char* service_id)
{
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			(*iter)->stream_status = PAUSE;
			BoostSleep(500);
			cout << __FUNCTION__ << ": " << t_service_id << " pause!" << endl;
			return;
		}
	}
	cout << __FUNCTION__ << " No service named " << service_id << " exist!" << endl;
}

void VasDecodeMasterPro::Resume()
{
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id;
		(*iter)->stream_status = PLAY;
		BoostSleep(500);
		cout << __FUNCTION__ << ": " << t_service_id << " resume!" << endl;
	}
}


void VasDecodeMasterPro::Resume(const char* service_id)
{
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			(*iter)->stream_status = PLAY;
			BoostSleep(500);
			cout << __FUNCTION__ << ": " << t_service_id << " resume!" << endl;
			return;
		}
	}
	cout << __FUNCTION__ << "No service named " << service_id << " exist!" << endl;
}

bool VasDecodeMasterPro::Seek(const char* service_id, float pos)
{
	bool seek_ok = false;
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			seek_ok = Seek(*iter, pos);
			BoostSleep(500);
			int sec, min;
			min = int(pos*(*iter)->total_ms) / (1000 * 60);
			sec = (int(pos*(*iter)->total_ms) / 1000) % 60;
			if (seek_ok)
			{
				cout << __FUNCTION__ << ": " << service_id << " skip to " << min << ":" << sec << " success!" << endl;
			}
			else
			{
				cout << __FUNCTION__ << ": " << service_id << " skip to " << min << ":" << sec << " failed!" << endl;
			}
			return seek_ok;
		}
	}
	cout << __FUNCTION__ << " No service named " << service_id << " exist!" << endl;
	return false;

}

bool VasDecodeMasterPro::GetDecodedFrame(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)
{
	
	shared_ptr<VideoStream> ptr_video_stream = GetVideoStreamPtr(service_id);
	if (ptr_video_stream&&ptr_video_stream->frame_queue.size()>0)
	{
		DecodedFrame t_decoded_frame = ptr_video_stream->frame_queue.pop();

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


void VasDecodeMasterPro::SetSize(const char *service_id, const int width, const int height)
{
	if (width<400||height<300)
	{
		cout << __FUNCTION__ << ": " << service_id << ": set size faild, please set the size greater than (400,300)!";
		return;
	}
	shared_ptr<VideoStream> ptr_video_stream = GetVideoStreamPtr(service_id);
	boost::mutex::scoped_lock lock(decode_mutex_);
	if (ptr_video_stream)
	{
		ptr_video_stream->dst_height = height;
		ptr_video_stream->dst_width = width;
		cout << __FUNCTION__ << ": " << service_id << " dst size switch to (" << width << ","<<height<<")" << endl;;
	}
	else
	{
		cout << __FUNCTION__ << " No service named " << service_id << " exist!" << endl;
	}
	lock.unlock();
}

const CvSize& VasDecodeMasterPro::GetDstSize(const char *service_id)
{
	CvSize size = cvSize(0, 0);
	shared_ptr<VideoStream> ptr_video_stream = GetVideoStreamPtr(service_id);
	if (ptr_video_stream)
	{
		size=cvSize(ptr_video_stream->dst_width, ptr_video_stream->dst_height);
	}
	else
	{
		cout << __FUNCTION__ << " No service named " << service_id << " exist!" << endl;
	}
	return size;
}

int VasDecodeMasterPro::GetIntervalTime(const char *service_id)
{
	shared_ptr<VideoStream> ptr_video_stream = GetVideoStreamPtr(service_id);
	if (ptr_video_stream)
	{
		return ptr_video_stream->interval_time;
	}
	else
	{
		cout << __FUNCTION__ << " No service named " << service_id << " exist!" << endl;
		return -1;
	}
}

void VasDecodeMasterPro::SetIntervalTime(const char *service_id, unsigned int interval_time)
{
	shared_ptr<VideoStream> ptr_video_stream = GetVideoStreamPtr(service_id);
	if (ptr_video_stream)
	{
		ptr_video_stream->interval_time = interval_time;
	}
	else
	{
		cout << __FUNCTION__ << " No service named " << service_id << " exist!" << endl;
	}
}

void VasDecodeMasterPro::SetIntervalTime(unsigned int interval_time)
{
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		(*iter)->interval_time = interval_time;
	}
}

void VasDecodeMasterPro::SetDecodeMode(const char *service_id, int decode_mode)
{
	shared_ptr<VideoStream> ptr_video_stream = GetVideoStreamPtr(service_id);
	bool b_s=false;
	if (ptr_video_stream)
	{
		if (decode_mode == 0 && ptr_video_stream->decode_mode == USE_GPU)
		{
			ptr_video_stream->decode_mode = USE_CPU;
			b_s = true;
		}
		else if (decode_mode == 1 && ptr_video_stream->decode_mode == USE_CPU)
		{
			ptr_video_stream->decode_mode = USE_GPU;
			b_s = true;
		}

		if (b_s)
		{
			ptr_video_stream->stream_status = SWTICH_DECODE_MODE;
		}
	}
}

void VasDecodeMasterPro::SetDecodeMode(int decode_mode)
{
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		bool b_s = false;
		if (decode_mode == 0 && (*iter)->decode_mode == USE_GPU)
		{
			(*iter)->decode_mode = USE_CPU;
			b_s = true;
		}
		else if (decode_mode == 1 && (*iter)->decode_mode == USE_CPU)
		{
			(*iter)->decode_mode = USE_GPU;
			b_s = true;
		}

		if (b_s)
		{
			(*iter)->stream_status = SWTICH_DECODE_MODE;
		}
	}

}


/*====================================Interface========================================*/


/*
*
*/
void VasDecodeMasterPro::VideoCapture(shared_ptr<VideoStream> ptr_video_stream)
{
	while (true)
	{
		if (ptr_video_stream->stream_status == STOP)
		{
			break;
		}

		int ret = CreateDecodeContext(ptr_video_stream);

		//stream_status_signal_(ptr_video_stream->service_id,GetStreamStatus(ptr_video_stream->service_id.c_str()));
		//cout << "OK " << endl;

		if (ret < 0)
		{
			cout << __FUNCTION__ << ": " << ptr_video_stream->service_id << " create decode thread failed!" << endl;
			//Notice
			DestoryDecodeContext(ptr_video_stream);
			return;
		}

		AVPacket packet;
		int pts = -1;
		int64_t max_pts;
		if (ptr_video_stream->video_source == VIDEO_FILE)
		{
			max_pts = ptr_video_stream->total_ms;
		}
		else max_pts = MAX_INT_64;
		
		cout << __FUNCTION__ << ": " << ptr_video_stream->service_id << " create decode thread success!" << endl;

		if (ptr_video_stream->last_ms>0 && ptr_video_stream->video_source==VIDEO_FILE)
		{
			float skip_pos = (float)ptr_video_stream->last_ms / ptr_video_stream->total_ms;
			ptr_video_stream->last_ms = 0;
			Seek(ptr_video_stream, skip_pos);

		}

		int frame_no=0;
		while (pts<max_pts)
		{
			//cout << ptr_video_stream->service_id << " get frame " << frame_no++ << endl;
			if (ptr_video_stream->stream_status == PAUSE)
			{
				BoostSleep(500);
				continue;
			}

			packet = ReadPacket(ptr_video_stream);
			pts = Pts(ptr_video_stream, &packet);
			if (ptr_video_stream->stream_status == STOP || ptr_video_stream->stream_status == SWTICH_DECODE_MODE)
			{
				if (ptr_video_stream->stream_status == SWTICH_DECODE_MODE)
				{
					cout << __FUNCTION__ << ": " << ptr_video_stream->service_id << " decode mode change to " << ptr_video_stream->decode_mode << endl;
				}
				ptr_video_stream->last_ms = pts;
				av_packet_unref(&packet);
				break;
			}
			
			if (pts==max_pts)
			{
				ptr_video_stream->stream_status = STOP;
				break;
			}

			if (pts<0)
			{
				break;
			}
			if (packet.data)
			{
				Decode(ptr_video_stream, &packet);
				av_packet_unref(&packet);
				if (ptr_video_stream->decode_ctx.p_decoded_frame)
				{
					DecodedFrame t_decoded_frame(ptr_video_stream->dst_width, ptr_video_stream->dst_height);
					bool convert = ConvertFrame(ptr_video_stream, (char*)t_decoded_frame.frame.data, ptr_video_stream->dst_width, ptr_video_stream->dst_height);
					if (convert)
					{
						ptr_video_stream->decode_count++;
						t_decoded_frame.pts = pts;
						t_decoded_frame.No = ptr_video_stream->decode_count;
					
				
						if (ptr_video_stream->frame_queue.size()<10)
						{
							ptr_video_stream->frame_queue.push(t_decoded_frame);
						}
						
						if (ptr_video_stream->interval_time>0)
						{
							BoostSleep(ptr_video_stream->interval_time);
						}
					}
				}
			}
			av_packet_unref(&packet);
			
		}

		DestoryDecodeContext(ptr_video_stream);

		BoostSleep(500);
	}
	
}


int VasDecodeMasterPro::SetUpVideoFile(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);

	const char *file_name = ptr_video_stream->file_name.c_str();

	ptr_video_stream->decode_ctx.p_format_ctx = NULL;
	int ret = avformat_open_input(&ptr_video_stream->decode_ctx.p_format_ctx, file_name, NULL, 0);

	AVFormatContext *fc = ptr_video_stream->decode_ctx.p_format_ctx;

	if (ret != 0)
	{
		lock.unlock();
		av_strerror(ret, err_buf_, sizeof(err_buf_));
		av_log(NULL, AV_LOG_ERROR, "%s : avformat_open_input error:%s\n",__FUNCTION__, err_buf_);
		ptr_video_stream->stream_status = FILE_FAULT;
		return -1;
	}

	//总时长
	ptr_video_stream->total_ms = ((fc->duration / AV_TIME_BASE) * 1000);
	cout << __FUNCTION__ << ": " << ptr_video_stream->service_id << " total_ms："<<ptr_video_stream->total_ms << endl;


	ret = avformat_find_stream_info(fc, NULL);//取出流信息
	if (ret != 0)
	{
		lock.unlock();
		av_strerror(ret, err_buf_, sizeof(err_buf_));
		av_log(NULL, AV_LOG_ERROR, "%s : avformat_open_input error:%s\n", __FUNCTION__, err_buf_);
		ptr_video_stream->stream_status = FILE_FAULT;
		return -1;
	}

	av_dump_format(fc, 0, file_name, 0);
	lock.unlock();

	return 0;
}


int VasDecodeMasterPro::SetUpRtspStream(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);

	ptr_video_stream->decode_mode = USE_GPU;
	ptr_video_stream->stream_status = CONNECTING;

	//Init AVDictionary
	av_dict_set(&ptr_video_stream->decode_ctx.p_dict, "rtsp_transport", "tcp", 0);

	//Set Max wait time
	char c_time[10];
	_itoa_s(time_out_, c_time, 10);
	string time_cout_s(c_time);
	av_dict_set(&ptr_video_stream->decode_ctx.p_dict, "stimeout", time_cout_s.c_str(), 0);

	const char *rtsp_addr = ptr_video_stream->rtsp_addr.c_str();

	int ret = avformat_open_input(&ptr_video_stream->decode_ctx.p_format_ctx, rtsp_addr, NULL, &ptr_video_stream->decode_ctx.p_dict);

	AVFormatContext *fc = ptr_video_stream->decode_ctx.p_format_ctx;

	if (ret<0)
	{
		ptr_video_stream->stream_status = NETWORK_FAULT;
		av_strerror(ret, err_buf_, sizeof(err_buf_));
		av_log(NULL, AV_LOG_ERROR, "%s avformat_open_input error:%s\n", __FUNCTION__, err_buf_);
		lock.unlock();

		if (ptr_video_stream->connect_times>3)
		{
			ptr_video_stream->stream_status = STOP;
		}
		return -1;
	}

	ret = avformat_find_stream_info(fc, NULL);//取出流信息
	if (ret < 0)
	{
		ptr_video_stream->stream_status = NETWORK_FAULT;
		av_strerror(ret, err_buf_, sizeof(err_buf_));
		av_log(NULL, AV_LOG_ERROR, "%s avformat_find_stream_info error:%s\n", __FUNCTION__, err_buf_);
		lock.unlock();

		if (ptr_video_stream->connect_times>3)
		{
			ptr_video_stream->stream_status = STOP;
		}
		return -1;
	}

	av_dump_format(fc, 0, rtsp_addr, 0);
	lock.unlock();
	return 0;

}

int VasDecodeMasterPro::FindCpuDecoder(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);

	AVFormatContext *fc = ptr_video_stream->decode_ctx.p_format_ctx;
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
				av_log(NULL, AV_LOG_ERROR, "[VasDecodeMasterPro.cpp] %s: avcodec_find_decoder error\n", __FUNCTION__);
				ptr_video_stream->stream_status = CPU_DECODER_ERROR;
				return -1;
			}

			ret = avcodec_open2(enc, codec, NULL);
			if (ret != 0)
			{
				lock.unlock();
				av_strerror(ret, err_buf_, sizeof(err_buf_));
				av_log(NULL, AV_LOG_ERROR, "[VasDecodeMasterPro.cpp] %s: avcodec_open2 error:%s\n", __FUNCTION__, err_buf_);
				ptr_video_stream->stream_status = CPU_DECODER_ERROR;
				return -1;
			}

			//Setup
			ptr_video_stream->decode_ctx.video_stream_index = i;
			ptr_video_stream->decode_ctx.p_codec_ctx = fc->streams[i]->codec;
			ptr_video_stream->decode_ctx.p_codec_ctx->codec = codec;
			ptr_video_stream->decode_ctx.fps = r2d(ptr_video_stream->decode_ctx.p_format_ctx->streams[i]->avg_frame_rate);
			cout << __FUNCTION__ << ": " << ptr_video_stream->service_id << ": fps " << ptr_video_stream->decode_ctx.fps << endl;
			av_log(NULL, AV_LOG_INFO, "[VasDecodeMasterPro.cpp] %s: set up %s CPU decode context success\n", __FUNCTION__, ptr_video_stream->service_id.c_str());
			lock.unlock();
			return 0;
		}
	}
	lock.unlock();
	return -1;
}

int VasDecodeMasterPro::FindGpuDecoder(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);
	AVFormatContext *fc = ptr_video_stream->decode_ctx.p_format_ctx;
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
				av_log(NULL, AV_LOG_ERROR, "%s: avcodec_find_decoder error\n", __FUNCTION__);
				ptr_video_stream->stream_status = GPU_DECODER_ERROR;
				return -1;
			}

			ret = avcodec_open2(enc, codec, NULL);
			if (ret != 0)
			{
				lock.unlock();
				av_strerror(ret, err_buf_, sizeof(err_buf_));
				av_log(NULL, AV_LOG_ERROR, "%s: avcodec_open2 error:%s\n", __FUNCTION__, err_buf_);
				ptr_video_stream->stream_status = GPU_DECODER_ERROR;
				return -1;
			}

			//Setup
			ptr_video_stream->decode_ctx.video_stream_index = i;
			ptr_video_stream->decode_ctx.p_codec_ctx = fc->streams[i]->codec;
			ptr_video_stream->decode_ctx.p_codec_ctx->codec = codec;
			ptr_video_stream->decode_ctx.fps = r2d(ptr_video_stream->decode_ctx.p_format_ctx->streams[i]->avg_frame_rate);
			cout << __FUNCTION__ << ": " << ptr_video_stream->service_id << ": fps: " << ptr_video_stream->decode_ctx.fps << endl;

			//Acc
			ptr_video_stream->decode_ctx.input_stream.hwaccel_id = HWACCEL_DXVA2;
			ptr_video_stream->decode_ctx.input_stream.active_hwaccel_id = HWACCEL_DXVA2;
			ptr_video_stream->decode_ctx.input_stream.hwaccel_device = 0;
			ptr_video_stream->decode_ctx.input_stream.hwaccel_ctx = NULL;
			ptr_video_stream->decode_ctx.input_stream.hwaccel_get_buffer = NULL;
			ptr_video_stream->decode_ctx.input_stream.hwaccel_retrieve_data = NULL;
			ptr_video_stream->decode_ctx.input_stream.hwaccel_uninit = NULL;

			AVCodecContext *codecctx = ptr_video_stream->decode_ctx.p_codec_ctx;

			codecctx->opaque = &ptr_video_stream->decode_ctx.input_stream;
			codecctx->get_format = get_format;
			codecctx->get_buffer2 = get_buffer;
			codecctx->thread_safe_callbacks = 1;

			if (!p_dxva2_master_)
			{
				lock.unlock();
				av_log(NULL, AV_LOG_ERROR, "%s: dxva2 decoder setup error:%s\n", __FUNCTION__);
				ptr_video_stream->stream_status = GPU_DECODER_ERROR;
				return -1;
			}

			dxva2_init(codecctx, p_dxva2_master_);

			ret = avcodec_open2(codecctx, codec, NULL);

			if (ret<0)
			{
				lock.unlock();
				av_log(NULL, AV_LOG_ERROR, "%s: dxva2 decoder open error:%s\n", __FUNCTION__);
				ptr_video_stream->stream_status = GPU_DECODER_ERROR;
				return -1;
			}

			av_log(NULL, AV_LOG_INFO, "%s: set up %s GPU decode context success\n", __FUNCTION__, ptr_video_stream->service_id.c_str());
			lock.unlock();
			return 0;
		}
	}

	lock.unlock();
	return -1;

}


/*
*Create Decode Context
*/
int VasDecodeMasterPro::CreateDecodeContext(shared_ptr<VideoStream> ptr_video_stream)
{
	int ret = -1;
	if (ptr_video_stream->video_source==VIDEO_FILE)
	{
		ret = SetUpVideoFile(ptr_video_stream);
	}
	else if (ptr_video_stream->video_source == RTSP_STREAM)
	{
		ret = SetUpRtspStream(ptr_video_stream);
	}
	if (ret<0)
	{
		return ret;
	}

	if (ptr_video_stream->decode_mode == USE_CPU)
	{
		ret = FindCpuDecoder(ptr_video_stream);
	}
	else
	{
		ret = FindGpuDecoder(ptr_video_stream);
		if (ret<0)
		{
			ret = FindCpuDecoder(ptr_video_stream);
			if (ret==0)
			{
				ptr_video_stream->decode_mode = USE_CPU;
				cout << __FUNCTION__ << ptr_video_stream->service_id.c_str() << "switch to cpu decode mode " << endl;
			}
		}

	}
	if (ret==0)
	{
		ptr_video_stream->stream_status = PLAY;
	}
	LOG(INFO) << "Create " << ptr_video_stream->service_id << " decode context success!" << endl;
	return ret;
}

void VasDecodeMasterPro::DestoryDecodeContext(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);
	if (ptr_video_stream->decode_ctx.p_sws_ctx)
	{
		sws_freeContext(ptr_video_stream->decode_ctx.p_sws_ctx);
		ptr_video_stream->decode_ctx.p_sws_ctx = NULL;
	}

	if (ptr_video_stream->decode_ctx.p_codec_ctx)
	{
		if (ptr_video_stream->decode_ctx.p_codec_ctx->opaque)
		{
			dxva2_uninit(ptr_video_stream->decode_ctx.p_codec_ctx);
		}
		avcodec_close(ptr_video_stream->decode_ctx.p_codec_ctx);
		ptr_video_stream->decode_ctx.p_codec_ctx = NULL;
	}
	if (ptr_video_stream->decode_ctx.p_format_ctx)
	{
		avformat_close_input(&ptr_video_stream->decode_ctx.p_format_ctx);
	}
	if (ptr_video_stream->decode_ctx.p_decoded_frame)
	{
		av_frame_free(&ptr_video_stream->decode_ctx.p_decoded_frame);
	}
	if (ptr_video_stream->frame_queue.size()>0)
	{
		while (ptr_video_stream->frame_queue.size()>0)
		{
			DecodedFrame d_frame= ptr_video_stream->frame_queue.pop();
			d_frame.frame.release();
		}
		ptr_video_stream->frame_queue.clear();
	}
	cout << __FUNCTION__<<": " << ptr_video_stream->service_id << " destroy decode context " << endl;
	lock.unlock();
}


//需要自己释放ReadPacket
AVPacket VasDecodeMasterPro::ReadPacket(shared_ptr<VideoStream> ptr_video_stream)
{
	boost::mutex::scoped_lock lock(decode_mutex_);
	AVPacket pkt;
	memset(&pkt, 0, sizeof(AVPacket));
	AVFormatContext *fc = ptr_video_stream->decode_ctx.p_format_ctx;
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

void VasDecodeMasterPro::Decode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt)
{
	if (ptr_video_stream->decode_mode==USE_CPU)
	{
		CpuDecode(ptr_video_stream, p_pkt);
	}
	else if (ptr_video_stream->decode_mode == USE_GPU)
	{
		GpuDecode(ptr_video_stream, p_pkt);
	}
}

void VasDecodeMasterPro::CpuDecode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt)
{
	boost::mutex::scoped_lock lock(decode_mutex_);

	AVFormatContext *fc = ptr_video_stream->decode_ctx.p_format_ctx;

	if (!fc)
	{
		lock.unlock();
		ptr_video_stream->decode_ctx.p_decoded_frame = NULL;
		return;
	}
	if (ptr_video_stream->decode_ctx.p_decoded_frame == NULL)
	{
		ptr_video_stream->decode_ctx.p_decoded_frame = av_frame_alloc();
	}
	int got_picture=0;
	int btsize = avcodec_decode_video2(ptr_video_stream->decode_ctx.p_codec_ctx,
		ptr_video_stream->decode_ctx.p_decoded_frame,
		&got_picture, p_pkt);
	if (btsize < 0)
	{
		lock.unlock();
		av_log(NULL, AV_LOG_INFO, "%s: decode failed \n", __FUNCTION__);
		if (ptr_video_stream->decode_ctx.p_decoded_frame)
		{
			av_frame_free(&ptr_video_stream->decode_ctx.p_decoded_frame);
			ptr_video_stream->decode_ctx.p_decoded_frame = NULL;
		}
		return;
	}
	lock.unlock();
}

void VasDecodeMasterPro::GpuDecode(shared_ptr<VideoStream> ptr_video_stream, AVPacket *p_pkt)
{
	
	AVFormatContext *fc = ptr_video_stream->decode_ctx.p_format_ctx;

	if (!fc)
	{
		ptr_video_stream->decode_ctx.p_decoded_frame = NULL;
		return;
	}
	if (ptr_video_stream->decode_ctx.p_decoded_frame == NULL)
	{
		ptr_video_stream->decode_ctx.p_decoded_frame = av_frame_alloc();
	}
	int got_picture = 0;
	int btsize = avcodec_decode_video2(ptr_video_stream->decode_ctx.p_codec_ctx,
		ptr_video_stream->decode_ctx.p_decoded_frame,
		&got_picture, p_pkt);
	if (btsize < 0)
	{
		//av_log(NULL, AV_LOG_INFO, "%s: decode failed \n", __FUNCTION__);
		if (ptr_video_stream->decode_ctx.p_decoded_frame)
		{
			av_frame_free(&ptr_video_stream->decode_ctx.p_decoded_frame);
			ptr_video_stream->decode_ctx.p_decoded_frame = NULL;
		}
		return;
	}
	if (got_picture)
	{
		if (ptr_video_stream->decode_ctx.input_stream.hwaccel_retrieve_data 
			&& ptr_video_stream->decode_ctx.p_decoded_frame->format == ptr_video_stream->decode_ctx.input_stream.hwaccel_pix_fmt)
		{
			boost::mutex::scoped_lock lock(decode_mutex_);
			int err = ptr_video_stream->decode_ctx.input_stream.hwaccel_retrieve_data(ptr_video_stream->decode_ctx.p_codec_ctx,
																				      ptr_video_stream->decode_ctx.p_decoded_frame);
			lock.unlock();
			if (err < 0)
			{ 
				printf("%s DecodeMaster: dxva2 Decode failed\n",__FUNCTION__);
				if (ptr_video_stream->decode_ctx.p_decoded_frame)
				{
					av_frame_free(&ptr_video_stream->decode_ctx.p_decoded_frame);
					ptr_video_stream->decode_ctx.p_decoded_frame = NULL;
				}
				return;
			}
		}
	}
	
}

bool VasDecodeMasterPro::ConvertFrame(shared_ptr<VideoStream> ptr_video_stream, char *out_data, int out_width, int out_height)
{
	boost::mutex::scoped_lock lock(decode_mutex_);

	if (!ptr_video_stream->decode_ctx.p_format_ctx)
	{
		lock.unlock();
		return false;
	}
	AVFormatContext *fc = ptr_video_stream->decode_ctx.p_format_ctx;
	AVCodecContext *video_ctx = fc->streams[ptr_video_stream->decode_ctx.video_stream_index]->codec;

	ptr_video_stream->decode_ctx.p_sws_ctx = sws_getCachedContext(ptr_video_stream->decode_ctx.p_sws_ctx, video_ctx->width, video_ctx->height, video_ctx->pix_fmt, out_width, out_height,
		AV_PIX_FMT_BGR24,
		SWS_BICUBIC,
		NULL, NULL, NULL);

	if (!ptr_video_stream->decode_ctx.p_sws_ctx)
	{
		lock.unlock();
		printf("%s: sws_getCachedContext failed:\n", __FUNCTION__);
		return false;
	}

	uint8_t *data[AV_NUM_DATA_POINTERS] = { 0 };
	data[0] = (uint8_t *)out_data;
	int linesize[AV_NUM_DATA_POINTERS] = { 0 };

	linesize[0] = out_width * 3;

	sws_scale(ptr_video_stream->decode_ctx.p_sws_ctx,
		ptr_video_stream->decode_ctx.p_decoded_frame->data,
		ptr_video_stream->decode_ctx.p_decoded_frame->linesize,
		0, video_ctx->height, data, linesize);


	lock.unlock();
	return true;
}


double VasDecodeMasterPro::r2d(AVRational r)
{
	return r.num == 0 || r.den == 0 ? 0. : (double)r.num / (double)r.den;
}


int VasDecodeMasterPro::Pts(shared_ptr<VideoStream> ptr_video_stream, const AVPacket *p_pkt)
{
	boost::mutex::scoped_lock lock(decode_mutex_);
	if (!ptr_video_stream->decode_ctx.p_codec_ctx)
	{
		lock.unlock();
		return -1;
	}
	int pts = (p_pkt->pts *r2d(ptr_video_stream->decode_ctx.p_format_ctx->streams[p_pkt->stream_index]->time_base)) * 1000;
	lock.unlock();
	return pts;
}

bool VasDecodeMasterPro::Seek(shared_ptr<VideoStream> ptr_video_stream, float pos)
{
	boost::mutex::scoped_lock lock(decode_mutex_);
	if (!ptr_video_stream->decode_ctx.p_format_ctx)
	{
		lock.unlock();
		return false;
	}
	int64_t stamp = 0;
	stamp = pos * ptr_video_stream->decode_ctx.p_format_ctx->streams[ptr_video_stream->decode_ctx.video_stream_index]->duration;
	int re = av_seek_frame(ptr_video_stream->decode_ctx.p_format_ctx, ptr_video_stream->decode_ctx.video_stream_index, stamp,
		AVSEEK_FLAG_BACKWARD | AVSEEK_FLAG_FRAME);
	avcodec_flush_buffers(ptr_video_stream->decode_ctx.p_format_ctx->streams[ptr_video_stream->decode_ctx.video_stream_index]->codec);

	lock.unlock();

	if (re >= 0)
		return true;
	return false;
}

void VasDecodeMasterPro::BoostSleep(int ms)
{
	boost::this_thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(ms));
}

shared_ptr<VideoStream> VasDecodeMasterPro::GetVideoStreamPtr(const char* service_id)
{
	vector<shared_ptr<VideoStream>>::iterator iter;
	for (iter = video_stream_list_.begin(); iter != video_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			return *iter;
		}
	}
	return NULL;
}

