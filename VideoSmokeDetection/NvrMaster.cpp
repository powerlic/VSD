#include "stdafx.h"
#include "NvrMaster.h"
#include"VasGpuBoost.h"

/*-------------------------Nvr call back-----------------------------*/

Mat g_frame[MAX_CAM_NUM];
char *gbuf_[MAX_CAM_NUM] = {0};
bool b_can_copy[MAX_CAM_NUM] = { true };

uchar* table_r = 0;
uchar* table_g = 0;
uchar* table_b = 0;

void InitColorTable()
{
	table_r = new uchar[256 * 256];
	table_g = new uchar[256 * 256 * 256];
	table_b = new uchar[256 * 256];

	for (int i = 0; i < 256; ++i)
	{
		for (int j = 0; j < 256; ++j)
		{
			int cc = i + 1.13983 * (j - 128);
			table_r[(i << 8) + j] = min(max(cc, 0), 255);
		}
	}

	for (int i = 0; i < 256; ++i)
	{
		for (int j = 0; j < 256; ++j)
		{
			int cc = i + 2.03211 * (j - 128);
			table_b[(i << 8) + j] = min(max(cc, 0), 255);
		}
	}

	for (int i = 0; i < 256; ++i)
	{
		for (int j = 0; j < 256; ++j)
			for (int k = 0; k < 256; ++k)
			{
				int cc = i - 0.39465 * (j - 128) - 0.58060 * (k - 128);
				table_g[(i << 16) + (j << 8) + k] = min(max(cc, 0), 255);
			}
	}
}


void CALLBACK decFunction(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, long nReserved1, long nReserved2)//解码回调  
{
	if (nPort<MAX_CAM_NUM)
	{
		if (g_frame[nPort].empty())
		{
			g_frame[nPort] = cv::Mat::zeros(pFrameInfo->nHeight, pFrameInfo->nWidth, CV_8UC3);
			gbuf_[nPort] = new char[nSize];
		}

		if (b_can_copy[nPort])
		{
			//CopyRawImage((unsigned char*)pBuf, 1, nSize);
			b_can_copy[nPort] = false;
			memcpy(gbuf_[nPort], pBuf, nSize);
		}
	}
}

void CALLBACK g_RealDataCallBack_V30(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
{
	BOOL iRet = FALSE;
	//get player port to protect the process safe  
	NvrVideoStream *nvr_video_stream = (NvrVideoStream *)pUser;
	LONG LPort = (LONG)nvr_video_stream->cam_id_;

	switch (dwDataType)
	{
	case NET_DVR_SYSHEAD://coming the stream header, open stream  
		//soft decode  
		if (LPort == -1)
		{
			if (!PlayM4_GetPort(&LPort))
			{
				return;
			}
		}

		if (dwBufSize > 0)
		{
			//set as stream mode, real-time stream under preview  
			if (!PlayM4_SetStreamOpenMode(LPort, STREAME_REALTIME))
			{
				//TRACE("Set StreamMode Error!");
				LOG(ERROR) << "Set StreamMode Error!" << endl;
				return;
			}
			//start player  
			if (!PlayM4_OpenStream(LPort, pBuffer, dwBufSize, 2 * 1024 * 1024))
			{
				//TRACE("PlayM4_OpenStream err");
				LOG(ERROR) << "PlayM4_OpenStream err!" << endl;
				return;
			}
			if (!PlayM4_SetDecCallBack(LPort, decFunction))
			{
				LOG(ERROR) << "PlayM4_SetDecCallBack err!" << endl;
				return;
			}
			if (!PlayM4_SetDisplayBuf(LPort, 6))
			{
				//TRACE("PlayM4_SetDisplayBuf err");
				LOG(ERROR) << "PlayM4_SetDisplayBuf err!" << endl;
				return;
			}
			//start play, set play window  
			if (!PlayM4_Play(LPort, NULL))
			{
				//TRACE("PlayM4_Play err");
				LOG(ERROR) << "PlayM4_Play err!" << endl;
				break;
			}


		}
		break;

	case NET_DVR_STREAMDATA:

		//start soft decode  

		if (dwBufSize > 0 && LPort != -1)
		{
			if (!PlayM4_InputData(LPort, pBuffer, dwBufSize))
			{
				//TRACE("PlayM4_InputData err");
				LOG(ERROR) << "PlayM4_InputData err!" << endl;
			}

		}
		break;
		//case NET_DVR_REALPLAYNETCLOSE:  
	case EXCEPTION_PREVIEW:
		break;

	default:
		break;
	}

	return;
}

void ConvertYUV422pToRGB24(const uchar *src, uchar *dst, const int width, const int height)
{
	int size = width*height;

	int tw = width / 2 + height % 2;
	//int cr, cg, cb;
	int r, c;

	uchar* y = (uchar*)src;
	uchar* u = y + size;
	uchar* v = y + size + ((width / 2 + width % 2) * (height / 2 + height % 2));

	for (r = 0; r < height; ++r)
	{
		for (c = 0; c < width; ++c)
		{
			*dst++ = table_r[((int)*y << 8) + *v];
			*dst++ = table_g[((int)*y << 16) + ((int)*u << 8) + *v];
			*dst++ = table_b[((int)*y << 8) + *u];
			if (c % 2 == 1)
			{
				++u;
				++v;
			}
			++y;
		}

		if (r % 2 == 0)
		{
			u -= tw;
			v -= tw;
		}
	}

}
/*----------------------------------NVR Channel----------------------------------------------*/
NvrVideoStream::NvrVideoStream(const vas::NvrChannel &param)
{
	channel_param_.CopyFrom(param);
	CheckParameter(param);
	frame_queue_.clear();
	service_id_ = channel_param_.service_id();
	play_handle_ = -1;
	memset(&preview_info_, 0, sizeof(NET_DVR_PREVIEWINFO));
}

NvrVideoStream::NvrVideoStream(const char *proto_file_name)
{
	CHECK(vas::ReadProtoFromTextFile(proto_file_name, &channel_param_));
	CheckParameter(channel_param_);
	frame_queue_.clear();
	service_id_ = channel_param_.service_id();
	play_handle_ = -1;
	memset(&preview_info_, 0, sizeof(NET_DVR_PREVIEWINFO));

}
NvrVideoStream::~NvrVideoStream()
{
	channel_param_.Clear();
	frame_queue_.clear();
	play_handle_ = -1;
}


void NvrVideoStream::CheckParameter(const vas::NvrChannel &param)
{

}


/*---------------------------------------NvrMaster--------------------------------*/
NvrMaster::NvrMaster(const string &ip, const string &user_name, const string &password, const int64_t &port)
{
	nvr_param_.set_user_name(user_name);
	nvr_param_.set_password(password);
	nvr_param_.set_ip(ip);
	nvr_param_.set_port(port);
	CheckNvrParam(nvr_param_);
	NET_DVR_Init();
	NET_DVR_SetConnectTime(2000, 1);
	NET_DVR_SetReconnect(10000, true);
	InitColorTable();
	Login();
}
NvrMaster::NvrMaster(const char*nvr_param_proto_file)
{
	CHECK(vas::ReadProtoFromTextFile(nvr_param_proto_file, &nvr_param_));
	CheckNvrParam(nvr_param_);
	NET_DVR_Init();
	NET_DVR_SetConnectTime(2000, 1);
	NET_DVR_SetReconnect(10000, true);
	InitColorTable();
	Login();
	
}
NvrMaster::NvrMaster(const vas::NvrParameter&param)
{
	nvr_param_.CopyFrom(param);
	NET_DVR_Init();
	NET_DVR_Init();
	NET_DVR_SetConnectTime(2000, 1);
	NET_DVR_SetReconnect(10000, true);
	InitColorTable();
	Login();
}

void NvrMaster::CheckNvrParam(const vas::NvrParameter &param)
{

}


NvrMaster::~NvrMaster()
{
	if (login_)
	{
		Logout();
	}
	
	NET_DVR_Cleanup();
}

bool NvrMaster::GetDecodedFrame(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)
{
	shared_ptr<NvrVideoStream> ptr_video_stream = GetNvrVideoStreamPtr(service_id);
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

//void NvrMaster::SetServiceCommandReplyProcessFunction(signal_service_command_reply::slot_function_type fun)
//{
//	service_command_reply_signal_.connect(fun);
//}
void NvrMaster::SetStreamStatusErrorFunction(cam_signal_t::slot_function_type fun)
{
	sig_connect_ = sig_.connect(fun);
}

bool NvrMaster::AddNvrVideoStream(const vas::NvrChannel &channel_param)
{
	if (login_)
	{
		shared_ptr<NvrVideoStream> ptr_nvr_stream = shared_ptr<NvrVideoStream>(new NvrVideoStream(channel_param));
		ptr_nvr_stream->cam_id_ = channel_param.channel_no();
		ptr_nvr_stream->service_id_ = channel_param.service_id();
		ptr_nvr_stream->preview_info_.byPreviewMode = 0;
		ptr_nvr_stream->preview_info_.bBlocked = 1;
		ptr_nvr_stream->preview_info_.hPlayWnd = NULL;
		ptr_nvr_stream->preview_info_.dwLinkMode = 0;
		ptr_nvr_stream->preview_info_.lChannel = nvr_device_info_.byStartDChan + channel_param.channel_no();

		ptr_nvr_stream->play_handle_ = NET_DVR_RealPlay_V40(nvr_user_id_, &ptr_nvr_stream->preview_info_, g_RealDataCallBack_V30, ptr_nvr_stream.get());

		if (ptr_nvr_stream->play_handle_<0)
		{
			LOG(ERROR) << "NET_DVR_RealPlay_V40 error" << endl;
			return false;
		}

		nvr_stream_list_.push_back(ptr_nvr_stream);

		return true;
	}
	else return false;
}

bool NvrMaster::DeleteNvrVideoStream(const char* service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<NvrVideoStream>>::iterator iter;
	for (iter = nvr_stream_list_.begin(); iter != nvr_stream_list_.end(); iter++)
	{
		string t_service_id = (*iter)->service_id_;
		if (t_service_id.compare(String(service_id)) == 0)
		{
			NET_DVR_StopRealPlay((*iter)->play_handle_);
			lio::BoostSleep(500);
			(*iter).reset();
			nvr_stream_list_.erase(iter);
			LOG(INFO) << service_id << " delete!" << endl;
			lock.unlock();
			return true;
		}
	}
	lock.unlock();
	LOG(INFO) << " No service named " << service_id << " exist!" << endl;
	return false;
}
bool NvrMaster::DeleteNvrVideoStream(const string &service_id)
{
	return DeleteNvrVideoStream(service_id.c_str());
}

void NvrMaster::CheckFun()
{
	while (true)
	{
		if (nvr_stream_list_.size()>0)
		{
			NET_DVR_IPPARACFG_V40 IPAccessCfgV40;
			DWORD dwReturned;
			NET_DVR_GetDVRConfig(nvr_user_id_, NET_DVR_GET_IPPARACFG_V40, 0, &IPAccessCfgV40, sizeof(NET_DVR_IPPARACFG_V40), &dwReturned);
			for (size_t i = 0; i < nvr_stream_list_.size(); i++)
			{
				shared_ptr<NvrVideoStream> ptr_video_stream = nvr_stream_list_[i];
				int channel_no = ptr_video_stream->channel_param_.channel_no();
				if (IPAccessCfgV40.struIPDevInfo[channel_no].byEnable == 1 && IPAccessCfgV40.struStreamMode[channel_no].uGetStream.struChanInfo.byEnable==1)
				{
					ptr_video_stream->channel_param_.set_cam_status(vas::CamStatus::CAM_ONLINE);
				}
				else
				{
					ptr_video_stream->channel_param_.set_cam_status(vas::CamStatus::CAM_OFFLINE);
				}
				//通知外部当前相机状态
				sig_(ptr_video_stream->service_id_.c_str(), ptr_video_stream->channel_param_.cam_status());
			}
			boost::this_thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(500));

		}
		lio::BoostSleep(5000);
	}
}

void NvrMaster::FrameProcessFun()
{
	while (true)
	{
		for (size_t i = 0; i < MAX_CAM_NUM&&i<nvr_stream_list_.size(); i++)
		{
			if (!b_can_copy[i]&&gbuf_[i]!= NULL)
			{
				if (nvr_stream_list_[i]->frame_queue_.size()<10)
				{
					uchar *image_data = (uchar*)g_frame[i].data;
					VasGpuBoost::ColorConvert::Get()->ConvertYUV422pToRGB24((uchar*)gbuf_[i], image_data, g_frame[i].cols, g_frame[i].rows);
					//ConvertYUV422pToRGB24((uchar*)gbuf_[i], image_data, g_frame[i].cols, g_frame[i].rows);
					Mat in_frame;
					g_frame[i].copyTo(in_frame);
					Size dst_size = Size(nvr_stream_list_[i]->channel_param_.dst_width(), nvr_stream_list_[i]->channel_param_.dst_height());
					DecodedFrame decode_frame(nvr_stream_list_[i]->channel_param_.dst_width(), nvr_stream_list_[i]->channel_param_.dst_height());
					resize(in_frame, decode_frame.frame, dst_size);
					decode_frame.No = 0;
					decode_frame.pts = 0;
					nvr_stream_list_[i]->frame_queue_.push(decode_frame);
					b_can_copy[i] = true;
				}
			
			}
			lio::BoostSleep(5);
			
		}
		lio::BoostSleep(20);
	}
}

shared_ptr<NvrVideoStream> NvrMaster::GetNvrVideoStreamPtr(const char* service_id)
{
	boost::mutex::scoped_lock lock(mutex_);
	vector<shared_ptr<NvrVideoStream>>::iterator iter;
	for (iter = nvr_stream_list_.begin(); iter != nvr_stream_list_.end(); iter++)
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

void NvrMaster::Login()
{
	if (!login_)
	{
		nvr_user_id_ = NET_DVR_Login_V30((char*)nvr_param_.ip().c_str(), nvr_param_.port(), (char*)nvr_param_.user_name().c_str(), (char*)nvr_param_.password().c_str(), &nvr_device_info_);
		if (nvr_user_id_<0)
		{
			LONG error_code;
			LOG(ERROR)<<NET_DVR_GetErrorMsg(&error_code)<<endl;
		}
		else
		{
			login_ = true;
			frame_process_thread_ = boost::thread(&NvrMaster::FrameProcessFun, this);
			status_check_thread_ = boost::thread(&NvrMaster::CheckFun, this);
		}
	}

}
void NvrMaster::Logout()
{
	if (login_)
	{
		NET_DVR_Logout(nvr_user_id_);
		frame_process_thread_.interrupt();
		status_check_thread_.interrupt();
		login_ = false;
	}
}