#include "stdafx.h"
#include "VsdMaster.h"


VsdMaster::VsdMaster(const char*caffe_model_path_proto_file)
{
	ptr_vas = shared_ptr<VasP>(CreateVasP(caffe_model_path_proto_file));
	ptr_vas->SetRegResultProcessFunction(boost::bind(&VsdMaster::RegResultProcessFun, this, _1, _2, _3, _4, _5, _6));
	ptr_vas->SetServiceCommandReplyProcessFunction(boost::bind(&VsdMaster::ServiceCommandReplyProcessFun, this, _1, _2, _3));
	ptr_vas->SetStreamStatusErrorFunction(boost::bind(&VsdMaster::StreamErrProcessFun, this, _1, _2));
}


VsdMaster::~VsdMaster()
{
}


/*
�����������ʵ�ֽ�������Ϣ�ŵ���Ϣ������ȥ
*/
void VsdMaster::RegResultProcessFun(const char* service_id, const vector<CvRect> &reg_rects,
	const uint32_t dst_width, const uint32_t dst_height,
	const vector<float> &scores, const vector<string> &labels)
{
	LOG(INFO) << "---------service_id " << service_id << " reg_results------" << endl;
	for (size_t i = 0; i < reg_rects.size(); i++)
	{
		LOG(INFO) << "rect " << i + 1 << " x:" << reg_rects[i].x << " y:" << reg_rects[i].y << " with:" << reg_rects[i].width << " height:" << reg_rects[i].height << endl;
		LOG(INFO) << "socre " << scores[i] << endl;
		LOG(INFO) << "labels " << labels[i] << endl;
	}
}

void VsdMaster::CheckSmokeRegFun(const char* service_id, const vector<CvRect> &reg_rects,
	const uint32_t dst_width, const uint32_t dst_height,
	const vector<float> &scores, const vector<string> &labels)
{
	LOG(INFO) << "---------service_id " << service_id << " reg_results------" << endl;
	for (size_t i = 0; i < reg_rects.size(); i++)
	{
		LOG(INFO) << "rect " << i + 1 << " x:" << reg_rects[i].x << " y:" << reg_rects[i].y << " with:" << reg_rects[i].width << " height:" << reg_rects[i].height << endl;
		LOG(INFO) << "socre " << scores[i] << endl;
		LOG(INFO) << "labels " << labels[i] << endl;

		if (labels[i].compare("smoke") == 0)
		{
			LOG(INFO) << " has detect out smoke!" << endl;
			ptr_vas->DeleteService(service_id);
			break;
		}
	}
}

/*
�����������Է��������������ɾ��ָ����յ��ķ�����Ϣ
*/
void VsdMaster::ServiceCommandReplyProcessFun(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)
{
	LOG(INFO) << service_id << "" << " info: " << info << endl;
}

/*
��������������Ĵ�����Ϣ�����磬���жϺ󣬿��Թر�һ�����񣬽���֪ͨ�������ȹرգ�Ȼ��������
*/
void VsdMaster::StreamErrProcessFun(const char *service_id, vas::StreamStatus status)
{
	switch (status)
	{
	case vas::STREAM_UNKNOWN:
		break;
	case vas::STREAM_CONNECTING:
		break;
	case vas::STREAM_NORMAL:
		break;
	case vas::STREAM_FINISH:
		cout << service_id << " stream finish" << endl;
		ptr_vas->DeleteService(service_id);
		break;
	case vas::STREAM_STOP:
		break;
	case vas::STREAM_PAUSE:
		break;
	case vas::STREAM_NETWORK_FAULT:
		cout << service_id << " stream network fault" << endl;
		ptr_vas->DeleteService(service_id);
		break;
	case vas::FILE_FAULT:
		cout << service_id << " file fault" << endl;
		ptr_vas->DeleteService(service_id);
		break;
	case vas::STREAM_CPU_DECODE_FAULT:
		cout << service_id << " cpu decode fault" << endl;
		ptr_vas->DeleteService(service_id);
		break;
	case vas::STREAM_GPU_DECODE_FAULT:
		cout << service_id << " gpu decode fault" << endl;
		ptr_vas->DeleteService(service_id);
		break;
	default:
		break;
	}
}

/*------------------------------------------NVR-------------------------------------------------------*/
VsdNvrMaster::VsdNvrMaster(const char*caffe_model_path_proto_file, const char*nvr_access_proto_file)
{
	ptr_vas_nvr_ = shared_ptr<VasNvrP>(CreateVasNvrP(caffe_model_path_proto_file, nvr_access_proto_file));
	ptr_vas_nvr_->SetRegResultProcessFunction(boost::bind(&VsdNvrMaster::RegResultProcessFun, this, _1, _2, _3, _4, _5, _6));
	ptr_vas_nvr_->SetServiceCommandReplyProcessFunction(boost::bind(&VsdNvrMaster::ServiceCommandReplyProcessFun, this, _1, _2, _3));
	ptr_vas_nvr_->SetStreamStatusErrorFunction(boost::bind(&VsdNvrMaster::StreamErrProcessFun, this, _1, _2));
}
VsdNvrMaster::~VsdNvrMaster()
{

}

void VsdNvrMaster::RegResultProcessFun(const char* service_id, const vector<CvRect> &reg_rects,
	const uint32_t dst_width, const uint32_t dst_height,
	const vector<float> &scores, const vector<string> &labels)
{
	LOG(INFO) << "---------service_id " << service_id << " reg_results------" << endl;
	for (size_t i = 0; i < reg_rects.size(); i++)
	{
		LOG(INFO) << "rect " << i + 1 << " x:" << reg_rects[i].x << " y:" << reg_rects[i].y << " with:" << reg_rects[i].width << " height:" << reg_rects[i].height << endl;
		LOG(INFO) << "socre " << scores[i] << endl;
		LOG(INFO) << "labels " << labels[i] << endl;
	}
}

void VsdNvrMaster::ServiceCommandReplyProcessFun(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)
{
	LOG(INFO) << service_id << "" << " info: " << info << endl;
}
void VsdNvrMaster::StreamErrProcessFun(const char *service_id, vas::CamStatus status)
{
	switch (status)
	{
	case vas::CAM_ONLINE:
		break;
	case vas::CAM_OFFLINE:
		break;
	default:
		break;
	}
}