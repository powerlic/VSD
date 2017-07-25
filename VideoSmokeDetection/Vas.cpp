#include "stdafx.h"
#include "Vas.h"


Vas::Vas()
{
	ptr_decode_master = shared_ptr<VasDecodeMasterPro>(new VasDecodeMasterPro);
	ptr_detecter_master = shared_ptr<VasDetecterMasterPro>(new VasDetecterMasterPro);

	InitCaffeModel("new_set1.ini");
}


Vas::~Vas()
{

}

void Vas::test()
{
	const char *video_path_1 = "C:\\Users\\Administrator\\Desktop\\208\\CH0005\\CH0001T20170502153054942.mp4";
	//const char *video_path_2 = "E:\\Video\\yangpu.mp4";
	//const char *video_path_3 = "E:\\Video\\\smoke\\\down_2017-01-04\\3.mp4";
	//const char *video_path_4 = "E:\\Video\\\smoke\\\down_2017-01-04\\5.mp4";
	//const char *video_path_1 = "G:\\VideoCapture\\chizhou\\smoke\\5.mp4";
	//const char *video_path_2 = "G:\\VideoCapture\\chizhou\\smoke\\6.mp4";
	//const char *video_path_3 = "G:\\VideoCapture\\dongzhi\\dongzhi_monitor\\chitouqiao.mp4";
	//const char *video_path_4 = "G:\\VideoCapture\\dongzhi\\dongzhi_monitor\\CVR1486003460_D3D0E326.mp4";
	const char *rstp_stream = "rtsp://admin:sfb@12345@61.189.189.109:554/h264/ch6/main/av_stream";

	const char *service_id_cpu_video_file_1 = "cpu_video_file_1";
	const char *service_id_cpu_video_file_2 = "cpu_video_file_2";
	const char *service_id_cpu_video_file_3 = "cpu_video_file_3";
	const char *service_id_cpu_video_file_4 = "cpu_video_file_4";
	const char *service_id_cpu_video_file_5 = "cpu_video_file_5";

	int _cpu = 0;
	int _gpu = 1;
	int _rtsp_stream = 0;
	int _video_file = 1;
	int w(800), h(576);

	/*bool iv1 = ptr_decode_master->InitVideoStream(service_id_cpu_video_file_1, video_path_1, _video_file, _cpu, w, h);
	bool iv2 = ptr_decode_master->InitVideoStream(service_id_cpu_video_file_2, video_path_2, _video_file, _cpu, w, h);
	bool iv3 = ptr_decode_master->InitVideoStream(service_id_cpu_video_file_3, video_path_3, _video_file, _cpu, w, h);
	bool iv4 = ptr_decode_master->InitVideoStream(service_id_cpu_video_file_4, video_path_4, _video_file, _cpu, w, h);
	bool id1 = ptr_detecter_master->InitDetecter(service_id_cpu_video_file_1, "new_set1.ini");
	bool id2 = ptr_detecter_master->InitDetecter(service_id_cpu_video_file_2, "new_set1.ini");
	bool id3 = ptr_detecter_master->InitDetecter(service_id_cpu_video_file_3, "new_set1.ini");
	bool id4 = ptr_detecter_master->InitDetecter(service_id_cpu_video_file_4, "new_set1.ini");*/

	//ptr_decode_master->ConnectSignal(boost::bind(&Vas::SlotProcessStreamStatus, this, _1,_2));

	ptr_detecter_master->SetCaffeRegFunction(std::bind(&Vas::Predict, this, std::placeholders::_1, std::placeholders::_2));

	ptr_detecter_master->SetGetFrameFunction(std::bind(&VasDecodeMasterPro::GetDecodedFrame, ptr_decode_master, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

	/*ptr_decode_master->Start();
	ptr_detecter_master->Start();*/


	int count = 0;
	Mat frame_1, frame_2, frame_3, frame_4, frame_5;
	int64_t pts_1, pts_2, pts_3, pts_4;
	int64_t No_1, No_2, No_3, No_4;
	vector<CvRect> rects_1, rects_2, rects_3, rects_4;
	vector<String> labels_1, labels_2, labels_3, labels_4;
	vector<float> probs_1, probs_2, probs_3, probs_4;
	while (count<500001)
	{
		count++;
		//cout << "reg " << count << endl;


		/*if (count==201)
		{
			ptr_decode_master->Pause(service_id_cpu_video_file_1);
			ptr_detecter_master->Pause(service_id_cpu_video_file_1);
		}
		if (count==401)
		{ 
			ptr_decode_master->Resume(service_id_cpu_video_file_1);
			ptr_detecter_master->Resume(service_id_cpu_video_file_1);
		}

		if (count==501)
		{
			ptr_decode_master->Stop();
			ptr_detecter_master->Stop();
		}

		if (count == 601)
		{
			ptr_decode_master->Start();
			ptr_detecter_master->Start();
		}*/
		
		if (count ==100)
		{

			ptr_decode_master->AddVideoStream(service_id_cpu_video_file_1, video_path_1, _video_file, _cpu, w, h);
			ptr_detecter_master->AddDetecter(service_id_cpu_video_file_1, "new_set1.ini");
			cout << endl;
			//ptr_detecter_master->SetCaffeRegFunction(std::bind(&Vas::Predict, this, std::placeholders::_1, std::placeholders::_2));
			//ptr_detecter_master->SetGetFrameFunction()
			//ptr_detecter_master->SetGetFrameFunction(std::bind(&VasDecodeMasterPro::GetDecodedFrame, ptr_decode_master, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
			

		}

		//if (count %600==0)
		//{
		//	ptr_detecter_master->DeleteDetecter(service_id_cpu_video_file_5);
		//	ptr_decode_master->DeleteVideoStream(service_id_cpu_video_file_5);
		//	//cout << endl;
		//	
		//	//ptr_detecter_master->SetCaffeRegFunction(std::bind(&Vas::Predict, this, std::placeholders::_1, std::placeholders::_2));
		//	//ptr_detecter_master->SetGetFrameFunction()
		//	//ptr_detecter_master->SetGetFrameFunction(std::bind(&VasDecodeMasterPro::GetDecodedFrame, ptr_decode_master, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
		//	//ptr_detecter_master->SetCaffeRegFunction(service_id_cpu_video_file_5, std::bind(&Vas::Predict, this, std::placeholders::_1, std::placeholders::_2));

		//}
		

		boost::this_thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(20));




		/*ptr_detecter_master->GetRegResultShowFrame(service_id_cpu_video_file_1, frame_1);
		ptr_detecter_master->GetRegResultShowFrame(service_id_cpu_video_file_2, frame_2);
		ptr_detecter_master->GetRegResultShowFrame(service_id_cpu_video_file_3, frame_3);
		ptr_detecter_master->GetRegResultShowFrame(service_id_cpu_video_file_4, frame_4);*/


		ptr_detecter_master->GetRegResultShowFrame(service_id_cpu_video_file_1, frame_1);

		if (frame_1.data)
		{
			imshow(service_id_cpu_video_file_1, frame_1);
		}
		if (frame_2.data)
		{
			imshow(service_id_cpu_video_file_2, frame_2);
		}
		if (frame_3.data)
		{
			imshow(service_id_cpu_video_file_3, frame_3);

		}
		if (frame_4.data)
		{
			imshow(service_id_cpu_video_file_4, frame_4);
		}

		if (frame_5.data)
		{
			imshow(service_id_cpu_video_file_5, frame_5);
		}

		//destroyAllWindows();
		
	
		
		waitKey(50);
	}
	_CrtDumpMemoryLeaks();
	system("pause");
}
void Vas::InitCaffeModel(const char *file_name)
{
	string deploy_file_path = ZIni::readString("CaffeModel", "deploy_file", "null", file_name);
	string trained_file_path = ZIni::readString("CaffeModel", "trained_file", "null", file_name);
	string mean_file_path = ZIni::readString("CaffeModel", "mean_file", "null", file_name);
	string label_file_path = ZIni::readString("CaffeModel", "label_file", "null", file_name);

	int set_reg_width = ZIni::readInt("CaffeModel", "reg_width", 0, file_name);
	int set_reg_height = ZIni::readInt("CaffeModel", "reg_height", 0, file_name);
	int is_use_gpu = ZIni::readInt("CaffeModel", "is_use_gpu", 0, file_name);

	if (deploy_file_path.compare("null") == 0 ||
		mean_file_path.compare("null") == 0 ||
		label_file_path.compare("null") == 0 ||
		trained_file_path.compare("null") == 0)
	{
		cout << __FUNCTION__<<": Caffemodel Path is Not Right" << endl;
		exit(1);
	}

	ptr_classifier = shared_ptr<Classifier>(CreateClassifier(deploy_file_path, trained_file_path, mean_file_path, label_file_path, is_use_gpu));

	if (!ptr_classifier)
	{
		cout << "[DetectMaster.cpp] InitCaffeModel: Caffemodel Init Failed" << endl;
		exit(1);
	}


}

vector<Prediction> Vas::Predict(const Mat &image_mat, const int &N)
{
	boost::mutex::scoped_lock lock(caffe_mutex);
	std::vector<Prediction> predictions;
	predictions = ptr_classifier->Classify(image_mat, N);
	return predictions;
}