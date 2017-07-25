#include "stdafx.h"
#include "Test.h"


#ifdef _DEBUG
#pragma comment(lib,"caffe-classifier_d.lib")
#else 
#pragma comment(lib,"caffe-classifier.lib")
#endif




boost::mutex caffe_mutex;
shared_ptr<Classifier> ptr_classifier=NULL;
concurrent_queue<Mat> frame_queue;



void TestBgVibeReadFromTxtFile()
{
	cout << "-----TestBgVibeReadFromTxtFile-----" << endl;
	vas::BgParameter bg_paramter;
	vas::ReadProtoFromTextFile("..\\proto\\bgVibe.prototxt", &bg_paramter);

	cout <<"bg_height "<<bg_paramter.bg_height() << endl;
	cout << "bg_width" << bg_paramter.bg_width() << endl;
	cout << "stable_threshold " << bg_paramter.stable_threshold() << endl;
	cout << "morphology_open_times " << bg_paramter.bg_operation().morphology_open_times() << endl;
	cout << "dilate_times " << bg_paramter.bg_operation().dilate_times() << endl;

	cout << "vibe_parameter" << endl;
	cout << " num_sample " << bg_paramter.vibe_parameter().num_samples() << endl;
	cout << " min_match " << bg_paramter.vibe_parameter().min_match() << endl;
	cout << " radius " << bg_paramter.vibe_parameter().radius() << endl;
	cout << " subsample_factor " << bg_paramter.vibe_parameter().subsample_factor() << endl;
	cout << " bg2_delay " << bg_paramter.vibe_parameter().bg2_delay() << endl;
	cout << " max_mismatch_count " << bg_paramter.vibe_parameter().max_mismatch_count() << endl;
	cout << endl;
}
void TestBgGuassianReadFromTxtFile()
{
	cout << "-----TestBgGuassianReadFromTxtFile-----" << endl;
	vas::BgParameter bg_paramter;
	vas::ReadProtoFromTextFile("..\\proto\\BgGuassian.prototxt", &bg_paramter);

	cout << "use_gpu " << bg_paramter.bg_method() << endl;
	cout << "bg_height " << bg_paramter.bg_height() << endl;
	cout << "bg_width " << bg_paramter.bg_width() << endl;
	cout << "stable_threshold " << bg_paramter.stable_threshold() << endl;
	cout << "morphology_open_times " << bg_paramter.bg_operation().morphology_open_times() << endl;
	cout << "dilate_times " << bg_paramter.bg_operation().dilate_times() << endl;


	cout << "guassian_parameter" << endl;
	cout << " learn_rate " << bg_paramter.guassian_parameter().learn_rate() << endl;
	cout << " num_history " << bg_paramter.guassian_parameter().num_history() << endl;
	cout << " shadow_detection " << bg_paramter.guassian_parameter().shadow_detection() << endl;
	cout << " var_threshold " << bg_paramter.guassian_parameter().var_threshold() << endl;
	cout << endl;
}


void TestBg()
{

}
void TestOpencvReadVideo()
{
	const string path = "F:\\VideoCapture\\chizhou\\smoke\\1.mp4";
	cv::VideoCapture capture(path);
	CHECK(!capture.isOpened()) << path << " does not exist!" << endl;
	Mat frame;
	while (!true)
	{
		if (!capture.read(frame))
			break;
		cv::imshow("show_frame", frame);
		
		waitKey(10);
	}
}
void TestBgGuassian()
{
	const string path = "F:\\VideoCapture\\chizhou\\smoke\\1.mp4";
	cv::VideoCapture capture(path);
	CHECK(capture.isOpened()) << path << " does not exist!" << endl;

	shared_ptr<Background> ptr_bg = shared_ptr<BgGuassian>(new BgGuassian("..\\proto\\bgGuassian.prototxt"));

	Mat frame;
	while (true)
	{
		if (!capture.read(frame))
			break;
		
		resize(frame, frame, cvSize(800, 576));
		ptr_bg->Update(frame);
		if (ptr_bg->BgStatus()==vas::BG_UPDATING)
		{
			imshow("mask",ptr_bg->Mask());
		}
		cv::imshow("show_frame", frame);
		waitKey(10);
	}
}
void TestFilter()
{
	const string path = "F:\\VideoCapture\\chizhou\\smoke\\1.mp4";
	cv::VideoCapture capture(path);
	CHECK(capture.isOpened()) << path << " does not exist!" << endl;

	shared_ptr<Filter> ptr_filter = shared_ptr<Filter>(new Filter("..\\proto\\filter.prototxt"));
	shared_ptr<Background> ptr_bg = shared_ptr<BgVibe>(new BgVibe("..\\proto\\bgVibe.prototxt"));

	Mat frame;
	while (true)
	{
		if (!capture.read(frame))
			break;

		resize(frame, frame, cvSize(800, 576));

		ptr_bg->Update(frame);
		if (ptr_bg->BgStatus() == vas::BG_UPDATING)
		{
			Mat filter_mask;
			vector<vector<Point>> contours;
			vector<CvRect> rects;
			ptr_filter->Filtrate(frame, ptr_bg->Mask(), filter_mask, contours, rects);

			imshow("mask", ptr_bg->Mask());
			imshow("filter_mask", filter_mask);
			imshow("frame", frame);
			waitKey(10);
		}

	}
}
void TestBgVibe()
{
	const string path = "F:\\VideoCapture\\chizhou\\smoke\\1.mp4";
	cv::VideoCapture capture(path);
	CHECK(capture.isOpened()) << path << " does not exist!" << endl;

	shared_ptr<Background> ptr_bg = shared_ptr<BgVibe>(new BgVibe("..\\proto\\bgVibe.prototxt"));

	Mat frame;
	while (true)
	{
		if (!capture.read(frame))
			break;

		resize(frame, frame, cvSize(800, 576));
		ptr_bg->Update(frame);
		if (ptr_bg->BgStatus() == vas::BG_UPDATING)
		{
			imshow("mask", ptr_bg->Mask());
		}
		cv::imshow("show_frame", frame);
		waitKey(10);
	}
}

void TestDetectMaster()
{
	//const string path = "F:\\VideoCapture\\chizhou\\smoke\\6.mp4";
	const string path = "F:\\smokeVideo\\208\\IMG_7942.MP4";
	cv::VideoCapture capture(path);
	CHECK(capture.isOpened()) << path << " does not exist!" << endl;

	std::function<vector<Prediction>(const Mat&, const int&)> caffe_reg_fun = std::bind(&Predict, std::placeholders::_1, std::placeholders::_2);
	std::function<bool(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)> get_frame_fun = std::bind(&TestGetFrame, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);


	shared_ptr<VasDetectMaster> ptr_detect_master = shared_ptr<VasDetectMaster>(new VasDetectMaster(caffe_reg_fun, get_frame_fun));
	

	ptr_detect_master->AddDetecter("..\\proto\\detecter.prototxt");

	Mat frame;
	while (true)
	{
		if (!capture.read(frame))
		{ 
			break;
			ptr_detect_master->DeleteDetecter("smoke_detect_instance_1");
		}

		resize(frame, frame, cvSize(800, 576));
		if (frame_queue.size()<10)
		{
			frame_queue.push(frame);
		}
		waitKey(10);
	}

}


void RegResultProcessFun(const char* service_id, const vector<CvRect> &reg_rects,
	                     const uint32_t dst_width, const uint32_t dst_height,
	                     const vector<float> &scores, const vector<string> &labels)
{
	LOG(INFO) << "---------service_id " << service_id << " reg_results------" << endl;
	for (size_t i = 0; i < reg_rects.size(); i++)
	{
		LOG(INFO) << "rect " << i+1 << " x:" << reg_rects[i].x << " y:" << reg_rects[i].y<< " with:" << reg_rects[i].width<< " height:" << reg_rects[i].height << endl;
		LOG(INFO) << "socre " << scores[i] << endl;
		LOG(INFO) << "labels " << labels[i] << endl;
	}
}
void ServiceCommandReplyProcessFun(const char* service_id, const vas::ServiceCommandReply &reply, const string &info)
{
	LOG(INFO) << service_id << "" << " info: "<<info << endl;
}
void StreamErrProcessFun(const char *service_id, vas::StreamStatus status)
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
		break;
	case vas::STREAM_STOP:
		break;
	case vas::STREAM_PAUSE:
		break;
	case vas::STREAM_NETWORK_FAULT:
		cout << service_id << " stream network fault" << endl;
		break;
	case vas::FILE_FAULT:
		cout << service_id << " file fault" << endl;
		break;
	case vas::STREAM_CPU_DECODE_FAULT:
		break;
	case vas::STREAM_GPU_DECODE_FAULT:
		break;
	default:
		break;
	}
}
void TestDecodeMaster()
{
	shared_ptr<VasDecodeMaster> ptr_decode_master = shared_ptr<VasDecodeMaster>(new VasDecodeMaster);

	vas::DecodeParameter param_1, param_2;
	//vas::ReadProtoFromTextFile("..\\proto\\decoder_rtsp.prototxt", &param_1);
	vas::ReadProtoFromTextFile("..\\proto\\decoder_video_file.prototxt", &param_2);

	string service_id_1 = param_1.service_id();
	string service_id_2 = param_2.service_id();

	//ptr_decode_master->AddVideoStream("..\\proto\\decoder_rtsp.prototxt");
	ptr_decode_master->AddVideoStream("..\\proto\\decoder_video_file.prototxt");
	ptr_decode_master->SetStreamStatusErrorFunction(boost::bind(StreamErrProcessFun, _1, _2));

	while (true)
	{
		Mat frame_1, frame_2;
		int64_t pts_1, pts_2, No_1, No_2;
	/*	if (ptr_decode_master->GetDecodedFrame(service_id_1, frame_1, pts_1, No_1));
		{
			if (!frame_1.empty())
			{
				imshow(service_id_1, frame_1);
			}
			
		}*/
		if (ptr_decode_master->GetDecodedFrame(service_id_2.c_str(), frame_2,pts_2, No_2)&&!frame_2.empty());
		{
			if (!frame_2.empty())
			{
				imshow(service_id_2, frame_2);
			}
		}
		waitKey(20);
	}
}

void TestVas()
{
	shared_ptr<VasService> ptr_vas = shared_ptr<VasService>(new VasService);
	//ptr_vas->SetRegResultProcessFunction(boost::bind(&RegResultProcessFun, _1, _2, _3, _4));
	ptr_vas->SetStreamStatusErrorFunction(boost::bind(StreamErrProcessFun, _1, _2));
	ptr_vas->SetServiceCommandReplyProcessFunction(boost::bind(ServiceCommandReplyProcessFun, _1, _2, _3));
	ptr_vas->AddService("..\\proto\\service_video_file.prototxt");

	vector<string> reg_types;
	reg_types.push_back("smoke");
	reg_types.push_back("human");
	reg_types.push_back("car");

	ptr_vas->AddService("service_id_2", "E:\\Video\\smoke\\down_2017-01-04\\1.mp4", 1, 1, 800, 600,reg_types);
	
	while (true)
	{
		lio::BoostSleep(1000);
	}
}
void TestVasServices()
{
	shared_ptr<VasService> ptr_vas = shared_ptr<VasService>(new VasService);
	int count = 0;
	while (true)
	{
		if (count % 100 == 0 && count % 200 != 0)
		{
			ptr_vas->StartServicesFromProtoFile("..\\proto\\service_list.prototxt");
		}
		if (count % 200 == 0)
		{
			ptr_vas->DeleteService();
		}

		//if (count == 0)
		/*{
			ptr_vas->StartServicesFromProtoFile("..\\proto\\service_list.prototxt");
		}*/

		count++;
		lio::BoostSleep(100);
	}

}

bool TestGetFrame(const char *service_id, Mat &frame, int64_t &pts, int64_t &No)
{
	if (frame_queue.size()>0)
	{
		frame = frame_queue.pop();
		pts = 0;
		No = 0;
		return true;
	}
	return false;
}
void TestMergeRects()
{
	vector<CvRect> rect_list;
	CvRect rect_1 = cvRect(0, 0, 20, 20);
	CvRect rect_2 = cvRect(19, 19, 20, 20);
	CvRect rect_3= cvRect(15, 15, 20, 20);

	CvRect rect_4 = cvRect(100, 100, 20, 20);
	CvRect rect_5 = cvRect(110, 110, 20, 20);
	CvRect rect_6 = cvRect(115, 115, 20, 20);
	CvRect rect_7 = cvRect(120, 130, 20, 20);

	CvRect rect_8 = cvRect(200, 200, 20, 20);
	CvRect rect_9 = cvRect(10, 10, 20, 20);
	rect_list.push_back(rect_1);
	rect_list.push_back(rect_2);
	rect_list.push_back(rect_3);
	rect_list.push_back(rect_4);
	rect_list.push_back(rect_5);
	rect_list.push_back(rect_6);
	rect_list.push_back(rect_7);
	rect_list.push_back(rect_8);
	rect_list.push_back(rect_9);

	map<CvRect, set<int>> rect_map;

	vector<CvRect> d_list;
	vector<set<int>> d_rect_number_set;


	lio::MergeRects(rect_list, d_list, d_rect_number_set, 500, 500);
}


vector<Prediction> Predict(const Mat &image_mat, const int &N)
{
	if (!ptr_classifier)
	{
		string deploy_file_path="..\\caffemodel\\yangpu_0427\\yangpu_0427_googlenet_16_b.prototxt";
		string trained_file_path="..\\caffemodel\\yangpu_0427\\b_bvlc_googlenet_smoke_12_iter_80000.caffemodel";
		string mean_file_path="..\\caffemodel\\yangpu_0427\\yangpu_0427_googlenet_16_b.binaryproto";
		string label_file_path="..\\caffemodel\\yangpu_0427\\yangpu_0427_googlenet_16_b.meta.txt";
		bool is_use_gpu=true;
		ptr_classifier = shared_ptr<Classifier>(CreateClassifier(deploy_file_path, trained_file_path, mean_file_path, label_file_path, is_use_gpu));
	}

	boost::mutex::scoped_lock lock(caffe_mutex);
	std::vector<Prediction> predictions;
	predictions = ptr_classifier->Classify(image_mat, N);
	return predictions;
}

