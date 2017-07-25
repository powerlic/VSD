#include "stdafx.h"
#include "BgVibeBoost.h"
#include "VasIO.h"

BgVibeBoost::BgVibeBoost()
{
}


BgVibeBoost::~BgVibeBoost()
{
	Reset();
	bg_parameter_.Clear();
}

void BgVibeBoost::CheckParameter(const vas::BgParameter &param)
{
	CHECK(param.bg_method() == vas::VIBE_CPU || param.bg_method() == vas::VIBE_GPU) << " BgMethod is not Vibe!" << endl;
	CHECK(param.has_bg_height() && param.has_bg_width()) << " BgVibe bg size is not set!" << endl;
	CHECK_GE(param.bg_height(), 300) << "Bg bg height must be greater than 300" << endl;
	CHECK_GE(param.bg_width(), 400) << "Bg bg width must be greater than 400" << endl;
	CHECK(param.has_stable_threshold()) << " BgVibe stable threshold is not set!" << endl;
	CHECK_GE(param.stable_threshold(), 0.2) << "BgVibe stable threshold must be greater than or equal to 0.2" << endl;
	CHECK(param.has_bg_operation()) << " BgVibe operation is not defined!" << endl;
	CHECK(param.has_vibe_parameter()) << " BgVibe parameter is not defined!" << endl;
	CHECK_LT(param.vibe_parameter().min_match(), 8) << " BgVibe min_match must be less than 8!" << endl;
	CHECK_GE(param.vibe_parameter().num_samples(), 10) << " BgVibe num_samples must be greater than or equal to 10" << endl;
	CHECK_GE(param.vibe_parameter().subsample_factor(), 10) << " BgVibe num_samples must be greater than or equal to 10" << endl;
	if (param.vibe_parameter().double_bg())
	{
		CHECK_GE(param.vibe_parameter().bg2_delay(), 50) << " BgVibe num_samples must be greater than or equal to 50" << endl;
	}
}

BgVibeBoost::BgVibeBoost(const char *file_name)
{
	bool success = vas::ReadProtoFromTextFile(file_name, &bg_parameter_);
	if (success)
	{
		bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);
	}

	CheckParameter(bg_parameter_);

	bg_setup1_ = false;
	bg_setup2_ = false;

	int c_yoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };
	memcpy(off_, c_yoff, sizeof(int) * 9);
}

BgVibeBoost::BgVibeBoost(   const CvSize &bg_size,
							const uint32_t morphology_open_times, const uint32_t dilate_times,
							const uint32_t num_samples, const uint32_t min_match,
							const uint32_t radius,
							const uint32_t sub_sample_factor,
							const bool use_gpu,
							const float stable_threahold,
							const uint32_t bg2_delay,
							const uint32_t max_mismatch_count,
							const bool double_bg)
{
	bg_parameter_ = vas::BgParameter();

	if (use_gpu) bg_parameter_.set_bg_method(vas::VIBE_GPU);
	else bg_parameter_.set_bg_method(vas::VIBE_CPU);

	bg_parameter_.set_stable_threshold(stable_threahold);
	bg_parameter_.set_bg_height(bg_size.height);
	bg_parameter_.set_bg_width(bg_size.width);

	bg_parameter_.mutable_bg_operation()->set_dilate_times(dilate_times);
	bg_parameter_.mutable_bg_operation()->set_morphology_open_times(morphology_open_times);

	bg_parameter_.mutable_vibe_parameter()->set_num_samples(num_samples);
	bg_parameter_.mutable_vibe_parameter()->set_min_match(min_match);
	bg_parameter_.mutable_vibe_parameter()->set_radius(radius);
	bg_parameter_.mutable_vibe_parameter()->set_subsample_factor(sub_sample_factor);
	bg_parameter_.mutable_vibe_parameter()->set_bg2_delay(bg2_delay);
	bg_parameter_.mutable_vibe_parameter()->set_max_mismatch_count(max_mismatch_count);
	bg_parameter_.mutable_vibe_parameter()->set_double_bg(double_bg);

	CheckParameter(bg_parameter_);

	bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);

	frame_count_ = 0;

	bg_setup1_ = false;
	bg_setup2_ = false;

	int c_yoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };
	memcpy(off_, c_yoff, sizeof(int) * 9);
}

BgVibeBoost::BgVibeBoost(const vas::BgParameter &other)
{
	bg_parameter_.CopyFrom(other);
	bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);

	CheckParameter(bg_parameter_);

	bg_setup1_ = false;
	bg_setup2_ = false;

	int c_yoff[9] = { -1, 0, 1, -1, 1, -1, 0, 1, 0 };
	memcpy(off_, c_yoff, sizeof(int) * 9);
}

void BgVibeBoost::Init()
{
	if (bg_parameter_.vibe_parameter().double_bg())
	{
		InitDouble();
	}
	else
	{
		InitSingle();
	}
}
void BgVibeBoost::InitSingle()
{
	Reset();
	CvSize bg_size = cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height());
	size_ = bg_size.height*bg_size.width;
	int num_sample = bg_parameter_.vibe_parameter().num_samples();

	if (bg_parameter_.bg_method()==vas::VIBE_CPU)
	{
		sample_list1_ = new uchar[num_sample*size_];
		memset(sample_list1_, 0, sizeof(uchar)*num_sample*size_);

		foreground_match_count_mat1_ = new uchar[size_];
		memset(foreground_match_count_mat1_, 0, sizeof(uchar)*size_);

		mask_ = Mat::zeros(bg_size, CV_8UC1);
	}
	else if (bg_parameter_.bg_method() == vas::VIBE_GPU)
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_sample_list1_, num_sample*size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_sample_list1_, 0, num_sample*size_ * sizeof(uchar)));

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_foreground_match_count_mat1_, size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_foreground_match_count_mat1_, 0, size_ * sizeof(uchar)));

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_mask1_, size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_mask1_, 0, size_ * sizeof(uchar)));

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_frame_, size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_frame_, 0, size_ * sizeof(uchar)));

		/*CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_rand_mat_, RandNum* size_ * sizeof(float)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_rand_mat_, 0, RandNum * size_ * sizeof(float)));

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_first_frame_rand_mat_, 2 * size_ * sizeof(float)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_first_frame_rand_mat_, 0, 2 * size_* num_sample* sizeof(float)));*/

		mask_ = Mat::zeros(bg_size, CV_8UC1);
	}
	else
	{
		LOG(ERROR) << " the bg method is not VIBE_CPU or VIBE_GPU" << endl;
	}
	frame_count_ = 0;
	bg_parameter_.set_bg_status(vas::BG_SETUP);
}
void BgVibeBoost::InitDouble()
{
	Reset();
	CvSize bg_size = cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height());
	size_ = bg_size.height*bg_size.width;
	int num_sample = bg_parameter_.vibe_parameter().num_samples();

	if (bg_parameter_.bg_method() == vas::VIBE_CPU)
	{	
		//fore 1
		sample_list1_ = new uchar[num_sample*size_];
		memset(sample_list1_, 0, sizeof(uchar)*num_sample*size_);

		foreground_match_count_mat1_ = new uchar[size_];
		memset(foreground_match_count_mat1_, 0, sizeof(uchar)*size_);

	
		//utified
		mask_  = Mat::zeros(bg_size, CV_8UC1);

		//fore 2
		if (bg_parameter_.vibe_parameter().double_bg())
		{
			sample_list2_ = new uchar[num_sample*size_];
			memset(sample_list2_, 0, sizeof(uchar)*num_sample*size_);

			foreground_match_count_mat2_ = new uchar[size_];
			memset(foreground_match_count_mat2_, 0, sizeof(uchar)*size_);

			mask1_ = Mat::zeros(bg_size, CV_8UC1);
			mask2_ = Mat::zeros(bg_size, CV_8UC1);
		}

		
	}
	else if (bg_parameter_.bg_method() == vas::VIBE_GPU)
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_sample_list1_, num_sample*size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_foreground_match_count_mat1_, size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_mask1_, size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_sample_list1_, 0, num_sample*size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_foreground_match_count_mat1_, 0, size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_mask1_, 0, size_ * sizeof(uchar)));

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_mask_, size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_mask_, 0, size_ * sizeof(uchar)));

		CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_frame_, size_ * sizeof(uchar)));
		CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_frame_, 0, size_ * sizeof(uchar)));

		mask_  = Mat::zeros(bg_size, CV_8UC1);

		if (bg_parameter_.vibe_parameter().double_bg())
		{
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_sample_list2_, num_sample*size_ * sizeof(uchar)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_foreground_match_count_mat2_, size_ * sizeof(uchar)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void**)&d_mask2_, size_ * sizeof(uchar)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_sample_list2_, 0, num_sample*size_ * sizeof(uchar)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_foreground_match_count_mat2_, 0, size_ * sizeof(uchar)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_mask2_, 0, size_ * sizeof(uchar)));

			mask1_ = Mat::zeros(bg_size, CV_8UC1);
			mask2_ = Mat::zeros(bg_size, CV_8UC1);
		}
		

	}
	else
	{
		LOG(ERROR) << " the bg method is not VIBE_CPU or VIBE_GPU" << endl;
	}

	frame_count_ = 0;
	bg_parameter_.set_bg_status(vas::BG_SETUP);

}

void BgVibeBoost::Reset()
{
	if (!mask_.empty())
	{
		mask_.release();
	}
	if (!mask1_.empty())
	{
		mask1_.release();
	}
	if (!mask2_.empty())
	{
		mask2_.release();
	}

	if (sample_list1_)
	{
		delete sample_list1_;
		sample_list1_ = NULL;
	}
	if (sample_list2_)
	{
		delete sample_list2_;
		sample_list2_ = NULL;
	}

	if (foreground_match_count_mat1_)
	{
		delete foreground_match_count_mat1_;
		foreground_match_count_mat1_ = NULL;
	}
	if (foreground_match_count_mat2_)
	{
		delete foreground_match_count_mat2_;
		foreground_match_count_mat2_ = NULL;
	}
	
	//GPU
	if (d_sample_list1_)
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_sample_list1_));
		d_sample_list1_ = NULL;
	}
	if (d_sample_list2_)
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_sample_list2_));
		d_sample_list2_ = NULL;
	}
	if (d_foreground_match_count_mat1_)
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_foreground_match_count_mat1_));
		d_foreground_match_count_mat1_ = NULL;
	}
	if (d_foreground_match_count_mat2_)
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_foreground_match_count_mat2_));
		d_foreground_match_count_mat2_ = NULL;
	}
	if (!d_mask1_)
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_mask1_));
		d_mask1_ = NULL;
	}
	if (!d_mask2_)
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_mask2_));
		d_mask2_ = NULL;
	}
	if (!d_mask_)
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_mask_));
		d_mask_ = NULL;
	}
	if (!d_frame_)
	{
		CUDA_SAFE_CALL_NO_SYNC(cudaFree(d_frame_));
		d_frame_ = NULL;
	}
}

void BgVibeBoost::Update(const Mat &gray_frame)
{
	if (bg_parameter_.vibe_parameter().double_bg())
	{
		UpdateDouble(gray_frame);
	}
	else
	{
		UpdateSingle(gray_frame);
	}
}
void BgVibeBoost::SetBgSize(const CvSize& value)
{
	CvSize bg_size = cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height());
	if (bg_size.width != value.width || bg_size.height != value.height)
	{
		bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);
		bg_parameter_.set_bg_height(value.height);
		bg_parameter_.set_bg_width(value.width);
		bg_setup1_ = false;
		bg_setup2_ = false;
	}
}

void BgVibeBoost::ProcessFirstFrameCPU(const Mat& gray_frame, uchar *sample_list)
{
	CvSize bg_size = cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height());
	RNG rng((unsigned)time(NULL));
	Mat resize_frame;
	int size = bg_size.height*bg_size.width;
	int num_samples = bg_parameter_.vibe_parameter().num_samples();
	CHECK(!gray_frame.empty()) << "Input frame is empty!" << endl;
	if (gray_frame.cols != bg_size.width || gray_frame.rows != bg_size.height)
	{
		resize(gray_frame, resize_frame, bg_size);
	}
	else resize_frame = gray_frame;

	for (int i = 0; i < bg_size.height; i++)
	for (int j = 0; j < bg_size.width; j++)
	{
			for (int k = 0; k < num_samples; k++)
			{
				int random = rng.uniform(0, 9);

				int row = i + off_[random];
				if (row < 0)
					row = 0;
				if (row >= bg_size.height)
					row = bg_size.height - 1;

				int col = j + off_[random];
				if (col < 0)
					col = 0;
				if (col >= bg_size.width)
					col = bg_size.width - 1;
				*(sample_list+ k*size + i*bg_size.width + j) = *(resize_frame.data + row*bg_size.width + col);
			}
	}
}

void BgVibeBoost::UpdateSamplist(uchar matches, uchar pixel_value, int x, int y, uchar *foreground_match_count_mat, uchar *mask, uchar *sample_list)
{
	RNG rng((unsigned)time(NULL));
	CvSize bg_size = cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height());
	int offset = x + y*bg_size.width;
	int size = bg_size.height*bg_size.width;
	if (matches >= bg_parameter_.vibe_parameter().min_match())
	{
		*(foreground_match_count_mat + offset) = 0;
		*(mask + offset) = 0;

		int int_rand_num = rng.uniform(0, bg_parameter_.vibe_parameter().subsample_factor());
		if (int_rand_num<2)
		{
			int_rand_num = rng.uniform(0, bg_parameter_.vibe_parameter().num_samples());
			*(sample_list+int_rand_num*size + offset) = pixel_value;

			int xx = x + off_[rng.uniform(0, 9)];
			if (xx < 0)
				xx = 0;
			if (xx >= bg_size.width)
				xx = bg_size.width - 1;


			int yy = y + off_[rng.uniform(0, 9)];
			if (yy < 0)
				yy = 0;
			if (yy >= bg_size.height)
				yy = bg_size.height - 1;

			*(sample_list + int_rand_num*size + xx + yy*bg_size.width) = pixel_value;
		}
	}
	else
	{
		*(foreground_match_count_mat + offset) = *(foreground_match_count_mat + offset) + 1;
		*(mask + offset) = 255;

		if (*(foreground_match_count_mat + offset) > bg_parameter_.vibe_parameter().max_mismatch_count())
		{
			*(mask + offset) = 0;
			*(foreground_match_count_mat+ offset) = 0;

			for (size_t i = 0; i < bg_parameter_.vibe_parameter().min_match(); i++)
			{
				int int_rand_num = rng.uniform(0, bg_parameter_.vibe_parameter().num_samples());
				*(sample_list + int_rand_num*size + offset) = pixel_value;
			}
		}
	}
}
void BgVibeBoost::TestAndUpdateSingleCPU(const Mat& gray_frame)
{
	Mat resize_frame;
	CvSize bg_size = cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height());
	int size = bg_size.width*bg_size.height;
	RNG rng((unsigned)time(NULL));
	if (gray_frame.cols != bg_size.width || gray_frame.rows != bg_size.height)
	{
		resize(gray_frame, resize_frame, bg_size);
	}
	else resize_frame = gray_frame;

	uint8_t num_samples = bg_parameter_.vibe_parameter().num_samples();
	uint8_t radius = bg_parameter_.vibe_parameter().radius();
	uint8_t min_match = bg_parameter_.vibe_parameter().min_match();

	for (size_t ii = 0; ii < bg_size.height; ii++)
	for (size_t jj = 0; jj < bg_size.width; jj++)
	{
		uint8_t matches_1(0), matches_2(0), count(0);
		uint32_t offset = jj + ii * bg_size.width;
		uint16_t dist_1;
		uint8_t int_rand_num;
		uchar pixel_value = *(resize_frame.data + offset);
		while (count < num_samples)
		{
			dist_1 = abs(*(sample_list1_+count*size + offset) - pixel_value);
			if (dist_1 < radius)
			{
				matches_1++;
			}
			if (matches_1 >= min_match)
			{
				break;
			}
			count++;
		}

		UpdateSamplist(matches_1, pixel_value, jj, ii, foreground_match_count_mat1_, mask_.data, sample_list1_);
	}
}
void BgVibeBoost::TestAndUpdateDoubleCPU(const Mat& gray_frame)
{
	Mat resize_frame;
	CvSize bg_size = cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height());
	int size = bg_size.height*bg_size.width;
	RNG rng((unsigned)time(NULL));
	if (gray_frame.cols != bg_size.width || gray_frame.rows != bg_size.height)
	{
		resize(gray_frame, resize_frame, bg_size);
	}
	else resize_frame = gray_frame;

	uint8_t num_samples = bg_parameter_.vibe_parameter().num_samples();
	uint8_t radius = bg_parameter_.vibe_parameter().radius();
	uint8_t min_match = bg_parameter_.vibe_parameter().min_match();

	time_t t1, t2;
	t1 = clock();

	for (size_t ii = 0; ii < bg_size.height; ii++)
	for (size_t jj = 0; jj < bg_size.width; jj++)
	{
		uint8_t matches_1(0), matches_2(0), count(0);
		uint32_t offset = jj + ii * bg_size.width;
		uint16_t dist_1, dist_2;
		uint8_t int_rand_num;
		uchar pixel_value = *(resize_frame.data + offset);
		while (count < num_samples)
		{
			dist_1 = abs(*(sample_list1_ + count*size + offset) - pixel_value);
			dist_2 = abs(*(sample_list2_ + count*size + offset) - pixel_value);

			if (dist_1 < radius)
			{
				matches_1++;
			}
			if (dist_2 < radius)
			{
				matches_2++;
			}
			if (matches_1 >= min_match && matches_2 >= min_match)
			{
				break;
			}
			count++;
		}

		UpdateSamplist(matches_1, pixel_value, jj, ii, foreground_match_count_mat1_, mask1_.data, sample_list1_);
		UpdateSamplist(matches_2, pixel_value, jj, ii, foreground_match_count_mat2_, mask2_.data, sample_list2_);

		if (*(mask1_.data + offset) && *(mask1_.data + +offset))
		{
			*(mask_.data + offset) = 255;
		}
		else *(mask_.data + offset) = 0;
	}
}
Mat& BgVibeBoost::mask1()
{
	if (bg_parameter_.vibe_parameter().double_bg() && !mask1_.empty())
	{
		return mask1_;
	}
	else
	{
		Mat zero_mat = Mat::zeros(cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height()), CV_8UC1);
		return zero_mat;
	}
}
Mat& BgVibeBoost::mask2()
{
	if (bg_parameter_.vibe_parameter().double_bg() && !mask2_.empty())
	{
		return mask2_;
	}
	else
	{
		Mat zero_mat = Mat::zeros(cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height()), CV_8UC1);
		return zero_mat;
	}
}

void BgVibeBoost::UpdateSingle(const Mat &frame)
{
	frame_count_++;
	CvSize bg_size = cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height());
	Mat bgMat;
	Mat resizeMask;
	CHECK(!frame.empty()) << "frame is empty!" << endl;

	Mat gray_frame;
	if (frame.channels() == 3)
	{
		cvtColor(frame, gray_frame, CV_RGB2GRAY);
	}
	else gray_frame = Mat(frame);

	if (gray_frame.cols != bg_size.width || gray_frame.rows != bg_size.height)
	{
		resize(gray_frame, bgMat, bg_size);
	}
	else gray_frame.copyTo(bgMat);


	if (bg_parameter_.bg_status() == vas::BG_UNINIALIZED)
	{
		InitSingle();
		return;
	}
	if (!bg_setup1_)
	{
		try
		{
			if (bg_parameter_.bg_method() == vas::VIBE_GPU)
			{
				ProcessFirstFrameGPU(bgMat, d_sample_list1_);
			}
			else ProcessFirstFrameCPU(bgMat, sample_list1_);
			bg_setup1_ = true;
			bg_parameter_.set_bg_status(vas::BG_SETUP);
			return;
		}
		catch (...)
		{
			LOG(ERROR) << "Vibe bg 1 set up error" << endl;
			return;
		}
	}
	if (bg_setup1_)
	{
		try
		{
			if (bg_parameter_.bg_method() == vas::VIBE_GPU)
			{
				TestAndUpdateSingleGPU(bgMat);
			}
			else
			{
				TestAndUpdateSingleCPU(bgMat);
			}
			if (Stable(mask_))
			{
				bg_parameter_.set_bg_status(vas::BG_UPDATING);
				MaskOperation(mask_);
			}
			else
			{
				bg_setup1_ = false;
				frame_count_ = 0;
				bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);
			}
		}
		catch (...)
		{
			cout << "[BgVibe.cpp] Update: Vibe Bg Setup error" << endl;
			return;
		}
	}



}
void BgVibeBoost::UpdateDouble(const Mat &frame)
{
	frame_count_++;
	CvSize bg_size = cvSize(bg_parameter_.bg_width(), bg_parameter_.bg_height());
	Mat bgMat;
	Mat resizeMask;

	CHECK(!frame.empty()) << "Frame is empty!" << endl;

	Mat gray_frame;
	if (frame.channels() == 3)
	{
		cvtColor(frame, gray_frame, CV_RGB2GRAY);
	}
	else gray_frame = Mat(frame);

	if (gray_frame.cols != bg_size.width || gray_frame.rows != bg_size.height)
	{
		resize(gray_frame, bgMat, bg_size);
	}
	else gray_frame.copyTo(bgMat);

	if (bg_parameter_.bg_status() == vas::BG_UNINIALIZED)
	{
		InitDouble();
		return;
	}
	if (!bg_setup1_)
	{
		try
		{
			if (bg_parameter_.bg_method() == vas::VIBE_GPU)
			{
				ProcessFirstFrameGPU(bgMat, d_sample_list1_);
			}
			else ProcessFirstFrameCPU(bgMat, sample_list1_);
			bg_setup1_ = true;
			bg_parameter_.set_bg_status(vas::BG_SETUP);
			return;
		}
		catch (...)
		{
			LOG(ERROR) << "Vibe bg 1 set up error" << endl;
			return;
		}
	}
	else if (!bg_setup2_&&frame_count_> bg_parameter_.vibe_parameter().bg2_delay())
	{
		try
		{
			if (bg_parameter_.bg_method() == vas::VIBE_GPU)
			{
				ProcessFirstFrameGPU(bgMat, d_sample_list2_);
			}
			else ProcessFirstFrameCPU(bgMat, sample_list2_);
			bg_setup2_ = true;
			bg_parameter_.set_bg_status(vas::BG_SETUP);
			return;
		}
		catch (...)
		{
			LOG(ERROR) << "Vibe bg 2 set up error" << endl;
			return;
		}
	}
	if (bg_setup1_&&bg_setup2_)
	{
		try
		{
			if (bg_parameter_.bg_method() == vas::VIBE_GPU)
			{
				TestAndUpdateDoubleGPU(bgMat);
			}
			else
			{
				TestAndUpdateDoubleCPU(bgMat);
			}
			if (Stable(mask_))
			{
				bg_parameter_.set_bg_status(vas::BG_UPDATING);
				MaskOperation(mask_);
			}
			else
			{
				bg_setup1_ = false;
				bg_setup2_ = false;
				frame_count_ = 0;
				bg_parameter_.set_bg_status(vas::BG_UNINIALIZED);
			}
		}
		catch (...)
		{
			cout << "[BgVibe.cpp] Update: Vibe Bg Setup error" << endl;
			return;
		}
	}

}