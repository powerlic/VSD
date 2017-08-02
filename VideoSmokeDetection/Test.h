#pragma once
#include "stdafx.h"
#include"VasIO.h"
#include"Background.h"
#include"BgGuassian.h"
#include"BgVibe.h"
#include"Filter.h"
#include "VasDetectMaster.h"
#include"IClassification.h"
#include "concurrent_queue.h"
#include "VasDecodeMaster.h"
#include"VasService.h"


void TestBgVibeReadFromTxtFile();
void TestBgGuassianReadFromTxtFile();
void TestBg();
void TestOpencvReadVideo();
void TestBgVibe();
void TestBgGuassian();
void TestFilter();
void TestDetectMaster();
void TestDvrDetect();
void TestDecodeMaster();
void TestVas();
void TestVasServices();


bool TestGetFrame(const char *service_id, Mat &frame, int64_t &pts, int64_t &No);

vector<Prediction> Predict(const Mat &image_mat, const int &N);


void TestMergeRects();
