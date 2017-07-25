// stdafx.h : ��׼ϵͳ�����ļ��İ����ļ���
// ���Ǿ���ʹ�õ��������ĵ�
// �ض�����Ŀ�İ����ļ�
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>



//TODO:  �ڴ˴����ó�����Ҫ������ͷ�ļ�
#include <boost/utility.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/filesystem.hpp>
#include <boost/signal.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>


#include <string>
#include<direct.h>

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "glog/logging.h"
#ifdef _DEBUG
#pragma comment(lib, "glog/libglogd.lib")
#else
#pragma comment(lib, "glog/libglog.lib")
#endif 


#ifdef _DEBUG
#pragma comment(lib, "gflagsd.lib")
#pragma comment(lib, "gflags_nothreadsd.lib")
#else
#pragma comment(lib, "gflags.lib")
#pragma comment(lib, "gflags_nothreads.lib")
#endif 


#define MAX_INT_64 9223372036854775807





