// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>


#define WIN32_LEAN_AND_MEAN 
#include<windows.h>


#include <boost/utility.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/filesystem.hpp>
#include <boost/signal.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>


#ifdef _DEBUG
#pragma comment(lib,"libprotobufd.lib")
#pragma comment(lib,"libprotocd.lib")
#else
#pragma comment(lib,"libprotobuf.lib")
#pragma comment(lib,"libprotoc.lib")
#endif


#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "glog/logging.h"
#ifdef _DEBUG
#pragma comment(lib, "libglogd.lib")
#else
#pragma comment(lib, "libglog.lib")
#endif 



// TODO:  在此处引用程序需要的其他头文件
