// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>



//TODO:  在此处引用程序需要的其他头文件
#include <boost/utility.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/filesystem.hpp>
#include <boost/signal.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>


#include"zini.h"
#include <string>

#include<direct.h>

#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "glog/logging.h"
 #ifdef _DEBUG
     #pragma comment(lib, "glog/libglogd.lib")
 #else
     #pragma comment(lib, "glog/libglog.lib")
 #endif 


#define MAX_INT_64 9223372036854775807




