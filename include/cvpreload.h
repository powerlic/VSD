#ifndef _CVPRELOAD_H_
#define _CVPRELOAD_H_

#pragma once
#include "cv.h"
#include "opencv2/core/version.hpp"
#include <opencv2/opencv.hpp>

#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#if CV_MAJOR_VERSION >= 3 
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/cudaoptflow.hpp>
#endif

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif

#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )
#pragma comment( lib, cvLIB("video") )
#pragma comment( lib, cvLIB("objdetect") )
#pragma comment( lib, cvLIB("gpu") )

#if CV_MAJOR_VERSION >= 3 
#pragma comment( lib, cvLIB("cudaarithm") )
#pragma comment( lib, cvLIB("cudaoptflow") )
#pragma comment( lib, cvLIB("cudabgsegm") )
#pragma comment( lib, cvLIB("cudaimgproc") )

#pragma comment( lib, cvLIB("imgcodecs") )
#pragma comment( lib, cvLIB("calib3d") )
#pragma comment( lib, cvLIB("videoio") )
#endif

#endif