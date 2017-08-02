#include"stdafx.h"
#include "VasUnit.h"
int _tmain(int argc, _TCHAR* argv[])
{

	//shared_ptr<VasUnit> ptr_vas_unit = shared_ptr<VasUnit>(new VasUnit("../proto/caffe_model_0523.prototxt","../proto/rtsp_service_list.prototxt","../proto/client_info.prototxt"));
	shared_ptr<VasUnit> ptr_vas_unit = shared_ptr<VasUnit>(new VasUnit("../proto/caffe_model_0523.prototxt", "../proto/service_list.prototxt", "../proto/client_info.prototxt"));
	ptr_vas_unit->Start();
	while (true)
	{
		Sleep(10000);
	}
	return 0;
}
