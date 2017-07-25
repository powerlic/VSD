#include"stdafx.h"
#define _DLL_VASP
#include"VasP.h"
#include"VasService.h"
#include"NvrService.h"

BOOL APIENTRY DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

DLL_VASP_API VasP* APIENTRY CreateVasP(const char* caffe_model_proto_file_name)
{
	return new VasService(caffe_model_proto_file_name);
}

DLL_VASP_API VasNvrP* APIENTRY CreateVasNvrP(const char* caffe_model_proto_file_name, const char*nvr_access_proto_file)
{
	return new NvrService(caffe_model_proto_file_name, nvr_access_proto_file);
}
