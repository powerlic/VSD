#include"stdafx.h"
#define _DLL_VasClientP
#include "VasClientP.h"
#include "VasClient.h"

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

DLL_VasClientP_API VasClientP* APIENTRY CreateVasClientP(const char* proto_file_name)
{
	return new VasClient(proto_file_name);
}