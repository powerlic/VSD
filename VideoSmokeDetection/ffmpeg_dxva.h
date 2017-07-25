#pragma once
#ifndef FFMPEG_DXVA2_H_
#define FFMPEG_DXVA2_H_
#define WIN32_LEAN_AND_MEAN//不加的话boost库会和windows库winsock冲突
#define COBJMACROS
#define inline __inline
#include<windows.h>
extern "C"
{
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libavutil/avutil.h"
#include "libavutil/dict.h"
#include "libswscale/swscale.h"
#include "libavcodec/dxva2.h"
#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
}

#include <d3d9.h>
#include <dxva2api.h>
#include<string>




enum HWAccelID
{
	HWACCEL_NONE = 0,
	HWACCEL_AUTO,
	HWACCEL_VDPAU,
	HWACCEL_DXVA2,
	HWACCEL_VDA,
	HWACCEL_VIDEOTOOLBOX,
	HWACCEL_QSV,
};

typedef struct InputStream
{
	int stream_index;
	AVStream *st;
	enum HWAccelID hwaccel_id;
	char  *hwaccel_device;
	/* hwaccel context */
	enum HWAccelID active_hwaccel_id;
	void  *hwaccel_ctx;
	void  *hwaccel_master;
	void (*hwaccel_uninit)(AVCodecContext *s);
	int(*hwaccel_get_buffer)(AVCodecContext *s, AVFrame *frame, int flags);
	int(*hwaccel_retrieve_data)(AVCodecContext *s, AVFrame *frame);
	enum AVPixelFormat hwaccel_pix_fmt;
	InputStream()
	{
		st = NULL;
		hwaccel_ctx = NULL;
		hwaccel_master = NULL;
	}
}InputStream;

typedef struct dxva2_mode
{
	const GUID     *guid;
	enum AVCodecID codec;
} dxva2_mode;

typedef struct surface_info
{
	int used;
	uint64_t age;
} surface_info;

typedef struct DXVA2Master
{
	HMODULE d3dlib;
	HMODULE dxva2lib;

	HANDLE  deviceHandle;

	IDirect3D9                  *d3d9;
	IDirect3DDevice9            *d3d9device;
	IDirect3DDeviceManager9     *d3d9devmgr;
	IDirectXVideoDecoderService *decoder_service;
};



typedef struct DXVA2Context
{
	IDirectXVideoDecoder        *decoder;

	GUID                        decoder_guid;
	DXVA2_ConfigPictureDecode   decoder_config;

	LPDIRECT3DSURFACE9          *surfaces;
	surface_info                *surface_infos;
	uint32_t                    num_surfaces;
	uint64_t                    surface_age;
	D3DFORMAT                   surface_format;

	AVFrame                     *tmp_frame;
} DXVA2Context;

typedef struct DXVA2SurfaceWrapper
{
	DXVA2Context         *ctx;
	LPDIRECT3DSURFACE9   surface;
	IDirectXVideoDecoder *decoder;
} DXVA2SurfaceWrapper;

typedef struct HWAccel
{
	const char *name;
	//int(*init)(AVCodecContext *s, const DXVA2Master*dxva2_master);
	int(*init)(AVCodecContext *s, const DXVA2Master *dxva2_master);
	enum HWAccelID id;
	enum AVPixelFormat pix_fmt;
} HWAccel;

void dxva2_uninit_master(DXVA2Master *dxva2_master);
DXVA2Master* dxva2_init_master();
int dxva2_alloc(AVCodecContext *s);
int dxva2_init(AVCodecContext *s, const DXVA2Master *dxva2_master);

enum AVPixelFormat get_format(AVCodecContext *s, const enum AVPixelFormat *pix_fmts);
int get_buffer(AVCodecContext *s, AVFrame *frame, int flags);
void dxva2_destroy_decoder(AVCodecContext *s);
void dxva2_uninit(AVCodecContext *s);
void dxva2_release_buffer(void *opaque, uint8_t *data);
int dxva2_get_buffer(AVCodecContext *s, AVFrame *frame, int flags);
int dxva2_retrieve_data(AVCodecContext *s, AVFrame *frame);
int dxva2_create_decoder(AVCodecContext *s);
int dxva2_get_decoder_configuration(AVCodecContext *s, const GUID *device_guid, const DXVA2_VideoDesc *desc, DXVA2_ConfigPictureDecode *config);

const HWAccel *get_hwaccel(enum AVPixelFormat pix_fmt);

#endif

