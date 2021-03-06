#include "stdafx.h"
#include "ffmpeg_dxva.h"

#include <initguid.h>


#pragma comment(lib,"avutil.lib")
#pragma comment(lib,"avcodec.lib")
#pragma comment(lib,"avformat.lib")
#pragma comment(lib,"swscale.lib")
#pragma comment(lib,"avfilter.lib")
#pragma comment(lib,"swresample.lib")
#pragma comment(lib,"avdevice.lib")
#pragma comment(lib,"postproc.lib")
#pragma comment(lib,"d3dx9.lib")


#define HAVE_DXVA2_LIB 1

typedef IDirect3D9* WINAPI pDirect3DCreate9(UINT);
typedef HRESULT WINAPI pCreateDeviceManager9(UINT *, IDirect3DDeviceManager9 **);


DEFINE_GUID(IID_IDirectXVideoDecoderService, 0xfc51a551, 0xd5e7, 0x11d9, 0xaf, 0x55, 0x00, 0x05, 0x4e, 0x43, 0xff, 0x02);
DEFINE_GUID(DXVA2_ModeMPEG2_VLD, 0xee27417f, 0x5e28, 0x4e65, 0xbe, 0xea, 0x1d, 0x26, 0xb5, 0x08, 0xad, 0xc9);
DEFINE_GUID(DXVA2_ModeMPEG2and1_VLD, 0x86695f12, 0x340e, 0x4f04, 0x9f, 0xd3, 0x92, 0x53, 0xdd, 0x32, 0x74, 0x60);
DEFINE_GUID(DXVA2_ModeH264_E, 0x1b81be68, 0xa0c7, 0x11d3, 0xb9, 0x84, 0x00, 0xc0, 0x4f, 0x2e, 0x73, 0xc5);
DEFINE_GUID(DXVA2_ModeH264_F, 0x1b81be69, 0xa0c7, 0x11d3, 0xb9, 0x84, 0x00, 0xc0, 0x4f, 0x2e, 0x73, 0xc5);
DEFINE_GUID(DXVADDI_Intel_ModeH264_E, 0x604F8E68, 0x4951, 0x4C54, 0x88, 0xFE, 0xAB, 0xD2, 0x5C, 0x15, 0xB3, 0xD6);
DEFINE_GUID(DXVA2_ModeVC1_D, 0x1b81beA3, 0xa0c7, 0x11d3, 0xb9, 0x84, 0x00, 0xc0, 0x4f, 0x2e, 0x73, 0xc5);
DEFINE_GUID(DXVA2_ModeVC1_D2010, 0x1b81beA4, 0xa0c7, 0x11d3, 0xb9, 0x84, 0x00, 0xc0, 0x4f, 0x2e, 0x73, 0xc5);
DEFINE_GUID(DXVA2_ModeHEVC_VLD_Main, 0x5b11d51b, 0x2f4c, 0x4452, 0xbc, 0xc3, 0x09, 0xf2, 0xa1, 0x16, 0x0c, 0xc0);
DEFINE_GUID(DXVA2_ModeHEVC_VLD_Main10, 0x107af0e0, 0xef1a, 0x4d19, 0xab, 0xa8, 0x67, 0xa1, 0x63, 0x07, 0x3d, 0x13);
DEFINE_GUID(DXVA2_ModeVP9_VLD_Profile0, 0x463707f8, 0xa1d0, 0x4585, 0x87, 0x6d, 0x83, 0xaa, 0x6d, 0x60, 0xb8, 0x9e);
DEFINE_GUID(DXVA2_NoEncrypt, 0x1b81beD0, 0xa0c7, 0x11d3, 0xb9, 0x84, 0x00, 0xc0, 0x4f, 0x2e, 0x73, 0xc5);
DEFINE_GUID(GUID_NULL, 0x00000000, 0x0000, 0x0000, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);


static const dxva2_mode dxva2_modes[] =
{
	/* MPEG-2 */
	{ &DXVA2_ModeMPEG2_VLD, AV_CODEC_ID_MPEG2VIDEO },
	{ &DXVA2_ModeMPEG2and1_VLD, AV_CODEC_ID_MPEG2VIDEO },

	/* H.264 */
	{ &DXVA2_ModeH264_F, AV_CODEC_ID_H264 },
	{ &DXVA2_ModeH264_E, AV_CODEC_ID_H264 },
	/* Intel specific H.264 mode */
	{ &DXVADDI_Intel_ModeH264_E, AV_CODEC_ID_H264 },

	/* VC-1 / WMV3 */
	{ &DXVA2_ModeVC1_D2010, AV_CODEC_ID_VC1 },
	{ &DXVA2_ModeVC1_D2010, AV_CODEC_ID_WMV3 },
	{ &DXVA2_ModeVC1_D, AV_CODEC_ID_VC1 },
	{ &DXVA2_ModeVC1_D, AV_CODEC_ID_WMV3 },

	/* HEVC/H.265 */
	{ &DXVA2_ModeHEVC_VLD_Main, AV_CODEC_ID_HEVC },
	{ &DXVA2_ModeHEVC_VLD_Main10, AV_CODEC_ID_HEVC },

	/* VP8/9 */
	{ &DXVA2_ModeVP9_VLD_Profile0, AV_CODEC_ID_VP9 },

	{ NULL, AV_CODEC_ID_NONE },
};

const HWAccel hwaccels[] =
{
#if HAVE_VDPAU_X11
		{ "vdpau", vdpau_init, HWACCEL_VDPAU, AV_PIX_FMT_VDPAU },
#endif

#if HAVE_DXVA2_LIB
		{ "dxva2", dxva2_init, HWACCEL_DXVA2, AV_PIX_FMT_DXVA2_VLD },
#endif

#if CONFIG_VDA
		{ "vda", videotoolbox_init, HWACCEL_VDA, AV_PIX_FMT_VDA },
#endif

#if CONFIG_VIDEOTOOLBOX
		{ "videotoolbox", videotoolbox_init, HWACCEL_VIDEOTOOLBOX, AV_PIX_FMT_VIDEOTOOLBOX },
#endif

#if CONFIG_LIBMFX
		{ "qsv", qsv_init, HWACCEL_QSV, AV_PIX_FMT_QSV },
#endif

		{ 0 },
};

boost::mutex dxva2_mutex;

 const HWAccel *get_hwaccel(enum AVPixelFormat pix_fmt)
{
	int i;
	for (i = 0; hwaccels[i].name; i++)
		if (hwaccels[i].pix_fmt == pix_fmt)
			return &hwaccels[i];
	return NULL;
}

 DXVA2Master* dxva2_init_master()
{
	int loglevel = AV_LOG_VERBOSE;
	pDirect3DCreate9      *createD3D = NULL;
	pCreateDeviceManager9 *createDeviceManager = NULL;

	HRESULT hr;
	D3DPRESENT_PARAMETERS d3dpp = { 0 };
	D3DDISPLAYMODE        d3ddm;
	unsigned resetToken = 0;
	UINT adapter = D3DADAPTER_DEFAULT;

	DXVA2Master* dxva2_master = (DXVA2Master*)av_mallocz(sizeof(*dxva2_master));
	if (!dxva2_master)
	{
		return NULL;
	}

	dxva2_master->deviceHandle = INVALID_HANDLE_VALUE;

	
	dxva2_master->d3dlib = LoadLibrary(L"d3d9.dll");
	if (!dxva2_master->d3dlib)
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to load D3D9 library\n");
		goto fail;
	}

	dxva2_master->dxva2lib = LoadLibrary(L"dxva2.dll");
	if (!dxva2_master->dxva2lib)
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to load DXVA2 library\n");
		goto fail;
	}

	createD3D = (pDirect3DCreate9 *)GetProcAddress(dxva2_master->d3dlib, "Direct3DCreate9");
	if (!createD3D)
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to locate Direct3DCreate9\n");
		goto fail;
	}

	createDeviceManager = (pCreateDeviceManager9 *)GetProcAddress(dxva2_master->dxva2lib, "DXVA2CreateDirect3DDeviceManager9");
	if (!createDeviceManager)
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to locate DXVA2CreateDirect3DDeviceManager9\n");
		goto fail;
	}

	dxva2_master->d3d9 = createD3D(D3D_SDK_VERSION);
	if (!dxva2_master->d3d9)
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to create IDirect3D object\n");
		goto fail;
	}

	IDirect3D9_GetAdapterDisplayMode(dxva2_master->d3d9, adapter, &d3ddm);

	d3dpp.Windowed = TRUE;
	d3dpp.BackBufferWidth = 800;
	d3dpp.BackBufferHeight = 576;
	d3dpp.BackBufferCount = 0;
	d3dpp.BackBufferFormat = d3ddm.Format;
	d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
	d3dpp.Flags = D3DPRESENTFLAG_VIDEO;


	hr = IDirect3D9_CreateDevice(dxva2_master->d3d9, adapter, D3DDEVTYPE_HAL, GetDesktopWindow(),
		D3DCREATE_SOFTWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED | D3DCREATE_FPU_PRESERVE,
		&d3dpp, &dxva2_master->d3d9device);
	if (FAILED(hr))
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to create Direct3D device\n");
		goto fail;
	}

	hr = createDeviceManager(&resetToken, &dxva2_master->d3d9devmgr);
	if (FAILED(hr))
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to create Direct3D device manager\n");
		goto fail;
	}

	dxva2_master->d3d9devmgr->ResetDevice(dxva2_master->d3d9device, resetToken);
	if (FAILED(hr)) {
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to bind Direct3D device to device manager\n");
		goto fail;
	}

	hr = dxva2_master->d3d9devmgr->OpenDeviceHandle(&dxva2_master->deviceHandle);
	if (FAILED(hr)) {
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to open device handle\n");
		goto fail;
	}

	hr = dxva2_master->d3d9devmgr->GetVideoService(dxva2_master->deviceHandle, IID_IDirectXVideoDecoderService, (void **)&dxva2_master->decoder_service);
	if (FAILED(hr)) {
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to create IDirectXVideoDecoderService\n");
		goto fail;
	}
	return dxva2_master;

fail:
	dxva2_uninit_master(dxva2_master);
	return NULL;
}
void dxva2_uninit_master(DXVA2Master *dxva2_master)
{
	if (dxva2_master->decoder_service)
		dxva2_master->decoder_service->Release();

	if (dxva2_master->d3d9devmgr && dxva2_master->deviceHandle != INVALID_HANDLE_VALUE)
		dxva2_master->d3d9devmgr->CloseDeviceHandle(dxva2_master->deviceHandle);

	if (dxva2_master->d3d9devmgr)
		dxva2_master->d3d9devmgr->Release();

	if (dxva2_master->d3d9device)
		IDirect3DDevice9_Release(dxva2_master->d3d9device);

	if (dxva2_master->d3d9)
		IDirect3D9_Release(dxva2_master->d3d9);

	if (dxva2_master->d3dlib)
		FreeLibrary(dxva2_master->d3dlib);

	if (dxva2_master->dxva2lib)
		FreeLibrary(dxva2_master->dxva2lib);


}
int dxva2_alloc(AVCodecContext *s)
{
	boost::mutex::scoped_lock lock(dxva2_mutex);

	InputStream  *ist = (InputStream *)s->opaque;
	int loglevel = (ist->hwaccel_id == HWACCEL_AUTO) ? AV_LOG_VERBOSE : AV_LOG_ERROR;


	DXVA2Context *ctx;
	ctx = (DXVA2Context*)av_mallocz(sizeof(*ctx));
	if (!ctx)
	{
		lock.unlock();
		return AVERROR(ENOMEM);
	}

	ist->hwaccel_ctx = ctx;
	ist->hwaccel_uninit = dxva2_uninit;
	ist->hwaccel_get_buffer = dxva2_get_buffer;
	ist->hwaccel_retrieve_data = dxva2_retrieve_data;

	ctx->tmp_frame = av_frame_alloc();
	if (!ctx->tmp_frame)
		goto fail;

	s->hwaccel_context = av_mallocz(sizeof(struct dxva_context));
	if (!s->hwaccel_context)
		goto fail;

	lock.unlock();
	return 0;

fail:
	lock.unlock();
	dxva2_uninit(s);
	return AVERROR(EINVAL);
}

int dxva2_init(AVCodecContext *s, const DXVA2Master *dxva2_master)
{
	
	InputStream *ist = (InputStream *)s->opaque;
	int loglevel = (ist->hwaccel_id == HWACCEL_AUTO) ? AV_LOG_VERBOSE : AV_LOG_ERROR;

	ist->hwaccel_master = (void*)dxva2_master;

	DXVA2Context *ctx;

	int ret;
	if(!ist->hwaccel_ctx) 
	{
		ret = dxva2_alloc(s);
		if (ret < 0)
		{
			return ret;
		}
	}

	ctx = (DXVA2Context*)ist->hwaccel_ctx;

	if (s->codec_id == AV_CODEC_ID_H264 && (s->profile & ~FF_PROFILE_H264_CONSTRAINED) > FF_PROFILE_H264_HIGH)
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Unsupported H.264 profile for DXVA2 HWAccel: %d\n", s->profile);
		return AVERROR(EINVAL);
	}

	if (s->codec_id == AV_CODEC_ID_HEVC &&s->profile != FF_PROFILE_HEVC_MAIN && s->profile != FF_PROFILE_HEVC_MAIN_10)
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Unsupported HEVC profile for DXVA2 HWAccel: %d\n", s->profile);
		return AVERROR(EINVAL);
	}

	if (ctx->decoder)
	{ 
		dxva2_destroy_decoder(s);
	}

	ret = dxva2_create_decoder(s);

	if (ret < 0)
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Error creating the DXVA2 decoder\n");
		return ret;
	}

	return 0;
}
int dxva2_create_decoder(AVCodecContext *s)
{
	//boost::mutex::scoped_lock lock(dxva2_mutex);

	InputStream  *ist = (InputStream*)s->opaque;
	int loglevel = (ist->hwaccel_id == HWACCEL_AUTO) ? AV_LOG_VERBOSE : AV_LOG_ERROR;
	DXVA2Context *ctx = (DXVA2Context*)ist->hwaccel_ctx;
	DXVA2Master *dxva2_master = (DXVA2Master*)ist->hwaccel_master;
	struct dxva_context *dxva_ctx = (dxva_context*)s->hwaccel_context;
	GUID *guid_list = NULL;
	unsigned guid_count = 0, i, j;
	GUID device_guid = GUID_NULL;
	D3DFORMAT t_surface_format;


	const D3DFORMAT surface_format = (s->sw_pix_fmt == AV_PIX_FMT_YUV420P10) ? (D3DFORMAT)MKTAG('P', '0', '1', '0') : (D3DFORMAT)MKTAG('N', 'V', '1', '2');
	D3DFORMAT target_format = D3DFMT_UNKNOWN;
	DXVA2_VideoDesc desc = { 0 };
	DXVA2_ConfigPictureDecode config;
	HRESULT hr;
	int surface_alignment;
	int ret;

	hr = dxva2_master->decoder_service->GetDecoderDeviceGuids(&guid_count, &guid_list);
	if (FAILED(hr))
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to retrieve decoder device GUIDs\n");
		goto fail;
	}

	for (i = 0; dxva2_modes[i].guid; i++)
	{
		D3DFORMAT *target_list = NULL;
		unsigned target_count = 0;
		const dxva2_mode *mode = &dxva2_modes[i];
		if (mode->codec != s->codec_id)
			continue;

		for (j = 0; j < guid_count; j++) {
			if (IsEqualGUID(*mode->guid, guid_list[j]))
				break;
		}
		if (j == guid_count)
			continue;

		//hr = IDirectXVideoDecoderService_GetDecoderRenderTargets(ctx->decoder_service, mode->guid, &target_count, &target_list);
		dxva2_master->decoder_service->GetDecoderRenderTargets(*mode->guid, &target_count, &target_list);
		if (FAILED(hr))
		{
			continue;
		}
		for (j = 0; j < target_count; j++)
		{
			const D3DFORMAT format = target_list[j];
			if (format == surface_format) {
				target_format = format;
				break;
			}
		}
		CoTaskMemFree(target_list);
		if (target_format)
		{
			device_guid = *mode->guid;
			break;
		}
	}

	CoTaskMemFree(guid_list);

	if (IsEqualGUID(device_guid, GUID_NULL))
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] No decoder device for codec found\n");
		goto fail;
	}

	desc.SampleWidth = s->coded_width;
	desc.SampleHeight = s->coded_height;
	desc.Format = target_format;


	ret = dxva2_get_decoder_configuration(s, &device_guid, &desc, &config);
	if (ret < 0)
	{
		goto fail;
	}

	if (s->codec_id == AV_CODEC_ID_MPEG2VIDEO)
		surface_alignment = 32;

	else if (s->codec_id == AV_CODEC_ID_HEVC)
		surface_alignment = 128;
	else
		surface_alignment = 16;

	ctx->num_surfaces = 4;

	if (s->codec_id == AV_CODEC_ID_H264 || s->codec_id == AV_CODEC_ID_HEVC)
		ctx->num_surfaces += 16;
	else if (s->codec_id == AV_CODEC_ID_VP9)
		ctx->num_surfaces += 8;
	else
		ctx->num_surfaces += 2;

	if (s->active_thread_type & FF_THREAD_FRAME)
		ctx->num_surfaces += s->thread_count;

	ctx->surfaces = (LPDIRECT3DSURFACE9*)av_mallocz(ctx->num_surfaces * sizeof(*ctx->surfaces));
	ctx->surface_infos = (surface_info*)av_mallocz(ctx->num_surfaces * sizeof(*ctx->surface_infos));
	ctx->surface_format = target_format;

	if (!ctx->surfaces || !ctx->surface_infos)
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Unable to allocate surface arrays\n");
		goto fail;
	}

	hr = dxva2_master->decoder_service->CreateSurface(FFALIGN(s->coded_width, surface_alignment),
		FFALIGN(s->coded_height, surface_alignment),
		ctx->num_surfaces - 1,
		target_format, D3DPOOL_DEFAULT, 0,
		DXVA2_VideoDecoderRenderTarget,
		ctx->surfaces, NULL);

	if (FAILED(hr))
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to create %d video surfaces\n", ctx->num_surfaces);
		goto fail;
	}

	hr = dxva2_master->decoder_service->CreateVideoDecoder(device_guid, &desc, &config, ctx->surfaces, ctx->num_surfaces, &ctx->decoder);

	if (FAILED(hr))
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Failed to create DXVA2 video decoder\n");
		goto fail;
	}

	ctx->decoder_guid = device_guid;
	ctx->decoder_config = config;

	dxva_ctx->cfg = &ctx->decoder_config;
	dxva_ctx->decoder = ctx->decoder;
	dxva_ctx->surface = ctx->surfaces;
	dxva_ctx->surface_count = ctx->num_surfaces;

	if (IsEqualGUID(ctx->decoder_guid, DXVADDI_Intel_ModeH264_E))
		dxva_ctx->workaround |= FF_DXVA2_WORKAROUND_INTEL_CLEARVIDEO;

	//lock.unlock();
	return 0;

fail:
	//lock.unlock();
	dxva2_destroy_decoder(s);
	return AVERROR(EINVAL);

}

int dxva2_get_decoder_configuration(AVCodecContext *s, const GUID *device_guid, const DXVA2_VideoDesc *desc, DXVA2_ConfigPictureDecode *config)
{

	InputStream  *ist = (InputStream*)s->opaque;
	int loglevel = (ist->hwaccel_id == HWACCEL_AUTO) ? AV_LOG_VERBOSE : AV_LOG_ERROR;
	DXVA2Context *ctx = (DXVA2Context*)ist->hwaccel_ctx;
	DXVA2Master *dxva2_master = (DXVA2Master*)ist->hwaccel_master;

	unsigned cfg_count = 0, best_score = 0;
	DXVA2_ConfigPictureDecode *cfg_list = NULL;
	DXVA2_ConfigPictureDecode best_cfg = { { 0 } };
	HRESULT hr;
	int i;

	hr = dxva2_master->decoder_service->GetDecoderConfigurations((*device_guid), desc, NULL, &cfg_count, &cfg_list);
	if (FAILED(hr)) {
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] Unable to retrieve decoder configurations\n");
		return AVERROR(EINVAL);
	}

	for (i = 0; i < cfg_count; i++)
	{
		DXVA2_ConfigPictureDecode *cfg = &cfg_list[i];

		unsigned score;
		if (cfg->ConfigBitstreamRaw == 1)
			score = 1;
		else if (s->codec_id == AV_CODEC_ID_H264 && cfg->ConfigBitstreamRaw == 2)
			score = 2;
		else
			continue;
		if (IsEqualGUID(cfg->guidConfigBitstreamEncryption, DXVA2_NoEncrypt))
			score += 16;
		if (score > best_score) {
			best_score = score;
			best_cfg = *cfg;
		}
	}
	CoTaskMemFree(cfg_list);

	if (!best_score)
	{
		av_log(NULL, loglevel, "[ffmpeg_dxva.cpp] No valid decoder configuration available\n");
		return AVERROR(EINVAL);
	}

	*config = best_cfg;
	return 0;

}

void dxva2_uninit(AVCodecContext *s)
{

	InputStream  *ist = (InputStream  *)s->opaque;
	DXVA2Context *ctx = (DXVA2Context *)ist->hwaccel_ctx;

	ist->hwaccel_uninit = NULL;
	ist->hwaccel_get_buffer = NULL;
	ist->hwaccel_retrieve_data = NULL;
	ist->hwaccel_master = NULL;

	if (ctx->decoder)
		dxva2_destroy_decoder(s);

	av_frame_free(&ctx->tmp_frame);
	av_freep(&ist->hwaccel_ctx);
	av_freep(&s->hwaccel_context);

}
void dxva2_destroy_decoder(AVCodecContext *s)
{
	boost::mutex::scoped_lock lock(dxva2_mutex);
	InputStream  *ist = (InputStream *)s->opaque;
	DXVA2Context *ctx = (DXVA2Context *)ist->hwaccel_ctx;
	int i;

	if (ctx->surfaces)
	{
		for (i = 0; i < ctx->num_surfaces; i++)
		{
			if (ctx->surfaces[i])
				IDirect3DSurface9_Release(ctx->surfaces[i]);
		}
	}
	av_freep(&ctx->surfaces);
	av_freep(&ctx->surface_infos);
	ctx->num_surfaces = 0;
	ctx->surface_age = 0;

	if (ctx->decoder)
	{
		ctx->decoder->Release();
		ctx->decoder = NULL;
	}
	lock.unlock();
}
void dxva2_release_buffer(void *opaque, uint8_t *data)
{
	boost::mutex::scoped_lock lock(dxva2_mutex);

	DXVA2SurfaceWrapper *w = (DXVA2SurfaceWrapper *)opaque;
	DXVA2Context        *ctx = w->ctx;
	int i;

	for (i = 0; i < ctx->num_surfaces; i++)
	{
		if (ctx->surfaces[i] == w->surface)
		{
			ctx->surface_infos[i].used = 0;
			break;
		}
	}
	IDirect3DSurface9_Release(w->surface);
	w->decoder->Release();
	av_free(w);

	lock.unlock();

}
int dxva2_get_buffer(AVCodecContext *s, AVFrame *frame, int flags)
{
	boost::mutex::scoped_lock lock(dxva2_mutex);

	InputStream  *ist = (InputStream  *)s->opaque;
	DXVA2Context *ctx = (DXVA2Context *)ist->hwaccel_ctx;
	int i, old_unused = -1;
	LPDIRECT3DSURFACE9 surface;
	DXVA2SurfaceWrapper *w = NULL;

	av_assert0(frame->format == AV_PIX_FMT_DXVA2_VLD);

	for (i = 0; i < ctx->num_surfaces; i++)
	{
		surface_info *info = &ctx->surface_infos[i];
		if (!info->used && (old_unused == -1 || info->age < ctx->surface_infos[old_unused].age))
			old_unused = i;
	}
	if (old_unused == -1)
	{
		av_log(NULL, AV_LOG_ERROR, "[ffmpeg_dxva.cpp] No free DXVA2 surface!\n");
		lock.unlock();
		return AVERROR(ENOMEM);
	}
	i = old_unused;

	surface = ctx->surfaces[i];

	w = (DXVA2SurfaceWrapper*)av_mallocz(sizeof(*w));
	if (!w)
	{
		lock.unlock();
		return AVERROR(ENOMEM);
	}

	frame->buf[0] = av_buffer_create((uint8_t*)surface, 0,
		dxva2_release_buffer, w,
		AV_BUFFER_FLAG_READONLY);
	if (!frame->buf[0])
	{
		av_free(w);
		lock.unlock();
		return AVERROR(ENOMEM);
	}

	w->ctx = ctx;
	w->surface = surface;
	IDirect3DSurface9_AddRef(w->surface);
	w->decoder = ctx->decoder;
	//IDirectXVideoDecoder_AddRef(w->decoder);
	w->decoder->AddRef();

	ctx->surface_infos[i].used = 1;
	ctx->surface_infos[i].age = ctx->surface_age++;

	frame->data[3] = (uint8_t *)surface;

	lock.unlock();
	return 0;
}

int dxva2_retrieve_data(AVCodecContext *s, AVFrame *frame)
{
	boost::mutex::scoped_lock lock(dxva2_mutex);

	LPDIRECT3DSURFACE9 surface = (LPDIRECT3DSURFACE9)frame->data[3];
	InputStream        *ist = (InputStream*)s->opaque;
	DXVA2Context       *ctx = (DXVA2Context*)ist->hwaccel_ctx;
	D3DSURFACE_DESC    surfaceDesc;
	D3DLOCKED_RECT     LockedRect;
	HRESULT            hr;
	int                ret, nbytes;

	IDirect3DSurface9_GetDesc(surface, &surfaceDesc);

	ctx->tmp_frame->width = frame->width;
	ctx->tmp_frame->height = frame->height;
	switch (ctx->surface_format)
	{
	case MKTAG('N', 'V', '1', '2'):
		ctx->tmp_frame->format = AV_PIX_FMT_NV12;
		nbytes = 1;
		break;
	case MKTAG('P', '0', '1', '0'):
		ctx->tmp_frame->format = AV_PIX_FMT_P010;
		nbytes = 2;
		break;
	default:
		av_assert0(0);
	}

	ret = av_frame_get_buffer(ctx->tmp_frame, 32);
	if (ret < 0)
	{
		lock.unlock();
		return ret;
	}

	hr = IDirect3DSurface9_LockRect(surface, &LockedRect, NULL, D3DLOCK_READONLY);
	if (FAILED(hr))
	{
		av_log(NULL, AV_LOG_ERROR, "[ffmpeg_dxva.cpp] Unable to lock DXVA2 surface\n");
		lock.unlock();
		return AVERROR_UNKNOWN;
	}

	av_image_copy_plane(ctx->tmp_frame->data[0], ctx->tmp_frame->linesize[0],
		(uint8_t*)LockedRect.pBits,
		LockedRect.Pitch, frame->width * nbytes, frame->height);

	av_image_copy_plane(ctx->tmp_frame->data[1], ctx->tmp_frame->linesize[1],
		(uint8_t*)LockedRect.pBits + LockedRect.Pitch * surfaceDesc.Height,
		LockedRect.Pitch, frame->width * nbytes, frame->height / 2);

	IDirect3DSurface9_UnlockRect(surface);

	ret = av_frame_copy_props(ctx->tmp_frame, frame);
	if (ret < 0)
		goto fail;

	av_frame_unref(frame);
	av_frame_move_ref(frame, ctx->tmp_frame);
	lock.unlock();
	return 0;
fail:
	lock.unlock();
	av_frame_unref(ctx->tmp_frame);
	return ret;
}
enum AVPixelFormat get_format(AVCodecContext *s, const enum AVPixelFormat *pix_fmts)
{
	boost::mutex::scoped_lock lock(dxva2_mutex);

	InputStream *ist = (InputStream *)s->opaque;
	const enum AVPixelFormat *p;
	int ret;

	const DXVA2Master *dxva2_master = (DXVA2Master *)ist->hwaccel_master;

	for (p = pix_fmts; *p != -1; p++)
	{
		const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(*p);
		const HWAccel *hwaccel;

		if (!(desc->flags & AV_PIX_FMT_FLAG_HWACCEL))
			break;

		hwaccel = get_hwaccel(*p);
		if (!hwaccel ||
			(ist->active_hwaccel_id && ist->active_hwaccel_id != hwaccel->id) ||
			(ist->hwaccel_id != HWACCEL_AUTO && ist->hwaccel_id != hwaccel->id))
			continue;

		ret = hwaccel->init(s, dxva2_master);
		if (ret < 0)
		{
			if (ist->hwaccel_id == hwaccel->id)
			{
				av_log(NULL, AV_LOG_FATAL,
					"[ffmpeg_dxva.cpp] %s hwaccel requested for input stream #%d:%d, "
					"but cannot be initialized.\n", hwaccel->name,
					ist->stream_index, ist->st->index);

				lock.unlock();
				return AV_PIX_FMT_NONE;
			}
			continue;
		}
		ist->active_hwaccel_id = hwaccel->id;
		ist->hwaccel_pix_fmt = *p;
		break;
	}
	lock.unlock();
	return *p;
}

int get_buffer(AVCodecContext *s, AVFrame *frame, int flags)
{
	boost::mutex::scoped_lock lock(dxva2_mutex);
	InputStream *ist = (InputStream *)s->opaque;

	if (ist->hwaccel_get_buffer && frame->format == ist->hwaccel_pix_fmt)
	{
		lock.unlock();
		return ist->hwaccel_get_buffer(s, frame, flags);
	}

	lock.unlock();
	return avcodec_default_get_buffer2(s, frame, flags);
}