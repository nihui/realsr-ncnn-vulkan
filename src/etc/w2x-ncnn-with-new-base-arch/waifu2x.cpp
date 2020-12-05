// waifu2x implemented with ncnn library


const int downsamptype = 3;
extern bool i16out;

#include "waifu2x.h"

#include <algorithm>
#include <vector>
#include "simptool.h"

//#define stdmin(X, Y) (((X) < (Y)) ? (X) : (Y))
//#define stdmax(X, Y) (((X) > (Y)) ? (X) : (Y))

//static int spvsizes[2];
static char spvnames[] =".\\spv\\W2XA0m\0";  //9,10,11
static const int largestspv=0x800;
static uint32_t preproc_spv[largestspv];
//static uint32_t postproc_spv[largestspv];


#if _WIN32
const wchar_t* modeldirs[] = { NULL, L".\\models-cunet",L".\\models-upconv_7_anime_style_art_rgb", L".\\models-upconv_7_photo" };

wchar_t parampath[128];
wchar_t modelpath[128];
#else // _WIN32
const char* modeldirs[] = { NULL, "./models-cunet","./models-upconv_7_anime_style_art_rgb", "./models-upconv_7_photo" };

#endif // _WIN32


inline int FillNetPack(ncnnNetPack& pack,int model,  int scale, int noise)
{
	//int prepadding = 0;
	pack.scale = scale;
	pack.noise = noise;
	if (model == 1)
	{
		if (noise == -1)
		{
			pack.prepadding = 18;
		}
		else if (scale == 1)
		{
			pack.prepadding = 28;
		}
		else if (scale == 2)
		{
			pack.prepadding = 18;
		}
	}
	else if (model == 2 || model == 3)
	{
		pack.prepadding = 7;
	}
	else
	{
		fprintf(stderr, "unknown model dir type\n");
		return -1;
	}


	if (noise == -1)
	{
		swprintf(parampath, 256, L"%s/scale2.0x_model.param", modeldirs[model]);
		swprintf(modelpath, 256, L"%s/scale2.0x_model.bin", modeldirs[model]);
	}
	else if (scale == 1)
	{
		swprintf(parampath, 256, L"%s/noise%d_model.param", modeldirs[model], noise);
		swprintf(modelpath, 256, L"%s/noise%d_model.bin", modeldirs[model], noise);
	}
	else if (scale == 2)
	{
		swprintf(parampath, 256, L"%s/noise%d_scale2.0x_model.param", modeldirs[model], noise);
		swprintf(modelpath, 256, L"%s/noise%d_scale2.0x_model.bin", modeldirs[model], noise);
	}


	//printf("mdl=%ls", modelpath);
	{
		FILE* fp = _wfopen(parampath, L"rb");
		if (!fp)
		{
			fwprintf(stderr, L"_wfopen %ls failed\n", parampath);
			return -1;
		}

		pack.net.load_param(fp);

		fclose(fp);
	}
	{
		FILE* fp = _wfopen(modelpath, L"rb");
		if (!fp)
		{
			fwprintf(stderr, L"_wfopen %ls failed\n", modelpath);
			return -1;
		}

		pack.net.load_model(fp);

		fclose(fp);
	}



	return 0;
}




Waifu2x::Waifu2x(int gpuid, bool _tta_mode)
{
	for (int vv = 0; vv < 3; vv++)
	{
		nets[vv].net.opt.use_vulkan_compute = true;
		nets[vv].net.opt.use_fp16_packed = true;
		nets[vv].net.opt.use_fp16_storage = true;
		nets[vv].net.opt.use_fp16_arithmetic = false;
		nets[vv].net.opt.use_int8_storage = true;
		nets[vv].net.opt.use_int8_arithmetic = false;

		nets[vv].net.set_vulkan_device(gpuid);
	}

	for (int kkk = 0; kkk < 2; kkk++)
	{
		waifu2x_preproc[kkk] = 0;
		waifu2x_postproc[kkk] = 0;
	}
    bicubic_2x = 0;
    tta_mode = _tta_mode;
}

Waifu2x::~Waifu2x()
{
    // cleanup preprocess and postprocess pipeline
    {
		for (int kkk = 0; kkk < 2; kkk++)
		{
			delete waifu2x_preproc[kkk];
			delete waifu2x_postproc[kkk];
		}
    }

    bicubic_2x->destroy_pipeline(nets[0].net.opt);
    delete bicubic_2x;
}



int Waifu2x::load(const int model, int scale, int noise,char dst)
{
	return FillNetPack(nets[dst], model, scale, noise);
}

int Waifu2x::load(const int model, int scale, int noise)
{
	FillNetPack(nets[0], model, scale, noise);

	//auto net = nets[0].net;
   //prepad..etc

    // initialize preprocess and postprocess pipeline
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);

        specializations[0].i = 1;

		for (int kkk = 0; kkk < 2; kkk++)
		{
			waifu2x_preproc[kkk] = new ncnn::Pipeline(nets[0].net.vulkan_device());
			waifu2x_preproc[kkk]->set_optimal_local_size_xyz(32, 32, 3);

			waifu2x_postproc[kkk] = new ncnn::Pipeline(nets[0].net.vulkan_device());
			waifu2x_postproc[kkk]->set_optimal_local_size_xyz(32, 32, 3);
		}



        int rszz;

        if (tta_mode)
        {


            if (nets[0].net.opt.use_fp16_storage && nets[0].net.opt.use_int8_storage)
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_tta_,_byt8_,largestspv);
            else if (nets[0].net.opt.use_fp16_storage)
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_tta_,_fp16_,largestspv);
            else
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_tta_,_fp32_,largestspv);

            waifu2x_preproc[0]->create(preproc_spv, rszz, specializations);
			rszz = LoadShaderSimp(preproc_spv, spvnames, 'Y', _tta_, _byt8_, largestspv);
			waifu2x_preproc[1]->create(preproc_spv, rszz, std::vector<ncnn::vk_specialization_type>(0));


			if (nets[0].net.opt.use_fp16_storage && nets[0].net.opt.use_int8_storage)
			{
				rszz = LoadShaderSimp(preproc_spv, spvnames, _postproc_, _tta_, 'g', largestspv);
				if (rszz > 0)
				{
					i16out = true;
				}
				else
				{
					i16out = false;
					rszz = LoadShaderSimp(preproc_spv, spvnames, _postproc_, _tta_, _byt8_, largestspv);
				}


			}
            else if (nets[0].net.opt.use_fp16_storage)
                rszz=LoadShaderSimp(preproc_spv,spvnames,_postproc_,_tta_,_fp16_,largestspv);
            else
                rszz=LoadShaderSimp(preproc_spv,spvnames,_postproc_,_tta_,_fp32_,largestspv);

			if (i16out)
			{
				waifu2x_postproc[0]->create(preproc_spv, rszz, std::vector<ncnn::vk_specialization_type>(0));
			}
			else
			{
				waifu2x_postproc[0]->create(preproc_spv, rszz, specializations);
			}

			rszz = LoadShaderSimp(preproc_spv, spvnames, 'B', _tta_, _byt8_, largestspv);
			waifu2x_postproc[1]->create(preproc_spv, rszz, std::vector<ncnn::vk_specialization_type>(0));
        }
        else
        {
            if (nets[0].net.opt.use_fp16_storage && nets[0].net.opt.use_int8_storage)
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_notta_,_byt8_,largestspv);
            else if (nets[0].net.opt.use_fp16_storage)
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_notta_,_fp16_,largestspv);
            else
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_notta_,_fp32_,largestspv);

            waifu2x_preproc[0]->create(preproc_spv, rszz, specializations);
			rszz = LoadShaderSimp(preproc_spv, spvnames, 'Y', _notta_, _byt8_, largestspv);
			waifu2x_preproc[1]->create(preproc_spv, rszz, std::vector<ncnn::vk_specialization_type>(0));

			if (nets[0].net.opt.use_fp16_storage && nets[0].net.opt.use_int8_storage)
			{
				rszz = LoadShaderSimp(preproc_spv, spvnames, _postproc_, _notta_, 'g', largestspv);
				if (rszz > 0)
				{
					i16out = true;
				}
				else
				{
					i16out = false;
					rszz = LoadShaderSimp(preproc_spv, spvnames, _postproc_, _notta_, _byt8_, largestspv);
				}
			}
            else if (nets[0].net.opt.use_fp16_storage)
                rszz=LoadShaderSimp(preproc_spv,spvnames,_postproc_,_notta_,_fp16_,largestspv);
            else
                rszz=LoadShaderSimp(preproc_spv,spvnames,_postproc_,_notta_,_fp32_,largestspv);

			if (i16out)
			{
				waifu2x_postproc[0]->create(preproc_spv, rszz, std::vector<ncnn::vk_specialization_type>(0));
			}
			else
			{
				waifu2x_postproc[0]->create(preproc_spv, rszz, specializations);
			}

			rszz = LoadShaderSimp(preproc_spv, spvnames, 'B', _notta_, _byt8_, largestspv);
			waifu2x_postproc[1]->create(preproc_spv, rszz, std::vector<ncnn::vk_specialization_type>(0));
        }


/*
        if (tta_mode)
        {


            if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_tta_,_byt8_,largestspv);
            else if (net.opt.use_fp16_storage)
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_tta_,_fp16_,largestspv);
            else
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_tta_,_fp32_,largestspv);

            waifu2x_preproc->create(preproc_spv, rszz, specializations);

            if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                rszz=LoadShaderSimp(postproc_spv,spvnames,_postproc_,_tta_,_byt8_,largestspv);
            else if (net.opt.use_fp16_storage)
                rszz=LoadShaderSimp(postproc_spv,spvnames,_postproc_,_tta_,_fp16_,largestspv);
            else
                rszz=LoadShaderSimp(postproc_spv,spvnames,_postproc_,_tta_,_fp32_,largestspv);

            waifu2x_postproc->create(postproc_spv, rszz, specializations);
        }
        else
        {
            if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_notta_,_byt8_,largestspv);
            else if (net.opt.use_fp16_storage)
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_notta_,_fp16_,largestspv);
            else
                rszz=LoadShaderSimp(preproc_spv,spvnames,_preproc_,_notta_,_fp32_,largestspv);

            waifu2x_preproc->create(preproc_spv, rszz, specializations);

            if (net.opt.use_fp16_storage && net.opt.use_int8_storage)
                rszz=LoadShaderSimp(postproc_spv,spvnames,_postproc_,_notta_,_byt8_,largestspv);
            else if (net.opt.use_fp16_storage)
                rszz=LoadShaderSimp(postproc_spv,spvnames,_postproc_,_notta_,_fp16_,largestspv);
            else
                rszz=LoadShaderSimp(postproc_spv,spvnames,_postproc_,_notta_,_fp32_,largestspv);

            waifu2x_postproc->create(postproc_spv, rszz, specializations);
        }
        */

    }

    // bicubic 2x for alpha channel
    {
        bicubic_2x = ncnn::create_layer("Interp");
        bicubic_2x->vkdev = nets[0].net.vulkan_device();

        ncnn::ParamDict pd;
        pd.set(0, downsamptype);// bicubic
        pd.set(1, 2.f);
        pd.set(2, 2.f);
        bicubic_2x->load_param(pd);

        bicubic_2x->create_pipeline(nets[0].net.opt);
    }

    return 0;
}

int Waifu2x::process(const ncnn::Mat& inimage, ncnn::Mat& outimage, const ScaleParam sparam, const int InOutType) const
{

	
	//==============
	//memcpy(outimage.data, inimage.data, inimage.w*inimage.h*inimage.elempack);
	//return 0;
	//==============


	int yoffcache = 0;
	const int scale = nets[sparam.model].scale;
	const int noise = nets[sparam.model].noise;
	const int prepadding = nets[sparam.model].prepadding;

    //printf("skale=%d\n",scale);

    if (noise == -1 && scale == 1)
    {
        outimage = inimage;
        return 0;
    }

    const unsigned char* pixeldata = (const unsigned char*)inimage.data;
    const int w = inimage.w;
    const int h = inimage.h;
    int channels = inimage.elempack;



	//printf("\npass=%d, channel=%d\n", pss, channels);
	//pss++;

	int preproctype = 0;
	int postproctype = 0;

	if (InOutType == _first8to32_)
	{
		preproctype = 0;
		postproctype = 1;
	}
	else if (InOutType == _mid32to32_)
	{
		preproctype = 1;
		postproctype = 1;
	}
	else if (InOutType == _end32to8_)
	{
		preproctype = 1;
		postproctype = 0;
	}

	if (preproctype == 1)
	{
		//printf("\nchar=%d\n", channels);
		channels = inimage.c;
	}

	bool doPostScale = false;
	const int pre_srcchansize = inimage.cstep*inimage.elemsize;
	const int post_dstchansize = outimage.cstep*outimage.elemsize;

	double postscale;
	const int basicinimages = inimage.w*scale;
	if (outimage.w != basicinimages)
	{
		postscale = (double)outimage.w / (double)basicinimages;

		//printf("\npostscale=%f\n", postscale);
		doPostScale = true;

	}

    const int TILE_SIZE_X = tilesize;
    const int TILE_SIZE_Y = tilesize;

    ncnn::VkAllocator* blob_vkallocator = nets[sparam.model].net.vulkan_device()->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = nets[sparam.model].net.vulkan_device()->acquire_staging_allocator();

    ncnn::Option opt = nets[sparam.model].net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // each tile 400x400
    const int xtiles = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, h) - yi * TILE_SIZE_Y;


		//printf("\ny=%d, h_nopad=%d\n", yi, tile_h_nopad);

        int prepadding_bottom = prepadding;
        if (scale == 1)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale >= 2)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }


        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, h);

        ncnn::Mat in;
        if (opt.use_fp16_storage && opt.use_int8_storage)
        {

			if (preproctype==1)
			{

				in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (int)channels, (size_t)fp16size);


				int pre_dstchansize = in.cstep*in.elemsize;



				int offzet = in_tile_y0 * w * fp16size;
				for (int iiu = 0; iiu < channels; iiu++)
				{
					memcpy((unsigned char*)(in.data) + pre_dstchansize * iiu, pixeldata + pre_srcchansize * iiu + offzet, pre_dstchansize);
				}


			}
			else
			{
				in = ncnn::Mat(w, (in_tile_y1 - in_tile_y0), (unsigned char*)pixeldata + in_tile_y0 * w * channels, (size_t)channels, 1);



			}


        }
        else
        {
            if (channels == 3)
            {

                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGR2RGB, w, (in_tile_y1 - in_tile_y0));

            }
            if (channels == 4)
            {

                in = ncnn::Mat::from_pixels(pixeldata + in_tile_y0 * w * channels, ncnn::Mat::PIXEL_BGRA2RGBA, w, (in_tile_y1 - in_tile_y0));

            }
        }

        ncnn::VkCompute cmd(nets[sparam.model].net.vulkan_device());

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h);

        ncnn::VkMat out_gpu;
        if (opt.use_fp16_storage && opt.use_int8_storage)
        {
			if (postproctype==1)
			{
				out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t)2u, 1, blob_vkallocator);
			}
			else if (i16out)
			{
				out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)channels*fp16size, fp16size, blob_vkallocator);
				//out_gpu.elemsize /= 2;
			}
			else
			{
				out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, (size_t)channels, 1, blob_vkallocator);
			}

        }
        else
        {
            out_gpu.create(w * scale, (out_tile_y1 - out_tile_y0) * scale, channels, (size_t)4u, 1, blob_vkallocator);
        }

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, w) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 1)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale >= 2)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }



            if (tta_mode)
            {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                    if (channels == 4)
                    {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];
                    bindings[9] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(waifu2x_preproc[preproctype], bindings, constants, dispatcher);
                }

                // waifu2x

                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++)
                {

                    ncnn::Extractor ex = nets[sparam.model].net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("Input1", in_tile_gpu[ti]);
                    ex.extract("Eltwise4", out_tile_gpu[ti], cmd);

                }

                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4)
                {
                    if (scale == 1)
                    {
                        out_alpha_tile_gpu = in_alpha_tile_gpu;
                    }
                    if (scale == 2)
                    {
                        bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                }

                // postproc
                {
                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = out_tile_gpu[0];
                    bindings[1] = out_tile_gpu[1];
                    bindings[2] = out_tile_gpu[2];
                    bindings[3] = out_tile_gpu[3];
                    bindings[4] = out_tile_gpu[4];
                    bindings[5] = out_tile_gpu[5];
                    bindings[6] = out_tile_gpu[6];
                    bindings[7] = out_tile_gpu[7];
                    bindings[8] = out_alpha_tile_gpu;
                    bindings[9] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = out_tile_gpu[0].w;
                    constants[1].i = out_tile_gpu[0].h;
                    constants[2].i = out_tile_gpu[0].cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = channels;
                    constants[9].i = out_alpha_tile_gpu.w;
                    constants[10].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(waifu2x_postproc[postproctype], bindings, constants, dispatcher);
                }
            }
            else
            {
                // preproc
                ncnn::VkMat in_tile_gpu;
                ncnn::VkMat in_alpha_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, w) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, h) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                    if (channels == 4)
                    {
                        in_alpha_tile_gpu.create(tile_w_nopad, tile_h_nopad, 1, in_out_tile_elemsize, 1, blob_vkallocator);
                    }

                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    bindings[2] = in_alpha_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(13);

                    //printf("in=%d\n",in_tile_gpu.w);

                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = channels;
                    constants[11].i = in_alpha_tile_gpu.w;
                    constants[12].i = in_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(waifu2x_preproc[preproctype], bindings, constants, dispatcher);
                }



                ncnn::VkMat out_tile_gpu;


                    ncnn::Extractor ex = nets[sparam.model].net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("Input1", in_tile_gpu);



                    ex.extract("Eltwise4", out_tile_gpu, cmd);



                ncnn::VkMat out_alpha_tile_gpu;
                if (channels == 4)
                {
                    if (scale == 1)
                    {
                        out_alpha_tile_gpu = in_alpha_tile_gpu;
                    }
                    if (scale == 2)
                    {
                        bicubic_2x->forward(in_alpha_tile_gpu, out_alpha_tile_gpu, cmd, opt);
                    }
                }

                // postproc
                {
                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = out_tile_gpu;
                    bindings[1] = out_alpha_tile_gpu;
                    bindings[2] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);



                    constants[0].i = out_tile_gpu.w;
                    constants[1].i = out_tile_gpu.h;
                    constants[2].i = out_tile_gpu.cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;

                    auto offset_x =xi * TILE_SIZE_X * scale;
                    constants[6].i = offset_x;

                    auto gx_max = std::min(TILE_SIZE_X * scale   ,   out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[7].i = gx_max;
                    constants[8].i = channels;
                    constants[9].i = out_alpha_tile_gpu.w;
                    constants[10].i = out_alpha_tile_gpu.h;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = channels;

                    cmd.record_pipeline(waifu2x_postproc[postproctype], bindings, constants, dispatcher);
                }
            }//tta-end

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }   //===xloop



		ncnn::VkMat out_skl_gpu;
		if (doPostScale)
		{
			double dsthi = (double)out_gpu.h*postscale+0.5;
			ncnn::ParamDict pd;
			pd.set(0, downsamptype);// downsamp
			pd.set(3, (int)dsthi);
			pd.set(4, outimage.w);
			bicubic_2x->load_param(pd);

			//puts("whyrecre1");
			//bicubic_2x->create_pipeline(nets[0].net.opt);
			//puts("whyrecre2");


			bicubic_2x->forward(out_gpu, out_skl_gpu, cmd, opt);
		}


        // download
        {
            ncnn::Mat out;

            unsigned char* otptr = (unsigned char*)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels;



            if (opt.use_fp16_storage && opt.use_int8_storage)
            {
				if (postproctype==1)
				{

				}
				else if (i16out)
				{
					out = ncnn::Mat(out_gpu.w, out_gpu.h, (unsigned short*)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, (size_t)channels*fp16size, fp16size);
				}
				else
				{
					out = ncnn::Mat(out_gpu.w, out_gpu.h, otptr, (size_t)channels, 1);
				}

            }


			if (doPostScale)
			{
				cmd.record_clone(out_skl_gpu, out, opt);
			}
			else
			{
				cmd.record_clone(out_gpu, out, opt);
			}


            cmd.submit_and_wait();

			if(postproctype==1)
			{

				int post_srcchansize = out.cstep*out.elemsize;

				unsigned char* outimgdata = (unsigned char*)(outimage.data);



				int offzet = yi * scale * TILE_SIZE_Y * w * scale * fp16size;
				int cpsiz = post_srcchansize;
				if (doPostScale)
				{

					offzet = yoffcache* outimage.w* fp16size;
					yoffcache += out.h;
					int extraha = outimage.h - yoffcache;
					if (extraha < 0)
					{
						cpsiz += extraha * outimage.w*fp16size;
					}
				}




				for (int iiu = 0; iiu < channels; iiu++)
				{
					memcpy(outimgdata + post_dstchansize * iiu + offzet, (unsigned char*)(out.data) + post_srcchansize * iiu, cpsiz);
				}


			}
			



            if (!(opt.use_fp16_storage && opt.use_int8_storage))
            {

				puts("\ndontGohere=939\n");
                if (channels == 3)
                {
                    //puts("prettoxi");
                    out.to_pixels((unsigned char*)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGB2BGR);
                    //puts("aftrrttoxi");
                }
                if (channels == 4)
                {

                    out.to_pixels((unsigned char*)outimage.data + yi * scale * TILE_SIZE_Y * w * scale * channels, ncnn::Mat::PIXEL_RGBA2BGRA);

                }
            }
        }
    }

	nets[sparam.model].net.vulkan_device()->reclaim_blob_allocator(blob_vkallocator);
	nets[sparam.model].net.vulkan_device()->reclaim_staging_allocator(staging_vkallocator);



    return 0;
}


