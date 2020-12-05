// waifu2x implemented with ncnn library
#define DLL_EXPORT extern "C" __declspec(dllexport)


#ifndef WAIFU2X_H
#define WAIFU2X_H

const int fp16size = 2;
typedef struct scalpp
{
	float DstSize;
	char model;
	char mdl;
	char skl;
	char noiz;
} ScaleParam;


enum
{
	_first8to32_ = 1,
	_mid32to32_ = 0,
	_end32to8_ = 2,
	_simp8to8_ = 3

};


#include <string>

// ncnn
#include <net.h>
#include <gpu.h>
#include <layer.h>


class ncnnNetPack
{
public:
	ncnn::Net net;
	int scale;
	int prepadding;
	int noise;
};


class Waifu2x
{
public:
    Waifu2x(int gpuid, bool tta_mode = false);
    ~Waifu2x();


    int load(const int model, int scale, int noise);
	int load(const int model, int scale, int noise, char dst);


    int process(const ncnn::Mat& inimage, ncnn::Mat& outimage, const ScaleParam sparam, const int InOutType) const;



public:
    // waifu2x parameters


    int tilesize;

	ncnnNetPack nets[3];

private:

    ncnn::Pipeline* waifu2x_preproc[2];		//A and Y
    ncnn::Pipeline* waifu2x_postproc[2];	//B and Z
    ncnn::Layer* bicubic_2x;
    bool tta_mode;
};

#endif // WAIFU2X_H
