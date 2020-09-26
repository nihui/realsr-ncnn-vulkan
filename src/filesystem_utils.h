#ifndef FILESYSTEM_UTILS_H
#define FILESYSTEM_UTILS_H

#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>

#if _WIN32
#include <windows.h>
#include "win32dirent.h"
#else // _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#endif // _WIN32

#if _WIN32
typedef std::wstring path_t;
#define PATHSTR(X) L##X
#else
typedef std::string path_t;
#define PATHSTR(X) X
#endif


#ifdef BuildDLL
#define FillPathListpp FillPathListFunc
BOOL WINAPI DllMain(
	HINSTANCE hinstDLL,  // handle to DLL module
	DWORD fdwReason,     // reason for calling function
	LPVOID lpReserved)  // reserved
{

	return TRUE;  // Successful DLL_PROCESS_ATTACH.
}
#else
#define FillPathListpp FillPathList
#endif


typedef struct whpack
{
	unsigned int w;
	unsigned int h;
} WHpack;

inline WHpack multipwh(const WHpack src,const float scale,const int mdlskl)
{
	WHpack rrtt;
	/*
	if (scale > 1.0f)
	{
		printf("\nrawwidth=%f\n", scale);
		rrtt.w = (unsigned int)scale;
		double rskl = ((double)scale) / ((double)src.w);
		rrtt.h = (unsigned int)(((double)src.h)*rskl);

		printf("\ndsth=%d\n", rrtt.h);
	}else
		*/
	if (scale == 0)
	{
		rrtt.w = src.w * mdlskl;
		rrtt.h = src.h * mdlskl;
	}
	else
	{
		
		float wji = src.w * mdlskl;
		float hji = src.h * mdlskl;
		wji = wji * scale + 0.5f;
		hji = hji * scale + 0.5f;

		rrtt.w = (unsigned int)wji;
		rrtt.h = (unsigned int)hji;

	}


	return rrtt;
}

inline char FillScaleParam(ScaleParam* dst,float skale,const char* modelset)
{

	FILE* fi = fopen(modelset, "rb");

	int relrd=fread(dst, 1, 0x40, fi)/8;
	fclose(fi);

	char maxuse = 0;
	for (int i = 0; i < relrd; i++)
	{


		if (dst[i].model == (char)0xFF)
		{
			stepinit_fixed = i;
			return maxuse;
		}
		else if (dst[i].model > maxuse)
		{
			maxuse = dst[i].model;

		}

	}

	stepinit_fixed = relrd;
	return maxuse;

	
}




#if _WIN32
#define Sleep_1024_ Sleep(1024)
WIN32_FIND_DATA FindFileData;
static bool FileExist(const path_t& path)
{
	
	HANDLE hFind;

	hFind = FindFirstFileExW(path.c_str(), FindExInfoStandard, &FindFileData,
		FindExSearchNameMatch, NULL, 0);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		//printf("\nNOT=%ls\n", path);
		return false;
	}
	else
	{

		//printf("\nYES=%ls\n", path);
		FindClose(hFind);
		return true;
	}
}

static bool path_is_directory(const path_t& path)
{
    DWORD attr = GetFileAttributesW(path.c_str());
    return (attr != INVALID_FILE_ATTRIBUTES) && (attr & FILE_ATTRIBUTE_DIRECTORY);
}

static int list_directory(const path_t& dirpath, std::vector<path_t>& imagepaths)
{
    imagepaths.clear();

    _WDIR* dir = _wopendir(dirpath.c_str());
    if (!dir)
    {
        fwprintf(stderr, L"opendir failed %ls\n", dirpath.c_str());
        return -1;
    }

    struct _wdirent* ent = 0;
    while ((ent = _wreaddir(dir)))
    {
        if (ent->d_type != DT_REG)
            continue;

        imagepaths.push_back(path_t(ent->d_name));
    }

    _wclosedir(dir);
    std::sort(imagepaths.begin(), imagepaths.end());

    return 0;
}
#else // _WIN32
#define Sleep_1024_ sleep(1)
#include <glob.h>
glob_t buf;
static bool FileExist(const path_t& path)
{
	buf.gl_pathc = 0;

	glob(path.c_str(), 0, NULL, &buf);
	if (buf.gl_pathc==0)
	{
		globfree(&buf);
		return false;
	}
	else
	{

		
		globfree(&buf);
		return true;
	}
}


static bool path_is_directory(const path_t& path)
{
    struct stat s;
    if (stat(path.c_str(), &s) != 0)
        return false;
    return S_ISDIR(s.st_mode);
}

static int list_directory(const path_t& dirpath, std::vector<path_t>& imagepaths)
{
    imagepaths.clear();

    DIR* dir = opendir(dirpath.c_str());
    if (!dir)
    {
        fprintf(stderr, "opendir failed %s\n", dirpath.c_str());
        return -1;
    }

    struct dirent* ent = 0;
    while ((ent = readdir(dir)))
    {
        if (ent->d_type != DT_REG)
            continue;

        imagepaths.push_back(path_t(ent->d_name));
    }

    closedir(dir);
    std::sort(imagepaths.begin(), imagepaths.end());

    return 0;
}
#endif // _WIN32

char tpstr[0x20];

#if _WIN32
void PGM16save(const wchar_t* outname, int w, int h, int chan,const unsigned char* data)
{
	FILE* fi = _wfopen(outname, L"wb");
#else // _WIN32
void PGM16save(const char* outname, int w, int h, int chan,const unsigned char* data)
{
	FILE* fi = fopen(outname, "wb");
#endif // _WIN32
	int hdrrll = sprintf(tpstr, "P6\n%d\n%d\n65535\n\0", w, h);
	fwrite(tpstr, 1, hdrrll, fi);
	fwrite(data, 1, w*h*chan * 2, fi);
	fclose(fi);

}


static path_t get_file_name_without_extension(const path_t& path)
{
    size_t dot = path.rfind(PATHSTR('.'));
    if (dot == path_t::npos)
        return path;

    return path.substr(0, dot);
}

inline int lastindexofslashs(path_t instr)
{
	int lyn = instr.size()-1;

	for (int i = lyn; i >= 0; i--)
	{
		if (instr[i] == PATHSTR('\\') || instr[i] == PATHSTR('/'))
			return i;



	}
    return path_t::npos;
}

static path_t extract_file_name_without_extension(const path_t& path)
{

	int lastslash = lastindexofslashs(path);

		if (lastslash == path_t::npos)
		{
			lastslash = 0;
		}


	size_t dot = path.rfind(PATHSTR('.'));
	if (dot == path_t::npos)
		return path;

	return path.substr(lastslash, dot- lastslash);
}

static path_t get_file_extension(const path_t& path)
{
    size_t dot = path.rfind(PATHSTR('.'));
    if (dot == path_t::npos)
        return path_t();

    return path.substr(dot + 1);
}

#if _WIN32
static path_t get_executable_directory()
{
    wchar_t filepath[256];
    GetModuleFileNameW(NULL, filepath, 256);

    wchar_t* backslash = wcsrchr(filepath, L'\\');
    backslash[1] = L'\0';

    return path_t(filepath);
}
#else // _WIN32
static path_t get_executable_directory()
{
    char filepath[256];
    readlink("/proc/self/exe", filepath, 256);

    char* slash = strrchr(filepath, '/');
    slash[1] = '\0';

    return path_t(filepath);
}
#endif // _WIN32

static bool filepath_is_readable(const path_t& path)
{
#if _WIN32
    FILE* fp = _wfopen(path.c_str(), L"rb");
#else // _WIN32
    FILE* fp = fopen(path.c_str(), "rb");
#endif // _WIN32
    if (!fp)
        return false;

    fclose(fp);
    return true;
}

static path_t sanitize_filepath(const path_t& path)
{
    if (filepath_is_readable(path))
        return path;

    return get_executable_directory() + path;
}


bool isDIR = false;

static int FillPathList(const path_t inputpath, const path_t outputpath, const path_t format, std::vector<path_t>* inout)
{
	if (inout[0].size() >0)
	{
		std::vector<path_t>().swap(inout[0]);
	}
	if (inout[1].size() > 0)
	{
		std::vector<path_t>().swap(inout[1]);
	}
	//printf("\nAFTERSWAPszin=%d, szout=%d\n", inout[0].size(), inout[1].size());


	bool inDir = true;
	bool outDir = true;

	if (!isDIR)
	{
		inDir = path_is_directory(inputpath);
		outDir = path_is_directory(outputpath);
	}

	if(isDIR ||(inDir && outDir))
	{
		isDIR = true;

		std::vector<path_t> filenames;
		int lr = list_directory(inputpath, filenames);
		if (lr != 0)
			return -1;

		//puts("\ndoloop000\n");
		const int count = filenames.size();
		//inout[0].resize(count);
		//inout[1].resize(count);

		//puts("\ndoloop001\n");
		path_t last_filename;
		path_t last_filename_noext;

		int realadd = 0;
		for (int i = 0; i < count; i++)
		{
			path_t filename = filenames[i];
			path_t filename_noext = get_file_name_without_extension(filename);
			path_t output_filename = filename_noext;

			// filename list is sorted, check if output image path conflicts
			if (filename_noext == last_filename_noext)
			{
				path_t output_filename2 = filename;

#if _WIN32
				fwprintf(stderr, L"both %ls and %ls output %ls ! %ls will output %ls\n", filename.c_str(), last_filename.c_str(), output_filename.c_str(), filename.c_str(), output_filename2.c_str());
#else
				fprintf(stderr, "both %s and %s output %s ! %s will output %s\n", filename.c_str(), last_filename.c_str(), output_filename.c_str(), filename.c_str(), output_filename2.c_str());
#endif

				output_filename = output_filename2;
			}
			else
			{
				last_filename = filename;
				last_filename_noext = filename_noext;
			}
			//puts("crahs?0");

			auto otfi = outputpath + PATHSTR('/') + output_filename;
			if (!FileExist(otfi + PATHSTR(".*")))
			{

				inout[0].push_back(inputpath + PATHSTR('/') + filename);
				inout[1].push_back(otfi+ PATHSTR('.') + format);
				realadd++;
			}
		}
		//puts("crahs?1");
		//printf("\nrealadd=%d\n",realadd);

		//printf("\nINREALADDszin=%d, szout=%d\n", inout[0].size(), inout[1].size());

		if (realadd == 0)
		{
			inout[0].clear();
			inout[1].clear();
		}
		else if (realadd < count)
		{
			inout[0].resize(realadd);
			inout[1].resize(realadd);

		}

		//puts("\ndoloop003\n");
		return inout[0].size();
	}
	else
	{
		if (inputpath[0] == '.')
			goto syrp;
		if(!FileExist(inputpath))
		{
			fprintf(stderr, "inputpath and outputpath must be either file or directory at the same time\n");
			return -1;
		}

		syrp:
		inout[0].push_back(inputpath);
		if (outDir)
		{

			inout[1].push_back(outputpath + PATHSTR('/') + extract_file_name_without_extension(inputpath) + PATHSTR('.') + format);
		}
		else
		{

			inout[1].push_back(outputpath);
		}



	}


	return inout[0].size();
}


#endif // FILESYSTEM_UTILS_H
