//#include <stdio.h>
//#include "simptoolextern.h"

#ifndef SIMPTOOL_H
#define SIMPTOOL_H

#define _fp16_ 'm'
#define _fp32_ 'b'
#define _byt8_ 's'

#define _tta_ '1'
#define _notta_ '0'

#define _preproc_ 'A'
#define _postproc_ 'Z'


int stepinit_fixed = 0;
ScaleParam ScaleSteps[8];


#if defined(_MSC_VER) // needs to be first because msvc doesn't short-circuit after failing defined(__has_builtin)
#  define bswap16(x)     _byteswap_ushort((x))
#  define bswap32(x)     _byteswap_ulong((x))
#  define bswap64(x)     _byteswap_uint64((x))
#elif (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
#  define bswap16(x)     __builtin_bswap16((x))
#  define bswap32(x)     __builtin_bswap32((x))
#  define bswap64(x)     __builtin_bswap64((x))
#elif defined(__has_builtin) && __has_builtin(__builtin_bswap64)  /* for clang; gcc 5 fails on this and && shortcircuit fails; must be after GCC check */
#  define bswap16(x)     __builtin_bswap16((x))
#  define bswap32(x)     __builtin_bswap32((x))
#  define bswap64(x)     __builtin_bswap64((x))
#endif


char tmpname[0x20];
void dumpper(char* name,unsigned char* ptr,int wXc,int h)
{
    FILE* fi=fopen(name,"wb");
    fwrite(tmpname,1,0xF,fi);
    fwrite(ptr,1,wXc*h,fi);
    fclose(fi);

}



inline int LoadShaderSimp(uint32_t* dst,char* Name,char proc,char withTTA,char valuetype,int rdsize)
{
    Name[9]=proc;
    Name[10]=withTTA;
    Name[11]=valuetype;
    FILE* fi=fopen(Name,"rb");
	if (fi == 0)
	{
		return -1;
	}
    int rsz=fread(dst,1,rdsize*4,fi);
    fclose(fi);
    return rsz;

}




#endif
