### Modifications
- Add models-ESRGAN for converted ESRGAN 4x models (if any), load them with char 'c' in 0modelset.rsr (See below).
- 
- Move spirv shaders to ./spv, instead of embedding them in the executable.
- spirv naming rule:  @@@xyz,
-    
- @@@=W2X(waifu2x) or RSR(RealSR),
- x=A(preproc) or Z(postproc) or B(middle-postproc) or Y(middle-preproc),
- y=1(with tta) or 0(no tta),
- z=b(fp32) or m(fp16, no int8) or s(fp16, with int8) or g(fp16in, int16out)
-    
- 6 extra .comp shader variants are added to src/etc, compile them with glslangValidator from https://github.com/KhronosGroup/glslang/releases
- command: glslangValidator -V -Os -o %1.spv %1
- Feel free to merge them with #ifdef .etc (I can't)
- 
- Pass /DBuildDLL (VC syntax) to build C api dll for C# .etc. I can't write cmakes.
- So I made a ZRealSR.cbp with both EXE and DLL targets, for CodeBlocks.
- SRnet.cs is an example C# prog to show you how to override FileSearching, imgDec, imgEnc funcs (if necessary), and run the proc.
- 
- When ./spv/RSRZ0g (no tta) or ./spv/RSRZ1g (with tta) WITHOUT FILE EXTENSION exists, it will produce 16bit PPM.
- 16bit PPM can be opened with Photoshop, irfanview.
- 
- When input output paths are both directories,  it won't exit after the firsttime processing was over. As there were new pics go into the input dir, it will process them automatically,  aka "Hot Folder" mode. ( you can remove the processed ones from input dir to reduce the file searching overhead).
- 
- Now we use 0modelset.rsr for model loading. see 3steps,2models.png for its structure.
- In this 8 byte struct, first 5 bytes are ScaleParam (see SRnet.cs), DstScale is the downsample factor for every step, and the last step can not be downsampled. (That means you can only get >4x results with multi steps)
- eg. for 7x upscaling, put 0.4375 in first DstScale, and 0 in the second one (0 means no downsampling, see the codes ), 4*0.4375*4=7.
- (byte)model is the index for model_list, 0xFF is the break sig for steps.
-
- Other 3 bytes are model_list, they are [model_id, nx, noise], nx and noise are not used in realsr.
- For model_id: 'a'=models-DF2K_JPEG, 'b'=models-DF2K, 'c'=models-ESRGAN. must be lower case. 0xFF is the break sig for model_list.



