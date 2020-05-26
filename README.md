# RealSR ncnn Vulkan

ncnn implementation of Real-World Super-Resolution via Kernel Estimation and Noise Injection super resolution.

realsr-ncnn-vulkan uses [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.

## Usages

### Example Command

```shell
realsr-ncnn-vulkan.exe -i input.jpg -o output.png -s 4
```

### Full Usages

```console
Usage: realsr-ncnn-vulkan -i infile -o outfile [options]...

  -h                   show this help
  -v                   verbose output
  -i input-path        input image path (jpg/png) or directory
  -o output-path       output image path (png) or directory
  -s scale             upscale ratio (4, default=4)
  -t tile-size         tile size (>=32/0=auto, default=0)
  -m model-path        realsr model path (default=models-DF2K_JPEG)
  -g gpu-id            gpu device to use (default=0)
  -j load:proc:save    thread count for load/proc/save (default=1:2:2)
  -x                   enable tta mode
```

- `input-path` and `output-path` accept either file path or directory path
- `scale` = scale level, 4=upscale 4x
- `tile-size` = tile size, use smaller value to reduce GPU memory usage, default is 400
- `load:proc:save` = thread count for the three stages (image decoding + realsr upscaling + image encoding), use larger value may increase GPU utility and consume more GPU memory. You can tune this configuration as "4:4:4" for many small-size images, and "2:2:2" for large-size images. The default setting usually works fine for most situations. If you find that your GPU is hungry, do increase thread count to achieve faster processing.

If you encounter crash or error, try to upgrade your GPU driver

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx

## Sample Images

### Original Image

![origin](images/0.png)

### Upscale 4x with ImageMagick Lanczo4 Filter

```shell
convert origin.jpg -resize 400% output.png
```

![browser](images/im.png)

### Upscale 4x with srmd scale=4 noise=-1

```shell
srmd-ncnn-vulkan.exe -i origin.jpg -o 4x.png -s 4 -n -1
```

![waifu2x](images/srmd.png)

### Upscale 4x with realsr model=DF2K scale=4 tta=1

```shell
realsr-ncnn-vulkan.exe -i origin.jpg -o output.png -s 4 -x -m models-DF2K
```

![realsr](images/2.png)

## Original RealSR Project

- https://github.com/jixiaozhong/RealSR

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
- https://github.com/nothings/stb for decoding and encoding image on Linux / MacOS
- https://github.com/tronkko/dirent for listing files in directory on Windows
