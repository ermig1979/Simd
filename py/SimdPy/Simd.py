import argparse
from dataclasses import dataclass
import os
import ctypes
import pathlib
import sys
import array
import enum

Simd = sys.modules[__name__]

###################################################################################################

## @ingroup python
# Describes type of description which can return function Simd.Lib.CpuDesc.
class CpuDesc(enum.Enum) :
	## A CPU model name.
	Model = 0 

## @ingroup python
# Describes type of information which can return function Simd.Lib.CpuInfo.
class CpuInfo(enum.Enum) :	
	## A system sockets number.
	Sockets = 0
	## A number of physical CPU cores.
	Cores = 1
	## A number of logical CPU cores.
	Threads = 2
	## A size of level 1 data cache.
	CacheL1 = 3
	## A size of level 2 cache.
	CacheL2 = 4
	## A size of level 3 cache.
	CacheL3 = 5
	## A size of system memory.
	RAM = 6
	## Enabling of SSE, SSE2, SSE3, SSSE3, SSE4.1 CPU extensions (x86 specific).
	SSE41 = 7
	## Enabling of AVX CPU extensions (x86 specific).
	AVX = 8
	## Enabling of AVX2, FMA CPU extensions (x86 specific).
	AVX2 = 9
	## Enabling of AVX-512F, AVX-512BW CPU extensions (x86 specific).
	AVX512BW = 10
	## Enabling of AVX-512VNNI CPU extensions (x86 specific).
	AVX512VNNI = 11
	## Enabling of AVX-512BF16 CPU extensions (x86 specific).
	AVX512BF16 = 12
	## Enabling of AMX CPU extensions (x86 specific).
	AMX = 13
	## Enabling of VMX (Altivec) CPU extensions (PowerPC specific).
	VMX = 14
	## Enabling of VSX (Power 7) CPU extensions (PowerPC specific).
	VSX = 15
	## Enabling of NEON CPU extensions (ARM specific).
	NEON = 16
	
## @ingroup python
# Describes formats of image file. It is used in functions Simd.ImageSaveToMemory and Simd.ImageSaveToFile.
class ImageFile(enum.Enum) :	
    ## An undefined image file format (format auto choice).
    Undefined = 0
    ## A PGM (Portable Gray Map) text (P2) image file format.
    PgmTxt = 1
    ## A PGM (Portable Gray Map) binary (P5) image file format.
    PgmBin = 2
    ## A PGM (Portable Pixel Map) text (P3) image file format.
    PpmTxt = 3
    ## A PGM (Portable Pixel Map) binary (P6) image file format.
    PpmBin = 4
    ## A PNG (Portable Network Graphics) image file format.
    Png = 5
    ## A JPEG (Joint Photographic Experts Group) image file format.
    Jpeg = 6

## @ingroup python
# Describes pixel format type. It is used in Simd.Image.
class PixelFormat(enum.Enum) :
	## Undefined pixel format.
	Empty = 0
	## A 8-bit gray pixel format.
	Gray8 = 1
    ## A 16-bit (2 8-bit channels) pixel format (UV plane of NV12 pixel format).
	Uv16 = 2
    ## A 24-bit (3 8-bit channels) BGR (Blue, Green, Red) pixel format.
	Bgr24 = 3
    ## A 32-bit (4 8-bit channels) BGRA (Blue, Green, Red, Alpha) pixel format.
	Bgra32 = 4
    ## A single channel 16-bit integer pixel format.
	Int16 = 5
    ## A single channel 32-bit integer pixel format.
	Int32 = 6
    ## A single channel 64-bit integer pixel format.
	Int64 = 7
    ## A single channel 32-bit float point pixel format.
	Float = 8
    ## A single channel 64-bit float point pixel format.
	Double = 9
    ## A 8-bit Bayer pixel format (GRBG).
	BayerGrbg = 10
    ## A 8-bit Bayer pixel format (GBRG).
	BayerGbrg = 11
    ## A 8-bit Bayer pixel format (RGGB).
	BayerRggb = 12
    ## A 8-bit Bayer pixel format (BGGR).
	BayerBggr = 13
    ## A 24-bit (3 8-bit channels) HSV (Hue, Saturation, Value) pixel format.
	Hsv24 = 14
    ## A 24-bit (3 8-bit channels) HSL (Hue, Saturation, Lightness) pixel format.
	Hsl24 = 15
    ## A 24-bit (3 8-bit channels) RGB (Red, Green, Blue) pixel format.
	Rgb24 = 16
    ## A 32-bit (4 8-bit channels) RGBA (Red, Green, Blue, Alpha) pixel format.
	Rgba32 = 17
    ## A 16-bit (2 8-bit channels) UYVY422 pixel format.
	Uyvy16 = 18
    ## A 32-bit (4 8-bit channels) ARGB (Alpha, Red, Green, Blue) pixel format.
	Argb32 = 19
	
	## Gets pixel size in bytes.
	# @return pixel size in bytes.	
	def PixelSize(self) -> int :
		if self == Simd.PixelFormat.Empty : return 0
		elif self == Simd.PixelFormat.Gray8 : return 1
		elif self == Simd.PixelFormat.Uv16 : return 2
		elif self == Simd.PixelFormat.Bgr24 : return 3
		elif self == Simd.PixelFormat.Bgra32 : return 4
		elif self == Simd.PixelFormat.Int16 : return 2
		elif self == Simd.PixelFormat.Int32 : return 4
		elif self == Simd.PixelFormat.Int64 : return 8
		elif self == Simd.PixelFormat.Float : return 4
		elif self == Simd.PixelFormat.Double : return 8
		elif self == Simd.PixelFormat.BayerGrbg : return 1	
		elif self == Simd.PixelFormat.BayerGbrg : return 1
		elif self == Simd.PixelFormat.BayerRggb : return 1
		elif self == Simd.PixelFormat.BayerBggr : return 1
		elif self == Simd.PixelFormat.Hsv24 : return 3
		elif self == Simd.PixelFormat.Hsl24 : return 3
		elif self == Simd.PixelFormat.Rgb24 : return 3
		elif self == Simd.PixelFormat.Rgba32 : return 4
		elif self == Simd.PixelFormat.Uyvy16 : return 2
		elif self == Simd.PixelFormat.Argb32 : return 4
		else : return 0
		
	## Gets channel size in bytes.
	# @return channel size in bytes.	
	def ChannelSize(self) -> int :
		if self == Simd.PixelFormat.Empty : return 0
		elif self == Simd.PixelFormat.Gray8 : return 1
		elif self == Simd.PixelFormat.Uv16 : return 1
		elif self == Simd.PixelFormat.Bgr24 : return 1
		elif self == Simd.PixelFormat.Bgra32 : return 1
		elif self == Simd.PixelFormat.Int16 : return 2
		elif self == Simd.PixelFormat.Int32 : return 4
		elif self == Simd.PixelFormat.Int64 : return 8
		elif self == Simd.PixelFormat.Float : return 4
		elif self == Simd.PixelFormat.Double : return 8
		elif self == Simd.PixelFormat.BayerGrbg : return 1	
		elif self == Simd.PixelFormat.BayerGbrg : return 1
		elif self == Simd.PixelFormat.BayerRggb : return 1
		elif self == Simd.PixelFormat.BayerBggr : return 1
		elif self == Simd.PixelFormat.Hsv24 : return 1
		elif self == Simd.PixelFormat.Hsl24 : return 1
		elif self == Simd.PixelFormat.Rgb24 : return 1
		elif self == Simd.PixelFormat.Rgba32 : return 1
		elif self == Simd.PixelFormat.Uyvy16 : return 1
		elif self == Simd.PixelFormat.Argb32 : return 1
		else : return 0
		
	## Gets channels count.
	# @return channels count.	
	def ChannelCount(self) -> int :
		if self == Simd.PixelFormat.Empty : return 0
		elif self == Simd.PixelFormat.Gray8 : return 1
		elif self == Simd.PixelFormat.Uv16 : return 2
		elif self == Simd.PixelFormat.Bgr24 : return 3
		elif self == Simd.PixelFormat.Bgra32 : return 4
		elif self == Simd.PixelFormat.Int16 : return 1
		elif self == Simd.PixelFormat.Int32 : return 1
		elif self == Simd.PixelFormat.Int64 : return 1
		elif self == Simd.PixelFormat.Float : return 1
		elif self == Simd.PixelFormat.Double : return 1
		elif self == Simd.PixelFormat.BayerGrbg : return 1	
		elif self == Simd.PixelFormat.BayerGbrg : return 1
		elif self == Simd.PixelFormat.BayerRggb : return 1
		elif self == Simd.PixelFormat.BayerBggr : return 1
		elif self == Simd.PixelFormat.Hsv24 : return 3
		elif self == Simd.PixelFormat.Hsl24 : return 3
		elif self == Simd.PixelFormat.Rgb24 : return 3
		elif self == Simd.PixelFormat.Rgba32 : return 4
		elif self == Simd.PixelFormat.Uyvy16 : return 2
		elif self == Simd.PixelFormat.Argb32 : return 4
		else : return 0

## @ingroup python
# Describes the position of the child sub image relative to the parent image.
# This enum is used for creation of sub image view in method Simd.Image.RegionAt.
class Position(enum.Enum) :
	## A position in the top-left corner.
	TopLeft = 0
	## A position at the top center.
	TopCenter = 1 
	## A position in the top-right corner.
	TopRight = 2 
	## A position of the left in the middle.
	MiddleLeft = 3
	## A central position.
	MiddleCenter = 4 
	## A position of the right in the middle.
	MiddleRight = 5
	## A position in the bottom-left corner.
	BottomLeft = 6
	## A position at the bottom center.
	BottomCenter = 7 
	## A position in the bottom-right corner.
	BottomRight = 8 
	
## @ingroup python
# Describes resized image channel types.
class ResizeChannel(enum.Enum) :
    ## 8-bit integer channel type.
    Byte = 0
    ## 16-bit integer channel type.
    Short = 1
    ## 32-bit float channel type.
    Float = 2
	
## @ingroup python
# Describes methods used in order to resize image.
class ResizeMethod(enum.Enum) :
    ## Nearest method.
	Nearest = 0
    ## Nearest Pytorch compatible method.
	NearestPytorch = 1
    ## Bilinear method.
	Bilinear = 2
    ## Bilinear Caffe compatible method. It is relevant only for Simd.ResizeChannel.Float (32-bit float channel type).
	BilinearCaffe = 3
    ## Bilinear Pytorch compatible method. It is relevant only for Simd.ResizeChannel.Float (32-bit float channel type).
	BilinearPytorch = 4
    ## Bicubic method.
	Bicubic = 5
    ## Area method.
	Area = 6
    ## Area method for previously reduced in 2 times image.
	AreaFast = 7
	
## @ingroup python
# 4D-tensor format type.
class TensorFormat(enum.Enum) :
	## Unknown tensor format.
	Unknown = -1 
	## NCHW (N - batch, C - channels, H - height, W - width) 4D-tensor format of (input/output) image.
	Nchw = 0  
	## NHWC (N - batch, H - height, W - width, C - channels) 4D-tensor format of (input/output) image.
	Nhwc = 1
	
## @ingroup python
# Describes tensor data type.
class TensorData(enum.Enum) :
	## Unknown tensor data type.
	Unknown = -1 
	## 32-bit floating point (Single Precision).
	FP32 = 0 
	## 32-bit signed integer.
	INT32 = 0 
	## 8-bit signed integer.
	INT8 = 1 
	## 8-bit unsigned integer.
	UINT8 = 2 
	## 64-bit signed integer.
	INT64 = 3 
	## 64-bit unsigned integer.
	UINT64 = 4 
	## 8-bit Boolean.
	BOOL = 5 
	## 16-bit BFloat16 (Brain Floating Point).
	BF16 = 6 
	## 16-bit floating point (Half Precision).
	FP16 = 7 

###################################################################################################

## @ingroup python
# A wrapper around %Simd Library API.
class Lib():
	__lib : ctypes.CDLL
	lib : ctypes.CDLL
	
	## Initializes Simd.Lib class (loads %Simd Library binaries).
	# @note This method must be called before any using of this class.
	# @param dir - a directory with %Simd Library binaries (Simd.dll or libSimd.so).
	def Init(dir: str):
		if not os.path.isdir(dir):
			raise Exception("Directory '{0}' with binaries is not exist!".format(dir))
		name : str
		if sys.platform == 'win32':
			name = "Simd.dll"
		else :
			name = "libSimd.so"
		path = str(pathlib.Path(dir).absolute() / name)
		if not os.path.isfile(path):
			raise Exception("Binary file '{0}' is not exist!".format(path))

		Lib.__lib = ctypes.CDLL(path)
		
		Lib.__lib.SimdVersion.argtypes = []
		Lib.__lib.SimdVersion.restype = ctypes.c_char_p 
		
		Lib.__lib.SimdCpuDesc.argtypes = [ ctypes.c_int ]
		Lib.__lib.SimdCpuDesc.restype = ctypes.c_char_p 
		
		Lib.__lib.SimdCpuInfo.argtypes = [ ctypes.c_int ]
		Lib.__lib.SimdCpuInfo.restype = ctypes.c_size_t 
		
		Lib.__lib.SimdPerformanceStatistic.argtypes = []
		Lib.__lib.SimdPerformanceStatistic.restype = ctypes.c_char_p 
		
		Lib.__lib.SimdAllocate.argtypes = [ ctypes.c_size_t, ctypes.c_size_t ]
		Lib.__lib.SimdAllocate.restype = ctypes.c_void_p 
		
		Lib.__lib.SimdFree.argtypes = [ ctypes.c_void_p ]
		Lib.__lib.SimdFree.restype = None
		
		Lib.__lib.SimdAlign.argtypes = [ ctypes.c_size_t, ctypes.c_size_t ]
		Lib.__lib.SimdAlign.restype = ctypes.c_size_t 
		
		Lib.__lib.SimdAlignment.argtypes = []
		Lib.__lib.SimdAlignment.restype = ctypes.c_size_t 
		
		Lib.__lib.SimdRelease.argtypes = [ ctypes.c_void_p ]
		Lib.__lib.SimdRelease.restype = None
		
		Lib.__lib.SimdGetThreadNumber.argtypes = []
		Lib.__lib.SimdGetThreadNumber.restype = ctypes.c_size_t 

		Lib.__lib.SimdSetThreadNumber.argtypes = [ ctypes.c_size_t ]
		Lib.__lib.SimdSetThreadNumber.restype = None 
		
		Lib.__lib.SimdEmpty.argtypes = []
		Lib.__lib.SimdEmpty.restype = None
		
		Lib.__lib.SimdGetFastMode.argtypes = []
		Lib.__lib.SimdGetFastMode.restype = ctypes.c_bool
		
		Lib.__lib.SimdSetFastMode.argtypes = [ ctypes.c_bool ]
		Lib.__lib.SimdSetFastMode.restype = None

		Lib.__lib.SimdCrc32.argtypes = [ ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdCrc32.restype = ctypes.c_uint32

		Lib.__lib.SimdCrc32c.argtypes = [ ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdCrc32c.restype = ctypes.c_uint32
		
		Lib.__lib.SimdImageSaveToFile.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p ]
		Lib.__lib.SimdImageSaveToFile.restype = ctypes.c_int32
		
		Lib.__lib.SimdImageLoadFromFile.argtypes = [ ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int32) ]
		Lib.__lib.SimdImageLoadFromFile.restype = ctypes.c_void_p
		
		Lib.__lib.SimdResizerInit.argtypes = [ ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32 ]
		Lib.__lib.SimdResizerInit.restype = ctypes.c_void_p
		
		Lib.__lib.SimdResizerRun.argtypes = [ ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdResizerRun.restype = None
		
		Lib.__lib.SimdSynetSetInput.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdSynetSetInput.restype = None
	
	## Gets version of %Simd Library.
	# @return A string with version.
	def Version() -> str: 
		ptr = Lib.__lib.SimdVersion()
		return str(ptr, encoding='utf-8')
	
	## Gets string with CPU description.
	# @param type - a type of CPU description.
	# @return A string with system description.
	def CpuDesc(type: Simd.CpuDesc) -> str: 
		ptr = Lib.__lib.SimdCpuDesc(type.value)
		return str(ptr, encoding='utf-8')
	
	## Gets information about CPU.
	# @param type - a type of CPU information.
	# @return integer value of given CPU parameter.
	def CpuInfo(type: Simd.CpuInfo) -> int: 
		return Lib.__lib.SimdCpuInfo(type.value)

	## Gets string with CPU and %Simd Library description.
	# @return string with CPU and %Simd Library description.	
	def SysInfo() -> str: 
		info = ""
		info += "Simd Library: {0}".format(Lib.Version())
		info += "; CPU: {0}".format(Lib.CpuDesc(Simd.CpuDesc.Model))
		info += "; System sockets: {0}".format(Lib.CpuInfo(Simd.CpuInfo.Sockets))
		info += ", Cores: {0}".format(Lib.CpuInfo(Simd.CpuInfo.Cores))
		info += ", Threads: {0}".format(Lib.CpuInfo(Simd.CpuInfo.Threads))
		info += "; Cache L1D: {:.0f} KB".format(Lib.CpuInfo(Simd.CpuInfo.CacheL1) / 1024)
		info += ", L2: {:.0f} KB".format(Lib.CpuInfo(Simd.CpuInfo.CacheL2) / 1024)
		info += ", L3: {:.1f} MB".format(Lib.CpuInfo(Simd.CpuInfo.CacheL3) / 1024 / 1024)
		info += ", RAM: {:.1f} GB".format(Lib.CpuInfo(Simd.CpuInfo.RAM) / 1024 / 1024 / 1024)
		info += "; Available SIMD:"
		if Lib.CpuInfo(Simd.CpuInfo.AMX) > 0 :
			info += " AMX"
		if Lib.CpuInfo(Simd.CpuInfo.AVX512BF16) > 0 :
			info += " AVX-512VBF16"
		if Lib.CpuInfo(Simd.CpuInfo.AVX512VNNI) > 0 :
			info += " AVX-512VNNI"
		if Lib.CpuInfo(Simd.CpuInfo.AVX512BW) > 0 :
			info += " AVX-512BW AVX-512F"
		if Lib.CpuInfo(Simd.CpuInfo.AVX2) > 0 :
			info += " AVX2 FMA"
		if Lib.CpuInfo(Simd.CpuInfo.AVX) > 0 :
			info += " AVX"
		if Lib.CpuInfo(Simd.CpuInfo.SSE41) > 0 :
			info += " SSE4.1 SSSE3 SSE3 SSE2 SSE"
		if Lib.CpuInfo(Simd.CpuInfo.NEON) > 0 :
			info += " NEON"
		if Lib.CpuInfo(Simd.CpuInfo.VSX) > 0 :
			info += " VSX"
		if Lib.CpuInfo(Simd.CpuInfo.VMX) > 0 :
			info += " Altivec"
		return info
	
	## Gets string with internal %Simd Library performance statistics.
	# @note %Simd Library must be built with switched on SIMD_PERF flag.
	# @return string with internal %Simd Library performance statistics.	
	def PerformanceStatistic() -> str: 
		ptr = Lib.__lib.SimdPerformanceStatistic()
		return str(ptr, encoding='utf-8')
	
    ## Allocates aligned memory block.
    # @note The memory allocated by this function is must be deleted by function Simd.Lib.Free.
	# @param size - an original size.
    # @param  align - a required alignment.
	# @return return a pointer to allocated memory.
	def Allocate(size : int, align : int) -> ctypes.c_void_p :
		return Lib.__lib.SimdAllocate(size, align)
	
    ## Frees aligned memory block.
    # @note This function frees a memory allocated by function Simd.Lib.Allocate.
	# @param ptr - a pointer to the memory to be deleted.
	def Free(ptr : ctypes.c_void_p) :
		Lib.__lib.SimdFree(ptr)
	
    ## Gets aligned size.
	# @param size - an original size.
    # @param  align - a required alignment.
	# @return return an aligned size.
	def Align(size : int, align : int) -> int:
		return Lib.__lib.SimdAlign(size, align)
	
    ## Gets alignment required for the most productive work of Simd Library.
	# @return return a required alignment.
	def Alignment() -> int:
		return Lib.__lib.SimdAlignment()
	
    ## Releases context created with using of Simd Library API.
	# @param context - a context to be released.
	def Release(context : ctypes.c_void_p) :
		Lib.__lib.SimdRelease(context)
	
	## Gets number of threads used by %Simd Library to parallelize some algorithms.
	# @return thread number used by %Simd Library.	
	def GetThreadNumber() -> int: 
		return Lib.__lib.SimdGetThreadNumber()
	
	## Sets number of threads used by %Simd Library to parallelize some algorithms.
	# @param threadNumber - number used by %Simd Library.	
	def SetThreadNumber(threadNumber: int) : 
		Lib.__lib.SimdSetThreadNumber(threadNumber)
		
	## Clears MMX registers.
	# Clears MMX registers (runs EMMS instruction). It is x86 specific functionality.
	def ClearMmx(): 
		Lib.__lib.SimdEmpty()
		
	## Gets current CPU Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) flags. It is used in order to process subnormal numbers.
	# @return current 'fast' mode.	
	def GetFastMode() -> bool: 
		return Lib.__lib.SimdGetFastMode()
	
	## Sets current CPU Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) flags. It is used in order to process subnormal numbers.
	# @param fast - a value of 'fast' mode to set.	
	def SetFastMode(fast: bool) : 
		Lib.__lib.SimdSetFastMode(fast)
		
    ## Gets 32-bit cyclic redundancy check (CRC32) for current data.
	# Calculation is performed for polynomial 0xEDB88320.
	# @param src - a pointer to data.
    # @param  size - a size of the data.
    # @return 32-bit cyclic redundancy check (CRC32).
	def Crc32(src : ctypes.c_void_p, size : int) -> ctypes.c_uint32 :
		return Lib.__lib.SimdCrc32(src, size)
	
    ## Gets 32-bit cyclic redundancy check (CRC32c) for current data.
	# Calculation is performed for polynomial 0x1EDC6F41 (Castagnoli-crc).
	# @param src - a pointer to data.
    # @param  size - a size of the data.
    # @return 32-bit cyclic redundancy check (CRC32c).
	def Crc32c(src : ctypes.c_void_p, size : int) :
		return Lib.__lib.SimdCrc32c(src, size)
	
    ## Saves an image to file in given image file format.
    # @param src - a pointer to pixels data of input image.
    # @param stride - a row size of input image in bytes.
    # @param width - a width of input image.
    # @param height - a height of input image.
    # @param format - a pixel format of input image. Supported pixel formats: Simd.PixelFormat.Gray8, Simd.PixelFormat.Bgr24, Simd.PixelFormat.Bgra32, Simd.PixelFormat.Rgb24, Simd.PixelFormat.Rgba32.
    # @param file - a format of output image file. To auto choise format of output file set this parameter to Simd.ImageFile.Undefined.
    # @param quality - a parameter of compression quality (if file format supports it).
    # @param path - a path to output image file.
    # @return result of the operation.
	def ImageSaveToFile(src : ctypes.c_void_p, stride: int, width: int, height: int, format : Simd.PixelFormat, file : Simd.ImageFile, quality : int, path : str) -> bool :
		return Lib.__lib.SimdImageSaveToFile(src, stride, width, height, format.value, file.value, quality, path.encode('utf-8')) != 0

    ## Loads an image from file.
    # @param path - a path to input image file.
    # @param desiredFormat - a desired pixel format of output image. It can be Simd.PixelFormat.Gray8, Simd.PixelFormat.Bgr24, Simd.PixelFormat.Bgra32, 
    #                 Simd.PixelFormat.Rgb24, Simd.PixelFormat.Rgba32 or Simd.PixelFormat.Empty (use pixel format of input image file).
	# @return a pointer to pixel data, row size in bytes, image width, image height, output pixel format. The output pixel data mast be deleted after use by function Simd.Lib.Free.
	def ImageLoadFromFile(path : str, desiredFormat: Simd.PixelFormat):
		stride = ctypes.c_size_t()
		width = ctypes.c_size_t()
		height = ctypes.c_size_t()
		format = ctypes.c_int32(desiredFormat.value)
		data = Lib.__lib.SimdImageLoadFromFile(path.encode('utf-8'), ctypes.byref(stride), ctypes.byref(width), ctypes.byref(height), ctypes.byref(format))
		return data, stride.value, width.value, height.value, Simd.PixelFormat(format.value)
	
    ## Creates resize context. 
    # @param srcX - a width of the input image.
    # @param srcY - a height of the input image.
    # @param dstX - a width of the output image.
    # @param dstY - a height of the output image.
    # @param channels - a channel number of input and output image.
    # @param type - a type of input and output image channel.
    # @param method - a method used in order to resize image.
    # @return a pointer to resize context. On error it returns NULL. This pointer is used in functions ::SimdResizerRun. It must be released with using of function Simd.Release.
	def ResizerInit(srcX : int, srcY : int, dstX : int, dstY : int, channels : int,  type : Simd.ResizeChannel, method : Simd.ResizeMethod) -> ctypes.c_void_p :
		return Lib.__lib.SimdResizerInit(srcX, srcY, dstX, dstY, channels, type.value, method.value)
	
    ## Performs image resizing.
    # @param resizer - a resize context. It must be created by function Simd.ResizerInit and released by function Simd.Release.
    # @param src - a pointer to pixels data of the original input image.
    # @param srcStride - a row size (in bytes) of the input image.
    # @param dst - a pointer to pixels data of the resized output image.
    # @param dstStride - a row size (in bytes) of the output image.
	def ResizerRun(resizer : ctypes.c_void_p, src : ctypes.c_void_p, srcStride : int, dst : ctypes.c_void_p, dstStride : int) :
		Lib.__lib.SimdResizerRun(resizer, src, srcStride, dst, dstStride)

	
###################################################################################################

## @ingroup python
# The Image class provides storage and manipulation of images.
class Image():
	## Creates a new image.
	# @param format - image pixel format.
	# @param width - image width.
	# @param height - image height.
	# @param align - a row size alingnment in bytes (optional).
	# @param stride - a row size in bytes of image created on the base of external image (optional). 
	# @param data - a pointer to external image pixel data (optional). 
	def __init__(self, format = Simd.PixelFormat.Empty, width = 0, height = 0, align = 0, stride = 0, data = ctypes.c_void_p(0)) :
		self.__data = ctypes.c_void_p(0)
		self.__owner = False
		self.Clear()
		if format != Simd.PixelFormat.Empty and width > 0 and height > 0 :
			if data != 0 and stride != 0 :
				self.FromExternal(format, width, height, stride, data)
			else :
				self.Recreate(format, width, height, align)
	
	## Frees image data.
	# Releases image pixel data, sets to zero all fields.
	def __del__(self) :
		self.Clear()

	## Recreates the imagee.
	# @param format - a new image pixel format.
	# @param width - a new image width.
	# @param height - a new image height.	
	# @param align - a row size alingnment in bytes (optional).
	def Recreate(self, format : Simd.PixelFormat, width : int, height : int, align = 0) :
		if format == self.__format and width == self.__width and height == self.__height :
			return
		self.Clear()
		pixelSize = format.PixelSize()
		if align == 0 :
			align = Lib.Alignment()
		if pixelSize * width * height > 0 :
			self.__format = format
			self.__width = width
			self.__height = height
			self.__stride = Lib.Align(pixelSize * width, align)
			self.__data = Lib.Allocate(self.__stride * self.__height, Lib.Alignment())
			self.__owner = True
			
	## Recreates a new image on the base of external image.
	# @param format - a new image pixel format.
	# @param width - a new image width.
	# @param height - a new image height.
	# @param stride - a row size in bytes of external image. 
	# @param data - a pointer to external image pixel data. 
	def FromExternal(self, format : Simd.PixelFormat, width : int, height : int, stride : int, data : ctypes.c_void_p) :
		self.Clear()
		self.__width = width
		self.__height = height
		self.__stride = stride
		self.__format = format
		self.__data = data

	## Clears image.
	# Releases image pixel data, sets to zero all fields.	
	def Clear(self) :
		self.__width = 0
		self.__height = 0
		self.__stride = 0
		self.__format = Simd.PixelFormat.Empty
		if self.__data != 0 and self.__owner :
			Lib.Free(self.__data)
		self.__data = ctypes.c_void_p(0)
		self.__owner = False
		
	## Gets image width.
	# @return image width.	
	def Width(self) -> int :
		return self.__width
	
	## Gets image height.
	# @return image height.	
	def Height(self) -> int :
		return self.__height
	
	## Gets image row size in bytes.
	# @return image row size in bytes.	
	def Stride(self) -> int :
		return self.__stride
	
	## Gets image pixel format.
	# @return image pixel format.	
	def Format(self) -> Simd.PixelFormat :
		return self.__format
	
	## Gets pointer to image pixel data.
	# @return pointer to image pixel data.	
	def Data(self) -> ctypes.c_void_p :
		return self.__data

    ## Saves the image to file in given image file format.
    # @note Supported pixel formats: Simd.PixelFormat.Gray8, Simd.PixelFormat.Bgr24, Simd.PixelFormat.Bgra32, Simd.PixelFormat.Rgb24, Simd.PixelFormat.Rgba32.
    # @param path - a path to output image file.
    # @param file - a format of output image file. To auto choise format of output file set this parameter to Simd.ImageFile.Undefined.
    # @param quality - a parameter of compression quality (if file format supports it).
    # @return result of the operation.
	def Save(self, path : str, file = Simd.ImageFile.Undefined, quality = 100) -> bool:
		return Lib.ImageSaveToFile(self.Data(), self.Stride(), self.Width(), self.Height(), self.Format(), file, quality, path)
	
	## Loads an image from file.
    # @param path - a path to input image file.
    # @param desiredFormat - a desired pixel format of output image. It can be Simd.PixelFormat.Gray8, Simd.PixelFormat.Bgr24, Simd.PixelFormat.Bgra32, 
    #                 Simd.PixelFormat.Rgb24, Simd.PixelFormat.Rgba32 or Simd.PixelFormat.Empty (use pixel format of input image file).
    # @return result of the operation.
	def Load(self, path : str, desiredFormat = Simd.PixelFormat.Empty) -> bool:
		self.Clear()
		data, stride, width, height, format = Lib.ImageLoadFromFile(path, desiredFormat)
		if data != 0 :
			self.__width = width
			self.__height = height
			self.__stride = stride
			self.__format = format
			self.__data = data
			self.__owner = True
		return self.__owner
	
    ## Creates a new Simd.Image which points to the region of current image bounded by the rectangle with specified coordinates.
    # @param left - a left side of the region.
    # @param top - a top side of the region.
    # @param right - a right side of the region.
    # @param bottom - a bottom side of the region.
    # @return - a new Simd.Image which points to the subregion of current image.
	def Region(self, left: int, top : int, right : int, bottom : int) :
		if self.Data() == ctypes.c_void_p(0) or right <= left or bottom <= top :
			return Simd.Image()
		left = min(max(left, 0), self.Width());
		top = min(max(top, 0), self.Height());
		right = min(max(right, 0), self.Width());
		bottom = min(max(bottom, 0), self.Height());
		return Simd.Image(self.Format(), right - left, bottom - top, 0, self.Stride(), self.Data() + top * self.Stride() + left * self.Format().PixelSize());
	
    ## Creates a new Simd.Image which points to the region of current image with given size at current position.
    # @param width - a width of the region.
    # @param height - a height of the region.
    # @param position - a position of the region.
    # @return - a new Simd.Image which points to the subregion of current image.
	def RegionAt(self, width: int, height : int, position : Simd.Position) :
		if position == Position.TopLeft :
			return self.Region(0, 0, width, height)
		elif position == Position.TopCenter :
			return self.Region((self.Width() - width) // 2, 0, (self.Width() + width) // 2, height)
		elif position == Position.TopRight :
			return self.Region(self.Width() - width, 0, self.Width(), height)
		elif position == Position.MiddleLeft:
			return self.Region(0, (self.Height() - height) // 2, width, (self.Height() + height) // 2)
		elif position == Position.MiddleCenter:
			return self.Region((self.Width() - width) // 2, (self.Height() - height) // 2, (self.Width() + width) // 2, (self.Height() + height) // 2)
		elif position == Position.MiddleRight:
			return self.Region(self.Width() - width, (self.Height() - height) // 2, self.Width(), (self.Height() + height) // 2)
		elif position == Position.BottomLeft:
			return self.Region(0, self.Height() - height, width, self.Height());
		elif position == Position.BottomCenter:
			return self.Region((self.Width() - width) // 2, self.Height() - height, (self.Width() + width) // 2, self.Height())
		elif position == Position.BottomRight:
			return self.Region(self.Width() - width, self.Height() - height, self.Width(), self.Height())
		else :
			return Simd.Image()

###################################################################################################

def PixelFormatToResizeChannel(src) -> ResizeChannel :
	if src == Simd.PixelFormat.Gray8 : return ResizeChannel.Byte
	elif src == Simd.PixelFormat.Uv16 : return ResizeChannel.Byte
	elif src == Simd.PixelFormat.Bgr24 : return ResizeChannel.Byte
	elif src == Simd.PixelFormat.Bgra32 : return ResizeChannel.Byte
	elif src == Simd.PixelFormat.Int16 : return ResizeChannel.Short
	elif src == Simd.PixelFormat.Float : return ResizeChannel.Float
	elif src == Simd.PixelFormat.Hsv24 : return ResizeChannel.Byte
	elif src == Simd.PixelFormat.Hsl24 : return ResizeChannel.Byte
	elif src == Simd.PixelFormat.Rgb24 : return ResizeChannel.Byte
	elif src == Simd.PixelFormat.Rgba32 : return ResizeChannel.Byte
	elif src == Simd.PixelFormat.Argb32 : return ResizeChannel.Byte
	else : raise Exception("Can't {0} convert to Simd.ResizeChannel !".format(src))

##  @ingroup python
# The function performs image resizing.
# @param src - an original input image.
# @param dst - a resized output image.
# @param method - a resizing method. By default it is equal to Simd.ResizeMethod.Bilinear.
def Resize(src : Image, dst : Image, method = Simd.ResizeMethod.Bilinear) :
	if dst.Format() != src.Format() :
		raise Exception("Incompatible image pixel formats!")
	resizer = Lib.ResizerInit(src.Width(), src.Height(), dst.Width(), dst.Height(), src.Format().ChannelCount(), Simd.PixelFormatToResizeChannel(src.Format()), method)
	if resizer == ctypes.c_void_p(0) :
		raise Exception("Can't create resizer context !")
	Lib.ResizerRun(resizer, src.Data(), src.Stride(), dst.Data(), dst.Stride())
	Lib.Release(resizer)

##  @ingroup python
# The function gets resized image.
# @param src - an original input image.
# @param width - a width of output image.
# @param height - a height of output image.
# @param method - a resizing method. By default it is equal to Simd.ResizeMethod.Bilinear.
# @return - resized output image.
def Resized(src : Image, width :int, height: int, method = Simd.ResizeMethod.Bilinear) -> Simd.Image :
	dst = Image(src.Format(), width, height)
	Simd.Resize(src, dst, method)
	return dst

##  @ingroup python
# Sets image to the input of neural network of <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.
# @param src - an input image. There are following supported pixel format: aSimd.PixelFormat.Gray8, Simd.PixelFormat.Bgr24, Simd.PixelFormat.Bgra32, Simd.PixelFormat.Rgb24.
# @param lower - an array with lower bound of values of the output tensor. The size of the array have to correspond number of channels in the output image tensor.
# @param upper - an array with upper bound of values of the output tensor. The size of the array have to correspond number of channels in the output image tensor.
# @param dst - a pointer to the output 32-bit float image tensor.
# @param channels - a number of channels in the output image tensor. It can be 1 or 3.
# @param format - a format of output image tensor. There are supported following tensor formats: Simd.TensorFormat.Nchw, Simd.TensorFormat.Nhwc.
def SynetSetInput(src : Image, lower, upper, dst : ctypes.c_void_p, channels : int, format : Simd.TensorFormat) :
	if src.Format() != PixelFormat.Gray8 and src.Format() != PixelFormat.Bgr24 and src.Format() != PixelFormat.Bgra32 and src.Format() != PixelFormat.Rgb24 :
		raise Exception("Incompatible image pixel format: {0}!".format(src.Format()))
	if channels != 1 and channels != 3 :
		raise Exception("Incompatible channel value: {0} !".format(channels))
	lo = (ctypes.c_float * len(lower))()
	for i in range(len(lower)) :
		lo[i] = lower[i]
	up = (ctypes.c_float * len(upper))()
	for i in range(len(upper)) :
		up[i] = upper[i]
	Lib._Lib__lib.SimdSynetSetInput(src.Data(), src.Width(), src.Height(), src.Stride(), src.Format().value, lo, up, dst, channels, format.value)
