###################################################################################################
# Simd Library (http://ermig1979.github.io/Simd).
#
# Copyright (c) 2011-2025 Yermalayeu Ihar.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or 
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###################################################################################################

import argparse
from dataclasses import dataclass
import os
import ctypes
import pathlib
import sys
import array
import enum

import numpy

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
	## Enabling of AVX, AVX2, FMA CPU extensions (x86 specific).
	AVX2 = 8
	## Enabling of AVX-512F, AVX-512BW CPU extensions (x86 specific).
	AVX512BW = 9
	## Enabling of AVX-512VNNI CPU extensions (x86 specific).
	AVX512VNNI = 10
	## Enabling of AVX-512BF16, AMX-BF16, AMX-INT8 CPU extensions (x86 specific).
	AMXBF16 = 11
	## Enabling of NEON CPU extensions (ARM specific).
	NEON = 12

## @ingroup python
# Describes frame format type. It is used in Simd.Frame.
class FrameFormat(enum.Enum) :	
	## An undefined pixel format.
	Empty = 0
	## Two planes (8-bit full size Y plane, 16-bit interlived half size UV plane) NV12 pixel format.
	Nv12 = 1
	## Three planes (8-bit full size Y plane, 8-bit half size U plane, 8-bit half size V plane) YUV420P pixel format.
	Yuv420p = 2
	## One plane 32-bit (4 8-bit channels) BGRA (Blue, Green, Red, Alpha) pixel format.
	Bgra32 = 3
	## One plane 24-bit (3 8-bit channels) BGR (Blue, Green, Red) pixel format.
	Bgr24 = 4
	## One plane 8-bit gray pixel format.
	Gray8 = 5
	## One plane 24-bit (3 8-bit channels) RGB (Red, Green, Blue) pixel format.
	Rgb24 = 6
	## One plane 32-bit (4 8-bit channels) RGBA (Red, Green, Blue, Alpha) pixel format.
	Rgba32 = 7
	## One plane 24-bit (3 8-bit channels) LAB (CIELAB) pixel format.
	Lab24 = 8
	
	## Gets number of planes for current frame format.
	# @return number of planes.	
	def PlaneCount(self) -> int :
		if self == Simd.FrameFormat.Empty : return 0
		elif self == Simd.FrameFormat.Nv12: return 2
		elif self == Simd.FrameFormat.Yuv420p: return 3
		elif self == Simd.FrameFormat.Bgra32: return 1
		elif self == Simd.FrameFormat.Bgr24: return 1
		elif self == Simd.FrameFormat.Gray8: return 1
		elif self == Simd.FrameFormat.Rgb24: return 1
		elif self == Simd.FrameFormat.Rgba32: return 1
		elif self == Simd.FrameFormat.Lab24: return 1
		else : return 0

	
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
    ## A BMP (BitMap Picture) image file format.
    Bmp = 7

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
	## A 24-bit (3 8-bit channels) LAB (CIELAB) pixel format.
	Lab24 = 20
	
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
		elif self == Simd.PixelFormat.Lab24 : return 3
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
		elif self == Simd.PixelFormat.Lab24 : return 1
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
		elif self == Simd.PixelFormat.Lab24 : return 3
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
	
## @ingroup python
# Describes Warp Affine flags. This type used in function Simd.WarpAffineInit.
class WarpAffineFlags(enum.Flag) :
	## Default Warp Affine flags.
    Default = 0 
	## 8-bit integer channel type.
    ChannelByte = 0 
	## Bit mask of channel type.
    ChannelMask = 1 
	## Nearest pixel interpolation method.
    InterpNearest = 0
	## Bilinear pixel interpolation method.
    InterpBilinear = 2 
	## Bit mask of pixel interpolation options.
    InterpMask = 2 
	## Nearest pixel interpolation method.
    BorderConstant = 0
	## Bilinear pixel interpolation method.
    BorderTransparent = 4 
	## Bit mask of pixel interpolation options.
    BorderMask = 4 

	
## @ingroup python
# Describes YUV format type. It is uses in YUV to BGR forward and backward conversions.
class YuvType(enum.Enum) :
    ## Unknown YUV standard.
    Unknown = -1  
    ## Corresponds to BT.601 standard. Uses Kr=0.299, Kb=0.114. Restricts Y to range [16..235], U and V to [16..240].
    Bt601 = 0
    ## Corresponds to BT.709 standard. Uses Kr=0.2126, Kb=0.0722. Restricts Y to range [16..235], U and V to [16..240].
    Bt709 = 1 
    ## Corresponds to BT.2020 standard. Uses Kr=0.2627, Kb=0.0593. Restricts Y to range [16..235], U and V to [16..240].
    Bt2020 = 2 
    ## Corresponds to T-REC-T.871 standard. Uses Kr=0.299, Kb=0.114. Y, U and V use full range [0..255].
    Trect871 = 3 

###################################################################################################

## @ingroup python
# A wrapper around %Simd Library API.
class Lib():
	__lib : ctypes.CDLL
	lib : ctypes.CDLL
	
	## Initializes Simd.Lib class (loads %Simd Library binaries).
	# @note This method must be called before any using of this class.
	# @param dir - a directory with %Simd Library binaries (Simd.dll or libSimd.so). By default it is empty string. That means that binaries will be searched in the directory with current Python file.
	def Init(dir = ""):
		if dir == "" :
			dir = pathlib.Path(__file__).parent.resolve()
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
		
		Lib.__lib.SimdAbsDifference.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t ]
		Lib.__lib.SimdAbsDifference.restype = None
		
		Lib.__lib.SimdAbsDifferenceSum.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64) ]
		Lib.__lib.SimdAbsDifferenceSum.restype = None
		
		Lib.__lib.SimdAbsDifferenceSumMasked.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint8, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64) ]
		Lib.__lib.SimdAbsDifferenceSumMasked.restype = None
		
		Lib.__lib.SimdAbsDifferenceSums3x3.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64) ]
		Lib.__lib.SimdAbsDifferenceSums3x3.restype = None
		
		Lib.__lib.SimdAbsDifferenceSums3x3Masked.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint8, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64) ]
		Lib.__lib.SimdAbsDifferenceSums3x3Masked.restype = None
		
		Lib.__lib.SimdAbsGradientSaturatedSum.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdAbsGradientSaturatedSum.restype = None
		
		Lib.__lib.SimdAddFeatureDifference.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint16, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdAddFeatureDifference.restype = None
		
		Lib.__lib.SimdAlphaBlending.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdAlphaBlending.restype = None
		
		Lib.__lib.SimdAlphaBlending2x.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdAlphaBlending2x.restype = None

		Lib.__lib.SimdAlphaBlendingBgraToYuv420p.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdAlphaBlendingBgraToYuv420p.restype = None

		Lib.__lib.SimdAlphaBlendingUniform.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint8, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdAlphaBlendingUniform.restype = None

		Lib.__lib.SimdAlphaFilling.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdAlphaFilling.restype = None
		
		
		Lib.__lib.SimdBgraToBgr.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdBgraToBgr.restype = None
		
		Lib.__lib.SimdBgraToGray.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdBgraToGray.restype = None
		
		Lib.__lib.SimdBgraToRgb.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdBgraToRgb.restype = None
		
		Lib.__lib.SimdBgraToRgba.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdBgraToRgba.restype = None
		
		Lib.__lib.SimdBgraToYuv420pV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdBgraToYuv420pV2.restype = None
		
		Lib.__lib.SimdBgraToYuv422pV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdBgraToYuv422pV2.restype = None

		Lib.__lib.SimdBgraToYuv444pV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdBgraToYuv444pV2.restype = None

		
		Lib.__lib.SimdBgrToBgra.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint8 ]
		Lib.__lib.SimdBgrToBgra.restype = None
		
		
		Lib.__lib.SimdBgrToGray.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdBgrToGray.restype = None
		
		
		Lib.__lib.SimdBgrToLab.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdBgrToLab.restype = None
		
		
		Lib.__lib.SimdBgrToRgb.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdBgrToRgb.restype = None
		
		Lib.__lib.SimdBgrToYuv420pV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdBgrToYuv420pV2.restype = None
		
		Lib.__lib.SimdBgrToYuv422pV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdBgrToYuv422pV2.restype = None

		Lib.__lib.SimdBgrToYuv444pV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdBgrToYuv444pV2.restype = None

		
		Lib.__lib.SimdCopy.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdCopy.restype = None
		
		
		Lib.__lib.SimdDeinterleaveUv.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdDeinterleaveUv.restype = None

		
		Lib.__lib.SimdFillPixel.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdFillPixel.restype = None
		
		
		Lib.__lib.SimdGrayToBgra.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint8 ]
		Lib.__lib.SimdGrayToBgra.restype = None
		
		Lib.__lib.SimdGrayToBgr.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdGrayToBgr.restype = None

		Lib.__lib.SimdGrayToY.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdGrayToY.restype = None

		
		Lib.__lib.SimdImageSaveToFile.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p ]
		Lib.__lib.SimdImageSaveToFile.restype = ctypes.c_int32
		
		Lib.__lib.SimdImageLoadFromFile.argtypes = [ ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int32) ]
		Lib.__lib.SimdImageLoadFromFile.restype = ctypes.c_void_p
		

		Lib.__lib.SimdInterleaveUv.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdInterleaveUv.restype = None

		
		Lib.__lib.SimdRgbToBgra.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint8 ]
		Lib.__lib.SimdRgbToBgra.restype = None
		
		Lib.__lib.SimdRgbToGray.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdRgbToGray.restype = None
		
		Lib.__lib.SimdRgbaToGray.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdRgbaToGray.restype = None

		
		Lib.__lib.SimdResizerInit.argtypes = [ ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int32, ctypes.c_int32 ]
		Lib.__lib.SimdResizerInit.restype = ctypes.c_void_p
		
		Lib.__lib.SimdResizerRun.argtypes = [ ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdResizerRun.restype = None

		
		Lib.__lib.SimdSynetSetInput.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdSynetSetInput.restype = None

		
		Lib.__lib.SimdWarpAffineInit.argtypes = [ ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_void_p ]
		Lib.__lib.SimdWarpAffineInit.restype = ctypes.c_void_p
		
		Lib.__lib.SimdWarpAffineRun.argtypes = [ ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p ]
		Lib.__lib.SimdWarpAffineRun.restype = None
		
		
		Lib.__lib.SimdYToGray.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t ]
		Lib.__lib.SimdYToGray.restype = None
		
		
		Lib.__lib.SimdYuv420pToBgrV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdYuv420pToBgrV2.restype = None

		Lib.__lib.SimdYuv422pToBgrV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdYuv422pToBgrV2.restype = None

		Lib.__lib.SimdYuv444pToBgrV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdYuv444pToBgrV2.restype = None

		
		Lib.__lib.SimdYuv420pToBgraV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint8, ctypes.c_int32 ]
		Lib.__lib.SimdYuv420pToBgraV2.restype = None

		Lib.__lib.SimdYuv422pToBgraV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint8, ctypes.c_int32 ]
		Lib.__lib.SimdYuv422pToBgraV2.restype = None

		Lib.__lib.SimdYuv444pToBgraV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint8, ctypes.c_int32 ]
		Lib.__lib.SimdYuv444pToBgraV2.restype = None
		

		Lib.__lib.SimdYuv420pToRgbV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdYuv420pToRgbV2.restype = None

		Lib.__lib.SimdYuv422pToRgbV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdYuv422pToRgbV2.restype = None

		Lib.__lib.SimdYuv444pToRgbV2.argtypes = [ ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32 ]
		Lib.__lib.SimdYuv444pToRgbV2.restype = None

	
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
		if Lib.CpuInfo(Simd.CpuInfo.AMXBF16) > 0 :
			info += " AMX-BF16 AMX-INT8 AVX-512VBF16"
		if Lib.CpuInfo(Simd.CpuInfo.AVX512VNNI) > 0 :
			info += " AVX-512VNNI"
		if Lib.CpuInfo(Simd.CpuInfo.AVX512BW) > 0 :
			info += " AVX-512BW AVX-512F"
		if Lib.CpuInfo(Simd.CpuInfo.AVX2) > 0 :
			info += " AVX2 FMA AVX"
		if Lib.CpuInfo(Simd.CpuInfo.SSE41) > 0 :
			info += " SSE4.1 SSSE3 SSE3 SSE2 SSE"
		if Lib.CpuInfo(Simd.CpuInfo.NEON) > 0 :
			info += " NEON"
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
	
    ## Puts to destination 8-bit gray image saturated sum of absolute gradient for every point of source 8-bit gray image.
    # @param src - a pointer to pixels data of input 8-bit gray image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 8-bit gray image.
    # @param dstStride - a row size of output image in bytes.
	def AbsGradientSaturatedSum(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdAbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride)
		
    ## Converts 32-bit BGRA image to 24-bit BGR image. Also it can be used for 32-bit RGBA to 24-bit RGB conversion.
    # @param src - a pointer to pixels data of input 32-bit BGRA (or 32-bit RGBA) image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 24-bit BGR (or 24-bit RGB) image.
    # @param dstStride - a row size of output image in bytes.
	def BgraToBgr(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdBgraToBgr(src, width, height, srcStride, dst, dstStride)
		
    ## Converts 32-bit BGRA image to 8-bit gray image. 
    # @param src - a pointer to pixels data of input 32-bit BGRA.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 8-bit gray image.
    # @param dstStride - a row size of output image in bytes.
	def BgraToGray(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdBgraToGray(src, width, height, srcStride, dst, dstStride)
		
    ## Converts 32-bit BGRA image to 24-bit RGB image. Also it can be used for 32-bit RGBA to 24-bit BGR conversion.
    # @param src - a pointer to pixels data of input 32-bit BGRA (or 32-bit RGBA) image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 24-bit RGB (or 24-bit BGR) image.
    # @param dstStride - a row size of output image in bytes.
	def BgraToRgb(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdBgraToRgb(src, width, height, srcStride, dst, dstStride)
		
    ## Converts 32-bit BGRA image to 32-bit RGBA image and back.
    # @param src - a pointer to pixels data of input 32-bit BGRA (or 32-bit RGBA) image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 32-bit RGBA (or 32-bit BGRA) image.
    # @param dstStride - a row size of output image in bytes.
	def BgraToRgba(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdBgraToRgba(src, width, height, srcStride, dst, dstStride)
		
    ## Converts 32-bit BGRA image to YUV420P.
    # The input BGRA and output Y images must have the same width and height.
    # The output U and V images must have the same width and height (half size relative to Y component).
    # @param src - a pointer to pixels data of input 32-bit BGRA image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param y - a pointer to pixels data of output 8-bit image with Y color plane.
    # @param yStride - a row size of the y image.
    # @param u - a pointer to pixels data of output 8-bit image with U color plane.
    # @param uStride - a row size of the u image.
    # @param v - a pointer to pixels data of output 8-bit image with V color plane.
    # @param vStride - a row size of the v image.
    # @param yuvType - a type of output YUV image (see descriprion of Simd.YuvType).
	def BgraToYuv420p(src : ctypes.c_void_p, srcStride: int, width: int, height: int, y : ctypes.c_void_p, yStride: int, u : ctypes.c_void_p, uStride: int, v : ctypes.c_void_p, vStride: int, yuvType = Simd.YuvType.Bt601) :
		Lib.__lib.SimdBgraToYuv420pV2(src, srcStride, width, height, y, yStride, u, uStride, v, vStride, yuvType.value)
		
    ## Converts 24-bit BGR to 32-bit BGRA image image. Also it can be used for 24-bit RGB to 32-bit RGBA conversion.
    # @param src - a pointer to pixels data of input 24-bit BGR (or 24-bit RGB) image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 32-bit BGRA (or 32-bit RGBA) image.
    # @param dstStride - a row size of output image in bytes.
    # @param alpha - a value of alpha channel. By default it is equal to 255.
	def BgrToBgra(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int, alpha = 255) :
		Lib.__lib.SimdBgrToBgra(src, width, height, srcStride, dst, dstStride, alpha)
		
    ## Converts 24-bit BGR image to 8-bit gray image. 
    # @param src - a pointer to pixels data of input 24-bit BGR.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 8-bit gray image.
    # @param dstStride - a row size of output image in bytes.
	def BgrToGray(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdBgrToGray(src, width, height, srcStride, dst, dstStride)
		
    ## Converts 24-bit BGR image to 24-bit LAB image. 
    # @param src - a pointer to pixels data of input 24-bit BGR image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 24-bit LAB (CIELAB) image.
    # @param dstStride - a row size of output image in bytes.
	def BgrToLab(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdBgrToLab(src, srcStride, width, height, dst, dstStride)
		
    ## Converts 24-bit BGR image to 24-bit RGB image and back. 
    # @param src - a pointer to pixels data of input 24-bit BGR (or 24-bit RGB) image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 24-bit RGB (or 24-bit BGR) image.
    # @param dstStride - a row size of output image in bytes.
	def BgrToRgb(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdBgrToRgb(src, width, height, srcStride, dst, dstStride)
		
    ## Converts 24-bit BGR image to YUV420P.
    # The input BGR and output Y images must have the same width and height.
    # The output U and V images must have the same width and height (half size relative to Y component).
    # @param src - a pointer to pixels data of input 24-bit BGR image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param y - a pointer to pixels data of output 8-bit image with Y color plane.
    # @param yStride - a row size of the y image.
    # @param u - a pointer to pixels data of output 8-bit image with U color plane.
    # @param uStride - a row size of the u image.
    # @param v - a pointer to pixels data of output 8-bit image with V color plane.
    # @param vStride - a row size of the v image.
    # @param yuvType - a type of output YUV image (see descriprion of Simd.YuvType).
	def BgrToYuv420p(src : ctypes.c_void_p, srcStride: int, width: int, height: int, y : ctypes.c_void_p, yStride: int, u : ctypes.c_void_p, uStride: int, v : ctypes.c_void_p, vStride: int, yuvType = Simd.YuvType.Bt601) :
		Lib.__lib.SimdBgrToYuv420pV2(src, srcStride, width, height, y, yStride, u, uStride, v, vStride, yuvType.value)
		
    ## Deinterleaves 16-bit UV interleaved image into separated 8-bit U and V planar images.
    # All images must have the same width and height.
    # This function used for NV12 to YUV420P conversion.
    # @param src - a pointer to pixels data of input 16-bit UV image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param u - a pointer to pixels data of output 8-bit image with U color plane.
    # @param uStride - a row size of the u image.
    # @param v - a pointer to pixels data of output 8-bit image with V color plane.
    # @param vStride - a row size of the v image.
	def DeinterleaveUv(src : ctypes.c_void_p, srcStride: int, width: int, height: int, u : ctypes.c_void_p, uStride: int, v : ctypes.c_void_p, vStride: int) :
		Lib.__lib.SimdDeinterleaveUv(src, srcStride, width, height, u, uStride, v, vStride)
		
    ## Copies an image.
    # @param src - a pointer to pixels data of input image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param pixelSize - a pixel size of input/output image.
    # @param dst - a pointer to pixels data of output image.
    # @param dstStride - a row size of output image in bytes.
	def Copy(src : ctypes.c_void_p, srcStride: int, width: int, height: int, pixelSize: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdCopy(src, srcStride, width, height, pixelSize, dst, dstStride)
	
    ## Fills image by value of given pixel.
    # @param dst - a pointer to pixels data of output image.
    # @param stride - a row size of output image in bytes.
    # @param width - a width of output image.
    # @param height - a height of output image.
    # @param pixel - an array of unsigned 8-bit integer width pixel channels. Its size is in range [1..4].
	def FillPixel(dst : ctypes.c_void_p, stride: int, width: int, height: int, pixel : array.array('B')) :
		size = len(pixel)
		if size < 1 or size > 4 :
			raise Exception("Incompatible pixel size: {0} !".format(size))
		Lib.__lib.SimdFillPixel(dst, stride, width, height, (ctypes.c_uint8 * size)(*pixel), size)	
		
    ## Converts 8-bit gray to 32-bit BGRA (32-bit RGBA) image.
    # @param src - a pointer to pixels data of input 8-bit gray.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 32-bit BGRA (or 32-bit RGBA) image.
    # @param dstStride - a row size of output image in bytes.
    # @param alpha - a value of alpha channel. By default it is equal to 255.
	def GrayToBgra(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int, alpha = 255) :
		Lib.__lib.SimdGrayToBgra(src, width, height, srcStride, dst, dstStride, alpha)
		
    ## Converts 8-bit gray image to 24-bit BGR (24-bit RGB) image and back. 
    # @param src - a pointer to pixels data of input 8-bit gray image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 24-bit BGR (or 24-bit RGB) image.
    # @param dstStride - a row size of output image in bytes.
	def GrayToBgr(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdGrayToBgr(src, width, height, srcStride, dst, dstStride)
		
    ## Converts 8-bit gray image to 8-bit Y-plane of YUV image. 
    # @param src - a pointer to pixels data of input 8-bit gray image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 8-bit Y-plane of YUV image.
    # @param dstStride - a row size of output image in bytes.
	def GrayToY(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdGrayToY(src, srcStride, width, height, dst, dstStride)
		
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
	
    ## Interleaves separate 8-bit U and V planar images into 16-bit UV interleaved image.
    # All images must have the same width and height.
    # This function used for YUV420P to NV12 conversion.
    # @param u - a pointer to pixels data of input 8-bit image with U color plane.
    # @param uStride - a row size of the u image.
    # @param v - a pointer to pixels data of input 8-bit image with V color plane.
    # @param vStride - a row size of the v image.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param uv - a pointer to pixels data of output 16-bit UV image.
    # @param uvStride - a row size of output image in bytes.
	def InterleaveUv(u : ctypes.c_void_p, uStride: int, v : ctypes.c_void_p, vStride: int, width: int, height: int, uv : ctypes.c_void_p, uvStride: int) :
		Lib.__lib.SimdInterleaveUv(u, uStride, v, vStride, width, height, uv, uvStride)
	
    ## Converts 24-bit RGB to 32-bit BGRA image image. Also it can be used for 24-bit BGR to 32-bit RGBA conversion.
    # @param src - a pointer to pixels data of input 24-bit RGB (or 24-bit BGR) image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 32-bit BGRA (or 32-bit RGBA) image.
    # @param dstStride - a row size of output image in bytes.
    # @param alpha - a value of alpha channel. By default it is equal to 255.
	def RgbToBgra(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int, alpha = 255) :
		Lib.__lib.SimdRgbToBgra(src, width, height, srcStride, dst, dstStride, alpha)
		
    ## Converts 24-bit RGB image to 8-bit gray image. 
    # @param src - a pointer to pixels data of input 24-bit RGB.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 8-bit gray image.
    # @param dstStride - a row size of output image in bytes.
	def RgbToGray(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdRgbToGray(src, width, height, srcStride, dst, dstStride)
		
    ## Converts 32-bit RGBA image to 8-bit gray image. 
    # @param src - a pointer to pixels data of input 32-bit RGBA.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 8-bit gray image.
    # @param dstStride - a row size of output image in bytes.
	def RgbaToGray(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdRgbaToGray(src, width, height, srcStride, dst, dstStride)
	
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
		
	## Sets image to the input of neural network of <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.
    # @param src - a pointer to pixels data of input image.
    # @param height - a height of input image.
    # @param stride - a row size of input image in bytes.
    # @param width - a width of input image.
    # @param srcFormat - a pixel format of input image. Supported pixel formats: Simd.PixelFormat.Gray8, Simd.PixelFormat.Bgr24, Simd.PixelFormat.Bgra32, Simd.PixelFormat.Rgb24, Simd.PixelFormat.Rgba32.
	# @param lower - an array with lower bound of values of the output tensor. The size of the array have to correspond number of channels in the output image tensor.
	# @param upper - an array with upper bound of values of the output tensor. The size of the array have to correspond number of channels in the output image tensor.
	# @param dst - a pointer to the output 32-bit float image tensor.
	# @param channels - a number of channels in the output image tensor. It can be 1 or 3.
	# @param dstFormat - a format of output image tensor. There are supported following tensor formats: Simd.TensorFormat.Nchw, Simd.TensorFormat.Nhwc.
	# @param isRgb - is channel order of output tensor is RGB or BGR. Its default value is false.
	def SynetSetInput(src : ctypes.c_void_p, width: int, height: int, stride: int, srcFormat : Simd.PixelFormat, lower : array.array('f'), upper : array.array('f'), dst : ctypes.c_void_p, channels : int, dstFormat : Simd.TensorFormat, isRgb = False) :
		if srcFormat != PixelFormat.Gray8 and srcFormat != PixelFormat.Bgr24 and srcFormat != PixelFormat.Bgra32 and srcFormat != PixelFormat.Rgb24 and srcFormat != PixelFormat.Rgba32 :
			raise Exception("Incompatible image pixel format: {0}!".format(srcFormat))
		if channels != 1 and channels != 3 :
			raise Exception("Incompatible channel value: {0} !".format(channels))
		lo = (ctypes.c_float * len(lower))(*lower)
		up = (ctypes.c_float * len(upper))(*upper)
		sF = srcFormat;
		if srcFormat == PixelFormat.Bgr24 and isRgb : 
			sF = PixelFormat.Rgb24
		elif srcFormat == PixelFormat.Rgb24 and isRgb : 
			sF = PixelFormat.Bgr24
		elif srcFormat == PixelFormat.Bgra32 and isRgb : 
			sF = PixelFormat.Rgba32
		elif srcFormat == PixelFormat.Rgba32 and isRgb : 
			sF = PixelFormat.Bgra32
		Lib.__lib.SimdSynetSetInput(src, width, height, stride, sF.value, lo, up, dst, channels, dstFormat.value)
		
    ## Creates wrap affine context.
    # @param srcW - a width of input image.
    # @param srcH - a height of input image.
    # @param srcS - a row size (in bytes) of the input image.
    # @param dstW - a width of output image.
    # @param dstH - a height of output image.
    # @param dstS - a row size (in bytes) of the output image.
    # @param channels - a channel number of input and output image. Its value must be in range [1..4].
    # @param mat - an array with coefficients of affine warp (2x3 matrix).
    # @param flags - a flags of algorithm parameters.
    # @param border - an array with color of border. The size of the array mast be equal to channels.
    #                 It parameter is actual for SimdWarpAffineBorderConstant flag. It can be NULL.
    # @return a pointer to warp affine context. On error it returns NULL.
    #         This pointer is used in functions Simd.WarpAffineRun.
    #         It must be released with using of function Simd.Release. 
	def WarpAffineInit(srcW : int, srcH : int, srcS : int, dstW : int, dstH : int, dstS : int, channels, mat: array.array('f'), flags : Simd.WarpAffineFlags, border: array.array('B')) -> ctypes.c_void_p :
		return Lib.__lib.SimdWarpAffineInit(srcW, srcH, srcS, dstW, dstH, dstS, channels, (ctypes.c_float * len(mat))(*mat), flags.value, (ctypes.c_byte * len(border))(*border))
	
    ## Performs warp affine for current image.
    # @param context - a warp affine context. It must be created by function Simd.WarpAffineInit and released by function Simd.Release.
    # @param src - a pointer to pixels data of the original input image.
    # @param dst - a pointer to pixels data of the filtered output image.
	def WarpAffineRun(context : ctypes.c_void_p, src : ctypes.c_void_p, dst : ctypes.c_void_p) :
		Lib.__lib.SimdWarpAffineRun(context, src, dst)
		
    ## Converts 8-bit Y-plane of YUV image to 8-bit gray image. 
    # @param src - a pointer to pixels data of input 8-bit Y-plane of YUV image.
    # @param srcStride - a row size of input image in bytes.
    # @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 8-bit gray image.
    # @param dstStride - a row size of output image in bytes.
	def YToGray(src : ctypes.c_void_p, srcStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int) :
		Lib.__lib.SimdYToGray(src, srcStride, width, height, dst, dstStride)
		
    ## Converts YUV420P image to 24-bit BGR.
    # The input Y and output BGR images must have the same width and height.
    # The input U and V images must have the same width and height (half size relative to Y component).
	# @param y - a pointer to pixels data of input 8-bit image with Y color plane.
    # @param yStride - a row size of the y image.
    # @param u - a pointer to pixels data of input 8-bit image with U color plane.
    # @param uStride - a row size of the u image.
    # @param v - a pointer to pixels data of input 8-bit image with V color plane.
    # @param vStride - a row size of the v image.
	# @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 24-bit BGR image.
    # @param dstStride - a row size of output image in bytes.
    # @param yuvType - a type of input YUV image (see descriprion of Simd.YuvType).
	def Yuv420pToBgr(y : ctypes.c_void_p, yStride: int, u : ctypes.c_void_p, uStride: int, v : ctypes.c_void_p, vStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int, yuvType = Simd.YuvType.Bt601) :
		Lib.__lib.SimdYuv420pToBgrV2(y, yStride, u, uStride, v, vStride, width, height, dst, dstStride, yuvType.value)
		
    ## Converts YUV420P image to 32-bit BGRA.
    # The input Y and output BGRA images must have the same width and height.
    # The input U and V images must have the same width and height (half size relative to Y component).
	# @param y - a pointer to pixels data of input 8-bit image with Y color plane.
    # @param yStride - a row size of the y image.
    # @param u - a pointer to pixels data of input 8-bit image with U color plane.
    # @param uStride - a row size of the u image.
    # @param v - a pointer to pixels data of input 8-bit image with V color plane.
    # @param vStride - a row size of the v image.
	# @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 32-bit BGRA image.
    # @param dstStride - a row size of output image in bytes.
    # @param alpha - a value of alpha channel. By default it is equal to 255.
    # @param yuvType - a type of input YUV image (see descriprion of Simd.YuvType).
	def Yuv420pToBgra(y : ctypes.c_void_p, yStride: int, u : ctypes.c_void_p, uStride: int, v : ctypes.c_void_p, vStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int, alpha = 255, yuvType = Simd.YuvType.Bt601) :
		Lib.__lib.SimdYuv420pToBgraV2(y, yStride, u, uStride, v, vStride, width, height, dst, dstStride, alpha, yuvType.value)
		
    ## Converts YUV420P image to 24-bit RGB.
    # The input Y and output BGR images must have the same width and height.
    # The input U and V images must have the same width and height (half size relative to Y component).
	# @param y - a pointer to pixels data of input 8-bit image with Y color plane.
    # @param yStride - a row size of the y image.
    # @param u - a pointer to pixels data of input 8-bit image with U color plane.
    # @param uStride - a row size of the u image.
    # @param v - a pointer to pixels data of input 8-bit image with V color plane.
    # @param vStride - a row size of the v image.
	# @param width - a width of input/output image.
    # @param height - a height of input/output image.
    # @param dst - a pointer to pixels data of output 24-bit RGB image.
    # @param dstStride - a row size of output image in bytes.
    # @param yuvType - a type of input YUV image (see descriprion of Simd.YuvType).
	def Yuv420pToRgb(y : ctypes.c_void_p, yStride: int, u : ctypes.c_void_p, uStride: int, v : ctypes.c_void_p, vStride: int, width: int, height: int, dst : ctypes.c_void_p, dstStride: int, yuvType = Simd.YuvType.Bt601) :
		Lib.__lib.SimdYuv420pToRgbV2(y, yStride, u, uStride, v, vStride, width, height, dst, dstStride, yuvType.value)

	
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

	## Recreates the image.
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
	
	## Check if current and other image have the same size and pixel format.
	# @param other - other image.
	# @return result of checking.
	def Compatible(self, other) -> bool :
		return self.Format() == other.Format() and self.Width() == other.Width() and self.Height() == other.Height()

	## Check if current and other image have the same size.
	# @param other - other image.
	# @return result of checking.
	def EqualSize(self, other) -> bool :
		return self.Width() == other.Width() and self.Height() == other.Height()
		
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
	
    ## Saves the image to file in given image file format.
    # @note Supported pixel formats: Simd.PixelFormat.Gray8, Simd.PixelFormat.Bgr24, Simd.PixelFormat.Bgra32, Simd.PixelFormat.Rgb24, Simd.PixelFormat.Rgba32.
    # @param path - a path to output image file.
    # @param file - a format of output image file. To auto choise format of output file set this parameter to Simd.ImageFile.Undefined.
    # @param quality - a parameter of compression quality (if file format supports it).
    # @return result of the operation.
	def Save(self, path : str, file = Simd.ImageFile.Undefined, quality = 100) -> bool:
		return Lib.ImageSaveToFile(self.Data(), self.Stride(), self.Width(), self.Height(), self.Format(), file, quality, path)
	
    ## Creates a new Simd.Image which points to the region of current image bounded by the rectangle with specified coordinates.
    # @param left - a left side of the region.
    # @param top - a top side of the region.
    # @param right - a right side of the region.
    # @param bottom - a bottom side of the region.
    # @return - a new Simd.Image which points to the subregion of current image.
	def Region(self, left: int, top : int, right : int, bottom : int) :
		if self.Data() == ctypes.c_void_p(0) or right <= left or bottom <= top :
			return Simd.Image()
		left = min(max(left, 0), self.Width())
		top = min(max(top, 0), self.Height())
		right = min(max(right, 0), self.Width())
		bottom = min(max(bottom, 0), self.Height())
		return Simd.Image(self.Format(), right - left, bottom - top, 0, self.Stride(), self.Data() + top * self.Stride() + left * self.Format().PixelSize())
	
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
			return self.Region(0, self.Height() - height, width, self.Height())
		elif position == Position.BottomCenter:
			return self.Region((self.Width() - width) // 2, self.Height() - height, (self.Width() + width) // 2, self.Height())
		elif position == Position.BottomRight:
			return self.Region(self.Width() - width, self.Height() - height, self.Width(), self.Height())
		else :
			return Simd.Image()
		
	## Copies current image.
	# @param dst - an output image. It can be empty.
	# @return copied image.
	def Copy(self, dst = None) :
		if dst == None :
			dst = Image(self.Format(), self.Width(), self.Height())
		if not self.Compatible(dst) :
			raise Exception("Current and output images are incompatible!")
		Lib.Copy(self.Data(), self.Stride(), self.Width(), self.Height(), self.Format().PixelSize(), dst.Data(), dst.Stride())
		return dst
	
	## Converts current image to output image.
	# Current image must be in Gray8, BGR-24, BGRA-32, RGB-24, RGBA32 format.
	# @param dst - an output image in Gray8, BGR-24, BGRA-32, RGB-24, RGBA32 format.
	# @param alpha - a value of output alpha channel (optional).
	def Convert(self, dst, alpha = 255) :
		sf = self.Format()
		if sf != PixelFormat.Gray8 and sf != PixelFormat.Bgr24 and sf != PixelFormat.Bgra32 and sf != PixelFormat.Rgb24 and sf != PixelFormat.Rgba32 :
			raise Exception("Unsupported current pixel format {0}!".format(sf))
		df = dst.Format()
		if df != PixelFormat.Gray8 and df != PixelFormat.Bgr24 and df != PixelFormat.Bgra32 and df != PixelFormat.Rgb24 and df != PixelFormat.Rgba32 :
			raise Exception("Unsupported output pixel format {0}!".format(df))
		if not self.EqualSize(dst) :
			raise Exception("Current and output image have different size!")
		if self.Format() == dst.Format() :
			self.Copy(dst)
			return
		if sf == PixelFormat.Gray8 :
			if df == PixelFormat.Bgr24 :
				Lib.GrayToBgr(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())
			elif df == PixelFormat.Bgra32 :
				Lib.GrayToBgra(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride(), alpha)		
			elif df == PixelFormat.Rgb24 :
				Lib.GrayToBgr(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())		
			elif df == PixelFormat.Rgba32 :
				Lib.GrayToBgra(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride(), alpha)		
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		elif sf == PixelFormat.Bgr24 :
			if df == PixelFormat.Bgra32 :
				Lib.BgrToBgra(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride(), alpha)		
			elif df == PixelFormat.Gray8 :
				Lib.BgrToGray(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())
			elif df == PixelFormat.Rgb24 :
				Lib.BgrToRgb(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())		
			elif df == PixelFormat.Rgba32 :
				Lib.RgbToBgra(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride(), alpha)		
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		elif sf == PixelFormat.Bgra32 :
			if df == PixelFormat.Gray8 :
				Lib.BgraToGray(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())
			elif df == PixelFormat.Bgr24 :
				Lib.BgraToBgr(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())		
			elif df == PixelFormat.Rgb24 :
				Lib.BgraToRgb(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())
			elif df == PixelFormat.Rgba32 :
				Lib.BgraToRgba(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		elif sf == PixelFormat.Rgb24 :
			if df == PixelFormat.Bgr24 :
				Lib.BgrToRgb(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())	
			elif df == PixelFormat.Bgra32 :
				Lib.RgbToBgra(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride(), alpha)		
			elif df == PixelFormat.Gray8 :
				Lib.RgbToGray(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())	
			elif df == PixelFormat.Rgba32 :
				Lib.BgrToBgra(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride(), alpha)		
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		elif sf == PixelFormat.Rgba32 :
			if df == PixelFormat.Bgr24 :
				Lib.BgraToRgb(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())
			elif df == PixelFormat.Bgra32 :
				Lib.BgraToRgba(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())
			elif df == PixelFormat.Gray8 :
				Lib.RgbaToGray(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())	
			elif df == PixelFormat.Rgb24 :
				Lib.BgraToBgr(self.Data(), self.Stride(), self.Width(), self.Height(), dst.Data(), dst.Stride())
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		else :
			raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		
	## Gets converted current image in given format.
	# Current image must be in Gray8, BGR-24, BGRA-32, RGB-24 or RGBA32 format.
	# @param format - a format of output image. It can be Gray8, BGR-24, BGRA-32, RGB-24 or RGBA32.
	# @param alpha - a value of output alpha channel (optional).
	# @return - converted output image in given format.
	def Converted(self, format : PixelFormat, alpha = 255) :
		dst = Image(format, self.Width(), self.Height())
		self.Convert(dst, alpha)
		return dst
	
	## Fills image by value of given pixel.
	# @param pixel - an array of unsigned 8-bit integer with pixel channels. Its size is in range [1..4]. 
	def Fill(self, pixel : array.array('B')) :
		fmt = self.Format()
		if fmt.ChannelSize() != 1 :
			raise Exception("Image.Fill supports only 8-bits channel image!")
		size = len(pixel)
		if size < 1 or size > 4 or size != fmt.ChannelCount() :
			raise Exception("Incompatible pixel size {0} and image type {1} !".format(size, fmt))
		Lib.FillPixel(self.Data(), self.Stride(), self.Width(), self.Height(), pixel)
	
	## Copy image to output numpy.array.
	# @param dst - an output numpy.array with image copy.
	# @return - output numpy.array with image copy. 
	def CopyToNumpyArray(self, dst = None) :
		if (dst == None) or (not dst.shape == [self.Height(), self.Width(), self.Format().PixelSize()]) :
			dst = numpy.empty([self.Height(), self.Width(), self.Format().PixelSize()], dtype = numpy.ubyte)
		size = self.Width() * self.Format().PixelSize()
		for y in range(self.Height()) :
			ctypes.memmove(dst.ctypes.data + size * y, self.Data() + y * self.Stride(), size)
		return dst
		
		
###################################################################################################

## @ingroup python
# The ImageFrame class provides storage and manipulation of frames (multiplanar images).
class ImageFrame():
	## Creates a new frame.
	# @param format - a frame format.
	# @param width - a frame width.
	# @param height - a frame height.
	# @param timestamp - a timestamp of created frame.
	# @param yuvType - a YUV format type of created frame.
	def __init__(self, format = Simd.FrameFormat.Empty, width = 0, height = 0, timestamp = 0.0, yuvType = YuvType.Unknown) :
		self.Clear()
		self.Recreate(format, width, height, yuvType)
		self.__timestamp = timestamp
			
	## Clears frame.
	# Releases plane images, sets to zero all fields.	
	def Clear(self) :
		self.__width = 0
		self.__height = 0
		self.__format = Simd.FrameFormat.Empty
		self.__timestamp = 0.0
		self.__yuvType = YuvType.Unknown
		self.__planes = [ Image(), Image(), Image(), Image() ]
	
	## Recreates the imagee.
	# @param format - a new frame format.
	# @param width - a new frame width.
	# @param height - a new frame height.
	# @param yuvType - a new frame YUV format type.	
	def Recreate(self, format : Simd.FrameFormat, width : int, height : int, yuvType = YuvType.Unknown) :
		if format == self.__format and width == self.__width and height == self.__height :
			return
		self.__format = format
		self.__width = width
		self.__height = height
		self.__yuvType = yuvType
		for plane in self.__planes :
			plane.Clear() 
		if format == Simd.FrameFormat.Nv12 :
			if width % 2 != 0 or height % 2 != 0 :
				raise Exception("Width: {0} and Height: {1} must be even for NV12!".format(width, height))
			self.__planes[0].Recreate(Simd.PixelFormat.Gray8, width, height)
			self.__planes[1].Recreate(Simd.PixelFormat.Uv16, width // 2, height // 2)
			if self.__yuvType == YuvType.Unknown :
				self.__yuvType = YuvType.Bt601
		elif format == Simd.FrameFormat.Yuv420p :
			if width % 2 != 0 or height % 2 != 0 :
				raise Exception("Width: {0} and Height: {1} must be even for YUV420P!".format(width, height))
			self.__planes[0].Recreate(Simd.PixelFormat.Gray8, width, height)
			self.__planes[1].Recreate(Simd.PixelFormat.Gray8, width // 2, height // 2)
			self.__planes[2].Recreate(Simd.PixelFormat.Gray8, width // 2, height // 2)
			if self.__yuvType == YuvType.Unknown :
				self.__yuvType = YuvType.Bt601
		elif format == Simd.FrameFormat.Bgra32 :
			self.__planes[0].Recreate(Simd.PixelFormat.Bgra32, width, height)
			if self.__yuvType != YuvType.Unknown :
				self.__yuvType = YuvType.Unknown
		elif format == Simd.FrameFormat.Bgr24 :
			self.__planes[0].Recreate(Simd.PixelFormat.Bgr24, width, height)
			if self.__yuvType != YuvType.Unknown :
				self.__yuvType = YuvType.Unknown
		elif format == Simd.FrameFormat.Gray8 :
			self.__planes[0].Recreate(Simd.PixelFormat.Gray8, width, height)
			if self.__yuvType != YuvType.Unknown :
				self.__yuvType = YuvType.Unknown
		elif format == Simd.FrameFormat.Rgb24 :
			self.__planes[0].Recreate(Simd.PixelFormat.Rgb24, width, height)
			if self.__yuvType != YuvType.Unknown :
				self.__yuvType = YuvType.Unknown
		elif format == Simd.FrameFormat.Rgba32 :
			self.__planes[0].Recreate(Simd.PixelFormat.Rgba32, width, height)
			if self.__yuvType != YuvType.Unknown :
				self.__yuvType = YuvType.Unknown
		elif format == Simd.FrameFormat.Lab24 :
			self.__planes[0].Recreate(Simd.PixelFormat.Lab24, width, height)
			if self.__yuvType != YuvType.Unknown :
				self.__yuvType = YuvType.Unknown
		else :
			raise Exception("Unsupported {0} frame format!".format(format))
	
	## Gets frame format.
	# @return frame format.	
	def Format(self) -> Simd.FrameFormat :
		return self.__format	
	
	## Gets frame width.
	# @return frame width.	
	def Width(self) -> int :
		return self.__width
	
	## Gets frame height.
	# @return frame height.	
	def Height(self) -> int :
		return self.__height
	
	## Gets frame timestamp.
	# @return frame timestamp.	
	def Timestamp(self) -> float :
		return self.__timestamp
	
	## Gets frame YUV format type.
	# @return frame YUV format type.	
	def GetYuvType(self) -> Simd.YuvType :
		return self.__yuvType
	
	## Gets plane images.
	# @return plane images.	
	def Planes(self)  :
		return self.__planes
	
	## Check if current and other image frame have the same size and pixel format.
	# @param other - other image frame.
	# @return result of checking.
	def Compatible(self, other) -> bool :
		return self.Format() == other.Format() and self.Width() == other.Width() and self.Height() == other.Height() and self.GetYuvType() == other.GetYuvType()

	## Check if current and other image frame have the same size.
	# @param other - other image frame.
	# @return result of checking.
	def EqualSize(self, other) -> bool :
		return self.Width() == other.Width() and self.Height() == other.Height()
	
	## Copies current image frame.
	# @param dst - an output image frame. It can be empty.
	# @return copied image frame.
	def Copy(self, dst = None) :
		if dst == None :
			dst = ImageFrame(self.Format(), self.Width(), self.Height(), self.GetYuvType())
		if not self.Compatible(dst) :
			raise Exception("Current and output images are incompatible!")
		for p in range(len(self.__planes)) :
			self.Planes()[p].Copy(dst.Planes()[p])
		dst.__timestamp = self.__timestamp
		return dst
	
	## Converts current image frame to output image frame.
	# @param dst - an output image frame.
	# @param alpha - a value of output alpha channel (optional).
	def Convert(self, dst, alpha = 255) :
		sf = self.Format()
		df = dst.Format()
		if not self.EqualSize(dst) :
			raise Exception("Current and output image frame have different size!")
		if sf == df :
			self.Copy(dst)
			return
		sy = self.GetYuvType()
		dy = dst.GetYuvType()
		sp = self.Planes()
		dp = dst.Planes()
		if sf == FrameFormat.Nv12 :
			if df == FrameFormat.Yuv420p :
				sp[0].Copy(dp[0])
				Lib.DeinterleaveUv(sp[1].Data(), sp[1].Stride(), sp[1].Width(), sp[1].Height(), dp[1].Data(), dp[1].Stride(), dp[2].Data(), dp[2].Stride())
			elif df == FrameFormat.Bgra32 or df == FrameFormat.Bgr24 or df == FrameFormat.Rgb24 or df == FrameFormat.Rgba32 or df == FrameFormat.Lab24 :
				u = Image(PixelFormat.Gray8, self.Width() // 2, self.Height() // 2)
				v = Image(PixelFormat.Gray8, self.Width() // 2, self.Height() // 2)
				Lib.DeinterleaveUv(sp[1].Data(), sp[1].Stride(), sp[1].Width(), sp[1].Height(), u.Data(), u.Stride(), v.Data(), v.Stride())
				if df == FrameFormat.Bgra32 :
					Lib.Yuv420pToBgra(sp[0].Data(), sp[0].Stride(), u.Data(), u.Stride(), v.Data(), v.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), alpha, sy)
				elif df == FrameFormat.Bgr24 :
					Lib.Yuv420pToBgr(sp[0].Data(), sp[0].Stride(), u.Data(), u.Stride(), v.Data(), v.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), sy)
				elif df == FrameFormat.Rgb24 :
					Lib.Yuv420pToRgb(sp[0].Data(), sp[0].Stride(), u.Data(), u.Stride(), v.Data(), v.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), sy)
				elif df == FrameFormat.Rgba32 :
					bgra = Image(PixelFormat.Bgra32, self.Width(), self.Height())
					Lib.Yuv420pToBgra(sp[0].Data(), sp[0].Stride(), u.Data(), u.Stride(), v.Data(), v.Stride(), self.Width(), self.Height(), bgra.Data(), bgra.Stride(), alpha, sy)
					bgra.Convert(dp[0])
				elif df == FrameFormat.Lab24 :
					bgr = Image(PixelFormat.Bgr24, self.Width(), self.Height())
					Lib.Yuv420pToBgr(sp[0].Data(), sp[0].Stride(), u.Data(), u.Stride(), v.Data(), v.Stride(), self.Width(), self.Height(), bgr.Data(), bgr.Stride(), sy)
					Lib.BgrToLab(bgr.Data(), bgr.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
				else :
					raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
			elif df == FrameFormat.Gray8 :
				if sy == Simd.YuvType.Trect871 :
					sp[0].Copy(dp[0])
				else :
					Lib.YToGray(sp[0].Data(), sp[0].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		elif sf == FrameFormat.Yuv420p :
			if df == FrameFormat.Nv12 :
				sp[0].Copy(dp[0])
				Lib.InterleaveUv(sp[1].Data(), sp[1].Stride(), sp[2].Data(), sp[2].Stride(), sp[1].Width(), sp[1].Height(), dp[1].Data(), dp[1].Stride())
			elif df == FrameFormat.Bgra32 or df == FrameFormat.Bgr24 or df == FrameFormat.Rgb24  or df == FrameFormat.Rgba32 or df == FrameFormat.Lab24 :
				if df == FrameFormat.Bgra32 :
					Lib.Yuv420pToBgra(sp[0].Data(), sp[0].Stride(), sp[1].Data(), sp[1].Stride(), sp[2].Data(), sp[2].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), alpha, sy)
				elif df == FrameFormat.Bgr24 :
					Lib.Yuv420pToBgr(sp[0].Data(), sp[0].Stride(), sp[1].Data(), sp[1].Stride(), sp[2].Data(), sp[2].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), sy)
				elif df == FrameFormat.Rgb24 :
					Lib.Yuv420pToRgb(sp[0].Data(), sp[0].Stride(), sp[1].Data(), sp[1].Stride(), sp[2].Data(), sp[2].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), sy)
				elif df == FrameFormat.Rgba32 :
					bgra = Image(PixelFormat.Bgra32, self.Width(), self.Height())
					Lib.Yuv420pToBgra(sp[0].Data(), sp[0].Stride(), sp[1].Data(), sp[1].Stride(), sp[2].Data(), sp[2].Stride(), self.Width(), self.Height(), bgra.Data(), bgra.Stride(), alpha, sy)
					bgra.Convert(dp[0])
				elif df == FrameFormat.Lab24 :
					bgr = Image(PixelFormat.Bgr24, self.Width(), self.Height())
					Lib.Yuv420pToBgr(sp[0].Data(), sp[0].Stride(), sp[1].Data(), sp[1].Stride(), sp[2].Data(), sp[2].Stride(), self.Width(), self.Height(), bgr.Data(), bgr.Stride(), sy)
					Lib.BgrToLab(bgr.Data(), bgr.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
				else :
					raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
			elif df == FrameFormat.Gray8 :
				if sy == Simd.YuvType.Trect871 :
					sp[0].Copy(dp[0])
				else :
					Lib.YToGray(sp[0].Data(), sp[0].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		elif sf == FrameFormat.Bgra32 :
			if df == FrameFormat.Nv12 :
				u = Image(PixelFormat.Gray8, self.Width() // 2, self.Height() // 2)
				v = Image(PixelFormat.Gray8, self.Width() // 2, self.Height() // 2)
				Lib.BgraToYuv420p(sp[0].Data(), sp[0].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), u.Data(), u.Stride(), v.Data(), v.Stride(), dy)
				Lib.InterleaveUv(u.Data(), u.Stride(), v.Data(), v.Stride(), u.Width(), u.Height(), dp[1].Data(), dp[1].Stride())
			elif df == FrameFormat.Yuv420p :
				Lib.BgraToYuv420p(sp[0].Data(), sp[0].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), dp[1].Data(), dp[1].Stride(), dp[2].Data(), dp[2].Stride(), dy)
			elif df == FrameFormat.Bgr24 or df == FrameFormat.Gray8 or df == FrameFormat.Rgb24 or df == FrameFormat.Rgba32:
				sp[0].Convert(dp[0], alpha)
			elif df == FrameFormat.Lab24 :
				bgr = sp[0].Converted(PixelFormat.Bgr24);
				Lib.BgrToLab(bgr.Data(), bgr.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		elif sf == FrameFormat.Bgr24 :
			if df == FrameFormat.Nv12 :
				u = Image(PixelFormat.Gray8, self.Width() // 2, self.Height() // 2)
				v = Image(PixelFormat.Gray8, self.Width() // 2, self.Height() // 2)
				Lib.BgrToYuv420p(sp[0].Data(), sp[0].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), u.Data(), u.Stride(), v.Data(), v.Stride(), dy)
				Lib.InterleaveUv(u.Data(), u.Stride(), v.Data(), v.Stride(), u.Width(), u.Height(), dp[1].Data(), dp[1].Stride())
			elif df == FrameFormat.Yuv420p :
				Lib.BgrToYuv420p(sp[0].Data(), sp[0].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), dp[1].Data(), dp[1].Stride(), dp[2].Data(), dp[2].Stride(), dy)
			elif df == FrameFormat.Bgra32 or df == FrameFormat.Gray8 or df == FrameFormat.Rgb24 or df == FrameFormat.Rgba32:
				sp[0].Convert(dp[0], alpha)
			elif df == FrameFormat.Lab24 :
				Lib.BgrToLab(sp[0].Data(), sp[0].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		elif sf == FrameFormat.Gray8 :
			if df == FrameFormat.Nv12 :
				if dy == Simd.YuvType.Trect871 :
					sp[0].Copy(dp[0])
				else :
					Lib.GrayToY(sp[0].Data(), sp[0].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
				dp[1].Fill([128, 128])
			elif df == FrameFormat.Yuv420p :
				if dy == Simd.YuvType.Trect871 :
					sp[0].Copy(dp[0])
				else :
					Lib.GrayToY(sp[0].Data(), sp[0].Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
				dp[1].Fill([128])
				dp[2].Fill([128])
			elif df == FrameFormat.Bgra32 or df == FrameFormat.Bgr24 or df == FrameFormat.Rgb24 or df == FrameFormat.Rgba32:
				sp[0].Convert(dp[0], alpha)
			elif df == FrameFormat.Lab24 :
				bgr = sp[0].Converted(PixelFormat.Bgr24);
				Lib.BgrToLab(bgr.Data(), bgr.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		elif sf == FrameFormat.Rgb24 :
			if df == FrameFormat.Nv12 :
				bgr = dp[0].Converted(PixelFormat.Bgr24)
				u = Image(PixelFormat.Gray8, self.Width() // 2, self.Height() // 2)
				v = Image(PixelFormat.Gray8, self.Width() // 2, self.Height() // 2)
				Lib.BgrToYuv420p(bgr.Data(), bgr.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), u.Data(), u.Stride(), v.Data(), v.Stride(), dy)
				Lib.InterleaveUv(u.Data(), u.Stride(), v.Data(), v.Stride(), u.Width(), u.Height(), dp[1].Data(), dp[1].Stride())
			elif df == FrameFormat.Yuv420p :
				bgr = dp[0].Converted(PixelFormat.Bgr24)
				Lib.BgrToYuv420p(bgr.Data(), bgr.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), dp[1].Data(), dp[1].Stride(), dp[2].Data(), dp[2].Stride(), dy)
			elif df == FrameFormat.Bgra32 or df == FrameFormat.Bgr24 or df == FrameFormat.Gray8 or df == FrameFormat.Rgba32:
				sp[0].Convert(dp[0], alpha)
			elif df == FrameFormat.Lab24 :
				bgr = sp[0].Converted(PixelFormat.Bgr24);
				Lib.BgrToLab(bgr.Data(), bgr.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		elif sf == FrameFormat.Rgba32 :
			if df == FrameFormat.Nv12 :
				bgra = dp[0].Converted(PixelFormat.Bgra32)
				u = Image(PixelFormat.Gray8, self.Width() // 2, self.Height() // 2)
				v = Image(PixelFormat.Gray8, self.Width() // 2, self.Height() // 2)
				Lib.BgraToYuv420p(bgra.Data(), bgra.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), u.Data(), u.Stride(), v.Data(), v.Stride(), dy)
				Lib.InterleaveUv(u.Data(), u.Stride(), v.Data(), v.Stride(), u.Width(), u.Height(), dp[1].Data(), dp[1].Stride())
			elif df == FrameFormat.Yuv420p :
				bgra = dp[0].Converted(PixelFormat.Bgra32)
				Lib.BgraToYuv420p(bgra.Data(), bgra.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride(), dp[1].Data(), dp[1].Stride(), dp[2].Data(), dp[2].Stride(), dy)
			elif df == FrameFormat.Bgra32 or df == FrameFormat.Bgr24 or df == FrameFormat.Gray8 or df == FrameFormat.Rgb24:
				sp[0].Convert(dp[0], alpha)
			elif df == FrameFormat.Lab24 :
				bgr = sp[0].Converted(PixelFormat.Bgr24);
				Lib.BgrToLab(bgr.Data(), bgr.Stride(), self.Width(), self.Height(), dp[0].Data(), dp[0].Stride())
			else :
				raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		else :
			raise Exception("Not implemented conversion {0} to {1} !".format(sf, df))
		
	## Gets converted current image frame in given format.
	# @param format - a format of output image frame.
	# @param alpha - a value of output alpha channel (optional).
    # @param yuvType - a type of output YUV image (optional).
	# @return - converted output image frame in given format.
	def Converted(self, format : FrameFormat, alpha = 255, yuvType = YuvType.Unknown) :
		dst = ImageFrame(format, self.Width(), self.Height(), yuvType)
		self.Convert(dst, alpha)
		return dst

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
	elif src == Simd.PixelFormat.Lab24 : return ResizeChannel.Byte
	else : raise Exception("Can't {0} convert to Simd.ResizeChannel !".format(src))

###################################################################################################
	
## @ingroup python
# Gets 8-bit gray image saturated sum of absolute gradient for every point of source 8-bit gray image.
# @param src - an input 8-bit gray image.
# @param dst - an output 8-bit gray image with sum of absolute gradient. Can be empty.
# @return - output 8-bit gray image with sum of absolute gradient.
def AbsGradientSaturatedSum(src : Image, dst : Image) -> Image :
	if src.Format() != Simd.PixelFormat.Gray8 :
		raise Exception("Unsupported input pixel format {0} != Simd.PixelFormat.Gray8!".format(src.Format()))
	if dst.Format() == Simd.PixelFormat.Empty :
		dst.Recreate(src.Format(), src.Width(), src.Height())
	if not src.Compatible(dst) :
		raise Exception("Input and output images are incompatible!")
	Lib.AbsGradientSaturatedSum(src.Data(), src.Stride(), src.Width(), src.Height(), dst.Data(), dst.Stride())
	return dst

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
def Resized(src : Image, width :int, height: int, method = Simd.ResizeMethod.Bilinear) -> Image :
	dst = Image(src.Format(), width, height)
	Simd.Resize(src, dst, method)
	return dst

##  @ingroup python
# Sets image to the input of neural network of <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.
# @param src - an input image. There are following supported pixel format: aSimd.PixelFormat.Gray8, Simd.PixelFormat.Bgr24, Simd.PixelFormat.Bgra32, Simd.PixelFormat.Rgb24, Simd.PixelFormat.Rgba32.
# @param lower - an array with lower bound of values of the output tensor. The size of the array have to correspond number of channels in the output image tensor.
# @param upper - an array with upper bound of values of the output tensor. The size of the array have to correspond number of channels in the output image tensor.
# @param dst - a pointer to the output 32-bit float image tensor.
# @param channels - a number of channels in the output image tensor. It can be 1 or 3.
# @param format - a format of output image tensor. There are supported following tensor formats: Simd.TensorFormat.Nchw, Simd.TensorFormat.Nhwc.
# @param isRgb - is channel order of output tensor is RGB or BGR. Its default value is false.
def SynetSetInput(src : Image, lower : array.array('f'), upper : array.array('f'), dst : ctypes.c_void_p, channels : int, format : Simd.TensorFormat, isRgb = False) :
	Lib.SynetSetInput(src.Data(), src.Width(), src.Height(), src.Stride(), src.Format(), lower, upper, dst, channels, format, isRgb)
	
##  @ingroup python
# Performs warp affine for current image.
# @param src - an input image to warp affine.
# @param mat - a pointer to 2x3 matrix with coefficients of affine warp.
# @param dst - a background input/output image.
# @param flags - a flags of algorithm parameters. By default is equal to Simd.WarpAffineFlags.ChannelByte | Simd.WarpAffineFlags.InterpBilinear | Simd.WarpAffineFlags.BorderConstant.
# @param  border - an array with color of border. The size of the array must be equal to channels.
#                  It parameter is actual for Simd.WarpAffineFlags.BorderConstant flag. 
def WarpAffine(src : Image, mat: array.array('f'), dst : Image, flags = (Simd.WarpAffineFlags.ChannelByte | Simd.WarpAffineFlags.InterpBilinear | Simd.WarpAffineFlags.BorderConstant), border = array.array('B', [])) :
	if src.Format() != dst.Format() or src.Format().ChannelSize() != 1 :
		raise Exception("Uncompartible image format for Warp Affine!")
	if (flags & Simd.WarpAffineFlags.ChannelMask) != Simd.WarpAffineFlags.ChannelByte :
		raise Exception("Uncompartible Warp Affine flag!")
	
	context = Lib.WarpAffineInit(src.Width(), src.Height(), src.Stride(), dst.Width(), dst.Height(), dst.Stride(), src.Format().ChannelCount(), mat, flags, border)
	if context == ctypes.c_void_p(0) :
		raise Exception("Can't create Warp Affine context !")
	
	Lib.WarpAffineRun(context, src.Data(), dst.Data())
	Lib.Release(context)

