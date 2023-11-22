import argparse
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
# Describes pixel format type.	
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
    ## A single channel 64-bit integer pixel format. */
	Int64 = 7
    ## A single channel 32-bit float point pixel format. */
	Float = 8
    ## A single channel 64-bit float point pixel format. */
	Double = 9
    ## A 8-bit Bayer pixel format (GRBG). */
	BayerGrbg = 10
    ## A 8-bit Bayer pixel format (GBRG). */
	BayerGbrg = 11
    ## A 8-bit Bayer pixel format (RGGB). */
	BayerRggb = 12
    ## A 8-bit Bayer pixel format (BGGR). */
	BayerBggr = 13
    ## A 24-bit (3 8-bit channels) HSV (Hue, Saturation, Value) pixel format. */
	Hsv24 = 14
    ## A 24-bit (3 8-bit channels) HSL (Hue, Saturation, Lightness) pixel format. */
	Hsl24 = 15
    ## A 24-bit (3 8-bit channels) RGB (Red, Green, Blue) pixel format. */
	Rgb24 = 16
    ## A 32-bit (4 8-bit channels) RGBA (Red, Green, Blue, Alpha) pixel format. */
	Rgba32 = 17
    ## A 16-bit (2 8-bit channels) UYVY422 pixel format. */
	Uyvy16 = 18
    ## A 32-bit (4 8-bit channels) ARGB (Alpha, Red, Green, Blue) pixel format. */
	Argb32 = 19

###################################################################################################

## @ingroup python
# A wrapper around %Simd Library API.
class Lib():
	lib : ctypes.CDLL
	
	## Simd.Lib constructor
	# @param dir - a directory with %Simd Library binaries (Simd.dll or libSimd.so).
	def __init__(self, dir: str):
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

		self.lib = ctypes.CDLL(path)
		
		self.lib.SimdVersion.argtypes = []
		self.lib.SimdVersion.restype = ctypes.c_char_p 
		
		self.lib.SimdCpuDesc.argtypes = [ ctypes.c_int ]
		self.lib.SimdCpuDesc.restype = ctypes.c_char_p 
		
		self.lib.SimdCpuInfo.argtypes = [ ctypes.c_int ]
		self.lib.SimdCpuInfo.restype = ctypes.c_size_t 
		
		self.lib.SimdPerformanceStatistic.argtypes = []
		self.lib.SimdPerformanceStatistic.restype = ctypes.c_char_p 
		
		self.lib.SimdAllocate.argtypes = [ ctypes.c_size_t, ctypes.c_size_t ]
		self.lib.SimdAllocate.restype = ctypes.c_void_p 
		
		self.lib.SimdFree.argtypes = [ ctypes.c_void_p ]
		self.lib.SimdFree.restype = None
		
		self.lib.SimdAlign.argtypes = [ ctypes.c_size_t, ctypes.c_size_t ]
		self.lib.SimdAlign.restype = ctypes.c_size_t 
		
		self.lib.SimdAlignment.argtypes = []
		self.lib.SimdAlignment.restype = ctypes.c_size_t 
		
		self.lib.SimdRelease.argtypes = [ ctypes.c_void_p ]
		self.lib.SimdRelease.restype = None
		
		self.lib.SimdGetThreadNumber.argtypes = []
		self.lib.SimdGetThreadNumber.restype = ctypes.c_size_t 

		self.lib.SimdSetThreadNumber.argtypes = [ ctypes.c_size_t ]
		self.lib.SimdSetThreadNumber.restype = None 
		
		self.lib.SimdEmpty.argtypes = []
		self.lib.SimdEmpty.restype = None
		
		self.lib.SimdGetFastMode.argtypes = []
		self.lib.SimdGetFastMode.restype = ctypes.c_bool
		
		self.lib.SimdSetFastMode.argtypes = [ ctypes.c_bool ]
		self.lib.SimdSetFastMode.restype = None
		
	## Gets verion of %Simd Library.
	# @return A string with version.
	def Version(self) -> str: 
		ptr = self.lib.SimdVersion()
		return str(ptr, encoding='utf-8')
	
	## Gets string with CPU description.
	# @param type - a type of CPU description.
	# @return A string with system description.
	def CpuDesc(self, type: Simd.CpuDesc) -> str: 
		ptr = self.lib.SimdCpuDesc(type.value)
		return str(ptr, encoding='utf-8')
	
	## Gets information about CPU.
	# @param type - a type of CPU information.
	# @return integer value of given CPU parameter.
	def CpuInfo(self, type: Simd.CpuInfo) -> int: 
		return self.lib.SimdCpuInfo(type.value)

	## Gets string with CPU and %Simd Library description.
	# @return string with CPU and %Simd Library description.	
	def SysInfo(self) -> str: 
		info = ""
		info += "Simd Library: {0}".format(self.Version())
		info += "; CPU: {0}".format(self.CpuDesc(Simd.CpuDesc.Model))
		info += "; System sockets: {0}".format(self.CpuInfo(Simd.CpuInfo.Sockets))
		info += ", Cores: {0}".format(self.CpuInfo(Simd.CpuInfo.Cores))
		info += ", Threads: {0}".format(self.CpuInfo(Simd.CpuInfo.Threads))
		info += "; Cache L1D: {:.0f} KB".format(self.CpuInfo(Simd.CpuInfo.CacheL1) / 1024)
		info += ", L2: {:.0f} KB".format(self.CpuInfo(Simd.CpuInfo.CacheL2) / 1024)
		info += ", L3: {:.1f} MB".format(self.CpuInfo(Simd.CpuInfo.CacheL3) / 1024 / 1024)
		info += ", RAM: {:.1f} GB".format(self.CpuInfo(Simd.CpuInfo.RAM) / 1024 / 1024 / 1024)
		info += "; Available SIMD:"
		if self.CpuInfo(Simd.CpuInfo.AMX) > 0 :
			info += " AMX"
		if self.CpuInfo(Simd.CpuInfo.AVX512BF16) > 0 :
			info += " AVX-512VBF16"
		if self.CpuInfo(Simd.CpuInfo.AVX512VNNI) > 0 :
			info += " AVX-512VNNI"
		if self.CpuInfo(Simd.CpuInfo.AVX512BW) > 0 :
			info += " AVX-512BW AVX-512F"
		if self.CpuInfo(Simd.CpuInfo.AVX2) > 0 :
			info += " AVX2 FMA"
		if self.CpuInfo(Simd.CpuInfo.AVX) > 0 :
			info += " AVX"
		if self.CpuInfo(Simd.CpuInfo.SSE41) > 0 :
			info += " SSE4.1 SSSE3 SSE3 SSE2 SSE"
		if self.CpuInfo(Simd.CpuInfo.NEON) > 0 :
			info += " NEON"
		if self.CpuInfo(Simd.CpuInfo.VSX) > 0 :
			info += " VSX"
		if self.CpuInfo(Simd.CpuInfo.VMX) > 0 :
			info += " Altivec"
		return info
	
	## Gets string with internal %Simd Library performance statistics.
	# @note %Simd Library must be built with switched on SIMD_PERF flag.
	# @return string with internal %Simd Library performance statistics.	
	def PerformanceStatistic(self) -> str: 
		ptr = self.lib.SimdPerformanceStatistic()
		return str(ptr, encoding='utf-8')
	
	## Gets number of threads used by %Simd Library to parallelize some algorithms.
	# @return thread number used by %Simd Library.	
	def GetThreadNumber(self) -> int: 
		return self.lib.SimdGetThreadNumber()
	
	## Sets number of threads used by %Simd Library to parallelize some algorithms.
	# @param threadNumber - number used by %Simd Library.	
	def SetThreadNumber(self, threadNumber: int) : 
		self.lib.SimdSetThreadNumber(threadNumber)
		
	## Gets current CPU Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) flags. It is used in order to process subnormal numbers.
	# @return current 'fast' mode.	
	def GetFastMode(self) -> bool: 
		return self.lib.SimdGetFastMode()
	
	## Sets current CPU Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) flags. It is used in order to process subnormal numbers.
	# @param fast - a value of 'fast' mode to set.	
	def SetFastMode(self, fast: bool) : 
		self.lib.SimdSetFastMode(fast)
	
###################################################################################################

class Image():
	lib : Simd.Lib
	
	def __init__(self, lib: Simd.Lib) :
		self.lib = lib
