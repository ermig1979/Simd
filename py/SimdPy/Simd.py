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
	# return a pointer to allocated memory.
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
	# return an aligned size.
	def Align(size : int, align : int) -> int:
		return Lib.__lib.SimdAlign(size, align)
	
    ## Gets alignment required for the most productive work of Simd Library.
	# return a required alignment.
	def Alignment() -> int:
		return Lib.__lib.SimdAlignment()
	
	## Gets number of threads used by %Simd Library to parallelize some algorithms.
	# @return thread number used by %Simd Library.	
	def GetThreadNumber() -> int: 
		return Lib.__lib.SimdGetThreadNumber()
	
	## Sets number of threads used by %Simd Library to parallelize some algorithms.
	# @param threadNumber - number used by %Simd Library.	
	def SetThreadNumber(threadNumber: int) : 
		Lib.__lib.SimdSetThreadNumber(threadNumber)
		
	## Gets current CPU Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) flags. It is used in order to process subnormal numbers.
	# @return current 'fast' mode.	
	def GetFastMode() -> bool: 
		return Lib.__lib.SimdGetFastMode()
	
	## Sets current CPU Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) flags. It is used in order to process subnormal numbers.
	# @param fast - a value of 'fast' mode to set.	
	def SetFastMode(fast: bool) : 
		Lib.__lib.SimdSetFastMode(fast)
	
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
	
		
