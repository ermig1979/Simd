import argparse
import os
import ctypes
import pathlib
import sys
import array
import enum

Simd = sys.modules[__name__]

###################################################################################################

class CpuDesc(enum.Enum) :
	Model = 0

###################################################################################################

class CpuInfo(enum.Enum) :	
    Sockets = 0
    Cores = 1
    Threads = 2
    CacheL1 = 3
    CacheL2 = 4
    CacheL3 = 5
    RAM = 6
    SSE41 = 7
    AVX = 8
    AVX2 = 9
    AVX512BW = 10
    AVX512VNNI = 11
    AVX512BF16 = 12
    AMX = 13
    VMX = 14
    VSX = 15
    NEON = 16

###################################################################################################

class Lib():
	lib : ctypes.CDLL
	
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
		
		self.lib.SimdRelease.argtypes = [ ctypes.c_void_p ]
		
	def Version(self) -> str: 
		ptr = self.lib.SimdVersion()
		return str(ptr, encoding='utf-8')
	
	def CpuDesc(self, type: Simd.CpuDesc) -> str: 
		ptr = self.lib.SimdCpuDesc(type.value)
		return str(ptr, encoding='utf-8')
	
	def CpuInfo(self, type: Simd.CpuInfo) -> int: 
		return self.lib.SimdCpuInfo(type.value)
	
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
	
###################################################################################################
