import argparse
import os
import ctypes
import pathlib
import sys
import array
import enum

###################################################################################################

class Simd():
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
		
		self.lib.SimdRelease.argtypes = [ ctypes.c_void_p ]
		
	def Version(self) -> str: 
		ptr = self.lib.SimdVersion()
		return str(ptr, encoding='utf-8')
	
	
###################################################################################################
