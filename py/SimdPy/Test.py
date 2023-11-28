import argparse
from mimetypes import init
import os
import ctypes
import pathlib
import sys
import array

import Simd

###################################################################################################

def PrintInfoTest(args) :
	#print("Simd version: {0}.".format(Simd.Lib.Version()))
	#print("CPU model: {0}.".format(Simd.Lib.CpuDesc(Simd.CpuDesc.Model)))
	print("{0}\n".format(Simd.Lib.SysInfo()))

###################################################################################################

def GetSetParamsTest(args) :
	threads = Simd.Lib.GetThreadNumber()
	mode = Simd.Lib.GetFastMode()
	
	Simd.Lib.SetThreadNumber(-1)
	Simd.Lib.SetFastMode(True)
	
	print("Simd thread number: {0}, fast mode: {1}. \n".format(Simd.Lib.GetThreadNumber(), Simd.Lib.GetFastMode()))
	
	Simd.Lib.SetFastMode(mode)
	Simd.Lib.SetThreadNumber(threads)
	
###################################################################################################

def ImageTest(args) :
	image = Simd.Image(Simd.PixelFormat.Bgr24, 120, 90)
	image.Load("city.jpg")
	crc32 = Simd.Lib.Crc32(image.Data(), image.Height() * image.Stride())
	print("Creates image: {0} {1}x{2}, Crc32: {3:X}".format(image.Format(), image.Width(), image.Height(), crc32))
	center = image.RegionAt(image.Width() // 2, image.Height() // 2, Simd.Position.MiddleCenter)
	center.Save("center.jpg", Simd.ImageFile.Jpeg, 85)
	resized = Simd.Resized(image, image.Width() // 2, image.Height() // 2)
	resized.Save("resized.jpg", Simd.ImageFile.Jpeg, 85)
	
###################################################################################################

def SynetSetInputTest(args) :
	print("\nSynetSetInputTest:")
	width = 128
	height = 128
	channels = 3
	image = Simd.Image()
	image.Load("city.jpg")
	resized = Simd.Resized(image, width, height, Simd.ResizeMethod.Area)
	lower = [0.0, 0.0, 0.0]
	upper = [1.0, 1.0, 1.0]
	input = Simd.Lib.Allocate(channels * height * width * 4, Simd.Lib.Alignment())
	Simd.SynetSetInput(resized, lower, upper, input, channels, Simd.TensorFormat.Nhwc)
	Simd.Lib.Free(input)
	

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="Simd", description="Simd Python Wrapper.")
	parser.add_argument("-b", "--bin", help="Directory with binary files.", required=False, type=str, default=".")
	args = parser.parse_args()
	
	print("Start testing of Simd Python Wrapper:\n")
	
	Simd.Lib.Init(args.bin)
	
	PrintInfoTest(args)
	
	GetSetParamsTest(args)
	
	ImageTest(args)
	
	SynetSetInputTest(args) 
	
	print("\nSimd Python Wrapper test ended successfully!")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
