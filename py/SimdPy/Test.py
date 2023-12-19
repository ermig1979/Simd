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
	print("\nPrintInfoTest: ", end="")
	#print("Simd version: {0}.".format(Simd.Lib.Version()))
	#print("CPU model: {0}.".format(Simd.Lib.CpuDesc(Simd.CpuDesc.Model)))
	print("{0}. ".format(Simd.Lib.SysInfo()), end="")
	print("OK.")

###################################################################################################

def GetSetParamsTest(args) :
	print("\nGetSetParamsTest: ", end="")
	
	threads = Simd.Lib.GetThreadNumber()
	mode = Simd.Lib.GetFastMode()
	
	Simd.Lib.SetThreadNumber(-1)
	Simd.Lib.SetFastMode(True)
	
	print("Simd thread number: {0}, fast mode: {1}. ".format(Simd.Lib.GetThreadNumber(), Simd.Lib.GetFastMode()), end="")
	
	Simd.Lib.SetFastMode(mode)
	Simd.Lib.SetThreadNumber(threads)
	print("OK.")
	
###################################################################################################

def ImagePaintTest(args) :
	print("\nImagePaintTest: ", end="")
	image = Simd.Image(Simd.PixelFormat.Bgr24, 120, 90)
	image.Load("city.jpg")
	crc32 = Simd.Lib.Crc32(image.Data(), image.Height() * image.Stride())
	print("Creates image: {0} {1}x{2}, Crc32: {3:X}. ".format(image.Format(), image.Width(), image.Height(), crc32), end="")
	Simd.FillPixel(image.Region(100, 100, 200, 200), [0, 0, 255])
	Simd.FillPixel(image.RegionAt(300, 300, Simd.Position.MiddleCenter), [0, 255, 0])
	image.Save("painted.jpg")
	print("OK.")
	
###################################################################################################

def ImageFrameTest(args) :
	print("\nImageFrameTest: ", end="")
	image = Simd.Image()
	image.Load("city.jpg", Simd.PixelFormat.Bgra32)
	frame = Simd.ImageFrame(Simd.FrameFormat.Bgra32, image.Width(), image.Height())
	frame.Planes()[0] = image.Copy();
	copy = frame.Copy()
	yuv = frame.Converted(Simd.FrameFormat.Yuv420p)
	#resized = Simd.Resized(image, image.Width() // 4, image.Height() // 4, Simd.ResizeMethod.Area)
	#resized.Save("resized.jpg", Simd.ImageFile.Jpeg, 85)
	print("OK.")
	
###################################################################################################

def ImageAbsGradientSaturatedSumTest(args) :
	print("\nImageAbsGradientSaturatedSumTest: ", end="")
	image = Simd.Image()
	image.Load("city.jpg", Simd.PixelFormat.Gray8)
	agss = Simd.AbsGradientSaturatedSum(image)
	agss.Save("AbsGradientSaturatedSum.jpg")
	print("OK.")
	
###################################################################################################

def ConvertImageTest(args) :
	print("\nConvertImageTest: ", end="")
	image = Simd.Image()
	image.Load("city.jpg", Simd.PixelFormat.Rgba32)
	converted = image.Converted(Simd.PixelFormat.Gray8)
	converted.Save("converted.jpg", Simd.ImageFile.Jpeg, 85)
	print("OK.")
	
###################################################################################################

def ImageResizeTest(args) :
	print("\nImageResizeTest: ", end="")
	image = Simd.Image()
	image.Load("city.jpg")
	resized = Simd.Resized(image, image.Width() // 4, image.Height() // 4, Simd.ResizeMethod.Area)
	resized.Save("resized.jpg", Simd.ImageFile.Jpeg, 85)
	print("OK.")
	
###################################################################################################

def ImageWarpAffineTest(args) :
	print("\nImageWarpAffineTest: ", end="")
	image = Simd.Image(Simd.PixelFormat.Bgr24, 120, 90)
	image.Load("city.jpg")
	center = image.RegionAt(image.Width() // 2, image.Height() // 2, Simd.Position.MiddleCenter).Copy()
	mat = [ 0.7, -0.7, float(image.Width() / 4), 0.7, 0.7, float(-image.Width() / 4)]
	Simd.WarpAffine(center, mat, image, Simd.WarpAffineFlags.ChannelByte | Simd.WarpAffineFlags.InterpBilinear | Simd.WarpAffineFlags.BorderTransparent)
	image.Save("warp_affine.jpg")
	print("OK.")
	
###################################################################################################

def SynetSetInputTest(args) :
	print("\nSynetSetInputTest: ", end="")
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
	print("OK.")

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="Simd", description="Simd Python Wrapper.")
	parser.add_argument("-b", "--bin", help="Directory with binary files.", required=False, type=str, default=".")
	args = parser.parse_args()
	
	print("Start testing of Simd Python Wrapper:")
	
	Simd.Lib.Init(args.bin)
	
	PrintInfoTest(args)
	
	GetSetParamsTest(args)
	
	ImagePaintTest(args)
	
	ImageAbsGradientSaturatedSumTest(args)
	
	ConvertImageTest(args)

	ImageResizeTest(args)
	
	ImageFrameTest(args)
	
	ImageWarpAffineTest(args)
	
	SynetSetInputTest(args) 
	
	print("\nSimd Python Wrapper test ended successfully!")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
