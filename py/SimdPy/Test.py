###################################################################################################
# Simd Library (http://ermig1979.github.io/Simd).
#
# Copyright (c) 2011-2024 Yermalayeu Ihar.
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
import os
import ctypes
import pathlib
import sys
import array
import numpy

import Simd

###################################################################################################

def LoadTestImage(args, fmt = Simd.PixelFormat.Rgb24) -> Simd.Image :
	if not os.path.isdir(args.root):
		raise Exception("Project root directory '{0}' is not exist!".format(args.root))
	path = "{0}/data/image/city.jpg".format(args.root)
	image = Simd.Image()
	if not image.Load(path, fmt) :
		raise Exception("Can't load image '{0}' in {1} format!".format(path, fmt))
	return image

###################################################################################################

def PrintInfoTest(args) :
	#print("Simd version: {0}.".format(Simd.Lib.Version()))
	#print("CPU model: {0}.".format(Simd.Lib.CpuDesc(Simd.CpuDesc.Model)))
	print("{0}. ".format(Simd.Lib.SysInfo()), end="")

###################################################################################################

def GetSetParamsTest(args) :
	threads = Simd.Lib.GetThreadNumber()
	mode = Simd.Lib.GetFastMode()
	
	Simd.Lib.SetThreadNumber(-1)
	Simd.Lib.SetFastMode(True)
	
	print("Simd thread number: {0}, fast mode: {1}. ".format(Simd.Lib.GetThreadNumber(), Simd.Lib.GetFastMode()), end="")
	
	Simd.Lib.SetFastMode(mode)
	Simd.Lib.SetThreadNumber(threads)
	
###################################################################################################

def ImagePaintTest(args) :
	image = LoadTestImage(args)
	crc32 = Simd.Lib.Crc32(image.Data(), image.Height() * image.Stride())
	print("Creates image: {0} {1}x{2}, Crc32: {3:X}. ".format(image.Format(), image.Width(), image.Height(), crc32), end="")
	image.Region(100, 100, 200, 200).Fill([0, 0, 255])
	image.RegionAt(300, 300, Simd.Position.MiddleCenter).Fill([0, 255, 0])
	image.Save("painted.jpg")
	
###################################################################################################

def ImageFrameTest(args) :
	formats = [Simd.FrameFormat.Nv12, Simd.FrameFormat.Yuv420p, Simd.FrameFormat.Bgra32, Simd.FrameFormat.Bgr24, Simd.FrameFormat.Gray8, Simd.FrameFormat.Rgb24, Simd.FrameFormat.Rgba32]
	image = LoadTestImage(args)
	frame = Simd.ImageFrame(Simd.FrameFormat.Rgb24, image.Width(), image.Height())
	frame.Planes()[0] = image.Copy()
	for i in range(len(formats)) :
		frameI = frame.Converted(formats[i])
		for j in range(len(formats)) :
			frameJ = frameI.Converted(formats[j])
			#frameJ.Save("converted_image_{0}_to_{1}.jpg".format(formats[i], formats[j]), Simd.ImageFile.Jpeg, 85)
	
###################################################################################################

def ImageAbsGradientSaturatedSumTest(args) :
	image = LoadTestImage(args, Simd.PixelFormat.Gray8)
	agss = Simd.AbsGradientSaturatedSum(image, Simd.Image())
	agss.Save("AbsGradientSaturatedSum.jpg")
	
###################################################################################################

def ConvertImageTest(args) :
	formats = [Simd.PixelFormat.Gray8, Simd.PixelFormat.Bgr24, Simd.PixelFormat.Bgra32, Simd.PixelFormat.Rgb24, Simd.PixelFormat.Rgba32]
	orig = LoadTestImage(args)
	for i in range(len(formats)) :
		imgI = orig.Converted(formats[i])
		for j in range(len(formats)) :
			imgJ = imgI.Converted(formats[j])
			#imgJ.Save("converted_image_{0}_to_{1}.jpg".format(formats[i], formats[j]), Simd.ImageFile.Jpeg, 85)
	
###################################################################################################

def ImageResizeTest(args) :
	image = LoadTestImage(args)
	resized = Simd.Resized(image, image.Width() // 4, image.Height() // 4, Simd.ResizeMethod.Area)
	resized.Save("resized.jpg", Simd.ImageFile.Jpeg, 85)
	
###################################################################################################

def ImageWarpAffineTest(args) :
	image = LoadTestImage(args)
	center = image.RegionAt(image.Width() // 2, image.Height() // 2, Simd.Position.MiddleCenter).Copy()
	mat = [ 0.7, -0.7, float(image.Width() / 4), 0.7, 0.7, float(-image.Width() / 4)]
	Simd.WarpAffine(center, mat, image, Simd.WarpAffineFlags.ChannelByte | Simd.WarpAffineFlags.InterpBilinear | Simd.WarpAffineFlags.BorderTransparent)
	image.Save("warp_affine.jpg")

###################################################################################################

def ImageToNumpyArrayTest(args) :
	image = LoadTestImage(args)
	array = image.CopyToNumpyArray()
	copy = Simd.Image(image.Format(), image.Width(), image.Height(), 0, image.Width()* image.Format().PixelSize(), array.ctypes.data)
	#print(array.shape())
	copy.Save("numpy.array.jpg")

	
###################################################################################################

def SynetSetInputTest(args) :
	width = 128
	height = 128
	channels = 3
	image = LoadTestImage(args)
	resized = Simd.Resized(image, width, height, Simd.ResizeMethod.Area)
	lower = [0.0, 0.0, 0.0]
	upper = [1.0, 1.0, 1.0]
	input = Simd.Lib.Allocate(channels * height * width * 4, Simd.Lib.Alignment())
	Simd.SynetSetInput(resized, lower, upper, input, channels, Simd.TensorFormat.Nhwc, True)
	Simd.Lib.Free(input)

###################################################################################################

def InitTestList(args) :
	tests = []
	tests.append(PrintInfoTest)
	tests.append(GetSetParamsTest)
	tests.append(ImagePaintTest)
	tests.append(ImageAbsGradientSaturatedSumTest)
	tests.append(ConvertImageTest)
	tests.append(ImageResizeTest)
	tests.append(ImageFrameTest)
	tests.append(ImageWarpAffineTest)
	tests.append(ImageToNumpyArrayTest)
	tests.append(SynetSetInputTest) 
	
	filtered = []
	for test in tests:
		if len(args.include) > 0 :
			skip = True
			for include in args.include :
				if test.__name__.find(include) != -1 :
					skip = False
			if skip :
				continue
			
		if len(args.exclude) > 0 :
			skip = False
			for exclude in args.exclude :
				if test.__name__.find(exclude) != -1 :
					skip = True
			if skip :
				continue
		filtered.append(test)
	return filtered

###################################################################################################

def RunTests(args, tests) :
	print("Start testing of Simd Python Wrapper:")
	
	for test in tests :
		print("\n{0}: ".format(test.__name__), end="")
		test(args)
		print("OK.")
	
	print("\nSimd Python Wrapper test ended successfully!\n")

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="Simd", description="Simd Python Wrapper.")
	parser.add_argument("-b", "--bin", help="Directory with binary files.", required=False, type=str, default="")
	parser.add_argument("-r", "--root", help="Simd Library root directory.", required=False, type=str, default=".")
	parser.add_argument("-i", "--include", help="Include tests filter.", required=False, default=[], action="append")
	parser.add_argument("-e", "--exclude", help="Exclude tests filter.", required=False, default=[], action="append")
	args = parser.parse_args()
	
	Simd.Lib.Init(args.bin)
	
	RunTests(args, InitTestList(args))
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
