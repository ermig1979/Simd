import argparse
import os
import ctypes
import pathlib
import sys

import Simd

###################################################################################################

def PrintInfoTest(simd: Simd.Lib, args) :
	#print("Simd version: {0}.".format(simd.Version()))
	#print("CPU model: {0}.".format(simd.CpuDesc(Simd.CpuDesc.Model)))
	print("{0}\n".format(simd.SysInfo()))

###################################################################################################

def GetSetParamsTest(simd: Simd.Lib, args) :
	threads = simd.GetThreadNumber()
	mode = simd.GetFastMode()
	
	simd.SetThreadNumber(-1)
	simd.SetFastMode(True)
	
	print("Simd thread number: {0}, fast mode: {1}.".format(simd.GetThreadNumber(), simd.GetFastMode()))
	
	simd.SetFastMode(mode)
	simd.SetThreadNumber(threads)

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="Simd", description="Simd Python Wrapper.")
	parser.add_argument("-b", "--bin", help="Directory with binary files.", required=False, type=str, default=".")
	args = parser.parse_args()
	
	print("Start testing of Simd Python Wrapper:\n")
	
	simd = Simd.Lib(args.bin)
	
	PrintInfoTest(simd, args)
	
	GetSetParamsTest(simd, args)
	
	print("\nSimd Python Wrapper test ended successfully!")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
