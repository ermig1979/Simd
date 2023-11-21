import argparse
import os
import ctypes
import pathlib
import sys

import Simd

###################################################################################################

def main():
	parser = argparse.ArgumentParser(prog="Simd", description="Simd Python Wrapper.")
	parser.add_argument("-b", "--bin", help="Directory with binary files.", required=False, type=str, default=".")
	args = parser.parse_args()
	
	simd = Simd.Lib(args.bin)
	
	print("Simd version: {0}.".format(simd.Version()))
	print("CPU model: {0}.".format(simd.CpuDesc(Simd.CpuDesc.Model)))
	print(simd.SysInfo())
	
	print("\nSimd Python Wrapper test ended successfully!")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
