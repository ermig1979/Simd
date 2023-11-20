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
	
	simd = Simd.Simd(args.bin)
	
	print("Simd version: {0}. \n".format(simd.Version()))
	
	print("\nSimd Python Wrapper ended successfully!")
	
	return 0
	
###################################################################################################
	
if __name__ == "__main__":
	main()
