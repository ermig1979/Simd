import sys

# The Simd wrapper directory (where Simd.py and libSimd.so live) is passed as argv[1]
# to avoid any shell/string-literal quoting of the package path.
wrapper_dir = sys.argv[1]
sys.path.insert(0, wrapper_dir)

import Simd

Simd.Lib.Init(wrapper_dir)
print("Simd from Python:", Simd.Lib.Version())
