/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy 
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
* copies of the Software, and to permit persons to whom the Software is 
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in 
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#ifndef __SimdDefs_h__
#define __SimdDefs_h__

#include "Simd/SimdConfig.h"

#include <cstddef>

#include <stdlib.h>
#include <assert.h>
#include <math.h>

#if defined(_MSC_VER) 

#if _MSC_VER >= 1600
#include <stdint.h>
#else
#define SIMD_OWN_STDINT
#endif

#define SIMD_INLINE __forceinline

#if !defined(SIMD_SSE2_DEPRECATE) && _MSC_VER >= 1300 
#define SIMD_SSE2_ENABLE
#endif

#if !defined(SIMD_SSE42_DEPRECATE) && _MSC_VER >= 1500 
#define SIMD_SSE42_ENABLE
#endif

#elif defined(__GNUC__)

#include <stdint.h>

#define SIMD_INLINE inline __attribute__ ((always_inline))

#if !defined(SIMD_SSE2_DEPRECATE) && defined(__SSE2__)
#define SIMD_SSE2_ENABLE
#endif

#if !defined(SIMD_SSE42_DEPRECATE) && defined(__SSE4_2__)
#define SIMD_SSE42_ENABLE
#endif

#else

#error This platform is unsupported!

#endif

#ifdef SIMD_SSE2_ENABLE
#include <emmintrin.h>
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSE42_ENABLE
#include <nmmintrin.h>
#endif// SIMD_SSE42_ENABLE

#endif//__SimdDefs_h__
