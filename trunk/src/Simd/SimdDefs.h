/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include <memory.h>
#include <limits.h>

#if defined(_MSC_VER) && defined(_MSC_FULL_VER)

#define SIMD_INLINE __forceinline

#define SIMD_ALIGNED(x) __declspec(align(x))

#ifdef _M_X64
#define SIMD_X64_ENABLE
#endif

#if !defined(SIMD_SSE2_DEPRECATE) && _MSC_VER >= 1300
#define SIMD_SSE2_ENABLE
#endif

#if !defined(SIMD_SSSE3_DEPRECATE) && _MSC_VER >= 1500
#define SIMD_SSSE3_ENABLE
#endif

#if !defined(SIMD_SSE41_DEPRECATE) && _MSC_VER >= 1500
#define SIMD_SSE41_ENABLE
#endif

#if !defined(SIMD_SSE42_DEPRECATE) && _MSC_VER >= 1500
#define SIMD_SSE42_ENABLE
#endif

#if !defined(SIMD_AVX_DEPRECATE) && _MSC_FULL_VER >= 160040219
#define SIMD_AVX_ENABLE
#endif

#if !defined(SIMD_AVX2_DEPRECATE) && _MSC_VER >= 1700
#define SIMD_AVX2_ENABLE
#endif

#elif defined(__GNUC__)

#define SIMD_INLINE inline __attribute__ ((always_inline))

#define SIMD_ALIGNED(x) __attribute__ ((aligned(x)))

#ifdef __x86_64__
#define SIMD_X64_ENABLE
#endif

#if !defined(SIMD_SSE2_DEPRECATE) && defined(__SSE2__)
#define SIMD_SSE2_ENABLE
#endif

#if !defined(SIMD_SSSE3_DEPRECATE) && defined(__SSSE3__)
#define SIMD_SSSE3_ENABLE
#endif

#if !defined(SIMD_SSE41_DEPRECATE) && defined(__SSE4_1__)
#define SIMD_SSE41_ENABLE
#endif

#if !defined(SIMD_SSE42_DEPRECATE) && defined(__SSE4_2__)
#define SIMD_SSE42_ENABLE
#endif

#if !defined(SIMD_AVX_DEPRECATE) && defined(__AVX__)
#define SIMD_AVX_ENABLE
#endif

#if !defined(SIMD_AVX2_DEPRECATE) && defined(__AVX2__)
#define SIMD_AVX2_ENABLE
#endif

#else

#error This platform is unsupported!

#endif

#ifdef SIMD_SSE2_ENABLE
#include <emmintrin.h>
#endif

#ifdef SIMD_SSSE3_ENABLE
#include <tmmintrin.h>
#endif

#ifdef SIMD_SSE41_ENABLE
#include <smmintrin.h>
#endif

#ifdef SIMD_SSE42_ENABLE
#include <nmmintrin.h>
#endif

#if defined(SIMD_AVX_ENABLE) || defined(SIMD_AVX2_ENABLE)
#include <immintrin.h>
#endif

#if defined(SIMD_AVX_ENABLE) || defined(SIMD_AVX2_ENABLE)
#define SIMD_ALIGN 32
#elif defined(SIMD_SSE2_ENABLE) || defined(SIMD_SSSE3_ENABLE) || defined(SIMD_SSE41_ENABLE) || defined(SIMD_SSE42_ENABLE)
#define SIMD_ALIGN 16
#elif defined (SIMD_X64_ENABLE)
#define SIMD_ALIGN 8 
#else
#define SIMD_ALIGN 4 
#endif

#endif//__SimdDefs_h__
