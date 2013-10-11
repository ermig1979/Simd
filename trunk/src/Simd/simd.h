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
#ifndef __SIMD_H__
#define __SIMD_H__

#include <stdlib.h>

#if defined(WIN32) && !defined(SIMD_STATIC)
#  ifdef SIMD_EXPORTS
#    define SIMD_API __declspec(dllexport)
#  else//SIMD_EXPORTS
#    define SIMD_API __declspec(dllimport)
#  endif//SIMD_EXPORTS
#else //WIN32
#    define SIMD_API
#endif//WIN32

#ifdef __cplusplus
extern "C" 
{
#endif//__cplusplus

    // Boolean values:
#define SIMD_TRUE   1
#define SIMD_FALSE  0

    SIMD_API const char * simd_version();

    SIMD_API int simd_sse2_enable();
    SIMD_API int simd_sse42_enable();
    SIMD_API int simd_avx_enable();
    SIMD_API int simd_avx2_enable();

    SIMD_API void simd_abs_difference_sum(const unsigned char *a, size_t a_stride, const unsigned char * b, size_t b_stride, 
        size_t width, size_t height, unsigned long long * sum);

    SIMD_API void simd_masked_abs_difference_sum(const unsigned char *a, size_t a_stride, const unsigned char *b, size_t b_stride, 
        const unsigned char *mask, size_t mask_stride, unsigned char index, size_t width, size_t height, unsigned long long * sum);

    SIMD_API void simd_abs_gradient_saturated_sum(const unsigned char * src, size_t src_stride, size_t width, size_t height, 
        unsigned char * dst, size_t dst_stride);

#ifdef __cplusplus 
}
#endif // __cplusplus

#endif//__SIMD_H__
