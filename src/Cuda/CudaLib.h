/*
* Cuda Library.
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

/** \file CudaLib.h
* This file contains a Cuda Library API functions.
*/

#ifndef __CudaLib_h__
#define __CudaLib_h__

#include "Simd/SimdTypes.h"

#if defined(WIN32) && !defined(CUDA_STATIC)
#  ifdef CUDA_EXPORTS
#    define CUDA_API __declspec(dllexport)
#  else//CUDA_EXPORTS
#    define CUDA_API __declspec(dllimport)
#  endif//CUDA_EXPORTS
#else //WIN32
#    define CUDA_API
#endif//WIN32

#ifdef __cplusplus
extern "C"
{
#endif//__cplusplus

    /**
    * \fn const char * CudaVersion();
    *
    * \short Gets version of Cuda Library.
    *
    * \return string with version of Cuda Library (major version number, minor version number, release number, number of SVN's commits).
    */
    CUDA_API const char * CudaVersion();

    /**
    * \fn void CudaMedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);
    *
    * \short Performs median filtration of input image (filter window is a rhomb 3x3). 
    *
    * All images must have the same width, height and format (8-bit gray, 24-bit BGR or 32-bit BGRA). 
    *
    * \param [in] src - a pointer to pixels data of original input image.
    * \param [in] srcStride - a row size of src image.
    * \param [in] width - an image width.
    * \param [in] height - an image height.
    * \param [in] channelCount - a channel count.
    * \param [out] dst - a pointer to pixels data of filtered output image.
    * \param [in] dstStride - a row size of dst image.
    */
    CUDA_API void CudaMedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif//__CudaLib_h__
