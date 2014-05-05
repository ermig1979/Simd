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
#ifndef __CudaUtils_h__
#define __CudaUtils_h__

#include "Simd/SimdView.h"

#include "Cuda/CudaAllocator.h"
#include "Cuda/CudaLib.h"

namespace Cuda
{
    //typedef Simd::View<Simd::Allocator> View;
    typedef Simd::View<Cuda::DeviceAllocator> DView;
    typedef Simd::View<Cuda::HostAllocator> HView;

    //-----------------------------------------------------------------------

    CUDA_INLINE void Copy(const HView & src, DView & dst)
    {
        assert(Simd::Compatible(src, dst));

        size_t rowSize = src.width*src.PixelSize();
        if(Simd::AlignHi(rowSize, SIMD_ALIGN) == src.stride && src.stride == dst.stride)
        {
            ::cudaError_t error = ::cudaMemcpy(dst.data, src.data, src.DataSize(), ::cudaMemcpyHostToDevice);
            assert(error == ::cudaSuccess);
        }
        else
        {
            for(size_t row = 0; row < src.height; ++row)
            {
                ::cudaError_t error = ::cudaMemcpy(dst.data + dst.stride*row, 
                    src.data + src.stride*row, rowSize, ::cudaMemcpyHostToDevice);
                assert(error == ::cudaSuccess);
            }
        }
    }

    CUDA_INLINE void Copy(const DView & src, HView & dst)
    {
        assert(Simd::Compatible(src, dst));

        size_t rowSize = src.width*src.PixelSize();
        if(Simd::AlignHi(rowSize, SIMD_ALIGN) == src.stride && src.stride == dst.stride)
        {
            ::cudaError_t error = ::cudaMemcpy(dst.data, src.data, src.DataSize(), ::cudaMemcpyDeviceToHost);
            assert(error == ::cudaSuccess);
        }
        else
        {
            for(size_t row = 0; row < src.height; ++row)
            {
                ::cudaError_t error = ::cudaMemcpy(dst.data + dst.stride*row, 
                    src.data + src.stride*row, rowSize, ::cudaMemcpyDeviceToHost);
                assert(error == ::cudaSuccess);
            }
        }
    }
}

#endif//__CudaUtils_h__