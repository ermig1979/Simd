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
#ifndef __CudaAllocator_h__
#define __CudaAllocator_h__

#include "Cuda/CudaDefs.h"

namespace Cuda
{
    struct DeviceAllocator
    {
        static void * Allocate(size_t size, size_t align);

        static void Free(void * p);
    };

    struct HostAllocator
    {
        static void * Allocate(size_t size, size_t align);

        static void Free(void * p);
    };

    //-------------------------------------------------------------------------

    // struct DeviceAllocator implementation:

    CUDA_INLINE void * DeviceAllocator::Allocate(size_t size, size_t align)
    {
        assert(size == Simd::AlignHi(size, align));
        void * p;
        ::cudaError_t error = ::cudaMalloc(&p, size);
        assert(error == ::cudaSuccess);
        return p;
    }

    CUDA_INLINE void DeviceAllocator::Free(void * p)
    {
        ::cudaError_t error = ::cudaFree(p);
        assert(error == ::cudaSuccess);
    }

    // struct HostAllocator implementation:

    CUDA_INLINE void * HostAllocator::Allocate(size_t size, size_t align)
    {
        assert(size == Simd::AlignHi(size, align));
        void * p;
        ::cudaError_t error = ::cudaMallocHost(&p, size);
        assert(error == ::cudaSuccess);
        return p;
    }

    CUDA_INLINE void HostAllocator::Free(void * p)
    {
        ::cudaError_t error = ::cudaFreeHost(p);
        assert(error == ::cudaSuccess);
    }
}

#endif//__CudaAllocator_h__
