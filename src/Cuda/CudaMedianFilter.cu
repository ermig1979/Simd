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
//#include "Simd/SimdMath.h"

#include "Cuda/CudaMath.h"
#include "Cuda/CudaAlg.h"
#include <device_functions.h>

namespace Cuda
{
    template <class T> CUDA_INLINE __device__ T Max(T a, T b)
    {
        return a > b ? a : b;
    }

    template <class T> CUDA_INLINE __device__ T Min(T a, T b)
    {
        return a < b ? a : b;
    }

    template <class T> CUDA_INLINE __device__ void Sort(T & a, T & b)
    {
        if(a > b)
        {
            T t = a;
            a = b;
            b = t;
        }
    }

    template <class T> CUDA_INLINE __device__ void LoadRhomb3x3(const uint8_t * y[3], size_t x[3], size_t c, T a[5])
    {
        a[0] = y[0][x[1] + c];
        a[1] = y[1][x[0] + c]; a[2] = y[1][x[1] + c]; a[3] = y[1][x[2] + c];
        a[4] = y[2][x[1] + c]; 
    }

    template <class T> CUDA_INLINE __device__ void PartialSort5(T a[5])
    {
        Sort(a[2], a[3]); 
        Sort(a[1], a[2]);
        Sort(a[2], a[3]); 
        a[4] = Max(a[1], a[4]); 
        a[0] = Min(a[0], a[3]); 
        Sort(a[2], a[0]); 
        a[2] = Max(a[4], a[2]); 
        a[2] = Min(a[2], a[0]);
    }

    __global__ void MedianFilterRhomb3x3Kernel(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        size_t channelCount, uint8_t * dst, size_t dstStride)
    {
        uint8_t a[5];
        const uint8_t * y[3];
        size_t x[3];

        const int row = blockDim.x * blockIdx.x + threadIdx.x;
        const int col = blockDim.y * blockIdx.y + threadIdx.y;

        if(row >= height || col >= width)
            return;    

        y[0] = src + srcStride*Max(row - 1, 0);
        y[1] = src + srcStride*row;
        y[2] = src + srcStride*Min<int>(row + 1, height - 1);

        x[0] = channelCount*Max(col - 1, 0);
        x[1] = channelCount*col;
        x[2] = channelCount*Min<int>(col + 1, width - 1);

        #pragma unroll
        for(size_t channel = 0; channel < channelCount; channel++)
        {
            LoadRhomb3x3(y, x, channel, a);
            PartialSort5(a);
            dst[row*dstStride + col*channelCount + channel] = (uint8_t)a[2];
        }
    }

    void MedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        size_t channelCount, uint8_t * dst, size_t dstStride)
    {
        const int n = 1;
        const int m = 128;
        const ::dim3 grid((height + n - 1)/n, (width + m - 1)/m, 1);      
        const ::dim3 block(n, m, 1); 
       
        MedianFilterRhomb3x3Kernel<<<grid, block>>>(src, srcStride, width, height, channelCount, dst, dstStride);
        ::cudaDeviceSynchronize();
    }
}