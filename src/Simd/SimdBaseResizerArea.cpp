/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdResizer.h"
#include "Simd/SimdCopyPixel.h"

namespace Simd
{
    namespace Base
    {
        ResizerByteArea::ResizerByteArea(const ResParam & param)
            : Resizer(param)
        {
            _ay.Resize(_param.dstH + 1);
            _iy.Resize(_param.dstH + 1);
            EstimateParams(_param.srcH, _param.dstH, Base::AREA_RANGE, _ay.data, _iy.data);

            _ax.Resize(_param.dstW + 1);
            _ix.Resize(_param.dstW + 1);
            EstimateParams(_param.srcW, _param.dstW, Base::AREA_RANGE, _ax.data, _ix.data);
        }

        void ResizerByteArea::EstimateParams(size_t srcSize, size_t dstSize, size_t range, int32_t * alpha, int32_t * index)
        {
            float scale = (float)srcSize / dstSize;

            for (size_t ds = 0; ds <= dstSize; ++ds)
            {
                float a = (float)ds*scale;
                size_t i = (size_t)::floor(a);
                a -= i;
                if (i == srcSize)
                {
                    i--;
                    a = 1.0f;
                }
                alpha[ds] = int32_t(range * (1.0f - a) / scale);
                index[ds] = int32_t(i);
            }
        }

        template<class T, size_t N> SIMD_INLINE void ResizerByteAreaSet(const T * src, int32_t value, int32_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] = src[c] * value;
        }

        template<class T, size_t N> SIMD_INLINE void ResizerByteAreaAdd(const T * src, int32_t value, int32_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] += src[c] * value;
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaPixelRowSum(const uint8_t * src, size_t size, int32_t nose, int32_t body, int32_t tail, int32_t * dst)
        {
            ResizerByteAreaSet<uint8_t, N>(src, nose, dst);
            for (size_t i = 0; i < size; ++i)
            {
                src += N;
                ResizerByteAreaAdd<uint8_t, N>(src, body, dst);
            }
            ResizerByteAreaAdd<uint8_t, N>(src, tail, dst);
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaRes(const int32_t * src, uint8_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] = uint8_t((src[c] + Base::AREA_ROUND) >> Base::AREA_SHIFT);
        }

        template<size_t N> void ResizerByteArea::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            int32_t ts[N], rs[N];
            int32_t ayb = _ay.data[0], axb = _ax.data[0];
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t by = _iy.data[dy], ey = _iy.data[dy + 1];
                int32_t ayn = _ay.data[dy], ayt = - _ay.data[dy + 1];
                for (size_t dx = 0; dx < _param.dstW; dx++)
                {
                    size_t bx = _ix.data[dx], sx = _ix.data[dx + 1] - bx;
                    int32_t axn = _ax.data[dx], axt = - _ax.data[dx + 1];
                    const uint8_t * s = src + by * srcStride + bx * N;
                    ResizerByteAreaPixelRowSum<N>(s, sx, axn, axb, axt, rs);
                    ResizerByteAreaSet<int32_t, N>(rs, ayn, ts);
                    for (size_t sy = by; sy < ey; sy++)
                    {
                        s += srcStride;
                        ResizerByteAreaPixelRowSum<N>(s, sx, axn, axb, axt, rs);
                        ResizerByteAreaAdd<int32_t, N>(rs, ayb, ts);
                    }
                    ResizerByteAreaPixelRowSum<N>(s, sx, axn, axb, axt, rs);
                    ResizerByteAreaAdd<int32_t, N>(rs, ayt, ts);
                    ResizerByteAreaRes<N>(ts, dst + dx * N);
                }
            }
        }

        void ResizerByteArea::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            switch (_param.channels)
            {
            case 1: Run<1>(src, srcStride, dst, dstStride); return;
            case 2: Run<2>(src, srcStride, dst, dstStride); return;
            case 3: Run<3>(src, srcStride, dst, dstStride); return;
            case 4: Run<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }
    }
}

