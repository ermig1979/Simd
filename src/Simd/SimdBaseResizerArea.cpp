/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdResizerCommon.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
    namespace Base
    {
        ResizerByteArea::ResizerByteArea(const ResParam& param)
            : Resizer(param)
        {
            _ay.Resize(_param.dstH + 1);
            _iy.Resize(_param.dstH + 1);
            _ax.Resize(_param.dstW + 1);
            _ix.Resize(_param.dstW + 1);
        }

        void ResizerByteArea::EstimateParams(size_t srcSize, size_t dstSize, size_t range, int32_t* alpha, int32_t* index)
        {
            float scale = (float)srcSize / dstSize;

            for (size_t ds = 0; ds <= dstSize; ++ds)
            {
                float a = (float)ds * scale;
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

        //---------------------------------------------------------------------------------------------

        ResizerByteArea1x1::ResizerByteArea1x1(const ResParam & param)
            : ResizerByteArea(param)
        {
            EstimateParams(_param.srcH, _param.dstH, Base::AREA_RANGE, _ay.data, _iy.data);
            EstimateParams(_param.srcW, _param.dstW, Base::AREA_RANGE, _ax.data, _ix.data);
            _by.Resize(AlignHi(_param.srcW * _param.channels, _param.align), false, _param.align);
        }

        template<size_t N, UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t* src, int32_t val, int32_t* dst)
        {
            for (size_t c = 0; c < N; ++c)
                Update<update>(dst + c, src[c] * val);
        }

        template<size_t N, UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t* src, size_t size, int32_t val, int32_t* dst)
        {
            if (update == UpdateAdd && val == 0)
                return;
            for (size_t i = 0; i < size; i += N, dst += N)
                ResizerByteArea1x1RowUpdate<N, update>(src + i, val, dst);
        }

        template<size_t N> SIMD_INLINE void ResizerByteArea1x1RowSum(const uint8_t* src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, int32_t* dst)
        {
            if (count)
            {
                size_t c = 0;
                ResizerByteArea1x1RowUpdate<N, UpdateSet>(src, size, curr, dst), src += stride, c += 1;
                for (; c < count; c += 1, src += stride)
                    ResizerByteArea1x1RowUpdate<N, UpdateAdd>(src, size, zero, dst);
                ResizerByteArea1x1RowUpdate<N, UpdateAdd>(src, size, zero - next, dst);
            }
            else
                ResizerByteArea1x1RowUpdate<N, UpdateSet>(src, size, curr - next, dst);
        }

        template<size_t N> void ResizerByteArea1x1::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t dstW = _param.dstW, rowSize = _param.srcW * N, rowRest = dstStride - dstW * N;
            const int32_t* iy = _iy.data, * ix = _ix.data, * ay = _ay.data, * ax = _ax.data;
            int32_t ay0 = ay[0], ax0 = ax[0];
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += rowRest)
            {
                int32_t* buf = _by.data;
                size_t yn = iy[dy + 1] - iy[dy];
                ResizerByteArea1x1RowSum<N>(src, srcStride, yn, rowSize, ay[dy], ay0, ay[dy + 1], buf), src += yn * srcStride;
                for (size_t dx = 0; dx < dstW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
                }
            }
        }

        void ResizerByteArea1x1::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
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

        //---------------------------------------------------------------------------------------------

        ResizerByteArea2x2::ResizerByteArea2x2(const ResParam& param)
            : ResizerByteArea(param)
        {
            EstimateParams(DivHi(_param.srcH, 2), _param.dstH, Base::AREA_RANGE / 2, _ay.data, _iy.data);
            EstimateParams(DivHi(_param.srcW, 2), _param.dstW, Base::AREA_RANGE / 2, _ax.data, _ix.data);
            _by.Resize(AlignHi(DivHi(_param.srcW, 2) * _param.channels, _param.align) + _param.align, false, _param.align);
        }

        template<size_t N, UpdateType update> SIMD_INLINE void ResizerByteArea2x2RowUpdate(const uint8_t* src0, const uint8_t* src1, size_t size, int32_t val, int32_t* dst)
        {
            if (update == UpdateAdd && val == 0)
                return;
            size_t size2N = AlignLoAny(size, 2 * N);
            size_t i = 0;
            for (; i < size2N; i += 2 * N, dst += N)
                ResizerByteArea2x2RowUpdate<N, N, update>(src0 + i, src1 + i, val, dst);
            if(i < size)
                ResizerByteArea2x2RowUpdate<N, 0, update>(src0 + i, src1 + i, val, dst);
        }

        template<size_t N> SIMD_INLINE void ResizerByteArea2x2RowSum(const uint8_t* src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, bool tail, int32_t* dst)
        {
            size_t c = 0;
            if (count)
            {
                ResizerByteArea2x2RowUpdate<N, UpdateSet>(src, src + stride, size, curr, dst), src += 2 * stride, c += 2;
                for (; c < count; c += 2, src += 2 * stride)
                    ResizerByteArea2x2RowUpdate<N, UpdateAdd>(src, src + stride, size, zero, dst);
                ResizerByteArea2x2RowUpdate<N, UpdateAdd>(src, tail ? src : src + stride, size, zero - next, dst);
            }
            else
                ResizerByteArea2x2RowUpdate<N, UpdateSet>(src, tail ? src : src + stride, size, curr - next, dst);
        }

        template<size_t N> void ResizerByteArea2x2::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t dstW = _param.dstW, rowSize = _param.srcW * N, rowRest = dstStride - dstW * N;
            const int32_t* iy = _iy.data, * ix = _ix.data, * ay = _ay.data, * ax = _ax.data;
            int32_t ay0 = ay[0], ax0 = ax[0];
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += rowRest)
            {
                int32_t* buf = _by.data;
                size_t yn = (iy[dy + 1] - iy[dy]) * 2;
                bool tail = (dy == _param.dstH - 1) && (_param.srcH & 1);
                ResizerByteArea2x2RowSum<N>(src, srcStride, yn, rowSize, ay[dy], ay0, ay[dy + 1], tail, buf), src += yn * srcStride;
                for (size_t dx = 0; dx < dstW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
                }
            }
        }

        void ResizerByteArea2x2::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
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

