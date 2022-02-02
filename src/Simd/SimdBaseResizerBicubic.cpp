/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
        ResizerByteBicubic::ResizerByteBicubic(const ResParam & param)
            : Resizer(param)
        {
        }
        
        void ResizerByteBicubic::EstimateIndexAlpha(size_t sizeS, size_t sizeD, size_t N, Array32i& index, Array32i& alpha)
        {
            index.Resize(sizeD);
            alpha.Resize(sizeD * 4);
            float scale = float(sizeS) / float(sizeD);
            for (size_t i = 0; i < sizeD; ++i)
            {
                float pos = (float)((i + 0.5f) * scale - 0.5f);
                int idx = (int)::floor(pos);
                float d = pos - idx;
                if (idx < 0)
                {
                    idx = 0;
                    d = 0.0f;
                }
                if (idx > (int)sizeS - 2)
                {
                    idx = (int)sizeS - 2;
                    d = 1.0f;
                }
                index[i] = idx * (int)N;
                alpha[i * 4 + 0] = Round(BICUBIC_RANGE * (2.0f - d) * (1.0f - d) * d / 6.0f);
                alpha[i * 4 + 1] = Round(BICUBIC_RANGE * (d - 2.0f) * (d + 1.0f) * (1.0f - d) / 2.0f);
                alpha[i * 4 + 2] = Round(BICUBIC_RANGE * (d - 2.0f) * (d + 1.0f) * d / 2.0f);
                alpha[i * 4 + 3] = Round(BICUBIC_RANGE * (1.0f + d) * (1.0f - d) * d / 6.0f);
            }
        } 

        void ResizerByteBicubic::Init(bool sparse)
        {
            if (_iy.data)
                return;
            EstimateIndexAlpha(_param.srcH, _param.dstH, 1, _iy, _ay);
            EstimateIndexAlpha(_param.srcW, _param.dstW, _param.channels, _ix, _ax);
            if (!sparse)
            {
                for (int i = 0; i < 4; ++i)
                    _bx[i].Resize(_param.dstW * _param.channels);
            }
            _sxl = (_param.srcW - 2) * _param.channels;
            for (_xn = 0; _ix[_xn] == 0; _xn++);
            for (_xt = _param.dstW; _ix[_xt - 1] == _sxl; _xt--);
        }

        template<int N, int F, int L> SIMD_INLINE int32_t CubicSumX(const uint8_t* src, const int32_t* ax)
        {
            return ax[0] * src[F * N] + ax[1] * src[0 * N] + ax[2] * src[1 * N] + ax[3] * src[L * N];
        }

        template<int N, int F, int L> SIMD_INLINE void BicubicInt(const uint8_t* src0, const uint8_t* src1,
            const uint8_t* src2, const uint8_t* src3, size_t sx, const int32_t* ax, const int32_t* ay, uint8_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
            {
                int32_t rs0 = CubicSumX<N, F, L>(src0 + sx + c, ax);
                int32_t rs1 = CubicSumX<N, F, L>(src1 + sx + c, ax);
                int32_t rs2 = CubicSumX<N, F, L>(src2 + sx + c, ax);
                int32_t rs3 = CubicSumX<N, F, L>(src3 + sx + c, ax);
                int32_t sum = ay[0] * rs0 + ay[1] * rs1 + ay[2] * rs2 + ay[3] * rs3;
                dst[c] = RestrictRange((sum + BICUBIC_ROUND) >> BICUBIC_SHIFT, 0, 255);
            }
        }

        template<int N> void ResizerByteBicubic::RunS(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t sy = _iy[dy];
                const uint8_t* src1 = src + sy * srcStride;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src0 = sy ? src1 - srcStride : src1;
                const uint8_t* src3 = sy < _param.srcH - 2 ? src2 + srcStride : src2;
                const int32_t* ay = _ay.data + dy * 4;
                size_t dx = 0;
                for (; dx < _xn; dx++)
                    BicubicInt<N, 0, 2>(src0, src1, src2, src3, _ix[dx], _ax.data + dx * 4, ay, dst + dx * N);
                for (; dx < _xt; dx++)
                    BicubicInt<N, -1, 2>(src0, src1, src2, src3, _ix[dx], _ax.data + dx * 4, ay, dst + dx * N);
                for (; dx < _param.dstW; dx++)
                    BicubicInt<N, -1, 1>(src0, src1, src2, src3, _ix[dx], _ax.data + dx * 4, ay, dst + dx * N);
            }
        }

        template<int N, int F, int L> SIMD_INLINE void PixelCubicSumX(const uint8_t* src, const int32_t* ax, int32_t* dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] = CubicSumX<N, F, L>(src + c, ax);
        }

        template<int N> SIMD_INLINE void RowCubicSumX(const uint8_t* src, size_t nose, size_t body, size_t tail, const int32_t* ix, const int32_t* ax, int32_t* dst)
        {
            size_t dx = 0;
            for (; dx < nose; dx++, ax += 4, dst += N)
                PixelCubicSumX<N, 0, 2>(src + ix[dx], ax, dst);
            for (; dx < body; dx++, ax += 4, dst += N)
                PixelCubicSumX<N, -1, 2>(src + ix[dx], ax, dst);
            for (; dx < tail; dx++, ax += 4, dst += N)
                PixelCubicSumX<N, -1, 1>(src + ix[dx], ax, dst);
        }

        SIMD_INLINE void BicubicRowInt(const int32_t* src0, const int32_t* src1, const int32_t* src2, const int32_t* src3, size_t n, const int32_t* ay, uint8_t* dst)
        {
            for (size_t i = 0; i < n; ++i)
            {
                int32_t sum = ay[0] * src0[i] + ay[1] * src1[i] + ay[2] * src2[i] + ay[3] * src3[i];
                dst[i] = RestrictRange((sum + BICUBIC_ROUND) >> BICUBIC_SHIFT, 0, 255);
            }
        }

        template<int N> void ResizerByteBicubic::RunB(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            int32_t prev = -1;
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                int32_t sy = _iy[dy], next = prev;
                for (int32_t curr = sy - 1, end = sy + 3; curr < end; ++curr)
                {
                    if (curr < prev)
                        continue;
                    const uint8_t* ps = src + RestrictRange(curr, 0, (int)_param.srcH - 1) * srcStride;
                    int32_t* pb = _bx[(curr + 1) & 3].data;
                    RowCubicSumX<N>(ps, _xn, _xt, _param.dstW, _ix.data, _ax.data, pb);
                    next++;
                }
                prev = next;

                const int32_t* ay = _ay.data + dy * 4;
                int32_t* pb0 = _bx[(sy + 0) & 3].data;
                int32_t* pb1 = _bx[(sy + 1) & 3].data;
                int32_t* pb2 = _bx[(sy + 2) & 3].data;
                int32_t* pb3 = _bx[(sy + 3) & 3].data;
                BicubicRowInt(pb0, pb1, pb2, pb3, _bx[0].size, ay, dst);
            }
        }

        void ResizerByteBicubic::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            bool sparse = _param.dstH * 4.0 <= _param.srcH;
            Init(sparse);
            switch (_param.channels)
            {
            case 1: sparse ? RunS<1>(src, srcStride, dst, dstStride) : RunB<1>(src, srcStride, dst, dstStride); return;
            case 2: sparse ? RunS<2>(src, srcStride, dst, dstStride) : RunB<2>(src, srcStride, dst, dstStride); return;
            case 3: sparse ? RunS<3>(src, srcStride, dst, dstStride) : RunB<3>(src, srcStride, dst, dstStride); return;
            case 4: sparse ? RunS<4>(src, srcStride, dst, dstStride) : RunB<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }
    }
}

