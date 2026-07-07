/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        ResizerByteBicubic::ResizerByteBicubic(const ResParam& param)
            : Base::ResizerByteBicubic(param)
        {
        }

        template<int N, int F, int L> SIMD_INLINE int32_t CubicSumX(const uint8_t* src, const int32_t* ax)
        {
            return ax[0] * src[F * N] + ax[1] * src[0 * N] + ax[2] * src[1 * N] + ax[3] * src[L * N];
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

        SIMD_INLINE void BicubicRowInt(const int32_t* src0, const int32_t* src1, const int32_t* src2, const int32_t* src3, size_t size, const int32_t* ay, uint8_t* dst)
        {
            size_t i = 0, F = svcntw();
            svint32_t ay0 = svdup_n_s32(ay[0]);
            svint32_t ay1 = svdup_n_s32(ay[1]);
            svint32_t ay2 = svdup_n_s32(ay[2]);
            svint32_t ay3 = svdup_n_s32(ay[3]);
            svint32_t round = svdup_n_s32(Base::BICUBIC_ROUND);
            for (; i < size; i += F)
            {
                svbool_t mask = svwhilelt_b32(i, size);
                svint32_t sum = svmul_s32_x(mask, svld1_s32(mask, src0 + i), ay0);
                sum = svmla_s32_x(mask, sum, svld1_s32(mask, src1 + i), ay1);
                sum = svmla_s32_x(mask, sum, svld1_s32(mask, src2 + i), ay2);
                sum = svmla_s32_x(mask, sum, svld1_s32(mask, src3 + i), ay3);
                sum = svasr_n_s32_x(mask, svadd_s32_x(mask, sum, round), Base::BICUBIC_SHIFT);
                sum = svmin_n_s32_x(mask, svmax_n_s32_x(mask, sum, 0), 255);
                svst1b_u32(mask, dst + i, svreinterpret_u32_s32(sum));
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

        void ResizerByteBicubic::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            bool sparse = _param.dstH * 4.0 <= _param.srcH;
            Init(sparse);
            switch (_param.channels)
            {
            case 1: sparse ? Base::ResizerByteBicubic::RunS<1>(src, srcStride, dst, dstStride) : RunB<1>(src, srcStride, dst, dstStride); return;
            case 2: sparse ? Base::ResizerByteBicubic::RunS<2>(src, srcStride, dst, dstStride) : RunB<2>(src, srcStride, dst, dstStride); return;
            case 3: sparse ? Base::ResizerByteBicubic::RunS<3>(src, srcStride, dst, dstStride) : RunB<3>(src, srcStride, dst, dstStride); return;
            case 4: sparse ? Base::ResizerByteBicubic::RunS<4>(src, srcStride, dst, dstStride) : RunB<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }
    }
#endif
}

