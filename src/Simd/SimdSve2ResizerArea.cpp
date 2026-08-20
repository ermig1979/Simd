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
#include "Simd/SimdResizerCommon.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        ResizerByteArea1x1::ResizerByteArea1x1(const ResParam& param)
            : Base::ResizerByteArea1x1(param)
        {
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1Update(const svbool_t& mask, int32_t* dst, const svint32_t& value)
        {
            svst1_s32(mask, dst, value);
        }

        template<> SIMD_INLINE void ResizerByteArea1x1Update<UpdateAdd>(const svbool_t& mask, int32_t* dst, const svint32_t& value)
        {
            svst1_s32(mask, dst, svadd_s32_x(mask, svld1_s32(mask, dst), value));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t* src0, size_t size, int32_t a0, int32_t* dst)
        {
            svint32_t _a0 = svdup_n_s32(a0);
            size_t F = svcntw();
            for (size_t i = 0; i < size; i += F)
            {
                svbool_t mask = svwhilelt_b32(i, size);
                svint32_t sum = svmul_s32_x(mask, svld1ub_s32(mask, src0 + i), _a0);
                ResizerByteArea1x1Update<update>(mask, dst + i, sum);
            }
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t* src0, size_t stride, size_t size, int32_t a0, int32_t a1, int32_t* dst)
        {
            svint32_t _a0 = svdup_n_s32(a0);
            svint32_t _a1 = svdup_n_s32(a1);
            const uint8_t* src1 = src0 + stride;
            size_t F = svcntw();
            for (size_t i = 0; i < size; i += F)
            {
                svbool_t mask = svwhilelt_b32(i, size);
                svint32_t sum = svmul_s32_x(mask, svld1ub_s32(mask, src0 + i), _a0);
                sum = svmla_s32_x(mask, sum, svld1ub_s32(mask, src1 + i), _a1);
                ResizerByteArea1x1Update<update>(mask, dst + i, sum);
            }
        }

        SIMD_INLINE void ResizerByteArea1x1RowSum(const uint8_t* src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, int32_t* dst)
        {
            if (count)
            {
                size_t i = 0;
                ResizerByteArea1x1RowUpdate<UpdateSet>(src, stride, size, curr, count == 1 ? zero - next : zero, dst), src += 2 * stride, i += 2;
                for (; i < count; i += 2, src += 2 * stride)
                    ResizerByteArea1x1RowUpdate<UpdateAdd>(src, stride, size, zero, i == count - 1 ? zero - next : zero, dst);
                if (i == count)
                    ResizerByteArea1x1RowUpdate<UpdateAdd>(src, size, zero - next, dst);
            }
            else
                ResizerByteArea1x1RowUpdate<UpdateSet>(src, size, curr - next, dst);
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
                ResizerByteArea1x1RowSum(src, srcStride, yn, rowSize, ay[dy], ay0, ay[dy + 1], buf), src += yn * srcStride;
                for (size_t dx = 0; dx < dstW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    Base::ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
                }
            }
        }

        void ResizerByteArea1x1::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
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
            : Base::ResizerByteArea2x2(param)
        {
        }

        SIMD_INLINE svuint8_t ResizerByteArea2x2ShuffleRc2()
        {
            const svbool_t all = svptrue_b8();
            svuint8_t i = svindex_u8(0, 1);
            svuint8_t base = svand_n_u8_x(all, i, 0xFC);
            svuint8_t even = svlsl_n_u8_x(all, svand_n_u8_x(all, i, 1), 1);
            svuint8_t odd = svlsr_n_u8_x(all, svand_n_u8_x(all, i, 2), 1);
            return svadd_u8_x(all, base, svadd_u8_x(all, even, odd));
        }

        SIMD_INLINE svuint8_t ResizerByteArea2x2ShuffleRc4()
        {
            const svbool_t all = svptrue_b8();
            svuint8_t i = svindex_u8(0, 1);
            svuint8_t base = svand_n_u8_x(all, i, 0xF8);
            svuint8_t p = svand_n_u8_x(all, i, 7);
            svuint8_t lo = svlsl_n_u8_x(all, svand_n_u8_x(all, p, 1), 2);
            svuint8_t hi = svlsr_n_u8_x(all, p, 1);
            return svadd_u8_x(all, base, svadd_u8_x(all, lo, hi));
        }

        SIMD_INLINE svuint8_t ResizerByteArea2x2ShuffleBgr()
        {
            uint8_t index[SIMD_SVE2_VECTOR_SIZE_MAX];
            const size_t A = svcntb();
            const size_t srcStep = AlignLoAny(A, size_t(6));
            for (size_t i = 0; i < A; ++i)
            {
                if (i < srcStep)
                    index[i] = uint8_t((i / 6) * 6 + ((i % 6) % 2) * 3 + (i % 6) / 2);
                else
                    index[i] = 0;
            }
            return svld1_u8(svptrue_b8(), index);
        }

        template<size_t N> SIMD_INLINE svuint8_t ResizerByteArea2x2Shuffle()
        {
            if (N == 2)
                return ResizerByteArea2x2ShuffleRc2();
            else if (N == 3)
                return ResizerByteArea2x2ShuffleBgr();
            else if (N == 4)
                return ResizerByteArea2x2ShuffleRc4();
            else
                return svindex_u8(0, 1);
        }

        template<size_t N> SIMD_INLINE svuint8_t ResizerByteArea2x2ShuffleSrc(const svuint8_t& src, const svuint8_t& index)
        {
            if (N == 1)
                return src;
            else
                return svtbl_u8(src, index);
        }

        template<size_t N> SIMD_INLINE size_t ResizerByteArea2x2SrcStep()
        {
            const size_t A = svcntb();
            return N == 3 ? AlignLoAny(A, size_t(6)) : A;
        }

        SIMD_INLINE svuint16_t ResizerByteArea2x2PairSum(const svuint8_t& s0, const svuint8_t& s1)
        {
            const svbool_t mask = svptrue_b16();
            svuint16_t sum0 = svaddlb_u16(s0, svext_u8(s0, s0, 1));
            svuint16_t sum1 = svaddlb_u16(s1, svext_u8(s1, s1, 1));
            return svadd_u16_x(mask, sum0, sum1);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea2x2Update(const svbool_t& mask, int32_t* dst, const svint32_t& value)
        {
            svst1_s32(mask, dst, value);
        }

        template<> SIMD_INLINE void ResizerByteArea2x2Update<UpdateAdd>(const svbool_t& mask, int32_t* dst, const svint32_t& value)
        {
            svst1_s32(mask, dst, svadd_s32_x(mask, svld1_s32(mask, dst), value));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea2x2StoreSum(const svuint16_t& sum, const svint32_t& val, int32_t* dst, size_t dstN)
        {
            const size_t F = svcntw();
            svbool_t maskLo = svwhilelt_b32((size_t)0, dstN);
            svbool_t maskHi = svwhilelt_b32(F, dstN);
            ResizerByteArea2x2Update<update>(maskLo, dst + 0, svmul_s32_x(maskLo, svreinterpret_s32_u32(svunpklo_u32(sum)), val));
            ResizerByteArea2x2Update<update>(maskHi, dst + F, svmul_s32_x(maskHi, svreinterpret_s32_u32(svunpkhi_u32(sum)), val));
        }

        template<size_t N, UpdateType update> SIMD_INLINE void ResizerByteArea2x2RowUpdate(const uint8_t* src0, const uint8_t* src1, size_t size, int32_t val, int32_t* dst, const svuint8_t& index)
        {
            if (update == UpdateAdd && val == 0)
                return;
            const size_t size2N = AlignLoAny(size, 2 * N);
            const size_t dstSize = size2N / 2;
            const size_t srcStep = ResizerByteArea2x2SrcStep<N>();
            const size_t dstStep = srcStep / 2;
            svint32_t _val = svdup_n_s32(val);
            size_t i = 0, j = 0;
            for (; j < dstSize; i += srcStep, j += dstStep)
            {
                size_t srcN = size2N - i < srcStep ? size2N - i : srcStep;
                size_t dstN = dstSize - j < dstStep ? dstSize - j : dstStep;
                svbool_t srcMask = svwhilelt_b8((size_t)0, srcN);
                svuint8_t s0 = ResizerByteArea2x2ShuffleSrc<N>(svld1_u8(srcMask, src0 + i), index);
                svuint8_t s1 = ResizerByteArea2x2ShuffleSrc<N>(svld1_u8(srcMask, src1 + i), index);
                ResizerByteArea2x2StoreSum<update>(ResizerByteArea2x2PairSum(s0, s1), _val, dst + j, dstN);
            }
            if (size2N < size)
                Base::ResizerByteArea2x2RowUpdate<N, 0, update>(src0 + size2N, src1 + size2N, val, dst + dstSize);
        }

        template<size_t N> SIMD_INLINE void ResizerByteArea2x2RowSum(const uint8_t* src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, bool tail, int32_t* dst)
        {
            svuint8_t index = ResizerByteArea2x2Shuffle<N>();
            size_t c = 0;
            if (count)
            {
                ResizerByteArea2x2RowUpdate<N, UpdateSet>(src, src + stride, size, curr, dst, index), src += 2 * stride, c += 2;
                for (; c < count; c += 2, src += 2 * stride)
                    ResizerByteArea2x2RowUpdate<N, UpdateAdd>(src, src + stride, size, zero, dst, index);
                ResizerByteArea2x2RowUpdate<N, UpdateAdd>(src, tail ? src : src + stride, size, zero - next, dst, index);
            }
            else
                ResizerByteArea2x2RowUpdate<N, UpdateSet>(src, tail ? src : src + stride, size, curr - next, dst, index);
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaResult(const int32_t* src, size_t count, int32_t curr, int32_t zero, int32_t next, uint8_t* dst)
        {
            svbool_t mask = svwhilelt_b32((size_t)0, N);
            svint32_t _zero = svdup_n_s32(zero);
            svint32_t sum = svmul_s32_x(mask, svld1_s32(mask, src), svdup_n_s32(curr));
            for (size_t i = 0; i < count; ++i)
            {
                src += N;
                sum = svmla_s32_x(mask, sum, svld1_s32(mask, src), _zero);
            }
            sum = svmla_s32_x(mask, sum, svld1_s32(mask, src), svdup_n_s32(-next));
            sum = svasr_n_s32_x(mask, svadd_n_s32_x(mask, sum, Base::AREA_ROUND), Base::AREA_SHIFT);
            svst1b_u32(mask, dst, svreinterpret_u32_s32(sum));
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
#endif
}

