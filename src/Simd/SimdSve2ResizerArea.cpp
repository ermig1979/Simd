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

        template<UpdateType update> SIMD_INLINE void ResizerByteAreaUpdate(const svbool_t& mask, int32_t* dst, const svint32_t& value)
        {
            svst1_s32(mask, dst, value);
        }

        template<> SIMD_INLINE void ResizerByteAreaUpdate<UpdateAdd>(const svbool_t& mask, int32_t* dst, const svint32_t& value)
        {
            svst1_s32(mask, dst, svadd_s32_x(mask, svld1_s32(mask, dst), value));
        }

        SIMD_INLINE void ResizerByteAreaUnpackU8(const svuint8_t& src, svint32_t& d0, svint32_t& d1, svint32_t& d2, svint32_t& d3)
        {
            svuint16_t lo = svunpklo_u16(src);
            svuint16_t hi = svunpkhi_u16(src);
            d0 = svreinterpret_s32_u32(svunpklo_u32(lo));
            d1 = svreinterpret_s32_u32(svunpkhi_u32(lo));
            d2 = svreinterpret_s32_u32(svunpklo_u32(hi));
            d3 = svreinterpret_s32_u32(svunpkhi_u32(hi));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1Store(int32_t* dst, const svint32_t& v0, const svint32_t& v1,
            const svint32_t& v2, const svint32_t& v3, size_t count)
        {
            const size_t F = svcntw();
            ResizerByteAreaUpdate<update>(svwhilelt_b32((size_t)0, count), dst + 0 * F, v0);
            ResizerByteAreaUpdate<update>(svwhilelt_b32(F, count), dst + 1 * F, v1);
            ResizerByteAreaUpdate<update>(svwhilelt_b32(2 * F, count), dst + 2 * F, v2);
            ResizerByteAreaUpdate<update>(svwhilelt_b32(3 * F, count), dst + 3 * F, v3);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1Block(const uint8_t* src0, int32_t* dst, const svint32_t& a0,
            const svbool_t& mask8, const svbool_t& mask32, size_t count)
        {
            svint32_t d0, d1, d2, d3;
            ResizerByteAreaUnpackU8(svld1_u8(mask8, src0), d0, d1, d2, d3);
            ResizerByteArea1x1Store<update>(dst,
                svmul_s32_x(mask32, d0, a0),
                svmul_s32_x(mask32, d1, a0),
                svmul_s32_x(mask32, d2, a0),
                svmul_s32_x(mask32, d3, a0),
                count);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1Block(const uint8_t* src0, const uint8_t* src1, int32_t* dst,
            const svint32_t& a0, const svint32_t& a1, const svbool_t& mask8, const svbool_t& mask32, size_t count)
        {
            svint32_t s00, s01, s02, s03, s10, s11, s12, s13;
            ResizerByteAreaUnpackU8(svld1_u8(mask8, src0), s00, s01, s02, s03);
            ResizerByteAreaUnpackU8(svld1_u8(mask8, src1), s10, s11, s12, s13);
            ResizerByteArea1x1Store<update>(dst,
                svmla_s32_x(mask32, svmul_s32_x(mask32, s00, a0), s10, a1),
                svmla_s32_x(mask32, svmul_s32_x(mask32, s01, a0), s11, a1),
                svmla_s32_x(mask32, svmul_s32_x(mask32, s02, a0), s12, a1),
                svmla_s32_x(mask32, svmul_s32_x(mask32, s03, a0), s13, a1),
                count);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t* src0, size_t size, int32_t a0, int32_t* dst)
        {
            if (update == UpdateAdd && a0 == 0)
                return;
            const size_t A = svcntb();
            const svbool_t mask8 = svptrue_b8();
            const svbool_t mask32 = svptrue_b32();
            svint32_t _a0 = svdup_n_s32(a0);
            size_t i = 0;
            for (; i + 2 * A <= size; i += 2 * A, dst += 2 * A)
            {
                ResizerByteArea1x1Block<update>(src0 + i, dst, _a0, mask8, mask32, A);
                ResizerByteArea1x1Block<update>(src0 + i + A, dst + A, _a0, mask8, mask32, A);
            }
            for (; i + A <= size; i += A, dst += A)
                ResizerByteArea1x1Block<update>(src0 + i, dst, _a0, mask8, mask32, A);
            if (i < size)
                ResizerByteArea1x1Block<update>(src0 + i, dst, _a0, svwhilelt_b8((size_t)0, size - i), mask32, size - i);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t* src0, size_t stride, size_t size, int32_t a0, int32_t a1, int32_t* dst)
        {
            const size_t A = svcntb();
            const svbool_t mask8 = svptrue_b8();
            const svbool_t mask32 = svptrue_b32();
            svint32_t _a0 = svdup_n_s32(a0);
            svint32_t _a1 = svdup_n_s32(a1);
            const uint8_t* src1 = src0 + stride;
            size_t i = 0;
            for (; i + 2 * A <= size; i += 2 * A, dst += 2 * A)
            {
                ResizerByteArea1x1Block<update>(src0 + i, src1 + i, dst, _a0, _a1, mask8, mask32, A);
                ResizerByteArea1x1Block<update>(src0 + i + A, src1 + i + A, dst + A, _a0, _a1, mask8, mask32, A);
            }
            for (; i + A <= size; i += A, dst += A)
                ResizerByteArea1x1Block<update>(src0 + i, src1 + i, dst, _a0, _a1, mask8, mask32, A);
            if (i < size)
                ResizerByteArea1x1Block<update>(src0 + i, src1 + i, dst, _a0, _a1, svwhilelt_b8((size_t)0, size - i), mask32, size - i);
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
                    ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
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

        SIMD_INLINE bool InitResizerByteArea2x2Index(uint8_t index[4][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            const size_t A = svcntb();
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t i = 0; i < A; ++i)
            {
                index[0][i] = uint8_t((i & 0xFC) + ((i & 1) << 1) + ((i & 2) >> 1));
                size_t p = i & 7;
                index[1][i] = uint8_t((i & 0xF8) + ((p & 1) << 2) + (p >> 1));
            }
            const size_t srcStep = AlignLoAny(2 * A, size_t(6));
            for (size_t i = 0; i < 2 * A; ++i)
            {
                uint8_t value = 0;
                if (i < srcStep)
                    value = uint8_t((i / 6) * 6 + ((i % 6) % 2) * 3 + (i % 6) / 2);
                if (i < A)
                    index[2][i] = value;
                else
                    index[3][i - A] = value;
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t RESIZER_BYTE_AREA_2X2_INDEX[4][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool RESIZER_BYTE_AREA_2X2_INDEX_INITED = InitResizerByteArea2x2Index(RESIZER_BYTE_AREA_2X2_INDEX);

        SIMD_INLINE svuint8_t ResizerByteArea2x2ShuffleRc2()
        {
            return svld1_u8(svptrue_b8(), RESIZER_BYTE_AREA_2X2_INDEX[0]);
        }

        SIMD_INLINE svuint8_t ResizerByteArea2x2ShuffleRc4()
        {
            return svld1_u8(svptrue_b8(), RESIZER_BYTE_AREA_2X2_INDEX[1]);
        }

        SIMD_INLINE svuint16_t ResizerByteArea2x2PairSum(const svuint8_t& s0, const svuint8_t& s1)
        {
            const svbool_t mask = svptrue_b16();
            svuint16_t sum = svadalp_u16_x(mask, svdup_n_u16(0), s0);
            return svadalp_u16_x(mask, sum, s1);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea2x2StoreSum(const svuint16_t& sum, const svint32_t& val, int32_t* dst, size_t dstN)
        {
            const size_t F = svcntw();
            svbool_t maskLo = svwhilelt_b32((size_t)0, dstN);
            svbool_t maskHi = svwhilelt_b32(F, dstN);
            ResizerByteAreaUpdate<update>(maskLo, dst + 0, svmul_s32_x(maskLo, svreinterpret_s32_u32(svunpklo_u32(sum)), val));
            ResizerByteAreaUpdate<update>(maskHi, dst + F, svmul_s32_x(maskHi, svreinterpret_s32_u32(svunpkhi_u32(sum)), val));
        }

        template<size_t N> SIMD_INLINE svuint8_t ResizerByteArea2x2LoadColor(const uint8_t* src, const svuint8_t& index, const svbool_t& mask)
        {
            svuint8_t value = svld1_u8(mask, src);
            if (N == 1)
                return value;
            else
                return svtbl_u8(value, index);
        }

        template<size_t N, UpdateType update> SIMD_INLINE void ResizerByteArea2x2RowUpdateColor(const uint8_t* src0, const uint8_t* src1,
            size_t size2N, size_t dstSize, int32_t* dst, const svint32_t& val, const svuint8_t& index)
        {
            const size_t A = svcntb();
            const size_t HA = A / 2;
            const svbool_t mask8 = svptrue_b8();
            size_t i = 0, j = 0;
            for (; i + 2 * A <= size2N; i += 2 * A, j += A)
            {
                svuint8_t s00 = ResizerByteArea2x2LoadColor<N>(src0 + i, index, mask8);
                svuint8_t s10 = ResizerByteArea2x2LoadColor<N>(src1 + i, index, mask8);
                svuint8_t s01 = ResizerByteArea2x2LoadColor<N>(src0 + i + A, index, mask8);
                svuint8_t s11 = ResizerByteArea2x2LoadColor<N>(src1 + i + A, index, mask8);
                ResizerByteArea2x2StoreSum<update>(ResizerByteArea2x2PairSum(s00, s10), val, dst + j, HA);
                ResizerByteArea2x2StoreSum<update>(ResizerByteArea2x2PairSum(s01, s11), val, dst + j + HA, HA);
            }
            for (; i + A <= size2N; i += A, j += HA)
            {
                svuint8_t s0 = ResizerByteArea2x2LoadColor<N>(src0 + i, index, mask8);
                svuint8_t s1 = ResizerByteArea2x2LoadColor<N>(src1 + i, index, mask8);
                ResizerByteArea2x2StoreSum<update>(ResizerByteArea2x2PairSum(s0, s1), val, dst + j, HA);
            }
            if (i < size2N)
            {
                size_t srcN = size2N - i;
                svbool_t srcMask = svwhilelt_b8((size_t)0, srcN);
                svuint8_t s0 = ResizerByteArea2x2LoadColor<N>(src0 + i, index, srcMask);
                svuint8_t s1 = ResizerByteArea2x2LoadColor<N>(src1 + i, index, srcMask);
                ResizerByteArea2x2StoreSum<update>(ResizerByteArea2x2PairSum(s0, s1), val, dst + j, dstSize - j);
            }
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea2x2RowUpdateBgr(const uint8_t* src0, const uint8_t* src1,
            size_t size2N, size_t dstSize, int32_t* dst, const svint32_t& val)
        {
            const size_t A = svcntb();
            const size_t HA = A / 2;
            const size_t srcStep = AlignLoAny(2 * A, size_t(6));
            const size_t dstStep = srcStep / 2;
            const svbool_t mask8 = svptrue_b8();
            svuint8_t index0 = svld1_u8(mask8, RESIZER_BYTE_AREA_2X2_INDEX[2]);
            svuint8_t index1 = svld1_u8(mask8, RESIZER_BYTE_AREA_2X2_INDEX[3]);
            size_t i = 0, j = 0;
            for (; i + srcStep <= size2N; i += srcStep, j += dstStep)
            {
                svbool_t mask1 = svwhilelt_b8((size_t)0, srcStep - A);
                svuint8x2_t t0 = svcreate2_u8(svld1_u8(mask8, src0 + i), svld1_u8(mask1, src0 + i + A));
                svuint8x2_t t1 = svcreate2_u8(svld1_u8(mask8, src1 + i), svld1_u8(mask1, src1 + i + A));
                ResizerByteArea2x2StoreSum<update>(ResizerByteArea2x2PairSum(svtbl2_u8(t0, index0), svtbl2_u8(t1, index0)), val, dst + j, HA);
                ResizerByteArea2x2StoreSum<update>(ResizerByteArea2x2PairSum(svtbl2_u8(t0, index1), svtbl2_u8(t1, index1)), val, dst + j + HA, dstStep - HA);
            }
            if (i < size2N)
            {
                size_t srcN = size2N - i;
                size_t dstN = dstSize - j;
                svbool_t mask0 = svwhilelt_b8((size_t)0, srcN);
                svbool_t mask1 = svwhilelt_b8((size_t)0, srcN > A ? srcN - A : 0);
                svuint8x2_t t0 = svcreate2_u8(svld1_u8(mask0, src0 + i), svld1_u8(mask1, src0 + i + A));
                svuint8x2_t t1 = svcreate2_u8(svld1_u8(mask0, src1 + i), svld1_u8(mask1, src1 + i + A));
                size_t dst0 = dstN < HA ? dstN : HA;
                ResizerByteArea2x2StoreSum<update>(ResizerByteArea2x2PairSum(svtbl2_u8(t0, index0), svtbl2_u8(t1, index0)), val, dst + j, dst0);
                ResizerByteArea2x2StoreSum<update>(ResizerByteArea2x2PairSum(svtbl2_u8(t0, index1), svtbl2_u8(t1, index1)), val, dst + j + HA, dstN - dst0);
            }
        }

        template<size_t N, UpdateType update> SIMD_INLINE void ResizerByteArea2x2RowUpdate(const uint8_t* src0, const uint8_t* src1, size_t size, int32_t val, int32_t* dst)
        {
            if (update == UpdateAdd && val == 0)
                return;
            const size_t size2N = AlignLoAny(size, 2 * N);
            const size_t dstSize = size2N / 2;
            svint32_t _val = svdup_n_s32(val);
            if (N == 3)
                ResizerByteArea2x2RowUpdateBgr<update>(src0, src1, size2N, dstSize, dst, _val);
            else if (N == 4)
                ResizerByteArea2x2RowUpdateColor<N, update>(src0, src1, size2N, dstSize, dst, _val, ResizerByteArea2x2ShuffleRc4());
            else if (N == 2)
                ResizerByteArea2x2RowUpdateColor<N, update>(src0, src1, size2N, dstSize, dst, _val, ResizerByteArea2x2ShuffleRc2());
            else
                ResizerByteArea2x2RowUpdateColor<N, update>(src0, src1, size2N, dstSize, dst, _val, svindex_u8(0, 1));
            if (size2N < size)
                Base::ResizerByteArea2x2RowUpdate<N, 0, update>(src0 + size2N, src1 + size2N, val, dst + dstSize);
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
#endif
}
