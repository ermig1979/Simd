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
#include "Simd/SimdResizerCommon.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE 
    namespace Avx512bw
    {
        ResizerByteArea1x1::ResizerByteArea1x1(const ResParam & param)
            : Avx2::ResizerByteArea1x1(param)
        {
        }

        template<UpdateType update, bool mask> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t * src0, __m512i alpha, int32_t * dst, __mmask64 tail = -1)
        {
            __m512i s0 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src0, tail)));
            __m512i i0 = UnpackU8<0>(s0);
            __m512i i1 = UnpackU8<1>(s0);
            Update<update, true>(dst + 0 * F, _mm512_madd_epi16(alpha, UnpackU8<0>(i0)));
            Update<update, true>(dst + 1 * F, _mm512_madd_epi16(alpha, UnpackU8<1>(i0)));
            Update<update, true>(dst + 2 * F, _mm512_madd_epi16(alpha, UnpackU8<0>(i1)));
            Update<update, true>(dst + 3 * F, _mm512_madd_epi16(alpha, UnpackU8<1>(i1)));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t * src0, size_t size, size_t aligned, int32_t a, int32_t * dst, __mmask64 tail)
        {
            __m512i alpha = SetInt16(a, a);
            size_t i = 0;
            for (; i < aligned; i += A, dst += A, src0 += A)
                ResizerByteArea1x1RowUpdate<update, false>(src0, alpha, dst);
            if(i < size)
                ResizerByteArea1x1RowUpdate<update, true>(src0, alpha, dst, tail);
        }

        template<UpdateType update, bool mask> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t * src0, const uint8_t * src1, __m512i alpha, int32_t * dst, __mmask64 tail = -1)
        {
            __m512i s0 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src0, tail)));
            __m512i s1 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src1, tail)));
            __m512i i0 = UnpackU8<0>(s0, s1);
            __m512i i1 = UnpackU8<1>(s0, s1);
            Update<update, true>(dst + 0 * F, _mm512_madd_epi16(alpha, UnpackU8<0>(i0)));
            Update<update, true>(dst + 1 * F, _mm512_madd_epi16(alpha, UnpackU8<1>(i0)));
            Update<update, true>(dst + 2 * F, _mm512_madd_epi16(alpha, UnpackU8<0>(i1)));
            Update<update, true>(dst + 3 * F, _mm512_madd_epi16(alpha, UnpackU8<1>(i1)));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t * src0, size_t stride, size_t size, size_t aligned, int32_t a0, int32_t a1, int32_t * dst, __mmask64 tail = -1)
        {
            __m512i alpha = SetInt16(a0, a1);
            const uint8_t * src1 = src0 + stride;
            size_t i = 0;
            for (; i < aligned; i += A, dst += A)
                ResizerByteArea1x1RowUpdate<update, false>(src0 + i, src1 + i, alpha, dst);
            if (i < size)
                ResizerByteArea1x1RowUpdate<update, true>(src0 + i, src1 + i, alpha, dst, tail);
        }

        template<UpdateType update, bool mask> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t * src0, const uint8_t * src1,
            const uint8_t * src2, const uint8_t * src3, __m512i a01, __m512i a23, int32_t * dst, __mmask64 tail = -1)
        {
            __m512i s0 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src0, tail)));
            __m512i s1 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src1, tail)));
            __m512i t010 = _mm512_maddubs_epi16(UnpackU8<0>(s0, s1), a01);
            __m512i t011 = _mm512_maddubs_epi16(UnpackU8<1>(s0, s1), a01);
            __m512i s2 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src2, tail)));
            __m512i s3 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<false, mask>(src3, tail)));
            __m512i t230 = _mm512_maddubs_epi16(UnpackU8<0>(s2, s3), a23);
            __m512i t231 = _mm512_maddubs_epi16(UnpackU8<1>(s2, s3), a23);
            Update<update, true>(dst + 0 * F, _mm512_madd_epi16(K16_0001, UnpackU16<0>(t010, t230)));
            Update<update, true>(dst + 1 * F, _mm512_madd_epi16(K16_0001, UnpackU16<1>(t010, t230)));
            Update<update, true>(dst + 2 * F, _mm512_madd_epi16(K16_0001, UnpackU16<0>(t011, t231)));
            Update<update, true>(dst + 3 * F, _mm512_madd_epi16(K16_0001, UnpackU16<1>(t011, t231)));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t * src0, size_t stride, size_t size, size_t aligned, int32_t a0, int32_t a12, int32_t a3, int32_t * dst, __mmask64 tail = -1)
        {
            __m512i a01 = SetInt8(a0, a12);
            __m512i a23 = SetInt8(a12, a3);
            const uint8_t * src1 = src0 + stride;
            const uint8_t * src2 = src1 + stride;
            const uint8_t * src3 = src2 + stride;
            size_t i = 0;
            for (; i < aligned; i += A, dst += A)
                ResizerByteArea1x1RowUpdate<update, false>(src0 + i, src1 + i, src2 + i, src3 + i, a01, a23, dst);
            if (i < size)
                ResizerByteArea1x1RowUpdate<update, true>(src0 + i, src1 + i, src2 + i, src3 + i, a01, a23, dst, tail);
        }

        SIMD_INLINE void ResizerByteArea1x1RowSum(const uint8_t * src, size_t stride, size_t count, size_t size, size_t aligned, int32_t curr, int32_t zero, int32_t next, int32_t * dst, __mmask64 tail)
        {
            if (count)
            {
                size_t i = 0;
                ResizerByteArea1x1RowUpdate<UpdateSet>(src, stride, size, aligned, curr, count == 1 ? zero - next : zero, dst, tail), src += 2 * stride, i += 2;
                for (; i < count; i += 2, src += 2 * stride)
                    ResizerByteArea1x1RowUpdate<UpdateAdd>(src, stride, size, aligned, zero, i == count - 1 ? zero - next : zero, dst, tail);
                if (i == count)
                    ResizerByteArea1x1RowUpdate<UpdateAdd>(src, size, aligned, zero - next, dst, tail);
            }
            else
                ResizerByteArea1x1RowUpdate<UpdateSet>(src, size, aligned, curr - next, dst, tail);
        }

        template<size_t N> void ResizerByteArea1x1::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t bodyW = _param.dstW - (N == 3 ? 1 : 0), rowSize = _param.srcW * N, rowRest = dstStride - _param.dstW * N;
            const int32_t * iy = _iy.data, *ix = _ix.data, *ay = _ay.data, *ax = _ax.data;
            int32_t ay0 = ay[0], ax0 = ax[0];
            size_t rowSizeA = AlignLo(rowSize, A);
            __mmask64 tail = TailMask64(rowSize - rowSizeA);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += rowRest)
            {
                int32_t * buf = _by.data;
                size_t yn = iy[dy + 1] - iy[dy];
                ResizerByteArea1x1RowSum(src, srcStride, yn, rowSize, rowSizeA, ay[dy], ay0, ay[dy + 1], buf, tail), src += yn * srcStride;
                size_t dx = 0;
                for (; dx < bodyW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    Sse41::ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
                }
                for (; dx < _param.dstW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    Base::ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
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
            : Avx2::ResizerByteArea2x2(param)
        {
        }

        template<size_t N> SIMD_INLINE __m512i ShuffleColor(__m512i val)
        {
            return val;
        }

        template<> SIMD_INLINE __m512i ShuffleColor<2>(__m512i val)
        {
            static const __m512i IDX = SIMD_MM512_SETR_EPI8(
                0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
                0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
                0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
                0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);
            return _mm512_shuffle_epi8(val, IDX);
        }

        template<> SIMD_INLINE __m512i ShuffleColor<4>(__m512i val)
        {
            static const __m512i IDX = SIMD_MM512_SETR_EPI8(
                0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
                0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
                0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
                0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);
            return _mm512_shuffle_epi8(val, IDX);
        }

        template<size_t N, UpdateType update> SIMD_INLINE void ResizerByteArea2x2RowUpdateColor(const uint8_t* src0, const uint8_t* src1, size_t size, int32_t val, int32_t* dst)
        {
            if (update == UpdateAdd && val == 0)
                return;
            size_t size2N = AlignLoAny(size, 2 * N);
            size_t i = 0;
            size_t size4F = AlignLoAny(size, 4 * F);
            __m512i _val = _mm512_set1_epi16(val);
            for (; i < size4F; i += 4 * F, dst += 2 * F)
            {
                __m512i s0 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, Load<false>(src0 + i));
                __m512i s1 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, Load<false>(src1 + i));
                s0 = _mm512_maddubs_epi16(ShuffleColor<N>(s0), K8_01);
                s1 = _mm512_maddubs_epi16(ShuffleColor<N>(s1), K8_01);
                Update<update, false>(dst + 0, _mm512_madd_epi16(_mm512_unpacklo_epi16(s0, s1), _val));
                Update<update, false>(dst + F, _mm512_madd_epi16(_mm512_unpackhi_epi16(s0, s1), _val));
            }
            if (size4F < size2N)
            {
                ptrdiff_t srcTail = size2N - size4F, dstTail = srcTail / 2;
                __mmask64 srcMask = TailMask64(srcTail);
                __m512i s0 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<false, true>(src0 + i, srcMask)));
                __m512i s1 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<false, true>(src1 + i, srcMask)));
                s0 = _mm512_maddubs_epi16(ShuffleColor<N>(s0), K8_01);
                s1 = _mm512_maddubs_epi16(ShuffleColor<N>(s1), K8_01);
                Update<update, false, true>(dst + 0, _mm512_madd_epi16(_mm512_unpacklo_epi16(s0, s1), _val), TailMask16(dstTail - 0));
                Update<update, false, true>(dst + F, _mm512_madd_epi16(_mm512_unpackhi_epi16(s0, s1), _val), TailMask16(dstTail - F));
                i += srcTail;
                dst += dstTail;
            }
            if (i < size)
                Base::ResizerByteArea2x2RowUpdate<N, 0, update>(src0 + i, src1 + i, val, dst);
        }

        template<size_t N> SIMD_INLINE void ResizerByteArea2x2RowSum(const uint8_t* src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, bool tail, int32_t* dst)
        {
            size_t c = 0;
            if (count)
            {
                ResizerByteArea2x2RowUpdateColor<N, UpdateSet>(src, src + stride, size, curr, dst), src += 2 * stride, c += 2;
                for (; c < count; c += 2, src += 2 * stride)
                    ResizerByteArea2x2RowUpdateColor<N, UpdateAdd>(src, src + stride, size, zero, dst);
                ResizerByteArea2x2RowUpdateColor<N, UpdateAdd>(src, tail ? src : src + stride, size, zero - next, dst);
            }
            else
                ResizerByteArea2x2RowUpdateColor<N, UpdateSet>(src, tail ? src : src + stride, size, curr - next, dst);
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea2x2RowUpdateBgr(const uint8_t* src0, const uint8_t* src1, size_t size, int32_t val, int32_t* dst)
        {
            if (update == UpdateAdd && val == 0)
                return;
            size_t size6 = AlignLoAny(size, 6);
            size_t i = 0;
            static const __m256i K32_PRM0 = SIMD_MM256_SETR_EPI32(0x0, 0x1, 0x2, 0x0, 0x3, 0x4, 0x5, 0x0);
            static const __m256i K32_PRM1 = SIMD_MM256_SETR_EPI32(0x2, 0x3, 0x4, 0x0, 0x5, 0x6, 0x7, 0x0);
            static const __m256i K8_SHFL = SIMD_MM256_SETR_EPI8(
                0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, -1, -1, -1, -1);
            static const __m256i K32_PRM2 = SIMD_MM256_SETR_EPI32(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x3, 0x7);
            static const __m256i K32_PRM3 = SIMD_MM256_SETR_EPI32(0x2, 0x4, 0x5, 0x6, 0x3, 0x7, 0x0, 0x1);
            __m256i _val = _mm256_set1_epi32(val);
            size_t size48 = AlignLoAny(size, 48);
            for (; i < size48; i += 48, dst += 24)
            {
                __m256i s00 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(Avx2::Load<false>((__m256i*)(src0 + i + 0 * Avx2::F)), K32_PRM0), K8_SHFL);
                __m256i s01 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(Avx2::Load<false>((__m256i*)(src0 + i + 2 * Avx2::F)), K32_PRM1), K8_SHFL);
                __m256i s10 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(Avx2::Load<false>((__m256i*)(src1 + i + 0 * Avx2::F)), K32_PRM0), K8_SHFL);
                __m256i s11 = _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(Avx2::Load<false>((__m256i*)(src1 + i + 2 * Avx2::F)), K32_PRM1), K8_SHFL);
                __m256i s0 = _mm256_add_epi16(_mm256_maddubs_epi16(s00, Avx2::K8_01), _mm256_maddubs_epi16(s10, Avx2::K8_01));
                __m256i s1 = _mm256_add_epi16(_mm256_maddubs_epi16(s01, Avx2::K8_01), _mm256_maddubs_epi16(s11, Avx2::K8_01));
                __m256i d0 = _mm256_permutevar8x32_epi32(s0, K32_PRM2);
                __m256i d2 = _mm256_permutevar8x32_epi32(s1, K32_PRM3);
                Avx2::Update<update, false>(dst + 0 * Avx2::F, _mm256_madd_epi16(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(d0)), _val));
                Avx2::Update<update, false>(dst + 1 * Avx2::F, _mm256_madd_epi16(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(_mm256_or_si256(d0, d2), 1)), _val));
                Avx2::Update<update, false>(dst + 2 * Avx2::F, _mm256_madd_epi16(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(d2)), _val));
            }
            for (; i < size6; i += 6, dst += 3)
                Base::ResizerByteArea2x2RowUpdate<3, 3, update>(src0 + i, src1 + i, val, dst);
            if (i < size)
                Base::ResizerByteArea2x2RowUpdate<3, 0, update>(src0 + i, src1 + i, val, dst);
        }

        template<> SIMD_INLINE void ResizerByteArea2x2RowSum<3>(const uint8_t* src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, bool tail, int32_t* dst)
        {
            size_t c = 0;
            if (count)
            {
                ResizerByteArea2x2RowUpdateBgr<UpdateSet>(src, src + stride, size, curr, dst), src += 2 * stride, c += 2;
                for (; c < count; c += 2, src += 2 * stride)
                    ResizerByteArea2x2RowUpdateBgr<UpdateAdd>(src, src + stride, size, zero, dst);
                ResizerByteArea2x2RowUpdateBgr<UpdateAdd>(src, tail ? src : src + stride, size, zero - next, dst);
            }
            else
                ResizerByteArea2x2RowUpdateBgr<UpdateSet>(src, tail ? src : src + stride, size, curr - next, dst);
        }

        template<size_t N> void ResizerByteArea2x2::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t bodyW = _param.dstW - (N == 3 ? 1 : 0), rowSize = _param.srcW * N, rowRest = dstStride - _param.dstW * N;
            const int32_t* iy = _iy.data, * ix = _ix.data, * ay = _ay.data, * ax = _ax.data;
            int32_t ay0 = ay[0], ax0 = ax[0];
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += rowRest)
            {
                int32_t* buf = _by.data;
                size_t yn = (iy[dy + 1] - iy[dy]) * 2;
                bool tail = (dy == _param.dstH - 1) && (_param.srcH & 1);
                ResizerByteArea2x2RowSum<N>(src, srcStride, yn, rowSize, ay[dy], ay0, ay[dy + 1], tail, buf), src += yn * srcStride;
                size_t dx = 0;
                for (; dx < bodyW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    Sse41::ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
                }
                for (; dx < _param.dstW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    Base::ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
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
#endif //SIMD_AVX512BW_ENABLE 
}

