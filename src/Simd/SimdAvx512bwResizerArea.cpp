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
            size_t dstW = _param.dstW, rowSize = _param.srcW*N, rowRest = dstStride - dstW * N;
            const int32_t * iy = _iy.data, *ix = _ix.data, *ay = _ay.data, *ax = _ax.data;
            int32_t ay0 = ay[0], ax0 = ax[0];
            size_t rowSizeA = AlignLo(rowSize, A);
            __mmask64 tail = TailMask64(rowSize - rowSizeA);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += rowRest)
            {
                int32_t * buf = _by.data;
                size_t yn = iy[dy + 1] - iy[dy];
                ResizerByteArea1x1RowSum(src, srcStride, yn, rowSize, rowSizeA, ay[dy], ay0, ay[dy + 1], buf, tail), src += yn * srcStride;
                for (size_t dx = 0; dx < dstW; dx++, dst += N)
                {
                    size_t xn = ix[dx + 1] - ix[dx];
                    Sse41::ResizerByteAreaResult<N>(buf, xn, ax[dx], ax0, ax[dx + 1], dst), buf += xn * N;
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
    }
#endif //SIMD_AVX512BW_ENABLE 
}

