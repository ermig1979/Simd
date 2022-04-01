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
#ifdef SIMD_AVX2_ENABLE 
    namespace Avx2
    {
        ResizerByteArea1x1::ResizerByteArea1x1(const ResParam & param)
            : Sse41::ResizerByteArea1x1(param)
        {
        }

        SIMD_INLINE __m256i SaveLoadTail(const uint8_t * ptr, size_t tail)
        {
            uint8_t buffer[DA];
            _mm256_storeu_si256((__m256i*)(buffer), _mm256_loadu_si256((__m256i*)(ptr + tail - A)));
            return _mm256_loadu_si256((__m256i*)(buffer + A - tail));
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t * src0, size_t size, int32_t a, int32_t * dst)
        {
            __m256i alpha = SetInt16(a, a);
            size_t sizeA = AlignLo(size, A);
            size_t i = 0;
            for (; i < sizeA; i += A, dst += A)
            {
                __m256i s0 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)(src0 + i)), K32_TWO_UNPACK_PERMUTE);
                __m256i i0 = UnpackU8<0>(s0);
                __m256i i1 = UnpackU8<1>(s0);
                Update<update, true>(dst + 0 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
            if (i < size)
            {
                __m256i s0 = _mm256_permutevar8x32_epi32(SaveLoadTail(src0 + i, size - i), K32_TWO_UNPACK_PERMUTE);
                __m256i i0 = UnpackU8<0>(s0);
                __m256i i1 = UnpackU8<1>(s0);
                Update<update, true>(dst + 0 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
        }

        template<UpdateType update> SIMD_INLINE void ResizerByteArea1x1RowUpdate(const uint8_t * src0, size_t stride, size_t size, int32_t a0, int32_t a1, int32_t * dst)
        {
            __m256i alpha = SetInt16(a0, a1);
            const uint8_t * src1 = src0 + stride;
            size_t sizeA = AlignLo(size, A);
            size_t i = 0;
            for (; i < sizeA; i += A, dst += A)
            {
                __m256i s0 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)(src0 + i)), K32_TWO_UNPACK_PERMUTE);
                __m256i s1 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)(src1 + i)), K32_TWO_UNPACK_PERMUTE);
                __m256i i0 = UnpackU8<0>(s0, s1);
                __m256i i1 = UnpackU8<1>(s0, s1);
                Update<update, true>(dst + 0 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
            if (i < size)
            {
                __m256i s0 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)(src0 + i)), K32_TWO_UNPACK_PERMUTE);
                __m256i s1 = _mm256_permutevar8x32_epi32(SaveLoadTail(src1 + i, size - i), K32_TWO_UNPACK_PERMUTE);
                __m256i i0 = UnpackU8<0>(s0, s1);
                __m256i i1 = UnpackU8<1>(s0, s1);
                Update<update, true>(dst + 0 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i0)));
                Update<update, true>(dst + 1 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i0)));
                Update<update, true>(dst + 2 * F, _mm256_madd_epi16(alpha, UnpackU8<0>(i1)));
                Update<update, true>(dst + 3 * F, _mm256_madd_epi16(alpha, UnpackU8<1>(i1)));
            }
        }

        SIMD_INLINE void ResizerByteArea1x1RowSum(const uint8_t * src, size_t stride, size_t count, size_t size, int32_t curr, int32_t zero, int32_t next, int32_t * dst)
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

        template<size_t N> void ResizerByteArea1x1::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t dstW = _param.dstW, rowSize = _param.srcW*N, rowRest = dstStride - dstW * N;
            const int32_t * iy = _iy.data, *ix = _ix.data, *ay = _ay.data, *ax = _ax.data;
            int32_t ay0 = ay[0], ax0 = ax[0];
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += rowRest)
            {
                int32_t * buf = _by.data;
                size_t yn = iy[dy + 1] - iy[dy];
                ResizerByteArea1x1RowSum(src, srcStride, yn, rowSize, ay[dy], ay0, ay[dy + 1], buf), src += yn * srcStride;
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
#endif //SIMD_AVX2_ENABLE 
}

