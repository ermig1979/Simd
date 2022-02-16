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
#include "Simd/SimdExtract.h"

#if defined(_MSC_VER) && defined(NDEBUG)
#define SIMD_AVX512BW_RESIZER_BYTE_BICUBIC_MSVS_COMPER_ERROR
#endif

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE 
    namespace Avx512bw
    {
        ResizerByteBicubic::ResizerByteBicubic(const ResParam& param)
            : Avx2::ResizerByteBicubic(param)
        {
        }

#ifndef SIMD_AVX512BW_RESIZER_BYTE_BICUBIC_MSVS_COMPER_ERROR
        template<int N> __m512i LoadAx(const int8_t* ax);

        template<> SIMD_INLINE __m512i LoadAx<1>(const int8_t* ax)
        {
            return _mm512_loadu_si512((__m512i*)ax);
        }

        template<> SIMD_INLINE __m512i LoadAx<2>(const int8_t* ax)
        {
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
            return _mm512_permutexvar_epi32(PERMUTE, _mm512_castsi256_si512(_mm256_loadu_si256((__m256i*)ax)));
        }

        template<> SIMD_INLINE __m512i LoadAx<3>(const int8_t* ax)
        {
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0);
            return _mm512_permutexvar_epi32(PERMUTE, _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)ax)));
        }

        template<> SIMD_INLINE __m512i LoadAx<4>(const int8_t* ax)
        {
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
            return _mm512_permutexvar_epi32(PERMUTE, _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)ax)));
        }

        template<int N> __m512i CubicSumX(const uint8_t* src, const int32_t* ix, __m512i ax, __m512i ay);

        template<> SIMD_INLINE __m512i CubicSumX<1>(const uint8_t* src, const int32_t* ix, __m512i ax, __m512i ay)
        {
            __m512i _src = _mm512_i32gather_epi32(_mm512_loadu_si512((__m512i*)ix), (int32_t*)src, 1);
            return  _mm512_madd_epi16(_mm512_maddubs_epi16(_src, ax), ay);
        }

        template<> SIMD_INLINE __m512i CubicSumX<2>(const uint8_t* src, const int32_t* ix, __m512i ax, __m512i ay)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF);
            __m512i _src = _mm512_shuffle_epi8(_mm512_i32gather_epi64(_mm256_loadu_si256((__m256i*)ix), (long long*)src, 1), SHUFFLE);
            return _mm512_madd_epi16(_mm512_maddubs_epi16(_src, ax), ay);
        }

        template<> SIMD_INLINE __m512i CubicSumX<3>(const uint8_t* src, const int32_t* ix, __m512i ax, __m512i ay)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1);
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0);
            __m512i _src = _mm512_permutexvar_epi32(PERMUTE, _mm512_shuffle_epi8(
                Load<false>((__m128i*)(src + ix[0]), (__m128i*)(src + ix[1]), 
                    (__m128i*)(src + ix[2]), (__m128i*)(src + ix[3])), SHUFFLE));
            return _mm512_madd_epi16(_mm512_maddubs_epi16(_src, ax), ay);
        }

        template<> SIMD_INLINE __m512i CubicSumX<4>(const uint8_t* src, const int32_t* ix, __m512i ax, __m512i ay)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
            __m512i _src = _mm512_shuffle_epi8(Load<false>((__m128i*)(src + ix[0]), 
                (__m128i*)(src + ix[1]), (__m128i*)(src + ix[2]), (__m128i*)(src + ix[3])), SHUFFLE);
            return _mm512_madd_epi16(_mm512_maddubs_epi16(_src, ax), ay);
        }

        template <int N> SIMD_INLINE void StoreBicubicInt(__m512i val, uint8_t* dst)
        {
            _mm_storeu_si128((__m128i*)dst, _mm512_cvtusepi32_epi8(_mm512_max_epi32(val, _mm512_setzero_si512())));
        }

        template <int N> SIMD_INLINE void BicubicInt(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, const int32_t* ix, const int8_t* ax, const __m512i* ay, uint8_t* dst)
        {
            static const __m512i ROUND = SIMD_MM512_SET1_EPI32(Base::BICUBIC_ROUND);
            __m512i _ax = LoadAx<N>(ax);
            __m512i say0 = CubicSumX<N>(src0 - N, ix, _ax, ay[0]);
            __m512i say1 = CubicSumX<N>(src1 - N, ix, _ax, ay[1]);
            __m512i say2 = CubicSumX<N>(src2 - N, ix, _ax, ay[2]);
            __m512i say3 = CubicSumX<N>(src3 - N, ix, _ax, ay[3]);
            __m512i sum = _mm512_add_epi32(_mm512_add_epi32(say0, say1), _mm512_add_epi32(say2, say3));
            __m512i dst0 = _mm512_srai_epi32(_mm512_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
            StoreBicubicInt<N>(dst0, dst);
        }

        template<int N> void ResizerByteBicubic::RunS(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            assert(_xn == 0 && _xt == _param.dstW);
            size_t step = 4 / N * 4;
            size_t body = AlignLoAny(_param.dstW, step);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t sy = _iy[dy];
                const uint8_t* src1 = src + sy * srcStride;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src0 = sy ? src1 - srcStride : src1;
                const uint8_t* src3 = sy < _param.srcH - 2 ? src2 + srcStride : src2;
                const int32_t* ay = _ay.data + dy * 4;
                __m512i ays[4];
                ays[0] = _mm512_set1_epi16(ay[0]);
                ays[1] = _mm512_set1_epi16(ay[1]);
                ays[2] = _mm512_set1_epi16(ay[2]);
                ays[3] = _mm512_set1_epi16(ay[3]);
                size_t dx = 0;
                for (; dx < body; dx += step)
                    BicubicInt<N>(src0, src1, src2, src3, _ix.data + dx, _ax.data + dx * 4, ays, dst + dx * N);
                for (; dx < _param.dstW; dx++)
                    Base::BicubicInt<N, -1, 2>(src0, src1, src2, src3, _ix[dx], _ax.data + dx * 4, ay, dst + dx * N);
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m512i LoadAx1(const int8_t* ax, __mmask16 mask)
        {
            return _mm512_maskz_loadu_epi32(mask, ax);
        }

        SIMD_INLINE __m512i CubicSumX1(const uint8_t* src, const int32_t* ix, __m512i ax, __m512i ay, __mmask16 mask)
        {
            __m512i _src = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, _mm512_maskz_loadu_epi32(mask, ix), (int32_t*)src, 1);
            return  _mm512_madd_epi16(_mm512_maddubs_epi16(_src, ax), ay);
        }

        SIMD_INLINE void BicubicInt1(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, 
            const int32_t* ix, const int8_t* ax, const __m512i* ay, uint8_t* dst, __mmask16 mask)
        {
            static const __m512i ROUND = SIMD_MM512_SET1_EPI32(Base::BICUBIC_ROUND);
            __m512i _ax = LoadAx1(ax, mask);
            __m512i say0 = CubicSumX1(src0 - 1, ix, _ax, ay[0], mask);
            __m512i say1 = CubicSumX1(src1 - 1, ix, _ax, ay[1], mask);
            __m512i say2 = CubicSumX1(src2 - 1, ix, _ax, ay[2], mask);
            __m512i say3 = CubicSumX1(src3 - 1, ix, _ax, ay[3], mask);
            __m512i sum = _mm512_add_epi32(_mm512_add_epi32(say0, say1), _mm512_add_epi32(say2, say3));
            __m512i dst0 = _mm512_srai_epi32(_mm512_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
            _mm_mask_storeu_epi8(dst, mask, _mm512_cvtusepi32_epi8(_mm512_max_epi32(dst0, _mm512_setzero_si512())));
        }

        template<> void ResizerByteBicubic::RunS<1>(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            assert(_xn == 0 && _xt == _param.dstW);
            size_t step = 16;
            size_t body = AlignLoAny(_param.dstW, step);
            __mmask16 tail = TailMask16(_param.dstW - body);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t sy = _iy[dy];
                const uint8_t* src1 = src + sy * srcStride;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src0 = sy ? src1 - srcStride : src1;
                const uint8_t* src3 = sy < _param.srcH - 2 ? src2 + srcStride : src2;
                const int32_t* ay = _ay.data + dy * 4;
                __m512i ays[4];
                ays[0] = _mm512_set1_epi16(ay[0]);
                ays[1] = _mm512_set1_epi16(ay[1]);
                ays[2] = _mm512_set1_epi16(ay[2]);
                ays[3] = _mm512_set1_epi16(ay[3]);
                size_t dx = 0;
                for (; dx < body; dx += step)
                    BicubicInt<1>(src0, src1, src2, src3, _ix.data + dx, _ax.data + dx * 4, ays, dst + dx);
                if(tail)
                    BicubicInt1(src0, src1, src2, src3, _ix.data + dx, _ax.data + dx * 4, ays, dst + dx, tail);
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m512i LoadAx2(const int8_t* ax, __mmask8 mask = __mmask8(-1))
        {
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
            return _mm512_permutexvar_epi32(PERMUTE, _mm512_castsi256_si512(_mm256_maskz_loadu_epi32(mask, ax)));
        }

        SIMD_INLINE __m512i CubicSumX2(const uint8_t* src, const int32_t* ix, __m512i ax, __m512i ay, __mmask8 mask = __mmask8(-1))
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF);
            __m512i _src = _mm512_mask_i32gather_epi64(_mm512_setzero_si512(), mask, _mm256_maskz_loadu_epi32(mask, ix), (long long*)src, 1);
            return _mm512_madd_epi16(_mm512_maddubs_epi16(_mm512_shuffle_epi8(_src, SHUFFLE), ax), ay);
        }

        SIMD_INLINE void BicubicInt2(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, 
            const int32_t* ix, const int8_t* ax, const __m512i* ay, uint8_t* dst, __mmask8 mask = __mmask8(-1))
        {
            static const __m512i ROUND = SIMD_MM512_SET1_EPI32(Base::BICUBIC_ROUND);
            __m512i _ax = LoadAx2(ax, mask);
            __m512i say0 = CubicSumX2(src0 - 2, ix, _ax, ay[0], mask);
            __m512i say1 = CubicSumX2(src1 - 2, ix, _ax, ay[1], mask);
            __m512i say2 = CubicSumX2(src2 - 2, ix, _ax, ay[2], mask);
            __m512i say3 = CubicSumX2(src3 - 2, ix, _ax, ay[3], mask);
            __m512i sum = _mm512_add_epi32(_mm512_add_epi32(say0, say1), _mm512_add_epi32(say2, say3));
            __m512i dst0 = _mm512_srai_epi32(_mm512_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
            _mm_mask_storeu_epi16((int16_t*)dst, mask, _mm512_cvtusepi32_epi8(_mm512_max_epi32(dst0, _mm512_setzero_si512())));
        }

        template<> void ResizerByteBicubic::RunS<2>(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            assert(_xn == 0 && _xt == _param.dstW);
            size_t step = 8;
            size_t body = AlignLoAny(_param.dstW, step);
            __mmask8 tail = TailMask8(_param.dstW - body);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t sy = _iy[dy];
                const uint8_t* src1 = src + sy * srcStride;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src0 = sy ? src1 - srcStride : src1;
                const uint8_t* src3 = sy < _param.srcH - 2 ? src2 + srcStride : src2;
                const int32_t* ay = _ay.data + dy * 4;
                __m512i ays[4];
                ays[0] = _mm512_set1_epi16(ay[0]);
                ays[1] = _mm512_set1_epi16(ay[1]);
                ays[2] = _mm512_set1_epi16(ay[2]);
                ays[3] = _mm512_set1_epi16(ay[3]);
                size_t dx = 0;
                for (; dx < body; dx += step)
                    BicubicInt2(src0, src1, src2, src3, _ix.data + dx, _ax.data + dx * 4, ays, dst + dx * 2);
                if (tail)
                    BicubicInt2(src0, src1, src2, src3, _ix.data + dx, _ax.data + dx * 4, ays, dst + dx * 2, tail);
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m512i LoadAx3(const int8_t* ax, __mmask8 srcMask)
        {
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0);
            return _mm512_permutexvar_epi32(PERMUTE, _mm512_castsi128_si512(_mm_maskz_loadu_epi32(srcMask, ax)));
        }

        SIMD_INLINE __m512i CubicSumX3(const uint8_t* src, const int32_t* ix, __m512i ax, __m512i ay, __mmask8 * srcMask)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1);
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0);
            __m128i src0 = _mm_maskz_loadu_epi32(srcMask[0], src + ix[0]);
            __m128i src1 = _mm_maskz_loadu_epi32(srcMask[1], src + ix[1]);
            __m128i src2 = _mm_maskz_loadu_epi32(srcMask[2], src + ix[2]);
            __m128i src3 = _mm_maskz_loadu_epi32(srcMask[3], src + ix[3]);
            __m512i _src = _mm512_permutexvar_epi32(PERMUTE, _mm512_shuffle_epi8(Set(src0, src1, src2, src3), SHUFFLE));
            return _mm512_madd_epi16(_mm512_maddubs_epi16(_src, ax), ay);
        }

        SIMD_INLINE void BicubicInt3(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3, 
            const int32_t* ix, const int8_t* ax, const __m512i* ay, uint8_t* dst, __mmask8 srcMask[5], __mmask16 dstMask)
        {
            static const __m512i ROUND = SIMD_MM512_SET1_EPI32(Base::BICUBIC_ROUND);
            __m512i _ax = LoadAx3(ax, srcMask[4]);
            __m512i say0 = CubicSumX3(src0 - 3, ix, _ax, ay[0], srcMask);
            __m512i say1 = CubicSumX3(src1 - 3, ix, _ax, ay[1], srcMask);
            __m512i say2 = CubicSumX3(src2 - 3, ix, _ax, ay[2], srcMask);
            __m512i say3 = CubicSumX3(src3 - 3, ix, _ax, ay[3], srcMask);
            __m512i sum = _mm512_add_epi32(_mm512_add_epi32(say0, say1), _mm512_add_epi32(say2, say3));
            __m512i dst0 = _mm512_srai_epi32(_mm512_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
            _mm_mask_storeu_epi8(dst, dstMask, _mm512_cvtusepi32_epi8(_mm512_max_epi32(dst0, _mm512_setzero_si512())));
        }

        template<> void ResizerByteBicubic::RunS<3>(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            assert(_xn == 0 && _xt == _param.dstW);
            size_t step = 4;
            size_t body = AlignLoAny(_param.dstW, step), tail = _param.dstW - body;
            __mmask8 srcMaskTail[5];
            srcMaskTail[0] = tail > 0 ? 0x7 : 0x0;
            srcMaskTail[1] = tail > 1 ? 0x7 : 0x0;
            srcMaskTail[2] = tail > 2 ? 0x7 : 0x0;
            srcMaskTail[3] = tail > 3 ? 0x7 : 0x0;
            srcMaskTail[4] = TailMask8(tail);
            __mmask16 dstMaskTail = TailMask16(tail * 3);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t sy = _iy[dy];
                const uint8_t* src1 = src + sy * srcStride;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src0 = sy ? src1 - srcStride : src1;
                const uint8_t* src3 = sy < _param.srcH - 2 ? src2 + srcStride : src2;
                const int32_t* ay = _ay.data + dy * 4;
                __m512i ays[4];
                ays[0] = _mm512_set1_epi16(ay[0]);
                ays[1] = _mm512_set1_epi16(ay[1]);
                ays[2] = _mm512_set1_epi16(ay[2]);
                ays[3] = _mm512_set1_epi16(ay[3]);
                size_t dx = 0;
                for (; dx < body; dx += step)
                    BicubicInt<3>(src0, src1, src2, src3, _ix.data + dx, _ax.data + dx * 4, ays, dst + dx * 3);
                if (tail)
                    BicubicInt3(src0, src1, src2, src3, _ix.data + dx, _ax.data + dx * 4, ays, dst + dx * 3, srcMaskTail, dstMaskTail);
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m512i LoadAx4(const int8_t* ax, __mmask8 srcMask)
        {
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
            return _mm512_permutexvar_epi32(PERMUTE, _mm512_castsi128_si512(_mm_maskz_loadu_epi32(srcMask, ax)));
        }

        SIMD_INLINE __m512i CubicSumX4(const uint8_t* src, const int32_t* ix, __m512i ax, __m512i ay, __mmask8* srcMask)
        {
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
            __m128i src0 = _mm_maskz_loadu_epi32(srcMask[0], src + ix[0]);
            __m128i src1 = _mm_maskz_loadu_epi32(srcMask[1], src + ix[1]);
            __m128i src2 = _mm_maskz_loadu_epi32(srcMask[2], src + ix[2]);
            __m128i src3 = _mm_maskz_loadu_epi32(srcMask[3], src + ix[3]);
            __m512i _src = _mm512_shuffle_epi8(Set(src0, src1, src2, src3), SHUFFLE);
            return _mm512_madd_epi16(_mm512_maddubs_epi16(_src, ax), ay);
        }

        SIMD_INLINE void BicubicInt4(const uint8_t* src0, const uint8_t* src1, const uint8_t* src2, const uint8_t* src3,
            const int32_t* ix, const int8_t* ax, const __m512i* ay, uint8_t* dst, __mmask8 srcMask[5], __mmask16 dstMask)
        {
            static const __m512i ROUND = SIMD_MM512_SET1_EPI32(Base::BICUBIC_ROUND);
            __m512i _ax = LoadAx4(ax, srcMask[4]);
            __m512i say0 = CubicSumX4(src0 - 4, ix, _ax, ay[0], srcMask);
            __m512i say1 = CubicSumX4(src1 - 4, ix, _ax, ay[1], srcMask);
            __m512i say2 = CubicSumX4(src2 - 4, ix, _ax, ay[2], srcMask);
            __m512i say3 = CubicSumX4(src3 - 4, ix, _ax, ay[3], srcMask);
            __m512i sum = _mm512_add_epi32(_mm512_add_epi32(say0, say1), _mm512_add_epi32(say2, say3));
            __m512i dst0 = _mm512_srai_epi32(_mm512_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
            _mm_mask_storeu_epi8(dst, dstMask, _mm512_cvtusepi32_epi8(_mm512_max_epi32(dst0, _mm512_setzero_si512())));
        }

        template<> void ResizerByteBicubic::RunS<4>(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            assert(_xn == 0 && _xt == _param.dstW);
            size_t step = 4;
            size_t body = AlignLoAny(_param.dstW, step), tail = _param.dstW - body;
            __mmask8 srcMaskTail[5];
            srcMaskTail[0] = tail > 0 ? 0xF : 0x0;
            srcMaskTail[1] = tail > 1 ? 0xF : 0x0;
            srcMaskTail[2] = tail > 2 ? 0xF : 0x0;
            srcMaskTail[3] = tail > 3 ? 0xF : 0x0;
            srcMaskTail[4] = TailMask8(tail);
            __mmask16 dstMaskTail = TailMask16(tail * 4);
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                size_t sy = _iy[dy];
                const uint8_t* src1 = src + sy * srcStride;
                const uint8_t* src2 = src1 + srcStride;
                const uint8_t* src0 = sy ? src1 - srcStride : src1;
                const uint8_t* src3 = sy < _param.srcH - 2 ? src2 + srcStride : src2;
                const int32_t* ay = _ay.data + dy * 4;
                __m512i ays[4];
                ays[0] = _mm512_set1_epi16(ay[0]);
                ays[1] = _mm512_set1_epi16(ay[1]);
                ays[2] = _mm512_set1_epi16(ay[2]);
                ays[3] = _mm512_set1_epi16(ay[3]);
                size_t dx = 0;
                for (; dx < body; dx += step)
                    BicubicInt<4>(src0, src1, src2, src3, _ix.data + dx, _ax.data + dx * 4, ays, dst + dx * 4);
                if (tail)
                    BicubicInt4(src0, src1, src2, src3, _ix.data + dx, _ax.data + dx * 4, ays, dst + dx * 4, srcMaskTail, dstMaskTail);
            }
        }

        //-----------------------------------------------------------------------------------------

        template<int F> SIMD_INLINE void PixelCubicSumX(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst);

        template<> SIMD_INLINE void PixelCubicSumX<1>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            __m512i _src = _mm512_i32gather_epi32(_mm512_loadu_si512((__m512i*)ix), (int32_t*)src, 1);
            __m512i _ax = _mm512_loadu_si512((__m512i*)ax);
            _mm512_storeu_si512((__m512i*)dst, _mm512_madd_epi16(_mm512_maddubs_epi16(_src, _ax), K16_0001));
        }

        template<> SIMD_INLINE void PixelCubicSumX<2>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
            __m512i _ax = _mm512_permutexvar_epi32(PERMUTE, _mm512_castsi256_si512(_mm256_loadu_si256((__m256i*)ax)));
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF);
            __m512i _src = _mm512_shuffle_epi8(_mm512_i32gather_epi64(_mm256_loadu_si256((__m256i*)ix), (long long*)src, 1), SHUFFLE);
            _mm512_storeu_si512((__m512i*)dst, _mm512_madd_epi16(_mm512_maddubs_epi16(_src, _ax), K16_0001));
        }

        template<> SIMD_INLINE void PixelCubicSumX<3>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            static const __m512i PERM_1 = SIMD_MM512_SETR_EPI32(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0);
            __m512i _ax = _mm512_permutexvar_epi32(PERM_1, _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)ax)));
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
                0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1);
            static const __m512i PERM_2 = SIMD_MM512_SETR_EPI32(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0);
            __m512i _src = _mm512_permutexvar_epi32(PERM_2, _mm512_shuffle_epi8(
                Load<false>((__m128i*)(src + ix[0]), (__m128i*)(src + ix[1]), (__m128i*)(src + ix[2]), (__m128i*)(src + ix[3])), SHUFFLE));
            _mm512_storeu_si512((__m512i*)dst, _mm512_madd_epi16(_mm512_maddubs_epi16(_src, _ax), K16_0001));
        }

        template<> SIMD_INLINE void PixelCubicSumX<4>(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            static const __m512i PERMUTE = SIMD_MM512_SETR_EPI32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
            __m512i _ax = _mm512_permutexvar_epi32(PERMUTE, _mm512_castsi128_si512(_mm_loadu_si128((__m128i*)ax)));
            static const __m512i SHUFFLE = SIMD_MM512_SETR_EPI8(
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);
            __m512i _src = _mm512_shuffle_epi8(Load<false>((__m128i*)(src + ix[0]), (__m128i*)(src + ix[1]), (__m128i*)(src + ix[2]), (__m128i*)(src + ix[3])), SHUFFLE);
            _mm512_storeu_si512((__m512i*)dst, _mm512_madd_epi16(_mm512_maddubs_epi16(_src, _ax), K16_0001));
        }

        template<int N> SIMD_INLINE void RowCubicSumX(const uint8_t* src, size_t nose, size_t body, size_t tail, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            size_t step = 4 / N * 4;
            size_t bodyS = nose + AlignLoAny(body - nose, step);

            size_t dx = 0;
            for (; dx < nose; dx++, ax += 4, dst += N)
                Base::PixelCubicSumX<N, 0, 2>(src + ix[dx], ax, dst);
            for (; dx < bodyS; dx += step, ax += 4 * step, dst += N * step)
                PixelCubicSumX<N>(src - N, ix + dx, ax, dst);
            for (; dx < body; dx++, ax += 4, dst += N)
                Base::PixelCubicSumX<N, -1, 2>(src + ix[dx], ax, dst);
            for (; dx < tail; dx++, ax += 4, dst += N)
                Base::PixelCubicSumX<N, -1, 1>(src + ix[dx], ax, dst);
        }

        SIMD_INLINE void BicubicRowInt(const int32_t* src0, const int32_t* src1, const int32_t* src2, const int32_t* src3, const int32_t* ay, size_t body, __mmask16 tail, uint8_t* dst)
        {
            static const __m512i ROUND = SIMD_MM512_SET1_EPI32(Base::BICUBIC_ROUND);
            __m512i ay0 = _mm512_set1_epi32(ay[0]);
            __m512i ay1 = _mm512_set1_epi32(ay[1]);
            __m512i ay2 = _mm512_set1_epi32(ay[2]);
            __m512i ay3 = _mm512_set1_epi32(ay[3]);
            size_t i = 0;
            for (; i < body; i += F)
            {
                __m512i say0 = _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(src0 + i)), ay0);
                __m512i say1 = _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(src1 + i)), ay1);
                __m512i say2 = _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(src2 + i)), ay2);
                __m512i say3 = _mm512_mullo_epi32(_mm512_loadu_si512((__m512i*)(src3 + i)), ay3);
                __m512i sum = _mm512_add_epi32(_mm512_add_epi32(say0, say1), _mm512_add_epi32(say2, say3));
                __m512i dst0 = _mm512_srai_epi32(_mm512_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
                _mm_storeu_si128((__m128i*)(dst + i), _mm512_cvtusepi32_epi8(_mm512_max_epi32(dst0, _mm512_setzero_si512())));
            }
            if (tail)
            {
                __m512i say0 = _mm512_mullo_epi32(_mm512_maskz_loadu_epi32(tail, src0 + i), ay0);
                __m512i say1 = _mm512_mullo_epi32(_mm512_maskz_loadu_epi32(tail, src1 + i), ay1);
                __m512i say2 = _mm512_mullo_epi32(_mm512_maskz_loadu_epi32(tail, src2 + i), ay2);
                __m512i say3 = _mm512_mullo_epi32(_mm512_maskz_loadu_epi32(tail, src3 + i), ay3);
                __m512i sum = _mm512_add_epi32(_mm512_add_epi32(say0, say1), _mm512_add_epi32(say2, say3));
                __m512i dst0 = _mm512_srai_epi32(_mm512_add_epi32(sum, ROUND), Base::BICUBIC_SHIFT);
                _mm_mask_storeu_epi8(dst + i, tail, _mm512_cvtusepi32_epi8(_mm512_max_epi32(dst0, _mm512_setzero_si512())));
            }
        }

        template<int N> void ResizerByteBicubic::RunB(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t rowBody = AlignLo(_bx[0].size, F);
            __mmask16 rowTail = TailMask16(_bx[0].size - rowBody);

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
                BicubicRowInt(pb0, pb1, pb2, pb3, ay, rowBody, rowTail, dst);
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void PixelCubicSumX1(const uint8_t* src, const int32_t* ix, const int8_t* ax, int32_t* dst, __mmask16 mask)
        {
            __m512i _src = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mask, _mm512_maskz_loadu_epi32(mask, ix), (int32_t*)src, 1);
            __m512i _ax = _mm512_maskz_loadu_epi32(mask, ax);
            _mm512_mask_storeu_epi32(dst, mask, _mm512_madd_epi16(_mm512_maddubs_epi16(_src, _ax), K16_0001));
        }

        SIMD_INLINE void RowCubicSumX1(const uint8_t* src, size_t nose, size_t body, size_t tail, const int32_t* ix, const int8_t* ax, int32_t* dst)
        {
            size_t bodyS = nose + AlignLoAny(body - nose, 16);
            size_t bodyTail = body - bodyS;
            __mmask16 bodyTailMask = TailMask16(bodyTail);

            size_t dx = 0;
            for (; dx < nose; dx++, ax += 4, dst += 1)
                Base::PixelCubicSumX<1, 0, 2>(src + ix[dx], ax, dst);
            for (; dx < bodyS; dx += 16, ax += 64, dst += 16)
                PixelCubicSumX<1>(src - 1, ix + dx, ax, dst);
            for (; dx < body; dx += bodyTail, ax += 4 * bodyTail, dst += bodyTail)
                PixelCubicSumX1(src - 1, ix + dx, ax, dst, bodyTailMask);
            for (; dx < tail; dx++, ax += 4, dst += 1)
                Base::PixelCubicSumX<1, -1, 1>(src + ix[dx], ax, dst);
        }

        template<> void ResizerByteBicubic::RunB<1>(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t rowBody = AlignLo(_bx[0].size, F);
            __mmask16 rowTail = TailMask16(_bx[0].size - rowBody);

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
                    RowCubicSumX1(ps, _xn, _xt, _param.dstW, _ix.data, _ax.data, pb);
                    next++;
                }
                prev = next;

                const int32_t* ay = _ay.data + dy * 4;
                int32_t* pb0 = _bx[(sy + 0) & 3].data;
                int32_t* pb1 = _bx[(sy + 1) & 3].data;
                int32_t* pb2 = _bx[(sy + 2) & 3].data;
                int32_t* pb3 = _bx[(sy + 3) & 3].data;
                BicubicRowInt(pb0, pb1, pb2, pb3, ay, rowBody, rowTail, dst);
            }
        }

        //-----------------------------------------------------------------------------------------

        void ResizerByteBicubic::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            bool sparse = _param.dstH * 3.0 <= _param.srcH;
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
#else // SIMD_AVX512BW_RESIZER_BYTE_BICUBIC_MSVS_COMPER_ERROR
        void ResizerByteBicubic::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            Avx2::ResizerByteBicubic::Run(src, srcStride, dst, dstStride);
        }
#endif // SIMD_AVX512BW_RESIZER_BYTE_BICUBIC_MSVS_COMPER_ERROR
    }
#endif //SIMD_AVX512BW_ENABLE 
}

