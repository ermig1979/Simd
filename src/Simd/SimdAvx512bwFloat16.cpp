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
#include "Simd/SimdExtract.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<bool mask> SIMD_INLINE void Float32ToFloat16(const float * src, uint16_t * dst, const __mmask16 * srcTails, __mmask32 dstTail)
        {
            __m256i lo = _mm512_cvtps_ph(mask ? _mm512_maskz_loadu_ps(srcTails[0], src + 0) : _mm512_loadu_ps(src + 0), 0);
            __m256i hi = _mm512_cvtps_ph(mask ? _mm512_maskz_loadu_ps(srcTails[1], src + F) : _mm512_loadu_ps(src + F), 0);
            __m512i dstValue = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            if (mask)
                _mm512_mask_storeu_epi16(dst, dstTail, dstValue);
            else
                _mm512_storeu_si512(dst, dstValue);
        }

        SIMD_INLINE void Float32ToFloat16x2(const float * src, uint16_t * dst)
        {
            _mm512_storeu_si512(dst + 0 * HA, _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtps_ph(_mm512_loadu_ps(src + 0 * F), 0)), _mm512_cvtps_ph(_mm512_loadu_ps(src + 1 * F), 0), 1));
            _mm512_storeu_si512(dst + 1 * HA, _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtps_ph(_mm512_loadu_ps(src + 2 * F), 0)), _mm512_cvtps_ph(_mm512_loadu_ps(src + 3 * F), 0), 1));
        }

        void Float32ToFloat16(const float * src, size_t size, uint16_t * dst)
        {
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            size_t alignedSize = Simd::AlignLo(size, DF);
            __mmask16 srcTailMasks[2];
            for (size_t c = 0; c < 2; ++c)
                srcTailMasks[c] = TailMask16(size - alignedSize - F*c);
            __mmask32 dstTailMask = TailMask32(size - alignedSize);

            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
                Float32ToFloat16x2(src + i, dst + i);
            for (; i < alignedSize; i += DF)
                Float32ToFloat16<false>(src + i, dst + i, srcTailMasks, dstTailMask);
            if (i < size)
                Float32ToFloat16<true>(src + i, dst + i, srcTailMasks, dstTailMask);
        }

        template<bool mask> SIMD_INLINE void Float16ToFloat32(const uint16_t * src, float * dst, __mmask32 srcTail, const __mmask16 * dstTails)
        {
            __m512i _src = mask ? _mm512_maskz_loadu_epi16(srcTail, src) : _mm512_loadu_si512(src);
            __m512 dst0 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_src, 0));
            __m512 dst1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_src, 1));
            if (mask)
            {
                _mm512_mask_storeu_ps(dst + 0, dstTails[0], dst0);
                _mm512_mask_storeu_ps(dst + F, dstTails[1], dst1);
            }
            else
            {
                _mm512_storeu_ps(dst + 0, dst0);
                _mm512_storeu_ps(dst + F, dst1);
            }
        }

        SIMD_INLINE void Float16ToFloat32x2(const uint16_t * src, float * dst)
        {
#if defined(_MSC_VER)
            const __m512i src0 = _mm512_loadu_si512(src + 00);
            _mm512_storeu_ps(dst + 0 * F, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(src0, 0)));
            _mm512_storeu_ps(dst + 1 * F, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(src0, 1)));
            const __m512i src1 = _mm512_loadu_si512(src + HA);
            _mm512_storeu_ps(dst + 2 * F, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(src1, 0)));
            _mm512_storeu_ps(dst + 3 * F, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(src1, 1)));
#else
            _mm512_storeu_ps(dst + 0 * F, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)src + 0)));
            _mm512_storeu_ps(dst + 1 * F, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)src + 1)));
            _mm512_storeu_ps(dst + 2 * F, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)src + 2)));
            _mm512_storeu_ps(dst + 3 * F, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)src + 3)));
#endif
        }

        void Float16ToFloat32(const uint16_t * src, size_t size, float * dst)
        {
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            size_t alignedSize = Simd::AlignLo(size, DF);
            __mmask32 srcTailMask = TailMask32(size - alignedSize);
            __mmask16 dstTailMasks[2];
            for (size_t c = 0; c < 2; ++c)
                dstTailMasks[c] = TailMask16(size - alignedSize - F*c);

            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
                Float16ToFloat32x2(src + i, dst + i);
            for (; i < alignedSize; i += DF)
                Float16ToFloat32<false>(src + i, dst + i, srcTailMask, dstTailMasks);
            if (i < size)
                Float16ToFloat32<true>(src + i, dst + i, srcTailMask, dstTailMasks);
        }

        template <int part> SIMD_INLINE void SquaredDifferenceSum16f(const __m512i & a, const __m512i & b, __m512 * sums)
        {
            __m512 _a = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a, part));
            __m512 _b = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b, part));
            __m512 _d = _mm512_sub_ps(_a, _b);
            sums[part] = _mm512_fmadd_ps(_d, _d, sums[part]);
        }

        template <bool mask> SIMD_INLINE void SquaredDifferenceSum16f2(const uint16_t * a, const uint16_t * b, __m512 * sums, __mmask32 tail = -1)
        {
            __m512i a0 = mask ? _mm512_maskz_loadu_epi16(tail, a) : _mm512_loadu_si512(a);
            __m512i b0 = mask ? _mm512_maskz_loadu_epi16(tail, b) : _mm512_loadu_si512(b);
            SquaredDifferenceSum16f<0>(a0, b0, sums);
            SquaredDifferenceSum16f<1>(a0, b0, sums);
        }

        SIMD_INLINE void SquaredDifferenceSum16f4(const uint16_t * a, const uint16_t * b, __m512 * sums)
        {
#if defined(_MSC_VER)
            __m512i a0 = _mm512_loadu_si512(a + 00);
            __m512i b0 = _mm512_loadu_si512(b + 00);
            SquaredDifferenceSum16f<0>(a0, b0, sums);
            SquaredDifferenceSum16f<1>(a0, b0, sums);
            __m512i a1 = _mm512_loadu_si512(a + HA);
            __m512i b1 = _mm512_loadu_si512(b + HA);
            SquaredDifferenceSum16f<0>(a1, b1, sums);
            SquaredDifferenceSum16f<1>(a1, b1, sums);
#else
            __m512 a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)a + 0));
            __m512 b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)b + 0));
            __m512 d0 = _mm512_sub_ps(a0, b0);
            sums[0] = _mm512_fmadd_ps(d0, d0, sums[0]);

            __m512 a1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)a + 1));
            __m512 b1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)b + 1));
            __m512 d1 = _mm512_sub_ps(a1, b1);
            sums[1] = _mm512_fmadd_ps(d1, d1, sums[1]);

            __m512 a2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)a + 2));
            __m512 b2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)b + 2));
            __m512 d2 = _mm512_sub_ps(a2, b2);
            sums[0] = _mm512_fmadd_ps(d2, d2, sums[0]);

            __m512 a3 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)a + 3));
            __m512 b3 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)b + 3));
            __m512 d3 = _mm512_sub_ps(a3, b3);
            sums[1] = _mm512_fmadd_ps(d3, d3, sums[1]);
#endif
        }

        void SquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum)
        {
            size_t alignedSize = AlignLo(size, DF);
            __mmask32 tailMask = TailMask32(size - alignedSize);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            __m512 sums[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
            for (; i < fullAlignedSize; i += QF)
                SquaredDifferenceSum16f4(a + i, b + i, sums);
            for (; i < alignedSize; i += DF)
                SquaredDifferenceSum16f2<false>(a + i, b + i, sums);
            if (i < size)
                SquaredDifferenceSum16f2<true>(a + i, b + i, sums, tailMask);
            sums[0] = _mm512_add_ps(sums[0], sums[1]);
            *sum = ExtractSum(sums[0]);
        }

        template <int part> SIMD_INLINE void CosineDistance16f(const __m512i & a, const __m512i & b, __m512 * aa, __m512 * ab, __m512 * bb)
        {
            __m512 a0 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a, part));
            __m512 b0 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b, part));
            aa[part] = _mm512_fmadd_ps(a0, a0, aa[part]);
            ab[part] = _mm512_fmadd_ps(a0, b0, ab[part]);
            bb[part] = _mm512_fmadd_ps(b0, b0, bb[part]);
        }

        template <bool mask> SIMD_INLINE void CosineDistance16f2(const uint16_t * a, const uint16_t * b, __m512 * aa, __m512 * ab, __m512 * bb, __mmask32 tail = -1)
        {
            __m512i a0 = mask ? _mm512_maskz_loadu_epi16(tail, a) : _mm512_loadu_si512(a);
            __m512i b0 = mask ? _mm512_maskz_loadu_epi16(tail, b) : _mm512_loadu_si512(b);
            CosineDistance16f<0>(a0, b0, aa, ab, bb);
            CosineDistance16f<1>(a0, b0, aa, ab, bb);
        }

        SIMD_INLINE void CosineDistance16f4(const uint16_t * a, const uint16_t * b, __m512 * aa, __m512 * ab, __m512 * bb)
        {
            __m512i a0 = _mm512_loadu_si512(a + 00);
            __m512i b0 = _mm512_loadu_si512(b + 00);
            CosineDistance16f<0>(a0, b0, aa, ab, bb);
            CosineDistance16f<1>(a0, b0, aa, ab, bb);
            __m512i a1 = _mm512_loadu_si512(a + HA);
            __m512i b1 = _mm512_loadu_si512(b + HA);
            CosineDistance16f<0>(a1, b1, aa, ab, bb);
            CosineDistance16f<1>(a1, b1, aa, ab, bb);
        }

        void CosineDistance16f(const uint16_t * a, const uint16_t * b, size_t size, float * distance)
        {
            size_t alignedSize = AlignLo(size, DF);
            __mmask32 tailMask = TailMask32(size - alignedSize);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            __m512 _aa[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
            __m512 _ab[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
            __m512 _bb[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
            for (; i < fullAlignedSize; i += QF)
                CosineDistance16f4(a + i, b + i, _aa, _ab, _bb);
            for (; i < alignedSize; i += DF)
                CosineDistance16f2<false>(a + i, b + i, _aa, _ab, _bb);
            if (i < size)
                CosineDistance16f2<true>(a + i, b + i, _aa, _ab, _bb, tailMask);
            float aa = ExtractSum(_mm512_add_ps(_aa[0], _aa[1]));
            float ab = ExtractSum(_mm512_add_ps(_ab[0], _ab[1]));
            float bb = ExtractSum(_mm512_add_ps(_bb[0], _bb[1]));
            *distance = 1.0f - ab / ::sqrt(aa*bb);
        }

        SIMD_INLINE __m512 Tail(size_t tail)
        {
            const int32_t mask[DF] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
            return _mm512_loadu_ps((float*)(mask + tail));
        }

        static void Squares(size_t M, size_t K, const uint16_t * const * A, float * squares)
        {
            size_t M4 = AlignLo(M, 4);
            size_t KF = AlignLo(K, F);
            __m512 mask = Tail(K - KF);
            size_t i = 0;
            for (; i < M4; i += 4)
            {
                __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                for (size_t k = 0; k < KF; k += F)
                {
                    __m512 a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[i + 0] + k)));
                    __m512 a1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[i + 1] + k)));
                    __m512 a2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[i + 2] + k)));
                    __m512 a3 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[i + 3] + k)));
                    sums[0] = _mm512_fmadd_ps(a0, a0, sums[0]);
                    sums[1] = _mm512_fmadd_ps(a1, a1, sums[1]);
                    sums[2] = _mm512_fmadd_ps(a2, a2, sums[2]);
                    sums[3] = _mm512_fmadd_ps(a3, a3, sums[3]);
                }
                if (KF < K)
                {
                    size_t k = K - F;
                    __m512 a0 = _mm512_and_ps(mask, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[i + 0] + k))));
                    __m512 a1 = _mm512_and_ps(mask, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[i + 1] + k))));
                    __m512 a2 = _mm512_and_ps(mask, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[i + 2] + k))));
                    __m512 a3 = _mm512_and_ps(mask, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[i + 3] + k))));
                    sums[0] = _mm512_fmadd_ps(a0, a0, sums[0]);
                    sums[1] = _mm512_fmadd_ps(a1, a1, sums[1]);
                    sums[2] = _mm512_fmadd_ps(a2, a2, sums[2]);
                    sums[3] = _mm512_fmadd_ps(a3, a3, sums[3]);
                }
                _mm_storeu_ps(squares + i, Extract4Sums(sums));
            }
            for (; i < M; i += 1)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t k = 0; k < KF; k += F)
                {
                    __m512 a = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[i] + k)));
                    sum = _mm512_fmadd_ps(a, a, sum);
                }
                if (KF < K)
                {
                    size_t k = K - F;
                    __m512 a = _mm512_and_ps(mask, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[i] + k))));
                    sum = _mm512_fmadd_ps(a, a, sum);
                }
                squares[i] = ExtractSum(sum);
            }
        }

        static void MicroCosineDistances6x4(size_t K, const uint16_t * const * A, const uint16_t * const * B, const float * aa, const float * bb, float * distances, size_t stride)
        {
            size_t K16 = K & (~15);
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c03 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c12 = _mm512_setzero_ps();
            __m512 c13 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c22 = _mm512_setzero_ps();
            __m512 c23 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c31 = _mm512_setzero_ps();
            __m512 c32 = _mm512_setzero_ps();
            __m512 c33 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c41 = _mm512_setzero_ps();
            __m512 c42 = _mm512_setzero_ps();
            __m512 c43 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 c51 = _mm512_setzero_ps();
            __m512 c52 = _mm512_setzero_ps();
            __m512 c53 = _mm512_setzero_ps();
            __m512 a0, b0, b1, b2, b3;
            for (size_t k = 0; k < K16; k += 16)
            {
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[0] + k)));
                b1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[1] + k)));
                b2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[2] + k)));
                b3 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[3] + k)));
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[0] + k)));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                c02 = _mm512_fmadd_ps(a0, b2, c02);
                c03 = _mm512_fmadd_ps(a0, b3, c03);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[1] + k)));
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                c12 = _mm512_fmadd_ps(a0, b2, c12);
                c13 = _mm512_fmadd_ps(a0, b3, c13);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[2] + k)));
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                c22 = _mm512_fmadd_ps(a0, b2, c22);
                c23 = _mm512_fmadd_ps(a0, b3, c23);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[3] + k)));
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                c32 = _mm512_fmadd_ps(a0, b2, c32);
                c33 = _mm512_fmadd_ps(a0, b3, c33);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[4] + k)));
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                c41 = _mm512_fmadd_ps(a0, b1, c41);
                c42 = _mm512_fmadd_ps(a0, b2, c42);
                c43 = _mm512_fmadd_ps(a0, b3, c43);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[5] + k)));
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                c51 = _mm512_fmadd_ps(a0, b1, c51);
                c52 = _mm512_fmadd_ps(a0, b2, c52);
                c53 = _mm512_fmadd_ps(a0, b3, c53);
            }
            if (K16 < K)
            {
                size_t k = K - 16;
                __m512 tail = Tail(K - K16);
                b0 = _mm512_and_ps(tail, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[0] + k))));
                b1 = _mm512_and_ps(tail, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[1] + k))));
                b2 = _mm512_and_ps(tail, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[2] + k))));
                b3 = _mm512_and_ps(tail, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[3] + k))));
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[0] + k)));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                c02 = _mm512_fmadd_ps(a0, b2, c02);
                c03 = _mm512_fmadd_ps(a0, b3, c03);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[1] + k)));
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                c11 = _mm512_fmadd_ps(a0, b1, c11);
                c12 = _mm512_fmadd_ps(a0, b2, c12);
                c13 = _mm512_fmadd_ps(a0, b3, c13);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[2] + k)));
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                c21 = _mm512_fmadd_ps(a0, b1, c21);
                c22 = _mm512_fmadd_ps(a0, b2, c22);
                c23 = _mm512_fmadd_ps(a0, b3, c23);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[3] + k)));
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                c31 = _mm512_fmadd_ps(a0, b1, c31);
                c32 = _mm512_fmadd_ps(a0, b2, c32);
                c33 = _mm512_fmadd_ps(a0, b3, c33);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[4] + k)));
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                c41 = _mm512_fmadd_ps(a0, b1, c41);
                c42 = _mm512_fmadd_ps(a0, b2, c42);
                c43 = _mm512_fmadd_ps(a0, b3, c43);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[5] + k)));
                c50 = _mm512_fmadd_ps(a0, b0, c50);
                c51 = _mm512_fmadd_ps(a0, b1, c51);
                c52 = _mm512_fmadd_ps(a0, b2, c52);
                c53 = _mm512_fmadd_ps(a0, b3, c53);
            }
            __m128 _bb = _mm_loadu_ps(bb);
            __m128 _1 = _mm_set1_ps(1.0f);
            _mm_storeu_ps(distances + 0 * stride, _mm_sub_ps(_1, _mm_div_ps(Extract4Sums(c00, c01, c02, c03), _mm_sqrt_ps(_mm_mul_ps(_bb, _mm_set1_ps(aa[0]))))));
            _mm_storeu_ps(distances + 1 * stride, _mm_sub_ps(_1, _mm_div_ps(Extract4Sums(c10, c11, c12, c13), _mm_sqrt_ps(_mm_mul_ps(_bb, _mm_set1_ps(aa[1]))))));
            _mm_storeu_ps(distances + 2 * stride, _mm_sub_ps(_1, _mm_div_ps(Extract4Sums(c20, c21, c22, c23), _mm_sqrt_ps(_mm_mul_ps(_bb, _mm_set1_ps(aa[2]))))));
            _mm_storeu_ps(distances + 3 * stride, _mm_sub_ps(_1, _mm_div_ps(Extract4Sums(c30, c31, c32, c33), _mm_sqrt_ps(_mm_mul_ps(_bb, _mm_set1_ps(aa[3]))))));
            _mm_storeu_ps(distances + 4 * stride, _mm_sub_ps(_1, _mm_div_ps(Extract4Sums(c40, c41, c42, c43), _mm_sqrt_ps(_mm_mul_ps(_bb, _mm_set1_ps(aa[4]))))));
            _mm_storeu_ps(distances + 5 * stride, _mm_sub_ps(_1, _mm_div_ps(Extract4Sums(c50, c51, c52, c53), _mm_sqrt_ps(_mm_mul_ps(_bb, _mm_set1_ps(aa[5]))))));
        }

        static void MicroCosineDistances6x1(size_t K, const uint16_t * const * A, const uint16_t * const * B, const float * aa, const float * bb, float * distances, size_t stride)
        {
            size_t K16 = K & (~15);
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps();
            __m512 c40 = _mm512_setzero_ps();
            __m512 c50 = _mm512_setzero_ps();
            __m512 a0, b0;
            for (size_t k = 0; k < K16; k += 16)
            {
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[0] + k)));
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[0] + k)));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[1] + k)));
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[2] + k)));
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[3] + k)));
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[4] + k)));
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[5] + k)));
                c50 = _mm512_fmadd_ps(a0, b0, c50);
            }
            if (K16 < K)
            {
                size_t k = K - 16;
                __m512 tail = Tail(K - K16);
                b0 = _mm512_and_ps(tail, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[0] + k))));
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[0] + k)));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[1] + k)));
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[2] + k)));
                c20 = _mm512_fmadd_ps(a0, b0, c20);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[3] + k)));
                c30 = _mm512_fmadd_ps(a0, b0, c30);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[4] + k)));
                c40 = _mm512_fmadd_ps(a0, b0, c40);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[5] + k)));
                c50 = _mm512_fmadd_ps(a0, b0, c50);
            }
            distances[0 * stride] = 1.0f - ExtractSum(c00) / sqrt(bb[0] * aa[0]);
            distances[1 * stride] = 1.0f - ExtractSum(c10) / sqrt(bb[0] * aa[1]);
            distances[2 * stride] = 1.0f - ExtractSum(c20) / sqrt(bb[0] * aa[2]);
            distances[3 * stride] = 1.0f - ExtractSum(c30) / sqrt(bb[0] * aa[3]);
            distances[4 * stride] = 1.0f - ExtractSum(c40) / sqrt(bb[0] * aa[4]);
            distances[5 * stride] = 1.0f - ExtractSum(c50) / sqrt(bb[0] * aa[5]);
        }

        static void MicroCosineDistances3x4(size_t K, const uint16_t * const * A, const uint16_t * const * B, const float * aa, const float * bb, float * distances, size_t stride)
        {
            size_t K16 = K & (~15);
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c03 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c11 = _mm512_setzero_ps();
            __m512 c12 = _mm512_setzero_ps();
            __m512 c13 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 c21 = _mm512_setzero_ps();
            __m512 c22 = _mm512_setzero_ps();
            __m512 c23 = _mm512_setzero_ps();
            __m512 a0, a1, a2, b0;
            for (size_t k = 0; k < K16; k += 16)
            {
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[0] + k)));
                a1 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[1] + k)));
                a2 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[2] + k)));
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[0] + k)));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[1] + k)));
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                c11 = _mm512_fmadd_ps(a1, b0, c11);
                c21 = _mm512_fmadd_ps(a2, b0, c21);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[2] + k)));
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                c12 = _mm512_fmadd_ps(a1, b0, c12);
                c22 = _mm512_fmadd_ps(a2, b0, c22);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[3] + k)));
                c03 = _mm512_fmadd_ps(a0, b0, c03);
                c13 = _mm512_fmadd_ps(a1, b0, c13);
                c23 = _mm512_fmadd_ps(a2, b0, c23);
            }
            if (K16 < K)
            {
                size_t k = K - 16;
                __m512 tail = Tail(K - K16);
                a0 = _mm512_and_ps(tail, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[0] + k))));
                a1 = _mm512_and_ps(tail, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[1] + k))));
                a2 = _mm512_and_ps(tail, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[2] + k))));
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[0] + k)));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[1] + k)));
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                c11 = _mm512_fmadd_ps(a1, b0, c11);
                c21 = _mm512_fmadd_ps(a2, b0, c21);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[2] + k)));
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                c12 = _mm512_fmadd_ps(a1, b0, c12);
                c22 = _mm512_fmadd_ps(a2, b0, c22);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[3] + k)));
                c03 = _mm512_fmadd_ps(a0, b0, c03);
                c13 = _mm512_fmadd_ps(a1, b0, c13);
                c23 = _mm512_fmadd_ps(a2, b0, c23);
            }
            __m128 _bb = _mm_loadu_ps(bb);
            __m128 _1 = _mm_set1_ps(1.0f);
            _mm_storeu_ps(distances + 0 * stride, _mm_sub_ps(_1, _mm_div_ps(Extract4Sums(c00, c01, c02, c03), _mm_sqrt_ps(_mm_mul_ps(_bb, _mm_set1_ps(aa[0]))))));
            _mm_storeu_ps(distances + 1 * stride, _mm_sub_ps(_1, _mm_div_ps(Extract4Sums(c10, c11, c12, c13), _mm_sqrt_ps(_mm_mul_ps(_bb, _mm_set1_ps(aa[1]))))));
            _mm_storeu_ps(distances + 2 * stride, _mm_sub_ps(_1, _mm_div_ps(Extract4Sums(c20, c21, c22, c23), _mm_sqrt_ps(_mm_mul_ps(_bb, _mm_set1_ps(aa[2]))))));
        }

        static void MicroCosineDistances3x1(size_t K, const uint16_t * const * A, const uint16_t * const * B, const float * aa, const float * bb, float * distances, size_t stride)
        {
            size_t K16 = K & (~15);
            __m512 c00 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps();
            __m512 a0, b0;
            for (size_t k = 0; k < K16; k += 16)
            {
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[0] + k)));
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[0] + k)));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[1] + k)));
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[2] + k)));
                c20 = _mm512_fmadd_ps(a0, b0, c20);
            }
            if (K16 < K)
            {
                size_t k = K - 16;
                __m512 tail = Tail(K - K16);
                b0 = _mm512_and_ps(tail, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[0] + k))));
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[0] + k)));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[1] + k)));
                c10 = _mm512_fmadd_ps(a0, b0, c10);
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[2] + k)));
                c20 = _mm512_fmadd_ps(a0, b0, c20);
            }
            distances[0 * stride] = 1.0f - ExtractSum(c00) / sqrt(bb[0] * aa[0]);
            distances[1 * stride] = 1.0f - ExtractSum(c10) / sqrt(bb[0] * aa[1]);
            distances[2 * stride] = 1.0f - ExtractSum(c20) / sqrt(bb[0] * aa[2]);
        }

        static void MicroCosineDistances1x4(size_t K, const uint16_t * const * A, const uint16_t * const * B, const float * aa, const float * bb, float * distances, size_t stride)
        {
            size_t K16 = K & (~15);
            __m512 c00 = _mm512_setzero_ps();
            __m512 c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps();
            __m512 c03 = _mm512_setzero_ps();
            __m512 a0, b0;
            for (size_t k = 0; k < K16; k += 16)
            {
                a0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[0] + k)));
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[0] + k)));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[1] + k)));
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[2] + k)));
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[3] + k)));
                c03 = _mm512_fmadd_ps(a0, b0, c03);
            }
            if (K16 < K)
            {
                size_t k = K - 16;
                __m512 tail = Tail(K - K16);
                a0 = _mm512_and_ps(tail, _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(A[0] + k))));
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[0] + k)));
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[1] + k)));
                c01 = _mm512_fmadd_ps(a0, b0, c01);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[2] + k)));
                c02 = _mm512_fmadd_ps(a0, b0, c02);
                b0 = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(B[3] + k)));
                c03 = _mm512_fmadd_ps(a0, b0, c03);
            }
            __m128 _bb = _mm_loadu_ps(bb);
            __m128 _1 = _mm_set1_ps(1.0f);
            _mm_storeu_ps(distances + 0 * stride, _mm_sub_ps(_1, _mm_div_ps(Extract4Sums(c00, c01, c02, c03), _mm_sqrt_ps(_mm_mul_ps(_bb, _mm_set1_ps(aa[0]))))));
        }

        static void MacroCosineDistances(size_t M, size_t N, size_t K, const uint16_t * const * A, const uint16_t * const * B, const float * aa, const float * bb, float * distances, size_t stride)
        {
            size_t M3 = AlignLoAny(M, 3);
            size_t M6 = AlignLoAny(M, 6);
            size_t N4 = AlignLo(N, 4);
            size_t i = 0;
            for (; i < M6; i += 6)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistances6x4(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                for (; j < N; j += 1)
                    MicroCosineDistances6x1(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                distances += 6 * stride;
            }
            for (; i < M3; i += 3)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistances3x4(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                for (; j < N; j += 1)
                    MicroCosineDistances3x1(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                distances += 3 * stride;
            }
            for (; i < M; i++)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistances1x4(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                for (; j < N; j += 1)
                    CosineDistance16f(A[i], B[j], K, distances + j);
                distances += 1 * stride;
            }
        }

        void CosineDistancesMxNa16f(size_t M, size_t N, size_t K, const uint16_t* const* A, const uint16_t* const* B, float* distances)
        {
            const size_t L2 = Base::AlgCacheL2();
            size_t mN = AlignLoAny(L2 / 2 / K, 4);
            size_t mM = AlignLoAny(L2 / 2 / K, 6);
            Array32f aa(mM), bb(N);
            for (size_t i = 0; i < M; i += mM)
            {
                size_t dM = Simd::Min(M, i + mM) - i;
                Squares(dM, K, A + i, aa.data);
                for (size_t j = 0; j < N; j += mN)
                {
                    size_t dN = Simd::Min(N, j + mN) - j;
                    if (i == 0)
                        Squares(dN, K, B + j, bb.data + j);
                    MacroCosineDistances(dM, dN, K, A + i, B + j, aa.data, bb.data + j, distances + i * N + j, N);
                }
            }
        }

        void CosineDistancesMxNp16f(size_t M, size_t N, size_t K, const uint16_t* A, const uint16_t* B, float* distances)
        {
            const size_t L2 = Base::AlgCacheL2();
            size_t mN = AlignLoAny(L2 / 2 / K, 4);
            size_t mM = AlignLoAny(L2 / 2 / K, 6);
            Array32f aa(mM), bb(N);
            Array16ucp ap(mM), bp(N);
            for (size_t i = 0; i < M; i += mM)
            {
                size_t dM = Simd::Min(M, i + mM) - i;
                for (size_t k = 0; k < dM; ++k)
                    ap[k] = A + k * K;
                Squares(dM, K, ap.data, aa.data);
                for (size_t j = 0; j < N; j += mN)
                {
                    size_t dN = Simd::Min(N, j + mN) - j;
                    if (i == 0)
                    {
                        for (size_t k = j, n = j + dN; k < n; ++k)
                            bp[k] = B + k * K;
                        Squares(dN, K, bp.data + j, bb.data + j);
                    }
                    MacroCosineDistances(dM, dN, K, ap.data, bp.data + j, aa.data, bb.data + j, distances + i * N + j, N);
                }
                A += dM * K;
            }
        }

        void VectorNormNa16f(size_t N, size_t K, const uint16_t* const* A, float* norms)
        {
            Squares(N, K, A, norms);
            size_t N16 = AlignLo(N, 16);
            for (size_t j = 0; j < N16; j += 16)
            {
                __m512 sum = _mm512_loadu_ps(norms + j);
                _mm512_storeu_ps(norms + j, _mm512_sqrt_ps(sum));
            }
            if (N16 < N)
            {
                __mmask16 tail = TailMask16(N - N16);
                __m512 sum = _mm512_maskz_loadu_ps(tail, norms + N16);
                _mm512_mask_storeu_ps(norms + N16, tail, _mm512_sqrt_ps(sum));
            }
        }

        void VectorNormNp16f(size_t N, size_t K, const uint16_t* A, float* norms)
        {
            Array16ucp a(N);
            for (size_t j = 0; j < N; ++j)
                a[j] = A + j * K;
            VectorNormNa16f(N, K, a.data, norms);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
