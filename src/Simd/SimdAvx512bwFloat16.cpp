/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<bool align, bool mask> SIMD_INLINE void Float32ToFloat16(const float * src, uint16_t * dst, const __mmask16 * srcTails, __mmask32 dstTail)
        {
            __m256i lo = _mm512_cvtps_ph((Avx512f::Load<align, mask>(src + 0, srcTails[0])), 0);
            __m256i hi = _mm512_cvtps_ph((Avx512f::Load<align, mask>(src + F, srcTails[1])), 0);
            Store<align, mask>(dst, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1), dstTail);
        }

        template<bool align> SIMD_INLINE void Float32ToFloat16x2(const float * src, uint16_t * dst)
        {
            Store<align>(dst + 0 * HA, _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtps_ph(Avx512f::Load<align>(src + 0 * F), 0)), _mm512_cvtps_ph(Avx512f::Load<align>(src + 1 * F), 0), 1));
            Store<align>(dst + 1 * HA, _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtps_ph(Avx512f::Load<align>(src + 2 * F), 0)), _mm512_cvtps_ph(Avx512f::Load<align>(src + 3 * F), 0), 1));
        }

        template <bool align> void Float32ToFloat16(const float * src, size_t size, uint16_t * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            size_t alignedSize = Simd::AlignLo(size, DF);
            __mmask16 srcTailMasks[2];
            for (size_t c = 0; c < 2; ++c)
                srcTailMasks[c] = TailMask16(size - alignedSize - F*c);
            __mmask32 dstTailMask = TailMask32(size - alignedSize);

            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
                Float32ToFloat16x2<align>(src + i, dst + i);
            for (; i < alignedSize; i += DF)
                Float32ToFloat16<align, false>(src + i, dst + i, srcTailMasks, dstTailMask);
            if (i < size)
                Float32ToFloat16<align, true>(src + i, dst + i, srcTailMasks, dstTailMask);
        }

        void Float32ToFloat16(const float * src, size_t size, uint16_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Float32ToFloat16<true>(src, size, dst);
            else
                Float32ToFloat16<false>(src, size, dst);
        }

        template<bool align, bool mask> SIMD_INLINE void Float16ToFloat32(const uint16_t * src, float * dst, __mmask32 srcTail, const __mmask16 * dstTails)
        {
            __m512i _src = Load<align, mask>(src, srcTail);
            Avx512f::Store<align, mask>(dst + 0, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_src, 0)), dstTails[0]);
            Avx512f::Store<align, mask>(dst + F, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_src, 1)), dstTails[1]);
        }

        template<bool align> SIMD_INLINE void Float16ToFloat32x2(const uint16_t * src, float * dst)
        {
#if defined(_MSC_VER)
            const __m512i src0 = Load<align>(src + 00);
            Avx512f::Store<align>(dst + 0 * F, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(src0, 0)));
            Avx512f::Store<align>(dst + 1 * F, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(src0, 1)));
            const __m512i src1 = Load<align>(src + HA);
            Avx512f::Store<align>(dst + 2 * F, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(src1, 0)));
            Avx512f::Store<align>(dst + 3 * F, _mm512_cvtph_ps(_mm512_extracti64x4_epi64(src1, 1)));
#else
            Avx512f::Store<align>(dst + 0 * F, _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)src + 0)));
            Avx512f::Store<align>(dst + 1 * F, _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)src + 1)));
            Avx512f::Store<align>(dst + 2 * F, _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)src + 2)));
            Avx512f::Store<align>(dst + 3 * F, _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)src + 3)));
#endif
        }

        template <bool align> void Float16ToFloat32(const uint16_t * src, size_t size, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            size_t alignedSize = Simd::AlignLo(size, DF);
            __mmask32 srcTailMask = TailMask32(size - alignedSize);
            __mmask16 dstTailMasks[2];
            for (size_t c = 0; c < 2; ++c)
                dstTailMasks[c] = TailMask16(size - alignedSize - F*c);

            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
                Float16ToFloat32x2<align>(src + i, dst + i);
            for (; i < alignedSize; i += DF)
                Float16ToFloat32<align, false>(src + i, dst + i, srcTailMask, dstTailMasks);
            if (i < size)
                Float16ToFloat32<align, true>(src + i, dst + i, srcTailMask, dstTailMasks);
        }

        void Float16ToFloat32(const uint16_t * src, size_t size, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Float16ToFloat32<true>(src, size, dst);
            else
                Float16ToFloat32<false>(src, size, dst);
        }

        template <int part> SIMD_INLINE void SquaredDifferenceSum16f(const __m512i & a, const __m512i & b, __m512 * sums)
        {
            __m512 _a = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(a, part));
            __m512 _b = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(b, part));
            __m512 _d = _mm512_sub_ps(_a, _b);
            sums[part] = _mm512_fmadd_ps(_d, _d, sums[part]);
        }

        template <bool align, bool mask> SIMD_INLINE void SquaredDifferenceSum16f2(const uint16_t * a, const uint16_t * b, __m512 * sums, __mmask32 tail = -1)
        {
            __m512i a0 = Load<align, mask>(a, tail);
            __m512i b0 = Load<align, mask>(b, tail);
            SquaredDifferenceSum16f<0>(a0, b0, sums);
            SquaredDifferenceSum16f<1>(a0, b0, sums);
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceSum16f4(const uint16_t * a, const uint16_t * b, __m512 * sums)
        {
#if defined(_MSC_VER)
            __m512i a0 = Load<align>(a + 00);
            __m512i b0 = Load<align>(b + 00);
            SquaredDifferenceSum16f<0>(a0, b0, sums);
            SquaredDifferenceSum16f<1>(a0, b0, sums);
            __m512i a1 = Load<align>(a + HA);
            __m512i b1 = Load<align>(b + HA);
            SquaredDifferenceSum16f<0>(a1, b1, sums);
            SquaredDifferenceSum16f<1>(a1, b1, sums);
#else
            __m512 a0 = _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)a + 0));
            __m512 b0 = _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)b + 0));
            __m512 d0 = _mm512_sub_ps(a0, b0);
            sums[0] = _mm512_fmadd_ps(d0, d0, sums[0]);

            __m512 a1 = _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)a + 1));
            __m512 b1 = _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)b + 1));
            __m512 d1 = _mm512_sub_ps(a1, b1);
            sums[1] = _mm512_fmadd_ps(d1, d1, sums[1]);

            __m512 a2 = _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)a + 2));
            __m512 b2 = _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)b + 2));
            __m512 d2 = _mm512_sub_ps(a2, b2);
            sums[0] = _mm512_fmadd_ps(d2, d2, sums[0]);

            __m512 a3 = _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)a + 3));
            __m512 b3 = _mm512_cvtph_ps(Avx2::Load<align>((__m256i*)b + 3));
            __m512 d3 = _mm512_sub_ps(a3, b3);
            sums[1] = _mm512_fmadd_ps(d3, d3, sums[1]);
#endif
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            size_t alignedSize = AlignLo(size, DF);
            __mmask32 tailMask = TailMask32(size - alignedSize);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            __m512 sums[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
            for (; i < fullAlignedSize; i += QF)
                SquaredDifferenceSum16f4<align>(a + i, b + i, sums);
            for (; i < alignedSize; i += DF)
                SquaredDifferenceSum16f2<align, false>(a + i, b + i, sums);
            if (i < size)
                SquaredDifferenceSum16f2<align, true>(a + i, b + i, sums, tailMask);
            sums[0] = _mm512_add_ps(sums[0], sums[1]);
            *sum = Avx512f::ExtractSum(sums[0]);
        }

        void SquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum)
        {
            if (Aligned(a) && Aligned(b))
                SquaredDifferenceSum16f<true>(a, b, size, sum);
            else
                SquaredDifferenceSum16f<false>(a, b, size, sum);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
