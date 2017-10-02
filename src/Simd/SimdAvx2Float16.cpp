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
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template<bool align> SIMD_INLINE void Float32ToFloat16(const float * src, uint16_t * dst)
        {
            Sse2::Store<align>((__m128i*)dst, _mm256_cvtps_ph(Avx::Load<align>(src), 0));
        }

        template <bool align> void Float32ToFloat16(const float * src, size_t size, uint16_t * dst)
        {
            assert(size >= F);
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            size_t partialAlignedSize = Simd::AlignLo(size, F);

            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                Float32ToFloat16<align>(src + i + F * 0, dst + i + F * 0);
                Float32ToFloat16<align>(src + i + F * 1, dst + i + F * 1);
                Float32ToFloat16<align>(src + i + F * 2, dst + i + F * 2);
                Float32ToFloat16<align>(src + i + F * 3, dst + i + F * 3);
            }
            for (; i < partialAlignedSize; i += F)
                Float32ToFloat16<align>(src + i, dst + i);
            if (partialAlignedSize != size)
                Float32ToFloat16<false>(src + size - F, dst + size - F);
        }

        void Float32ToFloat16(const float * src, size_t size, uint16_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Float32ToFloat16<true>(src, size, dst);
            else
                Float32ToFloat16<false>(src, size, dst);
        }

        template<bool align> SIMD_INLINE void Float16ToFloat32(const uint16_t * src, float * dst)
        {
            Avx::Store<align>(dst, _mm256_cvtph_ps(Sse2::Load<align>((__m128i*)src)));
        }

        template <bool align> void Float16ToFloat32(const uint16_t * src, size_t size, float * dst)
        {
            assert(size >= F);
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            size_t partialAlignedSize = Simd::AlignLo(size, F);

            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                Float16ToFloat32<align>(src + i + F * 0, dst + i + F * 0);
                Float16ToFloat32<align>(src + i + F * 1, dst + i + F * 1);
                Float16ToFloat32<align>(src + i + F * 2, dst + i + F * 2);
                Float16ToFloat32<align>(src + i + F * 3, dst + i + F * 3);
            }
            for (; i < partialAlignedSize; i += F)
                Float16ToFloat32<align>(src + i, dst + i);
            if (partialAlignedSize != size)
                Float16ToFloat32<false>(src + size - F, dst + size - F);
        }

        void Float16ToFloat32(const uint16_t * src, size_t size, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Float16ToFloat32<true>(src, size, dst);
            else
                Float16ToFloat32<false>(src, size, dst);
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t offset, __m256 & sum)
        {
            __m256 _a = _mm256_cvtph_ps(Sse2::Load<align>((__m128i*)(a + offset)));
            __m256 _b = _mm256_cvtph_ps(Sse2::Load<align>((__m128i*)(b + offset)));
            __m256 _d = _mm256_sub_ps(_a, _b);
            sum = _mm256_fmadd_ps(_d, _d, sum);
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum)
        {
            assert(size >= F);
            if (align)
                assert(Aligned(a) && Aligned(b));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
            if (fullAlignedSize)
            {
                for (; i < fullAlignedSize; i += QF)
                {
                    SquaredDifferenceSum16f<align>(a, b, i + F * 0, sums[0]);
                    SquaredDifferenceSum16f<align>(a, b, i + F * 1, sums[1]);
                    SquaredDifferenceSum16f<align>(a, b, i + F * 2, sums[2]);
                    SquaredDifferenceSum16f<align>(a, b, i + F * 3, sums[3]);
                }
                sums[0] = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
            }
            for (; i < partialAlignedSize; i += F)
                SquaredDifferenceSum16f<align>(a, b, i, sums[0]);
            if (partialAlignedSize != size)
            {
                __m256 mask = RightNotZero(size - partialAlignedSize);
                __m256 _a = _mm256_cvtph_ps(Sse2::Load<false>((__m128i*)(a + size - F)));
                __m256 _b = _mm256_cvtph_ps(Sse2::Load<false>((__m128i*)(b + size - F)));
                __m256 _d = _mm256_and_ps(_mm256_sub_ps(_a, _b), mask);
                sums[0] = _mm256_fmadd_ps(_d, _d, sums[0]);
            }
            *sum = Avx::ExtractSum(sums[0]);
        }

        void SquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum)
        {
            if (Aligned(a) && Aligned(b))
                SquaredDifferenceSum16f<true>(a, b, size, sum);
            else
                SquaredDifferenceSum16f<false>(a, b, size, sum);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
