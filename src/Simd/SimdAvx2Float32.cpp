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
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> SIMD_INLINE __m256i Float32ToUint8(const float * src, const __m256 & lower, const __m256 & upper, const __m256 & boost)
        {
            return _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(_mm256_min_ps(_mm256_max_ps(Avx::Load<align>(src), lower), upper), lower), boost));
        }

        template <bool align> SIMD_INLINE void Float32ToUint8(const float * src, const __m256 & lower, const __m256 & upper, const __m256 & boost, uint8_t * dst)
        {
            __m256i d0 = Float32ToUint8<align>(src + F * 0, lower, upper, boost);
            __m256i d1 = Float32ToUint8<align>(src + F * 1, lower, upper, boost);
            __m256i d2 = Float32ToUint8<align>(src + F * 2, lower, upper, boost);
            __m256i d3 = Float32ToUint8<align>(src + F * 3, lower, upper, boost);
            Store<align>((__m256i*)dst, PackU16ToU8(PackU32ToI16(d0, d1), PackU32ToI16(d2, d3)));
        }

        template <bool align> void Float32ToUint8(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst)
        {
            assert(size >= A);
            if (align)
                assert(Aligned(src) && Aligned(dst));

            __m256 _lower = _mm256_set1_ps(lower[0]);
            __m256 _upper = _mm256_set1_ps(upper[0]);
            __m256 boost = _mm256_set1_ps(255.0f / (upper[0] - lower[0]));

            size_t alignedSize = AlignLo(size, A);
            for (size_t i = 0; i < alignedSize; i += A)
                Float32ToUint8<align>(src + i, _lower, _upper, boost, dst + i);
            if (alignedSize != size)
                Float32ToUint8<false>(src + size - A, _lower, _upper, boost, dst + size - A);
        }

        void Float32ToUint8(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Float32ToUint8<true>(src, size, lower, upper, dst);
            else
                Float32ToUint8<false>(src, size, lower, upper, dst);
        }

        SIMD_INLINE __m256 Uint8ToFloat32(const __m128i & value, const __m256 & lower, const __m256 & boost)
        {
            return _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(value)), boost), lower);
        }

        template <bool align> SIMD_INLINE void Uint8ToFloat32(const uint8_t * src, const __m256 & lower, const __m256 & boost, float * dst)
        {
            __m128i _src = Sse2::Load<align>((__m128i*)src);
            Avx::Store<align>(dst + 0, Uint8ToFloat32(_src, lower, boost));
            Avx::Store<align>(dst + F, Uint8ToFloat32(_mm_srli_si128(_src, 8), lower, boost));
        }

        template <bool align> void Uint8ToFloat32(const uint8_t * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            assert(size >= HA);
            if (align)
                assert(Aligned(src) && Aligned(dst));

            __m256 _lower = _mm256_set1_ps(lower[0]);
            __m256 boost = _mm256_set1_ps((upper[0] - lower[0]) / 255.0f);

            size_t alignedSize = AlignLo(size, HA);
            for (size_t i = 0; i < alignedSize; i += HA)
                Uint8ToFloat32<align>(src + i, _lower, boost, dst + i);
            if (alignedSize != size)
                Uint8ToFloat32<false>(src + size - HA, _lower, boost, dst + size - HA);
        }

        void Uint8ToFloat32(const uint8_t * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Uint8ToFloat32<true>(src, size, lower, upper, dst);
            else
                Uint8ToFloat32<false>(src, size, lower, upper, dst);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
