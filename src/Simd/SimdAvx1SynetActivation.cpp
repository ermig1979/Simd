/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        template<bool align> SIMD_INLINE void SynetHswish32f(const float * src, __m256 shift, __m256 scale, float * dst, size_t offset)
        {
            __m256 value = Load<align>(src + offset);
            Store<align>(dst + offset, _mm256_mul_ps(_mm256_mul_ps(_mm256_max_ps(_mm256_add_ps(_mm256_min_ps(value, shift), shift), _mm256_setzero_ps()), scale), value));
        }

        template<bool align> void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            __m256 _shift = _mm256_set1_ps(shift[0]);
            __m256 _scale = _mm256_set1_ps(scale[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 0 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 1 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 2 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHswish32f<align>(src, _shift, _scale, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetHswish32f(src[i], shift[0], scale[0]);
        }

        void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetHswish32f<true>(src, size, shift, scale, dst);
            else
                SynetHswish32f<false>(src, size, shift, scale, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            assert(lower[0] <= upper[0]);
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float min = *lower;
            float max = *upper;
            __m256 _min = _mm256_set1_ps(min);
            __m256 _max = _mm256_set1_ps(max);
            size_t sizeF = Simd::AlignLo(size, F);
            size_t sizeQF = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Store<align>(dst + i + 0 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 0 * F)), _max));
                Store<align>(dst + i + 1 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 1 * F)), _max));
                Store<align>(dst + i + 2 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 2 * F)), _max));
                Store<align>(dst + i + 3 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 3 * F)), _max));
            }
            for (; i < sizeF; i += F)
                Store<align>(dst + i, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i)), _max));
            for (; i < size; ++i)
                dst[i] = Simd::RestrictRange(src[i], min, max);
        }

        void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetRestrictRange32f<true>(src, size, lower, upper, dst);
            else
                SynetRestrictRange32f<false>(src, size, lower, upper, dst);
        }
    }
#endif// SIMD_AVX_ENABLE
}
