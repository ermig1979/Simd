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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    namespace Neon
    {
        template<bool align> SIMD_INLINE void Float32ToFloat16(const float * src, uint16_t * dst)
        {
            Store<align>(dst, (uint16x4_t)vcvt_f16_f32(Load<align>(src)));
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
            Store<align>(dst, vcvt_f32_f16((float16x4_t)LoadHalf<align>(src)));
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

        template <bool align> SIMD_INLINE void SquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t offset, float32x4_t & sum)
        {
            float32x4_t _a = vcvt_f32_f16((float16x4_t)LoadHalf<align>(a + offset));
            float32x4_t _b = vcvt_f32_f16((float16x4_t)LoadHalf<align>(b + offset));
            float32x4_t _d = vsubq_f32(_a, _b);
            sum = vmlaq_f32(sum, _d, _d);
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum)
        {
            assert(size >= F);
            if (align)
                assert(Aligned(a) && Aligned(b));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, DF);
            size_t i = 0;
            float32x4_t sums[2] = { vdupq_n_f32(0), vdupq_n_f32(0) };
            if (fullAlignedSize)
            {
                for (; i < fullAlignedSize; i += DF)
                {
                    SquaredDifferenceSum16f<align>(a, b, i + F * 0, sums[0]);
                    SquaredDifferenceSum16f<align>(a, b, i + F * 1, sums[1]);
                }
                sums[0] = vaddq_f32(sums[0], sums[1]);
            }
            for (; i < partialAlignedSize; i += F)
                SquaredDifferenceSum16f<align>(a, b, i, sums[0]);
            if (partialAlignedSize != size)
            {
                float32x4_t tailMask = RightNotZero(size - partialAlignedSize);
                float32x4_t _a = vcvt_f32_f16((float16x4_t)LoadHalf<align>(a + size - F));
                float32x4_t _b = vcvt_f32_f16((float16x4_t)LoadHalf<align>(a + size - F));
                float32x4_t _d = And(vsubq_f32(_a, _b), tailMask);
                sums[0] = vaddq_f32(sums[0], vmulq_f32(_d, _d));
            }
            *sum = ExtractSum32f(sums[0]);
        }

        void SquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum)
        {
            if (Aligned(a) && Aligned(b))
                SquaredDifferenceSum16f<true>(a, b, size, sum);
            else
                SquaredDifferenceSum16f<false>(a, b, size, sum);
        }
    }
#endif // defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
}
