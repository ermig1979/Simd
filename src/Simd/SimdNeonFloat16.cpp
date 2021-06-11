/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdCpu.h"

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
                float32x4_t tailMask = RightNotZero32f(size - partialAlignedSize);
                float32x4_t _a = vcvt_f32_f16((float16x4_t)LoadHalf<false>(a + size - F));
                float32x4_t _b = vcvt_f32_f16((float16x4_t)LoadHalf<false>(a + size - F));
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

        template<bool align> void CosineDistance16f(const uint16_t * a, const uint16_t * b, size_t size, float * distance)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, DF);
            size_t i = 0;
            float32x4_t _aa[2] = { vdupq_n_f32(0), vdupq_n_f32(0) };
            float32x4_t _ab[2] = { vdupq_n_f32(0), vdupq_n_f32(0) };
            float32x4_t _bb[2] = { vdupq_n_f32(0), vdupq_n_f32(0) };
            if (fullAlignedSize)
            {
                for (; i < fullAlignedSize; i += DF)
                {
                    float32x4_t a0 = vcvt_f32_f16((float16x4_t)LoadHalf<align>(a + i + 0));
                    float32x4_t b0 = vcvt_f32_f16((float16x4_t)LoadHalf<align>(b + i + 0));
                    _aa[0] = vmlaq_f32(_aa[0], a0, a0);
                    _ab[0] = vmlaq_f32(_ab[0], a0, b0);
                    _bb[0] = vmlaq_f32(_bb[0], b0, b0);
                    float32x4_t a1 = vcvt_f32_f16((float16x4_t)LoadHalf<align>(a + i + F));
                    float32x4_t b1 = vcvt_f32_f16((float16x4_t)LoadHalf<align>(b + i + F));
                    _aa[1] = vmlaq_f32(_aa[1], a1, a1);
                    _ab[1] = vmlaq_f32(_ab[1], a1, b1);
                    _bb[1] = vmlaq_f32(_bb[1], b1, b1);
                }
                _aa[0] = vaddq_f32(_aa[0], _aa[1]);
                _ab[0] = vaddq_f32(_ab[0], _ab[1]);
                _bb[0] = vaddq_f32(_bb[0], _bb[1]);
            }
            for (; i < partialAlignedSize; i += F)
            {
                float32x4_t a0 = vcvt_f32_f16((float16x4_t)LoadHalf<align>(a + i + 0));
                float32x4_t b0 = vcvt_f32_f16((float16x4_t)LoadHalf<align>(b + i + 0));
                _aa[0] = vmlaq_f32(_aa[0], a0, a0);
                _ab[0] = vmlaq_f32(_ab[0], a0, b0);
                _bb[0] = vmlaq_f32(_bb[0], b0, b0);
            }
            if (partialAlignedSize != size)
            {
                float32x4_t tailMask = RightNotZero32f(size - partialAlignedSize);
                float32x4_t a0 = And(vcvt_f32_f16((float16x4_t)LoadHalf<align>(a + i + 0)), tailMask);
                float32x4_t b0 = And(vcvt_f32_f16((float16x4_t)LoadHalf<align>(b + i + 0)), tailMask);
                _aa[0] = vmlaq_f32(_aa[0], a0, a0);
                _ab[0] = vmlaq_f32(_ab[0], a0, b0);
                _bb[0] = vmlaq_f32(_bb[0], b0, b0);
            }
            float aa = ExtractSum32f(_aa[0]), ab = ExtractSum32f(_ab[0]), bb = ExtractSum32f(_bb[0]);
            *distance = 1.0f - ab / ::sqrt(aa*bb);
        }

        void CosineDistance16f(const uint16_t * a, const uint16_t * b, size_t size, float * distance)
        {
            if (Aligned(a) && Aligned(b))
                CosineDistance16f<true>(a, b, size, distance);
            else
                CosineDistance16f<false>(a, b, size, distance);
        }

        SIMD_INLINE float32x4_t Tail(size_t tail)
        {
            const int32_t mask[DF] = { 0, 0, 0, 0, -1, -1, -1, -1 };
            return Load<false>((float*)(mask + tail));
        }

        static void Squares(size_t M, size_t K, const uint16_t * const * A, float * squares)
        {
            size_t M4 = AlignLo(M, 4);
            size_t KF = AlignLo(K, F);
            float32x4_t mask = Tail(K - KF);
            size_t i = 0;
            for (; i < M4; i += 4)
            {
                float32x4_t sums[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
                for (size_t k = 0; k < KF; k += F)
                {
                    float32x4_t a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[i + 0] + k)));
                    float32x4_t a1 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[i + 1] + k)));
                    float32x4_t a2 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[i + 2] + k)));
                    float32x4_t a3 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[i + 3] + k)));
                    sums[0] = vmlaq_f32(sums[0], a0, a0);
                    sums[1] = vmlaq_f32(sums[1], a1, a1);
                    sums[2] = vmlaq_f32(sums[2], a2, a2);
                    sums[3] = vmlaq_f32(sums[3], a3, a3);
                }
                if (KF < K)
                {
                    size_t k = K - F;
                    float32x4_t a0 = And(mask, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[i + 0] + k))));
                    float32x4_t a1 = And(mask, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[i + 1] + k))));
                    float32x4_t a2 = And(mask, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[i + 2] + k))));
                    float32x4_t a3 = And(mask, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[i + 3] + k))));
                    sums[0] = vmlaq_f32(sums[0], a0, a0);
                    sums[1] = vmlaq_f32(sums[1], a1, a1);
                    sums[2] = vmlaq_f32(sums[2], a2, a2);
                    sums[3] = vmlaq_f32(sums[3], a3, a3);
                }
                Store<false>(squares + i, Extract4Sums(sums));
            }
            for (; i < M; i += 1)
            {
                float32x4_t sum = vdupq_n_f32(0.0f);
                for (size_t k = 0; k < KF; k += F)
                {
                    float32x4_t a = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[i] + k)));
                    sum = vmlaq_f32(sum, a, a);
                }
                if (KF < K)
                {
                    size_t k = K - F;
                    float32x4_t a = And(mask, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[i] + k))));
                    sum = vmlaq_f32(sum, a, a);
                }
                squares[i] = ExtractSum32f(sum);
            }
        }

#if defined(SIMD_ARM64_ENABLE)
        static void MicroCosineDistances6x4(size_t K, const uint16_t* const* A, const uint16_t* const* B, const float* aa, const float* bb, float* distances, size_t stride)
        {
            size_t K4 = K & (~3);
            float32x4_t c00 = vdupq_n_f32(0.0f);
            float32x4_t c01 = vdupq_n_f32(0.0f);
            float32x4_t c02 = vdupq_n_f32(0.0f);
            float32x4_t c03 = vdupq_n_f32(0.0f);
            float32x4_t c10 = vdupq_n_f32(0.0f);
            float32x4_t c11 = vdupq_n_f32(0.0f);
            float32x4_t c12 = vdupq_n_f32(0.0f);
            float32x4_t c13 = vdupq_n_f32(0.0f);
            float32x4_t c20 = vdupq_n_f32(0.0f);
            float32x4_t c21 = vdupq_n_f32(0.0f);
            float32x4_t c22 = vdupq_n_f32(0.0f);
            float32x4_t c23 = vdupq_n_f32(0.0f);
            float32x4_t c30 = vdupq_n_f32(0.0f);
            float32x4_t c31 = vdupq_n_f32(0.0f);
            float32x4_t c32 = vdupq_n_f32(0.0f);
            float32x4_t c33 = vdupq_n_f32(0.0f);
            float32x4_t c40 = vdupq_n_f32(0.0f);
            float32x4_t c41 = vdupq_n_f32(0.0f);
            float32x4_t c42 = vdupq_n_f32(0.0f);
            float32x4_t c43 = vdupq_n_f32(0.0f);
            float32x4_t c50 = vdupq_n_f32(0.0f);
            float32x4_t c51 = vdupq_n_f32(0.0f);
            float32x4_t c52 = vdupq_n_f32(0.0f);
            float32x4_t c53 = vdupq_n_f32(0.0f);
            float32x4_t a0, a1, a2, a3, a4, a5, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k)));
                a1 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[1] + k)));
                a2 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[2] + k)));
                a3 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[3] + k)));
                a4 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[4] + k)));
                a5 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[5] + k)));
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                c30 = vmlaq_f32(c30, a3, b0);
                c40 = vmlaq_f32(c40, a4, b0);
                c50 = vmlaq_f32(c50, a5, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[1] + k)));
                c01 = vmlaq_f32(c01, a0, b0);
                c11 = vmlaq_f32(c11, a1, b0);
                c21 = vmlaq_f32(c21, a2, b0);
                c31 = vmlaq_f32(c31, a3, b0);
                c41 = vmlaq_f32(c41, a4, b0);
                c51 = vmlaq_f32(c51, a5, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[2] + k)));
                c02 = vmlaq_f32(c02, a0, b0);
                c12 = vmlaq_f32(c12, a1, b0);
                c22 = vmlaq_f32(c22, a2, b0);
                c32 = vmlaq_f32(c32, a3, b0);
                c42 = vmlaq_f32(c42, a4, b0);
                c52 = vmlaq_f32(c52, a5, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[3] + k)));
                c03 = vmlaq_f32(c03, a0, b0);
                c13 = vmlaq_f32(c13, a1, b0);
                c23 = vmlaq_f32(c23, a2, b0);
                c33 = vmlaq_f32(c33, a3, b0);
                c43 = vmlaq_f32(c43, a4, b0);
                c53 = vmlaq_f32(c53, a5, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k))));
                a1 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[1] + k))));
                a2 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[2] + k))));
                a3 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[3] + k))));
                a4 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[4] + k))));
                a5 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[5] + k))));
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                c30 = vmlaq_f32(c30, a3, b0);
                c40 = vmlaq_f32(c40, a4, b0);
                c50 = vmlaq_f32(c50, a5, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[1] + k)));
                c01 = vmlaq_f32(c01, a0, b0);
                c11 = vmlaq_f32(c11, a1, b0);
                c21 = vmlaq_f32(c21, a2, b0);
                c31 = vmlaq_f32(c31, a3, b0);
                c41 = vmlaq_f32(c41, a4, b0);
                c51 = vmlaq_f32(c51, a5, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[2] + k)));
                c02 = vmlaq_f32(c02, a0, b0);
                c12 = vmlaq_f32(c12, a1, b0);
                c22 = vmlaq_f32(c22, a2, b0);
                c32 = vmlaq_f32(c32, a3, b0);
                c42 = vmlaq_f32(c42, a4, b0);
                c52 = vmlaq_f32(c52, a5, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[3] + k)));
                c03 = vmlaq_f32(c03, a0, b0);
                c13 = vmlaq_f32(c13, a1, b0);
                c23 = vmlaq_f32(c23, a2, b0);
                c33 = vmlaq_f32(c33, a3, b0);
                c43 = vmlaq_f32(c43, a4, b0);
                c53 = vmlaq_f32(c53, a5, b0);
            }
            float32x4_t _bb = Load<false>(bb);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            Store<false>(distances + 0 * stride, vmlsq_f32(_1, ReciprocalSqrt<1>(vmulq_f32(_bb, vdupq_n_f32(aa[0]))), Extract4Sums(c00, c01, c02, c03)));
            Store<false>(distances + 1 * stride, vmlsq_f32(_1, ReciprocalSqrt<1>(vmulq_f32(_bb, vdupq_n_f32(aa[1]))), Extract4Sums(c10, c11, c12, c13)));
            Store<false>(distances + 2 * stride, vmlsq_f32(_1, ReciprocalSqrt<1>(vmulq_f32(_bb, vdupq_n_f32(aa[2]))), Extract4Sums(c20, c21, c22, c23)));
            Store<false>(distances + 3 * stride, vmlsq_f32(_1, ReciprocalSqrt<1>(vmulq_f32(_bb, vdupq_n_f32(aa[3]))), Extract4Sums(c30, c31, c32, c33)));
            Store<false>(distances + 4 * stride, vmlsq_f32(_1, ReciprocalSqrt<1>(vmulq_f32(_bb, vdupq_n_f32(aa[4]))), Extract4Sums(c40, c41, c42, c43)));
            Store<false>(distances + 5 * stride, vmlsq_f32(_1, ReciprocalSqrt<1>(vmulq_f32(_bb, vdupq_n_f32(aa[5]))), Extract4Sums(c50, c51, c52, c53)));
        }

        static void MicroCosineDistances6x1(size_t K, const uint16_t* const* A, const uint16_t* const* B, const float* aa, const float* bb, float* distances, size_t stride)
        {
            size_t K4 = K & (~3);
            float32x4_t c00 = vdupq_n_f32(0.0f);
            float32x4_t c10 = vdupq_n_f32(0.0f);
            float32x4_t c20 = vdupq_n_f32(0.0f);
            float32x4_t c30 = vdupq_n_f32(0.0f);
            float32x4_t c40 = vdupq_n_f32(0.0f);
            float32x4_t c50 = vdupq_n_f32(0.0f);
            float32x4_t a0, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k)));
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[1] + k)));
                c10 = vmlaq_f32(c10, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[2] + k)));
                c20 = vmlaq_f32(c20, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[3] + k)));
                c30 = vmlaq_f32(c30, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[4] + k)));
                c40 = vmlaq_f32(c40, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[5] + k)));
                c50 = vmlaq_f32(c50, a0, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                b0 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k))));
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[1] + k)));
                c10 = vmlaq_f32(c10, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[2] + k)));
                c20 = vmlaq_f32(c20, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[3] + k)));
                c30 = vmlaq_f32(c30, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[4] + k)));
                c40 = vmlaq_f32(c40, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[5] + k)));
                c50 = vmlaq_f32(c50, a0, b0);
            }
            distances[0 * stride] = 1.0f - ExtractSum32f(c00) / sqrt(bb[0] * aa[0]);
            distances[1 * stride] = 1.0f - ExtractSum32f(c10) / sqrt(bb[0] * aa[1]);
            distances[2 * stride] = 1.0f - ExtractSum32f(c20) / sqrt(bb[0] * aa[2]);
            distances[3 * stride] = 1.0f - ExtractSum32f(c30) / sqrt(bb[0] * aa[3]);
            distances[4 * stride] = 1.0f - ExtractSum32f(c40) / sqrt(bb[0] * aa[4]);
            distances[5 * stride] = 1.0f - ExtractSum32f(c50) / sqrt(bb[0] * aa[5]);
        }
#else
        static void MicroCosineDistances3x4(size_t K, const uint16_t * const * A, const uint16_t * const * B, const float * aa, const float * bb, float * distances, size_t stride)
        {
            size_t K4 = K & (~3);
            float32x4_t c00 = vdupq_n_f32(0.0f);
            float32x4_t c01 = vdupq_n_f32(0.0f);
            float32x4_t c02 = vdupq_n_f32(0.0f);
            float32x4_t c03 = vdupq_n_f32(0.0f);
            float32x4_t c10 = vdupq_n_f32(0.0f);
            float32x4_t c11 = vdupq_n_f32(0.0f);
            float32x4_t c12 = vdupq_n_f32(0.0f);
            float32x4_t c13 = vdupq_n_f32(0.0f);
            float32x4_t c20 = vdupq_n_f32(0.0f);
            float32x4_t c21 = vdupq_n_f32(0.0f);
            float32x4_t c22 = vdupq_n_f32(0.0f);
            float32x4_t c23 = vdupq_n_f32(0.0f);
            float32x4_t a0, a1, a2, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k)));
                a1 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[1] + k)));
                a2 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[2] + k)));
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[1] + k)));
                c01 = vmlaq_f32(c01, a0, b0);
                c11 = vmlaq_f32(c11, a1, b0);
                c21 = vmlaq_f32(c21, a2, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[2] + k)));
                c02 = vmlaq_f32(c02, a0, b0);
                c12 = vmlaq_f32(c12, a1, b0);
                c22 = vmlaq_f32(c22, a2, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[3] + k)));
                c03 = vmlaq_f32(c03, a0, b0);
                c13 = vmlaq_f32(c13, a1, b0);
                c23 = vmlaq_f32(c23, a2, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k))));
                a1 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[1] + k))));
                a2 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[2] + k))));
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
                c10 = vmlaq_f32(c10, a1, b0);
                c20 = vmlaq_f32(c20, a2, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[1] + k)));
                c01 = vmlaq_f32(c01, a0, b0);
                c11 = vmlaq_f32(c11, a1, b0);
                c21 = vmlaq_f32(c21, a2, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[2] + k)));
                c02 = vmlaq_f32(c02, a0, b0);
                c12 = vmlaq_f32(c12, a1, b0);
                c22 = vmlaq_f32(c22, a2, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[3] + k)));
                c03 = vmlaq_f32(c03, a0, b0);
                c13 = vmlaq_f32(c13, a1, b0);
                c23 = vmlaq_f32(c23, a2, b0);
            }
            float32x4_t _bb = Load<false>(bb);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            Store<false>(distances + 0 * stride, vmlsq_f32(_1, ReciprocalSqrt<1>(vmulq_f32(_bb, vdupq_n_f32(aa[0]))), Extract4Sums(c00, c01, c02, c03)));
            Store<false>(distances + 1 * stride, vmlsq_f32(_1, ReciprocalSqrt<1>(vmulq_f32(_bb, vdupq_n_f32(aa[1]))), Extract4Sums(c10, c11, c12, c13)));
            Store<false>(distances + 2 * stride, vmlsq_f32(_1, ReciprocalSqrt<1>(vmulq_f32(_bb, vdupq_n_f32(aa[2]))), Extract4Sums(c20, c21, c22, c23)));
        }

        static void MicroCosineDistances3x1(size_t K, const uint16_t * const * A, const uint16_t * const * B, const float * aa, const float * bb, float * distances, size_t stride)
        {
            size_t K4 = K & (~3);
            float32x4_t c00 = vdupq_n_f32(0.0f);
            float32x4_t c10 = vdupq_n_f32(0.0f);
            float32x4_t c20 = vdupq_n_f32(0.0f);
            float32x4_t a0, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k)));
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[1] + k)));
                c10 = vmlaq_f32(c10, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[2] + k)));
                c20 = vmlaq_f32(c20, a0, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                b0 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k))));
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[1] + k)));
                c10 = vmlaq_f32(c10, a0, b0);
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[2] + k)));
                c20 = vmlaq_f32(c20, a0, b0);
            }
            distances[0 * stride] = 1.0f - ExtractSum32f(c00) / sqrt(bb[0] * aa[0]);
            distances[1 * stride] = 1.0f - ExtractSum32f(c10) / sqrt(bb[0] * aa[1]);
            distances[2 * stride] = 1.0f - ExtractSum32f(c20) / sqrt(bb[0] * aa[2]);
        }
#endif

        static void MicroCosineDistances1x4(size_t K, const uint16_t * const * A, const uint16_t * const * B, const float * aa, const float * bb, float * distances, size_t stride)
        {
            size_t K4 = K & (~3);
            float32x4_t c00 = vdupq_n_f32(0.0f);
            float32x4_t c01 = vdupq_n_f32(0.0f);
            float32x4_t c02 = vdupq_n_f32(0.0f);
            float32x4_t c03 = vdupq_n_f32(0.0f);
            float32x4_t a0, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k)));
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[1] + k)));
                c01 = vmlaq_f32(c01, a0, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[2] + k)));
                c02 = vmlaq_f32(c02, a0, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[3] + k)));
                c03 = vmlaq_f32(c03, a0, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k))));
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[1] + k)));
                c01 = vmlaq_f32(c01, a0, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[2] + k)));
                c02 = vmlaq_f32(c02, a0, b0);
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[3] + k)));
                c03 = vmlaq_f32(c03, a0, b0);
            }
            float32x4_t _bb = Load<false>(bb);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            Store<false>(distances + 0 * stride, vmlsq_f32(_1, ReciprocalSqrt<1>(vmulq_f32(_bb, vdupq_n_f32(aa[0]))), Extract4Sums(c00, c01, c02, c03)));
        }

        static void MicroCosineDistances1x1(size_t K, const uint16_t* const* A, const uint16_t* const* B, const float* aa, const float* bb, float* distances, size_t stride)
        {
            size_t K4 = K & (~3);
            float32x4_t c00 = vdupq_n_f32(0.0f);
            float32x4_t a0, b0;
            for (size_t k = 0; k < K4; k += 4)
            {
                a0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k)));
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
            }
            if (K4 < K)
            {
                size_t k = K - 4;
                float32x4_t tail = Tail(K - K4);
                a0 = And(tail, vcvt_f32_f16((float16x4_t)LoadHalf<false>((A[0] + k))));
                b0 = vcvt_f32_f16((float16x4_t)LoadHalf<false>((B[0] + k)));
                c00 = vmlaq_f32(c00, a0, b0);
            }
            distances[0 * stride] = 1.0f - ExtractSum32f(c00) / sqrt(bb[0] * aa[0]);
        }

#if defined(SIMD_ARM64_ENABLE)
        const size_t MicroM = 6;
        static void MacroCosineDistances(size_t M, size_t N, size_t K, const uint16_t* const* A, const uint16_t* const* B, const float* aa, const float* bb, float* distances, size_t stride)
        {
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
            for (; i < M; i++)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistances1x4(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                for (; j < N; j += 1)
                    MicroCosineDistances1x1(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                distances += 1 * stride;
            }
        }
#else
        const size_t MicroM = 3;
        static void MacroCosineDistances(size_t M, size_t N, size_t K, const uint16_t * const * A, const uint16_t * const * B, const float * aa, const float * bb, float * distances, size_t stride)
        {
            size_t M3 = AlignLoAny(M, 3);
            size_t N4 = AlignLo(N, 4);
            size_t i = 0;
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
                    MicroCosineDistances1x1(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                distances += 1 * stride;
            }
        }
#endif

        void CosineDistancesMxNa16f(size_t M, size_t N, size_t K, const uint16_t* const* A, const uint16_t* const* B, float* distances)
        {
            const size_t L2 = Base::AlgCacheL2();
            size_t mN = AlignLoAny(L2 / 2 / K, 4);
            size_t mM = AlignLoAny(L2 / 2 / K, MicroM);
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
            size_t mM = AlignLoAny(L2 / 2 / K, MicroM);
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
            for (size_t j = 0; j < N; ++j)
                norms[j] = sqrt(norms[j]);
        }

        void VectorNormNp16f(size_t N, size_t K, const uint16_t* A, float* norms)
        {
            Array16ucp a(N);
            for (size_t j = 0; j < N; ++j)
                a[j] = A + j * K;
            VectorNormNa16f(N, K, a.data, norms);
        }
    }
#endif // defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
}
