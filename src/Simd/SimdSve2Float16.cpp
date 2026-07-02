/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE void Float32ToFloat16(const float* src, const svbool_t& lo, const svbool_t& hi, const svbool_t& store, uint16_t* dst)
        {
            size_t F = svlen(svfloat32_t());
            svuint16_t _lo = svreinterpret_u16_f16(svcvt_f16_f32_x(lo, svld1_f32(lo, src + 0)));
            svuint16_t _hi = svreinterpret_u16_f16(svcvt_f16_f32_x(hi, svld1_f32(hi, src + F)));
            svst1_u16(store, dst, svuzp1_u16(_lo, _hi));
        }

        void Float32ToFloat16(const float* src, size_t size, uint16_t* dst)
        {
            size_t A = svlen(svuint16_t()), F = svlen(svfloat32_t()), sizeA = AlignLo(size, A), i = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            for (; i < sizeA; i += A)
                Float32ToFloat16(src + i, body32, body32, body16, dst + i);
            if (i < size)
            {
                size_t tail = size - i;
                Float32ToFloat16(src + i, svwhilelt_b32(size_t(0), Simd::Min(tail, F)),
                    svwhilelt_b32(F, tail), svwhilelt_b16(size_t(0), tail), dst + i);
            }
        }

        SIMD_INLINE void Float16ToFloat32(const uint16_t* src, const svbool_t& load, const svbool_t& even, const svbool_t& odd,
            const svbool_t& lo, const svbool_t& hi, float* dst)
        {
            size_t F = svlen(svfloat32_t());
            svfloat16_t _src = svreinterpret_f16_u16(svld1_u16(load, src));
            svfloat32_t _even = svcvt_f32_f16_x(even, _src);
            svfloat32_t _odd = svcvtlt_f32_f16_x(odd, _src);
            svst1_f32(lo, dst + 0, svzip1_f32(_even, _odd));
            svst1_f32(hi, dst + F, svzip2_f32(_even, _odd));
        }

        void Float16ToFloat32(const uint16_t* src, size_t size, float* dst)
        {
            size_t A = svlen(svuint16_t()), F = svlen(svfloat32_t()), sizeA = AlignLo(size, A), i = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            for (; i < sizeA; i += A)
                Float16ToFloat32(src + i, body16, body32, body32, body32, body32, dst + i);
            if (i < size)
            {
                size_t tail = size - i;
                Float16ToFloat32(src + i, svwhilelt_b16(size_t(0), tail),
                    svwhilelt_b32(size_t(0), (tail + 1) / 2), svwhilelt_b32(size_t(0), tail / 2),
                    svwhilelt_b32(size_t(0), Simd::Min(tail, F)), svwhilelt_b32(F, tail), dst + i);
            }
        }

        SIMD_INLINE void SquaredDifferenceSum16f(const uint16_t* a, const uint16_t* b, const svbool_t& load, const svbool_t& lo,
            const svbool_t& hi, svfloat32_t& sum0, svfloat32_t& sum1)
        {
            svfloat16_t _a = svreinterpret_f16_u16(svld1_u16(load, a));
            svfloat16_t _b = svreinterpret_f16_u16(svld1_u16(load, b));
            svfloat32_t d0 = svsub_f32_x(lo, svcvt_f32_f16_x(lo, _a), svcvt_f32_f16_x(lo, _b));
            svfloat32_t d1 = svsub_f32_x(hi, svcvtlt_f32_f16_x(hi, _a), svcvtlt_f32_f16_x(hi, _b));
            sum0 = svmla_f32_x(lo, sum0, d0, d0);
            sum1 = svmla_f32_x(hi, sum1, d1, d1);
        }

        void SquaredDifferenceSum16f(const uint16_t* a, const uint16_t* b, size_t size, float* sum)
        {
            size_t A = svlen(svuint16_t()), sizeA = AlignLo(size, A), i = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            svfloat32_t sum0 = svdup_n_f32(0.0f), sum1 = svdup_n_f32(0.0f);
            for (; i < sizeA; i += A)
                SquaredDifferenceSum16f(a + i, b + i, body16, body32, body32, sum0, sum1);
            if (i < size)
            {
                const svbool_t tail16 = svwhilelt_b16(i, size);
                SquaredDifferenceSum16f(a + i, b + i, tail16, body32, body32, sum0, sum1);
            }
            *sum = svaddv_f32(body32, svadd_f32_x(body32, sum0, sum1));
        }

        SIMD_INLINE void CosineDistance16f(const uint16_t* a, const uint16_t* b, const svbool_t& load, const svbool_t& lo, const svbool_t& hi,
            svfloat32_t& aa0, svfloat32_t& aa1, svfloat32_t& ab0, svfloat32_t& ab1, svfloat32_t& bb0, svfloat32_t& bb1)
        {
            svfloat16_t _a = svreinterpret_f16_u16(svld1_u16(load, a));
            svfloat16_t _b = svreinterpret_f16_u16(svld1_u16(load, b));
            svfloat32_t a0 = svcvt_f32_f16_x(lo, _a);
            svfloat32_t b0 = svcvt_f32_f16_x(lo, _b);
            aa0 = svmla_f32_x(lo, aa0, a0, a0);
            ab0 = svmla_f32_x(lo, ab0, a0, b0);
            bb0 = svmla_f32_x(lo, bb0, b0, b0);
            svfloat32_t a1 = svcvtlt_f32_f16_x(hi, _a);
            svfloat32_t b1 = svcvtlt_f32_f16_x(hi, _b);
            aa1 = svmla_f32_x(hi, aa1, a1, a1);
            ab1 = svmla_f32_x(hi, ab1, a1, b1);
            bb1 = svmla_f32_x(hi, bb1, b1, b1);
        }

        void CosineDistance16f(const uint16_t* a, const uint16_t* b, size_t size, float* distance)
        {
            size_t A = svlen(svuint16_t()), sizeA = AlignLo(size, A), i = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            svfloat32_t aa0 = svdup_n_f32(0.0f), aa1 = svdup_n_f32(0.0f);
            svfloat32_t ab0 = svdup_n_f32(0.0f), ab1 = svdup_n_f32(0.0f);
            svfloat32_t bb0 = svdup_n_f32(0.0f), bb1 = svdup_n_f32(0.0f);
            for (; i < sizeA; i += A)
                CosineDistance16f(a + i, b + i, body16, body32, body32, aa0, aa1, ab0, ab1, bb0, bb1);
            if (i < size)
            {
                const svbool_t tail16 = svwhilelt_b16(i, size);
                CosineDistance16f(a + i, b + i, tail16, body32, body32, aa0, aa1, ab0, ab1, bb0, bb1);
            }
            float _aa = svaddv_f32(body32, svadd_f32_x(body32, aa0, aa1));
            float _ab = svaddv_f32(body32, svadd_f32_x(body32, ab0, ab1));
            float _bb = svaddv_f32(body32, svadd_f32_x(body32, bb0, bb1));
            *distance = 1.0f - _ab / ::sqrt(_aa * _bb);
        }

        SIMD_INLINE void Load16f(const uint16_t* src, const svbool_t& load, const svbool_t& lo, const svbool_t& hi, svfloat32_t& dst0, svfloat32_t& dst1)
        {
            svfloat16_t _src = svreinterpret_f16_u16(svld1_u16(load, src));
            dst0 = svcvt_f32_f16_x(lo, _src);
            dst1 = svcvtlt_f32_f16_x(hi, _src);
        }

        SIMD_INLINE void Square16f(const uint16_t* a, const svbool_t& load, const svbool_t& lo, const svbool_t& hi, svfloat32_t& sum0, svfloat32_t& sum1)
        {
            svfloat32_t a0, a1;
            Load16f(a, load, lo, hi, a0, a1);
            sum0 = svmla_f32_m(lo, sum0, a0, a0);
            sum1 = svmla_f32_m(hi, sum1, a1, a1);
        }

        SIMD_INLINE void Product16f(const svbool_t& lo, const svbool_t& hi, const svfloat32_t& a0, const svfloat32_t& a1,
            const svfloat32_t& b0, const svfloat32_t& b1, svfloat32_t& sum0, svfloat32_t& sum1)
        {
            sum0 = svmla_f32_m(lo, sum0, a0, b0);
            sum1 = svmla_f32_m(hi, sum1, a1, b1);
        }

        SIMD_INLINE float Sum16f(const svbool_t& body, const svfloat32_t& sum0, const svfloat32_t& sum1)
        {
            return svaddv_f32(body, svadd_f32_x(body, sum0, sum1));
        }

        static void Squares(size_t M, size_t K, const uint16_t* const* A, float* squares)
        {
            size_t A_ = svlen(svuint16_t()), KA = AlignLoAny(K, A_);
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            for (size_t i = 0; i < M; ++i)
            {
                svfloat32_t sum0 = svdup_n_f32(0.0f), sum1 = svdup_n_f32(0.0f);
                size_t k = 0;
                for (; k < KA; k += A_)
                    Square16f(A[i] + k, body16, body32, body32, sum0, sum1);
                if (k < K)
                {
                    size_t tail = K - k;
                    Square16f(A[i] + k, svwhilelt_b16(size_t(0), tail),
                        svwhilelt_b32(size_t(0), (tail + 1) / 2), svwhilelt_b32(size_t(0), tail / 2), sum0, sum1);
                }
                squares[i] = Sum16f(body32, sum0, sum1);
            }
        }

        SIMD_INLINE void StoreDistance4(const svbool_t& body, const float* aa, const float* bb, float* distances,
            const svfloat32_t& c0l, const svfloat32_t& c0h, const svfloat32_t& c1l, const svfloat32_t& c1h,
            const svfloat32_t& c2l, const svfloat32_t& c2h, const svfloat32_t& c3l, const svfloat32_t& c3h)
        {
            distances[0] = 1.0f - Sum16f(body, c0l, c0h) / ::sqrt(aa[0] * bb[0]);
            distances[1] = 1.0f - Sum16f(body, c1l, c1h) / ::sqrt(aa[0] * bb[1]);
            distances[2] = 1.0f - Sum16f(body, c2l, c2h) / ::sqrt(aa[0] * bb[2]);
            distances[3] = 1.0f - Sum16f(body, c3l, c3h) / ::sqrt(aa[0] * bb[3]);
        }

        static void MicroCosineDistances2x4(size_t K, const uint16_t* const* A, const uint16_t* const* B, const float* aa, const float* bb, float* distances, size_t stride)
        {
            size_t A_ = svlen(svuint16_t()), KA = AlignLoAny(K, A_), k = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            svfloat32_t c00l = svdup_n_f32(0.0f), c00h = svdup_n_f32(0.0f), c01l = c00l, c01h = c00h, c02l = c00l, c02h = c00h, c03l = c00l, c03h = c00h;
            svfloat32_t c10l = svdup_n_f32(0.0f), c10h = svdup_n_f32(0.0f), c11l = c10l, c11h = c10h, c12l = c10l, c12h = c10h, c13l = c10l, c13h = c10h;
            for (; k < KA; k += A_)
            {
                svfloat32_t a0l, a0h, a1l, a1h, b0l, b0h;
                Load16f(A[0] + k, body16, body32, body32, a0l, a0h);
                Load16f(A[1] + k, body16, body32, body32, a1l, a1h);
                Load16f(B[0] + k, body16, body32, body32, b0l, b0h);
                Product16f(body32, body32, a0l, a0h, b0l, b0h, c00l, c00h);
                Product16f(body32, body32, a1l, a1h, b0l, b0h, c10l, c10h);
                Load16f(B[1] + k, body16, body32, body32, b0l, b0h);
                Product16f(body32, body32, a0l, a0h, b0l, b0h, c01l, c01h);
                Product16f(body32, body32, a1l, a1h, b0l, b0h, c11l, c11h);
                Load16f(B[2] + k, body16, body32, body32, b0l, b0h);
                Product16f(body32, body32, a0l, a0h, b0l, b0h, c02l, c02h);
                Product16f(body32, body32, a1l, a1h, b0l, b0h, c12l, c12h);
                Load16f(B[3] + k, body16, body32, body32, b0l, b0h);
                Product16f(body32, body32, a0l, a0h, b0l, b0h, c03l, c03h);
                Product16f(body32, body32, a1l, a1h, b0l, b0h, c13l, c13h);
            }
            if (k < K)
            {
                size_t tail = K - k;
                svbool_t load = svwhilelt_b16(size_t(0), tail);
                svbool_t lo = svwhilelt_b32(size_t(0), (tail + 1) / 2);
                svbool_t hi = svwhilelt_b32(size_t(0), tail / 2);
                svfloat32_t a0l, a0h, a1l, a1h, b0l, b0h;
                Load16f(A[0] + k, load, lo, hi, a0l, a0h);
                Load16f(A[1] + k, load, lo, hi, a1l, a1h);
                Load16f(B[0] + k, load, lo, hi, b0l, b0h);
                Product16f(lo, hi, a0l, a0h, b0l, b0h, c00l, c00h);
                Product16f(lo, hi, a1l, a1h, b0l, b0h, c10l, c10h);
                Load16f(B[1] + k, load, lo, hi, b0l, b0h);
                Product16f(lo, hi, a0l, a0h, b0l, b0h, c01l, c01h);
                Product16f(lo, hi, a1l, a1h, b0l, b0h, c11l, c11h);
                Load16f(B[2] + k, load, lo, hi, b0l, b0h);
                Product16f(lo, hi, a0l, a0h, b0l, b0h, c02l, c02h);
                Product16f(lo, hi, a1l, a1h, b0l, b0h, c12l, c12h);
                Load16f(B[3] + k, load, lo, hi, b0l, b0h);
                Product16f(lo, hi, a0l, a0h, b0l, b0h, c03l, c03h);
                Product16f(lo, hi, a1l, a1h, b0l, b0h, c13l, c13h);
            }
            StoreDistance4(body32, aa + 0, bb, distances + 0 * stride, c00l, c00h, c01l, c01h, c02l, c02h, c03l, c03h);
            StoreDistance4(body32, aa + 1, bb, distances + 1 * stride, c10l, c10h, c11l, c11h, c12l, c12h, c13l, c13h);
        }

        static void MicroCosineDistances2x1(size_t K, const uint16_t* const* A, const uint16_t* const* B, const float* aa, const float* bb, float* distances, size_t stride)
        {
            size_t A_ = svlen(svuint16_t()), KA = AlignLoAny(K, A_), k = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            svfloat32_t c00l = svdup_n_f32(0.0f), c00h = svdup_n_f32(0.0f), c10l = c00l, c10h = c00h;
            for (; k < KA; k += A_)
            {
                svfloat32_t a0l, a0h, a1l, a1h, b0l, b0h;
                Load16f(B[0] + k, body16, body32, body32, b0l, b0h);
                Load16f(A[0] + k, body16, body32, body32, a0l, a0h);
                Product16f(body32, body32, a0l, a0h, b0l, b0h, c00l, c00h);
                Load16f(A[1] + k, body16, body32, body32, a1l, a1h);
                Product16f(body32, body32, a1l, a1h, b0l, b0h, c10l, c10h);
            }
            if (k < K)
            {
                size_t tail = K - k;
                svbool_t load = svwhilelt_b16(size_t(0), tail);
                svbool_t lo = svwhilelt_b32(size_t(0), (tail + 1) / 2);
                svbool_t hi = svwhilelt_b32(size_t(0), tail / 2);
                svfloat32_t a0l, a0h, a1l, a1h, b0l, b0h;
                Load16f(B[0] + k, load, lo, hi, b0l, b0h);
                Load16f(A[0] + k, load, lo, hi, a0l, a0h);
                Product16f(lo, hi, a0l, a0h, b0l, b0h, c00l, c00h);
                Load16f(A[1] + k, load, lo, hi, a1l, a1h);
                Product16f(lo, hi, a1l, a1h, b0l, b0h, c10l, c10h);
            }
            distances[0 * stride] = 1.0f - Sum16f(body32, c00l, c00h) / ::sqrt(aa[0] * bb[0]);
            distances[1 * stride] = 1.0f - Sum16f(body32, c10l, c10h) / ::sqrt(aa[1] * bb[0]);
        }

        static void MicroCosineDistances1x4(size_t K, const uint16_t* const* A, const uint16_t* const* B, const float* aa, const float* bb, float* distances, size_t stride)
        {
            size_t A_ = svlen(svuint16_t()), KA = AlignLoAny(K, A_), k = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            svfloat32_t c00l = svdup_n_f32(0.0f), c00h = svdup_n_f32(0.0f), c01l = c00l, c01h = c00h, c02l = c00l, c02h = c00h, c03l = c00l, c03h = c00h;
            for (; k < KA; k += A_)
            {
                svfloat32_t a0l, a0h, b0l, b0h;
                Load16f(A[0] + k, body16, body32, body32, a0l, a0h);
                Load16f(B[0] + k, body16, body32, body32, b0l, b0h);
                Product16f(body32, body32, a0l, a0h, b0l, b0h, c00l, c00h);
                Load16f(B[1] + k, body16, body32, body32, b0l, b0h);
                Product16f(body32, body32, a0l, a0h, b0l, b0h, c01l, c01h);
                Load16f(B[2] + k, body16, body32, body32, b0l, b0h);
                Product16f(body32, body32, a0l, a0h, b0l, b0h, c02l, c02h);
                Load16f(B[3] + k, body16, body32, body32, b0l, b0h);
                Product16f(body32, body32, a0l, a0h, b0l, b0h, c03l, c03h);
            }
            if (k < K)
            {
                size_t tail = K - k;
                svbool_t load = svwhilelt_b16(size_t(0), tail);
                svbool_t lo = svwhilelt_b32(size_t(0), (tail + 1) / 2);
                svbool_t hi = svwhilelt_b32(size_t(0), tail / 2);
                svfloat32_t a0l, a0h, b0l, b0h;
                Load16f(A[0] + k, load, lo, hi, a0l, a0h);
                Load16f(B[0] + k, load, lo, hi, b0l, b0h);
                Product16f(lo, hi, a0l, a0h, b0l, b0h, c00l, c00h);
                Load16f(B[1] + k, load, lo, hi, b0l, b0h);
                Product16f(lo, hi, a0l, a0h, b0l, b0h, c01l, c01h);
                Load16f(B[2] + k, load, lo, hi, b0l, b0h);
                Product16f(lo, hi, a0l, a0h, b0l, b0h, c02l, c02h);
                Load16f(B[3] + k, load, lo, hi, b0l, b0h);
                Product16f(lo, hi, a0l, a0h, b0l, b0h, c03l, c03h);
            }
            StoreDistance4(body32, aa, bb, distances, c00l, c00h, c01l, c01h, c02l, c02h, c03l, c03h);
        }

        static void MicroCosineDistances1x1(size_t K, const uint16_t* const* A, const uint16_t* const* B, const float* aa, const float* bb, float* distances, size_t stride)
        {
            size_t A_ = svlen(svuint16_t()), KA = AlignLoAny(K, A_), k = 0;
            const svbool_t body16 = svptrue_b16();
            const svbool_t body32 = svptrue_b32();
            svfloat32_t c00l = svdup_n_f32(0.0f), c00h = svdup_n_f32(0.0f);
            for (; k < KA; k += A_)
            {
                svfloat32_t a0l, a0h, b0l, b0h;
                Load16f(A[0] + k, body16, body32, body32, a0l, a0h);
                Load16f(B[0] + k, body16, body32, body32, b0l, b0h);
                Product16f(body32, body32, a0l, a0h, b0l, b0h, c00l, c00h);
            }
            if (k < K)
            {
                size_t tail = K - k;
                svbool_t load = svwhilelt_b16(size_t(0), tail);
                svbool_t lo = svwhilelt_b32(size_t(0), (tail + 1) / 2);
                svbool_t hi = svwhilelt_b32(size_t(0), tail / 2);
                svfloat32_t a0l, a0h, b0l, b0h;
                Load16f(A[0] + k, load, lo, hi, a0l, a0h);
                Load16f(B[0] + k, load, lo, hi, b0l, b0h);
                Product16f(lo, hi, a0l, a0h, b0l, b0h, c00l, c00h);
            }
            distances[0 * stride] = 1.0f - Sum16f(body32, c00l, c00h) / ::sqrt(aa[0] * bb[0]);
        }

        const size_t MicroM = 2;
        static void MacroCosineDistances(size_t M, size_t N, size_t K, const uint16_t* const* A, const uint16_t* const* B, const float* aa, const float* bb, float* distances, size_t stride)
        {
            size_t M2 = AlignLoAny(M, 2);
            size_t N4 = AlignLoAny(N, 4);
            size_t i = 0;
            for (; i < M2; i += 2)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistances2x4(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                for (; j < N; j += 1)
                    MicroCosineDistances2x1(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                distances += 2 * stride;
            }
            for (; i < M; ++i)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistances1x4(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                for (; j < N; j += 1)
                    MicroCosineDistances1x1(K, A + i, B + j, aa + i, bb + j, distances + j, stride);
                distances += 1 * stride;
            }
        }

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
            size_t F = svcntw(), NF = AlignLo(N, F), j = 0;
            const svbool_t body = svptrue_b32();
            for (; j < NF; j += F)
                svst1_f32(body, norms + j, svsqrt_f32_x(body, svld1_f32(body, norms + j)));
            if (j < N)
            {
                svbool_t tail = svwhilelt_b32(j, N);
                svst1_f32(tail, norms + j, svsqrt_f32_x(tail, svld1_f32(tail, norms + j)));
            }
        }
    }
#endif
}
