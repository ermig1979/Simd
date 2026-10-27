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
#include "Simd/SimdSynetDynamicQuantizedInnerProduct16b.h"
#include "Simd/SimdSynetDynamicQuantizedActivation.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        typedef Simd::DynamicQuantizedInnerProduct16bParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::AlgParam AlgParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::GemmLastPtr GemmLastPtr;

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE svuint32_t Float32ToBFloat16(const svfloat32_t& value, const svbool_t& mask)
        {
            svuint32_t bits = svreinterpret_u32_f32(value);
            svuint32_t round = svadd_n_u32_x(mask, svand_n_u32_x(mask, svlsr_n_u32_x(mask, bits, Base::Bf16::SHIFT), 1), Base::Bf16::ROUND);
            return svlsr_n_u32_x(mask, svadd_u32_x(mask, bits, round), Base::Bf16::SHIFT);
        }

        template<bool bf16> SIMD_INLINE svfloat32_t Load16b(const void* ptr, const svbool_t& mask)
        {
            if (bf16)
                return svreinterpret_f32_u32(svlsl_n_u32_z(mask, svld1uh_u32(mask, (const uint16_t*)ptr), Base::Bf16::SHIFT));
            else
                return svld1_f32(mask, (const float*)ptr);
        }

        //-----------------------------------------------------------------------------------------

        template<bool bf16> void Dqip16bGemmV0_ConvertA(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t M, uint8_t* dst, float* scale, int32_t* zero)
        {
            const size_t F = svcntw(), E = bf16 ? 2 : 4;
            const svbool_t body = svptrue_b32();
            size_t K = p.K, KA = a.aK;
            for (size_t i = 0; i < M; ++i)
            {
                const uint8_t* src = src8 + i * K * E;
                svfloat32_t vmin = svdup_n_f32(0.0f);
                svfloat32_t vmax = svdup_n_f32(0.0f);
                size_t k = 0;
                for (; k + F <= K; k += F)
                {
                    svfloat32_t s = Load16b<bf16>(src + k * E, body);
                    vmin = svmin_f32_x(body, vmin, s);
                    vmax = svmax_f32_x(body, vmax, s);
                }
                if (k < K)
                {
                    svbool_t tail = svwhilelt_b32(k, K);
                    svfloat32_t s = Load16b<bf16>(src + k * E, tail);
                    vmin = svmin_f32_m(tail, vmin, s);
                    vmax = svmax_f32_m(tail, vmax, s);
                }
                float min = svminv_f32(body, vmin);
                float max = svmaxv_f32(body, vmax);
                float rowScale = (min == max) ? 1.0f : (max - min) / 255.0f;
                scale[i] = rowScale;
                float z = -min / rowScale;
                zero[i] = Base::NearByInt(Simd::Max(0.0f, Simd::Min(255.0f, z)));
                svfloat32_t norm = svdup_n_f32(1.0f / rowScale);
                svint32_t _zero = svdup_n_s32(zero[i]);
                for (k = 0; k + F <= K; k += F)
                {
                    svbool_t m = body;
                    svint32_t i32 = svadd_s32_x(m, NearbyInt(svmul_f32_x(m, Load16b<bf16>(src + k * E, m), norm), m), _zero);
                    i32 = svmin_n_s32_x(m, svmax_n_s32_x(m, i32, 0), 255);
                    svst1b_u32(m, dst + k, svreinterpret_u32_s32(i32));
                }
                if (k < K)
                {
                    svbool_t m = svwhilelt_b32(k, K);
                    svint32_t i32 = svadd_s32_x(m, NearbyInt(svmul_f32_x(m, Load16b<bf16>(src + k * E, m), norm), m), _zero);
                    i32 = svmin_n_s32_x(m, svmax_n_s32_x(m, i32, 0), 255);
                    svst1b_u32(m, dst + k, svreinterpret_u32_s32(i32));
                }
                for (k = K; k < KA; ++k)
                    dst[k] = 0;
                dst += KA;
            }
        }

        void Dqip16bGemmV0_ConvertAf(const uint8_t* src, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t M, uint8_t* dst, float* scale, int32_t* zero)
        {
            Dqip16bGemmV0_ConvertA<false>(src, p, a, M, dst, scale, zero);
        }

        void Dqip16bGemmV0_ConvertAb(const uint8_t* src, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t M, uint8_t* dst, float* scale, int32_t* zero)
        {
            Dqip16bGemmV0_ConvertA<true>(src, p, a, M, dst, scale, zero);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void QuantizeB4(const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2, const svfloat32_t& s3,
            const svfloat32_t& norm, const svint32_t& lo, const svint32_t& hi, svint32_t& sums, int8_t* dst, const svbool_t& mask)
        {
            svint32_t v0 = svmin_s32_z(mask, svmax_s32_z(mask, NearbyInt(svmul_f32_z(mask, s0, norm), mask), lo), hi);
            svint32_t v1 = svmin_s32_z(mask, svmax_s32_z(mask, NearbyInt(svmul_f32_z(mask, s1, norm), mask), lo), hi);
            svint32_t v2 = svmin_s32_z(mask, svmax_s32_z(mask, NearbyInt(svmul_f32_z(mask, s2, norm), mask), lo), hi);
            svint32_t v3 = svmin_s32_z(mask, svmax_s32_z(mask, NearbyInt(svmul_f32_z(mask, s3, norm), mask), lo), hi);
            sums = svsub_s32_z(mask, sums, svadd_s32_z(mask, svadd_s32_z(mask, v0, v1), svadd_s32_z(mask, v2, v3)));
            svint32_t b0 = svand_n_s32_z(mask, v0, 0xFF);
            svint32_t b1 = svlsl_n_s32_z(mask, svand_n_s32_z(mask, v1, 0xFF), 8);
            svint32_t b2 = svlsl_n_s32_z(mask, svand_n_s32_z(mask, v2, 0xFF), 16);
            svint32_t b3 = svlsl_n_s32_z(mask, svand_n_s32_z(mask, v3, 0xFF), 24);
            svint32_t packed = svorr_s32_z(mask, svorr_s32_z(mask, b0, b1), svorr_s32_z(mask, b2, b3));
            svst1_s8(svptrue_b8(), dst, svreinterpret_s8_s32(packed));
        }

        template<bool bf16> void Dqip16bGemmV0_ConvertB(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t N, int8_t* dst, float* scale, int32_t* sums)
        {
            const size_t F = svcntw(), E = bf16 ? 2 : 4;
            const svbool_t body = svptrue_b32();
            const svbool_t body8 = svptrue_b8();
            size_t KA = a.aK, K = p.K, K4 = AlignLo(K, 4), pN = p.N;
            bool overflow = p.overflow || a.overflow;
            svint32_t lo = svdup_n_s32(overflow ? -64 : -128), hi = svdup_n_s32(overflow ? 63 : 127);
            svfloat32_t eps = svdup_n_f32(0.000001f), q = svdup_n_f32(overflow ? 63.0f : 127.0f);
            svfloat32_t zero = svdup_n_f32(0.0f);
            svint8_t zeroB = svdup_n_s8(0);
            for (size_t j = 0; j < N; j += F)
            {
                svbool_t mask = j + F <= N ? body : svwhilelt_b32(j, N);
                const uint8_t* ps = src8 + j * E;
                svfloat32_t vmax = eps;
                for (size_t k = 0; k < K; ++k, ps += pN * E)
                    vmax = svmax_f32_m(mask, vmax, svabs_f32_x(mask, Load16b<bf16>(ps, mask)));
                svfloat32_t norm = svdiv_f32_z(mask, q, vmax);
                svst1_f32(mask, scale + j, svdiv_f32_z(mask, vmax, q));
                svint32_t _sums = svdup_n_s32(0);
                ps = src8 + j * E;
                size_t k = 0;
                for (; k < K4; k += 4, ps += pN * E * 4)
                {
                    QuantizeB4(Load16b<bf16>(ps + 0 * pN * E, mask), Load16b<bf16>(ps + 1 * pN * E, mask),
                        Load16b<bf16>(ps + 2 * pN * E, mask), Load16b<bf16>(ps + 3 * pN * E, mask), norm, lo, hi, _sums, dst, mask);
                    dst += F * 4;
                }
                if (k < K)
                {
                    size_t tK = K - k;
                    svfloat32_t s0 = Load16b<bf16>(ps + 0 * pN * E, mask);
                    svfloat32_t s1 = tK > 1 ? Load16b<bf16>(ps + 1 * pN * E, mask) : zero;
                    svfloat32_t s2 = tK > 2 ? Load16b<bf16>(ps + 2 * pN * E, mask) : zero;
                    svfloat32_t s3 = tK > 3 ? Load16b<bf16>(ps + 3 * pN * E, mask) : zero;
                    QuantizeB4(s0, s1, s2, s3, norm, lo, hi, _sums, dst, mask);
                    dst += F * 4;
                    k += 4;
                }
                for (; k < KA; k += 4, dst += F * 4)
                    svst1_s8(body8, dst, zeroB);
                svst1_s32(mask, sums + j, _sums);
            }
        }

        void Dqip16bGemmV0_ConvertBf(const uint8_t* src, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t N, int8_t* dst, float* scale, int32_t* sums)
        {
            Dqip16bGemmV0_ConvertB<false>(src, p, a, N, dst, scale, sums);
        }

        void Dqip16bGemmV0_ConvertBb(const uint8_t* src, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t N, int8_t* dst, float* scale, int32_t* sums)
        {
            Dqip16bGemmV0_ConvertB<true>(src, p, a, N, dst, scale, sums);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void StoreBody1(int32_t* ptr, const svint32_t& val0)
        {
            svst1_s32(svptrue_b32(), ptr, val0);
        }

        SIMD_INLINE void StoreBody2(int32_t* ptr, const svint32_t& val0, const svint32_t& val1)
        {
            const size_t F = svcntw();
            svst1_s32(svptrue_b32(), ptr + 0, val0);
            svst1_s32(svptrue_b32(), ptr + F, val1);
        }

        template<int M> void Dqip16bGemmV0_GemmBody_2xM(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum)
        {
            const size_t F = svcntw(), A = F * 4;
            const svbool_t body8 = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            svint32_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, cA0, cA1, cB0, cB1;
            svuint8_t a0;
            svint8_t b0, b1;
            size_t dS = a.cN, dA = a.aK;
            const int8_t* B1 = B0 + a.aK * F;
            const uint8_t* A1 = A0 + 1 * dA;
            const uint8_t* A2 = A0 + 2 * dA;
            const uint8_t* A3 = A0 + 3 * dA;
            const uint8_t* A4 = A0 + 4 * dA;
            const uint8_t* A5 = A0 + 5 * dA;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0x0) c00 = svld1_s32(body32, sum + 0x0 * dS + 0), c01 = svld1_s32(body32, sum + 0x0 * dS + F);
                    if (M > 0x1) c10 = svld1_s32(body32, sum + 0x1 * dS + 0), c11 = svld1_s32(body32, sum + 0x1 * dS + F);
                    if (M > 0x2) c20 = svld1_s32(body32, sum + 0x2 * dS + 0), c21 = svld1_s32(body32, sum + 0x2 * dS + F);
                    if (M > 0x3) c30 = svld1_s32(body32, sum + 0x3 * dS + 0), c31 = svld1_s32(body32, sum + 0x3 * dS + F);
                    if (M > 0x4) c40 = svld1_s32(body32, sum + 0x4 * dS + 0), c41 = svld1_s32(body32, sum + 0x4 * dS + F);
                    if (M > 0x5) c50 = svld1_s32(body32, sum + 0x5 * dS + 0), c51 = svld1_s32(body32, sum + 0x5 * dS + F);
                    if (M > 0x6) c60 = svld1_s32(body32, sum + 0x6 * dS + 0), c61 = svld1_s32(body32, sum + 0x6 * dS + F);
                    if (M > 0x7) c70 = svld1_s32(body32, sum + 0x7 * dS + 0), c71 = svld1_s32(body32, sum + 0x7 * dS + F);
                    if (M > 0x8) c80 = svld1_s32(body32, sum + 0x8 * dS + 0), c81 = svld1_s32(body32, sum + 0x8 * dS + F);
                    if (M > 0x9) c90 = svld1_s32(body32, sum + 0x9 * dS + 0), c91 = svld1_s32(body32, sum + 0x9 * dS + F);
                    if (M > 0xA) cA0 = svld1_s32(body32, sum + 0xA * dS + 0), cA1 = svld1_s32(body32, sum + 0xA * dS + F);
                    if (M > 0xB) cB0 = svld1_s32(body32, sum + 0xB * dS + 0), cB1 = svld1_s32(body32, sum + 0xB * dS + F);
                }
                else
                {
                    if (M > 0x0) c00 = svdup_n_s32(0), c01 = svdup_n_s32(0);
                    if (M > 0x1) c10 = svdup_n_s32(0), c11 = svdup_n_s32(0);
                    if (M > 0x2) c20 = svdup_n_s32(0), c21 = svdup_n_s32(0);
                    if (M > 0x3) c30 = svdup_n_s32(0), c31 = svdup_n_s32(0);
                    if (M > 0x4) c40 = svdup_n_s32(0), c41 = svdup_n_s32(0);
                    if (M > 0x5) c50 = svdup_n_s32(0), c51 = svdup_n_s32(0);
                    if (M > 0x6) c60 = svdup_n_s32(0), c61 = svdup_n_s32(0);
                    if (M > 0x7) c70 = svdup_n_s32(0), c71 = svdup_n_s32(0);
                    if (M > 0x8) c80 = svdup_n_s32(0), c81 = svdup_n_s32(0);
                    if (M > 0x9) c90 = svdup_n_s32(0), c91 = svdup_n_s32(0);
                    if (M > 0xA) cA0 = svdup_n_s32(0), cA1 = svdup_n_s32(0);
                    if (M > 0xB) cB0 = svdup_n_s32(0), cB1 = svdup_n_s32(0);
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = svld1_s8(body8, B0);
                    b1 = svld1_s8(body8, B1);
                    if (M > 0x0) a0 = Set4(A0 + k0), Madd4<false>(c00, a0, b0), Madd4<false>(c01, a0, b1);
                    if (M > 0x1) a0 = Set4(A1 + k0), Madd4<false>(c10, a0, b0), Madd4<false>(c11, a0, b1);
                    if (M > 0x2) a0 = Set4(A2 + k0), Madd4<false>(c20, a0, b0), Madd4<false>(c21, a0, b1);
                    if (M > 0x3) a0 = Set4(A3 + k0), Madd4<false>(c30, a0, b0), Madd4<false>(c31, a0, b1);
                    if (M > 0x4) a0 = Set4(A4 + k0), Madd4<false>(c40, a0, b0), Madd4<false>(c41, a0, b1);
                    if (M > 0x5) a0 = Set4(A5 + k0), Madd4<false>(c50, a0, b0), Madd4<false>(c51, a0, b1);
                    if (M > 0x6) a0 = Set4(A0 + k6), Madd4<false>(c60, a0, b0), Madd4<false>(c61, a0, b1);
                    if (M > 0x7) a0 = Set4(A1 + k6), Madd4<false>(c70, a0, b0), Madd4<false>(c71, a0, b1);
                    if (M > 0x8) a0 = Set4(A2 + k6), Madd4<false>(c80, a0, b0), Madd4<false>(c81, a0, b1);
                    if (M > 0x9) a0 = Set4(A3 + k6), Madd4<false>(c90, a0, b0), Madd4<false>(c91, a0, b1);
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(cA0, a0, b0), Madd4<false>(cA1, a0, b1);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cB0, a0, b0), Madd4<false>(cB1, a0, b1);
                    B0 += A, B1 += A;
                }
                if (M > 0x0) StoreBody2(sum + 0x0 * dS, c00, c01);
                if (M > 0x1) StoreBody2(sum + 0x1 * dS, c10, c11);
                if (M > 0x2) StoreBody2(sum + 0x2 * dS, c20, c21);
                if (M > 0x3) StoreBody2(sum + 0x3 * dS, c30, c31);
                if (M > 0x4) StoreBody2(sum + 0x4 * dS, c40, c41);
                if (M > 0x5) StoreBody2(sum + 0x5 * dS, c50, c51);
                if (M > 0x6) StoreBody2(sum + 0x6 * dS, c60, c61);
                if (M > 0x7) StoreBody2(sum + 0x7 * dS, c70, c71);
                if (M > 0x8) StoreBody2(sum + 0x8 * dS, c80, c81);
                if (M > 0x9) StoreBody2(sum + 0x9 * dS, c90, c91);
                if (M > 0xA) StoreBody2(sum + 0xA * dS, cA0, cA1);
                if (M > 0xB) StoreBody2(sum + 0xB * dS, cB0, cB1);
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = svld1_s32(body32, sum + 0x0 * dS);
                    if (M > 0x1) c10 = svld1_s32(body32, sum + 0x1 * dS);
                    if (M > 0x2) c20 = svld1_s32(body32, sum + 0x2 * dS);
                    if (M > 0x3) c30 = svld1_s32(body32, sum + 0x3 * dS);
                    if (M > 0x4) c40 = svld1_s32(body32, sum + 0x4 * dS);
                    if (M > 0x5) c50 = svld1_s32(body32, sum + 0x5 * dS);
                    if (M > 0x6) c60 = svld1_s32(body32, sum + 0x6 * dS);
                    if (M > 0x7) c70 = svld1_s32(body32, sum + 0x7 * dS);
                    if (M > 0x8) c80 = svld1_s32(body32, sum + 0x8 * dS);
                    if (M > 0x9) c90 = svld1_s32(body32, sum + 0x9 * dS);
                    if (M > 0xA) cA0 = svld1_s32(body32, sum + 0xA * dS);
                    if (M > 0xB) cB0 = svld1_s32(body32, sum + 0xB * dS);
                }
                else
                {
                    if (M > 0x0) c00 = svdup_n_s32(0);
                    if (M > 0x1) c10 = svdup_n_s32(0);
                    if (M > 0x2) c20 = svdup_n_s32(0);
                    if (M > 0x3) c30 = svdup_n_s32(0);
                    if (M > 0x4) c40 = svdup_n_s32(0);
                    if (M > 0x5) c50 = svdup_n_s32(0);
                    if (M > 0x6) c60 = svdup_n_s32(0);
                    if (M > 0x7) c70 = svdup_n_s32(0);
                    if (M > 0x8) c80 = svdup_n_s32(0);
                    if (M > 0x9) c90 = svdup_n_s32(0);
                    if (M > 0xA) cA0 = svdup_n_s32(0);
                    if (M > 0xB) cB0 = svdup_n_s32(0);
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = svld1_s8(body8, B0);
                    if (M > 0x0) a0 = Set4(A0 + k0), Madd4<false>(c00, a0, b0);
                    if (M > 0x1) a0 = Set4(A1 + k0), Madd4<false>(c10, a0, b0);
                    if (M > 0x2) a0 = Set4(A2 + k0), Madd4<false>(c20, a0, b0);
                    if (M > 0x3) a0 = Set4(A3 + k0), Madd4<false>(c30, a0, b0);
                    if (M > 0x4) a0 = Set4(A4 + k0), Madd4<false>(c40, a0, b0);
                    if (M > 0x5) a0 = Set4(A5 + k0), Madd4<false>(c50, a0, b0);
                    if (M > 0x6) a0 = Set4(A0 + k6), Madd4<false>(c60, a0, b0);
                    if (M > 0x7) a0 = Set4(A1 + k6), Madd4<false>(c70, a0, b0);
                    if (M > 0x8) a0 = Set4(A2 + k6), Madd4<false>(c80, a0, b0);
                    if (M > 0x9) a0 = Set4(A3 + k6), Madd4<false>(c90, a0, b0);
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(cA0, a0, b0);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cB0, a0, b0);
                    B0 += A;
                }
                if (M > 0x0) StoreBody1(sum + 0x0 * dS, c00);
                if (M > 0x1) StoreBody1(sum + 0x1 * dS, c10);
                if (M > 0x2) StoreBody1(sum + 0x2 * dS, c20);
                if (M > 0x3) StoreBody1(sum + 0x3 * dS, c30);
                if (M > 0x4) StoreBody1(sum + 0x4 * dS, c40);
                if (M > 0x5) StoreBody1(sum + 0x5 * dS, c50);
                if (M > 0x6) StoreBody1(sum + 0x6 * dS, c60);
                if (M > 0x7) StoreBody1(sum + 0x7 * dS, c70);
                if (M > 0x8) StoreBody1(sum + 0x8 * dS, c80);
                if (M > 0x9) StoreBody1(sum + 0x9 * dS, c90);
                if (M > 0xA) StoreBody1(sum + 0xA * dS, cA0);
                if (M > 0xB) StoreBody1(sum + 0xB * dS, cB0);
            }
        }

        typedef void(*Dqip16bGemmV0_GemmBody_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum);

        SIMD_INLINE Dqip16bGemmV0_GemmBody_2xM_Ptr GetDqip16bGemmV0_GemmBody_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return Dqip16bGemmV0_GemmBody_2xM<0x1>;
            case 0x2: return Dqip16bGemmV0_GemmBody_2xM<0x2>;
            case 0x3: return Dqip16bGemmV0_GemmBody_2xM<0x3>;
            case 0x4: return Dqip16bGemmV0_GemmBody_2xM<0x4>;
            case 0x5: return Dqip16bGemmV0_GemmBody_2xM<0x5>;
            case 0x6: return Dqip16bGemmV0_GemmBody_2xM<0x6>;
            case 0x7: return Dqip16bGemmV0_GemmBody_2xM<0x7>;
            case 0x8: return Dqip16bGemmV0_GemmBody_2xM<0x8>;
            case 0x9: return Dqip16bGemmV0_GemmBody_2xM<0x9>;
            case 0xA: return Dqip16bGemmV0_GemmBody_2xM<0xA>;
            case 0xB: return Dqip16bGemmV0_GemmBody_2xM<0xB>;
            case 0xC: return Dqip16bGemmV0_GemmBody_2xM<0xC>;
            }
            assert(0);
            return NULL;
        }

        static void Dqip16bGemmV0_GemmBody(const uint8_t* A, const DqipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum)
        {
            const size_t F = svcntw(), DF = F * 2;
            size_t n = 12;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.cN, dA = a.aK;
            Dqip16bGemmV0_GemmBody_2xM_Ptr gemm_2xN = GetDqip16bGemmV0_GemmBody_2xM(n);
            Dqip16bGemmV0_GemmBody_2xM_Ptr gemm_2xM = GetDqip16bGemmV0_GemmBody_2xM(m);

            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, sum + i * dS);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, sum + i * dS);
                B += a.aK * DF;
                sum += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, const svint32_t& sum,
            int32_t aZero, const svint32_t& bSum, float aScale, const svfloat32_t& bScale, const svfloat32_t& bias,
            const svfloat32_t& param0, const svfloat32_t& param1, const svbool_t& mask)
        {
            const size_t F = svcntw();
            svfloat32_t val = svmul_n_f32_x(mask, svcvt_f32_s32_x(mask, svadd_s32_x(mask, sum, svmul_n_s32_x(mask, bSum, aZero))), aScale);
            val = Activate<type>(svmla_f32_x(mask, bias, val, bScale), param0, param1, index, mask);
            if (term == Term16bLast32f)
                svst1_f32(mask, (float*)dst + index * F, val);
            else if (term == Term16bLast16b)
                svst1h_u32(mask, (uint16_t*)dst + index * F, Float32ToBFloat16(val, mask));
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, const svint32_t& sum,
            int32_t aZero, const svint32_t& bSum, float aScale, const svfloat32_t& bScale, const svfloat32_t& bias,
            const svfloat32_t& param0, const svfloat32_t& param1)
        {
            Save<term, type, 0>(dst, sum, aZero, bSum, aScale, bScale, bias, param0, param1, svptrue_b32());
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, const svint32_t& sum,
            int32_t aZero, const svint32_t& bSum, float aScale, const svfloat32_t& bScale, const svfloat32_t& bias,
            const svfloat32_t& param0, const svfloat32_t& param1, size_t tail)
        {
            Save<term, type, 0>(dst, sum, aZero, bSum, aScale, bScale, bias, param0, param1, svwhilelt_b32((size_t)0, tail));
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, const svint32_t& sum0, const svint32_t& sum1,
            int32_t aZero, const svint32_t& bSum0, const svint32_t& bSum1, float aScale, const svfloat32_t& bScale0, const svfloat32_t& bScale1,
            const svfloat32_t& bias0, const svfloat32_t& bias1, const svfloat32_t& param0, const svfloat32_t& param1)
        {
            Save<term, type, 0>(dst, sum0, aZero, bSum0, aScale, bScale0, bias0, param0, param1, svptrue_b32());
            Save<term, type, 1>(dst, sum1, aZero, bSum1, aScale, bScale1, bias1, param0, param1, svptrue_b32());
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, const svint32_t& sum0, const svint32_t& sum1,
            int32_t aZero, const svint32_t& bSum0, const svint32_t& bSum1, float aScale, const svfloat32_t& bScale0, const svfloat32_t& bScale1,
            const svfloat32_t& bias0, const svfloat32_t& bias1, const svfloat32_t& param0, const svfloat32_t& param1, size_t tail)
        {
            Save<term, type, 0>(dst, sum0, aZero, bSum0, aScale, bScale0, bias0, param0, param1, svptrue_b32());
            Save<term, type, 1>(dst, sum1, aZero, bSum1, aScale, bScale1, bias1, param0, param1, svwhilelt_b32((size_t)0, tail));
        }

        template<Term16bType term, SimdConvolutionActivationType type, int M> void Dqip16bGemmV0_GemmLast_2xM(const uint8_t* A0,
            const DqipParam& p, const AlgParam& a, size_t K, size_t N, int update, const int8_t* B0, int32_t* sum, int32_t* buf,
            const int32_t* aZero, const svint32_t& bSum0, const svint32_t& bSum1, const float* aScale, const svfloat32_t& bScale0, const svfloat32_t& bScale1,
            const svfloat32_t& bias0, const svfloat32_t& bias1, const svfloat32_t& param0, const svfloat32_t& param1, uint8_t* C)
        {
            const size_t F = svcntw(), A = F * 4;
            const svbool_t body8 = svptrue_b8();
            const svbool_t body32 = svptrue_b32();
            svint32_t c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, cA0, cA1, cB0, cB1;
            svuint8_t a0;
            svint8_t b0, b1;
            size_t dS = a.cN, dA = a.aK, dC = p.N * a.eC;
            const int8_t* B1 = B0 + a.aK * F;
            const uint8_t* A1 = A0 + 1 * dA;
            const uint8_t* A2 = A0 + 2 * dA;
            const uint8_t* A3 = A0 + 3 * dA;
            const uint8_t* A4 = A0 + 4 * dA;
            const uint8_t* A5 = A0 + 5 * dA;
            (void)buf;
            if (N > F)
            {
                if (update)
                {
                    if (M > 0x0) c00 = svld1_s32(body32, sum + 0x0 * dS + 0), c01 = svld1_s32(body32, sum + 0x0 * dS + F);
                    if (M > 0x1) c10 = svld1_s32(body32, sum + 0x1 * dS + 0), c11 = svld1_s32(body32, sum + 0x1 * dS + F);
                    if (M > 0x2) c20 = svld1_s32(body32, sum + 0x2 * dS + 0), c21 = svld1_s32(body32, sum + 0x2 * dS + F);
                    if (M > 0x3) c30 = svld1_s32(body32, sum + 0x3 * dS + 0), c31 = svld1_s32(body32, sum + 0x3 * dS + F);
                    if (M > 0x4) c40 = svld1_s32(body32, sum + 0x4 * dS + 0), c41 = svld1_s32(body32, sum + 0x4 * dS + F);
                    if (M > 0x5) c50 = svld1_s32(body32, sum + 0x5 * dS + 0), c51 = svld1_s32(body32, sum + 0x5 * dS + F);
                    if (M > 0x6) c60 = svld1_s32(body32, sum + 0x6 * dS + 0), c61 = svld1_s32(body32, sum + 0x6 * dS + F);
                    if (M > 0x7) c70 = svld1_s32(body32, sum + 0x7 * dS + 0), c71 = svld1_s32(body32, sum + 0x7 * dS + F);
                    if (M > 0x8) c80 = svld1_s32(body32, sum + 0x8 * dS + 0), c81 = svld1_s32(body32, sum + 0x8 * dS + F);
                    if (M > 0x9) c90 = svld1_s32(body32, sum + 0x9 * dS + 0), c91 = svld1_s32(body32, sum + 0x9 * dS + F);
                    if (M > 0xA) cA0 = svld1_s32(body32, sum + 0xA * dS + 0), cA1 = svld1_s32(body32, sum + 0xA * dS + F);
                    if (M > 0xB) cB0 = svld1_s32(body32, sum + 0xB * dS + 0), cB1 = svld1_s32(body32, sum + 0xB * dS + F);
                }
                else
                {
                    if (M > 0x0) c00 = svdup_n_s32(0), c01 = svdup_n_s32(0);
                    if (M > 0x1) c10 = svdup_n_s32(0), c11 = svdup_n_s32(0);
                    if (M > 0x2) c20 = svdup_n_s32(0), c21 = svdup_n_s32(0);
                    if (M > 0x3) c30 = svdup_n_s32(0), c31 = svdup_n_s32(0);
                    if (M > 0x4) c40 = svdup_n_s32(0), c41 = svdup_n_s32(0);
                    if (M > 0x5) c50 = svdup_n_s32(0), c51 = svdup_n_s32(0);
                    if (M > 0x6) c60 = svdup_n_s32(0), c61 = svdup_n_s32(0);
                    if (M > 0x7) c70 = svdup_n_s32(0), c71 = svdup_n_s32(0);
                    if (M > 0x8) c80 = svdup_n_s32(0), c81 = svdup_n_s32(0);
                    if (M > 0x9) c90 = svdup_n_s32(0), c91 = svdup_n_s32(0);
                    if (M > 0xA) cA0 = svdup_n_s32(0), cA1 = svdup_n_s32(0);
                    if (M > 0xB) cB0 = svdup_n_s32(0), cB1 = svdup_n_s32(0);
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = svld1_s8(body8, B0);
                    b1 = svld1_s8(body8, B1);
                    if (M > 0x0) a0 = Set4(A0 + k0), Madd4<false>(c00, a0, b0), Madd4<false>(c01, a0, b1);
                    if (M > 0x1) a0 = Set4(A1 + k0), Madd4<false>(c10, a0, b0), Madd4<false>(c11, a0, b1);
                    if (M > 0x2) a0 = Set4(A2 + k0), Madd4<false>(c20, a0, b0), Madd4<false>(c21, a0, b1);
                    if (M > 0x3) a0 = Set4(A3 + k0), Madd4<false>(c30, a0, b0), Madd4<false>(c31, a0, b1);
                    if (M > 0x4) a0 = Set4(A4 + k0), Madd4<false>(c40, a0, b0), Madd4<false>(c41, a0, b1);
                    if (M > 0x5) a0 = Set4(A5 + k0), Madd4<false>(c50, a0, b0), Madd4<false>(c51, a0, b1);
                    if (M > 0x6) a0 = Set4(A0 + k6), Madd4<false>(c60, a0, b0), Madd4<false>(c61, a0, b1);
                    if (M > 0x7) a0 = Set4(A1 + k6), Madd4<false>(c70, a0, b0), Madd4<false>(c71, a0, b1);
                    if (M > 0x8) a0 = Set4(A2 + k6), Madd4<false>(c80, a0, b0), Madd4<false>(c81, a0, b1);
                    if (M > 0x9) a0 = Set4(A3 + k6), Madd4<false>(c90, a0, b0), Madd4<false>(c91, a0, b1);
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(cA0, a0, b0), Madd4<false>(cA1, a0, b1);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cB0, a0, b0), Madd4<false>(cB1, a0, b1);
                    B0 += A, B1 += A;
                }
                if (N == F * 2)
                {
                    if (M > 0x0) Save2<term, type>(C + 0x0 * dC, c00, c01, aZero[0x0], bSum0, bSum1, aScale[0x0], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0x1) Save2<term, type>(C + 0x1 * dC, c10, c11, aZero[0x1], bSum0, bSum1, aScale[0x1], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0x2) Save2<term, type>(C + 0x2 * dC, c20, c21, aZero[0x2], bSum0, bSum1, aScale[0x2], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0x3) Save2<term, type>(C + 0x3 * dC, c30, c31, aZero[0x3], bSum0, bSum1, aScale[0x3], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0x4) Save2<term, type>(C + 0x4 * dC, c40, c41, aZero[0x4], bSum0, bSum1, aScale[0x4], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0x5) Save2<term, type>(C + 0x5 * dC, c50, c51, aZero[0x5], bSum0, bSum1, aScale[0x5], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0x6) Save2<term, type>(C + 0x6 * dC, c60, c61, aZero[0x6], bSum0, bSum1, aScale[0x6], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0x7) Save2<term, type>(C + 0x7 * dC, c70, c71, aZero[0x7], bSum0, bSum1, aScale[0x7], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0x8) Save2<term, type>(C + 0x8 * dC, c80, c81, aZero[0x8], bSum0, bSum1, aScale[0x8], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0x9) Save2<term, type>(C + 0x9 * dC, c90, c91, aZero[0x9], bSum0, bSum1, aScale[0x9], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0xA) Save2<term, type>(C + 0xA * dC, cA0, cA1, aZero[0xA], bSum0, bSum1, aScale[0xA], bScale0, bScale1, bias0, bias1, param0, param1);
                    if (M > 0xB) Save2<term, type>(C + 0xB * dC, cB0, cB1, aZero[0xB], bSum0, bSum1, aScale[0xB], bScale0, bScale1, bias0, bias1, param0, param1);
                }
                else
                {
                    size_t tail = N - F;
                    if (M > 0x0) Save2<term, type>(C + 0x0 * dC, c00, c01, aZero[0x0], bSum0, bSum1, aScale[0x0], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0x1) Save2<term, type>(C + 0x1 * dC, c10, c11, aZero[0x1], bSum0, bSum1, aScale[0x1], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0x2) Save2<term, type>(C + 0x2 * dC, c20, c21, aZero[0x2], bSum0, bSum1, aScale[0x2], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0x3) Save2<term, type>(C + 0x3 * dC, c30, c31, aZero[0x3], bSum0, bSum1, aScale[0x3], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0x4) Save2<term, type>(C + 0x4 * dC, c40, c41, aZero[0x4], bSum0, bSum1, aScale[0x4], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0x5) Save2<term, type>(C + 0x5 * dC, c50, c51, aZero[0x5], bSum0, bSum1, aScale[0x5], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0x6) Save2<term, type>(C + 0x6 * dC, c60, c61, aZero[0x6], bSum0, bSum1, aScale[0x6], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0x7) Save2<term, type>(C + 0x7 * dC, c70, c71, aZero[0x7], bSum0, bSum1, aScale[0x7], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0x8) Save2<term, type>(C + 0x8 * dC, c80, c81, aZero[0x8], bSum0, bSum1, aScale[0x8], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0x9) Save2<term, type>(C + 0x9 * dC, c90, c91, aZero[0x9], bSum0, bSum1, aScale[0x9], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0xA) Save2<term, type>(C + 0xA * dC, cA0, cA1, aZero[0xA], bSum0, bSum1, aScale[0xA], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                    if (M > 0xB) Save2<term, type>(C + 0xB * dC, cB0, cB1, aZero[0xB], bSum0, bSum1, aScale[0xB], bScale0, bScale1, bias0, bias1, param0, param1, tail);
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = svld1_s32(body32, sum + 0x0 * dS);
                    if (M > 0x1) c10 = svld1_s32(body32, sum + 0x1 * dS);
                    if (M > 0x2) c20 = svld1_s32(body32, sum + 0x2 * dS);
                    if (M > 0x3) c30 = svld1_s32(body32, sum + 0x3 * dS);
                    if (M > 0x4) c40 = svld1_s32(body32, sum + 0x4 * dS);
                    if (M > 0x5) c50 = svld1_s32(body32, sum + 0x5 * dS);
                    if (M > 0x6) c60 = svld1_s32(body32, sum + 0x6 * dS);
                    if (M > 0x7) c70 = svld1_s32(body32, sum + 0x7 * dS);
                    if (M > 0x8) c80 = svld1_s32(body32, sum + 0x8 * dS);
                    if (M > 0x9) c90 = svld1_s32(body32, sum + 0x9 * dS);
                    if (M > 0xA) cA0 = svld1_s32(body32, sum + 0xA * dS);
                    if (M > 0xB) cB0 = svld1_s32(body32, sum + 0xB * dS);
                }
                else
                {
                    if (M > 0x0) c00 = svdup_n_s32(0);
                    if (M > 0x1) c10 = svdup_n_s32(0);
                    if (M > 0x2) c20 = svdup_n_s32(0);
                    if (M > 0x3) c30 = svdup_n_s32(0);
                    if (M > 0x4) c40 = svdup_n_s32(0);
                    if (M > 0x5) c50 = svdup_n_s32(0);
                    if (M > 0x6) c60 = svdup_n_s32(0);
                    if (M > 0x7) c70 = svdup_n_s32(0);
                    if (M > 0x8) c80 = svdup_n_s32(0);
                    if (M > 0x9) c90 = svdup_n_s32(0);
                    if (M > 0xA) cA0 = svdup_n_s32(0);
                    if (M > 0xB) cB0 = svdup_n_s32(0);
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = svld1_s8(body8, B0);
                    if (M > 0x0) a0 = Set4(A0 + k0), Madd4<false>(c00, a0, b0);
                    if (M > 0x1) a0 = Set4(A1 + k0), Madd4<false>(c10, a0, b0);
                    if (M > 0x2) a0 = Set4(A2 + k0), Madd4<false>(c20, a0, b0);
                    if (M > 0x3) a0 = Set4(A3 + k0), Madd4<false>(c30, a0, b0);
                    if (M > 0x4) a0 = Set4(A4 + k0), Madd4<false>(c40, a0, b0);
                    if (M > 0x5) a0 = Set4(A5 + k0), Madd4<false>(c50, a0, b0);
                    if (M > 0x6) a0 = Set4(A0 + k6), Madd4<false>(c60, a0, b0);
                    if (M > 0x7) a0 = Set4(A1 + k6), Madd4<false>(c70, a0, b0);
                    if (M > 0x8) a0 = Set4(A2 + k6), Madd4<false>(c80, a0, b0);
                    if (M > 0x9) a0 = Set4(A3 + k6), Madd4<false>(c90, a0, b0);
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(cA0, a0, b0);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cB0, a0, b0);
                    B0 += A;
                }
                if (N == F)
                {
                    if (M > 0x0) Save1<term, type>(C + 0x0 * dC, c00, aZero[0x0], bSum0, aScale[0x0], bScale0, bias0, param0, param1);
                    if (M > 0x1) Save1<term, type>(C + 0x1 * dC, c10, aZero[0x1], bSum0, aScale[0x1], bScale0, bias0, param0, param1);
                    if (M > 0x2) Save1<term, type>(C + 0x2 * dC, c20, aZero[0x2], bSum0, aScale[0x2], bScale0, bias0, param0, param1);
                    if (M > 0x3) Save1<term, type>(C + 0x3 * dC, c30, aZero[0x3], bSum0, aScale[0x3], bScale0, bias0, param0, param1);
                    if (M > 0x4) Save1<term, type>(C + 0x4 * dC, c40, aZero[0x4], bSum0, aScale[0x4], bScale0, bias0, param0, param1);
                    if (M > 0x5) Save1<term, type>(C + 0x5 * dC, c50, aZero[0x5], bSum0, aScale[0x5], bScale0, bias0, param0, param1);
                    if (M > 0x6) Save1<term, type>(C + 0x6 * dC, c60, aZero[0x6], bSum0, aScale[0x6], bScale0, bias0, param0, param1);
                    if (M > 0x7) Save1<term, type>(C + 0x7 * dC, c70, aZero[0x7], bSum0, aScale[0x7], bScale0, bias0, param0, param1);
                    if (M > 0x8) Save1<term, type>(C + 0x8 * dC, c80, aZero[0x8], bSum0, aScale[0x8], bScale0, bias0, param0, param1);
                    if (M > 0x9) Save1<term, type>(C + 0x9 * dC, c90, aZero[0x9], bSum0, aScale[0x9], bScale0, bias0, param0, param1);
                    if (M > 0xA) Save1<term, type>(C + 0xA * dC, cA0, aZero[0xA], bSum0, aScale[0xA], bScale0, bias0, param0, param1);
                    if (M > 0xB) Save1<term, type>(C + 0xB * dC, cB0, aZero[0xB], bSum0, aScale[0xB], bScale0, bias0, param0, param1);
                }
                else
                {
                    size_t tail = N;
                    if (M > 0x0) Save1<term, type>(C + 0x0 * dC, c00, aZero[0x0], bSum0, aScale[0x0], bScale0, bias0, param0, param1, tail);
                    if (M > 0x1) Save1<term, type>(C + 0x1 * dC, c10, aZero[0x1], bSum0, aScale[0x1], bScale0, bias0, param0, param1, tail);
                    if (M > 0x2) Save1<term, type>(C + 0x2 * dC, c20, aZero[0x2], bSum0, aScale[0x2], bScale0, bias0, param0, param1, tail);
                    if (M > 0x3) Save1<term, type>(C + 0x3 * dC, c30, aZero[0x3], bSum0, aScale[0x3], bScale0, bias0, param0, param1, tail);
                    if (M > 0x4) Save1<term, type>(C + 0x4 * dC, c40, aZero[0x4], bSum0, aScale[0x4], bScale0, bias0, param0, param1, tail);
                    if (M > 0x5) Save1<term, type>(C + 0x5 * dC, c50, aZero[0x5], bSum0, aScale[0x5], bScale0, bias0, param0, param1, tail);
                    if (M > 0x6) Save1<term, type>(C + 0x6 * dC, c60, aZero[0x6], bSum0, aScale[0x6], bScale0, bias0, param0, param1, tail);
                    if (M > 0x7) Save1<term, type>(C + 0x7 * dC, c70, aZero[0x7], bSum0, aScale[0x7], bScale0, bias0, param0, param1, tail);
                    if (M > 0x8) Save1<term, type>(C + 0x8 * dC, c80, aZero[0x8], bSum0, aScale[0x8], bScale0, bias0, param0, param1, tail);
                    if (M > 0x9) Save1<term, type>(C + 0x9 * dC, c90, aZero[0x9], bSum0, aScale[0x9], bScale0, bias0, param0, param1, tail);
                    if (M > 0xA) Save1<term, type>(C + 0xA * dC, cA0, aZero[0xA], bSum0, aScale[0xA], bScale0, bias0, param0, param1, tail);
                    if (M > 0xB) Save1<term, type>(C + 0xB * dC, cB0, aZero[0xB], bSum0, aScale[0xB], bScale0, bias0, param0, param1, tail);
                }
            }
        }

        typedef void(*Dqip16bGemmV0_GemmLast_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, size_t N, int update, const int8_t* B0,
            int32_t* sum, int32_t* buf, const int32_t* aZero, const svint32_t& bSum0, const svint32_t& bSum1, const float* aScale,
            const svfloat32_t& bScale0, const svfloat32_t& bScale1, const svfloat32_t& bias0, const svfloat32_t& bias1,
            const svfloat32_t& param0, const svfloat32_t& param1, uint8_t* C);

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE Dqip16bGemmV0_GemmLast_2xM_Ptr GetDqip16bGemmV0_GemmLast_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0x1>;
            case 0x2: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0x2>;
            case 0x3: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0x3>;
            case 0x4: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0x4>;
            case 0x5: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0x5>;
            case 0x6: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0x6>;
            case 0x7: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0x7>;
            case 0x8: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0x8>;
            case 0x9: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0x9>;
            case 0xA: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0xA>;
            case 0xB: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0xB>;
            case 0xC: return Dqip16bGemmV0_GemmLast_2xM<term, type, 0xC>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Dqip16bGemmV0_GemmLast(const uint8_t* A, const DqipParam& p,
            const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf,
            const int32_t* aZero, const int32_t* bSums, const float* aScale, const float* bScale, const float* bias, const float* params, uint8_t* C)
        {
            const size_t F = svcntw(), DF = F * 2;
            const svbool_t body = svptrue_b32();
            size_t n = 12;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.cN, dA = a.aK, dC = p.N * a.eC;
            Dqip16bGemmV0_GemmLast_2xM_Ptr gemm_2xN = GetDqip16bGemmV0_GemmLast_2xM<term, type>(n);
            Dqip16bGemmV0_GemmLast_2xM_Ptr gemm_2xM = GetDqip16bGemmV0_GemmLast_2xM<term, type>(m);

            svfloat32_t param0 = svdup_n_f32(0.0f), param1 = svdup_n_f32(0.0f);
            if (type != SimdConvolutionActivationIdentity)
            {
                param0 = svdup_n_f32(params[0]);
                param1 = svdup_n_f32(params[1]);
            }
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                svbool_t mask0 = dN >= F ? body : svwhilelt_b32((size_t)0, dN);
                svint32_t bSum0 = svld1_s32(mask0, bSums + j);
                svfloat32_t bScale0 = svld1_f32(mask0, bScale + j);
                svfloat32_t bias0 = svld1_f32(mask0, bias + j);
                svint32_t bSum1 = svdup_n_s32(0);
                svfloat32_t bScale1 = svdup_n_f32(0.0f);
                svfloat32_t bias1 = svdup_n_f32(0.0f);
                if (dN > F)
                {
                    svbool_t mask1 = dN >= DF ? body : svwhilelt_b32((size_t)0, dN - F);
                    bSum1 = svld1_s32(mask1, bSums + j + F);
                    bScale1 = svld1_f32(mask1, bScale + j + F);
                    bias1 = svld1_f32(mask1, bias + j + F);
                }
                if (type == SimdConvolutionActivationPrelu)
                {
                    param0 = svld1_f32(mask0, params + j);
                    if (dN > F)
                    {
                        svbool_t mask1 = dN >= DF ? body : svwhilelt_b32((size_t)0, dN - F);
                        param1 = svld1_f32(mask1, params + j + F);
                    }
                }
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, aZero + i, bSum0, bSum1, aScale + i, bScale0, bScale1, bias0, bias1, param0, param1, C + i * dC);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, aZero + i, bSum0, bSum1, aScale + i, bScale0, bScale1, bias0, bias1, param0, param1, C + i * dC);
                B += a.aK * DF;
                sum += DF;
                C += DF * a.eC;
            }
        }

        template<Term16bType term> void SetGemmLast(const DqipParam& p, GemmLastPtr& gemmLast)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: gemmLast = Dqip16bGemmV0_GemmLast<term, SimdConvolutionActivationGelu>; break;
            default:
                assert(0);
            }
        }

        void SetGemmLast(const DqipParam& p, GemmLastPtr& gemmLast)
        {
            if (p.typeC == SimdTensorData16b)
                SetGemmLast<Term16bLast16b>(p, gemmLast);
            else if (p.typeC == SimdTensorData32f)
                SetGemmLast<Term16bLast32f>(p, gemmLast);
        }

        //-----------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProduct16bGemmV0::SynetDynamicQuantizedInnerProduct16bGemmV0(const DynamicQuantizedInnerProduct16bParam& p)
            : Base::SynetDynamicQuantizedInnerProduct16bGemmV0(p)
        {
            const size_t F = svcntw();
            SetAlgParam(F, 12, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (p.typeA == SimdTensorData32f)
                _convertA = Dqip16bGemmV0_ConvertAf;
            else if (p.typeA == SimdTensorData16b)
                _convertA = Dqip16bGemmV0_ConvertAb;
            if (p.typeB == SimdTensorData32f)
                _convertB = Dqip16bGemmV0_ConvertBf;
            else if (p.typeB == SimdTensorData16b)
                _convertB = Dqip16bGemmV0_ConvertBb;
            _gemmBody = Dqip16bGemmV0_GemmBody;
            SetGemmLast(p, _gemmLast);
        }
    }
#endif
}
