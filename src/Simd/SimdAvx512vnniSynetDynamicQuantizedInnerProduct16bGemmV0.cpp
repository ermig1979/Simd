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
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdMinMax.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#if defined(SIMD_AVX512VNNI_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512vnni
    {
        typedef Simd::DynamicQuantizedInnerProduct16bParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::AlgParam AlgParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::GemmLastPtr GemmLastPtr;

        //-----------------------------------------------------------------------------------------

        static SIMD_INLINE __m512 Abs(__m512 value)
        {
            return _mm512_andnot_ps(_mm512_set1_ps(-0.0f), value);
        }

        const __m512i K8_SFL_IDX = SIMD_MM512_SETR_EPI8(
            0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
            0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
            0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
            0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);

        static SIMD_INLINE void QuantizeB4(__m512 s0, __m512 s1, __m512 s2, __m512 s3, __m512 norm, __m512i lo, __m512i hi, __m512i& sums, int8_t* dst)
        {
            __m512i v0 = _mm512_max_epi32(lo, _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(s0, norm)), hi));
            __m512i v1 = _mm512_max_epi32(lo, _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(s1, norm)), hi));
            __m512i v2 = _mm512_max_epi32(lo, _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(s2, norm)), hi));
            __m512i v3 = _mm512_max_epi32(lo, _mm512_min_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(s3, norm)), hi));
            sums = _mm512_sub_epi32(sums, _mm512_add_epi32(_mm512_add_epi32(v0, v1), _mm512_add_epi32(v2, v3)));
            __m512i d0123 = _mm512_packs_epi16(_mm512_packs_epi32(v0, v1), _mm512_packs_epi32(v2, v3));
            _mm512_storeu_si512((__m512i*)dst, _mm512_shuffle_epi8(d0123, K8_SFL_IDX));
        }

        void Dqip16bGemmV0_ConvertBf(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t N, int8_t* dst, float* scale, int32_t* sums)
        {
            const float* src = (float*)src8;
            size_t KA = a.aK, K = p.K, K4 = AlignLo(K, 4), NF = AlignLo(N, F), pN = p.N, j = 0, k;
            __m512i _lo = _mm512_set1_epi32(p.overflow ? -64 : -128), _hi = _mm512_set1_epi32(p.overflow ? 63 : 127);
            __m512 _eps = _mm512_set1_ps(0.000001f), _q = _mm512_set1_ps(p.overflow ? 63.0f : 127.0f);
            for (; j < NF; j += F)
            {
                const float* ps = src + j;
                __m512 _max = _eps;
                for (k = 0; k < K; ++k, ps += pN)
                    _max = _mm512_max_ps(_max, Abs(_mm512_loadu_ps(ps)));
                __m512 _norm = _mm512_div_ps(_q, _max);
                _mm512_storeu_ps(scale + j, _mm512_div_ps(_max, _q));
                __m512i _sums = _mm512_setzero_si512();
                ps = src + j;
                for (k = 0; k < K4; k += 4, ps += pN * 4)
                {
                    QuantizeB4(_mm512_loadu_ps(ps + 0 * pN), _mm512_loadu_ps(ps + 1 * pN), _mm512_loadu_ps(ps + 2 * pN), _mm512_loadu_ps(ps + 3 * pN), _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < K; k += 4, ps += pN * 4)
                {
                    size_t tK = Simd::Min<size_t>(4, K - k);
                    __m512 s0 = _mm512_loadu_ps(ps + 0 * pN);
                    __m512 s1 = tK > 1 ? _mm512_loadu_ps(ps + 1 * pN) : _mm512_setzero_ps();
                    __m512 s2 = tK > 2 ? _mm512_loadu_ps(ps + 2 * pN) : _mm512_setzero_ps();
                    __m512 s3 = tK > 3 ? _mm512_loadu_ps(ps + 3 * pN) : _mm512_setzero_ps();
                    QuantizeB4(s0, s1, s2, s3, _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < KA; k += 4, dst += F * 4)
                    SetZero((uint8_t*)dst);
                _mm512_storeu_si512((__m512i*)(sums + j), _sums);
            }
            for (; j < N; j += F)
            {
                __mmask16 tailN = TailMask16(N - j);
                const float* ps = src + j;
                __m512 _max = _eps;
                for (k = 0; k < K; ++k, ps += pN)
                    _max = _mm512_max_ps(_max, Abs(_mm512_maskz_loadu_ps(tailN, ps)));
                __m512 _norm = _mm512_div_ps(_q, _max);
                _mm512_mask_storeu_ps(scale + j, tailN, _mm512_div_ps(_max, _q));
                __m512i _sums = _mm512_setzero_si512();
                ps = src + j;
                for (k = 0; k < K4; k += 4, ps += pN * 4)
                {
                    QuantizeB4(_mm512_maskz_loadu_ps(tailN, ps + 0 * pN), _mm512_maskz_loadu_ps(tailN, ps + 1 * pN),
                        _mm512_maskz_loadu_ps(tailN, ps + 2 * pN), _mm512_maskz_loadu_ps(tailN, ps + 3 * pN), _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < K; k += 4, ps += pN * 4)
                {
                    size_t tK = Simd::Min<size_t>(4, K - k);
                    __m512 s0 = _mm512_maskz_loadu_ps(tailN, ps + 0 * pN);
                    __m512 s1 = tK > 1 ? _mm512_maskz_loadu_ps(tailN, ps + 1 * pN) : _mm512_setzero_ps();
                    __m512 s2 = tK > 2 ? _mm512_maskz_loadu_ps(tailN, ps + 2 * pN) : _mm512_setzero_ps();
                    __m512 s3 = tK > 3 ? _mm512_maskz_loadu_ps(tailN, ps + 3 * pN) : _mm512_setzero_ps();
                    QuantizeB4(s0, s1, s2, s3, _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < KA; k += 4, dst += F * 4)
                    SetZero((uint8_t*)dst);
                _mm512_mask_storeu_epi32(sums + j, tailN, _sums);
            }
        }

        //-----------------------------------------------------------------------------------------

        static SIMD_INLINE __m512 LoadBf16ToFp32(const uint16_t* ptr)
        {
            return BFloat16ToFloat32(_mm256_loadu_si256((__m256i*)ptr));
        }

        static SIMD_INLINE __m512 LoadBf16ToFp32(const uint16_t* ptr, __mmask16 mask)
        {
            return BFloat16ToFloat32(_mm256_maskz_loadu_epi16(mask, ptr));
        }

        void Dqip16bGemmV0_ConvertBb(const uint8_t* src8, const DynamicQuantizedInnerProduct16bParam& p,
            const AlgParam& a, size_t N, int8_t* dst, float* scale, int32_t* sums)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t KA = a.aK, K = p.K, K4 = AlignLo(K, 4), NF = AlignLo(N, F), pN = p.N, j = 0, k;
            __m512i _lo = _mm512_set1_epi32(p.overflow ? -64 : -128), _hi = _mm512_set1_epi32(p.overflow ? 63 : 127);
            __m512 _eps = _mm512_set1_ps(0.000001f), _q = _mm512_set1_ps(p.overflow ? 63.0f : 127.0f);
            for (; j < NF; j += F)
            {
                const uint16_t* ps = src + j;
                __m512 _max = _eps;
                for (k = 0; k < K; ++k, ps += pN)
                    _max = _mm512_max_ps(_max, Abs(LoadBf16ToFp32(ps)));
                __m512 _norm = _mm512_div_ps(_q, _max);
                _mm512_storeu_ps(scale + j, _mm512_div_ps(_max, _q));
                __m512i _sums = _mm512_setzero_si512();
                ps = src + j;
                for (k = 0; k < K4; k += 4, ps += pN * 4)
                {
                    QuantizeB4(LoadBf16ToFp32(ps + 0 * pN), LoadBf16ToFp32(ps + 1 * pN), LoadBf16ToFp32(ps + 2 * pN), LoadBf16ToFp32(ps + 3 * pN), _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < K; k += 4, ps += pN * 4)
                {
                    size_t tK = Simd::Min<size_t>(4, K - k);
                    __m512 s0 = LoadBf16ToFp32(ps + 0 * pN);
                    __m512 s1 = tK > 1 ? LoadBf16ToFp32(ps + 1 * pN) : _mm512_setzero_ps();
                    __m512 s2 = tK > 2 ? LoadBf16ToFp32(ps + 2 * pN) : _mm512_setzero_ps();
                    __m512 s3 = tK > 3 ? LoadBf16ToFp32(ps + 3 * pN) : _mm512_setzero_ps();
                    QuantizeB4(s0, s1, s2, s3, _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < KA; k += 4, dst += F * 4)
                    SetZero((uint8_t*)dst);
                _mm512_storeu_si512((__m512i*)(sums + j), _sums);
            }
            for (; j < N; j += F)
            {
                __mmask16 tailN = TailMask16(N - j);
                const uint16_t* ps = src + j;
                __m512 _max = _eps;
                for (k = 0; k < K; ++k, ps += pN)
                    _max = _mm512_max_ps(_max, Abs(LoadBf16ToFp32(ps, tailN)));
                __m512 _norm = _mm512_div_ps(_q, _max);
                _mm512_mask_storeu_ps(scale + j, tailN, _mm512_div_ps(_max, _q));
                __m512i _sums = _mm512_setzero_si512();
                ps = src + j;
                for (k = 0; k < K4; k += 4, ps += pN * 4)
                {
                    QuantizeB4(LoadBf16ToFp32(ps + 0 * pN, tailN), LoadBf16ToFp32(ps + 1 * pN, tailN),
                        LoadBf16ToFp32(ps + 2 * pN, tailN), LoadBf16ToFp32(ps + 3 * pN, tailN), _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < K; k += 4, ps += pN * 4)
                {
                    size_t tK = Simd::Min<size_t>(4, K - k);
                    __m512 s0 = LoadBf16ToFp32(ps + 0 * pN, tailN);
                    __m512 s1 = tK > 1 ? LoadBf16ToFp32(ps + 1 * pN, tailN) : _mm512_setzero_ps();
                    __m512 s2 = tK > 2 ? LoadBf16ToFp32(ps + 2 * pN, tailN) : _mm512_setzero_ps();
                    __m512 s3 = tK > 3 ? LoadBf16ToFp32(ps + 3 * pN, tailN) : _mm512_setzero_ps();
                    QuantizeB4(s0, s1, s2, s3, _norm, _lo, _hi, _sums, dst);
                    dst += F * 4;
                }
                for (; k < KA; k += 4, dst += F * 4)
                    SetZero((uint8_t*)dst);
                _mm512_mask_storeu_epi32(sums + j, tailN, _sums);
            }
        }

        //-----------------------------------------------------------------------------------------

        static SIMD_INLINE void StoreBody1(int32_t* ptr, __m512i val0)
        {
            _mm512_storeu_si512((__m512i*)ptr + 0, val0);
        }

        static SIMD_INLINE void StoreBody2(int32_t* ptr, __m512i val0, __m512i val1)
        {
            _mm512_storeu_si512((__m512i*)ptr + 0, val0);
            _mm512_storeu_si512((__m512i*)ptr + 1, val1);
        }

        template<int M> void Dqip16bGemmV0_GemmBody_2xM(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum)
        {
            __m512i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1, a0, b0, b1;
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
                    if (M > 0x0) c00 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 0), c01 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 1);
                    if (M > 0x1) c10 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 0), c11 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 1);
                    if (M > 0x2) c20 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 0), c21 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 1);
                    if (M > 0x3) c30 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 0), c31 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 1);
                    if (M > 0x4) c40 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 0), c41 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 1);
                    if (M > 0x5) c50 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 0), c51 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 1);
                    if (M > 0x6) c60 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 0), c61 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 1);
                    if (M > 0x7) c70 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 0), c71 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 1);
                    if (M > 0x8) c80 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 0), c81 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 1);
                    if (M > 0x9) c90 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 0), c91 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 1);
                    if (M > 0xA) ca0 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 0), ca1 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 1);
                    if (M > 0xB) cb0 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 0), cb1 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 1);
                }
                else
                {
                    if (M > 0x0) c00 = _mm512_setzero_si512(), c01 = _mm512_setzero_si512();
                    if (M > 0x1) c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512();
                    if (M > 0x2) c20 = _mm512_setzero_si512(), c21 = _mm512_setzero_si512();
                    if (M > 0x3) c30 = _mm512_setzero_si512(), c31 = _mm512_setzero_si512();
                    if (M > 0x4) c40 = _mm512_setzero_si512(), c41 = _mm512_setzero_si512();
                    if (M > 0x5) c50 = _mm512_setzero_si512(), c51 = _mm512_setzero_si512();
                    if (M > 0x6) c60 = _mm512_setzero_si512(), c61 = _mm512_setzero_si512();
                    if (M > 0x7) c70 = _mm512_setzero_si512(), c71 = _mm512_setzero_si512();
                    if (M > 0x8) c80 = _mm512_setzero_si512(), c81 = _mm512_setzero_si512();
                    if (M > 0x9) c90 = _mm512_setzero_si512(), c91 = _mm512_setzero_si512();
                    if (M > 0xA) ca0 = _mm512_setzero_si512(), ca1 = _mm512_setzero_si512();
                    if (M > 0xB) cb0 = _mm512_setzero_si512(), cb1 = _mm512_setzero_si512();
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = _mm512_loadu_si512((__m512i*)B0);
                    b1 = _mm512_loadu_si512((__m512i*)B1);
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
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(ca0, a0, b0), Madd4<false>(ca1, a0, b1);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cb0, a0, b0), Madd4<false>(cb1, a0, b1);
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
                if (M > 0xA) StoreBody2(sum + 0xA * dS, ca0, ca1);
                if (M > 0xB) StoreBody2(sum + 0xB * dS, cb0, cb1);
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 0);
                    if (M > 0x1) c10 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 0);
                    if (M > 0x2) c20 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 0);
                    if (M > 0x3) c30 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 0);
                    if (M > 0x4) c40 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 0);
                    if (M > 0x5) c50 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 0);
                    if (M > 0x6) c60 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 0);
                    if (M > 0x7) c70 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 0);
                    if (M > 0x8) c80 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 0);
                    if (M > 0x9) c90 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 0);
                    if (M > 0xA) ca0 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 0);
                    if (M > 0xB) cb0 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 0);
                }
                else
                {
                    if (M > 0x0) c00 = _mm512_setzero_si512();
                    if (M > 0x1) c10 = _mm512_setzero_si512();
                    if (M > 0x2) c20 = _mm512_setzero_si512();
                    if (M > 0x3) c30 = _mm512_setzero_si512();
                    if (M > 0x4) c40 = _mm512_setzero_si512();
                    if (M > 0x5) c50 = _mm512_setzero_si512();
                    if (M > 0x6) c60 = _mm512_setzero_si512();
                    if (M > 0x7) c70 = _mm512_setzero_si512();
                    if (M > 0x8) c80 = _mm512_setzero_si512();
                    if (M > 0x9) c90 = _mm512_setzero_si512();
                    if (M > 0xA) ca0 = _mm512_setzero_si512();
                    if (M > 0xB) cb0 = _mm512_setzero_si512();
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = _mm512_loadu_si512((__m512i*)B0);
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
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(ca0, a0, b0);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cb0, a0, b0);
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
                if (M > 0xA) StoreBody1(sum + 0xA * dS, ca0);
                if (M > 0xB) StoreBody1(sum + 0xB * dS, cb0);
            }
        }

        typedef void(*Dqip16bGemmV0_GemmBody_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a,
            size_t K, size_t N, int update, const int8_t* B0, int32_t* sum);

        static SIMD_INLINE Dqip16bGemmV0_GemmBody_2xM_Ptr GetDqip16bGemmV0_GemmBody_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return Dqip16bGemmV0_GemmBody_2xM<1>;
            case 0x2: return Dqip16bGemmV0_GemmBody_2xM<2>;
            case 0x3: return Dqip16bGemmV0_GemmBody_2xM<3>;
            case 0x4: return Dqip16bGemmV0_GemmBody_2xM<4>;
            case 0x5: return Dqip16bGemmV0_GemmBody_2xM<5>;
            case 0x6: return Dqip16bGemmV0_GemmBody_2xM<6>;
            case 0x7: return Dqip16bGemmV0_GemmBody_2xM<7>;
            case 0x8: return Dqip16bGemmV0_GemmBody_2xM<8>;
            case 0x9: return Dqip16bGemmV0_GemmBody_2xM<9>;
            case 0xA: return Dqip16bGemmV0_GemmBody_2xM<10>;
            case 0xB: return Dqip16bGemmV0_GemmBody_2xM<11>;
            case 0xC: return Dqip16bGemmV0_GemmBody_2xM<12>;
            }
            assert(0);
            return NULL;
        }

        static void Dqip16bGemmV0_GemmBody(const uint8_t* A, const DqipParam& p, const AlgParam& a,
            size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum)
        {
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

        template<Term16bType term, SimdConvolutionActivationType type, int M> void Dqip16bGemmV0_GemmLast_2xM(const uint8_t* A0,
            const DqipParam& p, const AlgParam& a, size_t K, size_t N, int update, const int8_t* B0, int32_t* sum, int32_t* buf,
            const int32_t* aZero, const __m512i* bSums, const float* aScale, const __m512* bScale, const __m512* bias, const __m512* params, uint8_t* C)
        {
            __m512i c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51, c60, c61, c70, c71, c80, c81, c90, c91, ca0, ca1, cb0, cb1, a0, b0, b1;
            size_t dS = a.cN, dA = a.aK, dC = p.N * a.eC;
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
                    if (M > 0x0) c00 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 0), c01 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 1);
                    if (M > 0x1) c10 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 0), c11 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 1);
                    if (M > 0x2) c20 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 0), c21 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 1);
                    if (M > 0x3) c30 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 0), c31 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 1);
                    if (M > 0x4) c40 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 0), c41 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 1);
                    if (M > 0x5) c50 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 0), c51 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 1);
                    if (M > 0x6) c60 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 0), c61 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 1);
                    if (M > 0x7) c70 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 0), c71 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 1);
                    if (M > 0x8) c80 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 0), c81 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 1);
                    if (M > 0x9) c90 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 0), c91 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 1);
                    if (M > 0xA) ca0 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 0), ca1 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 1);
                    if (M > 0xB) cb0 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 0), cb1 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 1);
                }
                else
                {
                    if (M > 0x0) c00 = _mm512_setzero_si512(), c01 = _mm512_setzero_si512();
                    if (M > 0x1) c10 = _mm512_setzero_si512(), c11 = _mm512_setzero_si512();
                    if (M > 0x2) c20 = _mm512_setzero_si512(), c21 = _mm512_setzero_si512();
                    if (M > 0x3) c30 = _mm512_setzero_si512(), c31 = _mm512_setzero_si512();
                    if (M > 0x4) c40 = _mm512_setzero_si512(), c41 = _mm512_setzero_si512();
                    if (M > 0x5) c50 = _mm512_setzero_si512(), c51 = _mm512_setzero_si512();
                    if (M > 0x6) c60 = _mm512_setzero_si512(), c61 = _mm512_setzero_si512();
                    if (M > 0x7) c70 = _mm512_setzero_si512(), c71 = _mm512_setzero_si512();
                    if (M > 0x8) c80 = _mm512_setzero_si512(), c81 = _mm512_setzero_si512();
                    if (M > 0x9) c90 = _mm512_setzero_si512(), c91 = _mm512_setzero_si512();
                    if (M > 0xA) ca0 = _mm512_setzero_si512(), ca1 = _mm512_setzero_si512();
                    if (M > 0xB) cb0 = _mm512_setzero_si512(), cb1 = _mm512_setzero_si512();
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = _mm512_loadu_si512((__m512i*)B0);
                    b1 = _mm512_loadu_si512((__m512i*)B1);
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
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(ca0, a0, b0), Madd4<false>(ca1, a0, b1);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cb0, a0, b0), Madd4<false>(cb1, a0, b1);
                    B0 += A, B1 += A;
                }
                __mmask16 tail = TailMask16(N - F);
                if (M > 0x0) Save2<term, type>(C + 0x0 * dC, c00, c01, aZero[0x0], bSums, aScale[0x0], bScale, bias, params, tail);
                if (M > 0x1) Save2<term, type>(C + 0x1 * dC, c10, c11, aZero[0x1], bSums, aScale[0x1], bScale, bias, params, tail);
                if (M > 0x2) Save2<term, type>(C + 0x2 * dC, c20, c21, aZero[0x2], bSums, aScale[0x2], bScale, bias, params, tail);
                if (M > 0x3) Save2<term, type>(C + 0x3 * dC, c30, c31, aZero[0x3], bSums, aScale[0x3], bScale, bias, params, tail);
                if (M > 0x4) Save2<term, type>(C + 0x4 * dC, c40, c41, aZero[0x4], bSums, aScale[0x4], bScale, bias, params, tail);
                if (M > 0x5) Save2<term, type>(C + 0x5 * dC, c50, c51, aZero[0x5], bSums, aScale[0x5], bScale, bias, params, tail);
                if (M > 0x6) Save2<term, type>(C + 0x6 * dC, c60, c61, aZero[0x6], bSums, aScale[0x6], bScale, bias, params, tail);
                if (M > 0x7) Save2<term, type>(C + 0x7 * dC, c70, c71, aZero[0x7], bSums, aScale[0x7], bScale, bias, params, tail);
                if (M > 0x8) Save2<term, type>(C + 0x8 * dC, c80, c81, aZero[0x8], bSums, aScale[0x8], bScale, bias, params, tail);
                if (M > 0x9) Save2<term, type>(C + 0x9 * dC, c90, c91, aZero[0x9], bSums, aScale[0x9], bScale, bias, params, tail);
                if (M > 0xA) Save2<term, type>(C + 0xA * dC, ca0, ca1, aZero[0xA], bSums, aScale[0xA], bScale, bias, params, tail);
                if (M > 0xB) Save2<term, type>(C + 0xB * dC, cb0, cb1, aZero[0xB], bSums, aScale[0xB], bScale, bias, params, tail);
            }
            else
            {
                if (update)
                {
                    if (M > 0x0) c00 = _mm512_loadu_si512((__m512i*)(sum + 0x0 * dS) + 0);
                    if (M > 0x1) c10 = _mm512_loadu_si512((__m512i*)(sum + 0x1 * dS) + 0);
                    if (M > 0x2) c20 = _mm512_loadu_si512((__m512i*)(sum + 0x2 * dS) + 0);
                    if (M > 0x3) c30 = _mm512_loadu_si512((__m512i*)(sum + 0x3 * dS) + 0);
                    if (M > 0x4) c40 = _mm512_loadu_si512((__m512i*)(sum + 0x4 * dS) + 0);
                    if (M > 0x5) c50 = _mm512_loadu_si512((__m512i*)(sum + 0x5 * dS) + 0);
                    if (M > 0x6) c60 = _mm512_loadu_si512((__m512i*)(sum + 0x6 * dS) + 0);
                    if (M > 0x7) c70 = _mm512_loadu_si512((__m512i*)(sum + 0x7 * dS) + 0);
                    if (M > 0x8) c80 = _mm512_loadu_si512((__m512i*)(sum + 0x8 * dS) + 0);
                    if (M > 0x9) c90 = _mm512_loadu_si512((__m512i*)(sum + 0x9 * dS) + 0);
                    if (M > 0xA) ca0 = _mm512_loadu_si512((__m512i*)(sum + 0xA * dS) + 0);
                    if (M > 0xB) cb0 = _mm512_loadu_si512((__m512i*)(sum + 0xB * dS) + 0);
                }
                else
                {
                    if (M > 0x0) c00 = _mm512_setzero_si512();
                    if (M > 0x1) c10 = _mm512_setzero_si512();
                    if (M > 0x2) c20 = _mm512_setzero_si512();
                    if (M > 0x3) c30 = _mm512_setzero_si512();
                    if (M > 0x4) c40 = _mm512_setzero_si512();
                    if (M > 0x5) c50 = _mm512_setzero_si512();
                    if (M > 0x6) c60 = _mm512_setzero_si512();
                    if (M > 0x7) c70 = _mm512_setzero_si512();
                    if (M > 0x8) c80 = _mm512_setzero_si512();
                    if (M > 0x9) c90 = _mm512_setzero_si512();
                    if (M > 0xA) ca0 = _mm512_setzero_si512();
                    if (M > 0xB) cb0 = _mm512_setzero_si512();
                }
                for (size_t k0 = 0, k6 = 6 * dA; k0 < K; k0 += 4, k6 += 4)
                {
                    b0 = _mm512_loadu_si512((__m512i*)B0);
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
                    if (M > 0xA) a0 = Set4(A4 + k6), Madd4<false>(ca0, a0, b0);
                    if (M > 0xB) a0 = Set4(A5 + k6), Madd4<false>(cb0, a0, b0);
                    B0 += A;
                }
                __mmask16 tail = TailMask16(N);
                if (M > 0x0) Save1<term, type>(C + 0x0 * dC, c00, aZero[0x0], bSums, aScale[0x0], bScale, bias, params, tail);
                if (M > 0x1) Save1<term, type>(C + 0x1 * dC, c10, aZero[0x1], bSums, aScale[0x1], bScale, bias, params, tail);
                if (M > 0x2) Save1<term, type>(C + 0x2 * dC, c20, aZero[0x2], bSums, aScale[0x2], bScale, bias, params, tail);
                if (M > 0x3) Save1<term, type>(C + 0x3 * dC, c30, aZero[0x3], bSums, aScale[0x3], bScale, bias, params, tail);
                if (M > 0x4) Save1<term, type>(C + 0x4 * dC, c40, aZero[0x4], bSums, aScale[0x4], bScale, bias, params, tail);
                if (M > 0x5) Save1<term, type>(C + 0x5 * dC, c50, aZero[0x5], bSums, aScale[0x5], bScale, bias, params, tail);
                if (M > 0x6) Save1<term, type>(C + 0x6 * dC, c60, aZero[0x6], bSums, aScale[0x6], bScale, bias, params, tail);
                if (M > 0x7) Save1<term, type>(C + 0x7 * dC, c70, aZero[0x7], bSums, aScale[0x7], bScale, bias, params, tail);
                if (M > 0x8) Save1<term, type>(C + 0x8 * dC, c80, aZero[0x8], bSums, aScale[0x8], bScale, bias, params, tail);
                if (M > 0x9) Save1<term, type>(C + 0x9 * dC, c90, aZero[0x9], bSums, aScale[0x9], bScale, bias, params, tail);
                if (M > 0xA) Save1<term, type>(C + 0xA * dC, ca0, aZero[0xA], bSums, aScale[0xA], bScale, bias, params, tail);
                if (M > 0xB) Save1<term, type>(C + 0xB * dC, cb0, aZero[0xB], bSums, aScale[0xB], bScale, bias, params, tail);
            }
        }

        typedef void(*Dqip16bGemmV0_GemmLast_2xM_Ptr)(const uint8_t* A0, const DqipParam& p, const AlgParam& a, size_t K, size_t N, int update, const int8_t* B0, int32_t* sum,
            int32_t* buf, const int32_t* aZero, const __m512i* bSums, const float* aScale, const __m512* bScale, const __m512* bias, const __m512* params, uint8_t* C);

        template<Term16bType term, SimdConvolutionActivationType type> static SIMD_INLINE Dqip16bGemmV0_GemmLast_2xM_Ptr GetDqip16bGemmV0_GemmLast_2xM(size_t M)
        {
            switch (M)
            {
            case 0x0: return NULL;
            case 0x1: return Dqip16bGemmV0_GemmLast_2xM<term, type, 1>;
            case 0x2: return Dqip16bGemmV0_GemmLast_2xM<term, type, 2>;
            case 0x3: return Dqip16bGemmV0_GemmLast_2xM<term, type, 3>;
            case 0x4: return Dqip16bGemmV0_GemmLast_2xM<term, type, 4>;
            case 0x5: return Dqip16bGemmV0_GemmLast_2xM<term, type, 5>;
            case 0x6: return Dqip16bGemmV0_GemmLast_2xM<term, type, 6>;
            case 0x7: return Dqip16bGemmV0_GemmLast_2xM<term, type, 7>;
            case 0x8: return Dqip16bGemmV0_GemmLast_2xM<term, type, 8>;
            case 0x9: return Dqip16bGemmV0_GemmLast_2xM<term, type, 9>;
            case 0xA: return Dqip16bGemmV0_GemmLast_2xM<term, type, 10>;
            case 0xB: return Dqip16bGemmV0_GemmLast_2xM<term, type, 11>;
            case 0xC: return Dqip16bGemmV0_GemmLast_2xM<term, type, 12>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Dqip16bGemmV0_GemmLast(const uint8_t* A, const DqipParam& p,
            const AlgParam& a, size_t M, size_t N, size_t K, int update, const int8_t* B, int32_t* sum, int32_t* buf,
            const int32_t* aZero, const int32_t* bSums, const float* aScale, const float* bScale, const float* bias, const float* params, uint8_t* C)
        {
            size_t n = 12;
            size_t Mn = AlignLoAny(M, n), m = M - Mn;
            size_t dS = a.cN, dA = a.aK, dC = p.N * a.eC;
            Dqip16bGemmV0_GemmLast_2xM_Ptr gemm_2xN = GetDqip16bGemmV0_GemmLast_2xM<term, type>(n);
            Dqip16bGemmV0_GemmLast_2xM_Ptr gemm_2xM = GetDqip16bGemmV0_GemmLast_2xM<term, type>(m);

            __m512i _bSums[2];
            __m512 _bScale[2], _bias[2], _params[2];
            if (type != SimdConvolutionActivationIdentity)
            {
                _params[0] = _mm512_set1_ps(params[0]);
                _params[1] = _mm512_set1_ps(params[1]);
            }
            for (size_t j = 0; j < N; j += DF)
            {
                size_t dN = Simd::Min(DF, N - j);
                _bSums[0] = _mm512_loadu_si512((__m512i*)(bSums + j) + 0);
                _bScale[0] = _mm512_loadu_ps(bScale + j + 0);
                _bias[0] = _mm512_loadu_ps(bias + j + 0);
                if (dN > F)
                {
                    _bSums[1] = _mm512_loadu_si512((__m512i*)(bSums + j) + 1);
                    _bScale[1] = _mm512_loadu_ps(bScale + j + F);
                    _bias[1] = _mm512_loadu_ps(bias + j + F);
                }
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + j + 0);
                    if (dN > F)
                        _params[1] = _mm512_loadu_ps(params + j + F);
                }
                size_t i = 0;
                for (; i < Mn; i += n)
                    gemm_2xN(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, aZero + i, _bSums, aScale + i, _bScale, _bias, _params, C + i * dC);
                for (; i < M; i += m)
                    gemm_2xM(A + i * dA, p, a, K, dN, update, B, sum + i * dS, buf, aZero + i, _bSums, aScale + i, _bScale, _bias, _params, C + i * dC);
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
            : Avx512bw::SynetDynamicQuantizedInnerProduct16bGemmV0(p)
        {
            SetAlgParam(F, 12, F * 2, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            _alg.overflow = 0;
            _gemmBody = Dqip16bGemmV0_GemmBody;
            SetGemmLast(p, _gemmLast);
        }
    }
#endif
}
