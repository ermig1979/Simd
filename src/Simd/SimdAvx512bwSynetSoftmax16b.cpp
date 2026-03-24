/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdInterleave.h"
#include "Simd/SimdDeinterleave.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)     
    namespace Avx512bw
    {
        void SynetSoftmax16b21(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            Exp exp;
            size_t outerF = Simd::AlignLo(outer, F), tail = outer - outerF;
            for (size_t o = 0; o < outerF; o += F)
            {
                __m512i s01 = _mm512_loadu_si512((__m512i*)src);
                __m512 ss0 = BFloat16ToFloat32Even(s01);
                __m512 ss1 = BFloat16ToFloat32Odd(s01);
                __m512 max = _mm512_max_ps(ss0, ss1);
                __m512 exp0 = exp.Exponent(_mm512_sub_ps(ss0, max));
                __m512 exp1 = exp.Exponent(_mm512_sub_ps(ss1, max));
                __m512 sum = _mm512_add_ps(exp0, exp1);
                __m512 d0 = _mm512_div_ps(exp0, sum);
                __m512 d1 = _mm512_div_ps(exp1, sum);
                _mm512_storeu_si512((__m512i*)dst, Float32ToBFloat16Interlived(d0, d1));
                src += DF;
                dst += DF;
            }
            if (tail)
            {
                __mmask16 mask = TailMask32(tail * 2);
                __m512i s01 = _mm512_maskz_loadu_epi16(mask, src);
                __m512 ss0 = BFloat16ToFloat32Even(s01);
                __m512 ss1 = BFloat16ToFloat32Odd(s01);
                __m512 max = _mm512_max_ps(ss0, ss1);
                __m512 exp0 = exp.Exponent(_mm512_sub_ps(ss0, max));
                __m512 exp1 = exp.Exponent(_mm512_sub_ps(ss1, max));
                __m512 sum = _mm512_add_ps(exp0, exp1);
                __m512 d0 = _mm512_div_ps(exp0, sum);
                __m512 d1 = _mm512_div_ps(exp1, sum);
                _mm512_mask_storeu_epi16(dst, mask, Float32ToBFloat16Interlived(d0, d1));
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetSoftmax16b31To32f(const __m512i src[2], __m512 dst[3])
        {
            static const __m512i IDX0 = SIMD_MM512_SETR_EPI16(
                0x30, 0x00, 0x30, 0x03, 0x30, 0x06, 0x30, 0x09, 0x30, 0x0C, 0x30, 0x0F, 0x30, 0x12, 0x30, 0x15,
                0x30, 0x18, 0x30, 0x1B, 0x30, 0x1E, 0x30, 0x21, 0x30, 0x24, 0x30, 0x27, 0x30, 0x2A, 0x30, 0x2D);
            static const __m512i IDX1 = SIMD_MM512_SETR_EPI16(
                0x30, 0x01, 0x30, 0x04, 0x30, 0x07, 0x30, 0x0A, 0x30, 0x0D, 0x30, 0x10, 0x30, 0x13, 0x30, 0x16,
                0x30, 0x19, 0x30, 0x1C, 0x30, 0x1F, 0x30, 0x22, 0x30, 0x25, 0x30, 0x28, 0x30, 0x2B, 0x30, 0x2E);
            static const __m512i IDX2 = SIMD_MM512_SETR_EPI16(
                0x30, 0x02, 0x30, 0x05, 0x30, 0x08, 0x30, 0x0B, 0x30, 0x0E, 0x30, 0x11, 0x30, 0x14, 0x30, 0x17,
                0x30, 0x1A, 0x30, 0x1D, 0x30, 0x20, 0x30, 0x23, 0x30, 0x26, 0x30, 0x29, 0x30, 0x2C, 0x30, 0x2F);
            dst[0] = _mm512_castsi512_ps(_mm512_permutex2var_epi16(src[0], IDX0, src[1]));
            dst[1] = _mm512_castsi512_ps(_mm512_permutex2var_epi16(src[0], IDX1, src[1]));
            dst[2] = _mm512_castsi512_ps(_mm512_permutex2var_epi16(src[0], IDX2, src[1]));
        }

        SIMD_INLINE void SynetSoftmax16b31To16b(const __m512 src[3], __m512i dst[2])
        {
            __m512i s01 = Float32ToBFloat16Interlived(src[0], src[1]);
            __m512i s2 = Float32ToBFloat16(src[2]);
            static const __m512i IDX0 = SIMD_MM512_SETR_EPI16(
                0x00, 0x01, 0x20, 0x02, 0x03, 0x22, 0x04, 0x05, 0x24, 0x06, 0x07, 0x26, 0x08, 0x09, 0x28, 0x0A,
                0x0B, 0x2A, 0x0C, 0x0D, 0x2C, 0x0E, 0x0F, 0x2E, 0x10, 0x11, 0x30, 0x12, 0x13, 0x32, 0x14, 0x15);
            static const __m512i IDX1 = SIMD_MM512_SETR_EPI16(
                0x34, 0x16, 0x17, 0x36, 0x18, 0x19, 0x38, 0x1A, 0x1B, 0x3A, 0x1C, 0x1D, 0x3C, 0x1E, 0x1F, 0x3E,
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
            dst[0] = _mm512_permutex2var_epi16(s01, IDX0, s2);
            dst[1] = _mm512_permutex2var_epi16(s01, IDX1, s2);
        }

        SIMD_INLINE void SynetSoftmax16b31(const Exp& exp, __m512 buf[3])
        {
            __m512 max = _mm512_max_ps(buf[0], _mm512_max_ps(buf[1], buf[2]));
            buf[0] = exp.Exponent(_mm512_sub_ps(buf[0], max));
            buf[1] = exp.Exponent(_mm512_sub_ps(buf[1], max));
            buf[2] = exp.Exponent(_mm512_sub_ps(buf[2], max));
            __m512 sum = _mm512_add_ps(buf[0], _mm512_add_ps(buf[1], buf[2]));
            buf[0] = _mm512_div_ps(buf[0], sum);
            buf[1] = _mm512_div_ps(buf[1], sum);
            buf[2] = _mm512_div_ps(buf[2], sum);
        }

        void SynetSoftmax16b31(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            Exp exp;
            __m512i b16b[2];
            __m512 b32f[3];
            size_t outerF = Simd::AlignLo(outer, F), tail = outer - outerF;
            for (size_t o = 0; o < outerF; o += F)
            {
                b16b[0] = Load(src, src + F, __mmask16(-1));
                b16b[1] = _mm512_castsi256_si512(_mm256_loadu_si256((__m256i*)src + 2));
                SynetSoftmax16b31To32f(b16b, b32f);
                SynetSoftmax16b31(exp, b32f);
                SynetSoftmax16b31To16b(b32f, b16b);
                _mm512_storeu_si512((__m512i*)dst, b16b[0]);
                _mm256_storeu_si256((__m256i*)dst + 2, _mm512_castsi512_si256(b16b[1]));
                src += 3 * F;
                dst += 3 * F;
            }
            if (tail)
            {
                __mmask32 mask01 = TailMask32(tail * 3);
                __mmask16 mask2 = TailMask16(tail * 3 - DF);
                b16b[0] = _mm512_maskz_loadu_epi16(mask01, src);
                b16b[1] = _mm512_castsi256_si512(_mm256_maskz_loadu_epi16(mask2, src + DF));
                SynetSoftmax16b31To32f(b16b, b32f);
                SynetSoftmax16b31(exp, b32f);
                SynetSoftmax16b31To16b(b32f, b16b);
                _mm512_mask_storeu_epi16(dst, mask01, b16b[0]);
                _mm256_mask_storeu_epi16(dst + DF, mask2, _mm512_castsi512_si256(b16b[1]));
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void LoadTansp16x16(const uint16_t* src, size_t srcStride, size_t cols, float* dst, __m512& max)
        {
            __m512 a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aA, aB, aC, aD, aE, aF, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, bA, bB, bC, bD, bE, bF;

            static const __m256i def = PackFloat32ToBFloat16(_mm512_set1_ps(-FLT_MAX));
            __mmask16 srcMask = __mmask16(-1) >> (16 - cols);
            a0 = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0x0 * srcStride));
            a1 = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0x1 * srcStride));
            a2 = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0x2 * srcStride));
            a3 = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0x3 * srcStride));
            a4 = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0x4 * srcStride));
            a5 = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0x5 * srcStride));
            a6 = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0x6 * srcStride));
            a7 = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0x7 * srcStride));
            a8 = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0x8 * srcStride));
            a9 = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0x9 * srcStride));
            aA = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0xA * srcStride));
            aB = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0xB * srcStride));
            aC = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0xC * srcStride));
            aD = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0xD * srcStride));
            aE = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0xE * srcStride));
            aF = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + 0xF * srcStride));

            b0 = _mm512_unpacklo_ps(a0, a2);
            b1 = _mm512_unpacklo_ps(a1, a3);
            b2 = _mm512_unpackhi_ps(a0, a2);
            b3 = _mm512_unpackhi_ps(a1, a3);
            b4 = _mm512_unpacklo_ps(a4, a6);
            b5 = _mm512_unpacklo_ps(a5, a7);
            b6 = _mm512_unpackhi_ps(a4, a6);
            b7 = _mm512_unpackhi_ps(a5, a7);
            b8 = _mm512_unpacklo_ps(a8, aA);
            b9 = _mm512_unpacklo_ps(a9, aB);
            bA = _mm512_unpackhi_ps(a8, aA);
            bB = _mm512_unpackhi_ps(a9, aB);
            bC = _mm512_unpacklo_ps(aC, aE);
            bD = _mm512_unpacklo_ps(aD, aF);
            bE = _mm512_unpackhi_ps(aC, aE);
            bF = _mm512_unpackhi_ps(aD, aF);

            a0 = _mm512_unpacklo_ps(b0, b1);
            a1 = _mm512_unpackhi_ps(b0, b1);
            a2 = _mm512_unpacklo_ps(b2, b3);
            a3 = _mm512_unpackhi_ps(b2, b3);
            a4 = _mm512_unpacklo_ps(b4, b5);
            a5 = _mm512_unpackhi_ps(b4, b5);
            a6 = _mm512_unpacklo_ps(b6, b7);
            a7 = _mm512_unpackhi_ps(b6, b7);
            a8 = _mm512_unpacklo_ps(b8, b9);
            a9 = _mm512_unpackhi_ps(b8, b9);
            aA = _mm512_unpacklo_ps(bA, bB);
            aB = _mm512_unpackhi_ps(bA, bB);
            aC = _mm512_unpacklo_ps(bC, bD);
            aD = _mm512_unpackhi_ps(bC, bD);
            aE = _mm512_unpacklo_ps(bE, bF);
            aF = _mm512_unpackhi_ps(bE, bF);

            b0 = _mm512_shuffle_f32x4(a0, a4, 0x44);
            b1 = _mm512_shuffle_f32x4(a1, a5, 0x44);
            b2 = _mm512_shuffle_f32x4(a2, a6, 0x44);
            b3 = _mm512_shuffle_f32x4(a3, a7, 0x44);
            b4 = _mm512_shuffle_f32x4(a0, a4, 0xEE);
            b5 = _mm512_shuffle_f32x4(a1, a5, 0xEE);
            b6 = _mm512_shuffle_f32x4(a2, a6, 0xEE);
            b7 = _mm512_shuffle_f32x4(a3, a7, 0xEE);
            b8 = _mm512_shuffle_f32x4(a8, aC, 0x44);
            b9 = _mm512_shuffle_f32x4(a9, aD, 0x44);
            bA = _mm512_shuffle_f32x4(aA, aE, 0x44);
            bB = _mm512_shuffle_f32x4(aB, aF, 0x44);
            bC = _mm512_shuffle_f32x4(a8, aC, 0xEE);
            bD = _mm512_shuffle_f32x4(a9, aD, 0xEE);
            bE = _mm512_shuffle_f32x4(aA, aE, 0xEE);
            bF = _mm512_shuffle_f32x4(aB, aF, 0xEE);

            a0 = _mm512_shuffle_f32x4(b0, b8, 0x88);
            a1 = _mm512_shuffle_f32x4(b1, b9, 0x88);
            a2 = _mm512_shuffle_f32x4(b2, bA, 0x88);
            a3 = _mm512_shuffle_f32x4(b3, bB, 0x88);
            a4 = _mm512_shuffle_f32x4(b0, b8, 0xDD);
            a5 = _mm512_shuffle_f32x4(b1, b9, 0xDD);
            a6 = _mm512_shuffle_f32x4(b2, bA, 0xDD);
            a7 = _mm512_shuffle_f32x4(b3, bB, 0xDD);
            a8 = _mm512_shuffle_f32x4(b4, bC, 0x88);
            a9 = _mm512_shuffle_f32x4(b5, bD, 0x88);
            aA = _mm512_shuffle_f32x4(b6, bE, 0x88);
            aB = _mm512_shuffle_f32x4(b7, bF, 0x88);
            aC = _mm512_shuffle_f32x4(b4, bC, 0xDD);
            aD = _mm512_shuffle_f32x4(b5, bD, 0xDD);
            aE = _mm512_shuffle_f32x4(b6, bE, 0xDD);
            aF = _mm512_shuffle_f32x4(b7, bF, 0xDD);

            max = _mm512_max_ps(max, a0);
            max = _mm512_max_ps(max, a1);
            max = _mm512_max_ps(max, a2);
            max = _mm512_max_ps(max, a3);
            max = _mm512_max_ps(max, a4);
            max = _mm512_max_ps(max, a5);
            max = _mm512_max_ps(max, a6);
            max = _mm512_max_ps(max, a7);
            max = _mm512_max_ps(max, a8);
            max = _mm512_max_ps(max, a9);
            max = _mm512_max_ps(max, aA);
            max = _mm512_max_ps(max, aB);
            max = _mm512_max_ps(max, aC);
            max = _mm512_max_ps(max, aD);
            max = _mm512_max_ps(max, aE);
            max = _mm512_max_ps(max, aF);

            _mm512_storeu_ps(dst + 0x0 * F, a0);
            _mm512_storeu_ps(dst + 0x1 * F, a1);
            _mm512_storeu_ps(dst + 0x2 * F, a2);
            _mm512_storeu_ps(dst + 0x3 * F, a3);
            _mm512_storeu_ps(dst + 0x4 * F, a4);
            _mm512_storeu_ps(dst + 0x5 * F, a5);
            _mm512_storeu_ps(dst + 0x6 * F, a6);
            _mm512_storeu_ps(dst + 0x7 * F, a7);
            _mm512_storeu_ps(dst + 0x8 * F, a8);
            _mm512_storeu_ps(dst + 0x9 * F, a9);
            _mm512_storeu_ps(dst + 0xA * F, aA);
            _mm512_storeu_ps(dst + 0xB * F, aB);
            _mm512_storeu_ps(dst + 0xC * F, aC);
            _mm512_storeu_ps(dst + 0xD * F, aD);
            _mm512_storeu_ps(dst + 0xE * F, aE);
            _mm512_storeu_ps(dst + 0xF * F, aF);
        }

        SIMD_INLINE void LoadTansp16x16(const uint16_t* src, size_t srcStride, size_t cols, size_t rows, float* dst, __m512& max)
        {
            __m512 a[16], b[16];

            __mmask16 srcMask = __mmask16(-1) >> (16 - cols);
            static const __m256i def = PackFloat32ToBFloat16(_mm512_set1_ps(-FLT_MAX));
            for (size_t r = 0; r < rows; ++r)
                a[r] = BFloat16ToFloat32(_mm256_mask_loadu_epi16(def, srcMask, src + r * srcStride));

            for (size_t r = 0; r < rows; r += 4)
            {
                b[r + 0] = _mm512_unpacklo_ps(a[r + 0], a[r + 2]);
                b[r + 1] = _mm512_unpacklo_ps(a[r + 1], a[r + 3]);
                b[r + 2] = _mm512_unpackhi_ps(a[r + 0], a[r + 2]);
                b[r + 3] = _mm512_unpackhi_ps(a[r + 1], a[r + 3]);
            }

            for (size_t r = 0; r < rows; r += 4)
            {
                a[r + 0] = _mm512_unpacklo_ps(b[r + 0], b[r + 1]);
                a[r + 1] = _mm512_unpackhi_ps(b[r + 0], b[r + 1]);
                a[r + 2] = _mm512_unpacklo_ps(b[r + 2], b[r + 3]);
                a[r + 3] = _mm512_unpackhi_ps(b[r + 2], b[r + 3]);
            }

            for (size_t i = 0; i < 4; i += 1)
            {
                b[0x0 + i] = _mm512_shuffle_f32x4(a[0x0 + i], a[0x4 + i], 0x44);
                b[0x4 + i] = _mm512_shuffle_f32x4(a[0x0 + i], a[0x4 + i], 0xEE);
                b[0x8 + i] = _mm512_shuffle_f32x4(a[0x8 + i], a[0xC + i], 0x44);
                b[0xC + i] = _mm512_shuffle_f32x4(a[0x8 + i], a[0xC + i], 0xEE);
            }

            for (size_t i = 0; i < 4; i += 1)
            {
                a[0x0 + i] = _mm512_shuffle_f32x4(b[0x0 + i], b[0x8 + i], 0x88);
                a[0x4 + i] = _mm512_shuffle_f32x4(b[0x0 + i], b[0x8 + i], 0xDD);
                a[0x8 + i] = _mm512_shuffle_f32x4(b[0x4 + i], b[0xC + i], 0x88);
                a[0xC + i] = _mm512_shuffle_f32x4(b[0x4 + i], b[0xC + i], 0xDD);
            }

            for (size_t c = 0; c < cols; ++c)
            {
                max = _mm512_max_ps(max, a[c]);
                _mm512_storeu_ps(dst + c * F, a[c]);
            }
        }

        SIMD_INLINE void StoreTansp16x16(const float* src, size_t cols, __m512 k, uint16_t* dst, size_t dstStride)
        {
            __m512 a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aA, aB, aC, aD, aE, aF, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, bA, bB, bC, bD, bE, bF;

            a0 = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0x0 * F));
            a1 = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0x1 * F));
            a2 = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0x2 * F));
            a3 = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0x3 * F));
            a4 = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0x4 * F));
            a5 = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0x5 * F));
            a6 = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0x6 * F));
            a7 = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0x7 * F));
            a8 = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0x8 * F));
            a9 = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0x9 * F));
            aA = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0xA * F));
            aB = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0xB * F));
            aC = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0xC * F));
            aD = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0xD * F));
            aE = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0xE * F));
            aF = _mm512_mul_ps(k, _mm512_loadu_ps(src + 0xF * F));

            b0 = _mm512_unpacklo_ps(a0, a2);
            b1 = _mm512_unpacklo_ps(a1, a3);
            b2 = _mm512_unpackhi_ps(a0, a2);
            b3 = _mm512_unpackhi_ps(a1, a3);
            b4 = _mm512_unpacklo_ps(a4, a6);
            b5 = _mm512_unpacklo_ps(a5, a7);
            b6 = _mm512_unpackhi_ps(a4, a6);
            b7 = _mm512_unpackhi_ps(a5, a7);
            b8 = _mm512_unpacklo_ps(a8, aA);
            b9 = _mm512_unpacklo_ps(a9, aB);
            bA = _mm512_unpackhi_ps(a8, aA);
            bB = _mm512_unpackhi_ps(a9, aB);
            bC = _mm512_unpacklo_ps(aC, aE);
            bD = _mm512_unpacklo_ps(aD, aF);
            bE = _mm512_unpackhi_ps(aC, aE);
            bF = _mm512_unpackhi_ps(aD, aF);

            a0 = _mm512_unpacklo_ps(b0, b1);
            a1 = _mm512_unpackhi_ps(b0, b1);
            a2 = _mm512_unpacklo_ps(b2, b3);
            a3 = _mm512_unpackhi_ps(b2, b3);
            a4 = _mm512_unpacklo_ps(b4, b5);
            a5 = _mm512_unpackhi_ps(b4, b5);
            a6 = _mm512_unpacklo_ps(b6, b7);
            a7 = _mm512_unpackhi_ps(b6, b7);
            a8 = _mm512_unpacklo_ps(b8, b9);
            a9 = _mm512_unpackhi_ps(b8, b9);
            aA = _mm512_unpacklo_ps(bA, bB);
            aB = _mm512_unpackhi_ps(bA, bB);
            aC = _mm512_unpacklo_ps(bC, bD);
            aD = _mm512_unpackhi_ps(bC, bD);
            aE = _mm512_unpacklo_ps(bE, bF);
            aF = _mm512_unpackhi_ps(bE, bF);

            b0 = _mm512_shuffle_f32x4(a0, a4, 0x44);
            b1 = _mm512_shuffle_f32x4(a1, a5, 0x44);
            b2 = _mm512_shuffle_f32x4(a2, a6, 0x44);
            b3 = _mm512_shuffle_f32x4(a3, a7, 0x44);
            b4 = _mm512_shuffle_f32x4(a0, a4, 0xEE);
            b5 = _mm512_shuffle_f32x4(a1, a5, 0xEE);
            b6 = _mm512_shuffle_f32x4(a2, a6, 0xEE);
            b7 = _mm512_shuffle_f32x4(a3, a7, 0xEE);
            b8 = _mm512_shuffle_f32x4(a8, aC, 0x44);
            b9 = _mm512_shuffle_f32x4(a9, aD, 0x44);
            bA = _mm512_shuffle_f32x4(aA, aE, 0x44);
            bB = _mm512_shuffle_f32x4(aB, aF, 0x44);
            bC = _mm512_shuffle_f32x4(a8, aC, 0xEE);
            bD = _mm512_shuffle_f32x4(a9, aD, 0xEE);
            bE = _mm512_shuffle_f32x4(aA, aE, 0xEE);
            bF = _mm512_shuffle_f32x4(aB, aF, 0xEE);

            a0 = _mm512_shuffle_f32x4(b0, b8, 0x88);
            a1 = _mm512_shuffle_f32x4(b1, b9, 0x88);
            a2 = _mm512_shuffle_f32x4(b2, bA, 0x88);
            a3 = _mm512_shuffle_f32x4(b3, bB, 0x88);
            a4 = _mm512_shuffle_f32x4(b0, b8, 0xDD);
            a5 = _mm512_shuffle_f32x4(b1, b9, 0xDD);
            a6 = _mm512_shuffle_f32x4(b2, bA, 0xDD);
            a7 = _mm512_shuffle_f32x4(b3, bB, 0xDD);
            a8 = _mm512_shuffle_f32x4(b4, bC, 0x88);
            a9 = _mm512_shuffle_f32x4(b5, bD, 0x88);
            aA = _mm512_shuffle_f32x4(b6, bE, 0x88);
            aB = _mm512_shuffle_f32x4(b7, bF, 0x88);
            aC = _mm512_shuffle_f32x4(b4, bC, 0xDD);
            aD = _mm512_shuffle_f32x4(b5, bD, 0xDD);
            aE = _mm512_shuffle_f32x4(b6, bE, 0xDD);
            aF = _mm512_shuffle_f32x4(b7, bF, 0xDD);

            __mmask16 dstMask = __mmask16(-1) >> (16 - cols);
            _mm256_mask_storeu_epi16(dst + 0x0 * dstStride, dstMask, PackFloat32ToBFloat16(a0));
            _mm256_mask_storeu_epi16(dst + 0x1 * dstStride, dstMask, PackFloat32ToBFloat16(a1));
            _mm256_mask_storeu_epi16(dst + 0x2 * dstStride, dstMask, PackFloat32ToBFloat16(a2));
            _mm256_mask_storeu_epi16(dst + 0x3 * dstStride, dstMask, PackFloat32ToBFloat16(a3));
            _mm256_mask_storeu_epi16(dst + 0x4 * dstStride, dstMask, PackFloat32ToBFloat16(a4));
            _mm256_mask_storeu_epi16(dst + 0x5 * dstStride, dstMask, PackFloat32ToBFloat16(a5));
            _mm256_mask_storeu_epi16(dst + 0x6 * dstStride, dstMask, PackFloat32ToBFloat16(a6));
            _mm256_mask_storeu_epi16(dst + 0x7 * dstStride, dstMask, PackFloat32ToBFloat16(a7));
            _mm256_mask_storeu_epi16(dst + 0x8 * dstStride, dstMask, PackFloat32ToBFloat16(a8));
            _mm256_mask_storeu_epi16(dst + 0x9 * dstStride, dstMask, PackFloat32ToBFloat16(a9));
            _mm256_mask_storeu_epi16(dst + 0xA * dstStride, dstMask, PackFloat32ToBFloat16(aA));
            _mm256_mask_storeu_epi16(dst + 0xB * dstStride, dstMask, PackFloat32ToBFloat16(aB));
            _mm256_mask_storeu_epi16(dst + 0xC * dstStride, dstMask, PackFloat32ToBFloat16(aC));
            _mm256_mask_storeu_epi16(dst + 0xD * dstStride, dstMask, PackFloat32ToBFloat16(aD));
            _mm256_mask_storeu_epi16(dst + 0xE * dstStride, dstMask, PackFloat32ToBFloat16(aE));
            _mm256_mask_storeu_epi16(dst + 0xF * dstStride, dstMask, PackFloat32ToBFloat16(aF));
        }

        SIMD_INLINE void StoreTansp16x16(const float* src, size_t cols, size_t rows, __m512 k, uint16_t* dst, size_t dstStride)
        {
            __m512 a[16], b[16];

            for (size_t c = 0; c < cols; ++c)
                a[c] = _mm512_mul_ps(k, _mm512_loadu_ps(src + c * F));

            for (size_t i = 0; i < 4; i += 1)
            {
                b[0x0 + i] = _mm512_shuffle_f32x4(a[0x0 + i], a[0x4 + i], 0x44);
                b[0x4 + i] = _mm512_shuffle_f32x4(a[0x0 + i], a[0x4 + i], 0xEE);
                b[0x8 + i] = _mm512_shuffle_f32x4(a[0x8 + i], a[0xC + i], 0x44);
                b[0xC + i] = _mm512_shuffle_f32x4(a[0x8 + i], a[0xC + i], 0xEE);
            }

            for (size_t i = 0; i < 4; i += 1)
            {
                a[0x0 + i] = _mm512_shuffle_f32x4(b[0x0 + i], b[0x8 + i], 0x88);
                a[0x4 + i] = _mm512_shuffle_f32x4(b[0x0 + i], b[0x8 + i], 0xDD);
                a[0x8 + i] = _mm512_shuffle_f32x4(b[0x4 + i], b[0xC + i], 0x88);
                a[0xC + i] = _mm512_shuffle_f32x4(b[0x4 + i], b[0xC + i], 0xDD);
            }

            for (size_t r = 0; r < rows; r += 4)
            {
                b[r + 0] = _mm512_unpacklo_ps(a[r + 0], a[r + 2]);
                b[r + 1] = _mm512_unpacklo_ps(a[r + 1], a[r + 3]);
                b[r + 2] = _mm512_unpackhi_ps(a[r + 0], a[r + 2]);
                b[r + 3] = _mm512_unpackhi_ps(a[r + 1], a[r + 3]);
            }

            for (size_t r = 0; r < rows; r += 4)
            {
                a[r + 0] = _mm512_unpacklo_ps(b[r + 0], b[r + 1]);
                a[r + 1] = _mm512_unpackhi_ps(b[r + 0], b[r + 1]);
                a[r + 2] = _mm512_unpacklo_ps(b[r + 2], b[r + 3]);
                a[r + 3] = _mm512_unpackhi_ps(b[r + 2], b[r + 3]);
            }

            __mmask16 dstMask = __mmask16(-1) >> (16 - cols);
            for (size_t r = 0; r < rows; ++r)
                _mm256_mask_storeu_epi16(dst + r * dstStride, dstMask, PackFloat32ToBFloat16(a[r]));
        }

        void SynetSoftmax16bX1(const uint16_t* src, size_t outer, size_t count, uint16_t* dst)
        {
            size_t o = 0, c = 0, outerF = AlignLo(outer, F), countF = AlignLo(count, F);
            Array32f buf(AlignHi(count, F) * F);
            Exp exp;
            for (; o < outerF; o += F)
            {
                __m512 _max = _mm512_set1_ps(-FLT_MAX);
                for (c = 0; c < countF; c += F)
                    LoadTansp16x16(src + c, count, F, buf.data + c * F, _max);
                if (c < count)
                    LoadTansp16x16(src + c, count, count - c, buf.data + c * F, _max);
                __m512 _sum = _mm512_setzero_ps();
                for (c = 0; c < count; ++c)
                {
                    __m512 _exp = exp.Exponent(_mm512_sub_ps(_mm512_loadu_ps(buf.data + c * F), _max));
                    _sum = _mm512_add_ps(_sum, _exp);
                    _mm512_storeu_ps(buf.data + c * F, _exp);
                }
                __m512 _k = _mm512_div_ps(_mm512_set1_ps(1.0f), _sum);
                for (c = 0; c < countF; c += F)
                    StoreTansp16x16(buf.data + c * F, F, _k, dst + c, count);
                if (c < count)
                    StoreTansp16x16(buf.data + c * F, count - c, _k, dst + c, count);
                src += count * F;
                dst += count * F;
            }
            if (o < outer)
            {
                buf.Clear();
                __m512 _max = _mm512_set1_ps(-FLT_MAX);
                for (c = 0; c < countF; c += F)
                    LoadTansp16x16(src + c, count, F, outer - o, buf.data + c * F, _max);
                if (c < count)
                    LoadTansp16x16(src + c, count, count - c, outer - o, buf.data + c * F, _max);
                __m512 _sum = _mm512_setzero_ps();
                for (c = 0; c < count; ++c)
                {
                    __m512 _exp = exp.Exponent(_mm512_sub_ps(_mm512_loadu_ps(buf.data + c * F), _max));
                    _sum = _mm512_add_ps(_sum, _exp);
                    _mm512_storeu_ps(buf.data + c * F, _exp);
                }
                __m512 _k = _mm512_div_ps(_mm512_set1_ps(1.0f), _sum);
                for (c = 0; c < countF; c += F)
                    StoreTansp16x16(buf.data + c * F, F, outer - o, _k, dst + c, count);
                if (c < count)
                    StoreTansp16x16(buf.data + c * F, count - c, outer - o, _k, dst + c, count);
            }
        }

        //--------------------------------------------------------------------------------------------------

        void SynetSoftmax16b(const uint16_t* src, size_t outer, size_t count, size_t inner, uint16_t* dst)
        {
            if (inner == 1)
            {
                if (count == 2)
                    SynetSoftmax16b21(src, outer, dst);
                else if (count == 3)
                    SynetSoftmax16b31(src, outer, dst);
                else
                    SynetSoftmax16bX1(src, outer, count, dst);
            }
            else
            {
                Exp exp;
                size_t innerF = Simd::AlignLo(inner, F), innerDF = Simd::AlignLo(inner, DF);
                Array32f _buf(inner * (count + 2));
                float* max = _buf.data, * sum = _buf.data + inner, * buf = sum + inner, * b;
                __mmask16 tail = TailMask16(inner - innerF);
                for (size_t o = 0; o < outer; ++o)
                {
                    BFloat16ToFloat32(src, count * inner, buf);
                    memcpy(max, buf, inner * sizeof(float));
                    b = buf + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < innerF; i += F)
                            _mm512_storeu_ps(max + i, _mm512_max_ps(_mm512_loadu_ps(b + i), _mm512_loadu_ps(max + i)));
                        if (i < inner)
                            _mm512_mask_storeu_ps(max + i, tail, _mm512_max_ps(_mm512_maskz_loadu_ps(tail, b + i), _mm512_maskz_loadu_ps(tail, max + i)));
                        b += inner;
                    }

                    b = buf;
                    memset(sum, 0, inner * sizeof(float));
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < innerF; i += F)
                        {
                            __m512 _d = exp.Exponent(_mm512_sub_ps(_mm512_loadu_ps(b + i), _mm512_loadu_ps(max + i)));
                            _mm512_storeu_ps(b + i, _d);
                            _mm512_storeu_ps(sum + i, _mm512_add_ps(_d, _mm512_loadu_ps(sum + i)));
                        }
                        if (i < inner)
                        {
                            __m512 _d = exp.Exponent(_mm512_sub_ps(_mm512_maskz_loadu_ps(tail, b + i), _mm512_maskz_loadu_ps(tail, max + i)));
                            _mm512_mask_storeu_ps(b + i, tail, _d);
                            _mm512_mask_storeu_ps(sum + i, tail, _mm512_add_ps(_d, _mm512_maskz_loadu_ps(tail, sum + i)));
                        }
                        b += inner;
                    }

                    b = buf;
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < innerDF; i += DF)
                            _mm512_storeu_si512((__m512i*)(dst + i), Float32ToBFloat16(
                                _mm512_div_ps(_mm512_loadu_ps(b + i + 0), _mm512_loadu_ps(sum + i + 0)), _mm512_div_ps(_mm512_loadu_ps(b + i + F), _mm512_loadu_ps(sum + i + F))));
                        for (; i < innerF; i += F)
                            _mm256_storeu_si256((__m256i*)(dst + i), PackFloat32ToBFloat16(_mm512_div_ps(_mm512_loadu_ps(b + i), _mm512_loadu_ps(sum + i))));
                        if (i < inner)
                            _mm256_mask_storeu_epi16(dst + i, tail, PackFloat32ToBFloat16(_mm512_div_ps(_mm512_maskz_loadu_ps(tail, b + i), _mm512_maskz_loadu_ps(tail, sum + i))));
                        b += inner;
                        dst += inner;
                    }
                    src += count * inner;
                }
            }
        }
    }
#endif
}
