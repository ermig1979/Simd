/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdPrefetch.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        SIMD_INLINE void GemmPackA_4x16(const float* src, size_t stride, float* dst)
        {
            __m512 s0 = _mm512_loadu_ps(src + 0 * stride);
            __m512 s1 = _mm512_loadu_ps(src + 1 * stride);
            __m512 s2 = _mm512_loadu_ps(src + 2 * stride);
            __m512 s3 = _mm512_loadu_ps(src + 3 * stride);
            __m512 s020 = Interleave<0>(s0, s2);
            __m512 s021 = Interleave<1>(s0, s2);
            __m512 s130 = Interleave<0>(s1, s3);
            __m512 s131 = Interleave<1>(s1, s3);
            _mm512_storeu_ps(dst + 0x00, Interleave<0>(s020, s130));
            _mm512_storeu_ps(dst + 0x10, Interleave<1>(s020, s130));
            _mm512_storeu_ps(dst + 0x20, Interleave<0>(s021, s131));
            _mm512_storeu_ps(dst + 0x30, Interleave<1>(s021, s131));
        }

        SIMD_INLINE void GemmPackA_4x8(const float* src, size_t stride, float* dst)
        {
            __m256 s0 = _mm256_loadu_ps(src + 0 * stride);
            __m256 s1 = _mm256_loadu_ps(src + 1 * stride);
            __m256 s2 = _mm256_loadu_ps(src + 2 * stride);
            __m256 s3 = _mm256_loadu_ps(src + 3 * stride);
            __m256 s00 = _mm256_unpacklo_ps(s0, s2);
            __m256 s01 = _mm256_unpacklo_ps(s1, s3);
            __m256 s10 = _mm256_unpackhi_ps(s0, s2);
            __m256 s11 = _mm256_unpackhi_ps(s1, s3);
            __m256 d0 = _mm256_unpacklo_ps(s00, s01);
            __m256 d1 = _mm256_unpackhi_ps(s00, s01);
            __m256 d2 = _mm256_unpacklo_ps(s10, s11);
            __m256 d3 = _mm256_unpackhi_ps(s10, s11);
            _mm256_storeu_ps(dst + 0x00, _mm256_permute2f128_ps(d0, d1, 0x20));
            _mm256_storeu_ps(dst + 0x08, _mm256_permute2f128_ps(d2, d3, 0x20));
            _mm256_storeu_ps(dst + 0x10, _mm256_permute2f128_ps(d0, d1, 0x31));
            _mm256_storeu_ps(dst + 0x18, _mm256_permute2f128_ps(d2, d3, 0x31));
        }

        SIMD_INLINE void GemmPackA_4x4(const float* src, size_t stride, float* dst)
        {
            __m128 s0 = _mm_loadu_ps(src + 0 * stride);
            __m128 s1 = _mm_loadu_ps(src + 1 * stride);
            __m128 s2 = _mm_loadu_ps(src + 2 * stride);
            __m128 s3 = _mm_loadu_ps(src + 3 * stride);
            __m128 s00 = _mm_unpacklo_ps(s0, s2);
            __m128 s01 = _mm_unpacklo_ps(s1, s3);
            __m128 s10 = _mm_unpackhi_ps(s0, s2);
            __m128 s11 = _mm_unpackhi_ps(s1, s3);
            _mm_storeu_ps(dst + 0, _mm_unpacklo_ps(s00, s01));
            _mm_storeu_ps(dst + 4, _mm_unpackhi_ps(s00, s01));
            _mm_storeu_ps(dst + 8, _mm_unpacklo_ps(s10, s11));
            _mm_storeu_ps(dst + 12, _mm_unpackhi_ps(s10, s11));
        }

        const __m512i K32_PACKA6_0 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x04, 0x05, 0x06, 0x07, 0x12, 0x13, 0x08, 0x09, 0x0A, 0x0B);
        const __m512i K32_PACKA6_1 = SIMD_MM512_SETR_EPI32(0x06, 0x07, 0x12, 0x13, 0x08, 0x09, 0x0A, 0x0B, 0x14, 0x15, 0x0C, 0x0D, 0x0E, 0x0F, 0x16, 0x17);
        const __m512i K32_PACKA6_2 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x03, 0x18, 0x19, 0x04, 0x05, 0x06, 0x07, 0x1A, 0x1B, 0x08, 0x09, 0x0A, 0x0B);
        const __m512i K32_PACKA6_3 = SIMD_MM512_SETR_EPI32(0x06, 0x07, 0x1A, 0x1B, 0x08, 0x09, 0x0A, 0x0B, 0x1C, 0x1D, 0x0C, 0x0D, 0x0E, 0x0F, 0x1E, 0x1F);

        SIMD_INLINE void GemmPackA_6x16(const float* src, size_t stride, float* dst)
        {
            __m512 s0 = _mm512_loadu_ps(src + 0 * stride);
            __m512 s1 = _mm512_loadu_ps(src + 1 * stride);
            __m512 s2 = _mm512_loadu_ps(src + 2 * stride);
            __m512 s3 = _mm512_loadu_ps(src + 3 * stride);
            __m512 s4 = _mm512_loadu_ps(src + 4 * stride);
            __m512 s5 = _mm512_loadu_ps(src + 5 * stride);
            __m512 s02_0 = Interleave<0>(s0, s2);
            __m512 s02_1 = Interleave<1>(s0, s2);
            __m512 s13_0 = Interleave<0>(s1, s3);
            __m512 s13_1 = Interleave<1>(s1, s3);
            __m512 s45_0 = Interleave<0>(s4, s5);
            __m512 s45_1 = Interleave<1>(s4, s5);
            __m512 s0123_0 = Interleave<0>(s02_0, s13_0);
            __m512 s0123_1 = Interleave<1>(s02_0, s13_0);
            __m512 s0123_2 = Interleave<0>(s02_1, s13_1);
            __m512 s0123_3 = Interleave<1>(s02_1, s13_1);
            _mm512_mask_storeu_ps(dst + 0x00, 0x0FFF, _mm512_permutex2var_ps(s0123_0, K32_PACKA6_0, s45_0));
            _mm512_mask_storeu_ps(dst + 0x08, 0xFFF0, _mm512_permutex2var_ps(s0123_0, K32_PACKA6_1, s45_0));
            _mm512_mask_storeu_ps(dst + 0x18, 0x0FFF, _mm512_permutex2var_ps(s0123_1, K32_PACKA6_2, s45_0));
            _mm512_mask_storeu_ps(dst + 0x20, 0xFFF0, _mm512_permutex2var_ps(s0123_1, K32_PACKA6_3, s45_0));
            _mm512_mask_storeu_ps(dst + 0x30, 0x0FFF, _mm512_permutex2var_ps(s0123_2, K32_PACKA6_0, s45_1));
            _mm512_mask_storeu_ps(dst + 0x38, 0xFFF0, _mm512_permutex2var_ps(s0123_2, K32_PACKA6_1, s45_1));
            _mm512_mask_storeu_ps(dst + 0x48, 0x0FFF, _mm512_permutex2var_ps(s0123_3, K32_PACKA6_2, s45_1));
            _mm512_mask_storeu_ps(dst + 0x50, 0xFFF0, _mm512_permutex2var_ps(s0123_3, K32_PACKA6_3, s45_1));
        }

        SIMD_INLINE void GemmPackA_6x4(const float* src, size_t stride, float* dst)
        {
            __m128 s0 = _mm_loadu_ps(src + 0 * stride);
            __m128 s1 = _mm_loadu_ps(src + 1 * stride);
            __m128 s2 = _mm_loadu_ps(src + 2 * stride);
            __m128 s3 = _mm_loadu_ps(src + 3 * stride);
            __m128 s4 = _mm_loadu_ps(src + 4 * stride);
            __m128 s5 = _mm_loadu_ps(src + 5 * stride);
            __m128 s00 = _mm_unpacklo_ps(s0, s2);
            __m128 s01 = _mm_unpacklo_ps(s1, s3);
            __m128 s10 = _mm_unpackhi_ps(s0, s2);
            __m128 s11 = _mm_unpackhi_ps(s1, s3);
            __m128 s20 = _mm_unpacklo_ps(s4, s5);
            __m128 s21 = _mm_unpackhi_ps(s4, s5);
            _mm_storeu_ps(dst + 0, _mm_unpacklo_ps(s00, s01));
            _mm_storel_pi((__m64*)(dst + 4), s20);
            _mm_storeu_ps(dst + 6, _mm_unpackhi_ps(s00, s01));
            _mm_storeh_pi((__m64*)(dst + 10), s20);
            _mm_storeu_ps(dst + 12, _mm_unpacklo_ps(s10, s11));
            _mm_storel_pi((__m64*)(dst + 16), s21);
            _mm_storeu_ps(dst + 18, _mm_unpackhi_ps(s10, s11));
            _mm_storeh_pi((__m64*)(dst + 22), s21);
        }

        SIMD_INLINE void GemmPackA_8x16(const float* src, size_t stride, float* dst)
        {
            __m512 s0 = _mm512_loadu_ps(src + 0 * stride);
            __m512 s1 = _mm512_loadu_ps(src + 1 * stride);
            __m512 s2 = _mm512_loadu_ps(src + 2 * stride);
            __m512 s3 = _mm512_loadu_ps(src + 3 * stride);
            __m512 s4 = _mm512_loadu_ps(src + 4 * stride);
            __m512 s5 = _mm512_loadu_ps(src + 5 * stride);
            __m512 s6 = _mm512_loadu_ps(src + 6 * stride);
            __m512 s7 = _mm512_loadu_ps(src + 7 * stride);
            __m512 s04_0 = Interleave<0>(s0, s4);
            __m512 s04_1 = Interleave<1>(s0, s4);
            __m512 s15_0 = Interleave<0>(s1, s5);
            __m512 s15_1 = Interleave<1>(s1, s5);
            __m512 s26_0 = Interleave<0>(s2, s6);
            __m512 s26_1 = Interleave<1>(s2, s6);
            __m512 s37_0 = Interleave<0>(s3, s7);
            __m512 s37_1 = Interleave<1>(s3, s7);
            __m512 s0246_0 = Interleave<0>(s04_0, s26_0);
            __m512 s0246_1 = Interleave<1>(s04_0, s26_0);
            __m512 s0246_2 = Interleave<0>(s04_1, s26_1);
            __m512 s0246_3 = Interleave<1>(s04_1, s26_1);
            __m512 s1357_0 = Interleave<0>(s15_0, s37_0);
            __m512 s1357_1 = Interleave<1>(s15_0, s37_0);
            __m512 s1357_2 = Interleave<0>(s15_1, s37_1);
            __m512 s1357_3 = Interleave<1>(s15_1, s37_1);
            _mm512_storeu_ps(dst + 0x00, Interleave<0>(s0246_0, s1357_0));
            _mm512_storeu_ps(dst + 0x10, Interleave<1>(s0246_0, s1357_0));
            _mm512_storeu_ps(dst + 0x20, Interleave<0>(s0246_1, s1357_1));
            _mm512_storeu_ps(dst + 0x30, Interleave<1>(s0246_1, s1357_1));
            _mm512_storeu_ps(dst + 0x40, Interleave<0>(s0246_2, s1357_2));
            _mm512_storeu_ps(dst + 0x50, Interleave<1>(s0246_2, s1357_2));
            _mm512_storeu_ps(dst + 0x60, Interleave<0>(s0246_3, s1357_3));
            _mm512_storeu_ps(dst + 0x70, Interleave<1>(s0246_3, s1357_3));
        }

        SIMD_INLINE void GemmPackA_8x4(const float* src, size_t stride, float* dst)
        {
            __m128 s0 = _mm_loadu_ps(src + 0 * stride);
            __m128 s1 = _mm_loadu_ps(src + 1 * stride);
            __m128 s2 = _mm_loadu_ps(src + 2 * stride);
            __m128 s3 = _mm_loadu_ps(src + 3 * stride);
            __m128 s4 = _mm_loadu_ps(src + 4 * stride);
            __m128 s5 = _mm_loadu_ps(src + 5 * stride);
            __m128 s6 = _mm_loadu_ps(src + 6 * stride);
            __m128 s7 = _mm_loadu_ps(src + 7 * stride);
            __m128 s02_0 = _mm_unpacklo_ps(s0, s2);
            __m128 s02_1 = _mm_unpackhi_ps(s0, s2);
            __m128 s13_0 = _mm_unpacklo_ps(s1, s3);
            __m128 s13_1 = _mm_unpackhi_ps(s1, s3);
            __m128 s46_0 = _mm_unpacklo_ps(s4, s6);
            __m128 s46_1 = _mm_unpackhi_ps(s4, s6);
            __m128 s57_0 = _mm_unpacklo_ps(s5, s7);
            __m128 s57_1 = _mm_unpackhi_ps(s5, s7);
            _mm_storeu_ps(dst + 0x00, _mm_unpacklo_ps(s02_0, s13_0));
            _mm_storeu_ps(dst + 0x04, _mm_unpacklo_ps(s46_0, s57_0));
            _mm_storeu_ps(dst + 0x08, _mm_unpackhi_ps(s02_0, s13_0));
            _mm_storeu_ps(dst + 0x0C, _mm_unpackhi_ps(s46_0, s57_0));
            _mm_storeu_ps(dst + 0x10, _mm_unpacklo_ps(s02_1, s13_1));
            _mm_storeu_ps(dst + 0x14, _mm_unpacklo_ps(s46_1, s57_1));
            _mm_storeu_ps(dst + 0x18, _mm_unpackhi_ps(s02_1, s13_1));
            _mm_storeu_ps(dst + 0x1C, _mm_unpackhi_ps(s46_1, s57_1));
        }

        const __m512i K32_PACKA9_0 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x10, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E);
        const __m512i K32_PACKA9_1 = SIMD_MM512_SETR_EPI32(0x00, 0x11, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x12, 0x09, 0x0A, 0x0B, 0x0C, 0x0D);
        const __m512i K32_PACKA9_2 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x13, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x14, 0x0B, 0x0C, 0x0D);
        const __m512i K32_PACKA9_3 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x03, 0x04, 0x15, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x16, 0x0D);
        const __m512i K32_PACKA9_4 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x17, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E);
        const __m512i K32_PACKA9_5 = SIMD_MM512_SETR_EPI32(0x18, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x19, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D);
        const __m512i K32_PACKA9_6 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x1A, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x1B, 0x0A, 0x0B, 0x0C, 0x0D);
        const __m512i K32_PACKA9_7 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x03, 0x1C, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x1D, 0x0C, 0x0D);
        const __m512i K32_PACKA9_8 = SIMD_MM512_SETR_EPI32(0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x1E, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x1F);

        SIMD_INLINE void GemmPackA_9x16(const float* src, size_t stride, float* dst)
        {
            __m512 a[9], b[8];
            a[0] = _mm512_loadu_ps(src + 0 * stride);
            a[1] = _mm512_loadu_ps(src + 1 * stride);
            a[2] = _mm512_loadu_ps(src + 2 * stride);
            a[3] = _mm512_loadu_ps(src + 3 * stride);
            a[4] = _mm512_loadu_ps(src + 4 * stride);
            a[5] = _mm512_loadu_ps(src + 5 * stride);
            a[6] = _mm512_loadu_ps(src + 6 * stride);
            a[7] = _mm512_loadu_ps(src + 7 * stride);
            a[8] = _mm512_loadu_ps(src + 8 * stride);
            b[0] = Interleave<0>(a[0], a[4]);
            b[1] = Interleave<1>(a[0], a[4]);
            b[2] = Interleave<0>(a[1], a[5]);
            b[3] = Interleave<1>(a[1], a[5]);
            b[4] = Interleave<0>(a[2], a[6]);
            b[5] = Interleave<1>(a[2], a[6]);
            b[6] = Interleave<0>(a[3], a[7]);
            b[7] = Interleave<1>(a[3], a[7]);
            a[0] = Interleave<0>(b[0], b[4]);
            a[1] = Interleave<1>(b[0], b[4]);
            a[2] = Interleave<0>(b[1], b[5]);
            a[3] = Interleave<1>(b[1], b[5]);
            a[4] = Interleave<0>(b[2], b[6]);
            a[5] = Interleave<1>(b[2], b[6]);
            a[6] = Interleave<0>(b[3], b[7]);
            a[7] = Interleave<1>(b[3], b[7]);
            b[0] = Interleave<0>(a[0], a[4]);
            b[1] = Interleave<1>(a[0], a[4]);
            b[2] = Interleave<0>(a[1], a[5]);
            b[3] = Interleave<1>(a[1], a[5]);
            b[4] = Interleave<0>(a[2], a[6]);
            b[5] = Interleave<1>(a[2], a[6]);
            b[6] = Interleave<0>(a[3], a[7]);
            b[7] = Interleave<1>(a[3], a[7]);
            _mm512_storeu_ps(dst + 0x00, _mm512_permutex2var_ps(Alignr<0x0>(b[0], b[1]), K32_PACKA9_0, a[8]));
            _mm512_storeu_ps(dst + 0x10, _mm512_permutex2var_ps(Alignr<0xF>(b[0], b[1]), K32_PACKA9_1, a[8]));
            _mm512_storeu_ps(dst + 0x20, _mm512_permutex2var_ps(Alignr<0xD>(b[1], b[2]), K32_PACKA9_2, a[8]));
            _mm512_storeu_ps(dst + 0x30, _mm512_permutex2var_ps(Alignr<0xB>(b[2], b[3]), K32_PACKA9_3, a[8]));
            _mm512_storeu_ps(dst + 0x40, _mm512_permutex2var_ps(Alignr<0x9>(b[3], b[4]), K32_PACKA9_4, a[8]));
            _mm512_storeu_ps(dst + 0x50, _mm512_permutex2var_ps(Alignr<0x8>(b[4], b[5]), K32_PACKA9_5, a[8]));
            _mm512_storeu_ps(dst + 0x60, _mm512_permutex2var_ps(Alignr<0x6>(b[5], b[6]), K32_PACKA9_6, a[8]));
            _mm512_storeu_ps(dst + 0x70, _mm512_permutex2var_ps(Alignr<0x4>(b[6], b[7]), K32_PACKA9_7, a[8]));
            _mm512_storeu_ps(dst + 0x80, _mm512_permutex2var_ps(Alignr<0x0>(b[7], b[7]), K32_PACKA9_8, a[8]));
        }

        SIMD_INLINE void GemmPackA_9x4(const float* src, size_t stride, float* dst)
        {
            __m128 s0 = _mm_loadu_ps(src + 0 * stride);
            __m128 s1 = _mm_loadu_ps(src + 1 * stride);
            __m128 s2 = _mm_loadu_ps(src + 2 * stride);
            __m128 s3 = _mm_loadu_ps(src + 3 * stride);
            __m128 s4 = _mm_loadu_ps(src + 4 * stride);
            __m128 s5 = _mm_loadu_ps(src + 5 * stride);
            __m128 s6 = _mm_loadu_ps(src + 6 * stride);
            __m128 s7 = _mm_loadu_ps(src + 7 * stride);
            __m128 s02_0 = _mm_unpacklo_ps(s0, s2);
            __m128 s02_1 = _mm_unpackhi_ps(s0, s2);
            __m128 s13_0 = _mm_unpacklo_ps(s1, s3);
            __m128 s13_1 = _mm_unpackhi_ps(s1, s3);
            __m128 s46_0 = _mm_unpacklo_ps(s4, s6);
            __m128 s46_1 = _mm_unpackhi_ps(s4, s6);
            __m128 s57_0 = _mm_unpacklo_ps(s5, s7);
            __m128 s57_1 = _mm_unpackhi_ps(s5, s7);
            src += 8 * stride;
            _mm_storeu_ps(dst + 0x00, _mm_unpacklo_ps(s02_0, s13_0));
            _mm_storeu_ps(dst + 0x04, _mm_unpacklo_ps(s46_0, s57_0));
            dst[0x08] = src[0];
            _mm_storeu_ps(dst + 0x09, _mm_unpackhi_ps(s02_0, s13_0));
            _mm_storeu_ps(dst + 0x0D, _mm_unpackhi_ps(s46_0, s57_0));
            dst[0x11] = src[1];
            _mm_storeu_ps(dst + 0x12, _mm_unpacklo_ps(s02_1, s13_1));
            _mm_storeu_ps(dst + 0x16, _mm_unpacklo_ps(s46_1, s57_1));
            dst[0x1A] = src[2];
            _mm_storeu_ps(dst + 0x1B, _mm_unpackhi_ps(s02_1, s13_1));
            _mm_storeu_ps(dst + 0x1F, _mm_unpackhi_ps(s46_1, s57_1));
            dst[0x23] = src[3];
        }

        const __m512i K32_PACKA12_0 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x10, 0x11, 0x12, 0x13, 0x08, 0x09, 0x0A, 0x0B);
        const __m512i K32_PACKA12_1 = SIMD_MM512_SETR_EPI32(0x10, 0x11, 0x12, 0x13, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x14, 0x15, 0x16, 0x17);
        const __m512i K32_PACKA12_2 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x18, 0x19, 0x1A, 0x1B, 0x08, 0x09, 0x0A, 0x0B);
        const __m512i K32_PACKA12_3 = SIMD_MM512_SETR_EPI32(0x18, 0x19, 0x1A, 0x1B, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x1C, 0x1D, 0x1E, 0x1F);

        SIMD_INLINE void GemmPackA_12x16(const float* src, size_t stride, float* dst)
        {
            __m512 a[12], b[12];
            a[0] = _mm512_loadu_ps(src + 0 * stride);
            a[1] = _mm512_loadu_ps(src + 1 * stride);
            a[2] = _mm512_loadu_ps(src + 2 * stride);
            a[3] = _mm512_loadu_ps(src + 3 * stride);
            a[4] = _mm512_loadu_ps(src + 4 * stride);
            a[5] = _mm512_loadu_ps(src + 5 * stride);
            a[6] = _mm512_loadu_ps(src + 6 * stride);
            a[7] = _mm512_loadu_ps(src + 7 * stride);
            a[8] = _mm512_loadu_ps(src + 8 * stride);
            a[9] = _mm512_loadu_ps(src + 9 * stride);
            a[10] = _mm512_loadu_ps(src + 10 * stride);
            a[11] = _mm512_loadu_ps(src + 11 * stride);
            b[0] = Interleave<0>(a[0], a[4]);
            b[1] = Interleave<1>(a[0], a[4]);
            b[2] = Interleave<0>(a[1], a[5]);
            b[3] = Interleave<1>(a[1], a[5]);
            b[4] = Interleave<0>(a[2], a[6]);
            b[5] = Interleave<1>(a[2], a[6]);
            b[6] = Interleave<0>(a[3], a[7]);
            b[7] = Interleave<1>(a[3], a[7]);
            b[8] = Interleave<0>(a[8], a[10]);
            b[9] = Interleave<1>(a[8], a[10]);
            b[10] = Interleave<0>(a[9], a[11]);
            b[11] = Interleave<1>(a[9], a[11]);
            a[0] = Interleave<0>(b[0], b[4]);
            a[1] = Interleave<1>(b[0], b[4]);
            a[2] = Interleave<0>(b[1], b[5]);
            a[3] = Interleave<1>(b[1], b[5]);
            a[4] = Interleave<0>(b[2], b[6]);
            a[5] = Interleave<1>(b[2], b[6]);
            a[6] = Interleave<0>(b[3], b[7]);
            a[7] = Interleave<1>(b[3], b[7]);
            a[8] = Interleave<0>(b[8], b[10]);
            a[9] = Interleave<1>(b[8], b[10]);
            a[10] = Interleave<0>(b[9], b[11]);
            a[11] = Interleave<1>(b[9], b[11]);
            b[0] = Interleave<0>(a[0], a[4]);
            b[1] = Interleave<1>(a[0], a[4]);
            b[2] = Interleave<0>(a[1], a[5]);
            b[3] = Interleave<1>(a[1], a[5]);
            b[4] = Interleave<0>(a[2], a[6]);
            b[5] = Interleave<1>(a[2], a[6]);
            b[6] = Interleave<0>(a[3], a[7]);
            b[7] = Interleave<1>(a[3], a[7]);
            _mm512_mask_storeu_ps(dst + 0x00, 0x0FFF, _mm512_permutex2var_ps(b[0], K32_PACKA12_0, a[8]));
            _mm512_mask_storeu_ps(dst + 0x08, 0xFFF0, _mm512_permutex2var_ps(b[0], K32_PACKA12_1, a[8]));
            _mm512_mask_storeu_ps(dst + 0x18, 0x0FFF, _mm512_permutex2var_ps(b[1], K32_PACKA12_2, a[8]));
            _mm512_mask_storeu_ps(dst + 0x20, 0xFFF0, _mm512_permutex2var_ps(b[1], K32_PACKA12_3, a[8]));
            _mm512_mask_storeu_ps(dst + 0x30, 0x0FFF, _mm512_permutex2var_ps(b[2], K32_PACKA12_0, a[9]));
            _mm512_mask_storeu_ps(dst + 0x38, 0xFFF0, _mm512_permutex2var_ps(b[2], K32_PACKA12_1, a[9]));
            _mm512_mask_storeu_ps(dst + 0x48, 0x0FFF, _mm512_permutex2var_ps(b[3], K32_PACKA12_2, a[9]));
            _mm512_mask_storeu_ps(dst + 0x50, 0xFFF0, _mm512_permutex2var_ps(b[3], K32_PACKA12_3, a[9]));
            _mm512_mask_storeu_ps(dst + 0x60, 0x0FFF, _mm512_permutex2var_ps(b[4], K32_PACKA12_0, a[10]));
            _mm512_mask_storeu_ps(dst + 0x68, 0xFFF0, _mm512_permutex2var_ps(b[4], K32_PACKA12_1, a[10]));
            _mm512_mask_storeu_ps(dst + 0x78, 0x0FFF, _mm512_permutex2var_ps(b[5], K32_PACKA12_2, a[10]));
            _mm512_mask_storeu_ps(dst + 0x80, 0xFFF0, _mm512_permutex2var_ps(b[5], K32_PACKA12_3, a[10]));
            _mm512_mask_storeu_ps(dst + 0x90, 0x0FFF, _mm512_permutex2var_ps(b[6], K32_PACKA12_0, a[11]));
            _mm512_mask_storeu_ps(dst + 0x98, 0xFFF0, _mm512_permutex2var_ps(b[6], K32_PACKA12_1, a[11]));
            _mm512_mask_storeu_ps(dst + 0xA8, 0x0FFF, _mm512_permutex2var_ps(b[7], K32_PACKA12_2, a[11]));
            _mm512_mask_storeu_ps(dst + 0xB0, 0xFFF0, _mm512_permutex2var_ps(b[7], K32_PACKA12_3, a[11]));
        }

        SIMD_INLINE void GemmPackA_12x4(const float * src, size_t stride, float * dst)
        {
            __m128 a[4], b[4];
            for (size_t j = 0; j < 3; ++j)
            {
                a[0] = _mm_loadu_ps(src + 0 * stride);
                a[1] = _mm_loadu_ps(src + 1 * stride);
                a[2] = _mm_loadu_ps(src + 2 * stride);
                a[3] = _mm_loadu_ps(src + 3 * stride);
                b[0] = _mm_unpacklo_ps(a[0], a[2]);
                b[1] = _mm_unpackhi_ps(a[0], a[2]);
                b[2] = _mm_unpacklo_ps(a[1], a[3]);
                b[3] = _mm_unpackhi_ps(a[1], a[3]);
                _mm_storeu_ps(dst + 0x00, _mm_unpacklo_ps(b[0], b[2]));
                _mm_storeu_ps(dst + 0x0C, _mm_unpackhi_ps(b[0], b[2]));
                _mm_storeu_ps(dst + 0x18, _mm_unpacklo_ps(b[1], b[3]));
                _mm_storeu_ps(dst + 0x24, _mm_unpackhi_ps(b[1], b[3]));
                src += 4 * stride;
                dst += 4;
            }
        }

        SIMD_INLINE void GemmPackA_14x16(const float* src, size_t stride, float* dst)
        {
            __m512 a[16], b[4];
            a[0] = _mm512_loadu_ps(src + 0 * stride);
            a[1] = _mm512_loadu_ps(src + 1 * stride);
            a[2] = _mm512_loadu_ps(src + 2 * stride);
            a[3] = _mm512_loadu_ps(src + 3 * stride);
            a[4] = _mm512_loadu_ps(src + 4 * stride);
            a[5] = _mm512_loadu_ps(src + 5 * stride);
            a[6] = _mm512_loadu_ps(src + 6 * stride);
            a[7] = _mm512_loadu_ps(src + 7 * stride);
            a[8] = _mm512_loadu_ps(src + 8 * stride);
            a[9] = _mm512_loadu_ps(src + 9 * stride);
            a[10] = _mm512_loadu_ps(src + 10 * stride);
            a[11] = _mm512_loadu_ps(src + 11 * stride);
            a[12] = _mm512_loadu_ps(src + 12 * stride);
            a[13] = _mm512_loadu_ps(src + 13 * stride);
            a[14] = _mm512_setzero_ps();
            a[15] = _mm512_setzero_ps();
            for (size_t i = 0; i < 4; ++i)
            {
                __m512* c = a + i;
                b[0] = Interleave<0>(c[0], c[8]);
                b[1] = Interleave<1>(c[0], c[8]);
                b[2] = Interleave<0>(c[4], c[12]);
                b[3] = Interleave<1>(c[4], c[12]);
                c[0] = Interleave<0>(b[0], b[2]);
                c[4] = Interleave<1>(b[0], b[2]);
                c[8] = Interleave<0>(b[1], b[3]);
                c[12] = Interleave<1>(b[1], b[3]);
            }
            for (size_t i = 0; i < 4; ++i)
            {
                const __m512 * c = a + i * 4;
                b[0] = Interleave<0>(c[0], c[2]);
                b[1] = Interleave<1>(c[0], c[2]);
                b[2] = Interleave<0>(c[1], c[3]);
                b[3] = Interleave<1>(c[1], c[3]);
                _mm512_mask_storeu_ps(dst + 00, 0x3FFF, Interleave<0>(b[0], b[2]));
                _mm512_mask_storeu_ps(dst + 14, 0x3FFF, Interleave<1>(b[0], b[2]));
                _mm512_mask_storeu_ps(dst + 28, 0x3FFF, Interleave<0>(b[1], b[3]));
                _mm512_mask_storeu_ps(dst + 42, 0x3FFF, Interleave<1>(b[1], b[3]));
                dst += 56;
            }
        }

        SIMD_INLINE void GemmPackA_14x4(const float* src, size_t stride, float* dst)
        {
            __m128 a[4], b[4];
            for (size_t j = 0; j < 3; ++j)
            {
                a[0] = _mm_loadu_ps(src + 0 * stride);
                a[1] = _mm_loadu_ps(src + 1 * stride);
                a[2] = _mm_loadu_ps(src + 2 * stride);
                a[3] = _mm_loadu_ps(src + 3 * stride);
                b[0] = _mm_unpacklo_ps(a[0], a[2]);
                b[1] = _mm_unpackhi_ps(a[0], a[2]);
                b[2] = _mm_unpacklo_ps(a[1], a[3]);
                b[3] = _mm_unpackhi_ps(a[1], a[3]);
                _mm_storeu_ps(dst + 0x00, _mm_unpacklo_ps(b[0], b[2]));
                _mm_storeu_ps(dst + 0x0E, _mm_unpackhi_ps(b[0], b[2]));
                _mm_storeu_ps(dst + 0x1C, _mm_unpacklo_ps(b[1], b[3]));
                _mm_storeu_ps(dst + 0x2A, _mm_unpackhi_ps(b[1], b[3]));
                src += 4 * stride;
                dst += 4;
            }
            a[0] = _mm_loadu_ps(src + 0 * stride);
            a[1] = _mm_loadu_ps(src + 1 * stride);
            b[0] = _mm_unpacklo_ps(a[0], a[1]);
            b[1] = _mm_unpackhi_ps(a[0], a[1]);
            _mm_storel_pi((__m64*)(dst + 0x00), b[0]);
            _mm_storeh_pi((__m64*)(dst + 0x0E), b[0]);
            _mm_storel_pi((__m64*)(dst + 0x1C), b[1]);
            _mm_storeh_pi((__m64*)(dst + 0x2A), b[1]);
        }

        void GemmPackA(const float * src, size_t stride, size_t M, size_t K, size_t cell, float* dst)
        {
            size_t K4 = AlignLo(K, 4), K8 = AlignLo(K, 8), K16 =  AlignLo(K, 16);
            for (size_t i = 0; i < M; i += cell)
            {
                size_t m = Simd::Min(cell, M - i), k = 0;
                if (cell == 4 && m == 4)
                {
                    for (; k < K16; k += 16, dst += 64)
                        GemmPackA_4x16(src + k, stride, dst);
                    for (; k < K8; k += 8, dst += 32)
                        GemmPackA_4x8(src + k, stride, dst);
                    for (; k < K4; k += 4, dst += 16)
                        GemmPackA_4x4(src + k, stride, dst);
                }
                else if (cell == 6 && m == 6)
                {
                    for (; k < K16; k += 16, dst += 96)
                        GemmPackA_6x16(src + k, stride, dst);
                    for (; k < K4; k += 4, dst += 24)
                        GemmPackA_6x4(src + k, stride, dst);
                }                
                else if (cell == 8 && m == 8)
                {
                    for (; k < K16; k += 16, dst += 128)
                        GemmPackA_8x16(src + k, stride, dst);
                    for (; k < K4; k += 4, dst += 32)
                        GemmPackA_8x4(src + k, stride, dst);
                }                
                else if (cell == 9 && m == 9)
                {
                    for (; k < K16; k += 16, dst += 144)
                        GemmPackA_9x16(src + k, stride, dst);
                    for (; k < K4; k += 4, dst += 36)
                        GemmPackA_9x4(src + k, stride, dst);
                }                
                else if (cell == 12 && m == 12)
                {
                    for (; k < K16; k += 16, dst += 192)
                        GemmPackA_12x16(src + k, stride, dst);
                    for (; k < K4; k += 4, dst += 48)
                        GemmPackA_12x4(src + k, stride, dst);
                }
                else if (cell == 14 && m == 14)
                {
                    for (; k < K16; k += 16, dst += 224)
                        GemmPackA_14x16(src + k, stride, dst);
                    for (; k < K4; k += 4, dst += 56)
                        GemmPackA_14x4(src + k, stride, dst);
                }
                for (; k < K; ++k)
                {
                    for (size_t c = 0; c < m; ++c)
                        *(dst++) = src[c*stride + k];
                }  
                src += cell * stride;
            }
        }

        //---------------------------------------------------------------------

        void GemmPackB(const float * B, size_t ldb, size_t K, size_t N, size_t microN, float * pB)
        {
            for (size_t j = 0; j < N; j += microN)
            {
                size_t n = Simd::Min(microN, N - j);
                if (microN == 1 * F)
                {
                    __mmask16 mask0 = TailMask16(n - 0 * F);
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * b = B + k * ldb;
                        _mm512_storeu_ps(pB + 0 * F, _mm512_maskz_loadu_ps(mask0, b + 0 * F));
                        pB += microN;
                    }
                }
                else if (microN == 2 * F)
                {
                    __mmask16 mask0 = TailMask16(n - 0 * F);
                    __mmask16 mask1 = TailMask16(n - 1 * F);
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * b = B + k * ldb;
                        _mm512_storeu_ps(pB + 0 * F, _mm512_maskz_loadu_ps(mask0, b + 0 * F));
                        _mm512_storeu_ps(pB + 1 * F, _mm512_maskz_loadu_ps(mask1, b + 1 * F));
                        pB += microN;
                    }
                }
                else if (microN == 3 * F)
                {
                    __mmask16 mask0 = TailMask16(n - 0 * F);
                    __mmask16 mask1 = TailMask16(n - 1 * F);
                    __mmask16 mask2 = TailMask16(n - 2 * F);
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * b = B + k * ldb;
                        _mm512_storeu_ps(pB + 0 * F, _mm512_maskz_loadu_ps(mask0, b + 0 * F));
                        _mm512_storeu_ps(pB + 1 * F, _mm512_maskz_loadu_ps(mask1, b + 1 * F));
                        _mm512_storeu_ps(pB + 2 * F, _mm512_maskz_loadu_ps(mask2, b + 2 * F));
                        pB += microN;
                    }
                }
                else
                {
                    for (size_t k = 0; k < K; ++k)
                    {
                        const float * b = B + k * ldb;
                        size_t c = 0;
                        for (; c < n; ++c)
                            *(pB++) = *(b++);
                        for (; c < microN; ++c)
                            *(pB++) = 0;
                    }
                }
                B += microN;
            }
        }

        //---------------------------------------------------------------------

        SIMD_INLINE void ScaleC(float * ptr, __m512 beta, __mmask16 mask = -1)
        {
            _mm512_mask_storeu_ps(ptr, mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, ptr), beta));
        }

        void GemmScaleC(size_t M, size_t N, float beta, float * C, size_t ldc)
        {
            if (beta == 1.0f)
                return;
            else if (beta == 0.0f)
            {
                for (size_t i = 0; i < M; ++i)
                    memset(C + i * ldc, 0, N * sizeof(float));
            }
            else
            {
                size_t NQF = AlignLo(N, QF);
                size_t NF = AlignLo(N, F);
                __m512 _beta = _mm512_set1_ps(beta);
                __mmask16 tail = TailMask16(N - NF);
                for (size_t i = 0; i < M; ++i)
                {
                    size_t j = 0;
                    for (; j < NQF; j += QF)
                    {
                        ScaleC(C + j + F * 0, _beta);
                        ScaleC(C + j + F * 1, _beta);
                        ScaleC(C + j + F * 2, _beta);
                        ScaleC(C + j + F * 3, _beta);
                    }
                    for (; j < NF; j += F)
                        ScaleC(C + j, _beta);
                    if (j < N)
                        ScaleC(C + j, _beta, tail);
                    C += ldc;
                }
            }
        }
    }
#endif// SIMD_AVX512F_ENABLE
}
