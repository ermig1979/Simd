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
#include "Simd/SimdMemory.h"
#include "Simd/SimdBase64.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        const __m256i K8_TO_BASE64_SHUFFLE = SIMD_MM256_SETR_EPI8(
            0x1, 0x0, 0x2, 0x1, 0x4, 0x3, 0x5, 0x4, 0x7, 0x6, 0x8, 0x7, 0xA, 0x9, 0xB, 0xA,
            0x5, 0x4, 0x6, 0x5, 0x8, 0x7, 0x9, 0x8, 0xB, 0xA, 0xC, 0xB, 0xE, 0xD, 0xF, 0xE);

        const __m256i K16_TO_BASE64_MULLO = SIMD_MM256_SET2_EPI16(0x0010, 0x0100);

        const __m256i K16_TO_BASE64_MULHI = SIMD_MM256_SET2_EPI16(0x0040, 0x0400);

        const __m256i K16_003F = SIMD_MM256_SET1_EPI16(0x003F);
        const __m256i K16_3F00 = SIMD_MM256_SET1_EPI16(0x3F00);

        const __m256i K8_UPP_ADD = SIMD_MM256_SET1_EPI8('A');
        const __m256i K8_UPP_END = SIMD_MM256_SET1_EPI8(26);
        const __m256i K8_LOW_ADD = SIMD_MM256_SET1_EPI8('a' - 26);
        const __m256i K8_LOW_END = SIMD_MM256_SET1_EPI8(52);

        const __m256i K8_DIG_SHUFFLE = SIMD_MM256_SETR_EPI8(
            -1, -1, -1, -1, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/',
            -1, -1, -1, -1, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/');

        void Base64Encode24(const uint8_t* src, uint8_t* dst)
        {
            __m256i _src = _mm256_loadu_si256((__m256i*)src);
            __m256i permute = _mm256_permute4x64_epi64(_src, 0x94);
            __m256i shuffle = _mm256_shuffle_epi8(permute, K8_TO_BASE64_SHUFFLE);
            __m256i mullo = _mm256_mullo_epi16(shuffle, K16_TO_BASE64_MULLO);
            __m256i mulhi = _mm256_mulhi_epi16(shuffle, K16_TO_BASE64_MULHI);
            __m256i index = _mm256_or_si256(_mm256_and_si256(mullo, K16_3F00), _mm256_and_si256(mulhi, K16_003F));
            __m256i uppMask = _mm256_cmpgt_epi8(K8_UPP_END, index);
            __m256i letMask = _mm256_cmpgt_epi8(K8_LOW_END, index);
            __m256i uppValue = _mm256_and_si256(_mm256_add_epi8(index, K8_UPP_ADD), uppMask);
            __m256i lowValue = _mm256_and_si256(_mm256_add_epi8(index, K8_LOW_ADD), _mm256_andnot_si256(uppMask, letMask));
            __m256i digValue = _mm256_shuffle_epi8(K8_DIG_SHUFFLE, index);
            __m256i _dst = _mm256_or_si256(_mm256_or_si256(uppValue, lowValue), _mm256_andnot_si256(letMask, digValue));
            _mm256_storeu_si256((__m256i*)dst, _dst);
        }

        void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst)
        {
            size_t size3 = AlignLoAny(size, 3);
            size_t size24 = AlignLoAny(size - 23, 24);
            for (const uint8_t* body24 = src + size24; src < body24; src += 24, dst += 32)
                Base64Encode24(src, dst);
            for (const uint8_t* body3 = src + size3 - size24; src < body3; src += 3, dst += 4)
                Base::Base64Encode3(src, dst);
            if(size - size3)
                Base::Base64EncodeTail(src, size - size3, dst);
        }
    }
#endif
}

