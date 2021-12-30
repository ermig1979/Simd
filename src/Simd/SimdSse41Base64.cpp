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
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        const __m128i K8_TO_BASE64_SHUFFLE = SIMD_MM_SETR_EPI8(0x1, 0x0, 0x2, 0x1, 0x4, 0x3, 0x5, 0x4, 0x7, 0x6, 0x8, 0x7, 0xA, 0x9, 0xB, 0xA);

        const __m128i K16_TO_BASE64_MULLO = SIMD_MM_SET2_EPI16(0x0010, 0x0100);

        const __m128i K16_TO_BASE64_MULHI = SIMD_MM_SET2_EPI16(0x0040, 0x0400);

        const __m128i K16_003F = SIMD_MM_SET1_EPI16(0x003F);
        const __m128i K16_3F00 = SIMD_MM_SET1_EPI16(0x3F00);

        const __m128i K8_UPP_ADD = SIMD_MM_SET1_EPI8('A');
        const __m128i K8_UPP_END = SIMD_MM_SET1_EPI8(26);
        const __m128i K8_LOW_ADD = SIMD_MM_SET1_EPI8('a' - 26);
        const __m128i K8_LOW_END = SIMD_MM_SET1_EPI8(52);

        const __m128i K8_DIG_SHUFFLE = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/');

        void Base64Encode12(const uint8_t* src, uint8_t* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i shuffle = _mm_shuffle_epi8(_src, K8_TO_BASE64_SHUFFLE);
            __m128i mullo = _mm_mullo_epi16(shuffle, K16_TO_BASE64_MULLO);
            __m128i mulhi = _mm_mulhi_epi16(shuffle, K16_TO_BASE64_MULHI);
            __m128i index = _mm_or_si128(_mm_and_si128(mullo, K16_3F00), _mm_and_si128(mulhi, K16_003F));
            __m128i uppMask = _mm_cmpgt_epi8(K8_UPP_END, index);
            __m128i letMask = _mm_cmpgt_epi8(K8_LOW_END, index);
            __m128i uppValue = _mm_and_si128(_mm_add_epi8(index, K8_UPP_ADD), uppMask);
            __m128i lowValue = _mm_and_si128(_mm_add_epi8(index, K8_LOW_ADD), _mm_andnot_si128(uppMask, letMask));
            __m128i digValue = _mm_shuffle_epi8(K8_DIG_SHUFFLE, index);
            __m128i _dst = _mm_or_si128(_mm_or_si128(uppValue, lowValue), _mm_andnot_si128(letMask, digValue));
            _mm_storeu_si128((__m128i*)dst, _dst);
        }

        void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst)
        {
            size_t size3 = AlignLoAny(size, 3);
            size_t size12 = AlignLoAny(size - 11, 12);
            for (const uint8_t* body12 = src + size12; src < body12; src += 12, dst += 16)
                Base64Encode12(src, dst);
            for (const uint8_t* body3 = src + size3 - size12; src < body3; src += 3, dst += 4)
                Base::Base64Encode3(src, dst);
            if(size - size3)
                Base::Base64EncodeTail(src, size - size3, dst);
        }
    }
#endif
}

