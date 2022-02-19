/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdCompare.h"
#include "Simd/SimdMath.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        const __m128i K16_003F = SIMD_MM_SET1_EPI16(0x003F);
        const __m128i K16_3F00 = SIMD_MM_SET1_EPI16(0x3F00);

        const __m128i K8_UPP_ADD = SIMD_MM_SET1_EPI8('A');
        const __m128i K8_LOW_ADD = SIMD_MM_SET1_EPI8('a' - 26);

        const __m128i K8_9 = SIMD_MM_SET1_EPI8('9');
        const __m128i K8_Z = SIMD_MM_SET1_EPI8('Z');

        const __m128i K8_PLUS = SIMD_MM_SET1_EPI8('+');

        const __m128i K16_FROM_MULLO = SIMD_MM_SET2_EPI16(0x0400, 0x0040);
        const __m128i K16_FROM_MULHI = SIMD_MM_SET2_EPI16(0x1000, 0x0100);

        const __m128i K8_FROM_SHUFFLE_LO = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x2, 0x5, 0x7, 0x6, 0x9, 0xB, 0xA, 0xD, 0xF, 0xE, -1, -1, -1, -1);
        const __m128i K8_FROM_SHUFFLE_HI = SIMD_MM_SETR_EPI8(0x1, 0x0, 0x2, 0x5, 0x4, 0x6, 0x9, 0x8, 0xA, 0xD, 0xC, 0xE, -1, -1, -1, -1);

        const __m128i K8_FROM_DIG_SHUFFLE = SIMD_MM_SETR_EPI8(62, -1, -1, -1, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1);

        SIMD_INLINE void Base64Decode12(const uint8_t* src, uint8_t* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i letMask = _mm_cmpgt_epi8(_src, K8_9);
            __m128i lowMask = _mm_cmpgt_epi8(_src, K8_Z);
            __m128i lowValue = _mm_and_si128(lowMask, _mm_sub_epi8(_src, K8_LOW_ADD));
            __m128i uppValue = _mm_and_si128(_mm_andnot_si128(lowMask, letMask), _mm_sub_epi8(_src, K8_UPP_ADD));
            __m128i digValue = _mm_andnot_si128(letMask, _mm_shuffle_epi8(K8_FROM_DIG_SHUFFLE, _mm_sub_epi8(_src, K8_PLUS)));
            __m128i from = _mm_or_si128(_mm_or_si128(uppValue, lowValue), digValue);
            assert(TestZ(GreaterOrEqual8u(from, _mm_set1_epi8(64))));
            __m128i mullo = _mm_mullo_epi16(_mm_and_si128(from, K16_003F), K16_FROM_MULLO);
            __m128i mulhi = _mm_mulhi_epi16(_mm_and_si128(from, K16_3F00), K16_FROM_MULHI);
            __m128i shuffleHi = _mm_shuffle_epi8(mullo, K8_FROM_SHUFFLE_LO);
            __m128i shuffleLo = _mm_shuffle_epi8(mulhi, K8_FROM_SHUFFLE_HI);
            __m128i _dst = _mm_or_si128(shuffleLo, shuffleHi);
            _mm_storeu_si128((__m128i*)dst, _dst);
        }

        void Base64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t* dstSize)
        {
            assert(srcSize % 4 == 0 && srcSize >= 4);
            size_t srcSize16 = srcSize >= 15 ? AlignLoAny(srcSize - 15, 16) : 0;
            for (const uint8_t* body16 = src + srcSize16; src < body16; src += 16, dst += 12)
                Base64Decode12(src, dst);
            for (const uint8_t* body = src + srcSize - srcSize16 - 4; src < body; src += 4, dst += 3)
                Base::Base64Decode3(src, dst);
            *dstSize = srcSize / 4 * 3 + Base::Base64DecodeTail(src, dst) - 3;
        }

        //---------------------------------------------------------------------------------------------

        const __m128i K8_TO_SHUFFLE = SIMD_MM_SETR_EPI8(0x1, 0x0, 0x2, 0x1, 0x4, 0x3, 0x5, 0x4, 0x7, 0x6, 0x8, 0x7, 0xA, 0x9, 0xB, 0xA);

        const __m128i K16_TO_MULLO = SIMD_MM_SET2_EPI16(0x0010, 0x0100);

        const __m128i K16_TO_MULHI = SIMD_MM_SET2_EPI16(0x0040, 0x0400);

        const __m128i K8_UPP_END = SIMD_MM_SET1_EPI8(26);
        const __m128i K8_LOW_END = SIMD_MM_SET1_EPI8(52);

        const __m128i K8_TO_DIG_SHUFFLE = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/');

        SIMD_INLINE void Base64Encode12(const uint8_t* src, uint8_t* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i shuffle = _mm_shuffle_epi8(_src, K8_TO_SHUFFLE);
            __m128i mullo = _mm_mullo_epi16(shuffle, K16_TO_MULLO);
            __m128i mulhi = _mm_mulhi_epi16(shuffle, K16_TO_MULHI);
            __m128i index = _mm_or_si128(_mm_and_si128(mullo, K16_3F00), _mm_and_si128(mulhi, K16_003F));
            __m128i uppMask = _mm_cmpgt_epi8(K8_UPP_END, index);
            __m128i letMask = _mm_cmpgt_epi8(K8_LOW_END, index);
            __m128i uppValue = _mm_and_si128(_mm_add_epi8(index, K8_UPP_ADD), uppMask);
            __m128i lowValue = _mm_and_si128(_mm_add_epi8(index, K8_LOW_ADD), _mm_andnot_si128(uppMask, letMask));
            __m128i digValue = _mm_shuffle_epi8(K8_TO_DIG_SHUFFLE, index);
            __m128i _dst = _mm_or_si128(_mm_or_si128(uppValue, lowValue), _mm_andnot_si128(letMask, digValue));
            _mm_storeu_si128((__m128i*)dst, _dst);
        }

        void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst)
        {
            size_t size3 = AlignLoAny(size, 3);
            size_t size12 = size >= 11 ? AlignLoAny(size - 11, 12) : 0;
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

