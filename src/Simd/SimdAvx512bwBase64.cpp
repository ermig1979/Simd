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
#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        const __m512i K32_TO_BASE64_PERMUTE = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);

        const __m512i K8_TO_BASE64_SHUFFLE = SIMD_MM512_SETR_EPI8(
            0x1, 0x0, 0x2, 0x1, 0x4, 0x3, 0x5, 0x4, 0x7, 0x6, 0x8, 0x7, 0xA, 0x9, 0xB, 0xA,
            0x1, 0x0, 0x2, 0x1, 0x4, 0x3, 0x5, 0x4, 0x7, 0x6, 0x8, 0x7, 0xA, 0x9, 0xB, 0xA,
            0x1, 0x0, 0x2, 0x1, 0x4, 0x3, 0x5, 0x4, 0x7, 0x6, 0x8, 0x7, 0xA, 0x9, 0xB, 0xA,
            0x1, 0x0, 0x2, 0x1, 0x4, 0x3, 0x5, 0x4, 0x7, 0x6, 0x8, 0x7, 0xA, 0x9, 0xB, 0xA);

        const __m512i K16_TO_BASE64_MULLO = SIMD_MM512_SET2_EPI16(0x0010, 0x0100);

        const __m512i K16_TO_BASE64_MULHI = SIMD_MM512_SET2_EPI16(0x0040, 0x0400);

        const __m512i K16_003F = SIMD_MM512_SET1_EPI16(0x003F);
        const __m512i K16_3F00 = SIMD_MM512_SET1_EPI16(0x3F00);

        const __m512i K8_UPP_ADD = SIMD_MM512_SET1_EPI8('A');
        const __m512i K8_UPP_END = SIMD_MM512_SET1_EPI8(26);
        const __m512i K8_LOW_ADD = SIMD_MM512_SET1_EPI8('a' - 26);
        const __m512i K8_LOW_END = SIMD_MM512_SET1_EPI8(52);

        const __m512i K8_DIG_SHUFFLE = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/',
            -1, -1, -1, -1, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/',
            -1, -1, -1, -1, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/',
            -1, -1, -1, -1, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/');

        void Base64Encode48(const uint8_t* src, __mmask64 srcMask, uint8_t* dst, __mmask64 dstMask)
        {
            __m512i _src = _mm512_maskz_loadu_epi8(srcMask, src);
            __m512i permute = _mm512_permutexvar_epi32(K32_TO_BASE64_PERMUTE, _src);
            __m512i shuffle = _mm512_shuffle_epi8(permute, K8_TO_BASE64_SHUFFLE);
            __m512i mullo = _mm512_mullo_epi16(shuffle, K16_TO_BASE64_MULLO);
            __m512i mulhi = _mm512_mulhi_epi16(shuffle, K16_TO_BASE64_MULHI);
            __m512i index = _mm512_or_si512(_mm512_and_si512(mullo, K16_3F00), _mm512_and_si512(mulhi, K16_003F));
            __mmask64 uppMask = _mm512_cmp_epi8_mask(K8_UPP_END, index, _MM_CMPINT_NLE);
            __mmask64 letMask = _mm512_cmp_epi8_mask(K8_LOW_END, index, _MM_CMPINT_NLE);
            __m512i uppValue = _mm512_maskz_add_epi8(uppMask, index, K8_UPP_ADD);
            __m512i lowValue = _mm512_maskz_add_epi8((~uppMask)&letMask, index, K8_LOW_ADD);
            __m512i digValue = _mm512_maskz_shuffle_epi8(~letMask, K8_DIG_SHUFFLE, index);
            __m512i _dst = _mm512_or_si512(_mm512_or_si512(uppValue, lowValue), digValue);
            _mm512_mask_storeu_epi8(dst, dstMask, _dst);
        }

        void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst)
        {
            size_t size3 = AlignLoAny(size, 3);
            size_t size48 = AlignLoAny(size, 48);
            for (const uint8_t* body48 = src + size48; src < body48; src += 48, dst += 64)
                Base64Encode48(src, 0x0000FFFFFFFFFFFF, dst, 0xFFFFFFFFFFFFFFFF);
            if (size48 < size3)
            {
                size_t srcTail = size3 - size48, dstTail = (size3 - size48) / 3 * 4;
                __mmask64 srcMask = __mmask64(-1) >> (64 - srcTail);
                __mmask64 dstMask = __mmask64(-1) >> (64 - dstTail);
                Base64Encode48(src, srcMask, dst, dstMask);
                src += srcTail;
                dst += dstTail;
            }
            if (size - size3)
                Base::Base64EncodeTail(src, size - size3, dst);
        }
    }
#endif
}

