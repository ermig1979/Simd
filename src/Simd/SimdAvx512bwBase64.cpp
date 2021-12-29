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
            0x0, 0x1, -1, -1, 0x1, 0x2, -1, -1, 0x3, 0x4, -1, -1, 0x4, 0x5, -1, -1);

        const __m512i K8_TO_BASE64_SHUFFLE = SIMD_MM512_SETR_EPI8(
            0x0, -1, 0x1, 0x0, 0x2, 0x1, 0x2, -1, 0x3, -1, 0x4, 0x3, 0x5, 0x4, 0x5, -1,
            0x2, -1, 0x3, 0x2, 0x4, 0x3, 0x4, -1, 0x5, -1, 0x6, 0x5, 0x7, 0x6, 0x7, -1,
            0x0, -1, 0x1, 0x0, 0x2, 0x1, 0x2, -1, 0x3, -1, 0x4, 0x3, 0x5, 0x4, 0x5, -1,
            0x2, -1, 0x3, 0x2, 0x4, 0x3, 0x4, -1, 0x5, -1, 0x6, 0x5, 0x7, 0x6, 0x7, -1);

        const __m512i K16_TO_BASE64_SHIFT = SIMD_MM512_SETR_EPI16(
            0x2, 0x4, 0x6, 0x0, 0x2, 0x4, 0x6, 0x0, 0x2, 0x4, 0x6, 0x0, 0x2, 0x4, 0x6, 0x0,
            0x2, 0x4, 0x6, 0x0, 0x2, 0x4, 0x6, 0x0, 0x2, 0x4, 0x6, 0x0, 0x2, 0x4, 0x6, 0x0);

        const __m512i K16_TO_BASE64_MASK = SIMD_MM512_SET1_EPI16(0x3F);

        const __m512i K16_TO_BASE64_0 = SIMD_MM512_SETR_EPI16(
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f');
        const __m512i K16_TO_BASE64_1 = SIMD_MM512_SETR_EPI16(
            'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/');

        void Base64Encode24(const uint8_t* src, __mmask32 srcMask, uint8_t* dst, __mmask32 dstMask)
        {
            __m512i _src = _mm512_castsi256_si512(_mm256_maskz_loadu_epi8(srcMask, src));
            __m512i permuted = _mm512_permutexvar_epi32(K32_TO_BASE64_PERMUTE, _src);
            __m512i shuffled = _mm512_shuffle_epi8(permuted, K8_TO_BASE64_SHUFFLE);
            __m512i shifted = _mm512_srlv_epi16(shuffled, K16_TO_BASE64_SHIFT);
            __m512i masked = _mm512_and_si512(shifted, K16_TO_BASE64_MASK);
            __m512i _dst = _mm512_permutex2var_epi16(K16_TO_BASE64_0, masked, K16_TO_BASE64_1);
            _mm256_mask_storeu_epi8(dst, dstMask, _mm512_cvtepi16_epi8(_dst));
        }

        void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst)
        {
            size_t size3 = AlignLoAny(size, 3);
            size_t size24 = AlignLoAny(size, 24);
            for (const uint8_t* body24 = src + size24; src < body24; src += 24, dst += 32)
                Base64Encode24(src, 0xFFFFFF, dst, 0xFFFFFFFF);
            if (size24 < size3)
            {
                size_t srcTail = size3 - size24, dstTail = (size3 - size24) / 3 * 4;
                __mmask32 srcMask = __mmask32(-1) >> (32 - srcTail);
                __mmask32 dstMask = __mmask32(-1) >> (32 - dstTail);
                Base64Encode24(src, srcMask, dst, dstMask);
                src += srcTail;
                dst += dstTail;
            }
            if(size - size3)
                Base::Base64EncodeTail(src, size - size3, dst);
        }
    }
#endif
}

