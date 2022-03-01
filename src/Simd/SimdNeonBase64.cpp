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
#include "Simd/SimdCompare.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase64.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdShuffle.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        const uint8x16_t K8_3F_00 = SIMD_VEC_SET2_EPI8(0x3F, 0x00);
        const uint8x16_t K8_00_3F = SIMD_VEC_SET2_EPI8(0x00, 0x3F);

        const uint8x16_t K8_UPP_ADD = SIMD_VEC_SET1_EPI8('A');
        const uint8x16_t K8_LOW_ADD = SIMD_VEC_SET1_EPI8('a' - 26);

        const uint8x16_t K8_9 = SIMD_VEC_SET1_EPI8('9');
        const uint8x16_t K8_Z = SIMD_VEC_SET1_EPI8('Z');

        const uint8x16_t K8_PLUS = SIMD_VEC_SET1_EPI8('+');

        const int16x8_t K16_FROM_MULLO = SIMD_VEC_SET2_EPI16(10, 6);
        const int16x8_t K16_FROM_MULHI = SIMD_VEC_SET2_EPI16(-4, -8);

        const uint8x16_t K8_FROM_SHUFFLE_LO = SIMD_VEC_SETR_EPI8(0x1, 0x3, 0x2, 0x5, 0x7, 0x6, 0x9, 0xB, 0xA, 0xD, 0xF, 0xE, 0xFF, 0xFF, 0xFF, 0xFF);
        const uint8x16_t K8_FROM_SHUFFLE_HI = SIMD_VEC_SETR_EPI8(0x1, 0x0, 0x2, 0x5, 0x4, 0x6, 0x9, 0x8, 0xA, 0xD, 0xC, 0xE, 0xFF, 0xFF, 0xFF, 0xFF);

        const uint8x16_t K8_FROM_DIG_SHUFFLE = SIMD_VEC_SETR_EPI8(62, 0xFF, 0xFF, 0xFF, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0xFF);

        SIMD_INLINE void Base64Decode12(const uint8_t* src, uint8_t* dst)
        {
            uint8x16_t _src = Load<false>(src);
            uint8x16_t letMask = vcgtq_u8(_src, K8_9);
            uint8x16_t lowMask = vcgtq_u8(_src, K8_Z);
            uint8x16_t lowValue = vandq_u8(lowMask, vsubq_u8(_src, K8_LOW_ADD));
            uint8x16_t uppValue = vandq_u8(vandq_u8(vmvnq_u8(lowMask), letMask), vsubq_u8(_src, K8_UPP_ADD));
            uint8x16_t digValue = vandq_u8(vmvnq_u8(letMask), Shuffle(K8_FROM_DIG_SHUFFLE, vsubq_u8(_src, K8_PLUS)));
            uint8x16_t from = vorrq_u8(vorrq_u8(uppValue, lowValue), digValue);
            //assert(TestZ(GreaterOrEqual8u(from, _mm_set1_epi8(64))));
            uint8x16_t mullo = (uint8x16_t)vshlq_u16((uint16x8_t)vandq_u8(from, K8_3F_00), K16_FROM_MULLO);
            uint8x16_t mulhi = (uint8x16_t)vshlq_u16((uint16x8_t)vandq_u8(from, K8_00_3F), K16_FROM_MULHI);
            uint8x16_t shuffleHi = Shuffle(mullo, K8_FROM_SHUFFLE_LO);
            uint8x16_t shuffleLo = Shuffle(mulhi, K8_FROM_SHUFFLE_HI);
            uint8x16_t _dst = vorrq_u8(shuffleLo, shuffleHi);
            Store<false>(dst, _dst);
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
    }
#endif
}

