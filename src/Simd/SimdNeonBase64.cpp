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
        const uint8x16_t K8_UPP_ADD = SIMD_VEC_SET1_EPI8('A');
        const uint8x16_t K8_LOW_ADD = SIMD_VEC_SET1_EPI8('a' - 26);

        const uint8x16_t K8_9 = SIMD_VEC_SET1_EPI8('9');
        const uint8x16_t K8_Z = SIMD_VEC_SET1_EPI8('Z');

        const uint8x16_t K8_PLUS = SIMD_VEC_SET1_EPI8('+');

        const uint8x16_t K8_FROM_DIG_SHUFFLE = SIMD_VEC_SETR_EPI8(62, 0xFF, 0xFF, 0xFF, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 0xFF);

        const uint8x16_t K8_0F = SIMD_VEC_SET1_EPI8(0x0F);
        const uint8x16_t K8_30 = SIMD_VEC_SET1_EPI8(0x30);
        const uint8x16_t K8_3C = SIMD_VEC_SET1_EPI8(0x3C);
        const uint8x16_t K8_3F = SIMD_VEC_SET1_EPI8(0x3F);

        SIMD_INLINE uint8x16_t FromBase64(uint8x16_t src)
        {
            uint8x16_t letMask = vcgtq_u8(src, K8_9);
            uint8x16_t lowMask = vcgtq_u8(src, K8_Z);
            uint8x16_t lowValue = vandq_u8(lowMask, vsubq_u8(src, K8_LOW_ADD));
            uint8x16_t uppValue = vandq_u8(vandq_u8(vmvnq_u8(lowMask), letMask), vsubq_u8(src, K8_UPP_ADD));
            uint8x16_t digValue = vandq_u8(vmvnq_u8(letMask), Shuffle(K8_FROM_DIG_SHUFFLE, vsubq_u8(src, K8_PLUS)));
            uint8x16_t dst = vorrq_u8(vorrq_u8(uppValue, lowValue), digValue);
            assert(TestZ((uint32x4_t)vcgeq_u8(dst, K8_40)));
            return dst;
        }

        SIMD_INLINE void Base64Decode48(const uint8_t* src, uint8_t* dst)
        {
            uint8x16x4_t _src = Load4<false>(src);
            _src.val[0] = FromBase64(_src.val[0]);
            _src.val[1] = FromBase64(_src.val[1]);
            _src.val[2] = FromBase64(_src.val[2]);
            _src.val[3] = FromBase64(_src.val[3]);
            uint8x16x3_t _dst;
            _dst.val[0] = vorrq_u8(vshlq_n_u8(_src.val[0], 2), vshrq_n_u8(vandq_u8(_src.val[1], K8_30), 4));
            _dst.val[1] = vorrq_u8(vshlq_n_u8(vandq_u8(_src.val[1], K8_0F), 4), vshrq_n_u8(vandq_u8(_src.val[2], K8_3C), 2));
            _dst.val[2] = vorrq_u8(vshlq_n_u8(vandq_u8(_src.val[2], K8_03), 6), _src.val[3]);// vshrq_n_u8(vandq_u8(_src.val[2], K8_03), 6);
            Store3<false>(dst, _dst);
        }

        void Base64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t* dstSize)
        {
            assert(srcSize % 4 == 0 && srcSize >= 4);
            size_t srcSize64 = srcSize >= 63 ? AlignLoAny(srcSize - 63, 64) : 0;
            for (const uint8_t* body64 = src + srcSize64; src < body64; src += 64, dst += 48)
                Base64Decode48(src, dst);
            for (const uint8_t* body = src + srcSize - srcSize64 - 4; src < body; src += 4, dst += 3)
                Base::Base64Decode3(src, dst);
            *dstSize = srcSize / 4 * 3 + Base::Base64DecodeTail(src, dst) - 3;
        }

        //---------------------------------------------------------------------------------------------

        const uint8x16_t K8_C0 = SIMD_VEC_SET1_EPI8(0xC0);
        const uint8x16_t K8_F0 = SIMD_VEC_SET1_EPI8(0xF0);
        const uint8x16_t K8_FC = SIMD_VEC_SET1_EPI8(0xFC);

        const uint8x16_t K8_UPP_END = SIMD_VEC_SET1_EPI8(26);
        const uint8x16_t K8_LOW_END = SIMD_VEC_SET1_EPI8(52);

        const uint8x16_t K8_TO_DIG_SHUFFLE = SIMD_VEC_SETR_EPI8('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/', 0xFF, 0xFF, 0xFF, 0xFF);

        SIMD_INLINE uint8x16_t ToBase64(uint8x16_t src)
        {
            uint8x16_t uppMask = vcgtq_u8(K8_UPP_END, src);
            uint8x16_t letMask = vcgtq_u8(K8_LOW_END, src);
            uint8x16_t uppValue = vandq_u8(vaddq_u8(src, K8_UPP_ADD), uppMask);
            uint8x16_t lowValue = vandq_u8(vaddq_u8(src, K8_LOW_ADD), vandq_u8(vmvnq_u8(uppMask), letMask));
            uint8x16_t digValue = Shuffle(K8_TO_DIG_SHUFFLE, vsubq_u8(src, K8_LOW_END));
            return vorrq_u8(vorrq_u8(uppValue, lowValue), vandq_u8(vmvnq_u8(letMask), digValue));
        }

        SIMD_INLINE void Base64Encode48(const uint8_t* src, uint8_t* dst)
        {
            uint8x16x3_t _src = Load3<false>(src);
            uint8x16x4_t _dst;
            _dst.val[0] = vshrq_n_u8(vandq_u8(_src.val[0], K8_FC), 2);
            _dst.val[1] = vorrq_u8(vshlq_n_u8(vandq_u8(_src.val[0], K8_03), 4), vshrq_n_u8(vandq_u8(_src.val[1], K8_F0), 4));
            _dst.val[2] = vorrq_u8(vshlq_n_u8(vandq_u8(_src.val[1], K8_0F), 2), vshrq_n_u8(vandq_u8(_src.val[2], K8_C0), 6));
            _dst.val[3] = vandq_u8(_src.val[2], K8_3F);
            _dst.val[0] = ToBase64(_dst.val[0]);
            _dst.val[1] = ToBase64(_dst.val[1]);
            _dst.val[2] = ToBase64(_dst.val[2]);
            _dst.val[3] = ToBase64(_dst.val[3]);
            Store4<false>(dst, _dst);
        }

        void Base64Encode(const uint8_t* src, size_t size, uint8_t* dst)
        {
            size_t size3 = AlignLoAny(size, 3);
            size_t size48 = size >= 47 ? AlignLoAny(size - 47, 48) : 0;
            for (const uint8_t* body48 = src + size48; src < body48; src += 48, dst += 64)
                Base64Encode48(src, dst);
            for (const uint8_t* body3 = src + size3 - size48; src < body3; src += 3, dst += 4)
                Base::Base64Encode3(src, dst);
            if (size - size3)
                Base::Base64EncodeTail(src, size - size3, dst);
        }
    }
#endif
}

