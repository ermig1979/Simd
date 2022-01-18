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
#ifndef __SimdBase64_h__
#define __SimdBase64_h__

#include "Simd/SimdInit.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int32_t FromBase64Table(uint8_t src)
        {
            static const int32_t fromBase64Table[256] =
            {
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  62, 255, 255, 255,  63,
                52,  53,  54,  55,  56,  57,  58,  59,  60,  61, 255, 255, 255, 255, 255, 255,
                255,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
                15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 255, 255, 255, 255,  63,
                255,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
                41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51, 255, 255, 255, 255, 255
            };
            return fromBase64Table[src];
        }

        SIMD_INLINE void Base64Decode3(const uint8_t* src, uint8_t* dst)
        {
            int32_t s0 = FromBase64Table(src[0]);
            int32_t s1 = FromBase64Table(src[1]);
            int32_t s2 = FromBase64Table(src[2]);
            int32_t s3 = FromBase64Table(src[3]);
            assert(s0 < 64 && s1 < 64 && s2 < 64 && s3 < 64);
            int32_t n = s0 << 18 | s1 << 12 | s2 << 6 | s3;
            dst[0] = n >> 16;
            dst[1] = n >> 8 & 0xFF;
            dst[2] = n & 0xFF;
        }

        SIMD_INLINE size_t Base64DecodeTail(const uint8_t* src, uint8_t* dst)
        {
            int32_t s0 = FromBase64Table(src[0]);
            int32_t s1 = FromBase64Table(src[1]);
            assert(s0 < 64 && s1 < 64);
            int32_t n = s0 << 18 | s1 << 12;
            dst[0] = n >> 16;
            size_t tail = 1;
            if (src[2] != '=')
            {
                int32_t s2 = FromBase64Table(src[2]);
                assert(s2 < 64);
                n |= s2 << 6;
                dst[1] = n >> 8 & 0xFF;
                tail++;
                if (src[3] != '=')
                {
                    int32_t s3 = FromBase64Table(src[3]);
                    assert(s3 < 64);
                    n |= s3;
                    dst[2] = n & 0xFF;
                    tail++;
                }
            }
            return tail;
        }

        //---------------------------------------------------------------------------------------------

        SIMD_INLINE uint8_t ToBase64Table(uint8_t src)
        {
            assert(src < 64);
            static const uint8_t toBase64Table[] =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz"
                "0123456789+/";
            return toBase64Table[src];
        }

        SIMD_INLINE void Base64Encode3(const uint8_t* src, uint8_t* dst)
        {
            dst[0] = ToBase64Table((src[0] & 0xfc) >> 2);
            dst[1] = ToBase64Table(((src[0] & 0x03) << 4) | ((src[1] & 0xf0) >> 4));
            dst[2] = ToBase64Table(((src[1] & 0x0f) << 2) | ((src[2] & 0xc0) >> 6));
            dst[3] = ToBase64Table(src[2] & 0x3f);
        }

        SIMD_INLINE void Base64EncodeTail(const uint8_t* src, size_t tail, uint8_t* dst)
        {
            uint8_t src1 = (tail > 1 ? src[1] : 0);
            uint8_t src2 = (tail > 2 ? src[2] : 0);
            dst[0] = ToBase64Table((src[0] & 0xfc) >> 2);
            dst[1] = ToBase64Table(((src[0] & 0x03) << 4) | ((src1 & 0xf0) >> 4));
            dst[2] = tail > 1 ? ToBase64Table(((src1 & 0x0f) << 2) | ((src2 & 0xc0) >> 6)) : '=';
            dst[3] = tail > 2 ? ToBase64Table(src2 & 0x3f) : '=';
        }
    }
}

#endif//__SimdBase64_h__
