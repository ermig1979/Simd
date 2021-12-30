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
#ifndef __SimdBase64_h__
#define __SimdBase64_h__

#include "Simd/SimdInit.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE uint8_t ToBase64(uint8_t src)
        {
            assert(src < 64);
            static uint8_t toBase64[] =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz"
                "0123456789+/";
            return toBase64[src];
        }

        SIMD_INLINE void Base64Encode3(const uint8_t* src, uint8_t* dst)
        {
            dst[0] = ToBase64((src[0] & 0xfc) >> 2);
            dst[1] = ToBase64(((src[0] & 0x03) << 4) | ((src[1] & 0xf0) >> 4));
            dst[2] = ToBase64(((src[1] & 0x0f) << 2) | ((src[2] & 0xc0) >> 6));
            dst[3] = ToBase64(src[2] & 0x3f);
        }

        SIMD_INLINE void Base64EncodeTail(const uint8_t* src, size_t tail, uint8_t* dst)
        {
            uint8_t src1 = (tail > 1 ? src[1] : 0);
            uint8_t src2 = (tail > 2 ? src[2] : 0);
            dst[0] = ToBase64((src[0] & 0xfc) >> 2);
            dst[1] = ToBase64(((src[0] & 0x03) << 4) | ((src1 & 0xf0) >> 4));
            dst[2] = tail > 1 ? ToBase64(((src1 & 0x0f) << 2) | ((src2 & 0xc0) >> 6)) : '=';
            dst[3] = tail > 2 ? ToBase64(src2 & 0x3f) : '=';
        }
    }
}

#endif//__SimdBase64_h__
