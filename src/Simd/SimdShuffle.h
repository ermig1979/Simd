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
#ifndef __SimdShuffle_h__
#define __SimdShuffle_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        SIMD_INLINE uint8x16_t Shuffle(const uint8x16_t& src, const uint8x16_t& shuffle)
        {
            return vcombine_u8(vtbl2_u8((const uint8x8x2_t&)src, vget_low_u8(shuffle)), vtbl2_u8((const uint8x8x2_t&)src, vget_high_u8(shuffle)));
        }

        SIMD_INLINE uint8x16_t Shuffle(const uint8x16x2_t& src, const uint8x16_t& shuffle)
        {
            return vcombine_u8(vtbl4_u8((const uint8x8x4_t&)src, vget_low_u8(shuffle)), vtbl4_u8((const uint8x8x4_t&)src, vget_high_u8(shuffle)));
        }
    }
#endif// SIMD_NEON_ENABLE
}

#endif//__SimdShuffle_h__
