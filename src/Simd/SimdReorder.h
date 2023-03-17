/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#ifndef __SimdReorder_h__
#define __SimdReorder_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void Reorder16bit(const uint8_t * src, uint8_t * dst)
        {
            uint16_t value = *(uint16_t*)src;
            *(uint16_t*)dst = value >> 8 | value << 8;
        }

        SIMD_INLINE void Reorder32bit(const uint8_t * src, uint8_t * dst)
        {
            uint32_t value = *(uint32_t*)src;
            *(uint32_t*)dst =
                (value & 0x000000FF) << 24 | (value & 0x0000FF00) << 8 |
                (value & 0x00FF0000) >> 8 | (value & 0xFF000000) >> 24;
        }

        SIMD_INLINE void Reorder64bit(const uint8_t * src, uint8_t * dst)
        {
            uint64_t value = *(uint64_t*)src;
            *(uint64_t*)dst =
                (value & 0x00000000000000FF) << 56 | (value & 0x000000000000FF00) << 40 |
                (value & 0x0000000000FF0000) << 24 | (value & 0x00000000FF000000) << 8 |
                (value & 0x000000FF00000000) >> 8 | (value & 0x0000FF0000000000) >> 24 |
                (value & 0x00FF000000000000) >> 40 | (value & 0xFF00000000000000) >> 56;
        }
    }
}
#endif//__SimdReorder_h__
