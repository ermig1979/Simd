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

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        SIMD_INLINE void Crc32c(size_t & crc, const size_t * p, const size_t * end)
        {
            while (p < end)
            {
#ifdef SIMD_X64_ENABLE
                crc = _mm_crc32_u64(crc, *p++);
#else
                crc = _mm_crc32_u32(crc, *p++);
#endif
            }
        }

        SIMD_INLINE void Crc32c(size_t & crc, const uint8_t * p, const uint8_t * end)
        {
            while (p < end)
                crc = _mm_crc32_u8((uint32_t)crc, *p++);
        }

        uint32_t Crc32c(const void *src, size_t size)
        {
            uint8_t * nose = (uint8_t*)src;
            size_t * body = (size_t*)AlignHi(nose, sizeof(size_t));
            size_t * tail = (size_t*)AlignLo(nose + size, sizeof(size_t));

            size_t crc = 0xFFFFFFFF;
            Crc32c(crc, nose, (uint8_t*)body);
            Crc32c(crc, body, tail);
            Crc32c(crc, (uint8_t*)tail, nose + size);
            return ~(uint32_t)crc;
        }
    }
#endif// SIMD_SSE41_ENABLE
}
