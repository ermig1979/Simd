/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"

#if defined(SIMD_NEON_ENABLE) && defined(SIMD_ARM64_ENABLE)
#include <arm_acle.h>
#endif

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_ARM64_ENABLE)
    namespace Neon
    {
        SIMD_INLINE void Crc32(uint32_t& crc, const uint64_t* p, const uint64_t* end)
        {
            while (p < end)
                crc = __crc32d(crc, *p++);
        }

        SIMD_INLINE void Crc32(uint32_t& crc, const uint8_t* p, const uint8_t* end)
        {
            while (p < end)
                crc = __crc32b(crc, *p++);
        }

        uint32_t Crc32(const void* src, size_t size)
        {
            uint8_t* nose = (uint8_t*)src;
            uint64_t* body = (uint64_t*)AlignHi(nose, sizeof(uint64_t));
            uint64_t* tail = (uint64_t*)AlignLo(nose + size, sizeof(uint64_t));

            uint32_t crc = 0xFFFFFFFF;
            Crc32(crc, nose, (uint8_t*)body);
            Crc32(crc, body, tail);
            Crc32(crc, (uint8_t*)tail, nose + size);
            return ~crc;
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void Crc32c(uint32_t& crc, const uint64_t* p, const uint64_t* end)
        {
            while (p < end)
                crc = __crc32cd(crc, *p++);
        }

        SIMD_INLINE void Crc32c(uint32_t& crc, const uint8_t* p, const uint8_t* end)
        {
            while (p < end)
                crc = __crc32cb(crc, *p++);
        }

        uint32_t Crc32c(const void* src, size_t size)
        {
            uint8_t* nose = (uint8_t*)src;
            uint64_t* body = (uint64_t*)AlignHi(nose, sizeof(uint64_t));
            uint64_t* tail = (uint64_t*)AlignLo(nose + size, sizeof(uint64_t));

            uint32_t crc = 0xFFFFFFFF;
            Crc32c(crc, nose, (uint8_t*)body);
            Crc32c(crc, body, tail);
            Crc32c(crc, (uint8_t*)tail, nose + size);
            return ~crc;
        }
    }
#endif
}
