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
#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void Reorder16bitX(const uint8_t * src, uint8_t * dst)
        {
            size_t value = *(size_t*)src;
#if defined (SIMD_X64_ENABLE) || defined(SIMD_PPC64_ENABLE) || defined (SIMD_ARM64_ENABLE)
            *(size_t*)dst = (value & 0xFF00FF00FF00FF00) >> 8 | (value & 0x00FF00FF00FF00FF) << 8;
#else
            *(size_t*)dst = (value & 0xFF00FF00) >> 8 | (value & 0x00FF00FF) << 8;
#endif
        }

        void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size % 2 == 0);

            size_t alignedSize = AlignLo(size, sizeof(size_t));
            for (size_t i = 0; i < alignedSize; i += sizeof(size_t))
                Reorder16bitX(src + i, dst + i);
            for (size_t i = alignedSize; i < size; i += 2)
                Reorder16bit(src + i, dst + i);
        }

        SIMD_INLINE void Reorder32bitX(const uint8_t * src, uint8_t * dst)
        {
#if defined (SIMD_X64_ENABLE) || defined(SIMD_PPC64_ENABLE) || defined (SIMD_ARM64_ENABLE)
            size_t value = *(size_t*)src;
            *(size_t*)dst =
                (value & 0x000000FF000000FF) << 24 | (value & 0x0000FF000000FF00) << 8 |
                (value & 0x00FF000000FF0000) >> 8 | (value & 0xFF000000FF000000) >> 24;
#else
            Reorder32bit(src, dst);
#endif
        }

        void Reorder32bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size % 4 == 0);

            size_t alignedSize = AlignLo(size, sizeof(size_t));
            for (size_t i = 0; i < alignedSize; i += sizeof(size_t))
                Reorder32bitX(src + i, dst + i);
            for (size_t i = alignedSize; i < size; i += 4)
                Reorder32bit(src + i, dst + i);
        }

        void Reorder64bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size % 8 == 0);

            for (size_t i = 0; i < size; i += 8)
                Reorder64bit(src + i, dst + i);
        }
    }
}
