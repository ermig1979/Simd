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
#ifndef __SimdCopyPixel_h__
#define __SimdCopyPixel_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
    namespace Base
    {
        template<size_t N> SIMD_INLINE void CopyPixel(const uint8_t* src, uint8_t* dst)
        {
            for (size_t i = 0; i < N; ++i)
                dst[i] = src[i];
        }

        template<> SIMD_INLINE void CopyPixel<1>(const uint8_t* src, uint8_t* dst)
        {
            dst[0] = src[0];
        }

        template<> SIMD_INLINE void CopyPixel<2>(const uint8_t* src, uint8_t* dst)
        {
            ((uint16_t*)dst)[0] = ((uint16_t*)src)[0];
        }

        template<> SIMD_INLINE void CopyPixel<3>(const uint8_t* src, uint8_t* dst)
        {
            ((uint16_t*)dst)[0] = ((uint16_t*)src)[0];
            dst[2] = src[2];
        }

        template<> SIMD_INLINE void CopyPixel<4>(const uint8_t* src, uint8_t* dst)
        {
            ((uint32_t*)dst)[0] = ((uint32_t*)src)[0];
        }

        template<> SIMD_INLINE void CopyPixel<6>(const uint8_t* src, uint8_t* dst)
        {
            ((uint32_t*)dst)[0] = ((uint32_t*)src)[0];
            ((uint16_t*)dst)[2] = ((uint16_t*)src)[2];
        }

        template<> SIMD_INLINE void CopyPixel<8>(const uint8_t* src, uint8_t* dst)
        {
            ((uint64_t*)dst)[0] = ((uint64_t*)src)[0];
        }

        template<> SIMD_INLINE void CopyPixel<12>(const uint8_t* src, uint8_t* dst)
        {
            ((uint64_t*)dst)[0] = ((uint64_t*)src)[0];
            ((uint32_t*)dst)[2] = ((uint32_t*)src)[2];
        }
    }
}

#endif//__SimdCopyPixel_h__
