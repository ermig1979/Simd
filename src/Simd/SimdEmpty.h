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
#ifndef __SimdEmpty_h__
#define __SimdEmpty_h__

#include "Simd/SimdEnable.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        SIMD_INLINE void Empty()
        {
#if defined(_MSC_VER) && defined(SIMD_X64_ENABLE)
#else
            _mm_empty();
#endif
        }

        struct EmptyCaller
        {
            SIMD_INLINE ~EmptyCaller()
            {
                if (Enable)
                    Empty();
            }
        };
    }
#endif
}

#if defined(SIMD_SSE41_ENABLE) && !(defined(_MSC_VER) && defined(SIMD_X64_ENABLE))
#define SIMD_EMPTY() Simd::Sse41::EmptyCaller emptyCaller;
#else
#define SIMD_EMPTY() 
#endif

#endif
