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

#ifndef __SimdPrefetch_h__
#define __SimdPrefetch_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#if defined(SIMD_AVX_ENABLE)
    namespace Avx
    {
        const size_t PREFETCH_SIZE = 2048;

        SIMD_INLINE void PrefetchL1(const void* ptr)
        {
            _mm_prefetch((const char*)ptr + PREFETCH_SIZE, _MM_HINT_T0);
        }
    }
#endif

#if defined(SIMD_AVX512F_ENABLE)
    namespace Avx512f
    {
        const size_t PREFETCH_SIZE = 4096;

        SIMD_INLINE void PrefetchL1(const void* ptr)
        {
            _mm_prefetch((const char*)ptr + PREFETCH_SIZE, _MM_HINT_T0);
        }
    }
#endif
}

#endif//__SimdPrefetch_h__

