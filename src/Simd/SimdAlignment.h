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
#ifndef __SimdAlignment_h__
#define __SimdAlignment_h__

#include "Simd/SimdEnable.h"

namespace Simd
{
    SIMD_INLINE size_t GetAlignment()
    {
#ifdef SIMD_AVX512VNNI_ENABLE
        if (Avx512vnni::Enable)
            return sizeof(__m512i);
        else
#endif
#ifdef SIMD_AVX512BW_ENABLE
        if (Avx512bw::Enable)
            return sizeof(__m512i);
        else
#endif
#ifdef SIMD_AVX512F_ENABLE
        if (Avx512f::Enable)
            return sizeof(__m512);
        else
#endif
#ifdef SIMD_AVX2_ENABLE
        if (Avx2::Enable)
            return sizeof(__m256i);
        else
#endif
#ifdef SIMD_AVX_ENABLE
        if (Avx::Enable)
            return sizeof(__m256);
        else
#endif
#ifdef SIMD_SSE41_ENABLE
        if (Sse41::Enable)
            return sizeof(__m128i);
        else
#endif
#ifdef SIMD_SSE2_ENABLE
        if (Sse2::Enable)
            return sizeof(__m128i);
        else
#endif
#ifdef SIMD_VSX_ENABLE
        if (Vsx::Enable)
            return sizeof(__vector uint8_t);
        else
#endif
#ifdef SIMD_VMX_ENABLE
        if (Vmx::Enable)
            return sizeof(__vector uint8_t);
        else
#endif
#ifdef SIMD_NEON_ENABLE
        if (Neon::Enable)
            return sizeof(uint8x16_t);
        else
#endif
            return sizeof(void *);
    }

    extern const size_t ALIGNMENT;

    SIMD_INLINE size_t Alignment()
    {
#if defined(WIN32)
        return GetAlignment();
#else
        return ALIGNMENT;
#endif
    }
}

#endif//__SimdAlignment_h__
