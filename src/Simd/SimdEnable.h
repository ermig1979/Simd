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
#ifndef __SimdEnable_h__
#define __SimdEnable_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        bool GetEnable();

        const bool Enable = GetEnable();
    }
#endif

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        bool GetEnable();

        const bool Enable = GetEnable();
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        bool GetEnable();

        const bool Enable = GetEnable();
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        bool GetEnable();

        const bool Enable = GetEnable();
    }
#endif

#ifdef SIMD_AVX512VNNI_ENABLE
    namespace Avx512vnni
    {
        bool GetEnable();

        const bool Enable = GetEnable();
    }
#endif

#if defined(SIMD_AVX512BF16_ENABLE) && !defined(SIMD_AMX_EMULATE)
    namespace Avx512bf16
    {
        bool GetEnable();

        const bool Enable = GetEnable();
    }
#endif

#if defined(SIMD_AMX_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
    namespace Amx
    {
        bool GetEnable();

        const bool Enable = GetEnable();
    }
#endif

#ifdef SIMD_VMX_ENABLE
    namespace Vmx
    {
        bool GetEnable();

        const bool Enable = GetEnable();
    }
#endif

#ifdef SIMD_VSX_ENABLE
    namespace Vsx
    {
        bool GetEnable();

        const bool Enable = GetEnable();
    }
#endif

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        bool GetEnable();

        const bool Enable = GetEnable();
    }
#endif
}

#define SIMD_BASE_FUNC(func) Simd::Base::func

#ifdef SIMD_SSE41_ENABLE
#define SIMD_SSE41_FUNC(func) Simd::Sse41::Enable ? Simd::Sse41::func : 
#else
#define SIMD_SSE41_FUNC(func) 
#endif

#ifdef SIMD_AVX_ENABLE
#define SIMD_AVX_FUNC(func) Simd::Avx::Enable ? Simd::Avx::func : 
#else
#define SIMD_AVX_FUNC(func)
#endif

#ifdef SIMD_AVX2_ENABLE
#define SIMD_AVX2_FUNC(func) Simd::Avx2::Enable ? Simd::Avx2::func : 
#else
#define SIMD_AVX2_FUNC(func)
#endif

#ifdef SIMD_AVX512BW_ENABLE
#define SIMD_AVX512BW_FUNC(func) Simd::Avx512bw::Enable ? Simd::Avx512bw::func : 
#else
#define SIMD_AVX512BW_FUNC(func)
#endif

#ifdef SIMD_AVX512VNNI_ENABLE
#define SIMD_AVX512VNNI_FUNC(func) Simd::Avx512vnni::Enable ? Simd::Avx512vnni::func : 
#else
#define SIMD_AVX512VNNI_FUNC(func)
#endif

#if defined(SIMD_AVX512BF16_ENABLE) && !defined(SIMD_AMX_EMULATE)
#define SIMD_AVX512BF16_FUNC(func) Simd::Avx512bf16::Enable ? Simd::Avx512bf16::func : 
#else
#define SIMD_AVX512BF16_FUNC(func)
#endif

#if defined(SIMD_AMX_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
#define SIMD_AMX_FUNC(func) Simd::Amx::Enable ? Simd::Amx::func : 
#else
#define SIMD_AMX_FUNC(func)
#endif

#ifdef SIMD_VMX_ENABLE
#define SIMD_VMX_FUNC(func) Simd::Vmx::Enable ? Simd::Vmx::func : 
#else
#define SIMD_VMX_FUNC(func)
#endif

#ifdef SIMD_VSX_ENABLE
#define SIMD_VSX_FUNC(func) Simd::Vsx::Enable ? Simd::Vsx::func : 
#else
#define SIMD_VSX_FUNC(func)
#endif

#ifdef SIMD_NEON_ENABLE
#define SIMD_NEON_FUNC(func) Simd::Neon::Enable ? Simd::Neon::func : 
#else
#define SIMD_NEON_FUNC(func)
#endif

#define SIMD_FUNC0(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC1(func, EXT1) EXT1(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC2(func, EXT1, EXT2) EXT1(func) EXT2(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC3(func, EXT1, EXT2, EXT3) EXT1(func) EXT2(func) EXT3(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC4(func, EXT1, EXT2, EXT3, EXT4) EXT1(func) EXT2(func) EXT3(func) EXT4(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC5(func, EXT1, EXT2, EXT3, EXT4, EXT5) EXT1(func) EXT2(func) EXT3(func) EXT4(func) EXT5(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC6(func, EXT1, EXT2, EXT3, EXT4, EXT5, EXT6) EXT1(func) EXT2(func) EXT3(func) EXT4(func) EXT5(func) EXT6(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC7(func, EXT1, EXT2, EXT3, EXT4, EXT5, EXT6, EXT7) EXT1(func) EXT2(func) EXT3(func) EXT4(func) EXT5(func) EXT6(func) EXT7(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC8(func, EXT1, EXT2, EXT3, EXT4, EXT5, EXT6, EXT7, EXT8) EXT1(func) EXT2(func) EXT3(func) EXT4(func) EXT5(func) EXT6(func) EXT7(func) EXT8(func) SIMD_BASE_FUNC(func)
#define SIMD_FUNC9(func, EXT1, EXT2, EXT3, EXT4, EXT5, EXT6, EXT7, EXT8, EXT9) EXT1(func) EXT2(func) EXT3(func) EXT4(func) EXT5(func) EXT6(func) EXT7(func) EXT8(func) EXT9(func) SIMD_BASE_FUNC(func)

#endif//__SimdEnable_h__
