/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        bool SupportedByCPU();
        bool SupportedByOS();

        const bool Enable = SupportedByCPU() && SupportedByOS();
    }
    
#define SIMD_SSE2_INIT_FUNCTION_PTR(type, sse2, base) \
    (Simd::Sse2::Enable ? (type)(sse2) : (type)(base))

#else// SIMD_SSE2_ENABLE

#define SIMD_SSE2_INIT_FUNCTION_PTR(type, sse2, base) \
    (type)(base)

#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSE42_ENABLE
    namespace Sse42
    {
        bool SupportedByCPU();
        bool SupportedByOS();

        const bool Enable = SupportedByCPU() && SupportedByOS();
    }

#define SIMD_SSE42_INIT_FUNCTION_PTR(type, sse42, base) \
    (Simd::Sse42::Enable ? (type)(sse42) : (type)(base))

#else// SIMD_SSE42_ENABLE

#define SIMD_SSE42_INIT_FUNCTION_PTR(type, sse2, base) \
    (type)(base)

#endif// SIMD_SSE42_ENABLE

#ifdef SIMD_AVX_ENABLE
	namespace Avx
	{
		bool SupportedByCPU();
		bool SupportedByOS();

		const bool Enable = SupportedByCPU() && SupportedByOS();
	}

#define SIMD_AVX_INIT_FUNCTION_PTR(type, avx, base) \
	(Simd::Avx::Enable ? (type)(avx) : (type)(base))

#else// SIMD_AVX_ENABLE

#define SIMD_AVX_INIT_FUNCTION_PTR(type, avx, base) \
	(type)(base)

#endif// SIMD_AVX_ENABLE
}
#endif//__SimdEnable_h__