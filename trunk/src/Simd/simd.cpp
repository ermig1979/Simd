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

#include "Simd/simd.h"

#include "Simd/SimdEnable.h"
#include "Simd/SimdVersion.h"
#include "Simd/SimdConst.h"

#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdAvx2.h"

#ifdef WIN32
#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD dwReasonForCall, LPVOID lpReserved)
{
    switch(dwReasonForCall)
    {
    case DLL_PROCESS_DETACH:
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        return TRUE;
    }
    return TRUE;
}
#endif//WIN32

SIMD_API const char * simd_version()
{
    return SIMD_VERSION;
}

SIMD_API int simd_sse2_enable()
{
#ifdef SIMD_SSE2_ENABLE
    return Simd::Sse2::Enable ? SIMD_TRUE : SIMD_FALSE;
#else
    return SIMD_FALSE;
#endif
}

SIMD_API int simd_sse42_enable()
{
#ifdef SIMD_SSE42_ENABLE
    return Simd::Sse42::Enable ? SIMD_TRUE : SIMD_FALSE;
#else
    return SIMD_FALSE;
#endif
}

SIMD_API int simd_avx_enable()
{
#ifdef SIMD_AVX_ENABLE
    return Simd::Avx::Enable ? SIMD_TRUE : SIMD_FALSE;
#else
    return SIMD_FALSE;
#endif
}

SIMD_API int simd_avx2_enable()
{
#ifdef SIMD_AVX2_ENABLE
    return Simd::Avx2::Enable ? SIMD_TRUE : SIMD_FALSE;
#else
    return SIMD_FALSE;
#endif
}

SIMD_API void simd_abs_difference_sum(const unsigned char *a, size_t a_stride, const unsigned char * b, size_t b_stride, 
    size_t width, size_t height, unsigned long long * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Simd::Avx2::Enable && width >= Simd::Avx2::A)
        Simd::Avx2::AbsDifferenceSum(a, a_stride, b, b_stride, width, height, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Simd::Sse2::Enable && width >= Simd::Sse2::A)
        Simd::Sse2::AbsDifferenceSum(a, a_stride, b, b_stride, width, height, sum);
    else
#endif
        Simd::Base::AbsDifferenceSum(a, a_stride, b, b_stride, width, height, sum);
}

SIMD_API void simd_masked_abs_difference_sum(const unsigned char *a, size_t a_stride, const unsigned char *b, size_t b_stride, 
    const unsigned char *mask, size_t mask_stride, unsigned char index, size_t width, size_t height, unsigned long long * sum)
{
#ifdef SIMD_AVX2_ENABLE
    if(Simd::Avx2::Enable && width >= Simd::Avx2::A)
        Simd::Avx2::AbsDifferenceSum(a, a_stride, b, b_stride, mask, mask_stride, index, width, height, sum);
    else
#endif
#ifdef SIMD_SSE2_ENABLE
    if(Simd::Sse2::Enable && width >= Simd::Sse2::A)
        Simd::Sse2::AbsDifferenceSum(a, a_stride, b, b_stride, mask, mask_stride, index, width, height, sum);
    else
#endif
        Simd::Base::AbsDifferenceSum(a, a_stride, b, b_stride, mask, mask_stride, index, width, height, sum);
}

SIMD_API void simd_abs_gradient_saturated_sum(const unsigned char * src, size_t src_stride, size_t width, size_t height, 
                                             unsigned char * dst, size_t dst_stride)
{
#ifdef SIMD_AVX2_ENABLE
    if(Simd::Avx2::Enable && width >= Simd::Avx2::A)
        Simd::Avx2::AbsGradientSaturatedSum(src, src_stride, width, height, dst, dst_stride);
    else
#endif//SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
    if(Simd::Sse2::Enable && width >= Simd::Sse2::A)
        Simd::Sse2::AbsGradientSaturatedSum(src, src_stride, width, height, dst, dst_stride);
    else
#endif//SIMD_SSE2_ENABLE
        Simd::Base::AbsGradientSaturatedSum(src, src_stride, width, height, dst, dst_stride);
}







