/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#ifndef __SimdMemory_h__
#define __SimdMemory_h__

#include "Simd/SimdDefs.h"
#include "Simd/SimdMath.h"

namespace Simd
{
    SIMD_INLINE size_t AlignHi(size_t size, size_t align)
    {
        return (size + align - 1) & ~(align - 1);
    }

    SIMD_INLINE void* AlignHi(const void *p, size_t align)
    {
        return (void*)((((size_t)p) + align - 1) & ~(align - 1));
    }

    SIMD_INLINE size_t AlignLo(size_t size, size_t align)
    {
        return size & ~(align - 1);
    }

    SIMD_INLINE void* AlignLo(const void * p, size_t align)
    {
        return (void*)(((size_t)p) & ~(align - 1));
    }

    SIMD_INLINE bool Aligned(size_t size, size_t align)
    {
        return size%align == 0;
    }

    SIMD_INLINE bool Aligned(const void * p, size_t align)
    {
        return ((size_t)p)%align == 0;
    }

	SIMD_INLINE void * Allocate(size_t size, size_t align = SIMD_ALIGN)
	{
#if defined(_MSC_VER) 
        return _aligned_malloc(size, align);
#elif defined(__GNUC__)
        align = AlignHi(align, sizeof(void*));
        size = AlignHi(size, align);
        void * ptr;
        int result = ::posix_memalign(&ptr, align, size);
#ifdef SIMD_ALLOCATE_ASSERT
        assert(result == 0);
#endif
        return result ? NULL : ptr;
#else
		return malloc(size);
#endif
	}

	SIMD_INLINE void Free(void * p)
	{
#if defined(_MSC_VER) 
        _aligned_free(p);
#else
		free(p);
#endif
	}

#ifdef SIMD_SSE_ENABLE
    namespace Sse
    {
        SIMD_INLINE bool Aligned(size_t size, size_t align = sizeof(__m128))
        {
            return Simd::Aligned(size, align);
        }

        SIMD_INLINE bool Aligned(const void *p, size_t align = sizeof(__m128))
        {
            return Simd::Aligned(p, align);
        }
    }
#endif// SIMD_SSE_ENABLE

#ifdef SIMD_SSE2_ENABLE
	namespace Sse2
	{
        using Sse::Aligned;
	}
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSSE3_ENABLE
    namespace Ssse3
    {
        using Sse::Aligned;
    }
#endif// SIMD_SSSE3_ENABLE

#ifdef SIMD_SSE42_ENABLE
	namespace Sse42
	{
	}
#endif// SIMD_SSE42_ENABLE

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        SIMD_INLINE bool Aligned(size_t size, size_t align = sizeof(__m256))
        {
            return Simd::Aligned(size, align);
        }

        SIMD_INLINE bool Aligned(const void *p, size_t align = sizeof(__m256))
        {
            return Simd::Aligned(p, align);
        }
    }
#endif// SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE
	namespace Avx2
	{
        using Avx::Aligned;
    }
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_VMX_ENABLE
    namespace Vmx
    {
        SIMD_INLINE bool Aligned(size_t size, size_t align = sizeof(vec_uchar16))
        {
            return Simd::Aligned(size, align);
        }

        SIMD_INLINE bool Aligned(const void *p, size_t align = sizeof(vec_uchar16))
        {
            return Simd::Aligned(p, align);
        }
    }
#endif// SIMD_VMX_ENABLE

#ifdef SIMD_VSX_ENABLE
    namespace Vsx
    {
        using Vmx::Aligned;
    }
#endif// SIMD_VSX_ENABLE
}

#endif//__SimdMemory_h__
