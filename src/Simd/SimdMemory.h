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
#ifndef __SimdMemory_h__
#define __SimdMemory_h__

#include "Simd/SimdDefs.h"

#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
    #define SIMD_USE_POSIX_MEMALIGN
#endif

#ifdef __unix__
    #include <alloca.h>
#endif

namespace Simd
{
#if defined(SIMD_SSE2_ENABLE) || defined(SIMD_SSE42_ENABLE)
    const size_t DEFAULT_MEMORY_ALIGN = sizeof(__m128i);
#else
    const size_t DEFAULT_MEMORY_ALIGN = 1;
#endif

    //-------------------------------------------------------------------------

    SIMD_INLINE void* Allocate(size_t size, size_t align = DEFAULT_MEMORY_ALIGN) 
    {
#ifdef SIMD_USE_POSIX_MEMALIGN
        void* memptr = NULL;
        posix_memalign(&memptr, align, size);
        return memptr;
#else
        return _aligned_malloc(size, align);
#endif
    }

    SIMD_INLINE void Free(void *p) 
    {
#ifdef SIMD_USE_POSIX_MEMALIGN
        free(p);
#else
        _aligned_free(p);
#endif
    }

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

    SIMD_INLINE void* AlignLo(const void *p, size_t align)
    {
        return (void*)(((size_t)p) & ~(align - 1));
    }

    SIMD_INLINE bool Aligned(size_t size, size_t align)
    {
        return size%align == 0;
    }

    SIMD_INLINE bool Aligned(const void *p, size_t align)
    {
        return ((size_t)p)%align == 0;
    }

#ifdef SIMD_SSE2_ENABLE    
	namespace Sse2
	{
		SIMD_INLINE bool Aligned(size_t size, size_t align = sizeof(__m128i))
		{
			return Simd::Aligned(size, align);
		}

		SIMD_INLINE bool Aligned(const void *p, size_t align = sizeof(__m128i))
		{
			return Simd::Aligned(p, align);
		}
	}
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSE42_ENABLE    
	namespace Sse42
	{
		using namespace Sse2;
	}
#endif// SIMD_SSE42_ENABLE
}

#if defined(_WIN64) || defined(_WIN32) 
    #define SIMD_ALLOCA(type, size) \
        (type*) Simd::AlignHi(_alloca(Simd::AlignHi((size)*sizeof(type), Simd::DEFAULT_MEMORY_ALIGN) + Simd::DEFAULT_MEMORY_ALIGN))
#elif defined __unix__
    #define SIMD_ALLOCA(type, size) \
        (type*) Simd::AlignHi(alloca(Simd::AlignHi((size)*sizeof(type), Simd::DEFAULT_MEMORY_ALIGN) + Simd::DEFAULT_MEMORY_ALIGN))
#else
    #error Do not know how to allocate memory on stack
#endif

#define SIMD_ARRAY(type, name, size) \
	type * name = SIMD_ALLOCA(type, size)

#endif//__SimdMemory_h__
