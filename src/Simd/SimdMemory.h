/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
*               2016-2016 Sintegrial Technologies.
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

#if defined(SIMD_ALLOCATE_ERROR_MESSAGE)
#include <iostream>
#endif
#include <memory>

namespace Simd
{
    SIMD_INLINE size_t DivHi(size_t value, size_t divider)
    {
        return (value + divider - 1) / divider;
    }

    SIMD_INLINE size_t Pow2Hi(size_t value)
    {
        size_t pow2 = 1;
        for (; pow2 < value; pow2 *= 2);
        return pow2;
    }

    SIMD_INLINE size_t AlignHiAny(size_t size, size_t align)
    {
        return (size + align - 1) / align * align;
    }

    SIMD_INLINE size_t AlignLoAny(size_t size, size_t align)
    {
        return size / align * align;
    }

    SIMD_INLINE size_t AlignHi(size_t size, size_t align)
    {
        return (size + align - 1) & ~(align - 1);
    }

    SIMD_INLINE void * AlignHi(const void * ptr, size_t align)
    {
        return (void *)((((size_t)ptr) + align - 1) & ~(align - 1));
    }

    SIMD_INLINE size_t AlignLo(size_t size, size_t align)
    {
        return size & ~(align - 1);
    }

    SIMD_INLINE void * AlignLo(const void * ptr, size_t align)
    {
        return (void *)(((size_t)ptr) & ~(align - 1));
    }

    SIMD_INLINE bool Aligned(size_t size, size_t align)
    {
        return size == AlignLo(size, align);
    }

    SIMD_INLINE bool Aligned(const void * ptr, size_t align)
    {
        return ptr == AlignLo(ptr, align);
    }

    SIMD_INLINE void * Allocate(size_t size, size_t align = SIMD_ALIGN)
    {
#ifdef SIMD_NO_MANS_LAND
        size += 2 * SIMD_NO_MANS_LAND;
#endif
        void * ptr = NULL;
#if defined(_MSC_VER) 
        ptr = _aligned_malloc(size, align);
#elif defined(__MINGW32__) || defined(__MINGW64__)
        ptr = __mingw_aligned_malloc(size, align);
#elif defined(__GNUC__)
        align = AlignHi(align, sizeof(void *));
        size = AlignHi(size, align);
        int result = ::posix_memalign(&ptr, align, size);
        if (result != 0)
            ptr = NULL;
#else
        ptr = malloc(size);
#endif
#ifdef SIMD_ALLOCATE_ERROR_MESSAGE
        if (ptr == NULL)
            std::cout << "The function posix_memalign can't allocate " << size << " bytes with align " << align << " !" << std::endl << std::flush;
#endif
#ifdef SIMD_ALLOCATE_ASSERT
        assert(ptr);
#endif
#ifdef SIMD_NO_MANS_LAND
        if (ptr)
            ptr = (char*)ptr + SIMD_NO_MANS_LAND;
#endif
        return ptr;
    }

    template<class T> T* Allocate(uint8_t*& buffer, size_t size, size_t align = SIMD_ALIGN)
    {
        T* ptr = (T*)buffer;
        buffer = buffer + AlignHi(size * sizeof(T), align);
        return ptr;
    }

    SIMD_INLINE void Free(void * ptr)
    {
#ifdef SIMD_NO_MANS_LAND
        if (ptr)
            ptr = (char*)ptr - SIMD_NO_MANS_LAND;
#endif
#if defined(_MSC_VER) 
        _aligned_free(ptr);
#elif defined(__MINGW32__) || defined(__MINGW64__)
        return __mingw_aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    //---------------------------------------------------------------------------------------------

    struct Deletable
    {
        virtual ~Deletable() {}
    };

    //---------------------------------------------------------------------------------------------

#if defined(SIMD_CPP_2011_ENABLE)
    template<class T> using Holder = std::unique_ptr<T>;
#else
    template <class T> class Holder
    {
        T* _ptr;

    public:
        Holder(T* ptr)
            : _ptr(ptr)
        {
        }

        ~Holder()
        {
            if (_ptr)
                delete _ptr;
        }

        T& operator * ()
        {
            return *_ptr;
        }

        const T& operator * () const
        {
            return *_ptr;
        }

        T* operator -> ()
        {
            return _ptr;
        }

        const T* operator -> () const
        {
            return _ptr;
        }

        operator bool() const 
        {
            return _ptr != NULL;
        }
    };
#endif

    //---------------------------------------------------------------------------------------------


#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        SIMD_INLINE bool Aligned(size_t size, size_t align = sizeof(__m128))
        {
            return Simd::Aligned(size, align);
        }

        SIMD_INLINE bool Aligned(const void * ptr, size_t align = sizeof(__m128))
        {
            return Simd::Aligned(ptr, align);
        }        
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        using Sse2::Aligned;
    }
#endif// SIMD_SSE41_ENABLE

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        SIMD_INLINE bool Aligned(size_t size, size_t align = sizeof(__m256))
        {
            return Simd::Aligned(size, align);
        }

        SIMD_INLINE bool Aligned(const void * ptr, size_t align = sizeof(__m256))
        {
            return Simd::Aligned(ptr, align);
        }
    }
#endif// SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        using Avx::Aligned;
    }
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE
    namespace Avx512f
    {
        SIMD_INLINE bool Aligned(size_t size, size_t align = sizeof(__m512))
        {
            return Simd::Aligned(size, align);
        }

        SIMD_INLINE bool Aligned(const void * ptr, size_t align = sizeof(__m512))
        {
            return Simd::Aligned(ptr, align);
        }
    }
#endif// SIMD_AVX512F_ENABLE

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        using Avx512f::Aligned;
    }
#endif// SIMD_AVX512BW_ENABLE

#ifdef SIMD_VMX_ENABLE
    namespace Vmx
    {
        SIMD_INLINE bool Aligned(size_t size, size_t align = sizeof(vec_uchar16))
        {
            return Simd::Aligned(size, align);
        }

        SIMD_INLINE bool Aligned(const void * ptr, size_t align = sizeof(vec_uchar16))
        {
            return Simd::Aligned(ptr, align);
        }
    }
#endif// SIMD_VMX_ENABLE

#ifdef SIMD_VSX_ENABLE
    namespace Vsx
    {
        using Vmx::Aligned;
    }
#endif// SIMD_VSX_ENABLE

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        SIMD_INLINE bool Aligned(size_t size, size_t align = sizeof(uint8x16_t))
        {
            return Simd::Aligned(size, align);
        }

        SIMD_INLINE bool Aligned(const void * ptr, size_t align = sizeof(uint8x16_t))
        {
            return Simd::Aligned(ptr, align);
        }
    }
#endif// SIMD_NEON_ENABLE
}

#endif//__SimdMemory_h__
