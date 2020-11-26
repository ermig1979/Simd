/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#ifndef __SimdAllocator_hpp__
#define __SimdAllocator_hpp__

#include "Simd/SimdLib.h"

#include <memory>

namespace Simd
{
    /*! @ingroup cpp_allocator

        \short Aligned memory allocator.

        Performs allocation and deletion of aligned memory.

        \note Also it can be used as an allocator for STL containers.
    */
    template <class T> struct Allocator
    {
        /*!
            \fn void * Allocate(size_t size, size_t align);

            \short Allocates aligned memory block.

            \note The memory allocated by this function is must be deleted by function Simd::Allocator::Free.

            \param [in] size - a size of required memory block.
            \param [in] align - an align of allocated memory address.
            \return a pointer to allocated memory.
        */
        static SIMD_INLINE void * Allocate(size_t size, size_t align)
        {
#ifdef __SimdMemory_h__
            return Simd::Allocate(size, align);
#else
            return SimdAllocate(size, align);
#endif
        }

        /*!
            \fn void Free(void * ptr);

            \short Frees aligned memory block.

            \note This function frees a memory allocated by function Simd::Allocator::Allocate.

            \param [in] ptr - a pointer to the memory to be deleted.
        */
        static SIMD_INLINE void Free(void * ptr)
        {
#ifdef __SimdMemory_h__
            Simd::Free(ptr);
#else
            SimdFree(ptr);
#endif
        }

        /*!
            \fn size_t Align(size_t size, size_t align);

            \short Gets aligned size.

            \param [in] size - an original size.
            \param [in] align - a required alignment.

            \return an aligned size.
        */
        static SIMD_INLINE size_t Align(size_t size, size_t align)
        {
#ifdef __SimdMemory_h__
            return Simd::AlignHi(size, align);
#else
            return SimdAlign(size, align);
#endif
        }

        /*!
            \fn void * Align(void * ptr, size_t align);

            \short Gets aligned address.

            \param [in] ptr - an original pointer.
            \param [in] align - a required alignment.

            \return an aligned address.
        */
        static SIMD_INLINE void * Align(void * ptr, size_t align)
        {
#ifdef __SimdMemory_h__
            return Simd::AlignHi(ptr, align);
#else
            return (void *)SimdAlign((size_t)ptr, align);
#endif
        }

        /*!
            \fn size_t Alignment();

            \short Gets memory alignment required for the most productive work.

            \return a required memory alignment.
        */
        static SIMD_INLINE size_t Alignment()
        {
#if defined(__SimdAlignment_h__) && defined(WIN32)
            return Simd::Alignment();
#else
            return SimdAlignment();
#endif
        }

        //---------------------------------------------------------------------
        // STL allocator interface implementation:

        typedef T value_type;
        typedef T * pointer;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef T & reference;
        typedef const T & const_reference;
        typedef const T * const_pointer;

        template <typename U>
        struct rebind
        {
            typedef Allocator<U> other;
        };

        SIMD_INLINE Allocator()
        {
        }

        template <typename U> SIMD_INLINE Allocator(const Allocator<U> & a)
        {
        }

        SIMD_INLINE const_pointer address(const_reference value) const
        {
#if defined(SIMD_CPP_2011_ENABLE)
            return std::addressof(value);
#else
            return (reinterpret_cast<const_pointer>(&const_cast<char&>(reinterpret_cast<const volatile char&>(value))));
#endif
        }

        SIMD_INLINE pointer address(reference value) const
        {
#if defined(SIMD_CPP_2011_ENABLE)
            return std::addressof(value);
#else
            return (reinterpret_cast<pointer>(&const_cast<char&>(reinterpret_cast<const volatile char&>(value))));
#endif
        }

        SIMD_INLINE pointer allocate(size_type size, const void * ptr = NULL)
        {
            return static_cast<pointer>(Allocate(size * sizeof(T), Alignment()));
        }

        SIMD_INLINE size_type max_size() const
        {
            return ~static_cast<std::size_t>(0) / sizeof(T);
        }

        SIMD_INLINE void deallocate(pointer ptr, size_type size)
        {
            Free(ptr);
        }

        template<class U, class V> SIMD_INLINE void construct(U * ptr, const V & value)
        {
            ::new((void*)ptr) U(value);
        }

#if defined(SIMD_CPP_2011_ENABLE)
        template<class U, class... Args> SIMD_INLINE void construct(U * ptr, Args &&... args)
        {
            ::new((void*)ptr) U(std::forward<Args>(args)...);
        }
#endif

        template<class U> SIMD_INLINE void construct(U * ptr)
        {
            ::new((void*)ptr) U();
        }

        template<class U> SIMD_INLINE void destroy(U * ptr)
        {
            ptr->~U();
        }
    };

    template<typename T1, typename T2> SIMD_INLINE bool operator == (const Allocator<T1> & a1, const Allocator<T2> & a2)
    {
        return true;
    }

    template<typename T1, typename T2> SIMD_INLINE bool operator != (const Allocator<T1> & a1, const Allocator<T2> & a2)
    {
        return false;
    }
}

#endif//__SimdAllocator_hpp__
