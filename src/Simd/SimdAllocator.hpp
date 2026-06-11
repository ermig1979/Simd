/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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

        \short Aligned memory allocator for Simd C++ types and STL containers.

        The allocator is a stateless wrapper over Simd aligned memory functions. It is used
        by Simd C++ types such as View, Frame, Pyramid, Detection, ImageMatcher and
        ShiftDetector to allocate image data, temporary buffers and other owned storage with
        the alignment required by optimized SIMD code.

        It also implements the standard allocator interface, so containers such as
        <tt>std::vector<T, Simd::Allocator<T> ></tt> can keep their elements in aligned
        memory. All instances of this allocator are interchangeable because they do not own
        allocator state.
    */
    template <class T> struct Allocator
    {
        /*!
            \fn void * Allocate(size_t size, size_t align);

            \short Allocates an aligned memory block.

            Allocates a contiguous block of at least \a size bytes whose start address is a
            multiple of \a align. This function is used directly by Simd C++ classes when
            they allocate owned buffers. The STL allocator method allocate() calls it with
            <tt>size * sizeof(T)</tt> bytes and the default value returned by Alignment().

            \note The memory allocated by this function must be released by Free(). Do not
                  release it with \c free or \c delete.

            \param [in] size - the number of bytes to allocate.
            \param [in] align - the required alignment in bytes. It must be a power of two.
                                Use Alignment() to obtain the preferred alignment for the
                                current platform.
            \return a pointer to the allocated memory block, or \c NULL if allocation fails.
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

            \short Frees an aligned memory block.

            Releases a memory block returned by Allocate(). Simd C++ classes call this
            function when they destroy or recreate owned buffers, and the STL allocator
            method deallocate() delegates to it.

            \param [in] ptr - a pointer to the memory block to free. Passing \c NULL is safe.
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

            \short Rounds a size up to the requested alignment.

            Returns the smallest value that is both greater than or equal to \a size and a
            multiple of \a align. Simd::View uses this helper to compute aligned image
            strides from row byte sizes.

            \param [in] size - the original size in bytes or elements.
            \param [in] align - the required alignment in bytes. It must be a positive power
                                of two.

            \return the aligned size.
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

            \short Rounds a pointer up to the requested alignment.

            Returns the first address at or after \a ptr that is a multiple of \a align.
            Simd::View uses this helper when it is created over an external buffer and must
            align the stored data pointer to the requested boundary. The function does not
            allocate memory and does not change ownership of the buffer.

            \param [in] ptr - the original pointer.
            \param [in] align - the required alignment in bytes. It must be a positive power
                                of two.

            \return the aligned address.
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

            \short Returns the preferred memory alignment for optimized Simd code.

            The value reflects the active SIMD implementation on the current platform and is
            used as the default alignment for Simd C++ owned buffers and STL container
            allocations made through this allocator.

            \return the preferred memory alignment in bytes.
        */
        static SIMD_INLINE size_t Alignment()
        {
#if defined(__SimdAlignment_h__) && defined(_WIN32)
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
