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
#ifndef __SimdAllocator_hpp__
#define __SimdAllocator_hpp__

#include "Simd/SimdLib.h"

namespace Simd
{
    /*! @ingroup cpp_allocator

        \short Aligned memory allocator.

        Performs allocation and deletion of aligned memory.
    */
    struct Allocator
    {
        /*!
            \fn void * Allocate(size_t size, size_t align);

            \short Allocates aligned memory block.

            \note The memory allocated by this function is must be deleted by function Simd::Allocator::Free.

            \param [in] size - a size of required memory block.
            \param [in] align - an align of allocated memory address.
            \return a pointer to allocated memory.
        */
        static void * Allocate(size_t size, size_t align);

        /*!
            \fn void Free(void * ptr);

            \short Frees aligned memory block.

            \note This function frees a memory allocated by function Simd::Allocator::Allocate.

            \param [in] ptr - a pointer to the memory to be deleted.
        */
        static void Free(void * ptr);

        /*!
            \fn size_t Align(size_t size, size_t align);

            \short Gets aligned size.

            \param [in] size - an original size.
            \param [in] align - a required alignment.

            \return an aligned size.
        */
        static size_t Align(size_t size, size_t align);

        /*!
            \fn void * Align(void * ptr, size_t align);

            \short Gets aligned address.

            \param [in] ptr - an original pointer.
            \param [in] align - a required alignment.

            \return an aligned address.
        */
        static void * Align(void * ptr, size_t align);

        /*!
            \fn size_t Alignment();

            \short Gets memory alignment required for the most productive work.

            \return a required memory alignment.
        */
        static size_t Alignment();
    };

    //-------------------------------------------------------------------------

    // struct Allocator implementation:

    SIMD_INLINE void * Allocator::Allocate(size_t size, size_t align)
    {
        return SimdAllocate(size, align);
    }

    SIMD_INLINE void Allocator::Free(void * ptr)
    {
        SimdFree(ptr);
    }

    SIMD_INLINE size_t Allocator::Align(size_t size, size_t align)
    {
        return SimdAlign(size, align);
    }

    SIMD_INLINE void * Allocator::Align(void * ptr, size_t align)
    {
        return (void *) SimdAlign((size_t)ptr, align);
    }

    SIMD_INLINE size_t Allocator::Alignment()
    {
        return SimdAlignment();
    }
}

#endif//__SimdAllocator_hpp__
