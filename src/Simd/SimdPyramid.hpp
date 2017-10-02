/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#ifndef __SimdPyramid_hpp__
#define __SimdPyramid_hpp__

#include "Simd/SimdView.hpp"

#include <vector>

namespace Simd
{
    /*! @ingroup cpp_pyramid

        \short The Pyramid structure provides storage and manipulation of pyramid images.

        The pyramid is a series of gray 8-bit images.
        Every image in the series is lesser in two times than previous.
        The structure is useful for image analysis.

        \ref cpp_pyramid_functions.
    */
    template <template<class> class A> struct Pyramid
    {
        typedef A<uint8_t> Allocator; /*!< Allocator type definition. */

        /*!
            Creates a new empty Pyramid structure.
        */
        Pyramid();

        /*!
            Creates a new Pyramid structure with specified size.

            \param [in] size - a size of pyramid's base (lowest and biggest image).
            \param [in] levelCount - a number of pyramid levels.
        */
        Pyramid(const Point<ptrdiff_t> & size, size_t levelCount);

        /*!
            Creates a new Pyramid structure with specified size.

            \param [in] width - a width of pyramid's base (lowest and biggest image).
            \param [in] height - a height of pyramid's base (lowest and biggest image).
            \param [in] levelCount - a number of pyramid levels.
        */
        Pyramid(size_t width, size_t height, size_t levelCount);

        /*!
            Re-create a Pyramid structure with specified size.

            \param [in] size - a size of pyramid's base (lowest and biggest image).
            \param [in] levelCount - a number of pyramid levels.
        */
        void Recreate(Point<ptrdiff_t> size, size_t levelCount);

        /*!
            Re-create a Pyramid structure with specified size.

            \param [in] width - a width of pyramid's base (lowest and biggest image).
            \param [in] height - a height of pyramid's base (lowest and biggest image).
            \param [in] levelCount - a number of pyramid levels.
            */
        void Recreate(size_t width, size_t height, size_t levelCount);

        /*!
            Gets number of levels in the pyramid.

            \return - number of levels in the pyramid.
        */
        size_t Size() const;

        /*!
            Gets an image at given level of the pyramid.

            \param [in] level - a level of the pyramid.
            \return - a reference to the image at given level of the pyramid.
        */
        View<A> & operator [] (size_t level);

        /*!
            Gets an image at given level of the pyramid.

            \param [in] level - a level of the pyramid.
            \return - a constant reference to the image at given level of the pyramid.
        */
        const View<A> & operator [] (size_t level) const;

        /*!
            Gets an image at given level of the pyramid.

            \param [in] level - a level of the pyramid.
            \return - a reference to the image at given level of the pyramid.
        */
        View<A> & At(size_t level);

        /*!
            Gets an image at on given level of the pyramid.

            \param [in] level - a level of the pyramid.
            \return - a constant reference to the image at given level of the pyramid.
        */
        const View<A> & At(size_t level) const;

        /*!
            Gets an image at top level of the pyramid.

            \return - a reference to the image at top level of the pyramid.
        */
        View<A> & Top();

        /*!
            Gets an image at top level of the pyramid.

            \return - a constant reference to the image at top level of the pyramid.
        */
        const View<A> & Top() const;

        /*!
            Swaps two pyramids.

            \param [in] pyramid - other pyramid.
        */
        void Swap(Pyramid<A> & pyramid);

    private:
        std::vector< View<A> > _views;
    };

    /*! @ingroup cpp_pyramid_functions

        \fn Point<ptrdiff_t> Scale(Point<ptrdiff_t> size, int scale = 2);

        \short Scales size of an image.

        \note This function is useful for Pyramid structure.

        \param [in] size - an original image size.
        \param [in] scale - a scale. It must be a multiple of 2. By default it is equal to 2.
        \return scaled size.
    */
    Point<ptrdiff_t> Scale(Point<ptrdiff_t> size, int scale = 2);

    //-------------------------------------------------------------------------

    // struct Pyramid implementation:

    template <template<class> class A>
    SIMD_INLINE Pyramid<A>::Pyramid()
    {
    }

    template <template<class> class A>
    SIMD_INLINE Pyramid<A>::Pyramid(const Point<ptrdiff_t> & size, size_t levelCount)
    {
        Recreate(size, levelCount);
    }

    template <template<class> class A>
    SIMD_INLINE Pyramid<A>::Pyramid(size_t width, size_t height, size_t levelCount)
    {
        Recreate(width, height, levelCount);
    }

    template <template<class> class A>
    SIMD_INLINE void Pyramid<A>::Recreate(Point<ptrdiff_t> size, size_t levelCount)
    {
        if (_views.size() && size == _views[0].Size())
            return;
        _views.resize(levelCount);
        for (size_t level = 0; level < levelCount; ++level)
        {
            _views[level].Recreate(size, View<A>::Gray8);
            size = Scale(size);
        }
    }

    template <template<class> class A>
    SIMD_INLINE void Pyramid<A>::Recreate(size_t width, size_t height, size_t levelCount)
    {
        Recreate(Point<ptrdiff_t>(width, height), levelCount);
    }

    template <template<class> class A>
    SIMD_INLINE size_t Pyramid<A>::Size() const
    {
        return _views.size();
    }

    template <template<class> class A>
    SIMD_INLINE View<A> & Pyramid<A>::operator [] (size_t level)
    {
        return _views[level];
    }

    template <template<class> class A>
    SIMD_INLINE const View<A> & Pyramid<A>::operator [] (size_t level) const
    {
        return _views[level];
    }

    template <template<class> class A>
    SIMD_INLINE View<A> & Pyramid<A>::At(size_t level)
    {
        return _views[level];
    }

    template <template<class> class A>
    SIMD_INLINE const View<A> & Pyramid<A>::At(size_t level) const
    {
        return _views[level];
    }

    template <template<class> class A>
    SIMD_INLINE View<A> & Pyramid<A>::Top()
    {
        return _views.back();
    }

    template <template<class> class A>
    SIMD_INLINE const View<A> & Pyramid<A>::Top() const
    {
        return _views.back();
    }

    template <template<class> class A>
    SIMD_INLINE void Pyramid<A>::Swap(Pyramid & pyramid)
    {
        _views.swap(pyramid._views);
    }

    // Pyramid utilities implementation:

    SIMD_INLINE Point<ptrdiff_t> Scale(Point<ptrdiff_t> size, int scale)
    {
        while (scale > 1)
        {
            assert(scale % 2 == 0);
            size.x = (size.x + 1) >> 1;
            size.y = (size.y + 1) >> 1;
            scale >>= 1;
        }
        return size;
    }
}

#endif//__SimdPyramid_hpp__
