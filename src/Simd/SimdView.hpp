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
#ifndef __SimdView_hpp__
#define __SimdView_hpp__

#include "Simd/SimdRectangle.hpp"
#include "Simd/SimdAllocator.hpp"

#include <memory.h>
#include <assert.h>

namespace Simd
{
    /*! @ingroup cpp_view

        \short The View structure provides storage and manipulation of images.

        \ref cpp_view_functions.
    */
    template <class A>
    struct View
    {
        typedef A Allocator; /*!< Allocator type definition. */

        /*!
            \enum Format
            Describes pixel format types of an image view.
            \note This type is corresponds to C type ::SimdPixelFormatType.
        */
        enum Format
        {
            /*! An undefined pixel format. */
            None = 0,
            /*! A 8-bit gray pixel format. */
            Gray8,
            /*! A 16-bit (2 8-bit channels) pixel format (UV plane of NV12 pixel format). */
            Uv16,
            /*! A 24-bit (3 8-bit channels) BGR (Blue, Green, Red) pixel format. */
            Bgr24,
            /*! A 32-bit (4 8-bit channels) BGRA (Blue, Green, Red, Alpha) pixel format. */
            Bgra32,
            /*! A single channel 16-bit integer pixel format. */
            Int16,
            /*! A single channel 32-bit integer pixel format. */
            Int32,
            /*! A single channel 64-bit integer pixel format. */
            Int64,
            /*! A single channel 32-bit float point pixel format. */
            Float,
            /*! A single channel 64-bit float point pixel format. */
            Double,
            /*! A 8-bit Bayer pixel format (GRBG). */
            BayerGrbg,
            /*! A 8-bit Bayer pixel format (GBRG). */
            BayerGbrg,
            /*! A 8-bit Bayer pixel format (RGGB). */
            BayerRggb,
            /*! A 8-bit Bayer pixel format (BGGR). */
            BayerBggr,
            /*! A 24-bit (3 8-bit channels) HSV (Hue, Saturation, Value) pixel format. */
            Hsv24,
            /*! A 24-bit (3 8-bit channels) HSL (Hue, Saturation, Lightness) pixel format. */
            Hsl24,
        };

        /*!
            \enum Position
            Describes the position of the child image view to the parent image view.
            This enum is used for creation of sub image view in method Simd::View::Region.
        */
        enum Position
        {
            TopLeft, /*!< A position in the top-left corner. */
            TopCenter, /*!< A position at the top center. */
            TopRight, /*!< A position in the top-right corner. */
            MiddleLeft, /*!< A position of the left in the middle. */
            MiddleCenter, /*!< A central position. */
            MiddleRight, /*!< A position of the right in the middle. */
            BottomLeft, /*!< A position in the bottom-left corner. */
            BottomCenter, /*!< A position at the bottom center. */
            BottomRight, /*!< A position in the bottom-right corner. */
        };

        const size_t width; /*!< \brief A width of the image. */
        const size_t height; /*!< \brief A height of the image. */
        const ptrdiff_t stride; /*!< \brief A row size of the image in bytes. */
        const Format format; /*!< \brief A pixel format types of the image. */
        uint8_t * const data; /*!< \brief A pointer to the pixel data (first row) of the image. */

        /*!
            Creates a new empty View structure. 
        */
        View();

        /*!
            Creates a new View structure on the base of the image view.

            \note This constructor is not create new image view! It only creates a reference to the same image. If you want to create a copy then must use method Simd::View::Clone.

            \param [in] view - an original image view. 
        */
        View(const View & view);

        /*!
            Creates a new View structure with specified width, height, row size, pixel format and pointer to pixel data.

            \param [in] w - a width of created image view. 
            \param [in] h - a height of created image view. 
            \param [in] s - a stride (row size) of created image view. 
            \param [in] f - a pixel format of created image view. 
            \param [in] d - a pointer to the external buffer with pixel data. If this pointer is NULL then will be created own buffer.
        */
        View(size_t w, size_t h, ptrdiff_t s, Format f, void * d);

        /*!
            Creates a new View structure with specified width, height, pixel format, pointer to pixel data and memory alignment.

            \param [in] w - a width of created image view. 
            \param [in] h - a height of created image view. 
            \param [in] f - a pixel format of created image view. 
            \param [in] d - a pointer to the external buffer with pixel data. If this pointer is NULL then will be created own buffer.
            \param [in] align - a required memory alignment. Its default value is determined by function Allocator::Alignment.
        */
        View(size_t w, size_t h, Format f, void * d = NULL, size_t align = Allocator::Alignment());

        /*!
            Creates a new View structure with specified width, height and pixel format.

            \param [in] size - a size (width and height) of created image view. 
            \param [in] f - a pixel format of created image view. 
        */
        View(const Point<ptrdiff_t> & size, Format f);

        /*!
            A View destructor.
        */
        ~View();

        /*!
            Gets a copy of current image view.

            \return a pointer to the new View structure. The user must free this pointer after usage.
        */
        View * Clone() const;

        /*!
            Creates reference to other View structure.

            \note This function is not create copy of image view! It only create a reference to the same image.

            \param [in] view - an original image view. 
            \return a reference to itself. 
        */
        View & operator = (const View & view);

        /*!
            Creates reference to itself. 

            \return a reference to itself. 
        */
        View & Ref();

        /*!
            Re-creates a View structure with specified width, height, pixel format, pointer to pixel data and memory alignment.

            \param [in] w - a width of re-created image view. 
            \param [in] h - a height of re-created image view. 
            \param [in] f - a pixel format of re-created image view. 
            \param [in] d - a pointer to the external buffer with pixel data. If this pointer is NULL then will be created own buffer.
            \param [in] align - a required memory alignment. Its default value is determined by function Allocator::Alignment.
        */
        void Recreate(size_t w, size_t h, Format f, void * d = NULL, size_t align = Allocator::Alignment());
        
        /*!
            Re-creates a View structure with specified width, height and pixel format.

            \param [in] size - a size (width and height) of re-created image view. 
            \param [in] f - a pixel format of re-created image view. 
        */
        void Recreate(const Point<ptrdiff_t> & size, Format f);

        /*!
            Creates a new View structure which points to the region of current image bounded by the rectangle with specified coordinates.

            \param [in] left - a left side of the region. 
            \param [in] top - a top side of the region. 
            \param [in] right - a right side of the region. 
            \param [in] bottom - a bottom side of the region.
            \return - a new View structure which points to the region of current image.
        */
        View Region(ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom) const;

        /*!
            Creates a new View structure which points to the region of current image bounded by the rectangle with specified coordinates.

            \param [in] topLeft - a top-left corner of the region. 
            \param [in] bottomRight - a bottom-right corner of the region.
            \return - a new View structure which points to the region of current image.
        */
        View Region(const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight) const;
        
        /*!
            Creates a new View structure which points to the region of current image bounded by the rectangle with specified coordinates.

            \param [in] rect - a rectangle which bound the region. 
            \return - a new View structure which points to the region of current image.
        */
        View Region(const Rectangle<ptrdiff_t> & rect) const;

        /*!
            Creates a new View structure which points to the region of current image bounded by the rectangle with specified coordinates.

            \param [in] size - a size (width and height) of the region. 
            \param [in] position - a value represents the position of the region (see Simd::View::Position). 
            \return - a new View structure which points to the region of current image.
        */
        View Region(const Point<ptrdiff_t> & size, Position position) const;

        /*!
            Creates a new View structure which points to the vertically flipped image.

            \return - a new View structure which points to the flipped image.
        */
        View Flipped() const;

        /*!
            Gets size (width and height) of the image.

            \return - a new Point structure with image width and height.
        */
        Point<ptrdiff_t> Size() const;

        /*!
            Gets size in bytes required to store pixel data of current View structure.

            \return - a size of data pixels in bytes.
        */
        size_t DataSize() const;

        /*!
            Gets area in pixels of of current View structure.

            \return - a area of current View in pixels.
        */
        size_t Area() const;

        /*!
            Gets constant reference to the pixel of arbitrary type into current view with specified coordinates.

            \param [in] x - a x-coordinate of the pixel. 
            \param [in] y - a y-coordinate of the pixel. 
            \return - a const reference to pixel of arbitrary type.
        */
        template <class T> const T & At(size_t x, size_t y) const;

        /*!
            Gets reference to the pixel of arbitrary type into current view with specified coordinates.

            \param [in] x - a x-coordinate of the pixel. 
            \param [in] y - a y-coordinate of the pixel. 
            \return - a reference to pixel of arbitrary type.
        */
        template <class T> T & At(size_t x, size_t y);

        /*!
            Gets constant reference to the pixel of arbitrary type into current view with specified coordinates.

            \param [in] p - a point with coordinates of the pixel. 
            \return - a const reference to pixel of arbitrary type.
        */
        template <class T> const T & At(const Point<ptrdiff_t> & p) const;

        /*!
            Gets reference to the pixel of arbitrary type into current view with specified coordinates.

            \param [in] p - a point with coordinates of the pixel. 
            \return - a reference to pixel of arbitrary type.
        */
        template <class T> T & At(const Point<ptrdiff_t> & p);

        /*!
            \fn size_t PixelSize(Format format);

            Gets pixel size in bytes for current pixel format.

            \param [in] format - a pixel format. 
            \return - a pixel size in bytes.
        */
        static size_t PixelSize(Format format);

        /*!
            Gets pixel size in bytes for current image.

            \return - a pixel size in bytes.
        */
        size_t PixelSize() const;

        /*!
            \fn size_t ChannelSize(Format format);

            Gets pixel channel size in bytes for current pixel format.

            \param [in] format - a pixel format. 
            \return - a pixel channel size in bytes.
        */
        static size_t ChannelSize(Format format);

        /*!
            Gets pixel channel size in bytes for current image.

            \return - a pixel channel size in bytes.
        */
        size_t ChannelSize() const;

        /*!
            \fn size_t ChannelCount(Format format);

            Gets number of channels in the pixel for current pixel format.

            \param [in] format - a pixel format. 
            \return - a number of channels.
        */
        static size_t ChannelCount(Format format);

        /*!
            Gets number of channels in the pixel for current image.

            \return - a number of channels.
        */
        size_t ChannelCount() const;

    private:
        bool _owner;
    };

    /*! @ingroup cpp_view_functions

        \fn template <class A, class B> bool EqualSize(const View<A> & a, const View<B> & b);

        Checks two image views on the same size.

        \param [in] a - a first image. 
        \param [in] b - a second image. 
        \return - a result of checking.
    */
    template <class A, class B> bool EqualSize(const View<A> & a, const View<B> & b);

    /*! @ingroup cpp_view_functions

        \fn template <class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c);

        Checks three image views on the same size.

        \param [in] a - a first image. 
        \param [in] b - a second image. 
        \param [in] c - a third image. 
        \return - a result of checking.
    */
    template <class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c);

    /*! @ingroup cpp_view_functions

        \fn template <class A, class B> bool Compatible(const View<A> & a, const View<B> & b);

        Checks two image views on compatibility (the images must have the same size and pixel format).

        \param [in] a - a first image. 
        \param [in] b - a second image. 
        \return - a result of checking.
    */
    template <class A, class B> bool Compatible(const View<A> & a, const View<B> & b);

    /*! @ingroup cpp_view_functions

        \fn template <class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c);

        Checks three image views on compatibility (the images must have the same size and pixel format).

        \param [in] a - a first image. 
        \param [in] b - a second image. 
        \param [in] c - a third image. 
        \return - a result of checking.
    */
    template <class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c);

    /*! @ingroup cpp_view_functions

        \fn template <class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);

        Checks four image views on compatibility (the images must have the same size and pixel format).

        \param [in] a - a first image. 
        \param [in] b - a second image. 
        \param [in] c - a third image. 
        \param [in] d - a fourth image. 
        \return - a result of checking.
    */
    template <class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);

    /*! @ingroup cpp_view_functions

        \fn template <class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d, const View<A> & e);

        Checks five image views on compatibility (the images must have the same size and pixel format).

        \param [in] a - a first image. 
        \param [in] b - a second image. 
        \param [in] c - a third image. 
        \param [in] d - a fourth image. 
        \param [in] e - a fifth image. 
        \return - a result of checking.
    */
    template <class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d, const View<A> & e);

    //-------------------------------------------------------------------------

    // struct View implementation:

    template <class A> SIMD_INLINE View<A>::View()
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
    }

    /*! \cond */
    template <class A> SIMD_INLINE View<A>::View(const View<A> & view)
        : width(view.width)
        , height(view.height)
        , stride(view.stride)
        , format(view.format)
        , data(view.data)
        , _owner(false)
    {
    }
    /*! \endcond */

    template <class A> SIMD_INLINE View<A>::View(size_t w, size_t h, ptrdiff_t s, Format f, void * d)
        : width(w)
        , height(h)
        , stride(s)
        , format(f)
        , data((uint8_t*)d)
        , _owner(false)
    {
        if(data == NULL && height && width && stride && format != None)
        {
            *(void**)&data = Allocator::Allocate(height*stride, Allocator::Alignment());
            _owner = true;
        }
    }

    template <class A> SIMD_INLINE View<A>::View(size_t w, size_t h, Format f, void * d, size_t align)
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
        Recreate(w, h, f, d, align);
    }

    template <class A> SIMD_INLINE View<A>::View(const Point<ptrdiff_t> & size, Format f)
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
        Recreate(size.x, size.y, f);
    }

    template <class A> SIMD_INLINE View<A>::~View()
    {
        if(_owner && data)
        {
            Allocator::Free(data);
        }
    }

    template <class A> SIMD_INLINE View<A> * View<A>::Clone() const
    {
        View<A> * view = new View<A>(width, height, format);
        size_t size = width*PixelSize();
        for(size_t row = 0; row < height; ++row)
            memcpy(view->data + view->stride*row, data + stride*row, size);
        return view;
    }

    template <class A> SIMD_INLINE View<A> & View<A>::operator = (const View<A> & view)
    {
        if(this != &view)
        {
            if(_owner && data)
            {
                Allocator::Free(data);
                assert(0);
            }
            *(size_t*)&width = view.width;
            *(size_t*)&height = view.height;
            *(Format*)&format = view.format;
            *(ptrdiff_t*)&stride = view.stride;
            *(unsigned char**)&data = view.data;
            _owner = false;
        }
        return *this;
    }

    template <class A> SIMD_INLINE View<A> & View<A>::Ref()
    {
        return *this;
    }

    template <class A> SIMD_INLINE void View<A>::Recreate(size_t w, size_t h, Format f, void * d, size_t align)
    {
        if(_owner && data)
        {
            Allocator::Free(data);
            *(void**)&data = NULL;
            _owner = false;
        }
        *(size_t*)&width = w;
        *(size_t*)&height = h;
        *(Format*)&format = f;
        *(ptrdiff_t*)&stride = Allocator::Align(width*PixelSize(format), align);
        if(d)
        {
            *(void**)&data = Allocator::Align(d, align);
            _owner = false;
        }
        else
        {
            *(void**)&data = Allocator::Allocate(height*stride, align);
            _owner = true;
        }
    }

    template <class A> SIMD_INLINE void View<A>::Recreate(const Point<ptrdiff_t> & size, Format f)
    {
        Recreate(size.x, size.y, f);
    }

    template <class A> SIMD_INLINE View<A> View<A>::Region(ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom) const
    {
        if(data != NULL && right >= left && bottom >= top)
        {
            left = std::min<ptrdiff_t>(std::max<ptrdiff_t>(left, 0), width);
            top = std::min<ptrdiff_t>(std::max<ptrdiff_t>(top, 0), height);
            right = std::min<ptrdiff_t>(std::max<ptrdiff_t>(right, 0), width);
            bottom = std::min<ptrdiff_t>(std::max<ptrdiff_t>(bottom, 0), height);
            return View<A>(right - left, bottom - top, stride, format, data + top*stride + left*PixelSize(format));
        }
        else
            return View<A>();
    }

    template <class A> SIMD_INLINE View<A> View<A>::Region(const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight) const
    {
        return Region(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    }

    template <class A> SIMD_INLINE View<A> View<A>::Region(const Rectangle<ptrdiff_t> & rect) const
    {
        return Region(rect.Left(), rect.Top(), rect.Right(), rect.Bottom());
    }

    template <class A> SIMD_INLINE View<A> View<A>::Region(const Point<ptrdiff_t> & size, Position position) const
    {
        switch(position)
        {
        case TopLeft:
            return Region(0, 0, size.x, size.y);
        case TopCenter:
            return Region((width - size.x)/2, 0, (width + size.x)/2, size.y);
        case TopRight:
            return Region(width - size.x, 0, width, size.y);
        case MiddleLeft:
            return Region(0, (height - size.y)/2, size.x, (height + size.y)/2);
        case MiddleCenter:
            return Region((width - size.x)/2, (height - size.y)/2, (width + size.x)/2, (height + size.y)/2);
        case MiddleRight:
            return Region(width - size.x, (height - size.y)/2, width, (height + size.y)/2);
        case BottomLeft:
            return Region(0, height - size.y, size.x, height);
        case BottomCenter:
            return Region((width - size.x)/2, height - size.y, (width + size.x)/2, height);
        case BottomRight:
            return Region(width - size.x, height - size.y, width, height);
        default:
            assert(0);
        }
        return View<A>();
    }

    template <class A> SIMD_INLINE View<A> View<A>::Flipped() const
    {
        return View<A>(width, height, -stride, format, data + (height - 1)*stride);
    }

    template <class A> SIMD_INLINE Point<ptrdiff_t> View<A>::Size() const
    {
        return Point<ptrdiff_t>(width, height);
    }

    template <class A> SIMD_INLINE size_t View<A>::DataSize() const
    {
        return stride*height;
    }

    template <class A> SIMD_INLINE size_t View<A>::Area() const
    {
        return width*height;
    }

    template <class A> template<class T> SIMD_INLINE const T & View<A>::At(size_t x, size_t y) const
    {
        assert(x < width && y < height);
        return ((const T*)(data + y*stride))[x];
    }

    template <class A> template<class T> SIMD_INLINE T & View<A>::At(size_t x, size_t y)
    {
        assert(x < width && y < height);
        return ((T*)(data + y*stride))[x];
    }

    template <class A> template<class T> SIMD_INLINE const T & View<A>::At(const Point<ptrdiff_t> & p) const
    {
        return At<T>(p.x, p.y);
    }

    template <class A> template<class T> SIMD_INLINE T & View<A>::At(const Point<ptrdiff_t> & p)
    {
        return At<T>(p.x, p.y);
    }

    template <class A> SIMD_INLINE size_t View<A>::PixelSize(Format format)
    {
        switch(format)
        {
        case None:      return 0;
        case Gray8:     return 1;
        case Uv16:      return 2;
        case Bgr24:     return 3;
        case Bgra32:    return 4;
        case Int16:     return 2;
        case Int32:     return 4;
        case Int64:     return 8;
        case Float:     return 4;
        case Double:    return 8;
        case BayerGrbg: return 1;
        case BayerGbrg: return 1;
        case BayerRggb: return 1;
        case BayerBggr: return 1;
        case Hsv24:     return 3;
        case Hsl24:     return 3;
        default: assert(0); return 0;
        }
    }

    template <class A> SIMD_INLINE size_t View<A>::PixelSize() const
    {
        return PixelSize(format);
    }

    template <class A> SIMD_INLINE size_t View<A>::ChannelSize(Format format)
    {
        switch(format)
        {
        case None:      return 0;
        case Gray8:     return 1;
        case Uv16:      return 1;
        case Bgr24:     return 1;
        case Bgra32:    return 1;
        case Int16:     return 2;
        case Int32:     return 4;
        case Int64:     return 8;
        case Float:     return 4;
        case Double:    return 8;
        case BayerGrbg: return 1;
        case BayerGbrg: return 1;
        case BayerRggb: return 1;
        case BayerBggr: return 1;
        case Hsv24:     return 1;
        case Hsl24:     return 1;
        default: assert(0); return 0;
        }
    }

    template <class A> SIMD_INLINE size_t View<A>::ChannelSize() const
    {
        return ChannelSize(format);
    }

    template <class A> SIMD_INLINE size_t View<A>::ChannelCount(Format format)
    {
        switch(format)
        {
        case None:      return 0;
        case Gray8:     return 1;
        case Uv16:      return 2;
        case Bgr24:     return 3;
        case Bgra32:    return 4;
        case Int16:     return 1;
        case Int32:     return 1;
        case Int64:     return 1;
        case Float:     return 1;
        case Double:    return 1;
        case BayerGrbg: return 1;
        case BayerGbrg: return 1;
        case BayerRggb: return 1;
        case BayerBggr: return 1;
        case Hsv24:     return 3;
        case Hsl24:     return 3;
        default: assert(0); return 0;
        }
    }

    template <class A> SIMD_INLINE size_t View<A>::ChannelCount() const
    {
        return ChannelCount(format);
    }

    // View utilities implementation:

    template <class A, class B> SIMD_INLINE bool EqualSize(const View<A> & a, const View<B> & b)
    {
        return
            (a.width == b.width && a.height == b.height);
    }

    template <class A> SIMD_INLINE bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c)
    {
        return
            (a.width == b.width && a.height == b.height) &&
            (a.width == c.width && a.height == c.height);
    }

    template <class A, class B> SIMD_INLINE bool Compatible(const View<A> & a, const View<B> & b)
    {
        typedef typename View<A>::Format Format;

        return
            (a.width == b.width && a.height == b.height && a.format == (Format)b.format);
    }

    template <class A> SIMD_INLINE bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c)
    {
        return
            (a.width == b.width && a.height == b.height && a.format == b.format) &&
            (a.width == c.width && a.height == c.height && a.format == c.format);
    }

    template <class A> SIMD_INLINE bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d)
    {
        return
            (a.width == b.width && a.height == b.height && a.format == b.format) &&
            (a.width == c.width && a.height == c.height && a.format == c.format) &&
            (a.width == d.width && a.height == d.height && a.format == d.format);
    }

    template <class A> SIMD_INLINE bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d, const View<A> & e)
    {
        return
            (a.width == b.width && a.height == b.height && a.format == b.format) &&
            (a.width == c.width && a.height == c.height && a.format == c.format) &&
            (a.width == d.width && a.height == d.height && a.format == d.format) &&
            (a.width == e.width && a.height == e.height && a.format == e.format);
    }
}

#endif//__SimdView_hpp__
