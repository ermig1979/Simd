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
    template <class A>
    struct View
    {
        typedef A Allocator;

        enum Format
        {
            None = 0,
            Gray8,
            Uv16,
            Bgr24,
            Bgra32,
            Int16,
            Int32,
            Int64,
            Float,
            Double,
            BayerGrbg,
            BayerGbrg,
            BayerRggb,
            BayerBggr,
            Hsv24,
            Hsl24,
        };

        enum Position
        {
            TopLeft,
            TopCenter,
            TopRight,
            MiddleLeft,
            MiddleCenter,
            MiddleRight,
            BottomLeft,
            BottomCenter,
            BottomRight,
        };

        const size_t width;
        const size_t height;
        const ptrdiff_t stride;
        const Format format;
        unsigned char * const data;

        View();
        View(const View & view);
        View(size_t w, size_t h, ptrdiff_t s, Format f, void * d);
        View(size_t w, size_t h, Format f, void * d = NULL, size_t align = Allocator::Alignment());
        View(const Point<ptrdiff_t> & size, Format f);

        ~View();

        View * Clone() const;

        View & operator = (const View & view);

        View & Ref();

        void Recreate(size_t w, size_t h, Format f, void * d = NULL, size_t align = Allocator::Alignment());
        void Recreate(const Point<ptrdiff_t> & size, Format f);

        View Region(ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom) const;
        View Region(const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight) const;
        View Region(const Rectangle<ptrdiff_t> & rect) const;
        View Region(const Point<ptrdiff_t> & size, Position position) const;

        View Flipped() const;

        Point<ptrdiff_t> Size() const;
        size_t DataSize() const;
        size_t Area() const;

        template <class T> const T & At(size_t x, size_t y) const;
        template <class T> T & At(size_t x, size_t y);

        template <class T> const T & At(const Point<ptrdiff_t> & p) const;
        template <class T> T & At(const Point<ptrdiff_t> & p);

        static size_t PixelSize(Format format);
        size_t PixelSize() const;

        static size_t ChannelSize(Format format);
        size_t ChannelSize() const;

        static size_t ChannelCount(Format format);
        size_t ChannelCount() const;

    private:
        bool _owner;
    };

    template <class A, class B> bool EqualSize(const View<A> & a, const View<B> & b);
    template <class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c);

    template <class A, class B> bool Compatible(const View<A> & a, const View<B> & b);
    template <class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c);
    template <class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);
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

    template <class A> SIMD_INLINE View<A>::View(const View<A> & view)
        : width(view.width)
        , height(view.height)
        , stride(view.stride)
        , format(view.format)
        , data(view.data)
        , _owner(false)
    {
    }

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
