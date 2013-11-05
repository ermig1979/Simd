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
#ifndef __SimdView_h__
#define __SimdView_h__

#include "Simd/SimdRectangle.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
    struct View
    {
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
        View(size_t w, size_t h, Format f, void * d = NULL, size_t align = Simd::DEFAULT_MEMORY_ALIGN);
        View(const Point<ptrdiff_t> size, Format f);

        ~View();

        View * Clone() const;

        View & operator = (const View & view);

        View & Ref();

        void Recreate(size_t w, size_t h, Format f, void * d = NULL, size_t align = Simd::DEFAULT_MEMORY_ALIGN);
        void Recreate(Point<ptrdiff_t> size, Format f);

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

    bool EqualSize(const View & a, const View & b);
    bool EqualSize(const View & a, const View & b, const View & c);

    bool Compatible(const View & a, const View & b);
    bool Compatible(const View & a, const View & b, const View & c);
    bool Compatible(const View & a, const View & b, const View & c, const View & d);
    bool Compatible(const View & a, const View & b, const View & c, const View & d, const View & e);

    //-------------------------------------------------------------------------

    // struct View implementation:

    SIMD_INLINE View::View()
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
    }

    SIMD_INLINE View::View(const View & view)
        : width(view.width)
        , height(view.height)
        , stride(view.stride)
        , format(view.format)
        , data(view.data)
        , _owner(false)
    {
    }

    SIMD_INLINE View::View(size_t w, size_t h, ptrdiff_t s, Format f, void * d)
        : width(w)
        , height(h)
        , stride(s)
        , format(f)
        , data(d ? (unsigned char*)d : (unsigned char*)Simd::Allocate(height*stride))
        , _owner(data == NULL)
    {
    }

    SIMD_INLINE View::View(size_t w, size_t h, Format f, void * d, size_t align)
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
        Recreate(w, h, f, d, align);
    }

    SIMD_INLINE View::View(const Point<ptrdiff_t> size, Format f)
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
        Recreate(size.x, size.y, f);
    }

    SIMD_INLINE View::~View()
    {
        if(_owner && data)
        {
            Simd::Free(data);
        }
    }

    SIMD_INLINE View * View::Clone() const
    {
        View * view = new View(width, height, format);
        size_t size = width*PixelSize();
        for(size_t row = 0; row < height; ++row)
            memcpy(view->data + view->stride*row, data + stride*row, size);
        return view;
    }

    SIMD_INLINE View & View::operator = (const View & view)
    {
        if(this != &view)
        {
            if(_owner && data)
            {
                Simd::Free(data);
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

    SIMD_INLINE View & View::Ref()
    {
        return *this;
    }

    SIMD_INLINE void View::Recreate(size_t w, size_t h, Format f, void * d, size_t align)
    {
        if(_owner && data)
        {
            Simd::Free(data);
            *(void**)&data = NULL;
            _owner = false;
        }
        *(size_t*)&width = w;
        *(size_t*)&height = h;
        *(Format*)&format = f;
        *(ptrdiff_t*)&stride = Simd::AlignHi(width*PixelSize(format), align);
        if(d)
        {
            *(void**)&data = Simd::AlignHi(d, align);
            _owner = false;
        }
        else
        {
            *(void**)&data = Simd::Allocate(height*stride, align);
            _owner = true;
        }
    }

    SIMD_INLINE void View::Recreate(Point<ptrdiff_t> size, Format f)
    {
        Recreate(size.x, size.y, f);
    }

    SIMD_INLINE View View::Region(ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom) const
    {
        if(data != NULL && right >= left && bottom >= top)
        {
            left = RestrictRange<ptrdiff_t>(left, 0, width);
            top = RestrictRange<ptrdiff_t>(top, 0, height);
            right = RestrictRange<ptrdiff_t>(right, 0, width);
            bottom = RestrictRange<ptrdiff_t>(bottom, 0, height);
            return View(right - left, bottom - top, stride, format, data + top*stride + left*PixelSize(format));
        }
        else
            return View();
    }

    SIMD_INLINE View View::Region(const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight) const
    {
        return Region(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    }

    SIMD_INLINE View View::Region(const Rectangle<ptrdiff_t> & rect) const
    {
        return Region(rect.Left(), rect.Top(), rect.Right(), rect.Bottom());
    }

    SIMD_INLINE View View::Region(const Point<ptrdiff_t> & size, Position position) const
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
        return View();
    }

    SIMD_INLINE View View::Flipped() const
    {
        return View(width, height, -stride, format, data + (height - 1)*stride);
    }

    SIMD_INLINE Point<ptrdiff_t> View::Size() const
    {
        return Point<ptrdiff_t>(width, height);
    }

    SIMD_INLINE size_t View::DataSize() const
    {
        return stride*height;
    }

    SIMD_INLINE size_t View::Area() const
    {
        return width*height;
    }

    template <class T>
    SIMD_INLINE const T & View::At(size_t x, size_t y) const
    {
        return ((const T*)(data + y*stride))[x];
    }

    template <class T>
    SIMD_INLINE T & View::At(size_t x, size_t y)
    {
        return ((T*)(data + y*stride))[x];
    }

    template <class T>
    SIMD_INLINE const T & View::At(const Point<ptrdiff_t> & p) const
    {
        return At<T>(p.x, p.y);
    }

    template <class T>
    SIMD_INLINE T & View::At(const Point<ptrdiff_t> & p)
    {
        return At<T>(p.x, p.y);
    }

    SIMD_INLINE size_t View::PixelSize(Format format)
    {
        switch(format)
        {
        case None:   return 0;
        case Gray8:  return 1;
        case Uv16:   return 2;
        case Bgr24:  return 3;
        case Bgra32: return 4;
        case Int16:  return 2;
        case Int32:  return 4;
        case Int64:  return 8;
        case Float:  return 4;
        case Double: return 8;
        default: assert(0); return 0;
        }
    }

    SIMD_INLINE size_t View::PixelSize() const
    {
        return PixelSize(format);
    }

    SIMD_INLINE size_t View::ChannelSize(Format format)
    {
        switch(format)
        {
        case None:   return 0;
        case Gray8:  return 1;
        case Uv16:   return 1;
        case Bgr24:  return 1;
        case Bgra32: return 1;
        case Int16:  return 2;
        case Int32:  return 4;
        case Int64:  return 8;
        case Float:  return 4;
        case Double: return 8;
        default: assert(0); return 0;
        }
    }

    SIMD_INLINE size_t View::ChannelSize() const
    {
        return ChannelSize(format);
    }

    SIMD_INLINE size_t View::ChannelCount(Format format)
    {
        switch(format)
        {
        case None:   return 0;
        case Gray8:  return 1;
        case Uv16:   return 2;
        case Bgr24:  return 3;
        case Bgra32: return 4;
        case Int16:  return 1;
        case Int32:  return 1;
        case Int64:  return 1;
        case Float:  return 1;
        case Double: return 1;
        default: assert(0); return 0;
        }
    }

    SIMD_INLINE size_t View::ChannelCount() const
    {
        return ChannelCount(format);
    }

    // View utilities implementation:

    SIMD_INLINE bool EqualSize(const View & a, const View & b)
    {
        return
            (a.width == b.width && a.height == b.height);
    }

    SIMD_INLINE bool EqualSize(const View & a, const View & b, const View & c)
    {
        return
            (a.width == b.width && a.height == b.height) &&
            (a.width == c.width && a.height == c.height);
    }

    SIMD_INLINE bool Compatible(const View & a, const View & b)
    {
        return
            (a.width == b.width && a.height == b.height && a.format == b.format);
    }

    SIMD_INLINE bool Compatible(const View & a, const View & b, const View & c)
    {
        return
            (a.width == b.width && a.height == b.height && a.format == b.format) &&
            (a.width == c.width && a.height == c.height && a.format == c.format);
    }

    SIMD_INLINE bool Compatible(const View & a, const View & b, const View & c, const View & d)
    {
        return
            (a.width == b.width && a.height == b.height && a.format == b.format) &&
            (a.width == c.width && a.height == c.height && a.format == c.format) &&
            (a.width == d.width && a.height == d.height && a.format == d.format);
    }

    SIMD_INLINE bool Compatible(const View & a, const View & b, const View & c, const View & d, const View & e)
    {
        return
            (a.width == b.width && a.height == b.height && a.format == b.format) &&
            (a.width == c.width && a.height == c.height && a.format == c.format) &&
            (a.width == d.width && a.height == d.height && a.format == d.format) &&
            (a.width == e.width && a.height == e.height && a.format == e.format);
    }
}

#endif//__SimdView_h__
