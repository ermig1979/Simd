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

        static const size_t s_pixelSizes[];
        static const size_t s_channelSizes[];
        static const size_t s_channelCounts[];
    };

    bool EqualSize(const View & a, const View & b);
    bool EqualSize(const View & a, const View & b, const View & c);

    bool Compatible(const View & a, const View & b);
    bool Compatible(const View & a, const View & b, const View & c);
    bool Compatible(const View & a, const View & b, const View & c, const View & d);
    bool Compatible(const View & a, const View & b, const View & c, const View & d, const View & e);

    //-------------------------------------------------------------------------

    // struct View implementation:

    SIMD_INLINE void View::Recreate(Point<ptrdiff_t> size, Format f)
    {
        Recreate(size.x, size.y, f);
    }

    SIMD_INLINE View View::Region(const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight) const
    {
        return Region(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    }

    SIMD_INLINE View View::Region(const Rectangle<ptrdiff_t> & rect) const
    {
        return Region(rect.Left(), rect.Top(), rect.Right(), rect.Bottom());
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

    SIMD_INLINE size_t View::PixelSize() const
    {
        return PixelSize(format);
    }

    SIMD_INLINE size_t View::ChannelSize() const
    {
        return ChannelSize(format);
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