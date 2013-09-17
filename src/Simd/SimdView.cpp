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
#include "Simd/SimdMath.h"
#include "Simd/SimdView.h"

namespace Simd
{
    View::View()
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
    }

    View::View(const View & view)
        : width(view.width)
        , height(view.height)
        , stride(view.stride)
        , format(view.format)
        , data(view.data)
        , _owner(false)
    {
    }

    View::View(size_t w, size_t h, ptrdiff_t s, Format f, void * d)
        : width(w)
        , height(h)
        , stride(s)
        , format(f)
        , data(d ? (unsigned char*)d : (unsigned char*)Simd::Allocate(height*stride))
        , _owner(data == NULL)
    {
    }

    View::View(size_t w, size_t h, Format f, void * d, size_t align)
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
        Recreate(w, h, f, d, align);
    }

    View::View(const Point<ptrdiff_t> size, Format f)
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
        Recreate(size.x, size.y, f);
    }

    View::~View()
    {
        if(_owner && data)
        {
            Simd::Free(data);
        }
    }

    View * View::Clone() const
    {
        View * view = new View(width, height, format);
        size_t size = width*PixelSize();
        for(size_t row = 0; row < height; ++row)
            memcpy(view->data + view->stride*row, data + stride*row, size);
        return view;
    }

    View & View::operator = (const View & view)
    {
        if(this !=  &view)
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

    void View::Recreate(size_t w, size_t h, Format f, void * d, size_t align)
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

    View View::Region(ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom) const
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

    View View::Region(const Point<ptrdiff_t> & size, Position position) const
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

    size_t View::PixelSize(Format format)
    {
        assert(format >= None && format <= Double);
        return s_pixelSizes[format];
    }

    size_t View::ChannelSize(Format format)
    {
        assert(format >= None && format <= Double);
        return s_channelSizes[format];
    }

    size_t View::ChannelCount(Format format)
    {
        assert(format >= None && format <= Double);
        return s_channelCounts[format];
    }

    //                                     {None, Gray8, Uv16, Bgr24, Bgra32, Int32, Float, Int64, Double}
    const size_t View::s_pixelSizes[]    = {   0,     1,    2,     3,      4,     4,     4,     8,      8};
    const size_t View::s_channelSizes[]  = {   0,     1,    1,     1,      1,     4,     4,     8,      8};
    const size_t View::s_channelCounts[] = {   0,     1,    2,     3,      4,     1,     1,     1,      1};
}