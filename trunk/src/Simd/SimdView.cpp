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
    size_t View::SizeOf(Format format)
    {
        switch(format)
        {
        case None:
            return 0;
        case Gray8:
            return 1;
		case Uv16:
			return 2;
        case Bgr24:
            return 3;
        case Bgra32:
        case Int32:
            return 4;
        case Int64:
            return 8;
        default:
            return 0;
        }
    }

    View::View()
    {
        width = 0;
        height = 0;
        stride = 0;
        format = None;
        data = NULL;
        _owner = false;
    }

    View::View(size_t w, size_t h, ptrdiff_t s, Format f, void* d)
    {
        width = w;
        height = h;
        stride = s;
        format = f;
        if(d)
        {
            data = (uchar*)d;
            _owner = false;
        }
        else
        {
            data = (uchar*)Allocate(height*stride);
            _owner = true;
        }
    }

    View::View(size_t w, size_t h, Format f, void * d, size_t align)
    {
        width = w;
        height = h;
        format = f;
        stride = AlignHi(width*SizeOf(format), align);
        if(d)
        {
            data = (uchar*)AlignHi(d, align);
            _owner = false;
        }
        else
        {
            data = (uchar*)Allocate(height*stride, align);
            _owner = true;
        }
    }

    View::~View()
    {
        if(_owner && data)
        {
            Free(data);
        }
    }

    View View::Region(ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom) const
    {
        if(data != NULL && right >= left && bottom >= top)
        {
			left = RestrictRange(left, (ptrdiff_t)0, (ptrdiff_t)width);
			top = RestrictRange(top, (ptrdiff_t)0, (ptrdiff_t)height);
			right = RestrictRange(right, (ptrdiff_t)0, (ptrdiff_t)width);
			bottom = RestrictRange(bottom, (ptrdiff_t)0, (ptrdiff_t)height);
            return View(right - left, bottom - top, stride, format, data + top*stride + left*SizeOf(format));
        }
        else
            return View();
    }
}