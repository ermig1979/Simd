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

#include "Simd/SimdTypes.h"
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
            Int64
        };
        static size_t SizeOf(Format format);

        size_t width;
        size_t height;
        ptrdiff_t stride;
        Format format;
        uchar * data;

        View(); 
        View(size_t w, size_t h, ptrdiff_t s, Format f, void* d); 
        View(size_t w, size_t h, Format f, void* d, size_t align = DEFAULT_MEMORY_ALIGN);

        ~View(); 

        View Region(ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom) const;

    private:
        bool _owner;
    };
}

#endif//__SimdView_h__