/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#ifndef __SimdPixel_h__
#define __SimdPixel_h__

#include "Simd/SimdView.h"

namespace Simd
{
    namespace Pixel
    {
        struct Bgr24;
        struct Bgra32;

        //-------------------------------------------------------------------------

        struct Bgr24
        {
            uint8_t blue;
            uint8_t green;
            uint8_t red;

            Bgr24(const uint8_t & b = uint8_t(0), const uint8_t & g = uint8_t(0), const uint8_t & r = uint8_t(0));
            Bgr24(const Bgra32 & p);
            Bgr24(const Bgr24 & p);

            template <class A> static const Bgr24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);
            template <class A> static Bgr24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);
        };

        struct Bgra32
        {
            uint8_t blue;
            uint8_t green;
            uint8_t red;
            uint8_t alpha;

            Bgra32(const uint8_t & b = uint8_t(0), const uint8_t & g = uint8_t(0), const uint8_t & r = uint8_t(0), const uint8_t & a = uint8_t(255));
            Bgra32(const Bgr24 & p, const uint8_t & a = uint8_t(255));
            Bgra32(const Bgra32 & p);

            template <class A> static const Bgra32 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);
            template <class A> static Bgra32 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);
        };

        //-------------------------------------------------------------------------

        // struct Bgr24 implementation:

        SIMD_INLINE Bgr24::Bgr24(const uint8_t & b, const uint8_t & g, const uint8_t & r)
            : blue(b)
            , green(g)
            , red(r)
        {
        }

        SIMD_INLINE Bgr24::Bgr24(const Bgra32 & p)
            : blue(p.blue)
            , green(p.green)
            , red(p.red)
        {
        }

        SIMD_INLINE Bgr24::Bgr24(const Bgr24 & p)
            : blue(p.blue)
            , green(p.green)
            , red(p.red)
        {
        }

        template <class A> SIMD_INLINE const Bgr24 & Bgr24::At(const View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Bgr24);

            return view.At<Bgr24>(col, row);
        }

        template <class A> SIMD_INLINE Bgr24 & Bgr24::At(View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Bgr24);

            return view.At<Bgr24>(col, row);
        }

        // struct Bgra32 implementation:

        SIMD_INLINE Bgra32::Bgra32(const uint8_t & b, const uint8_t & g, const uint8_t & r, const uint8_t & a)
            : blue(b)
            , green(g)
            , red(r)
            , alpha(a)
        {
        }

        SIMD_INLINE Bgra32::Bgra32(const Bgr24 & p, const uint8_t & a)
            : blue(p.blue)
            , green(p.green)
            , red(p.red)
            , alpha(a)
        {
        }

        SIMD_INLINE Bgra32::Bgra32(const Bgra32 & p)
            : blue(p.blue)
            , green(p.green)
            , red(p.red)
            , alpha(p.alpha)
        {
        }

        template <class A> SIMD_INLINE const Bgra32 & Bgra32::At(const View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Bgra32);

            return view.At<Bgra32>(col, row);
        }

        template <class A> SIMD_INLINE Bgra32 & Bgra32::At(View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Bgra32);

            return view.At<Bgra32>(col, row);
        }
    }
}

#endif//__SimdPixel_h__
