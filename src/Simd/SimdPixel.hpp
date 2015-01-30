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
#ifndef __SimdPixel_hpp__
#define __SimdPixel_hpp__

#include "Simd/SimdView.hpp"

namespace Simd
{
    namespace Pixel
    {
        struct Bgr24;
        struct Bgra32;
        struct Hsv24;
        struct Hsl24;

        //-------------------------------------------------------------------------

        struct Bgr24
        {
            uint8_t blue;
            uint8_t green;
            uint8_t red;

            Bgr24(const uint8_t & gray = uint8_t(0));
            Bgr24(const uint8_t & b, const uint8_t & g, const uint8_t & r);
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

            Bgra32(const uint8_t & gray = uint8_t(0), const uint8_t & a = uint8_t(255));
            Bgra32(const uint8_t & b, const uint8_t & g, const uint8_t & r, const uint8_t & a = uint8_t(255));
            Bgra32(const Bgr24 & p, const uint8_t & a = uint8_t(255));
            Bgra32(const Bgra32 & p);

            template <class A> static const Bgra32 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);
            template <class A> static Bgra32 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);
        };

        struct Hsv24
        {
            uint8_t hue;
            uint8_t saturation;
            uint8_t value;

            Hsv24(const uint8_t & gray = uint8_t(0));
            Hsv24(const uint8_t & h, const uint8_t & s, const uint8_t & v);
            Hsv24(const Hsv24 & p);

            template <class A> static const Hsv24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);
            template <class A> static Hsv24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);
        };

        struct Hsl24
        {
            uint8_t hue;
            uint8_t saturation;
            uint8_t lightness;

            Hsl24(const uint8_t & gray = uint8_t(0));
            Hsl24(const uint8_t & h, const uint8_t & s, const uint8_t & l);
            Hsl24(const Hsl24 & p);

            template <class A> static const Hsl24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);
            template <class A> static Hsl24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);
        };

        //-------------------------------------------------------------------------

        // struct Bgr24 implementation:

        SIMD_INLINE Bgr24::Bgr24(const uint8_t & gray)
            : blue(gray)
            , green(gray)
            , red(gray)
        {
        }

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

        SIMD_INLINE Bgra32::Bgra32(const uint8_t & gray, const uint8_t & a)
            : blue(gray)
            , green(gray)
            , red(gray)
            , alpha(a)
        {
        }

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

        // struct Hsv24 implementation:

        SIMD_INLINE Hsv24::Hsv24(const uint8_t & gray)
            : hue(0)
            , saturation(0)
            , value(gray)
        {
        }

        SIMD_INLINE Hsv24::Hsv24(const uint8_t & h, const uint8_t & s, const uint8_t & v)
            : hue(h)
            , saturation(s)
            , value(v)
        {
        }

        SIMD_INLINE Hsv24::Hsv24(const Hsv24 & p)
            : hue(p.hue)
            , saturation(p.saturation)
            , value(p.value)
        {
        }

        template <class A> SIMD_INLINE const Hsv24 & Hsv24::At(const View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Hsv24);

            return view.At<Hsv24>(col, row);
        }

        template <class A> SIMD_INLINE Hsv24 & Hsv24::At(View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Hsv24);

            return view.At<Hsv24>(col, row);
        }

        // struct Hsl24 implementation:

        SIMD_INLINE Hsl24::Hsl24(const uint8_t & gray)
            : hue(0)
            , saturation(0)
            , lightness(gray)
        {
        }

        SIMD_INLINE Hsl24::Hsl24(const uint8_t & h, const uint8_t & s, const uint8_t & l)
            : hue(h)
            , saturation(s)
            , lightness(l)
        {
        }

        SIMD_INLINE Hsl24::Hsl24(const Hsl24 & p)
            : hue(p.hue)
            , saturation(p.saturation)
            , lightness(p.lightness)
        {
        }

        template <class A> SIMD_INLINE const Hsl24 & Hsl24::At(const View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Hsl24);

            return view.At<Hsl24>(col, row);
        }

        template <class A> SIMD_INLINE Hsl24 & Hsl24::At(View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Hsl24);

            return view.At<Hsl24>(col, row);
        }
    }
}

#endif//__SimdPixel_hpp__
