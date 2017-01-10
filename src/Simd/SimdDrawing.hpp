/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
#ifndef __SimdDrawing_hpp__
#define __SimdDrawing_hpp__

#include "Simd/SimdView.hpp"

namespace Simd
{
    /*! @ingroup cpp_drawing

        \fn void DrawLine(View<A> & canvas, ptrdiff_t x1, ptrdiff_t y1, ptrdiff_t x2, ptrdiff_t y2, const Color & color)

        \short Draws a line at the image.

        \param [out] canvas - a canvas (image where we draw line).
        \param [in] x1 - X coordinate of the first point of the line.
        \param [in] y1 - Y coordinate of the first point of the line.
        \param [in] x2 - X coordinate of the second point of the line.
        \param [in] y2 - Y coordinate of the second point of the line.
        \param [in] color - a color of the line.
    */
    template<class A, class Color> SIMD_INLINE void DrawLine(View<A> & canvas, ptrdiff_t x1, ptrdiff_t y1, ptrdiff_t x2, ptrdiff_t y2, const Color & color)
    {
        assert(canvas.PixelSize() == sizeof(Color));

        const bool inverse = std::abs(y2 - y1) > std::abs(x2 - x1);
        if (inverse)
        {
            std::swap(x1, y1);
            std::swap(x2, y2);
        }

        if (x1 > x2)
        {
            std::swap(x1, x2);
            std::swap(y1, y2);
        }

        const double dx = double(x2 - x1);
        const double dy = (double)std::abs(y2 - y1);

        double error = dx / 2.0f;
        const ptrdiff_t ystep = (y1 < y2) ? 1 : -1;
        ptrdiff_t y = y1;

        const ptrdiff_t maxX = x2;

        for (ptrdiff_t x = x1; x < x2; x++)
        {
            if (inverse)
            {
                canvas.At<Color>(y, x) = color;
            }
            else
            {
                canvas.At<Color>(x, y) = color;
            }

            error -= dy;
            if (error < 0)
            {
                y += ystep;
                error += dx;
            }
        }
    }

    /*! @ingroup cpp_drawing

        \fn void DrawLine(View<A> & canvas, const Point<ptrdiff_t> & p1, const Point<ptrdiff_t> & p2, const Color & color)

        \short Draws a line at the image.

        \param [out] canvas - a canvas (image where we draw line).
        \param [in] p1 - the first point of the line.
        \param [in] p2 - the second point of the line.
        \param [in] color - a color of the line.
    */
    template<class A, class Color> SIMD_INLINE void DrawLine(View<A> & canvas, const Point<ptrdiff_t> & p1, const Point<ptrdiff_t> & p2, const Color & color)
    {
        DrawLine(canvas, p1.x, p1.y, p2.x, p2.y, color);
    }
}

#endif//__SimdDrawing_hpp__
