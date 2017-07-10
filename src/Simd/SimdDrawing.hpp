/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

#include <vector>

namespace Simd
{
    /*! @ingroup cpp_drawing

        \fn void DrawLine(View<A> & canvas, ptrdiff_t x1, ptrdiff_t y1, ptrdiff_t x2, ptrdiff_t y2, const Color & color, size_t width = 1)

        \short Draws a line at the image.

        \param [out] canvas - a canvas (image where we draw line).
        \param [in] x1 - X coordinate of the first point of the line.
        \param [in] y1 - Y coordinate of the first point of the line.
        \param [in] x2 - X coordinate of the second point of the line.
        \param [in] y2 - Y coordinate of the second point of the line.
        \param [in] color - a color of the line.
        \param [in] width - a width of the line. By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawLine(View<A> & canvas, ptrdiff_t x1, ptrdiff_t y1, ptrdiff_t x2, ptrdiff_t y2, const Color & color, size_t width = 1)
    {
        assert(canvas.PixelSize() == sizeof(Color));

        const ptrdiff_t w = canvas.width - 1;
        const ptrdiff_t h = canvas.height - 1;

        if (x1 < 0 || y1 < 0 || x1 > w || y1 > h || x2 < 0 || y2 < 0 || x2 > w || y2 > h)
        {
            if ((x1 < 0 && x2 < 0) || (y1 < 0 && y2 < 0) || (x1 > w && x2 > w) || (y1 > h && y2 > h))
                return;

            if (y1 == y2)
            {
                x1 = std::min<ptrdiff_t>(std::max<ptrdiff_t>(x1, 0), w);
                x2 = std::min<ptrdiff_t>(std::max<ptrdiff_t>(x2, 0), w);
            }
            else if (x1 == x2)
            {
                y1 = std::min<ptrdiff_t>(std::max<ptrdiff_t>(y1, 0), h);
                y2 = std::min<ptrdiff_t>(std::max<ptrdiff_t>(y2, 0), h);
            }
            else
            {
                ptrdiff_t x0 = (x1*y2 - y1*x2) / (y2 - y1);
                ptrdiff_t y0 = (y1*x2 - x1*y2) / (x2 - x1);
                ptrdiff_t xh = (x1*y2 - y1*x2 + h*(x2 - x1)) / (y2 - y1);
                ptrdiff_t yw = (y1*x2 - x1*y2 + w*(y2 - y1)) / (x2 - x1);

                if (x1 < 0)
                {
                    x1 = 0;
                    y1 = y0;
                }
                if (x2 < 0)
                {
                    x2 = 0;
                    y2 = y0;
                }
                if (x1 > w)
                {
                    x1 = w;
                    y1 = yw;
                }
                if (x2 > w)
                {
                    x2 = w;
                    y2 = yw;
                }
                if ((y1 < 0 && y2 < 0) || (y1 > h && y2 > h))
                    return;

                if (y1 < 0)
                {
                    x1 = x0;
                    y1 = 0;
                }
                if (y2 < 0)
                {
                    x2 = x0;
                    y2 = 0;
                }

                if (y1 > h)
                {
                    x1 = xh;
                    y1 = h;
                }
                if (y2 > h)
                {
                    x2 = xh;
                    y2 = h;
                }
            }
        }

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
        ptrdiff_t y0 = y1 - width/2;

        for (ptrdiff_t x = x1; x <= x2; x++)
        {
            for (size_t i = 0; i < width; ++i)
            {
                ptrdiff_t y = y0 + i;
                if (y >= 0)
                {
                    if (inverse)
                    {
                        if (y < w)
                            At<A, Color>(canvas, y, x) = color;
                    }
                    else
                    {
                        if (y < h)
                            At<A, Color>(canvas, x, y) = color;
                    }
                }

            }

            error -= dy;
            if (error < 0)
            {
                y0 += ystep;
                error += dx;
            }
        }
    }

    /*! @ingroup cpp_drawing

        \fn void DrawLine(View<A> & canvas, const Point<ptrdiff_t> & p1, const Point<ptrdiff_t> & p2, const Color & color, size_t width = 1)

        \short Draws a line at the image.

        \param [out] canvas - a canvas (image where we draw line).
        \param [in] p1 - the first point of the line.
        \param [in] p2 - the second point of the line.
        \param [in] color - a color of the line.
        \param [in] width - a width of the line. By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawLine(View<A> & canvas, const Point<ptrdiff_t> & p1, const Point<ptrdiff_t> & p2, const Color & color, size_t width = 1)
    {
        DrawLine(canvas, p1.x, p1.y, p2.x, p2.y, color, width);
    }

    /*! @ingroup cpp_drawing

        \fn void DrawRectangle(View<A> & canvas, const Rectangle<ptrdiff_t> & rect, const Color & color, size_t width = 1)

        \short Draws a rectangle at the image.

        \param [out] canvas - a canvas (image where we draw rectangle).
        \param [in] rect - a rectangle.
        \param [in] color - a color of the rectangle's frame.
        \param [in] width - a width of the rectangle's frame. By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawRectangle(View<A> & canvas, const Rectangle<ptrdiff_t> & rect, const Color & color, size_t width = 1)
    {
        DrawLine(canvas, rect.left, rect.top, rect.right, rect.top, color, width);
        DrawLine(canvas, rect.right, rect.top, rect.right, rect.bottom, color, width);
        DrawLine(canvas, rect.right, rect.bottom, rect.left, rect.bottom, color, width);
        DrawLine(canvas, rect.left, rect.bottom, rect.left, rect.top, color, width);
    }

    /*! @ingroup cpp_drawing

        \fn void DrawFilledPolygon(View<A> & canvas, const std::vector<Simd::Point<ptrdiff_t>> & polygon, const Color & color)

        \short Draws a filled polygon at the image.

        \param [out] canvas - a canvas (image where we draw filled polygon).
        \param [in] polygon - a polygon.
        \param [in] color - a color of the rectangle's frame.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawFilledPolygon(View<A> & canvas, const std::vector<Simd::Point<ptrdiff_t>> & polygon, const Color & color)
    {
        assert(canvas.PixelSize() == sizeof(color));

        typedef Simd::Point<ptrdiff_t> Point;
        typedef std::vector<ptrdiff_t> Vector;

        ptrdiff_t top = canvas.height, bottom = 0;
        for (size_t i = 0; i < polygon.size(); ++i)
        {
            top = std::min(top, polygon[i].y);
            bottom = std::max(bottom, polygon[i].y);
        }
        top = std::max<ptrdiff_t>(0, top);
        bottom = std::min<ptrdiff_t>(bottom, canvas.height);

        for (ptrdiff_t y = top; y < bottom; ++y)
        {
            Vector intersections;
            for (size_t i = 0; i < polygon.size(); ++i)
            {
                const Point & p0 = (i ? polygon[i - 1] : polygon.back()), p1 = polygon[i];
                if ((y >= p0.y && y < p1.y) || (y >= p1.y && y < p0.y))
                    intersections.push_back(p0.x + (y - p0.y)*(p1.x - p0.x) / (p1.y - p0.y));
            }
            assert(intersections.size() % 2 == 0);
            std::sort(intersections.begin(), intersections.end());
            for (size_t i = 0; i < intersections.size(); i += 2)
            {
                ptrdiff_t left = std::max<ptrdiff_t>(0, intersections[i + 0]);
                ptrdiff_t right = std::min<ptrdiff_t>(canvas.width, intersections[i + 1]);
                Color * dst = & At<A, Color>(canvas, 0, y);
                for (ptrdiff_t x = left; x < right; ++x)
                    dst[x] = color;
            }
        }
    }
}

#endif//__SimdDrawing_hpp__
