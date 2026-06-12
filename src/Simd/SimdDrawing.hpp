/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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

        \short Draws a clipped line segment on an image.

        The function draws a line from (x1, y1) to (x2, y2) into canvas. Coordinates use the usual
        image coordinate system: X grows to the right and Y grows downward. The segment is clipped
        to the canvas bounds; if it is completely outside, the function does nothing. Pixel size of
        canvas must be equal to sizeof(Color). Only colors with 1, 2, 3 or 4 bytes are supported.

        \note This function is a C++ wrapper for function ::SimdDrawLine.

        \param [out] canvas - a canvas image.
        \param [in] x1 - X coordinate of the first point of the line.
        \param [in] y1 - Y coordinate of the first point of the line.
        \param [in] x2 - X coordinate of the second point of the line.
        \param [in] y2 - Y coordinate of the second point of the line.
        \param [in] color - a color of the line.
        \param [in] width - a line width (in pixels). By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawLine(View<A> & canvas, ptrdiff_t x1, ptrdiff_t y1, ptrdiff_t x2, ptrdiff_t y2, const Color & color, size_t width = 1)
    {
        assert(canvas.PixelSize() == sizeof(Color));

        SimdDrawLine(canvas.data, canvas.stride, canvas.width, canvas.height, sizeof(Color), x1, y1, x2, y2, (const uint8_t*)&color, width);
    }

    /*! @ingroup cpp_drawing

        \fn void DrawLine(View<A> & canvas, const Point<ptrdiff_t> & p1, const Point<ptrdiff_t> & p2, const Color & color, size_t width = 1)

        \short Draws a clipped line segment on an image.

        The function draws a line from p1 to p2 into canvas. Coordinates use the usual image
        coordinate system: X grows to the right and Y grows downward. The segment is clipped to the
        canvas bounds; if it is completely outside, the function does nothing. Pixel size of canvas
        must be equal to sizeof(Color). Only colors with 1, 2, 3 or 4 bytes are supported.

        \note This function calls Simd::DrawLine(View<A> & canvas, ptrdiff_t x1, ptrdiff_t y1, ptrdiff_t x2, ptrdiff_t y2, const Color & color, size_t width = 1).

        \param [out] canvas - a canvas image.
        \param [in] p1 - the first point of the line.
        \param [in] p2 - the second point of the line.
        \param [in] color - a color of the line.
        \param [in] width - a line width (in pixels). By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawLine(View<A> & canvas, const Point<ptrdiff_t> & p1, const Point<ptrdiff_t> & p2, const Color & color, size_t width = 1)
    {
        DrawLine(canvas, p1.x, p1.y, p2.x, p2.y, color, width);
    }

    /*! @ingroup cpp_drawing

        \fn void DrawRectangle(View<A> & canvas, ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, const Color & color, size_t width = 1)

        \short Draws a clipped rectangle frame on an image.

        The function draws four clipped line segments: (left, top)-(right, top),
        (right, top)-(right, bottom), (right, bottom)-(left, bottom) and
        (left, bottom)-(left, top). Coordinates use the usual image coordinate system:
        X grows to the right and Y grows downward. Pixel size of canvas must be equal
        to sizeof(Color). Only colors with 1, 2, 3 or 4 bytes are supported.

        \note This function is a C++ wrapper for function ::SimdDrawRectangle.

        \param [out] canvas - a canvas image.
        \param [in] left - X coordinate of the left side of the rectangle.
        \param [in] top - Y coordinate of the top side of the rectangle.
        \param [in] right - X coordinate of the right side of the rectangle.
        \param [in] bottom - Y coordinate of the bottom side of the rectangle.
        \param [in] color - a color of the rectangle frame.
        \param [in] width - a width of the rectangle frame (in pixels). By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawRectangle(View<A>& canvas, ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, const Color& color, size_t width = 1)
    {
        assert(canvas.PixelSize() == sizeof(Color));

        SimdDrawRectangle(canvas.data, canvas.stride, canvas.width, canvas.height, sizeof(Color), left, top, right, bottom, (const uint8_t*)&color, width);
    }

    /*! @ingroup cpp_drawing

        \fn void DrawRectangle(View<A> & canvas, const Rectangle<ptrdiff_t> & rect, const Color & color, size_t width = 1)

        \short Draws a clipped rectangle frame on an image.

        The function draws the rectangle frame using rect.left, rect.top, rect.right and rect.bottom
        as side coordinates. The frame sides are clipped to the canvas bounds. Pixel size of canvas
        must be equal to sizeof(Color). Only colors with 1, 2, 3 or 4 bytes are supported.

        \note This function is a C++ wrapper for function ::SimdDrawRectangle.

        \param [out] canvas - a canvas image.
        \param [in] rect - a rectangle frame.
        \param [in] color - a color of the rectangle frame.
        \param [in] width - a width of the rectangle frame (in pixels). By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawRectangle(View<A> & canvas, const Rectangle<ptrdiff_t> & rect, const Color & color, size_t width = 1)
    {
        assert(canvas.PixelSize() == sizeof(Color));

        SimdDrawRectangle(canvas.data, canvas.stride, canvas.width, canvas.height, sizeof(Color), rect.left, rect.top, rect.right, rect.bottom, (const uint8_t*)&color, width);
    }

    /*! @ingroup cpp_drawing

        \fn void DrawRectangle(View<A> & canvas, const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight, const Color & color, size_t width = 1)

        \short Draws a clipped rectangle frame on an image.

        The function draws the rectangle frame using topLeft and bottomRight as side coordinates.
        The frame sides are clipped to the canvas bounds. Pixel size of canvas must be equal to
        sizeof(Color). Only colors with 1, 2, 3 or 4 bytes are supported.

        \note This function calls Simd::DrawRectangle(View<A> & canvas, const Rectangle<ptrdiff_t> & rect, const Color & color, size_t width = 1).

        \param [out] canvas - a canvas image.
        \param [in] topLeft - a top-left corner of the rectangle.
        \param [in] bottomRight - a bottom-right corner of the rectangle.
        \param [in] color - a color of the rectangle frame.
        \param [in] width - a width of the rectangle frame (in pixels). By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawRectangle(View<A> & canvas, const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight, const Color & color, size_t width = 1)
    {
        DrawRectangle<A, Color>(canvas, Rectangle<ptrdiff_t>(topLeft, bottomRight), color, width);
    }

    /*! @ingroup cpp_drawing

        \fn void DrawFilledRectangle(View<A> & canvas, Rectangle<ptrdiff_t> rect, const Color & color)

        \short Draws a filled rectangle on an image.

        The function fills the rectangle area [rect.left, rect.right) x [rect.top, rect.bottom)
        with the specified color. The filled area is clipped to the canvas bounds. Pixel size of
        canvas must be equal to sizeof(Color).

        \param [out] canvas - a canvas image.
        \param [in] rect - a rectangle area to fill.
        \param [in] color - a color of the filled rectangle.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawFilledRectangle(View<A> & canvas, Rectangle<ptrdiff_t> rect, const Color & color)
    {
        assert(canvas.PixelSize() == sizeof(color));

        if (sizeof(Color) <= 4)
            Simd::FillPixel<A, Color>(canvas.Region(rect).Ref(), color);
        else
        {
            rect &= Rectangle<ptrdiff_t>(canvas.Size());
            for (ptrdiff_t row = rect.top; row < rect.bottom; ++row)
            {
                Color * dst = &At<A, Color>(canvas, 0, row);
                for (ptrdiff_t col = rect.left; col < rect.right; ++col)
                    dst[col] = color;
            }
        }
    }

    /*! @ingroup cpp_drawing

        \fn void DrawPolygon(View<A> & canvas, const std::vector<Simd::Point<ptrdiff_t>> & polygon, const Color & color, size_t width = 1)

        \short Draws a clipped polygon frame on an image.

        The function draws a closed polyline through all polygon points. The last point is connected
        with the first one. Each polygon side is drawn by Simd::DrawLine and clipped to the canvas
        bounds. Pixel size of canvas must be equal to sizeof(Color). Only colors with 1, 2, 3 or 4
        bytes are supported.

        \param [out] canvas - a canvas image.
        \param [in] polygon - polygon vertices in drawing order.
        \param [in] color - a color of the polygon frame.
        \param [in] width - a width of the polygon frame (in pixels). By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawPolygon(View<A> & canvas, const std::vector<Simd::Point<ptrdiff_t>> & polygon, const Color & color, size_t width = 1)
    {
        assert(canvas.PixelSize() == sizeof(color));

        typedef Simd::Point<ptrdiff_t> Point;

        for (size_t i = 0; i < polygon.size(); ++i)
        {
            const Point & p1 = (i ? polygon[i - 1] : polygon.back()), p2 = polygon[i];
            DrawLine(canvas, p1, p2, color, width);
        }
    }

    /*! @ingroup cpp_drawing

        \fn void DrawFilledPolygon(View<A> & canvas, const std::vector<Simd::Point<ptrdiff_t>> & polygon, const Color & color)

        \short Draws a filled polygon on an image.

        The function fills a polygon with a scanline algorithm. For every canvas row intersecting
        the polygon, edge intersections are sorted and paired, then pixels between each pair are
        filled. Parts outside the canvas bounds are clipped. Pixel size of canvas must be equal to
        sizeof(Color).

        \param [out] canvas - a canvas image.
        \param [in] polygon - polygon vertices in drawing order.
        \param [in] color - a color of the filled polygon.
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
                Color * dst = &At<A, Color>(canvas, 0, y);
                for (ptrdiff_t x = left; x < right; ++x)
                    dst[x] = color;
            }
        }
    }

    /*! @ingroup cpp_drawing

        \fn void DrawEllipse(View<A> & canvas, const Point<ptrdiff_t> & center, const Point<ptrdiff_t> & axes, double slope, const Color & color, size_t width = 1)

        \short Draws a clipped ellipse frame on an image.

        The function approximates the ellipse by short line segments and draws them by Simd::DrawLine.
        Axes contain ellipse radii along local X and Y directions. Slope is the rotation angle in
        radians. The resulting frame is clipped to the canvas bounds. Pixel size of canvas must be
        equal to sizeof(Color). Only colors with 1, 2, 3 or 4 bytes are supported.

        \param [out] canvas - a canvas image.
        \param [in] center - a center of the ellipse.
        \param [in] axes - ellipse radii along local X and Y directions.
        \param [in] slope - a rotation angle of the ellipse (in radians).
        \param [in] color - a color of the ellipse frame.
        \param [in] width - a width of the ellipse frame (in pixels). By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawEllipse(View<A> & canvas, const Point<ptrdiff_t> & center, const Point<ptrdiff_t> & axes, double slope, const Color & color, size_t width = 1)
    {
        assert(canvas.PixelSize() == sizeof(color));

        const size_t n = 8 * std::max((size_t)1, (size_t)::pow(axes.x*axes.x + axes.y*axes.y, 0.25));
        double ss = ::sin(slope);
        double sc = ::cos(slope);
        double px, py, da = 2 * M_PI / n;
        for (size_t i = 0; i <= n; ++i)
        {
            double a = i*da;
            double ax = ::sin(a)*axes.x;
            double ay = ::cos(a)*axes.y;
            double cx = ax*sc + ay*ss + center.x;
            double cy = ay*sc - ax*ss + center.y;
            if (i > 0)
                DrawLine(canvas, (ptrdiff_t)cx, (ptrdiff_t)cy, (ptrdiff_t)px, (ptrdiff_t)py, color, width);
            px = cx;
            py = cy;
        }
    }

    /*! @ingroup cpp_drawing

        \fn void DrawCircle(View<A> & canvas, const Point<ptrdiff_t> & center, ptrdiff_t radius, const Color & color, size_t width = 1)

        \short Draws a clipped circle frame on an image.

        The function approximates the circle by short line segments and draws them by Simd::DrawLine.
        The resulting frame is clipped to the canvas bounds. Pixel size of canvas must be equal to
        sizeof(Color). Only colors with 1, 2, 3 or 4 bytes are supported.

        \param [out] canvas - a canvas image.
        \param [in] center - a center of the circle.
        \param [in] radius - a radius of the circle.
        \param [in] color - a color of the circle frame.
        \param [in] width - a width of the circle frame (in pixels). By default it is equal to 1.
    */
    template<template<class> class A, class Color> SIMD_INLINE void DrawCircle(View<A> & canvas, const Point<ptrdiff_t> & center, ptrdiff_t radius, const Color & color, size_t width = 1)
    {
        assert(canvas.PixelSize() == sizeof(color));

        const size_t n = 8 * std::max((size_t)1, (size_t)::pow(radius, 0.5));
        double px, py, da = 2 * M_PI / n;
        for (size_t i = 0; i <= n; ++i)
        {
            double a = i*da;
            double cx = radius*::cos(a) + center.x;
            double cy = radius*::sin(a) + center.y;
            if (i > 0)
                DrawLine(canvas, (ptrdiff_t)cx, (ptrdiff_t)cy, (ptrdiff_t)px, (ptrdiff_t)py, color, width);
            px = cx;
            py = cy;
        }
    }
}

#endif//__SimdDrawing_hpp__
