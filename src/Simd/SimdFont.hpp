/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#ifndef __SimdFont_hpp__
#define __SimdFont_hpp__

#include "Simd/SimdLib.hpp"
#include "Simd/SimdDrawing.hpp"

#include <vector>
#include <string>
#include <algorithm>

namespace Simd
{
    /*! @ingroup cpp_drawing

        \short The Font class provides text drawing.

        Using example:
        \code
        #include "Simd/SimdFont.hpp"

        int main()
        {
            typedef Simd::Pixel::Bgra32 Color;
            typedef Simd::Font::View View;

            Simd::Font font(32);

            View image(320, 240, View::Bgra32);

            Simd::FillPixel(image, Color(128, 128, 0));

            font.Draw(image, "Hello, Simd!", View::MiddleCenter, Color(0, 0, 255));

            image.Save("HelloSimd.ppm");

            return 0;
        }
        \endcode
    */
    class Font
    {
    public:
        typedef std::string String; /*!< String type definition. */
        typedef Simd::Point<ptrdiff_t> Point; /*!< Point type definition. */
        typedef Simd::View<Simd::Allocator> View; /*!< Image time definition. */

        /*!
            Creates a new Font class with given height.

            \note The font supports ASCII characters only. It was generated on the base of the generic monospace font from Gdiplus.

            \param [in] height - initial height value. By default it is equal to 16.
        */
        Font(size_t height = 16)
            : _context(NULL)
        {
            _context = SimdFontInit();
            if (_context)
                SimdFontResize(_context, height);
        }

        /*!
            Font class destructor.
        */
        ~Font()
        {
            if (_context)
                SimdRelease(_context);
        }

        /*!
            Sets a new height value to font.

            \param [in] height - a new height value. 

            \return a result of the operation.
        */
        bool Resize(size_t height)
        {
            if (_context)
                return SimdFontResize(_context, height) != SimdFalse;
            return false;
        }

        /*!
            Gets height of the font.

            \return current height of the font.
        */
        size_t Height() const
        {
            if (_context)
                return SimdFontHeight(_context);
            return 0;
        }

        /*!
            Measures a size of region is need to draw given text.

            \param [in] text - a text to draw.

            \return measured size.
        */
        Point Measure(const String & text) const
        {
            Point size;
            if (_context)
                SimdFontMeasure(_context, text.c_str(), (size_t*)&size.x, (size_t*)&size.y);
            return size;
        }

        /*!
            Draws a text at the image.

            \param [out] canvas - a canvas (image where we draw text).
            \param [in] text - a text to draw.
            \param [in] position - a start position to draw text.
            \param [in] color - a color of the text.

            \return a result of the operation.
        */
        template <class Color> bool Draw(View & canvas, const String & text, const Point & position, const Color & color) const
        {
            assert(sizeof(color) == canvas.PixelSize());

            if (_context)
            {
                SimdFontDraw(_context, canvas.data, canvas.stride, canvas.width, canvas.height, canvas.PixelSize(), text.c_str(), position.x, position.y, (uint8_t*)&color);
                return true;
            }
            return false;
        }

        /*!
            Draws a text at the image.

            \param [out] canvas - a canvas (image where we draw text).
            \param [in] text - a text to draw.
            \param [in] position - a position to draw text (see Simd::View::Position).
            \param [in] color - a color of the text.

            \return a result of the operation.
        */
        template <class Color> bool Draw(View & canvas, const String & text, const View::Position & position, const Color & color) const
        {
            return Draw(canvas.Region(Measure(text), position).Ref(), text, Point(0, 0), color);
        }

        /*!
            Draws a text at the image. Fills the text background by given color.

            \param [out] canvas - a canvas (image where we draw text).
            \param [in] text - a text to draw.
            \param [in] position - a position to draw text (see Simd::View::Position).
            \param [in] color - a color of the text.
            \param [in] background - background color.

            \return a result of the operation.
        */
        template <class Color> bool Draw(View & canvas, const String & text, const View::Position & position, const Color & color, const Color & background) const
        {
            View region = canvas.Region(Measure(text), position);
            Simd::FillPixel(region, background);
            return Draw(region, text, Point(0, 0), color);
        }

    private:
        void* _context;
    };
}

#endif
