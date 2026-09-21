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

        \short The Font class is a C++ wrapper for drawing ASCII text on images.

        The class wraps C API functions ::SimdFontInit, ::SimdFontResize, ::SimdFontHeight,
        ::SimdFontMeasure and ::SimdFontDraw. It uses a built-in monospace-like font generated
        from the generic monospace font of Gdiplus. The font supports printable ASCII glyphs.
        The character '\n' starts a new line. Other characters are ignored.

        A typical usage creates one Font, optionally resizes it to a fraction of the canvas
        height (for example canvas.height / 32), and then draws labels at pixel coordinates
        or at a named Simd::View::Position. Height() is used as a line step when several
        strings are stacked, or to place a label above a rectangle. Measure() returns the
        pixel size required to draw a string.

        The canvas must have 1, 2, 3 or 4 eight-bit channels. Pixel size of canvas must be
        equal to sizeof(Color). Text is blended into the canvas through an 8-bit glyph
        alpha mask. Drawing is clipped to the canvas.

        Using example:
        \code
        #include "Simd/SimdFont.hpp"

        int main()
        {
            typedef Simd::Pixel::Bgra32 Color;
            typedef Simd::Font::View View;
            typedef Simd::Font::Point Point;
            typedef Simd::Font::String String;

            View image(320, 240, View::Bgra32);

            Simd::FillPixel(image, Color(128, 128, 0));

            Simd::Font font(32);
            font.Draw(image, "Hello, Simd!", View::MiddleCenter, Color(0, 0, 255));

            String text = "First_string,\nSecond-line.";
            font.Draw(image, text, View::BottomRight, Color(0, 0, 0));
            font.Resize(24);
            font.Draw(image, text, View::TopLeft, Color(0, 0, 0), Color(255, 255, 255));

            font.Resize(16);
            font.Draw(image, "id=1", Point(8, 8), Color(255, 255, 255));
            font.Draw(image, "in 1", Point(8, 8 + (ptrdiff_t)font.Height()), Color(255, 255, 0));

            image.Save("HelloSimd.ppm");

            return 0;
        }
        \endcode

        \note This is a wrapper around the low-level \ref drawing API.
    */
    class Font
    {
    public:
        typedef std::string String; /*!< Text string type used by Measure and Draw. */
        typedef Simd::Point<ptrdiff_t> Point; /*!< Point type used as a text position (x, y) and as a measured text size (width, height). */
        typedef Simd::View<Simd::Allocator> View; /*!< Image type used as a canvas for Draw. */

        /*!
            Creates a new Font class with the given glyph height.

            The constructor creates an internal font context by ::SimdFontInit and then calls
            Resize with the given height. The font contains embedded ASCII glyphs (originally
            generated from the generic monospace font of Gdiplus). Supported glyphs are
            printable ASCII characters. The character '\n' starts a new line. Other characters
            are ignored.

            \param [in] height - initial glyph height in pixels. By default it is equal to 16.
                                 The value must be inside the supported range of the embedded font.
        */
        Font(size_t height = 16)
            : _context(NULL)
        {
            _context = SimdFontInit();
            if (_context)
                SimdFontResize(_context, height);
        }

        /*!
            Releases the internal font context.

            The destructor calls ::SimdRelease for the context created by ::SimdFontInit.
        */
        ~Font()
        {
            if (_context)
                SimdRelease(_context);
        }

        /*!
            Sets a new glyph height.

            The function recreates internal 8-bit alpha glyph images from embedded font data.
            It returns false if height is outside the supported range of the embedded font.
            Reusing the current height is a successful no-op. Typical usage scales the font
            to the canvas, for example canvas.height / 32.

            \param [in] height - a new glyph height in pixels.
            \return true on success and false on failure.
        */
        bool Resize(size_t height)
        {
            if (_context)
                return SimdFontResize(_context, height) != SimdFalse;
            return false;
        }

        /*!
            Gets current glyph height in pixels.

            The value is used as a vertical step when several text lines are drawn one under
            another, and to place a label above a rectangle (top - Height()).

            \return current glyph height in pixels. It is equal to 0 if the font context is not created.
        */
        size_t Height() const
        {
            if (_context)
                return SimdFontHeight(_context);
            return 0;
        }

        /*!
            Measures the size of the rectangle required to draw the given text.

            Supported glyphs advance the current X position by the current glyph width.
            The '\n' character starts a new line and advances Y by the current glyph height.
            Unsupported characters are ignored. If the text contains at least one drawable
            glyph, the returned size also includes the font indentation on all sides.

            The measured size is used by Draw overloads that take Simd::View::Position in
            order to create a destination region with Simd::View::Region.

            \param [in] text - a text to measure.
            \return measured size (width in x, height in y). It is equal to (0, 0) if the
                    font context is not created or the text has no drawable glyphs.
        */
        Point Measure(const String & text) const
        {
            Point size;
            if (_context)
                SimdFontMeasure(_context, text.c_str(), (size_t*)&size.x, (size_t*)&size.y);
            return size;
        }

        /*!
            Draws text at the given pixel position.

            The position is the top-left corner of the measured text region; glyphs are
            shifted by the current font indentation inside it. Drawing is clipped to the
            canvas. Supported glyphs advance X by the current glyph width, '\n' starts a
            new line, and unsupported characters are ignored.

            The canvas must have 1, 2, 3 or 4 eight-bit channels. Pixel size of canvas must
            be equal to sizeof(Color). Text is blended into the canvas through an 8-bit
            glyph alpha mask.

            \note This function is a C++ wrapper for function ::SimdFontDraw.

            \param [out] canvas - a canvas image.
            \param [in] text - a text to draw.
            \param [in] position - the top-left position of the measured text region.
            \param [in] color - a color of the text. Pixel size of canvas must be equal to sizeof(Color).
            \return true if the font context exists; otherwise false.
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
            Draws text at a named position of the canvas.

            The function measures the text, takes a region of this size with Simd::View::Region
            at the given Simd::View::Position (for example View::TopLeft, View::MiddleCenter,
            View::BottomRight) and draws the text at the origin of this region.

            \param [out] canvas - a canvas image.
            \param [in] text - a text to draw.
            \param [in] position - a named position of the text region (see Simd::View::Position).
            \param [in] color - a color of the text. Pixel size of canvas must be equal to sizeof(Color).
            \return true if the font context exists; otherwise false.
        */
        template <class Color> bool Draw(View & canvas, const String & text, const View::Position & position, const Color & color) const
        {
            return Draw(canvas.Region(Measure(text), position).Ref(), text, Point(0, 0), color);
        }

        /*!
            Draws text at a named position and fills the text background.

            The function measures the text, takes a region of this size with Simd::View::Region
            at the given Simd::View::Position, fills this region by the background color with
            Simd::FillPixel and then draws the text.

            \param [out] canvas - a canvas image.
            \param [in] text - a text to draw.
            \param [in] position - a named position of the text region (see Simd::View::Position).
            \param [in] color - a color of the text. Pixel size of canvas must be equal to sizeof(Color).
            \param [in] background - a background color of the text region. Pixel size of canvas must be equal to sizeof(Color).
            \return true if the font context exists; otherwise false.
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
