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
#ifndef __SimdPixel_hpp__
#define __SimdPixel_hpp__

#include "Simd/SimdView.hpp"

namespace Simd
{
    /*! @ingroup cpp_pixels

        \short Contains pixel structures for Simd::View formats and drawing colors.

        Each structure is the memory layout of the matching Simd::View::Format
        (Bgr24, Bgra32, Hsv24, Hsl24, Rgb24, Rgba32). Channel fields are stored
        in that order and the structure size is the pixel size, so a value can be
        written into a view and passed as the color of Simd::DrawLine,
        Simd::DrawRectangle, Simd::DrawFilledRectangle, Simd::Font::Draw and
        Simd::FillPixel. Those functions require sizeof(color) to equal the
        canvas pixel size.

        OpenCV frames used by the motion and face-detection examples are BGR.
        They use Pixel::Bgr24 as the color type. TestMotion paints alarmed
        objects with Bgr24(0, 0, 255), other objects with Bgr24(0, 255, 255)
        and event text with Bgr24(255, 255, 255). DrawRectangle, DrawLine and
        Font::Draw take that color on the Bgr24 frame. UseFaceDetection draws
        each face with DrawRectangle(image, rect, Pixel::Bgr24(0, 255, 255)).
        The Font example uses Pixel::Bgra32 on a Bgra32 canvas, including
        FillPixel(image, Bgra32(128, 128, 0)).

        Bgr24 and Bgra32 constructors take channels as blue, green, red.
        Rgb24 and Rgba32 constructors take them as red, green, blue.
        TestRandom::GetColor therefore builds Bgr24(b, g, r) and Rgb24(r, g, b)
        from the same (b, g, r) triple, so "B", "G" and "R" labels have the same
        color on BGR and RGB canvases. Conversion constructors copy the matching
        channels and exchange red and blue between BGR and RGB layouts. Alpha is
        omitted when the result is 24-bit. A 24-bit source converted to 32 bits
        uses alpha 255 unless another value is passed. Bgra32 and Rgba32
        conversions copy alpha.

        Pixels are read and written with View::At and View::Row. The transform
        test assigns view.At<Pixel::Bgr24>(x, y) and view.At<Pixel::Bgra32>(x, y).
        The font test fills a row with View::Row<Pixel::Bgra32>. The static At
        methods do the same lookup and also require view.format to match the
        pixel type. col is the x coordinate and row is the y coordinate.

        Hsv24 and Hsl24 are the element types of images produced by Simd::BgrToHsv
        and Simd::BgrToHsl. Every component is an 8-bit value in [0, 255].

        Using example:
        \code
        #include "Simd/SimdFont.hpp"

        int main()
        {
            typedef Simd::View<Simd::Allocator> View;
            typedef Simd::Pixel::Bgr24 Color;
            typedef Simd::Rectangle<ptrdiff_t> Rect;
            typedef Simd::Point<ptrdiff_t> Point;

            View image(320, 240, View::Bgr24);
            const Color Red(0, 0, 255), Yellow(0, 255, 255);
            Rect object(40, 30, 120, 90);
            Simd::DrawRectangle(image, object, Yellow, 1);

            Simd::Font font(image.height / 32);
            font.Draw(image, "1", Point(object.left, object.top - (ptrdiff_t)font.Height()), Red);

            for (size_t y = 0; y < 8; ++y)
                for (size_t x = 0; x < image.width; ++x)
                {
                    Color & pixel = image.At<Color>(x, y);
                    pixel.blue = uint8_t(y);
                    pixel.green = uint8_t(x);
                    pixel.red = 255;
                }

            typedef Simd::Pixel::Bgra32 Color32;
            View bgra(320, 240, View::Bgra32);
            Simd::FillPixel(bgra, Color32(128, 128, 0));
            font.Draw(bgra, "Hello, Simd!", View::MiddleCenter, Color32(0, 0, 255));

            uint8_t blue = 0, green = 255, red = 255;
            Simd::DrawRectangle(image, Rect(0, 0, 16, 16), Color(blue, green, red));
            View rgbImage(16, 16, View::Rgb24);
            Simd::FillPixel(rgbImage, Simd::Pixel::Rgb24(red, green, blue));
            return 0;
        }
        \endcode
    */
    namespace Pixel
    {
        struct Bgr24;
        struct Bgra32;
        struct Hsv24;
        struct Hsl24;
        struct Rgb24;
        struct Rgba32;

        //-------------------------------------------------------------------------

        /*! @ingroup cpp_pixels

            \short 24-bit BGR pixel.

            Stores blue, green, red as three consecutive bytes. This is the element
            type of a View::Bgr24 image and the color type used on BGR canvases.
            UseMotionDetector and TestMotion alias it as Color and pass it to
            DrawRectangle, DrawLine and Font::Draw. UseFaceDetection passes
            Bgr24(0, 255, 255) to DrawRectangle. The bytes are the same order as
            Simd::FillBgr and as an OpenCV BGR cv::Mat.
        */
        struct Bgr24
        {
            uint8_t blue; /*!< \brief Blue channel. First byte of a View::Bgr24 pixel. */
            uint8_t green; /*!< \brief Green channel. Second byte of a View::Bgr24 pixel. */
            uint8_t red; /*!< \brief Red channel. Third byte of a View::Bgr24 pixel. Bgr24(0, 0, 255) is red. */

            /*!
                Creates a 24-bit BGR pixel with the same value in every channel.

                Bgr24() is black. The value is written to blue, green and red.

                \param [in] gray - initial value for blue, green and red. It is equal to 0 by default.
            */
            Bgr24(const uint8_t & gray = uint8_t(0));

            /*!
                Creates a 24-bit BGR pixel from blue, green and red.

                Argument order is blue, green, red. Motion annotation uses
                Bgr24(0, 0, 255) for red, Bgr24(0, 255, 255) for yellow and
                Bgr24(255, 255, 255) for white. Debug drawing inside Detector
                uses Bgr24(0, 255, 0) for moving regions and Bgr24(0, 255, 255)
                for the model ROI and tracked objects. The transform test writes
                Bgr24(y, x, red) through View::At, so blue follows the row and
                green follows the column.

                \param [in] b - initial value for the blue channel.
                \param [in] g - initial value for the green channel.
                \param [in] r - initial value for the red channel.
            */
            Bgr24(const uint8_t & b, const uint8_t & g, const uint8_t & r);

            /*!
                Creates a 24-bit BGR pixel from a 32-bit BGRA pixel.

                Copies blue, green and red. Alpha is omitted.

                \param [in] p - a 32-bit BGRA pixel.
            */
            Bgr24(const Bgra32 & p);

            /*!
                Creates a 24-bit BGR pixel from a 24-bit RGB pixel.

                Copies the channels by name, so the color is kept and the byte
                order becomes blue, green, red.

                \param [in] p - a 24-bit RGB pixel.
            */
            Bgr24(const Rgb24 & p);

            /*!
                Creates a 24-bit BGR pixel from a 32-bit RGBA pixel.

                Copies blue, green and red by name. Alpha is omitted.

                \param [in] p - a 32-bit RGBA pixel.
            */
            Bgr24(const Rgba32& p);

            /*!
                Creates a copy of a 24-bit BGR pixel.

                \param [in] p - a 24-bit BGR pixel.
            */
            Bgr24(const Bgr24 & p);

            /*!
                \fn template <template<class> class A> static const Bgr24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a constant reference to the BGR pixel at (col, row).

                Requires view.format == View::Bgr24 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view. Typical code uses the equivalent
                view.At<Pixel::Bgr24>(x, y).

                \param [in] view - an image view of 24-bit BGR pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a constant reference to the pixel.
            */
            template <template<class> class A> static const Bgr24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

            /*!
                \fn template <template<class> class A> static Bgr24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a reference to the BGR pixel at (col, row).

                Requires view.format == View::Bgr24 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view. The transform test assigns pixels with
                view.At<Pixel::Bgr24>(x, y) = Pixel::Bgr24(y, x, red).

                \param [in] view - an image view of 24-bit BGR pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a reference to the pixel.
            */
            template <template<class> class A> static Bgr24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);
        };

        /*! @ingroup cpp_pixels

            \short 32-bit BGRA pixel.

            Stores blue, green, red, alpha as four consecutive bytes. This is the
            element type of a View::Bgra32 image and the color type of the Font
            example and TestFont. FillPixel, Font::Draw and the drawing functions
            write these four bytes when the canvas pixel size is 4. The byte order
            matches Simd::FillBgra. TestRandom::GetColor calls Bgra32(b, g, r) and
            leaves alpha at 255.
        */
        struct Bgra32
        {
            uint8_t blue; /*!< \brief Blue channel. First byte of a View::Bgra32 pixel. */
            uint8_t green; /*!< \brief Green channel. Second byte of a View::Bgra32 pixel. */
            uint8_t red; /*!< \brief Red channel. Third byte of a View::Bgra32 pixel. Bgra32(0, 0, 255) is opaque red. */
            uint8_t alpha; /*!< \brief Alpha channel. Fourth byte of a View::Bgra32 pixel. 255 is opaque. */

            /*!
                Creates a 32-bit BGRA pixel with equal blue, green and red.

                Bgra32() is opaque black. The Font example fills a canvas with
                FillPixel(image, Bgra32(128, 128, 0)): blue = 128, green = 128,
                red = 0 and alpha = 255.

                \param [in] gray - initial value for blue, green and red. It is equal to 0 by default.
                \param [in] a - initial value for alpha. It is equal to 255 by default.
            */
            Bgra32(const uint8_t & gray = uint8_t(0), const uint8_t & a = uint8_t(255));

            /*!
                Creates a 32-bit BGRA pixel from blue, green, red and alpha.

                Argument order is blue, green, red, alpha. The Font example draws
                red text with Bgra32(0, 0, 255), white text with Bgra32(255, 255, 255)
                and cyan text with Bgra32(255, 255, 0). The font test fills each
                row with Bgra32(255, x, y), so blue is 255, green follows the column
                and red follows the row. The transform test assigns
                view.At<Pixel::Bgra32>(x, y) = Pixel::Bgra32(x, y, red, alpha).

                \param [in] b - initial value for the blue channel.
                \param [in] g - initial value for the green channel.
                \param [in] r - initial value for the red channel.
                \param [in] a - initial value for the alpha channel. It is equal to 255 by default.
            */
            Bgra32(const uint8_t & b, const uint8_t & g, const uint8_t & r, const uint8_t & a = uint8_t(255));

            /*!
                Creates a 32-bit BGRA pixel from a 24-bit BGR pixel.

                Copies blue, green and red. Alpha is 255 unless another value is passed.

                \param [in] p - a 24-bit BGR pixel.
                \param [in] a - initial value for the alpha channel. It is equal to 255 by default.
            */
            Bgra32(const Bgr24 & p, const uint8_t & a = uint8_t(255));

            /*!
                Creates a 32-bit BGRA pixel from a 24-bit RGB pixel.

                Copies blue, green and red by name, so the color is kept and the
                byte order becomes blue, green, red, alpha.

                \param [in] p - a 24-bit RGB pixel.
                \param [in] a - initial value for the alpha channel. It is equal to 255 by default.
            */
            Bgra32(const Rgb24 & p, const uint8_t & a = uint8_t(255));

            /*!
                Creates a 32-bit BGRA pixel from a 32-bit RGBA pixel.

                Copies blue, green, red and alpha by name.

                \param [in] p - a 32-bit RGBA pixel.
            */
            Bgra32(const Rgba32& p);

            /*!
                Creates a copy of a 32-bit BGRA pixel.

                \param [in] p - a 32-bit BGRA pixel.
            */
            Bgra32(const Bgra32 & p);

            /*!
                \fn template <template<class> class A> static const Bgra32 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a constant reference to the BGRA pixel at (col, row).

                Requires view.format == View::Bgra32 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view. Typical code uses view.At<Pixel::Bgra32>(x, y)
                or View::Row<Pixel::Bgra32>.

                \param [in] view - an image view of 32-bit BGRA pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a constant reference to the pixel.
            */
            template <template<class> class A> static const Bgra32 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

            /*!
                \fn template <template<class> class A> static Bgra32 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a reference to the BGRA pixel at (col, row).

                Requires view.format == View::Bgra32 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view. The transform test assigns
                view.At<Pixel::Bgra32>(x, y) = Pixel::Bgra32(x, y, red, alpha).

                \param [in] view - an image view of 32-bit BGRA pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a reference to the pixel.
            */
            template <template<class> class A> static Bgra32 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);
        };

        /*! @ingroup cpp_pixels

            \short 24-bit HSV pixel.

            Stores hue, saturation, value as three consecutive bytes. This is the
            element type of a View::Hsv24 image, including the output of
            Simd::BgrToHsv. Each component is an 8-bit value in [0, 255], in the
            same order as the C description of ::SimdPixelFormatHsv24. There is
            no constructor from Bgr24; convert a whole image with Simd::BgrToHsv
            and then read pixels with View::At<Pixel::Hsv24> or Hsv24::At.
        */
        struct Hsv24
        {
            uint8_t hue; /*!< \brief Hue channel. First byte of a View::Hsv24 pixel. Range [0, 255]. */
            uint8_t saturation; /*!< \brief Saturation channel. Second byte of a View::Hsv24 pixel. 0 is gray. */
            uint8_t value; /*!< \brief Value channel. Third byte of a View::Hsv24 pixel. */

            /*!
                Creates a gray 24-bit HSV pixel.

                Hue and saturation are set to 0. Only the value channel receives
                gray. Hsv24() is black.

                \param [in] gray - initial value of the value channel. It is equal to 0 by default.
            */
            Hsv24(const uint8_t & gray = uint8_t(0));

            /*!
                Creates a 24-bit HSV pixel from hue, saturation and value.

                The values are stored in that order. They use the same 8-bit
                encoding as Simd::BgrToHsv.

                \param [in] h - initial value for the hue channel.
                \param [in] s - initial value for the saturation channel.
                \param [in] v - initial value for the value channel.
            */
            Hsv24(const uint8_t & h, const uint8_t & s, const uint8_t & v);

            /*!
                Creates a copy of a 24-bit HSV pixel.

                \param [in] p - a 24-bit HSV pixel.
            */
            Hsv24(const Hsv24 & p);

            /*!
                \fn template <template<class> class A> static const Hsv24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a constant reference to the HSV pixel at (col, row).

                Requires view.format == View::Hsv24 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view. Use this after Simd::BgrToHsv, or use
                view.At<Pixel::Hsv24>(x, y).

                \param [in] view - an image view of 24-bit HSV pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a constant reference to the pixel.
            */
            template <template<class> class A> static const Hsv24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

            /*!
                \fn template <template<class> class A> static Hsv24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a reference to the HSV pixel at (col, row).

                Requires view.format == View::Hsv24 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view.

                \param [in] view - an image view of 24-bit HSV pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a reference to the pixel.
            */
            template <template<class> class A> static Hsv24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);
        };

        /*! @ingroup cpp_pixels

            \short 24-bit HSL pixel.

            Stores hue, saturation, lightness as three consecutive bytes. This is
            the element type of a View::Hsl24 image, including the output of
            Simd::BgrToHsl. Each component is an 8-bit value in [0, 255], in the
            same order as the C description of ::SimdPixelFormatHsl24. There is
            no constructor from Bgr24; convert a whole image with Simd::BgrToHsl
            and then read pixels with View::At<Pixel::Hsl24> or Hsl24::At.
        */
        struct Hsl24
        {
            uint8_t hue; /*!< \brief Hue channel. First byte of a View::Hsl24 pixel. Range [0, 255]. */
            uint8_t saturation; /*!< \brief Saturation channel. Second byte of a View::Hsl24 pixel. 0 is gray. */
            uint8_t lightness; /*!< \brief Lightness channel. Third byte of a View::Hsl24 pixel. */

            /*!
                Creates a gray 24-bit HSL pixel.

                Hue and saturation are set to 0. Only the lightness channel
                receives gray. Hsl24() is black.

                \param [in] gray - initial value of the lightness channel. It is equal to 0 by default.
            */
            Hsl24(const uint8_t & gray = uint8_t(0));

            /*!
                Creates a 24-bit HSL pixel from hue, saturation and lightness.

                The values are stored in that order. They use the same 8-bit
                encoding as Simd::BgrToHsl.

                \param [in] h - initial value for the hue channel.
                \param [in] s - initial value for the saturation channel.
                \param [in] l - initial value for the lightness channel.
            */
            Hsl24(const uint8_t & h, const uint8_t & s, const uint8_t & l);

            /*!
                Creates a copy of a 24-bit HSL pixel.

                \param [in] p - a 24-bit HSL pixel.
            */
            Hsl24(const Hsl24 & p);

            /*!
                \fn template <template<class> class A> static const Hsl24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a constant reference to the HSL pixel at (col, row).

                Requires view.format == View::Hsl24 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view. Use this after Simd::BgrToHsl, or use
                view.At<Pixel::Hsl24>(x, y).

                \param [in] view - an image view of 24-bit HSL pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a constant reference to the pixel.
            */
            template <template<class> class A> static const Hsl24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

            /*!
                \fn template <template<class> class A> static Hsl24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a reference to the HSL pixel at (col, row).

                Requires view.format == View::Hsl24 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view.

                \param [in] view - an image view of 24-bit HSL pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a reference to the pixel.
            */
            template <template<class> class A> static Hsl24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);
        };

        /*! @ingroup cpp_pixels

            \short 24-bit RGB pixel.

            Stores red, green, blue as three consecutive bytes. This is the
            element type of a View::Rgb24 image. Argument order is red, green,
            blue, which is the reverse of Bgr24. TestRandom::GetColor uses that
            difference: for an Rgb24 canvas it calls Rgb24(r, g, b) where the
            caller passed (b, g, r), so the painted color matches a Bgr24 canvas.
            Pass Rgb24 to DrawRectangle, Font::Draw and FillPixel only when the
            canvas format is View::Rgb24.
        */
        struct Rgb24
        {
            uint8_t red; /*!< \brief Red channel. First byte of a View::Rgb24 pixel. */
            uint8_t green; /*!< \brief Green channel. Second byte of a View::Rgb24 pixel. */
            uint8_t blue; /*!< \brief Blue channel. Third byte of a View::Rgb24 pixel. Rgb24(255, 0, 0) is red. */

            /*!
                Creates a 24-bit RGB pixel with the same value in every channel.

                Rgb24() is black. The value is written to red, green and blue.

                \param [in] gray - initial value for red, green and blue. It is equal to 0 by default.
            */
            Rgb24(const uint8_t & gray = uint8_t(0));

            /*!
                Creates a 24-bit RGB pixel from red, green and blue.

                Argument order is red, green, blue. TestRandom builds the blue,
                green and red labels with Rgb24(r, g, b) after receiving the
                channels as (b, g, r). A BGR color Bgr24(b, g, r) is the same
                color as Rgb24(r, g, b).

                \param [in] r - initial value for the red channel.
                \param [in] g - initial value for the green channel.
                \param [in] b - initial value for the blue channel.
            */
            Rgb24(const uint8_t & r, const uint8_t & g, const uint8_t & b);

            /*!
                Creates a 24-bit RGB pixel from a 32-bit BGRA pixel.

                Copies red, green and blue by name, so the color is kept and the
                byte order becomes red, green, blue. Alpha is omitted.

                \param [in] p - a 32-bit BGRA pixel.
            */
            Rgb24(const Bgra32 & p);

            /*!
                Creates a 24-bit RGB pixel from a 24-bit BGR pixel.

                Copies red, green and blue by name, so the color is kept and the
                byte order becomes red, green, blue.

                \param [in] p - a 24-bit BGR pixel.
            */
            Rgb24(const Bgr24 & p);

            /*!
                Creates a 24-bit RGB pixel from a 32-bit RGBA pixel.

                Copies red, green and blue. Alpha is omitted.

                \param [in] p - a 32-bit RGBA pixel.
            */
            Rgb24(const Rgba32& p);

            /*!
                Creates a copy of a 24-bit RGB pixel.

                \param [in] p - a 24-bit RGB pixel.
            */
            Rgb24(const Rgb24 & p);

            /*!
                \fn template <template<class> class A> static const Rgb24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a constant reference to the RGB pixel at (col, row).

                Requires view.format == View::Rgb24 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view. Typical code uses view.At<Pixel::Rgb24>(x, y).

                \param [in] view - an image view of 24-bit RGB pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a constant reference to the pixel.
            */
            template <template<class> class A> static const Rgb24 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

            /*!
                \fn template <template<class> class A> static Rgb24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a reference to the RGB pixel at (col, row).

                Requires view.format == View::Rgb24 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view.

                \param [in] view - an image view of 24-bit RGB pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a reference to the pixel.
            */
            template <template<class> class A> static Rgb24 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);
        };

        /*! @ingroup cpp_pixels

            \short 32-bit RGBA pixel.

            Stores red, green, blue, alpha as four consecutive bytes. This is the
            element type of a View::Rgba32 image. Argument order is red, green,
            blue, alpha, which is the reverse of the color channels of Bgra32.
            TestRandom::GetColor calls Rgba32(r, g, b) and leaves alpha at 255,
            so an RGBA canvas receives the same color as a BGRA canvas built with
            Bgra32(b, g, r). Pass Rgba32 to drawing and FillPixel only when the
            canvas format is View::Rgba32.
        */
        struct Rgba32
        {
            uint8_t red; /*!< \brief Red channel. First byte of a View::Rgba32 pixel. */
            uint8_t green; /*!< \brief Green channel. Second byte of a View::Rgba32 pixel. */
            uint8_t blue; /*!< \brief Blue channel. Third byte of a View::Rgba32 pixel. Rgba32(255, 0, 0) is opaque red. */
            uint8_t alpha; /*!< \brief Alpha channel. Fourth byte of a View::Rgba32 pixel. 255 is opaque. */

            /*!
                Creates a 32-bit RGBA pixel with equal red, green and blue.

                Rgba32() is opaque black. gray is written to red, green and blue.
                Alpha is independent and defaults to 255.

                \param [in] gray - initial value for red, green and blue. It is equal to 0 by default.
                \param [in] a - initial value for alpha. It is equal to 255 by default.
            */
            Rgba32(const uint8_t& gray = uint8_t(0), const uint8_t& a = uint8_t(255));

            /*!
                Creates a 32-bit RGBA pixel from red, green, blue and alpha.

                Argument order is red, green, blue, alpha. TestRandom passes
                Rgba32(r, g, b) when the source triple was (b, g, r).

                \param [in] r - initial value for the red channel.
                \param [in] g - initial value for the green channel.
                \param [in] b - initial value for the blue channel.
                \param [in] a - initial value for the alpha channel. It is equal to 255 by default.
            */
            Rgba32(const uint8_t& r, const uint8_t& g, const uint8_t& b, const uint8_t& a = uint8_t(255));

            /*!
                Creates a 32-bit RGBA pixel from a 32-bit BGRA pixel.

                Copies red, green, blue and alpha by name, so the color and alpha
                are kept and the byte order becomes red, green, blue, alpha.

                \param [in] p - a 32-bit BGRA pixel.
            */
            Rgba32(const Bgra32& p);

            /*!
                Creates a 32-bit RGBA pixel from a 24-bit BGR pixel.

                Copies red, green and blue by name. Alpha is 255 unless another
                value is passed.

                \param [in] p - a 24-bit BGR pixel.
                \param [in] a - initial value for the alpha channel. It is equal to 255 by default.
            */
            Rgba32(const Bgr24& p, const uint8_t& a = uint8_t(255));

            /*!
                Creates a 32-bit RGBA pixel from a 24-bit RGB pixel.

                Copies red, green and blue. Alpha is 255 unless another value is passed.

                \param [in] p - a 24-bit RGB pixel.
                \param [in] a - initial value for the alpha channel. It is equal to 255 by default.
            */
            Rgba32(const Rgb24& p, const uint8_t& a = uint8_t(255));

            /*!
                Creates a copy of a 32-bit RGBA pixel.

                \param [in] p - a 32-bit RGBA pixel.
            */
            Rgba32(const Rgba32& p);

            /*!
                \fn template <template<class> class A> static const Rgba32 & At(const View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a constant reference to the RGBA pixel at (col, row).

                Requires view.format == View::Rgba32 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view. Typical code uses view.At<Pixel::Rgba32>(x, y).

                \param [in] view - an image view of 32-bit RGBA pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a constant reference to the pixel.
            */
            template <template<class> class A> static const Rgba32& At(const View<A>& view, ptrdiff_t col, ptrdiff_t row);

            /*!
                \fn template <template<class> class A> static Rgba32 & At(View<A> & view, ptrdiff_t col, ptrdiff_t row);

                Gets a reference to the RGBA pixel at (col, row).

                Requires view.format == View::Rgba32 and then returns Simd::At.
                col is the x coordinate and row is the y coordinate; both must
                lie inside the view.

                \param [in] view - an image view of 32-bit RGBA pixel format.
                \param [in] col - x-coordinate of the pixel.
                \param [in] row - y-coordinate of the pixel.
                \return a reference to the pixel.
            */
            template <template<class> class A> static Rgba32& At(View<A>& view, ptrdiff_t col, ptrdiff_t row);
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

        SIMD_INLINE Bgr24::Bgr24(const Rgb24 & p)
            : blue(p.blue)
            , green(p.green)
            , red(p.red)
        {
        }

        SIMD_INLINE Bgr24::Bgr24(const Rgba32& p)
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

        template <template<class> class A> SIMD_INLINE const Bgr24 & Bgr24::At(const View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Bgr24);

            return Simd::At<A, Bgr24>(view, col, row);
        }

        template <template<class> class A> SIMD_INLINE Bgr24 & Bgr24::At(View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Bgr24);

            return Simd::At<A, Bgr24>(view, col, row);
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

        SIMD_INLINE Bgra32::Bgra32(const Rgb24 & p, const uint8_t & a)
            : blue(p.blue)
            , green(p.green)
            , red(p.red)
            , alpha(a)
        {
        }

        SIMD_INLINE Bgra32::Bgra32(const Rgba32& p)
            : blue(p.blue)
            , green(p.green)
            , red(p.red)
            , alpha(p.alpha)
        {
        }

        SIMD_INLINE Bgra32::Bgra32(const Bgra32 & p)
            : blue(p.blue)
            , green(p.green)
            , red(p.red)
            , alpha(p.alpha)
        {
        }

        template <template<class> class A> SIMD_INLINE const Bgra32 & Bgra32::At(const View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Bgra32);

            return Simd::At<A, Bgra32>(view, col, row);
        }

        template <template<class> class A> SIMD_INLINE Bgra32 & Bgra32::At(View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Bgra32);

            return Simd::At<A, Bgra32>(view, col, row);
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

        template <template<class> class A> SIMD_INLINE const Hsv24 & Hsv24::At(const View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Hsv24);

            return Simd::At<A, Hsv24>(view, col, row);
        }

        template <template<class> class A> SIMD_INLINE Hsv24 & Hsv24::At(View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Hsv24);

            return Simd::At<A, Hsv24>(view, col, row);
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

        template <template<class> class A> SIMD_INLINE const Hsl24 & Hsl24::At(const View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Hsl24);

            return Simd::At<A, Hsl24>(view, col, row);
        }

        template <template<class> class A> SIMD_INLINE Hsl24 & Hsl24::At(View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Hsl24);

            return Simd::At<A, Hsl24>(view, col, row);
        }

        // struct Rgb24 implementation:

        SIMD_INLINE Rgb24::Rgb24(const uint8_t & gray)
            : red(gray)
            , green(gray)
            , blue(gray)
        {
        }

        SIMD_INLINE Rgb24::Rgb24(const uint8_t & r, const uint8_t & g, const uint8_t & b)
            : red(r)
            , green(g)
            , blue(b)
        {
        }

        SIMD_INLINE Rgb24::Rgb24(const Bgra32 & p)
            : red(p.red)
            , green(p.green)
            , blue(p.blue)
        {
        }

        SIMD_INLINE Rgb24::Rgb24(const Bgr24 & p)
            : red(p.red)
            , green(p.green)
            , blue(p.blue)
        {
        }

        SIMD_INLINE Rgb24::Rgb24(const Rgba32& p)
            : red(p.red)
            , green(p.green)
            , blue(p.blue)
        {
        }

        SIMD_INLINE Rgb24::Rgb24(const Rgb24 & p)
            : red(p.red)
            , green(p.green)
            , blue(p.blue)
        {
        }

        template <template<class> class A> SIMD_INLINE const Rgb24 & Rgb24::At(const View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Rgb24);

            return Simd::At<A, Rgb24>(view, col, row);
        }

        template <template<class> class A> SIMD_INLINE Rgb24 & Rgb24::At(View<A> & view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Rgb24);

            return Simd::At<A, Rgb24>(view, col, row);
        }

        // struct Rgba32 implementation:

        SIMD_INLINE Rgba32::Rgba32(const uint8_t& gray, const uint8_t& a)
            : red(gray)
            , green(gray)
            , blue(gray)
            , alpha(a)
        {
        }

        SIMD_INLINE Rgba32::Rgba32(const uint8_t& r, const uint8_t& g, const uint8_t& b, const uint8_t& a)
            : red(r)
            , green(g)
            , blue(b)
            , alpha(a)
        {
        }

        SIMD_INLINE Rgba32::Rgba32(const Bgra32& p)
            : red(p.red)
            , green(p.green)
            , blue(p.blue)
            , alpha(p.alpha)
        {
        }        
        
        SIMD_INLINE Rgba32::Rgba32(const Bgr24& p, const uint8_t& a)
            : red(p.red)
            , green(p.green)
            , blue(p.blue)
            , alpha(a)
        {
        }

        SIMD_INLINE Rgba32::Rgba32(const Rgb24& p, const uint8_t& a)
            : red(p.red)
            , green(p.green)
            , blue(p.blue)
            , alpha(a)
        {
        }

        SIMD_INLINE Rgba32::Rgba32(const Rgba32& p)
            : red(p.red)
            , green(p.green)
            , blue(p.blue)
            , alpha(p.alpha)
        {
        }

        template <template<class> class A> SIMD_INLINE const Rgba32& Rgba32::At(const View<A>& view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Rgba32);

            return Simd::At<A, Rgba32>(view, col, row);
        }

        template <template<class> class A> SIMD_INLINE Rgba32& Rgba32::At(View<A>& view, ptrdiff_t col, ptrdiff_t row)
        {
            assert(view.format == View<A>::Rgba32);

            return Simd::At<A, Rgba32>(view, col, row);
        }
    }
}

#endif//__SimdPixel_hpp__
