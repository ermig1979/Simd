/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2014-2019 Antonenka Mikhail,
*               2019-2019 Artur Voronkov,
*               2022-2022 Souriya Trinh.
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
#ifndef __SimdFrame_hpp__
#define __SimdFrame_hpp__

#include "Simd/SimdLib.hpp"

namespace Simd
{
    /*! @ingroup cpp_frame

        \short The Frame structure provides storage and manipulation of frames (multiplanar images).

        The structure holds one or more Simd::View planes that together form a video frame.
        Packed formats (Gray8, Bgr24, Bgra32, Rgb24, Rgba32, Lab24) use planes[0].
        Nv12 uses two planes (full-size Y and half-size interleaved UV).
        Yuv420p uses three planes (full-size Y and half-size U, V).
        Yuv444p uses three full-size Y, U, V planes.

        A typical usage wraps a packed Simd::View (or an OpenCV cv::Mat through View)
        as a Frame with a timestamp, converts it to YUV, optionally resizes it, and
        converts it back. Simd::Motion::Detector::NextFrame takes Frame as input.
        Packed output is drawn or saved through planes[0].

        Copy constructor and assignment create a reference to the same planes
        (not a deep copy). Use Copy() or Clone() to duplicate pixel data.
        Ref() is used to pass a temporary Frame as a non-const reference, for
        example to Simd::Convert.

        Nv12 and Yuv420p require even width and height. For YUV formats
        yuvType defaults to ::SimdYuvBt601 when it is ::SimdYuvUnknown.
        Packed formats set yuvType to ::SimdYuvUnknown.

        Using example:
        \code
        #include "Simd/SimdFrame.hpp"

        int main()
        {
            typedef Simd::View<Simd::Allocator> View;
            typedef Simd::Frame<Simd::Allocator> Frame;
            typedef Simd::Point<ptrdiff_t> Size;

            View image(320, 240, View::Bgr24);
            Frame input(image, false, 0.040);

            Frame yuv(input.Size(), Frame::Yuv420p);
            Simd::Convert(input, yuv);

            Frame resized;
            Simd::Resize(yuv, resized, Size(160, 120), SimdResizeMethodBilinear);

            Frame rgb(resized.Size(), Frame::Rgb24);
            Simd::Convert(resized, rgb);

            rgb.planes[0].Save("frame.ppm");

            return 0;
        }
        \endcode

        \ref cpp_frame_functions.
    */
    template <template<class> class A>
    struct Frame
    {
        typedef A<uint8_t> Allocator; /*!< Allocator type definition. */

        /*! Maximal count of pixel planes in a frame. Packed formats use 1, Nv12 uses 2, Yuv420p and Yuv444p use 3. */
        static const size_t PLANE_COUNT_MAX = 4;

        /*!
            \enum Format
            Describes pixel format types of a frame.

            PlaneCount() returns how many entries of planes[] are used.
            Nv12 and Yuv420p store chroma at half width and half height and
            require even frame width and height.
        */
        enum Format
        {
            /*! An undefined pixel format. PlaneCount is 0. */
            None = 0,
            /*! Two planes: planes[0] is 8-bit full-size Y (View::Gray8), planes[1] is 16-bit interleaved half-size UV (View::Uv16). Width and height must be even. */
            Nv12,
            /*! Three planes: planes[0] is 8-bit full-size Y, planes[1] and planes[2] are 8-bit half-size U and V (View::Gray8). Width and height must be even. */
            Yuv420p,
            /*! One plane 32-bit (4 8-bit channels) BGRA (Blue, Green, Red, Alpha) pixel format in planes[0]. */
            Bgra32,
            /*! One plane 24-bit (3 8-bit channels) BGR (Blue, Green, Red) pixel format in planes[0]. */
            Bgr24,
            /*! One plane 8-bit gray pixel format in planes[0]. */
            Gray8,
            /*! One plane 24-bit (3 8-bit channels) RGB (Red, Green, Blue) pixel format in planes[0]. */
            Rgb24,
            /*! One plane 32-bit (4 8-bit channels) RGBA (Red, Green, Blue, Alpha) pixel format in planes[0]. */
            Rgba32,
            /*! Three planes: planes[0], planes[1] and planes[2] are 8-bit full-size Y, U, V (View::Gray8). */
            Yuv444p,
            /*! One plane 24-bit (3 8-bit channels) Lab (CIELAB) pixel format in planes[0]. */
            Lab24,
        };

        typedef void (*DeleterPtr)(void* context); /*!< Optional callback called from the destructor with context when the Frame wraps an external buffer. */

        const size_t width; /*!< \brief A width of the frame in pixels (luma / full-frame size). */
        const size_t height; /*!< \brief A height of the frame in pixels (luma / full-frame size). */
        const Format format; /*!< \brief A pixel format of the frame. */
        View<A> planes[PLANE_COUNT_MAX]; /*!< \brief Image planes of the frame. Used entries are [0, PlaneCount()). Packed formats store the image in planes[0]. */
        bool flipped; /*!< \brief A flag of a vertically flipped image (false - point (0, 0) is at the top-left corner, true - point (0, 0) is at the bottom-left corner). Compatible() and Convert() require the same value. */
        double timestamp; /*!< \brief A timestamp of the frame. Typical usage stores time in seconds (for example OpenCV CAP_PROP_POS_MSEC * 0.001). */
        const SimdYuvType yuvType; /*!< \brief A YUV color space of YUV formats. Defaults to ::SimdYuvBt601 for Nv12, Yuv420p and Yuv444p. Packed formats use ::SimdYuvUnknown. */ 

        /*!
            Creates a new empty Frame structure.

            Width and height are 0, format is None, yuvType is ::SimdYuvUnknown,
            flipped is false and timestamp is 0.
        */
        Frame();

        /*!
            Creates a new Frame structure on the base of the other frame.

            \note This constructor does not create a new frame! It only creates a reference to the same planes.
            If you want to create a copy then must use method Simd::Frame::Copy or Simd::Frame::Clone.

            \param [in] frame - an original frame.
        */
        Frame(const Frame & frame);

#ifdef SIMD_CPP_2011_ENABLE
        /*!
            Move constructor of Frame structure.

            Transfers planes and the optional deleter from frame. After the call
            frame is empty.

            \param [in] frame - a moved Frame.
        */
        Frame(Frame&& frame) noexcept;
#endif

        /*!
            Creates a new one-plane Frame structure on the base of the image view.

            Supported view formats are View::Gray8, View::Bgr24, View::Bgra32,
            View::Rgb24, View::Rgba32 and View::Lab24. Other formats assert.
            yuvType is set to ::SimdYuvUnknown.

            Typical usage wraps a video image (or an OpenCV cv::Mat through View)
            together with a timestamp before Simd::Convert or Motion::Detector::NextFrame.

            \note This constructor does not create a new image frame! It only creates a reference to the same image.
            If you want to create a copy then must use method Simd::Frame::Copy or Simd::Frame::Clone.

            \param [in] view - an original image view.
            \param [in] flipped_ - a flag of a vertically flipped image of the created frame. It is equal to false by default.
            \param [in] timestamp_ - a timestamp of the created frame. It is equal to 0 by default.
        */
        Frame(const View<A> & view, bool flipped_ = false, double timestamp_ = 0);

#ifdef SIMD_CPP_2011_ENABLE
        /*!
            Creates a new one-plane Frame structure on the base of the temporary image view.

            Supported view formats are the same as for the const View constructor.
            The view is moved into planes[0].

            \param [in] view - a temporary image view.
            \param [in] flipped_ - a flag of a vertically flipped image of the created frame. It is equal to false by default.
            \param [in] timestamp_ - a timestamp of the created frame. It is equal to 0 by default.
        */
        Frame(View<A>&& view, bool flipped_ = false, double timestamp_ = 0);
#endif

        /*!
            Creates a new Frame structure with specified width, height and pixel format.

            Allocates owned planes for format_. Nv12 and Yuv420p require even width and height.
            For YUV formats yuvType_ equal to ::SimdYuvUnknown is replaced by ::SimdYuvBt601.
            Packed formats set yuvType to ::SimdYuvUnknown.

            \param [in] width_ - a width of created frame.
            \param [in] height_ - a height of created frame.
            \param [in] format_ - a pixel format of created frame.
            \param [in] flipped_ - a flag of a vertically flipped image of the created frame. It is equal to false by default.
            \param [in] timestamp_ - a timestamp of the created frame. It is equal to 0 by default.
            \param [in] yuvType_ - a YUV format type of the created frame. It is equal to ::SimdYuvUnknown by default.
        */
        Frame(size_t width_, size_t height_, Format format_, bool flipped_ = false, double timestamp_ = 0, SimdYuvType yuvType_ = SimdYuvUnknown);

        /*!
            Creates a new Frame structure with specified width, height and pixel format.

            Allocates owned planes. See the width/height constructor for Nv12 / Yuv420p
            even-size and yuvType rules.

            \param [in] size - a size (width and height) of created frame.
            \param [in] format_ - a pixel format of created frame.
            \param [in] flipped_ - a flag of a vertically flipped image of the created frame. It is equal to false by default.
            \param [in] timestamp_ - a timestamp of the created frame. It is equal to 0 by default.
            \param [in] yuvType_ - a YUV format type of the created frame. It is equal to ::SimdYuvUnknown by default.
        */
        Frame(const Point<ptrdiff_t> & size, Format format_, bool flipped_ = false, double timestamp_ = 0, SimdYuvType yuvType_ = SimdYuvUnknown);

        /*!
            Creates a new Frame structure with specified width, height and pixel format around external buffers.

            The frame does not own the buffers. Unused plane pointers (data1, data2)
            are ignored for packed formats. Nv12 uses data0/data1, Yuv420p and Yuv444p
            use data0/data1/data2. If deleter is not NULL it is called from the destructor
            with context.

            \param [in] width_ - a width of created frame.
            \param [in] height_ - a height of created frame.
            \param [in] format_ - a pixel format of created frame.
            \param [in] data0 - a pointer to the pixel data of first image plane.
            \param [in] stride0 - a row size of first image plane.
            \param [in] data1 - a pointer to the pixel data of second image plane.
            \param [in] stride1 - a row size of second image plane.
            \param [in] data2 - a pointer to the pixel data of third image plane.
            \param [in] stride2 - a row size of third image plane.
            \param [in] flipped_ - a flag of a vertically flipped image of the created frame. It is equal to false by default.
            \param [in] timestamp_ - a timestamp of the created frame. It is equal to 0 by default.
            \param [in] yuvType_ - a YUV format type of the created frame. It is equal to ::SimdYuvUnknown by default.
            \param [in] deleter - an optional callback to delete the external buffer after using. It is equal to NULL by default.
            \param [in] context - a context of the callback to delete the external buffer after using. It is equal to NULL by default.
        */
        Frame(size_t width_, size_t height_, Format format_, uint8_t * data0, size_t stride0, uint8_t * data1, size_t stride1, uint8_t * data2, size_t stride2, 
            bool flipped_ = false, double timestamp_ = 0, SimdYuvType yuvType_ = SimdYuvUnknown, DeleterPtr deleter = NULL, void *context = NULL);

        /*!
            A Frame destructor.

            Calls deleter(context) when an external-buffer deleter was set.
        */
        ~Frame();

        /*!
            Gets a copy of current frame.

            Allocates a new Frame on the heap and copies pixel data of all used planes.
            Prefer Copy() when a stack object is enough.

            \return a pointer to the new Frame structure. The user must free this pointer after usage.
        */
        Frame * Clone() const;

        /*!
            Gets a copy of region of current frame which is bounded by the rectangle with specified coordinates.

            The region is taken with Region(rect) and then cloned. For Nv12 and Yuv420p
            the rectangle is aligned to even coordinates.

            \param [in] rect - a rectangle which bounds the region.
            \return - a pointer to the new Frame structure. The user must free this pointer after usage.
        */
        Frame * Clone(const Rectangle<ptrdiff_t>& rect) const;

        /*!
            Gets a copy of current frame using buffer as a storage.

            Grows buffer planes when they are smaller than the current planes.
            The returned Frame is not owner of pixel data.

            \param [in, out] buffer - an external frame as a buffer.
            \return a pointer to the new Frame structure (not owner). The user must free this pointer after usage.
        */
        Frame * Clone(Frame & buffer) const;

        /*!
            Gets a copy of current frame by value.

            The copy has the same width, height, format, flipped, timestamp and yuvType.
            Pixel data of all used planes are copied.

            \return a new Frame structure containing a copy of the frame.
        */
        Frame Copy() const;

        /*!
            Gets a copy of region of current frame bounded by the rectangle with specified coordinates, by value.

            The region is taken with Region(rect) and then copied. For Nv12 and Yuv420p
            the rectangle is aligned to even coordinates.

            \param [in] rect - a rectangle which bounds the region.
            \return a new Frame structure containing a copy of the region.
        */
        Frame Copy(const Rectangle<ptrdiff_t>& rect) const;

        /*!
            Creates reference to other Frame structure.

            \note This function does not create a copy of the frame! It only creates a reference to the same planes.

            \param [in] frame - an original frame.
            \return a reference to itself.
        */
        Frame & operator = (const Frame & frame);

#ifdef SIMD_CPP_2011_ENABLE
        /*!
            Moves Frame structure.

            Clears this frame and then swaps it with frame.

            \param [in] frame - a moved frame.
            \return a reference to itself.
        */
        Frame& operator = (Frame&& frame);
#endif

        /*!
            Creates reference to itself.

            It is used to pass a temporary Frame as a non-const argument, for example:
            \code
            Simd::Convert(input, Frame(grayView).Ref());
            \endcode

            \return a reference to itself.
        */
        Frame & Ref();

        /*!
            Re-creates a Frame structure with specified width, height and pixel format.

            Allocates owned planes for format_. Nv12 and Yuv420p require even width and height.
            For YUV formats yuvType_ equal to ::SimdYuvUnknown is replaced by ::SimdYuvBt601.
            Packed formats set yuvType to ::SimdYuvUnknown. flipped and timestamp are not changed.

            \param [in] width_ - a width of re-created frame.
            \param [in] height_ - a height of re-created frame.
            \param [in] format_ - a pixel format of re-created frame.
            \param [in] yuvType_ - a YUV format type of re-created frame. It is equal to ::SimdYuvUnknown by default.
        */
        void Recreate(size_t width_, size_t height_, Format format_, SimdYuvType yuvType_ = SimdYuvUnknown);

        /*!
            Re-creates a Frame structure with specified width, height and pixel format.

            See Recreate(width, height, format, yuvType) for Nv12 / Yuv420p even-size
            and yuvType rules.

            \param [in] size - a size (width and height) of re-created frame.
            \param [in] format_ - a pixel format of re-created frame.
            \param [in] yuvType_ - a YUV format type of re-created frame. It is equal to ::SimdYuvUnknown by default.
        */
        void Recreate(const Point<ptrdiff_t> & size, Format format_, SimdYuvType yuvType_ = SimdYuvUnknown);

        /*!
            Creates a new Frame structure which points to the region of current frame bounded by the rectangle with specified coordinates.

            The result is a reference to the same pixel data, not a copy. For Nv12 and Yuv420p
            the rectangle is aligned to even coordinates and chroma planes are taken at half size.

            \param [in] left - a left side of the region.
            \param [in] top - a top side of the region.
            \param [in] right - a right side of the region.
            \param [in] bottom - a bottom side of the region.
            \return - a new Frame structure which points to the region of frame.
        */
        Frame Region(const ptrdiff_t & left, const ptrdiff_t & top, const ptrdiff_t & right, const ptrdiff_t & bottom) const;

        /*!
            Creates a new Frame structure which points to the region of current frame bounded by the rectangle with specified coordinates.

            The arguments are clamped to the frame and, for Nv12 and Yuv420p, aligned to even
            coordinates. The actual region is written back.

            \param [in, out] left - a left side of the required region. Returns the left side of the actual region.
            \param [in, out] top - a top side of the required region. Returns the top side of the actual region.
            \param [in, out] right - a right side of the required region. Returns the right side of the actual region.
            \param [in, out] bottom - a bottom side of the required region. Returns the bottom side of the actual region.
            \return - a new Frame structure which points to the region of frame.
        */
        Frame Region(ptrdiff_t & left, ptrdiff_t & top, ptrdiff_t & right, ptrdiff_t & bottom) const;

        /*!
            Creates a new Frame structure which points to the region of frame bounded by the rectangle with specified coordinates.

            The result is a reference to the same pixel data, not a copy.

            \param [in] topLeft - a top-left corner of the region.
            \param [in] bottomRight - a bottom-right corner of the region.
            \return - a new Frame structure which points to the region of frame.
        */
        Frame Region(const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight) const;

        /*!
            Creates a new Frame structure which points to the region of frame bounded by the rectangle with specified coordinates.

            The arguments are clamped (and even-aligned for Nv12 and Yuv420p). The actual
            corners are written back.

            \param [in, out] topLeft - a top-left corner of the required region. Returns the top-left corner of the actual region.
            \param [in, out] bottomRight - a bottom-right corner of the required region. Returns the bottom-right corner of the actual region.
            \return - a new Frame structure which points to the region of frame.
        */
        Frame Region(Point<ptrdiff_t> & topLeft, Point<ptrdiff_t> & bottomRight) const;

        /*!
            Creates a new Frame structure which points to the region of frame bounded by the rectangle with specified coordinates.

            The result is a reference to the same pixel data, not a copy.

            \param [in] rect - a rectangle which bounds the region.
            \return - a new Frame structure which points to the region of frame.
        */
        Frame Region(const Rectangle<ptrdiff_t> & rect) const;

        /*!
            Creates a new Frame structure which points to the region of frame bounded by the rectangle with specified coordinates.

            The rectangle is clamped (and even-aligned for Nv12 and Yuv420p). The actual
            region is written back to rect.

            \param [in, out] rect - a rectangle which bounds the required region. Returns the actual region.
            \return - a new Frame structure which points to the region of frame.
        */
        Frame Region(Rectangle<ptrdiff_t> & rect) const;

        /*!
            Creates a new Frame structure which points to the vertically flipped frame.

            Each used plane is flipped. The flipped flag is toggled. Compatible() and
            Convert() require the same flipped value on both frames.

            \return - a new Frame structure which points to the flipped frame.
        */
        Frame Flipped() const;

        /*!
            Gets size (width and height) of the frame.

            Typical usage passes the result to another Frame constructor or to Simd::Resize.

            \return - a new Point structure with frame width and height.
        */
        Point<ptrdiff_t> Size() const;

        /*!
            Gets size in bytes required to store pixel data of current Frame structure.

            The value is the sum of DataSize() of all planes (including unused ones).

            \return - a size of data pixels in bytes.
        */
        size_t DataSize() const;

        /*!
            Gets area in pixels of current Frame structure.

            The value is width * height (luma / full-frame area).

            \return - an area of current Frame in pixels.
        */
        size_t Area() const;

        /*!
            \fn size_t PlaneCount(Format format);

            Gets number of planes in the frame for the given pixel format.

            None uses 0 planes, packed formats use 1, Nv12 uses 2,
            Yuv420p and Yuv444p use 3.

            \param [in] format - a pixel format.
            \return - a number of planes.
        */
        static size_t PlaneCount(Format format);

        /*!
            Gets number of planes for current frame.

            The value is PlaneCount(format). Used planes are planes[0] .. planes[PlaneCount() - 1].

            \return - a number of planes.
        */
        size_t PlaneCount() const;

        /*!
            Clears Frame structure (resets all public fields).

            Used planes are cleared. Width and height become 0, format becomes None,
            flipped becomes false, timestamp becomes 0, yuvType becomes ::SimdYuvUnknown.
         */
        void Clear();

        /*!
            Swaps content of two (this and other) Frame structures.

            All public fields and the optional deleter are exchanged.

            \param [in] other - an other frame.
        */
        void Swap(Frame& other);

        /*!
            Gets owner flag: do all used planes own their images?

            An empty frame (PlaneCount() == 0) is not an owner.

            \return - an owner flag.
        */
        bool Owner() const;

        /*!
            Captures image planes (copies to internal buffers) if this Frame is not owner of current image planes.

            Calls View::Capture for each used plane. After the call Owner() is true
            when the frame is not empty.
        */
        void Capture();

        private:
            DeleterPtr _deleter;
            void* _context;
    };

    /*! @ingroup cpp_frame_functions

        \fn template <template<class> class A, template<class> class B> bool EqualSize(const Frame<A> & a, const Frame<B> & b);

        Checks two frames on the same size.

        The frames must have the same width and height. Format, flipped and yuvType
        may differ. Convert() requires EqualSize.

        \param [in] a - a first frame.
        \param [in] b - a second frame.
        \return - a result of checking.
    */
    template <template<class> class A, template<class> class B> bool EqualSize(const Frame<A> & a, const Frame<B> & b);

    /*! @ingroup cpp_frame_functions

        \fn template <template<class> class A, template<class> class B> bool Compatible(const Frame<A> & a, const Frame<B> & b);

        Checks two frames on compatibility.

        The frames must have the same width, height, pixel format, flipped flag
        and yuvType. Copy() requires Compatible.

        \param [in] a - a first frame.
        \param [in] b - a second frame.
        \return - a result of checking.
    */
    template <template<class> class A, template<class> class B> bool Compatible(const Frame<A> & a, const Frame<B> & b);

    /*! @ingroup cpp_frame_functions

        \fn template <template<class> class A, template<class> class B> void Copy(const Frame<A> & src, Frame<B> & dst);

        \short Copies one frame to another frame.

        The frames must be Compatible (the same width, height, format, flipped and
        yuvType). Pixel data of all used planes are copied. Timestamp of dst is
        not changed.

        \param [in] src - an input frame.
        \param [out] dst - an output frame.
    */
    template <template<class> class A, template<class> class B> void Copy(const Frame<A> & src, Frame<B> & dst);

    /*! @ingroup cpp_frame_functions

        \fn template <template<class> class A> void Convert(const Frame<A> & src, Frame<A> & dst);

        \short Converts one frame to another frame.

        The frames must have the same width, height and flipped flag. Both formats
        must be defined (not None). The same format does Copy. YUV-to-YUV conversion
        requires the same yuvType. Typical usage converts packed Bgr24 to Yuv420p
        and back.

        Timestamp of dst is not changed.

        \param [in] src - an input frame.
        \param [out] dst - an output frame.
    */
    template <template<class> class A> void Convert(const Frame<A> & src, Frame<A> & dst);

    /*! @ingroup cpp_frame_functions

        \fn void Resize(const Frame<A> & src, Frame<A> & dst, ::SimdResizeMethodType method = ::SimdResizeMethodBilinear)

        \short Performs resizing of frame.

        The frames must have the same format (not None). Equal size does Copy.
        Nv12 and Yuv420p resize chroma planes at half resolution. Yuv444p resizes
        all three planes with the same resizer. Timestamp of dst is not changed.

        \param [in] src - an original input frame.
        \param [out] dst - a resized output frame.
        \param [in] method - a resizing method. By default it is equal to ::SimdResizeMethodBilinear.
    */
    template<template<class> class A> SIMD_INLINE void Resize(const Frame<A>& src, Frame<A>& dst, ::SimdResizeMethodType method = ::SimdResizeMethodBilinear);

    /*! @ingroup cpp_frame_functions

        \fn void Resize(const Frame<A> & src, Frame<A> & dst, const Point<ptrdiff_t> & size, ::SimdResizeMethodType method = ::SimdResizeMethodBilinear)

        \short Performs resizing of frame.

        Recreates dst when its size differs from size (format is taken from src).
        The input frame can be the output (in-place resize uses a temporary Frame).

        \param [in] src - an original input frame.
        \param [out] dst - a resized output frame. The input frame can be the output.
        \param [in] size - a size of output frame.
        \param [in] method - a resizing method. By default it is equal to ::SimdResizeMethodBilinear.
    */
    template<template<class> class A> SIMD_INLINE void Resize(const Frame<A>& src, Frame<A>& dst, const Point<ptrdiff_t>& size, ::SimdResizeMethodType method = ::SimdResizeMethodBilinear);

    //-------------------------------------------------------------------------------------------------

    // struct Frame implementation:

    template <template<class> class A> SIMD_INLINE Frame<A>::Frame()
        : width(0)
        , height(0)
        , format(None)
        , flipped(false)
        , timestamp(0)
        , yuvType(SimdYuvUnknown)
        , _deleter(NULL)
        , _context(NULL)
    {
    }

    template <template<class> class A> SIMD_INLINE Frame<A>::Frame(const Frame & frame)
        : width(frame.width)
        , height(frame.height)
        , format(frame.format)
        , flipped(frame.flipped)
        , timestamp(frame.timestamp)
        , yuvType(frame.yuvType)
        , _deleter(NULL)
        , _context(NULL)
    {
        for (size_t i = 0, n = PlaneCount(); i < n; ++i)
            planes[i] = frame.planes[i];
    }

#ifdef SIMD_CPP_2011_ENABLE
    template <template<class> class A> SIMD_INLINE Frame<A>::Frame(Frame && frame) noexcept
        : width(0)
        , height(0)
        , format(None)
        , flipped(false)
        , timestamp(0)
        , yuvType(SimdYuvUnknown)
        , _deleter(NULL)
        , _context(NULL)
    {
        Swap(frame);
    }
#endif

    template <template<class> class A> SIMD_INLINE Frame<A>::Frame(const View<A> & view, bool flipped_, double timestamp_)
        : width(view.width)
        , height(view.height)
        , format(None)
        , flipped(flipped_)
        , timestamp(timestamp_)
        , yuvType(SimdYuvUnknown)
        , _deleter(NULL)
        , _context(NULL)
    {
        switch (view.format)
        {
        case View<A>::Gray8: (Format&)format = Gray8; break;
        case View<A>::Bgr24: (Format&)format = Bgr24; break;
        case View<A>::Bgra32: (Format&)format = Bgra32; break;
        case View<A>::Rgb24: (Format&)format = Rgb24; break;
        case View<A>::Rgba32: (Format&)format = Rgba32; break;
        case View<A>::Lab24: (Format&)format = Lab24; break;
        default:
            assert(0);
        }
        planes[0] = view;
    }

#ifdef SIMD_CPP_2011_ENABLE
    template <template<class> class A> SIMD_INLINE Frame<A>::Frame(View<A>&& view, bool flipped_, double timestamp_)
        : width(view.width)
        , height(view.height)
        , format(None)
        , flipped(flipped_)
        , timestamp(timestamp_)
        , yuvType(SimdYuvUnknown)
        , _deleter(NULL)
        , _context(NULL)
    {
        switch (view.format)
        {
        case View<A>::Gray8: (Format&)format = Gray8; break;
        case View<A>::Bgr24: (Format&)format = Bgr24; break;
        case View<A>::Bgra32: (Format&)format = Bgra32; break;
        case View<A>::Rgb24: (Format&)format = Rgb24; break;
        case View<A>::Rgba32: (Format&)format = Rgba32; break;
        case View<A>::Lab24: (Format&)format = Lab24; break;
        default:
            assert(0);
        }
        planes[0] = std::move(view);
    }
#endif

    template <template<class> class A> SIMD_INLINE Frame<A>::Frame(size_t width_, size_t height_, Format format_, bool flipped_, double timestamp_, SimdYuvType yuvType_)
        : width(0)
        , height(0)
        , format(None)
        , flipped(flipped_)
        , timestamp(timestamp_)
        , yuvType(SimdYuvUnknown)
        , _deleter(NULL)
        , _context(NULL)
    {
        Recreate(width_, height_, format_, yuvType_);
    }

    template <template<class> class A> SIMD_INLINE Frame<A>::Frame(const Point<ptrdiff_t> & size, Format format_, bool flipped_, double timestamp_, SimdYuvType yuvType_)
        : width(0)
        , height(0)
        , format(None)
        , flipped(flipped_)
        , timestamp(timestamp_)
        , yuvType(SimdYuvUnknown)
        , _deleter(NULL)
        , _context(NULL)
    {
        Recreate(size, format_, yuvType_);
    }

    template <template<class> class A> SIMD_INLINE Frame<A>::Frame(size_t width_, size_t height_, Format format_, uint8_t * data0, size_t stride0,
        uint8_t * data1, size_t stride1, uint8_t * data2, size_t stride2, bool flipped_, double timestamp_, SimdYuvType yuvType_, DeleterPtr deleter, void* context)
        : width(width_)
        , height(height_)
        , format(format_)
        , flipped(flipped_)
        , timestamp(timestamp_)
        , yuvType(yuvType_)
        , _deleter(deleter)
        , _context(context)
    {
        switch (format)
        {
        case None:
            break;
        case Nv12:
            assert((width & 1) == 0 && (height & 1) == 0);
            planes[0] = View<A>(width, height, stride0, View<A>::Gray8, data0);
            planes[1] = View<A>(width / 2, height / 2, stride1, View<A>::Uv16, data1);
            if(yuvType == SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvBt601;
            break;
        case Yuv420p:
            assert((width & 1) == 0 && (height & 1) == 0);
            planes[0] = View<A>(width, height, stride0, View<A>::Gray8, data0);
            planes[1] = View<A>(width / 2, height / 2, stride1, View<A>::Gray8, data1);
            planes[2] = View<A>(width / 2, height / 2, stride2, View<A>::Gray8, data2);
            if (yuvType == SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvBt601;
            break;
        case Bgra32:
            planes[0] = View<A>(width, height, stride0, View<A>::Bgra32, data0);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        case Bgr24:
            planes[0] = View<A>(width, height, stride0, View<A>::Bgr24, data0);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        case Gray8:
            planes[0] = View<A>(width, height, stride0, View<A>::Gray8, data0);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        case Rgb24:
            planes[0] = View<A>(width, height, stride0, View<A>::Rgb24, data0);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        case Rgba32:
            planes[0] = View<A>(width, height, stride0, View<A>::Rgba32, data0);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        case Yuv444p:
            planes[0] = View<A>(width, height, stride0, View<A>::Gray8, data0);
            planes[1] = View<A>(width, height, stride1, View<A>::Gray8, data1);
            planes[2] = View<A>(width, height, stride2, View<A>::Gray8, data2);
            if (yuvType == SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvBt601;
            break;
        case Lab24:
            planes[0] = View<A>(width, height, stride0, View<A>::Lab24, data0);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        default:
            assert(0);
        }
    }

    template <template<class> class A> SIMD_INLINE Frame<A>::~Frame()
    {
        if (_deleter)
            _deleter(_context);
    }

    template <template<class> class A> SIMD_INLINE Frame<A> * Frame<A>::Clone() const
    {
        Frame<A> * clone = new Frame<A>(width, height, format, flipped, timestamp, yuvType);
        Simd::Copy(*this, *clone);
        return clone;
    }

    template <template<class> class A> SIMD_INLINE Frame<A>* Frame<A>::Clone(const Rectangle<ptrdiff_t>& rect) const
    {
        return Region(rect).Clone();
    }

    /*! \cond */
    template <template<class> class A> SIMD_INLINE Frame<A> * Frame<A>::Clone(Frame<A> & buffer) const
    {
        for (size_t i = 0; i < PlaneCount(); ++i)
        {
            if (buffer.planes[i].width < planes[i].width || buffer.planes[i].height < planes[i].height)
                buffer.planes[i].Recreate(planes[i].Size(), planes[i].format);
        }
        Frame<A> * clone = new Frame<A>(width, height, format,
                                        buffer.planes[0].data, buffer.planes[0].stride,
                                        buffer.planes[1].data, buffer.planes[1].stride,
                                        buffer.planes[2].data, buffer.planes[2].stride,
                                        flipped, timestamp, yuvType);
        Simd::Copy(*this, *clone);
        return clone;
    }

    template <template<class> class A> SIMD_INLINE Frame<A> Frame<A>::Copy() const
    {
        Frame<A> copy(width, height, format, flipped, timestamp, yuvType);
        Simd::Copy(*this, copy);
        return copy;
    }

    template <template<class> class A> SIMD_INLINE Frame<A> Frame<A>::Copy(const Rectangle<ptrdiff_t>& rect) const
    {
        return Region(rect).Copy();
    }

    template <template<class> class A> SIMD_INLINE Frame<A> & Frame<A>::operator = (const Frame<A> & frame)
    {
        if (this != &frame)
        {
            *(size_t*)&width = frame.width;
            *(size_t*)&height = frame.height;
            *(Format*)&format = frame.format;
            flipped = frame.flipped;
            timestamp = frame.timestamp;
            *(SimdYuvType*)&yuvType = frame.yuvType;
            for (size_t i = 0, n = PlaneCount(); i < n; ++i)
                planes[i] = frame.planes[i];
        }
        return *this;
    }

#ifdef SIMD_CPP_2011_ENABLE
    template <template<class> class A> SIMD_INLINE Frame<A>& Frame<A>::operator = (Frame<A>&& frame)
    {
        if (this != &frame)
        {
            Clear();
            Swap(frame);
        }
        return *this;
    }
#endif
    /*! \endcond */

    template <template<class> class A> SIMD_INLINE Frame<A> & Frame<A>::Ref()
    {
        return *this;
    }

    template <template<class> class A> SIMD_INLINE void Frame<A>::Recreate(size_t width_, size_t height_, Format format_, SimdYuvType yuvType_)
    {
        *(size_t*)&width = width_;
        *(size_t*)&height = height_;
        *(Format*)&format = format_;
        *(SimdYuvType*)&yuvType = yuvType_;

        for (size_t i = 0; i < PLANE_COUNT_MAX; ++i)
            planes[i].Recreate(0, 0, View<A>::None);

        switch (format)
        {
        case None:
            break;
        case Nv12:
            assert((width & 1) == 0 && (height & 1) == 0);
            planes[0].Recreate(width, height, View<A>::Gray8);
            planes[1].Recreate(width / 2, height / 2, View<A>::Uv16);
            if (yuvType == SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvBt601;
            break;
        case Yuv420p:
            assert((width & 1) == 0 && (height & 1) == 0);
            planes[0].Recreate(width, height, View<A>::Gray8);
            planes[1].Recreate(width / 2, height / 2, View<A>::Gray8);
            planes[2].Recreate(width / 2, height / 2, View<A>::Gray8);
            if (yuvType == SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvBt601;
            break;
        case Bgra32:
            planes[0].Recreate(width, height, View<A>::Bgra32);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        case Bgr24:
            planes[0].Recreate(width, height, View<A>::Bgr24);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        case Gray8:
            planes[0].Recreate(width, height, View<A>::Gray8);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        case Rgb24:
            planes[0].Recreate(width, height, View<A>::Rgb24);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        case Rgba32:
            planes[0].Recreate(width, height, View<A>::Rgba32);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        case Yuv444p:
            planes[0].Recreate(width, height, View<A>::Gray8);
            planes[1].Recreate(width, height, View<A>::Gray8);
            planes[2].Recreate(width, height, View<A>::Gray8);
            if (yuvType == SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvBt601;
            break;
        case Lab24:
            planes[0].Recreate(width, height, View<A>::Lab24);
            if (yuvType != SimdYuvUnknown)
                *(SimdYuvType*)&yuvType = SimdYuvUnknown;
            break;
        default:
            assert(0);
        }
    }

    template <template<class> class A> SIMD_INLINE void Frame<A>::Recreate(const Point<ptrdiff_t> & size, Format format_, SimdYuvType yuvType_)
    {
        Recreate(size.x, size.y, format_, yuvType_);
    }

    template <template<class> class A> SIMD_INLINE Frame<A> Frame<A>::Region(const ptrdiff_t & left, const ptrdiff_t & top, const ptrdiff_t & right, const ptrdiff_t & bottom) const
    {
        Rectangle<ptrdiff_t> rect(left, top, right, bottom);
        return Region(rect.left, rect.top, rect.right, rect.bottom);
    }

    template <template<class> class A> SIMD_INLINE Frame<A> Frame<A>::Region(ptrdiff_t & left, ptrdiff_t & top, ptrdiff_t & right, ptrdiff_t & bottom) const
    {
        if (format != None && right >= left && bottom >= top)
        {
            left = std::min<ptrdiff_t>(std::max<ptrdiff_t>(left, 0), width);
            top = std::min<ptrdiff_t>(std::max<ptrdiff_t>(top, 0), height);
            right = std::min<ptrdiff_t>(std::max<ptrdiff_t>(right, 0), width);
            bottom = std::min<ptrdiff_t>(std::max<ptrdiff_t>(bottom, 0), height);

            if (format == Nv12 || format == Yuv420p)
            {
                left = left & ~1;
                top = top & ~1;
                right = (right + 1) & ~1;
                bottom = (bottom + 1) & ~1;
            }

            Frame frame;
            *(size_t*)&frame.width = right - left;
            *(size_t*)&frame.height = bottom - top;
            *(Format*)&frame.format = format;
            frame.flipped = flipped;
            frame.timestamp = timestamp;
            *(SimdYuvType*)&frame.yuvType = yuvType;

            frame.planes[0] = planes[0].Region(left, top, right, bottom);

            if (format == Nv12 || format == Yuv420p)
                frame.planes[1] = planes[1].Region(left / 2, top / 2, right / 2, bottom / 2);

            if (format == Yuv420p)
                frame.planes[2] = planes[2].Region(left / 2, top / 2, right / 2, bottom / 2);

            if (format == Yuv444p)
            {
                frame.planes[1] = planes[1].Region(left, top, right, bottom);
                frame.planes[2] = planes[2].Region(left, top, right, bottom);
            }

            return frame;
        }
        else
            return Frame<A>();
    }

    template <template<class> class A> SIMD_INLINE Frame<A> Frame<A>::Region(const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight) const
    {
        return Region(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    }

    template <template<class> class A> SIMD_INLINE Frame<A> Frame<A>::Region(Point<ptrdiff_t> & topLeft, Point<ptrdiff_t> & bottomRight) const
    {
        return Region(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    }

    template <template<class> class A> SIMD_INLINE Frame<A> Frame<A>::Region(const Rectangle<ptrdiff_t> & rect) const
    {
        return Region(rect.left, rect.top, rect.right, rect.bottom);
    }

    template <template<class> class A> SIMD_INLINE Frame<A> Frame<A>::Region(Rectangle<ptrdiff_t> & rect) const
    {
        return Region(rect.left, rect.top, rect.right, rect.bottom);
    }

    template <template<class> class A> SIMD_INLINE Frame<A> Frame<A>::Flipped() const
    {
        Frame frame;
        *(size_t*)&frame.width = width;
        *(size_t*)&frame.height = height;
        *(Format*)&frame.format = format;
        frame.timestamp = timestamp;
        frame.flipped = !flipped;
        *(SimdYuvType*)&frame.yuvType = yuvType;
        for (size_t i = 0, n = PlaneCount(); i < n; ++i)
            frame.planes[i] = planes[i].Flipped();
        return frame;
    }

    template <template<class> class A> SIMD_INLINE Point<ptrdiff_t> Frame<A>::Size() const
    {
        return Point<ptrdiff_t>(width, height);
    }

    template <template<class> class A> SIMD_INLINE size_t Frame<A>::DataSize() const
    {
        size_t size = 0;
        for (size_t i = 0; i < PLANE_COUNT_MAX; ++i)
            size += planes[i].DataSize();
        return size;
    }

    template <template<class> class A> SIMD_INLINE size_t Frame<A>::Area() const
    {
        return width*height;
    }

    template <template<class> class A> SIMD_INLINE size_t Frame<A>::PlaneCount(Format format)
    {
        switch (format)
        {
        case None:    return 0;
        case Nv12:    return 2;
        case Yuv420p: return 3;
        case Bgra32:  return 1;
        case Bgr24:   return 1;
        case Gray8:   return 1;
        case Rgb24:   return 1;
        case Rgba32:  return 1;
        case Yuv444p: return 3;
        case Lab24:   return 1;
        default: assert(0); return 0;
        }
    }

    template <template<class> class A> SIMD_INLINE size_t Frame<A>::PlaneCount() const
    {
        return PlaneCount(format);
    }

    template <template<class> class A> SIMD_INLINE void Frame<A>::Clear()
    {
        for (size_t i = 0, n = PlaneCount(); i < n; ++i)
            planes[i].Clear();
        *(size_t*)&width = 0;
        *(size_t*)&height = 0;
        *(Format*)&format = None;
        flipped = false;
        timestamp = 0;
        *(SimdYuvType*)&yuvType = SimdYuvUnknown;
    }

    template <template<class> class A> SIMD_INLINE void Frame<A>::Swap(Frame<A>& other)
    {
        for (size_t i = 0; i < PLANE_COUNT_MAX; ++i)
            planes[i].Swap(other.planes[i]);
        std::swap((size_t&)width, (size_t&)other.width);
        std::swap((size_t&)height, (size_t&)other.height);
        std::swap((Format&)format, (Format&)other.format);
        std::swap(flipped, other.flipped);
        std::swap(timestamp, other.timestamp);
        std::swap((SimdYuvType&)yuvType, (SimdYuvType&)other.yuvType);
        std::swap(_deleter, other._deleter);
        std::swap(_context, other._context);
    }

    template <template<class> class A> SIMD_INLINE bool Frame<A>::Owner() const
    {
        size_t n = PlaneCount();
        bool owner = n > 0;
        for (size_t i = 0; i < n; ++i)
            owner = owner && planes[i].Owner();
        return owner;
    }

    template <template<class> class A> SIMD_INLINE void Frame<A>::Capture()
    {
        for (size_t i = 0, n = PlaneCount(); i < n; ++i)
            planes[i].Capture();
    }

    //-------------------------------------------------------------------------------------------------
    // View utilities implementation:

    template <template<class> class A, template<class> class B> SIMD_INLINE bool EqualSize(const Frame<A> & a, const Frame<B> & b)
    {
        return
            (a.width == b.width && a.height == b.height);
    }

    template <template<class> class A, template<class> class B> SIMD_INLINE bool Compatible(const Frame<A> & a, const Frame<B> & b)
    {
        typedef typename Frame<A>::Format Format;

        return
            (a.width == b.width && a.height == b.height && a.format == (Format)b.format && a.flipped == b.flipped && a.yuvType == b.yuvType);
    }

    template <template<class> class A, template<class> class B> SIMD_INLINE void Copy(const Frame<A> & src, Frame<B> & dst)
    {
        assert(Compatible(src, dst));

        if (src.format)
        {
            for (size_t i = 0, n = src.PlaneCount(); i < n; ++i)
                Simd::Copy(src.planes[i], dst.planes[i]);
        }
    }

    template <template<class> class A> SIMD_INLINE void Convert(const Frame<A> & src, Frame<A> & dst)
    {
        assert(EqualSize(src, dst) && src.format && dst.format && src.flipped == dst.flipped);

        if (src.format == dst.format)
        {
            Copy(src, dst);
            return;
        }

        switch (src.format)
        {
        case Frame<A>::Nv12:
            switch (dst.format)
            {
            case Frame<A>::Yuv420p:
                assert(src.yuvType == dst.yuvType);
                Copy(src.planes[0], dst.planes[0]);
                DeinterleaveUv(src.planes[1], dst.planes[1], dst.planes[2]);
                break;
            case Frame<A>::Bgra32:
            {
                View<A> u(src.Size(), View<A>::Gray8), v(src.Size(), View<A>::Gray8);
                DeinterleaveUv(src.planes[1], u, v);
                Yuv420pToBgra(src.planes[0], u, v, dst.planes[0], src.yuvType);
                break;
            }
            case Frame<A>::Bgr24:
            {
                View<A> u(src.Size(), View<A>::Gray8), v(src.Size(), View<A>::Gray8);
                DeinterleaveUv(src.planes[1], u, v);
                Yuv420pToBgr(src.planes[0], u, v, dst.planes[0], src.yuvType);
                break;
            }
            case Frame<A>::Gray8:
                if (src.yuvType == SimdYuvTrect871)
                    Copy(src.planes[0], dst.planes[0]);
                else
                    YToGray(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgb24:
            {
                View<A> u(src.Size(), View<A>::Gray8), v(src.Size(), View<A>::Gray8);
                DeinterleaveUv(src.planes[1], u, v);
                Yuv420pToRgb(src.planes[0], u, v, dst.planes[0], src.yuvType);
                break;
            }
            case Frame<A>::Rgba32:
            {
                View<A> u(src.Size(), View<A>::Gray8), v(src.Size(), View<A>::Gray8);
                DeinterleaveUv(src.planes[1], u, v);
                View<A> bgr(src.Size(), View<A>::Bgr24);
                Yuv420pToBgr(src.planes[0], u, v, bgr, src.yuvType);
                BgrToRgba(bgr, dst.planes[0]);
                break;
            }
            case Frame<A>::Yuv444p:
            {
                assert(src.yuvType == dst.yuvType);
                Copy(src.planes[0], dst.planes[0]);
                View<A> u(src.Size(), View<A>::Gray8), v(src.Size(), View<A>::Gray8);
                DeinterleaveUv(src.planes[1], u, v);
                Simd::StretchGray2x2(u, dst.planes[1]);
                Simd::StretchGray2x2(v, dst.planes[2]);
                break;
            }
            case Frame<A>::Lab24:
            {
                View<A> u(src.Size(), View<A>::Gray8), v(src.Size(), View<A>::Gray8);
                DeinterleaveUv(src.planes[1], u, v);
                View<A> bgr(src.Size(), View<A>::Bgr24);
                Yuv420pToBgr(src.planes[0], u, v, bgr, src.yuvType);
                BgrToLab(bgr, dst.planes[0]);
                break;
            }
            default:
                assert(0);
            }
            break;

        case Frame<A>::Yuv420p:
            switch (dst.format)
            {
            case Frame<A>::Nv12:
                assert(src.yuvType == dst.yuvType);
                Copy(src.planes[0], dst.planes[0]);
                InterleaveUv(src.planes[1], src.planes[2], dst.planes[1]);
                break;
            case Frame<A>::Bgra32:
                Yuv420pToBgra(src.planes[0], src.planes[1], src.planes[2], dst.planes[0], src.yuvType);
                break;
            case Frame<A>::Bgr24:
                Yuv420pToBgr(src.planes[0], src.planes[1], src.planes[2], dst.planes[0], src.yuvType);
                break;
            case Frame<A>::Gray8:
                if (src.yuvType == SimdYuvTrect871)
                    Copy(src.planes[0], dst.planes[0]);
                else
                    YToGray(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgb24:
                Yuv420pToRgb(src.planes[0], src.planes[1], src.planes[2], dst.planes[0], src.yuvType);
                break;
            case Frame<A>::Rgba32:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                Yuv420pToBgr(src.planes[0], src.planes[1], src.planes[2], bgr, src.yuvType);
                BgrToRgba(bgr, dst.planes[0]);
                break;
            }
            case Frame<A>::Yuv444p:
            {
                assert(src.yuvType == dst.yuvType);
                Copy(src.planes[0], dst.planes[0]);
                Simd::StretchGray2x2(src.planes[0], dst.planes[1]);
                Simd::StretchGray2x2(src.planes[1], dst.planes[2]);
                break;
            }
            case Frame<A>::Lab24:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                Yuv420pToBgr(src.planes[0], src.planes[1], src.planes[2], bgr, src.yuvType);
                BgrToLab(bgr, dst.planes[0]);
                break;
            }
            default:
                assert(0);
            }
            break;

        case Frame<A>::Bgra32:
            switch (dst.format)
            {
            case Frame<A>::Nv12:
            {
                View<A> u(src.Size(), View<A>::Gray8), v(src.Size(), View<A>::Gray8);
                BgraToYuv420p(src.planes[0], dst.planes[0], u, v, dst.yuvType);
                InterleaveUv(u, v, dst.planes[1]);
                break;
            }
            case Frame<A>::Yuv420p:
                BgraToYuv420p(src.planes[0], dst.planes[0], dst.planes[1], dst.planes[2], dst.yuvType);
                break;
            case Frame<A>::Bgr24:
                BgraToBgr(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Gray8:
                BgraToGray(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgb24:
                BgraToRgb(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgba32:
                BgraToRgba(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Yuv444p:
                BgraToYuv444p(src.planes[0], dst.planes[0], dst.planes[1], dst.planes[2], dst.yuvType);
                break;
            case Frame<A>::Lab24:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                BgraToBgr(src.planes[0], bgr);
                BgrToLab(bgr, dst.planes[0]);
                break;
            }
            default:
                assert(0);
            }
            break;

        case Frame<A>::Bgr24:
            switch (dst.format)
            {
            case Frame<A>::Nv12:
            {
                View<A> u(src.Size(), View<A>::Gray8), v(src.Size(), View<A>::Gray8);
                BgrToYuv420p(src.planes[0], dst.planes[0], u, v, dst.yuvType);
                InterleaveUv(u, v, dst.planes[1]);
                break;
            }
            case Frame<A>::Yuv420p:
                BgrToYuv420p(src.planes[0], dst.planes[0], dst.planes[1], dst.planes[2], dst.yuvType);
                break;
            case Frame<A>::Bgra32:
                BgrToBgra(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Gray8:
                BgrToGray(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgb24:
                BgrToRgb(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgba32:
                BgrToRgba(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Yuv444p:
                BgrToYuv444p(src.planes[0], dst.planes[0], dst.planes[1], dst.planes[2], dst.yuvType);
                break;
            case Frame<A>::Lab24:
                BgrToLab(src.planes[0], dst.planes[0]);
                break;
            default:
                assert(0);
            }
            break;

        case Frame<A>::Gray8:
            switch (dst.format)
            {
            case Frame<A>::Nv12:
                if (dst.yuvType == SimdYuvTrect871)
                    Copy(src.planes[0], dst.planes[0]);
                else
                    GrayToY(src.planes[0], dst.planes[0]);
                Fill(dst.planes[1], 128);
                break;
            case Frame<A>::Yuv420p:
            case Frame<A>::Yuv444p:
                if (dst.yuvType == SimdYuvTrect871)
                    Copy(src.planes[0], dst.planes[0]);
                else
                    GrayToY(src.planes[0], dst.planes[0]);
                Fill(dst.planes[1], 128);
                Fill(dst.planes[2], 128);
                break;
            case Frame<A>::Bgra32:
                GrayToBgra(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Bgr24:
                GrayToBgr(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgb24:
                GrayToRgb(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgba32:
                GrayToRgba(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Lab24:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                GrayToBgr(src.planes[0], bgr);
                BgrToLab(bgr, dst.planes[0]);
                break;
            }
            default:
                assert(0);
            }
            break;

        case Frame<A>::Rgb24:
            switch (dst.format)
            {
            case Frame<A>::Nv12:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                RgbToBgr(src.planes[0], bgr);
                View<A> u(src.Size(), View<A>::Gray8), v(src.Size(), View<A>::Gray8);
                BgrToYuv420p(bgr, dst.planes[0], u, v, dst.yuvType);
                InterleaveUv(u, v, dst.planes[1]);
                break;
            }
            case Frame<A>::Yuv420p:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                RgbToBgr(src.planes[0], bgr);
                BgrToYuv420p(bgr, dst.planes[0], dst.planes[1], dst.planes[2], dst.yuvType);
                break;
            }
            case Frame<A>::Bgra32:
                RgbToBgra(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Gray8:
                RgbToGray(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Bgr24:
                RgbToBgr(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgba32:
                RgbToRgba(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Yuv444p:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                RgbToBgr(src.planes[0], bgr);
                BgrToYuv444p(bgr, dst.planes[0], dst.planes[1], dst.planes[2], dst.yuvType);
                break;
            }
            case Frame<A>::Lab24:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                RgbToBgr(src.planes[0], bgr);
                BgrToLab(bgr, dst.planes[0]);
                break;
            }
            default:
                assert(0);
            }
            break;

        case Frame<A>::Rgba32:
            switch (dst.format)
            {
            case Frame<A>::Nv12:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                RgbaToBgr(src.planes[0], bgr);
                View<A> u(src.Size(), View<A>::Gray8), v(src.Size(), View<A>::Gray8);
                BgrToYuv420p(bgr, dst.planes[0], u, v, dst.yuvType);
                InterleaveUv(u, v, dst.planes[1]);
                break;
            }
            case Frame<A>::Yuv420p:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                RgbaToBgr(src.planes[0], bgr);
                BgrToYuv420p(bgr, dst.planes[0], dst.planes[1], dst.planes[2], dst.yuvType);
                break;
            }
            case Frame<A>::Bgra32:
                RgbaToBgra(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Gray8:
                RgbaToGray(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Bgr24:
                RgbaToBgr(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgb24:
                RgbaToRgb(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Yuv444p:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                RgbaToBgr(src.planes[0], bgr);
                BgrToYuv444p(bgr, dst.planes[0], dst.planes[1], dst.planes[2], dst.yuvType);
                break;
            }
            case Frame<A>::Lab24:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                RgbaToBgr(src.planes[0], bgr);
                BgrToLab(bgr, dst.planes[0]);
                break;
            }
            default:
                assert(0);
            }
            break;

        case Frame<A>::Yuv444p:
            switch (dst.format)
            {
            case Frame<A>::Nv12:
            {
                assert(src.yuvType == dst.yuvType);
                Copy(src.planes[0], dst.planes[0]);
                View<A> u(src.Size() / 2, View<A>::Gray8), v(src.Size() / 2, View<A>::Gray8);
                Simd::ReduceGray2x2(src.planes[0], u);
                Simd::ReduceGray2x2(src.planes[1], v);
                InterleaveUv(u, v, dst.planes[1]);
                break;
            }
            case Frame<A>::Yuv420p:
            {
                assert(src.yuvType == dst.yuvType);
                Copy(src.planes[0], dst.planes[0]);
                Simd::ReduceGray2x2(src.planes[0], dst.planes[1]);
                Simd::ReduceGray2x2(src.planes[1], dst.planes[2]);
                break;
            }            
            case Frame<A>::Bgra32:
                Yuv444pToBgra(src.planes[0], src.planes[1], src.planes[2], dst.planes[0], src.yuvType);
                break;
            case Frame<A>::Bgr24:
                Yuv444pToBgr(src.planes[0], src.planes[1], src.planes[2], dst.planes[0], src.yuvType);
                break;
            case Frame<A>::Gray8:
                if (src.yuvType == SimdYuvTrect871)
                    Copy(src.planes[0], dst.planes[0]);
                else
                    YToGray(src.planes[0], dst.planes[0]);
                break;
            case Frame<A>::Rgb24:
                Yuv444pToRgb(src.planes[0], src.planes[1], src.planes[2], dst.planes[0], src.yuvType);
                break;
            case Frame<A>::Rgba32:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                Yuv444pToBgr(src.planes[0], src.planes[1], src.planes[2], bgr, src.yuvType);
                BgrToRgba(bgr, dst.planes[0]);
                break;
            }
            case Frame<A>::Lab24:
            {
                View<A> bgr(src.Size(), View<A>::Bgr24);
                Yuv444pToBgr(src.planes[0], src.planes[1], src.planes[2], bgr, src.yuvType);
                BgrToLab(bgr, dst.planes[0]);
                break;
            }
            default:
                assert(0);
            }
            break;

        default:
            assert(0);
        }
    }

    //-------------------------------------------------------------------------------------------------

    template<template<class> class A> SIMD_INLINE void Resize(const Frame<A>& src, Frame<A>& dst, SimdResizeMethodType method)
    {
        assert(src.format == dst.format && src.format != Frame<A>::None);

        if (EqualSize(src, dst))
        {
            Copy(src, dst);
        }
        else
        {
            SimdResizeChannelType type = SimdResizeChannelByte;
            void* mainResizer = SimdResizerInit(src.planes[0].width, src.planes[0].height, dst.planes[0].width, dst.planes[0].height, src.planes[0].ChannelCount(), type, method);
            if (mainResizer)
            {
                SimdResizerRun(mainResizer, src.planes[0].data, src.planes[0].stride, dst.planes[0].data, dst.planes[0].stride);
                if (src.format == Frame<A>::Yuv444p)
                {
                    SimdResizerRun(mainResizer, src.planes[1].data, src.planes[1].stride, dst.planes[1].data, dst.planes[1].stride);
                    SimdResizerRun(mainResizer, src.planes[2].data, src.planes[2].stride, dst.planes[2].data, dst.planes[2].stride);
                }
                SimdRelease(mainResizer);
            }
            else
                assert(0);
            if (src.format == Frame<A>::Nv12 || src.format == Frame<A>::Yuv420p)
            {
                void* halfResizer = SimdResizerInit(src.planes[1].width, src.planes[1].height, dst.planes[1].width, dst.planes[1].height, src.planes[1].ChannelCount(), type, method);
                if (halfResizer)
                {
                    SimdResizerRun(halfResizer, src.planes[1].data, src.planes[1].stride, dst.planes[1].data, dst.planes[1].stride);
                    if (src.format == Frame<A>::Yuv420p)
                        SimdResizerRun(halfResizer, src.planes[2].data, src.planes[2].stride, dst.planes[2].data, dst.planes[2].stride);
                    SimdRelease(halfResizer);
                }
                else
                    assert(0);
            }
        }
    }

    template<template<class> class A> SIMD_INLINE void Resize(const Frame<A>& src, Frame<A>& dst, const Point<ptrdiff_t>& size, SimdResizeMethodType method)
    {
        if (&src == &dst)
        {
            if (src.Size() != size)
            {
                Frame<A> tmp(size, src.format);
                Resize(src, tmp, method);
                dst.Swap(tmp);
            }
        }
        else
        {
            if (dst.Size() != size)
                dst.Recreate(size, src.format);
            Resize(src, dst, method);
        }
    }
}

#endif
