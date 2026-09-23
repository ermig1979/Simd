/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2014-2019 Antonenka Mikhail,
*               2018-2019 Dmitry Fedorov,
*               2019-2019 Artur Voronkov,
*               2022-2022 Fabien Spindler,
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
#ifndef __SimdView_hpp__
#define __SimdView_hpp__

#include "Simd/SimdRectangle.hpp"
#include "Simd/SimdAllocator.hpp"

#include <memory.h>
#include <assert.h>
#include <algorithm>
#include <fstream>

namespace Simd
{
    /*! @ingroup cpp_view

        \short Image: pixel buffer, size, stride, format and ownership.

        View<A> is the C++ image used by the library. The template argument
        A is the allocator template of an owned buffer. In-tree code uses
        Simd::Allocator. TestCheckCpp, Detection, ContourDetector, Font,
        Motion, ShiftDetector and the examples in Simd::Point and
        Simd::Pyramid all write
        <tt>typedef Simd::View&lt;Simd::Allocator&gt; View</tt>.
        Simd::Frame stores one view in each plane. Simd::Pyramid stores one
        Gray8 view at each level. The wrappers in SimdLib.hpp pass
        <tt>data</tt>, <tt>stride</tt>, <tt>width</tt>, <tt>height</tt> and
        <tt>format</tt> through to the C API.

        <tt>width</tt> and <tt>height</tt> are the size in pixels.
        <tt>stride</tt> is the step in bytes from one row to the next. An
        owned row is aligned, so <tt>stride</tt> can be larger than
        <tt>width * PixelSize()</tt>. <tt>data</tt> points at the first row.
        These fields are const. Recreate, Load, Clear, Release, Swap and the
        assignment operators replace them. Pixels are written through
        <tt>data</tt>, At and Row. TestViewVector stores moved views in
        <tt>std::vector&lt;View&gt;</tt> and then writes
        <tt>views[i].data[i]</tt>.

        A view either owns its buffer or references another buffer. The
        destructor, Clear and Recreate free the buffer only when Owner() is
        true. Owning views come from the allocating constructors, Recreate,
        Load, Copy, Clone and Capture. Referencing views come from the copy
        constructor, copy assignment, Region, Flipped, an external pointer
        and, with SIMD_OPENCV_ENABLE, from <tt>cv::Mat</tt>. A referencing
        view stays valid while that buffer stays alive.

        The copy constructor and the copy assignment share the pixel buffer
        and leave the destination non-owning. TestCheckCpp does
        <tt>View sv; sv = vs;</tt> and <tt>View cp = sv</tt>. Copy assignment
        over a view that already owns its buffer frees that buffer and then
        executes <tt>assert(0)</tt>. A build that defines NDEBUG omits the
        assert, and the view still becomes a reference of the assigned
        image. Replace an owning view with Recreate,
        Simd::Copy, Swap or move assignment. Move construction and move
        assignment transfer ownership and leave the source empty.
        TestViewMove does <tt>b = std::move(a)</tt>. TestViewVector
        <tt>push_back</tt> of a temporary view relies on the move, so the
        vector owns the buffer. The move operations require
        SIMD_CPP_2011_ENABLE.

        Copy() and Clone() duplicate pixels. Copy() returns an owning view
        by value. Clone() returns a heap view that the caller deletes.
        Capture() turns a referencing view into an owner. TestCheckCpp calls
        <tt>cp.Capture()</tt> after copying <tt>sv</tt>, so <tt>cp</tt> keeps
        its own pixels. ImageLoader::Release calls Release() and returns the
        decoded buffer from ::SimdImageLoadFromFile. The caller frees that
        pointer with ::SimdFree.

        Region returns a referencing view of the half-open rectangle
        <tt>[left, right) x [top, bottom)</tt>, clipped to the image. The
        sub-view keeps the parent stride and format. ShiftDetector estimates
        motion from <tt>background.Region(region)</tt>. Simd::CopyFrame
        copies the four bands around an interior rectangle with Region.
        ContourDetector::Detect runs on <tt>src.Region(_roi)</tt>. Detection
        fills <tt>mask.Region(rect)</tt>, and TestDetection places a window
        with <tt>Region(size, View::MiddleCenter)</tt> and
        <tt>View::MiddleRight</tt>. Font::Draw measures the text and draws
        into <tt>canvas.Region(Measure(text), position)</tt>. A function
        that takes <tt>View&amp;</tt> cannot bind a temporary, so the
        destination is passed through Ref(). Simd::Copy and Simd::Fill in
        CopyFrame, AlphaBlending into <tt>bkg.Region(rect).Ref()</tt>, and
        ImageMatcher wrapping a hash buffer as
        <tt>View(main, main, main, View::Gray8, hash-&gt;main).Ref()</tt>
        all do this.

        Flipped() returns a referencing view whose stride is the negation of
        this stride and whose <tt>data</tt> points at the last row. Row 0 of
        that view is the last row of this view. Frame::Flipped assigns
        <tt>planes[i] = planes[i].Flipped()</tt>.

        At&lt;T&gt; and Row&lt;T&gt; address pixels. T is the pixel type of
        the format: <tt>uint8_t</tt> for Gray8 (detection masks and the shift
        test), Simd::Pixel::Bgr24 and Simd::Pixel::Bgra32 for color images
        (the transform and font tests), and <tt>float</tt> for View::Float
        (TestSynet). The coordinates must lie inside the view.

        EqualSize compares width and height. Compatible also compares
        format. Simd::Convert requires equal sizes and accepts different
        formats. TestCheckCpp converts a 6x6 Bgra32 view into a 6x6 Gray8
        view. Simd::Copy and the other same-format wrappers require
        Compatible images and accept different strides.

        Format has the same numeric values as ::SimdPixelFormatType. Gray8
        is the format of pyramids, shift detection, contours and
        <tt>lena.pgm</tt>. Bgr24 and Bgra32 are the formats of OpenCV frames
        and of drawing. Uv16 is the interleaved chroma plane of an Nv12
        frame.

        Load and Save wrap ::SimdImageLoadFromFile, ::SimdImageLoadFromMemory
        and ::SimdImageSaveToFile. The detection, contour and shift-detector
        examples load <tt>lena.pgm</tt> and save <tt>result.pgm</tt>.
        TestImageIO also decodes a file image from memory.

        With SIMD_OPENCV_ENABLE a view and a <tt>cv::Mat</tt> share one
        buffer. The <tt>cv::Mat</tt> constructor reads <tt>cols</tt>,
        <tt>rows</tt>, <tt>step[0]</tt> and OcvTo(type). <tt>operator cv::Mat</tt>
        builds a header over this buffer. <tt>CV_8UC3</tt> becomes Bgr24,
        which is OpenCV's channel order. TestAnyToAny passes views to
        <tt>cv::cvtColor</tt> through that conversion. TestCheckCpp assigns
        a view from a <tt>cv::Mat</tt> and a <tt>cv::Mat</tt> from a view.

        Using example:
        \code
        #include "Simd/SimdLib.hpp"

        int main()
        {
            typedef Simd::View<Simd::Allocator> View;
            typedef Simd::Rectangle<ptrdiff_t> Rect;

            View image;
            if (!image.Load("../../data/image/face/lena.pgm"))
                return 1;

            View canvas(image.Size(), View::Gray8);
            Simd::Copy(image, canvas);

            Rect window(80, 40, 240, 200);
            View crop = canvas.Region(window);
            View patch = crop.Copy();

            Simd::Fill(canvas.Region(window).Ref(), 0);
            Simd::Copy(patch, canvas.Region(window).Ref());

            uint8_t value = canvas.At<uint8_t>(window.left, window.top);
            canvas.Save("result.pgm");
            return value;
        }
        \endcode

        OpenCV conversion (define SIMD_OPENCV_ENABLE before including this header):
        \code
        #include "opencv2/core/core.hpp"
        #define SIMD_OPENCV_ENABLE
        #include "Simd/SimdView.hpp"

        int main()
        {
            typedef Simd::View<Simd::Allocator> View;

            View view(40, 30, View::Bgr24);
            cv::Mat mat(80, 60, CV_8UC3);

            View fromMat = mat;
            cv::Mat fromView = view;
            return (int)fromMat.width + fromView.cols;
        }
        \endcode

        \ref cpp_view_functions.
    */
    template <template<class> class A>
    struct View
    {
        typedef A<uint8_t> Allocator; /*!< Allocator of an owned buffer. In-tree code passes Simd::Allocator. Alignment() is the default row alignment. */

        /*!
            \enum Format
            Describes pixel format of an image view.

            The enumerators have the same values as ::SimdPixelFormatType.
            Load, Save and the C wrappers cast Format to that type.
            PixelSize, ChannelSize and ChannelCount describe the memory
            layout of each enumerator. Simd::Pixel holds the channel layout
            of Bgr24, Bgra32, Hsv24, Hsl24, Rgb24 and Rgba32.
        */
        enum Format
        {
            /*! Empty view and the failure format of Load. PixelSize is 0. */
            None = 0,
            /*! One 8-bit channel. PixelSize is 1. Pyramid levels, ShiftDetector, ContourDetector, motion masks and lena.pgm use Gray8. */
            Gray8,
            /*! Two 8-bit channels, interleaved UV. PixelSize is 2. Frame::Nv12 stores this plane at half resolution. */
            Uv16,
            /*! Three 8-bit channels in B, G, R order. PixelSize is 3. OpenCV frames, drawing and Motion annotation use Bgr24. */
            Bgr24,
            /*! Four 8-bit channels in B, G, R, A order. PixelSize is 4. The font example draws on a Bgra32 canvas. */
            Bgra32,
            /*! One signed 16-bit channel. PixelSize is 2. Simd::Int16ToGray reads this format. */
            Int16,
            /*! One signed 32-bit channel. PixelSize is 4. */
            Int32,
            /*! One signed 64-bit channel. PixelSize is 8. ToOcv has no OpenCV type for Int64. */
            Int64,
            /*! One 32-bit floating-point channel. PixelSize is 4. TestSynet addresses rows with Row&lt;float&gt;. Synet tests recreate float views with an explicit alignment. */
            Float,
            /*! One 64-bit floating-point channel. PixelSize is 8. */
            Double,
            /*! 8-bit Bayer mosaic, 2x2 tile G R / B G. PixelSize is 1. */
            BayerGrbg,
            /*! 8-bit Bayer mosaic, 2x2 tile G B / R G. PixelSize is 1. */
            BayerGbrg,
            /*! 8-bit Bayer mosaic, 2x2 tile R G / G B. PixelSize is 1. */
            BayerRggb,
            /*! 8-bit Bayer mosaic, 2x2 tile B G / G R. PixelSize is 1. */
            BayerBggr,
            /*! Three 8-bit channels in H, S, V order. PixelSize is 3. Simd::BgrToHsv writes this format. */
            Hsv24,
            /*! Three 8-bit channels in H, S, L order. PixelSize is 3. Simd::BgrToHsl writes this format. */
            Hsl24,
            /*! Three 8-bit channels in R, G, B order. PixelSize is 3. */
            Rgb24,
            /*! Four 8-bit channels in R, G, B, A order. PixelSize is 4. */
            Rgba32,
            /*! Packed UYVY422. PixelSize is 2: every four bytes store two pixels as U0, Y0, V0, Y1. Simd::Yuv420pToUyvy422 writes this format. */
            Uyvy16,
            /*! Four 8-bit channels in A, R, G, B order. PixelSize is 4. Simd::AlphaPremultiply treats Argb32 as premultiplied alpha at the first channel. */
            Argb32,
            /*! Three 8-bit channels in CIELAB order. PixelSize is 3. Simd::BgrToLab writes this format. */
            Lab24,
        };

        /*!
            \enum Position
            Names the place where Region(size, position) puts a window.

            The window has the requested size and is then clipped to the
            image. Font::Draw(canvas, text, position, color) measures the
            text and uses this placement. TestDetection fills
            <tt>Region(Size(W / 3, H / 2), MiddleRight)</tt> and crops an
            object with <tt>Region(obj.Size() * 5 / 7, MiddleCenter)</tt>.
        */
        enum Position
        {
            TopLeft, /*!< Window origin at (0, 0). */
            TopCenter, /*!< Window centered horizontally and placed at the top. */
            TopRight, /*!< Window placed at the top-right. */
            MiddleLeft, /*!< Window centered vertically and placed at the left. */
            MiddleCenter, /*!< Window centered horizontally and vertically. */
            MiddleRight, /*!< Window centered vertically and placed at the right. */
            BottomLeft, /*!< Window placed at the bottom-left. */
            BottomCenter, /*!< Window centered horizontally and placed at the bottom. */
            BottomRight, /*!< Window placed at the bottom-right. */
        };

        const size_t width; /*!< \brief Width in pixels. A region can be narrower than its parent. */
        const size_t height; /*!< \brief Height in pixels. A region can be shorter than its parent. */
        const ptrdiff_t stride; /*!< \brief Bytes from one row to the next, including alignment padding. Flipped() makes this negative. */
        const Format format; /*!< \brief Pixel format. None is an empty view. */
        uint8_t * const data; /*!< \brief Pointer to the first row. Flipped() points it at the last row of the parent buffer. */

        /*!
            Creates an empty view.

            Width, height and stride are 0, format is None, data is NULL and
            the view owns nothing. ImageMatcher starts from an empty view and
            then assigns a Gray8 source or calls Recreate. A failed Load and
            Clear leave this same state.
        */
        View();

        /*!
            Creates a view that references another view.

            The new view copies width, height, stride, format and data and
            leaves Owner() false. The source keeps ownership. Destroying the
            source frees the buffer while this view still points at it.
            TestCheckCpp uses <tt>View cp = sv</tt> and then Capture() to
            obtain an independent buffer. Duplicate pixels with Copy() or
            Clone().

            \param [in] view - an original image view.
        */
        View(const View & view);

#ifdef SIMD_CPP_2011_ENABLE
        /*!
            Moves a view.

            Ownership, geometry and the pixel pointer are transferred with
            Swap. After the call, <tt>view</tt> is empty. TestViewMove and
            <tt>std::vector&lt;View&gt;::push_back</tt> of a temporary view
            in TestViewVector use this constructor. Frame can move a
            temporary view into <tt>planes[0]</tt>.

            \param [in] view - a view whose buffer is transferred.
        */
        View(View&& view) noexcept;
#endif

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Creates a view that references an OpenCV matrix.

            Width is <tt>mat.cols</tt>, height is <tt>mat.rows</tt>, stride
            is <tt>mat.step[0]</tt> and data is <tt>mat.data</tt>. The format
            is OcvTo(mat.type()), or None when <tt>mat.data</tt> is empty.
            The view does not own the buffer. <tt>CV_8UC3</tt> becomes
            Bgr24. TestCheckCpp assigns <tt>sv = cm</tt> for a
            <tt>cv::Mat cm</tt>.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] mat - an OpenCV matrix.
        */
        View(const cv::Mat & mat);
#endif


        /*!
            Creates a view with an explicit stride.

            The pointer <tt>d</tt> is stored as given. This is the constructor
            used to wrap an external buffer whose row step is already known.
            ImageMatcher hashes into
            <tt>View(main, main, main, View::Gray8, hash-&gt;main)</tt>.
            Detection wraps an integral-image row the same way and passes the
            temporary through Ref(). Region and Flipped build their results
            with this constructor.

            When <tt>d</tt> is NULL and width, height, stride and format are
            all non-zero, the view allocates <tt>height * stride</tt> bytes
            with Allocator::Alignment() and becomes the owner. That
            allocation uses a positive stride. The stride value itself is
            not aligned. A zero dimension or format None leaves the view
            without a buffer.

            \param [in] w - width in pixels.
            \param [in] h - height in pixels.
            \param [in] s - stride in bytes. May be larger than the row payload, and is negative for a view produced by Flipped().
            \param [in] f - pixel format.
            \param [in] d - external pixel buffer. NULL allocates an owned buffer when the geometry is non-empty.
        */
        View(size_t w, size_t h, ptrdiff_t s, Format f, void * d);

        /*!
            Creates a view and aligns its rows.

            Calls Recreate(w, h, f, d, align). The stride is
            <tt>Allocator::Align(width * PixelSize(f), align)</tt>.
            <tt>View image(320, 240, View::Gray8)</tt> and the 128x96 Bgr24
            source in TestImageResize use this constructor with a NULL
            buffer, so the view owns the pixels. A non-NULL <tt>d</tt> is
            aligned upward and is not owned. The bytes before the aligned
            address and a full <tt>height * stride</tt> block must fit in
            the caller buffer.

            \param [in] w - width in pixels.
            \param [in] h - height in pixels.
            \param [in] f - pixel format.
            \param [in] d - external pixel buffer. NULL allocates an owned buffer. The default is NULL.
            \param [in] align - row and pointer alignment in bytes. The default is Allocator::Alignment().
        */
        View(size_t w, size_t h, Format f, void * d = NULL, size_t align = Allocator::Alignment());

        /*!
            Creates an owned view of the given size and format.

            Calls Recreate(size.x, size.y, f) with the default alignment.
            <tt>size.x</tt> is the width and <tt>size.y</tt> is the height.
            The pyramid example allocates <tt>View dst(image.Size(), View::Gray8)</tt>
            this way. The buffer is uninitialized.

            \param [in] size - width in <tt>x</tt> and height in <tt>y</tt>.
            \param [in] f - pixel format.
        */
        View(const Point<ptrdiff_t> & size, Format f);

        /*!
            Destroys the view.

            Frees <tt>data</tt> with Allocator::Free when Owner() is true.
            A referencing view does not free the buffer.
        */
        ~View();

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Creates an OpenCV header that references this image.

            The matrix is <tt>cv::Mat(height, width, ToOcv(format), data, stride)</tt>.
            It does not own the buffer. TestAnyToAny passes
            <tt>(cv::Mat)src</tt> and <tt>(cv::Mat)(dst.Ref())</tt> to
            <tt>cv::cvtColor</tt>. TestCheckCpp assigns <tt>cm = sv</tt>.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \return an OpenCV matrix header over this buffer.
        */
        operator cv::Mat() const;
#endif


#ifdef SIMD_TENSORFLOW_ENABLE
        /*!
            Writes this image into a rank-3 float tensor.

            The tensor is indexed <tt>(row, col, channel)</tt>. Bgr24 writes
            three channels. Bgra32 writes blue, green and red and skips
            alpha. Gray8 writes one channel, reading the first byte of each
            row for every column. The stored value is
            <tt>(sample + shift) * scale</tt>. Any other format leaves the
            tensor unchanged. The view buffer is not shared with the tensor.

            \note You have to define SIMD_TENSORFLOW_ENABLE in order to use this functionality.

            \param [out] tensor - destination tensor of rank 3.
            \param [in] shift - value added to each sample. The default is 0.
            \param [in] scale - value multiplied after the shift. The default is 1.
        */
        void ToTFTensor(tensorflow::Tensor & tensor, float shift = 0, float scale = 1) const;

        /*!
            Writes this image into one batch of a rank-4 float tensor.

            The tensor is indexed <tt>(batchIndex, row, col, channel)</tt>.
            The channel layout is the same as for the rank-3 overload.
            The default scale of this overload is 0.

            \note You have to define SIMD_TENSORFLOW_ENABLE in order to use this functionality.

            \param [out] tensor - destination tensor of rank 4.
            \param [in] batchIndex - batch coordinate written by this call.
            \param [in] shift - value added to each sample. The default is 0.
            \param [in] scale - value multiplied after the shift. The default is 0.
        */
        void ToTFTensor(tensorflow::Tensor & tensor, int batchIndex, float shift = 0, float scale = 0) const;
#endif

        /*!
            Duplicates this image on the heap.

            Allocates an owned view of the same width, height and format and
            copies <tt>width * PixelSize()</tt> bytes of each row. Row
            padding is not copied. The caller deletes the returned view.

            \return a heap view that owns the copied pixels.
        */
        View * Clone() const;

        /*!
            Duplicates a rectangular region on the heap.

            Equivalent to <tt>Region(rect).Clone()</tt>. The rectangle is
            half-open and clipped by Region.

            \param [in] rect - a rectangle which bounds the region.
            \return a heap view that owns the copied region.
        */
        View* Clone(const Rectangle<ptrdiff_t>& rect) const;

        /*!
            Copies this image into <tt>buffer</tt> and returns a view of that storage.

            When <tt>buffer</tt> is narrower or shorter than this view,
            <tt>buffer.Recreate(width, height, format)</tt> replaces it.
            The returned view is a non-owning header over <tt>buffer.data</tt>
            with the aligned stride of this width, which can differ from
            <tt>buffer.stride</tt>. Each row of <tt>width * PixelSize()</tt>
            bytes is copied into that header. The caller deletes the view
            object. The pixels remain in <tt>buffer</tt>.

            \param [in, out] buffer - storage for the copy. Recreated when it is too small.
            \return a heap view that references <tt>buffer</tt>.
        */
        View * Clone(View & buffer) const;

        /*!
            Duplicates this image by value.

            Allocates an owned view and copies <tt>width * PixelSize()</tt>
            bytes of each row. Row padding is not copied. With
            SIMD_CPP_2011_ENABLE the returned view can be moved, so the
            caller keeps the owned buffer. The example above stores a crop
            with <tt>View patch = crop.Copy()</tt>.

            \return an owning view with a copy of the pixels.
        */
        View Copy() const;

        /*!
            Duplicates a rectangular region by value.

            Equivalent to <tt>Region(rect).Copy()</tt>.

            \param [in] rect - a rectangle which bounds the region.
            \return an owning view with a copy of the region.
        */
        View Copy(const Rectangle<ptrdiff_t>& rect) const;

        /*!
            Makes this view reference another view.

            Self-assignment does nothing. Otherwise, when this view owns its
            buffer, that buffer is freed and <tt>assert(0)</tt> is executed.
            A build that defines NDEBUG omits the assert. In either build
            the destination then copies width, height, stride, format and
            data and becomes non-owning. TestCheckCpp assigns into a
            default-constructed view: <tt>View sv; sv = vs;</tt>. To replace
            the pixels of an owning view, use Recreate, Simd::Copy, Swap or
            move assignment.

            \param [in] view - an original image view.
            \return a reference to itself.
        */
        View & operator = (const View & view);

#ifdef SIMD_CPP_2011_ENABLE
        /*!
            Moves another view into this view.

            Swap exchanges both views, including ownership, and then Clear
            empties the source. An owned buffer previously held by this view
            is freed. TestViewMove replaces <tt>b</tt> with
            <tt>std::move(a)</tt>.

            \param [in] view - a view whose buffer is transferred.
            \return a reference to itself.
        */
        View& operator = (View&& view);
#endif

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Makes this view reference an OpenCV matrix.

            Equivalent to assigning <tt>View(mat)</tt>. The same ownership
            rule as copy assignment applies: assigning over an owning view
            frees its buffer and executes <tt>assert(0)</tt>.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] mat - an OpenCV matrix.
            \return a reference to itself.
        */
        View & operator = (const cv::Mat & mat);
#endif

        /*!
            Returns this view so a temporary can bind to <tt>View&amp;</tt>.

            Region, Flipped and a constructor expression are temporaries.
            Simd::Copy, Simd::Fill, Simd::AlphaBlending and the other output
            arguments take <tt>View&amp;</tt>, so the temporary is passed
            through Ref(). CopyFrame copies each outer band with
            <tt>dst.Region(...).Ref()</tt>. TestShift blends into
            <tt>bkg.Region(rect).Ref()</tt>. Font::Draw draws into
            <tt>canvas.Region(Measure(text), position).Ref()</tt>.

            \code
            #include "Simd/SimdLib.hpp"

            int main()
            {
                typedef Simd::View<Simd::Allocator> View;
                View a(100, 100, View::Gray8);
                View b(100, 100, View::Gray8);
                Simd::Copy(a.Region(20, 20, 80, 80), b.Region(20, 20, 80, 80).Ref());
                return 0;
            }
            \endcode

            \return a reference to itself.
        */
        View & Ref();

        /*!
            Replaces the geometry and the buffer of this view.

            An owned buffer is freed first. The stride becomes
            <tt>Allocator::Align(width * PixelSize(f), align)</tt>.
            A non-NULL <tt>d</tt> is aligned upward and is not owned. A NULL
            <tt>d</tt> with non-zero height and stride allocates
            <tt>height * stride</tt> bytes and the view becomes the owner.
            The new pixels are uninitialized. Pyramid::Recreate allocates
            each level with <tt>Recreate(size, Gray8)</tt>. ImageMatcher
            recreates a Gray8 view before Simd::Convert. Synet tests place
            a float row in <tt>Recreate(count, 1, View::Float, NULL, align)</tt>.

            \param [in] w - width in pixels.
            \param [in] h - height in pixels.
            \param [in] f - pixel format.
            \param [in] d - external pixel buffer. NULL allocates an owned buffer. The default is NULL.
            \param [in] align - row and pointer alignment in bytes. The default is Allocator::Alignment().
        */
        void Recreate(size_t w, size_t h, Format f, void * d = NULL, size_t align = Allocator::Alignment());

        /*!
            Replaces this view with an owned image of the given size and format.

            Calls <tt>Recreate(size.x, size.y, f)</tt>.

            \param [in] size - width in <tt>x</tt> and height in <tt>y</tt>.
            \param [in] f - pixel format.
        */
        void Recreate(const Point<ptrdiff_t> & size, Format f);

        /*!
            Returns a referencing sub-view of a half-open rectangle.

            The rectangle is <tt>[left, right) x [top, bottom)</tt>. When
            <tt>data</tt> is NULL, or <tt>right &lt; left</tt>, or
            <tt>bottom &lt; top</tt>, the result is an empty view. Otherwise
            each side is clamped to the image, and the result has
            width <tt>right - left</tt>, height <tt>bottom - top</tt>, the
            parent stride and format, and
            <tt>data + top * stride + left * PixelSize()</tt>.
            A rectangle that only partly overlaps the image is reduced to
            the overlap. The result does not own the pixels.

            ShiftDetector passes <tt>background.Region(region)</tt>.
            CopyFrame splits the image into the bands outside an interior
            rectangle. ContourDetector restricts metrics to
            <tt>src.Region(_roi)</tt>. An output region is bound with Ref().

            \param [in] left - left side of the region.
            \param [in] top - top side of the region.
            \param [in] right - right side of the region. The pixel column <tt>right</tt> is outside the region.
            \param [in] bottom - bottom side of the region. The pixel row <tt>bottom</tt> is outside the region.
            \return a referencing view of the clipped region.
        */
        View Region(ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom) const;

        /*!
            Returns a referencing sub-view between two corners.

            Calls <tt>Region(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)</tt>.
            TestDetection blends a patch with
            <tt>dst.Region(p, p + Size(s, s))</tt>.

            \param [in] topLeft - top-left corner of the region.
            \param [in] bottomRight - bottom-right corner of the region. This corner is outside the region.
            \return a referencing view of the clipped region.
        */
        View Region(const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight) const;

        /*!
            Returns a referencing sub-view of a rectangle.

            Calls <tt>Region(rect.Left(), rect.Top(), rect.Right(), rect.Bottom())</tt>.
            The rectangle uses the same half-open range as Simd::Rectangle.
            <tt>Region(Rectangle&lt;ptrdiff_t&gt;(view.Size()))</tt> is the
            full image. ShiftDetector shifts the rectangle before taking the
            region: <tt>background.Region(region.Shifted(shift))</tt>.

            \param [in] rect - a rectangle which bounds the region.
            \return a referencing view of the clipped region.
        */
        View Region(const Rectangle<ptrdiff_t> & rect) const;

        /*!
            Returns a referencing window placed at a named position.

            The window size is <tt>size</tt>. The position selects one of the
            nine placements: the horizontal origin is 0, <tt>width - size.x</tt>,
            or <tt>(width - size.x) / 2</tt>, and the vertical origin is 0,
            <tt>height - size.y</tt>, or <tt>(height - size.y) / 2</tt>.
            A centered span runs from <tt>(side - size) / 2</tt> to
            <tt>(side + size) / 2</tt>. The resulting rectangle is clipped
            by Region(left, top, right, bottom), so a window larger than the
            image is reduced to the image. Font::Draw and TestDetection use
            this overload.

            \param [in] size - width in <tt>x</tt> and height in <tt>y</tt> of the window.
            \param [in] position - placement of the window. See Simd::View::Position.
            \return a referencing view of the clipped window.
        */
        View Region(const Point<ptrdiff_t> & size, Position position) const;

        /*!
            Returns a referencing view of this image turned upside down.

            For a view with at least one row, the result has the same width,
            height and format, stride <tt>-stride</tt>, and data
            <tt>data + (height - 1) * stride</tt>. Row <tt>y</tt> of the
            result is row <tt>height - 1 - y</tt> of this view. The result
            does not own the pixels. Frame::Flipped stores
            <tt>planes[i].Flipped()</tt> and toggles Frame::flipped.
            DataSize() of a flipped view is not the allocation size, because
            the stride is negative.

            \return a referencing view with a negative stride.
        */
        View Flipped() const;

        /*!
            Returns the image size as a point.

            <tt>x</tt> is the width and <tt>y</tt> is the height.
            Detection::Init, ContourDetector::Init and Pyramid take this
            size. <tt>Rectangle&lt;ptrdiff_t&gt;(view.Size())</tt> is the
            full image, because that rectangle constructor treats the point
            as the bottom-right corner.

            \return a point with image width and height.
        */
        Point<ptrdiff_t> Size() const;

        /*!
            Returns <tt>stride * height</tt>.

            For an owned view created by Recreate or by an allocating
            constructor this is the number of allocated bytes, including
            row padding. A region reports the parent stride times the region
            height, which is larger than the region payload when the parent
            rows are padded. A flipped view has a negative stride, so this
            product is not the buffer size.

            \return stride times height, as <tt>size_t</tt>.
        */
        size_t DataSize() const;

        /*!
            Returns the number of pixels, <tt>width * height</tt>.

            \return the area of this view in pixels.
        */
        size_t Area() const;

        /*!
            Returns a const pixel at integer coordinates.

            The pixel is <tt>((const T*)(data + y * stride))[x]</tt>.
            <tt>sizeof(T)</tt> is the pixel size: <tt>uint8_t</tt> for Gray8,
            Simd::Pixel::Bgr24 for Bgr24, <tt>float</tt> for Float, and so on.
            The function asserts <tt>x &lt; width</tt> and <tt>y &lt; height</tt>.
            Coordinates are not clamped. Detection reads a mask with
            <tt>At&lt;uint8_t&gt;(col, row)</tt>.

            \param [in] x - x coordinate of the pixel.
            \param [in] y - y coordinate of the pixel.
            \return a const reference to the pixel.
        */
        template <class T> const T & At(size_t x, size_t y) const;

        /*!
            Returns a pixel at integer coordinates.

            The address is the same as for the const overload. The transform
            test assigns <tt>At&lt;Simd::Pixel::Bgr24&gt;(x, y)</tt> and
            <tt>At&lt;Simd::Pixel::Bgra32&gt;(x, y)</tt>.

            \param [in] x - x coordinate of the pixel.
            \param [in] y - y coordinate of the pixel.
            \return a reference to the pixel.
        */
        template <class T> T & At(size_t x, size_t y);

        /*!
            Returns a const pixel at a point.

            Calls <tt>At&lt;T&gt;(p.x, p.y)</tt>. Motion reads a neighbour
            produced by <tt>current + Point(-1, 0)</tt> this way. A negative
            coordinate converts to a large <tt>size_t</tt> and fails the
            bounds assert.

            \param [in] p - coordinates of the pixel.
            \return a const reference to the pixel.
        */
        template <class T> const T & At(const Point<ptrdiff_t> & p) const;

        /*!
            Returns a pixel at a point.

            Calls <tt>At&lt;T&gt;(p.x, p.y)</tt>.

            \param [in] p - coordinates of the pixel.
            \return a reference to the pixel.
        */
        template <class T> T & At(const Point<ptrdiff_t> & p);

        /*!
            Returns a const pointer to the first pixel of a row.

            The pointer is <tt>(const T*)(data + row * stride)</tt>.
            The function asserts <tt>row &lt; height</tt>. The font test
            scans <tt>Row&lt;uint8_t&gt;</tt>. TestSynet takes
            <tt>Row&lt;float&gt;</tt> of a Float view. The transform test
            writes a Uv16 row through <tt>Row&lt;uint8_t&gt;</tt> at
            indexes <tt>2 * x</tt> and <tt>2 * x + 1</tt>. A negative stride
            walks upward, which is how a flipped view addresses its rows.

            \param [in] row - row index.
            \return a const pointer to the first pixel of the row.
        */
        template <class T> const T * Row(size_t row) const;

        /*!
            Returns a pointer to the first pixel of a row.

            The address is the same as for the const overload.

            \param [in] row - row index.
            \return a pointer to the first pixel of the row.
        */
        template <class T> T * Row(size_t row);

        /*!
            Returns the pixel size in bytes of a format.

            The sizes are: 0 for None; 1 for Gray8 and every Bayer format;
            2 for Uv16, Int16 and Uyvy16; 3 for Bgr24, Hsv24, Hsl24, Rgb24
            and Lab24; 4 for Bgra32, Int32, Float, Rgba32 and Argb32; 8 for
            Int64 and Double. An unknown format asserts and returns 0.
            Simd::Copy passes this size to ::SimdCopy. Font::Draw requires
            <tt>sizeof(color)</tt> to equal PixelSize().

            \param [in] format - a pixel format.
            \return the pixel size in bytes.
        */
        static size_t PixelSize(Format format);

        /*!
            Returns the pixel size in bytes of this image.

            Calls <tt>PixelSize(format)</tt>.

            \return the pixel size in bytes.
        */
        size_t PixelSize() const;

        /*!
            Returns the channel size in bytes of a format.

            The sizes are: 0 for None; 1 for every 8-bit format, including
            multi-channel 8-bit formats and both Bayer and UYVY; 2 for Int16;
            4 for Int32 and Float; 8 for Int64 and Double. An unknown format
            asserts and returns 0.

            \param [in] format - a pixel format.
            \return the channel size in bytes.
        */
        static size_t ChannelSize(Format format);

        /*!
            Returns the channel size in bytes of this image.

            Calls <tt>ChannelSize(format)</tt>.

            \return the channel size in bytes.
        */
        size_t ChannelSize() const;

        /*!
            Returns the number of channels in a format.

            The counts are: 0 for None; 1 for Gray8, Int16, Int32, Int64,
            Float, Double and every Bayer format; 2 for Uv16 and Uyvy16;
            3 for Bgr24, Hsv24, Hsl24, Rgb24 and Lab24; 4 for Bgra32, Rgba32
            and Argb32. An unknown format asserts and returns 0.
            Simd::AlphaBlending passes ChannelCount() as the channel count
            of the color image.

            \param [in] format - a pixel format.
            \return the number of channels.
        */
        static size_t ChannelCount(Format format);

        /*!
            Returns the number of channels in this image.

            Calls <tt>ChannelCount(format)</tt>.

            \return the number of channels.
        */
        size_t ChannelCount() const;

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Converts a pixel format to an OpenCV matrix type.

            Gray8, None and every Bayer format return <tt>CV_8UC1</tt>.
            Uv16 and Uyvy16 return <tt>CV_8UC2</tt>. Bgr24, Hsv24, Hsl24,
            Rgb24 and Lab24 return <tt>CV_8UC3</tt>. Bgra32, Rgba32 and
            Argb32 return <tt>CV_8UC4</tt>. Int16 returns <tt>CV_16SC1</tt>,
            Int32 returns <tt>CV_32SC1</tt>, Float returns <tt>CV_32FC1</tt>
            and Double returns <tt>CV_64FC1</tt>. Int64 is not mapped and
            asserts. The OpenCV type does not record BGR versus RGB or a
            Bayer pattern. <tt>operator cv::Mat</tt> uses this mapping.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] format - a pixel format.
            \return an OpenCV matrix type.
        */
        static int ToOcv(Format format);

        /*!
            Converts an OpenCV matrix type to a pixel format.

            <tt>CV_8UC1</tt> returns Gray8, <tt>CV_8UC2</tt> returns Uv16,
            <tt>CV_8UC3</tt> returns Bgr24, <tt>CV_8UC4</tt> returns Bgra32,
            <tt>CV_16SC1</tt> returns Int16, <tt>CV_32SC1</tt> returns Int32,
            <tt>CV_32FC1</tt> returns Float and <tt>CV_64FC1</tt> returns
            Double. Any other type asserts and returns None. A 3-channel
            OpenCV matrix therefore becomes Bgr24, including when the caller
            stored RGB, HSV or Lab bytes. The <tt>cv::Mat</tt> constructor
            uses this mapping.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] type - an OpenCV matrix type.
            \return a pixel format.
        */
        static Format OcvTo(int type);
#endif

        /*!
            Exchanges this view with another view.

            Width, height, stride, format, data and ownership are swapped.
            TestCheckCpp calls <tt>sv.Swap(vs)</tt>. The move constructor
            and move assignment are implemented with Swap. Frame::Swap
            swaps every plane through this method.

            \param [in, out] other - the view to exchange with.
        */
        void Swap(View & other);

        /*!
            Decodes an image from a file into this view.

            Clear() runs first, so a previous owned buffer is freed. The
            file type is detected from the file content. PGM, PPM, PNG,
            JPEG and BMP are recognized. PGM and PPM comments are not
            supported, and a PGM or PPM max value other than 255 is
            rejected. On input, <tt>format</tt> requests Gray8, Bgr24,
            Bgra32, Rgb24, Rgba32, or None. None keeps the natural format
            of the file, which for <tt>lena.pgm</tt> is Gray8. On success
            the view owns the decoded buffer and the method returns true.
            The detection, contour and shift-detector examples load
            <tt>../../data/image/face/lena.pgm</tt>. TestYuvToAny loads a
            file as Bgr24. On failure the view is empty, format is None
            and the method returns false.

            \param [in] path - path to the image file.
            \param [in] format - requested pixel format. The default is None.
            \return true when the file was decoded.
        */
        bool Load(const std::string & path, Format format = None);

        /*!
            Decodes an image from a memory buffer into this view.

            The buffer is a file image, not raw pixels. Detection of the
            file type, the requested formats and the ownership of the
            result are the same as for Load(path). TestImageIO decodes
            two encoded buffers and compares the views.

            \param [in] src - pointer to the encoded file bytes.
            \param [in] size - number of bytes in <tt>src</tt>.
            \param [in] format - requested pixel format. The default is None.
            \return true when the buffer was decoded.
        */
        bool Load(const uint8_t * src, size_t size, Format format = None);

        /*!
            Encodes this image and writes it to a file.

            The view is not changed. <tt>type</tt> selects the encoder.
            ::SimdImageFileUndefined, the default, selects it from the
            path extension: <tt>.pgm</tt>, <tt>.ppm</tt>, <tt>.png</tt>,
            <tt>.jpg</tt>/<tt>.jpeg</tt> or <tt>.bmp</tt>. For a JPEG
            extension, quality 100 is written as quality 85. An
            unrecognized extension saves Gray8 as binary PGM and any other
            supported format as binary PPM. The detection and contour
            examples call <tt>Save("result.pgm")</tt>. TestDrawing saves
            <tt>draw_line.jpg</tt> and <tt>rectangles.pgm</tt>. TestResize
            saves a JPEG region with an explicit quality of 85.

            \param [in] path - path to the output file.
            \param [in] type - file format, or ::SimdImageFileUndefined to infer it from <tt>path</tt>. The default is ::SimdImageFileUndefined.
            \param [in] quality - compression quality for encoders that use it. The default is 100.
            \return true when the file was written.
        */
        bool Save(const std::string & path, SimdImageFileType type = SimdImageFileUndefined, int quality = 100) const;

        /*!
            Releases an owned buffer and makes the view empty.

            When Owner() is true the buffer is freed. Width, height and
            stride become 0, format becomes None, data becomes NULL and
            the view is not an owner. Load calls Clear before decoding.
            Move assignment calls Clear on the source after Swap.
        */
        void Clear();

        /*!
            Detaches the pixel buffer and makes the view empty.

            The buffer is not freed. When <tt>size</tt> is not NULL,
            <tt>*size</tt> receives DataSize() from before the fields are
            cleared. ImageLoader::Release reads the geometry and then calls
            Release() so ::SimdImageLoadFromFile can return the decoded
            pointer. That pointer was allocated by the Simd allocator and
            is freed with ::SimdFree. Release of a referencing view still
            detaches <tt>data</tt>; that external pointer is not an
            allocation to pass to ::SimdFree.

            \param [out] size - optional pointer that receives DataSize() of the detached buffer. May be NULL.
            \return the detached pixel pointer.
        */
        uint8_t* Release(size_t* size = NULL);

        /*!
            Reports whether the destructor will free the buffer.

            The allocating constructors, Recreate with a NULL buffer, Load,
            Copy, Clone(), Capture and a moved-from owner return true.
            The copy constructor, copy assignment, Region, Flipped, an
            external pointer and a <tt>cv::Mat</tt> wrapper return false.
            Frame::Owner is true only when every used plane is an owner.

            \return true when this view owns its buffer.
        */
        bool Owner() const;

        /*!
            Makes this view the owner of its pixels.

            A view that is already an owner, or whose <tt>data</tt> is NULL,
            is left unchanged. Otherwise, when <tt>copy</tt> is true, the
            pixels are copied into a new owned buffer and the previous
            pointer is left with its original owner. TestCheckCpp does this
            after <tt>View cp = sv</tt>, and Frame::Capture does it for
            every plane. When <tt>copy</tt> is false, the view takes
            ownership of the current pointer without copying. That pointer
            must be one Allocator::Free can release.

            \param [in] copy - true copies the pixels into a new buffer. False adopts the current pointer. The default is true.
        */
        void Capture(bool copy = true);

    private:
        bool _owner;
    };

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A, class T> const T & At(const View<A> & view, size_t x, size_t y);

        Returns a const pixel of a view.

        The pixel is <tt>((const T*)(view.data + y * view.stride))[x]</tt>.
        The function asserts <tt>x &lt; view.width</tt> and
        <tt>y &lt; view.height</tt>. <tt>sizeof(T)</tt> is the pixel size.
        View::At is the method form of this function, and Simd::Pixel::At
        checks the view format before using it.

        \param [in] view - an image.
        \param [in] x - x coordinate of the pixel.
        \param [in] y - y coordinate of the pixel.
        \return a const reference to the pixel.
    */
    template <template<class> class A, class T> const T & At(const View<A> & view, size_t x, size_t y);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A, class T> T & At(View<A> & view, size_t x, size_t y);

        Returns a pixel of a view.

        The address is the same as for the const overload.

        \param [in, out] view - an image.
        \param [in] x - x coordinate of the pixel.
        \param [in] y - y coordinate of the pixel.
        \return a reference to the pixel.
    */
    template <template<class> class A, class T> T & At(View<A> & view, size_t x, size_t y);


    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A, template<class> class B> bool EqualSize(const View<A> & a, const View<B> & b);

        Checks that two views have the same width and height.

        The formats and strides may differ. The two views may use different
        allocator templates. Simd::Convert asserts EqualSize and then
        converts between Gray8, Bgr24, Bgra32, Rgb24 and Rgba32.
        TestCheckCpp relies on that for a 6x6 Bgra32 source and a 6x6 Gray8
        destination.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \return true when the widths are equal and the heights are equal.
    */
    template <template<class> class A, template<class> class B> bool EqualSize(const View<A> & a, const View<B> & b);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c);

        Checks that three views have the same width and height.

        The formats and strides may differ. <tt>b</tt> and <tt>c</tt> are
        both compared with <tt>a</tt>.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] c - a third image.
        \return true when all three views have the size of <tt>a</tt>.
    */
    template <template<class> class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c);


    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);

        Checks that four views have the same width and height.

        The formats and strides may differ. <tt>b</tt>, <tt>c</tt> and
        <tt>d</tt> are compared with <tt>a</tt>.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] c - a third image.
        \param [in] d - a fourth image.
        \return true when all four views have the size of <tt>a</tt>.
    */
    template <template<class> class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A, template<class> class B> bool Compatible(const View<A> & a, const View<B> & b);

        Checks that two views have the same width, height and format.

        The strides may differ, and the views may use different allocator
        templates. Simd::Copy asserts Compatible and then copies
        <tt>width * PixelSize()</tt> bytes of each row. A Gray8 pyramid
        level and a Gray8 image of that size are compatible. A Bgra32 view
        and a Gray8 view of the same size are not; Simd::Convert accepts
        that pair because it checks EqualSize.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \return true when the sizes and formats are equal.
    */
    template <template<class> class A, template<class> class B> bool Compatible(const View<A> & a, const View<B> & b);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c);

        Checks that three views have the same width, height and format.

        The strides may differ. <tt>b</tt> and <tt>c</tt> are both compared
        with <tt>a</tt>.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] c - a third image.
        \return true when all three views have the size and format of <tt>a</tt>.
    */
    template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);

        Checks that four views have the same width, height and format.

        The strides may differ. <tt>b</tt>, <tt>c</tt> and <tt>d</tt> are
        compared with <tt>a</tt>.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] c - a third image.
        \param [in] d - a fourth image.
        \return true when all four views have the size and format of <tt>a</tt>.
    */
    template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d, const View<A> & e);

        Checks that five views have the same width, height and format.

        The strides may differ. <tt>b</tt>, <tt>c</tt>, <tt>d</tt> and
        <tt>e</tt> are compared with <tt>a</tt>.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] c - a third image.
        \param [in] d - a fourth image.
        \param [in] e - a fifth image.
        \return true when all five views have the size and format of <tt>a</tt>.
    */
    template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d, const View<A> & e);

    //-------------------------------------------------------------------------

    // struct View implementation:

    template <template<class> class A> SIMD_INLINE View<A>::View()
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
    }

    /*! \cond */
    template <template<class> class A> SIMD_INLINE View<A>::View(const View<A> & view)
        : width(view.width)
        , height(view.height)
        , stride(view.stride)
        , format(view.format)
        , data(view.data)
        , _owner(false)
    {
    }

#ifdef SIMD_CPP_2011_ENABLE
    template <template<class> class A> SIMD_INLINE View<A>::View(View<A> && view) noexcept
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
        Swap(view);
    }
#endif
    /*! \endcond */

#ifdef SIMD_OPENCV_ENABLE
    template <template<class> class A> SIMD_INLINE View<A>::View(const cv::Mat & mat)
        : width(mat.cols)
        , height(mat.rows)
        , stride(mat.step[0])
        , format(mat.data ? OcvTo(mat.type()) : None)
        , data(mat.data)
        , _owner(false)
    {
    }
#endif

#ifdef SIMD_TENSORFLOW_ENABLE
    template <template<class> class A> SIMD_INLINE void View<A>::ToTFTensor( tensorflow::Tensor & tensor, float shift, float scale) const
    {
        auto mapped = tensor.tensor<float, 3>();

        if (format == View<A>::Bgr24)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t * bgr = data + row*stride;
                for (size_t col = 0; col < width; ++col, bgr += 3)
                {
                    mapped(row, col, 0) = (bgr[0] + shift) * scale;
                    mapped(row, col, 1) = (bgr[1] + shift) * scale;
                    mapped(row, col, 2) = (bgr[2] + shift) * scale;
                }
            }
        } else if (format == View<A>::Bgra32)
        {

            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t * bgra = data + row*stride;
                for (size_t col = 0; col < width; ++col, bgra += 4)
                {
                    mapped(row, col, 0) = (bgra[0] + shift) * scale;
                    mapped(row, col, 1) = (bgra[1] + shift) * scale;
                    mapped(row, col, 2) = (bgra[2] + shift) * scale;
                }
            }
        } else if (format == View<A>::Gray8)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t * gray = data + row*stride;
                for (size_t col = 0; col < width; ++col)
                {
                    mapped(row, col, 0) = (gray[0] + shift) * scale;
                }
            }
        }
    }

    template <template<class> class A> SIMD_INLINE void View<A>::ToTFTensor( tensorflow::Tensor & tensor, int batchIndex, float shift, float scale) const
    {
        auto mapped = tensor.tensor<float, 4>();

        if (format == View<A>::Bgr24)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t * bgr = data + row*stride;
                for (size_t col = 0; col < width; ++col, bgr += 3)
                {
                    mapped(batchIndex, row, col, 0) = ((float)bgr[0] + shift) * scale;
                    mapped(batchIndex, row, col, 1) = ((float)bgr[1] + shift) * scale;
                    mapped(batchIndex, row, col, 2) = ((float)bgr[2] + shift) * scale;
                }
            }
        } else if (format == View<A>::Bgra32)
        {

            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t * bgra = data + row*stride;
                for (size_t col = 0; col < width; ++col, bgra += 4)
                {
                    mapped(batchIndex, row, col, 0) = ((float)bgra[0] + shift) * scale;
                    mapped(batchIndex, row, col, 1) = ((float)bgra[1] + shift) * scale;
                    mapped(batchIndex, row, col, 2) = ((float)bgra[2] + shift) * scale;
                }
            }
        } else if (format == View<A>::Gray8)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t * gray = data + row*stride;
                for (size_t col = 0; col < width; ++col)
                {
                    mapped(batchIndex, row, col, 0) = ((float)gray[0] + shift) * scale;
                }
            }
        }
    }
#endif

    template <template<class> class A> SIMD_INLINE View<A>::View(size_t w, size_t h, ptrdiff_t s, Format f, void * d)
        : width(w)
        , height(h)
        , stride(s)
        , format(f)
        , data((uint8_t*)d)
        , _owner(false)
    {
        if (data == NULL && height && width && stride && format != None)
        {
            *(void**)&data = Allocator::Allocate(height*stride, Allocator::Alignment());
            _owner = true;
        }
    }

    template <template<class> class A> SIMD_INLINE View<A>::View(size_t w, size_t h, Format f, void * d, size_t align)
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
        Recreate(w, h, f, d, align);
    }

    template <template<class> class A> SIMD_INLINE View<A>::View(const Point<ptrdiff_t> & size, Format f)
        : width(0)
        , height(0)
        , stride(0)
        , format(None)
        , data(NULL)
        , _owner(false)
    {
        Recreate(size.x, size.y, f);
    }

    template <template<class> class A> SIMD_INLINE View<A>::~View()
    {
        if (_owner && data)
        {
            Allocator::Free(data);
        }
    }

#ifdef SIMD_OPENCV_ENABLE
    template <template<class> class A> SIMD_INLINE View<A>::operator cv::Mat() const
    {
        return cv::Mat((int)height, (int)width, ToOcv(format), data, stride);
    }
#endif

    template <template<class> class A> SIMD_INLINE View<A> * View<A>::Clone() const
    {
        View<A> * view = new View<A>(width, height, format);
        size_t size = width*PixelSize();
        for (size_t row = 0; row < height; ++row)
            memcpy(view->data + view->stride*row, data + stride*row, size);
        return view;
    }

    template <template<class> class A> SIMD_INLINE View<A>* View<A>::Clone(const Rectangle<ptrdiff_t>& rect) const
    {
        return Region(rect).Clone();
    }

    template <template<class> class A> SIMD_INLINE View<A> * View<A>::Clone(View & buffer) const
    {
        if (buffer.width < width || buffer.height < height)
            buffer.Recreate(width, height, format);

        View<A> * view = new View<A>(width, height, format, buffer.data);
        size_t size = width*PixelSize();
        for (size_t row = 0; row < height; ++row)
            memcpy(view->data + view->stride*row, data + stride*row, size);
        return view;
    }

    template <template<class> class A> SIMD_INLINE View<A> View<A>::Copy() const
    {
        View<A> view(width, height, format);
        size_t size = width*PixelSize();
        for (size_t row = 0; row < height; ++row)
            memcpy(view.data + view.stride*row, data + stride*row, size);
        return view;
    }

    template <template<class> class A> SIMD_INLINE View<A> View<A>::Copy(const Rectangle<ptrdiff_t>& rect) const
    {
        return Region(rect).Copy();
    }

    /*! \cond */
    template <template<class> class A> SIMD_INLINE View<A> & View<A>::operator = (const View<A> & view)
    {
        if (this != &view)
        {
            if (_owner && data)
            {
                Allocator::Free(data);
                assert(0);
            }
            *(size_t*)&width = view.width;
            *(size_t*)&height = view.height;
            *(Format*)&format = view.format;
            *(ptrdiff_t*)&stride = view.stride;
            *(unsigned char**)&data = view.data;
            _owner = false;
        }
        return *this;
    }

#ifdef SIMD_CPP_2011_ENABLE
    template <template<class> class A> SIMD_INLINE View<A>& View<A>::operator = (View<A>&& view)
    {
        if (this != &view)
        {
            Swap(view);
            view.Clear();
        }
        return *this;
    }
#endif
    /*! \endcond */

#ifdef SIMD_OPENCV_ENABLE
    template <template<class> class A> SIMD_INLINE View<A> & View<A>::operator = (const cv::Mat & mat)
    {
        *this = View<A>(mat);
        return *this;
    }
#endif

    template <template<class> class A> SIMD_INLINE View<A> & View<A>::Ref()
    {
        return *this;
    }

    template <template<class> class A> SIMD_INLINE void View<A>::Recreate(size_t w, size_t h, Format f, void * d, size_t align)
    {
        if (_owner && data)
        {
            Allocator::Free(data);
            *(void**)&data = NULL;
            _owner = false;
        }
        *(size_t*)&width = w;
        *(size_t*)&height = h;
        *(Format*)&format = f;
        *(ptrdiff_t*)&stride = Allocator::Align(width*PixelSize(format), align);
        if (d)
        {
            *(void**)&data = Allocator::Align(d, align);
            _owner = false;
        }
        else if(height && stride)
        {
            *(void**)&data = Allocator::Allocate(height*stride, align);
            _owner = true;
        }
    }

    template <template<class> class A> SIMD_INLINE void View<A>::Recreate(const Point<ptrdiff_t> & size, Format f)
    {
        Recreate(size.x, size.y, f);
    }

    template <template<class> class A> SIMD_INLINE View<A> View<A>::Region(ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom) const
    {
        if (data != NULL && right >= left && bottom >= top)
        {
            left = std::min<ptrdiff_t>(std::max<ptrdiff_t>(left, 0), width);
            top = std::min<ptrdiff_t>(std::max<ptrdiff_t>(top, 0), height);
            right = std::min<ptrdiff_t>(std::max<ptrdiff_t>(right, 0), width);
            bottom = std::min<ptrdiff_t>(std::max<ptrdiff_t>(bottom, 0), height);
            return View<A>(right - left, bottom - top, stride, format, data + top*stride + left*PixelSize(format));
        }
        else
            return View<A>();
    }

    template <template<class> class A> SIMD_INLINE View<A> View<A>::Region(const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight) const
    {
        return Region(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y);
    }

    template <template<class> class A> SIMD_INLINE View<A> View<A>::Region(const Rectangle<ptrdiff_t> & rect) const
    {
        return Region(rect.Left(), rect.Top(), rect.Right(), rect.Bottom());
    }

    template <template<class> class A> SIMD_INLINE View<A> View<A>::Region(const Point<ptrdiff_t> & size, Position position) const
    {
        ptrdiff_t w = width, h = height;
        switch (position)
        {
        case TopLeft:
            return Region(0, 0, size.x, size.y);
        case TopCenter:
            return Region((w - size.x) / 2, 0, (w + size.x) / 2, size.y);
        case TopRight:
            return Region(w - size.x, 0, w, size.y);
        case MiddleLeft:
            return Region(0, (h - size.y) / 2, size.x, (h + size.y) / 2);
        case MiddleCenter:
            return Region((w - size.x) / 2, (h - size.y) / 2, (w + size.x) / 2, (h + size.y) / 2);
        case MiddleRight:
            return Region(w - size.x, (h - size.y) / 2, w, (h + size.y) / 2);
        case BottomLeft:
            return Region(0, h - size.y, size.x, h);
        case BottomCenter:
            return Region((w - size.x) / 2, h - size.y, (w + size.x) / 2, h);
        case BottomRight:
            return Region(w - size.x, h - size.y, w, h);
        default:
            assert(0);
        }
        return View<A>();
    }

    template <template<class> class A> SIMD_INLINE View<A> View<A>::Flipped() const
    {
        return View<A>(width, height, -stride, format, data + (height - 1)*stride);
    }

    template <template<class> class A> SIMD_INLINE Point<ptrdiff_t> View<A>::Size() const
    {
        return Point<ptrdiff_t>(width, height);
    }

    template <template<class> class A> SIMD_INLINE size_t View<A>::DataSize() const
    {
        return stride*height;
    }

    template <template<class> class A> SIMD_INLINE size_t View<A>::Area() const
    {
        return width*height;
    }

    template <template<class> class A> template<class T> SIMD_INLINE const T & View<A>::At(size_t x, size_t y) const
    {
        assert(x < width && y < height);
        return ((const T*)(data + y*stride))[x];
    }

    template <template<class> class A> template<class T> SIMD_INLINE T & View<A>::At(size_t x, size_t y)
    {
        assert(x < width && y < height);
        return ((T*)(data + y*stride))[x];
    }

    template <template<class> class A> template<class T> SIMD_INLINE const T & View<A>::At(const Point<ptrdiff_t> & p) const
    {
        return At<T>(p.x, p.y);
    }

    template <template<class> class A> template<class T> SIMD_INLINE T & View<A>::At(const Point<ptrdiff_t> & p)
    {
        return At<T>(p.x, p.y);
    }

    template <template<class> class A> template<class T> SIMD_INLINE const T * View<A>::Row(size_t row) const
    {
        assert(row < height);
        return ((const T*)(data + row*stride));
    }

    template <template<class> class A> template<class T> SIMD_INLINE T * View<A>::Row(size_t row)
    {
        assert(row < height);
        return ((T*)(data + row*stride));
    }

    template <template<class> class A> SIMD_INLINE size_t View<A>::PixelSize(Format format)
    {
        switch (format)
        {
        case None:      return 0;
        case Gray8:     return 1;
        case Uv16:      return 2;
        case Bgr24:     return 3;
        case Bgra32:    return 4;
        case Int16:     return 2;
        case Int32:     return 4;
        case Int64:     return 8;
        case Float:     return 4;
        case Double:    return 8;
        case BayerGrbg: return 1;
        case BayerGbrg: return 1;
        case BayerRggb: return 1;
        case BayerBggr: return 1;
        case Hsv24:     return 3;
        case Hsl24:     return 3;
        case Rgb24:     return 3;
        case Rgba32:    return 4;
        case Uyvy16:    return 2;
        case Argb32:    return 4;
        case Lab24:     return 3;
        default: assert(0); return 0;
        }
    }

    template <template<class> class A> SIMD_INLINE size_t View<A>::PixelSize() const
    {
        return PixelSize(format);
    }

    template <template<class> class A> SIMD_INLINE size_t View<A>::ChannelSize(Format format)
    {
        switch (format)
        {
        case None:      return 0;
        case Gray8:     return 1;
        case Uv16:      return 1;
        case Bgr24:     return 1;
        case Bgra32:    return 1;
        case Int16:     return 2;
        case Int32:     return 4;
        case Int64:     return 8;
        case Float:     return 4;
        case Double:    return 8;
        case BayerGrbg: return 1;
        case BayerGbrg: return 1;
        case BayerRggb: return 1;
        case BayerBggr: return 1;
        case Hsv24:     return 1;
        case Hsl24:     return 1;
        case Rgb24:     return 1;
        case Rgba32:    return 1;
        case Uyvy16:    return 1;
        case Argb32:    return 1;
        case Lab24:     return 1;
        default: assert(0); return 0;
        }
    }

    template <template<class> class A> SIMD_INLINE size_t View<A>::ChannelSize() const
    {
        return ChannelSize(format);
    }

    template <template<class> class A> SIMD_INLINE size_t View<A>::ChannelCount(Format format)
    {
        switch (format)
        {
        case None:      return 0;
        case Gray8:     return 1;
        case Uv16:      return 2;
        case Bgr24:     return 3;
        case Bgra32:    return 4;
        case Int16:     return 1;
        case Int32:     return 1;
        case Int64:     return 1;
        case Float:     return 1;
        case Double:    return 1;
        case BayerGrbg: return 1;
        case BayerGbrg: return 1;
        case BayerRggb: return 1;
        case BayerBggr: return 1;
        case Hsv24:     return 3;
        case Hsl24:     return 3;
        case Rgb24:     return 3;
        case Rgba32:    return 4;
        case Uyvy16:    return 2;
        case Argb32:    return 4;
        case Lab24:     return 3;
        default: assert(0); return 0;
        }
    }

    template <template<class> class A> SIMD_INLINE size_t View<A>::ChannelCount() const
    {
        return ChannelCount(format);
    }

#ifdef SIMD_OPENCV_ENABLE
    template <template<class> class A> SIMD_INLINE int View<A>::ToOcv(Format format)
    {
        switch (format)
        {
        case None:      return CV_8UC1;
        case Gray8:     return CV_8UC1;
        case Uv16:      return CV_8UC2;
        case Bgr24:     return CV_8UC3;
        case Bgra32:    return CV_8UC4;
        case Int16:     return CV_16SC1;
        case Int32:     return CV_32SC1;
        case Float:     return CV_32FC1;
        case Double:    return CV_64FC1;
        case BayerGrbg: return CV_8UC1;
        case BayerGbrg: return CV_8UC1;
        case BayerRggb: return CV_8UC1;
        case BayerBggr: return CV_8UC1;
        case Hsv24:     return CV_8UC3;
        case Hsl24:     return CV_8UC3;
        case Rgb24:     return CV_8UC3;
        case Rgba32:    return CV_8UC4;
        case Uyvy16:    return CV_8UC2;
        case Argb32:    return CV_8UC4;
        case Lab24:     return CV_8UC3;
        default: assert(0); return 0;
        }
    }

    template <template<class> class A> SIMD_INLINE typename View<A>::Format View<A>::OcvTo(int type)
    {
        switch (type)
        {
        case CV_8UC1:   return Gray8;
        case CV_8UC2:   return Uv16;
        case CV_8UC3:   return Bgr24;
        case CV_8UC4:   return Bgra32;
        case CV_16SC1:  return Int16;
        case CV_32SC1:  return Int32;
        case CV_32FC1:  return Float;
        case CV_64FC1:  return Double;
        default: assert(0); return None;
        }
    }
#endif

    template <template<class> class A> SIMD_INLINE void View<A>::Swap(View<A> & other)
    {
        std::swap((size_t&)width, (size_t&)other.width);
        std::swap((size_t&)height, (size_t&)other.height);
        std::swap((ptrdiff_t&)stride, (ptrdiff_t&)other.stride);
        std::swap((Format&)format, (Format&)other.format);
        std::swap((uint8_t*&)data, (uint8_t*&)other.data);
        std::swap((bool&)_owner, (bool&)other._owner);
    }

    template <template<class> class A> SIMD_INLINE bool View<A>::Load(const std::string & path, Format format_)
    {
        Clear();
        (Format&)format = format_;
        *(uint8_t**)&data = SimdImageLoadFromFile(path.c_str(), (size_t*)&stride, (size_t*)&width, (size_t*)&height, (SimdPixelFormatType*)&format);
        if (data)
            _owner = true;
        else
            (Format&)format = None;
        return _owner;
    }

    template <template<class> class A> SIMD_INLINE bool View<A>::Load(const uint8_t * src, size_t size, Format format_)
    {
        Clear();
        (Format&)format = format_;
        *(uint8_t**)&data = SimdImageLoadFromMemory(src, size, (size_t*)&stride, (size_t*)&width, (size_t*)&height, (SimdPixelFormatType*)&format);
        if (data)
            _owner = true;
        else
            (Format&)format = None;
        return _owner;
    }

    template <template<class> class A> SIMD_INLINE bool View<A>::Save(const std::string & path, SimdImageFileType type, int quality) const
    {
        return SimdImageSaveToFile(data, stride, width, height, (SimdPixelFormatType)format, type, quality, path.c_str()) == SimdTrue;
    }

    template <template<class> class A> SIMD_INLINE void View<A>::Clear()
    {
        if (_owner && data)
            Allocator::Free(data);
#ifdef SIMD_CPP_2011_ENABLE
        *(void**)&data = nullptr;
#else
        *(void**)&data = NULL;
#endif
        _owner = false;
        *(size_t*)&width = 0;
        *(size_t*)&height = 0;
        *(ptrdiff_t *)&stride = 0;
#ifdef SIMD_CPP_2011_ENABLE
        *(Format*)&format = Format::None;
#else
        *(Format*)&format = (Format)(0); // Modified for c++ 98
#endif
    }

    template <template<class> class A> SIMD_INLINE uint8_t* View<A>::Release(size_t* size)
    {
        uint8_t* released = data;
        if (size)
            *size = DataSize();
        _owner = false;
        Clear();
        return released;
    }

    template <template<class> class A> SIMD_INLINE bool View<A>::Owner() const
    {
        return _owner;
    }

    template <template<class> class A> SIMD_INLINE void View<A>::Capture(bool copy)
    {
        if (data && _owner == false)
        {
            if (copy)
            {
                View<A> buffer(width, height, format);
                size_t size = width * PixelSize();
                for (size_t row = 0; row < height; ++row)
                    memcpy(buffer.data + buffer.stride * row, data + stride * row, size);
                Swap(buffer);
            }
            else
                _owner = true;
        }
    }

    // View utilities implementation:

    template <template<class> class A, class T> const T & At(const View<A> & view, size_t x, size_t y)
    {
        assert(x < view.width && y < view.height);

        return ((const T*)(view.data + y*view.stride))[x];
    }

    template <template<class> class A, class T> T & At(View<A> & view, size_t x, size_t y)
    {
        assert(x < view.width && y < view.height);

        return ((T*)(view.data + y*view.stride))[x];
    }

    template <template<class> class A, template<class> class B> SIMD_INLINE bool EqualSize(const View<A> & a, const View<B> & b)
    {
        return
            (a.width == b.width && a.height == b.height);
    }

    template <template<class> class A> SIMD_INLINE bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c)
    {
        return
            (a.width == b.width && a.height == b.height) &&
            (a.width == c.width && a.height == c.height);
    }

    template <template<class> class A> SIMD_INLINE bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d)
    {
        return
            (a.width == b.width && a.height == b.height) &&
            (a.width == c.width && a.height == c.height) &&
            (a.width == d.width && a.height == d.height);
    }

    template <template<class> class A, template<class> class B> SIMD_INLINE bool Compatible(const View<A> & a, const View<B> & b)
    {
        typedef typename View<A>::Format Format;

        return
            (a.width == b.width && a.height == b.height && a.format == (Format)b.format);
    }

    template <template<class> class A> SIMD_INLINE bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c)
    {
        return
            (a.width == b.width && a.height == b.height && a.format == b.format) &&
            (a.width == c.width && a.height == c.height && a.format == c.format);
    }

    template <template<class> class A> SIMD_INLINE bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d)
    {
        return
            (a.width == b.width && a.height == b.height && a.format == b.format) &&
            (a.width == c.width && a.height == c.height && a.format == c.format) &&
            (a.width == d.width && a.height == d.height && a.format == d.format);
    }

    template <template<class> class A> SIMD_INLINE bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d, const View<A> & e)
    {
        return
            (a.width == b.width && a.height == b.height && a.format == b.format) &&
            (a.width == c.width && a.height == c.height && a.format == c.format) &&
            (a.width == d.width && a.height == d.height && a.format == d.format) &&
            (a.width == e.width && a.height == e.height && a.format == e.format);
    }
}

#endif//__SimdView_hpp__
