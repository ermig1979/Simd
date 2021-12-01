/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar,
*               2014-2019 Antonenka Mikhail,
*               2018-2019 Dmitry Fedorov,
*               2019-2019 Artur Voronkov.
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

        \short The View structure provides storage and manipulation of images.

        In order to have mutual conversion with OpenCV image type (cv::Mat) you have to define macro SIMD_OPENCV_ENABLE:
        \verbatim
        #include "opencv2/core/core.hpp"
        #define SIMD_OPENCV_ENABLE
        #include "Simd/SimdView.hpp"

        int main()
        {
            typedef Simd::View<Simd::Allocator> View;

            View view1(40, 30, View::Bgr24);
            cv::Mat mat1(80, 60, CV_8UC3)

            View view2 = mat1; // view2 will be refer to mat1, it is not a copy!
            cv::Mat mat2 = view1; // mat2 will be refer to view1, it is not a copy!

            return 0;
        }
        \endverbatim

        \ref cpp_view_functions.
    */
    template <template<class> class A>
    struct View
    {
        typedef A<uint8_t> Allocator; /*!< Allocator type definition. */

        /*!
            \enum Format
            Describes pixel format types of an image view.
            \note This type is corresponds to C type ::SimdPixelFormatType.
        */
        enum Format
        {
            /*! An undefined pixel format. */
            None = 0,
            /*! A 8-bit gray pixel format. */
            Gray8,
            /*! A 16-bit (2 8-bit channels) pixel format (UV plane of NV12 pixel format). */
            Uv16,
            /*! A 24-bit (3 8-bit channels) BGR (Blue, Green, Red) pixel format. */
            Bgr24,
            /*! A 32-bit (4 8-bit channels) BGRA (Blue, Green, Red, Alpha) pixel format. */
            Bgra32,
            /*! A single channel 16-bit integer pixel format. */
            Int16,
            /*! A single channel 32-bit integer pixel format. */
            Int32,
            /*! A single channel 64-bit integer pixel format. */
            Int64,
            /*! A single channel 32-bit float point pixel format. */
            Float,
            /*! A single channel 64-bit float point pixel format. */
            Double,
            /*! A 8-bit Bayer pixel format (GRBG). */
            BayerGrbg,
            /*! A 8-bit Bayer pixel format (GBRG). */
            BayerGbrg,
            /*! A 8-bit Bayer pixel format (RGGB). */
            BayerRggb,
            /*! A 8-bit Bayer pixel format (BGGR). */
            BayerBggr,
            /*! A 24-bit (3 8-bit channels) HSV (Hue, Saturation, Value) pixel format. */
            Hsv24,
            /*! A 24-bit (3 8-bit channels) HSL (Hue, Saturation, Lightness) pixel format. */
            Hsl24,
            /*! A 24-bit (3 8-bit channels) RGB (Red, Green, Blue) pixel format. */
            Rgb24,
            /*! A 32-bit (4 8-bit channels) RGBA (Red, Green, Blue, Alpha) pixel format. */
            Rgba32,
            /*! A 16-bit (2 8-bit channels) UYVY422 pixel format. */
            Uyvy16,
        };

        /*!
            \enum Position
            Describes the position of the child image view to the parent image view.
            This enum is used for creation of sub image view in method Simd::View::Region.
        */
        enum Position
        {
            TopLeft, /*!< A position in the top-left corner. */
            TopCenter, /*!< A position at the top center. */
            TopRight, /*!< A position in the top-right corner. */
            MiddleLeft, /*!< A position of the left in the middle. */
            MiddleCenter, /*!< A central position. */
            MiddleRight, /*!< A position of the right in the middle. */
            BottomLeft, /*!< A position in the bottom-left corner. */
            BottomCenter, /*!< A position at the bottom center. */
            BottomRight, /*!< A position in the bottom-right corner. */
        };

        const size_t width; /*!< \brief A width of the image. */
        const size_t height; /*!< \brief A height of the image. */
        const ptrdiff_t stride; /*!< \brief A row size of the image in bytes. */
        const Format format; /*!< \brief A pixel format types of the image. */
        uint8_t * const data; /*!< \brief A pointer to the pixel data (first row) of the image. */

        /*!
            Creates a new empty View structure.
        */
        View();

        /*!
            Creates a new View structure on the base of the image view.

            \note This constructor is not create new image view! It only creates a reference to the same image. If you want to create a copy then must use method Simd::View::Clone.

            \param [in] view - an original image view.
        */
        View(const View & view);

        /*!
            Move constructor of View structure.

            \param [in] view - a moved View.
        */
        View(View&& view) noexcept;

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Creates a new View structure on the base of OpenCV Mat type.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] mat - an OpenCV Mat.
        */
        View(const cv::Mat & mat);
#endif


        /*!
            Creates a new View structure with specified width, height, row size, pixel format and pointer to pixel data.

            \param [in] w - a width of created image view.
            \param [in] h - a height of created image view.
            \param [in] s - a stride (row size) of created image view.
            \param [in] f - a pixel format of created image view.
            \param [in] d - a pointer to the external buffer with pixel data. If this pointer is NULL then will be created own buffer.
        */
        View(size_t w, size_t h, ptrdiff_t s, Format f, void * d);

        /*!
            Creates a new View structure with specified width, height, pixel format, pointer to pixel data and memory alignment.

            \param [in] w - a width of created image view.
            \param [in] h - a height of created image view.
            \param [in] f - a pixel format of created image view.
            \param [in] d - a pointer to the external buffer with pixel data. If this pointer is NULL then will be created own buffer.
            \param [in] align - a required memory alignment. Its default value is determined by function Allocator::Alignment.
        */
        View(size_t w, size_t h, Format f, void * d = NULL, size_t align = Allocator::Alignment());

        /*!
            Creates a new View structure with specified width, height and pixel format.

            \param [in] size - a size (width and height) of created image view.
            \param [in] f - a pixel format of created image view.
        */
        View(const Point<ptrdiff_t> & size, Format f);

        /*!
            A View destructor.
        */
        ~View();

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Creates an OpenCV Mat which references this image.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \return an OpenCV Mat which references to this image.
        */
        operator cv::Mat() const;
#endif


#ifdef SIMD_TENSORFLOW_ENABLE
        /*!
            Creates an Tensorflow Tensor which references this image.

            \note You have to define SIMD_TENSORFLOW_ENABLE in order to use this functionality.

            \return an Tensorflow Tensor which references to this image.
        */
        void ToTFTensor(tensorflow::Tensor & tensor, float shift = 0, float scale = 1) const;


        /*!
           Creates an Tensorflow Tensor which references this image.

           \note You have to define SIMD_TENSORFLOW_ENABLE in order to use this functionality.

           \return an Tensorflow Tensor which references to this image.
       */
        void ToTFTensor(tensorflow::Tensor & tensor, int batchIndex, float shift = 0, float scale = 0) const;
#endif

        /*!
            Gets a copy of current image view.

            \return a pointer to the new View structure. The user must free this pointer after usage.
        */
        View * Clone() const;

        /*!
            Gets a copy of region of current image view which bounded by the rectangle with specified coordinates.

            \param [in] rect - a rectangle which bound the region.
            \return - a pointer to the new View structure. The user must free this pointer after usage.
        */
        View* Clone(const Rectangle<ptrdiff_t>& rect) const;

        /*!
            Gets a copy of current image view using buffer as a storage.

            \param [in] buffer - an external view as a buffer.
            \return a pointer to the new View structure (not owner). The user must free this pointer after usage.
        */
        View * Clone(View & buffer) const;

        /*!
            Creates view which references to other View structure.

            \note This function does not create a copy of image view! It only creates a reference to the same image.

            \param [in] view - an original image view.
            \return a reference to itself.
        */
        View & operator = (const View & view);

        /*!
            Moves View structure.

            \param [in] view - a moved image view.
            \return a reference to itself.
        */
        View& operator = (View&& view);

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Creates view which references to an OpenCV Mat.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] mat - an OpenCV Mat.
            \return a reference to itself.
        */
        View & operator = (const cv::Mat & mat);
#endif

        /*!
            Creates reference to itself. 
            It may be useful if we need to create reference to the temporary object:
            \verbatim
            #include "Simd/SimdLib.hpp"

            int main()
            {
                typedef Simd::View<Simd::Allocator> View;
                View a(100, 100, View::Gray8);
                View b(100, 100, View::Gray8);
                // Copying of a central part of a to the center of b:
                Simd::Copy(a.Region(20, 20, 80, 80), b.Region(20, 20, 80, 80).Ref());
                return 0;
            }
            \endverbatim

            \return a reference to itself.
        */
        View & Ref();

        /*!
            Re-creates a View structure with specified width, height, pixel format, pointer to pixel data and memory alignment.

            \param [in] w - a width of re-created image view.
            \param [in] h - a height of re-created image view.
            \param [in] f - a pixel format of re-created image view.
            \param [in] d - a pointer to the external buffer with pixel data. If this pointer is NULL then will be created own buffer.
            \param [in] align - a required memory alignment. Its default value is determined by function Allocator::Alignment.
        */
        void Recreate(size_t w, size_t h, Format f, void * d = NULL, size_t align = Allocator::Alignment());

        /*!
            Re-creates a View structure with specified width, height and pixel format.

            \param [in] size - a size (width and height) of re-created image view.
            \param [in] f - a pixel format of re-created image view.
        */
        void Recreate(const Point<ptrdiff_t> & size, Format f);

        /*!
            Creates a new View structure which points to the region of current image bounded by the rectangle with specified coordinates.

            \param [in] left - a left side of the region.
            \param [in] top - a top side of the region.
            \param [in] right - a right side of the region.
            \param [in] bottom - a bottom side of the region.
            \return - a new View structure which points to the region of current image.
        */
        View Region(ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom) const;

        /*!
            Creates a new View structure which points to the region of current image bounded by the rectangle with specified coordinates.

            \param [in] topLeft - a top-left corner of the region.
            \param [in] bottomRight - a bottom-right corner of the region.
            \return - a new View structure which points to the region of current image.
        */
        View Region(const Point<ptrdiff_t> & topLeft, const Point<ptrdiff_t> & bottomRight) const;

        /*!
            Creates a new View structure which points to the region of current image bounded by the rectangle with specified coordinates.

            \param [in] rect - a rectangle which bound the region.
            \return - a new View structure which points to the region of current image.
        */
        View Region(const Rectangle<ptrdiff_t> & rect) const;

        /*!
            Creates a new View structure which points to the region of current image bounded by the rectangle with specified coordinates.

            \param [in] size - a size (width and height) of the region.
            \param [in] position - a value represents the position of the region (see Simd::View::Position).
            \return - a new View structure which points to the region of current image.
        */
        View Region(const Point<ptrdiff_t> & size, Position position) const;

        /*!
            Creates a new View structure which points to the vertically flipped image.

            \return - a new View structure which points to the flipped image.
        */
        View Flipped() const;

        /*!
            Gets size (width and height) of the image.

            \return - a new Point structure with image width and height.
        */
        Point<ptrdiff_t> Size() const;

        /*!
            Gets size in bytes required to store pixel data of current View structure.

            \return - a size of data pixels in bytes.
        */
        size_t DataSize() const;

        /*!
            Gets area in pixels of of current View structure.

            \return - a area of current View in pixels.
        */
        size_t Area() const;

        /*!
            Gets constant reference to the pixel of arbitrary type into current view with specified coordinates.

            \param [in] x - a x-coordinate of the pixel.
            \param [in] y - a y-coordinate of the pixel.
            \return - a constant reference to pixel of arbitrary type.
        */
        template <class T> const T & At(size_t x, size_t y) const;

        /*!
            Gets reference to the pixel of arbitrary type into current view with specified coordinates.

            \param [in] x - a x-coordinate of the pixel.
            \param [in] y - a y-coordinate of the pixel.
            \return - a reference to pixel of arbitrary type.
        */
        template <class T> T & At(size_t x, size_t y);

        /*!
            Gets constant reference to the pixel of arbitrary type into current view with specified coordinates.

            \param [in] p - a point with coordinates of the pixel.
            \return - a constant reference to pixel of arbitrary type.
        */
        template <class T> const T & At(const Point<ptrdiff_t> & p) const;

        /*!
            Gets reference to the pixel of arbitrary type into current view with specified coordinates.

            \param [in] p - a point with coordinates of the pixel.
            \return - a reference to pixel of arbitrary type.
        */
        template <class T> T & At(const Point<ptrdiff_t> & p);

        /*!
            Gets constant pointer to the first pixel of specified row.

            \param [in] row - a row of the image.
            \return - a constant pointer to the first pixel.
        */
        template <class T> const T * Row(size_t row) const;

        /*!
            Gets pointer to the first pixel of specified row.

            \param [in] row - a row of the image.
            \return - a pointer to the first pixel.
        */
        template <class T> T * Row(size_t row);

        /*!
            \fn size_t PixelSize(Format format);

            Gets pixel size in bytes for current pixel format.

            \param [in] format - a pixel format.
            \return - a pixel size in bytes.
        */
        static size_t PixelSize(Format format);

        /*!
            Gets pixel size in bytes for current image.

            \return - a pixel size in bytes.
        */
        size_t PixelSize() const;

        /*!
            \fn size_t ChannelSize(Format format);

            Gets pixel channel size in bytes for current pixel format.

            \param [in] format - a pixel format.
            \return - a pixel channel size in bytes.
        */
        static size_t ChannelSize(Format format);

        /*!
            Gets pixel channel size in bytes for current image.

            \return - a pixel channel size in bytes.
        */
        size_t ChannelSize() const;

        /*!
            \fn size_t ChannelCount(Format format);

            Gets number of channels in the pixel for current pixel format.

            \param [in] format - a pixel format.
            \return - a number of channels.
        */
        static size_t ChannelCount(Format format);

        /*!
            Gets number of channels in the pixel for current image.

            \return - a number of channels.
        */
        size_t ChannelCount() const;

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Converts Simd Library pixel format to OpenCV Matrix type.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] format - a Simd Library pixel format.
            \return - an OpenCV Matrix type.
        */
        static int ToOcv(Format format);

        /*!
            Converts OpenCV Matrix type to Simd Library pixel format.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] type - an OpenCV Matrix type.
            \return - a Simd Library pixel format.
        */
        static Format OcvTo(int type);
#endif

        /*!
            Swaps content of two (this and other) View  structures.

            \param [in] other - an other image view.
        */
        void Swap(View & other);

        /*!
            Loads image from file.
            
            Supported formats are described by ::SimdImageFileType enumeration.

            \note PGM and PPM files with comments are not supported.

            \param [in] path - a path to image file.
            \param [in] format - a desired format of loaded image. 
                Supported values are View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24, View::Rgba32 and View::None.
                Default value is View::None (loads image in native pixel format of image file).
            \return - a result of loading.
        */
        bool Load(const std::string & path, Format format = None);

        /*!
            Loads image from memory buffer.

            Supported formats are described by ::SimdImageFileType enumeration.

            \note PGM and PPM files with comments are not supported.

            \param [in] src - a pointer to memory buffer.
            \param [in] size - a buffer size.
            \param [in] format - a desired format of loaded image.
                Supported values are View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24, View::Rgba32 and View::None.
                Default value is View::None (loads image in native pixel format of image file).
            \return - a result of loading.
        */
        bool Load(const uint8_t * src, size_t size, Format format = None);

        /*!
            Saves image to file.
 
            \param [in] path - a path to file.
            \param [in] type - a image file format. By default is equal to ::SimdImageFileUndefined (format auto choice).
            \param [in] quality - a parameter of compression quality (if file format supports it).
            \return - a result of saving.
        */
        bool Save(const std::string & path, SimdImageFileType type = SimdImageFileUndefined, int quality = 100) const;

        /*!
            Clears View structure (reset all fields) and free memory if it's owner.
         */
        void Clear();

        /*!
            Releases pixel data and resets all fields.

            \param [out] size - a pointer to the size of released pixel data. Can be NULL.
            \return - a released pointer to pixel data. It must be deleted by function ::SimdFree.
        */
        uint8_t* Release(size_t* size = NULL);

    private:
        bool _owner;
    };

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A, class T> const T & At(const View<A> & view, size_t x, size_t y);

        Gets constant reference to the pixel of arbitrary type at the point at the image with specified coordinates.

        \param [in] view - an image.
        \param [in] x - a x-coordinate of the pixel.
        \param [in] y - a y-coordinate of the pixel.
        \return - a const reference to pixel of arbitrary type.
    */
    template <template<class> class A, class T> const T & At(const View<A> & view, size_t x, size_t y);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A, class T> T & At(View<A> & view, size_t x, size_t y);

        Gets reference to the pixel of arbitrary type at the point at the image with specified coordinates.

        \param [in] view - an image.
        \param [in] x - a x-coordinate of the pixel.
        \param [in] y - a y-coordinate of the pixel.
        \return - a reference to pixel of arbitrary type.
    */
    template <template<class> class A, class T> T & At(View<A> & view, size_t x, size_t y);


    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A, template<class> class B> bool EqualSize(const View<A> & a, const View<B> & b);

        Checks two image views on the same size.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \return - a result of checking.
    */
    template <template<class> class A, template<class> class B> bool EqualSize(const View<A> & a, const View<B> & b);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c);

        Checks three image views on the same size.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] c - a third image.
        \return - a result of checking.
    */
    template <template<class> class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c);


    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);

        Checks four image views on the same size.

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] c - a third image.
        \param [in] d - a fourth image.
        \return - a result of checking.
    */
    template <template<class> class A> bool EqualSize(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A, template<class> class B> bool Compatible(const View<A> & a, const View<B> & b);

        Checks two image views on compatibility (the images must have the same size and pixel format).

        \param [in] a - a first image.
        \param [in] b - a second image.
        \return - a result of checking.
    */
    template <template<class> class A, template<class> class B> bool Compatible(const View<A> & a, const View<B> & b);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c);

        Checks three image views on compatibility (the images must have the same size and pixel format).

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] c - a third image.
        \return - a result of checking.
    */
    template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);

        Checks four image views on compatibility (the images must have the same size and pixel format).

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] c - a third image.
        \param [in] d - a fourth image.
        \return - a result of checking.
    */
    template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d);

    /*! @ingroup cpp_view_functions

        \fn template <template<class> class A> bool Compatible(const View<A> & a, const View<A> & b, const View<A> & c, const View<A> & d, const View<A> & e);

        Checks five image views on compatibility (the images must have the same size and pixel format).

        \param [in] a - a first image.
        \param [in] b - a second image.
        \param [in] c - a third image.
        \param [in] d - a fourth image.
        \param [in] e - a fifth image.
        \return - a result of checking.
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
    /*! \endcond */

#ifdef SIMD_OPENCV_ENABLE
    template <template<class> class A> SIMD_INLINE View<A>::View(const cv::Mat & mat)
        : width(mat.cols)
        , height(mat.rows)
        , stride(mat.step[0])
        , format(OcvTo(mat.type()))
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

    template <template<class> class A> SIMD_INLINE View<A>& View<A>::operator = (View<A>&& view)
    {
        if (this != &view)
        {
            Swap(view);
            view.Clear();
        }
        return *this;
    }
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
        case Gray8:     return CV_8UC1;
        case Uv16:      return CV_8UC2;
        case Bgr24:     return CV_8UC3;
        case Bgra32:    return CV_8UC4;
        case Int16:     return CV_16SC1;
        case Int32:     return CV_32SC1;
        case Float:     return CV_32FC1;
        case Double:    return CV_64FC1;
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
        *(void**)&data = nullptr;
        _owner = false;
        *(size_t*)&width = 0;
        *(size_t*)&height = 0;
        *(ptrdiff_t *)&stride = 0;
        *(Format*)&format = Format::None;
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
