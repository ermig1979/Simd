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
#ifndef __SimdPyramid_hpp__
#define __SimdPyramid_hpp__

#include "Simd/SimdView.hpp"

#include <vector>

namespace Simd
{
    /*! @ingroup cpp_pyramid

        \short Gray8 image pyramid: a base image and its successively halved levels.

        Pyramid<A> stores 8-bit gray views. Level 0 is the base, the largest
        image. Each following level is half of the previous one on both sides.
        The half-size is Simd::Scale: each side n becomes (n + 1) / 2, so an
        odd side rounds up. A 16x16 pyramid of 3 levels is 16x16, 8x8 and 4x4
        (TestCheckCpp). A 15x9 pyramid of 3 levels is 15x9, 8x5 and 4x3.
        Those sizes are the sizes written by Simd::ReduceGray, so Simd::Build
        fills the views allocated here.

        The template argument A is the allocator template of the levels.
        In-tree code uses Simd::Allocator. Every level is a View<A> of format
        Gray8. Recreate allocates the buffers and leaves the pixels
        uninitialized. Callers write level 0 and then fill the upper levels
        with Simd::Build, or they write every level with Simd::Fill.
        Simd::Copy copies a whole pyramid onto another pyramid of the same
        level count. Fill, Copy and Build are declared in SimdLib.hpp.

        Size() is the number of levels. The base image size is
        operator[](0).Size(). At(level) and operator[](level) return the same
        view. The index must be less than Size(). Top() is the last level,
        the smallest image.

        Motion::Detector keeps several of these pyramids. Calibration walks
        the frame size with Simd::Scale until the top level is smaller than
        the top-level size minimum or the estimated object area is below
        its minimum. The number of skipped fine levels is then clamped by
        Options::CalibrationScaleLevelMax (the default 3 is an 8-times
        downscale). The scaled pyramid is recreated at the original frame
        size. SetFrame converts the input frame into scaled[0] and calls
        Simd::Build(scaled, SimdReduce2x2). The texture pyramid then copies
        scaled.Top() into its own level 0 and is reduced with SimdReduce4x4.
        The ROI mask, difference image and segmentation mask are pyramids of
        the working frame size. The ROI mask is either resized from
        Model::mask into level 0 or filled with Simd::Fill and a polygon on
        level 0, then built with SimdReduce4x4. Options::DebugDrawLevel
        selects the pyramid level copied into the debug inset (difference,
        gray, dx or dy, chosen by Options::DebugDrawBottomRight).

        ShiftDetector::ShiftDetector recreates a background pyramid and a
        current pyramid at the background size. SetBackground writes level 0
        and calls Simd::Build with SimdReduce2x2. Estimate writes the current
        image into a region of current[0] and reduces that region into the
        upper levels. The search starts at the coarsest level and doubles the
        shift on the way to the next finer level.

        Font::Resize builds a temporary pyramid when a glyph has to be
        reduced. Level 0 is a bilinear resize of the original glyph.
        Simd::Build with SimdReduce2x2 writes the reduced glyph at the last
        level, and that view is copied into the glyph cache.

        TestCheckCpp creates two pyramids of 16x16 and 3 levels, fills the
        first with the byte 1, builds it with SimdReduce2x2 and copies it
        onto the second.

        Recreate on an existing pyramid returns immediately when level 0
        already has the requested size. The level count is not part of that
        comparison, so a second Recreate with the same base size keeps the
        old levels and their pixels. A default-constructed pyramid is empty:
        Size() is 0, and operator[], At and Top require at least one level.

        Using example:
        \code
        #include "Simd/SimdLib.hpp"

        int main()
        {
            typedef Simd::View<Simd::Allocator> View;
            typedef Simd::Pyramid<Simd::Allocator> Pyramid;
            typedef Simd::Point<ptrdiff_t> Point;

            View image(320, 240, View::Gray8);
            Pyramid pyramid(image.Size(), 4);
            Simd::Copy(image, pyramid[0]);
            Simd::Build(pyramid, SimdReduce2x2);

            View coarse = pyramid.Top();
            Point half = Simd::Scale(image.Size());

            Pyramid copy(image.Size(), pyramid.Size());
            Simd::Copy(pyramid, copy);
            return (int)pyramid.Size() + (int)coarse.width + (int)half.x;
        }
        \endcode

        \ref cpp_pyramid_functions.
    */
    template <template<class> class A> struct Pyramid
    {
        typedef A<uint8_t> Allocator; /*!< Allocator of the Gray8 levels. In-tree code passes Simd::Allocator. */

        /*!
            Creates an empty pyramid.

            Size() is 0. Recreate, or a constructor with a base size and a
            level count, allocates the levels. operator[], At and Top require
            a pyramid that already has levels.
        */
        Pyramid();

        /*!
            Creates a pyramid and allocates its levels.

            Equivalent to an empty pyramid followed by Recreate(size, levelCount).
            Level 0 has the given size and format Gray8. Each next level has
            size Simd::Scale of the previous level. The pixel buffers are
            allocated and left uninitialized. Motion texture features allocate
            through Recreate. Font::Resize and TestCheckCpp use the
            width/height constructor.

            \param [in] size - width and height of level 0.
            \param [in] levelCount - number of levels, including level 0.
        */
        Pyramid(const Point<ptrdiff_t> & size, size_t levelCount);

        /*!
            Creates a pyramid and allocates its levels.

            Forwards to Recreate(Point<ptrdiff_t>(width, height), levelCount).
            TestCheckCpp uses Pyramid(16, 16, 3): level 0 is 16x16, level 1
            is 8x8 and level 2 is 4x4. Font::Resize uses
            Pyramid(size, level + 1), where size is the current glyph size
            shifted left by level, so the last level is the reduced glyph.

            \param [in] width - width of level 0.
            \param [in] height - height of level 0.
            \param [in] levelCount - number of levels, including level 0.
        */
        Pyramid(size_t width, size_t height, size_t levelCount);

        /*!
            Allocates or replaces the pyramid levels.

            When the pyramid already has levels and size equals the size of
            level 0, the function returns and leaves the levels and their
            pixels unchanged. The level count is not compared.
            Motion::Detector::Calibrate returns before Recreate when the
            input frame size is unchanged, and recreates the pyramids after
            the frame size changes. The first Recreate on an empty pyramid
            always allocates.

            Otherwise the pyramid is resized to levelCount views. Level 0 is
            recreated as Gray8 of the given size. Each next level is Gray8 of
            Simd::Scale of the previous size. The new buffers are
            uninitialized. ShiftDetector::ShiftDetector calls the width/height
            overload for the background and current pyramids. Motion::Detector
            recreates the scaled pyramid at the original frame size with
            scaleLevel + 1 levels, and recreates the texture, difference,
            buffer, segmentation mask and ROI mask pyramids at the working
            frame size.

            \param [in] size - width and height of level 0.
            \param [in] levelCount - number of levels, including level 0.
        */
        void Recreate(Point<ptrdiff_t> size, size_t levelCount);

        /*!
            Allocates or replaces the pyramid levels.

            Forwards to Recreate(Point<ptrdiff_t>(width, height), levelCount).
            The same early return applies: an existing pyramid whose level 0
            already has this width and height is left unchanged.

            \param [in] width - width of level 0.
            \param [in] height - height of level 0.
            \param [in] levelCount - number of levels, including level 0.
        */
        void Recreate(size_t width, size_t height, size_t levelCount);

        /*!
            Returns the number of levels.

            This is the level count. The image size of the base is
            operator[](0).Size(). An empty pyramid returns 0. Simd::Fill,
            Simd::Copy and Simd::Build loop while the index is less than
            Size(). Simd::Copy requires both pyramids to have the same
            Size(). ShiftDetector counts levels of the current pyramid while
            the search rectangle still has enough area.

            \return the number of levels.
        */
        size_t Size() const;

        /*!
            Returns the Gray8 view at the given level.

            Level 0 is the base. The index must be less than Size(). At(level)
            returns the same view. Motion, ShiftDetector and Font index
            pyramids with this operator: scaled[0] receives the converted
            frame, roiMask[0] receives the ROI image, and the font copies
            pyramid[level] after Simd::Build. Simd::Fill(difference[i], 0)
            fills that one view.

            \param [in] level - a level index, from 0 up to Size() - 1.
            \return a reference to the Gray8 image at that level.
        */
        View<A> & operator [] (size_t level);

        /*!
            Returns the Gray8 view at the given level.

            Equivalent to the non-constant operator[]. Difference estimation
            reads texture levels through a constant pyramid. The index must
            be less than Size().

            \param [in] level - a level index, from 0 up to Size() - 1.
            \return a constant reference to the Gray8 image at that level.
        */
        const View<A> & operator [] (size_t level) const;

        /*!
            Returns the Gray8 view at the given level.

            Equivalent to operator[](level). Simd::Fill, Simd::Copy and
            Simd::Build in SimdLib.hpp call At to walk the levels.

            \param [in] level - a level index, from 0 up to Size() - 1.
            \return a reference to the Gray8 image at that level.
        */
        View<A> & At(size_t level);

        /*!
            Returns the Gray8 view at the given level.

            Equivalent to the constant operator[](level).

            \param [in] level - a level index, from 0 up to Size() - 1.
            \return a constant reference to the Gray8 image at that level.
        */
        const View<A> & At(size_t level) const;

        /*!
            Returns the last level, the smallest image.

            The pyramid must contain at least one level. For levels
            0 .. Size() - 1 this is operator[](Size() - 1). Motion copies
            scaled.Top() into level 0 of the texture pyramid: that view is
            the working resolution after the calibration downscale.
            Font::Resize reads the reduced glyph from the last level, the
            same view Top() returns.

            \return a reference to the smallest Gray8 level.
        */
        View<A> & Top();

        /*!
            Returns the last level, the smallest image.

            Equivalent to the non-constant Top(). The pyramid must contain
            at least one level.

            \return a constant reference to the smallest Gray8 level.
        */
        const View<A> & Top() const;

        /*!
            Exchanges the levels of two pyramids.

            Both level vectors are swapped. Each pyramid then has the other
            pyramid's level count, image sizes and pixel buffers. An empty
            pyramid can be exchanged with a pyramid that already has levels.

            \param [in, out] pyramid - a pyramid whose levels are exchanged with this one.
        */
        void Swap(Pyramid<A> & pyramid);

    private:
        std::vector< View<A> > _views;
    };

    /*! @ingroup cpp_pyramid_functions

        \fn Point<ptrdiff_t> Scale(Point<ptrdiff_t> size, int scale = 2);

        \short Halves an image size one or more times.

        Each halving replaces a side n with (n + 1) >> 1, which is
        (n + 1) / 2. Odd sides round up: 15 becomes 8, 9 becomes 5, and 1
        stays 1. The function receives size by value and returns a new point.
        A right shift of Point truncates both coordinates; Scale adds 1
        before the shift, and that is the size Simd::ReduceGray writes.
        Pyramid::Recreate calls Scale with the default between consecutive
        levels.

        \a scale tells how many times the size can be halved. The default
        is 2, which halves each side once. The value has to be a positive
        power of two: every step requires the remaining scale to be even
        and then halves it. A scale less than or equal to 1 returns size
        unchanged. A scale of 4 halves the size twice.

        Motion::Detector calls Scale(size) while it counts how many levels
        still fit above the top-level size minimum, and calls it again to
        turn the original frame size into the working frame size. That
        working size is the base of the texture, difference and ROI pyramids.

        \param [in] size - width and height of the image to scale.
        \param [in] scale - a positive power of two. The default value 2 halves each side once.
        \return the scaled size.
    */
    Point<ptrdiff_t> Scale(Point<ptrdiff_t> size, int scale = 2);

    //-------------------------------------------------------------------------

    // struct Pyramid implementation:

    template <template<class> class A>
    SIMD_INLINE Pyramid<A>::Pyramid()
    {
    }

    template <template<class> class A>
    SIMD_INLINE Pyramid<A>::Pyramid(const Point<ptrdiff_t> & size, size_t levelCount)
    {
        Recreate(size, levelCount);
    }

    template <template<class> class A>
    SIMD_INLINE Pyramid<A>::Pyramid(size_t width, size_t height, size_t levelCount)
    {
        Recreate(width, height, levelCount);
    }

    template <template<class> class A>
    SIMD_INLINE void Pyramid<A>::Recreate(Point<ptrdiff_t> size, size_t levelCount)
    {
        if (_views.size() && size == _views[0].Size())
            return;
        _views.resize(levelCount);
        for (size_t level = 0; level < levelCount; ++level)
        {
            _views[level].Recreate(size, View<A>::Gray8);
            size = Scale(size);
        }
    }

    template <template<class> class A>
    SIMD_INLINE void Pyramid<A>::Recreate(size_t width, size_t height, size_t levelCount)
    {
        Recreate(Point<ptrdiff_t>(width, height), levelCount);
    }

    template <template<class> class A>
    SIMD_INLINE size_t Pyramid<A>::Size() const
    {
        return _views.size();
    }

    template <template<class> class A>
    SIMD_INLINE View<A> & Pyramid<A>::operator [] (size_t level)
    {
        return _views[level];
    }

    template <template<class> class A>
    SIMD_INLINE const View<A> & Pyramid<A>::operator [] (size_t level) const
    {
        return _views[level];
    }

    template <template<class> class A>
    SIMD_INLINE View<A> & Pyramid<A>::At(size_t level)
    {
        return _views[level];
    }

    template <template<class> class A>
    SIMD_INLINE const View<A> & Pyramid<A>::At(size_t level) const
    {
        return _views[level];
    }

    template <template<class> class A>
    SIMD_INLINE View<A> & Pyramid<A>::Top()
    {
        return _views.back();
    }

    template <template<class> class A>
    SIMD_INLINE const View<A> & Pyramid<A>::Top() const
    {
        return _views.back();
    }

    template <template<class> class A>
    SIMD_INLINE void Pyramid<A>::Swap(Pyramid & pyramid)
    {
        _views.swap(pyramid._views);
    }

    // Pyramid utilities implementation:

    SIMD_INLINE Point<ptrdiff_t> Scale(Point<ptrdiff_t> size, int scale)
    {
        while (scale > 1)
        {
            assert(scale % 2 == 0);
            size.x = (size.x + 1) >> 1;
            size.y = (size.y + 1) >> 1;
            scale >>= 1;
        }
        return size;
    }
}

#endif//__SimdPyramid_hpp__
