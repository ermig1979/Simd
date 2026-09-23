/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2014-2019 Antonenka Mikhail.
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
#ifndef __SimdShift_hpp__
#define __SimdShift_hpp__

#include "Simd/SimdLib.hpp"

#include <vector>
#include <float.h>

namespace Simd
{
    /*! @ingroup cpp_shift

        \short Estimates the translation of a Gray8 region relative to a background.

        ShiftDetector<A> finds the translation that moves a rectangle onto the
        background window matching a current Gray8 image. The usual type is
        ShiftDetector<Simd::Allocator>. View is Simd::View<A>, so the background
        and the current image use that allocator. Point is an integer pixel
        shift and also the frame size (x is the width, y is the height).
        FPoint is the sub-pixel shift. Rect is the half-open correlation window
        [left, right) x [top, bottom).

        The structure owns a context created by ::SimdShiftDetectorInitBuffers
        and released by ::SimdRelease. The context keeps a background pyramid
        and a current pyramid. Each upper level is the 2x reduction of the
        level below (SimdReduce2x2). levelCount includes the full-resolution
        level. ShiftDetectorFileSpecialTest uses 4 levels.
        ShiftDetectorRandSpecialTest uses 6 levels for a 1920x1080 frame.

        Call InitBuffers, then SetBackground, then Estimate. Estimate returns
        false while the context is missing, while the window area is below
        regionAreaMin, or while the coarse-to-fine search cannot place the
        window. After a true result, Shift(), RefinedShift(), Stability() and
        Correlation() read the base pyramid level. The integer shift is the
        translation from the initial rectangle to the matching background
        window. The match is region.Shifted(Shift()), so the current image
        lies at (region.left + shift.x, region.top + shift.y).
        ShiftDetectorFileSpecialTest loads a Gray8 image, takes the current
        view from background.Region(region.Shifted(10, 10)) with
        region = Rect(64, 64, 192, 192), and calls Estimate(current, region, 32).
        That shift is (10, 10). ShiftDetectorRandSpecialTest builds a
        1920x1080 Gray8 frame, places a 256-pixel ring, shrinks the window
        with AddBorder(-32), and calls
        Estimate(background.Region(region), region.Shifted(ss), ms * 2)
        with ms = region.Width() / 4. The first argument is the background
        crop of the unshifted window. The second argument is that window
        translated by ss, so the matching translation is -ss.

        The initial rectangle lies inside the background: Left() >= 0,
        Top() >= 0, Right() <= frame width and Bottom() <= frame height.
        current.Size() equals region.Size(). The search may then move the
        window. A candidate that leaves more than half of the window outside
        the frame makes Estimate return false. A smaller hidden part is scored
        with hiddenAreaPenalty.

        TextureGray compares the gray pixels. TextureGrad compares the
        saturated sum of absolute X and Y gradients from
        Simd::AbsGradientSaturatedSum (border pixels of that texture are 0).
        AbsDifference minimizes the mean absolute difference and reports
        correlation 1 - difference/255. SquaredDifference minimizes the mean
        squared difference and reports correlation 1 - sqrt(difference)/255.
        Equal images give correlation 1. The file test keeps the defaults
        TextureGray and AbsDifference. The random test passes TextureGray and
        SquaredDifference.

        The search starts at the coarsest level whose window area is still at
        least regionAreaMin. The shift begins at (0, 0), and shift.x and
        shift.y are doubled on the way to the next finer level. At level i a
        candidate is limited to max((maxShift >> i) + 1, 2) pixels of that
        level on each axis. Visiting a candidate outside that limit makes
        Estimate return false. The coarsest level hill-climbs from (0, 0)
        for up to that many steps. Each finer level hill-climbs for up to 3
        steps around the doubled shift. Estimate returns false when the
        center is still moving after those steps.

        Using example:
        \code
        #include "Simd/SimdShift.hpp"
        #include <iostream>

        int main()
        {
            typedef Simd::ShiftDetector<Simd::Allocator> ShiftDetector;

            ShiftDetector::View background;
            background.Load("../../data/image/face/lena.pgm");

            ShiftDetector detector;
            detector.InitBuffers(background.Size(), 4, ShiftDetector::TextureGray, ShiftDetector::AbsDifference);
            detector.SetBackground(background);

            ShiftDetector::Rect region(64, 64, 192, 192);
            ShiftDetector::View current = background.Region(region.Shifted(10, 10));

            if (detector.Estimate(current, region, 32))
            {
                ShiftDetector::Point shift = detector.Shift();
                ShiftDetector::FPoint refined = detector.RefinedShift();
                std::cout << "Shift = (" << shift.x << ", " << shift.y << "). " << std::endl;
                std::cout << "Refined = (" << refined.x << ", " << refined.y << "). " << std::endl;
                std::cout << "Stability = " << detector.Stability()
                    << ", correlation = " << detector.Correlation() << ". " << std::endl;
            }
            else
                std::cout << "Can't find shift for current image!" << std::endl;

            return 0;
        }
        \endcode

        \note This is a C++ wrapper around ::SimdShiftDetectorInitBuffers,
              ::SimdShiftDetectorSetBackground, ::SimdShiftDetectorEstimate and
              ::SimdShiftDetectorGetShift. The Python class Simd.ShiftingDetector
              calls those C functions with the same texture, difference, level
              count, penalty and area arguments.
    */
    template <template<class> class A>
    struct ShiftDetector
    {
        typedef A<uint8_t> Allocator; /*!< Allocator of the caller's Gray8 views. Simd::Allocator is the type used by the tests. */
        typedef Simd::View<A> View; /*!< Gray8 image. SetBackground requires this format. The frame size is View::Size(). */
        typedef Simd::Point<ptrdiff_t> Point; /*!< Integer shift (x, y), or a frame size (width, height). Point() is an unset frame size. */
        typedef Simd::Point<double> FPoint; /*!< Sub-pixel shift returned by RefinedShift(). TestShift wraps an integer point in this type before SquaredDistance. */
        typedef Rectangle<ptrdiff_t> Rect; /*!< Half-open correlation window [left, right) x [top, bottom). */

        /*!
            \enum TextureType

            Texture stored in the background and current pyramids.

            InitBuffers keeps this choice for the life of the context. A
            different value recreates the context. TextureGray is the default
            and is the texture used by both special tests.
            ::SimdShiftDetectorTextureGray and ::SimdShiftDetectorTextureGrad
            are the C API names of these values. The Python enumeration
            Simd.ShiftDetectorTexture uses Gray and Grad in the same order.
        */
        enum TextureType
        {
            /*!
                Compare the original Gray8 pixels.

                SetBackground with makeCopy == false stores a view of the
                caller's background at pyramid level 0. The file test and the
                random test both use TextureGray.
            */
            TextureGray,
            /*!
                Compare Simd::AbsGradientSaturatedSum of the Gray8 image.

                For an inner pixel the texture is min(|src[x+1,y] - src[x-1,y]|
                + |src[x,y+1] - src[x,y-1]|, 255). Border pixels are 0.
                SetBackground always writes this texture into the owned pyramid.
            */
            TextureGrad,
        };

        /*!
            \enum DifferenceType

            Difference minimized by Estimate, and the source of Correlation().

            The stored difference is the sum of per-pixel differences divided
            by the area of the compared window. A partly hidden window is then
            multiplied by the hidden-area factor from Estimate. AbsDifference
            is the default. The random test passes SquaredDifference.
            ::SimdShiftDetectorAbsDifference and
            ::SimdShiftDetectorSquaredDifference are the C API names. The
            Python enumeration Simd.ShiftDetectorDifference uses Abs and
            Squared in the same order.
        */
        enum DifferenceType
        {
            /*!
                Mean absolute difference.

                The search uses Simd::AbsDifferenceSum and, for a fully visible
                3x3 neighborhood, Simd::AbsDifferenceSums3x3. Correlation() is
                1 - difference/255.
            */
            AbsDifference,
            /*!
                Mean squared difference.

                The search uses Simd::SquaredDifferenceSum.
                ShiftDetectorRandSpecialTest selects this metric.
                Correlation() is 1 - sqrt(difference)/255.
            */
            SquaredDifference,
        };

        /*!
            Creates an empty detector.

            The texture is TextureGray, the difference is AbsDifference, the
            level count is 0 and the frame size is Point(). The context is
            empty, so Estimate returns false until InitBuffers.
        */
        ShiftDetector()
            : _textureType(TextureGray)
            , _differenceType(AbsDifference)
            , _levelCount(0)
            , _context(NULL)
        {

        }

        /*!
            Releases the shift-detector context.

            Calls ::SimdRelease when InitBuffers has created a context.
        */
        ~ShiftDetector()
        {
            if (_context)
                SimdRelease(_context);
        }

        /*!
            Creates the pyramids for a background of the given size.

            Calls ::SimdShiftDetectorInitBuffers. The frame size is the
            background size: x is the width and y is the height, the same
            point as View::Size(). levelCount is the number of pyramid levels
            including the full-resolution level. The file test passes
            background.Size() and 4. The random test passes background.Size(),
            6, TextureGray and SquaredDifference.

            A second call with the same frame size, level count, texture and
            difference keeps the existing context and the background already
            stored in it. Any other combination releases that context and
            creates a new one. SetBackground is required again after a new
            context is created.

            \param [in] frameSize - background size in pixels. x is the width, y is the height.
            \param [in] levelCount - number of pyramid levels, including level 0. Each next level is half the size.
            \param [in] textureType - texture compared by Estimate. The default is TextureGray.
            \param [in] differenceType - difference minimized by Estimate. The default is AbsDifference.
        */
        void InitBuffers(const Point & frameSize, size_t levelCount, TextureType textureType = TextureGray, DifferenceType differenceType = AbsDifference)
        {
            if (_levelCount == levelCount && _frameSize == frameSize && _textureType == textureType && _differenceType == differenceType && _context)
                return;

            _levelCount = levelCount;
            _frameSize = frameSize;
            _textureType = textureType;
            _differenceType = differenceType;

            if(_context)
                SimdRelease(_context);
            _context = SimdShiftDetectorInitBuffers(_frameSize.x, _frameSize.y, _levelCount, (SimdShiftDetectorTextureType)textureType, (SimdShiftDetectorDifferenceType)differenceType);
        }

        /*!
            Builds the background pyramid from a Gray8 image.

            The image size equals the frame size passed to InitBuffers, and
            background.format is View::Gray8. Both special tests pass the same
            view that was measured by InitBuffers, with the default copy.

            TextureGray and makeCopy == true copy the pixels into pyramid
            level 0. TextureGray and makeCopy == false make level 0 a view of
            background.data. That buffer then stays alive and unchanged until
            the next SetBackground or until the destructor. Upper levels are
            still owned images. TextureGrad writes
            Simd::AbsGradientSaturatedSum into the owned level 0. The call
            then builds the upper levels with Simd::Build and SimdReduce2x2.

            \param [in] background - Gray8 background of the size passed to InitBuffers.
            \param [in] makeCopy - copy a TextureGray background into the context. The default is true.
        */
        void SetBackground(const View & background, bool makeCopy = true)
        {
            assert(_levelCount && _frameSize == background.Size() && background.format == View::Gray8);

            if (_context)
                SimdShiftDetectorSetBackground(_context, background.data, background.stride, makeCopy ? SimdTrue : SimdFalse);
        }

        /*!
            Minimal area of a correlation window, in pixels.

            Estimate uses this value when regionAreaMin is omitted. The file
            test and the random test rely on that default. A window with a
            smaller area makes Estimate return false. A pyramid level is used
            only while its window area is still at least this value. The
            Python wrapper passes 25, the same constant.
        */
        static const ptrdiff_t REGION_CORRELATION_AREA_MIN = 25;

        /*!
            Estimates the translation of the current image relative to the background.

            Calls ::SimdShiftDetectorEstimate. current is Gray8 and
            current.Size() equals region.Size(). region is the initial
            position of that image in background coordinates. The returned
            shift moves this rectangle onto the match:
            background.Region(region.Shifted(shift)) is the window that
            corresponds to current. The file test builds current as
            background.Region(region.Shifted(10, 10)) and passes the
            unshifted region, so the integer shift is (10, 10). The random
            test passes the unshifted crop as current and region.Shifted(ss)
            as the initial window.

            The initial rectangle lies inside the frame. Estimate returns
            false when the context is missing, when region.Area() is below
            regionAreaMin, or when the search leaves the per-level shift
            limit. It also returns false when a candidate keeps less than
            half of the window inside the frame. A debug build requires the
            frame size to differ from Point(), which is the state left by the
            default constructor.

            maxShift is the largest absolute translation, in full-resolution
            pixels, accepted along each axis. At pyramid level i the limit in
            that level's pixels is max((maxShift >> i) + 1, 2). The file test
            passes the integer overload, which uses the same limit on both
            axes. This overload is the one that can give X and Y different
            limits. The C API receives them as maxShiftX and maxShiftY.

            hiddenAreaPenalty weights a window that crosses the frame border.
            The visible part is the shifted window clipped by the frame and
            moved back. Its mean difference is multiplied by
            1 + (initialArea - visibleArea) * hiddenAreaPenalty / initialArea.
            The default 0, used by both special tests, leaves the mean
            difference unchanged. A positive penalty lowers Correlation()
            because the stored difference includes this factor.

            regionAreaMin selects the pyramid levels. The base level and every
            coarser level whose area is still at least this value take part
            in the search. The search starts at the coarsest of those levels.

            \param [in] current - Gray8 image of the same size as region.
            \param [in] region - initial half-open window inside the background. The shift is applied to this rectangle.
            \param [in] maxShift - maximal absolute shift along X (maxShift.x) and Y (maxShift.y), in full-resolution pixels.
            \param [in] hiddenAreaPenalty - extra weight of the hidden fraction of the window. The default is 0.
            \param [in] regionAreaMin - minimal window area. The default is REGION_CORRELATION_AREA_MIN (25).
            \return true when a shift was found. Read Shift(), RefinedShift(), Stability() and Correlation() after a true result.
        */
        bool Estimate(const View & current, const Rect & region, const Point & maxShift, double hiddenAreaPenalty = 0, ptrdiff_t regionAreaMin = REGION_CORRELATION_AREA_MIN)
        {
            assert(current.Size() == region.Size() && region.Area() > 0);
            assert(_frameSize != Point() && _frameSize.x >= (ptrdiff_t)current.width && _frameSize.y >= (ptrdiff_t)current.height);

            if (_context == 0 || region.Area() < regionAreaMin)
                return false;

            return SimdShiftDetectorEstimate(_context, current.data, current.stride, current.width, current.height, 
                region.left, region.top, maxShift.x, maxShift.y, &hiddenAreaPenalty, regionAreaMin) == SimdTrue;
        }

        /*!
            Estimates the translation with one limit for both axes.

            Forwards to Estimate(current, region, Point(maxShift, maxShift),
            hiddenAreaPenalty, regionAreaMin). ShiftDetectorFileSpecialTest
            calls Estimate(current, region, 32). ShiftDetectorRandSpecialTest
            calls Estimate(background.Region(region), region.Shifted(ss), ms * 2),
            where ms is region.Width() / 4.

            \param [in] current - Gray8 image of the same size as region.
            \param [in] region - initial half-open window inside the background.
            \param [in] maxShift - maximal absolute shift along both X and Y, in full-resolution pixels.
            \param [in] hiddenAreaPenalty - extra weight of the hidden fraction of the window. The default is 0.
            \param [in] regionAreaMin - minimal window area. The default is REGION_CORRELATION_AREA_MIN (25).
            \return true when a shift was found. Read Shift(), RefinedShift(), Stability() and Correlation() after a true result.
        */
        bool Estimate(const View & current, const Rect & region, int maxShift, double hiddenAreaPenalty = 0, ptrdiff_t regionAreaMin = REGION_CORRELATION_AREA_MIN)
        {
            return Estimate(current, region, Point(maxShift, maxShift), hiddenAreaPenalty, regionAreaMin);
        }

        /*!
            Returns the integer shift from the last successful Estimate.

            The point is the base-level translation in full-resolution pixels.
            region.Shifted(Shift()) is the matching background window. The
            file test prints this point. An empty detector returns Point().
            Read this value when Estimate has returned true.

            \return integer shift (x, y).
        */
        Point Shift() const
        {
            Point shift;
            if (_context)
                SimdShiftDetectorGetShift(_context, (ptrdiff_t*)&shift, NULL, NULL, NULL);
            return shift;
        }

        /*!
            Returns the sub-pixel shift from the last successful Estimate.

            The value is the integer shift plus an offset fitted from the 3x3
            difference neighborhood of the base level. The fit is a parabola
            on the averaged rows and columns of that neighborhood. When the
            minimum touches the border of the searched neighborhood, or a
            neighbor is missing, the fitted offset is (-1, -1) and Stability()
            is 0. TestShift wraps an integer point in FPoint and passes
            RefinedShift() to Simd::SquaredDistance. An empty detector
            returns FPoint().

            \return sub-pixel shift (x, y).
        */
        FPoint RefinedShift() const
        {
            FPoint refinedShift;
            if (_context)
                SimdShiftDetectorGetShift(_context, NULL, (double*)&refinedShift, NULL, NULL);
            return refinedShift;
        }

        /*!
            Returns the stability of the shift from the last successful Estimate.

            The value is 0 when the best difference sits on the border of the
            searched neighborhood, when a 3x3 neighbor is missing, or when the
            fitted sub-pixel offset lies outside [-0.75, 0.75]. Otherwise it
            is (average of five samples on the far side of the minimum -
            interpolated minimum) / that average. A value close to 1 is a
            sharp minimum. An empty detector returns 0.

            \return stability of the found shift.
        */
        double Stability() const
        {
            double stability = 0;
            if (_context)
                SimdShiftDetectorGetShift(_context, NULL, NULL, &stability, NULL);
            return stability;
        }

        /*!
            Returns the correlation of the best shift from the last successful Estimate.

            The correlation is taken from the best average difference at the
            base level. AbsDifference uses 1 - difference/255.
            SquaredDifference uses 1 - sqrt(difference)/255. Equal images give
            1. hiddenAreaPenalty is included in the stored difference, so a
            positive penalty on a partly hidden window lowers this value.
            An empty detector returns 0.

            \return correlation of the background and the current image.
        */
        double Correlation() const
        {
            double correlation = 0;
            if (_context)
                SimdShiftDetectorGetShift(_context, NULL, NULL, NULL, &correlation);
            return correlation;
        }

    private:
        Point _frameSize;
        size_t _levelCount;
        TextureType _textureType;
        DifferenceType _differenceType;
        void* _context;
    };
}
#endif
