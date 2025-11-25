/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar,
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

        \short ShiftDetector structure provides shift detection of given region at the image.

        Using example:
        \verbatim
        #include "Simd/SimdShift.hpp"
        #include <iostream>

        int main()
        {
            typedef Simd::ShiftDetector<Simd::Allocator> ShiftDetector;

            ShiftDetector::View background;
            background.Load("../../data/image/face/lena.pgm");

            ShiftDetector detector;

            detector.InitBuffers(background.Size(), 4);

            detector.SetBackground(background);

            ShiftDetector::Rect region(64, 64, 192, 192);

            ShiftDetector::View current = background.Region(region.Shifted(10, 10));

            if (detector.Estimate(current, region, 32))
                std::cout << "Shift = (" << detector.Shift().x << ", " << detector.Shift().y << "). " << std::endl;
            else
                std::cout << "Can't find shift for current image!" << std::endl;

            return 0;
        }
        \endverbatim
    */
    template <template<class> class A>
    struct ShiftDetector
    {
        typedef A<uint8_t> Allocator; /*!< Allocator type definition. */
        typedef Simd::View<A> View; /*!< An image type definition. */
        typedef Simd::Point<ptrdiff_t> Point; /*!< A point with integer coordinates. */
        typedef Simd::Point<double> FPoint; /*!< A point with float point coordinates. */
        typedef Rectangle<ptrdiff_t> Rect; /*!< A rectangle type definition. */

        /*!
            \enum TextureType

            Describes types of texture which used to find correlation between background and current image.
        */
        enum TextureType
        {
            /*!
                Original grayscale image.
            */
            TextureGray,
            /*!
                Saturated sum of absolute gradients along X and Y axes.
            */
            TextureGrad,
        };

        /*!
            \enum DifferenceType

            Describes types of function which used to find correlation between background and current image.
        */
        enum DifferenceType
        {
            /*!
                Sum of absolute differences of points of two images.
            */
            AbsDifference,
            /*!
                Sum of squared differences of points of two images.
            */
            SquaredDifference,
        };

        /*!
            Creates a new empty ShiftDetector structure.
        */
        ShiftDetector()
            : _textureType(TextureGray)
            , _differenceType(AbsDifference)
            , _levelCount(0)
            , _context(NULL)
        {

        }

        /*!
            A ShiftDetector destructor.
        */
        ~ShiftDetector()
        {
            if (_context)
                SimdRelease(_context);
        }

        /*!
            Initializes internal buffers of ShiftDetector structure. It allows it to work with image of given size.

            \param [in] frameSize - a size of background image.
            \param [in] levelCount - number of levels in the internal image pyramids used to find shift.
            \param [in] textureType - type of textures used to detect shift.
            \param [in] differenceType - type of correlation functions used to detect shift.
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
            Sets a background image. Size of background image must be equal to frameSize (see function ShiftDetector::InitBuffers).

            \param [in] background - background image.
            \param [in] makeCopy - if true, copy of the background will be created.
        */
        void SetBackground(const View & background, bool makeCopy = true)
        {
            assert(_levelCount && _frameSize == background.Size() && background.format == View::Gray8);

            if (_context)
                SimdShiftDetectorSetBackground(_context, background.data, background.stride, makeCopy ? SimdTrue : SimdFalse);
        }

        static const ptrdiff_t REGION_CORRELATION_AREA_MIN = 25;

        /*!
            Estimates shift of current image relative to background image.

            \param [in] current - current image.
            \param [in] region - a region at the background where the algorithm start to search current image. Estimated shift is taken relative of the region.
            \param [in] maxShift - a 2D-point which characterizes maximal possible shift of the region (along X and Y axes).
            \param [in] hiddenAreaPenalty - a parameter used to restrict searching of the shift at the border of background image.
            \param [in] regionAreaMin - a parameter used to set minimal area of region use for shift estimation. By default is equal to 25.
            \return a result of shift estimation.
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
            Estimates shift of current image relative to background image.

            \param [in] current - current image.
            \param [in] region - a region at the background where the algorithm start to search current image. Estimated shift is taken relative of the region.
            \param [in] maxShift - a maximal distance which characterizes maximal possible shift of the region.
            \param [in] hiddenAreaPenalty - a parameter used to restrict searching of the shift at the border of background image.
            \param [in] regionAreaMin - a parameter used to set minimal area of region use for shift estimation. By default is equal to 25.
            \return a result of shift estimation.
        */
        bool Estimate(const View & current, const Rect & region, int maxShift, double hiddenAreaPenalty = 0, ptrdiff_t regionAreaMin = REGION_CORRELATION_AREA_MIN)
        {
            return Estimate(current, region, Point(maxShift, maxShift), hiddenAreaPenalty, regionAreaMin);
        }

        /*!
            Gets estimated integer shift of current image relative to background image.

            \return estimated integer shift.
        */
        Point Shift() const
        {
            Point shift;
            if (_context)
                SimdShiftDetectorGetShift(_context, (ptrdiff_t*)&shift, NULL, NULL, NULL);
            return shift;
        }

        /*!
            Gets refined (with sub-pixel accuracy) shift of current image relative to background image.

            \return refined shift with sub-pixel accuracy.
        */
        FPoint RefinedShift() const
        {
            FPoint refinedShift;
            if (_context)
                SimdShiftDetectorGetShift(_context, NULL, (double*)&refinedShift, NULL, NULL);
            return refinedShift;
        }

        /*!
            Gets a value which characterizes stability (reliability) of found shift.

            \return stability (reliability) of found shift.
        */
        double Stability() const
        {
            double stability = 0;
            if (_context)
                SimdShiftDetectorGetShift(_context, NULL, NULL, &stability, NULL);
            return stability;
        }

        /*!
            Gets the best correlation of background and current image.

            \return the best correlation of background and current image.
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
