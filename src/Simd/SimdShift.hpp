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
            Initializes internal buffers of ShiftDetector structure. It allows it to work with image of given size.

            \param [in] frameSize - a size of background image.
            \param [in] levelCount - number of levels in the internal image pyramids used to find shift.
            \param [in] textureType - type of textures used to detect shift.
            \param [in] differenceType - type of correlation functions used to detect shift.
        */
        void InitBuffers(const Point & frameSize, size_t levelCount, TextureType textureType = TextureGray, DifferenceType differenceType = AbsDifference)
        {
            if (_background.Size() && _background.Size() == levelCount && _background[0].Size() == frameSize)
                return;

            _textureType = textureType;
            _differenceType = differenceType;
            _background.Recreate(frameSize, levelCount);
            _current.Recreate(frameSize, levelCount);
        }

        /*!
            Sets a background image. Size of background image must be equal to frameSize (see function ShiftDetector::InitBuffers).

            \param [in] background - background image.
            \param [in] makeCopy - if true, copy of the background will be created.
        */
        void SetBackground(const View & background, bool makeCopy = true)
        {
            assert(_background.Size() && _background[0].Size() == background.Size() && background.format == _background[0].format);

            if (_textureType == TextureGray)
            {
                if (makeCopy)
                {
                    Simd::Copy(background, _background[0]);
                }
                else
                {
                    _background[0].Clear();
                    _background[0] = View(background.width, background.height, background.stride, background.format, background.data);
                }
            }
            else if (_textureType == TextureGrad)
            {
                Simd::AbsGradientSaturatedSum(background, _background[0]);
            }
            else
            {
                assert(0);
            }
            Build(_background, ::SimdReduce2x2);
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
            assert(_current.Size() && _current[0].width >= current.width && _current[0].height >= current.height);

            if (region.Area() < regionAreaMin)
                return false;

            InitLevels(region, maxShift, regionAreaMin);
            SetCurrent(current, region);

            Point shift;
            for (ptrdiff_t i = _levels.size() - 1; i >= 0; i--)
            {
                shift.x *= 2;
                shift.y *= 2;
                if (!SearchLocalMin(_levels[i], shift, hiddenAreaPenalty))
                    return false;
                shift = _levels[i].shift;
            }
            return true;
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
            return _levels[0].shift;
        }

        /*!
            Gets proximate (with sub-pixel accuracy) shift of current image relative to background image.

            \return proximate shift with sub-pixel accuracy.
        */
        FPoint ProximateShift() const
        {
            return FPoint(_levels[0].shift) + _levels[0].differences.Refinement();
        }

        /*!
            Gets a value which characterizes stability (reliability) of found shift.

            \return stability (reliability) of found shift.
        */
        double Stability() const
        {
            return _levels[0].differences.Stability();
        }

        /*!
            Gets the best correlation of background and current image.

            \return the best correlation of background and current image.
        */
        double Correlation() const
        {
            double difference = _levels[0].differences.At(_levels[0].shift);
            if (_differenceType == AbsDifference)
                return 1.0 - difference / 255;
            else
                return 1.0 - ::sqrt(difference) / 255;
        }


    private:
        typedef Simd::Pyramid<A> Pyramid;

        Pyramid _background;
        Pyramid _current;

        struct Differences
        {
            void Init(const Point & neighborhood, const Point & origin = Point())
            {
                _neighborhood = neighborhood;
                _origin = origin;
                _size = 2 * _neighborhood + Point(1, 1);
                _table.resize(_size.x*_size.y);
                std::fill(_table.begin(), _table.end(), DBL_MAX);
            }

            double & At(const Point & shift)
            {
                return _table[Index(shift.x, shift.y)];
            }

            const double & At(const Point & shift) const
            {
                return _table[Index(shift.x, shift.y)];
            }

            bool Empty(const Point & shift) const
            {
                return (_table[Index(shift.x, shift.y)] == DBL_MAX);
            }

            FPoint Refinement() const
            {
                ptrdiff_t minX = 0, minY = 0;
                double minValue = DBL_MAX;
                for (ptrdiff_t y = 0; y < _size.y; y++)
                {
                    for (ptrdiff_t x = 0; x < _size.x; x++)
                    {
                        size_t offset = y*_size.x + x;
                        double value = _table[offset];
                        if (value < minValue)
                        {
                            minX = x;
                            minY = y;
                            minValue = value;
                        }
                    }
                }
                if (minX == 0 || minY == 0 || minX == _size.x - 1 || minY == _size.y - 1)
                {
                    return FPoint(-1, -1);
                }

                double z[3][3];
                for (ptrdiff_t y = 0; y < 3; y++)
                {
                    for (ptrdiff_t x = 0; x < 3; x++)
                    {
                        ptrdiff_t offset = (minY - 1 + y)*_size.x + minX - 1 + x;
                        double value = _table[offset];
                        if (value == DBL_MAX)
                            return FPoint(-1, -1);
                        z[y][x] = value;
                    }
                }

                double zx[3], zy[3];
                for (ptrdiff_t i = 0; i < 3; i++)
                {
                    zy[i] = (z[i][0] + z[i][1] + z[i][2]) / 3;
                    zx[i] = (z[0][i] + z[1][i] + z[2][i]) / 3;
                }
                return FPoint((zx[0] - zx[2]) / 2 / (zx[0] + zx[2] - 2 * zx[1]), (zy[0] - zy[2]) / 2 / (zy[0] + zy[2] - 2 * zy[1]));
            }

            double Stability() const
            {
                ptrdiff_t minX = 0, minY = 0;
                double minValue = DBL_MAX;
                for (ptrdiff_t y = 0; y < _size.y; y++)
                {
                    for (ptrdiff_t x = 0; x < _size.x; x++)
                    {
                        ptrdiff_t offset = y*_size.x + x;
                        double value = _table[offset];
                        if (value < minValue)
                        {
                            minX = x;
                            minY = y;
                            minValue = value;
                        }
                    }
                }
                if (minX == 0 || minY == 0 || minX == _size.x - 1 || minY == _size.y - 1)
                {
                    return 0;
                }

                double z[3][3];
                for (ptrdiff_t y = 0; y < 3; y++)
                {
                    for (ptrdiff_t x = 0; x < 3; x++)
                    {
                        ptrdiff_t offset = (minY - 1 + y)*_size.x + minX - 1 + x;
                        double value = _table[offset];
                        if (value == DBL_MAX)
                            return 0;
                        z[y][x] = value;
                    }
                }
                FPoint refinement = Refinement();

                static const double maxRefinement = 0.75;//0.5 + 0.25 (permissible error).
                if (!std::isfinite(refinement.x) || !std::isfinite(refinement.y) || std::isnan(refinement.x) || std::isnan(refinement.y) ||
                    refinement.x < -maxRefinement || refinement.x > maxRefinement || refinement.y < -maxRefinement || refinement.y > maxRefinement)
                {
                    return 0;
                }

                double neighborhoodSum = 0, minSum = 0;
                if (refinement.x < 0)
                {
                    double dx = -refinement.x;
                    neighborhoodSum += z[0][2] + z[1][2] + z[2][2];
                    if (refinement.y < 0)
                    {
                        double dy = -refinement.y;
                        neighborhoodSum += z[2][0] + z[2][1];
                        minSum = (1 - dx)*(1 - dy)*z[1][1] + dx*dy*z[0][0] + (1 - dx)*dy*z[0][1] + dx*(1 - dy)*z[1][0];
                    }
                    else
                    {
                        double dy = refinement.y;
                        neighborhoodSum += z[0][0] + z[0][1];
                        minSum = (1 - dx)*(1 - dy)*z[1][1] + dx*dy*z[2][0] + (1 - dx)*dy*z[2][1] + dx*(1 - dy)*z[1][0];
                    }
                }
                else
                {
                    double dx = refinement.x;
                    neighborhoodSum += z[0][0] + z[1][0] + z[2][0];
                    if (refinement.y < 0)
                    {
                        double dy = -refinement.y;
                        neighborhoodSum += z[2][1] + z[2][2];
                        minSum = (1 - dx)*(1 - dy)*z[1][1] + dx*dy*z[0][2] + (1 - dx)*dy*z[0][1] + dx*(1 - dy)*z[1][2];
                    }
                    else
                    {
                        double dy = refinement.y;
                        neighborhoodSum += z[0][1] + z[0][2];
                        minSum = (1 - dx)*(1 - dy)*z[1][1] + dx*dy*z[2][2] + (1 - dx)*dy*z[2][1] + dx*(1 - dy)*z[1][2];
                    }
                }
                double averageNeighborhood = neighborhoodSum / 5;
                return (averageNeighborhood - minSum) / averageNeighborhood;
            }

        private:
            std::vector<double> _table;
            Point _size;
            Point _neighborhood;
            Point _origin;

            size_t Index(size_t x, size_t y) const
            {
                return (y - _origin.y + _neighborhood.y)*_size.x + x - _origin.x + _neighborhood.x;
            }
        };

        struct Level
        {
            Point neighborhood;
            Point maxShift;
            Rect buildRegion;
            Rect searchRegion;
            Differences differences;
            View background;
            View current;
            Point shift;
        };
        typedef std::vector<Level> Levels;
        Levels _levels;
        TextureType _textureType;
        DifferenceType _differenceType;

        SIMD_INLINE size_t AlignHi(size_t size, size_t align)
        {
            return (size + align - 1) & ~(align - 1);
        }

        SIMD_INLINE size_t AlignLo(size_t size, size_t align) const
        {
            return size & ~(align - 1);
        }

        void InitLevels(const Rect & region, const Point & maxShift, ptrdiff_t regionAreaMin)
        {
            assert(region.Left() >= 0 && region.Top() >= 0 && region.Right() <= _current[0].Size().x && region.Bottom() <= _current[0].Size().y);

            size_t levelCount = 0;
            for (size_t i = 0; i < _current.Size(); ++i)
            {
                Rect rect(region.left >> i, region.top >> i, region.right >> i, region.bottom >> i);
                if (rect.Area() >= regionAreaMin)
                    levelCount = i + 1;
            }
            assert(levelCount);

            const ptrdiff_t maxShiftMin = 2;

            _levels.resize(levelCount);
            size_t buildRegionAlign = size_t(1) << (levelCount - 1);
            for (size_t i = 0; i < _levels.size(); ++i)
            {
                _levels[i].maxShift.x = std::max<ptrdiff_t>((maxShift.x >> i) + 1, maxShiftMin);
                _levels[i].maxShift.y = std::max<ptrdiff_t>((maxShift.y >> i) + 1, maxShiftMin);

                _levels[i].neighborhood.x = (i == _levels.size() - 1 ? _levels[i].maxShift.x : maxShiftMin + 1);
                _levels[i].neighborhood.y = (i == _levels.size() - 1 ? _levels[i].maxShift.y : maxShiftMin + 1);

                _levels[i].searchRegion.left = region.left >> i;
                _levels[i].searchRegion.top = region.top >> i;
                _levels[i].searchRegion.right = region.right >> i;
                _levels[i].searchRegion.bottom = region.bottom >> i;

                _levels[i].current = _current[i];
                _levels[i].background = _background[i];

                _levels[i].shift = Point(INT_MAX, INT_MAX);

                _levels[i].buildRegion.left = AlignLo(region.left, buildRegionAlign) >> i;
                _levels[i].buildRegion.top = AlignLo(region.top, buildRegionAlign) >> i;
                _levels[i].buildRegion.right = AlignHi(region.right, buildRegionAlign) >> i;
                _levels[i].buildRegion.bottom = AlignHi(region.bottom, buildRegionAlign) >> i;
            }
        }

        void SetCurrent(const View & current, const Rect & region)
        {
            if (_textureType == TextureGray)
                Simd::Copy(current, _current[0].Region(region).Ref());
            else if (_textureType == TextureGrad)
                Simd::AbsGradientSaturatedSum(current, _current[0].Region(region).Ref());
            else
                assert(0);
            for (size_t i = 1; i < _levels.size(); ++i)
                Simd::ReduceGray2x2(_current[i - 1].Region(_levels[i - 1].buildRegion), _current[i].Region(_levels[i].buildRegion).Ref());
        }

        double GetDifference(const View & background, const View & current, const Point & shift, const Rect & region)
        {
            View _background = background.Region(region.Shifted(shift));
            View _current = current.Region(region);
            uint64_t difference = 0;
            if (_differenceType == AbsDifference)
                Simd::AbsDifferenceSum(_background, _current, difference);
            else
                Simd::SquaredDifferenceSum(_background, _current, difference);
            return double(difference) / region.Area();
        }

        void GetDifferences3x3(Level & level, const Point & shift)
        {
            Point size = level.current.Size();
            Rect enlarged = level.searchRegion;
            enlarged.AddBorder(1);
            if (enlarged.left < 0 || enlarged.right > size.x || enlarged.top < 0 || enlarged.bottom > size.y)
                return;

            Rect shifted = enlarged.Shifted(shift);
            if (shifted.left < 0 || shifted.right > size.x || shifted.top < 0 || shifted.bottom > size.y)
                return;

            if (_differenceType == AbsDifference)
            {
                uint64_t differences[9];
                double area = (double)level.searchRegion.Area();
                Simd::AbsDifferenceSums3x3(level.current.Region(enlarged), level.background.Region(shifted), differences);
                for (ptrdiff_t dy = -1; dy <= 1; ++dy)
                    for (ptrdiff_t dx = -1; dx <= 1; ++dx)
                        level.differences.At(Point(shift.x + dx, shift.y + dy)) = differences[3 * dy + dx + 4] / area;
            }
            else
            {
                for (ptrdiff_t dy = -1; dy <= 1; ++dy)
                    for (ptrdiff_t dx = -1; dx <= 1; ++dx)
                        level.differences.At(Point(shift.x + dx, shift.y + dy)) = GetDifference(level.current, level.background, Point(shift.x + dx, shift.y + dy), level.searchRegion);
            }
        }

        bool SearchLocalMin(Level & level, const Point & shift, double hiddenAreaPenalty)
        {
            Differences & differences = level.differences;
            differences.Init(level.neighborhood, shift);
            Point minShift(shift);
            Point stageShift(shift);
            double minDifference = DBL_MAX;

            for (ptrdiff_t stage = 0, stageCount = std::max<ptrdiff_t>(level.neighborhood.x, level.neighborhood.y); stage < stageCount; ++stage)
            {
                if (stage == 0)
                    GetDifferences3x3(level, shift);

                for (ptrdiff_t dy = -1; dy <= 1; ++dy)
                {
                    for (ptrdiff_t dx = -1; dx <= 1; ++dx)
                    {
                        Point currentShift(stageShift.x + dx, stageShift.y + dy);

                        if (currentShift.x > level.maxShift.x || currentShift.y > level.maxShift.y ||
                            currentShift.x < -level.maxShift.x || currentShift.y < -level.maxShift.y)
                        {
                            return false;
                        }

                        double & diffAtCurShift = differences.At(currentShift);

                        if (diffAtCurShift == DBL_MAX)
                        {
                            Rect region = level.searchRegion.Shifted(currentShift);
                            region &= Rect(level.current.Size());
                            region.Shift(-currentShift);

                            ptrdiff_t initialArea = level.searchRegion.Area();
                            ptrdiff_t currentArea = region.Area();
                            if (currentArea * 2 < initialArea)
                                return false;

                            diffAtCurShift = GetDifference(level.background, level.current, currentShift, region) *
                                (1.0 + (initialArea - currentArea)*hiddenAreaPenalty / initialArea);
                        }

                        if (minDifference > diffAtCurShift)
                        {
                            minDifference = diffAtCurShift;
                            minShift = currentShift;
                        }
                    }
                }

                if (stageShift == minShift)
                {
                    level.shift = minShift;
                    return true;
                }

                stageShift = minShift;
            }

            return false;
        }
    };
}
#endif//__SimdShift_hpp__
