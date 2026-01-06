/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#define SIMD_STATIC
#include "Simd/SimdShiftDetector.h"

#include "Simd/SimdLib.hpp"

namespace Simd
{
    namespace Base
    {
        typedef Simd::Allocator<uint8_t> Allocator;
        typedef Simd::View<Simd::Allocator> View;
        typedef Simd::Point<ptrdiff_t> Point;
        typedef Simd::Point<double> FPoint;
        typedef Simd::Rectangle<ptrdiff_t> Rect;
        typedef Simd::Pyramid<Simd::Allocator> Pyramid;

        //-------------------------------------------------------------------------------------------------

        struct ShiftDetectorDifferences
        {
            void Init(const Point& neighborhood, const Point& origin = Point())
            {
                _neighborhood = neighborhood;
                _origin = origin;
                _size = 2 * _neighborhood + Point(1, 1);
                _table.resize(_size.x * _size.y);
                std::fill(_table.begin(), _table.end(), DBL_MAX);
            }

            double& At(const Point& shift)
            {
                return _table[Index(shift.x, shift.y)];
            }

            const double& At(const Point& shift) const
            {
                return _table[Index(shift.x, shift.y)];
            }

            bool Empty(const Point& shift) const
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
                        size_t offset = y * _size.x + x;
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
                        ptrdiff_t offset = (minY - 1 + y) * _size.x + minX - 1 + x;
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
                        ptrdiff_t offset = y * _size.x + x;
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
                        ptrdiff_t offset = (minY - 1 + y) * _size.x + minX - 1 + x;
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
                        minSum = (1 - dx) * (1 - dy) * z[1][1] + dx * dy * z[0][0] + (1 - dx) * dy * z[0][1] + dx * (1 - dy) * z[1][0];
                    }
                    else
                    {
                        double dy = refinement.y;
                        neighborhoodSum += z[0][0] + z[0][1];
                        minSum = (1 - dx) * (1 - dy) * z[1][1] + dx * dy * z[2][0] + (1 - dx) * dy * z[2][1] + dx * (1 - dy) * z[1][0];
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
                        minSum = (1 - dx) * (1 - dy) * z[1][1] + dx * dy * z[0][2] + (1 - dx) * dy * z[0][1] + dx * (1 - dy) * z[1][2];
                    }
                    else
                    {
                        double dy = refinement.y;
                        neighborhoodSum += z[0][1] + z[0][2];
                        minSum = (1 - dx) * (1 - dy) * z[1][1] + dx * dy * z[2][2] + (1 - dx) * dy * z[2][1] + dx * (1 - dy) * z[1][2];
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
                return (y - _origin.y + _neighborhood.y) * _size.x + x - _origin.x + _neighborhood.x;
            }
        };

        //-------------------------------------------------------------------------------------------------

        struct ShiftDetectorLevel
        {
            Point neighborhood;
            Point maxShift;
            Rect buildRegion;
            Rect searchRegion;
            ShiftDetectorDifferences differences;
            View background;
            View current;
            Point shift;
        };
        typedef std::vector<ShiftDetectorLevel> ShiftDetectorLevels;

        //-------------------------------------------------------------------------------------------------

        struct ShiftDetectorImpl
        {
            Pyramid _background;
            Pyramid _current;
            ShiftDetectorLevels _levels;
            SimdShiftDetectorTextureType _textureType;
            SimdShiftDetectorDifferenceType _differenceType;

            void InitLevels(const Rect& region, const Point& maxShift, ptrdiff_t regionAreaMin)
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

            void SetCurrent(const View& current, const Rect& region)
            {
                if (_textureType == SimdShiftDetectorTextureGray)
                    Simd::Copy(current, _current[0].Region(region).Ref());
                else if (_textureType == SimdShiftDetectorTextureGrad)
                    Simd::AbsGradientSaturatedSum(current, _current[0].Region(region).Ref());
                else
                    assert(0);
                for (size_t i = 1; i < _levels.size(); ++i)
                    Simd::ReduceGray2x2(_current[i - 1].Region(_levels[i - 1].buildRegion), _current[i].Region(_levels[i].buildRegion).Ref());
            }

            double GetDifference(const View& background, const View& current, const Point& shift, const Rect& region)
            {
                View _background = background.Region(region.Shifted(shift));
                View _current = current.Region(region);
                uint64_t difference = 0;
                if (_differenceType == SimdShiftDetectorAbsDifference)
                    Simd::AbsDifferenceSum(_background, _current, difference);
                else
                    Simd::SquaredDifferenceSum(_background, _current, difference);
                return double(difference) / region.Area();
            }

            void GetDifferences3x3(ShiftDetectorLevel& level, const Point& shift)
            {
                Point size = level.current.Size();
                Rect enlarged = level.searchRegion;
                enlarged.AddBorder(1);
                if (enlarged.left < 0 || enlarged.right > size.x || enlarged.top < 0 || enlarged.bottom > size.y)
                    return;

                Rect shifted = enlarged.Shifted(shift);
                if (shifted.left < 0 || shifted.right > size.x || shifted.top < 0 || shifted.bottom > size.y)
                    return;

                if (_differenceType == SimdShiftDetectorAbsDifference)
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

            bool SearchLocalMin(ShiftDetectorLevel& level, const Point& shift, double hiddenAreaPenalty)
            {
                ShiftDetectorDifferences& differences = level.differences;
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

                            double& diffAtCurShift = differences.At(currentShift);

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
                                    (1.0 + (initialArea - currentArea) * hiddenAreaPenalty / initialArea);
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

        //-------------------------------------------------------------------------------------------------

        ShiftDetector::ShiftDetector(size_t bkgWidth, size_t bkgHeight, size_t levelCount, SimdShiftDetectorTextureType textureType, SimdShiftDetectorDifferenceType differenceType)
            :_impl(NULL)
        {
            _impl = new ShiftDetectorImpl();
            _impl->_textureType = textureType;
            _impl->_differenceType = differenceType;
            _impl->_background.Recreate(bkgWidth, bkgHeight, levelCount);
            _impl->_current.Recreate(bkgWidth, bkgHeight, levelCount);
        }

        ShiftDetector::~ShiftDetector()
        {
            if (_impl)
                delete _impl;
        }

        void ShiftDetector::SetBackground(const uint8_t* bkg, size_t bkgStride, SimdBool makeCopy)
        {
            View background(_impl->_background[0].width, _impl->_background[0].height, bkgStride, View::Gray8, (uint8_t*)bkg);
            if (_impl->_textureType == SimdShiftDetectorTextureGray)
            {
                if (makeCopy)
                {
                    Simd::Copy(background, _impl->_background[0]);
                }
                else
                {
                    _impl->_background[0].Clear();
                    _impl->_background[0] = View(background.width, background.height, background.stride, background.format, background.data);
                }
            }
            else if (_impl->_textureType == SimdShiftDetectorTextureGrad)
            {
                Simd::AbsGradientSaturatedSum(background, _impl->_background[0]);
            }
            else
            {
                assert(0);
            }
            Simd::Build(_impl->_background, ::SimdReduce2x2);
        }

        SimdBool ShiftDetector::Estimate(const uint8_t* curr, size_t currStride, size_t currWidth, size_t currHeight,
            size_t initShiftX, size_t initShiftY, size_t maxShiftX, size_t maxShiftY, const double* hiddenAreaPenalty, ptrdiff_t regionAreaMin)
        {
            View current(currWidth, currHeight, currStride, View::Gray8, (uint8_t*)curr);
            Rect region(initShiftX, initShiftY, initShiftX + currWidth, initShiftY + currHeight);
            Point maxShift(maxShiftX, maxShiftY);

            if (_impl->_current[0].width < current.width || _impl->_current[0].height < current.height)
                return SimdFalse;
            if (region.Area() < regionAreaMin)
                return SimdFalse;

            _impl->InitLevels(region, maxShift, regionAreaMin);
            _impl->SetCurrent(current, region);

            Point shift;
            for (ptrdiff_t i = _impl->_levels.size() - 1; i >= 0; i--)
            {
                shift.x *= 2;
                shift.y *= 2;
                if (!_impl->SearchLocalMin(_impl->_levels[i], shift, *hiddenAreaPenalty))
                    return SimdFalse;
                shift = _impl->_levels[i].shift;
            }
            return SimdTrue;
        }

        void ShiftDetector::GetShift(ptrdiff_t* shift, double* refinedShift, double* stability, double* correlation)
        {
            if (shift)
            {
                shift[0] = _impl->_levels[0].shift.x;
                shift[1] = _impl->_levels[0].shift.y;
            }
            if (refinedShift)
            {
                FPoint refinement = _impl->_levels[0].differences.Refinement();
                refinedShift[0] = _impl->_levels[0].shift.x + refinement.x;
                refinedShift[1] = _impl->_levels[0].shift.y + refinement.y;
            }
            if (stability)
            {
                stability[0] = _impl->_levels[0].differences.Stability();
            }
            if (correlation)
            {
                double difference = _impl->_levels[0].differences.At(_impl->_levels[0].shift);
                if (_impl->_differenceType == SimdShiftDetectorAbsDifference)
                    correlation[0] = 1.0 - difference / 255;
                else
                    correlation[0] = 1.0 - ::sqrt(difference) / 255;
            }
        }
    }
}
