/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar, 
*               2014-2021 Antonenka Mikhail.
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
#ifndef __SimdContour_hpp__
#define __SimdContour_hpp__

#include "Simd/SimdLib.hpp"

#include <vector>

namespace Simd
{
    /*! @ingroup cpp_contour

        \short ContourDetector structure provides detection of contours at the image.

        Using example:
        \verbatim
        #include "Simd/SimdContour.hpp"
        #include "Simd/SimdDrawing.hpp"

        int main()
        {
            typedef Simd::ContourDetector<Simd::Allocator> ContourDetector;

            ContourDetector::View image;
            image.Load("../../data/image/face/lena.pgm");

            ContourDetector contourDetector;

            contourDetector.Init(image.Size());

            ContourDetector::Contours contours;
            contourDetector.Detect(image, contours);

            for (size_t i = 0; i < contours.size(); ++i)
            {
                for (size_t j = 1; j < contours[i].size(); ++j)
                    Simd::DrawLine(image, contours[i][j - 1], contours[i][j], uint8_t(255));
            }
            image.Save("result.pgm");

            return 0;
        }
        \endverbatim

    */
    template <template<class> class A>
    struct ContourDetector
    {
        typedef A<uint8_t> Allocator; /*!< Allocator type definition. */
        typedef Simd::View<A> View; /*!< An image type definition. */
        typedef Simd::Point<ptrdiff_t> Size; /*!< An image size type definition. */
        typedef Simd::Point<ptrdiff_t> Point; /*!< A point type definition. */
        typedef Rectangle<ptrdiff_t> Rect; /*!< A rectangle type definition. */
        typedef std::vector<Point> Contour; /*!< A contour type definition. */
        typedef std::vector<Contour> Contours; /*!< A vector of contours type definition. */

        /*!
            Prepares ContourDetector structure to work with image of given size.

            \param [in] size - a size of input image.
        */
        void Init(Size size)
        {
            _m.Recreate(size, View::Int16);
            _a.Recreate(size, View::Gray8);
            _e.Recreate(size, View::Gray8);
        }

        /*!
            Detects contours at given image.

            \param [in] src - a input image.
            \param [out] contours - detected contours.
            \param [in] mask - an image with the mask. It is used to restrict region of contours detection. By default it is not used.
            \param [in] indexMin - a minimal index in the mask. By default is equal 3.
            \param [in] roi - Region Of Interest. This is Another way to restrict region of contours detection. By default it is not used.
            \param [in] gradientThreshold - a gradient threshold for contour detection. If this parameter is negative it will be estimated automatically. By default is equal to 40.
            \param [in] anchorThreshold - a anchor threshold for contour detection. By default is equal to 0.
            \param [in] anchorScanInterval - the anchor scan interval. This parameter affects to performance. By default is equal to 2.
            \param [in] minSegmentLength - the minimal length of detected contour. By default is equal to 2.
            \return a result of this operation.
        */
        bool Detect(const View & src, Contours & contours, const View & mask = View(), uint8_t indexMin = 3, const Rect & roi = Rect(),
            int gradientThreshold = 40, int anchorThreshold = 0, int anchorScanInterval = 2, int minSegmentLength = 2)
        {
            if (!Simd::Compatible(src, _a))
                return false;
            if (mask.format != View::None && !Simd::Compatible(mask, _a))
                return false;

            _roi = roi.Empty() ? Rect(src.Size()) : roi;
            _roi.Intersection(Rect(src.Size()));

            ContourMetrics(src, mask, indexMin);

            if (gradientThreshold < 0)
                gradientThreshold = EstimateAdaptiveThreshold();

            ContourAnchors(anchorThreshold, anchorScanInterval);

            PerformSmartRouting(contours, minSegmentLength, gradientThreshold * 2);

            return true;
        }

    private:

        enum Direction
        {
            Unknown = -1,
            Up,
            Down,
            Right,
            Left,
        };

        struct Anchor
        {
            Point p;
            uint16_t val;
            Anchor(const Point & p_, uint16_t val_)
                : p(p_)
                , val(val_)
            {}

            static SIMD_INLINE bool Compare(const Anchor & a, const Anchor & b)
            {
                return a.val > b.val;
            }
        };
        typedef std::vector<Anchor> Anchors;

        Rect _roi;
        View _m;
        View _a;
        View _e;
        Anchors _anchors;
        Contour _contour;

        void ContourMetrics(const View & src, const View & mask, uint8_t indexMin)
        {
            if (mask.format == View::Gray8)
                Simd::ContourMetrics(src.Region(_roi), mask.Region(_roi), indexMin, _m.Region(_roi).Ref());
            else
                Simd::ContourMetrics(src.Region(_roi), _m.Region(_roi).Ref());
        }

        void ContourAnchors(int anchorThreshold, int anchorScanInterval)
        {
            Simd::ContourAnchors(_m.Region(_roi), anchorScanInterval, anchorThreshold, _a.Region(_roi).Ref());

            _anchors.clear();
            for (ptrdiff_t row = _roi.Top() + 1; row < _roi.Bottom() - 1; row += anchorScanInterval)
            {
                const uint8_t * a = &At<A, uint8_t>(_a, 0, row);
                for (ptrdiff_t col = _roi.Left() + 1; col < _roi.Right() - 1; col += anchorScanInterval)
                {
                    if (a[col])
                        _anchors.push_back(Anchor(Point(col, row), At<A, uint16_t>(_m, col, row) / 2));
                }
            }

            std::stable_sort(_anchors.begin(), _anchors.end(), Anchor::Compare);
        }

        void PerformSmartRouting(Contours & contours, size_t minSegmentLength, uint16_t gradientThreshold)
        {
            View e = _e.Region(_roi);
            Rect frame(1, 1, e.width - 1, e.height - 1);
            Simd::Fill(e.Region(frame).Ref(), 0);
            Simd::FillFrame(e, frame, 255);

            _contour.reserve(200);
            for (size_t i = 0; i < _anchors.size(); i++)
            {
                const Anchor & anchor = _anchors[i];
                if (anchor.val > 0)
                {
                    _contour.clear();
                    SmartRoute(contours, _contour, anchor.p.x, anchor.p.y, minSegmentLength, gradientThreshold, Unknown);
                    if (_contour.size() > minSegmentLength)
                        contours.push_back(_contour);
                }
            }
        }

        void SmartRoute(Contours & contours, Contour & contour, ptrdiff_t x, ptrdiff_t y, size_t minSegmentLength, uint16_t gradientThreshold, Direction direction)
        {
            switch (direction)
            {
            case Unknown:
                break;
            case Left:
                while (CheckMetricsForMagnitudeAndDirection(x, y, gradientThreshold, 1))
                {
                    if (At<A, uint8_t>(_e, x, y) == 0)
                    {
                        At<A, uint8_t>(_e, x, y) = 255;
                        if (!contour.empty() && (std::abs(contour.back().x - x) > 1 || std::abs(contour.back().y - y) > 1))
                        {
                            if (contour.size() > minSegmentLength)
                                contours.push_back(contour);
                            contour.clear();
                        }
                        contour.push_back(Point(x, y));
                    }
                    if (CheckMetricsForMagnitudeMaximum(x - 1, y - 1, x - 1, y, x - 1, y + 1))
                    {
                        x--;
                        y--;
                    }
                    else if (CheckMetricsForMagnitudeMaximum(x - 1, y + 1, x - 1, y, x - 1, y - 1))
                    {
                        x--;
                        y++;
                    }
                    else
                        x--;
                    if (At<A, uint8_t>(_e, x, y) != 0)
                        break;
                }
                break;
            case Right:
                while (CheckMetricsForMagnitudeAndDirection(x, y, gradientThreshold, 1))
                {
                    if (At<A, uint8_t>(_e, x, y) == 0)
                    {
                        At<A, uint8_t>(_e, x, y) = 255;
                        if (!contour.empty() && (std::abs(contour.back().x - x) > 1 || std::abs(contour.back().y - y) > 1))
                        {
                            if (contour.size() > minSegmentLength)
                                contours.push_back(contour);
                            contour.clear();
                        }
                        contour.push_back(Point(x, y));
                    }
                    if (CheckMetricsForMagnitudeMaximum(x + 1, y - 1, x + 1, y, x + 1, y + 1))
                    {
                        x++;
                        y--;
                    }
                    else if (CheckMetricsForMagnitudeMaximum(x + 1, y + 1, x + 1, y, x + 1, y - 1))
                    {
                        x++;
                        y++;
                    }
                    else
                        x++;
                    if (At<A, uint8_t>(_e, x, y) != 0)
                        break;
                }
                break;
            case Up:
                while (CheckMetricsForMagnitudeAndDirection(x, y, gradientThreshold, 0))
                {
                    if (At<A, uint8_t>(_e, x, y) == 0)
                    {
                        At<A, uint8_t>(_e, x, y) = 255;
                        if (!contour.empty() && (std::abs(contour.back().x - x) > 1 || std::abs(contour.back().y - y) > 1))
                        {
                            if (contour.size() > minSegmentLength)
                                contours.push_back(contour);
                            contour.clear();
                        }
                        contour.push_back(Point(x, y));
                    }
                    if (CheckMetricsForMagnitudeMaximum(x - 1, y - 1, x, y - 1, x + 1, y - 1))
                    {
                        x--;
                        y--;
                    }
                    else if (CheckMetricsForMagnitudeMaximum(x + 1, y - 1, x, y - 1, x - 1, y - 1))
                    {
                        x++;
                        y--;
                    }
                    else
                        y--;
                    if (At<A, uint8_t>(_e, x, y) != 0)
                        break;
                }
                break;
            case Down:
                while (CheckMetricsForMagnitudeAndDirection(x, y, gradientThreshold, 0))
                {
                    if (At<A, uint8_t>(_e, x, y) == 0)
                    {
                        At<A, uint8_t>(_e, x, y) = 255;
                        if (!contour.empty() && (std::abs(contour.back().x - x) > 1 || std::abs(contour.back().y - y) > 1))
                        {
                            if (contour.size() > minSegmentLength)
                                contours.push_back(contour);
                            contour.clear();
                        }
                        contour.push_back(Point(x, y));
                    }
                    if (CheckMetricsForMagnitudeMaximum(x + 1, y + 1, x, y + 1, x - 1, y + 1))
                    {
                        x++;
                        y++;
                    }
                    else if (CheckMetricsForMagnitudeMaximum(x - 1, y + 1, x, y + 1, x + 1, y + 1))
                    {
                        x--;
                        y++;
                    }
                    else
                        y++;
                    if (At<A, uint8_t>(_e, x, y) != 0)
                        break;
                }
                break;
            }

            if (At<A, uint8_t>(_e, x, y) != 0 || At<A, uint16_t>(_m, x, y) < gradientThreshold)
                return;

            uint16_t d = At<A, uint16_t>(_m, x, y) & 1;
            if (d == 0)
            {
                SmartRoute(contours, contour, x, y, minSegmentLength, gradientThreshold, Up);
                SmartRoute(contours, contour, x, y, minSegmentLength, gradientThreshold, Down);
            }
            else if (d == 1)
            {
                SmartRoute(contours, contour, x, y, minSegmentLength, gradientThreshold, Right);
                SmartRoute(contours, contour, x, y, minSegmentLength, gradientThreshold, Left);
            }
        }

        bool CheckMetricsForMagnitudeAndDirection(ptrdiff_t x, ptrdiff_t y, int16_t gradientThreshold, int16_t direction) const
        {
            const uint16_t & m = At<A, uint16_t>(_m, x, y);
            return m >= gradientThreshold && (m & 1) == direction;
        }

        bool CheckMetricsForMagnitudeMaximum(ptrdiff_t x0, ptrdiff_t y0, ptrdiff_t x1, ptrdiff_t y1, ptrdiff_t x2, ptrdiff_t y2) const
        {
            const uint16_t m0 = At<A, uint16_t>(_m, x0, y0) | 1;
            const uint16_t m1 = At<A, uint16_t>(_m, x1, y1) | 1;
            const uint16_t m2 = At<A, uint16_t>(_m, x2, y2) | 1;
            return m0 > m1 && m0 > m2;
        }

        uint16_t EstimateAdaptiveThreshold()
        {
            Point roiSize = _roi.Size();
            Point mSize = _m.Size();
            if (roiSize.x >= mSize.x || roiSize.y >= mSize.y)
                assert(0);

            View m = _m.Region(_roi);
            Point size = m.Size();
            uint16_t value;
            uint32_t sum = 0;
            int count = 0;
            for (ptrdiff_t i = 0; i < size.x; ++i)
            {
                for (ptrdiff_t j = 0; j < size.y; ++j)
                {
                    value = At<A, uint16_t>(m, i, j);
                    if (value)
                    {
                        count++;
                        value = value >> 1;
                        sum += value;
                    }
                }
            }

            uint16_t meanThreshold = (uint16_t)((double)sum / count);
            return meanThreshold;
        }
    };
}
#endif//__SimdContour_hpp__
