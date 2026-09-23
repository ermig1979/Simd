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
#ifndef __SimdRectangle_hpp__
#define __SimdRectangle_hpp__

#include "Simd/SimdPoint.hpp"

#include <algorithm>

namespace Simd
{
    /*! @ingroup cpp_rectangle

        \short Axis-aligned half-open rectangle [left, right) x [top, bottom).

        Rectangle<T> stores the four sides of an axis-aligned rectangle. The
        right and bottom sides are exclusive: the covered area is
        [left, right) x [top, bottom). Width() is right - left and Height()
        is bottom - top. Image coordinates have their origin at the top-left
        corner, X grows to the right and Y grows downward. Contains(x, y),
        DrawFilledRectangle, View::Region, Frame::Region, CopyFrame and
        FillFrame all use that half-open range. A point with x == right or
        y == bottom lies outside the rectangle.

        Rectangle<ptrdiff_t> is the image rectangle. View::Region, View::Clone,
        View::Copy and Frame::Region take it as a region of interest.
        Simd::DrawRectangle and Simd::DrawFilledRectangle draw it.
        Simd::CopyFrame and Simd::FillFrame leave its interior unchanged and
        process the border around it. Simd::ShiftBilinear shifts pixels inside
        the crop and copies the area outside the crop from the source.
        Simd::SegmentationShrinkRegion rewrites the rectangle as the bounding
        box of one mask index. Motion::Rect is an object box or a moving
        region. Detection::Rect is a search window or an object bound.
        ContourDetector::Rect is the region of interest. ShiftDetector::Rect
        is the correlation window.

        Rectangle() is (0, 0, 0, 0). Its area is 0, so Empty() is true. Motion
        starts a moving region from Rect() and grows it with operator |= and
        each accepted pixel. ContourDetector replaces an empty ROI with
        Rect(src.Size()). Detection skips a level whose search rectangle is
        empty. SegmentationShrinkRegion writes (0, 0, 0, 0) when the index is
        absent.

        Rectangle(point) sets the top-left corner to (0, 0) and the
        bottom-right corner to the point, so Rectangle(view.Size()) is the
        whole image [0, width) x [0, height). TestImageMatcher passes
        Rect(src.Size()) as the crop of Simd::ShiftBilinear. Detection stores
        level.rect = Rect(level.roi.Size()) and then lets
        SegmentationShrinkRegion tighten it to the ROI mask.
        DrawFilledRectangle clips the fill with
        rect &= Rectangle(canvas.Size()).

        Values passed to a constructor are converted to T. Conversion of float
        or double to ptrdiff_t rounds to the nearest integer, with halves away
        from zero. Other conversions are a C-style cast. Assignment from
        another Rectangle and the converting constructor copy the four sides
        the same way. operator* and operator/ scale every side and then
        construct the result, so a floating factor applied to
        Rectangle<ptrdiff_t> is rounded. Rect(0, 0, 10, 20) * 1.5 is
        (0, 0, 15, 30). Rect(0, 0, 11, 21) / 2 uses integer division and is
        (0, 0, 5, 10). Detection scales a window by the pyramid level with
        Rect(col, row, col + size.x, row + size.y) * scale and brings a motion
        region back with rects[i] / level.scale. Grouped detections and a
        motion trajectory are averaged by adding rectangles with operator +=
        and dividing by the count.

        operator |= with a point grows the half-open box so that the pixel is
        inside. An empty rectangle becomes the one-pixel cell
        [x, x + 1) x [y, y + 1). Motion flood-fill writes
        region->rect |= current for every accepted pixel. operator |= with a
        rectangle is the bounding union. Font::Draw unions glyph cells into
        the alpha rectangle. Detection unions motion regions and then clips
        them with operator &=. operator &= with a rectangle is the in-place
        intersection. An empty rectangle is left unchanged. An empty argument
        replaces this rectangle. ShiftDetector clips a shifted window with
        region &= Rect(image.Size()) and then Shift(-shift) moves it back.
        Motion clips a tracked object with object->rect &= Rect(frameSize).
        Intersection returns a new rectangle and leaves this one unchanged.
        A disjoint pair produces an empty rectangle whose width and height are
        non-negative. Detection builds the scanned window with
        rect.Shifted(-size / 2).Intersection(Rect(dst.Size() - size)). Font
        skips a glyph when canvas.Intersection(shifted).Empty().

        AddBorder adds the same margin on every side: left and top decrease,
        right and bottom increase. A negative margin shrinks the rectangle.
        Motion::ShrinkRoi calls AddBorder(1) after SegmentationShrinkRegion.
        ExpandRoi expands the parent ROI with AddBorder(1) and insets the
        child mask with AddBorder(-1) before operator &=. Enlarged grows a
        box by a fraction of (Width() + Height()) / 2 so Contains can link a
        center that lies just outside the original box. TestShift shrinks the
        correlation window with AddBorder(-hs / 4).

        Shift adds a translation to all four sides. Shifted returns the
        translated rectangle and leaves this one unchanged. ShiftDetector
        compares background.Region(region.Shifted(shift)) with
        current.Region(region). Motion recenters a tracked box with
        object->rect.Shift(nearest->rect.Center() - object->rect.Center()).
        Font places a glyph with current.Shifted(shift + indent) and moves
        the alpha box to the origin with alphaRect.Shift(-alphaRect.TopLeft()).

        With SIMD_OPENCV_ENABLE, a rectangle converts to and from cv::Rect_<T>.
        OpenCV stores (x, y, width, height). This structure stores
        (left, top, right, bottom). Assignment from cv::Rect sets
        left = x, top = y, right = x + width and bottom = y + height.
        Assignment to cv::Rect sets x = left, y = top, width = right - left
        and height = bottom - top.

        Using example:
        \code
        #include "Simd/SimdLib.hpp"
        #include "Simd/SimdDrawing.hpp"

        int main()
        {
            typedef Simd::Point<ptrdiff_t> Point;
            typedef Simd::Rectangle<ptrdiff_t> Rect;
            typedef Simd::View<Simd::Allocator> View;

            View image(320, 240, View::Gray8);
            Rect imageRect(image.Size());
            Rect window(10, 20, 110, 80);

            View crop = image.Region(window);
            Rect grown = window;
            grown.AddBorder(2);
            grown &= imageRect;

            Rect bounds;
            bounds |= Point(window.left, window.top);
            bounds |= Point(window.right - 1, window.bottom - 1);

            if (imageRect.Contains(window.Center()))
                Simd::DrawRectangle(image, window, uint8_t(255));

            View dst(image.Size(), View::Gray8);
            Simd::ShiftBilinear(image, image, Simd::Point<double>(0.5, -0.25), imageRect, dst);

            return (int)crop.width + (int)grown.Width() + (int)bounds.Area();
        }
        \endcode

        OpenCV conversion (define SIMD_OPENCV_ENABLE before including this header):
        \code
        #include "opencv2/core/core.hpp"
        #define SIMD_OPENCV_ENABLE
        #include "Simd/SimdRectangle.hpp"

        int main()
        {
            typedef Simd::Rectangle<ptrdiff_t> Rect;

            cv::Rect cvRect(10, 20, 100, 80);
            Rect simdRect = cvRect;
            cvRect = simdRect;

            return simdRect.Width();
        }
        \endcode

        \ref cpp_rectangle_functions.
    */
    template <typename T>
    struct Rectangle
    {
        typedef T Type; /*!< Side coordinate type. ptrdiff_t is a pixel rectangle. */

        T left; /*!< \brief Inclusive X of the left side. Grows to the right. */
        T top; /*!< \brief Inclusive Y of the top side. Grows downward. */
        T right; /*!< \brief Exclusive X of the right side. Width is right - left. */
        T bottom; /*!< \brief Exclusive Y of the bottom side. Height is bottom - top. */

        /*!
            Creates the empty rectangle (0, 0, 0, 0).

            Empty() is true. Motion starts a moving region from Rect() and
            grows it with operator |=. ContourDetector treats an empty ROI as
            the whole image. Detection skips a search rectangle that is empty.
        */
        Rectangle();

        /*!
            Creates a rectangle from the four sides.

            Each side is converted to T. float and double values converted to
            ptrdiff_t are rounded to the nearest integer, with halves rounded
            away from zero. The rectangle is half-open: [l, r) x [t, b).
            TestShift builds the correlation window as
            Rect(c.x - hs, c.y - hs, c.x + hs, c.y + hs). Detection places an
            object at Rect(col, row, col + size.x, row + size.y) before
            scaling it by the pyramid level. CopyFrame and FillFrame require
            0 <= left <= right <= width and 0 <= top <= bottom <= height.

            \param [in] l - initial left side. Pixels with x >= left are inside.
            \param [in] t - initial top side. Pixels with y >= top are inside.
            \param [in] r - initial right side. Pixels with x >= right are outside.
            \param [in] b - initial bottom side. Pixels with y >= bottom are outside.
        */
        template <typename TL, typename TT, typename TR, typename TB> Rectangle(TL l, TT t, TR r, TB b);

        /*!
            Creates a rectangle from its top-left and bottom-right corners.

            left and top are taken from lt. right and bottom are taken from rb.
            Each coordinate is converted to T. DrawRectangle(topLeft, bottomRight)
            builds this rectangle and draws the frame. Font::Draw builds a glyph
            cell as Rect(curr, curr + glyphSize), where curr + glyphSize is the
            exclusive bottom-right corner.

            \param [in] lt - top-left corner. Its x and y become left and top.
            \param [in] rb - bottom-right corner. Its x and y become right and bottom.
        */
        template <typename TLT, typename TRB> Rectangle(const Point<TLT> & lt, const Point<TRB> & rb);

        /*!
            Creates a rectangle [0, rb.x) x [0, rb.y).

            The top-left corner is (0, 0). The point is the exclusive
            bottom-right corner, so Rectangle(view.Size()) is the whole image.
            TestImageMatcher passes Rect(src.Size()) as the ShiftBilinear crop.
            Detection initializes a level with Rect(level.roi.Size()).
            SegmentationShrinkRegion checks Contains against
            Rectangle(mask.Size()). DrawFilledRectangle and ShiftDetector clip
            a rectangle with &= Rectangle(image.Size()).

            \param [in] rb - exclusive bottom-right corner. x becomes right and y becomes bottom.
        */
        template <typename TRB> Rectangle(const Point<TRB> & rb);

        /*!
            Creates a rectangle by copying the four sides of another rectangle type.

            The source must be a class template with left, top, right and bottom.
            Each side is converted to T. This copies between
            Rectangle<ptrdiff_t> and Rectangle<double>. With SIMD_OPENCV_ENABLE
            the cv::Rect_ constructor below is a separate overload: OpenCV
            stores width and height, and this constructor does not read them.

            \param [in] r - a rectangle of arbitrary type with left, top, right and bottom.
        */
        template <class TR, template<class> class TRectangle> Rectangle(const TRectangle<TR> & r);

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Creates a rectangle from an OpenCV rectangle.

            cv::Rect_ stores (x, y, width, height). The Simd rectangle stores
            the half-open sides left = x, top = y, right = x + width and
            bottom = y + height. Each value is converted to T, so a floating
            OpenCV rectangle assigned to Rectangle<ptrdiff_t> is rounded.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] r - an OpenCV rectangle (x, y, width, height).
        */
        template <class TR> Rectangle(const cv::Rect_<TR> & r);
#endif

        /*!
            Destroys the rectangle. The destructor has no side effects.
        */
        ~Rectangle();

        /*!
            Converts this rectangle to another rectangle type constructed from the four sides.

            The target is constructed as TRectangle<TR>(left, top, right, bottom).
            Each side is converted to TR. With SIMD_OPENCV_ENABLE, conversion to
            cv::Rect_ uses the dedicated operator below, which passes width and
            height rather than right and bottom.

            \return a rectangle of arbitrary type.
        */
        template <class TR, template<class> class TRectangle> operator TRectangle<TR>() const;

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Converts this rectangle to an OpenCV rectangle.

            The result is cv::Rect_<TR>(left, top, right - left, bottom - top).
            OpenCV therefore receives x, y, width and height. Conversion to
            ptrdiff_t rounds float and double; conversion to any other type is
            a C-style cast. A Rectangle<ptrdiff_t> converted to cv::Rect_<double>
            keeps the integer sides.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \return an OpenCV rectangle (x, y, width, height).
        */
        template <class TR> operator cv::Rect_<TR>() const;
#endif

        /*!
            Copies the four sides of another rectangle.

            Each side is converted to T. Assigning Rectangle<double> to
            Rectangle<ptrdiff_t> rounds every side.

            \param [in] r - a rectangle of arbitrary coordinate type.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & operator = (const Rectangle<TR> & r);

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Copies an OpenCV rectangle into this rectangle.

            left = x, top = y, right = x + width and bottom = y + height.
            Each value is converted to T.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] r - an OpenCV rectangle (x, y, width, height).
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & operator = (const cv::Rect_<TR> & r);
#endif

        /*!
            Sets the left side.

            The value is converted to T. Pixels with x >= left are candidates
            for the interior. The method does not move the other sides.

            \param [in] l - a new left side.
            \return a reference to itself.
        */
        template <typename TL> Rectangle<T> & SetLeft(const TL & l);

        /*!
            Sets the top side.

            The value is converted to T. Pixels with y >= top are candidates
            for the interior. The method does not move the other sides.

            \param [in] t - a new top side.
            \return a reference to itself.
        */
        template <typename TT> Rectangle<T> & SetTop(const TT & t);

        /*!
            Sets the right side.

            The value is converted to T. The right side is exclusive: pixels
            with x >= right are outside. The method does not move the other
            sides.

            \param [in] r - a new right side.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & SetRight(const TR & r);

        /*!
            Sets the bottom side.

            The value is converted to T. The bottom side is exclusive: pixels
            with y >= bottom are outside. The method does not move the other
            sides.

            \param [in] b - a new bottom side.
            \return a reference to itself.
        */
        template <typename TB> Rectangle<T> & SetBottom(const TB & b);

        /*!
            Sets the top-left corner.

            left and top are converted from the point. right and bottom stay
            unchanged. Motion::ExpandRoi maps a parent ROI onto the next
            pyramid level with
            SetTopLeft(parent.TopLeft() * 2 - Point(1, 1)).

            \param [in] topLeft - a new top-left corner.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & SetTopLeft(const Point<TP> & topLeft);

        /*!
            Sets the top-right corner.

            right and top are converted from the point. left and bottom stay
            unchanged. The right coordinate remains exclusive.

            \param [in] topRight - a new top-right corner.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & SetTopRight(const Point<TP> & topRight);

        /*!
            Sets the bottom-left corner.

            left and bottom are converted from the point. right and top stay
            unchanged. The bottom coordinate remains exclusive.

            \param [in] bottomLeft - a new bottom-left corner.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & SetBottomLeft(const Point<TP> & bottomLeft);

        /*!
            Sets the bottom-right corner.

            right and bottom are converted from the point. left and top stay
            unchanged. Motion::ExpandRoi writes the child corner with
            SetBottomRight(parent.BottomRight() * 2 + Point(1, 1)).

            \param [in] bottomRight - a new bottom-right corner.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & SetBottomRight(const Point<TP> & bottomRight);

        /*!
            Returns the inclusive left side.

            View::Region(rect) reads Left(), Top(), Right() and Bottom().
            ShiftDetector::InitLevels requires Left() >= 0 and Right() within
            the image width.

            \return the left side.
        */
        T Left() const;

        /*!
            Returns the inclusive top side.

            \return the top side.
        */
        T Top() const;

        /*!
            Returns the exclusive right side.

            \return the right side.
        */
        T Right() const;

        /*!
            Returns the exclusive bottom side.

            \return the bottom side.
        */
        T Bottom() const;

        /*!
            Returns the top-left corner (left, top).

            Motion::ExpandRoi scales this corner onto the next pyramid level
            with TopLeft() * 2 - Point(1, 1). Font::Draw shifts glyph cells by
            -alphaRect.TopLeft() so the alpha image starts at the origin.

            \return the top-left corner.
        */
        Point<T> TopLeft() const;

        /*!
            Returns the top-right corner (right, top).

            right is the exclusive edge, so this corner lies outside the
            rectangle when the width is positive.

            \return the top-right corner.
        */
        Point<T> TopRight() const;

        /*!
            Returns the bottom-left corner (left, bottom).

            bottom is the exclusive edge, so this corner lies outside the
            rectangle when the height is positive.

            \return the bottom-left corner.
        */
        Point<T> BottomLeft() const;

        /*!
            Returns the bottom-right corner (right, bottom).

            Both coordinates are exclusive. Motion::ExpandRoi scales this
            corner with BottomRight() * 2 + Point(1, 1).

            \return the bottom-right corner.
        */
        Point<T> BottomRight() const;

        /*!
            Returns the width, right - left.

            The width is negative when right < left. CopyFrame requires
            Width() >= 0. ShiftDetector compares current.Size() with
            region.Size(), whose x is this width. LBP detection steps across
            a feature with Width() * column. TestResize builds the UV
            rectangle as the luma rectangle divided by 2, which divides this
            width as well.

            \return the width.
        */
        T Width() const;

        /*!
            Returns the height, bottom - top.

            The height is negative when bottom < top. CopyFrame requires
            Height() >= 0. SegmentationShrinkRegion requires Height() > 0
            before it searches the mask.

            \return the height.
        */
        T Height() const;

        /*!
            Returns the area, Width() * Height().

            Empty() is true when this product is 0. A negative width and a
            negative height give a positive area. ShiftDetector rejects a
            window with Area() < regionAreaMin and divides a pixel difference
            by region.Area(). Detection runs a cascade in parallel when
            rect.Area() reaches 10000 for Haar and 30000 for LBP. Haar
            normalization uses the area of Rect(1, 1, win.x - 1, win.y - 1),
            the window without its one-pixel border. Motion drops a region
            whose Area() is at most areaRegionMinEstimated.

            \return the area. It is 0 for the default rectangle.
        */
        T Area() const;

        /*!
            Returns true when Width() * Height() is 0.

            The default rectangle is empty. A rectangle with right == left or
            bottom == top is empty even when the other side is positive.
            ContourDetector replaces an empty ROI with the whole image.
            Detection skips an empty search rectangle. Motion keeps a region
            only when the tightened box is not empty. Font skips the alpha
            blit when the glyph intersection is empty. operator |= replaces an
            empty receiver with the argument and leaves a non-empty receiver
            unchanged when the argument is empty. operator &= leaves an empty
            receiver unchanged.

            \return true when the area is 0.
        */
        bool Empty() const;

        /*!
            Returns the size as Point<T>(Width(), Height()).

            x is the width and y is the height. Font::Draw recreates the glyph
            alpha image with alpha.Recreate(alphaRect.Size()). ShiftDetector
            requires current.Size() == region.Size().

            \return the size point.
        */
        Point<T> Size() const;

        /*!
            Returns the center, rounded to T.

            The coordinates are (left + right) / 2 and (top + bottom) / 2,
            computed in double and then converted to T. Halves round away from
            zero. For a non-negative one-pixel cell [x, x + 1) x [y, y + 1)
            the rounded center is (x + 1, y + 1), which Contains does not
            include. Motion
            stores region->point = region->rect.Center() and picks the nearest
            tracked object by SquaredDistance of the two centers. After
            averaging a trajectory it recenters the box on that point with
            Shift. TestRandom uses Center() as the middle of a rhombus mask.

            \return the center point.
        */
        Point<T> Center() const;

        /*!
            Returns true when the pixel (x, y) lies inside the half-open rectangle.

            The pixel is inside when x >= left && x < right && y >= top &&
            y < bottom. The coordinates are converted to T before the
            comparison, so a floating point passed to Rectangle<ptrdiff_t> is
            rounded first. The right column and the bottom row are outside.

            \param [in] x - x-coordinate of the checked point.
            \param [in] y - y-coordinate of the checked point.
            \return true when the point is inside.
        */
        template <typename TX, typename TY> bool Contains(TX x, TY y) const;

        /*!
            Returns true when the point lies inside the half-open rectangle.

            This calls Contains(p.x, p.y). Motion::LinkObjects accepts a
            region when the enlarged region contains the object center or the
            enlarged object contains the region center.

            \param [in] p - a checked point.
            \return true when the point is inside.
        */
        template <typename TP> bool Contains(const Point<TP> & p) const;

        /*!
            Returns true when the half-open rectangle [l, r) x [t, b) lies inside this rectangle.

            The four sides are converted to T. The result is true when
            l >= left && r <= right && t >= top && b <= bottom. The tested
            right and bottom may touch this rectangle's exclusive edges.
            A tested rectangle that sticks out by one pixel is outside.

            \param [in] l - left side of the checked rectangle.
            \param [in] t - top side of the checked rectangle.
            \param [in] r - right side of the checked rectangle.
            \param [in] b - bottom side of the checked rectangle.
            \return true when the checked rectangle is inside.
        */
        template <typename TL, typename TT, typename TR, typename TB> bool Contains(TL l, TT t, TR r, TB b) const;

        /*!
            Returns true when the other rectangle lies inside this rectangle.

            This calls Contains(r.left, r.top, r.right, r.bottom).
            SegmentationShrinkRegion requires
            Rectangle(mask.Size()).Contains(rect) before it searches.
            TestRandom checks the same condition for a mask rhombus.

            \param [in] r - a checked rectangle.
            \return true when the checked rectangle is inside.
        */
        template <typename TR> bool Contains(const Rectangle <TR> & r) const;

        /*!
            Translates this rectangle by a point.

            All four sides move by the same converted offset. The width and
            the height stay the same. This method writes into this rectangle.
            Shifted returns a new rectangle. Motion recenters a tracked box
            with Shift(region.Center() - object.Center()). ShiftDetector moves
            a clipped window back with Shift(-currentShift). Font moves the
            alpha box to the origin with Shift(-TopLeft()).

            \param [in] shift - a point with the translation. x moves left and right, y moves top and bottom.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & Shift(const Point<TP> & shift);

        /*!
            Translates this rectangle by two offsets.

            The offsets are converted to T through Point<T> and then added to
            every side. A floating offset applied to Rectangle<ptrdiff_t> is
            rounded. TestShift builds the current view as
            background.Region(region.Shifted(10, 10)); the Shifted overload
            uses this translation and leaves the original region in place.

            \param [in] shiftX - translation along X.
            \param [in] shiftY - translation along Y.
            \return a reference to itself.
        */
        template <typename TX, typename TY> Rectangle<T> & Shift(TX shiftX, TY shiftY);

        /*!
            Returns this rectangle translated by a point.

            This rectangle is not modified. ShiftDetector reads
            background.Region(region.Shifted(shift)) while the correlation
            window itself stays on the current image. Font places a glyph with
            current.Shifted(shift + indent).

            \param [in] shift - a point with the translation.
            \return the translated rectangle.
        */
        template <typename TP> Rectangle<T> Shifted(const Point<TP> & shift) const;

        /*!
            Returns this rectangle translated by two offsets.

            This rectangle is not modified. The offsets are converted to T
            before they are added. TestShift passes region.Shifted(ss) to
            ShiftDetector::Estimate and draws both the original window and the
            shifted window.

            \param [in] shiftX - translation along X.
            \param [in] shiftY - translation along Y.
            \return the translated rectangle.
        */
        template <typename TX, typename TY> Rectangle<T> Shifted(TX shiftX, TY shiftY) const;

        /*!
            Grows or shrinks the rectangle by the same margin on every side.

            left and top decrease by the margin. right and bottom increase by
            the margin. The margin is converted to T. A positive margin makes
            the rectangle larger. A negative margin shrinks it and can make it
            empty. Motion::ShrinkRoi calls AddBorder(1) so the mask index
            found by SegmentationShrinkRegion has a one-pixel neighbourhood.
            ExpandRoi calls AddBorder(1) on the scaled ROI and AddBorder(-1)
            on the child image rectangle. Enlarged adds
            ceil(((Width() + Height()) / 2) * TrackingAdditionalLinking),
            where the inner division is integer division.
            ShiftDetector enlarges the 3x3 search window with AddBorder(1).
            TestShift shrinks the correlation window with AddBorder(-hs / 4).

            \param [in] border - a margin added on every side. A negative value shrinks the rectangle.
            \return a reference to itself.
        */
        template <typename TB> Rectangle<T> & AddBorder(TB border);

        /*!
            Returns the intersection of this rectangle and another rectangle.

            This rectangle is not modified. The other rectangle is converted
            to T. The result has left = max(left, r.left),
            top = max(top, r.top), right = max(left, min(right, r.right)) and
            bottom = max(top, min(bottom, r.bottom)), so its width and height
            are non-negative. A disjoint pair is empty. Detection scans
            rect.Shifted(-size / 2).Intersection(Rect(dst.Size() - size)).
            Font::Draw skips a glyph when canvas.Intersection(shifted).Empty().
            operator &= writes an intersection into this rectangle and treats
            an empty receiver differently: an empty receiver stays unchanged.

            \param [in] r - the other rectangle.
            \return the intersection. It is empty when the rectangles do not overlap.
        */
        template <typename TR> Rectangle<T> Intersection(const Rectangle<TR> & r) const;

        /*!
            Replaces this rectangle with its intersection with the one-pixel cell of a point.

            The point is converted to T. When Contains(p) is true, the
            rectangle becomes [p.x, p.x + 1) x [p.y, p.y + 1). Otherwise it
            becomes empty while keeping its previous left and top
            (right = left and bottom = top). TestCheckCpp intersects two
            rectangles and then applies this operator to a point inside the
            result. Clipping to an image uses operator &= with
            Rectangle(image.Size()), which keeps the overlapping area.

            \param [in] p - a point.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & operator &= (const Point<TP> & p);

        /*!
            Replaces this rectangle with its intersection with another rectangle.

            When this rectangle is empty, it stays unchanged. When the argument
            is empty, this rectangle becomes a copy of that argument. Otherwise
            each side is clipped and the result keeps a non-negative width and
            height. DrawFilledRectangle clips the fill with
            rect &= Rectangle(canvas.Size()) before the per-pixel loop.
            ShiftDetector clips a shifted window with
            region &= Rect(level.current.Size()). Motion clips propagated
            regions with &= rectChild and clips a tracked object with
            &= Rect(frameSize). Detection clips the union of motion regions
            with &= level.rect. Intersection returns a new rectangle instead
            of writing into this one.

            \param [in] r - the other rectangle.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & operator &= (const Rectangle<TR> & r);

        /*!
            Grows this rectangle so that the pixel of the point is inside.

            The point is converted to T. An empty rectangle becomes the
            one-pixel cell [p.x, p.x + 1) x [p.y, p.y + 1). A point already
            inside does not change the rectangle. A point with p.x >= right
            sets right to p.x + 1, and a point with p.y >= bottom sets bottom
            to p.y + 1, because those edges are exclusive. Motion flood-fill
            writes region->rect |= current for every mask pixel it accepts,
            starting from Rect().

            \param [in] p - a point to include.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & operator |= (const Point<TP> & p);

        /*!
            Replaces this rectangle with the bounding union of two rectangles.

            An empty receiver becomes a copy of the argument. An empty argument
            leaves this rectangle unchanged. Otherwise left and top become the
            minima and right and bottom become the maxima. Font::Draw unions
            visible glyph cells into alphaRect and canvasRect. Detection
            unions motion regions with rect |= r before clipping to the level.

            \param [in] r - the other rectangle.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & operator |= (const Rectangle<TR> & r);

        /*!
            Adds the four sides of another rectangle to this rectangle.

            left, top, right and bottom each grow by the converted side of the
            argument. This is a sum of coordinates. The bounding union is
            operator |=. Motion averages a trajectory by accumulating
            sum += region.rect and then dividing by the number of positions.
            Detection groups overlapping objects the same way:
            buffer[cls].rect += src[i].rect, then divides by the group weight.
            Adding to the default rectangle copies the argument when the
            coordinate types match.

            \param [in] r - the rectangle whose sides are added.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & operator += (const Rectangle<TR> & r);

        /*!
            Returns true when the two half-open rectangles have a common point.

            For rectangles of non-negative size this is
            left < r.right && right > r.left && top < r.bottom && bottom > r.top.
            Edges that only touch do not overlap: right == r.left is outside
            both interiors. The implementation compares the two horizontal
            tests with each other and the two vertical tests with each other.
            Font and Detection test overlap with Intersection and operator &=.

            \param [in] r - the other rectangle.
            \return true when the rectangles overlap.
        */
        bool Overlaps(const Rectangle<T> & r) const;
    };

    /*! @ingroup cpp_rectangle_functions

        \fn template <typename T> bool operator == (const Rectangle<T> & r1, const Rectangle<T> & r2);

        \short Compares two rectangles by all four sides.

        left, top, right and bottom are compared independently. Two rectangles
        of equal width and height compare equal only when they also share the
        same origin. TestDetection uses operator != to compare an object
        rectangle from two detector runs.

        \param [in] r1 - a first rectangle.
        \param [in] r2 - a second rectangle.
        \return true when all four sides are equal.
    */
    template <typename T> bool operator == (const Rectangle<T> & r1, const Rectangle<T> & r2);

    /*! @ingroup cpp_rectangle_functions

        \fn template <typename T> bool operator != (const Rectangle<T> & r1, const Rectangle<T> & r2);

        \short Compares two rectangles by any differing side.

        The result is true when left, top, right or bottom differs.
        TestDetection reports a mismatch when os[i].rect != om[i].rect.

        \param [in] r1 - a first rectangle.
        \param [in] r2 - a second rectangle.
        \return true when any side differs.
    */
    template <typename T> bool operator != (const Rectangle<T> & r1, const Rectangle<T> & r2);

    /*! @ingroup cpp_rectangle_functions

        \fn template<class T1, class T2> Rectangle<T1> operator / (const Rectangle<T1> & rect, const T2 & value);

        \short Divides every side by a scalar.

        The result is Rectangle<T1>(left / value, top / value, right / value,
        bottom / value). Construction converts each quotient back to T1, so
        Rectangle<ptrdiff_t> divided by a floating value is rounded to the
        nearest integer. Integer division truncates toward zero.
        TestResize builds the chroma rectangle as the luma rectangle / 2.
        Detection maps a motion region onto a pyramid level with
        rects[i] / level.scale and averages a group with rect / weight.

        \param [in] rect - a rectangle.
        \param [in] value - a non-zero scalar.
        \return the rectangle with divided sides.
    */
    template<class T1, class T2> Rectangle<T1> operator / (const Rectangle<T1> & rect, const T2 & value);

    /*! @ingroup cpp_rectangle_functions

        \fn template<class T1, class T2> Rectangle<T1> operator * (const Rectangle<T1> & rect, const T2 & value);

        \short Multiplies every side by a scalar.

        The result is Rectangle<T1>(left * value, top * value, right * value,
        bottom * value). A floating factor applied to Rectangle<ptrdiff_t> is
        rounded by the constructor. Detection lifts a detection window to
        input-image coordinates with
        Rect(col, row, col + size.x, row + size.y) * scale. Motion debug
        drawing paints object.rect * scale on the full-resolution frame.

        \param [in] rect - a rectangle.
        \param [in] value - a scalar factor.
        \return the rectangle with multiplied sides.
    */
    template<class T1, class T2> Rectangle<T1> operator * (const Rectangle<T1> & rect, const T2 & value);

    /*! @ingroup cpp_rectangle_functions

        \fn template<class T1, class T2> Rectangle<T1> operator * (const T2 & value, const Rectangle<T1> & rect);

        \short Multiplies a scalar by every side of a rectangle.

        The result is the same as rect * value. A floating factor applied to
        Rectangle<ptrdiff_t> is rounded by the constructor.

        \param [in] value - a scalar factor.
        \param [in] rect - a rectangle.
        \return the rectangle with multiplied sides.
    */
    template<class T1, class T2> Rectangle<T1> operator * (const T2 & value, const Rectangle<T1> & rect);

    /*! @ingroup cpp_rectangle_functions

        \fn template <typename T> Rectangle<T> operator + (const Rectangle<T> & r1, const Rectangle<T> & r2);

        \short Adds the corresponding sides of two rectangles.

        The result is (r1.left + r2.left, r1.top + r2.top, r1.right + r2.right,
        r1.bottom + r2.bottom). This is the same sum as operator +=. The
        bounding union is operator |=. Motion and Detection accumulate this
        sum and then divide by the number of rectangles to average a box.

        \param [in] r1 - a first rectangle.
        \param [in] r2 - a second rectangle.
        \return the rectangle with summed sides.
    */
    template <typename T> Rectangle<T> operator + (const Rectangle<T> & r1, const Rectangle<T> & r2);

    //-------------------------------------------------------------------------

    // struct Rectangle<T> implementation:

    template <typename T>
    SIMD_INLINE Rectangle<T>::Rectangle()
        : left(0)
        , top(0)
        , right(0)
        , bottom(0)
    {
    }

    template <typename T> template <typename TL, typename TT, typename TR, typename TB>
    SIMD_INLINE Rectangle<T>::Rectangle(TL l, TT t, TR r, TB b)
        : left(Convert<T, TL>(l))
        , top(Convert<T, TT>(t))
        , right(Convert<T, TR>(r))
        , bottom(Convert<T, TB>(b))
    {
    }

    template <typename T> template <typename TLT, typename TRB>
    SIMD_INLINE Rectangle<T>::Rectangle(const Point<TLT> & lt, const Point<TRB> & rb)
        : left(Convert<T, TLT>(lt.x))
        , top(Convert<T, TLT>(lt.y))
        , right(Convert<T, TRB>(rb.x))
        , bottom(Convert<T, TRB>(rb.y))
    {
    }

    template <typename T> template <typename TRB>
    SIMD_INLINE Rectangle<T>::Rectangle(const Point<TRB> & rb)
        : left(0)
        , top(0)
        , right(Convert<T, TRB>(rb.x))
        , bottom(Convert<T, TRB>(rb.y))
    {
    }

    template <typename T> template <class TR, template<class> class TRectangle>
    SIMD_INLINE Rectangle<T>::Rectangle(const TRectangle<TR> & r)
        : left(Convert<T, TR>(r.left))
        , top(Convert<T, TR>(r.top))
        , right(Convert<T, TR>(r.right))
        , bottom(Convert<T, TR>(r.bottom))
    {
    }

#ifdef SIMD_OPENCV_ENABLE
    template <typename T> template <class TR>
    SIMD_INLINE Rectangle<T>::Rectangle(const cv::Rect_<TR> & r)
        : left(Convert<T, TR>(r.x))
        , top(Convert<T, TR>(r.y))
        , right(Convert<T, TR>(r.x + r.width))
        , bottom(Convert<T, TR>(r.y + r.height))
    {
    }
#endif

    template <typename T>
    SIMD_INLINE Rectangle<T>::~Rectangle()
    {
    }

    template <typename T> template <class TR, template<class> class TRectangle>
    SIMD_INLINE Rectangle<T>::operator TRectangle<TR>() const
    {
        return TRectangle<TR>(Convert<TR, T>(left), Convert<TR, T>(top),
            Convert<TR, T>(right), Convert<TR, T>(bottom));
    }

#ifdef SIMD_OPENCV_ENABLE
    template <typename T> template <class TR>
    SIMD_INLINE Rectangle<T>::operator cv::Rect_<TR>() const
    {
        return cv::Rect_<TR>(Convert<TR, T>(left), Convert<TR, T>(top),
            Convert<TR, T>(right - left), Convert<TR, T>(bottom - top));
    }
#endif

    template <typename T> template <typename TR>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::operator = (const Rectangle<TR> & r)
    {
        left = Convert<T, TR>(r.left);
        top = Convert<T, TR>(r.top);
        right = Convert<T, TR>(r.right);
        bottom = Convert<T, TR>(r.bottom);
        return *this;
    }

#ifdef SIMD_OPENCV_ENABLE
    template <typename T> template <class TR>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::operator = (const cv::Rect_<TR> & r)
    {
        left = Convert<T, TR>(r.x);
        top = Convert<T, TR>(r.y);
        right = Convert<T, TR>(r.x + r.width);
        bottom = Convert<T, TR>(r.y + r.height);
        return *this;
    }
#endif

    template <typename T> template <typename TL>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetLeft(const TL & l)
    {
        left = Convert<T, TL>(l);
        return *this;
    }

    template <typename T> template <typename TT>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetTop(const TT & t)
    {
        top = Convert<T, TT>(t);
        return *this;
    }

    template <typename T> template <typename TR>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetRight(const TR & r)
    {
        right = Convert<T, TR>(r);
        return *this;
    }

    template <typename T> template <typename TB>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetBottom(const TB & b)
    {
        bottom = Convert<T, TB>(b);
        return *this;
    }

    template <typename T> template <typename TP>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetTopLeft(const Point<TP> & topLeft)
    {
        left = Convert<T, TP>(topLeft.x);
        top = Convert<T, TP>(topLeft.y);
        return *this;
    }

    template <typename T> template <typename TP>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetTopRight(const Point<TP> & topRight)
    {
        right = Convert<T, TP>(topRight.x);
        top = Convert<T, TP>(topRight.y);
        return *this;
    }

    template <typename T> template <typename TP>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetBottomLeft(const Point<TP> & bottomLeft)
    {
        left = Convert<T, TP>(bottomLeft.x);
        bottom = Convert<T, TP>(bottomLeft.y);
        return *this;
    }

    template <typename T> template <typename TP>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetBottomRight(const Point<TP> & bottomRight)
    {
        right = Convert<T, TP>(bottomRight.x);
        bottom = Convert<T, TP>(bottomRight.y);
        return *this;
    }

    template <typename T>
    SIMD_INLINE T Rectangle<T>::Left() const
    {
        return left;
    }

    template <typename T>
    SIMD_INLINE T Rectangle<T>::Top() const
    {
        return top;
    }

    template <typename T>
    SIMD_INLINE T Rectangle<T>::Right() const
    {
        return right;
    }

    template <typename T>
    SIMD_INLINE T Rectangle<T>::Bottom() const
    {
        return bottom;
    }

    template <typename T>
    SIMD_INLINE Point<T> Rectangle<T>::TopLeft() const
    {
        return Point<T>(left, top);
    }

    template <typename T>
    SIMD_INLINE Point<T> Rectangle<T>::TopRight() const
    {
        return Point<T>(right, top);
    }

    template <typename T>
    SIMD_INLINE Point<T> Rectangle<T>::BottomLeft() const
    {
        return Point<T>(left, bottom);
    }

    template <typename T>
    SIMD_INLINE Point<T> Rectangle<T>::BottomRight() const
    {
        return Point<T>(right, bottom);
    }

    template <typename T>
    SIMD_INLINE T Rectangle<T>::Width() const
    {
        return right - left;
    }

    template <typename T>
    SIMD_INLINE T Rectangle<T>::Height() const
    {
        return bottom - top;
    }

    template <typename T>
    SIMD_INLINE T Rectangle<T>::Area() const
    {
        return Width()*Height();
    }

    template <typename T>
    SIMD_INLINE bool Rectangle<T>::Empty() const
    {
        return Area() == 0;
    }

    template <typename T>
    SIMD_INLINE Point<T> Rectangle<T>::Size() const
    {
        return Point<T>(Width(), Height());
    }

    template <typename T>
    SIMD_INLINE Point<T> Rectangle<T>::Center() const
    {
        return Point<T>((left + right) / 2.0, (top + bottom) / 2.0);
    }

    template <typename T> template <typename TX, typename TY>
    SIMD_INLINE bool Rectangle<T>::Contains(TX x, TY y) const
    {
        Point<T> p(x, y);
        return p.x >= left && p.x < right && p.y >= top && p.y < bottom;
    }

    template <typename T> template <typename TP>
    SIMD_INLINE bool Rectangle<T>::Contains(const Point<TP> & p) const
    {
        return Contains(p.x, p.y);
    }

    template <typename T> template <typename TL, typename TT, typename TR, typename TB>
    SIMD_INLINE bool Rectangle<T>::Contains(TL l, TT t, TR r, TB b) const
    {
        Rectangle<T> rect(l, t, r, b);
        return rect.left >= left && rect.right <= right && rect.top >= top && rect.bottom <= bottom;
    }

    template <typename T> template <typename TR>
    SIMD_INLINE bool Rectangle<T>::Contains(const Rectangle <TR> & r) const
    {
        return Contains(r.left, r.top, r.right, r.bottom);
    }

    template <typename T> template <typename TP>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::Shift(const Point<TP> & shift)
    {
        return Shift(shift.x, shift.y);
    }

    template <typename T> template <typename TX, typename TY>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::Shift(TX shiftX, TY shiftY)
    {
        Point<T> shift(shiftX, shiftY);
        left += shift.x;
        top += shift.y;
        right += shift.x;
        bottom += shift.y;
        return *this;
    }

    template <typename T> template <typename TP>
    SIMD_INLINE Rectangle<T> Rectangle<T>::Shifted(const Point<TP> & shift) const
    {
        return Shifted(shift.x, shift.y);
    }

    template <typename T> template <typename TX, typename TY>
    SIMD_INLINE Rectangle<T> Rectangle<T>::Shifted(TX shiftX, TY shiftY) const
    {
        Point<T> shift(shiftX, shiftY);
        return Rectangle<T>(left + shift.x, top + shift.y, right + shift.x, bottom + shift.y);
    }

    template <typename T> template <typename TB>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::AddBorder(TB border)
    {
        T _border = Convert<T, TB>(border);
        left -= _border;
        top -= _border;
        right += _border;
        bottom += _border;
        return *this;
    }

    template <typename T> template <typename TR>
    SIMD_INLINE Rectangle<T> Rectangle<T>::Intersection(const Rectangle<TR> & rect) const
    {
        Rectangle<T> _rect(rect);
        T l = std::max(left, _rect.left);
        T t = std::max(top, _rect.top);
        T r = std::max(l, std::min(right, _rect.right));
        T b = std::max(t, std::min(bottom, _rect.bottom));
        return Rectangle(l, t, r, b);
    }

    /*! \cond PRIVATE */
    template <typename T> template <typename TP>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::operator &= (const Point<TP> & p)
    {
        Point<T> _p(p);
        if (Contains(_p))
        {
            left = _p.x;
            top = _p.y;
            right = _p.x + 1;
            bottom = _p.y + 1;
        }
        else
        {
            bottom = top;
            right = left;
        }
        return *this;
    }

    template <typename T> template <typename TR>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::operator &= (const Rectangle<TR> & r)
    {
        if (Empty())
            return *this;
        if (r.Empty())
            return this->operator=(r);

        Rectangle<T> _r(r);
        if (left < _r.left)
            left = std::min(_r.left, right);
        if (top < _r.top)
            top = std::min(_r.top, bottom);
        if (right > _r.right)
            right = std::max(_r.right, left);
        if (bottom > _r.bottom)
            bottom = std::max(_r.bottom, top);
        return *this;
    }
    /*! \endcond */

    template <typename T> template <typename TP>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::operator |= (const Point<TP> & p)
    {
        Point<T> _p(p);
        if (Empty())
        {
            left = _p.x;
            top = _p.y;
            right = _p.x + 1;
            bottom = _p.y + 1;
        }
        else
        {
            if (left > _p.x)
                left = _p.x;
            if (top > _p.y)
                top = _p.y;
            if (right <= _p.x)
                right = _p.x + 1;
            if (bottom <= _p.y)
                bottom = _p.y + 1;
        }
        return *this;
    }

    template <typename T> template <typename TR>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::operator |= (const Rectangle<TR> & r)
    {
        if (Empty())
            return this->operator=(r);
        if (r.Empty())
            return *this;

        Rectangle<T> _r(r);
        left = std::min(left, _r.left);
        top = std::min(top, _r.top);
        right = std::max(right, _r.right);
        bottom = std::max(bottom, _r.bottom);
        return *this;
    }

    template <typename T> template <typename TR>
    SIMD_INLINE Rectangle<T> & Rectangle<T>::operator += (const Rectangle<TR> & r)
    {
        left += Convert<T, TR>(r.left);
        top += Convert<T, TR>(r.top);
        right += Convert<T, TR>(r.right);
        bottom += Convert<T, TR>(r.bottom);
        return *this;
    }

    template <typename T>
    SIMD_INLINE bool Rectangle<T>::Overlaps(const Rectangle<T> & r) const
    {
        bool lr = left < r.right;
        bool rl = right > r.left;
        bool tb = top < r.bottom;
        bool bt = bottom > r.top;
        return (lr == rl) && (tb == bt);
    }

    // Rectangle<T> utilities implementation:

    template <typename T>
    SIMD_INLINE bool operator == (const Rectangle<T> & r1, const Rectangle<T> & r2)
    {
        return r1.left == r2.left && r1.top == r2.top && r1.right == r2.right && r1.bottom == r2.bottom;
    }

    template <typename T>
    SIMD_INLINE bool operator != (const Rectangle<T> & r1, const Rectangle<T> & r2)
    {
        return r1.left != r2.left || r1.top != r2.top || r1.right != r2.right || r1.bottom != r2.bottom;
    }

    template<class T1, class T2>
    SIMD_INLINE Rectangle<T1> operator / (const Rectangle<T1> & rect, const T2 & value)
    {
        return Rectangle<T1>(rect.left / value, rect.top / value, rect.right / value, rect.bottom / value);
    }

    template<class T1, class T2>
    SIMD_INLINE Rectangle<T1> operator * (const Rectangle<T1> & rect, const T2 & value)
    {
        return Rectangle<T1>(rect.left*value, rect.top*value, rect.right*value, rect.bottom*value);
    }

    template<class T1, class T2>
    SIMD_INLINE Rectangle<T1> operator * (const T2 & value, const Rectangle<T1> & rect)
    {
        return Rectangle<T1>(rect.left*value, rect.top*value, rect.right*value, rect.bottom*value);
    }

    template<class T>
    SIMD_INLINE Rectangle<T> operator + (const Rectangle<T> & r1, const Rectangle<T> & r2)
    {
        return Rectangle<T>(r1.left + r2.left, r1.top + r2.top, r1.right + r2.right, r1.bottom + r2.bottom);
    }
}
#endif//__SimdRectangle_hpp__
