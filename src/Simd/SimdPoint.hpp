/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2024-2024 Sergey Chezhin.
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
#ifndef __SimdPoint_hpp__
#define __SimdPoint_hpp__

#include "Simd/SimdLib.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#ifdef SIMD_OPENCV_ENABLE
#include "opencv2/core/core.hpp"
#endif

namespace Simd
{
    /*! @ingroup cpp_point

        \short 2D point (x, y) or size (width, height).

        Point<T> stores two coordinates of type T. The usual image coordinate
        system has its origin at the top-left corner: X grows to the right and
        Y grows downward. That is how Simd::DrawLine, Simd::DrawRectangle,
        Simd::Font::Draw, View::At and contour anchors use a
        Point<ptrdiff_t>. The same type is also a size: x is the width and y
        is the height. View::Size(), Detection::Size, Motion::Size,
        Pyramid::Recreate, Simd::Resize and Simd::TransformSize use it that
        way. Rectangle(point) treats the point as the bottom-right corner and
        sets the top-left corner to (0, 0), so Rectangle(view.Size()) is the
        full image. TestImageMatcher passes Rect(src.Size()) as the crop of
        Simd::ShiftBilinear.

        Point() is the origin (0, 0). Motion measures the squared diagonal of a
        frame with SquaredDistance(model.frameSize, Point()), which is
        width*width + height*height. ShiftDetector treats Point() as an unset
        frame size. Simd::TransformSize returns Point<ptrdiff_t>() for an
        unknown transform.

        Coordinates passed to the constructor are converted to T. Conversion
        of float or double to ptrdiff_t rounds to the nearest integer (half
        away from zero). Other conversions are a C-style cast.
        TestCheckCpp builds Point<ptrdiff_t>(1.4, 2.6), which becomes (1, 3),
        and Point<double>(1.4, 3.6), which keeps the fractional values.
        Assignment and the converting constructor copy x and y the same way.
        TestShift builds ShiftDetector::FPoint from an integer Point to compare
        a refined sub-pixel shift.

        Point<double> is a sub-pixel shift. Simd::ShiftBilinear takes
        Point<double> as the shift along X and Y. The image-matcher test uses
        FPoint(Random() * 2 - 1, Random() * 2 - 1). ShiftDetector::FPoint is
        the refined shift added to the integer shift. Motion::FPoint is a
        different use of the same type: an ONVIF coordinate in [-1, 1] whose
        origin is the screen center and whose Y axis grows upward. Motion::FSize
        stores an ONVIF size in [0, 2]. Those ranges belong to the Motion
        aliases, not to Point itself.

        Point<float> is the source position of a destination corner after the
        affine matrix in WarpAffine. The mapped point is
        Point(x * m[0] + y * m[1] + m[2], x * m[3] + y * m[4] + m[5]).

        Arithmetic is component-wise. Motion flood-fill walks
        current + Point(-1, 0) and the other 4-neighbours, then reads
        View::At(neighbour). ExpandRoi scales a parent corner onto the next
        pyramid level with TopLeft() * 2 - Point(1, 1) and
        BottomRight() * 2 + Point(1, 1). Simd::SquaredDistance compares object
        centers and trajectory points without a square root. Pyramid::Scale
        halves a size with (x + 1) >> 1 on each coordinate; the Point shift
        operators shift both coordinates and do not add 1.

        With SIMD_OPENCV_ENABLE, a point converts to and from cv::Point (x, y)
        and cv::Size (width, height), and it converts to cv::Point2f.

        Using example:
        \code
        #include "Simd/SimdLib.hpp"
        #include "Simd/SimdDrawing.hpp"

        int main()
        {
            typedef Simd::Point<ptrdiff_t> Point;
            typedef Simd::Point<double> FPoint;
            typedef Simd::View<Simd::Allocator> View;
            typedef Simd::Rectangle<ptrdiff_t> Rect;

            View image(320, 240, View::Gray8);
            Point size = image.Size();
            View dst(size.x / 2, size.y / 2, View::Gray8);
            Simd::Resize(image, dst, Point(160, 120), SimdResizeMethodArea);

            Point a(10, 20), b(40, 80);
            if (Simd::SquaredDistance(a, b) < 10000)
                Simd::DrawLine(image, a, b, uint8_t(255), 1);

            Point neighbour = a + Point(1, 0);
            Point child = a * 2 - Point(1, 1);
            uint8_t value = image.At<uint8_t>(neighbour);

            Point rounded(1.4, 2.6);
            FPoint shift(0.5, -0.25);
            View shifted(image.Size(), View::Gray8);
            Simd::ShiftBilinear(image, image, shift, Rect(image.Size()), shifted);

            return rounded.x + child.y + (int)value;
        }
        \endcode

        OpenCV conversion (define SIMD_OPENCV_ENABLE before including this header):
        \code
        #include "opencv2/core/core.hpp"
        #define SIMD_OPENCV_ENABLE
        #include "Simd/SimdPoint.hpp"

        int main()
        {
            typedef Simd::Point<ptrdiff_t> Point;

            cv::Size cvSize;
            cv::Point cvPoint;
            Point simdPoint;

            simdPoint = cvPoint;
            simdPoint = cvSize;
            cvSize = simdPoint;
            cvPoint = simdPoint;
            return 0;
        }
        \endcode

        \ref cpp_point_functions.
    */
    template <typename T>
    struct Point
    {
        typedef T Type; /*!< Coordinate type. ptrdiff_t is a pixel or a size. double is a sub-pixel shift. */

        T x; /*!< \brief Horizontal coordinate, or width when the point is a size. Grows to the right. */
        T y; /*!< \brief Vertical coordinate, or height when the point is a size. Grows downward in image space. */

        /*!
            Creates a point (0, 0).

            Point() is the image origin and an empty size. Motion measures the
            squared diagonal of a frame with SquaredDistance(frameSize, Point()).
            ShiftDetector compares the frame size with Point() before estimation.
        */
        Point();

        /*!
            Creates a point from two coordinates.

            Each coordinate is converted to T. float and double values converted
            to ptrdiff_t are rounded to the nearest integer, with halves rounded
            away from zero. Point<ptrdiff_t>(1.4, 2.6) is therefore (1, 3).
            Point<double> keeps the fractional values. Drawing, font placement,
            contour anchors and motion screen points pass integer pixel
            coordinates. Detection::Init and Simd::Resize pass a size as
            Point(width, height). Motion builds ONVIF corners as
            FPoint(-1.0, 1.0). WarpAffine builds Point<float> from the affine
            mapping of a corner.

            \param [in] tx - initial X value. It is the width when the point is a size.
            \param [in] ty - initial Y value. It is the height when the point is a size.
        */
       template <typename TX, typename TY> SIMD_CONSTEXPR Point(TX tx, TY ty);

        /*!
            Creates a point from another point type that has x and y.

            Both coordinates are converted to T. This copies between
            Point<ptrdiff_t> and Point<double>. TestShift wraps an integer
            shift in ShiftDetector::FPoint before SquaredDistance. With
            SIMD_OPENCV_ENABLE the same constructor reads cv::Point_<T>::x
            and cv::Point_<T>::y, so a Simd point can be assigned from cv::Point.

            \param [in] p - a point of arbitrary type with x and y fields.
        */
        template <class TP, template<class> class TPoint> Point(const TPoint<TP> & p);

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Creates a point from an OpenCV size.

            x is set from size.width and y from size.height. Both values are
            converted to T, so a floating OpenCV size assigned to
            Point<ptrdiff_t> is rounded. This is the conversion used by
            simdPoint = cvSize.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] size - an OpenCV size. width becomes x and height becomes y.
        */
        template <class TS> Point(const cv::Size_<TS> & size);
#endif

        /*!
            Destroys the point. The destructor has no side effects.
        */
        SIMD_CONSTEXPR ~Point();

        /*!
            Converts this point to another point type constructed from (x, y).

            Coordinates are converted to the target type. With SIMD_OPENCV_ENABLE
            this is how a Simd point is assigned to cv::Point and to cv::Size
            (cv::Size receives x as width and y as height).

            \return a point of arbitrary type.
        */
        template <class TP, template<class> class TPoint> operator TPoint<TP>() const;

        /*!
            Copies another point into this point.

            x and y are converted to T. Assigning Point<double> to
            Point<ptrdiff_t> rounds each coordinate.

            \param [in] p - a point of arbitrary coordinate type.
            \return a reference to itself.
        */
        template <typename TP> Point & operator = (const Point<TP> & p);

        /*!
            Adds another point to this point.

            The added coordinates are converted to T and then added to x and y.
            The free operator + used by the motion flood-fill
            (current + Point(dx, dy)) builds a new point with the same
            component-wise sum.

            \param [in] p - a point of arbitrary coordinate type.
            \return a reference to itself.
        */
        template <typename TP> Point & operator += (const Point<TP> & p);

        /*!
            Subtracts another point from this point.

            The subtracted coordinates are converted to T. The free operator -
            used by ExpandRoi (corner * 2 - Point(1, 1)) builds a new point
            with the same component-wise difference.

            \param [in] p - a point of arbitrary coordinate type.
            \return a reference to itself.
        */
        template <typename TP> Point & operator -= (const Point<TP> & p);

        /*!
            Multiplies both coordinates by a scalar.

            Each product is converted back to T. A floating factor applied to
            Point<ptrdiff_t> is rounded. The free operator * used by ExpandRoi
            (TopLeft() * 2) returns a new point and does not modify this one.
            ShiftDetector doubles a coarse shift by writing shift.x and shift.y,
            one pyramid level at a time, rather than by calling this operator.

            \param [in] a - a factor of arbitrary type.
            \return a reference to itself.
        */
        template <typename TA> Point & operator *= (const TA & a);

        /*!
            Divides both coordinates by a double precision value.

            Each quotient is converted back to T. Point<ptrdiff_t> therefore
            rounds the result, including halves away from zero. The divisor is
            double; there is no overload for an integer divisor. Dividing two
            points by each other is the free operator / and is component-wise.

            \param [in] a - a divider.
            \return a reference to itself.
        */
        Point & operator /= (double a);

        /*!
            Returns a point with both coordinates shifted left by the same bit count.

            The operator is meaningful for integer coordinate types. It returns
            a new point and does not modify this one. Pyramid::Scale does not
            use it: that function halves a size with (coordinate + 1) >> 1 so
            odd sizes round up.

            \param [in] shift - a non-negative bit count.
            \return a new point with shifted coordinates.
        */
        Point operator << (ptrdiff_t shift) const;

        /*!
            Returns a point with both coordinates shifted right by the same bit count.

            The operator is meaningful for integer coordinate types. It returns
            a new point and does not modify this one. A right shift by 1 halves
            both coordinates and truncates toward negative infinity for
            non-negative values. Pyramid::Scale adds 1 before that shift.

            \param [in] shift - a non-negative bit count.
            \return a new point with shifted coordinates.
        */
        Point operator >> (ptrdiff_t shift) const;

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Converts this point to cv::Point2f.

            x and y are passed to the cv::Point2f constructor. This conversion
            is available only when SIMD_OPENCV_ENABLE is defined. Assignment to
            cv::Point and cv::Size uses the converting operator to an arbitrary
            point type.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \return a cv::Point2f with the same coordinates.
        */
        operator cv::Point2f() const;
#endif
    };

    /*! @ingroup cpp_point_functions

        \fn template <typename T> bool operator == (const Point<T> & p1, const Point<T> & p2);

        \short Compares two points on equality of both coordinates.

        x and y are compared independently. Pyramid::Recreate skips rebuilding
        when the requested size compares equal to the size of pyramid level 0.
        Motion::Calibrate keeps the current model when the new frame size
        compares equal to originalFrameSize.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return true when both coordinates are equal.
    */
    template <typename T> bool operator == (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> bool operator != (const Point<T> & p1, const Point<T> & p2);

        \short Compares two points on inequality of either coordinate.

        ShiftDetector::Estimate requires the initialized frame size to differ
        from Point() before it accepts a current image.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return true when x or y differs.
    */
    template <typename T> bool operator != (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> Point<T> operator + (const Point<T> & p1, const Point<T> & p2);

        \short Adds two points component-wise.

        The result is (p1.x + p2.x, p1.y + p2.y). Motion segmentation walks the
        4-neighbourhood with current + Point(-1, 0), current + Point(0, -1),
        current + Point(1, 0) and current + Point(0, 1), then passes the sum
        to View::At.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return the component-wise sum.
    */
    template <typename T> Point<T> operator + (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> Point<T> operator - (const Point<T> & p1, const Point<T> & p2);

        \short Subtracts two points component-wise.

        The result is (p1.x - p2.x, p1.y - p2.y). ExpandRoi places the child
        top-left corner at parentTopLeft * 2 - Point(1, 1). SquaredDistance
        subtracts the two points before squaring the components.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return the component-wise difference.
    */
    template <typename T> Point<T> operator - (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> Point<T> operator * (const Point<T> & p1, const Point<T> & p2);

        \short Multiplies two points component-wise.

        The result is (p1.x * p2.x, p1.y * p2.y). This is not a dot product
        and not a scale of one point by a scalar. Use DotProduct for the sum
        of component products, and operator*(point, scalar) to scale both
        coordinates by one value.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return the component-wise product.
    */
    template <typename T> Point<T> operator * (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> Point<T> operator / (const Point<T> & p1, const Point<T> & p2);

        \short Divides two points component-wise.

        The result is (p1.x / p2.x, p1.y / p2.y). Integer coordinates use
        integer division and truncate toward zero. Use operator/(point, scalar)
        to divide both coordinates by one value.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point. Neither coordinate may be zero.
        \return the component-wise quotient.
    */
    template <typename T> Point<T> operator / (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> Point<T> operator - (const Point<T> & p);

        \short Returns the point with both coordinates negated.

        The result is (-p.x, -p.y). A one-pixel step toward the origin can be
        written as a negative point, for example Point(-1, 0), which the motion
        flood-fill stores in its neighbour table.

        \param [in] p - an original point.
        \return the negated point.
    */
    template <typename T> Point<T> operator - (const Point<T> & p);

    /*! @ingroup cpp_point_functions

        \fn template <typename TP, typename TA> Point<TP> operator / (const Point<TP> & p, const TA & a);

        \short Divides both coordinates by a scalar.

        The result is Point<TP>(p.x / a, p.y / a). Construction converts the
        quotients back to TP, so Point<ptrdiff_t> divided by a floating value
        is rounded to the nearest integer. The member operator /= always
        divides by double and writes the rounded values into the same point.

        \param [in] p - a point.
        \param [in] a - a non-zero scalar value.
        \return the scaled point.
    */
    template <typename TP, typename TA> Point<TP> operator / (const Point<TP> & p, const TA & a);

    /*! @ingroup cpp_point_functions

        \fn template <typename TP, typename TA> Point<TP> operator * (const Point<TP> & p, const TA & a);

        \short Multiplies both coordinates by a scalar.

        The result is Point<TP>(p.x * a, p.y * a). Motion::ExpandRoi maps a
        rectangle from a parent pyramid level to the child level with
        TopLeft() * 2 and BottomRight() * 2. A floating factor applied to
        Point<ptrdiff_t> is rounded by the Point constructor. The member
        operator *= writes the same products into the original point.

        \param [in] p - a point.
        \param [in] a - a scalar value.
        \return the scaled point.
    */
    template <typename TP, typename TA> Point<TP> operator * (const Point<TP> & p, const TA & a);

    /*! @ingroup cpp_point_functions

        \fn template <typename TP, typename TA> Point<TP> operator * (const TA & a, const Point<TP> & p);

        \short Multiplies a scalar by both coordinates of a point.

        The result is the same as p * a: Point<TP>(p.x * a, p.y * a).
        Point<ptrdiff_t> constructed from a floating product is rounded.

        \param [in] a - a scalar value.
        \param [in] p - a point.
        \return the scaled point.
    */
    template <typename TP, typename TA> Point<TP> operator * (const TA & a, const Point<TP> & p);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> T SquaredDistance(const Point<T> & p1, const Point<T> & p2);

        \short Returns the squared Euclidean distance between two points.

        The value is (p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y)
        and has type T. Callers compare distances without computing a square
        root. Motion picks the tracked object nearest to a moving region by the
        smallest SquaredDistance between object->center and region.rect.Center().
        An object becomes Moving when
        SquaredDistance(trajectory.back()->point, pointStart) reaches
        squareShiftMin. That threshold is
        SquaredDistance(frameSize, Point()) * shiftMin * shiftMin, the squared
        frame diagonal scaled by the relative shift. TestShift rejects a
        detection when SquaredDistance(FPoint(appliedShift), refinedShift) is
        greater than 1. For integer T the sum of squares can overflow.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return the squared distance. It is 0 when the points are equal.
    */
    template <typename T> T SquaredDistance(const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> double Distance(const Point<T> & p1, const Point<T> & p2);

        \short Returns the Euclidean distance between two points.

        The value is the square root of SquaredDistance, always as double.
        Motion tracking and the shift test compare SquaredDistance instead, so
        they avoid the square root and keep the comparison in the coordinate
        type.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return the distance. It is 0 when the points are equal.
    */
    template <typename T> double Distance(const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> T DotProduct(const Point<T> & p1, const Point<T> & p2);

        \short Returns the dot product of two points.

        The value is p1.x * p2.x + p1.y * p2.y and has type T. It is the sum
        of the component-wise products, not the point returned by operator*.
        For integer T the sum can overflow. This is unrelated to the
        depthwise-convolution dot product used by Synet.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return the dot product.
    */
    template <typename T> T DotProduct(const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> T CrossProduct(const Point<T> & p1, const Point<T> & p2);

        \short Returns the z component of the 2D cross product.

        The value is p1.x * p2.y - p1.y * p2.x and has type T. Its sign is the
        orientation of the two vectors and its magnitude is the area of the
        parallelogram they span. For integer T the products can overflow.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return the signed cross product.
    */
    template <typename T> T CrossProduct(const Point<T> & p1, const Point<T> & p2);

    //-------------------------------------------------------------------------

    // struct Point<T> implementation:

#ifndef SIMD_ROUND
#define SIMD_ROUND
    SIMD_INLINE int Round(double value)
    {
        return (int)(value + (value >= 0 ? 0.5 : -0.5));
    }
#endif

    template <class TD, class TS>
    SIMD_INLINE SIMD_CONSTEXPR TD Convert(TS src)
    {
        return (TD)src;
    }

    template <>
    SIMD_INLINE ptrdiff_t Convert<ptrdiff_t, double>(double src)
    {
        return Round(src);
    }

    template <>
    SIMD_INLINE ptrdiff_t Convert<ptrdiff_t, float>(float src)
    {
        return Round(src);
    }

    template <typename T>
    SIMD_INLINE Point<T>::Point()
        : x(0)
        , y(0)
    {
    }

    template <typename T> template <typename TX, typename TY>
    SIMD_INLINE SIMD_CONSTEXPR Point<T>::Point(TX tx, TY ty)
        : x(Convert<T, TX>(tx))
        , y(Convert<T, TY>(ty))
    {
    }

    template <typename T> template <class TP, template<class> class TPoint>
    SIMD_INLINE Point<T>::Point(const TPoint<TP> & p)
        : x(Convert<T, TP>(p.x))
        , y(Convert<T, TP>(p.y))
    {
    }

#ifdef SIMD_OPENCV_ENABLE
    template <typename T> template <class TS>
    SIMD_INLINE Point<T>::Point(const cv::Size_<TS> & size)
        : x(Convert<T, TS>(size.width))
        , y(Convert<T, TS>(size.height))
    {
    }
#endif

    template <typename T>
    SIMD_INLINE SIMD_CONSTEXPR Point<T>::~Point()
    {
    }

    template <typename T> template <class TP, template<class> class TPoint>
    SIMD_INLINE Point<T>::operator TPoint<TP>() const
    {
        return TPoint<TP>(Convert<TP, T>(x), Convert<TP, T>(y));
    }

    template <typename T> template <typename TP>
    SIMD_INLINE Point<T> & Point<T>::operator = (const Point<TP> & p)
    {
        x = Convert<T, TP>(p.x);
        y = Convert<T, TP>(p.y);
        return *this;
    }

    template <typename T> template <typename TP>
    SIMD_INLINE Point<T> & Point<T>::operator += (const Point<TP> & p)
    {
        x += Convert<T, TP>(p.x);
        y += Convert<T, TP>(p.y);
        return *this;
    }

    template <typename T> template <typename TP>
    SIMD_INLINE Point<T> & Point<T>::operator -= (const Point<TP> & p)
    {
        x -= Convert<T, TP>(p.x);
        y -= Convert<T, TP>(p.y);
        return *this;
    }

    template <typename T> template <typename TA>
    SIMD_INLINE Point<T> & Point<T>::operator *= (const TA & a)
    {
        x = Convert<T, TA>(x*a);
        y = Convert<T, TA>(y*a);
        return *this;
    }

    template <typename T>
    SIMD_INLINE Point<T> & Point<T>::operator /= (double a)
    {
        x = Convert<T, double>(x / a);
        y = Convert<T, double>(y / a);
        return *this;
    }

    template <typename T>
    SIMD_INLINE Point<T> Point<T>::operator << (ptrdiff_t shift) const
    {
        return Point<T>(x << shift, y << shift);
    }

    template <typename T>
    SIMD_INLINE Point<T> Point<T>::operator >> (ptrdiff_t shift) const
    {
        return Point<T>(x >> shift, y >> shift);
    }

#ifdef SIMD_OPENCV_ENABLE
    template<typename T>
    SIMD_INLINE Simd::Point<T>::operator cv::Point2f() const
    {
        return cv::Point2f(x, y);
    }
#endif //SIMD_OPENCV_ENABLE

    // Point<T> utilities implementation:

    template <typename T>
    SIMD_INLINE bool operator == (const Point<T> & p1, const Point<T> & p2)
    {
        return p1.x == p2.x && p1.y == p2.y;
    }

    template <typename T>
    SIMD_INLINE bool operator != (const Point<T> & p1, const Point<T> & p2)
    {
        return p1.x != p2.x || p1.y != p2.y;
    }

    template <typename T>
    SIMD_INLINE Point<T> operator + (const Point<T> & p1, const Point<T> & p2)
    {
        return Point<T>(p1.x + p2.x, p1.y + p2.y);
    }

    template <typename T>
    SIMD_INLINE Point<T> operator - (const Point<T> & p1, const Point<T> & p2)
    {
        return Point<T>(p1.x - p2.x, p1.y - p2.y);
    }

    template <typename T>
    SIMD_INLINE Point<T> operator * (const Point<T> & p1, const Point<T> & p2)
    {
        return Point<T>(p1.x * p2.x, p1.y * p2.y);
    }

    template <typename T>
    SIMD_INLINE Point<T> operator / (const Point<T> & p1, const Point<T> & p2)
    {
        return Point<T>(p1.x / p2.x, p1.y / p2.y);
    }

    template <typename T>
    SIMD_INLINE Point<T> operator - (const Point<T> & p)
    {
        return Point<T>(-p.x, -p.y);
    }

    template <typename TP, typename TA>
    SIMD_INLINE Point<TP> operator / (const Point<TP> & p, const TA & a)
    {
        return Point<TP>(p.x / a, p.y / a);
    }

    template <typename TP, typename TA>
    SIMD_INLINE Point<TP> operator * (const Point<TP> & p, const TA & a)
    {
        return Point<TP>(p.x*a, p.y*a);
    }

    template <typename TP, typename TA>
    SIMD_INLINE Point<TP> operator * (const TA & a, const Point<TP> & p)
    {
        return Point<TP>(p.x*a, p.y*a);
    }

    template <typename T>
    SIMD_INLINE T SquaredDistance(const Point<T> & p1, const Point<T> & p2)
    {
        Point<T> dp = p2 - p1;
        return dp.x*dp.x + dp.y*dp.y;
    }

    template <typename T>
    SIMD_INLINE double Distance(const Point<T> & p1, const Point<T> & p2)
    {
        return ::sqrt(double(SquaredDistance(p1, p2)));
    }

    template <typename T>
    SIMD_INLINE T DotProduct(const Point<T> & p1, const Point<T> & p2)
    {
        return (p1.x * p2.x + p1.y * p2.y);
    }

    template <typename T>
    SIMD_INLINE T CrossProduct(const Point<T> & p1, const Point<T> & p2)
    {
        return (p1.x * p2.y - p1.y * p2.x);
    }
}
#endif//__SimdPoint_hpp__
