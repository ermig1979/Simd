/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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

#include <math.h>

#ifdef SIMD_OPENCV_ENABLE
#include "opencv2/core/core.hpp"
#endif

namespace Simd
{
    /*! @ingroup cpp_point

        \short The Point structure defines the x- and y-coordinates of a point.

        In order to have mutual conversion with OpenCV cv::Point and cv::Size types you have to define macro SIMD_OPENCV_ENABLE:
        \verbatim
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
        \endverbatim

        \ref cpp_point_functions.
    */
    template <typename T>
    struct Point
    {
        typedef T Type; /*!< Type definition. */

        T x; /*!< \brief Specifies the x-coordinate of a point. */
        T y; /*!< \brief Specifies the y-coordinate of a point. */

        /*!
            Creates a new Point structure that contains the default (0, 0) coordinates.
        */
        Point();

        /*!
            Creates a new Point structure that contains the specified coordinates.

            \param [in] tx - initial X value.
            \param [in] ty - initial Y value.
        */
        template <typename TX, typename TY> Point(TX tx, TY ty);

        /*!
            Creates a new Point structure on the base of another point of arbitrary type.

            \param [in] p - a point of arbitrary type.
        */
        template <class TP, template<class> class TPoint> Point(const TPoint<TP> & p);

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Creates a new Point structure on the base of OpenCV size type.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] size - an OpenCV size.
        */
        template <class TS> Point(const cv::Size_<TS> & size);
#endif

        /*!
            A point destructor.
        */
        ~Point();

        /*!
            Converts itself to point of arbitrary type.

            \return a point of arbitrary type.
        */
        template <class TP, template<class> class TPoint> operator TPoint<TP>() const;

        /*!
            Performs copying from point of arbitrary type.

            \param [in] p - a point of arbitrary type.
            \return a reference to itself.
        */
        template <typename TP> Point & operator = (const Point<TP> & p);

        /*!
            Adds to itself point of arbitrary type.

            \param [in] p - a point of arbitrary type.
            \return a reference to itself.
        */
        template <typename TP> Point & operator += (const Point<TP> & p);

        /*!
            Subtracts from itself point of arbitrary type.

            \param [in] p - a point of arbitrary type.
            \return a reference to itself.
        */
        template <typename TP> Point & operator -= (const Point<TP> & p);

        /*!
            Multiplies itself by value of arbitrary type.

            \param [in] a - a factor of arbitrary type.
            \return a reference to itself.
        */
        template <typename TA> Point & operator *= (const TA & a);

        /*!
            Divides itself into given value.

            \param [in] a - a value of divider.
            \return a reference to itself.
        */
        Point & operator /= (double a);

        /*!
            Performs shift bit left for value of point coordinates.

            \note It function is actual for integer types of Point.

            \param [in] shift - a shift value.
            \return a new point with shifted coordinates.
        */
        Point operator << (ptrdiff_t shift) const;

        /*!
            Performs shift bit right for value of point coordinates.

            \note It function is actual for integer types of Point.

            \param [in] shift - a shift value.
            \return a new point with shifted coordinates.
        */
        Point operator >> (ptrdiff_t shift) const;

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Casts simd point to OpenCV Point2f

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.
        */
        operator cv::Point2f() const;
#endif
    };

    /*! @ingroup cpp_point_functions

        \fn template <typename T> bool operator == (const Point<T> & p1, const Point<T> & p2);

        \short Compares two points on equality.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return a result of comparison.
    */
    template <typename T> bool operator == (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> bool operator != (const Point<T> & p1, const Point<T> & p2);

        \short Compares two points on inequality.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return a result of comparison.
    */
    template <typename T> bool operator != (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> Point<T> operator + (const Point<T> & p1, const Point<T> & p2);

        \short Adds two points.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return a result of addition.
    */
    template <typename T> Point<T> operator + (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> Point<T> operator - (const Point<T> & p1, const Point<T> & p2);

        \short Subtracts two points.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return a result of subtraction.
    */
    template <typename T> Point<T> operator - (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> Point<T> operator * (const Point<T> & p1, const Point<T> & p2);

        \short Multiplies two points.

        \note Coordinates of the points are multiplied independently.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return a result of multiplication.
    */
    template <typename T> Point<T> operator * (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> Point<T> operator / (const Point<T> & p1, const Point<T> & p2);

        \short Divides two points.

        \note Coordinates of the points are divided independently.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return a result of division.
    */
    template <typename T> Point<T> operator / (const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> Point<T> operator - (const Point<T> & p);

        \short Returns point with coordinates with the opposite sign.

        \param [in] p - an original point.
        \return a result of the operation.
    */
    template <typename T> Point<T> operator - (const Point<T> & p);

    /*! @ingroup cpp_point_functions

        \fn template <typename TP, typename TA> Point<TP> operator / (const Point<TP> & p, const TA & a);

        \short Divides the point on the scalar value.

        \param [in] p - a point.
        \param [in] a - a scalar value.
        \return a result of division.
    */
    template <typename TP, typename TA> Point<TP> operator / (const Point<TP> & p, const TA & a);

    /*! @ingroup cpp_point_functions

        \fn template <typename TP, typename TA> Point<TP> operator * (const Point<TP> & p, const TA & a);

        \short Multiplies the point on the scalar value.

        \param [in] p - a point.
        \param [in] a - a scalar value.
        \return a result of multiplication.
    */
    template <typename TP, typename TA> Point<TP> operator * (const Point<TP> & p, const TA & a);

    /*! @ingroup cpp_point_functions

        \fn template <typename TP, typename TA> Point<TP> operator * (const TA & a, const Point<TP> & p);

        \short Multiplies the scalar value on the point.

        \param [in] a - a scalar value.
        \param [in] p - a point.
        \return a result of multiplication.
    */
    template <typename TP, typename TA> Point<TP> operator * (const TA & a, const Point<TP> & p);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> T SquaredDistance(const Point<T> & p1, const Point<T> & p2);

        \short Gets squared distance between two points.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return a squared distance between them.
    */
    template <typename T> T SquaredDistance(const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> double Distance(const Point<T> & p1, const Point<T> & p2);

        \short Gets distance between two points.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return a distance between them.
    */
    template <typename T> double Distance(const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> T DotProduct(const Point<T> & p1, const Point<T> & p2);

        \short Gets dot product of two points.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return a dot product.
    */
    template <typename T> T DotProduct(const Point<T> & p1, const Point<T> & p2);

    /*! @ingroup cpp_point_functions

        \fn template <typename T> T CrossProduct(const Point<T> & p1, const Point<T> & p2);

        \short Gets cross product of two points.

        \param [in] p1 - a first point.
        \param [in] p2 - a second point.
        \return a cross product.
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
    SIMD_INLINE TD Convert(TS src)
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
    SIMD_INLINE Point<T>::Point(TX tx, TY ty)
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
    SIMD_INLINE Point<T>::~Point()
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
