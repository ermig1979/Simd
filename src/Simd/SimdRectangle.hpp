/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

        \short The Rectangle structure defines the positions of left, top, right and bottom sides of a rectangle.

        In order to have mutual conversion with OpenCV rectangle you have to define macro SIMD_OPENCV_ENABLE:
        \verbatim
        #include "opencv2/core/core.hpp"
        #define SIMD_OPENCV_ENABLE
        #include "Simd/SimdRectangle.hpp"

        int main()
        {
            typedef Simd::Rectangle<ptrdiff_t> Rect;

            cv::Rect cvRect;
            Rect simdRect;

            simdRect = cvRect;
            cvRect = simdRect;

            return 0;
        }
        \endverbatim

        \ref cpp_rectangle_functions.
    */
    template <typename T>
    struct Rectangle
    {
        typedef T Type; /*!< Type definition. */

        T left; /*!< \brief Specifies the position of left side of a rectangle. */
        T top; /*!< \brief Specifies the position of top side of a rectangle. */
        T right; /*!< \brief Specifies the position of right side of a rectangle. */
        T bottom; /*!< \brief Specifies the position of bottom side of a rectangle. */

        /*!
            Creates a new Rectangle structure that contains the default (0, 0, 0, 0) positions of its sides.
        */
        Rectangle();

        /*!
            Creates a new Rectangle structure that contains the specified positions of its sides.

            \param [in] l - initial left value.
            \param [in] t - initial top value.
            \param [in] r - initial right value.
            \param [in] b - initial bottom value.
        */
        template <typename TL, typename TT, typename TR, typename TB> Rectangle(TL l, TT t, TR r, TB b);

        /*!
            Creates a new Rectangular structure that contains the specified coordinates of its left-top and right-bottom corners.

            \param [in] lt - initial coordinates of left-top corner.
            \param [in] rb - initial coordinates of right-bottom corner.
        */
        template <typename TLT, typename TRB> Rectangle(const Point<TLT> & lt, const Point<TRB> & rb);

        /*!
            Creates a new Rectangular structure that contains the specified coordinates of its right-bottom corner.
            The coordinates of left-top corner is set to (0, 0).

            \param [in] rb - initial coordinates of right-bottom corner.
        */
        template <typename TRB> Rectangle(const Point<TRB> & rb);

        /*!
            Creates a new Rectangle structure on the base of another rectangle of arbitrary type.

            \param [in] r - a rectangle of arbitrary type.
        */
        template <class TR, template<class> class TRectangle> Rectangle(const TRectangle<TR> & r);

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Creates a new Rectangle structure on the base of OpenCV rectangle.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] r - an OpenCV rectangle.
        */
        template <class TR> Rectangle(const cv::Rect_<TR> & r);
#endif

        /*!
            A rectangle destructor.
        */
        ~Rectangle();

        /*!
            Converts itself to rectangle of arbitrary type.

            \return a rectangle of arbitrary type.
        */
        template <class TR, template<class> class TRectangle> operator TRectangle<TR>() const;

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Converts itself to OpenCV rectangle.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \return an OpenCV rectangle.
        */
        template <class TR> operator cv::Rect_<TR>() const;
#endif

        /*!
            Performs copying from rectangle of arbitrary type.

            \param [in] r - a rectangle of arbitrary type.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & operator = (const Rectangle<TR> & r);

#ifdef SIMD_OPENCV_ENABLE
        /*!
            Performs copying from OpenCV rectangle.

            \note You have to define SIMD_OPENCV_ENABLE in order to use this functionality.

            \param [in] r - an OpenCV rectangle.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & operator = (const cv::Rect_<TR> & r);
#endif

        /*!
            Sets position of left side.

            \param [in] l - a new position of left side.
            \return a reference to itself.
        */
        template <typename TL> Rectangle<T> & SetLeft(const TL & l);

        /*!
            Sets position of top side.

            \param [in] t - a new position of top side.
            \return a reference to itself.
        */
        template <typename TT> Rectangle<T> & SetTop(const TT & t);

        /*!
            Sets position of right side.

            \param [in] r - a new position of right side.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & SetRight(const TR & r);

        /*!
            Sets position of bottom side.

            \param [in] b - a new position of bottom side.
            \return a reference to itself.
        */
        template <typename TB> Rectangle<T> & SetBottom(const TB & b);

        /*!
            Sets coordinates of top-left corner.

            \param [in] topLeft - a new coordinates of top-left corner.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & SetTopLeft(const Point<TP> & topLeft);

        /*!
            Sets coordinates of top-right corner.

            \param [in] topRight - a new coordinates of top-right corner.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & SetTopRight(const Point<TP> & topRight);

        /*!
            Sets coordinates of bottom-left corner.

            \param [in] bottomLeft - a new coordinates of bottom-left corner.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & SetBottomLeft(const Point<TP> & bottomLeft);

        /*!
            Sets coordinates of bottom-right corner.

            \param [in] bottomRight - a new coordinates of bottom-right corner.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & SetBottomRight(const Point<TP> & bottomRight);

        /*!
            Gets position of left side.

            \return a position of left side.
        */
        T Left() const;

        /*!
            Gets position of top side.

            \return a position of top side.
        */
        T Top() const;

        /*!
            Gets position of right side.

            \return a position of right side.
        */
        T Right() const;

        /*!
            Gets position of bottom side.

            \return a position of bottom side.
        */
        T Bottom() const;

        /*!
            Gets coordinates of top-left corner.

            \return a point with coordinates of top-left corner.
        */
        Point<T> TopLeft() const;

        /*!
            Gets coordinates of top-right corner.

            \return a point with coordinates of top-right corner.
        */
        Point<T> TopRight() const;

        /*!
            Gets coordinates of bottom-left corner.

            \return a point with coordinates of bottom-left corner.
        */
        Point<T> BottomLeft() const;

        /*!
            Gets coordinates of bottom-right corner.

            \return a point with coordinates of bottom-right corner.
        */
        Point<T> BottomRight() const;

        /*!
            Gets rectangle width.

            \return a rectangle width.
        */
        T Width() const;

        /*!
            Gets rectangle height.

            \return a rectangle height.
        */
        T Height() const;

        /*!
            Gets rectangle area.

            \return a rectangle area.
        */
        T Area() const;

        /*!
            Returns true if rectangle area is equal to zero.

            \return a boolean value.
        */
        bool Empty() const;

        /*!
            Gets size (width and height) of the rectangle.

            \return a point with rectangle size.
        */
        Point<T> Size() const;

        /*!
            Gets coordinates of rectangle center.

            \return a point with coordinates of rectangle center.
        */
        Point<T> Center() const;

        /*!
            Checks on the point with specified coordinates to belonging to the rectangle.

            \param [in] x - x-coordinate of checked point.
            \param [in] y - y-coordinate of checked point.
            \return a result of checking.
        */
        template <typename TX, typename TY> bool Contains(TX x, TY y) const;

        /*!
            Checks on the point to belonging to the rectangle.

            \param [in] p - a checked point.
            \return a result of checking.
        */
        template <typename TP> bool Contains(const Point<TP> & p) const;

        /*!
            Checks on the rectangle with specified coordinates to belonging to the rectangle.

            \param [in] l - a left side of checked rectangle.
            \param [in] t - a top side of checked rectangle.
            \param [in] r - a right side of checked rectangle.
            \param [in] b - a bottom side of checked rectangle.
            \return a result of checking.
        */
        template <typename TL, typename TT, typename TR, typename TB> bool Contains(TL l, TT t, TR r, TB b) const;

        /*!
            Checks on the rectangle to belonging to the rectangle.

            \param [in] r - a checked rectangle.
            \return a result of checking.
        */
        template <typename TR> bool Contains(const Rectangle <TR> & r) const;

        /*!
            Shifts a rectangle on the specific value.

            \param [in] shift - a point with shift value.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & Shift(const Point<TP> & shift);

        /*!
            Shifts a rectangle on the specific value.

            \param [in] shiftX - x-coordinate of the shift.
            \param [in] shiftY - y-coordinate of the shift.
            \return a reference to itself.
        */
        template <typename TX, typename TY> Rectangle<T> & Shift(TX shiftX, TY shiftY);

        /*!
            Gets a rectangle with shifted coordinates.

            \param [in] shift - a point with shift value.
            \return a shifted rectangle.
        */
        template <typename TP> Rectangle<T> Shifted(const Point<TP> & shift) const;

        /*!
            Gets a rectangle with shifted coordinates.

            \param [in] shiftX - x-coordinate of the shift.
            \param [in] shiftY - y-coordinate of the shift.
            \return a shifted rectangle.
        */
        template <typename TX, typename TY> Rectangle<T> Shifted(TX shiftX, TY shiftY) const;

        /*!
            Adds border to rectangle.

            \note The value of border can be negative.

            \param [in] border - a width of added border.
            \return a reference to itself.
        */
        template <typename TB> Rectangle<T> & AddBorder(TB border);

        /*!
            Gets an intersection of the two rectangles (current and specified).

            \param [in] r - specified rectangle.
            \return a rectangle with result of intersection.
        */
        template <typename TR> Rectangle<T> Intersection(const Rectangle<TR> & r) const;

        /*!
            Sets to the rectangle results of the intersection of the rectangle and specified point.

            \param [in] p - specified point.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & operator &= (const Point<TP> & p);

        /*!
            Sets to the rectangle results of the intersection of the rectangle and specified rectangle.

            \param [in] r - specified rectangle.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & operator &= (const Rectangle<TR> & r);

        /*!
            Sets to the rectangle results of the union of the rectangle and specified point.

            \param [in] p - specified point.
            \return a reference to itself.
        */
        template <typename TP> Rectangle<T> & operator |= (const Point<TP> & p);

        /*!
            Sets to the rectangle results of the union of the rectangle and specified rectangle.

            \param [in] r - specified rectangle.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & operator |= (const Rectangle<TR> & r);

        /*!
            Adds to the rectangle's coordinates corresponding coordinates of specified rectangle.

            \param [in] r - specified rectangle.
            \return a reference to itself.
        */
        template <typename TR> Rectangle<T> & operator += (const Rectangle<TR> & r);

        /*!
            Checks on overlapping of current rectangle and specified rectangle.

            \param [in] r - specified rectangle.
            \return a result of checking.
        */
        bool Overlaps(const Rectangle<T> & r) const;
    };

    /*! @ingroup cpp_rectangle_functions

        \fn template <typename T> bool operator == (const Rectangle<T> & r1, const Rectangle<T> & r2);

        \short Compares two rectangles on equality.

        \param [in] r1 - a first rectangle.
        \param [in] r2 - a second rectangle.
        \return a result of comparison.
    */
    template <typename T> bool operator == (const Rectangle<T> & r1, const Rectangle<T> & r2);

    /*! @ingroup cpp_rectangle_functions

        \fn template <typename T> bool operator != (const Rectangle<T> & r1, const Rectangle<T> & r2);

        \short Compares two rectangles on inequality.

        \param [in] r1 - a first rectangle.
        \param [in] r2 - a second rectangle.
        \return a result of comparison.
    */
    template <typename T> bool operator != (const Rectangle<T> & r1, const Rectangle<T> & r2);

    /*! @ingroup cpp_rectangle_functions

        \fn template<class T1, class T2> Rectangle<T1> operator / (const Rectangle<T1> & rect, const T2 & value);

        \short Divides the rectangle on the scalar value.

        \param [in] rect - a rectangle.
        \param [in] value - a scalar value.
        \return a result of division.
    */
    template<class T1, class T2> Rectangle<T1> operator / (const Rectangle<T1> & rect, const T2 & value);

    /*! @ingroup cpp_rectangle_functions

        \fn template<class T1, class T2> Rectangle<T1> operator * (const Rectangle<T1> & rect, const T2 & value);

        \short Multiplies the rectangle on the scalar value.

        \param [in] rect - a rectangle.
        \param [in] value - a scalar value.
        \return a result of multiplication.
    */
    template<class T1, class T2> Rectangle<T1> operator * (const Rectangle<T1> & rect, const T2 & value);

    /*! @ingroup cpp_rectangle_functions

        \fn template<class T1, class T2> Rectangle<T1> operator * (const T2 & value, const Rectangle<T1> & rect);

        \short Multiplies the scalar value on the rectangle.

        \param [in] value - a scalar value.
        \param [in] rect - a rectangle.
        \return a result of multiplication.
    */
    template<class T1, class T2> Rectangle<T1> operator * (const T2 & value, const Rectangle<T1> & rect);

    /*! @ingroup cpp_rectangle_functions

        \fn template <typename T> Rectangle<T> operator + (const Rectangle<T> & r1, const Rectangle<T> & r2);

        \short Sums the corresponding rectangle's coordinates of two rectangles..

        \param [in] r1 - a first rectangle.
        \param [in] r2 - a second rectangle.
        \return a rectangle with result coordinates.
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
