/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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

        /*!
            A rectangle destructor.
        */
		~Rectangle();

        /*!
            Converts itself to rectangle of arbitrary type.

            \return a rectangle of arbitrary type. 
        */
        template <class TR, template<class> class TRectangle> operator TRectangle<TR>() const;

        /*!
            Performs copying from rectangle of arbitrary type.

            \param [in] r - a rectangle of arbitrary type. 
            \return a reference to itself. 
        */
		template <typename TR> Rectangle<T> & operator = (const Rectangle<TR> & r);

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
		: left((T)l)
		, top((T)t)
		, right((T)r)
		, bottom((T)b)
	{
	}

	template <typename T> template <typename TLT, typename TRB> 
	SIMD_INLINE Rectangle<T>::Rectangle(const Point<TLT> & lt, const Point<TRB> & rb)
		: left((T)lt.x)
		, top((T)lt.y)
		, right((T)rb.x)
		, bottom((T)rb.y)
	{
	}

	template <typename T> template <typename TRB> 
	SIMD_INLINE Rectangle<T>::Rectangle(const Point<TRB> & rb)
		: left(0)
		, top(0)
		, right((T)rb.x)
		, bottom((T)rb.y)
	{
	}

	template <typename T> template <class TR, template<class> class TRectangle> 
	SIMD_INLINE Rectangle<T>::Rectangle(const TRectangle<TR> & r)
		: left((T)r.Left())
		, top((T)r.Top())
		, right((T)r.Right())
		, bottom((T)r.Bottom())
	{
	}

	template <typename T> 
	SIMD_INLINE Rectangle<T>::~Rectangle()
	{
	}

    template <typename T> template <class TR, template<class> class TRectangle> 
    SIMD_INLINE Rectangle<T>::operator TRectangle<TR>() const
    {
        return TRectangle<TR>((TR)left, (TR)top, (TR)right, (TR)bottom);
    }

	template <typename T> template <typename TR> 
	SIMD_INLINE Rectangle<T> & Rectangle<T>::operator = (const Rectangle<TR> & r)
	{
		left = (T)r.left;
		top = (T)r.top;
		right = (T)r.right;
		bottom = (T)r.bottom;
		return *this;
	}

    template <typename T> template <typename TL> 
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetLeft(const TL & l)
    {
        left = (T)l;
        return *this;
    }

    template <typename T> template <typename TT> 
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetTop(const TT & t)
    {
        top = (T)t;
        return *this;
    }

    template <typename T> template <typename TR> 
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetRight(const TR & r)
    {
        right = (T)r;
        return *this;
    }

    template <typename T> template <typename TB> 
    SIMD_INLINE Rectangle<T> & Rectangle<T>::SetBottom(const TB & b)
    {
        bottom = (T)b;
        return *this;
    }

	template <typename T> template <typename TP> 
	SIMD_INLINE Rectangle<T> & Rectangle<T>::SetTopLeft(const Point<TP> & topLeft)
	{
		left = (T)topLeft.x;
		top = (T)topLeft.y;
		return *this;
	}

	template <typename T> template <typename TP> 
	SIMD_INLINE Rectangle<T> & Rectangle<T>::SetTopRight(const Point<TP> & topRight)
	{
		right = (T)topRight.x;
		top = (T)topRight.y;
		return *this;
	}

	template <typename T> template <typename TP> 
	SIMD_INLINE Rectangle<T> & Rectangle<T>::SetBottomLeft(const Point<TP> & bottomLeft)
	{
		left = (T)bottomLeft.x;
		bottom = (T)bottomLeft.y;
		return *this;
	}

	template <typename T> template <typename TP> 
	SIMD_INLINE Rectangle<T> & Rectangle<T>::SetBottomRight(const Point<TP> & bottomRight)
	{
		right = (T)bottomRight.x;
		bottom = (T)bottomRight.y;
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
		return Point<T>((left + right)/2, (top + bottom)/2);
	}

	template <typename T> template <typename TX, typename TY> 
	SIMD_INLINE bool Rectangle<T>::Contains(TX x, TY y) const
	{
		return (T)x >= left && (T)x < right && (T)y >= top && (T)y < bottom;
	}

	template <typename T> template <typename TP> 
	SIMD_INLINE bool Rectangle<T>::Contains(const Point<TP> & p) const
	{
		return Contains(p.x, p.y);
	}

	template <typename T> template <typename TL, typename TT, typename TR, typename TB> 
	SIMD_INLINE bool Rectangle<T>::Contains(TL l, TT t, TR r, TB b) const
	{
		return (T)l >= left && (T)r <= right && (T)t >= top && (T)b <= bottom;
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
		left += (T)shiftX;
		top += (T)shiftY;
		right += (T)shiftX;
		bottom += (T)shiftY;
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
        return Rectangle<T>(left + (T)shiftX, top + (T)shiftY, right + (T)shiftX, bottom + (T)shiftY);
    }

    template <typename T> template <typename TB> 
    SIMD_INLINE Rectangle<T> & Rectangle<T>::AddBorder(TB border)
    {
        left -= (T)border;
        top -= (T)border;
        right += (T)border;
        bottom += (T)border;
        return *this;
    }

    template <typename T> template <typename TR> 
    SIMD_INLINE Rectangle<T> Rectangle<T>::Intersection(const Rectangle<TR> & rect) const
    {
        T l = std::max<T>(left, rect.left);
        T t = std::max<T>(top, rect.top);
        T r = std::max<T>(l, std::min<T>(right, rect.right));
        T b = std::max<T>(t, std::min<T>(bottom, rect.bottom));
        return Rectangle(l, t, r, b);
    }

    template <typename T> template <typename TP> 
    SIMD_INLINE Rectangle<T> & Rectangle<T>::operator &= (const Point<TP> & p)
    {
        if (Contains(p))
        {
            left = (T)p.x;
            top = (T)p.y;
            right = (T)p.x + (T)1;
            bottom = (T)p.y + (T)1;
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
        if(r.Empty())
            return this->operator=(r);

        if (left < (T)r.left)
            left = (T)r.left;
        if (top < (T)r.top)
            top = (T)r.top;
        if (right > (T)r.right)
            right = (T)r.right;
        if (bottom > (T)r.bottom)
            bottom = (T)r.bottom;
        return *this;
    }

    template <typename T> template <typename TP> 
    SIMD_INLINE Rectangle<T> & Rectangle<T>::operator |= (const Point<TP> & p)
    {
        if (Empty())
        {
            left = (T)p.x;
            top = (T)p.y;
            right = (T)p.x + (T)1;
            bottom = (T)p.y + (T)1;
        }
        else
        {
            if (left > (T)p.x)
                left = (T)p.x;
            if (top > (T)p.y)
                top = (T)p.y;
            if (right <= (T)p.x)
                right = (T)p.x + (T)1;
            if (bottom <= (T)p.y)
                bottom = (T)p.y + (T)1;
        }
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
        return Rectangle<T1>((T1)(rect.left/value), (T1)(rect.top/value), (T1)(rect.right/value), (T1)(rect.bottom/value));
    }

    template<class T1, class T2>
    SIMD_INLINE Rectangle<T1> operator * (const Rectangle<T1> & rect, const T2 & value)
    {
        return Rectangle<T1>((T1)(rect.left*value), (T1)(rect.top*value), (T1)(rect.right*value), (T1)(rect.bottom*value));
    }

    template<class T1, class T2>
    SIMD_INLINE Rectangle<T1> operator * (const T2 & value, const Rectangle<T1> & rect)
    {
        return Rectangle<T1>((T1)(rect.left*value), (T1)(rect.top*value), (T1)(rect.right*value), (T1)(rect.bottom*value));
    }
}
#endif//__SimdRectangle_hpp__