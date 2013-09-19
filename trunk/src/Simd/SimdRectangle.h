/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#ifndef __SimdRectangle_h__
#define __SimdRectangle_h__

#include "Simd/SimdPoint.h"
#include "Simd/SimdMath.h"

namespace Simd
{
	template <typename T> 
	struct Rectangle
	{
        typedef T Type;

		T left, top, right, bottom;

		Rectangle();
		template <typename TL, typename TT, typename TR, typename TB> Rectangle(TL l, TT t, TR r, TB b);
		template <typename TLT, typename TRB> Rectangle(const Point<TLT> & lt, const Point<TRB> & rb);
		template <typename TRB> Rectangle(const Point<TRB> & rb);
		template <class TR, template<class> class TRectangle> Rectangle(const TRectangle<TR> & r);
		~Rectangle();

        template <class TR, template<class> class TRectangle> operator TRectangle<TR>() const;

		template <typename TR> Rectangle<T> & operator = (const Rectangle<TR> & r);

        template <typename TL> Rectangle<T> & SetLeft(const TL & l);
        template <typename TT> Rectangle<T> & SetTop(const TT & t);
        template <typename TR> Rectangle<T> & SetRight(const TR & r);
        template <typename TB> Rectangle<T> & SetBottom(const TB & b);

		template <typename TP> Rectangle<T> & SetTopLeft(const Point<TP> & topLeft);
		template <typename TP> Rectangle<T> & SetTopRight(const Point<TP> & topRight);
		template <typename TP> Rectangle<T> & SetBottomLeft(const Point<TP> & bottomLeft);
		template <typename TP> Rectangle<T> & SetBottomRight(const Point<TP> & bottomRight);

        T Left() const;
        T Top() const;
        T Right() const;
        T Bottom() const; 

        Point<T> TopLeft() const;
		Point<T> TopRight() const;
		Point<T> BottomLeft() const;
		Point<T> BottomRight() const; 

		T Width() const;
		T Height() const;
		T Area() const;
		bool Empty() const;

		Point<T> Size() const;
		Point<T> Center() const;

		template <typename TX, typename TY> bool Contains(TX x, TY y) const;
		template <typename TP> bool Contains(const Point<TP> & p) const;
		template <typename TL, typename TT, typename TR, typename TB> bool Contains(TL l, TT t, TR r, TB b) const;
		template <typename TR> bool Contains(const Rectangle <TR> & r) const;

		template <typename TP> Rectangle<T> & Shift(const Point<TP> & shift);
		template <typename TX, typename TY> Rectangle<T> & Shift(TX shiftX, TY shiftY);
        template <typename TB> Rectangle<T> & AddBorder(TB border);

        template <typename TR> Rectangle<T> Intersection(const Rectangle<TR> & r) const;

        template <typename TP> Rectangle<T> & operator &= (const Point<TP> & p);
        template <typename TR> Rectangle<T> & operator &= (const Rectangle<TR> & r);
        template <typename TP> Rectangle<T> & operator |= (const Point<TP> & p);

        bool Overlaps(const Rectangle<T> & r) const;
	};

	template <typename T> bool operator == (const Rectangle<T> & r1, const Rectangle<T> & r2);
	template <typename T> bool operator != (const Rectangle<T> & r1, const Rectangle<T> & r2);

    template<class T1, class T2> Rectangle<T1> operator * (const Rectangle<T1> & rect, const T2 & value);
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
		left += (T)shift.x;
		top += (T)shift.y;
		right += (T)shift.x;
		bottom += (T)shift.y;
		return *this;
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
        T l = Max<T>(left, rect.left);
        T t = Max<T>(top, rect.top);
        T r = Max<T>(l, Min<T>(right, rect.right));
        T b = Max<T>(t, Min<T>(bottom, rect.bottom));
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
#endif//__SimdRectangle_h__