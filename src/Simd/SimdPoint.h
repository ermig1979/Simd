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
#ifndef __SimdPoint_h__
#define __SimdPoint_h__

#include "Simd/SimdDefs.h"
#include "Simd/SimdLib.h"

namespace Simd
{
	template <typename T> 
	struct Point
	{
        typedef T Type;

		T x, y;

		Point();
		template <typename TX, typename TY> Point(TX tx, TY ty);
        template <class TP, template<class> class TPoint> Point(const TPoint<TP> & p);
		~Point();

        template <class TP, template<class> class TPoint> operator TPoint<TP>() const;

		template <typename TP> Point & operator = (const Point<TP> & p);
		template <typename TP> Point & operator += (const Point<TP> & p);
		template <typename TP> Point & operator -= (const Point<TP> & p);

		template <typename TA> Point & operator *= (const TA & a);
		Point & operator /= (double a);

		Point operator << (ptrdiff_t shift) const;
		Point operator >> (ptrdiff_t shift) const;
	};

	template <typename T> bool operator == (const Point<T> & p1, const Point<T> & p2);
	template <typename T> bool operator != (const Point<T> & p1, const Point<T> & p2);
	template <typename T> Point<T> operator + (const Point<T> & p1, const Point<T> & p2);
	template <typename T> Point<T> operator - (const Point<T> & p1, const Point<T> & p2);
    template <typename T> Point<T> operator * (const Point<T> & p1, const Point<T> & p2);
    template <typename T> Point<T> operator / (const Point<T> & p1, const Point<T> & p2);
	template <typename T> Point<T> operator - (const Point<T> & p);
	template <typename T> Point<double> operator / (const Point<T> & p, double a);
	template <typename TP, typename TA> Point<TP> operator * (const Point<TP> & p, const TA & a);
	template <typename TP, typename TA> Point<TP> operator * (const TA & a, const Point<TP> & p);

    template <typename T> T SquaredDistance(const Point<T> & p1, const Point<T> & p2);
    template <typename T> T DotProduct(const Point<T> & p1, const Point<T> & p2);
    template <typename T> T CrossProduct(const Point<T> & p1, const Point<T> & p2);

	//-------------------------------------------------------------------------

	// struct Point<T> implementation:

	template <typename T> 
	SIMD_INLINE Point<T>::Point()
		: x(0)
		, y(0)
	{
	}

	template <typename T> template <typename TX, typename TY> 
	SIMD_INLINE Point<T>::Point(TX tx, TY ty)  
		: x((T)tx)
		, y((T)ty) 
	{
	}

    template <typename T> template <class TP, template<class> class TPoint> 
	SIMD_INLINE Point<T>::Point(const TPoint<TP> & p)
        : x((T)p.x)
        , y((T)p.y) 
    {
    }

	template <typename T> 
	SIMD_INLINE Point<T>::~Point()
	{
	}

    template <typename T> template <class TP, template<class> class TPoint> 
    SIMD_INLINE Point<T>::operator TPoint<TP>() const
    {
        return TPoint<TP>((TP)x, (TP)y);
    }

	template <typename T> template <typename TP> 
	SIMD_INLINE Point<T> & Point<T>::operator = (const Point<TP> & p) 
	{
		 x = (T)p.x; 
		 y = (T)p.y; 
		 return *this; 
	}

	template <typename T> template <typename TP> 
	SIMD_INLINE Point<T> & Point<T>::operator += (const Point<TP> & p) 
	{
		x += (T)p.x; 
		y += (T)p.y; 
		return *this; 
	}

	template <typename T> template <typename TP> 
	SIMD_INLINE Point<T> & Point<T>::operator -= (const Point<TP> & p) 
	{
		x -= (T)p.x; 
		y -= (T)p.y; 
		return *this; 
	}

	template <typename T> template <typename TA> 
	SIMD_INLINE Point<T> & Point<T>::operator *= (const TA & a) 
	{
		x = (T)(x*a); 
		y = (T)(y*a); 
		return *this; 
	}

	template <typename T> 
	SIMD_INLINE Point<T> & Point<T>::operator /= (double a) 
	{
		x = (T)(x/a); 
		y = (T)(y/a); 
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

	template <typename T> 
	SIMD_INLINE Point<double> operator / (const Point<T> & p, double a)
	{
		return Point<double>(p.x/a, p.y/a);
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
#endif//__SimdPoint_h__
