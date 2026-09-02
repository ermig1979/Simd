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
#ifndef __SimdShape_h__
#define __SimdShape_h__

#include "Simd/SimdMath.h"

#include <vector>

namespace Simd
{
    typedef std::vector<size_t> Shape;

    //--------------------------------------------------------------------------------------------------

    SIMD_INLINE Shape Shp()
    {
        return Shape();
    }

    SIMD_INLINE Shape Shp(size_t axis0)
    {
        return Shape({ axis0 });
    }

    SIMD_INLINE Shape Shp(size_t axis0, size_t axis1)
    {
        return Shape({ axis0, axis1 });
    }

    SIMD_INLINE Shape Shp(size_t axis0, size_t axis1, size_t axis2)
    {
        return Shape({ axis0, axis1, axis2 });
    }

    SIMD_INLINE Shape Shp(size_t axis0, size_t axis1, size_t axis2, size_t axis3)
    {
        return Shape({ axis0, axis1, axis2, axis3 });
    }

    SIMD_INLINE Shape Shp(size_t axis0, size_t axis1, size_t axis2, size_t axis3, size_t axis4)
    {
        return Shape({ axis0, axis1, axis2, axis3, axis4 });
    }

    template<class T> SIMD_INLINE Shape Shp(const std::vector<T>& vec)
    {
        Shape shape(vec.size());
        for (size_t i = 0; i < vec.size(); ++i)
            shape[i] = (size_t)vec[i];
        return shape;
    }

    template<class T> SIMD_INLINE Shape Shp(const T* data, size_t size)
    {
        Shape shape(size);
        for (size_t i = 0; i < size; ++i)
            shape[i] = (size_t)data[i];
        return shape;
    }

    //--------------------------------------------------------------------------------------------------

    SIMD_INLINE bool IsCompatible(const Shape& a, const Shape& b)
    {
        for (size_t i = 0, n = Max(a.size(), b.size()), a0 = n - a.size(), b0 = n - b.size(); i < n; ++i)
        {
            size_t ai = i < a0 ? 1 : a[i - a0];
            size_t bi = i < b0 ? 1 : b[i - b0];
            if (!(ai == bi || ai == 1 || bi == 1))
                return false;
        }
        return true;
    }

    SIMD_INLINE Shape OutputShape(const Shape& a, const Shape& b)
    {
        Shape d(Max(a.size(), b.size()), 1);
        for (size_t i = 0, n = d.size(), a0 = n - a.size(), b0 = n - b.size(); i < n; ++i)
        {
            size_t ai = i < a0 ? 1 : a[i - a0];
            size_t bi = i < b0 ? 1 : b[i - b0];
            d[i] = Max(ai, bi);
        }
        return d;
    }

    SIMD_INLINE Shape SourceSteps(const Shape& src, const Shape& dst)
    {
        Shape steps(dst.size(), 0);
        size_t step = 1;
        for (ptrdiff_t i = dst.size() - 1, s0 = dst.size() - src.size(); i >= 0; --i)
        {
            size_t si = i < s0 ? 1 : src[i - s0];
            steps[i] = si == 1 ? 0 : step;
            step *= si;
        }
        return steps;
    }

    SIMD_INLINE Shape FullSrcShape(const Shape& src, const Shape& dst)
    {
        Shape full(dst.size(), 1);
        for (size_t is = 0, id = dst.size() - src.size(); is < src.size(); is++, id++)
            full[id] = src[is];
        return full;
    }

    SIMD_INLINE int Relation(size_t a, size_t b, size_t d)
    {
        if (a < d)
            return -1;
        if (b < d)
            return 1;
        return 0;
    }

    SIMD_INLINE void CompactShapes(Shape& a, Shape& b, Shape& d)
    {
        Shape _a = FullSrcShape(a, d), _b = FullSrcShape(b, d), _d = d;
        a = Shp(_a[0]), b = Shp(_b[0]), d = Shp(_d[0]);
        for (size_t i = 1; i < _d.size(); ++i)
        {
            if (Relation(a.back(), b.back(), d.back()) == Relation(_a[i], _b[i], _d[i]) || d.back() == 1 || _d[i] == 1)
            {
                a.back() *= _a[i];
                b.back() *= _b[i];
                d.back() *= _d[i];
            }
            else
            {
                a.push_back(_a[i]);
                b.push_back(_b[i]);
                d.push_back(_d[i]);
            }
        }
    }
}

#endif
