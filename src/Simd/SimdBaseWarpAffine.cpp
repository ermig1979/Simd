/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdWarpAffine.h"

#include "Simd/SimdPoint.hpp"

namespace Simd
{
    static inline void SetInv(const float * mat, float * inv)
    {
        double D = mat[0] * mat[4] - mat[1] * mat[3];
        D = D != 0.0 ? 1.0 / D : 0.0;
        double A11 = mat[4] * D;
        double A22 = mat[0] * D;
        double A12 = -mat[1] * D;
        double A21 = -mat[3] * D;
        double b1 = -A11 * mat[2] - A12 * mat[5];
        double b2 = -A21 * mat[2] - A22 * mat[5];
        inv[0] = (float)A11;
        inv[1] = (float)A12;
        inv[2] = (float)b1;
        inv[3] = (float)A21;
        inv[4] = (float)A22;
        inv[5] = (float)b2;
    }

    WarpAffParam::WarpAffParam(size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border, size_t align)
    {
        this->srcW = srcW;
        this->srcH = srcH;
        this->dstW = dstW;
        this->dstH = dstH;
        this->channels = channels;
        memcpy(this->mat, mat, 6 * sizeof(float));
        this->flags = flags;
        if (border && (flags & SimdWarpAffineBorderMask) == SimdWarpAffineBorderConstant)
            memcpy(this->border, border, this->PixelSize());
        else
            memset(this->border, 0, BorderSizeMax);
        this->align = align;
        SetInv(this->mat, this->inv);
    }

    //---------------------------------------------------------------------------------------------

    namespace Base
    {
        typedef Simd::Point<float> Point;

        SIMD_INLINE Point Conv(float x, float y, const float* m)
        {
            return Point(x * m[0] + y * m[1] + m[2], x * m[3] + y * m[4] + m[5]);
        }

        //-----------------------------------------------------------------------------------------

        WarpAffineNearest::WarpAffineNearest(const WarpAffParam& param)
            : WarpAffine(param)
        {

        }

        void WarpAffineNearest::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            if(_empty)
                Init();
            const WarpAffParam& p = _param;
            uint16_t* index = _index.data;
            bool fill = p.NeedFill();
            for (size_t y = 0; y < p.dstH; ++y)
            {
                if (fill)
                {
                    for (int x = 0, xe = _beg[y]; x < xe; ++x)
                    {
                        for (size_t c = 0; c < p.channels; ++c)
                            dst[x * p.channels + c] = p.border[c];
                    }
                }
                for (int x = _beg[y], xe = _end[y]; x < xe; ++x)
                {

                    size_t offset = index[2 * x + 0] * srcStride + index[2 * x + 1] * p.channels;
                    for (size_t c = 0; c < p.channels; ++c)
                        dst[x * p.channels + c] = src[offset + c];
                }
                if (fill)
                {
                    for (int x = _end[y], xe = p.dstW; x < xe; ++x)
                    {
                        for (size_t c = 0; c < p.channels; ++c)
                            dst[x * p.channels + c] = p.border[c];
                    }
                }
                index += p.dstW * 2;
                dst += dstStride;
            }
        }

        void WarpAffineNearest::Init()
        {
            const WarpAffParam& p = _param;
            _beg.Resize(p.dstH);
            _end.Resize(p.dstH);
            SetRange();
            _index.Resize(p.dstH * p.dstW * 2);
            SetIndex();
            _empty = false;
        }

        void WarpAffineNearest::SetRange()
        {
            const WarpAffParam& p = _param;

            Point points[4];
            points[0] = Conv(0.0f, 0.0f, p.mat);
            points[1] = Conv((float)p.srcW - 1, 0.0f, p.mat);
            points[2] = Conv((float)p.srcW - 1, (float)p.srcH - 1, p.mat);
            points[3] = Conv(0.0f, (float)p.srcH - 1, p.mat);

            int W = (int)p.dstW;
            for (size_t y = 0; y < p.dstH; ++y)
            {
                _beg[y] = W;
                _end[y] = 0;
            }
            for (int v = 0; v < 4; ++v)
            {
                const Point& curr = points[v];
                const Point& next = points[(v + 1) & 3];
                float yMin = Simd::Max(Simd::Min(curr.y, next.y), 0.0f);
                float yMax = Simd::Min(Simd::Max(curr.y, next.y), (float)p.dstH);
                if (next.y == curr.y)
                    continue;
                float k = (next.x - curr.x) / (next.y - curr.y);
                for (int y = Round(yMin), ye = Round(yMax); y < ye; ++y)
                {
                    int x = Round(curr.x + (y - curr.y) * k);
                    _beg[y] = Simd::Min(_beg[y], Simd::Max(x, 0));
                    _end[y] = Simd::Max(_end[y], Simd::Min(x + 1, W));
                }
            }
        }

        void WarpAffineNearest::SetIndex()
        {
            const WarpAffParam& p = _param;
            uint16_t* index = _index.data;
            for (int y = 0, ye = (int)p.dstH; y < ye; ++y)
            {
                for (int x = _beg[y], xe = _end[y]; x < xe; ++x)
                {
                    Point i = Conv(x, y, p.inv);
                    index[2 * x + 0] = Simd::RestrictRange(Round(i.y), 0, (int)p.srcH - 1);
                    index[2 * x + 1] = Simd::RestrictRange(Round(i.x), 0, (int)p.srcW - 1);
                }
                index += p.dstW * 2;
            }
        }

        //-----------------------------------------------------------------------------------------

        WarpAffineByteBilinear::WarpAffineByteBilinear(const WarpAffParam& param)
            : WarpAffine(param)
        {
        }

        void WarpAffineByteBilinear::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {

        }

        //-----------------------------------------------------------------------------------------

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border)
        {
            WarpAffParam param(srcW, srcH, dstW, dstH, channels, mat, flags, border, 1);
            if (!param.Valid())
                return NULL;
            if (param.IsNearest())
                return new WarpAffineNearest(param);
            else if (param.IsByteBilinear())
                return new WarpAffineByteBilinear(param);
            else
                return NULL;
        }
    }
}
