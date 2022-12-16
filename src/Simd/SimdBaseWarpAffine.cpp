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
#include "Simd/SimdCopyPixel.h"

#include "Simd/SimdPoint.hpp"

namespace Simd
{
    static SIMD_INLINE void SetInv(const float * mat, float * inv)
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

    WarpAffParam::WarpAffParam(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border, size_t align)
    {
        this->srcW = srcW;
        this->srcH = srcH;
        this->srcS = srcS;
        this->dstW = dstW;
        this->dstH = dstH;
        this->dstS = dstS;
        this->channels = channels;
        memcpy(this->mat, mat, 6 * sizeof(float));
        this->flags = flags;
        memset(this->border, 0, BorderSizeMax);
        if (border && (flags & SimdWarpAffineBorderMask) == SimdWarpAffineBorderConstant)
            memcpy(this->border, border, this->PixelSize());
        this->align = align;
        SetInv(this->mat, this->inv);
    }

    //---------------------------------------------------------------------------------------------

    namespace Base
    {
        template<int N> void NearestRun(const WarpAffParam& p, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1;
            for (int y = 0; y < (int)p.dstH; ++y)
            {
                int nose = beg[y], tail = end[y];
                if (N == 3)
                {
                    if (fill)
                    {
                        int x = 0, nose1 = nose - 1;
                        for (; x < nose1; ++x)
                            Base::CopyPixel<4>(p.border, dst + x * 3);
                        for (; x < nose; ++x)
                            Base::CopyPixel<3>(p.border, dst + x * 3);
                    }
                    {
                        int x = nose, tail1 = tail - 1;
                        for (; x < tail1; ++x)
                            Base::CopyPixel<4>(src + NearestOffset<3>(x, y, p.inv, w, h, s), dst + x * 3);
                        for (; x < tail; ++x)
                            Base::CopyPixel<3>(src + NearestOffset<3>(x, y, p.inv, w, h, s), dst + x * 3);
                    }
                    if (fill)
                    {
                        int x = tail, width1 = width - 1;
                        for (; x < width1; ++x)
                            Base::CopyPixel<4>(p.border, dst + x * 3);
                        for (; x < width; ++x)
                            Base::CopyPixel<3>(p.border, dst + x * 3);
                    }
                }
                else
                {
                    if (fill)
                    {
                        for (int x = 0; x < nose; ++x)
                            CopyPixel<N>(p.border, dst + x * N);
                    }
                    {
                        for (int x = nose; x < tail; ++x)
                            CopyPixel<N>(src + NearestOffset<N>(x, y, p.inv, w, h, s), dst + x * N);
                    }
                    if (fill)
                    {
                        for (int x = tail; x < width; ++x)
                            CopyPixel<N>(p.border, dst + x * N);
                    }
                }
                dst += p.dstS;
            }
        }

        //---------------------------------------------------------------------------------------------

        WarpAffineNearest::WarpAffineNearest(const WarpAffParam& param)
            : WarpAffine(param)
        {
            switch (_param.channels)
            {
            case 1: _run = NearestRun<1>; break;
            case 2: _run = NearestRun<2>; break;
            case 3: _run = NearestRun<3>; break;
            case 4: _run = NearestRun<4>; break;
            }
        }

        void WarpAffineNearest::Run(const uint8_t* src, uint8_t* dst)
        {
            if(_first)
                Init();
            _run(_param, _beg.data, _end.data, src, dst, _buf.data);
            _first = false;
        }

        SIMD_INLINE Point Conv(float x, float y, const float* m)
        {
            return Point(x * m[0] + y * m[1] + m[2], x * m[3] + y * m[4] + m[5]);
        }

        void WarpAffineNearest::Init()
        {
            const WarpAffParam& p = _param;
            _beg.Resize(p.dstH);
            _end.Resize(p.dstH);
            _buf.Resize(p.dstW);
            float w = (float)(p.srcW - 1), h = (float)(p.srcH - 1);
            Point points[4];
            points[0] = Conv(0, 0, p.mat);
            points[1] = Conv(w, 0, p.mat);
            points[2] = Conv(w, h, p.mat);
            points[3] = Conv(0, h, p.mat);
            SetRange(points);
        }

        void WarpAffineNearest::SetRange(const Base::Point* points)
        {
            const WarpAffParam& p = _param;
            int w = (int)p.dstW;
            for (size_t y = 0; y < p.dstH; ++y)
            {
                _beg[y] = w;
                _end[y] = 0;
            }
            for (int v = 0; v < 4; ++v)
            {
                const Point& curr = points[v];
                const Point& next = points[(v + 1) & 3];
                float yMin = Simd::Max(Simd::Min(curr.y, next.y), 0.0f);
                float yMax = Simd::Min(Simd::Max(curr.y, next.y), (float)p.dstH);
                int yBeg = Round(yMin);
                int yEnd = Round(yMax);
                if (next.y == curr.y)
                    continue;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                if (abs(a) <= 1.0f)
                {
                    for (int y = yBeg; y < yEnd; ++y)
                    {
                        int x = Round(y * a + b);
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(x, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(x + 1, w));
                    }
                }
                else
                {
                    for (int y = yBeg; y < yEnd; ++y)
                    {
                        float xM = b + Simd::RestrictRange(float(y) - 0.5f, yMin, yMax) * a;
                        float xP = b + Simd::RestrictRange(float(y) + 0.5f, yMin, yMax) * a;
                        int xBeg = Round(Simd::Min(xM, xP));
                        int xEnd = Round(Simd::Max(xM, xP));
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(xBeg, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(xEnd + 1, w));
                    }
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        WarpAffineByteBilinear::WarpAffineByteBilinear(const WarpAffParam& param)
            : WarpAffine(param)
        {
        }

        void WarpAffineByteBilinear::Run(const uint8_t* src, uint8_t* dst)
        {
            if (_first)
                Init();
            _first = false;
        }

        void WarpAffineByteBilinear::Init()
        {
            const WarpAffParam& p = _param;
            _beg.Resize(p.dstH);
            _end.Resize(p.dstH);
            _buf.Resize(p.dstW);
            float w = (float)(p.srcW - 1), h = (float)(p.srcH - 1);
            Point points[4];
            points[0] = Conv(0, 0, p.mat);
            points[1] = Conv(w, 0, p.mat);
            points[2] = Conv(w, h, p.mat);
            points[3] = Conv(0, h, p.mat);
            //SetRange(points);
        }

        //-----------------------------------------------------------------------------------------

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border)
        {
            WarpAffParam param(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border, 1);
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
