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
#include "Simd/SimdWarpAffine.h"
#include "Simd/SimdWarpAffineCommon.h"
#include "Simd/SimdCopy.h"

#include "Simd/SimdPoint.hpp"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svint32_t Round(svbool_t mask, svfloat32_t value)
        {
            svbool_t positive = svcmpge_n_f32(mask, value, 0.0f);
            svfloat32_t round = svsel_f32(positive, svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svcvt_s32_f32_x(mask, svadd_f32_x(mask, value, round));
        }

        template<int N> SIMD_INLINE void FillBorder(uint8_t* dst, int count, const svuint8_t& bv, const uint8_t* bs)
        {
            size_t i = 0, size = count * N, A = svcntb(), sizeA = AlignLo(size, A);
            const svbool_t body = svptrue_b8();
            for (; i < sizeA; i += A)
                svst1_u8(body, dst + i, bv);
            for (; i < size; i += N)
                Base::CopyPixel<N>(bs, dst + i);
        }

        template<> SIMD_INLINE void FillBorder<3>(uint8_t* dst, int count, const svuint8_t& bv, const uint8_t* bs)
        {
            int i = 0, size = count * 3, size3 = size - 3;
            for (; i < size3; i += 3)
                Base::CopyPixel<4>(bs, dst + i);
            for (; i < size; i += 3)
                Base::CopyPixel<3>(bs, dst + i);
        }

        template<int N> SIMD_INLINE svuint8_t InitBorder(const uint8_t* border)
        {
            switch (N)
            {
            case 1: return svdup_n_u8(*border);
            case 2: return svreinterpret_u8_u16(svdup_n_u16(*(uint16_t*)border));
            case 3: return svdup_n_u8(0);
            case 4: return svreinterpret_u8_u32(svdup_n_u32(*(uint32_t*)border));
            }
            return svdup_n_u8(0);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE svuint32_t NearestOffset(svbool_t mask, svfloat32_t x, svfloat32_t y, const svfloat32_t& m0, const svfloat32_t& m1, const svfloat32_t& m2,
            const svfloat32_t& m3, const svfloat32_t& m4, const svfloat32_t& m5, const svint32_t& w, const svint32_t& h, int n, int s)
        {
            svfloat32_t dx = svmla_f32_x(mask, svmla_f32_x(mask, m2, x, m0), y, m1);
            svfloat32_t dy = svmla_f32_x(mask, svmla_f32_x(mask, m5, x, m3), y, m4);
            svint32_t ix = svmin_s32_x(mask, svmax_n_s32_x(mask, Round(mask, dx), 0), w);
            svint32_t iy = svmin_s32_x(mask, svmax_n_s32_x(mask, Round(mask, dy), 0), h);
            return svreinterpret_u32_s32(svmla_n_s32_x(mask, svmul_n_s32_x(mask, iy, s), ix, n));
        }

        //-----------------------------------------------------------------------------------------

        template<int N> void NearestRun(const WarpAffParam& p, int yBeg, int yEnd, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1;
            size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t m0 = svdup_n_f32(p.inv[0]);
            svfloat32_t m1 = svdup_n_f32(p.inv[1]);
            svfloat32_t m2 = svdup_n_f32(p.inv[2]);
            svfloat32_t m3 = svdup_n_f32(p.inv[3]);
            svfloat32_t m4 = svdup_n_f32(p.inv[4]);
            svfloat32_t m5 = svdup_n_f32(p.inv[5]);
            svint32_t _w = svdup_n_s32(w);
            svint32_t _h = svdup_n_s32(h);
            svuint8_t _border = InitBorder<N>(p.border);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int nose = beg[y], tail = end[y];
                {
                    int x = nose;
                    svfloat32_t _y = svdup_n_f32((float)y);
                    for (; x < tail; x += (int)F)
                    {
                        svfloat32_t _x = svcvt_f32_u32_x(body, svindex_u32((uint32_t)x, 1));
                        svst1_u32(body, buf + x, NearestOffset(body, _x, _y, m0, m1, m2, m3, m4, m5, _w, _h, N, s));
                    }
                }
                if (fill)
                    FillBorder<N>(dst, nose, _border, p.border);
                Base::NearestGather<N>(src, buf + nose, tail - nose, dst + N * nose);
                if (fill)
                    FillBorder<N>(dst + tail * N, width - tail, _border, p.border);
                dst += p.dstS;
            }
        }

        //-------------------------------------------------------------------------------------------------

        WarpAffineNearest::WarpAffineNearest(const WarpAffParam& param)
#ifdef SIMD_NEON_ENABLE
            : Neon::WarpAffineNearest(param)
#else
            : Base::WarpAffineNearest(param)
#endif
        {
            switch (_param.channels)
            {
            case 1: _run = NearestRun<1>; break;
            case 2: _run = NearestRun<2>; break;
            case 3: _run = NearestRun<3>; break;
            case 4: _run = NearestRun<4>; break;
            }
        }

        void WarpAffineNearest::SetRange(const Base::Point* points)
        {
            const WarpAffParam& p = _param;
            int w = (int)p.dstW, h = (int)p.dstH, F = (int)svcntw(), hF = (int)AlignLo(h, F);
            const svbool_t body = svptrue_b32();
            svint32_t _w = svdup_n_s32(w), _1 = svdup_n_s32(1), _0 = svdup_n_s32(0);
            int y = 0;
            for (; y < hF; y += F)
            {
                svst1_s32(body, _beg.data + y, _w);
                svst1_s32(body, _end.data + y, _0);
            }
            for (; y < h; ++y)
            {
                _beg[y] = w;
                _end[y] = 0;
            }
            for (int v = 0; v < 4; ++v)
            {
                const Base::Point& curr = points[v];
                const Base::Point& next = points[(v + 1) & 3];
                float yMin = Simd::Max(Simd::Min(curr.y, next.y), 0.0f);
                float yMax = Simd::Min(Simd::Max(curr.y, next.y), (float)p.dstH);
                int yBeg = Simd::Round(yMin);
                int yEnd = Simd::Round(yMax);
                int yEndF = (int)AlignLo(yEnd - yBeg, F) + yBeg;
                if (next.y == curr.y)
                    continue;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                svfloat32_t _a = svdup_n_f32(a);
                svfloat32_t _b = svdup_n_f32(b);
                if (abs(a) <= 1.0f)
                {
                    int y = yBeg;
                    for (; y < yEndF; y += F)
                    {
                        svfloat32_t _y = svcvt_f32_u32_x(body, svindex_u32((uint32_t)y, 1));
                        svint32_t _x = Round(body, svmla_f32_x(body, _b, _y, _a));
                        svint32_t xBeg = svld1_s32(body, _beg.data + y);
                        svint32_t xEnd = svld1_s32(body, _end.data + y);
                        xBeg = svmin_s32_x(body, xBeg, svmax_s32_x(body, _x, _0));
                        xEnd = svmax_s32_x(body, xEnd, svmin_s32_x(body, svadd_s32_x(body, _x, _1), _w));
                        svst1_s32(body, _beg.data + y, xBeg);
                        svst1_s32(body, _end.data + y, xEnd);
                    }
                    for (; y < yEnd; ++y)
                    {
                        int x = Simd::Round(y * a + b);
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(x, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(x + 1, w));
                    }
                }
                else
                {
                    int y = yBeg;
                    svfloat32_t _05 = svdup_n_f32(0.5f);
                    svfloat32_t _yMin = svdup_n_f32(yMin);
                    svfloat32_t _yMax = svdup_n_f32(yMax);
                    for (; y < yEndF; y += F)
                    {
                        svfloat32_t _y = svcvt_f32_u32_x(body, svindex_u32((uint32_t)y, 1));
                        svfloat32_t yM = svmin_f32_x(body, svmax_f32_x(body, svsub_f32_x(body, _y, _05), _yMin), _yMax);
                        svfloat32_t yP = svmin_f32_x(body, svmax_f32_x(body, svadd_f32_x(body, _y, _05), _yMin), _yMax);
                        svfloat32_t xM = svmla_f32_x(body, _b, yM, _a);
                        svfloat32_t xP = svmla_f32_x(body, _b, yP, _a);
                        svint32_t xBeg = svld1_s32(body, _beg.data + y);
                        svint32_t xEnd = svld1_s32(body, _end.data + y);
                        xBeg = svmin_s32_x(body, xBeg, svmax_s32_x(body, Round(body, svmin_f32_x(body, xM, xP)), _0));
                        xEnd = svmax_s32_x(body, xEnd, svmin_s32_x(body, svadd_s32_x(body, Round(body, svmax_f32_x(body, xM, xP)), _1), _w));
                        svst1_s32(body, _beg.data + y, xBeg);
                        svst1_s32(body, _end.data + y, xEnd);
                    }
                    for (; y < yEnd; ++y)
                    {
                        float xM = b + Simd::RestrictRange(float(y) - 0.5f, yMin, yMax) * a;
                        float xP = b + Simd::RestrictRange(float(y) + 0.5f, yMin, yMax) * a;
                        int xBeg = Simd::Round(Simd::Min(xM, xP));
                        int xEnd = Simd::Round(Simd::Max(xM, xP));
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(xBeg, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(xEnd + 1, w));
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border)
        {
            WarpAffParam param(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border, svcntb());
            if (!param.Valid())
                return NULL;
            if (param.IsNearest())
                return new WarpAffineNearest(param);
#ifdef SIMD_NEON_ENABLE
            else if (param.IsByteBilinear())
                return Neon::WarpAffineInit(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border);
#endif
            else
                return Base::WarpAffineInit(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border);
        }
    }
#endif
}
