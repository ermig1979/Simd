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
#include "Simd/SimdConst.h"
#include "Simd/SimdStore.h"

#include "Simd/SimdPoint.hpp"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        template<int N> SIMD_INLINE void FillBorder(uint8_t* dst, int count, const uint8x16_t& bv, const uint8_t* bs)
        {
            int i = 0, size = count * N, size16 = (int)AlignLo(size, A);
            for (; i < size16; i += A)
                vst1q_u8(dst + i, bv);
            for (; i < size; i += N)
                Base::CopyPixel<N>(bs, dst + i);
        }

        template<> SIMD_INLINE void FillBorder<3>(uint8_t* dst, int count, const uint8x16_t& bv, const uint8_t* bs)
        {
            int i = 0, size = count * 3, size3 = size - 3;
            for (; i < size3; i += 3)
                Base::CopyPixel<4>(bs, dst + i);
            for (; i < size; i += 3)
                Base::CopyPixel<3>(bs, dst + i);
        }

        template<int N> SIMD_INLINE uint8x16_t InitBorder(const uint8_t* border)
        {
            switch (N)
            {
            case 1: return vdupq_n_u8(*border);
            case 2: return (uint8x16_t)vdupq_n_u16(*(uint16_t*)border);
            case 3: return K8_00;
            case 4: return (uint8x16_t)vdupq_n_u32(*(uint32_t*)border);
            }
            return K8_00;
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE uint32x4_t NearestOffset(float32x4_t x, float32x4_t y, const float32x4_t* m, int32x4_t w, const int32x4_t& h, const int32x4_t& n, const int32x4_t& s)
        {
            float32x4_t dx = vaddq_f32(vaddq_f32(vmulq_f32(x, m[0]), vmulq_f32(y, m[1])), m[2]);
            float32x4_t dy = vaddq_f32(vaddq_f32(vmulq_f32(x, m[3]), vmulq_f32(y, m[4])), m[5]);
            int32x4_t zero = vdupq_n_s32(0);
            int32x4_t ix = vminq_s32(vmaxq_s32(Round(dx), zero), w);
            int32x4_t iy = vminq_s32(vmaxq_s32(Round(dy), zero), h);
            return vreinterpretq_u32_s32(vaddq_s32(vmulq_s32(ix, n), vmulq_s32(iy, s)));
        }

        //-----------------------------------------------------------------------------------------

        template<int N> void NearestRun(const WarpAffParam& p, int yBeg, int yEnd, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* buf)
        {
            bool fill = p.NeedFill();
            int width = (int)p.dstW, s = (int)p.srcS, w = (int)p.srcW - 1, h = (int)p.srcH - 1;
            const float32x4_t _4 = vdupq_n_f32(4.0f);
            const int32x4_t _0123 = SIMD_VEC_SETR_EPI32(0, 1, 2, 3);
            float32x4_t _m[6];
            for (int i = 0; i < 6; ++i)
                _m[i] = vdupq_n_f32(p.inv[i]);
            int32x4_t _w = vdupq_n_s32(w);
            int32x4_t _h = vdupq_n_s32(h);
            int32x4_t _n = vdupq_n_s32(N);
            int32x4_t _s = vdupq_n_s32(s);
            uint8x16_t _border = InitBorder<N>(p.border);
            dst += yBeg * p.dstS;
            for (int y = yBeg; y < yEnd; ++y)
            {
                int nose = beg[y], tail = end[y];
                {
                    int x = nose;
                    float32x4_t _y = vcvtq_f32_s32(vdupq_n_s32(y));
                    float32x4_t _x = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(x), _0123));
                    for (; x < tail; x += 4)
                    {
                        vst1q_u32(buf + x, NearestOffset(_x, _y, _m, _w, _h, _n, _s));
                        _x = vaddq_f32(_x, _4);
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
            : Base::WarpAffineNearest(param)
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
            int w = (int)p.dstW, h = (int)p.dstH, h4 = (int)AlignLo(h, 4);
            const int32x4_t _0123 = SIMD_VEC_SETR_EPI32(0, 1, 2, 3);
            int32x4_t _w = vdupq_n_s32(w), _1 = vdupq_n_s32(1), _0 = vdupq_n_s32(0);
            int y = 0;
            for (; y < h4; y += 4)
            {
                vst1q_s32(_beg.data + y, _w);
                vst1q_s32(_end.data + y, _0);
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
                int yBeg = Round(yMin);
                int yEnd = Round(yMax);
                int yEnd4 = (int)AlignLo(yEnd - yBeg, 4) + yBeg;
                if (next.y == curr.y)
                    continue;
                float a = (next.x - curr.x) / (next.y - curr.y);
                float b = curr.x - curr.y * a;
                float32x4_t _a = vdupq_n_f32(a);
                float32x4_t _b = vdupq_n_f32(b);
                if (abs(a) <= 1.0f)
                {
                    int y = yBeg;
                    for (; y < yEnd4; y += 4)
                    {
                        float32x4_t _y = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(y), _0123));
                        int32x4_t _x = Round(vaddq_f32(vmulq_f32(_y, _a), _b));
                        int32x4_t xBeg = vld1q_s32(_beg.data + y);
                        int32x4_t xEnd = vld1q_s32(_end.data + y);
                        xBeg = vminq_s32(xBeg, vmaxq_s32(_x, _0));
                        xEnd = vmaxq_s32(xEnd, vminq_s32(vaddq_s32(_x, _1), _w));
                        vst1q_s32(_beg.data + y, xBeg);
                        vst1q_s32(_end.data + y, xEnd);
                    }
                    for (; y < yEnd; ++y)
                    {
                        int x = Round(y * a + b);
                        _beg[y] = Simd::Min(_beg[y], Simd::Max(x, 0));
                        _end[y] = Simd::Max(_end[y], Simd::Min(x + 1, w));
                    }
                }
                else
                {
                    int y = yBeg;
                    float32x4_t _05 = vdupq_n_f32(0.5f);
                    float32x4_t _yMin = vdupq_n_f32(yMin);
                    float32x4_t _yMax = vdupq_n_f32(yMax);
                    for (; y < yEnd4; y += 4)
                    {
                        float32x4_t _y = vcvtq_f32_s32(vaddq_s32(vdupq_n_s32(y), _0123));
                        float32x4_t yM = vminq_f32(vmaxq_f32(vsubq_f32(_y, _05), _yMin), _yMax);
                        float32x4_t yP = vminq_f32(vmaxq_f32(vaddq_f32(_y, _05), _yMin), _yMax);
                        float32x4_t xM = vaddq_f32(vmulq_f32(yM, _a), _b);
                        float32x4_t xP = vaddq_f32(vmulq_f32(yP, _a), _b);
                        int32x4_t xBeg = vld1q_s32(_beg.data + y);
                        int32x4_t xEnd = vld1q_s32(_end.data + y);
                        xBeg = vminq_s32(xBeg, vmaxq_s32(Round(vminq_f32(xM, xP)), _0));
                        xEnd = vmaxq_s32(xEnd, vminq_s32(vaddq_s32(Round(vmaxq_f32(xM, xP)), _1), _w));
                        vst1q_s32(_beg.data + y, xBeg);
                        vst1q_s32(_end.data + y, xEnd);
                    }
                    for (; y < yEnd; ++y)
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

        //-------------------------------------------------------------------------------------------------

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border)
        {
            WarpAffParam param(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border, A);
            if (!param.Valid())
                return NULL;
            if (param.IsNearest())
                return new WarpAffineNearest(param);
            else
                return Base::WarpAffineInit(srcW, srcH, srcS, dstW, dstH, dstS, channels, mat, flags, border);
        }
    }
#endif
}
