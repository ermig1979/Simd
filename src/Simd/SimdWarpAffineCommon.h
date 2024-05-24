/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#ifndef __SimdWarpAffineCommon_h__
#define __SimdWarpAffineCommon_h__

#include "Simd/SimdWarpAffine.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
    namespace Base
    {
        template<int N> SIMD_INLINE uint32_t NearestOffset(int x, int y, const float* m, int w, int h, int s)
        {
            float sx = (float)x, sy = (float)y;
            float dx = sx * m[0] + sy * m[1] + m[2];
            float dy = sx * m[3] + sy * m[4] + m[5];
            int ix = Simd::RestrictRange(Round(dx), 0, w);
            int iy = Simd::RestrictRange(Round(dy), 0, h);
            return iy * s + ix * N;
        }

        //-----------------------------------------------------------------------------------------

        template<int N> SIMD_INLINE void NearestGather(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            int i = 0;
            for (; i < count; i++, dst += N)
                Base::CopyPixel<N>(src + offset[i], dst);
        }

        template<> SIMD_INLINE void NearestGather<3>(const uint8_t* src, uint32_t* offset, int count, uint8_t* dst)
        {
            int i = 0, count1 = count - 1;
            for (; i < count1; i++, dst += 3)
                Base::CopyPixel<4>(src + offset[i], dst);
            if (i < count)
                Base::CopyPixel<3>(src + offset[i], dst);
        }

        //-------------------------------------------------------------------------------------------------

        const int WA_LINEAR_SHIFT = 5;
        const int WA_BILINEAR_SHIFT = 2 * WA_LINEAR_SHIFT;
        const int WA_BILINEAR_ROUND_TERM = 1 << (WA_BILINEAR_SHIFT - 1);
        const int WA_FRACTION_RANGE = 1 << WA_LINEAR_SHIFT;

        SIMD_INLINE void ByteBilinearPrepMain(int x, int y, const float* m, int n, int s, uint32_t* offs, uint8_t* fx, uint16_t* fy)
        {
            float sx = (float)x, sy = (float)y;
            float dx = sx * m[0] + sy * m[1] + m[2];
            float dy = sx * m[3] + sy * m[4] + m[5];
            int ix = (int)floor(dx);
            int iy = (int)floor(dy);
            int fx1 = Round((dx - ix) * WA_FRACTION_RANGE);
            int fy1 = Round((dy - iy) * WA_FRACTION_RANGE);
            *offs = iy * s + ix * n;
            fx[0] = WA_FRACTION_RANGE - fx1;
            fx[1] = fx1;
            fy[0] = WA_FRACTION_RANGE - fy1;
            fy[1] = fy1;
        }

        template<int N> SIMD_INLINE void ByteBilinearInterpMain(const uint8_t* src0, const uint8_t* src1, const uint8_t* fx, const uint16_t* fy, uint8_t * dst)
        {
            int f00 = fy[0] * fx[0];
            int f01 = fy[0] * fx[1];
            int f10 = fy[1] * fx[0];
            int f11 = fy[1] * fx[1];
            for (int c = 0; c < N; c++)
                dst[c] = (src0[c] * f00 + src0[c + N] * f01 + src1[c] * f10 + src1[c + N] * f11 + WA_BILINEAR_ROUND_TERM) >> WA_BILINEAR_SHIFT;
        }

        //-------------------------------------------------------------------------------------------------

        template<int N> SIMD_INLINE void ByteBilinearInterpEdge(int x, int y, const float* m, int w, int h, int s, const uint8_t* src, const uint8_t* brd, uint8_t* dst)
        {
            float sx = (float)x, sy = (float)y;
            float dx = sx * m[0] + sy * m[1] + m[2];
            float dy = sx * m[3] + sy * m[4] + m[5];
            int ix = (int)floor(dx);
            int iy = (int)floor(dy);
            int fx = Round((dx - ix) * WA_FRACTION_RANGE);
            int fy = Round((dy - iy) * WA_FRACTION_RANGE);
            int f00 = (WA_FRACTION_RANGE - fy) * (WA_FRACTION_RANGE - fx);
            int f01 = (WA_FRACTION_RANGE - fy) * fx;
            int f10 = fy * (WA_FRACTION_RANGE - fx);
            int f11 = fy * fx;
            bool x0 = ix < 0, x1 = ix > w;
            bool y0 = iy < 0, y1 = iy > h;
            src += iy * s + ix * N;
            const uint8_t* s00 = y0 || x0 ? brd : src;
            const uint8_t* s01 = y0 || x1 ? brd : src + N;
            const uint8_t* s10 = y1 || x0 ? brd : src + s;
            const uint8_t* s11 = y1 || x1 ? brd : src + s + N;
            for (int c = 0; c < N; c++)
                dst[c] = (s00[c] * f00 + s01[c] * f01 + s10[c] * f10 + s11[c] * f11 + WA_BILINEAR_ROUND_TERM) >> WA_BILINEAR_SHIFT;
        }
        //-------------------------------------------------------------------------------------------------

        template<int N> SIMD_INLINE void ByteBilinearGather(const uint8_t* src0, const uint8_t* src1, uint32_t* offset, int count, uint8_t* dst0, uint8_t* dst1)
        {
            int i = 0;
            for (; i < count; i++, dst0 += 2 * N, dst1 += 2 * N)
            {
                int offs = offset[i];
                Base::CopyPixel<N * 2>(src0 + offs, dst0);
                Base::CopyPixel<N * 2>(src1 + offs, dst1);
            }
        }
    }

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
    }
#endif
}
#endif//__SimdWarpAffineCommon_h__
