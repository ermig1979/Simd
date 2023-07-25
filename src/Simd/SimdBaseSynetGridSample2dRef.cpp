/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynetGridSample.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        template <typename T, SimdBool align> SIMD_INLINE T Denormalize(T pos, ptrdiff_t dim)
        {
            if (align)
                return T((pos + 1) / 2.0f * (dim - 1));
            else
                return T(((pos + 1) * dim - 1) / 2.0f);
        }

        template <typename T> SIMD_INLINE T Reflect(T x, float min, float max)
        {
            float fx = float(x);
            float range = max - min;
            if (fx < min)
            {
                float dx = min - fx;
                int n = int(dx / range);
                float r = dx - n * range;
                return n % 2 == 0 ? T(min + r) : T(max - r);
            }
            else if (fx > max)
            {
                float dx = fx - max;
                int n = int(dx / range);
                float r = dx - n * range;
                return n % 2 == 0 ? T(max - r) : T(min + r);
            }
            else
                return T(fx);
        }

        SIMD_INLINE void CubicCoeffs(float x, float k[4])
        {
            static const float a = -0.75f;
            x = std::abs(x);
            k[0] = ((a * (x + 1.0f) - 5.0f * a) * (x + 1.0f) + 8.0f * a) * (x + 1.0f) - 4.0f * a;
            k[1] = ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
            k[2] = ((a + 2.0f) * (1.0f - x) - (a + 3.0f)) * (1.0f - x) * (1.0f - x) + 1.0f;
            k[3] = ((a * (2.0f - x) - 5.0f * a) * (2.0f - x) + 8.0f * a) * (2.0f - x) - 4.0f * a;
        }

        template <typename T> SIMD_INLINE T BicubicInterp(T p[4][4], float x, float y)
        {
            float v[4];
            float coeffs[4];
            CubicCoeffs(x, coeffs);
            for (int i = 0; i < 4; i++)
                v[i] = coeffs[0] * p[i][0] + coeffs[1] * p[i][1] + coeffs[2] * p[i][2] + coeffs[3] * p[i][3];
            CubicCoeffs(y, coeffs);
            return T(coeffs[0] * v[0] + coeffs[1] * v[1] + coeffs[2] * v[2] + coeffs[3] * v[3]);
        }

        template <typename T, SimdGridSamplePaddingType padding> SIMD_INLINE T PixelAtGrid(const T* src, ptrdiff_t y, ptrdiff_t x, ptrdiff_t H, ptrdiff_t W, float border[4])
        {
            if (padding == SimdGridSamplePaddingZeros)
                return x >= 0 && x < W&& y >= 0 && y < H ? src[y * W + x] : T(0);
            else if (padding == SimdGridSamplePaddingBorder)
            {
                x = Simd::RestrictRange<ptrdiff_t>(x, 0, W - 1);
                y = Simd::RestrictRange<ptrdiff_t>(y, 0, H - 1);
                return src[y * W + x];
            }
            else if (padding == SimdGridSamplePaddingReflect)
            {
                x = ptrdiff_t(Reflect(T(x), border[0], border[2]));
                y = ptrdiff_t(Reflect(T(y), border[1], border[3]));
                return src[y * W + x];
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<class T, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align>
        void GridSample2d(const uint8_t* src8, size_t batch, size_t channels, size_t srcH, size_t srcW, const uint8_t* grd8, size_t dstH, size_t dstW, uint8_t* dst8)
        {
            const T* src = (const T*)src8;
            const T* grd = (const T*)grd8;
            T* dst = (T*)dst8;
            float border[4];
            if (align)
            {
                border[0] = 0.0f;
                border[1] = 0.0f;
                border[2] = srcW - 1.0f;
                border[3] = srcH - 1.0f;
            }
            else
            {
                border[0] = -0.5f;
                border[1] = -0.5f;
                border[2] = srcW - 0.5f;
                border[3] = srcH - 0.5f;
            }
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    const T* gr = grd;
                    for (size_t dy = 0; dy < dstH; ++dy)
                    {
                        for (size_t dx = 0; dx < dstW; ++dx)
                        {
                            T x = Denormalize<T, align>(gr[0], srcW);
                            T y = Denormalize<T, align>(gr[1], srcH);
                            if (interp == SimdGridSampleInterpNearest)
                            {
                                x = T(Round(float(x)));
                                y = T(Round(float(y)));
                            }
                            if (x < border[0] || x > border[2] || y < border[1] || y > border[3])
                            {
                                if (padding == SimdGridSamplePaddingBorder)
                                {
                                    x = Simd::RestrictRange<T>(x, 0, T(srcW - 1));
                                    y = Simd::RestrictRange<T>(y, 0, T(srcH - 1));
                                }
                                else if (padding == SimdGridSamplePaddingReflect)
                                {
                                    x = Reflect(x, border[0], border[2]);
                                    y = Reflect(y, border[1], border[3]);
                                }
                            }

                            if (interp == SimdGridSampleInterpNearest)
                            {
                                dst[0] = PixelAtGrid<T, padding>(src, ptrdiff_t(y), ptrdiff_t(x), srcH, srcW, border);
                            }
                            if (interp == SimdGridSampleInterpBilinear)
                            {
                                ptrdiff_t x1 = ptrdiff_t(std::floor(x));
                                ptrdiff_t y1 = ptrdiff_t(std::floor(y));
                                ptrdiff_t x2 = x1 + 1;
                                ptrdiff_t y2 = y1 + 1;

                                T p11 = PixelAtGrid<T, padding>(src, y1, x1, srcH, srcW, border);
                                T p12 = PixelAtGrid<T, padding>(src, y1, x2, srcH, srcW, border);
                                T p21 = PixelAtGrid<T, padding>(src, y2, x1, srcH, srcW, border);
                                T p22 = PixelAtGrid<T, padding>(src, y2, x2, srcH, srcW, border);

                                T dx2 = T(x2) - x;
                                T dx1 = x - T(x1);
                                T dy2 = T(y2) - y;
                                T dy1 = y - T(y1);
                                dst[0] = dy2 * (dx2 * p11 + dx1 * p12) + dy1 * (dx2 * p21 + dx1 * p22);
                            }
                            if (interp == SimdGridSampleInterpBicubic)
                            {
                                ptrdiff_t x0 = ptrdiff_t(std::floor(x)) - 1;
                                ptrdiff_t y0 = ptrdiff_t(std::floor(y)) - 1;
                                T p[4][4];
                                for (ptrdiff_t h = 0; h < 4; h++)
                                    for (ptrdiff_t w = 0; w < 4; w++)
                                        p[h][w] = PixelAtGrid<T, padding>(src, h + y0, w + x0, srcH, srcW, border);
                                T dx1 = T(x - x0 - 1);
                                T dy1 = T(y - y0 - 1);
                                dst[0] = BicubicInterp(p, float(dx1), float(dy1));
                            }
                            gr += 2; dst += 1;
                        }
                    }
                    src += srcH * srcW;
                }
                grd += dstH * dstW * 2;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<class T, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding> SynetGridSample2dRef::GridSample2dPtr GetGridSample2d(SimdBool align)
        {
            return align ? GridSample2d<T, interp, padding, SimdTrue> : GridSample2d<T, interp, padding, SimdFalse>;
        }

        template<class T, SimdGridSampleInterpType interp> SynetGridSample2dRef::GridSample2dPtr GetGridSample2d(SimdGridSamplePaddingType padding, SimdBool align)
        {
            switch (padding)
            {
            case SimdGridSamplePaddingZeros: return GetGridSample2d<T, interp, SimdGridSamplePaddingZeros>(align);
            case SimdGridSamplePaddingBorder: return GetGridSample2d<T, interp, SimdGridSamplePaddingBorder>(align);
            case SimdGridSamplePaddingReflect: return GetGridSample2d<T, interp, SimdGridSamplePaddingReflect>(align);
            default:
                return NULL;
            }
        }

        template<class T> SynetGridSample2dRef::GridSample2dPtr GetGridSample2d(SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align)
        {
            switch (interp)
            {
            case SimdGridSampleInterpBilinear: return GetGridSample2d<T, SimdGridSampleInterpBilinear>(padding, align);
            case SimdGridSampleInterpNearest: return GetGridSample2d<T, SimdGridSampleInterpNearest>(padding, align);
            case SimdGridSampleInterpBicubic: return GetGridSample2d<T, SimdGridSampleInterpBicubic>(padding, align);
            default:
                return NULL;
            }
        }

        SynetGridSample2dRef::GridSample2dPtr GetGridSample2d(SimdTensorDataType type, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align)
        {
            switch (type)
            {
            case SimdTensorData32f: return GetGridSample2d<float>(interp, padding, align);
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetGridSample2dRef::SynetGridSample2dRef(const GridSample2dParam& param)
            : Simd::SynetGridSample2d(param)
        {
            _gridSample2d = GetGridSample2d(_param.type, _param.interp, _param.padding, _param.align);
        }

        void SynetGridSample2dRef::Forward(const uint8_t* src, const uint8_t* grd, uint8_t* dst)
        {
            _gridSample2d(src, _param.batch, _param.channels, _param.srcH, _param.srcW, grd, _param.dstH, _param.dstW, dst);
        }
    }
#endif
}
