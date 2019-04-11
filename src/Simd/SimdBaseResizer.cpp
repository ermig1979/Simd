/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdResizer.h"

namespace Simd
{
    namespace Base
    {
        ResizerByteBilinear::ResizerByteBilinear(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels)
            : Resizer(SimdResizeChannelByte, SimdResizeMethodBilinear)
            , _sx(srcX), _sy(srcY), _dx(dstX), _dy(dstY), _cn(channels)
        {
            _ay.Resize(_dy);
            _iy.Resize(_dy);
            EstimateIndexAlpha(_sy, _dy, _iy.data, _ay.data, 1);

            _rs = _dx * _cn;
            _ax.Resize(_rs);
            _ix.Resize(_rs);
            EstimateIndexAlpha(_sx, _dx, _ix.data, _ax.data, _cn);
        }

        void ResizerByteBilinear::EstimateIndexAlpha(size_t srcSize, size_t dstSize, int32_t * indices, int32_t * alphas, size_t channels)
        {
            float scale = (float)srcSize / dstSize;

            for (size_t i = 0; i < dstSize; ++i)
            {
                float alpha = (float)((i + 0.5f)*scale - 0.5f);
                ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                alpha -= index;

                if (index < 0)
                {
                    index = 0;
                    alpha = 0;
                }

                if (index >(ptrdiff_t)srcSize - 2)
                {
                    index = srcSize - 2;
                    alpha = 1;
                }

                for (size_t c = 0; c < channels; c++)
                {
                    size_t offset = i * channels + c;
                    indices[offset] = (int32_t)(channels*index + c);
                    alphas[offset] = (int32_t)(alpha * FRACTION_RANGE + 0.5f);
                }
            }
        }

        void ResizerByteBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride) const
        {
            Array32i bx[2];
            bx[0].Resize(_rs);
            bx[1].Resize(_rs);
            int32_t * pbx[2] = { bx[0].data, bx[1].data };
            int32_t prev = -2;
            for (size_t dy = 0; dy < _dy; dy++, dst += dstStride)
            {
                int32_t fy = _ay[dy];
                int32_t sy = _iy[dy];
                int32_t k = 0;

                if (sy == prev)
                    k = 2;
                else if (sy == prev + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                prev = sy;

                for (; k < 2; k++)
                {
                    int32_t * pb = pbx[k];
                    const uint8_t * ps = src + (sy + k)*srcStride;
                    for (size_t dx = 0; dx < _rs; dx++)
                    {
                        int32_t sx = _ix[dx];
                        int32_t fx = _ax[dx];
                        int32_t t = ps[sx];
                        pb[dx] = (t << LINEAR_SHIFT) + (ps[sx + _cn] - t)*fx;
                    }
                }

                if (fy == 0)
                    for (size_t dx = 0; dx < _rs; dx++)
                        dst[dx] = ((pbx[0][dx] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
                else if (fy == FRACTION_RANGE)
                    for (size_t dx = 0; dx < _rs; dx++)
                        dst[dx] = ((pbx[1][dx] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
                else
                {
                    for (size_t dx = 0; dx < _rs; dx++)
                    {
                        int32_t t = pbx[0][dx];
                        dst[dx] = ((t << LINEAR_SHIFT) + (pbx[1][dx] - t)*fy + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
                    }
                }
            }
        }

        //---------------------------------------------------------------------

        const int32_t AREA_COEFF = 1 << 7;
        const int32_t AREA_SHIFT = 14;

        ResizerByteArea::ResizerByteArea(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels)
            : Resizer(SimdResizeChannelByte, SimdResizeMethodArea)
            , _sx(srcX), _sy(srcY), _dx(dstX), _dy(dstY), _cn(channels)
        {
            _ay.Resize(_dy + 1);
            _iy.Resize(_dy + 1);
            EstimateParams(_sy, _dy, _ay.data, _iy.data);

            _rs = _dx * _cn;
            _ax.Resize(_dx + 1);
            _ix.Resize(_dx + 1);
            EstimateParams(_sx, _dx, _ax.data, _ix.data);
        }

        void ResizerByteArea::EstimateParams(size_t srcSize, size_t dstSize, int32_t * alpha, int32_t * index)
        {
            float k = (float)srcSize / dstSize;

            for (size_t ds = 0; ds <= dstSize; ++ds)
            {
                float a = (float)ds*k;
                size_t i = (size_t)::floor(a);
                a -= i;
                if (ds == dstSize)
                {
                    i--;
                    a = 1.0f;
                }
                alpha[ds] = int32_t(AREA_COEFF * (1.0f - a) / k);
                index[ds] = int32_t(i);
            }
        }

        template<class T, size_t N> SIMD_INLINE void ResizerByteAreaSet(const T * src, int32_t value, int32_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] = src[c] * value;
        }

        template<class T, size_t N> SIMD_INLINE void ResizerByteAreaAdd(const T * src, int32_t value, int32_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] += src[c] * value;
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaPixelRowSum(const uint8_t * src, size_t size, int32_t nose, int32_t body, int32_t tail, int32_t * dst)
        {
            ResizerByteAreaSet<uint8_t, N>(src, nose, dst);
            for (size_t i = 0; i < size; ++i)
            {
                src += N;
                ResizerByteAreaAdd<uint8_t, N>(src, body, dst);
            }
            ResizerByteAreaAdd<uint8_t, N>(src, tail, dst);
        }

        template<size_t N> SIMD_INLINE void ResizerByteAreaRes(const int32_t * src, int32_t roundTerm, int shift, uint8_t * dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] = uint8_t((src[c] + roundTerm) >> shift);
        }

        template<size_t N> void ResizerByteArea::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride) const
        {
            int32_t ts[N], rs[N];
            int32_t ayb = _ay.data[0], axb = _ax.data[0];
            int32_t rt = (ayb*axb) / 2;
            for (size_t dy = 0; dy < _dy; dy++, dst += dstStride)
            {
                size_t by = _iy.data[dy], ey = _iy.data[dy + 1];
                int32_t ayn = _ay.data[dy], ayt = - _ay.data[dy + 1];
                for (size_t dx = 0; dx < _dx; dx++)
                {
                    size_t bx = _ix.data[dx], sx = _ix.data[dx + 1] - bx;
                    int32_t axn = _ax.data[dx], axt = - _ax.data[dx + 1];
                    const uint8_t * s = src + by * srcStride + bx * N;
                    ResizerByteAreaPixelRowSum<N>(s, sx, axn, axb, axt, rs);
                    ResizerByteAreaSet<int32_t, N>(rs, ayn, ts);
                    for (size_t sy = by; sy < ey; sy++)
                    {
                        s += srcStride;
                        ResizerByteAreaPixelRowSum<N>(s, sx, axn, axb, axt, rs);
                        ResizerByteAreaAdd<int32_t, N>(rs, ayb, ts);
                    }
                    ResizerByteAreaPixelRowSum<N>(s, sx, axn, axb, axt, rs);
                    ResizerByteAreaAdd<int32_t, N>(rs, ayt, ts);
                    ResizerByteAreaRes<N>(ts, rt, AREA_SHIFT, dst + dx * N);
                }
            }
        }

        void ResizerByteArea::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride) const
        {
            switch (_cn)
            {
            case 1: Run<1>(src, srcStride, dst, dstStride); return;
            case 2: Run<2>(src, srcStride, dst, dstStride); return;
            case 3: Run<3>(src, srcStride, dst, dstStride); return;
            case 4: Run<4>(src, srcStride, dst, dstStride); return;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        ResizerFloatBilinear::ResizerFloatBilinear(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, size_t align, bool caffeInterp)
            : Resizer(SimdResizeChannelFloat, SimdResizeMethodBilinear)
            , _sx(srcX), _sy(srcY), _dx(dstX), _dy(dstY), _cn(channels)
        {
            _ay.Resize(_dy, false, align);
            _iy.Resize(_dy, false, align);
            EstimateIndexAlpha(_sy, _dy, _iy.data, _ay.data, 1, caffeInterp);

            _rs = _dx * _cn;
            _ax.Resize(_rs, false, align);
            _ix.Resize(_rs, false, align);
            EstimateIndexAlpha(_sx, _dx, _ix.data, _ax.data, _cn, caffeInterp);
        }

        void ResizerFloatBilinear::EstimateIndexAlpha(size_t srcSize, size_t dstSize, int32_t * indices, float * alphas, size_t channels, bool caffeInterp)
        {
            if (caffeInterp)
            {
                float scale = dstSize > 1 ? float(srcSize - 1) / float(dstSize - 1) : 0.0f;
                for (size_t i = 0; i < dstSize; ++i)
                {
                    float alpha = float(i)*scale;
                    ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                    alpha -= index;
                    if (index > (ptrdiff_t)srcSize - 2)
                    {
                        index = srcSize - 2;
                        alpha = 1;
                    }
                    for (size_t c = 0; c < channels; c++)
                    {
                        size_t offset = i * channels + c;
                        indices[offset] = (int32_t)(channels*index + c);
                        alphas[offset] = alpha;
                    }
                }
            }
            else
            {
                float scale = (float)srcSize / dstSize;
                for (size_t i = 0; i < dstSize; ++i)
                {
                    float alpha = (float)((i + 0.5f)*scale - 0.5f);
                    ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                    alpha -= index;
                    if (index < 0)
                    {
                        index = 0;
                        alpha = 0;
                    }
                    if (index >(ptrdiff_t)srcSize - 2)
                    {
                        index = srcSize - 2;
                        alpha = 1;
                    }
                    for (size_t c = 0; c < channels; c++)
                    {
                        size_t offset = i * channels + c;
                        indices[offset] = (int32_t)(channels*index + c);
                        alphas[offset] = alpha;
                    }
                }
            }
        }

        void ResizerFloatBilinear::Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride) const
        {
            Run((const float*)src, srcStride / sizeof(float), (float*)dst, dstStride / sizeof(float));
        }

        void ResizerFloatBilinear::Run(const float * src, size_t srcStride, float * dst, size_t dstStride) const
        {
            Array32f bx[2];
            bx[0].Resize(_rs);
            bx[1].Resize(_rs);
            float * pbx[2] = { bx[0].data, bx[1].data };
            int32_t prev = -2;
            for (size_t dy = 0; dy < _dy; dy++, dst += dstStride)
            {
                float fy1 = _ay[dy];
                float fy0 = 1.0f - fy1;
                int32_t sy = _iy[dy];
                int32_t k = 0;

                if (sy == prev)
                    k = 2;
                else if (sy == prev + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                prev = sy;

                for (; k < 2; k++)
                {
                    float * pb = pbx[k];
                    const float * ps = src + (sy + k)*srcStride;
                    for (size_t dx = 0; dx < _rs; dx++)
                    {
                        int32_t sx = _ix[dx];
                        float fx = _ax[dx];
                        pb[dx] = ps[sx]*(1.0f - fx) + ps[sx + _cn]*fx;
                    }
                }

                for (size_t dx = 0; dx < _rs; dx++)
                    dst[dx] = pbx[0][dx]*fy0 + pbx[1][dx]*fy1;
            }
        }

        //---------------------------------------------------------------------

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method)
        {
            if (type == SimdResizeChannelByte && method == SimdResizeMethodBilinear)
                return new ResizerByteBilinear(srcX, srcY, dstX, dstY, channels);
            else  if (type == SimdResizeChannelByte && method == SimdResizeMethodArea)
                return new ResizerByteArea(srcX, srcY, dstX, dstY, channels);
            else if (type == SimdResizeChannelFloat && method == SimdResizeMethodBilinear)
                return new ResizerFloatBilinear(srcX, srcY, dstX, dstY, channels, sizeof(void*), false);
            else if (type == SimdResizeChannelFloat && method == SimdResizeMethodCaffeInterp)
                return new ResizerFloatBilinear(srcX, srcY, dstX, dstY, channels, sizeof(void*), true);
            else
                return NULL;
        }
    }
}

