/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"

namespace Simd
{
    namespace Base
    {
        namespace
        {
            struct Buffer
            {
                const int size;
                float * cos, *sin;
                int * index;
                float * value;

                Buffer(size_t width, size_t quantization)
                    : size((int)quantization / 2)
                {
                    _p = Allocate(width*(sizeof(int) + sizeof(float)) + sizeof(float) * 2 * size);
                    index = (int*)_p;
                    value = (float*)index + width;
                    cos = value + width;
                    sin = cos + size;
                    for (int i = 0; i < size; ++i)
                    {
                        cos[i] = (float)::cos(i*M_PI / size);
                        sin[i] = (float)::sin(i*M_PI / size);
                    }
                }

                ~Buffer()
                {
                    Free(_p);
                }

            private:
                void *_p;
            };
        }

        void AddRowToHistograms(int * indexes, float * values, size_t row, size_t width, size_t height, size_t cellX, size_t cellY, size_t quantization, float * histograms)
        {
            int blockX = int(width / cellX);
            int blockY = int(height / cellY);
            int blockStride = int(quantization*blockX);

            float yp = ((float)row + 0.5f) / (float)cellY - 0.5f;
            int iyp = (int)floor(yp);
            float vy0 = yp - iyp;
            float vy1 = 1.0f - vy0;

            size_t noseEnd = cellX / 2;
            size_t bodyEnd = width - cellX / 2;

            if (iyp < 0)
            {
                float * h = histograms + (iyp + 1)*blockStride;
                for (size_t col = 1; col < width - 1; ++col)
                {
                    float value = values[col];
                    int index = indexes[col];

                    float xp = ((float)col + 0.5f) / (float)cellX - 0.5f;
                    int ixp = (int)floor(xp);
                    float vx0 = xp - ixp;
                    float vx1 = 1.0f - vx0;

                    if (ixp >= 0)
                        h[ixp*quantization + index] += vx1 * vy0*value;
                    if (ixp + 1 < blockX)
                        h[(ixp + 1)*quantization + index] += vx0 * vy0*value;
                }
            }
            else if (iyp + 1 == blockY)
            {
                float * h = histograms + iyp * blockStride;
                for (size_t col = 1; col < width - 1; ++col)
                {
                    float value = values[col];
                    int index = indexes[col];

                    float xp = ((float)col + 0.5f) / (float)cellX - 0.5f;
                    int ixp = (int)floor(xp);
                    float vx0 = xp - ixp;
                    float vx1 = 1.0f - vx0;

                    if (ixp >= 0)
                        h[ixp*quantization + index] += vx1 * vy1*value;
                    if (ixp + 1 < blockX)
                        h[(ixp + 1)*quantization + index] += vx0 * vy1*value;
                }
            }
            else
            {
                float * h0 = histograms + iyp * blockStride;
                float * h1 = histograms + (iyp + 1)*blockStride;
                size_t col = 1;
                for (; col < noseEnd; ++col)
                {
                    float value = values[col];
                    int index = indexes[col];

                    float xp = ((float)col + 0.5f) / (float)cellX - 0.5f;
                    int ixp = (int)floor(xp);
                    float vx0 = xp - ixp;

                    h0[(ixp + 1)*quantization + index] += vx0 * vy1*value;
                    h1[(ixp + 1)*quantization + index] += vx0 * vy0*value;
                }

                for (; col < bodyEnd; ++col)
                {
                    float value = values[col];
                    int index = indexes[col];

                    float xp = ((float)col + 0.5f) / (float)cellX - 0.5f;
                    int ixp = (int)floor(xp);
                    float vx0 = xp - ixp;
                    float vx1 = 1.0f - vx0;

                    h0[ixp*quantization + index] += vx1 * vy1*value;
                    h1[ixp*quantization + index] += vx1 * vy0*value;
                    h0[(ixp + 1)*quantization + index] += vx0 * vy1*value;
                    h1[(ixp + 1)*quantization + index] += vx0 * vy0*value;
                }

                for (; col < width - 1; ++col)
                {
                    float value = values[col];
                    int index = indexes[col];

                    float xp = ((float)col + 0.5f) / (float)cellX - 0.5f;
                    int ixp = (int)floor(xp);
                    float vx0 = xp - ixp;
                    float vx1 = 1.0f - vx0;

                    h0[ixp*quantization + index] += vx1 * vy1*value;
                    h1[ixp*quantization + index] += vx1 * vy0*value;
                }
            }
        }

        void HogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height,
            size_t cellX, size_t cellY, size_t quantization, float * histograms)
        {
            assert(width%cellX == 0 && height%cellY == 0 && quantization % 2 == 0);

            Buffer buffer(width, quantization);

            memset(histograms, 0, quantization*(width / cellX)*(height / cellY) * sizeof(float));

            for (size_t row = 1; row < height - 1; ++row)
            {
                const uint8_t * src1 = src + stride * row;
                const uint8_t * src0 = src1 - stride;
                const uint8_t * src2 = src1 + stride;

#if 1
                for (size_t col = 1; col < width - 1; ++col)
                {
                    float dy = (float)(src2[col] - src0[col]);
                    float dx = (float)(src1[col + 1] - src1[col - 1]);
                    float value = (float)::sqrt(dx*dx + dy * dy);

                    float bestDot = 0;
                    int index = 0;
                    for (int direction = 0; direction < buffer.size; direction++)
                    {
                        float dot = buffer.cos[direction] * dx + buffer.sin[direction] * dy;
                        if (dot > bestDot)
                        {
                            bestDot = dot;
                            index = direction;
                        }
                        else if (-dot > bestDot)
                        {
                            bestDot = -dot;
                            index = direction + buffer.size;
                        }
                    }

                    buffer.value[col] = value;
                    buffer.index[col] = index;
                }
#else
                size_t size = (buffer.size + 1) / 2;
                for (size_t col = 1; col < width - 1; ++col)
                {
                    float dy = (float)(src2[col] - src0[col]);
                    float dx = (float)(src1[col + 1] - src1[col - 1]);
                    float value = (float)::sqrt(dx*dx + dy * dy);
                    float ady = Simd::Abs(dy);
                    float adx = Simd::Abs(dx);

                    float bestDot = 0;
                    int index = 0;
                    for (int direction = 0; direction < size; direction++)
                    {
                        float dot = buffer.cos[direction] * adx + buffer.sin[direction] * ady;
                        if (dot > bestDot)
                        {
                            bestDot = dot;
                            index = direction;
                        }
                    }
                    if (dx < 0)
                        index = buffer.size - index;
                    if (dy < 0 && index != 0)
                        index = buffer.size * 2 - index - (dx == 0);

                    buffer.value[col] = value;
                    buffer.index[col] = index;
                }
#endif

                AddRowToHistograms(buffer.index, buffer.value, row, width, height, cellX, cellY, quantization, histograms);
            }
        }

        class HogFeatureExtractor
        {
            static const size_t C = 8;
            static const size_t Q = 9;
            static const size_t Q2 = 18;

            size_t _sx, _sy, _hs;

            float _cos[5];
            float _sin[5];
            float _k[C];

            Array32i _index;
            Array32f _value;
            Array32f _histogram;
            Array32f _norm;

            void Init(size_t w, size_t h)
            {
                _sx = w / C;
                _hs = _sx + 2;
                _sy = h / C;
                for (int i = 0; i < 5; ++i)
                {
                    _cos[i] = (float)::cos(i*M_PI / Q);
                    _sin[i] = (float)::sin(i*M_PI / Q);
                }
                for (int i = 0; i < C; ++i)
                    _k[i] = float((1 + i * 2) / 16.0f);
                _index.Resize(w);
                _value.Resize(w);
                _histogram.Resize((_sx + 2)*(_sy + 2)*Q2);
                _norm.Resize((_sx + 2)*(_sy + 2));
            }

            void AddRowToHistogram(size_t row, size_t width, size_t height)
            {
                size_t iyp = (row - 4) / C;
                float vy0 = _k[(row + 4) & 7];
                float vy1 = 1.0f - vy0;
                float * h0 = _histogram.data + ((iyp + 1)*_hs + 0)*Q2;
                float * h1 = _histogram.data + ((iyp + 2)*_hs + 0)*Q2;
                for (size_t col = 1, n = C, i = 5; col < width - 1; i = 0, n = Simd::Min<size_t>(C, width - col - 1))
                {
                    for (; i < n; ++i, ++col)
                    {
                        float value = _value[col];
                        int index = _index[col];
                        float vx0 = _k[i];
                        float vx1 = 1.0f - vx0;
                        h0[index] += vx1 * vy1*value;
                        h1[index] += vx1 * vy0*value;
                        h0[Q2 + index] += vx0 * vy1*value;
                        h1[Q2 + index] += vx0 * vy0*value;
                    }
                    h0 += Q2;
                    h1 += Q2;
                }
            }

            void EstimateHistogram(const uint8_t * src, size_t stride, size_t width, size_t height)
            {
                _histogram.Clear();
                for (size_t row = 1; row < height - 1; ++row)
                {
                    const uint8_t * src1 = src + stride * row;
                    const uint8_t * src0 = src1 - stride;
                    const uint8_t * src2 = src1 + stride;

                    for (size_t col = 1; col < width - 1; ++col)
                    {
                        float dy = (float)(src2[col] - src0[col]);
                        float dx = (float)(src1[col + 1] - src1[col - 1]);
                        float value = (float)::sqrt(dx*dx + dy * dy);
                        float ady = Simd::Abs(dy);
                        float adx = Simd::Abs(dx);

                        float bestDot = 0;
                        int index = 0;
                        for (int direction = 0; direction < 5; direction++)
                        {
                            float dot = _cos[direction] * adx + _sin[direction] * ady;
                            if (dot > bestDot)
                            {
                                bestDot = dot;
                                index = direction;
                            }
                        }
                        if (dx < 0)
                            index = Q - index;
                        if (dy < 0 && index != 0)
                            index = Q2 - index - (dx == 0);

                        _value[col] = value;
                        _index[col] = index;
                    }

                    AddRowToHistogram(row, width, height);
                }
            }

            void EstimateNorm()
            {
                _norm.Clear();
                for (size_t y = 0; y < _sy; ++y)
                {
                    const float * ph = _histogram.data + ((y + 1)*_hs + 1)*Q2;
                    float * pn = _norm.data + (y + 1)*_hs + 1;
                    for (size_t x = 0; x < _sx; ++x)
                    {
                        const float * h = ph + x * Q2;
                        for (int o = 0; o < Q; ++o)
                            pn[x] += Simd::Square(h[o] + h[o + Q]);
                    }
                }
            }

            void ExtractFeatures(float * features)
            {
                float eps = 0.0001f;
                for (size_t y = 0; y < _sy; y++)
                {
                    for (size_t x = 0; x < _sx; x++)
                    {
                        float * dst = features + (y*_sx + x) * 31;

                        float *psrc, n1, n2, n3, n4;

                        float * p0 = _norm.data + y * _hs + x;
                        float * p1 = p0 + _hs;
                        float * p2 = p1 + _hs;

                        n1 = 1.0f / sqrt(p1[1] + p1[2] + p2[1] + p2[2] + eps);
                        n2 = 1.0f / sqrt(p0[1] + p0[2] + p1[1] + p1[2] + eps);
                        n3 = 1.0f / sqrt(p1[0] + p1[1] + p2[0] + p2[1] + eps);
                        n4 = 1.0f / sqrt(p0[0] + p0[1] + p1[0] + p1[1] + eps);

                        float t1 = 0;
                        float t2 = 0;
                        float t3 = 0;
                        float t4 = 0;

                        psrc = _histogram.data + ((y + 1)*_hs + x + 1)*Q2;
                        for (int o = 0; o < Q2; o++)
                        {
                            float h1 = Simd::Min(*psrc * n1, 0.2f);
                            float h2 = Simd::Min(*psrc * n2, 0.2f);
                            float h3 = Simd::Min(*psrc * n3, 0.2f);
                            float h4 = Simd::Min(*psrc * n4, 0.2f);
                            *dst = 0.5f * (h1 + h2 + h3 + h4);
                            t1 += h1;
                            t2 += h2;
                            t3 += h3;
                            t4 += h4;
                            dst++;
                            psrc++;
                        }

                        psrc = _histogram.data + ((y + 1)*_hs + x + 1)*Q2;
                        for (int o = 0; o < Q; o++)
                        {
                            float sum = *psrc + *(psrc + Q);
                            float h1 = Simd::Min(sum * n1, 0.2f);
                            float h2 = Simd::Min(sum * n2, 0.2f);
                            float h3 = Simd::Min(sum * n3, 0.2f);
                            float h4 = Simd::Min(sum * n4, 0.2f);
                            *dst = 0.5f * (h1 + h2 + h3 + h4);
                            dst++;
                            psrc++;
                        }

                        *dst = 0.2357f * t1;
                        dst++;
                        *dst = 0.2357f * t2;
                        dst++;
                        *dst = 0.2357f * t3;
                        dst++;
                        *dst = 0.2357f * t4;
                    }
                }
            }

        public:
            void Run(const uint8_t * src, size_t stride, size_t width, size_t height, float * features)
            {
                Init(width, height);

                EstimateHistogram(src, stride, width, height);

                EstimateNorm();

                ExtractFeatures(features);
            }
        };

        void HogExtractFeatures(const uint8_t * src, size_t stride, size_t width, size_t height, float * features)
        {
            assert(width % 8 == 0 && height % 8 == 0 && width >= 16 && height >= 16);

            HogFeatureExtractor extractor;
            extractor.Run(src, stride, width, height, features);
        }

        namespace HogSeparableFilter_Detail
        {
            template <int add> void Set(float & dst, float value);

            template <> SIMD_INLINE void Set<0>(float & dst, float value)
            {
                dst = value;
            }

            template <> SIMD_INLINE void Set<1>(float & dst, float value)
            {
                dst += value;
            }
        }

        void HogDeinterleave(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const float * psrc = src + row * srcStride;
                size_t offset = row * dstStride;
                for (size_t col = 0; col < width; ++col)
                {
                    for (size_t i = 0; i < count; ++i)
                        dst[i][offset + col] = *psrc++;
                }
            }
        }

        class HogSeparableFilter
        {
            typedef Array<float> Array32f;

            size_t _w, _h;
            Array32f _buffer;

            void Init(size_t w, size_t h, size_t rs, size_t cs)
            {
                _w = w - rs + 1;
                _h = h - cs + 1;
                _buffer.Resize(_w*h);
            }

            void FilterRows(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                    {
                        const float * s = src + col;
                        float sum = 0;
                        for (size_t i = 0; i < size; ++i)
                            sum += s[i] * filter[i];
                        dst[col] = sum;
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }


            template <int add> void FilterCols(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                    {
                        const float * s = src + col;
                        float sum = 0;
                        for (size_t i = 0; i < size; ++i)
                            sum += s[i*srcStride] * filter[i];
                        HogSeparableFilter_Detail::Set<add>(dst[col], sum);
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }

        public:

            void Run(const float * src, size_t srcStride, size_t width, size_t height,
                const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add)
            {
                Init(width, height, rowSize, colSize);

                FilterRows(src, srcStride, _w, height, rowFilter, rowSize, _buffer.data, _w);

                if (add)
                    FilterCols<1>(_buffer.data, _w, _w, _h, colFilter, colSize, dst, dstStride);
                else
                    FilterCols<0>(_buffer.data, _w, _w, _h, colFilter, colSize, dst, dstStride);
            }
        };

        void HogFilterSeparable(const float * src, size_t srcStride, size_t width, size_t height,
            const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add)
        {
            assert(width >= rowSize - 1 && height >= colSize - 1);

            HogSeparableFilter filter;
            filter.Run(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
        }
    }
}
