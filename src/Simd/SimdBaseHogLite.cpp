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
#include "Simd/SimdArray.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
    namespace Base
    {
        template <size_t cell> class HogLiteFeatureExtractor
        {
            static const size_t FQ = 8;
            static const size_t HQ = FQ/2;

            typedef Array<int> Ints;
            typedef Array<float> Floats;

            size_t _hx, _fx;
            Ints _hi[2];
            Floats _hf[2], _nf[4];
            int _k0[cell], _k1[cell];

            SIMD_INLINE void Init(size_t width)
            {
                _hx = width / cell;
                _fx = _hx - 2;
                for (size_t i = 0; i < cell; ++i)
                {
                    _k0[i] = int(cell - i - 1) * 2 + 1;
                    _k1[i] = int(i) * 2 + 1;
                }
                for (size_t i = 0; i < 2; ++i)
                {
                    _hi[i].Resize(_hx*FQ, true);
                    _hf[i].Resize(_hx*FQ);
                }
                for (size_t i = 0; i < 4; ++i)
                    _nf[i].Resize(_hx);
            }

            SIMD_INLINE void UpdateIntegerHistogram(const uint8_t * src, size_t stride, size_t width, size_t rowI, size_t rowF)
            {
                int * h0 = _hi[(rowI + 0) & 1].data;
                int * h1 = _hi[(rowI + 1) & 1].data;
                int ky0 = _k0[rowF];
                int ky1 = _k1[rowF];
                for (size_t col = 0; col < width;)
                {
                    for (size_t colF = 0; colF < cell; ++colF, ++col)
                    {
                        int dy = src[col + stride] - src[col - stride];
                        int dx = src[col + 1] - src[col - 1];
                        int adx = Abs(dx);
                        int ady = Abs(dy);
                        int value = RestrictRange(Max(adx, ady) + (Min(adx, ady) + 1) / 2);

                        size_t index = (adx > ady ? 0 : 1);
                        index = (dx > 0 ? index : (HQ - 1) - index);
                        index = (dy > 0 ? index : (FQ - 1) - index);

                        h0[00 + index] += value*_k0[colF] * ky0;
                        h1[00 + index] += value*_k0[colF] * ky1;
                        h0[FQ + index] += value*_k1[colF] * ky0;
                        h1[FQ + index] += value*_k1[colF] * ky1;
                    }
                    h0 += FQ;
                    h1 += FQ;
                }
            }

            SIMD_INLINE void UpdateFloatHistogram(size_t rowI)
            {
                const float k = 1.0f / Simd::Square(cell * 2);
                Ints & hi = _hi[rowI & 1];
                Floats & hf = _hf[rowI & 1];
                Floats & nf = _nf[rowI & 3];

                for (size_t i = 0; i < hi.size; ++i)
                    hf.data[i] = float(hi.data[i])*k;
                hi.Clear();

                const float * h = hf.data;
                for (size_t x = 0; x < _hx; ++x, h += FQ)
                {
                    float sum = 0;
                    for (int i = 0; i < HQ; ++i)
                        sum += Simd::Square(h[i] + h[i + HQ]);
                    nf.data[x] = sum;
                }
            }

            SIMD_INLINE void SetFeatures(size_t rowI, float * dst)
            {
                const float eps = 0.0001f;
                float * hf = _hf[(rowI - 1) & 1].data + FQ;
                float * p0 = _nf[(rowI - 2) & 3].data;
                float * p1 = _nf[(rowI - 1) & 3].data;
                float * p2 = _nf[(rowI - 0) & 3].data;
                for (size_t x = 0; x < _fx; ++x, ++p0, ++p1, ++p2)
                {
                    float n1 = 1.0f / sqrt(p1[1] + p1[2] + p2[1] + p2[2] + eps);
                    float n2 = 1.0f / sqrt(p0[1] + p0[2] + p1[1] + p1[2] + eps);
                    float n3 = 1.0f / sqrt(p1[0] + p1[1] + p2[0] + p2[1] + eps);
                    float n4 = 1.0f / sqrt(p0[0] + p0[1] + p1[0] + p1[1] + eps);

                    float t1 = 0;
                    float t2 = 0;
                    float t3 = 0;
                    float t4 = 0;

                    float * src = hf + FQ*x;
                    for (size_t o = 0; o < FQ; o++)
                    {
                        float h1 = Simd::Min(*src * n1, 0.2f);
                        float h2 = Simd::Min(*src * n2, 0.2f);
                        float h3 = Simd::Min(*src * n3, 0.2f);
                        float h4 = Simd::Min(*src * n4, 0.2f);
                        *dst++ = 0.5f * (h1 + h2 + h3 + h4);
                        t1 += h1;
                        t2 += h2;
                        t3 += h3;
                        t4 += h4;
                        src++;
                    }

                    src = hf + FQ*x;
                    for (size_t o = 0; o < HQ; o++)
                    {
                        float sum = *src + *(src + HQ);
                        float h1 = Simd::Min(sum * n1, 0.2f);
                        float h2 = Simd::Min(sum * n2, 0.2f);
                        float h3 = Simd::Min(sum * n3, 0.2f);
                        float h4 = Simd::Min(sum * n4, 0.2f);
                        *dst++ = 0.5f * (h1 + h2 + h3 + h4);
                        src++;
                    }

                    *dst++ = 0.2357f * t1;
                    *dst++ = 0.2357f * t2;
                    *dst++ = 0.2357f * t3;
                    *dst++ = 0.2357f * t4;
                }
            }

        public:

            void Run(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * features, size_t featuresStride)
            {
                assert(cell == 8 || cell == 4);
                assert(width >= cell * 3 && height >= cell * 3);

                Init(width);

                src += (srcStride + 1)*cell / 2;
                height = (height/cell - 1)*cell;
                width = (width/cell - 1)*cell;

                for (size_t row = 0; row < height; ++row)
                {
                    size_t rowI = row / cell;
                    size_t rowF = row & (cell - 1);
                    UpdateIntegerHistogram(src, srcStride, width, rowI, rowF);
                    if (rowF == cell - 1)
                    {
                        UpdateFloatHistogram(rowI);
                        if (rowI >= 2)
                        {
                            SetFeatures(rowI, features);
                            features += featuresStride;
                        }
                    }
                    src += srcStride;
                }
                size_t rowI = height/cell;
                UpdateFloatHistogram(rowI);
                SetFeatures(rowI, features);
            }
        };

        void HogLiteExtractFeatures(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t cell, float * features, size_t featuresStride)
        {
            if (cell == 4)
            {
                HogLiteFeatureExtractor<4> extractor;
                extractor.Run(src, srcStride, width, height, features, featuresStride);
            }
            else
            {
                HogLiteFeatureExtractor<8> extractor;
                extractor.Run(src, srcStride, width, height, features, featuresStride);
            }
        }

        class HogLiteFeatureFilter
        {
            void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, float * dst, size_t dstStride)
            {
                size_t filterStride = featureSize*filterWidth;
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                    {
                        float sum = 0;
                        const float * pSrc = src + dstRow*srcStride + dstCol*featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                        {
                            for (size_t filterCol = 0; filterCol < filterStride; ++filterCol)
                                sum += pSrc[filterCol] * pFilter[filterCol];
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        dst[dstCol] = sum;
                    }
                    dst += dstStride;
                }
            }

            void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                size_t filterStride = featureSize*filterWidth;
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                    {
                        if (mask[dstCol])
                        {
                            float sum = 0;
                            const float * pSrc = src + dstRow*srcStride + dstCol*featureSize;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                for (size_t filterCol = 0; filterCol < filterStride; ++filterCol)
                                    sum += pSrc[filterCol] * pFilter[filterCol];
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            dst[dstCol] = sum;
                        }
                        else
                            dst[dstCol] = -FLT_MAX;
                    }
                    dst += dstStride;
                    mask += maskStride;
                }
            }
        public:
            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                assert(featureSize == 8 || featureSize == 16);
                assert(srcWidth >= filterWidth && srcHeight >= filterHeight);

                size_t dstWidth = srcWidth - filterWidth + 1;
                size_t dstHeight = srcHeight - filterHeight + 1;
                if (mask)
                    Filter(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                else
                    Filter(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, dst, dstStride);
            }
        };

        void HogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
        {
            HogLiteFeatureFilter featureFilter;
            featureFilter.Run(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
        }

        class HogLiteFeatureResizer
        {
            typedef Array<int> Ints;
            typedef Array<float> Floats;

            Ints _iy, _ix;
            Floats _ky, _kx;

            void InitIndexWeight(size_t srcSize, size_t dstSize, size_t dstStep, Ints & indexes, Floats & weights)
            {
                indexes.Resize(dstSize);
                weights.Resize(dstSize);

                float scale = float(srcSize) / float(dstSize);
                for (size_t i = 0; i < dstSize; ++i)
                {
                    float weight = (float)((i + 0.5f)*scale - 0.5f);
                    int index = (int)::floor(weight);
                    weight -= index;
                    if (index < 0)
                    {
                        index = 0;
                        weight = 0.0f;
                    }
                    if (index > (int)srcSize - 2)
                    {
                        index = (int)srcSize - 2;
                        weight = 1.0f;
                    }
                    indexes[i] = int(index*dstStep);
                    weights[i] = weight;
                }
            }

            void Resize(const float * src, size_t srcStride, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                for (size_t rowDst = 0; rowDst < dstHeight; ++rowDst)
                {
                    float ky1 = _ky[rowDst];
                    float ky0 = 1.0f - ky1;
                    const float * pSrc = src + _iy[rowDst];
                    float * pDst = dst + rowDst*dstStride;
                    for (size_t colDst = 0; colDst < dstWidth; ++colDst, pDst += featureSize)
                    {
                        float kx1 = _kx[colDst];
                        float kx0 = 1.0f - kx1;
                        float k00 = ky0*kx0, k01 = ky0*kx1, k10 = ky1*kx0, k11 = ky1*kx1;
                        const float * pSrc0 = pSrc + _ix[colDst];
                        const float * pSrc1 = pSrc0 + srcStride;
                        for (size_t i = 0; i < featureSize; ++i)
                            pDst[i] = pSrc0[i] * k00 + pSrc0[i + featureSize] * k01 + pSrc1[i] * k10 + pSrc1[i + featureSize] * k11;
                    }
                }
            }

        public:
            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                assert(featureSize == 8 || featureSize == 16);

                if (srcWidth == dstWidth && srcHeight == dstHeight)
                {
                    size_t size = sizeof(float)*srcWidth*featureSize;
                    for (size_t row = 0; row < dstHeight; ++row)
                        memcpy(dst + row*dstStride, src + row*srcStride, size);
                    return;
                }

                InitIndexWeight(srcWidth, dstWidth, featureSize, _ix, _kx);
                InitIndexWeight(srcHeight, dstHeight, srcStride, _iy, _ky);

                Resize(src, srcStride, featureSize, dst, dstStride, dstWidth, dstHeight);
            }
        };

        void HogLiteResizeFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
        {
            HogLiteFeatureResizer featureResizer;
            featureResizer.Run(src, srcStride, srcWidth, srcHeight, featureSize, dst, dstStride, dstWidth, dstHeight);
        }

        void HogLiteCompressFeatures(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const float * s = src;
                float * d = dst;
                for (size_t col = 0; col < width; ++col)
                {
                    const float * p = pca;
                    for (size_t i = 0; i < 8; ++i, p += 16)
                    {
                        float sum = 0;
                        for (size_t j = 0; j < 16; ++j)
                            sum += s[j] * p[j];
                        d[i] = sum;
                    }
                    s += 16;
                    d += 8;
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        class HogLiteSeparableFilter
        {
            typedef Array<float> Array32f;

            size_t _dstWidth, _dstHeight;
            Array32f _buffer;

            void Init(size_t srcWidth, size_t srcHeight, size_t hSize, size_t vSize)
            {
                _dstWidth = srcWidth - hSize + 1;
                _dstHeight = srcHeight - vSize + 1;
                _buffer.Resize(_dstWidth*srcHeight);
            }

            void FilterH(const float * src, size_t srcStride, size_t width, size_t height, size_t step, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                    {
                        const float * s = src + col*step;
                        float sum = 0;
                        for (size_t i = 0; i < size; ++i)
                            sum += s[i] * filter[i];
                        dst[col] = sum;
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template<UpdateType update> void FilterV(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                    {
                        const float * s = src + col;
                        float sum = 0;
                        for (size_t i = 0; i < size; ++i)
                            sum += s[i*srcStride] * filter[i];
                        Update<update>(dst + col, sum);
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }

        public:
            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add)
            {
                assert(featureSize == 8 || featureSize == 16);
                assert(srcWidth >= hSize && srcHeight >= vSize);

                Init(srcWidth, srcHeight, hSize, vSize);

                FilterH(src, srcStride, _dstWidth, srcHeight, featureSize, hFilter, hSize*featureSize, _buffer.data, _dstWidth);

                if(add)
                    FilterV<UpdateAdd>(_buffer.data, _dstWidth, _dstWidth, _dstHeight, vFilter, vSize, dst, dstStride);
                else
                    FilterV<UpdateSet>(_buffer.data, _dstWidth, _dstWidth, _dstHeight, vFilter, vSize, dst, dstStride);
            }
        };

        void HogLiteFilterSeparable(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add)
        {
            HogLiteSeparableFilter filter;
            filter.Run(src, srcStride, srcWidth, srcHeight, featureSize, hFilter, hSize, vFilter, vSize, dst, dstStride, add);
        }

        void HogLiteFindMax7x7(const float * a, size_t aStride, const float * b, size_t bStride, size_t height, float * pValue, size_t * pCol, size_t * pRow)
        {
            *pValue = -FLT_MAX;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < 7; ++col)
                {
                    float value = a[col] + b[col];
                    if (value > *pValue)
                    {
                        *pValue = value;
                        *pCol = col;
                        *pRow = row;
                    }
                }
                a += aStride;
                b += bStride;
            }
        }

        template<size_t size> void HogLiteCreateMask(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, uint32_t * dst, size_t dstStride)
        {
            size_t dstStartEnd = size - scale;
            size_t dstRowSize = (srcWidth*scale + size - scale) * sizeof(uint32_t);
            for (size_t srcRow = 0; srcRow < srcHeight; ++srcRow)
            {
                for (size_t dstRow = 0; dstRow < scale; ++dstRow)
                    memset(dst + (dstStartEnd + dstRow)*dstStride, 0, dstRowSize);

                for (size_t srcCol = 0; srcCol < srcWidth; ++srcCol)
                {
                    if (src[srcCol] > *threshold)
                    {
                        uint32_t * pDst = dst + srcCol * scale;
                        for (size_t dstRow = 0; dstRow < size; ++dstRow)
                        {
                            memset(pDst, -1, size * sizeof(uint32_t));
                            pDst += dstStride;
                        }
                    }
                }
                src += srcStride;
                dst += dstStride*scale;
            }
        }

        void HogLiteCreateMask(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride)
        {
            size_t dstStartEnd = size - scale;
            size_t dstRowSize = (srcWidth*scale + size - scale) * sizeof(uint32_t);
            for (size_t dstRow = 0; dstRow < dstStartEnd; ++dstRow)
                memset(dst + dstRow*dstStride, 0, dstRowSize);

            switch (size)
            {
            case 7: HogLiteCreateMask<7>(src, srcStride, srcWidth, srcHeight, threshold, scale, dst, dstStride); return;
            default: break;
            }

            for (size_t srcRow = 0; srcRow < srcHeight; ++srcRow)
            {
                for (size_t dstRow = 0; dstRow < scale; ++dstRow)
                    memset(dst + (dstStartEnd + dstRow)*dstStride, 0, dstRowSize);

                for (size_t srcCol = 0; srcCol < srcWidth; ++srcCol)
                {
                    if (src[srcCol] > *threshold)
                    {
                        uint32_t * pDst = dst + srcCol * scale;
                        for (size_t dstRow = 0; dstRow < size; ++dstRow)
                        {
                            for (size_t dstCol = 0; dstCol < size; ++dstCol)
                                pDst[dstCol] = -1;
                            pDst += dstStride;
                        }
                    }
                }
                src += srcStride;
                dst += dstStride*scale;
            }
        }
    }
}
