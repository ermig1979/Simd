/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdArray.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <size_t cell> class HogLiteFeatureExtractor
        {
            static const size_t FQ = 8;
            static const size_t HQ = FQ / 2;

            typedef Array<uint8_t> Bytes;
            typedef Array<int> Ints;
            typedef Array<float> Floats;

            size_t _hx, _fx, _w, _aw;
            Bytes _value, _index;
            Ints _hi[2];
            Floats _hf[2], _nf[4];
            int _k0[cell], _k1[cell];
            __m128i _kx8;

            SIMD_INLINE void Init(size_t width)
            {
                _w = (width / cell - 1)*cell;
                _aw = AlignLo(_w, A);
                _hx = width / cell;
                _fx = _hx - 2;
                _value.Resize(_aw + 3 * A, true);
                _index.Resize(_aw + 3 * A, true);
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
                _kx8 = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, 15, 13, 11, 9, 7, 5, 3, 1);
            }

            template<bool align> static SIMD_INLINE void SetIndexAndValue(const uint8_t * src, size_t stride, uint8_t * value, uint8_t * index)
            {
                __m128i y0 = Load<false>((__m128i*)(src - stride));
                __m128i y1 = Load<false>((__m128i*)(src + stride));
                __m128i x0 = Load<false>((__m128i*)(src - 1));
                __m128i x1 = Load<false>((__m128i*)(src + 1));

                __m128i ady = AbsDifferenceU8(y0, y1);
                __m128i adx = AbsDifferenceU8(x0, x1);

                __m128i max = _mm_max_epu8(ady, adx);
                __m128i min = _mm_min_epu8(ady, adx);
                __m128i val = _mm_adds_epu8(max, _mm_avg_epu8(min, K_ZERO));
                Store<align>((__m128i*)value, val);

                __m128i idx = _mm_blendv_epi8(K8_01, K_ZERO, Compare8u<SimdCompareGreater>(adx, ady));
                idx = _mm_blendv_epi8(_mm_sub_epi8(K8_03, idx), idx, Compare8u<SimdCompareGreater>(x1, x0));
                idx = _mm_blendv_epi8(_mm_sub_epi8(K8_07, idx), idx, Compare8u<SimdCompareGreater>(y1, y0));
                Store<align>((__m128i*)index, idx);
            }

            SIMD_INLINE void SetIndexAndValue(const uint8_t * src, size_t stride)
            {
                uint8_t * value = _value.data + A;
                uint8_t * index = _index.data + A;
                for (size_t col = 0; col < _aw; col += A)
                    SetIndexAndValue<true>(src + col, stride, value + col, index + col);
                if (_aw < _w)
                {
                    size_t col = _w - A;
                    SetIndexAndValue<false>(src + col, stride, value + col, index + col);
                }
            }

            SIMD_INLINE void UpdateIntegerHistogram4x4(size_t rowI, size_t rowF)
            {
                int * h0 = _hi[(rowI + 0) & 1].data;
                int * h1 = _hi[(rowI + 1) & 1].data;
                uint8_t * value = _value.data + A;
                uint8_t * index = _index.data + A;
                int ky0 = _k0[rowF];
                int ky1 = _k1[rowF];
                for (size_t col = 0; col < _w;)
                {
                    for (size_t colF = 0; colF < cell; ++colF, ++col)
                    {
                        int val = value[col];
                        int idx = index[col];
                        h0[00 + idx] += val*_k0[colF] * ky0;
                        h1[00 + idx] += val*_k0[colF] * ky1;
                        h0[FQ + idx] += val*_k1[colF] * ky0;
                        h1[FQ + idx] += val*_k1[colF] * ky1;
                    }
                    h0 += FQ;
                    h1 += FQ;
                }
            }

            SIMD_INLINE void UpdateIntegerHistogram8x8(size_t rowI, size_t rowF)
            {
                int * h0 = _hi[(rowI + 0) & 1].data;
                int * h1 = _hi[(rowI + 1) & 1].data;
                uint8_t * value = _value.data + A - cell;
                uint8_t * index = _index.data + A - cell;
                __m128i ky0 = _mm_set1_epi16((short)_k0[rowF]);
                __m128i ky1 = _mm_set1_epi16((short)_k1[rowF]);
                for (size_t col = 0; col <= _w; col += cell)
                {
                    __m128i val = Load<false>((__m128i*)(value + col));
                    __m128i idx = Load<false>((__m128i*)(index + col));
                    __m128i cur0 = K_ZERO;
                    __m128i cur1 = K8_01;
                    __m128i dirs[4];
                    for (size_t i = 0; i < 4; ++i)
                    {
                        __m128i dir0 = _mm_maddubs_epi16(_mm_and_si128(_mm_cmpeq_epi8(idx, cur0), val), _kx8);
                        __m128i dir1 = _mm_maddubs_epi16(_mm_and_si128(_mm_cmpeq_epi8(idx, cur1), val), _kx8);
                        dirs[i] = _mm_hadd_epi16(dir0, dir1);
                        cur0 = _mm_add_epi8(cur0, K8_02);
                        cur1 = _mm_add_epi8(cur1, K8_02);
                    }
                    dirs[0] = _mm_hadd_epi16(dirs[0], dirs[1]);
                    dirs[1] = _mm_hadd_epi16(dirs[2], dirs[3]);
                    Store<true>((__m128i*)h0 + 0, _mm_add_epi32(Load<true>((__m128i*)h0 + 0), _mm_madd_epi16(dirs[0], ky0)));
                    Store<true>((__m128i*)h0 + 1, _mm_add_epi32(Load<true>((__m128i*)h0 + 1), _mm_madd_epi16(dirs[1], ky0)));
                    Store<true>((__m128i*)h1 + 0, _mm_add_epi32(Load<true>((__m128i*)h1 + 0), _mm_madd_epi16(dirs[0], ky1)));
                    Store<true>((__m128i*)h1 + 1, _mm_add_epi32(Load<true>((__m128i*)h1 + 1), _mm_madd_epi16(dirs[1], ky1)));
                    h0 += FQ;
                    h1 += FQ;
                }
            }

            SIMD_INLINE void UpdateFloatHistogram(size_t rowI)
            {
                const float k = 1.0f / 256.0f;
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
                height = (height / cell - 1)*cell;

                float k = 1.0f / 256.0f, eps = 0.0001f;
                for (size_t row = 0; row < height; ++row)
                {
                    SetIndexAndValue(src, srcStride);
                    size_t rowI = row / cell;
                    size_t rowF = row & (cell - 1);
                    if(cell == 8)
                        UpdateIntegerHistogram8x8(rowI, rowF);
                    else
                        UpdateIntegerHistogram4x4(rowI, rowF);
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
                size_t rowI = height / cell;
                UpdateFloatHistogram(rowI);
                SetFeatures(rowI, features);
            }
        };

        void HogLiteExtractFeatures8x8(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * features, size_t featuresStride)
        {
            HogLiteFeatureExtractor<8> extractor;
            extractor.Run(src, srcStride, width, height, features, featuresStride);
        }
    }
#endif// SIMD_SSE41_ENABLE
}


