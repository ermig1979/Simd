/*
* Simd Library (http://simd.sourceforge.net).
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
#include "Simd/SimdSse2.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdEnable.h"
#include "Simd/SimdAllocator.hpp"

#include <vector>

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
	namespace Sse41
	{
        namespace Custom_8x8_18
        {
            struct Buffer
            {
                __m128i pos[5];
                __m128 cos[5], sin[5], kx[8], ky[8];
                    
                int * index;
                float * value;
                __m128 * hist;
                size_t hs;

                Buffer(size_t width)
                {
                    width = AlignHi(width, A / sizeof(float));
                    hs = (width / 8 + 1) * 18 * sizeof(__m128);
                    _p = Allocate(width*(sizeof(int) + sizeof(float)) + hs);
                    index = (int*)_p - 1;
                    value = (float*)index + width;
                    hist = (__m128*)(value + width + 1);

                    for (int i = 0; i < 5; ++i)
                    {
                        cos[i] = _mm_set1_ps((float)::cos(i*M_PI / 9));
                        sin[i] = _mm_set1_ps((float)::sin(i*M_PI / 9));
                        pos[i] = _mm_set1_epi32(i);
                    }
                    for (int i = 0; i < 8; ++i)
                    {
                        float k0 = float((15 - i * 2) / 16.0f);
                        float k1 = 1.0f - k0;
                        kx[i] = _mm_setr_ps(k0, k1, k0, k1);
                        ky[i] = _mm_setr_ps(k0, k0, k1, k1);
                    }
                    ClearHist();
                }

                ~Buffer()
                {
                    Free(_p);
                }

                void ClearHist()
                {
                    memset(hist, 0, hs);
                }

            private:
                void *_p;
            };

            const __m128i K32_1 = SIMD_MM_SET1_EPI32(1);
            const __m128i K32_9 = SIMD_MM_SET1_EPI32(9);
            const __m128i K32_18 = SIMD_MM_SET1_EPI32(18);

            template <bool align> SIMD_INLINE void HogDirectionHistograms(const __m128 & dx, const __m128 & dy, Buffer & buffer, size_t col)
            {
                __m128 bestDot = _mm_setzero_ps();
                __m128i bestIndex = _mm_setzero_si128();
                __m128 _0 = _mm_set1_ps(-0.0f);
                __m128 adx = _mm_andnot_ps(_0, dx);
                __m128 ady = _mm_andnot_ps(_0, dy);
                for (int i = 0; i < 5; ++i)
                {
                    __m128 dot = _mm_add_ps(_mm_mul_ps(adx, buffer.cos[i]), _mm_mul_ps(ady, buffer.sin[i]));
                    __m128 mask = _mm_cmpgt_ps(dot, bestDot);
                    bestDot = _mm_max_ps(dot, bestDot);
                    bestIndex = _mm_blendv_epi8(bestIndex, buffer.pos[i], _mm_castps_si128(mask));
                }
                __m128i maskDx = _mm_castps_si128(_mm_cmplt_ps(dx, _mm_setzero_ps()));
                bestIndex = _mm_blendv_epi8(bestIndex, _mm_sub_epi32(K32_9, bestIndex), maskDx);

                __m128i maskDy = _mm_castps_si128(_mm_cmplt_ps(dy, _mm_setzero_ps()));
                __m128i corr = _mm_and_si128(_mm_castps_si128(_mm_cmpeq_ps(adx, _mm_setzero_ps())), K32_1);
                bestIndex = _mm_blendv_epi8(bestIndex, _mm_sub_epi32(K32_18, _mm_add_epi32(bestIndex, corr)), maskDy);

                bestIndex = _mm_andnot_si128(_mm_cmpeq_epi32(bestIndex, K32_18), bestIndex);

                Store<align>((__m128i*)(buffer.index + col), bestIndex);
                Sse::Store<align>(buffer.value + col, Sse::Sqrt<0>(_mm_add_ps(_mm_mul_ps(adx, adx), _mm_mul_ps(ady, ady))));
            }

            template <bool align> SIMD_INLINE void HogDirectionHistograms(const __m128i & dx, const __m128i & dy, Buffer & buffer, size_t col)
            {
                HogDirectionHistograms<align>(_mm_cvtepi32_ps(UnpackI16<0>(dx)), _mm_cvtepi32_ps(UnpackI16<0>(dy)), buffer, col + 0);
                HogDirectionHistograms<align>(_mm_cvtepi32_ps(UnpackI16<1>(dx)), _mm_cvtepi32_ps(UnpackI16<1>(dy)), buffer, col + 4);
            }

            template <bool align> SIMD_INLINE void HogDirectionHistograms(const uint8_t * src, size_t stride, Buffer & buffer, size_t col)
            {
                const uint8_t * s = src + col;
                __m128i t = Load<false>((__m128i*)(s - stride));
                __m128i l = Load<false>((__m128i*)(s - 1));
                __m128i r = Load<false>((__m128i*)(s + 1));
                __m128i b = Load<false>((__m128i*)(s + stride));
                HogDirectionHistograms<align>(SubUnpackedU8<0>(r, l), SubUnpackedU8<0>(b, t), buffer, col + 0);
                HogDirectionHistograms<align>(SubUnpackedU8<1>(r, l), SubUnpackedU8<1>(b, t), buffer, col + 8);
            }

            void AddRowToBuffer(const uint8_t * src, size_t stride, Buffer & buffer, size_t row, size_t width, size_t aligned)
            {
                const uint8_t * s = src + stride*row;
                for (size_t col = 1; col < aligned; col += A)
                    HogDirectionHistograms<true>(s, stride, buffer, col);
                HogDirectionHistograms<false>(s, stride, buffer, width - 1 - A);

                __m128 ky = buffer.ky[(row + 4) & 7];
                __m128 * hist = buffer.hist;
                size_t cellEnd = width/8;

                for (size_t col = 1; col < 4; ++col)
                {
                    int index = buffer.index[col];
                    __m128 value = _mm_set1_ps(buffer.value[col]);
                    __m128 kx = buffer.kx[(col + 4)&7];
                    hist[index] = _mm_add_ps(hist[index], _mm_mul_ps(value, _mm_mul_ps(ky, kx)));
                }
                hist += 18;

                for (size_t cell = 1, col = 4; cell < cellEnd; ++cell)
                {
                    for (size_t i = 0; i < 8; ++i, ++col)
                    {
                        int index = buffer.index[col];
                        __m128 value = _mm_set1_ps(buffer.value[col]);
                        __m128 kx = buffer.kx[i];
                        hist[index] = _mm_add_ps(hist[index], _mm_mul_ps(value, _mm_mul_ps(ky, kx)));
                    }
                    hist += 18;
                }

                for (size_t col = width - 4; col < width - 1; ++col)
                {
                    int index = buffer.index[col];
                    __m128 value = _mm_set1_ps(buffer.value[col]);
                    __m128 kx = buffer.kx[(col + 4)&7];
                    hist[index] = _mm_add_ps(hist[index], _mm_mul_ps(value, _mm_mul_ps(ky, kx)));
                }
            }

            void AddToHistogram(Buffer & buffer, size_t row, size_t width, size_t height, float * histograms)
            {
                typedef float f18_t[18];

                float * src = (float*)buffer.hist;
                f18_t * h0 = (f18_t*)histograms + row*width - width - 1;
                f18_t * h1 = h0 + width;

                if (row == 0)
                {
                    for (size_t i = 0; i < 18; ++i)
                        h1[1][i] += src[i * 4 + 3];
                    h1++;
                    src += 72;
                    for (size_t cell = 1; cell < width; ++cell)
                    {
                        for (size_t i = 0; i < 18; ++i)
                        {
                            h1[0][i] += src[i * 4 + 2];
                            h1[1][i] += src[i * 4 + 3];
                        }
                        h1++;
                        src += 72;
                    }
                    for (size_t i = 0; i < 18; ++i)
                        h1[0][i] += src[i * 4 + 2];
                }
                else if (row == height)
                {
                    for (size_t i = 0; i < 18; ++i)
                        h0[1][i] += src[i * 4 + 1];
                    h0++;
                    src += 72;
                    for (size_t cell = 1; cell < width; ++cell)
                    {
                        for (size_t i = 0; i < 18; ++i)
                        {
                            h0[0][i] += src[i * 4 + 0];
                            h0[1][i] += src[i * 4 + 1];
                        }
                        h0++;
                        src += 72;
                    }
                    for (size_t i = 0; i < 18; ++i)
                        h0[0][i] += src[i * 4 + 0];
                }
                else
                {
                    for (size_t i = 0; i < 18; ++i)
                    {
                        h0[1][i] += src[i * 4 + 1];
                        h1[1][i] += src[i * 4 + 3];
                    }
                    h0++;
                    h1++;
                    src += 72;
                    for (size_t cell = 1; cell < width; ++cell)
                    {
                        __m128 * ps = (__m128*)src;
                        for (size_t i = 0; i < 16; i += 4)
                        {
                            __m128 s00 = _mm_unpacklo_ps(ps[i + 0], ps[i + 2]);
                            __m128 s01 = _mm_unpacklo_ps(ps[i + 1], ps[i + 3]);
                            __m128 s10 = _mm_unpackhi_ps(ps[i + 0], ps[i + 2]);
                            __m128 s11 = _mm_unpackhi_ps(ps[i + 1], ps[i + 3]);

                            _mm_storeu_ps(h0[0] + i, _mm_add_ps(_mm_loadu_ps(h0[0] + i), _mm_unpacklo_ps(s00, s01)));
                            _mm_storeu_ps(h0[1] + i, _mm_add_ps(_mm_loadu_ps(h0[1] + i), _mm_unpackhi_ps(s00, s01)));
                            _mm_storeu_ps(h1[0] + i, _mm_add_ps(_mm_loadu_ps(h1[0] + i), _mm_unpacklo_ps(s10, s11)));
                            _mm_storeu_ps(h1[1] + i, _mm_add_ps(_mm_loadu_ps(h1[1] + i), _mm_unpackhi_ps(s10, s11)));
                        }
                        for (size_t i = 16; i < 18; ++i)
                        {
                            h0[0][i] += src[i * 4 + 0];
                            h0[1][i] += src[i * 4 + 1];
                            h1[0][i] += src[i * 4 + 2];
                            h1[1][i] += src[i * 4 + 3];
                        }
                        h0++;
                        h1++;
                        src += 72;
                    }
                    for (size_t i = 0; i < 18; ++i)
                    {
                        h0[0][i] += src[i * 4 + 0];
                        h1[0][i] += src[i * 4 + 2];
                    }
                }
                buffer.ClearHist();
            }

            void HogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height, float * histograms)
            {
                const size_t quantization = 18;

                size_t sizeX = width / 8, sizeY = height/8;

                memset(histograms, 0, quantization*sizeX*sizeY*sizeof(float));

                Buffer buffer(width);

                size_t aligned = AlignLo(width - 2, A) + 1;

                for (size_t row = 1; row < 4; ++row)
                    AddRowToBuffer(src, stride, buffer, row, width, aligned);
                AddToHistogram(buffer, 0, sizeX, sizeY, histograms);
                for (size_t row = 4, cell = 1; row < height - 4; ++row)
                {
                    AddRowToBuffer(src, stride, buffer, row, width, aligned);
                    if ((row & 7) == 3)
                        AddToHistogram(buffer, cell++, sizeX, sizeY, histograms);
                }
                for (size_t row = height - 4; row < height - 1; ++row)
                    AddRowToBuffer(src, stride, buffer, row, width, aligned);
                AddToHistogram(buffer, sizeY, sizeX, sizeY, histograms);
            }
        }

        void HogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height,
            size_t cellX, size_t cellY, size_t quantization, float * histograms)
        {
            assert(width%cellX == 0 && height%cellY == 0 && quantization % 2 == 0);

            if (cellX == 8 && cellY == 8 && quantization == 18)
                Custom_8x8_18::HogDirectionHistograms(src, stride, width, height, histograms);
            else
                Sse2::HogDirectionHistograms(src, stride, width, height, cellX, cellY, quantization, histograms);
        }

        class HogFeatureExtractor
        {
            static const size_t C = 8;
            static const size_t Q = 9;
            static const size_t Q2 = 18;

            typedef std::vector<int, Simd::Allocator<int> > Vector32i;
            typedef std::vector<float, Simd::Allocator<float> > Vector32f;

            size_t _sx, _sy;

            __m128i _pos[5];
            __m128 _cos[5], _sin[5];
            __m128 _kx[8], _ky[8];
            __m128i _Q, _Q2;

            Vector32i _index;
            Vector32f _value;
            Vector32f _buffer;
            Vector32f _histogram;
            Vector32f _norm;

            void Init(size_t w, size_t h)
            {
                _sx = w / C;
                _sy = h / C;
                for (int i = 0; i < 5; ++i)
                {
                    _cos[i] = _mm_set1_ps((float)::cos(i*M_PI / Q));
                    _sin[i] = _mm_set1_ps((float)::sin(i*M_PI / Q));
                    _pos[i] = _mm_set1_epi32(i);
                }
                for (int i = 0; i < C; ++i)
                {
                    float k0 = float((15 - i * 2) / 16.0f);
                    float k1 = 1.0f - k0;
                    _kx[i] = _mm_setr_ps(k0, k1, k0, k1);
                    _ky[i] = _mm_setr_ps(k0, k0, k1, k1);
                }
                _Q = _mm_set1_epi32(Q);
                _Q2 = _mm_set1_epi32(Q2);

                _index.resize(w);
                _value.resize(w);
                _buffer.resize((_sx + 1)*4*Q2);
                _histogram.resize(_sx*_sy*Q2);
                _norm.resize(_sx*_sy);
            }

            template <bool align> SIMD_INLINE void GetHistogram(const __m128 & dx, const __m128 & dy, size_t col)
            {
                __m128 bestDot = _mm_setzero_ps();
                __m128i bestIndex = _mm_setzero_si128();
                __m128 _0 = _mm_set1_ps(-0.0f);
                __m128 adx = _mm_andnot_ps(_0, dx);
                __m128 ady = _mm_andnot_ps(_0, dy);
                for (int i = 0; i < 5; ++i)
                {
                    __m128 dot = _mm_add_ps(_mm_mul_ps(adx, _cos[i]), _mm_mul_ps(ady, _sin[i]));
                    __m128 mask = _mm_cmpgt_ps(dot, bestDot);
                    bestDot = _mm_max_ps(dot, bestDot);
                    bestIndex = _mm_blendv_epi8(bestIndex, _pos[i], _mm_castps_si128(mask));
                }
                __m128i maskDx = _mm_castps_si128(_mm_cmplt_ps(dx, _mm_setzero_ps()));
                bestIndex = _mm_blendv_epi8(bestIndex, _mm_sub_epi32(_Q, bestIndex), maskDx);

                __m128i maskDy = _mm_castps_si128(_mm_cmplt_ps(dy, _mm_setzero_ps()));
                __m128i corr = _mm_and_si128(_mm_castps_si128(_mm_cmpeq_ps(adx, _mm_setzero_ps())), K32_00000001);
                bestIndex = _mm_blendv_epi8(bestIndex, _mm_sub_epi32(_Q2, _mm_add_epi32(bestIndex, corr)), maskDy);

                bestIndex = _mm_andnot_si128(_mm_cmpeq_epi32(bestIndex, _Q2), bestIndex);

                Store<align>((__m128i*)(_index.data() + col), bestIndex);
                Sse::Store<align>(_value.data() + col, Sse::Sqrt<0>(_mm_add_ps(_mm_mul_ps(adx, adx), _mm_mul_ps(ady, ady))));
            }

            template <bool align> SIMD_INLINE void GetHistogram(const __m128i & dx, const __m128i & dy, size_t col)
            {
                GetHistogram<align>(_mm_cvtepi32_ps(UnpackI16<0>(dx)), _mm_cvtepi32_ps(UnpackI16<0>(dy)), col + 0);
                GetHistogram<align>(_mm_cvtepi32_ps(UnpackI16<1>(dx)), _mm_cvtepi32_ps(UnpackI16<1>(dy)), col + 4);
            }

            template <bool align> SIMD_INLINE void GetHistogram(const uint8_t * src, size_t stride, size_t col)
            {
                const uint8_t * s = src + col;
                __m128i t = Load<false>((__m128i*)(s - stride));
                __m128i l = Load<false>((__m128i*)(s - 1));
                __m128i r = Load<false>((__m128i*)(s + 1));
                __m128i b = Load<false>((__m128i*)(s + stride));
                GetHistogram<align>(SubUnpackedU8<0>(r, l), SubUnpackedU8<0>(b, t), col + 0);
                GetHistogram<align>(SubUnpackedU8<1>(r, l), SubUnpackedU8<1>(b, t), col + 8);
            }

            void AddRowToBuffer(const uint8_t * src, size_t stride, size_t row, size_t width, size_t aligned)
            {
                const uint8_t * s = src + stride*row;
                GetHistogram<false>(s, stride, 1);
                for (size_t col = A; col < aligned; col += A)
                    GetHistogram<true>(s, stride, col);
                GetHistogram<false>(s, stride, width - 1 - A);

                __m128 ky = _ky[(row + 4) & 7];
                __m128 * buffer = (__m128*)_buffer.data();
                size_t cellEnd = width / 8;

                for (size_t col = 1; col < 4; ++col)
                {
                    int index = _index[col];
                    __m128 value = _mm_set1_ps(_value[col]);
                    buffer[index] = _mm_add_ps(buffer[index], _mm_mul_ps(value, _mm_mul_ps(ky, _kx[(col + 4) & 7])));
                }
                buffer += 18;

                for (size_t cell = 1, col = 4; cell < cellEnd; ++cell)
                {
                    for (size_t i = 0; i < 8; ++i, ++col)
                    {
                        int index = _index[col];
                        __m128 value = _mm_set1_ps(_value[col]);
                        buffer[index] = _mm_add_ps(buffer[index], _mm_mul_ps(value, _mm_mul_ps(ky, _kx[i])));
                    }
                    buffer += 18;
                }

                for (size_t col = width - 4; col < width - 1; ++col)
                {
                    int index = _index[col];
                    __m128 value = _mm_set1_ps(_value[col]);
                    buffer[index] = _mm_add_ps(buffer[index], _mm_mul_ps(value, _mm_mul_ps(ky, _kx[(col + 4) & 7])));
                }
            }

            void AddToHistogram(size_t row, size_t width, size_t height)
            {
                typedef float f18_t[18];

                float * src = _buffer.data();
                f18_t * h0 = (f18_t*)_histogram.data() + row*width - width - 1;
                f18_t * h1 = h0 + width;

                if (row == 0)
                {
                    for (size_t i = 0; i < 18; ++i)
                        h1[1][i] += src[i * 4 + 3];
                    h1++;
                    src += 72;
                    for (size_t cell = 1; cell < width; ++cell)
                    {
                        for (size_t i = 0; i < 18; ++i)
                        {
                            h1[0][i] += src[i * 4 + 2];
                            h1[1][i] += src[i * 4 + 3];
                        }
                        h1++;
                        src += 72;
                    }
                    for (size_t i = 0; i < 18; ++i)
                        h1[0][i] += src[i * 4 + 2];
                }
                else if (row == height)
                {
                    for (size_t i = 0; i < 18; ++i)
                        h0[1][i] += src[i * 4 + 1];
                    h0++;
                    src += 72;
                    for (size_t cell = 1; cell < width; ++cell)
                    {
                        for (size_t i = 0; i < 18; ++i)
                        {
                            h0[0][i] += src[i * 4 + 0];
                            h0[1][i] += src[i * 4 + 1];
                        }
                        h0++;
                        src += 72;
                    }
                    for (size_t i = 0; i < 18; ++i)
                        h0[0][i] += src[i * 4 + 0];
                }
                else
                {
                    for (size_t i = 0; i < 18; ++i)
                    {
                        h0[1][i] += src[i * 4 + 1];
                        h1[1][i] += src[i * 4 + 3];
                    }
                    h0++;
                    h1++;
                    src += 72;
                    for (size_t cell = 1; cell < width; ++cell)
                    {
                        __m128 * ps = (__m128*)src;
                        for (size_t i = 0; i < 16; i += 4)
                        {
                            __m128 s00 = _mm_unpacklo_ps(ps[i + 0], ps[i + 2]);
                            __m128 s01 = _mm_unpacklo_ps(ps[i + 1], ps[i + 3]);
                            __m128 s10 = _mm_unpackhi_ps(ps[i + 0], ps[i + 2]);
                            __m128 s11 = _mm_unpackhi_ps(ps[i + 1], ps[i + 3]);

                            _mm_storeu_ps(h0[0] + i, _mm_add_ps(_mm_loadu_ps(h0[0] + i), _mm_unpacklo_ps(s00, s01)));
                            _mm_storeu_ps(h0[1] + i, _mm_add_ps(_mm_loadu_ps(h0[1] + i), _mm_unpackhi_ps(s00, s01)));
                            _mm_storeu_ps(h1[0] + i, _mm_add_ps(_mm_loadu_ps(h1[0] + i), _mm_unpacklo_ps(s10, s11)));
                            _mm_storeu_ps(h1[1] + i, _mm_add_ps(_mm_loadu_ps(h1[1] + i), _mm_unpackhi_ps(s10, s11)));
                        }
                        for (size_t i = 16; i < 18; ++i)
                        {
                            h0[0][i] += src[i * 4 + 0];
                            h0[1][i] += src[i * 4 + 1];
                            h1[0][i] += src[i * 4 + 2];
                            h1[1][i] += src[i * 4 + 3];
                        }
                        h0++;
                        h1++;
                        src += 72;
                    }
                    for (size_t i = 0; i < 18; ++i)
                    {
                        h0[0][i] += src[i * 4 + 0];
                        h1[0][i] += src[i * 4 + 2];
                    }
                }
                memset(_buffer.data(), 0, _buffer.size() * sizeof(float));
            }

            void GetHistogram(const uint8_t * src, size_t stride, size_t width, size_t height)
            {
                const size_t quantization = 18;

                memset(_histogram.data(), 0, 18*_sx*_sy*sizeof(float));

                size_t aligned = AlignHi(width - 1, A) - A;

                for (size_t row = 1; row < 4; ++row)
                    AddRowToBuffer(src, stride, row, width, aligned);
                AddToHistogram(0, _sx, _sy);
                for (size_t row = 4, cell = 1; row < height - 4; ++row)
                {
                    AddRowToBuffer(src, stride, row, width, aligned);
                    if ((row & 7) == 3)
                        AddToHistogram(cell++, _sx, _sy);
                }
                for (size_t row = height - 4; row < height - 1; ++row)
                    AddRowToBuffer(src, stride, row, width, aligned);
                AddToHistogram(_sy, _sx, _sy);
            }

        public:
            void Run(const uint8_t * src, size_t stride, size_t width, size_t height, float * features)
            {
                Init(width, height);

                GetHistogram(src, stride, width, height);

                for (size_t i = 0, n = _norm.size(); i < n; ++i)
                {
                    const float * h = _histogram.data() + i*Q2;
                    for (int o = 0; o < Q; ++o)
                        _norm[i] += Simd::Square(h[o] + h[o + Q]);
                }

                float eps = 0.0001f;
                for (size_t y = 0; y < _sy; y++)
                {
                    for (size_t x = 0; x < _sx; x++)
                    {
                        float * dst = features + (y*_sx + x) * 31;

                        float *psrc, *p, n1, n2, n3, n4,
                            x0y0 = 0.0f, x1y0 = 0.0f, x2y0 = 0.0f, x0y1 = 0.0f, x1y1 = 0.0f, x2y1 = 0.0f, x0y2 = 0.0f, x1y2 = 0.0f, x2y2 = 0.0f;
                        ptrdiff_t xx = x - 1, yy = y - 1;

                        p = _norm.data() + yy*_sx + xx;
                        if (xx > 0 && yy > 0) 		x0y0 = *p;
                        if (yy > 0)				x1y0 = *(p + 1);
                        if (xx + 2 < (int)_sx && yy > 0)	x2y0 = *(p + 2);
                        if (xx > 0) 				x0y1 = *(p + _sx);
                        x1y1 = *(p + _sx + 1);
                        if (xx + 2 < (int)_sx) 			x2y1 = *(p + 2 + _sx);
                        if (xx > 0 && yy + 2 < (int)_sy)	x0y2 = *(p + 2 * _sx);
                        if (yy + 2 < (int)_sy) 			    x1y2 = *(p + 1 + 2 * _sx);
                        if (xx + 2 < (int)_sx && yy + 2 < (int)_sy) x2y2 = *(p + 2 + 2 * _sx);

                        n1 = 1.0f / sqrt(x1y1 + x2y1 + x1y2 + x2y2 + eps);
                        n2 = 1.0f / sqrt(x1y0 + x2y0 + x1y1 + x2y1 + eps);
                        n3 = 1.0f / sqrt(x0y1 + x1y1 + x0y2 + x1y2 + eps);
                        n4 = 1.0f / sqrt(x0y0 + x1y0 + x0y1 + x1y1 + eps);

                        float t1 = 0;
                        float t2 = 0;
                        float t3 = 0;
                        float t4 = 0;

                        psrc = _histogram.data() + (y*_sx + x)*Q2;
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

                        psrc = _histogram.data() + (y*_sx + x)*Q2;
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
        };

        void HogExtractFeatures(const uint8_t * src, size_t stride, size_t width, size_t height, float * features)
        {
            assert(width % 8 == 0 && height % 8 == 0 && width >= 16 && height >= 16);

            HogFeatureExtractor extractor;
            extractor.Run(src, stride, width, height, features);
        }
	}
#endif// SIMD_SSE41_ENABLE
}