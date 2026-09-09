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
#include "Simd/SimdStore.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE void HogDeinterleave(const float* src, size_t count, float** dst, size_t offset, size_t i)
        {
            src += i;
            __m128 a0 = Load<false>(src + 0 * count);
            __m128 a1 = Load<false>(src + 1 * count);
            __m128 a2 = Load<false>(src + 2 * count);
            __m128 a3 = Load<false>(src + 3 * count);
            __m128 b0 = _mm_unpacklo_ps(a0, a2);
            __m128 b1 = _mm_unpackhi_ps(a0, a2);
            __m128 b2 = _mm_unpacklo_ps(a1, a3);
            __m128 b3 = _mm_unpackhi_ps(a1, a3);
            Store<false>(dst[i + 0] + offset, _mm_unpacklo_ps(b0, b2));
            Store<false>(dst[i + 1] + offset, _mm_unpackhi_ps(b0, b2));
            Store<false>(dst[i + 2] + offset, _mm_unpacklo_ps(b1, b3));
            Store<false>(dst[i + 3] + offset, _mm_unpackhi_ps(b1, b3));
        }

        void HogDeinterleave(const float* src, size_t srcStride, size_t width, size_t height, size_t count, float** dst, size_t dstStride)
        {
            assert(width >= F && count >= F);

            size_t alignedCount = AlignLo(count, F);
            size_t alignedWidth = AlignLo(width, F);

            for (size_t row = 0; row < height; ++row)
            {
                size_t rowOffset = row * dstStride;
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    const float* s = src + count * col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - F);
                }
                if (alignedWidth != width)
                {
                    size_t col = width - F;
                    const float* s = src + count * col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - F);
                }
                src += srcStride;
            }
        }

        class HogFeatureExtractor
        {
            static const size_t C = 8;
            static const size_t Q = 9;
            static const size_t Q2 = 18;

            size_t _sx, _sy, _hs;

            __m128i _pos[5];
            __m128 _cos[5], _sin[5];
            __m128 _kx[8], _ky[8];
            __m128i _Q, _Q2;

            Array32i _index;
            Array32f _value;
            Array32f _buffer;
            Array32f _histogram;
            Array32f _norm;

            void Init(size_t w, size_t h)
            {
                _sx = w / C;
                _hs = _sx + 2;
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

                _index.Resize(w);
                _value.Resize(w);
                _buffer.Resize((_sx + 1) * 4 * Q2);
                _histogram.Resize((_sx + 2)*(_sy + 2)*Q2);
                _norm.Resize((_sx + 2)*(_sy + 2));
            }

            template <bool align> SIMD_INLINE void GetHistogram(const __m128 & dx, const __m128 & dy, size_t col)
            {
                __m128 _0 = _mm_set1_ps(-0.0f);
                __m128 adx = _mm_andnot_ps(_0, dx);
                __m128 ady = _mm_andnot_ps(_0, dy);
                __m128 bestDot = _mm_add_ps(_mm_mul_ps(adx, _cos[0]), _mm_mul_ps(ady, _sin[0]));
                __m128i bestIndex = _pos[0];
                for (int i = 1; i < 5; ++i)
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

                Store<align>((__m128i*)(_index.data + col), bestIndex);
                Store<align>(_value.data + col, Sqrt<0>(_mm_add_ps(_mm_mul_ps(adx, adx), _mm_mul_ps(ady, ady))));
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
                const uint8_t * s = src + stride * row;
                GetHistogram<false>(s, stride, 1);
                for (size_t col = A; col < aligned; col += A)
                    GetHistogram<true>(s, stride, col);
                GetHistogram<false>(s, stride, width - 1 - A);

                __m128 * buffer = (__m128*)_buffer.data;
                __m128 ky = _ky[(row + 4) & 7];
                for (size_t col = 1, n = C, i = 5; col < width - 1; i = 0, n = Simd::Min<size_t>(C, width - col - 1))
                {
                    for (; i < n; ++i, ++col)
                    {
                        int index = _index[col];
                        __m128 value = _mm_set1_ps(_value[col]);
                        buffer[index] = _mm_add_ps(buffer[index], _mm_mul_ps(value, _mm_mul_ps(ky, _kx[i])));
                    }
                    buffer += Q2;
                }
            }

            void AddToHistogram(size_t row, size_t width, size_t height)
            {
                typedef float f18_t[18];
                float * src = _buffer.data;
                f18_t * h0 = (f18_t*)_histogram.data + row * _hs;
                f18_t * h1 = h0 + _hs;
                for (size_t cell = 0; cell <= width; ++cell)
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
                    __m128 s0 = _mm_add_ps(_mm_unpacklo_ps(ps[16], ps[17]), Load(h0[0] + 16, h0[1] + 16));
                    __m128 s1 = _mm_add_ps(_mm_unpackhi_ps(ps[16], ps[17]), Load(h1[0] + 16, h1[1] + 16));
                    StoreHalf<0>(h0[0] + 16, s0);
                    StoreHalf<1>(h0[1] + 16, s0);
                    StoreHalf<0>(h1[0] + 16, s1);
                    StoreHalf<1>(h1[1] + 16, s1);
                    h0++;
                    h1++;
                    src += 4 * Q2;
                }
                _buffer.Clear();
            }

            void EstimateHistogram(const uint8_t * src, size_t stride, size_t width, size_t height)
            {
                _histogram.Clear();

                size_t aligned = AlignHi(width - 1, A) - A;

                _buffer.Clear();
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

            SIMD_INLINE float GetNorm(const float * src)
            {
                __m128 _norm = _mm_setzero_ps();
                for (size_t i = 0; i < 8; i += 4)
                {
                    __m128 sum = _mm_add_ps(_mm_loadu_ps(src + i + 0), _mm_loadu_ps(src + i + Q));
                    _norm = _mm_add_ps(_norm, _mm_mul_ps(sum, sum));
                }
                _norm = _mm_hadd_ps(_norm, _norm);
                _norm = _mm_hadd_ps(_norm, _norm);
                float norm;
                _mm_store_ss(&norm, _norm);
                return norm + Simd::Square(src[Q - 1] + src[Q2 - 1]);
            }

            void EstimateNorm()
            {
                _norm.Clear();
                for (size_t y = 0, i = 0; y < _sy; y++)
                {
                    const float * h = _histogram.data + ((y + 1)*_hs + 1)*Q2;
                    float * n = _norm.data + (y + 1)*_hs + 1;
                    for (size_t x = 0; x < _sx; x++, i++)
                        n[x] = GetNorm(h + x * Q2);
                }
            }

            void ExtractFeatures(float * features)
            {
                __m128 _02 = _mm_set1_ps(0.2f);
                __m128 _05 = _mm_set1_ps(0.5f);
                __m128 _02357 = _mm_set1_ps(0.2357f);
                __m128 eps = _mm_set1_ps(0.0001f);
                for (size_t y = 0; y < _sy; y++)
                {
                    float * ph = _histogram.data + ((y + 1)*_hs + 1)*Q2;
                    for (size_t x = 0; x < _sx; x++)
                    {
                        float * dst = features + (y*_sx + x) * 31;

                        float * p0 = _norm.data + y * _hs + x;
                        float * p1 = p0 + _hs;
                        float * p2 = p1 + _hs;

                        __m128 n = _mm_setr_ps(
                            p1[1] + p1[2] + p2[1] + p2[2],
                            p0[1] + p0[2] + p1[1] + p1[2],
                            p1[0] + p1[1] + p2[0] + p2[1],
                            p0[0] + p0[1] + p1[0] + p1[1]);

                        n = _mm_rsqrt_ps(_mm_add_ps(n, eps));

                        __m128 t = _mm_setzero_ps();

                        float * src = ph + x * Q2;
                        for (int o = 0; o < 16; o += 4)
                        {
                            __m128 s = _mm_loadu_ps(src);
                            __m128 h0 = _mm_min_ps(_mm_mul_ps(Broadcast<0>(s), n), _02);
                            __m128 h1 = _mm_min_ps(_mm_mul_ps(Broadcast<1>(s), n), _02);
                            __m128 h2 = _mm_min_ps(_mm_mul_ps(Broadcast<2>(s), n), _02);
                            __m128 h3 = _mm_min_ps(_mm_mul_ps(Broadcast<3>(s), n), _02);
                            t = _mm_add_ps(t, _mm_add_ps(_mm_add_ps(h0, h1), _mm_add_ps(h2, h3)));
                            _mm_storeu_ps(dst, _mm_mul_ps(_05, _mm_hadd_ps(_mm_hadd_ps(h0, h1), _mm_hadd_ps(h2, h3))));
                            dst += 4;
                            src += 4;
                        }
                        {
                            __m128 h0 = _mm_min_ps(_mm_mul_ps(_mm_set1_ps(*src++), n), _02);
                            __m128 h1 = _mm_min_ps(_mm_mul_ps(_mm_set1_ps(*src++), n), _02);
                            t = _mm_add_ps(t, _mm_add_ps(h0, h1));
                            __m128 h = _mm_hadd_ps(h0, h1);
                            _mm_storeu_ps(dst, _mm_mul_ps(_05, _mm_hadd_ps(h, h)));
                            dst += 2;
                        }

                        src = ph + x * Q2;
                        for (int o = 0; o < 8; o += 4)
                        {
                            __m128 s = _mm_add_ps(_mm_loadu_ps(src), _mm_loadu_ps(src + Q));
                            __m128 h0 = _mm_min_ps(_mm_mul_ps(Broadcast<0>(s), n), _02);
                            __m128 h1 = _mm_min_ps(_mm_mul_ps(Broadcast<1>(s), n), _02);
                            __m128 h2 = _mm_min_ps(_mm_mul_ps(Broadcast<2>(s), n), _02);
                            __m128 h3 = _mm_min_ps(_mm_mul_ps(Broadcast<3>(s), n), _02);
                            _mm_storeu_ps(dst, _mm_mul_ps(_05, _mm_hadd_ps(_mm_hadd_ps(h0, h1), _mm_hadd_ps(h2, h3))));
                            dst += 4;
                            src += 4;
                        }
                        {
                            __m128 s = _mm_set1_ps(src[0] + src[Q]);
                            __m128 h = _mm_min_ps(_mm_mul_ps(s, n), _02);
                            h = _mm_dp_ps(_05, h, 0xF1);
                            _mm_store_ss(dst++, h);
                        }
                        _mm_storeu_ps(dst, _mm_mul_ps(t, _02357));
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
            assert(width >= A + 2);

            HogFeatureExtractor extractor;
            extractor.Run(src, stride, width, height, features);
        }

        //-----------------------------------------------------------------------------------------

        namespace HogSeparableFilter_Detail
        {
            template <int add, bool end> SIMD_INLINE void Set(float* dst, const __m128& value, const __m128& mask)
            {
                Store<false>(dst, value);
            }

            template <> SIMD_INLINE void Set<1, false>(float* dst, const __m128& value, const __m128& mask)
            {
                Store<false>(dst, _mm_add_ps(Load<false>(dst), value));
            }

            template <> SIMD_INLINE void Set<1, true>(float* dst, const __m128& value, const __m128& mask)
            {
                Store<false>(dst, _mm_add_ps(Load<false>(dst), _mm_and_ps(value, mask)));
            }
        }

        class HogSeparableFilter
        {
            size_t _w, _h, _s;
            Array32f _buffer;
            Array128f _filter;

            void Init(size_t w, size_t h, size_t rs, size_t cs)
            {
                _w = w - rs + 1;
                _s = AlignHi(_w, F);
                _h = h - cs + 1;
                _buffer.Resize(_s * h);
            }

            template <bool align> SIMD_INLINE void FilterRows(const float* src, const __m128* filter, size_t size, float* dst)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t i = 0; i < size; ++i)
                    sum = _mm_add_ps(sum, _mm_mul_ps(Load<false>(src + i), filter[i]));
                Store<align>(dst, sum);
            }

            void FilterRows(const float* src, size_t srcStride, size_t width, size_t height, const float* filter, size_t size, float* dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm_set1_ps(filter[i]);

                size_t alignedWidth = AlignLo(width, F);

                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += F)
                        FilterRows<true>(src + col, _filter.data, size, dst + col);
                    if (alignedWidth != width)
                        FilterRows<false>(src + width - F, _filter.data, size, dst + width - F);
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <int add, bool end> SIMD_INLINE void FilterCols(const float* src, size_t stride, const __m128* filter, size_t size, float* dst, const __m128& mask)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = _mm_add_ps(sum, _mm_mul_ps(Load<!end>(src), filter[i]));
                HogSeparableFilter_Detail::Set<add, end>(dst, sum, mask);
            }

            template <int add> void FilterCols(const float* src, size_t srcStride, size_t width, size_t height, const float* filter, size_t size, float* dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm_set1_ps(filter[i]);

                size_t alignedWidth = AlignLo(width, F);
                __m128 tailMask = RightNotZero32f(width - alignedWidth);

                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += F)
                        FilterCols<add, false>(src + col, srcStride, _filter.data, size, dst + col, tailMask);
                    if (alignedWidth != width)
                        FilterCols<add, true>(src + width - F, srcStride, _filter.data, size, dst + width - F, tailMask);
                    src += srcStride;
                    dst += dstStride;
                }
            }

        public:

            void Run(const float* src, size_t srcStride, size_t width, size_t height,
                const float* rowFilter, size_t rowSize, const float* colFilter, size_t colSize, float* dst, size_t dstStride, int add)
            {
                Init(width, height, rowSize, colSize);

                FilterRows(src, srcStride, _w, height, rowFilter, rowSize, _buffer.data, _s);

                if (add)
                    FilterCols<1>(_buffer.data, _s, _w, _h, colFilter, colSize, dst, dstStride);
                else
                    FilterCols<0>(_buffer.data, _s, _w, _h, colFilter, colSize, dst, dstStride);
            }
        };

        void HogFilterSeparable(const float* src, size_t srcStride, size_t width, size_t height,
            const float* rowFilter, size_t rowSize, const float* colFilter, size_t colSize, float* dst, size_t dstStride, int add)
        {
            assert(width >= F + rowSize - 1 && height >= colSize - 1);

            HogSeparableFilter filter;
            filter.Run(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
        }
    }
#endif
}
