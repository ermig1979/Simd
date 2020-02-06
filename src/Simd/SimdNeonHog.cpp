/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE void HogDeinterleave(const float * src, size_t count, float ** dst, size_t offset, size_t i)
        {
            src += i;
            float32x4x2_t a01 = vzipq_f32(Load<false>(src + 0 * count), Load<false>(src + 2 * count));
            float32x4x2_t a23 = vzipq_f32(Load<false>(src + 1 * count), Load<false>(src + 3 * count));
            float32x4x2_t b01 = vzipq_f32(a01.val[0], a23.val[0]);
            float32x4x2_t b23 = vzipq_f32(a01.val[1], a23.val[1]);
            Store<false>(dst[i + 0] + offset, b01.val[0]);
            Store<false>(dst[i + 1] + offset, b01.val[1]);
            Store<false>(dst[i + 2] + offset, b23.val[0]);
            Store<false>(dst[i + 3] + offset, b23.val[1]);
        }

        void HogDeinterleave(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride)
        {
            assert(width >= F && count >= F);

            size_t alignedCount = AlignLo(count, F);
            size_t alignedWidth = AlignLo(width, F);

            for (size_t row = 0; row < height; ++row)
            {
                size_t rowOffset = row*dstStride;
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    const float * s = src + count*col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - F);
                }
                if (alignedWidth != width)
                {
                    size_t col = width - F;
                    const float * s = src + count*col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += F)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - F);
                }
                src += srcStride;
            }
        }

        namespace
        {
            struct Buffer
            {
                const int size;
                float32x4_t * cos, *sin;
                int32x4_t * pos, *neg;
                int * index;
                float * value;

                Buffer(size_t width, size_t quantization)
                    : size((int)quantization / 2)
                {
                    width = AlignHi(width, A / sizeof(float));
                    _p = Allocate(width*(sizeof(int) + sizeof(float)) + (sizeof(int32x4_t) + sizeof(float32x4_t)) * 2 * size);
                    index = (int*)_p - 1;
                    value = (float*)index + width;
                    cos = (float32x4_t*)(value + width + 1);
                    sin = cos + size;
                    pos = (int32x4_t*)(sin + size);
                    neg = pos + size;
                    for (int i = 0; i < size; ++i)
                    {
                        cos[i] = vdupq_n_f32((float)::cos(i*M_PI / size));
                        sin[i] = vdupq_n_f32((float)::sin(i*M_PI / size));
                        pos[i] = vdupq_n_s32(i);
                        neg[i] = vdupq_n_s32(size + i);
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

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const float32x4_t & dx, const float32x4_t & dy, Buffer & buffer, size_t col)
        {
            float32x4_t bestDot = vdupq_n_f32(0);
            int32x4_t bestIndex = vdupq_n_s32(0);
            for (int i = 0; i < buffer.size; ++i)
            {
                float32x4_t dot = vaddq_f32(vmulq_f32(dx, buffer.cos[i]), vmulq_f32(dy, buffer.sin[i]));
                uint32x4_t mask = vcgtq_f32(dot, bestDot);
                bestDot = vmaxq_f32(dot, bestDot);
                bestIndex = vbslq_s32(mask, buffer.pos[i], bestIndex);

                dot = vnegq_f32(dot);
                mask = vcgtq_f32(dot, bestDot);
                bestDot = vmaxq_f32(dot, bestDot);
                bestIndex = vbslq_s32(mask, buffer.neg[i], bestIndex);
            }
            Store<align>(buffer.index + col, bestIndex);
            Store<align>(buffer.value + col, Sqrt<SIMD_NEON_RCP_ITER>(vaddq_f32(vmulq_f32(dx, dx), vmulq_f32(dy, dy))));
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const int16x8_t & dx, const int16x8_t & dy, Buffer & buffer, size_t col)
        {
            HogDirectionHistograms<align>(ToFloat<0>(dx), ToFloat<0>(dy), buffer, col + 0);
            HogDirectionHistograms<align>(ToFloat<1>(dx), ToFloat<1>(dy), buffer, col + 4);
        }

        template <bool align> SIMD_INLINE void HogDirectionHistograms(const uint8_t * src, size_t stride, Buffer & buffer, size_t col)
        {
            const uint8_t * s = src + col;
            uint8x16_t t = Load<false>(s - stride);
            uint8x16_t l = Load<false>(s - 1);
            uint8x16_t r = Load<false>(s + 1);
            uint8x16_t b = Load<false>(s + stride);
            HogDirectionHistograms<align>(Sub<0>(r, l), Sub<0>(b, t), buffer, col + 0);
            HogDirectionHistograms<align>(Sub<1>(r, l), Sub<1>(b, t), buffer, col + 8);
        }

        void HogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height,
            size_t cellX, size_t cellY, size_t quantization, float * histograms)
        {
            assert(width%cellX == 0 && height%cellY == 0 && quantization % 2 == 0);

            Buffer buffer(width, quantization);

            memset(histograms, 0, quantization*(width / cellX)*(height / cellY) * sizeof(float));

            size_t alignedWidth = AlignLo(width - 2, A) + 1;

            for (size_t row = 1; row < height - 1; ++row)
            {
                const uint8_t * s = src + stride*row;
                for (size_t col = 1; col < alignedWidth; col += A)
                    HogDirectionHistograms<true>(s, stride, buffer, col);
                HogDirectionHistograms<false>(s, stride, buffer, width - 1 - A);
                Base::AddRowToHistograms(buffer.index, buffer.value, row, width, height, cellX, cellY, quantization, histograms);
            }
        }

        class HogFeatureExtractor
        {
            static const size_t C = 8;
            static const size_t Q = 9;
            static const size_t Q2 = 18;

            typedef Array<int> Array32i;
            typedef Array<float> Array32f;

            size_t _sx, _sy, _hs;

            int32x4_t _pos[5];
            float32x4_t _cos[5], _sin[5];
            float32x4_t _kx[8], _ky[8];
            int32x4_t _Q, _Q2;

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
                    _cos[i] = vdupq_n_f32((float)::cos(i*M_PI / Q));
                    _sin[i] = vdupq_n_f32((float)::sin(i*M_PI / Q));
                    _pos[i] = vdupq_n_s32(i);
                }
                for (int i = 0; i < C; ++i)
                {
                    float k0 = float((15 - i * 2) / 16.0f);
                    float k1 = 1.0f - k0;
                    _kx[i] = SetF32(k0, k1, k0, k1);
                    _ky[i] = SetF32(k0, k0, k1, k1);
                }
                _Q = vdupq_n_s32(Q);
                _Q2 = vdupq_n_s32(Q2);

                _index.Resize(w);
                _value.Resize(w);
                _buffer.Resize((_sx + 1) * 4 * Q2);
                _histogram.Resize((_sx + 2)*(_sy + 2)*Q2);
                _norm.Resize((_sx + 2)*(_sy + 2));
            }

            template <bool align> SIMD_INLINE void GetHistogram(const float32x4_t & dx, const float32x4_t & dy, size_t col)
            {
                float32x4_t _0 = vdupq_n_f32(0);
                float32x4_t bestDot = _0;
                int32x4_t bestIndex = vdupq_n_s32(0);
                float32x4_t adx = vabsq_f32(dx);
                float32x4_t ady = vabsq_f32(dy);
                for (int i = 0; i < 5; ++i)
                {
                    float32x4_t dot = vmlaq_f32(vmulq_f32(adx, _cos[i]), ady, _sin[i]);
                    uint32x4_t mask = vcgtq_f32(dot, bestDot);
                    bestDot = vmaxq_f32(dot, bestDot);
                    bestIndex = vbslq_s32(mask, _pos[i], bestIndex);
                }
                uint32x4_t maskDx = vcltq_f32(dx, _0);
                bestIndex = vbslq_s32(maskDx, vsubq_s32(_Q, bestIndex), bestIndex);

                uint32x4_t maskDy = vcltq_f32(dy, _0);
                uint32x4_t corr = vandq_u32(vceqq_f32(adx, _0), K32_00000001);
                bestIndex = vbslq_s32(maskDy, vsubq_s32(_Q2, vaddq_s32(bestIndex, (int32x4_t)corr)), bestIndex);

                bestIndex = vbslq_s32(vceqq_s32(bestIndex, _Q2), (int32x4_t)K32_00000000, bestIndex);

                Store<false>(_index.data + col, bestIndex); // fixed program crash.
                Store<align>(_value.data + col, Sqrt<SIMD_NEON_RCP_ITER>(vmlaq_f32(vmulq_f32(adx, adx), ady, ady)));
            }

            template <bool align> SIMD_INLINE void GetHistogram(const int16x8_t & dx, const int16x8_t & dy, size_t col)
            {
                GetHistogram<align>(ToFloat<0>(dx), ToFloat<0>(dy), col + 0);
                GetHistogram<align>(ToFloat<1>(dx), ToFloat<1>(dy), col + 4);
            }

            template <bool align> SIMD_INLINE void GetHistogram(const uint8_t * src, size_t stride, size_t col)
            {
                const uint8_t * s = src + col;
                uint8x16_t t = Load<false>(s - stride);
                uint8x16_t l = Load<false>(s - 1);
                uint8x16_t r = Load<false>(s + 1);
                uint8x16_t b = Load<false>(s + stride);
                GetHistogram<align>(Sub<0>(r, l), Sub<0>(b, t), col + 0);
                GetHistogram<align>(Sub<1>(r, l), Sub<1>(b, t), col + 8);
            }

            void AddRowToBuffer(const uint8_t * src, size_t stride, size_t row, size_t width, size_t aligned)
            {
                const uint8_t * s = src + stride*row;
                GetHistogram<false>(s, stride, 1);
                for (size_t col = A; col < aligned; col += A)
                    GetHistogram<true>(s, stride, col);
                GetHistogram<false>(s, stride, width - 1 - A);

                float32x4_t * buffer = (float32x4_t*)_buffer.data;
                float32x4_t ky = _ky[(row + 4) & 7];
                for (size_t col = 1, n = C, i = 5; col < width - 1; i = 0, n = Simd::Min<size_t>(C, width - col - 1))
                {
                    for (; i < n; ++i, ++col)
                    {
                        int index = _index[col];
                        float32x4_t value = vdupq_n_f32(_value[col]);
                        buffer[index] = vmlaq_f32(buffer[index], value, vmulq_f32(ky, _kx[i]));
                    }
                    buffer += Q2;
                }
            }

            void AddToHistogram(size_t row, size_t width, size_t height)
            {
                typedef float f18_t[18];
                const float * src = _buffer.data;
                f18_t * h0 = (f18_t*)_histogram.data + row*_hs;
                f18_t * h1 = h0 + _hs;
                for (size_t cell = 0; cell <= width; ++cell)
                {
                    for (size_t i = 0; i < 16; i += 4)
                    {
                        float32x4x4_t s = Load4<true>(src + 4 * i);
                        Store<false>(h0[0] + i, vaddq_f32(Load<false>(h0[0] + i), s.val[0]));
                        Store<false>(h0[1] + i, vaddq_f32(Load<false>(h0[1] + i), s.val[1]));
                        Store<false>(h1[0] + i, vaddq_f32(Load<false>(h1[0] + i), s.val[2]));
                        Store<false>(h1[1] + i, vaddq_f32(Load<false>(h1[1] + i), s.val[3]));
                    }
                    float32x2x4_t s = LoadHalf4<true>(src + 64);
                    Store<false>(h0[0] + 16, vadd_f32(LoadHalf<false>(h0[0] + 16), s.val[0]));
                    Store<false>(h0[1] + 16, vadd_f32(LoadHalf<false>(h0[1] + 16), s.val[1]));
                    Store<false>(h1[0] + 16, vadd_f32(LoadHalf<false>(h1[0] + 16), s.val[2]));
                    Store<false>(h1[1] + 16, vadd_f32(LoadHalf<false>(h1[1] + 16), s.val[3]));
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
                float32x4_t norm = vdupq_n_f32(0);
                for (size_t i = 0; i < 8; i += 4)
                {
                    float32x4_t sum = vaddq_f32(Load<false>(src + i + 0), Load<false>(src + i + Q));
                    norm = vmlaq_f32(norm, sum, sum);
                }
                return ExtractSum32f(norm) + Simd::Square(src[Q - 1] + src[Q2 - 1]);
            }

            void EstimateNorm()
            {
                _norm.Clear();
                for (size_t y = 0, i = 0; y < _sy; y++)
                {
                    const float * h = _histogram.data + ((y + 1)*_hs + 1)*Q2;
                    float * n = _norm.data + (y + 1)*_hs + 1;
                    for (size_t x = 0; x < _sx; x++, i++)
                        n[x] = GetNorm(h + x*Q2);
                }
            }

            void ExtractFeatures(float * features)
            {
                float32x4_t _02 = vdupq_n_f32(0.2f);
                float32x4_t _05 = vdupq_n_f32(0.5f);
                float32x4_t _02357 = vdupq_n_f32(0.2357f);
                float32x4_t eps = vdupq_n_f32(0.0001f);
                for (size_t y = 0; y < _sy; y++)
                {
                    float * ph = _histogram.data + ((y + 1)*_hs + 1)*Q2;
                    for (size_t x = 0; x < _sx; x++)
                    {
                        float * dst = features + (y*_sx + x) * 31;

                        float * p0 = _norm.data + y*_hs + x;
                        float * p1 = p0 + _hs;
                        float * p2 = p1 + _hs;

                        float32x4_t n = SetF32(
                            p1[1] + p1[2] + p2[1] + p2[2],
                            p0[1] + p0[2] + p1[1] + p1[2],
                            p1[0] + p1[1] + p2[0] + p2[1],
                            p0[0] + p0[1] + p1[0] + p1[1]);

                        n = ReciprocalSqrt<SIMD_NEON_RCP_ITER>(vaddq_f32(n, eps));

                        float32x4_t t = vdupq_n_f32(0);

                        float * src = ph + x*Q2;
                        for (int o = 0; o < 16; o += 4)
                        {
                            float32x4_t s = Load<false>(src);
                            float32x4_t h0 = vminq_f32(vmulq_f32(Broadcast<0>(s), n), _02);
                            float32x4_t h1 = vminq_f32(vmulq_f32(Broadcast<1>(s), n), _02);
                            float32x4_t h2 = vminq_f32(vmulq_f32(Broadcast<2>(s), n), _02);
                            float32x4_t h3 = vminq_f32(vmulq_f32(Broadcast<3>(s), n), _02);
                            t = vaddq_f32(t, vaddq_f32(vaddq_f32(h0, h1), vaddq_f32(h2, h3)));
                            Store<false>(dst, vmulq_f32(_05, Hadd(Hadd(h0, h1), Hadd(h2, h3))));
                            dst += 4;
                            src += 4;
                        }
                        {
                            float32x4_t h0 = vminq_f32(vmulq_f32(vdupq_n_f32(*src++), n), _02);
                            float32x4_t h1 = vminq_f32(vmulq_f32(vdupq_n_f32(*src++), n), _02);
                            t = vaddq_f32(t, vaddq_f32(h0, h1));
                            float32x4_t h = Hadd(h0, h1);
                            Store<false>(dst, vmulq_f32(_05, Hadd(h, h)));
                            dst += 2;
                        }

                        src = ph + x*Q2;
                        for (int o = 0; o < 8; o += 4)
                        {
                            float32x4_t s = vaddq_f32(Load<false>(src), Load<false>(src + Q));
                            float32x4_t h0 = vminq_f32(vmulq_f32(Broadcast<0>(s), n), _02);
                            float32x4_t h1 = vminq_f32(vmulq_f32(Broadcast<1>(s), n), _02);
                            float32x4_t h2 = vminq_f32(vmulq_f32(Broadcast<2>(s), n), _02);
                            float32x4_t h3 = vminq_f32(vmulq_f32(Broadcast<3>(s), n), _02);
                            Store<false>(dst, vmulq_f32(_05, Hadd(Hadd(h0, h1), Hadd(h2, h3))));
                            dst += 4;
                            src += 4;
                        }
                        {
                            float32x4_t s = vdupq_n_f32(src[0] + src[Q]);
                            float32x4_t h = vminq_f32(vmulq_f32(s, n), _02);
                            h = vmulq_f32(_05, h);
                            *dst++ = ExtractSum32f(h);
                        }
                        Store<false>(dst, vmulq_f32(t, _02357));
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
            template <int add, bool end> SIMD_INLINE void Set(float * dst, const float32x4_t & value, const float32x4_t & mask)
            {
                Store<false>(dst, value);
            }

            template <> SIMD_INLINE void Set<1, false>(float * dst, const float32x4_t & value, const float32x4_t & mask)
            {
                Store<false>(dst, vaddq_f32(Load<false>(dst), value));
            }

            template <> SIMD_INLINE void Set<1, true>(float * dst, const float32x4_t & value, const float32x4_t & mask)
            {
                Store<false>(dst, vaddq_f32(Load<false>(dst), And(value, mask)));
            }
        }

        class HogSeparableFilter
        {
            typedef Array<float> Array32f;
            typedef Array<float32x4_t> Array128f;

            size_t _w, _h, _s;
            Array32f _buffer;
            Array128f _filter;

            void Init(size_t w, size_t h, size_t rs, size_t cs)
            {
                _w = w - rs + 1;
                _s = AlignHi(_w, F);
                _h = h - cs + 1;
                _buffer.Resize(_s*h);
            }

            template <bool align> void FilterRows(const float * src, const float32x4_t * filter, size_t size, float * dst)
            {
                float32x4_t sum = vdupq_n_f32(0);
                for (size_t i = 0; i < size; ++i)
                    sum = vmlaq_f32(sum, Load<false>(src + i), filter[i]);
                Store<align>(dst, sum);
            }

            void FilterRows(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = vdupq_n_f32(filter[i]);

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

            template <int add, bool end> void FilterCols(const float * src, size_t stride, const float32x4_t * filter, size_t size, float * dst, const float32x4_t & mask)
            {
                float32x4_t sum = vdupq_n_f32(0);
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = vmlaq_f32(sum, Load<!end>(src), filter[i]);
                HogSeparableFilter_Detail::Set<add, end>(dst, sum, mask);
            }

            template <int add> void FilterCols(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = vdupq_n_f32(filter[i]);

                size_t alignedWidth = AlignLo(width, F);
                float32x4_t tailMask = RightNotZero32f(width - alignedWidth);

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

            void Run(const float * src, size_t srcStride, size_t width, size_t height,
                const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add)
            {
                Init(width, height, rowSize, colSize);

                FilterRows(src, srcStride, _w, height, rowFilter, rowSize, _buffer.data, _s);

                if (add)
                    FilterCols<1>(_buffer.data, _s, _w, _h, colFilter, colSize, dst, dstStride);
                else
                    FilterCols<0>(_buffer.data, _s, _w, _h, colFilter, colSize, dst, dstStride);
            }
        };

        void HogFilterSeparable(const float * src, size_t srcStride, size_t width, size_t height,
            const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add)
        {
            assert(width >= F + rowSize - 1 && height >= colSize - 1);

            HogSeparableFilter filter;
            filter.Run(src, srcStride, width, height, rowFilter, rowSize, colFilter, colSize, dst, dstStride, add);
        }
    }
#endif// SIMD_NEON_ENABLE
}
