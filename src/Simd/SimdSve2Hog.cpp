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
#include "Simd/SimdArray.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        namespace
        {
            struct Buffer
            {
                const int size;
                float* cos;
                float* sin;
                int* index;
                float* value;

                Buffer(size_t width, size_t quantization)
                    : size((int)quantization / 2)
                {
                    _p = Allocate(width * (sizeof(int) + sizeof(float)) + sizeof(float) * 2 * size);
                    index = (int*)_p;
                    value = (float*)index + width;
                    cos = value + width;
                    sin = cos + size;
                    for (int i = 0; i < size; ++i)
                    {
                        cos[i] = (float)::cos(i * M_PI / size);
                        sin[i] = (float)::sin(i * M_PI / size);
                    }
                }

                ~Buffer()
                {
                    Free(_p);
                }

            private:
                void* _p;
            };
        }

        SIMD_INLINE svbool_t HogTailMask(size_t col, size_t end, const svuint32_t& offsets)
        {
            const svbool_t mask = svptrue_b32();
            return svcmplt_n_u32(mask, svadd_n_u32_x(mask, offsets, (uint32_t)col), (uint32_t)end);
        }

        SIMD_INLINE void HogDirectionHistograms32(const svint32_t& dx, const svint32_t& dy, Buffer& buffer, size_t col, const svuint32_t& offsets, const svbool_t& mask)
        {
            svfloat32_t _dx = svcvt_f32_s32_x(mask, dx);
            svfloat32_t _dy = svcvt_f32_s32_x(mask, dy);
            svfloat32_t bestDot = svdup_n_f32(0.0f);
            svint32_t bestIndex = svdup_n_s32(0);
            for (int i = 0; i < buffer.size; ++i)
            {
                svfloat32_t dot = svmla_f32_x(mask, svmul_n_f32_x(mask, _dx, buffer.cos[i]), _dy, svdup_n_f32(buffer.sin[i]));
                svbool_t positive = svcmpgt_f32(mask, dot, bestDot);
                bestDot = svmax_f32_x(mask, dot, bestDot);
                bestIndex = svsel_s32(positive, svdup_n_s32(i), bestIndex);

                dot = svneg_f32_x(mask, dot);
                svbool_t negative = svcmpgt_f32(mask, dot, bestDot);
                bestDot = svmax_f32_x(mask, dot, bestDot);
                bestIndex = svsel_s32(negative, svdup_n_s32(buffer.size + i), bestIndex);
            }
            svst1_scatter_u32index_s32(mask, buffer.index + col, offsets, bestIndex);
            svst1_scatter_u32index_f32(mask, buffer.value + col, offsets, svsqrt_f32_x(mask, svmla_f32_x(mask, svmul_f32_x(mask, _dx, _dx), _dy, _dy)));
        }

        SIMD_INLINE void HogDirectionHistograms16(const svint16_t& dx, const svint16_t& dy, Buffer& buffer, size_t col, const svuint32_t& loOffsets, const svuint32_t& hiOffsets, const svbool_t& maskLo, const svbool_t& maskHi)
        {
            HogDirectionHistograms32(svmovlb_s32(dx), svmovlb_s32(dy), buffer, col, loOffsets, maskLo);
            HogDirectionHistograms32(svmovlt_s32(dx), svmovlt_s32(dy), buffer, col, hiOffsets, maskHi);
        }

        SIMD_INLINE svint16_t HogDifferenceLo(const svuint8_t& a, const svuint8_t& b)
        {
            svbool_t mask = svptrue_b16();
            return svsub_s16_x(mask, svreinterpret_s16_u16(svmovlb_u16(a)), svreinterpret_s16_u16(svmovlb_u16(b)));
        }

        SIMD_INLINE svint16_t HogDifferenceHi(const svuint8_t& a, const svuint8_t& b)
        {
            svbool_t mask = svptrue_b16();
            return svsub_s16_x(mask, svreinterpret_s16_u16(svmovlt_u16(a)), svreinterpret_s16_u16(svmovlt_u16(b)));
        }

        SIMD_INLINE void HogDirectionHistograms(const uint8_t* src, size_t stride, Buffer& buffer, size_t col, size_t end)
        {
            const uint8_t* s = src + col;
            svbool_t mask = svwhilelt_b8(col, end);
            const svuint32_t offsets0 = svindex_u32(0, 4);
            const svuint32_t offsets1 = svindex_u32(2, 4);
            const svuint32_t offsets2 = svindex_u32(1, 4);
            const svuint32_t offsets3 = svindex_u32(3, 4);
            svuint8_t t = svld1_u8(mask, s - stride);
            svuint8_t l = svld1_u8(mask, s - 1);
            svuint8_t r = svld1_u8(mask, s + 1);
            svuint8_t b = svld1_u8(mask, s + stride);
            HogDirectionHistograms16(HogDifferenceLo(r, l), HogDifferenceLo(b, t), buffer, col, offsets0, offsets1, HogTailMask(col, end, offsets0), HogTailMask(col, end, offsets1));
            HogDirectionHistograms16(HogDifferenceHi(r, l), HogDifferenceHi(b, t), buffer, col, offsets2, offsets3, HogTailMask(col, end, offsets2), HogTailMask(col, end, offsets3));
        }

        void HogDirectionHistograms(const uint8_t* src, size_t stride, size_t width, size_t height,
            size_t cellX, size_t cellY, size_t quantization, float* histograms)
        {
            assert(width % cellX == 0 && height % cellY == 0 && quantization % 2 == 0);

            Buffer buffer(width, quantization);

            memset(histograms, 0, quantization * (width / cellX) * (height / cellY) * sizeof(float));

            size_t A = svcntb(), end = width - 1;
            for (size_t row = 1; row < height - 1; ++row)
            {
                const uint8_t* s = src + stride * row;
                for (size_t col = 1; col < end; col += A)
                    HogDirectionHistograms(s, stride, buffer, col, end);
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

            float _cos[5], _sin[5];
            float _kx[C][4], _ky[C][4];

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
                    _cos[i] = (float)::cos(i * M_PI / Q);
                    _sin[i] = (float)::sin(i * M_PI / Q);
                }
                for (int i = 0; i < (int)C; ++i)
                {
                    float k0 = float((15 - i * 2) / 16.0f);
                    float k1 = 1.0f - k0;
                    _kx[i][0] = k0;
                    _kx[i][1] = k1;
                    _kx[i][2] = k0;
                    _kx[i][3] = k1;
                    _ky[i][0] = k0;
                    _ky[i][1] = k0;
                    _ky[i][2] = k1;
                    _ky[i][3] = k1;
                }

                _index.Resize(w);
                _value.Resize(w);
                _buffer.Resize((_sx + 1) * 4 * Q2);
                _histogram.Resize((_sx + 2) * (_sy + 2) * Q2);
                _norm.Resize((_sx + 2) * (_sy + 2));
            }

            SIMD_INLINE void GetHistogram32(const svint32_t& dx, const svint32_t& dy, size_t col, const svuint32_t& offsets, const svbool_t& mask)
            {
                svfloat32_t _0 = svdup_n_f32(0.0f);
                svfloat32_t _dx = svcvt_f32_s32_x(mask, dx);
                svfloat32_t _dy = svcvt_f32_s32_x(mask, dy);
                svfloat32_t adx = svabs_f32_x(mask, _dx);
                svfloat32_t ady = svabs_f32_x(mask, _dy);
                svfloat32_t bestDot = _0;
                svint32_t bestIndex = svdup_n_s32(0);
                for (int i = 0; i < 5; ++i)
                {
                    svfloat32_t dot = svmla_f32_x(mask, svmul_n_f32_x(mask, adx, _cos[i]), ady, svdup_n_f32(_sin[i]));
                    svbool_t positive = svcmpgt_f32(mask, dot, bestDot);
                    bestDot = svmax_f32_x(mask, dot, bestDot);
                    bestIndex = svsel_s32(positive, svdup_n_s32(i), bestIndex);
                }

                svbool_t negativeDx = svcmplt_n_f32(mask, _dx, 0.0f);
                bestIndex = svsel_s32(negativeDx, svsub_s32_x(mask, svdup_n_s32((int)Q), bestIndex), bestIndex);

                svbool_t negativeDy = svcmplt_n_f32(mask, _dy, 0.0f);
                svint32_t correction = svsel_s32(svcmpeq_n_f32(mask, adx, 0.0f), svdup_n_s32(1), svdup_n_s32(0));
                bestIndex = svsel_s32(negativeDy, svsub_s32_x(mask, svdup_n_s32((int)Q2), svadd_s32_x(mask, bestIndex, correction)), bestIndex);
                bestIndex = svsel_s32(svcmpeq_n_s32(mask, bestIndex, (int)Q2), svdup_n_s32(0), bestIndex);

                svst1_scatter_u32index_s32(mask, _index.data + col, offsets, bestIndex);
                svst1_scatter_u32index_f32(mask, _value.data + col, offsets, svsqrt_f32_x(mask, svmla_f32_x(mask, svmul_f32_x(mask, adx, adx), ady, ady)));
            }

            SIMD_INLINE void GetHistogram16(const svint16_t& dx, const svint16_t& dy, size_t col, const svuint32_t& loOffsets, const svuint32_t& hiOffsets, const svbool_t& maskLo, const svbool_t& maskHi)
            {
                GetHistogram32(svmovlb_s32(dx), svmovlb_s32(dy), col, loOffsets, maskLo);
                GetHistogram32(svmovlt_s32(dx), svmovlt_s32(dy), col, hiOffsets, maskHi);
            }

            SIMD_INLINE void GetHistogram(const uint8_t* src, size_t stride, size_t col, size_t end)
            {
                const uint8_t* s = src + col;
                svbool_t mask = svwhilelt_b8(col, end);
                const svuint32_t offsets0 = svindex_u32(0, 4);
                const svuint32_t offsets1 = svindex_u32(2, 4);
                const svuint32_t offsets2 = svindex_u32(1, 4);
                const svuint32_t offsets3 = svindex_u32(3, 4);
                svuint8_t t = svld1_u8(mask, s - stride);
                svuint8_t l = svld1_u8(mask, s - 1);
                svuint8_t r = svld1_u8(mask, s + 1);
                svuint8_t b = svld1_u8(mask, s + stride);
                GetHistogram16(HogDifferenceLo(r, l), HogDifferenceLo(b, t), col, offsets0, offsets1, HogTailMask(col, end, offsets0), HogTailMask(col, end, offsets1));
                GetHistogram16(HogDifferenceHi(r, l), HogDifferenceHi(b, t), col, offsets2, offsets3, HogTailMask(col, end, offsets2), HogTailMask(col, end, offsets3));
            }

            void AddRowToBuffer(const uint8_t* src, size_t stride, size_t row, size_t width)
            {
                const uint8_t* s = src + stride * row;
                size_t A = svcntb(), end = width - 1;
                for (size_t col = 1; col < end; col += A)
                    GetHistogram(s, stride, col, end);

                float* buffer = _buffer.data;
                svbool_t mask = svwhilelt_b32(size_t(0), size_t(4));
                svfloat32_t ky = svld1_f32(mask, _ky[(row + 4) & 7]);
                for (size_t col = 1, n = C, i = 5; col < width - 1; i = 0, n = Simd::Min<size_t>(C, width - col - 1))
                {
                    for (; i < n; ++i, ++col)
                    {
                        int index = _index[col];
                        float* dst = buffer + index * 4;
                        svfloat32_t weight = svmul_f32_x(mask, ky, svld1_f32(mask, _kx[i]));
                        svst1_f32(mask, dst, svmla_f32_x(mask, svld1_f32(mask, dst), weight, svdup_n_f32(_value[col])));
                    }
                    buffer += 4 * Q2;
                }
            }

            void AddToHistogram(size_t row, size_t width, size_t height)
            {
                const float* src = _buffer.data;
                float* h0 = _histogram.data + row * _hs * Q2;
                float* h1 = h0 + _hs * Q2;
                size_t F = svcntw();
                for (size_t cell = 0; cell <= width; ++cell)
                {
                    for (size_t i = 0; i < Q2; i += F)
                    {
                        svbool_t mask = svwhilelt_b32(i, Q2);
                        svuint32_t offsets = svadd_n_u32_x(mask, svmul_n_u32_x(mask, svindex_u32(0, 1), 4), (uint32_t)(4 * i));
                        svfloat32_t s0 = svld1_gather_u32index_f32(mask, src, offsets);
                        svfloat32_t s1 = svld1_gather_u32index_f32(mask, src + 1, offsets);
                        svfloat32_t s2 = svld1_gather_u32index_f32(mask, src + 2, offsets);
                        svfloat32_t s3 = svld1_gather_u32index_f32(mask, src + 3, offsets);
                        svst1_f32(mask, h0 + i, svadd_f32_x(mask, svld1_f32(mask, h0 + i), s0));
                        svst1_f32(mask, h0 + Q2 + i, svadd_f32_x(mask, svld1_f32(mask, h0 + Q2 + i), s1));
                        svst1_f32(mask, h1 + i, svadd_f32_x(mask, svld1_f32(mask, h1 + i), s2));
                        svst1_f32(mask, h1 + Q2 + i, svadd_f32_x(mask, svld1_f32(mask, h1 + Q2 + i), s3));
                    }
                    h0 += Q2;
                    h1 += Q2;
                    src += 4 * Q2;
                }
                _buffer.Clear();
            }

            void EstimateHistogram(const uint8_t* src, size_t stride, size_t width, size_t height)
            {
                _histogram.Clear();

                _buffer.Clear();
                for (size_t row = 1; row < 4; ++row)
                    AddRowToBuffer(src, stride, row, width);
                AddToHistogram(0, _sx, _sy);
                for (size_t row = 4, cell = 1; row < height - 4; ++row)
                {
                    AddRowToBuffer(src, stride, row, width);
                    if ((row & 7) == 3)
                        AddToHistogram(cell++, _sx, _sy);
                }
                for (size_t row = height - 4; row < height - 1; ++row)
                    AddRowToBuffer(src, stride, row, width);
                AddToHistogram(_sy, _sx, _sy);
            }

            SIMD_INLINE float GetNorm(const float* src)
            {
                svfloat32_t norm = svdup_n_f32(0.0f);
                size_t F = svcntw();
                for (size_t i = 0; i < Q; i += F)
                {
                    svbool_t mask = svwhilelt_b32(i, Q);
                    svfloat32_t sum = svadd_f32_x(mask, svld1_f32(mask, src + i), svld1_f32(mask, src + i + Q));
                    norm = svmla_f32_m(mask, norm, sum, sum);
                }
                return svaddv_f32(svptrue_b32(), norm);
            }

            void EstimateNorm()
            {
                _norm.Clear();
                for (size_t y = 0; y < _sy; y++)
                {
                    const float* h = _histogram.data + ((y + 1) * _hs + 1) * Q2;
                    float* n = _norm.data + (y + 1) * _hs + 1;
                    for (size_t x = 0; x < _sx; x++)
                        n[x] = GetNorm(h + x * Q2);
                }
            }

            void ExtractFeatures(float* features)
            {
                svbool_t mask = svwhilelt_b32(size_t(0), size_t(4));
                svfloat32_t _02 = svdup_n_f32(0.2f);
                svfloat32_t _05 = svdup_n_f32(0.5f);
                svfloat32_t _02357 = svdup_n_f32(0.2357f);
                svfloat32_t _1 = svdup_n_f32(1.0f);
                svfloat32_t eps = svdup_n_f32(0.0001f);
                float ns[4];
                for (size_t y = 0; y < _sy; y++)
                {
                    const float* ph = _histogram.data + ((y + 1) * _hs + 1) * Q2;
                    for (size_t x = 0; x < _sx; x++)
                    {
                        float* dst = features + (y * _sx + x) * 31;

                        float* p0 = _norm.data + y * _hs + x;
                        float* p1 = p0 + _hs;
                        float* p2 = p1 + _hs;

                        ns[0] = p1[1] + p1[2] + p2[1] + p2[2];
                        ns[1] = p0[1] + p0[2] + p1[1] + p1[2];
                        ns[2] = p1[0] + p1[1] + p2[0] + p2[1];
                        ns[3] = p0[0] + p0[1] + p1[0] + p1[1];
                        svfloat32_t n = svdiv_f32_x(mask, _1, svsqrt_f32_x(mask, svadd_f32_x(mask, svld1_f32(mask, ns), eps)));
                        svfloat32_t t = svdup_n_f32(0.0f);

                        const float* src = ph + x * Q2;
                        for (int o = 0; o < (int)Q2; ++o)
                        {
                            svfloat32_t h = svmin_f32_x(mask, svmul_f32_x(mask, svdup_n_f32(src[o]), n), _02);
                            dst[o] = 0.5f * svaddv_f32(mask, h);
                            t = svadd_f32_x(mask, t, h);
                        }
                        dst += Q2;

                        for (int o = 0; o < (int)Q; ++o)
                        {
                            svfloat32_t h = svmin_f32_x(mask, svmul_f32_x(mask, svdup_n_f32(src[o] + src[o + Q]), n), _02);
                            dst[o] = 0.5f * svaddv_f32(mask, h);
                        }
                        dst += Q;

                        svst1_f32(mask, dst, svmul_f32_x(mask, t, _02357));
                    }
                }
            }

        public:
            void Run(const uint8_t* src, size_t stride, size_t width, size_t height, float* features)
            {
                Init(width, height);

                EstimateHistogram(src, stride, width, height);

                EstimateNorm();

                ExtractFeatures(features);
            }
        };

        void HogExtractFeatures(const uint8_t* src, size_t stride, size_t width, size_t height, float* features)
        {
            assert(width % 8 == 0 && height % 8 == 0 && width >= 16 && height >= 16);

            HogFeatureExtractor extractor;
            extractor.Run(src, stride, width, height, features);
        }

        SIMD_INLINE void HogDeinterleave(const float* src, const svuint32_t& offsets, const svbool_t& mask, float* dst)
        {
            svst1_f32(mask, dst, svld1_gather_u32index_f32(mask, src, offsets));
        }

        void HogDeinterleave(const float* src, size_t srcStride, size_t width, size_t height, size_t count, float** dst, size_t dstStride)
        {
            assert(width >= svcntw());

            size_t F = svcntw(), alignedWidth = AlignLo(width, F);
            const svbool_t body = svptrue_b32();
            const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), (uint32_t)count);

            for (size_t row = 0; row < height; ++row)
            {
                size_t rowOffset = row * dstStride, col = 0;
                for (; col < alignedWidth; col += F)
                {
                    const float* s = src + count * col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < count; ++i)
                        HogDeinterleave(s + i, offsets, body, dst[i] + offset);
                }
                if (col < width)
                {
                    const float* s = src + count * col;
                    size_t offset = rowOffset + col;
                    svbool_t tail = svwhilelt_b32(col, width);
                    for (size_t i = 0; i < count; ++i)
                        HogDeinterleave(s + i, offsets, tail, dst[i] + offset);
                }
                src += srcStride;
            }
        }
    }
#endif
}
