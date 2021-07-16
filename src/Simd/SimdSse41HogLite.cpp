/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        const __m128i K8_KX4 = SIMD_MM_SETR_EPI8(1, 3, 5, 7, 7, 5, 3, 1, 1, 3, 5, 7, 7, 5, 3, 1);
        const __m128i K8_KX8 = SIMD_MM_SETR_EPI8(1, 3, 5, 7, 9, 11, 13, 15, 15, 13, 11, 9, 7, 5, 3, 1);

        template <size_t cell> class HogLiteFeatureExtractor
        {
            static const size_t FQ = 8;
            static const size_t HQ = FQ / 2;
            static const size_t DQ = FQ * 2;

            typedef Array<uint8_t> Bytes;
            typedef Array<int> Ints;
            typedef Array<float> Floats;

            size_t _hx, _fx, _w, _aw;
            Bytes _value, _index;
            Ints _hi[2];
            Floats _hf[2], _nf[4], _nb;
            int _k0[cell], _k1[cell];
            __m128i _kx4, _kx8;
            __m128 _k, _02, _05, _02357, _eps;

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
                    _hi[i].Resize((_hx + 4)*FQ, true);
                    _hf[i].Resize(_hx*FQ);
                }
                for (size_t i = 0; i < 4; ++i)
                    _nf[i].Resize(_hx + DF);
                _nb.Resize(_hx * 4);
                _k = _mm_set1_ps(1.0f / Simd::Square(cell * 2));
                _02 = _mm_set1_ps(0.2f);
                _05 = _mm_set1_ps(0.5f);
                _02357 = _mm_set1_ps(0.2357f);
                _eps = _mm_set1_ps(0.0001f);
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

            static SIMD_INLINE void UpdateIntegerHistogram4x4(uint8_t * value, uint8_t * index, const __m128i & ky0, const __m128i & ky1, int * h0, int * h1)
            {
                __m128i val = Load<false>((__m128i*)value);
                __m128i idx = Load<false>((__m128i*)index);
                __m128i cur0 = K_ZERO;
                __m128i cur1 = K8_01;
                __m128i dirs[4];
                for (size_t i = 0; i < 4; ++i)
                {
                    __m128i dir0 = _mm_maddubs_epi16(_mm_and_si128(_mm_cmpeq_epi8(idx, cur0), val), K8_KX4);
                    __m128i dir1 = _mm_maddubs_epi16(_mm_and_si128(_mm_cmpeq_epi8(idx, cur1), val), K8_KX4);
                    dirs[i] = _mm_hadd_epi16(dir0, dir1);
                    cur0 = _mm_add_epi8(cur0, K8_02);
                    cur1 = _mm_add_epi8(cur1, K8_02);
                }
                __m128i hx0 = Shuffle32i<0x88>(dirs[0], dirs[1]);
                __m128i hx1 = Shuffle32i<0x88>(dirs[2], dirs[3]);
                __m128i hx2 = Shuffle32i<0xDD>(dirs[0], dirs[1]);
                __m128i hx3 = Shuffle32i<0xDD>(dirs[2], dirs[3]);
                Store<true>((__m128i*)h0 + 0, _mm_add_epi32(Load<true>((__m128i*)h0 + 0), _mm_madd_epi16(hx0, ky0)));
                Store<true>((__m128i*)h0 + 1, _mm_add_epi32(Load<true>((__m128i*)h0 + 1), _mm_madd_epi16(hx1, ky0)));
                Store<true>((__m128i*)h0 + 4, _mm_add_epi32(Load<true>((__m128i*)h0 + 4), _mm_madd_epi16(hx2, ky0)));
                Store<true>((__m128i*)h0 + 5, _mm_add_epi32(Load<true>((__m128i*)h0 + 5), _mm_madd_epi16(hx3, ky0)));
                Store<true>((__m128i*)h1 + 0, _mm_add_epi32(Load<true>((__m128i*)h1 + 0), _mm_madd_epi16(hx0, ky1)));
                Store<true>((__m128i*)h1 + 1, _mm_add_epi32(Load<true>((__m128i*)h1 + 1), _mm_madd_epi16(hx1, ky1)));
                Store<true>((__m128i*)h1 + 4, _mm_add_epi32(Load<true>((__m128i*)h1 + 4), _mm_madd_epi16(hx2, ky1)));
                Store<true>((__m128i*)h1 + 5, _mm_add_epi32(Load<true>((__m128i*)h1 + 5), _mm_madd_epi16(hx3, ky1)));
            }

            SIMD_INLINE void UpdateIntegerHistogram4x4(size_t rowI, size_t rowF)
            {
                int * h0 = _hi[(rowI + 0) & 1].data;
                int * h1 = _hi[(rowI + 1) & 1].data;
                uint8_t * value = _value.data + A - cell;
                uint8_t * index = _index.data + A - cell;
                __m128i ky0 = _mm_set1_epi16((short)_k0[rowF]);
                __m128i ky1 = _mm_set1_epi16((short)_k1[rowF]);
                for (size_t col = 0; col <= _w;)
                {
                    UpdateIntegerHistogram4x4(value + col, index + col, ky0, ky1, h0, h1);
                    col += cell;
                    h0 += FQ;
                    h1 += FQ;
                    UpdateIntegerHistogram4x4(value + col, index + col, ky0, ky1, h0, h1);
                    col += 3 * cell;
                    h0 += 3 * FQ;
                    h1 += 3 * FQ;
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
                        __m128i dir0 = _mm_maddubs_epi16(_mm_and_si128(_mm_cmpeq_epi8(idx, cur0), val), K8_KX8);
                        __m128i dir1 = _mm_maddubs_epi16(_mm_and_si128(_mm_cmpeq_epi8(idx, cur1), val), K8_KX8);
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
                Ints & hi = _hi[rowI & 1];
                Floats & hf = _hf[rowI & 1];
                Floats & nf = _nf[rowI & 3];

                for (size_t i = 0; i < hf.size; i += DF)
                {
                    Store<true>(hf.data + i + 0, _mm_mul_ps(_k, _mm_cvtepi32_ps(Load<true>((__m128i*)(hi.data + i + 0)))));
                    Store<true>(hf.data + i + F, _mm_mul_ps(_k, _mm_cvtepi32_ps(Load<true>((__m128i*)(hi.data + i + F)))));
                }
                hi.Clear();

                const float * h = hf.data;
                for (size_t x = 0; x < _hx; ++x, h += FQ)
                {
                    __m128 h0 = Load<true>(h + 00);
                    __m128 h1 = Load<true>(h + HQ);
                    __m128 sum = _mm_add_ps(h0, h1);
                    _mm_store_ss(nf.data + x, _mm_dp_ps(sum, sum, 0xF1));
                }
            }

            SIMD_INLINE void BlockNorm(size_t rowI)
            {
                const float * src0 = _nf[(rowI - 2) & 3].data;
                const float * src1 = _nf[(rowI - 1) & 3].data;
                const float * src2 = _nf[(rowI - 0) & 3].data;
                float * dst = _nb.data;
                for (size_t x = 0; x < _fx; x += 3, src0 += 3, src1 += 3, src2 += 3, dst += 3 * F)
                {
                    __m128 s00 = Load<false>(src0 + 0);
                    __m128 s01 = Load<false>(src0 + 1);
                    __m128 s10 = Load<false>(src1 + 0);
                    __m128 s11 = Load<false>(src1 + 1);
                    __m128 s20 = Load<false>(src2 + 0);
                    __m128 s21 = Load<false>(src2 + 1);
                    __m128 v00 = _mm_add_ps(s00, s10);
                    __m128 v01 = _mm_add_ps(s01, s11);
                    __m128 v10 = _mm_add_ps(s10, s20);
                    __m128 v11 = _mm_add_ps(s11, s21);
                    __m128 h0 = _mm_hadd_ps(v00, v01);
                    __m128 h1 = _mm_hadd_ps(v10, v11);
                    __m128 d0 = _mm_shuffle_ps(h0, h1, 0x88);
                    __m128 d1 = _mm_shuffle_ps(h0, h1, 0x99);
                    __m128 d2 = _mm_shuffle_ps(h0, h1, 0xDD);
                    Store<true>(dst + 0 * F, Shuffle32f<0x27>(d0));
                    Store<true>(dst + 1 * F, Shuffle32f<0x72>(d1));
                    Store<true>(dst + 2 * F, Shuffle32f<0x27>(d2));
                }
            }

            SIMD_INLINE void SetFeatures(size_t rowI, float * dst)
            {
                const float * hf = _hf[(rowI - 1) & 1].data + FQ;
                const float * nb = _nb.data;
                for (size_t x = 0; x < _fx; ++x, nb += 4)
                {
                    __m128 n = _mm_rsqrt_ps(_mm_add_ps(_mm_load_ps(nb), _eps));
                    __m128 t = _mm_setzero_ps();
                    const float * src = hf + x * FQ;
                    for (int o = 0; o < FQ; o += 4)
                    {
                        __m128 s = _mm_loadu_ps(src);
                        __m128 h0 = _mm_min_ps(_mm_mul_ps(Broadcast<0>(s), n), _02);
                        __m128 h1 = _mm_min_ps(_mm_mul_ps(Broadcast<1>(s), n), _02);
                        __m128 h2 = _mm_min_ps(_mm_mul_ps(Broadcast<2>(s), n), _02);
                        __m128 h3 = _mm_min_ps(_mm_mul_ps(Broadcast<3>(s), n), _02);
                        t = _mm_add_ps(t, _mm_add_ps(_mm_add_ps(h0, h1), _mm_add_ps(h2, h3)));
                        _mm_storeu_ps(dst, _mm_mul_ps(_05, _mm_hadd_ps(_mm_hadd_ps(h0, h1), _mm_hadd_ps(h2, h3))));
                        dst += F;
                        src += F;
                    }
                    src = hf + x * FQ;
                    __m128 s = _mm_add_ps(_mm_loadu_ps(src), _mm_loadu_ps(src + HQ));
                    __m128 h0 = _mm_min_ps(_mm_mul_ps(Broadcast<0>(s), n), _02);
                    __m128 h1 = _mm_min_ps(_mm_mul_ps(Broadcast<1>(s), n), _02);
                    __m128 h2 = _mm_min_ps(_mm_mul_ps(Broadcast<2>(s), n), _02);
                    __m128 h3 = _mm_min_ps(_mm_mul_ps(Broadcast<3>(s), n), _02);
                    _mm_storeu_ps(dst, _mm_mul_ps(_05, _mm_hadd_ps(_mm_hadd_ps(h0, h1), _mm_hadd_ps(h2, h3))));
                    dst += 4;
                    _mm_storeu_ps(dst, _mm_mul_ps(t, _02357));
                    dst += 4;
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

                for (size_t row = 0; row < height; ++row)
                {
                    SetIndexAndValue(src, srcStride);
                    size_t rowI = row / cell;
                    size_t rowF = row & (cell - 1);
                    if (cell == 4)
                        UpdateIntegerHistogram4x4(rowI, rowF);
                    else
                        UpdateIntegerHistogram8x8(rowI, rowF);
                    if (rowF == cell - 1)
                    {
                        UpdateFloatHistogram(rowI);
                        if (rowI >= 2)
                        {
                            BlockNorm(rowI);
                            SetFeatures(rowI, features);
                            features += featuresStride;
                        }
                    }
                    src += srcStride;
                }
                size_t rowI = height / cell;
                UpdateFloatHistogram(rowI);
                BlockNorm(rowI);
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
            template<bool align> SIMD_INLINE void ProductSum1x1(const float * src, const float * filter, __m128 & sum)
            {
                __m128 _src = Load<align>(src);
                __m128 _filter = Load<align>(filter);
                sum = _mm_add_ps(sum, _mm_mul_ps(_src, _filter));
            }

            template<bool align, size_t step> SIMD_INLINE void ProductSum1x4(const float * src, const float * filter, __m128 * sums)
            {
                __m128 _filter = Load<align>(filter);
                sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(Load<align>(src + 0 * step), _filter));
                sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(Load<align>(src + 1 * step), _filter));
                sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(Load<align>(src + 2 * step), _filter));
                sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(Load<align>(src + 3 * step), _filter));
            }

            template <bool align, size_t featureSize> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterWidth, size_t filterHeight, float * dst, size_t dstStride)
            {
                size_t filterStride = featureSize * filterWidth;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                        const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                        {
                            size_t filterCol = 0;
                            for (; filterCol < filterStride; filterCol += F)
                                ProductSum1x4<align, featureSize>(pSrc + filterCol, pFilter + filterCol, sums);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        _mm_storeu_ps(dst + dstCol, _mm_hadd_ps(_mm_hadd_ps(sums[0], sums[1]), _mm_hadd_ps(sums[2], sums[3])));
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        __m128 sum = _mm_setzero_ps();
                        const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                        {
                            for (size_t filterCol = 0; filterCol < filterStride; filterCol += F)
                                ProductSum1x1<align>(pSrc + filterCol, pFilter + filterCol, sum);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        dst[dstCol] = ExtractSum(sum);
                    }
                    dst += dstStride;
                }
            }

            template <bool align, size_t featureSize> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                size_t filterStride = featureSize * filterWidth;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                __m128 _min = _mm_set1_ps(-FLT_MAX);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        __m128 _mask = _mm_castsi128_ps(_mm_loadu_si128((__m128i*)(mask + dstCol)));
                        if (TestZ(_mask))
                            _mm_storeu_ps(dst + dstCol, _min);
                        else
                        {
                            __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                            const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                size_t filterCol = 0;
                                for (; filterCol < filterStride; filterCol += F)
                                    ProductSum1x4<align, featureSize>(pSrc + filterCol, pFilter + filterCol, sums);
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            _mm_storeu_ps(dst + dstCol, _mm_blendv_ps(_min, Extract4Sums(sums), _mask));
                        }
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        if (mask[dstCol])
                        {
                            __m128 sum = _mm_setzero_ps();
                            const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                for (size_t filterCol = 0; filterCol < filterStride; filterCol += F)
                                    ProductSum1x1<align>(pSrc + filterCol, pFilter + filterCol, sum);
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            dst[dstCol] = ExtractSum(sum);
                        }
                        else
                            dst[dstCol] = -FLT_MAX;
                    }
                    dst += dstStride;
                    mask += maskStride;
                }
            }

            template <bool align> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, float * dst, size_t dstStride)
            {
                if (featureSize == 16)
                    Filter<align, 16>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, dst, dstStride);
                else
                    Filter<align, 8>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, dst, dstStride);
            }

            template <bool align> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                if (featureSize == 16)
                    Filter<align, 16>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                else
                    Filter<align, 8>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
            }

        public:

            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                assert(featureSize == 8 || featureSize == 16);
                assert(srcWidth >= filterWidth && srcHeight >= filterHeight);

                size_t dstWidth = srcWidth - filterWidth + 1;
                size_t dstHeight = srcHeight - filterHeight + 1;

                if (mask)
                {
                    if (Aligned(src) && Aligned(srcStride) && Aligned(filter))
                        Filter<true>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                    else
                        Filter<false>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                }
                else
                {
                    if (Aligned(src) && Aligned(srcStride) && Aligned(filter))
                        Filter<true>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, dst, dstStride);
                    else
                        Filter<false>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, dst, dstStride);
                }
            }
        };

        void HogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
        {
            HogLiteFeatureFilter featureFilter;
            featureFilter.Run(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
        }

        namespace HogLiteFeatureResizerDetail
        {
            template <int size> struct Feature
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const __m128 k[2][2], float * dst);
            };

            template <> struct Feature<8>
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const __m128 k[2][2], float * dst)
                {
                    Store<align>(dst + 0 * F, _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(Load<align>(src0 + 0 * F), k[0][0]), _mm_mul_ps(Load<align>(src0 + 2 * F), k[0][1])),
                        _mm_add_ps(_mm_mul_ps(Load<align>(src1 + 0 * F), k[1][0]), _mm_mul_ps(Load<align>(src1 + 2 * F), k[1][1]))));
                    Store<align>(dst + 1 * F, _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(Load<align>(src0 + 1 * F), k[0][0]), _mm_mul_ps(Load<align>(src0 + 3 * F), k[0][1])),
                        _mm_add_ps(_mm_mul_ps(Load<align>(src1 + 1 * F), k[1][0]), _mm_mul_ps(Load<align>(src1 + 3 * F), k[1][1]))));
                }
            };

            template <> struct Feature<16>
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const __m128 k[2][2], float * dst)
                {
                    Store<align>(dst + 0 * F, _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(Load<align>(src0 + 0 * F), k[0][0]), _mm_mul_ps(Load<align>(src0 + 4 * F), k[0][1])),
                        _mm_add_ps(_mm_mul_ps(Load<align>(src1 + 0 * F), k[1][0]), _mm_mul_ps(Load<align>(src1 + 4 * F), k[1][1]))));
                    Store<align>(dst + 1 * F, _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(Load<align>(src0 + 1 * F), k[0][0]), _mm_mul_ps(Load<align>(src0 + 5 * F), k[0][1])),
                        _mm_add_ps(_mm_mul_ps(Load<align>(src1 + 1 * F), k[1][0]), _mm_mul_ps(Load<align>(src1 + 5 * F), k[1][1]))));
                    Store<align>(dst + 2 * F, _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(Load<align>(src0 + 2 * F), k[0][0]), _mm_mul_ps(Load<align>(src0 + 6 * F), k[0][1])),
                        _mm_add_ps(_mm_mul_ps(Load<align>(src1 + 2 * F), k[1][0]), _mm_mul_ps(Load<align>(src1 + 6 * F), k[1][1]))));
                    Store<align>(dst + 3 * F, _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(Load<align>(src0 + 3 * F), k[0][0]), _mm_mul_ps(Load<align>(src0 + 7 * F), k[0][1])),
                        _mm_add_ps(_mm_mul_ps(Load<align>(src1 + 3 * F), k[1][0]), _mm_mul_ps(Load<align>(src1 + 7 * F), k[1][1]))));
                }
            };
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

            template<bool align, size_t featureSize> void Resize(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                __m128 _1 = _mm_set1_ps(1.0f);
                for (size_t rowDst = 0; rowDst < dstHeight; ++rowDst)
                {
                    __m128 ky1 = _mm_set1_ps(_ky[rowDst]);
                    __m128 ky0 = _mm_sub_ps(_1, ky1);
                    const float * pSrc = src + _iy[rowDst];
                    float * pDst = dst + rowDst * dstStride;
                    for (size_t colDst = 0; colDst < dstWidth; ++colDst, pDst += featureSize)
                    {
                        __m128 kx1 = _mm_set1_ps(_kx[colDst]);
                        __m128 kx0 = _mm_sub_ps(_1, kx1);
                        __m128 k[2][2];
                        k[0][0] = _mm_mul_ps(ky0, kx0);
                        k[0][1] = _mm_mul_ps(ky0, kx1);
                        k[1][0] = _mm_mul_ps(ky1, kx0);
                        k[1][1] = _mm_mul_ps(ky1, kx1);
                        const float * pSrc0 = pSrc + _ix[colDst];
                        const float * pSrc1 = pSrc0 + srcStride;
                        HogLiteFeatureResizerDetail::Feature<featureSize>:: template Interpolate<align>(pSrc0, pSrc1, k, pDst);
                    }
                }
            }

            template<bool align> void Resize(const float * src, size_t srcStride, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                if (featureSize == 8)
                    Resize<align, 8>(src, srcStride, dst, dstStride, dstWidth, dstHeight);
                else
                    Resize<align, 16>(src, srcStride, dst, dstStride, dstWidth, dstHeight);
            }

        public:
            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                assert(featureSize == 8 || featureSize == 16);

                if (srcWidth == dstWidth && srcHeight == dstHeight)
                {
                    size_t size = sizeof(float)*srcWidth*featureSize;
                    for (size_t row = 0; row < dstHeight; ++row)
                        memcpy(dst + row * dstStride, src + row * srcStride, size);
                    return;
                }

                InitIndexWeight(srcWidth, dstWidth, featureSize, _ix, _kx);
                InitIndexWeight(srcHeight, dstHeight, srcStride, _iy, _ky);

                if (Aligned(src) && Aligned(dst))
                    Resize<true>(src, srcStride, featureSize, dst, dstStride, dstWidth, dstHeight);
                else
                    Resize<false>(src, srcStride, featureSize, dst, dstStride, dstWidth, dstHeight);
            }
        };

        void HogLiteResizeFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
        {
            HogLiteFeatureResizer featureResizer;
            featureResizer.Run(src, srcStride, srcWidth, srcHeight, featureSize, dst, dstStride, dstWidth, dstHeight);
        }

        template<bool align> void HogLiteCompressFeatures(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const float * s = src;
                float * d = dst;
                for (size_t col = 0; col < width; ++col)
                {
                    const float * p = pca;
                    for (size_t i = 0; i < 8; i += 4, p += 64)
                    {
                        __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                        for (size_t j = 0; j < 16; j += F)
                        {
                            __m128 _s = Load<align>(s + j);
                            sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(_s, Load<align>(p + j + 00)));
                            sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(_s, Load<align>(p + j + 16)));
                            sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(_s, Load<align>(p + j + 32)));
                            sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(_s, Load<align>(p + j + 48)));
                        }
                        Store<align>(d + i, _mm_hadd_ps(_mm_hadd_ps(sums[0], sums[1]), _mm_hadd_ps(sums[2], sums[3])));
                    }
                    s += 16;
                    d += 8;
                }
                src += srcStride;
                dst += dstStride;
            }

        }

        void HogLiteCompressFeatures(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(pca) && Aligned(dst))
                HogLiteCompressFeatures<true>(src, srcStride, width, height, pca, dst, dstStride);
            else
                HogLiteCompressFeatures<false>(src, srcStride, width, height, pca, dst, dstStride);
        }

        class HogLiteSeparableFilter
        {
            size_t _dstWidth, _dstHeight, _dstStride;
            Array32f _buffer;
            Array128f _filter;

            void Init(size_t srcWidth, size_t srcHeight, size_t hSize, size_t vSize)
            {
                _dstWidth = srcWidth - hSize + 1;
                _dstStride = AlignHi(_dstWidth, F);
                _dstHeight = srcHeight - vSize + 1;
                _buffer.Resize(_dstStride*srcHeight);
            }

            template<bool align> static SIMD_INLINE void FilterHx1(const float * src, const float * filter, __m128 & sum)
            {
                __m128 _src = Load<align>(src);
                __m128 _filter = Load<align>(filter);
                sum = _mm_add_ps(sum, _mm_mul_ps(_src, _filter));
            }

            template<bool align, size_t step> static SIMD_INLINE void FilterHx4(const float * src, const float * filter, __m128 * sums)
            {
                __m128 _filter = Load<align>(filter);
                sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(Load<align>(src + 0 * step), _filter));
                sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(Load<align>(src + 1 * step), _filter));
                sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(Load<align>(src + 2 * step), _filter));
                sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(Load<align>(src + 3 * step), _filter));
            }

            template <bool align, size_t step> void FilterH(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                size_t alignedWidth = AlignLo(width, 4);
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedWidth; col += 4)
                    {
                        __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += F)
                            FilterHx4<align, step>(s + i, filter + i, sums);
                        Store<true>(dst + col, Extract4Sums(sums));
                    }
                    for (; col < width; ++col)
                    {
                        __m128 sum = _mm_setzero_ps();
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += F)
                            FilterHx1<align>(s + i, filter + i, sum);
                        dst[col] = ExtractSum(sum);
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <bool align> void FilterH(const float * src, size_t srcStride, size_t width, size_t height, size_t step, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                if (step == 16)
                    FilterH<align, 16>(src, srcStride, width, height, filter, size, dst, dstStride);
                else
                    FilterH<align, 8>(src, srcStride, width, height, filter, size, dst, dstStride);
            }

            template <bool srcAlign, bool dstAlign, UpdateType update, bool masked> static SIMD_INLINE void FilterV(const float * src, size_t stride, const __m128 * filter, size_t size, float * dst, const __m128 & mask)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = _mm_add_ps(sum, _mm_mul_ps(Load<srcAlign>(src), filter[i]));
                Update<update, dstAlign>(dst, Masked<masked && update != UpdateSet>(sum, mask));
            }

            template <UpdateType update, bool align> void FilterV(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm_set1_ps(filter[i]);

                size_t alignedWidth = AlignLo(width, F);
                __m128 tailMask = RightNotZero32f(width - alignedWidth);

                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += F)
                        FilterV<true, align, update, false>(src + col, srcStride, _filter.data, size, dst + col, tailMask);
                    if (alignedWidth != width)
                        FilterV<false, false, update, true>(src + width - F, srcStride, _filter.data, size, dst + width - F, tailMask);
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <UpdateType update> void FilterV(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                if (Aligned(dst) && Aligned(dstStride))
                    FilterV<update, true>(src, srcStride, width, height, filter, size, dst, dstStride);
                else
                    FilterV<update, false>(src, srcStride, width, height, filter, size, dst, dstStride);
            }

        public:

            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add)
            {
                assert(featureSize == 8 || featureSize == 16);
                assert(srcWidth >= hSize && srcHeight >= vSize);

                Init(srcWidth, srcHeight, hSize, vSize);

                if (Aligned(src) && Aligned(srcStride) && Aligned(hFilter))
                    FilterH<true>(src, srcStride, _dstWidth, srcHeight, featureSize, hFilter, hSize*featureSize, _buffer.data, _dstStride);
                else
                    FilterH<false>(src, srcStride, _dstWidth, srcHeight, featureSize, hFilter, hSize*featureSize, _buffer.data, _dstStride);

                if (add)
                    FilterV<UpdateAdd>(_buffer.data, _dstStride, _dstWidth, _dstHeight, vFilter, vSize, dst, dstStride);
                else
                    FilterV<UpdateSet>(_buffer.data, _dstStride, _dstWidth, _dstHeight, vFilter, vSize, dst, dstStride);
            }
        };

        void HogLiteFilterSeparable(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add)
        {
            HogLiteSeparableFilter filter;
            filter.Run(src, srcStride, srcWidth, srcHeight, featureSize, hFilter, hSize, vFilter, vSize, dst, dstStride, add);
        }

        uint8_t g_tzcnt_table[256] =
        {
            8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            7, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
            4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        };

        void HogLiteFindMax7x7(const float * a, size_t aStride, const float * b, size_t bStride, size_t height, float * pValue, size_t * pCol, size_t * pRow)
        {
            __m128 sums[7][2];
            __m128 max = _mm_set1_ps(-FLT_MAX);
            for (size_t row = 0; row < height; ++row)
            {
                sums[row][0] = _mm_add_ps(Load<false>(a + 0), Load<false>(b + 0));
                sums[row][1] = _mm_add_ps(Load<false>(a + 3), Load<false>(b + 3));
                max = _mm_max_ps(max, _mm_max_ps(sums[row][0], sums[row][1]));
                a += aStride;
                b += bStride;
            }
            max = _mm_max_ps(Alignr<1>(max, max), max);
            max = _mm_max_ps(Alignr<2>(max, max), max);
            _mm_store_ss(pValue, max);
            for (size_t row = 0; row < height; ++row)
            {
                __m128i m03 = _mm_castps_si128(_mm_cmpeq_ps(max, sums[row][0]));
                __m128i m36 = _mm_castps_si128(_mm_cmpeq_ps(max, sums[row][1]));
                __m128i m06 = _mm_packs_epi32(m03, _mm_srli_si128(m36, 4));
                if (!_mm_testz_si128(m06, K_INV_ZERO))
                {
                    int mask = _mm_movemask_epi8(_mm_packs_epi16(m06, _mm_setzero_si128()));
                    *pRow = row;
                    *pCol = g_tzcnt_table[mask];
                    break;
                }
            }
        }

        template<bool align> SIMD_INLINE void Fill7x7(uint32_t * dst, size_t stride)
        {
            for (size_t row = 0; row < 7; ++row)
            {
                Store<align>((__m128i*)(dst + 0), Sse2::K_INV_ZERO);
                Store<align>((__m128i*)(dst + 3), Sse2::K_INV_ZERO);
                dst += stride;
            }
        }

        template <size_t scale> void HogLiteCreateMask7x7(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, uint32_t * dst, size_t dstStride)
        {
            size_t dstStartEnd = 7 - scale;
            size_t dstRowSize = (srcWidth*scale + 7 - scale) * sizeof(uint32_t);
            for (size_t dstRow = 0; dstRow < dstStartEnd; ++dstRow)
                memset(dst + dstRow * dstStride, 0, dstRowSize);

            size_t alignedSrcWidth = AlignLo(srcWidth, F);
            __m128 _threshold = _mm_set1_ps(*threshold);
            for (size_t srcRow = 0; srcRow < srcHeight; ++srcRow)
            {
                for (size_t dstRow = 0; dstRow < scale; ++dstRow)
                    memset(dst + (dstStartEnd + dstRow)*dstStride, 0, dstRowSize);

                size_t srcCol = 0;
                for (; srcCol < alignedSrcWidth; srcCol += F)
                {
                    int mask = _mm_movemask_ps(_mm_cmpgt_ps(Load<false>(src + srcCol), _threshold));
                    if (mask)
                    {
                        uint32_t * pDst = dst + srcCol * scale;
                        if (mask & 1)
                            Fill7x7<false>(pDst + 0 * scale, dstStride);
                        if (mask & 2)
                            Fill7x7<false>(pDst + 1 * scale, dstStride);
                        if (mask & 4)
                            Fill7x7<false>(pDst + 2 * scale, dstStride);
                        if (mask & 8)
                            Fill7x7<false>(pDst + 3 * scale, dstStride);
                    }
                }
                for (; srcCol < srcWidth; ++srcCol)
                {
                    if (src[srcCol] > *threshold)
                        Fill7x7<false>(dst + srcCol * scale, dstStride);
                }
                src += srcStride;
                dst += dstStride * scale;
            }
        }

        void HogLiteCreateMask(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride)
        {
            if (scale == 1 && size == 7)
                HogLiteCreateMask7x7<1>(src, srcStride, srcWidth, srcHeight, threshold, dst, dstStride);
            else if (scale == 2 && size == 7)
                HogLiteCreateMask7x7<2>(src, srcStride, srcWidth, srcHeight, threshold, dst, dstStride);
            else
                Base::HogLiteCreateMask(src, srcStride, srcWidth, srcHeight, threshold, scale, size, dst, dstStride);
        }
    }
#endif// SIMD_SSE41_ENABLE
}


