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
                for (size_t x = 0; x < _fx; x += 3, src0 += 3, src1 += 3, src2 += 3, dst += 3*F)
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
                    const float * src = hf + x*FQ;
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
                    src = hf + x*FQ;
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
                    if(cell == 4)
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
    }
#endif// SIMD_SSE41_ENABLE
}


