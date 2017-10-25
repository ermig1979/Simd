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
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        const __m256i K8_KX4 = SIMD_MM256_SETR_EPI8(
            1, 3, 5, 7, 7, 5, 3, 1, 1, 3, 5, 7, 7, 5, 3, 1,
            1, 3, 5, 7, 7, 5, 3, 1, 1, 3, 5, 7, 7, 5, 3, 1);
        const __m256i K8_KX8 = SIMD_MM256_SETR_EPI8(
            1, 3, 5, 7, 9, 11, 13, 15, 15, 13, 11, 9, 7, 5, 3, 1,
            1, 3, 5, 7, 9, 11, 13, 15, 15, 13, 11, 9, 7, 5, 3, 1);

        template <size_t cell> class HogLiteFeatureExtractor
        {
            static const size_t FQ = 8;
            static const size_t HQ = FQ / 2;
            static const size_t DQ = FQ * 2;
            static const size_t QQ = FQ * 4;

            typedef Array<uint8_t> Bytes;
            typedef Array<int> Ints;
            typedef Array<float> Floats;

            size_t _hx, _fx, _w, _aw;
            Bytes _value, _index;
            Ints _hi[2];
            Floats _hf[2], _nf[4], _nb;
            int _k0[cell], _k1[cell];
            __m256 _k, _02, _05, _02357, _eps;

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
                    _hi[i].Resize((_hx + 8)*FQ, true);
                    _hf[i].Resize(_hx*FQ);
                }
                for (size_t i = 0; i < 4; ++i)
                    _nf[i].Resize(_hx + F);
                _nb.Resize((_hx + 6)* 4 );
                _k = _mm256_set1_ps(1.0f / Simd::Square(cell * 2));
                _02 = _mm256_set1_ps(0.2f);
                _05 = _mm256_set1_ps(0.5f);
                _02357 = _mm256_set1_ps(0.2357f);
                _eps = _mm256_set1_ps(0.0001f);
            }

            template<bool align> static SIMD_INLINE void SetIndexAndValue(const uint8_t * src, size_t stride, uint8_t * value, uint8_t * index)
            {
                __m256i y0 = Load<false>((__m256i*)(src - stride));
                __m256i y1 = Load<false>((__m256i*)(src + stride));
                __m256i x0 = Load<false>((__m256i*)(src - 1));
                __m256i x1 = Load<false>((__m256i*)(src + 1));

                __m256i ady = AbsDifferenceU8(y0, y1);
                __m256i adx = AbsDifferenceU8(x0, x1);

                __m256i max = _mm256_max_epu8(ady, adx);
                __m256i min = _mm256_min_epu8(ady, adx);
                __m256i val = _mm256_adds_epu8(max, _mm256_avg_epu8(min, K_ZERO));
                Store<align>((__m256i*)value, val);

                __m256i idx = _mm256_blendv_epi8(K8_01, K_ZERO, Compare8u<SimdCompareGreater>(adx, ady));
                idx = _mm256_blendv_epi8(_mm256_sub_epi8(K8_03, idx), idx, Compare8u<SimdCompareGreater>(x1, x0));
                idx = _mm256_blendv_epi8(_mm256_sub_epi8(K8_07, idx), idx, Compare8u<SimdCompareGreater>(y1, y0));
                Store<align>((__m256i*)index, idx);
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

            static SIMD_INLINE void UpdateIntegerHistogram4x4(uint8_t * value, uint8_t * index, const __m256i & ky0, const __m256i & ky1, int * h0, int * h1)
            {
                __m256i val = Load<false>((__m256i*)value);
                __m256i idx = Load<false>((__m256i*)index);
                __m256i cur0 = K_ZERO;
                __m256i cur1 = K8_01;
                __m256i dirs[4];
                for (size_t i = 0; i < 4; ++i)
                {
                    __m256i dir0 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_cmpeq_epi8(idx, cur0), val), K8_KX4);
                    __m256i dir1 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_cmpeq_epi8(idx, cur1), val), K8_KX4);
                    dirs[i] = _mm256_hadd_epi16(dir0, dir1);
                    cur0 = _mm256_add_epi8(cur0, K8_02);
                    cur1 = _mm256_add_epi8(cur1, K8_02);
                }
                __m256i hx0 = Shuffle32i<0x88>(dirs[0], dirs[1]);
                __m256i hx1 = Shuffle32i<0x88>(dirs[2], dirs[3]);
                __m256i hx2 = Shuffle32i<0xDD>(dirs[0], dirs[1]);
                __m256i hx3 = Shuffle32i<0xDD>(dirs[2], dirs[3]);
                __m256i hx0p = _mm256_permute2x128_si256(hx0, hx1, 0x20);
                __m256i hx1p = _mm256_permute2x128_si256(hx0, hx1, 0x31);
                __m256i hx2p = _mm256_permute2x128_si256(hx2, hx3, 0x20);
                __m256i hx3p = _mm256_permute2x128_si256(hx2, hx3, 0x31);
                Store<true>((__m256i*)h0 + 0, _mm256_add_epi32(Load<true>((__m256i*)h0 + 0), _mm256_madd_epi16(hx0p, ky0)));
                Store<true>((__m256i*)h0 + 2, _mm256_add_epi32(Load<true>((__m256i*)h0 + 2), _mm256_madd_epi16(hx2p, ky0)));
                Store<true>((__m256i*)h0 + 4, _mm256_add_epi32(Load<true>((__m256i*)h0 + 4), _mm256_madd_epi16(hx1p, ky0)));
                Store<true>((__m256i*)h0 + 6, _mm256_add_epi32(Load<true>((__m256i*)h0 + 6), _mm256_madd_epi16(hx3p, ky0)));
                Store<true>((__m256i*)h1 + 0, _mm256_add_epi32(Load<true>((__m256i*)h1 + 0), _mm256_madd_epi16(hx0p, ky1)));
                Store<true>((__m256i*)h1 + 2, _mm256_add_epi32(Load<true>((__m256i*)h1 + 2), _mm256_madd_epi16(hx2p, ky1)));
                Store<true>((__m256i*)h1 + 4, _mm256_add_epi32(Load<true>((__m256i*)h1 + 4), _mm256_madd_epi16(hx1p, ky1)));
                Store<true>((__m256i*)h1 + 6, _mm256_add_epi32(Load<true>((__m256i*)h1 + 6), _mm256_madd_epi16(hx3p, ky1)));
            }

            SIMD_INLINE void UpdateIntegerHistogram4x4(size_t rowI, size_t rowF)
            {
                int * h0 = _hi[(rowI + 0) & 1].data;
                int * h1 = _hi[(rowI + 1) & 1].data;
                uint8_t * value = _value.data + A - cell;
                uint8_t * index = _index.data + A - cell;
                __m256i ky0 = _mm256_set1_epi16((short)_k0[rowF]);
                __m256i ky1 = _mm256_set1_epi16((short)_k1[rowF]);
                for (size_t col = 0; col <= _w;)
                {
                    UpdateIntegerHistogram4x4(value + col, index + col, ky0, ky1, h0, h1);
                    col += cell;
                    h0 += FQ;
                    h1 += FQ;
                    UpdateIntegerHistogram4x4(value + col, index + col, ky0, ky1, h0, h1);
                    col += 7 * cell;
                    h0 += 7 * FQ;
                    h1 += 7 * FQ;
                }
            }

            static SIMD_INLINE void UpdateIntegerHistogram8x8(uint8_t * value, uint8_t * index, const __m256i & ky0, const __m256i & ky1, int * h0, int * h1)
            {
                __m256i val = Load<false>((__m256i*)value);
                __m256i idx = Load<false>((__m256i*)index);
                __m256i cur0 = K_ZERO;
                __m256i cur1 = K8_01;
                __m256i dirs[4];
                for (size_t i = 0; i < 4; ++i)
                {
                    __m256i dir0 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_cmpeq_epi8(idx, cur0), val), K8_KX8);
                    __m256i dir1 = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_cmpeq_epi8(idx, cur1), val), K8_KX8);
                    dirs[i] = _mm256_hadd_epi16(dir0, dir1);
                    cur0 = _mm256_add_epi8(cur0, K8_02);
                    cur1 = _mm256_add_epi8(cur1, K8_02);
                }
                dirs[0] = _mm256_hadd_epi16(dirs[0], dirs[1]);
                dirs[1] = _mm256_hadd_epi16(dirs[2], dirs[3]);
                __m256i hx0 = _mm256_permute2x128_si256(dirs[0], dirs[1], 0x20);
                __m256i hx1 = _mm256_permute2x128_si256(dirs[0], dirs[1], 0x31);
                Store<true>((__m256i*)h0 + 0, _mm256_add_epi32(Load<true>((__m256i*)h0 + 0), _mm256_madd_epi16(hx0, ky0)));
                Store<true>((__m256i*)h0 + 2, _mm256_add_epi32(Load<true>((__m256i*)h0 + 2), _mm256_madd_epi16(hx1, ky0)));
                Store<true>((__m256i*)h1 + 0, _mm256_add_epi32(Load<true>((__m256i*)h1 + 0), _mm256_madd_epi16(hx0, ky1)));
                Store<true>((__m256i*)h1 + 2, _mm256_add_epi32(Load<true>((__m256i*)h1 + 2), _mm256_madd_epi16(hx1, ky1)));
            }

            SIMD_INLINE void UpdateIntegerHistogram8x8(size_t rowI, size_t rowF)
            {
                int * h0 = _hi[(rowI + 0) & 1].data;
                int * h1 = _hi[(rowI + 1) & 1].data;
                uint8_t * value = _value.data + A - cell;
                uint8_t * index = _index.data + A - cell;
                __m256i ky0 = _mm256_set1_epi16((short)_k0[rowF]);
                __m256i ky1 = _mm256_set1_epi16((short)_k1[rowF]);
                for (size_t col = 0; col <= _w;)
                {
                    UpdateIntegerHistogram8x8(value + col, index + col, ky0, ky1, h0, h1);
                    col += cell;
                    h0 += FQ;
                    h1 += FQ;
                    UpdateIntegerHistogram8x8(value + col, index + col, ky0, ky1, h0, h1);
                    col += 3 * cell;
                    h0 += 3 * FQ;
                    h1 += 3 * FQ;
                }
            }

            SIMD_INLINE void UpdateFloatHistogram(size_t rowI)
            {
                Ints & hi = _hi[rowI & 1];
                Floats & hf = _hf[rowI & 1];
                Floats & nf = _nf[rowI & 3];

                size_t alignedSize = AlignLo(hf.size, DF), i = 0;
                for (; i < alignedSize; i += DF)
                {
                    Avx::Store<true>(hf.data + i + 0, _mm256_mul_ps(_k, _mm256_cvtepi32_ps(Load<true>((__m256i*)(hi.data + i + 0)))));
                    Avx::Store<true>(hf.data + i + F, _mm256_mul_ps(_k, _mm256_cvtepi32_ps(Load<true>((__m256i*)(hi.data + i + F)))));
                }
                for (; i < hf.size; i += F)
                    Avx::Store<true>(hf.data + i, _mm256_mul_ps(_k, _mm256_cvtepi32_ps(Load<true>((__m256i*)(hi.data + i)))));
                hi.Clear();

                const float * h = hf.data;
                size_t ahx = AlignLo(_hx, 4), x = 0;
                for (; x < ahx; x += 4, h += QQ)
                {
                    __m256 h01 = Load<true>(h + 0 * FQ);
                    __m256 h23 = Load<true>(h + 1 * FQ);
                    __m256 h45 = Load<true>(h + 2 * FQ);
                    __m256 h67 = Load<true>(h + 3 * FQ);
                    __m256 s01 = _mm256_add_ps(_mm256_permute2f128_ps(h01, h23, 0x20), _mm256_permute2f128_ps(h01, h23, 0x31));
                    __m256 n01 = Permute4x64<0x88>(_mm256_dp_ps(s01, s01, 0xF1));
                    __m256 s23 = _mm256_add_ps(_mm256_permute2f128_ps(h45, h67, 0x20), _mm256_permute2f128_ps(h45, h67, 0x31));
                    __m256 n23 = Permute4x64<0x88>(_mm256_dp_ps(s23, s23, 0xF1));
                    _mm_storeu_ps(nf.data + x, _mm_shuffle_ps(_mm256_castps256_ps128(n01), _mm256_castps256_ps128(n23), 0x88));
                }
                for (; x < _hx; ++x, h += FQ)
                {
                    __m128 h0 = Sse::Load<true>(h + 00);
                    __m128 h1 = Sse::Load<true>(h + HQ);
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
                for (size_t x = 0; x < _fx; x += 3, src0 += 3, src1 += 3, src2 += 3, dst += 3* Sse::F)
                {
                    __m128 s00 = Sse::Load<false>(src0 + 0);
                    __m128 s01 = Sse::Load<false>(src0 + 1);
                    __m128 s10 = Sse::Load<false>(src1 + 0);
                    __m128 s11 = Sse::Load<false>(src1 + 1);
                    __m128 s20 = Sse::Load<false>(src2 + 0);
                    __m128 s21 = Sse::Load<false>(src2 + 1);
                    __m128 v00 = _mm_add_ps(s00, s10);
                    __m128 v01 = _mm_add_ps(s01, s11);
                    __m128 v10 = _mm_add_ps(s10, s20);
                    __m128 v11 = _mm_add_ps(s11, s21);
                    __m128 h0 = _mm_hadd_ps(v00, v01);
                    __m128 h1 = _mm_hadd_ps(v10, v11);
                    __m128 d0 = _mm_shuffle_ps(h0, h1, 0x88);
                    __m128 d1 = _mm_shuffle_ps(h0, h1, 0x99);
                    __m128 d2 = _mm_shuffle_ps(h0, h1, 0xDD);
                    Sse::Store<true>(dst + 0 * Sse::F, Sse2::Shuffle32f<0x27>(d0));
                    Sse::Store<true>(dst + 1 * Sse::F, Sse2::Shuffle32f<0x72>(d1));
                    Sse::Store<true>(dst + 2 * Sse::F, Sse2::Shuffle32f<0x27>(d2));
                }
            }

            SIMD_INLINE __m256 Features07(const __m256 & n, const __m256 & s, __m256 & t)
            {
                __m256 h0 = _mm256_min_ps(_mm256_mul_ps(Broadcast<0>(s), n), _02);
                __m256 h1 = _mm256_min_ps(_mm256_mul_ps(Broadcast<1>(s), n), _02);
                __m256 h2 = _mm256_min_ps(_mm256_mul_ps(Broadcast<2>(s), n), _02);
                __m256 h3 = _mm256_min_ps(_mm256_mul_ps(Broadcast<3>(s), n), _02);
                t = _mm256_add_ps(t, _mm256_add_ps(_mm256_add_ps(h0, h1), _mm256_add_ps(h2, h3)));
                return _mm256_mul_ps(_05, _mm256_hadd_ps(_mm256_hadd_ps(h0, h1), _mm256_hadd_ps(h2, h3)));
            }

            SIMD_INLINE __m256 Features8B(const __m256 & n, const __m256 & s)
            {
                __m256 h0 = _mm256_min_ps(_mm256_mul_ps(Broadcast<0>(s), n), _02);
                __m256 h1 = _mm256_min_ps(_mm256_mul_ps(Broadcast<1>(s), n), _02);
                __m256 h2 = _mm256_min_ps(_mm256_mul_ps(Broadcast<2>(s), n), _02);
                __m256 h3 = _mm256_min_ps(_mm256_mul_ps(Broadcast<3>(s), n), _02);
                return _mm256_mul_ps(_05, _mm256_hadd_ps(_mm256_hadd_ps(h0, h1), _mm256_hadd_ps(h2, h3)));
            }

            SIMD_INLINE void SetFeatures(size_t rowI, float * dst)
            {
                const float * hf = _hf[(rowI - 1) & 1].data + FQ;
                const float * nb = _nb.data;
                size_t x = 0, afx = AlignLo(_fx, 2);
                for (; x < afx; x += 2, nb += 8, dst += QQ)
                {
                    __m256 n = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_load_ps(nb), _eps));
                    __m256 t = _mm256_setzero_ps();
                    __m256 f[4];
                    const float * src = hf + x*FQ;
                    __m256 s0 = Avx::Load<false>(src + 0 * HQ, src + 2 * HQ);
                    __m256 s1 = Avx::Load<false>(src + 1 * HQ, src + 3 * HQ);
                    f[0] = Features07(n, s0, t);
                    f[1] = Features07(n, s1, t);
                    f[2] = Features8B(n, _mm256_add_ps(s0, s1));
                    f[3] = _mm256_mul_ps(t, _02357);
                    Avx::Store<false>(dst + 0 * F, _mm256_permute2f128_ps(f[0], f[1], 0x20));
                    Avx::Store<false>(dst + 1 * F, _mm256_permute2f128_ps(f[2], f[3], 0x20));
                    Avx::Store<false>(dst + 2 * F, _mm256_permute2f128_ps(f[0], f[1], 0x31));
                    Avx::Store<false>(dst + 3 * F, _mm256_permute2f128_ps(f[2], f[3], 0x31));
                }
                for (; x < _fx; ++x, nb += 4)
                {
                    __m128 n = _mm_rsqrt_ps(_mm_add_ps(_mm_load_ps(nb), _mm256_castps256_ps128(_eps)));
                    __m128 t = _mm_setzero_ps();
                    const float * src = hf + x*FQ;
                    for (int o = 0; o < FQ; o += 4)
                    {
                        __m128 s = _mm_loadu_ps(src);
                        __m128 h0 = _mm_min_ps(_mm_mul_ps(Sse2::Broadcast<0>(s), n), _mm256_castps256_ps128(_02));
                        __m128 h1 = _mm_min_ps(_mm_mul_ps(Sse2::Broadcast<1>(s), n), _mm256_castps256_ps128(_02));
                        __m128 h2 = _mm_min_ps(_mm_mul_ps(Sse2::Broadcast<2>(s), n), _mm256_castps256_ps128(_02));
                        __m128 h3 = _mm_min_ps(_mm_mul_ps(Sse2::Broadcast<3>(s), n), _mm256_castps256_ps128(_02));
                        t = _mm_add_ps(t, _mm_add_ps(_mm_add_ps(h0, h1), _mm_add_ps(h2, h3)));
                        _mm_storeu_ps(dst, _mm_mul_ps(_mm256_castps256_ps128(_05), _mm_hadd_ps(_mm_hadd_ps(h0, h1), _mm_hadd_ps(h2, h3))));
                        dst += Sse2::F;
                        src += Sse2::F;
                    }
                    src = hf + x*FQ;
                    __m128 s = _mm_add_ps(_mm_loadu_ps(src), _mm_loadu_ps(src + HQ));
                    __m128 h0 = _mm_min_ps(_mm_mul_ps(Sse2::Broadcast<0>(s), n), _mm256_castps256_ps128(_02));
                    __m128 h1 = _mm_min_ps(_mm_mul_ps(Sse2::Broadcast<1>(s), n), _mm256_castps256_ps128(_02));
                    __m128 h2 = _mm_min_ps(_mm_mul_ps(Sse2::Broadcast<2>(s), n), _mm256_castps256_ps128(_02));
                    __m128 h3 = _mm_min_ps(_mm_mul_ps(Sse2::Broadcast<3>(s), n), _mm256_castps256_ps128(_02));
                    _mm_storeu_ps(dst, _mm_mul_ps(_mm256_castps256_ps128(_05), _mm_hadd_ps(_mm_hadd_ps(h0, h1), _mm_hadd_ps(h2, h3))));
                    dst += 4;
                    _mm_storeu_ps(dst, _mm_mul_ps(t, _mm256_castps256_ps128(_02357)));
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

        namespace HogLiteFeatureFilterDetail
        {
            template <int size> struct Feature
            {
                template <bool align> static SIMD_INLINE void Sum4x4(const float * src, const float * filter, __m256 * sums);
            };

            template <> struct Feature<8>
            {
                template <bool align> static SIMD_INLINE void Sum4x4(const float * src, const float * filter, __m256 * sums)
                {
                    __m256 filter0 = Load<align>(filter + 0 * F);
                    __m256 src0 = Load<align>(src + 0 * F);
                    __m256 src1 = Load<align>(src + 1 * F);
                    __m256 src2 = Load<align>(src + 2 * F);
                    __m256 src3 = Load<align>(src + 3 * F);
                    sums[0] = _mm256_fmadd_ps(src0, filter0, sums[0]);
                    sums[1] = _mm256_fmadd_ps(src1, filter0, sums[1]);
                    sums[2] = _mm256_fmadd_ps(src2, filter0, sums[2]);
                    sums[3] = _mm256_fmadd_ps(src3, filter0, sums[3]);
                    __m256 filter1 = Load<align>(filter + 1 * F);
                    __m256 src4 = Load<align>(src + 4 * F);
                    sums[0] = _mm256_fmadd_ps(src1, filter1, sums[0]);
                    sums[1] = _mm256_fmadd_ps(src2, filter1, sums[1]);
                    sums[2] = _mm256_fmadd_ps(src3, filter1, sums[2]);
                    sums[3] = _mm256_fmadd_ps(src4, filter1, sums[3]);
                    __m256 filter2 = Load<align>(filter + 2 * F);
                    __m256 src5 = Load<align>(src + 5 * F);
                    sums[0] = _mm256_fmadd_ps(src2, filter2, sums[0]);
                    sums[1] = _mm256_fmadd_ps(src3, filter2, sums[1]);
                    sums[2] = _mm256_fmadd_ps(src4, filter2, sums[2]);
                    sums[3] = _mm256_fmadd_ps(src5, filter2, sums[3]);
                    __m256 filter3 = Load<align>(filter + 3 * F);
                    __m256 src6 = Load<align>(src + 6 * F);
                    sums[0] = _mm256_fmadd_ps(src3, filter3, sums[0]);
                    sums[1] = _mm256_fmadd_ps(src4, filter3, sums[1]);
                    sums[2] = _mm256_fmadd_ps(src5, filter3, sums[2]);
                    sums[3] = _mm256_fmadd_ps(src6, filter3, sums[3]);
                }
            };

            template <> struct Feature<16>
            {
                template <bool align> static SIMD_INLINE void Sum4x4(const float * src, const float * filter, __m256 * sums)
                {
                    __m256 filter0 = Load<align>(filter + 0 * F);
                    __m256 src0 = Load<align>(src + 0 * F);
                    __m256 src2 = Load<align>(src + 2 * F);
                    __m256 src4 = Load<align>(src + 4 * F);
                    __m256 src6 = Load<align>(src + 6 * F);
                    sums[0] = _mm256_fmadd_ps(src0, filter0, sums[0]);
                    sums[1] = _mm256_fmadd_ps(src2, filter0, sums[1]);
                    sums[2] = _mm256_fmadd_ps(src4, filter0, sums[2]);
                    sums[3] = _mm256_fmadd_ps(src6, filter0, sums[3]);
                    __m256 filter2 = Load<align>(filter + 2 * F);
                    __m256 src8 = Load<align>(src + 8 * F);
                    sums[0] = _mm256_fmadd_ps(src2, filter2, sums[0]);
                    sums[1] = _mm256_fmadd_ps(src4, filter2, sums[1]);
                    sums[2] = _mm256_fmadd_ps(src6, filter2, sums[2]);
                    sums[3] = _mm256_fmadd_ps(src8, filter2, sums[3]);
                    __m256 filter1 = Load<align>(filter + 1 * F);
                    __m256 src1 = Load<align>(src + 1 * F);
                    __m256 src3 = Load<align>(src + 3 * F);
                    __m256 src5 = Load<align>(src + 5 * F);
                    __m256 src7 = Load<align>(src + 7 * F);
                    sums[0] = _mm256_fmadd_ps(src1, filter1, sums[0]);
                    sums[1] = _mm256_fmadd_ps(src3, filter1, sums[1]);
                    sums[2] = _mm256_fmadd_ps(src5, filter1, sums[2]);
                    sums[3] = _mm256_fmadd_ps(src7, filter1, sums[3]);
                    __m256 filter3 = Load<align>(filter + 3 * F);
                    __m256 src9 = Load<align>(src + 9 * F);
                    sums[0] = _mm256_fmadd_ps(src3, filter3, sums[0]);
                    sums[1] = _mm256_fmadd_ps(src5, filter3, sums[1]);
                    sums[2] = _mm256_fmadd_ps(src7, filter3, sums[2]);
                    sums[3] = _mm256_fmadd_ps(src9, filter3, sums[3]);
                }
            };
        }

        class HogLiteFeatureFilter
        {
            template<bool align> SIMD_INLINE void ProductSum1x1(const float * src, const float * filter, __m256 & sum)
            {
                __m256 _src = Avx::Load<align>(src);
                __m256 _filter = Avx::Load<align>(filter);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(_src, _filter));
            }

            template<bool align, size_t step> SIMD_INLINE void ProductSum1x4(const float * src, const float * filter, __m256 * sums)
            {
                __m256 _filter = Avx::Load<align>(filter);
                sums[0] = _mm256_fmadd_ps(Avx::Load<align>(src + 0 * step), _filter, sums[0]);
                sums[1] = _mm256_fmadd_ps(Avx::Load<align>(src + 1 * step), _filter, sums[1]);
                sums[2] = _mm256_fmadd_ps(Avx::Load<align>(src + 2 * step), _filter, sums[2]);
                sums[3] = _mm256_fmadd_ps(Avx::Load<align>(src + 3 * step), _filter, sums[3]);
            }

            template <bool align, size_t featureSize> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterSize, float * dst, size_t dstStride)
            {
                size_t filterStride = featureSize*filterSize;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                size_t alignedFilterStride = AlignLo(filterStride, QF);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                        const float * pSrc = src + dstRow*srcStride + dstCol*featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterSize; ++filterRow)
                        {
                            size_t filterCol = 0;
                            for (; filterCol < alignedFilterStride; filterCol += QF)
                                HogLiteFeatureFilterDetail::Feature<featureSize>:: template Sum4x4<align>(pSrc + filterCol, pFilter + filterCol, sums);
                            for (; filterCol < filterStride; filterCol += F)
                                ProductSum1x4<align, featureSize>(pSrc + filterCol, pFilter + filterCol, sums);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        __m256 sum = _mm256_hadd_ps(_mm256_hadd_ps(sums[0], sums[1]), _mm256_hadd_ps(sums[2], sums[3]));
                        _mm_storeu_ps(dst + dstCol, _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1)));
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        __m256 sum = _mm256_setzero_ps();
                        const float * pSrc = src + dstRow*srcStride + dstCol*featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterSize; ++filterRow)
                        {
                            for (size_t filterCol = 0; filterCol < filterStride; filterCol += F)
                                ProductSum1x1<align>(pSrc + filterCol, pFilter + filterCol, sum);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        dst[dstCol] = Avx::ExtractSum(sum);
                    }
                    dst += dstStride;
                }
            }

            template <bool align> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterSize, float * dst, size_t dstStride)
            {
                if (featureSize == 16)
                    Filter<align, 16>(src, srcStride, dstWidth, dstHeight, filter, filterSize, dst, dstStride);
                else
                    Filter<align, 8>(src, srcStride, dstWidth, dstHeight, filter, filterSize, dst, dstStride);
            }

        public:

            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterSize, float * dst, size_t dstStride)
            {
                assert(featureSize == 8 || featureSize == 16);
                assert(srcWidth >= filterSize && srcHeight >= filterSize);

                size_t dstWidth = srcWidth - filterSize + 1;
                size_t dstHeight = srcHeight - filterSize + 1;

                if (Aligned(src) && Aligned(srcStride, F) && Aligned(filter))
                    Filter<true>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterSize, dst, dstStride);
                else
                    Filter<false>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterSize, dst, dstStride);
            }
        };

        void HogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterSize, float * dst, size_t dstStride)
        {
            HogLiteFeatureFilter featureFilter;
            featureFilter.Run(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterSize, dst, dstStride);
        }

        namespace HogLiteFeatureResizerDetail
        {
            template <int size> struct Feature
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const __m256 k[2][2], float * dst);
            };

            template <> struct Feature<8>
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const __m256 k[2][2], float * dst)
                {
                    Avx::Store<align>(dst + 0 * F, _mm256_add_ps(
                        _mm256_fmadd_ps(Load<align>(src0 + 0 * F), k[0][0], _mm256_mul_ps(Load<align>(src0 + 1 * F), k[0][1])),
                        _mm256_fmadd_ps(Load<align>(src1 + 0 * F), k[1][0], _mm256_mul_ps(Load<align>(src1 + 1 * F), k[1][1]))));
                }
            };

            template <> struct Feature<16>
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const __m256 k[2][2], float * dst)
                {
                    Avx::Store<align>(dst + 0 * F, _mm256_add_ps(
                        _mm256_fmadd_ps(Load<align>(src0 + 0 * F), k[0][0], _mm256_mul_ps(Load<align>(src0 + 2 * F), k[0][1])),
                        _mm256_fmadd_ps(Load<align>(src1 + 0 * F), k[1][0], _mm256_mul_ps(Load<align>(src1 + 2 * F), k[1][1]))));
                    Avx::Store<align>(dst + 1 * F, _mm256_add_ps(
                        _mm256_fmadd_ps(Load<align>(src0 + 1 * F), k[0][0], _mm256_mul_ps(Load<align>(src0 + 3 * F), k[0][1])),
                        _mm256_fmadd_ps(Load<align>(src1 + 1 * F), k[1][0], _mm256_mul_ps(Load<align>(src1 + 3 * F), k[1][1]))));
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
                    if (index >(int)srcSize - 2)
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
                __m256 _1 = _mm256_set1_ps(1.0f);
                for (size_t rowDst = 0; rowDst < dstHeight; ++rowDst)
                {
                    __m256 ky1 = _mm256_set1_ps(_ky[rowDst]);
                    __m256 ky0 = _mm256_sub_ps(_1, ky1);
                    const float * pSrc = src + _iy[rowDst];
                    float * pDst = dst + rowDst*dstStride;
                    for (size_t colDst = 0; colDst < dstWidth; ++colDst, pDst += featureSize)
                    {
                        __m256 kx1 = _mm256_set1_ps(_kx[colDst]);
                        __m256 kx0 = _mm256_sub_ps(_1, kx1);
                        __m256 k[2][2];
                        k[0][0] = _mm256_mul_ps(ky0, kx0);
                        k[0][1] = _mm256_mul_ps(ky0, kx1);
                        k[1][0] = _mm256_mul_ps(ky1, kx0);
                        k[1][1] = _mm256_mul_ps(ky1, kx1);
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
                        memcpy(dst + row*dstStride, src + row*srcStride, size);
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
    }
#endif// SIMD_AVX2_ENABLE
}


