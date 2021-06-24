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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i K8_KX4 = SIMD_MM512_SETR_EPI8(
            1, 3, 5, 7, 7, 5, 3, 1, 1, 3, 5, 7, 7, 5, 3, 1,
            1, 3, 5, 7, 7, 5, 3, 1, 1, 3, 5, 7, 7, 5, 3, 1,
            1, 3, 5, 7, 7, 5, 3, 1, 1, 3, 5, 7, 7, 5, 3, 1,
            1, 3, 5, 7, 7, 5, 3, 1, 1, 3, 5, 7, 7, 5, 3, 1);
        const __m512i K8_KX8 = SIMD_MM512_SETR_EPI8(
            1, 3, 5, 7, 9, 11, 13, 15, 15, 13, 11, 9, 7, 5, 3, 1,
            1, 3, 5, 7, 9, 11, 13, 15, 15, 13, 11, 9, 7, 5, 3, 1,
            1, 3, 5, 7, 9, 11, 13, 15, 15, 13, 11, 9, 7, 5, 3, 1,
            1, 3, 5, 7, 9, 11, 13, 15, 15, 13, 11, 9, 7, 5, 3, 1);

        const __m512i K32_PERMUTE_4 = SIMD_MM512_SETR_EPI32(0, 1, 2, 3, 1, 2, 3, 4, 4, 5, 6, 7, 5, 6, 7, 8);
        const __m512i K32_PERMUTE_8 = SIMD_MM512_SETR_EPI32(0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9);

        SIMD_INLINE __m512i Merge(__m512i a, __m512i b)
        {
            __m512i ab0 = Shuffle32i<0x88>(a, b);
            __m512i ab1 = Shuffle32i<0xDD>(a, b);
            return _mm512_add_epi16(ab0, ab1);
        }

        const __m512i K64_PERMUTE_0 = SIMD_MM512_SETR_EPI64(0x0, 0x1, 0x8, 0x9, 0x2, 0x3, 0xA, 0xB);
        const __m512i K64_PERMUTE_1 = SIMD_MM512_SETR_EPI64(0x4, 0x5, 0xC, 0xD, 0x6, 0x7, 0xE, 0xF);

        const __m512i K32_PERMUTE_BN_0 = SIMD_MM512_SETR_EPI32(0x11, 0x01, 0x10, 0x00, 0x12, 0x02, 0x11, 0x01, 0x13, 0x03, 0x12, 0x02, 0x14, 0x04, 0x13, 0x03);
        const __m512i K32_PERMUTE_BN_1 = SIMD_MM512_SETR_EPI32(0x15, 0x05, 0x14, 0x04, 0x16, 0x06, 0x15, 0x05, 0x17, 0x07, 0x16, 0x06, 0x18, 0x08, 0x17, 0x07);
        const __m512i K32_PERMUTE_BN_2 = SIMD_MM512_SETR_EPI32(0x19, 0x09, 0x18, 0x08, 0x1A, 0x0A, 0x19, 0x09, 0x1B, 0x0B, 0x1A, 0x0A, 0x1C, 0x0C, 0x1B, 0x0B);

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
            __mmask64 _wt;
            __mmask16 _ft[2];
            Bytes _value, _index;
            Ints _hi[2];
            Floats _hf[2], _nf[4], _nb;
            int _k0[cell], _k1[cell];
            __m256 _02, _05, _02357, _eps;
            __m512 _k;

            SIMD_INLINE void Init(size_t width)
            {
                _w = (width / cell - 1)*cell;
                _aw = AlignLo(_w, A);
                _wt = TailMask64(_w - _aw);
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
                    _hf[i].Resize(AlignHi(_hx*FQ, DF));
                }
                for (size_t i = 0; i < 4; ++i)
                    _nf[i].Resize(_hx + Avx2::F);
                _nb.Resize((_hx + 12) * 4);
                _k = _mm512_set1_ps(1.0f / Simd::Square(cell * 2));
                _02 = _mm256_set1_ps(0.2f);
                _05 = _mm256_set1_ps(0.5f);
                _02357 = _mm256_set1_ps(0.2357f);
                _eps = _mm256_set1_ps(0.0001f);
            }

            template<bool align, bool mask> static SIMD_INLINE void SetIndexAndValue(const uint8_t * src, size_t stride, uint8_t * value, uint8_t * index, __mmask64 tail = -1)
            {
                __m512i y0 = Load<false, mask>(src - stride, tail);
                __m512i y1 = Load<false, mask>(src + stride, tail);
                __m512i x0 = Load<false, mask>(src - 1, tail);
                __m512i x1 = Load<false, mask>(src + 1, tail);

                __m512i ady = AbsDifferenceU8(y0, y1);
                __m512i adx = AbsDifferenceU8(x0, x1);

                __m512i max = _mm512_max_epu8(ady, adx);
                __m512i min = _mm512_min_epu8(ady, adx);
                __m512i val = _mm512_adds_epu8(max, _mm512_avg_epu8(min, K_ZERO));
                Store<align, mask>(value, val, tail);

                __m512i idx = _mm512_mask_blend_epi8(_mm512_cmpgt_epu8_mask(adx, ady), K8_01, K_ZERO);
                idx = _mm512_mask_sub_epi8(idx, _mm512_cmple_epu8_mask(x1, x0), K8_03, idx);
                idx = _mm512_mask_sub_epi8(idx, _mm512_cmple_epu8_mask(y1, y0), K8_07, idx);
                Store<align, mask>(index, idx, tail);
            }

            SIMD_INLINE void SetIndexAndValue(const uint8_t * src, size_t stride)
            {
                uint8_t * value = _value.data + A;
                uint8_t * index = _index.data + A;
                size_t col = 0;
                for (; col < _aw; col += A)
                    SetIndexAndValue<true, false>(src + col, stride, value + col, index + col);
                if (col < _w)
                    SetIndexAndValue<true, true>(src + col, stride, value + col, index + col, _wt);
            }

            static SIMD_INLINE void UpdateIntegerHistogram4x4(uint8_t * value, uint8_t * index, const __m512i & ky0, const __m512i & ky1, int * h0, int * h1)
            {
                __m512i val = _mm512_permutexvar_epi32(K32_PERMUTE_4, Load<false>(value));
                __m512i idx = _mm512_permutexvar_epi32(K32_PERMUTE_4, Load<false>(index));
                __m512i cur0 = K_ZERO;
                __m512i cur1 = K8_01;
                __m512i dirs[4];
                for (size_t i = 0; i < 4; ++i)
                {
                    __m512i dir0 = _mm512_maddubs_epi16(_mm512_maskz_mov_epi8(_mm512_cmpeq_epi8_mask(idx, cur0), val), K8_KX4);
                    __m512i dir1 = _mm512_maddubs_epi16(_mm512_maskz_mov_epi8(_mm512_cmpeq_epi8_mask(idx, cur1), val), K8_KX4);
                    dirs[i] = Merge(dir0, dir1);
                    cur0 = _mm512_add_epi8(cur0, K8_02);
                    cur1 = _mm512_add_epi8(cur1, K8_02);
                }
                __m512i hx0 = Shuffle32i<0x88>(dirs[0], dirs[1]);
                __m512i hx1 = Shuffle32i<0x88>(dirs[2], dirs[3]);
                __m512i hx2 = Shuffle32i<0xDD>(dirs[0], dirs[1]);
                __m512i hx3 = Shuffle32i<0xDD>(dirs[2], dirs[3]);
                __m512i hx0p = _mm512_permutex2var_epi64(hx0, K64_PERMUTE_0, hx1);
                __m512i hx1p = _mm512_permutex2var_epi64(hx0, K64_PERMUTE_1, hx1);
                __m512i hx2p = _mm512_permutex2var_epi64(hx2, K64_PERMUTE_0, hx3);
                __m512i hx3p = _mm512_permutex2var_epi64(hx2, K64_PERMUTE_1, hx3);
                Store<true>(h0 + 0 * F, _mm512_add_epi32(Load<true>(h0 + 0 * F), _mm512_madd_epi16(hx0p, ky0)));
                Store<true>(h0 + 1 * F, _mm512_add_epi32(Load<true>(h0 + 1 * F), _mm512_madd_epi16(hx2p, ky0)));
                Store<true>(h0 + 2 * F, _mm512_add_epi32(Load<true>(h0 + 2 * F), _mm512_madd_epi16(hx1p, ky0)));
                Store<true>(h0 + 3 * F, _mm512_add_epi32(Load<true>(h0 + 3 * F), _mm512_madd_epi16(hx3p, ky0)));
                Store<true>(h1 + 0 * F, _mm512_add_epi32(Load<true>(h1 + 0 * F), _mm512_madd_epi16(hx0p, ky1)));
                Store<true>(h1 + 1 * F, _mm512_add_epi32(Load<true>(h1 + 1 * F), _mm512_madd_epi16(hx2p, ky1)));
                Store<true>(h1 + 2 * F, _mm512_add_epi32(Load<true>(h1 + 2 * F), _mm512_madd_epi16(hx1p, ky1)));
                Store<true>(h1 + 3 * F, _mm512_add_epi32(Load<true>(h1 + 3 * F), _mm512_madd_epi16(hx3p, ky1)));
            }

            SIMD_INLINE void UpdateIntegerHistogram4x4(size_t rowI, size_t rowF)
            {
                int * h0 = _hi[(rowI + 0) & 1].data;
                int * h1 = _hi[(rowI + 1) & 1].data;
                uint8_t * value = _value.data + A - cell;
                uint8_t * index = _index.data + A - cell;
                __m512i ky0 = _mm512_set1_epi16((short)_k0[rowF]);
                __m512i ky1 = _mm512_set1_epi16((short)_k1[rowF]);
                for (size_t col = 0; col <= _w;)
                {
                    UpdateIntegerHistogram4x4(value + col, index + col, ky0, ky1, h0, h1);
                    col += 8 * cell;
                    h0 += 8 * FQ;
                    h1 += 8 * FQ;
                }
            }

            static SIMD_INLINE void UpdateIntegerHistogram8x8(uint8_t * value, uint8_t * index, const __m512i & ky0, const __m512i & ky1, int * h0, int * h1)
            {
                __m512i val = _mm512_permutexvar_epi32(K32_PERMUTE_8, Load<false>(value));
                __m512i idx = _mm512_permutexvar_epi32(K32_PERMUTE_8, Load<false>(index));
                __m512i cur0 = K_ZERO;
                __m512i cur1 = K8_01;
                __m512i dirs[4];
                for (size_t i = 0; i < 4; ++i)
                {
                    __m512i dir0 = _mm512_maddubs_epi16(_mm512_maskz_mov_epi8(_mm512_cmpeq_epi8_mask(idx, cur0), val), K8_KX8);
                    __m512i dir1 = _mm512_maddubs_epi16(_mm512_maskz_mov_epi8(_mm512_cmpeq_epi8_mask(idx, cur1), val), K8_KX8);
                    dirs[i] = Merge(dir0, dir1);
                    cur0 = _mm512_add_epi8(cur0, K8_02);
                    cur1 = _mm512_add_epi8(cur1, K8_02);
                }
                dirs[0] = Merge(dirs[0], dirs[1]);
                dirs[1] = Merge(dirs[2], dirs[3]);
                __m512i hx0 = _mm512_permutex2var_epi64(dirs[0], K64_PERMUTE_0, dirs[1]);
                __m512i hx1 = _mm512_permutex2var_epi64(dirs[0], K64_PERMUTE_1, dirs[1]);
                Store<false>(h0 + 0, _mm512_add_epi32(Load<false>(h0 + 0), _mm512_madd_epi16(hx0, ky0)));
                Store<false>(h0 + F, _mm512_add_epi32(Load<false>(h0 + F), _mm512_madd_epi16(hx1, ky0)));
                Store<false>(h1 + 0, _mm512_add_epi32(Load<false>(h1 + 0), _mm512_madd_epi16(hx0, ky1)));
                Store<false>(h1 + F, _mm512_add_epi32(Load<false>(h1 + F), _mm512_madd_epi16(hx1, ky1)));
            }

            SIMD_INLINE void UpdateIntegerHistogram8x8(size_t rowI, size_t rowF)
            {
                int * h0 = _hi[(rowI + 0) & 1].data;
                int * h1 = _hi[(rowI + 1) & 1].data;
                uint8_t * value = _value.data + A - cell;
                uint8_t * index = _index.data + A - cell;
                __m512i ky0 = _mm512_set1_epi16((short)_k0[rowF]);
                __m512i ky1 = _mm512_set1_epi16((short)_k1[rowF]);
                for (size_t col = 0; col <= _w;)
                {
                    UpdateIntegerHistogram8x8(value + col, index + col, ky0, ky1, h0, h1);
                    col += 4 * cell;
                    h0 += 4 * FQ;
                    h1 += 4 * FQ;
                }
            }

            SIMD_INLINE void UpdateFloatHistogram(size_t rowI)
            {
                Ints & hi = _hi[rowI & 1];
                Floats & hf = _hf[rowI & 1];
                Floats & nf = _nf[rowI & 3];
                for (size_t i = 0; i < hf.size; i += DF)
                {
                    Avx512f::Store<true>(hf.data + i + 0, _mm512_mul_ps(_k, _mm512_cvtepi32_ps(Load<true>(hi.data + i + 0))));
                    Avx512f::Store<true>(hf.data + i + F, _mm512_mul_ps(_k, _mm512_cvtepi32_ps(Load<true>(hi.data + i + F))));
                }
                hi.Clear();

                const float * h = hf.data;
                size_t ahx = AlignLo(_hx, 4), x = 0;
                for (; x < ahx; x += 4, h += QQ)
                {
                    __m256 h01 = Avx2::Load<true>(h + 0 * FQ);
                    __m256 h23 = Avx2::Load<true>(h + 1 * FQ);
                    __m256 h45 = Avx2::Load<true>(h + 2 * FQ);
                    __m256 h67 = Avx2::Load<true>(h + 3 * FQ);
                    __m256 s01 = _mm256_add_ps(_mm256_permute2f128_ps(h01, h23, 0x20), _mm256_permute2f128_ps(h01, h23, 0x31));
                    __m256 n01 = Avx2::Permute4x64<0x88>(_mm256_dp_ps(s01, s01, 0xF1));
                    __m256 s23 = _mm256_add_ps(_mm256_permute2f128_ps(h45, h67, 0x20), _mm256_permute2f128_ps(h45, h67, 0x31));
                    __m256 n23 = Avx2::Permute4x64<0x88>(_mm256_dp_ps(s23, s23, 0xF1));
                    _mm_storeu_ps(nf.data + x, _mm_shuffle_ps(_mm256_castps256_ps128(n01), _mm256_castps256_ps128(n23), 0x88));
                }
                for (; x < _hx; ++x, h += FQ)
                {
                    __m128 h0 = Sse2::Load<true>(h + 00);
                    __m128 h1 = Sse2::Load<true>(h + HQ);
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
                for (size_t x = 0; x < _fx; x += 12, dst += 3 * F)
                {
                    __m512 s0 = Avx512f::Load<false>(src0 + x);
                    __m512 s1 = Avx512f::Load<false>(src1 + x);
                    __m512 s2 = Avx512f::Load<false>(src2 + x);
                    __m512 v0 = _mm512_add_ps(s0, s1);
                    __m512 v1 = _mm512_add_ps(s1, s2);
                    __m512 h0 = _mm512_add_ps(v0, Alignr<1>(v0, v0));
                    __m512 h1 = _mm512_add_ps(v1, Alignr<1>(v1, v1));
                    Avx512f::Store<true>(dst + 0 * F, _mm512_permutex2var_ps(h0, K32_PERMUTE_BN_0, h1));
                    Avx512f::Store<true>(dst + 1 * F, _mm512_permutex2var_ps(h0, K32_PERMUTE_BN_1, h1));
                    Avx512f::Store<true>(dst + 2 * F, _mm512_permutex2var_ps(h0, K32_PERMUTE_BN_2, h1));
                }
            }

            SIMD_INLINE __m256 Features07(const __m256 & n, const __m256 & s, __m256 & t)
            {
                __m256 h0 = _mm256_min_ps(_mm256_mul_ps(Avx2::Broadcast<0>(s), n), _02);
                __m256 h1 = _mm256_min_ps(_mm256_mul_ps(Avx2::Broadcast<1>(s), n), _02);
                __m256 h2 = _mm256_min_ps(_mm256_mul_ps(Avx2::Broadcast<2>(s), n), _02);
                __m256 h3 = _mm256_min_ps(_mm256_mul_ps(Avx2::Broadcast<3>(s), n), _02);
                t = _mm256_add_ps(t, _mm256_add_ps(_mm256_add_ps(h0, h1), _mm256_add_ps(h2, h3)));
                return _mm256_mul_ps(_05, _mm256_hadd_ps(_mm256_hadd_ps(h0, h1), _mm256_hadd_ps(h2, h3)));
            }

            SIMD_INLINE __m256 Features8B(const __m256 & n, const __m256 & s)
            {
                __m256 h0 = _mm256_min_ps(_mm256_mul_ps(Avx2::Broadcast<0>(s), n), _02);
                __m256 h1 = _mm256_min_ps(_mm256_mul_ps(Avx2::Broadcast<1>(s), n), _02);
                __m256 h2 = _mm256_min_ps(_mm256_mul_ps(Avx2::Broadcast<2>(s), n), _02);
                __m256 h3 = _mm256_min_ps(_mm256_mul_ps(Avx2::Broadcast<3>(s), n), _02);
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
                    const float * src = hf + x * FQ;
                    __m256 s0 = Avx::Load<false>(src + 0 * HQ, src + 2 * HQ);
                    __m256 s1 = Avx::Load<false>(src + 1 * HQ, src + 3 * HQ);
                    f[0] = Features07(n, s0, t);
                    f[1] = Features07(n, s1, t);
                    f[2] = Features8B(n, _mm256_add_ps(s0, s1));
                    f[3] = _mm256_mul_ps(t, _02357);
                    Avx::Store<false>(dst + 0 * Avx2::F, _mm256_permute2f128_ps(f[0], f[1], 0x20));
                    Avx::Store<false>(dst + 1 * Avx2::F, _mm256_permute2f128_ps(f[2], f[3], 0x20));
                    Avx::Store<false>(dst + 2 * Avx2::F, _mm256_permute2f128_ps(f[0], f[1], 0x31));
                    Avx::Store<false>(dst + 3 * Avx2::F, _mm256_permute2f128_ps(f[2], f[3], 0x31));
                }
                for (; x < _fx; ++x, nb += 4)
                {
                    __m128 n = _mm_rsqrt_ps(_mm_add_ps(_mm_load_ps(nb), _mm256_castps256_ps128(_eps)));
                    __m128 t = _mm_setzero_ps();
                    const float * src = hf + x * FQ;
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
                    src = hf + x * FQ;
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
            template<bool align, bool mask> SIMD_INLINE void ProductSum1x1(const float * src, const float * filter, __m512 & sum, __mmask16 tail = -1)
            {
                __m512 _src = Avx512f::Load<align, mask>(src, tail);
                __m512 _filter = Avx512f::Load<align, mask>(filter, tail);
                sum = _mm512_fmadd_ps(_src, _filter, sum);
            }

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

            template<bool align> SIMD_INLINE void ProductSum1x4x8(const float * src, const float * filter, __m512 * sums)
            {
                __m512 _filter = _mm512_broadcast_f32x8(Avx::Load<align>(filter));
                sums[0] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 0 * F), _filter, sums[0]);
                sums[1] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 1 * F), _filter, sums[1]);
            }

            template <bool align> static SIMD_INLINE void ProductSum4x4x8(const float * src, const float * filter, __m512 * sums)
            {
                __m512 filter0 = _mm512_broadcast_f32x8(Avx::Load<align>(filter + 0 * HF));
                __m512 src0 = Avx512f::Load<align>(src + 0 * HF);
                __m512 src2 = Avx512f::Load<align>(src + 2 * HF);
                sums[0] = _mm512_fmadd_ps(src0, filter0, sums[0]);
                sums[1] = _mm512_fmadd_ps(src2, filter0, sums[1]);
                __m512 filter2 = _mm512_broadcast_f32x8(Avx::Load<align>(filter + 2 * HF));
                __m512 src4 = Avx512f::Load<align>(src + 4 * HF);
                sums[0] = _mm512_fmadd_ps(src2, filter2, sums[0]);
                sums[1] = _mm512_fmadd_ps(src4, filter2, sums[1]);
                __m512 filter1 = _mm512_broadcast_f32x8(Avx::Load<align>(filter + 1 * HF));
                __m512 src1 = Alignr<8>(src0, src2);
                __m512 src3 = Alignr<8>(src2, src4);
                sums[0] = _mm512_fmadd_ps(src1, filter1, sums[0]);
                sums[1] = _mm512_fmadd_ps(src3, filter1, sums[1]);
                __m512 filter3 = _mm512_broadcast_f32x8(Avx::Load<align>(filter + 3 * HF));
                __m512 src5 = Avx512f::Load<false>(src + 5 * HF);
                sums[0] = _mm512_fmadd_ps(src3, filter3, sums[0]);
                sums[1] = _mm512_fmadd_ps(src5, filter3, sums[1]);
            }

            template <bool align> void Filter8(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterWidth, size_t filterHeight, float * dst, size_t dstStride)
            {
                size_t filterStride = 8 * filterWidth;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                size_t alignedFilterStride = AlignLo(filterStride, DF);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        __m512 sums[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
                        const float * pSrc = src + dstRow * srcStride + dstCol * 8;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                        {
                            size_t filterCol = 0;
                            for (; filterCol < alignedFilterStride; filterCol += DF)
                                ProductSum4x4x8<align>(pSrc + filterCol, pFilter + filterCol, sums);
                            for (; filterCol < filterStride; filterCol += HF)
                                ProductSum1x4x8<false>(pSrc + filterCol, pFilter + filterCol, sums);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        __m256 sum0 = _mm256_hadd_ps(_mm512_castps512_ps256(sums[0]), _mm512_castps512_ps256(Alignr<8>(sums[0], sums[0])));
                        __m256 sum1 = _mm256_hadd_ps(_mm512_castps512_ps256(sums[1]), _mm512_castps512_ps256(Alignr<8>(sums[1], sums[1])));
                        __m256 sum = _mm256_hadd_ps(sum0, sum1);
                        _mm_storeu_ps(dst + dstCol, _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1)));
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        __m256 sum = _mm256_setzero_ps();
                        const float * pSrc = src + dstRow * srcStride + dstCol * 8;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                        {
                            for (size_t filterCol = 0; filterCol < filterStride; filterCol += Avx::F)
                                ProductSum1x1<align>(pSrc + filterCol, pFilter + filterCol, sum);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        dst[dstCol] = Avx::ExtractSum(sum);
                    }
                    dst += dstStride;
                }
            }

            template <bool align> void Filter8(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                size_t filterStride = 8 * filterWidth;
                size_t alignedDstWidth = AlignLo(dstWidth, 8);
                size_t alignedFilterStride = AlignLo(filterStride, DF);
                __m128 _min = _mm_set1_ps(-FLT_MAX);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        __m128 _mask = _mm_castsi128_ps(_mm_loadu_si128((__m128i*)(mask + dstCol)));
                        if (Sse41::TestZ(_mask))
                            _mm_storeu_ps(dst + dstCol, _min);
                        else
                        {
                            __m512 sums[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
                            const float * pSrc = src + dstRow * srcStride + dstCol * 8;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                size_t filterCol = 0;
                                for (; filterCol < alignedFilterStride; filterCol += DF)
                                    ProductSum4x4x8<align>(pSrc + filterCol, pFilter + filterCol, sums);
                                for (; filterCol < filterStride; filterCol += HF)
                                    ProductSum1x4x8<false>(pSrc + filterCol, pFilter + filterCol, sums);
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            __m256 sum0 = _mm256_hadd_ps(_mm512_castps512_ps256(sums[0]), _mm512_castps512_ps256(Alignr<8>(sums[0], sums[0])));
                            __m256 sum1 = _mm256_hadd_ps(_mm512_castps512_ps256(sums[1]), _mm512_castps512_ps256(Alignr<8>(sums[1], sums[1])));
                            __m256 sum = _mm256_hadd_ps(sum0, sum1);
                            _mm_storeu_ps(dst + dstCol, _mm_blendv_ps(_min, _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1)), _mask));
                        }
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        if (mask[dstCol])
                        {
                            __m256 sum = _mm256_setzero_ps();
                            const float * pSrc = src + dstRow * srcStride + dstCol * 8;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                for (size_t filterCol = 0; filterCol < filterStride; filterCol += Avx::F)
                                    ProductSum1x1<align>(pSrc + filterCol, pFilter + filterCol, sum);
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            dst[dstCol] = Avx::ExtractSum(sum);
                        }
                        else
                            dst[dstCol] = -FLT_MAX;
                    }
                    dst += dstStride;
                    mask += maskStride;
                }
            }

            template<bool align> static SIMD_INLINE void ProductSum1x4x16(const float * src, const float * filter, __m512 * sums)
            {
                __m512 _filter = Avx512f::Load<align>(filter);
                sums[0] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 0 * F), _filter, sums[0]);
                sums[1] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 1 * F), _filter, sums[1]);
                sums[2] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 2 * F), _filter, sums[2]);
                sums[3] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 3 * F), _filter, sums[3]);
            }

            template <bool align> static SIMD_INLINE void ProductSum4x4x16(const float * src, const float * filter, __m512 * sums)
            {
                __m512 filter0 = Avx512f::Load<align>(filter + 0 * F);
                __m512 src0 = Avx512f::Load<align>(src + 0 * F);
                __m512 src1 = Avx512f::Load<align>(src + 1 * F);
                __m512 src2 = Avx512f::Load<align>(src + 2 * F);
                __m512 src3 = Avx512f::Load<align>(src + 3 * F);
                sums[0] = _mm512_fmadd_ps(src0, filter0, sums[0]);
                sums[1] = _mm512_fmadd_ps(src1, filter0, sums[1]);
                sums[2] = _mm512_fmadd_ps(src2, filter0, sums[2]);
                sums[3] = _mm512_fmadd_ps(src3, filter0, sums[3]);
                __m512 filter1 = Avx512f::Load<align>(filter + 1 * F);
                __m512 src4 = Avx512f::Load<align>(src + 4 * F);
                sums[0] = _mm512_fmadd_ps(src1, filter1, sums[0]);
                sums[1] = _mm512_fmadd_ps(src2, filter1, sums[1]);
                sums[2] = _mm512_fmadd_ps(src3, filter1, sums[2]);
                sums[3] = _mm512_fmadd_ps(src4, filter1, sums[3]);
            }

            template <bool align> void Filter16(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterWidth, size_t filterHeight, float * dst, size_t dstStride)
            {
                size_t filterStride = 16 * filterWidth;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                size_t alignedFilterStride = AlignLo(filterStride, DF);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                        const float * pSrc = src + dstRow * srcStride + dstCol * 16;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                        {
                            size_t filterCol = 0;
                            for (; filterCol < alignedFilterStride; filterCol += DF)
                                ProductSum4x4x16<align>(pSrc + filterCol, pFilter + filterCol, sums);
                            for (; filterCol < filterStride; filterCol += F)
                                ProductSum1x4x16<align>(pSrc + filterCol, pFilter + filterCol, sums);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        __m256 sum0 = _mm512_castps512_ps256(_mm512_add_ps(sums[0], Alignr<8>(sums[0], sums[0])));
                        __m256 sum1 = _mm512_castps512_ps256(_mm512_add_ps(sums[1], Alignr<8>(sums[1], sums[1])));
                        __m256 sum2 = _mm512_castps512_ps256(_mm512_add_ps(sums[2], Alignr<8>(sums[2], sums[2])));
                        __m256 sum3 = _mm512_castps512_ps256(_mm512_add_ps(sums[3], Alignr<8>(sums[3], sums[3])));
                        __m256 sum = _mm256_hadd_ps(_mm256_hadd_ps(sum0, sum1), _mm256_hadd_ps(sum2, sum3));
                        _mm_storeu_ps(dst + dstCol, _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1)));
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        __m512 sum = _mm512_setzero_ps();
                        const float * pSrc = src + dstRow * srcStride + dstCol * 16;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                        {
                            for (size_t filterCol = 0; filterCol < filterStride; filterCol += F)
                                ProductSum1x1<align, false>(pSrc + filterCol, pFilter + filterCol, sum);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        dst[dstCol] = Avx512f::ExtractSum(sum);
                    }
                    dst += dstStride;
                }
            }

            template <bool align> void Filter16(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                size_t filterStride = 16 * filterWidth;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                size_t alignedFilterStride = AlignLo(filterStride, DF);
                __m128 _min = _mm_set1_ps(-FLT_MAX);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        __m128 _mask = _mm_castsi128_ps(_mm_loadu_si128((__m128i*)(mask + dstCol)));
                        if (Sse41::TestZ(_mask))
                            _mm_storeu_ps(dst + dstCol, _min);
                        else
                        {
                            __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            const float * pSrc = src + dstRow * srcStride + dstCol * 16;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                size_t filterCol = 0;
                                for (; filterCol < alignedFilterStride; filterCol += DF)
                                    ProductSum4x4x16<align>(pSrc + filterCol, pFilter + filterCol, sums);
                                for (; filterCol < filterStride; filterCol += F)
                                    ProductSum1x4x16<align>(pSrc + filterCol, pFilter + filterCol, sums);
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            __m256 sum0 = _mm512_castps512_ps256(_mm512_add_ps(sums[0], Alignr<8>(sums[0], sums[0])));
                            __m256 sum1 = _mm512_castps512_ps256(_mm512_add_ps(sums[1], Alignr<8>(sums[1], sums[1])));
                            __m256 sum2 = _mm512_castps512_ps256(_mm512_add_ps(sums[2], Alignr<8>(sums[2], sums[2])));
                            __m256 sum3 = _mm512_castps512_ps256(_mm512_add_ps(sums[3], Alignr<8>(sums[3], sums[3])));
                            __m256 sum = _mm256_hadd_ps(_mm256_hadd_ps(sum0, sum1), _mm256_hadd_ps(sum2, sum3));
                            _mm_storeu_ps(dst + dstCol, _mm_blendv_ps(_min, _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1)), _mask));
                        }
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        if (mask[dstCol])
                        {
                            __m512 sum = _mm512_setzero_ps();
                            const float * pSrc = src + dstRow * srcStride + dstCol * 16;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                for (size_t filterCol = 0; filterCol < filterStride; filterCol += F)
                                    ProductSum1x1<align, false>(pSrc + filterCol, pFilter + filterCol, sum);
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            dst[dstCol] = Avx512f::ExtractSum(sum);
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
                    Filter16<align>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, dst, dstStride);
                else
                    Filter8<align>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, dst, dstStride);
            }

            template <bool align> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                if (featureSize == 16)
                    Filter16<align>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                else
                    Filter8<align>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
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

            template<bool align> void Resize8(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                __m512 _1 = _mm512_set1_ps(1.0f);
                size_t alignedDstWidth = AlignLo(dstWidth, 2);
                for (size_t rowDst = 0; rowDst < dstHeight; ++rowDst)
                {
                    __m512 ky1 = _mm512_set1_ps(_ky[rowDst]);
                    __m512 ky0 = _mm512_sub_ps(_1, ky1);
                    const float * pSrc = src + _iy[rowDst];
                    float * pDst = dst + rowDst * dstStride;
                    size_t colDst = 0;
                    for (; colDst < alignedDstWidth; colDst += 2, pDst += F)
                    {
                        __m512 kx1 = _mm512_insertf32x8(_mm512_set1_ps(_kx[colDst + 0]), _mm256_set1_ps(_kx[colDst + 1]), 1);
                        __m512 kx0 = _mm512_sub_ps(_1, kx1);
                        __m512 k00 = _mm512_mul_ps(ky0, kx0);
                        __m512 k01 = _mm512_mul_ps(ky0, kx1);
                        __m512 k10 = _mm512_mul_ps(ky1, kx0);
                        __m512 k11 = _mm512_mul_ps(ky1, kx1);
                        const float * pSrc00 = pSrc + _ix[colDst + 0];
                        const float * pSrc01 = pSrc + _ix[colDst + 1];
                        const float * pSrc10 = pSrc00 + srcStride;
                        const float * pSrc11 = pSrc01 + srcStride;
                        Avx512f::Store<align>(pDst, _mm512_add_ps(
                            _mm512_fmadd_ps(Load<align>(pSrc00, pSrc01), k00, _mm512_mul_ps(Load<align>(pSrc00 + Avx2::F, pSrc01 + Avx2::F), k01)),
                            _mm512_fmadd_ps(Load<align>(pSrc10, pSrc11), k10, _mm512_mul_ps(Load<align>(pSrc10 + Avx2::F, pSrc11 + Avx2::F), k11))));
                    }
                    for (; colDst < dstWidth; ++colDst, pDst += Avx2::F)
                    {
                        __m256 kx1 = _mm256_set1_ps(_kx[colDst]);
                        __m256 kx0 = _mm256_sub_ps(_mm512_castps512_ps256(_1), kx1);
                        __m256 k00 = _mm256_mul_ps(_mm512_castps512_ps256(ky0), kx0);
                        __m256 k01 = _mm256_mul_ps(_mm512_castps512_ps256(ky0), kx1);
                        __m256 k10 = _mm256_mul_ps(_mm512_castps512_ps256(ky1), kx0);
                        __m256 k11 = _mm256_mul_ps(_mm512_castps512_ps256(ky1), kx1);
                        const float * pSrc0 = pSrc + _ix[colDst];
                        const float * pSrc1 = pSrc0 + srcStride;
                        Avx::Store<align>(pDst, _mm256_add_ps(
                            _mm256_fmadd_ps(Avx::Load<align>(pSrc0), k00, _mm256_mul_ps(Avx::Load<align>(pSrc0 + Avx2::F), k01)),
                            _mm256_fmadd_ps(Avx::Load<align>(pSrc1), k10, _mm256_mul_ps(Avx::Load<align>(pSrc1 + Avx2::F), k11))));
                    }
                }
            }

            template<bool align> void Resize16(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                __m512 _1 = _mm512_set1_ps(1.0f);
                for (size_t rowDst = 0; rowDst < dstHeight; ++rowDst)
                {
                    __m512 ky1 = _mm512_set1_ps(_ky[rowDst]);
                    __m512 ky0 = _mm512_sub_ps(_1, ky1);
                    const float * pSrc = src + _iy[rowDst];
                    float * pDst = dst + rowDst * dstStride;
                    for (size_t colDst = 0; colDst < dstWidth; ++colDst, pDst += F)
                    {
                        __m512 kx1 = _mm512_set1_ps(_kx[colDst]);
                        __m512 kx0 = _mm512_sub_ps(_1, kx1);
                        __m512 k00 = _mm512_mul_ps(ky0, kx0);
                        __m512 k01 = _mm512_mul_ps(ky0, kx1);
                        __m512 k10 = _mm512_mul_ps(ky1, kx0);
                        __m512 k11 = _mm512_mul_ps(ky1, kx1);
                        const float * pSrc0 = pSrc + _ix[colDst];
                        const float * pSrc1 = pSrc0 + srcStride;
                        Avx512f::Store<align>(pDst, _mm512_add_ps(
                            _mm512_fmadd_ps(Avx512f::Load<align>(pSrc0), k00, _mm512_mul_ps(Avx512f::Load<align>(pSrc0 + F), k01)),
                            _mm512_fmadd_ps(Avx512f::Load<align>(pSrc1), k10, _mm512_mul_ps(Avx512f::Load<align>(pSrc1 + F), k11))));
                    }
                }
            }

            template<bool align> void Resize(const float * src, size_t srcStride, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                if (featureSize == 8)
                    Resize8<align>(src, srcStride, dst, dstStride, dstWidth, dstHeight);
                else
                    Resize16<align>(src, srcStride, dst, dstStride, dstWidth, dstHeight);
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
            if (align)
                assert(Aligned(src) && Aligned(pca) && Aligned(dst));

            SIMD_ALIGNED(64) float pca2[128];
            for (size_t i = 0; i < 8; ++i)
            {
                for (size_t j = 0; j < 8; ++j)
                    pca2[j * 16 + i + 0] = pca[i * 16 + j + 0];
                for (size_t j = 0; j < 8; ++j)
                    pca2[j * 16 + i + 8] = pca[i * 16 + j + 8];
            }
            __m512 _pca[8];
            for (size_t i = 0; i < 8; ++i)
                _pca[i] = Avx512f::Load<true>(pca2 + i * F);
            for (size_t row = 0; row < height; ++row)
            {
                const float * s = src;
                float * d = dst;
                for (size_t col = 0; col < width; ++col)
                {
                    __m512 sums[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
                    __m512 _src = Avx512f::Load<align>(s);
                    __m512 src0 = Shuffle2x<0x44>(_src);
                    __m512 src1 = Shuffle2x<0xEE>(_src);
                    sums[0] = _mm512_fmadd_ps(Broadcast<0>(src0), _pca[0], sums[0]);
                    sums[1] = _mm512_fmadd_ps(Broadcast<0>(src1), _pca[4], sums[1]);
                    sums[0] = _mm512_fmadd_ps(Broadcast<1>(src0), _pca[1], sums[0]);
                    sums[1] = _mm512_fmadd_ps(Broadcast<1>(src1), _pca[5], sums[1]);
                    sums[0] = _mm512_fmadd_ps(Broadcast<2>(src0), _pca[2], sums[0]);
                    sums[1] = _mm512_fmadd_ps(Broadcast<2>(src1), _pca[6], sums[1]);
                    sums[0] = _mm512_fmadd_ps(Broadcast<3>(src0), _pca[3], sums[0]);
                    sums[1] = _mm512_fmadd_ps(Broadcast<3>(src1), _pca[7], sums[1]);
                    sums[0] = _mm512_add_ps(sums[0], sums[1]);
                    sums[0] = _mm512_add_ps(sums[0], Avx512f::Alignr<8>(sums[0], _mm512_setzero_ps()));
                    Avx::Store<align>(d, _mm512_castps512_ps256(sums[0]));
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
            Array512f _filter;

            void Init(size_t srcWidth, size_t srcHeight, size_t hSize, size_t vSize)
            {
                _dstWidth = srcWidth - hSize + 1;
                _dstStride = AlignHi(_dstWidth, Avx512f::F);
                _dstHeight = srcHeight - vSize + 1;
                _buffer.Resize(_dstStride*srcHeight);
            }

            template<bool align> static SIMD_INLINE void FilterHx1x8(const float * src, const float * filter, __m256 & sum)
            {
                __m256 _src = Avx::Load<align>(src);
                __m256 _filter = Avx::Load<align>(filter);
                sum = _mm256_fmadd_ps(_src, _filter, sum);
            }

            template<bool align> static SIMD_INLINE void FilterHx4x8(const float * src, const float * filter, __m256 * sums)
            {
                __m256 _filter = Avx::Load<align>(filter);
                sums[0] = _mm256_fmadd_ps(Avx::Load<align>(src + 0 * Avx::F), _filter, sums[0]);
                sums[1] = _mm256_fmadd_ps(Avx::Load<align>(src + 1 * Avx::F), _filter, sums[1]);
                sums[2] = _mm256_fmadd_ps(Avx::Load<align>(src + 2 * Avx::F), _filter, sums[2]);
                sums[3] = _mm256_fmadd_ps(Avx::Load<align>(src + 3 * Avx::F), _filter, sums[3]);
            }

            template <bool align> void FilterHx8o(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                const size_t step = 8;
                size_t alignedWidth = AlignLo(width, 4);
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedWidth; col += 4)
                    {
                        __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += Avx::F)
                            FilterHx4x8<align>(s + i, filter + i, sums);
                        Sse2::Store<true>(dst + col, Avx::Extract4Sums(sums));
                    }
                    for (; col < width; ++col)
                    {
                        __m256 sum = _mm256_setzero_ps();
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += Avx::F)
                            FilterHx1x8<align>(s + i, filter + i, sum);
                        dst[col] = Avx::ExtractSum(sum);
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template<bool align> static SIMD_INLINE void FilterHx4x8(const float * src, const float * filter, __m512 * sums)
            {
                __m512 _filter = Avx512f::Load<align>(filter);
                sums[0] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 0 * HF), _filter, sums[0]);
                sums[1] = _mm512_fmadd_ps(Avx512f::Load<false>(src + 1 * HF), _filter, sums[1]);
                sums[2] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 2 * HF), _filter, sums[2]);
                sums[3] = _mm512_fmadd_ps(Avx512f::Load<false>(src + 3 * HF), _filter, sums[3]);
            }

            template <bool align> void FilterHx8e(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                const size_t step = 8;
                size_t alignedWidth = AlignLo(width, 4);
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedWidth; col += 4)
                    {
                        __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += F)
                            FilterHx4x8<align>(s + i, filter + i, sums);
                        Sse2::Store<true>(dst + col, Avx512f::Extract4Sums(sums));
                    }
                    for (; col < width; ++col)
                    {
                        __m256 sum = _mm256_setzero_ps();
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += Avx::F)
                            FilterHx1x8<align>(s + i, filter + i, sum);
                        dst[col] = Avx::ExtractSum(sum);
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template<bool align> static SIMD_INLINE void FilterHx1x16(const float * src, const float * filter, __m512 & sum)
            {
                __m512 _src = Avx512f::Load<align>(src);
                __m512 _filter = Avx512f::Load<align>(filter);
                sum = _mm512_fmadd_ps(_src, _filter, sum);
            }

            template<bool align> static SIMD_INLINE void FilterHx4x16(const float * src, const float * filter, __m512 * sums)
            {
                __m512 _filter = Avx512f::Load<align>(filter);
                sums[0] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 0 * F), _filter, sums[0]);
                sums[1] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 1 * F), _filter, sums[1]);
                sums[2] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 2 * F), _filter, sums[2]);
                sums[3] = _mm512_fmadd_ps(Avx512f::Load<align>(src + 3 * F), _filter, sums[3]);
            }

            template <bool align> static SIMD_INLINE void FilterHx4x16x2(const float * src, const float * filter, __m512 * sums)
            {
                __m512 filter0 = Avx512f::Load<align>(filter + 0 * F);
                __m512 src0 = Avx512f::Load<align>(src + 0 * F);
                __m512 src1 = Avx512f::Load<align>(src + 1 * F);
                __m512 src2 = Avx512f::Load<align>(src + 2 * F);
                __m512 src3 = Avx512f::Load<align>(src + 3 * F);
                sums[0] = _mm512_fmadd_ps(src0, filter0, sums[0]);
                sums[1] = _mm512_fmadd_ps(src1, filter0, sums[1]);
                sums[2] = _mm512_fmadd_ps(src2, filter0, sums[2]);
                sums[3] = _mm512_fmadd_ps(src3, filter0, sums[3]);
                __m512 filter1 = Avx512f::Load<align>(filter + 1 * F);
                __m512 src4 = Avx512f::Load<align>(src + 4 * F);
                sums[0] = _mm512_fmadd_ps(src1, filter1, sums[0]);
                sums[1] = _mm512_fmadd_ps(src2, filter1, sums[1]);
                sums[2] = _mm512_fmadd_ps(src3, filter1, sums[2]);
                sums[3] = _mm512_fmadd_ps(src4, filter1, sums[3]);
            }

            template <bool align> void FilterHx16(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                const size_t step = 16;
                size_t alignedWidth = AlignLo(width, 4);
                size_t alignedSize = AlignLo(size, 2);
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedWidth; col += 4)
                    {
                        __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                        const float * s = src + col * step;
                        size_t i = 0;
                        for (; i < alignedSize; i += DF)
                            FilterHx4x16x2<align>(s + i, filter + i, sums);
                        for (; i < size; i += F)
                            FilterHx4x16<align>(s + i, filter + i, sums);
                        _mm_storeu_ps(dst + col, Avx512f::Extract4Sums(sums));
                    }
                    for (; col < width; ++col)
                    {
                        __m512 sum = _mm512_setzero_ps();
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += F)
                            FilterHx1x16<align>(s + i, filter + i, sum);
                        dst[col] = Avx512f::ExtractSum(sum);
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <bool align> void FilterH(const float * src, size_t srcStride, size_t width, size_t height, size_t step, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                if (step == 16)
                    FilterHx16<align>(src, srcStride, width, height, filter, size, dst, dstStride);
                else
                {
                    if (size & 1)
                        FilterHx8o<align>(src, srcStride, width, height, filter, size, dst, dstStride);
                    else
                        FilterHx8e<align>(src, srcStride, width, height, filter, size, dst, dstStride);
                }
            }

            template <bool srcAlign, bool dstAlign, UpdateType update, bool mask> static SIMD_INLINE void FilterV(const float * src, size_t stride, const __m512 * filter, size_t size, float * dst, __mmask16 tail = -1)
            {
                __m512 sum = _mm512_setzero_ps();
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = _mm512_fmadd_ps(Avx512f::Load<srcAlign>(src), filter[i], sum);
                Avx512f::Update<update, dstAlign, mask>(dst, sum, tail);
            }

            template <UpdateType update, bool align> void FilterV(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm512_set1_ps(filter[i]);

                size_t alignedWidth = AlignLo(width, F);
                __mmask16 tailMask = TailMask16(width - alignedWidth);
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedWidth; col += F)
                        FilterV<true, align, update, false>(src + col, srcStride, _filter.data, size, dst + col);
                    if (col < width)
                        FilterV<true, align, update, true>(src + col, srcStride, _filter.data, size, dst + col, tailMask);
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

        const __m512i K64_15 = SIMD_MM512_SET1_EPI64(15);
        const __m512i K32_64_TO_32_2 = SIMD_MM512_SETR_EPI32(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);

        class HogLiteMaskCreater
        {
            typedef Simd::Array<uint32_t> Ints;
            Ints _sums[8];
            size_t _dstWidth, _alignedDstWidth, _dstHeight;
            __mmask16 _dstWidthTail;

            void Init(size_t srcWidth, size_t srcHeight, size_t scale, size_t size)
            {
                _dstWidth = srcWidth * scale + size - scale;
                _alignedDstWidth = AlignLo(_dstWidth, F);
                _dstWidthTail = TailMask16(_dstWidth - _alignedDstWidth);
                _dstHeight = srcHeight * scale + size - scale;
                size_t sumSize = AlignHi(_dstWidth, F) + 2 * F;
                for (size_t i = 0; i < 8; ++i)
                    _sums[i].Resize(sumSize, true);
            }

            template<size_t step, bool align, bool mask> SIMD_INLINE void SetDstRow(const uint32_t * sum0, const uint32_t * sum1, uint32_t * dst, __mmask16 tail = -1)
            {
                __m512i s00 = Load<false>(sum0 - step);
                __m512i s10 = Load<false>(sum1 - step);
                __m512i s01 = Load<true>(sum0);
                __m512i s11 = Load<true>(sum1);
                __m512i sum = _mm512_sub_epi32(_mm512_sub_epi32(s11, s10), _mm512_sub_epi32(s01, s00));
                __m512i value = _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(sum, K_ZERO));
                Store<align, mask>(dst, value, tail);
            }

            template<size_t step> SIMD_INLINE void SetDstRow(const uint32_t * sum0, const uint32_t * sum1, uint32_t * dst)
            {
                size_t dstCol = 0;
                for (; dstCol < _alignedDstWidth; dstCol += F)
                    SetDstRow<step, false, false>(sum0 + dstCol, sum1 + dstCol, dst + dstCol);
                if (dstCol < _dstWidth)
                    SetDstRow<step, false, true>(sum0 + dstCol, sum1 + dstCol, dst + dstCol, _dstWidthTail);
            }

            void CreateMask7x7x1(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, uint32_t * dst, size_t dstStride)
            {
                size_t alignedSrcWidth = AlignLo(srcWidth, F);
                __m512 _threshold = _mm512_set1_ps(*threshold);
                for (size_t row = 0; row < srcHeight; ++row)
                {
                    uint32_t * sum0 = _sums[(row + 0) & 7].data + F;
                    uint32_t * sum6 = _sums[(row + 6) & 7].data + F;
                    uint32_t * sum7 = _sums[(row + 7) & 7].data + F;

                    __m512i _rowSums = K_ZERO;
                    size_t col = 0;
                    for (; col < alignedSrcWidth; col += F)
                    {
                        __mmask16 mmask = _mm512_cmp_ps_mask(Avx512f::Load<false>(src + col), _threshold, _CMP_GT_OQ);

                        __mmask64 lo = ((mmask & 0xFF) * 0x0101010101010101ull) & 0xFF7F3F1F0F070301ull;
                        _rowSums = _mm512_add_epi32(_rowSums, _mm512_sad_epu8(_mm512_movm_epi8(lo), K_ZERO));
                        _mm256_storeu_si256((__m256i*)(sum7 + col + 00), _mm256_add_epi32(_mm512_cvtepi64_epi32(_rowSums), _mm256_loadu_si256((__m256i*)(sum6 + col + 00))));
                        _rowSums = _mm512_permutexvar_epi64(K64_15, _rowSums);

                        __mmask64 hi = ((mmask >> 8) * 0x0101010101010101ull) & 0xFF7F3F1F0F070301ull;
                        _rowSums = _mm512_add_epi32(_rowSums, _mm512_sad_epu8(_mm512_movm_epi8(hi), K_ZERO));
                        _mm256_storeu_si256((__m256i*)(sum7 + col + HF), _mm256_add_epi32(_mm512_cvtepi64_epi32(_rowSums), _mm256_loadu_si256((__m256i*)(sum6 + col + HF))));
                        _rowSums = _mm512_permutexvar_epi64(K64_15, _rowSums);
                    }
                    uint32_t rowSum = sum7[col - 1] - sum6[col - 1];
                    for (; col < srcWidth; ++col)
                    {
                        if (src[col] > *threshold)
                            rowSum += 0xFF;
                        sum7[col] = rowSum + sum6[col];
                    }
                    for (; col < _dstWidth; ++col)
                        sum7[col] = sum7[col - 1];

                    SetDstRow<7>(sum0, sum7, dst);

                    src += srcStride;
                    dst += dstStride;
                }

                for (size_t row = srcHeight; row < _dstHeight; ++row)
                {
                    uint32_t * sum0 = _sums[(row + 0) & 7].data + F;
                    uint32_t * sum7 = _sums[(srcHeight - 1 + 7) & 7].data + F;
                    SetDstRow<7>(sum0, sum7, dst);
                    dst += dstStride;
                }
            }

            void CreateMask7x7x2(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, uint32_t * dst, size_t dstStride)
            {
                size_t alignedSrcWidth = AlignLo(srcWidth, F);
                __m512 _threshold = _mm512_set1_ps(*threshold);
                for (size_t srcRow = 0; srcRow < srcHeight; ++srcRow)
                {
                    size_t dstRow = srcRow * 2;
                    uint32_t * sum0 = _sums[(srcRow + 0) & 7].data + F;
                    uint32_t * sum1 = _sums[(srcRow + 1) & 7].data + F;
                    uint32_t * sum3 = _sums[(srcRow + 3) & 7].data + F;
                    uint32_t * sum4 = _sums[(srcRow + 4) & 7].data + F;

                    __m512i _rowSums = K_ZERO;
                    size_t srcCol = 0, dstCol = 0;
                    for (; srcCol < alignedSrcWidth; srcCol += F, dstCol += DF)
                    {
                        __mmask16 mmask = _mm512_cmp_ps_mask(Avx512f::Load<false>(src + srcCol), _threshold, _CMP_GT_OQ);

                        __mmask64 lo = ((mmask & 0xFF) * 0x0101010101010101ull) & 0xFF7F3F1F0F070301ull;
                        _rowSums = _mm512_add_epi32(_rowSums, _mm512_sad_epu8(_mm512_movm_epi8(lo), K_ZERO));
                        Store<false>(sum4 + dstCol + 0, _mm512_add_epi32(_mm512_permutexvar_epi32(K32_64_TO_32_2, _rowSums), Load<false>(sum3 + dstCol + 0)));
                        _rowSums = _mm512_permutexvar_epi64(K64_15, _rowSums);

                        __mmask64 hi = ((mmask >> 8) * 0x0101010101010101ull) & 0xFF7F3F1F0F070301ull;
                        _rowSums = _mm512_add_epi32(_rowSums, _mm512_sad_epu8(_mm512_movm_epi8(hi), K_ZERO));
                        Store<false>(sum4 + dstCol + F, _mm512_add_epi32(_mm512_permutexvar_epi32(K32_64_TO_32_2, _rowSums), Load<false>(sum3 + dstCol + F)));
                        _rowSums = _mm512_permutexvar_epi64(K64_15, _rowSums);
                    }
                    uint32_t rowSum = sum4[dstCol - 1] - sum3[dstCol - 1];
                    for (; srcCol < srcWidth; srcCol += 1, dstCol += 2)
                    {
                        if (src[srcCol] > *threshold)
                            rowSum += 0xFF;
                        sum4[dstCol + 0] = rowSum + sum3[dstCol + 0];
                        sum4[dstCol + 1] = rowSum + sum3[dstCol + 1];
                    }
                    for (; dstCol < _dstWidth; ++dstCol)
                        sum4[dstCol] = sum4[dstCol - 1];

                    SetDstRow<7>(sum0, sum4, dst);
                    dst += dstStride;
                    SetDstRow<7>(sum1, sum4, dst);
                    dst += dstStride;
                    src += srcStride;
                }

                uint32_t * sum0 = _sums[(srcHeight + 0) & 7].data + F;
                uint32_t * sum1 = _sums[(srcHeight + 1) & 7].data + F;
                uint32_t * sum2 = _sums[(srcHeight + 2) & 7].data + F;
                uint32_t * sum3 = _sums[(srcHeight + 3) & 7].data + F;
                SetDstRow<7>(sum0, sum3, dst + 0 * dstStride);
                SetDstRow<7>(sum1, sum3, dst + 1 * dstStride);
                SetDstRow<7>(sum1, sum3, dst + 2 * dstStride);
                SetDstRow<7>(sum2, sum3, dst + 3 * dstStride);
                SetDstRow<7>(sum2, sum3, dst + 4 * dstStride);
            }

        public:

            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride)
            {
                if (size == 7 && (scale == 1 || scale == 2))
                {
                    Init(srcWidth, srcHeight, scale, size);
                    if (scale == 1)
                        CreateMask7x7x1(src, srcStride, srcWidth, srcHeight, threshold, dst, dstStride);
                    else
                        CreateMask7x7x2(src, srcStride, srcWidth, srcHeight, threshold, dst, dstStride);
                }
                else
                    Base::HogLiteCreateMask(src, srcStride, srcWidth, srcHeight, threshold, scale, size, dst, dstStride);
            }
        };

        void HogLiteCreateMask(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride)
        {
            HogLiteMaskCreater maskCreater;
            maskCreater.Run(src, srcStride, srcWidth, srcHeight, threshold, scale, size, dst, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}


