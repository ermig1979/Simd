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
#include "Simd/SimdMemory.h"
#include "Simd/SimdRecursiveBilateralFilter.h"
#include "Simd/SimdPerformance.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        typedef Base::RbfAlg RbfAlg;

        template<size_t channels> void RowRanges(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst);

        SIMD_INLINE void Ranges1(const uint8_t* src0, const uint8_t* src1, const float* ranges, float* dst)
        {
            __m128i s0 = _mm_loadu_si128((__m128i*)src0);
            __m128i s1 = _mm_loadu_si128((__m128i*)src1);
            __m128i d = _mm_sub_epi8(_mm_max_epu8(s0, s1), _mm_min_epu8(s0, s1));
            SIMD_ALIGNED(A) uint8_t diff[A];
            _mm_storeu_si128((__m128i*)diff, d);
            for (size_t i = 0; i < A; ++i)
                dst[i] = ranges[diff[i]];
        }

        template<> void RowRanges<1>(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
        {
            size_t widthA = AlignLo(width, A), x = 0;
            for (; x < widthA; x += A)
                Ranges1(src0 + x, src1 + x, ranges, dst + x);
            if(widthA < width)
            {
                x = width - A;
                Ranges1(src0 + x, src1 + x, ranges, dst + x);
            }
        }

        SIMD_INLINE void Ranges2(const uint8_t* src0, const uint8_t* src1, const float* ranges, float* dst)
        {
            __m128i s0 = _mm_loadu_si128((__m128i*)src0);
            __m128i s1 = _mm_loadu_si128((__m128i*)src1);
            __m128i d8 = _mm_sub_epi8(_mm_max_epu8(s0, s1), _mm_min_epu8(s0, s1));
            __m128i d16_0 = _mm_and_si128(d8, K16_00FF);
            __m128i d16_1 = _mm_and_si128(_mm_srli_si128(d8, 1), K16_00FF);
            __m128i a16 = _mm_srli_epi16(_mm_add_epi16(d16_0, d16_1), 1);
            SIMD_ALIGNED(A) uint16_t diff[HA];
            _mm_storeu_si128((__m128i*)diff, a16);
            for (size_t i = 0; i < HA; ++i)
                dst[i] = ranges[diff[i]];
        }

        template<> void RowRanges<2>(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
        {
            size_t widthHA = AlignLo(width, HA), x = 0, o = 0;
            for (; x < widthHA; x += HA, o += A)
                Ranges2(src0 + o, src1 + o, ranges, dst + x);
            if (widthHA < width)
            {
                x = width - HA, o = x * 2;
                Ranges2(src0 + o, src1 + o, ranges, dst + x);
            }
        }

        SIMD_INLINE void Ranges3(const uint8_t* src0, const uint8_t* src1, const float* ranges, float* dst)
        {
            static const __m128i K0 = SIMD_MM_SETR_EPI8(0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1);
            static const __m128i K1 = SIMD_MM_SETR_EPI8(0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xa, -1, -1, -1);
            static const __m128i K2 = SIMD_MM_SETR_EPI8(0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xb, -1, -1, -1);
            __m128i s0 = _mm_loadu_si128((__m128i*)src0);
            __m128i s1 = _mm_loadu_si128((__m128i*)src1);
            __m128i d8 = _mm_sub_epi8(_mm_max_epu8(s0, s1), _mm_min_epu8(s0, s1));
            __m128i d32_0 = _mm_shuffle_epi8(d8, K0);
            __m128i d32_1 = _mm_shuffle_epi8(d8, K1);
            __m128i d32_2 = _mm_shuffle_epi8(d8, K2);
            __m128i a32 = _mm_srli_epi16(_mm_add_epi32(_mm_add_epi32(d32_0, d32_1), _mm_add_epi32(d32_1, d32_2)), 2);
            SIMD_ALIGNED(A) uint32_t diff[F];
            _mm_storeu_si128((__m128i*)diff, a32);
            for (size_t i = 0; i < F; ++i)
                dst[i] = ranges[diff[i]];
        }

        template<> void RowRanges<3>(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
        {
            size_t widthF = AlignLo(width, F), x = 0, o = 0;
            for (; x < widthF; x += F, o += F * 3)
                Ranges3(src0 + o, src1 + o, ranges, dst + x);
            if (widthF < width)
            {
                x = width - F, o = x * 3;
                Ranges3(src0 + o, src1 + o, ranges, dst + x);
            }
        }

        SIMD_INLINE void Ranges4(const uint8_t* src0, const uint8_t* src1, const float* ranges, float* dst)
        {
            static const __m128i K0 = SIMD_MM_SETR_EPI8(0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xc, -1, -1, -1);
            static const __m128i K1 = SIMD_MM_SETR_EPI8(0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xd, -1, -1, -1);
            static const __m128i K2 = SIMD_MM_SETR_EPI8(0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xa, -1, -1, -1, 0xe, -1, -1, -1);
            __m128i s0 = _mm_loadu_si128((__m128i*)src0);
            __m128i s1 = _mm_loadu_si128((__m128i*)src1);
            __m128i d8 = _mm_sub_epi8(_mm_max_epu8(s0, s1), _mm_min_epu8(s0, s1));
            __m128i d32_0 = _mm_shuffle_epi8(d8, K0);
            __m128i d32_1 = _mm_shuffle_epi8(d8, K1);
            __m128i d32_2 = _mm_shuffle_epi8(d8, K2);
            __m128i a32 = _mm_srli_epi16(_mm_add_epi32(_mm_add_epi32(d32_0, d32_1), _mm_add_epi32(d32_1, d32_2)), 2);
            SIMD_ALIGNED(A) uint32_t diff[F];
            _mm_storeu_si128((__m128i*)diff, a32);
            for (size_t i = 0; i < F; ++i)
                dst[i] = ranges[diff[i]];
        }

        template<> void RowRanges<4>(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
        {
            size_t widthF = AlignLo(width, F), x = 0, o = 0;
            for (; x < widthF; x += F, o += A)
                Ranges4(src0 + o, src1 + o, ranges, dst + x);
            if (widthF < width)
            {
                x = width - F, o = x * 4;
                Ranges4(src0 + o, src1 + o, ranges, dst + x);
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t channels> SIMD_INLINE void SetOut(const float* bc, const float* bf, const float* ec, const float* ef, size_t width, uint8_t* dst)
        {
            for (size_t x = 0; x < width; x++)
            {
                float factor = 1.f / (bf[x] + ef[x]);
                for (size_t c = 0; c < channels; c++)
                    dst[c] = uint8_t(factor * (bc[c] + ec[c]));
                bc += channels;
                ec += channels;
                dst += channels;
            }
        }

        template<> SIMD_INLINE void SetOut<1>(const float* bc, const float* bf, const float* ec, const float* ef, size_t width, uint8_t* dst)
        {
            size_t widthF = AlignLo(width, F), x = 0;
            __m128 _1 = _mm_set1_ps(1.0f);
            for (; x < widthF; x += F)
            {
                __m128 _bf = _mm_loadu_ps(bf + x);
                __m128 _ef = _mm_loadu_ps(ef + x);
                __m128 factor = _mm_div_ps(_1, _mm_add_ps(_bf, _ef));
                __m128 _bc = _mm_loadu_ps(bc + x);
                __m128 _ec = _mm_loadu_ps(ec + x);
                __m128 f32 = _mm_mul_ps(factor, _mm_add_ps(_bc, _ec));
                __m128i i32 = _mm_cvtps_epi32(_mm_floor_ps(f32));
                __m128i u8 = _mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO);
                *(int32_t*)(dst + x) = _mm_cvtsi128_si32(u8);
            }
            for (; x < width; x++)
            {
                float factor = 1.0f / (bf[x] + ef[x]);
                dst[x] = uint8_t(factor * (bc[x] + ec[x]));
            }
        }

        template<> SIMD_INLINE void SetOut<2>(const float* bc, const float* bf, const float* ec, const float* ef, size_t width, uint8_t* dst)
        {
            size_t widthHF = AlignLo(width, 2), x = 0, o = 0;
            __m128 _1 = _mm_set1_ps(1.0f);
            for (; x < widthHF; x += HF, o += F)
            {
                __m128 _bf = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_loadl_pi(_mm_setzero_ps(), (__m64*)(bf + x))), 0x50));
                __m128 _ef = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(_mm_loadl_pi(_mm_setzero_ps(), (__m64*)(ef + x))), 0x50));
                __m128 factor = _mm_div_ps(_1, _mm_add_ps(_bf, _ef));
                __m128 _bc = _mm_loadu_ps(bc + o);
                __m128 _ec = _mm_loadu_ps(ec + o);
                __m128 f32 = _mm_mul_ps(factor, _mm_add_ps(_bc, _ec));
                __m128i i32 = _mm_cvtps_epi32(_mm_floor_ps(f32));
                __m128i u8 = _mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO);
                *(int32_t*)(dst + o) = _mm_cvtsi128_si32(u8);
            }
            for (; x < width; x++, o += 2)
            {
                float factor = 1.0f / (bf[x] + ef[x]);
                dst[o + 0] = uint8_t(factor * (bc[o + 0] + ec[o + 0]));
                dst[o + 1] = uint8_t(factor * (bc[o + 1] + ec[o + 1]));
            }
        }

        template<> SIMD_INLINE void SetOut<3>(const float* bc, const float* bf, const float* ec, const float* ef, size_t width, uint8_t* dst)
        {
            size_t width1 = width - 1, x = 0, o = 0;
            __m128 _1 = _mm_set1_ps(1.0f);
            for (; x < width1; x += 1, o += 3)
            {
                __m128 _bf = _mm_set1_ps(bf[x]);
                __m128 _ef = _mm_set1_ps(ef[x]);
                __m128 factor = _mm_div_ps(_1, _mm_add_ps(_bf, _ef));
                __m128 _bc = _mm_loadu_ps(bc + o);
                __m128 _ec = _mm_loadu_ps(ec + o);
                __m128 f32 = _mm_mul_ps(factor, _mm_add_ps(_bc, _ec));
                __m128i i32 = _mm_cvtps_epi32(_mm_floor_ps(f32));
                __m128i u8 = _mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO);
                *(int32_t*)(dst + o) = _mm_cvtsi128_si32(u8);
            }
            float factor = 1.0f / (bf[x] + ef[x]);
            dst[o + 0] = uint8_t(factor * (bc[o + 0] + ec[o + 0]));
            dst[o + 1] = uint8_t(factor * (bc[o + 1] + ec[o + 1]));
            dst[o + 2] = uint8_t(factor * (bc[o + 2] + ec[o + 2]));
        }

        template<> SIMD_INLINE void SetOut<4>(const float* bc, const float* bf, const float* ec, const float* ef, size_t width, uint8_t* dst)
        {
            __m128 _1 = _mm_set1_ps(1.0f);
            for (size_t x = 0, o = 0; x < width; x += 1, o += 4)
            {
                __m128 _bf = _mm_set1_ps(bf[x]);
                __m128 _ef = _mm_set1_ps(ef[x]);
                __m128 factor = _mm_div_ps(_1, _mm_add_ps(_bf, _ef));
                __m128 _bc = _mm_loadu_ps(bc + o);
                __m128 _ec = _mm_loadu_ps(ec + o);
                __m128 f32 = _mm_mul_ps(factor, _mm_add_ps(_bc, _ec));
                __m128i i32 = _mm_cvtps_epi32(_mm_floor_ps(f32));
                __m128i u8 = _mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO);
                *(int32_t*)(dst + o) = _mm_cvtsi128_si32(u8);
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t channels> void HorFilter(const RbfParam& p, RbfAlg& a, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t size = p.width * channels, cLast = size - 1, fLast = p.width - 1;
            for (size_t y = 0; y < p.height; y++)
            {
                const uint8_t* sl = src, * sr = src + cLast;
                float* lc = a.cb0.data, * rc = a.cb1.data + cLast;
                float* lf = a.fb0.data, * rf = a.fb1.data + fLast;
                *lf++ = 1.f;
                *rf-- = 1.f;
                for (int c = 0; c < channels; c++)
                {
                    *lc++ = *sl++;
                    *rc-- = *sr--;
                }
                RowRanges<channels>(src, src + channels, p.width - 1, a.ranges.data, a.rb0.data + 1);
                for (size_t x = 1; x < p.width; x++)
                {
                    float la = a.rb0[x];
                    float ra = a.rb0[p.width - x];
                    *lf++ = a.alpha + la * lf[-1];
                    *rf-- = a.alpha + ra * rf[+1];
                    for (int c = 0; c < channels; c++)
                    {
                        *lc++ = (a.alpha * (*sl++) + la * lc[-channels]);
                        *rc-- = (a.alpha * (*sr--) + ra * rc[+channels]);
                    }
                }
                SetOut<channels>(a.cb0.data, a.fb0.data, a.cb1.data, a.fb1.data, p.width, dst);
                src += srcStride;
                dst += dstStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t channels> void VerSetEdge(const uint8_t* src, size_t width, float* factor, float* colors)
        {
            size_t widthF = AlignLo(width, F);
            size_t x = 0;
            __m128 _1 = _mm_set1_ps(1.0f);
            for (; x < widthF; x += F)
                _mm_storeu_ps(factor + x, _1);
            for (; x < width; x++)
                factor[x] = 1.0f;

            size_t size = width * channels, sizeF = AlignLo(size, F);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m128i i32 = _mm_cvtepu8_epi32( _mm_cvtsi32_si128(*(int32_t*)(src + i)));
                _mm_storeu_ps(colors + i, _mm_cvtepi32_ps(i32));
            }
            for (; i < size; i++)
                colors[i] = src[i];
        }

        //-----------------------------------------------------------------------------------------

        template<size_t channels> void VerSetMain(const uint8_t* hor, size_t width,
            float alpha, const float* ranges, const float* pf, const float* pc, float* cf, float* cc)
        {
            for (size_t x = 0, o = 0; x < width; x++)
            {
                cf[x] = alpha + ranges[x] * pf[x];
                for (size_t e = o + channels; o < e; o++)
                    cc[o] = alpha * hor[o] + ranges[x] * pc[o];
            }
        }

        template<size_t channels> void VerFilter(const RbfParam& p, RbfAlg& a, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t size = p.width * channels, srcTail = srcStride - size, dstTail = dstStride - size;
            float* dcb = a.cb0.data, * dfb = a.fb0.data, * ucb = a.cb1.data, * ufb = a.fb1.data;

            const uint8_t* suc = src + srcStride * (p.height - 1);
            const uint8_t* duc = dst + dstStride * (p.height - 1);
            float* uf = ufb + p.width * (p.height - 1);
            float* uc = ucb + size * (p.height - 1);
            VerSetEdge<channels>(duc, p.width, uf, uc);
            for (size_t y = 1; y < p.height; y++)
            {
                duc -= dstStride;
                suc -= srcStride;
                uf -= p.width;
                uc -= size;
                RowRanges<channels>(suc, suc + srcStride, p.width, a.ranges.data, a.rb0.data);
                VerSetMain<channels>(duc, p.width, a.alpha, a.rb0.data, uf + p.width, uc + size, uf, uc);
            }

            VerSetEdge<channels>(dst, p.width, dfb, dcb);
            SetOut<channels>(dcb, dfb, ucb, ufb, p.width, dst);
            for (size_t y = 1; y < p.height; y++)
            {
                src += srcStride;
                dst += dstStride;
                float* dc = dcb + (y & 1) * size;
                float* df = dfb + (y & 1) * p.width;
                const float* dpc = dcb + ((y - 1) & 1) * size;
                const float* dpf = dfb + ((y - 1) & 1) * p.width;
                RowRanges<channels>(src, src - srcStride, p.width, a.ranges.data, a.rb0.data);
                VerSetMain<channels>(dst, p.width, a.alpha, a.rb0.data, dpf, dpc, df, dc);
                SetOut<channels>(dc, df, ucb + y * size, ufb + y * p.width, p.width, dst);
            }
        }

		//-----------------------------------------------------------------------------------------

        RecursiveBilateralFilterDefault::RecursiveBilateralFilterDefault(const RbfParam& param)
            :Base::RecursiveBilateralFilterDefault(param)
        {
            if (_param.width * _param.channels >= A)
            {
                switch (_param.channels)
                {
                case 1: _hFilter = HorFilter<1>; _vFilter = VerFilter<1>; break;
                case 2: _hFilter = HorFilter<2>; _vFilter = VerFilter<2>; break;
                case 3: _hFilter = HorFilter<3>; _vFilter = VerFilter<3>; break;
                case 4: _hFilter = HorFilter<4>; _vFilter = VerFilter<4>; break;
                default:
                    assert(0);
                }
            }
        }

        void RecursiveBilateralFilterDefault::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            InitBuf();
            _hFilter(_param, _alg, src, srcStride, dst, dstStride);
            _vFilter(_param, _alg, src, srcStride, dst, dstStride);
        }

        //-----------------------------------------------------------------------------------------

        void* RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, const float* sigmaSpatial, const float* sigmaRange)
        {
            RbfParam param(width, height, channels, sigmaSpatial, sigmaRange, sizeof(void*));
            if (!param.Valid())
                return NULL;
            return new RecursiveBilateralFilterDefault(param);
        }
    }
#endif
}

