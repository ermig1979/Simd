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
#include "Simd/SimdStore.h"
#include "Simd/SimdRecursiveBilateralFilter.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdFmadd.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        typedef RecursiveBilateralFilter::FilterPtr FilterPtr;

        SIMD_INLINE __m256i AbsDiff8u(const uint8_t* src0, const uint8_t* src1)
        {
            __m256i s0 = _mm256_loadu_si256((__m256i*)src0);
            __m256i s1 = _mm256_loadu_si256((__m256i*)src1);
            return _mm256_sub_epi8(_mm256_max_epu8(s0, s1), _mm256_min_epu8(s0, s1));
        }

        template<RbfDiffType type> SIMD_INLINE __m256i Diff(__m256i ch0, __m256i ch1)
        {
            switch (type)
            {
            case RbfDiffAvg: return _mm256_avg_epu8(ch0, ch1);
            case RbfDiffMax: return _mm256_max_epu8(ch0, ch1);
            case RbfDiffSum: return _mm256_adds_epu8(ch0, ch1);
            default:
                assert(0); return _mm256_setzero_si256();
            }
        }

        template<RbfDiffType type> SIMD_INLINE __m256i Diff(__m256i ch0, __m256i ch1, __m256i ch2)
        {
            switch (type)
            {
            case RbfDiffAvg: return _mm256_avg_epu8(ch1, _mm256_avg_epu8(ch0, ch2));
            case RbfDiffMax: return _mm256_max_epu8(_mm256_max_epu8(ch0, ch1), ch2);
            case RbfDiffSum: return _mm256_adds_epu8(ch0, _mm256_adds_epu8(ch1, ch2));
            default:
                assert(0); return _mm256_setzero_si256();
            }
        }

        //-----------------------------------------------------------------------------------------

        template<RbfDiffType type> SIMD_INLINE __m128i Diff3(__m128i src)
        {
            static const __m128i K0 = SIMD_MM_SETR_EPI8(0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1);
            static const __m128i K1 = SIMD_MM_SETR_EPI8(0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xa, -1, -1, -1);
            static const __m128i K2 = SIMD_MM_SETR_EPI8(0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xb, -1, -1, -1);
            return Diff<type>(_mm_shuffle_epi8(src, K0), _mm_shuffle_epi8(src, K1), _mm_shuffle_epi8(src, K2));
        }

        template<RbfDiffType type> SIMD_INLINE __m128i Diff4(__m128i src)
        {
            static const __m128i K0 = SIMD_MM_SETR_EPI8(0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xc, -1, -1, -1);
            static const __m128i K1 = SIMD_MM_SETR_EPI8(0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xd, -1, -1, -1);
            static const __m128i K2 = SIMD_MM_SETR_EPI8(0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xa, -1, -1, -1, 0xe, -1, -1, -1);
            return Diff<type>(_mm_shuffle_epi8(src, K0), _mm_shuffle_epi8(src, K1), _mm_shuffle_epi8(src, K2));
        }

        template<size_t channels, RbfDiffType type> SIMD_INLINE void RowDiff(const uint8_t* src0, const uint8_t* src1, size_t width, uint8_t* dst)
        {
            switch (channels)
            {
            case 1:
            {
                for (size_t x = 0; x < width; x += A)
                    _mm256_storeu_si256((__m256i*)(dst + x), AbsDiff8u(src0 + x, src1 + x));
                break;
            }
            //case 2:
            //{
            //    for (size_t x = 0, c = 0; x < width; x += A, c += 2 * A)
            //    {
            //        __m128i ad0 = AbsDiff8u(src0 + c + 0, src1 + c + 0);
            //        __m128i d0 = Diff<type>(_mm_and_si128(ad0, K16_00FF), _mm_and_si128(_mm_srli_si128(ad0, 1), K16_00FF));
            //        __m128i ad1 = AbsDiff8u(src0 + c + A, src1 + c + A);
            //        __m128i d1 = Diff<type>(_mm_and_si128(ad1, K16_00FF), _mm_and_si128(_mm_srli_si128(ad1, 1), K16_00FF));
            //        _mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi16(d0, d1));
            //    }
            //    break;
            //}
            //case 3:
            //{
            //    for (size_t x = 0, c = 0; x < width; x += A, c += 3 * A)
            //    {
            //        __m128i d0 = Diff3<type>(AbsDiff8u(src0 + c + 0 * 12, src1 + c + 0 * 12));
            //        __m128i d1 = Diff3<type>(AbsDiff8u(src0 + c + 1 * 12, src1 + c + 1 * 12));
            //        __m128i d2 = Diff3<type>(AbsDiff8u(src0 + c + 2 * 12, src1 + c + 2 * 12));
            //        __m128i d3 = Diff3<type>(AbsDiff8u(src0 + c + 3 * 12, src1 + c + 3 * 12));
            //        _mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
            //    }
            //    break;
            //}
            //case 4:
            //{
            //    for (size_t x = 0, c = 0; x < width; x += A, c += 4 * A)
            //    {
            //        __m128i d0 = Diff4<type>(AbsDiff8u(src0 + c + 0 * A, src1 + c + 0 * A));
            //        __m128i d1 = Diff4<type>(AbsDiff8u(src0 + c + 1 * A, src1 + c + 1 * A));
            //        __m128i d2 = Diff4<type>(AbsDiff8u(src0 + c + 2 * A, src1 + c + 2 * A));
            //        __m128i d3 = Diff4<type>(AbsDiff8u(src0 + c + 3 * A, src1 + c + 3 * A));
            //        _mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)));
            //    }
            //    break;
            //}
            default:
                for (size_t x = 0, o = 0; x < width; x += 1, o += channels)
                    dst[x] = Base::Diff<channels, type>(src0 + o, src1 + o);
            }
        }

        //-----------------------------------------------------------------------------------------

        namespace Fast
        {
            template<int dir> SIMD_INLINE void Set(int value, uint8_t* dst);

            template<> SIMD_INLINE void Set<+1>(int value, uint8_t* dst)
            {
                dst[0] = uint8_t(value);
            }

            template<> SIMD_INLINE void Set<-1>(int value, uint8_t* dst)
            {
                dst[0] = uint8_t((value + dst[0] + 1) / 2);
            }

            template<int dir> SIMD_INLINE void Set16(__m128i value, uint8_t* dst);

            template<> SIMD_INLINE void Set16<+1>(__m128i value, uint8_t* dst)
            {
                _mm_storeu_si128((__m128i*)dst, value);
            }

            template<> SIMD_INLINE void Set16<-1>(__m128i value, uint8_t* dst)
            {
                _mm_storeu_si128((__m128i*)dst, _mm_avg_epu8(_mm_loadu_si128((__m128i*)dst), value));
            }

            template<int dir> SIMD_INLINE void Set32(__m256i value, uint8_t* dst);

            template<> SIMD_INLINE void Set32<+1>(__m256i value, uint8_t* dst)
            {
                _mm256_storeu_si256((__m256i*)dst, value);
            }

            template<> SIMD_INLINE void Set32<-1>(__m256i value, uint8_t* dst)
            {
                _mm256_storeu_si256((__m256i*)dst, _mm256_avg_epu8(_mm256_loadu_si256((__m256i*)dst), value));
            }

            //-----------------------------------------------------------------------------------------

            template<size_t channels, int dir> void VerEdge(const uint8_t* src, size_t width, float* factor, float* colors, uint8_t* dst)
            {
                __m256 _1 = _mm256_set1_ps(1.0f);
                size_t widthF = AlignLo(width, F), x = 0;
                for (; x < widthF; x += F)
                    _mm256_storeu_ps(factor + x, _1);
                for (; x < width; x++)
                    factor[x] = 1.0f;

                size_t size = width * channels, sizeF = AlignLo(size, F), i = 0;
                for (; i < sizeF; i += F)
                {
                    __m256i i32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src + i)));
                    _mm256_storeu_ps(colors + i, _mm256_cvtepi32_ps(i32));
                }
                for (; i < size; i++)
                    colors[i] = src[i];

                size_t sizeA = AlignLo(size, A);
                for (i = 0; i < sizeA; i += A)
                    Set32<dir>(_mm256_loadu_si256((__m256i*)(src + i)), dst + i);
                for (; i < size; i += 1)
                    Set<dir>(src[i], dst + i);
            }

            //-----------------------------------------------------------------------------------------

            template<size_t channels> struct VerMain
            {
                template<int dir, bool nofma> static void Run(const uint8_t* src, const uint8_t* diff, size_t width, float alpha,
                    const float* ranges, float* factor, float* colors, uint8_t* dst)
                {
                    for (size_t x = 0, o = 0; x < width; x++)
                    {
                        float range = ranges[diff[x]];
                        factor[x] = alpha + range * factor[x];
                        for (size_t e = o + channels; o < e; o++)
                        {
                            colors[o] = alpha * src[o] + range * colors[o];
                            Set<dir>(int(colors[o] / factor[x]), dst + o);
                        }
                    }
                }
            };

            template<> struct VerMain<1>
            {
                template<int offs, bool nofma> static SIMD_INLINE __m128i RunHF(__m128i src, const uint8_t* diff,
                    __m128 alpha, const float* ranges, float* factor, float* colors)
                {
                    __m128 _range = _mm_i32gather_ps(ranges, _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)(diff + offs))), 4);
                    __m128 _factor = Fmadd<nofma>(_range, _mm_loadu_ps(factor + offs), alpha);
                    __m128 _src = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(src, offs)));
                    __m128 _colors = Fmadd<nofma>(alpha, _src, _range, _mm_loadu_ps(colors + offs));
                    _mm_storeu_ps(factor + offs, _factor);
                    _mm_storeu_ps(colors + offs, _colors);
                    return _mm_cvtps_epi32(_mm_floor_ps(_mm_div_ps(_colors, _factor)));
                }

                template<int part, bool nofma> static SIMD_INLINE __m256i RunF(__m256i src, const uint8_t* diff,
                    __m256 alpha, const float* ranges, float* factor, float* colors)
                {
                    __m256 _range = _mm256_i32gather_ps(ranges, _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(diff + part * F))), 4);
                    __m256 _factor = Fmadd<nofma>(_range, _mm256_loadu_ps(factor + part * F), alpha);
                    __m256 _src = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_permute4x64_epi64(src, 0x55 * part))));
                    __m256 _colors = Fmadd<nofma>(alpha, _src, _range, _mm256_loadu_ps(colors + part * F));
                    _mm256_storeu_ps(factor + part * F, _factor);
                    _mm256_storeu_ps(colors + part * F, _colors);
                    return _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors, _factor)));
                }

                template<int dir, bool nofma> static void Run(const uint8_t* src, const uint8_t* diff, size_t width, float alpha,
                    const float* ranges, float* factor, float* colors, uint8_t* dst)
                {
                    size_t widthA = AlignLo(width, A), widthHA = AlignLo(width, HA), x = 0;
                    __m256 _alpha = _mm256_set1_ps(alpha);
                    for (; x < widthA; x += A)
                    {
                        __m256i _src = _mm256_loadu_si256((__m256i*)(src + x));
                        __m256i d0 = RunF<0, nofma>(_src, diff + x, _alpha, ranges, factor + x, colors + x);
                        __m256i d1 = RunF<1, nofma>(_src, diff + x, _alpha, ranges, factor + x, colors + x);
                        __m256i d2 = RunF<2, nofma>(_src, diff + x, _alpha, ranges, factor + x, colors + x);
                        __m256i d3 = RunF<3, nofma>(_src, diff + x, _alpha, ranges, factor + x, colors + x);
                        Set32<dir>(PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)), dst + x);
                    }
                    for (; x < widthHA; x += HA)
                    {
                        __m128i _src = _mm_loadu_si128((__m128i*)(src + x));
                        __m128i d0 = RunHF<0 * 4, nofma>(_src, diff + x, _mm256_castps256_ps128(_alpha), ranges, factor + x, colors + x);
                        __m128i d1 = RunHF<1 * 4, nofma>(_src, diff + x, _mm256_castps256_ps128(_alpha), ranges, factor + x, colors + x);
                        __m128i d2 = RunHF<2 * 4, nofma>(_src, diff + x, _mm256_castps256_ps128(_alpha), ranges, factor + x, colors + x);
                        __m128i d3 = RunHF<3 * 4, nofma>(_src, diff + x, _mm256_castps256_ps128(_alpha), ranges, factor + x, colors + x);
                        Set16<dir>(_mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)), dst + x);
                    }
                    for (; x < width; x++)
                    {
                        float range = ranges[diff[x]];
                        factor[x] = alpha + range * factor[x];
                        colors[x] = alpha * src[x] + range * colors[x];
                        Set<dir>(int(colors[x] / factor[x]), dst + x);
                    }
                }
            };

            //-----------------------------------------------------------------------------------------

            template<size_t channels, RbfDiffType type, bool nofma> void VerFilter(const RbfParam& p, float* buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
            {
                size_t size = p.width * channels;
                uint8_t* diff = (uint8_t*)(buf + size + p.width);
                VerEdge<channels, +1>(src, p.width, buf + size, buf, dst);
                for (size_t y = 1; y < p.height; y++)
                {
                    src += srcStride;
                    dst += dstStride;
                    RowDiff<channels, type>(src, src - srcStride, p.width, diff);
                    VerMain<channels>::template Run<+1, nofma>(src, diff, p.width, p.alpha, p.ranges, buf + size, buf, dst);
                }
                VerEdge<channels, -1>(src, p.width, buf + size, buf, dst);
                for (size_t y = 1; y < p.height; y++)
                {
                    src -= srcStride;
                    dst -= dstStride;
                    RowDiff<channels, type>(src, src + srcStride, p.width, diff);
                    VerMain<channels>::template Run<-1, nofma>(src, diff, p.width, p.alpha, p.ranges, buf + size, buf, dst);
                }
            }
            
            //-----------------------------------------------------------------------------------------

            template <size_t channels, RbfDiffType type> void Set(const RbfParam& param, FilterPtr& horFilter, FilterPtr& verFilter)
            {
                //horFilter = HorFilter<channels, type>;
                verFilter = FmaAvoid(param.flags) ? VerFilter<channels, type, true> : VerFilter<channels, type, false>;
            }

            template <RbfDiffType type> void Set(const RbfParam& param, FilterPtr& horFilter, FilterPtr& verFilter)
            {
                switch (param.channels)
                {
                case 1: Set<1, type>(param, horFilter, verFilter); break;
                //case 2: Set<2, type>(param, horFilter, verFilter); break;
                //case 3: Set<3, type>(param, horFilter, verFilter); break;
                //case 4: Set<4, type>(param, horFilter, verFilter); break;
                default:
                    assert(0);
                }
            }

            void Set(const RbfParam& param, FilterPtr& horFilter, FilterPtr& verFilter)
            {
                switch (DiffType(param.flags))
                {
                case RbfDiffAvg: Set<RbfDiffAvg>(param, horFilter, verFilter); break;
                case RbfDiffMax: Set<RbfDiffAvg>(param, horFilter, verFilter); break;
                case RbfDiffSum: Set<RbfDiffAvg>(param, horFilter, verFilter); break;
                default:
                    assert(0);
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        RecursiveBilateralFilterFast::RecursiveBilateralFilterFast(const RbfParam& param)
            : Sse41::RecursiveBilateralFilterFast(param)
        {
            Fast::Set(_param, _hFilter, _vFilter);
        }

        //-----------------------------------------------------------------------------------------

        void* RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, 
            const float* sigmaSpatial, const float* sigmaRange, SimdRecursiveBilateralFilterFlags flags)
        {
            RbfParam param(width, height, channels, sigmaSpatial, sigmaRange, flags, A);
            if (!param.Valid())
                return NULL;
            if (Precise(flags))
                return new Sse41::RecursiveBilateralFilterPrecize(param);
            else
                return new RecursiveBilateralFilterFast(param);
        }
    }
#endif
}

