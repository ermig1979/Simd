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

        template<RbfDiffType type> SIMD_INLINE __m256i Diff3(__m256i src)
        {
            src = _mm256_permute4x64_epi64(src, 0x94);
            static const __m256i K0 = SIMD_MM256_SETR_EPI8(
                0x0, -1, -1, -1, 0x3, -1, -1, -1, 0x6, -1, -1, -1, 0x9, -1, -1, -1,
                0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1, 0xD, -1, -1, -1);
            static const __m256i K1 = SIMD_MM256_SETR_EPI8(
                0x1, -1, -1, -1, 0x4, -1, -1, -1, 0x7, -1, -1, -1, 0xA, -1, -1, -1,
                0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1, 0xE, -1, -1, -1);
            static const __m256i K2 = SIMD_MM256_SETR_EPI8(
                0x2, -1, -1, -1, 0x5, -1, -1, -1, 0x8, -1, -1, -1, 0xB, -1, -1, -1,
                0x6, -1, -1, -1, 0x9, -1, -1, -1, 0xC, -1, -1, -1, 0xF, -1, -1, -1);
            return Diff<type>(_mm256_shuffle_epi8(src, K0), _mm256_shuffle_epi8(src, K1), _mm256_shuffle_epi8(src, K2));
        }

        template<RbfDiffType type> SIMD_INLINE __m256i Diff4(__m256i src)
        {
            static const __m256i K0 = SIMD_MM256_SETR_EPI8(
                0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xc, -1, -1, -1,
                0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xc, -1, -1, -1);
            static const __m256i K1 = SIMD_MM256_SETR_EPI8(
                0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xd, -1, -1, -1,
                0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xd, -1, -1, -1);
            static const __m256i K2 = SIMD_MM256_SETR_EPI8(
                0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xa, -1, -1, -1, 0xe, -1, -1, -1,
                0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xa, -1, -1, -1, 0xe, -1, -1, -1);
            return Diff<type>(_mm256_shuffle_epi8(src, K0), _mm256_shuffle_epi8(src, K1), _mm256_shuffle_epi8(src, K2));
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
            case 2:
            {
                for (size_t x = 0, c = 0; x < width; x += A, c += 2 * A)
                {
                    __m256i ad0 = AbsDiff8u(src0 + c + 0, src1 + c + 0);
                    __m256i d0 = Diff<type>(_mm256_and_si256(ad0, K16_00FF), _mm256_and_si256(_mm256_srli_si256(ad0, 1), K16_00FF));
                    __m256i ad1 = AbsDiff8u(src0 + c + A, src1 + c + A);
                    __m256i d1 = Diff<type>(_mm256_and_si256(ad1, K16_00FF), _mm256_and_si256(_mm256_srli_si256(ad1, 1), K16_00FF));
                    _mm256_storeu_si256((__m256i*)(dst + x), PackI16ToU8(d0, d1));
                }
                break;
            }
            case 3:
            {
                for (size_t x = 0, c = 0; x < width; x += A, c += 3 * A)
                {
                    __m256i d0 = Diff3<type>(AbsDiff8u(src0 + c + 0 * 24, src1 + c + 0 * 24));
                    __m256i d1 = Diff3<type>(AbsDiff8u(src0 + c + 1 * 24, src1 + c + 1 * 24));
                    __m256i d2 = Diff3<type>(AbsDiff8u(src0 + c + 2 * 24, src1 + c + 2 * 24));
                    __m256i d3 = Diff3<type>(AbsDiff8u(src0 + c + 3 * 24, src1 + c + 3 * 24));
                    _mm256_storeu_si256((__m256i*)(dst + x), PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
                }
                break;
            }
            case 4:
            {
                for (size_t x = 0, c = 0; x < width; x += A, c += 4 * A)
                {
                    __m256i d0 = Diff4<type>(AbsDiff8u(src0 + c + 0 * A, src1 + c + 0 * A));
                    __m256i d1 = Diff4<type>(AbsDiff8u(src0 + c + 1 * A, src1 + c + 1 * A));
                    __m256i d2 = Diff4<type>(AbsDiff8u(src0 + c + 2 * A, src1 + c + 2 * A));
                    __m256i d3 = Diff4<type>(AbsDiff8u(src0 + c + 3 * A, src1 + c + 3 * A));
                    _mm256_storeu_si256((__m256i*)(dst + x), PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
                }
                break;
            }
            default:
                for (size_t x = 0, o = 0; x < width; x += 1, o += channels)
                    dst[x] = Base::Diff<channels, type>(src0 + o, src1 + o);
            }
        }

        template<size_t channels, RbfDiffType type, size_t rows> void RowDiffs(const uint8_t* src0, const uint8_t* src1, size_t srcStride, size_t width, uint8_t* dst, size_t dstStride)
        {
            for (size_t i = 0; i < rows; ++i)
            {
                RowDiff<channels, type>(src0, src1, width, dst);
                src0 += srcStride;
                src1 += srcStride;
                dst += dstStride;
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

            template<size_t channels, int dir, bool nofma> void HorRowRun(const uint8_t* src, size_t width, float alpha, const float* ranges, uint8_t* diff, uint8_t* dst)
            {
                if (dir == -1) diff += width - 2;
                float factor = 1.0f, colors[channels];
                for (int c = 0; c < channels; c++)
                {
                    colors[c] = src[c];
                    Set<dir>(src[c], dst + c);
                }
                for (size_t x = 1; x < width; x += 1)
                {
                    src += channels * dir;
                    dst += channels * dir;
                    float range = ranges[diff[0]];
                    factor = alpha + range * factor;
                    for (int c = 0; c < channels; c++)
                    {
                        colors[c] = alpha * src[c] + range * colors[c];
                        Set<dir>(int(colors[c] / factor), dst + c);
                    }
                    diff += dir;
                }
            }

            //-----------------------------------------------------------------------------------------

            template<size_t channels> struct HorRow
            {
                template<int dir, bool nofma> static void Run4x(const uint8_t* src, size_t srcStride, size_t width,
                    float alpha, const float* ranges, uint8_t* diff, uint8_t* dst, size_t dstStride)
                {
                    for(size_t row = 0; row < 4; ++row)
                        HorRowRun<channels, dir, nofma>(src + row * srcStride, width, alpha, ranges, diff + row * dstStride, dst + row * dstStride);
                }

                template<int dir, bool nofma> static void Run8x(const uint8_t* src, size_t srcStride, size_t width,
                    float alpha, const float* ranges, uint8_t* diff, uint8_t* dst, size_t dstStride)
                {
                    for (size_t row = 0; row < 8; ++row)
                        HorRowRun<channels, dir, nofma>(src + row * srcStride, width, alpha, ranges, diff + row * dstStride, dst + row * dstStride);
                }
            };

            template<> struct HorRow<1>
            {
                template<int dir, bool nofma> static void Run4x(const uint8_t* src, size_t srcStride, size_t width,
                    float alpha, const float* ranges, uint8_t* diff, uint8_t* dst, size_t dstStride)
                {
                    const size_t so0 = 0, so1 = srcStride, so2 = 2 * srcStride, so3 = 3 * srcStride;
                    const size_t do0 = 0, do1 = dstStride, do2 = 2 * dstStride, do3 = 3 * dstStride;
                    __m128 _factor = _mm_set1_ps(1.0f), _alpha = _mm_set1_ps(alpha);
                    __m128 _colors = _mm_setr_ps(src[so0], src[so1], src[so2], src[so3]);
                    if (dir == -1) diff += width - 2;
                    for (size_t x = 0; x < width; x += 1)
                    {
                        __m128 _range = _mm_setr_ps(ranges[diff[do0]], ranges[diff[do1]], ranges[diff[do2]], ranges[diff[do3]]);
                        __m128i _dst = _mm_cvtps_epi32(_mm_floor_ps(_mm_div_ps(_colors, _factor)));
                        if (dir == -1) _dst = _mm_avg_epu8(_dst, _mm_setr_epi32(dst[do0], dst[do1], dst[do2], dst[do3]));
                        dst[do0] = _mm_extract_epi32(_dst, 0);
                        dst[do1] = _mm_extract_epi32(_dst, 1);
                        dst[do2] = _mm_extract_epi32(_dst, 2);
                        dst[do3] = _mm_extract_epi32(_dst, 3);
                        src += dir, dst += dir, diff += dir;
                        __m128i _src = _mm_setr_epi32(src[so0], src[so1], src[so2], src[so3]);
                        _factor = Fmadd<nofma>(_range, _factor, _alpha);
                        _colors = Fmadd<nofma>(_alpha, _mm_cvtepi32_ps(_src), _range, _colors);
                    }
                }

                static SIMD_INLINE __m256i Load1x(const uint8_t* src, __m256i offs)
                {
                    return _mm256_and_si256(_mm256_i32gather_epi32((int*)src, offs, 1), K32_000000FF);
                }

                template<int dir> static SIMD_INLINE __m256i Load4x(const uint8_t* src, __m256i offs)
                {
                    if (dir == -1)
                        src -= 3;
                    return _mm256_i32gather_epi32((int*)src, offs, 1);
                }

                template<class T, int dir> static SIMD_INLINE void Store(__m256i val, uint8_t* dst, size_t stride)
                {
                    if (dir == -1 && sizeof(T) - 1)
                        dst -= sizeof(T) - 1;
                    __m128i val0 = _mm256_extractf128_si256(val, 0);
                    *(T*)(dst) = _mm_extract_epi32(val0, 0), dst += stride;
                    *(T*)(dst) = _mm_extract_epi32(val0, 1), dst += stride;
                    *(T*)(dst) = _mm_extract_epi32(val0, 2), dst += stride;
                    *(T*)(dst) = _mm_extract_epi32(val0, 3), dst += stride;
                    __m128i val1 = _mm256_extractf128_si256(val, 1);
                    *(T*)(dst) = _mm_extract_epi32(val1, 0), dst += stride;
                    *(T*)(dst) = _mm_extract_epi32(val1, 1), dst += stride;
                    *(T*)(dst) = _mm_extract_epi32(val1, 2), dst += stride;
                    *(T*)(dst) = _mm_extract_epi32(val1, 3), dst += stride;
                }

                template<int dir> static SIMD_INLINE __m256i Get(__m256i src, int idx)
                {
                    return _mm256_and_si256(_mm256_srli_epi32(src, (dir > 0 ? idx : 3 - idx) * 8), K32_000000FF);
                }

                template<int dir> static SIMD_INLINE void Set(__m256i &dst, __m256i src, int idx)
                {
                    dst = _mm256_or_si256(_mm256_slli_epi32(src, (dir > 0 ? idx : 3 - idx) * 8), dst);
                }

                template<int dir, bool nofma> static void Run8x(const uint8_t* src, size_t srcStride, size_t width,
                    float alpha, const float* ranges, uint8_t* diff, uint8_t* dst, size_t dstStride)
                {
                    static const __m256i _01234567 = SIMD_MM256_SETR_EPI32(0, 1, 2, 3, 4, 5, 6, 7);
                    __m256i srcOffs = _mm256_mullo_epi32(_mm256_set1_epi32((int)srcStride), _01234567);
                    __m256i dstOffs = _mm256_mullo_epi32(_mm256_set1_epi32((int)dstStride), _01234567);
                    __m256 _factor = _mm256_set1_ps(1.0f), _alpha = _mm256_set1_ps(alpha);
                    __m256 _colors = _mm256_cvtepi32_ps(Load1x(src, srcOffs));
                    if (dir == -1) diff += width - 2;
                    size_t x = 0, width4 = width & (~3);
                    for (; x < width4; x += 4)
                    {
                        __m256i diff4 = Load4x<dir>(diff, dstOffs);
                        __m256i src4 = Load4x<dir>(src + dir, srcOffs);
                        __m256i dst4 = _mm256_setzero_si256();
                        for (size_t i = 0; i < 4; ++i)
                        {
                            __m256 _range = _mm256_i32gather_ps(ranges, Get<dir>(diff4, i), 4);
                            Set<dir>(dst4, _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors, _factor))), i);
                            _factor = Fmadd<nofma>(_range, _factor, _alpha);
                            _colors = Fmadd<nofma>(_alpha, _mm256_cvtepi32_ps(Get<dir>(src4, i)), _range, _colors);
                        }
                        if (dir == -1) dst4 = _mm256_avg_epu8(dst4, Load4x<dir>(dst, dstOffs));
                        Store<uint32_t, dir>(dst4, dst, dstStride);
                        src += 4 * dir, dst += 4 * dir, diff += 4 * dir;
                    }
                    for (; x < width; x += 1)
                    {
                        __m256 _range = _mm256_i32gather_ps(ranges, Load1x(diff, dstOffs), 4);
                        __m256i _dst = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors, _factor)));
                        if (dir == -1) _dst = _mm256_avg_epu8(_dst, Load1x(dst, dstOffs));
                        Store<uint8_t, dir>(_dst, dst, dstStride);
                        src += dir, dst += dir, diff += dir;
                        _factor = Fmadd<nofma>(_range, _factor, _alpha);
                        _colors = Fmadd<nofma>(_alpha, _mm256_cvtepi32_ps(Load1x(src, srcOffs)), _range, _colors);
                    }
                }
            };

            template<> struct HorRow<2>
            {
                //static SIMD_INLINE __m128i Load(const uint8_t* src0, const uint8_t* src1)
                //{
                //    int lo = *(uint16_t*)src0, hi = *(uint16_t*)src1, val = lo | (hi << 16);
                //    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(val));
                //}

                static SIMD_INLINE __m128i LoadDiff4x1(const uint8_t* src, __m128i offs)
                {
                    return _mm_and_si128(_mm_i32gather_epi32((int*)src, offs, 1), Sse41::K32_000000FF);
                }

                template<int dir> static SIMD_INLINE __m128i LoadDiff4x4(const uint8_t* src, __m128i offs)
                {
                    if (dir == -1)
                        src -= 3;
                    return _mm_i32gather_epi32((int*)src, offs, 1);
                }

                static SIMD_INLINE __m256i Load4x1(const uint8_t* src, __m128i offs)
                {
                    static const __m256i SHFL = SIMD_MM256_SETR_EPI8(
                        0x0, -1, -1, -1, 0x1, -1, -1, -1, 0x8, -1, -1, -1, 0x9, -1, -1, -1,
                        0x0, -1, -1, -1, 0x1, -1, -1, -1, 0x8, -1, -1, -1, 0x9, -1, -1, -1);
                    return _mm256_shuffle_epi8(_mm256_i32gather_epi64((long long*)src, offs, 1), SHFL);
                }

                template<int dir> static SIMD_INLINE __m256i Load4x4(const uint8_t* src, __m128i offs)
                {
                    if (dir == -1)
                        src -= 6;
                    static const __m256i SHFL = SIMD_MM256_SETR_EPI8(
                        0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF,
                        0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7, 0x8, 0xA, 0xC, 0xE, 0x9, 0xB, 0xD, 0xF);
                    return _mm256_shuffle_epi8(_mm256_i32gather_epi64((long long*)src, offs, 1), SHFL);
                }

                static SIMD_INLINE __m256 UnpackLo(__m256 src)
                {
                    static const __m256i UNPACK = SIMD_MM256_SETR_EPI32(0, 0, 1, 1, 2, 2, 3, 3);
                    return _mm256_permutevar8x32_ps(src, UNPACK);
                }

                static SIMD_INLINE __m256 UnpackHi(__m256 src)
                {
                    static const __m256i UNPACK = SIMD_MM256_SETR_EPI32(4, 4, 5, 5, 6, 6, 7, 7);
                    return _mm256_permutevar8x32_ps(src, UNPACK);
                }

                static SIMD_INLINE void Store4x1(__m256i val, uint8_t* dst, size_t stride)
                {
                    val = _mm256_packus_epi16(_mm256_packus_epi32(val, _mm256_setzero_si256()), _mm256_setzero_si256());
                    __m128i val0 = _mm256_extractf128_si256(val, 0);
                    *(uint16_t*)(dst) = _mm_extract_epi16(val0, 0), dst += stride;
                    *(uint16_t*)(dst) = _mm_extract_epi16(val0, 1), dst += stride;
                    __m128i val1 = _mm256_extractf128_si256(val, 1);
                    *(uint16_t*)(dst) = _mm_extract_epi16(val1, 0), dst += stride;
                    *(uint16_t*)(dst) = _mm_extract_epi16(val1, 1), dst += stride;
                }

                template<int dir> static SIMD_INLINE void Store4x4(__m256i val, uint8_t* dst, size_t stride)
                {
                    if (dir == -1)
                        dst -= 6;
                    static const __m256i SHFL = SIMD_MM256_SETR_EPI8(
                        0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
                        0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);
                    val = _mm256_shuffle_epi8(val, SHFL);
                    __m128i val0 = _mm256_extractf128_si256(val, 0);
                    *(uint64_t*)(dst) = _mm_extract_epi64(val0, 0), dst += stride;
                    *(uint64_t*)(dst) = _mm_extract_epi64(val0, 1), dst += stride;
                    __m128i val1 = _mm256_extractf128_si256(val, 1);
                    *(uint64_t*)(dst) = _mm_extract_epi64(val1, 0), dst += stride;
                    *(uint64_t*)(dst) = _mm_extract_epi64(val1, 1), dst += stride;
                }

                template<int dir> static SIMD_INLINE __m128i Get(__m128i src, int idx)
                {
                    return _mm_and_si128(_mm_srli_epi32(src, (dir > 0 ? idx : 3 - idx) * 8), Sse41::K32_000000FF);
                }

                template<int dir> static SIMD_INLINE __m256i Get(__m256i src, int idx)
                {
                    return _mm256_and_si256(_mm256_srli_epi32(src, (dir > 0 ? idx : 3 - idx) * 8), K32_000000FF);
                }

                template<int dir> static SIMD_INLINE void Set(__m256i& dst, __m256i src, int idx)
                {
                    dst = _mm256_or_si256(_mm256_slli_epi32(src, (dir > 0 ? idx : 3 - idx) * 8), dst);
                }

                template<int dir, bool nofma> static void Run4x(const uint8_t* src, size_t srcStride, size_t width,
                    float alpha, const float* ranges, uint8_t* diff, uint8_t* dst, size_t dstStride)
                {
                    static const __m128i _0123 = SIMD_MM_SETR_EPI32(0, 1, 2, 3);
                    __m128i srcOffs = _mm_mullo_epi32(_mm_set1_epi32((int)srcStride), _0123);
                    __m128i dstOffs = _mm_mullo_epi32(_mm_set1_epi32((int)dstStride), _0123);
                    __m256 _factor = _mm256_set1_ps(1.0f), _alpha = _mm256_set1_ps(alpha);
                    __m256 _colors = _mm256_cvtepi32_ps(Load4x1(src, srcOffs));
                    if (dir == -1) diff += width - 2;
                    size_t x = 0, width4 = width& (~3);
                    for (; x < width4; x += 4)
                    {
                        __m128i diff4 = LoadDiff4x4<dir>(diff, dstOffs);
                        __m256i src4 = Load4x4<dir>(src + dir * 2, srcOffs);
                        __m256i dst4 = _mm256_setzero_si256();
                        for (size_t i = 0; i < 4; ++i)
                        {
                            __m256 _range = _mm256_castps128_ps256(_mm_i32gather_ps(ranges, Get<dir>(diff4, i), 4));
                            Set<dir>(dst4, _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors, UnpackLo(_factor)))), i);
                            _factor = Fmadd<nofma>(_range, _factor, _alpha);
                            _colors = Fmadd<nofma>(_alpha, _mm256_cvtepi32_ps(Get<dir>(src4, i)), UnpackLo(_range), _colors);
                        }
                        if (dir == -1) dst4 = _mm256_avg_epu8(dst4, Load4x4<dir>(dst, dstOffs));
                        Store4x4<dir>(dst4, dst, dstStride);
                        src += 8 * dir, dst += 8 * dir, diff += 4 * dir;
                    }
                    for (; x < width; x += 1)
                    {
                        __m256 _range = _mm256_castps128_ps256(_mm_i32gather_ps(ranges, LoadDiff4x1(diff, dstOffs), 4));
                        __m256i _dst = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors, UnpackLo(_factor))));
                        if (dir == -1) _dst = _mm256_avg_epu8(_dst, Load4x1(dst, dstOffs));
                        Store4x1(_dst, dst, dstStride);
                        src += dir * 2, dst += dir * 2, diff += dir;
                        _factor = Fmadd<nofma>(_range, _factor, _alpha);
                        _colors = Fmadd<nofma>(_alpha, _mm256_cvtepi32_ps(Load4x1(src, srcOffs)), UnpackLo(_range), _colors);
                    }
                }

                static SIMD_INLINE __m256i LoadDiff8x1(const uint8_t* src, __m256i offs)
                {
                    return _mm256_and_si256(_mm256_i32gather_epi32((int*)src, offs, 1), K32_000000FF);
                }

                template<int dir> static SIMD_INLINE __m256i LoadDiff8x4(const uint8_t* src, __m256i offs)
                {
                    if (dir == -1)
                        src -= 3;
                    return _mm256_i32gather_epi32((int*)src, offs, 1);
                }

                static SIMD_INLINE void Store8x1(__m256i lo, __m256i hi, uint8_t* dst, size_t stride)
                {
                    __m256i val = _mm256_packus_epi16(_mm256_packus_epi32(lo, hi), _mm256_setzero_si256());
                    __m128i val0 = _mm256_extractf128_si256(val, 0);
                    __m128i val1 = _mm256_extractf128_si256(val, 1);
                    *(uint16_t*)(dst) = _mm_extract_epi16(val0, 0), dst += stride;
                    *(uint16_t*)(dst) = _mm_extract_epi16(val0, 1), dst += stride;
                    *(uint16_t*)(dst) = _mm_extract_epi16(val1, 0), dst += stride;
                    *(uint16_t*)(dst) = _mm_extract_epi16(val1, 1), dst += stride;
                    *(uint16_t*)(dst) = _mm_extract_epi16(val0, 2), dst += stride;
                    *(uint16_t*)(dst) = _mm_extract_epi16(val0, 3), dst += stride;
                    *(uint16_t*)(dst) = _mm_extract_epi16(val1, 2), dst += stride;
                    *(uint16_t*)(dst) = _mm_extract_epi16(val1, 3), dst += stride;
                }

                template<int dir, bool nofma> static void Run8x(const uint8_t* src0, size_t srcStride, size_t width,
                    float alpha, const float* ranges, uint8_t* diff, uint8_t* dst0, size_t dstStride)
                {
                    const uint8_t* src1 = src0 + 4 * srcStride;
                    uint8_t* dst1 = dst0 + 4 * dstStride;
#if 0
                    Run4x<dir, nofma>(src0, srcStride, width, alpha, ranges, diff + 0 * dstStride, dst0, dstStride);
                    Run4x<dir, nofma>(src1, srcStride, width, alpha, ranges, diff + 4 * dstStride, dst1, dstStride);
#else
                    static const __m256i _01234567 = SIMD_MM256_SETR_EPI32(0, 1, 2, 3, 4, 5, 6, 7);
                    __m256i srcOffs = _mm256_mullo_epi32(_mm256_set1_epi32((int)srcStride), _01234567);
                    __m256i dstOffs = _mm256_mullo_epi32(_mm256_set1_epi32((int)dstStride), _01234567);
                    __m256 _factor = _mm256_set1_ps(1.0f), _alpha = _mm256_set1_ps(alpha);
                    __m256 _colors0 = _mm256_cvtepi32_ps(Load4x1(src0, _mm256_castsi256_si128(srcOffs)));
                    __m256 _colors1 = _mm256_cvtepi32_ps(Load4x1(src1, _mm256_castsi256_si128(srcOffs)));
                    if (dir == -1) diff += width - 2;
                    size_t x = 0, width4 = width & (~3);
                    for (; x < width4; x += 4)
                    {
                        __m256i _diff4 = LoadDiff8x4<dir>(diff, dstOffs);
                        __m256i _src40 = Load4x4<dir>(src0 + dir * 2, _mm256_castsi256_si128(srcOffs));
                        __m256i _src41 = Load4x4<dir>(src1 + dir * 2, _mm256_castsi256_si128(srcOffs));
                        __m256i _dst40 = _mm256_setzero_si256(), _dst41 = _mm256_setzero_si256();
                        for (size_t i = 0; i < 4; ++i)
                        {
                            __m256 _range = _mm256_i32gather_ps(ranges, Get<dir>(_diff4, i), 4);
                            Set<dir>(_dst40, _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors0, UnpackLo(_factor)))), i);
                            Set<dir>(_dst41, _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors1, UnpackHi(_factor)))), i);
                            _factor = Fmadd<nofma>(_range, _factor, _alpha);
                            _colors0 = Fmadd<nofma>(_alpha, _mm256_cvtepi32_ps(Get<dir>(_src40, i)), UnpackLo(_range), _colors0);
                            _colors1 = Fmadd<nofma>(_alpha, _mm256_cvtepi32_ps(Get<dir>(_src41, i)), UnpackHi(_range), _colors1);
                        }
                        if (dir == -1) _dst40 = _mm256_avg_epu8(_dst40, Load4x4<dir>(dst0, _mm256_castsi256_si128(dstOffs)));
                        if (dir == -1) _dst41 = _mm256_avg_epu8(_dst41, Load4x4<dir>(dst1, _mm256_castsi256_si128(dstOffs)));
                        Store4x4<dir>(_dst40, dst0, dstStride);
                        Store4x4<dir>(_dst41, dst1, dstStride);
                        src0 += dir * 8, src1 += dir * 8, dst0 += dir * 8, dst1 += dir * 8, diff += 4 * dir;
                    }
                    for (; x < width; x += 1)
                    {
                        __m256 _range = _mm256_i32gather_ps(ranges, LoadDiff8x1(diff, dstOffs), 4);
                        __m256i _dst0 = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors0, UnpackLo(_factor))));
                        __m256i _dst1 = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors1, UnpackHi(_factor))));
                        if (dir == -1) _dst0 = _mm256_avg_epu8(_dst0, Load4x1(dst0, _mm256_castsi256_si128(dstOffs)));
                        if (dir == -1) _dst1 = _mm256_avg_epu8(_dst1, Load4x1(dst1, _mm256_castsi256_si128(dstOffs)));
                        Store8x1(_dst0, _dst1, dst0, dstStride);
                        src0 += dir * 2, src1 += dir * 2, dst0 += dir * 2, dst1 += dir * 2, diff += dir;
                        _factor = Fmadd<nofma>(_range, _factor, _alpha);
                        _colors0 = Fmadd<nofma>(_alpha, _mm256_cvtepi32_ps(Load4x1(src0, _mm256_castsi256_si128(srcOffs))), UnpackLo(_range), _colors0);
                        _colors1 = Fmadd<nofma>(_alpha, _mm256_cvtepi32_ps(Load4x1(src1, _mm256_castsi256_si128(srcOffs))), UnpackHi(_range), _colors1);
                    }
#endif
                }
            };

            //-----------------------------------------------------------------------------------------

            template<size_t channels, RbfDiffType type, bool nofma> void HorFilter(const RbfParam& p, float* buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
            {
                size_t last = (p.width - 1) * channels, y = 0, height4 = AlignLo(p.height, 4), height8 = AlignLo(p.height, 8);
                uint8_t* diff = (uint8_t*)buf;
                for (; y < height8; y += 8)
                {
                    RowDiffs<channels, type, 8>(src, src + channels, srcStride, p.width - 1, diff, dstStride);
                    HorRow<channels>::template Run8x<+1, nofma>(src, srcStride, p.width, p.alpha, p.ranges, diff, dst, dstStride);
                    HorRow<channels>::template Run8x<-1, nofma>(src + last, srcStride, p.width, p.alpha, p.ranges, diff, dst + last, dstStride);
                    src += 8 * srcStride;
                    dst += 8 * dstStride;
                }
                for (; y < height4; y += 4)
                {
                    RowDiffs<channels, type, 4>(src, src + channels, srcStride, p.width - 1, diff, dstStride);
                    HorRow<channels>::template Run4x<+1, nofma>(src, srcStride, p.width, p.alpha, p.ranges, diff, dst, dstStride);
                    HorRow<channels>::template Run4x<-1, nofma>(src + last, srcStride, p.width, p.alpha, p.ranges, diff, dst + last, dstStride);
                    src += 4 * srcStride;
                    dst += 4 * dstStride;
                }
                for (; y < p.height; y++)
                {
                    RowDiff<channels, type>(src, src + channels, p.width - 1, diff);
                    HorRowRun<channels, +1, nofma>(src, p.width, p.alpha, p.ranges, diff, dst);
                    HorRowRun<channels, -1, nofma>(src + last, p.width, p.alpha, p.ranges, diff, dst + last);
                    src += srcStride;
                    dst += dstStride;
                }
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
                template<int offs, bool nofma> static SIMD_INLINE __m128i Run4(__m128i src, const uint8_t* diff,
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

                template<int part, bool nofma> static SIMD_INLINE __m256i Run8(__m256i src, const uint8_t* diff,
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
                        __m256i d0 = Run8<0, nofma>(_src, diff + x, _alpha, ranges, factor + x, colors + x);
                        __m256i d1 = Run8<1, nofma>(_src, diff + x, _alpha, ranges, factor + x, colors + x);
                        __m256i d2 = Run8<2, nofma>(_src, diff + x, _alpha, ranges, factor + x, colors + x);
                        __m256i d3 = Run8<3, nofma>(_src, diff + x, _alpha, ranges, factor + x, colors + x);
                        Set32<dir>(PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)), dst + x);
                    }
                    for (; x < widthHA; x += HA)
                    {
                        __m128i _src = _mm_loadu_si128((__m128i*)(src + x));
                        __m128i d0 = Run4<0 * 4, nofma>(_src, diff + x, _mm256_castps256_ps128(_alpha), ranges, factor + x, colors + x);
                        __m128i d1 = Run4<1 * 4, nofma>(_src, diff + x, _mm256_castps256_ps128(_alpha), ranges, factor + x, colors + x);
                        __m128i d2 = Run4<2 * 4, nofma>(_src, diff + x, _mm256_castps256_ps128(_alpha), ranges, factor + x, colors + x);
                        __m128i d3 = Run4<3 * 4, nofma>(_src, diff + x, _mm256_castps256_ps128(_alpha), ranges, factor + x, colors + x);
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

            template<> struct VerMain<2>
            {
                template<int offs, bool nofma> static SIMD_INLINE __m128i Run4(__m128i src, __m128 alpha, __m128 range, const __m128& factor, float* colors)
                {
                    __m128 _src = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(src, offs)));
                    __m128 _colors = Fmadd<nofma>(alpha, _src, range, _mm_loadu_ps(colors + offs));
                    _mm_storeu_ps(colors + offs, _colors);
                    return _mm_cvtps_epi32(_mm_floor_ps(_mm_div_ps(_colors, factor)));
                }

                template<int offs, bool nofma> static SIMD_INLINE __m128i Run8(__m128i src, const uint8_t* diff,
                    __m128 alpha, const float* ranges, float* factor, float* colors)
                {
                    __m128 _range = _mm_setr_ps(ranges[diff[0]], ranges[diff[1]], ranges[diff[2]], ranges[diff[3]]);
                    //__m128 _range = _mm_i32gather_ps(ranges, _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)diff)), 4);
                    __m128 _factor = Fmadd<nofma>(_range, _mm_loadu_ps(factor), alpha);
                    _mm_storeu_ps(factor, _factor);
                    __m128i dst0 = Run4<offs + 0, nofma>(src, alpha, Sse41::Shuffle32f<0x50>(_range), Sse41::Shuffle32f<0x50>(_factor), colors);
                    __m128i dst1 = Run4<offs + 4, nofma>(src, alpha, Sse41::Shuffle32f<0xFA>(_range), Sse41::Shuffle32f<0xFA>(_factor), colors);
                    return _mm_packs_epi32(dst0, dst1);
                }

                template<int part, bool nofma> static SIMD_INLINE __m256i Run8(__m256i src, __m256 alpha, __m256 range, const __m256& factor, float* colors)
                {
                    __m256 _src = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_permute4x64_epi64(src, 0x55 * part))));
                    __m256 _colors = Fmadd<nofma>(alpha, _src, range, _mm256_loadu_ps(colors + part * F));
                    _mm256_storeu_ps(colors + part * F, _colors);
                    return _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors, factor)));
                }

                template<int part, bool nofma> static SIMD_INLINE __m256i Run16(__m256i src, const uint8_t* diff,
                    __m256 alpha, const float* ranges, float* factor, float* colors)
                {
                    static const __m256i PART_0 = SIMD_MM256_SETR_EPI32(0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3);
                    static const __m256i PART_1 = SIMD_MM256_SETR_EPI32(0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7);
                    __m256 _range = _mm256_i32gather_ps(ranges, _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)diff)), 4);
                    __m256 _factor = Fmadd<nofma>(_range, _mm256_loadu_ps(factor), alpha);
                    _mm256_storeu_ps(factor, _factor);
                    __m256i dst0 = Run8<part + 0, nofma>(src, alpha, _mm256_permutevar8x32_ps(_range, PART_0), _mm256_permutevar8x32_ps(_factor, PART_0), colors);
                    __m256i dst1 = Run8<part + 1, nofma>(src, alpha, _mm256_permutevar8x32_ps(_range, PART_1), _mm256_permutevar8x32_ps(_factor, PART_1), colors);
                    return PackI32ToI16(dst0, dst1);
                }

                template<int dir, bool nofma> static void Run(const uint8_t* src, const uint8_t* diff, size_t width, float alpha,
                    const float* ranges, float* factor, float* colors, uint8_t* dst)
                {
                    size_t width8 = AlignLo(width, 8), width16 = AlignLo(width, 16), x = 0, o = 0;
                    __m256 _alpha = _mm256_set1_ps(alpha);
                    for (; x < width16; x += 16, o += 32)
                    {
                        __m256i _src = _mm256_loadu_si256((__m256i*)(src + o));
                        __m256i dst0 = Run16<0, nofma>(_src, diff + x + 0, _alpha, ranges, factor + x + 0, colors + o);
                        __m256i dst1 = Run16<2, nofma>(_src, diff + x + 8, _alpha, ranges, factor + x + 8, colors + o);
                        Set32<dir>(PackI16ToU8(dst0, dst1), dst + o);
                    }
                    for (; x < width8; x += 8, o += 16)
                    {
                        __m128i _src = _mm_loadu_si128((__m128i*)(src + o));
                        __m128i dst0 = Run8<0, nofma>(_src, diff + x + 0, _mm256_castps256_ps128(_alpha), ranges, factor + x + 0, colors + o);
                        __m128i dst1 = Run8<8, nofma>(_src, diff + x + 4, _mm256_castps256_ps128(_alpha), ranges, factor + x + 4, colors + o);
                        Set16<dir>(_mm_packus_epi16(dst0, dst1), dst + o);
                    }
                    for (; x < width; x++)
                    {
                        float range = ranges[diff[x]];
                        factor[x] = alpha + range * factor[x];
                        for (size_t e = o + 2; o < e; o++)
                        {
                            colors[o] = alpha * src[o] + range * colors[o];
                            Set<dir>(int(colors[o] / factor[x]), dst + o);
                        }
                    }
                }
            };

            template<> struct VerMain<3>
            {
                template<int offs, bool nofma> static SIMD_INLINE __m128i Run4C(__m128i src, __m128 alpha, __m128 range, const __m128& factor, float* colors)
                {
                    static const __m128i SHFL = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                    __m128 _src = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(src, offs)));
                    __m128 _colors = Fmadd<nofma>(alpha, _src, range, _mm_loadu_ps(colors + offs));
                    _mm_storeu_ps(colors + offs, _colors);
                    return _mm_shuffle_epi8(_mm_cvtps_epi32(_mm_floor_ps(_mm_div_ps(_colors, factor))), SHFL);
                }

                template<bool nofma> static SIMD_INLINE __m128i Run4(const uint8_t* src, const uint8_t* diff, __m128 alpha, const float* ranges, float* factor, float* colors)
                {
                    __m128i _src = _mm_loadu_si128((__m128i*)src);
                    __m128 _range = _mm_setr_ps(ranges[diff[0]], ranges[diff[1]], ranges[diff[2]], ranges[diff[3]]);
                    __m128 _factor = Fmadd<nofma>(_range, _mm_loadu_ps(factor), alpha);
                    _mm_storeu_ps(factor, _factor);
                    __m128i dst0 = Run4C<0, nofma>(_src, alpha, Sse41::Shuffle32f<0x40>(_range), Sse41::Shuffle32f<0x40>(_factor), colors);
                    __m128i dst1 = Run4C<4, nofma>(_src, alpha, Sse41::Shuffle32f<0xA5>(_range), Sse41::Shuffle32f<0xA5>(_factor), colors);
                    __m128i dst2 = Run4C<8, nofma>(_src, alpha, Sse41::Shuffle32f<0xFE>(_range), Sse41::Shuffle32f<0xFE>(_factor), colors);
                    return _mm_or_si128(dst0, _mm_or_si128(_mm_slli_si128(dst1, 4), _mm_slli_si128(dst2, 8)));
                }

                template<int part, bool nofma> static SIMD_INLINE __m256i Run8C(__m256i src, __m256 alpha, __m256 range, const __m256& factor, float* colors)
                {
                    static const __m256i SHFL = SIMD_MM256_SETR_EPI8(
                        0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                        0x0, 0x4, 0x8, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                    __m256 _src = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_permute4x64_epi64(src, 0x55 * part))));
                    __m256 _colors = Fmadd<nofma>(alpha, _src, range, _mm256_loadu_ps(colors + part * F));
                    _mm256_storeu_ps(colors + part * F, _colors);
                    return _mm256_shuffle_epi8(_mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors, factor))), SHFL);
                }

                template<bool nofma> static SIMD_INLINE __m256i Run8(const uint8_t* src, const uint8_t* diff, __m256 alpha, const float* ranges, float* factor, float* colors)
                {
                    static const __m256i PERM_0 = SIMD_MM256_SETR_EPI32(0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x2, 0x2);
                    static const __m256i PERM_1 = SIMD_MM256_SETR_EPI32(0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5);
                    static const __m256i PERM_2 = SIMD_MM256_SETR_EPI32(0x5, 0x5, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7);
                    static const __m256i PERM_3 = SIMD_MM256_SETR_EPI32(0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7);
                    __m256i _src = _mm256_loadu_si256((__m256i*)src);
                    __m256 _range = _mm256_i32gather_ps(ranges, _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)diff)), 4);
                    __m256 _factor = Fmadd<nofma>(_range, _mm256_loadu_ps(factor), alpha);
                    _mm256_storeu_ps(factor, _factor);
                    __m256i dst0 = Run8C<0, nofma>(_src, alpha, _mm256_permutevar8x32_ps(_range, PERM_0), _mm256_permutevar8x32_ps(_factor, PERM_0), colors);
                    __m256i dst1 = Run8C<1, nofma>(_src, alpha, _mm256_permutevar8x32_ps(_range, PERM_1), _mm256_permutevar8x32_ps(_factor, PERM_1), colors);
                    __m256i dst2 = Run8C<2, nofma>(_src, alpha, _mm256_permutevar8x32_ps(_range, PERM_2), _mm256_permutevar8x32_ps(_factor, PERM_2), colors);
                    return _mm256_permutevar8x32_epi32(_mm256_or_si256(dst0, _mm256_or_si256(_mm256_slli_si256(dst1, 4), _mm256_slli_si256(dst2, 8))), PERM_3);
                }

                template<int dir, bool nofma> static void Run(const uint8_t* src, const uint8_t* diff, size_t width, float alpha,
                    const float* ranges, float* factor, float* colors, uint8_t* dst)
                {
                    size_t width32 = AlignLo(width, 32), width16 = AlignLo(width, 16), x = 0, o = 0;
                    __m256 _alpha = _mm256_set1_ps(alpha);
                    for (; x < width32; x += 32, o += 96)
                    {
                        __m256i dst0 = Run8<nofma>(src + o + 0 * 24, diff + x + 0 * 8, _alpha, ranges, factor + x + 0 * 8, colors + o + 0 * 24);
                        __m256i dst1 = Run8<nofma>(src + o + 1 * 24, diff + x + 1 * 8, _alpha, ranges, factor + x + 1 * 8, colors + o + 1 * 24);
                        __m256i dst2 = Run8<nofma>(src + o + 2 * 24, diff + x + 2 * 8, _alpha, ranges, factor + x + 2 * 8, colors + o + 2 * 24);
                        __m256i dst3 = Run8<nofma>(src + o + 3 * 24, diff + x + 3 * 8, _alpha, ranges, factor + x + 3 * 8, colors + o + 3 * 24);
                        Set32<dir>(_mm256_or_si256(dst0, _mm256_permute4x64_epi64(dst1, 0x3F)), dst + o + 0 * 32);
                        Set32<dir>(_mm256_or_si256(_mm256_permute4x64_epi64(dst1, 0xF9), _mm256_permute4x64_epi64(dst2, 0x4F)), dst + o + 1 * 32);
                        Set32<dir>(_mm256_or_si256(_mm256_permute4x64_epi64(dst2, 0xFE), _mm256_permute4x64_epi64(dst3, 0x93)), dst + o + 2 * 32);
                    }
                    for (; x < width16; x += 16, o += 48)
                    {
                        __m128i dst0 = Run4<nofma>(src + o + 0 * 12, diff + x + 0 * 4, _mm256_castps256_ps128(_alpha), ranges, factor + x + 0 * 4, colors + o + 0 * 12);
                        __m128i dst1 = Run4<nofma>(src + o + 1 * 12, diff + x + 1 * 4, _mm256_castps256_ps128(_alpha), ranges, factor + x + 1 * 4, colors + o + 1 * 12);
                        __m128i dst2 = Run4<nofma>(src + o + 2 * 12, diff + x + 2 * 4, _mm256_castps256_ps128(_alpha), ranges, factor + x + 2 * 4, colors + o + 2 * 12);
                        __m128i dst3 = Run4<nofma>(src + o + 3 * 12, diff + x + 3 * 4, _mm256_castps256_ps128(_alpha), ranges, factor + x + 3 * 4, colors + o + 3 * 12);
                        Set16<dir>(_mm_or_si128(dst0, _mm_slli_si128(dst1, 12)), dst + o + 0 * 16);
                        Set16<dir>(_mm_or_si128(_mm_srli_si128(dst1, 4), _mm_slli_si128(dst2, 8)), dst + o + 1 * 16);
                        Set16<dir>(_mm_or_si128(_mm_srli_si128(dst2, 8), _mm_slli_si128(dst3, 4)), dst + o + 2 * 16);
                    }
                    for (; x < width; x++)
                    {
                        float range = ranges[diff[x]];
                        factor[x] = alpha + range * factor[x];
                        for (size_t e = o + 3; o < e; o++)
                        {
                            colors[o] = alpha * src[o] + range * colors[o];
                            Set<dir>(int(colors[o] / factor[x]), dst + o);
                        }
                    }
                }
            };

            template<> struct VerMain<4>
            {
                template<int offs, bool nofma> static SIMD_INLINE __m128i Run4C(__m128i src, __m128 alpha, __m128 range, const __m128& factor, float* colors)
                {
                    __m128 _src = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(src, offs)));
                    __m128 _colors = Fmadd<nofma>(alpha, _src, range, _mm_loadu_ps(colors + offs));
                    _mm_storeu_ps(colors + offs, _colors);
                    return _mm_cvtps_epi32(_mm_floor_ps(_mm_div_ps(_colors, factor)));
                }

                template<bool nofma> static SIMD_INLINE __m128i Run4(const uint8_t* src, const uint8_t* diff, __m128 alpha, const float* ranges, float* factor, float* colors)
                {
                    __m128i _src = _mm_loadu_si128((__m128i*)src);
                    __m128 _range = _mm_setr_ps(ranges[diff[0]], ranges[diff[1]], ranges[diff[2]], ranges[diff[3]]);
                    __m128 _factor = Fmadd<nofma>(_range, _mm_loadu_ps(factor), alpha);
                    _mm_storeu_ps(factor, _factor);
                    __m128i dst0 = Run4C<0x0, nofma>(_src, alpha, Sse41::Shuffle32f<0x00>(_range), Sse41::Shuffle32f<0x00>(_factor), colors);
                    __m128i dst1 = Run4C<0x4, nofma>(_src, alpha, Sse41::Shuffle32f<0x55>(_range), Sse41::Shuffle32f<0x55>(_factor), colors);
                    __m128i dst2 = Run4C<0x8, nofma>(_src, alpha, Sse41::Shuffle32f<0xAA>(_range), Sse41::Shuffle32f<0xAA>(_factor), colors);
                    __m128i dst3 = Run4C<0xC, nofma>(_src, alpha, Sse41::Shuffle32f<0xFF>(_range), Sse41::Shuffle32f<0xFF>(_factor), colors);
                    return _mm_packus_epi16(_mm_packs_epi32(dst0, dst1), _mm_packs_epi32(dst2, dst3));
                }

                template<int part, bool nofma> static SIMD_INLINE __m256i Run8C(__m256i src, __m256 alpha, __m256 range, const __m256& factor, float* colors)
                {
                    __m256 _src = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm256_castsi256_si128(_mm256_permute4x64_epi64(src, 0x55 * part))));
                    __m256 _colors = Fmadd<nofma>(alpha, _src, Broadcast<part>(range), _mm256_loadu_ps(colors + part * F));
                    _mm256_storeu_ps(colors + part * F, _colors);
                    return _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_div_ps(_colors, Broadcast<part>(factor))));
                }

                template<bool nofma> static SIMD_INLINE __m256i Run8(const uint8_t* src, const uint8_t* diff, __m256 alpha, const float* ranges, float* factor, float* colors)
                {
                    static const __m256i PERM = SIMD_MM256_SETR_EPI32(0x0, 0x2, 0x4, 0x6, 0x1, 0x3, 0x5, 0x7);
                    __m256i _src = _mm256_loadu_si256((__m256i*)src);
                    __m256 _range = _mm256_i32gather_ps(ranges, _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)diff)), 4);
                    __m256 _factor = Fmadd<nofma>(_range, _mm256_loadu_ps(factor), alpha);
                    _mm256_storeu_ps(factor, _factor);
                    _range = _mm256_permutevar8x32_ps(_range, PERM);
                    _factor = _mm256_permutevar8x32_ps(_factor, PERM);
                    __m256i dst0 = Run8C<0, nofma>(_src, alpha, _range, _factor, colors);
                    __m256i dst1 = Run8C<1, nofma>(_src, alpha, _range, _factor, colors);
                    __m256i dst2 = Run8C<2, nofma>(_src, alpha, _range, _factor, colors);
                    __m256i dst3 = Run8C<3, nofma>(_src, alpha, _range, _factor, colors);
                    return PackI16ToU8(PackI32ToI16(dst0, dst1), PackI32ToI16(dst2, dst3));
                }

                template<int dir, bool nofma> static void Run(const uint8_t* src, const uint8_t* diff, size_t width, float alpha,
                    const float* ranges, float* factor, float* colors, uint8_t* dst)
                {
                    size_t width8 = AlignLo(width, 8), width4 = AlignLo(width, 4), x = 0, o = 0;
                    __m256 _alpha = _mm256_set1_ps(alpha);
                    for (; x < width8; x += 8, o += 32)
                        Set32<dir>(Run8<nofma>(src + o, diff + x, _alpha, ranges, factor + x, colors + o), dst + o);
                    for (; x < width4; x += 4, o += 16)
                        Set16<dir>(Run4<nofma>(src + o, diff + x, _mm256_castps256_ps128(_alpha), ranges, factor + x, colors + o), dst + o);
                    for (; x < width; x++)
                    {
                        float range = ranges[diff[x]];
                        factor[x] = alpha + range * factor[x];
                        for (size_t e = o + 4; o < e; o++)
                        {
                            colors[o] = alpha * src[o] + range * colors[o];
                            Set<dir>(int(colors[o] / factor[x]), dst + o);
                        }
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
                if(channels <= 2)
                    horFilter = FmaAvoid(param.flags) ? HorFilter<channels, type, true> : HorFilter<channels, type, false>;
                verFilter = FmaAvoid(param.flags) ? VerFilter<channels, type, true> : VerFilter<channels, type, false>;
            }

            template <RbfDiffType type> void Set(const RbfParam& param, FilterPtr& horFilter, FilterPtr& verFilter)
            {
                switch (param.channels)
                {
                case 1: Set<1, type>(param, horFilter, verFilter); break;
                case 2: Set<2, type>(param, horFilter, verFilter); break;
                case 3: Set<3, type>(param, horFilter, verFilter); break;
                case 4: Set<4, type>(param, horFilter, verFilter); break;
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

