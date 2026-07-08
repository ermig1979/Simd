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
#include "Simd/SimdMemory.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdRecursiveBilateralFilter.h"
#include "Simd/SimdPerformance.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        typedef RecursiveBilateralFilter::FilterPtr FilterPtr;

        SIMD_INLINE uint8x16_t AbsDiff8u(const uint8_t* src0, const uint8_t* src1)
        {
            return vabdq_u8(vld1q_u8(src0), vld1q_u8(src1));
        }

        template<RbfDiffType type> SIMD_INLINE int Diff(int ch0, int ch1)
        {
            switch (type)
            {
            case RbfDiffAvg: return Base::Average(ch0, ch1);
            case RbfDiffMax: return Max(ch0, ch1);
            case RbfDiffSum: return Min(ch0 + ch1, 255);
            default:
                assert(0); return 0;
            }
        }

        template<RbfDiffType type> SIMD_INLINE int Diff(int ch0, int ch1, int ch2)
        {
            switch (type)
            {
            case RbfDiffAvg: return Base::Average(ch1, Base::Average(ch0, ch2));
            case RbfDiffMax: return Max(Max(ch0, ch1), ch2);
            case RbfDiffSum: return Min(ch0 + ch1 + ch2, 255);
            default:
                assert(0); return 0;
            }
        }

        SIMD_INLINE float32x4_t LoadAs32f(const uint8_t* src)
        {
            return vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8((uint8x8_t)vdup_n_u32(*(uint32_t*)src)))));
        }

        SIMD_INLINE void StoreAs8u(uint8_t* dst, const float32x4_t& value)
        {
            uint32x4_t u32 = vcvtq_u32_f32(value);
            uint16x4_t u16 = vqmovn_u32(u32);
            uint8x8_t u8 = vqmovn_u16(vcombine_u16(u16, vdup_n_u16(0)));
            ((uint32_t*)dst)[0] = vget_lane_u32(vreinterpret_u32_u8(u8), 0);
        }

        SIMD_INLINE float32x4_t LoadFactor(const float* factors, size_t channels, size_t offset)
        {
            SIMD_ALIGNED(16) float tmp[F];
            for (size_t i = 0; i < F; ++i)
                tmp[i] = factors[(offset + i) / channels];
            return vld1q_f32(tmp);
        }

        template<int channels, RbfDiffType type> SIMD_INLINE void RowRanges(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
        {
            for (size_t x = 0, o = 0; x < width; x += 1, o += channels)
                dst[x] = ranges[Base::Diff<channels, type>(src0 + o, src1 + o)];
        }

        //-----------------------------------------------------------------------------------------

        template<RbfDiffType type> SIMD_INLINE void Ranges1(const uint8_t* src0, const uint8_t* src1, const float* ranges, float* dst)
        {
            SIMD_ALIGNED(16) uint8_t diff[A];
            vst1q_u8(diff, AbsDiff8u(src0, src1));
            for (size_t i = 0; i < A; ++i)
                dst[i] = ranges[diff[i]];
        }

        template<RbfDiffType type> SIMD_INLINE void Ranges2(const uint8_t* src0, const uint8_t* src1, const float* ranges, float* dst)
        {
            SIMD_ALIGNED(16) uint8_t diff[A];
            vst1q_u8(diff, AbsDiff8u(src0, src1));
            for (size_t i = 0, o = 0; i < HA; ++i, o += 2)
                dst[i] = ranges[Diff<type>(diff[o + 0], diff[o + 1])];
        }

        template<RbfDiffType type> SIMD_INLINE void Ranges3(const uint8_t* src0, const uint8_t* src1, const float* ranges, float* dst)
        {
            SIMD_ALIGNED(16) uint8_t diff[A];
            vst1q_u8(diff, AbsDiff8u(src0, src1));
            for (size_t i = 0, o = 0; i < F; ++i, o += 3)
                dst[i] = ranges[Diff<type>(diff[o + 0], diff[o + 1], diff[o + 2])];
        }

        template<RbfDiffType type> SIMD_INLINE void Ranges4(const uint8_t* src0, const uint8_t* src1, const float* ranges, float* dst)
        {
            SIMD_ALIGNED(16) uint8_t diff[A];
            vst1q_u8(diff, AbsDiff8u(src0, src1));
            for (size_t i = 0, o = 0; i < F; ++i, o += 4)
                dst[i] = ranges[Diff<type>(diff[o + 0], diff[o + 1], diff[o + 2])];
        }

        //-----------------------------------------------------------------------------------------

        namespace Prec
        {
            template<int channels> struct RowRanges
            {
                template<RbfDiffType type> static void Run(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst);
            };

            template<> struct RowRanges<1>
            {
                template<RbfDiffType type> static void Run(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
                {
                    if (width < A)
                    {
                        Neon::RowRanges<1, type>(src0, src1, width, ranges, dst);
                        return;
                    }
                    size_t widthA = AlignLo(width, A), x = 0;
                    for (; x < widthA; x += A)
                        Ranges1<type>(src0 + x, src1 + x, ranges, dst + x);
                    if (widthA < width)
                    {
                        x = width - A;
                        Ranges1<type>(src0 + x, src1 + x, ranges, dst + x);
                    }
                }
            };

            template<> struct RowRanges<2>
            {
                template<RbfDiffType type> static void Run(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
                {
                    if (width < HA)
                    {
                        Neon::RowRanges<2, type>(src0, src1, width, ranges, dst);
                        return;
                    }
                    size_t widthHA = AlignLo(width, HA), x = 0, o = 0;
                    for (; x < widthHA; x += HA, o += A)
                        Ranges2<type>(src0 + o, src1 + o, ranges, dst + x);
                    if (widthHA < width)
                    {
                        x = width - HA, o = x * 2;
                        Ranges2<type>(src0 + o, src1 + o, ranges, dst + x);
                    }
                }
            };

            template<> struct RowRanges<3>
            {
                template<RbfDiffType type> static void Run(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
                {
                    if (width < F)
                    {
                        Neon::RowRanges<3, type>(src0, src1, width, ranges, dst);
                        return;
                    }
                    size_t widthF = AlignLo(width, F), x = 0, o = 0;
                    for (; x < widthF; x += F, o += F * 3)
                        Ranges3<type>(src0 + o, src1 + o, ranges, dst + x);
                    if (widthF < width)
                    {
                        x = width - F, o = x * 3;
                        Ranges3<type>(src0 + o, src1 + o, ranges, dst + x);
                    }
                }
            };

            template<> struct RowRanges<4>
            {
                template<RbfDiffType type> static void Run(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
                {
                    if (width < F)
                    {
                        Neon::RowRanges<4, type>(src0, src1, width, ranges, dst);
                        return;
                    }
                    size_t widthF = AlignLo(width, F), x = 0, o = 0;
                    for (; x < widthF; x += F, o += A)
                        Ranges4<type>(src0 + o, src1 + o, ranges, dst + x);
                    if (widthF < width)
                    {
                        x = width - F, o = x * 4;
                        Ranges4<type>(src0 + o, src1 + o, ranges, dst + x);
                    }
                }
            };

            //-----------------------------------------------------------------------------------------

            template<int channels> SIMD_INLINE void SetOut(const float* bc, const float* bf, const float* ec, const float* ef, size_t width, uint8_t* dst)
            {
                size_t widthF = AlignLo(width, F), x = 0, o = 0;
                SIMD_ALIGNED(16) float factors[F];
                for (; x < widthF; x += F, o += channels * F)
                {
                    for (size_t i = 0; i < F; ++i)
                        factors[i] = 1.0f / (bf[x + i] + ef[x + i]);
                    for (size_t i = 0; i < channels * F; i += F)
                    {
                        float32x4_t factor = LoadFactor(factors, channels, i);
                        float32x4_t colors = vaddq_f32(Load<false>(bc + o + i), Load<false>(ec + o + i));
                        StoreAs8u(dst + o + i, vmulq_f32(factor, colors));
                    }
                }
                for (; x < width; x++, o += channels)
                {
                    float factor = 1.0f / (bf[x] + ef[x]);
                    for (int c = 0; c < channels; c++)
                        dst[o + c] = uint8_t(factor * (bc[o + c] + ec[o + c]));
                }
            }

            //-----------------------------------------------------------------------------------------

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsequenced"
#endif
            template<int channels, RbfDiffType type> void HorFilter(const RbfParam& p, float * buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
            {
                //SIMD_PERF_FUNC();
                size_t size = p.width * channels, cLast = size - 1, fLast = p.width - 1;
                float* cb0 = buf, * cb1 = cb0 + size, * fb0 = cb1 + size, * fb1 = fb0 + p.width, * rb0 = fb1 + p.width;
                for (size_t y = 0; y < p.height; y++)
                {
                    const uint8_t* sl = src, * sr = src + cLast;
                    float* lc = cb0, * rc = cb1 + cLast;
                    float* lf = fb0, * rf = fb1 + fLast;
                    *lf++ = 1.f;
                    *rf-- = 1.f;
                    for (int c = 0; c < channels; c++)
                    {
                        *lc++ = *sl++;
                        *rc-- = *sr--;
                    }
                    RowRanges<channels>::template Run<type>(src, src + channels, p.width - 1, p.ranges, rb0 + 1);
                    for (size_t x = 1; x < p.width; x++)
                    {
                        float la = rb0[x];
                        float ra = rb0[p.width - x];
                        *lf++ = p.alpha + la * lf[-1];
                        *rf-- = p.alpha + ra * rf[+1];
                        for (int c = 0; c < channels; c++)
                        {
                            *lc++ = (p.alpha * (*sl++) + la * lc[-channels]);
                            *rc-- = (p.alpha * (*sr--) + ra * rc[+channels]);
                        }
                    }
                    SetOut<channels>(cb0, fb0, cb1, fb1, p.width, dst);
                    src += srcStride;
                    dst += dstStride;
                }
            }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

            //-----------------------------------------------------------------------------------------

            template<int channels> void VerSetEdge(const uint8_t* src, size_t width, float* factor, float* colors)
            {
                size_t widthF = AlignLo(width, F);
                size_t x = 0;
                float32x4_t _1 = vdupq_n_f32(1.0f);
                for (; x < widthF; x += F)
                    Store<false>(factor + x, _1);
                for (; x < width; x++)
                    factor[x] = 1.0f;

                size_t size = width * channels, sizeF = AlignLo(size, F);
                size_t i = 0;
                for (; i < sizeF; i += F)
                    Store<false>(colors + i, LoadAs32f(src + i));
                for (; i < size; i++)
                    colors[i] = src[i];
            }

            //-----------------------------------------------------------------------------------------

            template<int channels> void VerSetMain(const uint8_t* hor, size_t width,
                float alpha, const float* ranges, const float* pf, const float* pc, float* cf, float* cc)
            {
                size_t widthF = AlignLo(width, F), x = 0, o = 0;
                float32x4_t _alpha = vdupq_n_f32(alpha);
                for (; x < widthF; x += F, o += channels * F)
                {
                    float32x4_t _ranges = Load<false>(ranges + x);
                    Store<false>(cf + x, vaddq_f32(_alpha, vmulq_f32(_ranges, Load<false>(pf + x))));
                    for (size_t i = 0; i < channels * F; i += F)
                    {
                        float32x4_t range = LoadFactor(ranges + x, channels, i);
                        float32x4_t color = vaddq_f32(vmulq_f32(_alpha, LoadAs32f(hor + o + i)), vmulq_f32(range, Load<false>(pc + o + i)));
                        Store<false>(cc + o + i, color);
                    }
                }
                for (; x < width; x++, o += channels)
                {
                    cf[x] = alpha + ranges[x] * pf[x];
                    for (int c = 0; c < channels; c++)
                        cc[o + c] = alpha * hor[o + c] + ranges[x] * pc[o + c];
                }
            }

            //-----------------------------------------------------------------------------------------

            template<int channels, RbfDiffType type> void VerFilter(const RbfParam& p, float * buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
            {
                //SIMD_PERF_FUNC();
                size_t size = p.width * channels, srcTail = srcStride - size, dstTail = dstStride - size;
                float* rb0 = buf, * dcb = rb0 + p.width, * dfb = dcb + size * 2, * ucb = dfb + p.width * 2, * ufb = ucb + size * p.height;

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
                    RowRanges<channels>::template Run<type>(suc, suc + srcStride, p.width, p.ranges, rb0);
                    VerSetMain<channels>(duc, p.width, p.alpha, rb0, uf + p.width, uc + size, uf, uc);
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
                    RowRanges<channels>::template Run<type>(src, src - srcStride, p.width, p.ranges, rb0);
                    VerSetMain<channels>(dst, p.width, p.alpha, rb0, dpf, dpc, df, dc);
                    SetOut<channels>(dc, df, ucb + y * size, ufb + y * p.width, p.width, dst);
                }
            }

            //-----------------------------------------------------------------------------------------

            template <int channels, RbfDiffType type> void Set(FilterPtr& horFilter, FilterPtr& verFilter)
            {
                horFilter = HorFilter<channels, type>;
                verFilter = VerFilter<channels, type>;
            }

            template <RbfDiffType type> void Set(size_t channels, FilterPtr& horFilter, FilterPtr& verFilter)
            {
                switch (channels)
                {
                case 1: Set<1, type>(horFilter, verFilter); break;
                case 2: Set<2, type>(horFilter, verFilter); break;
                case 3: Set<3, type>(horFilter, verFilter); break;
                case 4: Set<4, type>(horFilter, verFilter); break;
                default:
                    assert(0);
                }
            }

            void Set(const RbfParam& param, FilterPtr& horFilter, FilterPtr& verFilter)
            {
                switch (DiffType(param.flags))
                {
                case RbfDiffAvg: Set<RbfDiffAvg>(param.channels, horFilter, verFilter); break;
                case RbfDiffMax: Set<RbfDiffAvg>(param.channels, horFilter, verFilter); break;
                case RbfDiffSum: Set<RbfDiffAvg>(param.channels, horFilter, verFilter); break;
                default:
                    assert(0);
                }
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

            template<int dir> SIMD_INLINE void Set16(uint8x16_t value, uint8_t* dst);

            template<> SIMD_INLINE void Set16<+1>(uint8x16_t value, uint8_t* dst)
            {
                Store<false>(dst, value);
            }

            template<> SIMD_INLINE void Set16<-1>(uint8x16_t value, uint8_t* dst)
            {
                Store<false>(dst, vrhaddq_u8(Load<false>(dst), value));
            }

            //-----------------------------------------------------------------------------------------

            template<int channels, RbfDiffType type> SIMD_INLINE void RowDiff(const uint8_t* src0, const uint8_t* src1, size_t width, uint8_t* dst)
            {
                switch (channels)
                {
                case 1:
                {
                    for (size_t x = 0; x < width; x += A)
                        Store<false>(dst + x, AbsDiff8u(src0 + x, src1 + x));
                    break;
                }
                case 2:
                {
                    SIMD_ALIGNED(16) uint8_t diff[A];
                    for (size_t x = 0, o = 0; x < width; x += A, o += 2 * A)
                    {
                        Store<false>(diff, AbsDiff8u(src0 + o + 0, src1 + o + 0));
                        for (size_t i = 0, c = 0; i < HA; ++i, c += 2)
                            dst[x + i] = (uint8_t)Diff<type>(diff[c + 0], diff[c + 1]);
                        Store<false>(diff, AbsDiff8u(src0 + o + A, src1 + o + A));
                        for (size_t i = 0, c = 0; i < HA; ++i, c += 2)
                            dst[x + HA + i] = (uint8_t)Diff<type>(diff[c + 0], diff[c + 1]);
                    }
                    break;
                }
                case 3:
                {
                    SIMD_ALIGNED(16) uint8_t diff[A];
                    for (size_t x = 0, o = 0; x < width; x += A, o += 3 * A)
                    {
                        for (size_t j = 0; j < 4; ++j)
                        {
                            Store<false>(diff, AbsDiff8u(src0 + o + j * 12, src1 + o + j * 12));
                            for (size_t i = 0, c = 0; i < F; ++i, c += 3)
                                dst[x + j * F + i] = (uint8_t)Diff<type>(diff[c + 0], diff[c + 1], diff[c + 2]);
                        }
                    }
                    break;
                }
                case 4:
                {
                    SIMD_ALIGNED(16) uint8_t diff[A];
                    for (size_t x = 0, o = 0; x < width; x += A, o += 4 * A)
                    {
                        for (size_t j = 0; j < 4; ++j)
                        {
                            Store<false>(diff, AbsDiff8u(src0 + o + j * A, src1 + o + j * A));
                            for (size_t i = 0, c = 0; i < F; ++i, c += 4)
                                dst[x + j * F + i] = (uint8_t)Diff<type>(diff[c + 0], diff[c + 1], diff[c + 2]);
                        }
                    }
                    break;
                }
                default:
                    for (size_t x = 0, o = 0; x < width; x += 1, o += channels)
                        dst[x] = (uint8_t)Base::Diff<channels, type>(src0 + o, src1 + o);
                }
            }

            template<int channels, RbfDiffType type> void RowDiff4x(const uint8_t* src0, const uint8_t* src1, size_t srcStride, size_t width, uint8_t* dst, size_t dstStride)
            {
                for (size_t i = 0; i < 4; ++i)
                {
                    RowDiff<channels, type>(src0, src1, width, dst);
                    src0 += srcStride;
                    src1 += srcStride;
                    dst += dstStride;
                }
            }

            //-----------------------------------------------------------------------------------------

            template<int channels, int dir> void HorRow(const uint8_t* src, size_t width, float alpha, const float* ranges, uint8_t* diff, uint8_t* dst)
            {
                if (dir == -1)
                    diff += width - 2;
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

            template<int channels, RbfDiffType type> void HorFilter(const RbfParam& p, float* buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
            {
                size_t last = (p.width - 1) * channels, height4 = AlignLo(p.height, 4), y = 0;
                uint8_t* diff = (uint8_t*)buf;
                for (; y < height4; y += 4)
                {
                    RowDiff4x<channels, type>(src, src + channels, srcStride, p.width - 1, diff, dstStride);
                    for (size_t i = 0; i < 4; ++i)
                    {
                        HorRow<channels, +1>(src + i * srcStride, p.width, p.alpha, p.ranges, diff + i * dstStride, dst + i * dstStride);
                        HorRow<channels, -1>(src + i * srcStride + last, p.width, p.alpha, p.ranges, diff + i * dstStride, dst + i * dstStride + last);
                    }
                    src += 4 * srcStride;
                    dst += 4 * dstStride;
                }
                for (; y < p.height; y++)
                {
                    RowDiff<channels, type>(src, src + channels, p.width - 1, diff);
                    HorRow<channels, +1>(src, p.width, p.alpha, p.ranges, diff, dst);
                    HorRow<channels, -1>(src + last, p.width, p.alpha, p.ranges, diff, dst + last);
                    src += srcStride;
                    dst += dstStride;
                }
            }

            //-----------------------------------------------------------------------------------------

            template<int channels, int dir> void VerEdge(const uint8_t* src, size_t width, float* factor, float* colors, uint8_t* dst)
            {
                size_t widthF = AlignLo(width, F), x = 0;
                float32x4_t _1 = vdupq_n_f32(1.0f);
                for (; x < widthF; x += F)
                    Store<false>(factor + x, _1);
                for (; x < width; x++)
                    factor[x] = 1.0f;

                size_t size = width * channels, sizeF = AlignLo(size, F), i = 0;
                for (; i < sizeF; i += F)
                    Store<false>(colors + i, LoadAs32f(src + i));
                for (; i < size; i++)
                    colors[i] = src[i];

                size_t sizeA = AlignLo(size, A);
                for (i = 0; i < sizeA; i += A)
                    Set16<dir>(Load<false>(src + i), dst + i);
                for (; i < size; i += 1)
                    Set<dir>(src[i], dst + i);
            }

            template<int channels, int dir> void VerMain(const uint8_t* src, const uint8_t* diff, size_t width, float alpha,
                const float* ranges, float* factor, float* colors, uint8_t* dst)
            {
                size_t widthF = AlignLo(width, F), x = 0, o = 0;
                float32x4_t _alpha = vdupq_n_f32(alpha);
                SIMD_ALIGNED(16) float range[F], factorValue[F], colorValue[F];
                for (; x < widthF; x += F, o += channels * F)
                {
                    for (size_t i = 0; i < F; ++i)
                        range[i] = ranges[diff[x + i]];
                    float32x4_t _range = Load<false>(range);
                    float32x4_t _factor = vaddq_f32(_alpha, vmulq_f32(_range, Load<false>(factor + x)));
                    Store<false>(factor + x, _factor);
                    Store<false>(factorValue, _factor);
                    for (size_t i = 0; i < channels * F; i += F)
                    {
                        float32x4_t _ranges = LoadFactor(range, channels, i);
                        float32x4_t _colors = vaddq_f32(vmulq_f32(_alpha, LoadAs32f(src + o + i)), vmulq_f32(_ranges, Load<false>(colors + o + i)));
                        Store<false>(colors + o + i, _colors);
                        Store<false>(colorValue, _colors);
                        for (size_t j = 0; j < F; ++j)
                            Set<dir>(int(colorValue[j] / factorValue[(i + j) / channels]), dst + o + i + j);
                    }
                }
                for (; x < width; x++)
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

            template<int channels, RbfDiffType type> void VerFilter(const RbfParam& p, float* buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
            {
                size_t size = p.width * channels;
                uint8_t* diff = (uint8_t*)(buf + size + p.width);
                VerEdge<channels, +1>(src, p.width, buf + size, buf, dst);
                for (size_t y = 1; y < p.height; y++)
                {
                    src += srcStride;
                    dst += dstStride;
                    RowDiff<channels, type>(src, src - srcStride, p.width, diff);
                    VerMain<channels, +1>(src, diff, p.width, p.alpha, p.ranges, buf + size, buf, dst);
                }
                VerEdge<channels, -1>(src, p.width, buf + size, buf, dst);
                for (size_t y = 1; y < p.height; y++)
                {
                    src -= srcStride;
                    dst -= dstStride;
                    RowDiff<channels, type>(src, src + srcStride, p.width, diff);
                    VerMain<channels, -1>(src, diff, p.width, p.alpha, p.ranges, buf + size, buf, dst);
                }
            }

            //-----------------------------------------------------------------------------------------

            template <int channels, RbfDiffType type> void Set(FilterPtr& horFilter, FilterPtr& verFilter)
            {
                horFilter = HorFilter<channels, type>;
                verFilter = VerFilter<channels, type>;
            }

            template <RbfDiffType type> void Set(size_t channels, FilterPtr& horFilter, FilterPtr& verFilter)
            {
                switch (channels)
                {
                case 1: Set<1, type>(horFilter, verFilter); break;
                case 2: Set<2, type>(horFilter, verFilter); break;
                case 3: Set<3, type>(horFilter, verFilter); break;
                case 4: Set<4, type>(horFilter, verFilter); break;
                default:
                    assert(0);
                }
            }

            void Set(const RbfParam& param, FilterPtr& horFilter, FilterPtr& verFilter)
            {
                switch (DiffType(param.flags))
                {
                case RbfDiffAvg: Set<RbfDiffAvg>(param.channels, horFilter, verFilter); break;
                case RbfDiffMax: Set<RbfDiffAvg>(param.channels, horFilter, verFilter); break;
                case RbfDiffSum: Set<RbfDiffAvg>(param.channels, horFilter, verFilter); break;
                default:
                    assert(0);
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        RecursiveBilateralFilterPrecize::RecursiveBilateralFilterPrecize(const RbfParam& param)
            : Base::RecursiveBilateralFilterPrecize(param)
        {
            if (_param.width * _param.channels >= A)
                Prec::Set(_param, _hFilter, _vFilter);
        }

        //-----------------------------------------------------------------------------------------

        RecursiveBilateralFilterFast::RecursiveBilateralFilterFast(const RbfParam& param)
            : Base::RecursiveBilateralFilterFast(param)
        {
            if (_param.width >= A)
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
                return new RecursiveBilateralFilterPrecize(param);
            else
                return new RecursiveBilateralFilterFast(param);
        }
    }
#endif
}
