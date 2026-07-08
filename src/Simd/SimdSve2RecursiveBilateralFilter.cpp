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
#include "Simd/SimdRecursiveBilateralFilter.h"
#include "Simd/SimdPerformance.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        typedef RecursiveBilateralFilter::FilterPtr FilterPtr;

        SIMD_INLINE svuint32_t Load8u(const uint8_t* src, const svuint32_t& offsets, const svbool_t& mask)
        {
            return svld1ub_gather_u32offset_u32(mask, src, offsets);
        }

        SIMD_INLINE svuint32_t AbsDiff8u(const uint8_t* src0, const uint8_t* src1, const svuint32_t& offsets, const svbool_t& mask)
        {
            svuint32_t s0 = Load8u(src0, offsets, mask);
            svuint32_t s1 = Load8u(src1, offsets, mask);
            return svsub_u32_x(mask, svmax_u32_x(mask, s0, s1), svmin_u32_x(mask, s0, s1));
        }

        template<RbfDiffType type> SIMD_INLINE svuint32_t Diff(const svuint32_t& ch0, const svuint32_t& ch1, const svbool_t& mask)
        {
            switch (type)
            {
            case RbfDiffAvg: return svlsr_n_u32_x(mask, svadd_n_u32_x(mask, svadd_u32_x(mask, ch0, ch1), 1), 1);
            case RbfDiffMax: return svmax_u32_x(mask, ch0, ch1);
            case RbfDiffSum: return svmin_n_u32_x(mask, svadd_u32_x(mask, ch0, ch1), 255);
            default:
                assert(0); return svdup_n_u32(0);
            }
        }

        template<RbfDiffType type> SIMD_INLINE svuint32_t Diff(const svuint32_t& ch0, const svuint32_t& ch1, const svuint32_t& ch2, const svbool_t& mask)
        {
            switch (type)
            {
            case RbfDiffAvg: return Diff<RbfDiffAvg>(ch1, Diff<RbfDiffAvg>(ch0, ch2, mask), mask);
            case RbfDiffMax: return svmax_u32_x(mask, svmax_u32_x(mask, ch0, ch1), ch2);
            case RbfDiffSum: return svmin_n_u32_x(mask, svadd_u32_x(mask, svadd_u32_x(mask, ch0, ch1), ch2), 255);
            default:
                assert(0); return svdup_n_u32(0);
            }
        }

        template<int channels, RbfDiffType type> SIMD_INLINE void RowRanges(const uint8_t* src0, const uint8_t* src1, size_t width, const float* ranges, float* dst)
        {
            size_t F = svcntw(), x = 0;
            const svbool_t body = svptrue_b32();
            const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), channels);
            for (; x < width; x += F)
            {
                svbool_t mask = svwhilelt_b32(x, width);
                const uint8_t* ps0 = src0 + x * channels;
                const uint8_t* ps1 = src1 + x * channels;
                svuint32_t diff = AbsDiff8u(ps0, ps1, offsets, mask);
                if (channels == 2)
                    diff = Diff<type>(diff, AbsDiff8u(ps0 + 1, ps1 + 1, offsets, mask), mask);
                if (channels >= 3)
                    diff = Diff<type>(diff, AbsDiff8u(ps0 + 1, ps1 + 1, offsets, mask), AbsDiff8u(ps0 + 2, ps1 + 2, offsets, mask), mask);
                svst1_f32(mask, dst + x, svld1_gather_u32index_f32(mask, ranges, diff));
            }
        }

        SIMD_INLINE svuint32_t Float32ToUint8(const svfloat32_t& value, const svbool_t& mask)
        {
            return svcvt_u32_f32_x(mask, value);
        }

        template<int channels> SIMD_INLINE void SetOut(const float* bc, const float* bf, const float* ec, const float* ef, size_t width, uint8_t* dst)
        {
            size_t F = svcntw(), x = 0;
            const svbool_t body = svptrue_b32();
            const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), channels);
            const svfloat32_t _1 = svdup_n_f32(1.0f);
            for (; x < width; x += F)
            {
                svbool_t mask = svwhilelt_b32(x, width);
                svfloat32_t factor = svdiv_f32_x(mask, _1, svadd_f32_x(mask, svld1_f32(mask, bf + x), svld1_f32(mask, ef + x)));
                size_t o = x * channels;
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t colors = svadd_f32_x(mask,
                        svld1_gather_u32index_f32(mask, bc + o + c, offsets),
                        svld1_gather_u32index_f32(mask, ec + o + c, offsets));
                    svst1b_scatter_u32offset_u32(mask, dst + o + c, offsets, Float32ToUint8(svmul_f32_x(mask, factor, colors), mask));
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        namespace Prec
        {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsequenced"
#endif
            template<int channels, RbfDiffType type> void HorFilter(const RbfParam& p, float* buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
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
                    RowRanges<channels, type>(src, src + channels, p.width - 1, p.ranges, rb0 + 1);
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
                size_t F = svcntw(), size = width * channels, x = 0, i = 0;
                const svbool_t body = svptrue_b32();
                const svfloat32_t _1 = svdup_n_f32(1.0f);
                for (; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    svst1_f32(mask, factor + x, _1);
                }
                for (; i < size; i += F)
                {
                    svbool_t mask = svwhilelt_b32(i, size);
                    svst1_f32(mask, colors + i, svcvt_f32_u32_x(mask, svld1ub_u32(mask, src + i)));
                }
            }

            //-----------------------------------------------------------------------------------------

            template<int channels> void VerSetMain(const uint8_t* hor, size_t width,
                float alpha, const float* ranges, const float* pf, const float* pc, float* cf, float* cc)
            {
                size_t F = svcntw(), x = 0;
                const svbool_t body = svptrue_b32();
                const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), channels);
                const svfloat32_t _alpha = svdup_n_f32(alpha);
                for (; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    svfloat32_t _ranges = svld1_f32(mask, ranges + x);
                    svst1_f32(mask, cf + x, svmla_f32_x(mask, _alpha, _ranges, svld1_f32(mask, pf + x)));
                    size_t o = x * channels;
                    for (size_t c = 0; c < channels; ++c)
                    {
                        svfloat32_t color = svmla_f32_x(mask,
                            svmul_f32_x(mask, _alpha, svcvt_f32_u32_x(mask, Load8u(hor + o + c, offsets, mask))),
                            _ranges, svld1_gather_u32index_f32(mask, pc + o + c, offsets));
                        svst1_scatter_u32index_f32(mask, cc + o + c, offsets, color);
                    }
                }
            }

            //-----------------------------------------------------------------------------------------

            template<int channels, RbfDiffType type> void VerFilter(const RbfParam& p, float* buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
            {
                //SIMD_PERF_FUNC();
                size_t size = p.width * channels;
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
                    RowRanges<channels, type>(suc, suc + srcStride, p.width, p.ranges, rb0);
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
                    RowRanges<channels, type>(src, src - srcStride, p.width, p.ranges, rb0);
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

        RecursiveBilateralFilterPrecize::RecursiveBilateralFilterPrecize(const RbfParam& param)
            : Base::RecursiveBilateralFilterPrecize(param)
        {
            Prec::Set(_param, _hFilter, _vFilter);
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

            template<int dir> SIMD_INLINE void Set(const svbool_t& mask, const svuint32_t& value, uint8_t* dst);

            template<> SIMD_INLINE void Set<+1>(const svbool_t& mask, const svuint32_t& value, uint8_t* dst)
            {
                svst1b_u32(mask, dst, value);
            }

            template<> SIMD_INLINE void Set<-1>(const svbool_t& mask, const svuint32_t& value, uint8_t* dst)
            {
                svuint32_t sum = svadd_u32_x(mask, svld1ub_u32(mask, dst), value);
                svst1b_u32(mask, dst, svlsr_n_u32_x(mask, svadd_n_u32_x(mask, sum, 1), 1));
            }

            template<int dir> SIMD_INLINE void Set(const svbool_t& mask, const svuint32_t& offsets, const svuint32_t& value, uint8_t* dst);

            template<> SIMD_INLINE void Set<+1>(const svbool_t& mask, const svuint32_t& offsets, const svuint32_t& value, uint8_t* dst)
            {
                svst1b_scatter_u32offset_u32(mask, dst, offsets, value);
            }

            template<> SIMD_INLINE void Set<-1>(const svbool_t& mask, const svuint32_t& offsets, const svuint32_t& value, uint8_t* dst)
            {
                svuint32_t sum = svadd_u32_x(mask, svld1ub_gather_u32offset_u32(mask, dst, offsets), value);
                svst1b_scatter_u32offset_u32(mask, dst, offsets, svlsr_n_u32_x(mask, svadd_n_u32_x(mask, sum, 1), 1));
            }

            //-----------------------------------------------------------------------------------------

            template<int channels, RbfDiffType type> SIMD_INLINE void RowDiff(const uint8_t* src0, const uint8_t* src1, size_t width, uint8_t* dst)
            {
                size_t F = svcntw(), x = 0;
                const svbool_t body = svptrue_b32();
                const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), channels);
                for (; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    const uint8_t* ps0 = src0 + x * channels;
                    const uint8_t* ps1 = src1 + x * channels;
                    svuint32_t diff = AbsDiff8u(ps0, ps1, offsets, mask);
                    if (channels == 2)
                        diff = Diff<type>(diff, AbsDiff8u(ps0 + 1, ps1 + 1, offsets, mask), mask);
                    if (channels >= 3)
                        diff = Diff<type>(diff, AbsDiff8u(ps0 + 1, ps1 + 1, offsets, mask), AbsDiff8u(ps0 + 2, ps1 + 2, offsets, mask), mask);
                    svst1b_u32(mask, dst + x, diff);
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
                if (dir == -1 && width > 1)
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
                size_t F = svcntw(), size = width * channels, x = 0, i = 0;
                const svfloat32_t _1 = svdup_n_f32(1.0f);
                for (; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    svst1_f32(mask, factor + x, _1);
                }
                for (; i < size; i += F)
                {
                    svbool_t mask = svwhilelt_b32(i, size);
                    svuint32_t src8 = svld1ub_u32(mask, src + i);
                    svst1_f32(mask, colors + i, svcvt_f32_u32_x(mask, src8));
                    Set<dir>(mask, src8, dst + i);
                }
            }

            template<int channels, int dir> void VerMain(const uint8_t* src, const uint8_t* diff, size_t width, float alpha,
                const float* ranges, float* factor, float* colors, uint8_t* dst)
            {
                size_t F = svcntw(), x = 0;
                const svbool_t body = svptrue_b32();
                const svuint32_t offsets = svmul_n_u32_x(body, svindex_u32(0, 1), channels);
                const svfloat32_t _alpha = svdup_n_f32(alpha);
                for (; x < width; x += F)
                {
                    svbool_t mask = svwhilelt_b32(x, width);
                    svfloat32_t _range = svld1_gather_u32index_f32(mask, ranges, svld1ub_u32(mask, diff + x));
                    svfloat32_t _factor = svmla_f32_x(mask, _alpha, _range, svld1_f32(mask, factor + x));
                    svst1_f32(mask, factor + x, _factor);
                    size_t o = x * channels;
                    for (size_t c = 0; c < channels; ++c)
                    {
                        svfloat32_t _color = svmla_f32_x(mask,
                            svmul_f32_x(mask, _alpha, svcvt_f32_u32_x(mask, Load8u(src + o + c, offsets, mask))),
                            _range, svld1_gather_u32index_f32(mask, colors + o + c, offsets));
                        svst1_scatter_u32index_f32(mask, colors + o + c, offsets, _color);
                        Set<dir>(mask, offsets, Float32ToUint8(svdiv_f32_x(mask, _color, _factor), mask), dst + o + c);
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

        RecursiveBilateralFilterFast::RecursiveBilateralFilterFast(const RbfParam& param)
            : Base::RecursiveBilateralFilterFast(param)
        {
            Fast::Set(_param, _hFilter, _vFilter);
        }

        //-----------------------------------------------------------------------------------------

        void* RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels,
            const float* sigmaSpatial, const float* sigmaRange, SimdRecursiveBilateralFilterFlags flags)
        {
            RbfParam param(width, height, channels, sigmaSpatial, sigmaRange, flags, svcntb());
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
