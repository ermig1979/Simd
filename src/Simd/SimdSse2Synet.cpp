/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template<int shift> SIMD_INLINE __m128 LoadAtEdge(const float * src)
        {
            static const int32_t mask[3 * F] = { 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0 };
            return _mm_and_ps(_mm_loadu_ps(src + shift), _mm_loadu_ps((float*)mask + F + shift));
        }

        SIMD_INLINE __m128 NoseSquareSum(const float * src)
        {
            return _mm_add_ps(_mm_add_ps(Sse::Square(LoadAtEdge<-2>(src)), Sse::Square(LoadAtEdge<-1>(src))),
                _mm_add_ps(Sse::Square(_mm_loadu_ps(src)), _mm_add_ps(Sse::Square(_mm_loadu_ps(src + 1)), Sse::Square(_mm_loadu_ps(src + 2)))));
        }

        SIMD_INLINE __m128 BodySquareSum(const float * src)
        {
            return _mm_add_ps(_mm_add_ps(Sse::Square(_mm_loadu_ps(src - 2)), Sse::Square(_mm_loadu_ps(src - 1))),
                _mm_add_ps(Sse::Square(_mm_loadu_ps(src)), _mm_add_ps(Sse::Square(_mm_loadu_ps(src + 1)), Sse::Square(_mm_loadu_ps(src + 2)))));
        }

        SIMD_INLINE __m128 TailSquareSum(const float * src)
        {
            return _mm_add_ps(_mm_add_ps(Sse::Square(LoadAtEdge<2>(src)), Sse::Square(LoadAtEdge<1>(src))),
                _mm_add_ps(Sse::Square(_mm_loadu_ps(src)), _mm_add_ps(Sse::Square(_mm_loadu_ps(src - 1)), Sse::Square(_mm_loadu_ps(src - 2)))));
        }

        template<bool align> void SynetLrnLayerCrossChannelsNchw(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            __m128 k0 = _mm_set1_ps(k[0]);
            __m128 k1 = _mm_set1_ps(k[1]);
            __m128 k2 = _mm_set1_ps(k[2]);
            Sse2::Pow pow;
            Array32f sum(spatial, true), zero(spatial, true);
            size_t aligned = AlignLo(spatial, F);
            for (size_t c = 0; c < half; ++c)
            {
                const float * pos = src + c * spatial;
                size_t s = 0;
                for (; s < aligned; s += F)
                {
                    __m128 _pos = Sse::Load<align>(pos + s);
                    Sse::Store<true>(sum.data + s, _mm_add_ps(Sse::Load<true>(sum.data + s), _mm_mul_ps(_pos, _pos)));
                }
                for (; s < spatial; ++s)
                    sum[s] += Simd::Square(pos[s]);
            }
            for (size_t c = 0; c < channels; ++c)
            {
                const float * pos = (c < channels - half) ? src + half * spatial : zero.data;
                const float * neg = (c > half) ? src - (half + 1) * spatial : zero.data;
                size_t s = 0;
                for (; s < aligned; s += F)
                {
                    __m128 _pos = Sse::Load<align>(pos + s);
                    __m128 _neg = Sse::Load<align>(neg + s);
                    __m128 _sum = Sse::Load<true>(sum.data + s);
                    _sum = _mm_add_ps(_sum, _mm_sub_ps(_mm_mul_ps(_pos, _pos), _mm_mul_ps(_neg, _neg)));
                    __m128 _src = Sse::Load<align>(src + s);
                    Sse::Store<true>(sum.data + s, _sum);
                    Sse::Store<align>(dst + s, _mm_mul_ps(_src, pow(_mm_add_ps(k0, _mm_mul_ps(k1, _sum)), k2)));
                }
                for (; s < spatial; ++s)
                {
                    sum[s] += Simd::Square(pos[s]);
                    sum[s] -= Simd::Square(neg[s]);
                    dst[s] = src[s] * Base::Pow(k[0] + k[1] * sum[s], k[2]);
                }
                src += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetLrnLayerCrossChannelsNchw(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            if (Aligned(src) && Aligned(dst) && Aligned(spatial, F))
                SynetLrnLayerCrossChannelsNchw<true>(src, half, channels, spatial, k, dst);
            else
                SynetLrnLayerCrossChannelsNchw<false>(src, half, channels, spatial, k, dst);
        }

        template<bool align> void SynetLrnLayerCrossChannelsNhwc2h(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            __m128 k0 = _mm_set1_ps(k[0]);
            __m128 k1 = _mm_set1_ps(k[1]);
            __m128 k2 = _mm_set1_ps(k[2]);
            Sse2::Pow pow;
            size_t aligned = AlignLo(channels - half, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                Sse::Store<align>(dst + 0, _mm_mul_ps(Sse::Load<align>(src + 0), pow(_mm_add_ps(k0, _mm_mul_ps(k1, NoseSquareSum(src + 0))), k2)));
                for (size_t c = F; c < aligned; c += F)
                    Sse::Store<align>(dst + c, _mm_mul_ps(Sse::Load<align>(src + c), pow(_mm_add_ps(k0, _mm_mul_ps(k1, BodySquareSum(src + c))), k2)));
                if (aligned != channels - half)
                {
                    size_t c = channels - half - F;
                    Sse::Store<false>(dst + c, _mm_mul_ps(Sse::Load<false>(src + c), pow(_mm_add_ps(k0, _mm_mul_ps(k1, BodySquareSum(src + c))), k2)));
                }
                size_t c = channels - F;
                Sse::Store<false>(dst + c, _mm_mul_ps(Sse::Load<false>(src + c), pow(_mm_add_ps(k0, _mm_mul_ps(k1, TailSquareSum(src + c))), k2)));
                src += channels;
                dst += channels;
            }
        }

        SIMD_INLINE void SynetLrnLayerCrossChannelsNhwc(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            if (half == 2 && channels >= F + half)
            {
                if (Aligned(src) && Aligned(dst) && Aligned(channels, F))
                    SynetLrnLayerCrossChannelsNhwc2h<true>(src, half, channels, spatial, k, dst);
                else
                    SynetLrnLayerCrossChannelsNhwc2h<false>(src, half, channels, spatial, k, dst);
            }
            else
                Base::SynetLrnLayerCrossChannels(src, half, channels, spatial, k, dst, SimdTensorFormatNhwc);
        }

        void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNchw)
                SynetLrnLayerCrossChannelsNchw(src, half, channels, spatial, k, dst);
            else if (format == SimdTensorFormatNhwc)
                SynetLrnLayerCrossChannelsNhwc(src, half, channels, spatial, k, dst);
            else
                Base::SynetLrnLayerCrossChannels(src, half, channels, spatial, k, dst, format);
        }

        //---------------------------------------------------------------------

        void SynetSoftmaxLayerForward21(const float* src, size_t outer, float* dst)
        {
            Sse2::Exp exp;
            size_t aligned = Simd::AlignLo(outer, F);
            size_t o = 0;
            for (; o < aligned; o += F)
            {
                __m128 s0 = _mm_loadu_ps(src + 0);
                __m128 s1 = _mm_loadu_ps(src + F);
                __m128 ss0 = _mm_shuffle_ps(s0, s1, 0x88);
                __m128 ss1 = _mm_shuffle_ps(s0, s1, 0xDD);
                __m128 max = _mm_max_ps(ss0, ss1);
                __m128 exp0 = exp.Exponent(_mm_sub_ps(ss0, max));
                __m128 exp1 = exp.Exponent(_mm_sub_ps(ss1, max));
                __m128 sum = _mm_add_ps(exp0, exp1);
                __m128 d0 = _mm_div_ps(exp0, sum);
                __m128 d1 = _mm_div_ps(exp1, sum);
                _mm_storeu_ps(dst + 0, _mm_unpacklo_ps(d0, d1));
                _mm_storeu_ps(dst + F, _mm_unpackhi_ps(d0, d1));
                src += DF;
                dst += DF;
            }
            for (; o < outer; ++o)
            {
                float max = Simd::Max(src[0], src[1]);
                float exp0 = ::exp(src[0] - max);
                float exp1 = ::exp(src[1] - max);
                float sum = exp0 + exp1;
                dst[0] = exp0 / sum;
                dst[1] = exp1 / sum;
                src += 2;
                dst += 2;
            }
        }

        SIMD_INLINE void SynetSoftmaxLayerForward31(const Sse2::Exp& exp, __m128 buf[3])
        {
            __m128 max = _mm_max_ps(buf[0], _mm_max_ps(buf[1], buf[2]));
            buf[0] = exp.Exponent(_mm_sub_ps(buf[0], max));
            buf[1] = exp.Exponent(_mm_sub_ps(buf[1], max));
            buf[2] = exp.Exponent(_mm_sub_ps(buf[2], max));
            __m128 sum = _mm_add_ps(buf[0], _mm_add_ps(buf[1], buf[2]));
            buf[0] = _mm_div_ps(buf[0], sum);
            buf[1] = _mm_div_ps(buf[1], sum);
            buf[2] = _mm_div_ps(buf[2], sum);
        }

        void SynetSoftmaxLayerForward31(const float* src, size_t outer, float* dst)
        {
            Sse2::Exp exp;
            __m128 buf[3];
            size_t aligned = Simd::AlignLo(outer, F);
            for (size_t o = 0; o < aligned; o += F)
            {
                buf[0] = Sse::Gather<3>(src + 0);
                buf[1] = Sse::Gather<3>(src + 1);
                buf[2] = Sse::Gather<3>(src + 2);
                SynetSoftmaxLayerForward31(exp, buf);
                Sse::Scater<3>(dst + 0, buf[0]);
                Sse::Scater<3>(dst + 1, buf[1]);
                Sse::Scater<3>(dst + 2, buf[2]);
                src += 3*F;
                dst += 3*F;
            }
            if (aligned < outer)
            {
                size_t tail = outer - aligned;
                buf[0] = Sse::Gather<3>(src + 0, tail);
                buf[1] = Sse::Gather<3>(src + 1, tail);
                buf[2] = Sse::Gather<3>(src + 2, tail);
                SynetSoftmaxLayerForward31(exp, buf);
                Sse::Scater<3>(dst + 0, buf[0], tail);
                Sse::Scater<3>(dst + 1, buf[1], tail);
                Sse::Scater<3>(dst + 2, buf[2], tail);
            }
        }

        void SynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst)
        {
            if (count == 2 && inner == 1)
                SynetSoftmaxLayerForward21(src, outer, dst);
            else if(count == 3 && inner == 1)
                SynetSoftmaxLayerForward31(src, outer, dst);
            else
            {
                Sse2::Exp exp;
                size_t aligned = Simd::AlignLo(inner, F);
                Array32f tmp(inner * 2);
                const float * s;
                float * max = tmp.data, *sum = tmp.data + inner, *d;
                for (size_t o = 0; o < outer; ++o)
                {
                    memcpy(max, src, inner * sizeof(float));
                    s = src + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += F)
                            _mm_storeu_ps(max + i, _mm_max_ps(_mm_loadu_ps(s + i), _mm_loadu_ps(max + i)));
                        for (; i < inner; ++i)
                            max[i] = Simd::Max(max[i], s[i]);
                        s += inner;
                    }

                    s = src;
                    d = dst;
                    memset(sum, 0, inner * sizeof(float));
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += F)
                        {
                            __m128 _d = exp.Exponent(_mm_sub_ps(_mm_loadu_ps(s + i), _mm_loadu_ps(max + i)));
                            _mm_storeu_ps(d + i, _d);
                            _mm_storeu_ps(sum + i, _mm_add_ps(_d, _mm_loadu_ps(sum + i)));
                        }
                        for (; i < inner; ++i)
                        {
                            d[i] = ::exp(s[i] - max[i]);
                            sum[i] += d[i];
                        }
                        s += inner;
                        d += inner;
                    }

                    d = dst;
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += F)
                            _mm_storeu_ps(d + i, _mm_div_ps(_mm_loadu_ps(d + i), _mm_loadu_ps(sum + i)));
                        for (; i < inner; ++i)
                            d[i] /= sum[i];
                        d += inner;
                    }
                    src += count * inner;
                    dst += count * inner;
                }
            }
        }

        //---------------------------------------------------------------------

        template<SimdSynetUnaryOperation32fType type> __m128 SynetUnaryOperation32f(__m128 value);

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(__m128 value)
        {
            return _mm_andnot_ps(_mm_set1_ps(-0.0f), value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(__m128 value)
        {
            return Exponent(value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(__m128 value)
        {
            return Logarithm(value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(__m128 value)
        {
            return _mm_sub_ps(_mm_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(__m128 value)
        {
            return _mm_rsqrt_ps(value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(__m128 value)
        {
            return _mm_sqrt_ps(value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(__m128 value)
        {
            return Tanh(value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(__m128 value)
        {
            return _mm_setzero_ps();
        }

        template<SimdSynetUnaryOperation32fType type, bool align> void SynetUnaryOperation32fLayerForward(const float* src, size_t size, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Sse::Store<align>(dst + i + 0 * F, SynetUnaryOperation32f<type>(Sse::Load<align>(src + i + 0 * F)));
                Sse::Store<align>(dst + i + 1 * F, SynetUnaryOperation32f<type>(Sse::Load<align>(src + i + 1 * F)));
                Sse::Store<align>(dst + i + 2 * F, SynetUnaryOperation32f<type>(Sse::Load<align>(src + i + 2 * F)));
                Sse::Store<align>(dst + i + 3 * F, SynetUnaryOperation32f<type>(Sse::Load<align>(src + i + 3 * F)));
            }
            for (; i < sizeF; i += F)
                Sse::Store<align>(dst + i, SynetUnaryOperation32f<type>(Sse::Load<align>(src + i)));
            for (; i < size; ++i)
                dst[i] = Base::SynetUnaryOperation32f<type>(src[i]);
        }

        template<bool align> void SynetUnaryOperation32fLayerForward(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
        {
            switch (type)
            {
            case SimdSynetUnaryOperation32fAbs: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fAbs, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fExp: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fExp, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fLog: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fLog, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNeg: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fNeg, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRsqrt: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fRsqrt, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSqrt: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fSqrt, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fTanh: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fTanh, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fZero: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fZero, align>(src, size, dst); break;
            default:
                assert(0);
            }
        }

        void SynetUnaryOperation32fLayerForward(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetUnaryOperation32fLayerForward<true>(src, size, type, dst);
            else
                SynetUnaryOperation32fLayerForward<false>(src, size, type, dst);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
