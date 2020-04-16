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
#include "Simd/SimdSynet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse1.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> void SynetEltwiseLayerForwardSum(const float * src0, const __m256 & weight0, const float * src1, const __m256 & weight1, float * dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, _mm256_fmadd_ps(Avx::Load<align>(src0 + offset), weight0, _mm256_mul_ps(Avx::Load<align>(src1 + offset), weight1)));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(const float * src, const __m256 & weight, float * dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, _mm256_fmadd_ps(Avx::Load<align>(src + offset), weight, Load<align>(dst + offset)));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float * src0 = src[0];
            const float * src1 = src[1];
            __m256 weight0 = _mm256_set1_ps(weight[0]);
            __m256 weight1 = _mm256_set1_ps(weight[1]);
            size_t j = 0;
            if (partial)
            {
                for (; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 0);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 1);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 2);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j);
            }
            for (; j < size; ++j)
                dst[j] = src0[j] * weight[0] + src1[j] * weight[1];
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                __m256 weighti = _mm256_set1_ps(weight[i]);
                size_t j = 0;
                if (partial)
                {
                    for (; j < aligned; j += QF)
                    {
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 0);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 1);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 2);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j);
                }
                for (; j < size; ++j)
                    dst[j] += srci[j] * weight[i];
            }
        }

        void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            if (type != SimdSynetEltwiseOperationSum)
            {
                Avx::SynetEltwiseLayerForward(src, weight, count, size, type, dst);
                return;
            }
            assert(count >= 2);
            bool aligned = Aligned(dst) && Aligned(src[0]) && Aligned(src[1]);
            for (size_t i = 2; i < count; ++i)
                aligned = aligned && Aligned(src[i]);
            if (aligned)
                SynetEltwiseLayerForwardSum<true>(src, weight, count, size, dst);
            else
                SynetEltwiseLayerForwardSum<false>(src, weight, count, size, dst);
        }

        SIMD_INLINE __m256 Tail(size_t tail)
        {
            const int32_t mask[DF] = { 0, 0, 0, 0, 0, 0, 0, 0 , -1, -1, -1, -1, -1, -1, -1, -1 };
            return _mm256_loadu_ps((float*)(mask + tail));
        }

        void SynetInnerProductLayerForward1(const float * S0, const float * W, const float * B, size_t K, float * D)
        {
            size_t K8 = K & (~7);
            size_t K32 = K & (~31);
            const float * W0 = W + 0 * K;
            __m256 d00, d01, d02, d03;
            __m256 s0, s1, s2, s3, w0, w1, w2, w3;
            size_t k = 0;
            d00 = _mm256_setzero_ps();
            if (K32)
            {
                d01 = _mm256_setzero_ps();
                d02 = _mm256_setzero_ps();
                d03 = _mm256_setzero_ps();
                for (; k < K32; k += 32)
                {
                    s0 = _mm256_loadu_ps(S0 + k + 0 * F);
                    s1 = _mm256_loadu_ps(S0 + k + 1 * F);
                    w0 = _mm256_loadu_ps(W0 + k + 0 * F);
                    w1 = _mm256_loadu_ps(W0 + k + 1 * F);
                    d00 = _mm256_fmadd_ps(s0, w0, d00);
                    d01 = _mm256_fmadd_ps(s1, w1, d01);
                    s2 = _mm256_loadu_ps(S0 + k + 2 * F);
                    s3 = _mm256_loadu_ps(S0 + k + 3 * F);
                    w2 = _mm256_loadu_ps(W0 + k + 2 * F);
                    w3 = _mm256_loadu_ps(W0 + k + 3 * F);
                    d02 = _mm256_fmadd_ps(s2, w2, d02);
                    d03 = _mm256_fmadd_ps(s3, w3, d03);
                }
                d00 = _mm256_add_ps(_mm256_add_ps(d00, d01), _mm256_add_ps(d02, d03));
            }
            for (; k < K8; k += 8)
            {
                s0 = _mm256_loadu_ps(S0 + k);
                w0 = _mm256_loadu_ps(W0 + k);
                d00 = _mm256_fmadd_ps(s0, w0, d00);
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                s0 = _mm256_and_ps(tail, _mm256_loadu_ps(S0 + k));
                w0 = _mm256_loadu_ps(W0 + k);
                d00 = _mm256_fmadd_ps(s0, w0, d00);
            }
            D[0] = Avx::ExtractSum(d00) + B[0];
        }

        void SynetInnerProductLayerForward4(const float * S0, const float * W, const float * B, size_t K, float * D)
        {
            size_t K8 = K & (~7);
            size_t K16 = K & (~15);
            const float * W0 = W + 0 * K;
            const float * W1 = W + 1 * K;
            const float * W2 = W + 2 * K;
            const float * W3 = W + 3 * K;
            __m256 d00, d01, d10, d11, d20, d21, d30, d31;
            __m256 s0, s1, w0, w1;
            size_t k = 0;
            d00 = _mm256_setzero_ps();
            d10 = _mm256_setzero_ps();
            d20 = _mm256_setzero_ps();
            d30 = _mm256_setzero_ps();
            if (K16)
            {
                d01 = _mm256_setzero_ps();
                d11 = _mm256_setzero_ps();
                d21 = _mm256_setzero_ps();
                d31 = _mm256_setzero_ps();
                for (; k < K16; k += 16)
                {
                    s0 = _mm256_loadu_ps(S0 + k + 0 * F);
                    s1 = _mm256_loadu_ps(S0 + k + 1 * F);
                    w0 = _mm256_loadu_ps(W0 + k + 0 * F);
                    w1 = _mm256_loadu_ps(W0 + k + 1 * F);
                    d00 = _mm256_fmadd_ps(s0, w0, d00);
                    d01 = _mm256_fmadd_ps(s1, w1, d01);
                    w0 = _mm256_loadu_ps(W1 + k + 0 * F);
                    w1 = _mm256_loadu_ps(W1 + k + 1 * F);
                    d10 = _mm256_fmadd_ps(s0, w0, d10);
                    d11 = _mm256_fmadd_ps(s1, w1, d11);
                    w0 = _mm256_loadu_ps(W2 + k + 0 * F);
                    w1 = _mm256_loadu_ps(W2 + k + 1 * F);
                    d20 = _mm256_fmadd_ps(s0, w0, d20);
                    d21 = _mm256_fmadd_ps(s1, w1, d21);
                    w0 = _mm256_loadu_ps(W3 + k + 0 * F);
                    w1 = _mm256_loadu_ps(W3 + k + 1 * F);
                    d30 = _mm256_fmadd_ps(s0, w0, d30);
                    d31 = _mm256_fmadd_ps(s1, w1, d31);
                }
                d00 = _mm256_add_ps(d00, d01);
                d10 = _mm256_add_ps(d10, d11);
                d20 = _mm256_add_ps(d20, d21);
                d30 = _mm256_add_ps(d30, d31);
            }
            for (; k < K8; k += 8)
            {
                s0 = _mm256_loadu_ps(S0 + k + 0 * F);
                w0 = _mm256_loadu_ps(W0 + k + 0 * F);
                d00 = _mm256_fmadd_ps(s0, w0, d00);
                w0 = _mm256_loadu_ps(W1 + k + 0 * F);
                d10 = _mm256_fmadd_ps(s0, w0, d10);
                w0 = _mm256_loadu_ps(W2 + k + 0 * F);
                d20 = _mm256_fmadd_ps(s0, w0, d20);
                w0 = _mm256_loadu_ps(W3 + k + 0 * F);
                d30 = _mm256_fmadd_ps(s0, w0, d30);
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                s0 = _mm256_and_ps(tail, _mm256_loadu_ps(S0 + k));
                w0 = _mm256_loadu_ps(W0 + k + 0 * F);
                d00 = _mm256_fmadd_ps(s0, w0, d00);
                w0 = _mm256_loadu_ps(W1 + k + 0 * F);
                d10 = _mm256_fmadd_ps(s0, w0, d10);
                w0 = _mm256_loadu_ps(W2 + k + 0 * F);
                d20 = _mm256_fmadd_ps(s0, w0, d20);
                w0 = _mm256_loadu_ps(W3 + k + 0 * F);
                d30 = _mm256_fmadd_ps(s0, w0, d30);
            }
            _mm_storeu_ps(D, _mm_add_ps(Extract4Sums(d00, d10, d20, d30), _mm_loadu_ps(B)));
        }

        void SynetInnerProductLayerForward(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst)
        {
            float _bias[4] = { 0, 0, 0, 0 };
            size_t count4 = AlignLo(count, 4);
            size_t i = 0;
            for (; i < count4; i += 4)
                SynetInnerProductLayerForward4(src, weight + i * size, (bias ? bias + i : _bias), size, dst + i);
            for (; i < count; ++i)
                SynetInnerProductLayerForward1(src, weight + i * size, (bias ? bias + i : _bias), size, dst + i);
        }

        //---------------------------------------------------------------------

        template<int shift> SIMD_INLINE __m256 LoadAtEdge(const float * src)
        {
            static const int32_t mask[3 * F] = { 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };
            return _mm256_and_ps(_mm256_loadu_ps(src + shift), _mm256_loadu_ps((float*)mask + F + shift));
        }

        SIMD_INLINE __m256 NoseSquareSum(const float * src)
        {
            return _mm256_add_ps(_mm256_add_ps(Avx::Square(LoadAtEdge<-2>(src)), Avx::Square(LoadAtEdge<-1>(src))),
                _mm256_add_ps(Avx::Square(_mm256_loadu_ps(src)), _mm256_add_ps(Avx::Square(_mm256_loadu_ps(src + 1)), Avx::Square(_mm256_loadu_ps(src + 2)))));
        }

        SIMD_INLINE __m256 BodySquareSum(const float * src)
        {
            return _mm256_add_ps(_mm256_add_ps(Avx::Square(_mm256_loadu_ps(src - 2)), Avx::Square(_mm256_loadu_ps(src - 1))),
                _mm256_add_ps(Avx::Square(_mm256_loadu_ps(src)), _mm256_add_ps(Avx::Square(_mm256_loadu_ps(src + 1)), Avx::Square(_mm256_loadu_ps(src + 2)))));
        }

        SIMD_INLINE __m256 TailSquareSum(const float * src)
        {
            return _mm256_add_ps(_mm256_add_ps(Avx::Square(LoadAtEdge<2>(src)), Avx::Square(LoadAtEdge<1>(src))),
                _mm256_add_ps(Avx::Square(_mm256_loadu_ps(src)), _mm256_add_ps(Avx::Square(_mm256_loadu_ps(src - 1)), Avx::Square(_mm256_loadu_ps(src - 2)))));
        }

        template<bool align> void SynetLrnLayerCrossChannelsNchw(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            __m256 k0 = _mm256_set1_ps(k[0]);
            __m256 k1 = _mm256_set1_ps(k[1]);
            __m256 k2 = _mm256_set1_ps(k[2]);
            Avx2::Pow pow;
            Array32f sum(spatial, true), zero(spatial, true);
            size_t aligned = AlignLo(spatial, F);
            for (size_t c = 0; c < half; ++c)
            {
                const float * pos = src + c * spatial;
                size_t s = 0;
                for (; s < aligned; s += F)
                {
                    __m256 _pos = Avx::Load<align>(pos + s);
                    Avx::Store<true>(sum.data + s, _mm256_add_ps(Avx::Load<true>(sum.data + s), _mm256_mul_ps(_pos, _pos)));
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
                    __m256 _pos = Avx::Load<align>(pos + s);
                    __m256 _neg = Avx::Load<align>(neg + s);
                    __m256 _sum = Avx::Load<true>(sum.data + s);
                    _sum = _mm256_add_ps(_sum, _mm256_sub_ps(_mm256_mul_ps(_pos, _pos), _mm256_mul_ps(_neg, _neg)));
                    __m256 _src = Avx::Load<align>(src + s);
                    Avx::Store<true>(sum.data + s, _sum);
                    Avx::Store<align>(dst + s, _mm256_mul_ps(_src, pow(_mm256_add_ps(k0, _mm256_mul_ps(k1, _sum)), k2)));
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
            __m256 k0 = _mm256_set1_ps(k[0]);
            __m256 k1 = _mm256_set1_ps(k[1]);
            __m256 k2 = _mm256_set1_ps(k[2]);
            Avx2::Pow pow;
            size_t aligned = AlignLo(channels - half, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                Avx::Store<align>(dst + 0, _mm256_mul_ps(Avx::Load<align>(src + 0), pow(_mm256_add_ps(k0, _mm256_mul_ps(k1, NoseSquareSum(src + 0))), k2)));
                for (size_t c = F; c < aligned; c += F)
                    Avx::Store<align>(dst + c, _mm256_mul_ps(Avx::Load<align>(src + c), pow(_mm256_add_ps(k0, _mm256_mul_ps(k1, BodySquareSum(src + c))), k2)));
                if (aligned != channels - half)
                {
                    size_t c = channels - half - F;
                    Avx::Store<false>(dst + c, _mm256_mul_ps(Avx::Load<false>(src + c), pow(_mm256_add_ps(k0, _mm256_mul_ps(k1, BodySquareSum(src + c))), k2)));
                }
                size_t c = channels - F;
                Avx::Store<false>(dst + c, _mm256_mul_ps(Avx::Load<false>(src + c), pow(_mm256_add_ps(k0, _mm256_mul_ps(k1, TailSquareSum(src + c))), k2)));
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
                Sse2::SynetLrnLayerCrossChannels(src, half, channels, spatial, k, dst, SimdTensorFormatNhwc);
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

        template <bool align, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, float * dst, size_t offset)
        {
            __m256 _src = Avx::Load<align>(src + offset);
            __m256 _scale = Avx::Load<align>(scale + offset);
            __m256 _bias = Avx::Load<align>(bias + offset);
            Avx::Store<align>(dst + offset, Fmadd<nofma>(_src, _scale, _bias));
        }

        template <bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, float* dst, size_t offset, __m256i tail)
        {
            __m256 _src = _mm256_maskload_ps(src + offset, tail);
            __m256 _scale = _mm256_maskload_ps(scale + offset, tail);
            __m256 _bias = _mm256_maskload_ps(bias + offset, tail);
            _mm256_maskstore_ps(dst + offset, tail, Fmadd<nofma>(_src, _scale, _bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, float* dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, _mm256_mul_ps(Avx::Load<align>(src + offset), Avx::Load<align>(scale + offset)));
        }

        template <bool align, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const __m256& scale, const __m256& bias, float* dst, size_t offset)
        {
            __m256 _src = Avx::Load<align>(src + offset);
            Avx::Store<align>(dst + offset, Fmadd<nofma>(_src, scale, bias));
        }

        template <bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, const __m256 & bias, float * dst, size_t offset, __m256i tail)
        {
            __m256 _src = _mm256_maskload_ps(src + offset, tail);
            _mm256_maskstore_ps(dst + offset, tail, Fmadd<nofma>(_src, scale, bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, float * dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, _mm256_mul_ps(Avx::Load<align>(src + offset), scale));
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNchw(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(width, F) && Aligned(dst));

            size_t widthQF = AlignLo(width, QF);
            size_t widthF = AlignLo(width, F);
            if (bias)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        size_t w = 0;
                        if (widthF)
                        {
                            __m256 _scale = _mm256_set1_ps(scale[c]);
                            __m256 _bias = _mm256_set1_ps(bias[c]);
                            for (; w < widthQF; w += QF)
                            {
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 0);
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 1);
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 2);
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 3);
                            }
                            for (; w < widthF; w += F)
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w);
                        }
                        for (; w < width; ++w)
                            dst[w] = src[w] * scale[c] + bias[c];
                        src += width;
                        dst += width;
                    }
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        size_t w = 0;
                        if (widthF)
                        {
                            __m256 _scale = _mm256_set1_ps(scale[c]);
                            for (; w < widthQF; w += QF)
                            {
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 0);
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 1);
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 2);
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 3);
                            }
                            for (; w < widthF; w += F)
                                SynetScaleLayerForward<align>(src, _scale, dst, w);
                        }
                        for (; w < width; ++w)
                            dst[w] = src[w] * scale[c];
                        src += width;
                        dst += width;
                    }
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNchw(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (!((compatibility & SimdSynetCompatibilityNoFma) && bias))
            {
                width = height * width;
                height = 1;
                if (Aligned(src) && Aligned(width, F) && Aligned(dst))
                    SynetScaleLayerForwardNchw<true, false>(src, scale, bias, channels, height, width, dst);
                else
                    SynetScaleLayerForwardNchw<false, false>(src, scale, bias, channels, height, width, dst);
            }
            else
            {
                if (Aligned(src) && Aligned(width, F) && Aligned(dst))
                    SynetScaleLayerForwardNchw<true, true>(src, scale, bias, channels, height, width, dst);
                else
                    SynetScaleLayerForwardNchw<false, true>(src, scale, bias, channels, height, width, dst);
            }
        }

        template <bool align, bool nofma, bool notail> void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsQF = AlignLo(channels, QF);
            if (bias)
            {
                size_t widthF = AlignLo(width, F);
                __m256i tail = LeftNotZero32i(channels - channelsF);
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<nofma>(src, scale, bias, dst, c, tail);
                        src += channels;
                        dst += channels;
                    }
                    for (; w < width; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<notail>(src, scale, bias, dst, c, tail);
                        src += channels;
                        dst += channels;
                    }
                }
            }
            else
            {
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 0);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 1);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 2);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align>(src, scale, dst, c);
                        for (; c < channels; ++c)
                            dst[c] = src[c] * scale[c];
                        src += channels;
                        dst += channels;
                    }
                }
            }
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForwardNhwc(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if ((compatibility & SimdSynetCompatibilityNoFma) && bias)
                SynetScaleLayerForwardNhwc<align, true, true>(src, scale, bias, channels, height, width, dst);
            else if((compatibility & SimdSynetCompatibilityNoFmaTail) && bias)
                SynetScaleLayerForwardNhwc<align, false, true>(src, scale, bias, channels, height, width, dst);
            else
                SynetScaleLayerForwardNhwc<align, false, false>(src, scale, bias, channels, height, width, dst);
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNhwc3(const float * src, const float * scale, const float * bias, size_t height, size_t width, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(width));

            size_t width3 = width * 3;
            size_t widthF3 = AlignLo(width, F) * 3;
            if (bias)
            {
                float _scale[F * 3], _bias[F * 3];
                for (size_t i = 0; i < F; ++i)
                    for (size_t c = 0; c < 3; ++c)
                        _scale[i * 3 + c] = scale[c], _bias[i * 3 + c] = bias[c];
                __m256 _scale0 = Load<false>(_scale + 0 * F);
                __m256 _scale1 = Load<false>(_scale + 1 * F);
                __m256 _scale2 = Load<false>(_scale + 2 * F);
                __m256 _bias0 = Load<false>(_bias + 0 * F);
                __m256 _bias1 = Load<false>(_bias + 1 * F);
                __m256 _bias2 = Load<false>(_bias + 2 * F);                
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF3; w += F * 3)
                    {
                        SynetScaleLayerForward<align, nofma>(src, _scale0, _bias0, dst, w + F * 0);
                        SynetScaleLayerForward<align, nofma>(src, _scale1, _bias1, dst, w + F * 1);
                        SynetScaleLayerForward<align, nofma>(src, _scale2, _bias2, dst, w + F * 2);
                    }
                    for (; w < width3; w += 3)
                    {
                        dst[w + 0] = src[w + 0] * scale[0] + bias[0];
                        dst[w + 1] = src[w + 1] * scale[1] + bias[1];
                        dst[w + 2] = src[w + 2] * scale[2] + bias[2];
                    }
                    src += width3;
                    dst += width3;
                }
            }
            else
            {
                float _scale[F * 3];
                for (size_t i = 0; i < F; ++i)
                    for (size_t c = 0; c < 3; ++c)
                        _scale[i * 3 + c] = scale[c];
                __m256 _scale0 = Load<false>(_scale + 0 * F);
                __m256 _scale1 = Load<false>(_scale + 1 * F);
                __m256 _scale2 = Load<false>(_scale + 2 * F);                
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF3; w += F * 3)
                    {
                        SynetScaleLayerForward<align>(src, _scale0, dst, w + F * 0);
                        SynetScaleLayerForward<align>(src, _scale1, dst, w + F * 1);
                        SynetScaleLayerForward<align>(src, _scale2, dst, w + F * 2);
                    }
                    for (; w < width3; w += 3)
                    {
                        dst[w + 0] = src[w + 0] * scale[0];
                        dst[w + 1] = src[w + 1] * scale[1];
                        dst[w + 2] = src[w + 2] * scale[2];
                    }
                    src += width3;
                    dst += width3;
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst, SimdSynetCompatibilityType compatibility)
        {
            if (!((compatibility & SimdSynetCompatibilityNoFmaTail) && bias))
            {
                width = height * width;
                height = 1;
            }
            if (channels == 3)
            {
                if ((compatibility & SimdSynetCompatibilityNoFma) && bias)
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(width))
                        SynetScaleLayerForwardNhwc3<true, true>(src, scale, bias, height, width, dst);
                    else
                        SynetScaleLayerForwardNhwc3<false, true>(src, scale, bias, height, width, dst);
                }
                else
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(width))
                        SynetScaleLayerForwardNhwc3<true, false>(src, scale, bias, height, width, dst);
                    else
                        SynetScaleLayerForwardNhwc3<false, false>(src, scale, bias, height, width, dst);
                }
            }
            else
            {
                if (Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                    SynetScaleLayerForwardNhwc<true>(src, scale, bias, channels, height, width, dst, compatibility);
                else
                    SynetScaleLayerForwardNhwc<false>(src, scale, bias, channels, height, width, dst, compatibility);
            }
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNchw8c(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            if (bias)
            {
                for (size_t c = 0; c < channels; c += F)
                {
                    __m256 _scale = Load<false>(scale + c);
                    __m256 _bias = Load<false>(bias + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 0);
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 1);
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 2);
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s);
                    src += spatialF;
                    dst += spatialF;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; c += F)
                {
                    __m256 _scale = Load<false>(scale + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 0);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 1);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 2);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetScaleLayerForward<align>(src, _scale, dst, s);
                    src += spatialF;
                    dst += spatialF;
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNchw8c(const float* src, const float* scale, const float* bias, size_t channels, size_t spatial, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (compatibility & SimdSynetCompatibilityNoFma)
            {
                if (Aligned(src) && Aligned(dst))
                    SynetScaleLayerForwardNchw8c<true, true>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNchw8c<false, true>(src, scale, bias, channels, spatial, dst);
            }
            else
            {
                if (Aligned(src) && Aligned(dst))
                    SynetScaleLayerForwardNchw8c<true, false>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNchw8c<false, false>(src, scale, bias, channels, spatial, dst);
            }
        }

        void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetScaleLayerForwardNchw(src, scale, bias, channels, width, height, dst, compatibility);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetScaleLayerForwardNhwc(src, scale, bias, channels, height, width, dst, compatibility);
            else if (format == SimdTensorFormatNchw4c)
                Sse::SynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
            else if (format == SimdTensorFormatNchw8c)
                SynetScaleLayerForwardNchw8c(src, scale, bias, channels, spatial, dst, compatibility);
            else
                Base::SynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
        }

        //---------------------------------------------------------------------

        void SynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst)
        {
            Avx2::Exp exp;
            if (inner == 1 && count == 2)
            {
                size_t aligned = Simd::AlignLo(outer, F);
                size_t o = 0;
                for (; o < aligned; o += F)
                {
                    __m256 s0 = _mm256_loadu_ps(src + 0);
                    __m256 s1 = _mm256_loadu_ps(src + F);
                    __m256 ss0 = _mm256_shuffle_ps(s0, s1, 0x88);
                    __m256 ss1 = _mm256_shuffle_ps(s0, s1, 0xDD);
                    __m256 max = _mm256_max_ps(ss0, ss1);
                    __m256 exp0 = exp.Exponent(_mm256_sub_ps(ss0, max));
                    __m256 exp1 = exp.Exponent(_mm256_sub_ps(ss1, max));
                    __m256 sum = _mm256_add_ps(exp0, exp1);
                    __m256 d0 = _mm256_div_ps(exp0, sum);
                    __m256 d1 = _mm256_div_ps(exp1, sum);
                    _mm256_storeu_ps(dst + 0, _mm256_unpacklo_ps(d0, d1));
                    _mm256_storeu_ps(dst + F, _mm256_unpackhi_ps(d0, d1));
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
            else
            {
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
                            _mm256_storeu_ps(max + i, _mm256_max_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(max + i)));
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
                            __m256 _d = exp.Exponent(_mm256_sub_ps(_mm256_loadu_ps(s + i), _mm256_loadu_ps(max + i)));
                            _mm256_storeu_ps(d + i, _d);
                            _mm256_storeu_ps(sum + i, _mm256_add_ps(_d, _mm256_loadu_ps(sum + i)));
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
                            _mm256_storeu_ps(d + i, _mm256_div_ps(_mm256_loadu_ps(d + i), _mm256_loadu_ps(sum + i)));
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

        template<SimdSynetUnaryOperation32fType type> __m256 SynetUnaryOperation32f(__m256 value);

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(__m256 value)
        {
            return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(__m256 value)
        {
            return Exponent(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(__m256 value)
        {
            return Logarithm(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(__m256 value)
        {
            return _mm256_sub_ps(_mm256_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(__m256 value)
        {
            return _mm256_rsqrt_ps(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(__m256 value)
        {
            return _mm256_sqrt_ps(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(__m256 value)
        {
            return Tanh(value);
        }

        template<> SIMD_INLINE __m256 SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(__m256 value)
        {
            return _mm256_setzero_ps();
        }

        template<SimdSynetUnaryOperation32fType type, bool align> void SynetUnaryOperation32fLayerForward(const float* src, size_t size, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Avx::Store<align>(dst + i + 0 * F, SynetUnaryOperation32f<type>(Avx::Load<align>(src + i + 0 * F)));
                Avx::Store<align>(dst + i + 1 * F, SynetUnaryOperation32f<type>(Avx::Load<align>(src + i + 1 * F)));
                Avx::Store<align>(dst + i + 2 * F, SynetUnaryOperation32f<type>(Avx::Load<align>(src + i + 2 * F)));
                Avx::Store<align>(dst + i + 3 * F, SynetUnaryOperation32f<type>(Avx::Load<align>(src + i + 3 * F)));
            }
            for (; i < sizeF; i += F)
                Avx::Store<align>(dst + i, SynetUnaryOperation32f<type>(Avx::Load<align>(src + i)));
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
#endif// SIMD_AVX2_ENABLE
}
