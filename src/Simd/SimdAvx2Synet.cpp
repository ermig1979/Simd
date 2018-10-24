/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"

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

        template <bool align> SIMD_INLINE void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t count, size_t size, const float * k, float * dst)
        {
            size_t aligned = AlignLo(size, F);
            Array32f sum(size, true), zero(size, true);

            for (size_t i = 0; i < half; ++i)
            {
                const float * pos = src + i * size;
                size_t j = 0;
                for (; j < aligned; j += F)
                {
                    __m256 _pos = Avx::Load<align>(pos + j);
                    Avx::Store<true>(sum.data + j, _mm256_fmadd_ps(_pos, _pos, Avx::Load<true>(sum.data + j)));
                }
                for (; j < size; ++j)
                    sum[j] += Simd::Square(pos[j]);
            }

            __m256 k0 = _mm256_set1_ps(k[0]);
            __m256 k1 = _mm256_set1_ps(k[1]);
            __m256 k2 = _mm256_set1_ps(k[2]);
            Avx2::Pow pow;
            for (size_t i = 0; i < count; ++i)
            {
                const float * pos = (i < count - half) ? src + half * size : zero.data;
                const float * neg = (i > half) ? src - (half + 1) * size : zero.data;
                size_t j = 0;
                for (; j < aligned; j += F)
                {
                    __m256 _pos = Avx::Load<align>(pos + j);
                    __m256 _neg = Avx::Load<align>(neg + j);
                    __m256 _sum = Avx::Load<true>(sum.data + j);
                    _sum = _mm256_fmadd_ps(_pos, _pos, _mm256_fnmadd_ps(_neg, _neg, _sum));
                    __m256 _src = Avx::Load<align>(src + j);
                    Avx::Store<true>(sum.data + j, _sum);
                    Avx::Store<align>(dst + j, _mm256_mul_ps(_src, pow(_mm256_fmadd_ps(k1, _sum, k0), k2)));
                }
                for (; j < size; ++j)
                {
                    sum[j] += Simd::Square(pos[j]);
                    sum[j] -= Simd::Square(neg[j]);
                    dst[j] = src[j] * Base::Pow(k[0] + k[1] * sum[j], k[2]);
                }
                src += size;
                dst += size;
            }
        }

        void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t count, size_t size, const float * k, float * dst)
        {
            if (Aligned(src) && Aligned(dst) && Aligned(size))
                SynetLrnLayerCrossChannels<true>(src, half, count, size, k, dst);
            else
                SynetLrnLayerCrossChannels<false>(src, half, count, size, k, dst);
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, const __m256 & bias, float * dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, _mm256_fmadd_ps(Avx::Load<align>(src + offset), scale, bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, float * dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, _mm256_mul_ps(Avx::Load<align>(src + offset), scale));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (bias)
            {
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        __m256 _scale = _mm256_set1_ps(scale[i]);
                        __m256 _bias = _mm256_set1_ps(bias[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 0);
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 1);
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 2);
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 3);
                        }
                        for (; j < partial; j += F)
                            SynetScaleLayerForward<align>(src, _scale, _bias, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = src[j] * scale[i] + bias[i];
                    src += size;
                    dst += size;
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        __m256 _scale = _mm256_set1_ps(scale[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetScaleLayerForward<align>(src, _scale, dst, j + F * 0);
                            SynetScaleLayerForward<align>(src, _scale, dst, j + F * 1);
                            SynetScaleLayerForward<align>(src, _scale, dst, j + F * 2);
                            SynetScaleLayerForward<align>(src, _scale, dst, j + F * 3);
                        }
                        for (; j < partial; j += F)
                            SynetScaleLayerForward<align>(src, _scale, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = src[j] * scale[i];
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst)
        {
            if (Aligned(dst) && Aligned(size))
                SynetScaleLayerForward<true>(src, scale, bias, count, size, dst);
            else
                SynetScaleLayerForward<false>(src, scale, bias, count, size, dst);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
