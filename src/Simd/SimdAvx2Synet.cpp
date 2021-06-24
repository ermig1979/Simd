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
#include "Simd/SimdSynet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdGather.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx2
    {
        template<int part> SIMD_INLINE __m256 Cvt8uTo32f(__m128i src)
        {
            return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(src, part * 8)));
        }

        template<bool nofma, int part> SIMD_INLINE __m256i SynetAdd8iNchw(__m128i a, __m128i b, __m256 scale[3], __m256 shift[3])
        {
            __m256 _a = Fmadd<nofma>(Cvt8uTo32f<part>(a), scale[0], shift[0]);
            __m256 _b = Fmadd<nofma>(Cvt8uTo32f<part>(b), scale[1], shift[1]);
            return _mm256_cvtps_epi32(Fmadd<nofma>(_mm256_add_ps(_a, _b), scale[2], shift[2]));
        }

        template <bool nofma> SIMD_INLINE void SynetAdd8iNchwDF(const uint8_t* a, const uint8_t* b, __m256 scale[3], __m256 shift[3], __m256i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = Sse2::Load<false>((__m128i*)(a + offset));
            __m128i _b = Sse2::Load<false>((__m128i*)(b + offset));
            __m256i c0 = SynetAdd8iNchw<nofma, 0>(_a, _b, scale, shift);
            __m256i c1 = SynetAdd8iNchw<nofma, 1>(_a, _b, scale, shift);
            Sse2::Store<false>((__m128i*)(c + offset), _mm256_extracti128_si256(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(c0, c1), K_ZERO), upper), 0));
        }

        template<bool nofma> SIMD_INLINE void SynetAdd8iNchwF(const uint8_t* a, const uint8_t* b, __m256 scale[3], __m256 shift[3], __m256i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = _mm_loadl_epi64((__m128i*)(a + offset));
            __m128i _b = _mm_loadl_epi64((__m128i*)(b + offset));
            __m256i c0 = SynetAdd8iNchw<nofma, 0>(_a, _b, scale, shift);
            *(int64_t*)(c + offset) = Extract64i<0>(_mm256_min_epu8(_mm256_packus_epi16(PackI32ToI16(c0, K_ZERO), K_ZERO), upper));
        }

        template <bool nofma> void SynetAdd8iNchw(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            assert(spatial >= F);

            size_t spatialDF = AlignLo(spatial, DF);
            size_t spatialF = AlignLo(spatial, F);
            __m256i _upper = _mm256_set1_epi8(upper);
            __m256 scale[3], shift[3];
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    scale[0] = _mm256_set1_ps(aScale[c]);
                    shift[0] = _mm256_set1_ps(aShift[c]);
                    scale[1] = _mm256_set1_ps(bScale[c]);
                    shift[1] = _mm256_set1_ps(bShift[c]);
                    scale[2] = _mm256_set1_ps(cScale[c]);
                    shift[2] = _mm256_set1_ps(cShift[c]);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        SynetAdd8iNchwDF<nofma>(aData, bData, scale, shift, _upper, cData, s);
                    for (; s < spatialF; s += F)
                        SynetAdd8iNchwF<nofma>(aData, bData, scale, shift, _upper, cData, s);
                    if (s < spatial)
                        SynetAdd8iNchwF<nofma>(aData, bData, scale, shift, _upper, cData, spatial - F);
                    aData += spatial;
                    bData += spatial;
                    cData += spatial;
                }
            }
        }

        template<int part, bool align, bool nofma> SIMD_INLINE __m256i SynetAdd8iNhwc(__m128i a, const float* aScale, const float* aShift,
            __m128i b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, size_t offset)
        {
            __m256 _a = Fmadd<nofma>(Cvt8uTo32f<part>(a), Load<align>(aScale + offset), Load<align>(aShift + offset));
            __m256 _b = Fmadd<nofma>(Cvt8uTo32f<part>(b), Load<align>(bScale + offset), Load<align>(bShift + offset));
            return _mm256_cvtps_epi32(Fmadd<nofma>(_mm256_add_ps(_a, _b), Load<align>(cScale + offset), Load<align>(cShift + offset)));
        }

        template <bool align, bool nofma> SIMD_INLINE void SynetAdd8iNhwcDF(const uint8_t* a, const float* aScale, const float* aShift,
            const uint8_t* b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, __m256i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = Sse2::Load<false>((__m128i*)(a + offset));
            __m128i _b = Sse2::Load<false>((__m128i*)(b + offset));
            __m256i c0 = SynetAdd8iNhwc<0, align, nofma>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 0 * F);
            __m256i c1 = SynetAdd8iNhwc<1, align, nofma>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 1 * F);
            Sse2::Store<false>((__m128i*)(c + offset), _mm256_extracti128_si256(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(c0, c1), K_ZERO), upper), 0));
        }

        template <bool align, bool nofma> SIMD_INLINE void SynetAdd8iNhwcF(const uint8_t* a, const float* aScale, const float* aShift,
            const uint8_t* b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, __m256i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = _mm_loadl_epi64((__m128i*)(a + offset));
            __m128i _b = _mm_loadl_epi64((__m128i*)(b + offset));
            __m256i c0 = SynetAdd8iNhwc<0, align, nofma>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 0 * F);
            *(int64_t*)(c + offset) = Extract64i<0>(_mm256_min_epu8(_mm256_packus_epi16(PackI32ToI16(c0, K_ZERO), K_ZERO), upper));
        }

        template <bool align, bool nofma> void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsDF = AlignLo(channels, DF);
            __m256i _upper = _mm256_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsDF; c += DF)
                        SynetAdd8iNhwcDF<align, nofma>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, c);
                    for (; c < channelsF; c += F)
                        SynetAdd8iNhwcF<align, nofma>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, c);
                    if (c < channels)
                        SynetAdd8iNhwcF<false, nofma>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, channels - F);
                    aData += channels;
                    bData += channels;
                    cData += channels;
                }
            }
        }

        template <bool nofma> SIMD_INLINE void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            if (Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift))
                SynetAdd8iNhwc<true, nofma>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else
                SynetAdd8iNhwc<false, nofma>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
        }

        void SynetAdd8i(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
            bool nofma = Base::FmaAvoid(compatibility);
            if (format == SimdTensorFormatNchw && spatial >= F)
            {
                if(nofma)
                    SynetAdd8iNchw<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
                else
                    SynetAdd8iNchw<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            }
            else if (format == SimdTensorFormatNhwc && channels >= F)
            {
                if (nofma)
                    SynetAdd8iNhwc<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
                else
                    SynetAdd8iNhwc<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            }
            else
                Sse41::SynetAdd8i(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, format, compatibility);
        }

        //---------------------------------------------------------------------

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
            //SIMD_PERF_FUNCF(count * size * 2);
            float _bias[4] = { 0, 0, 0, 0 };
            size_t count4 = AlignLo(count, 4);
            size_t i = 0;
            for (; i < count4; i += 4)
                SynetInnerProductLayerForward4(src, weight + i * size, (bias ? bias + i : _bias), size, dst + i);
            for (; i < count; ++i)
                SynetInnerProductLayerForward1(src, weight + i * size, (bias ? bias + i : _bias), size, dst + i);
        }

        //---------------------------------------------------------------------

        SIMD_INLINE __m256i LoadTail(const void* ptr, size_t tail)
        {
            int8_t buf[A];
            _mm256_storeu_si256((__m256i*)buf, _mm256_setzero_si256());
            for (size_t i = 0; i < tail; ++i)
                buf[i] = ((int8_t*)ptr)[i];
            return _mm256_loadu_si256((__m256i*)buf);
        }

        static SIMD_INLINE void Save4Sums(const __m256i& sum0, const __m256i sum1, const __m256i& sum2, const __m256i& sum3, int32_t* dst)
        {
            __m256i sum = _mm256_hadd_epi32(_mm256_hadd_epi32(sum0, sum1), _mm256_hadd_epi32(sum2, sum3));
            _mm_storeu_si128((__m128i*)dst, _mm_add_epi32(_mm256_extractf128_si256(sum, 0), _mm256_extractf128_si256(sum, 1)));
        }

        template<bool overflow> static void SynetInnerProduct8i1x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            __m256i d00 = _mm256_setzero_si256();
            __m256i s0, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm256_loadu_si256((__m256i*)(S0 + k));
                w0 = _mm256_loadu_si256((__m256i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
            }
            if (KA < K)
            {
                size_t tail = K - KA;
                s0 = LoadTail(S0 + KA, tail);
                w0 = LoadTail(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
            }
            D[0] = ExtractSum<uint32_t>(d00);
        }

        template<bool overflow> static void SynetInnerProduct8i1x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            __m256i d00 = _mm256_setzero_si256();
            __m256i d01 = _mm256_setzero_si256();
            __m256i d02 = _mm256_setzero_si256();
            __m256i d03 = _mm256_setzero_si256();
            __m256i s0, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm256_loadu_si256((__m256i*)(S0 + k));
                w0 = _mm256_loadu_si256((__m256i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                w0 = _mm256_loadu_si256((__m256i*)(W1 + k));
                Madd4<overflow>(d01, s0, w0);
                w0 = _mm256_loadu_si256((__m256i*)(W2 + k));
                Madd4<overflow>(d02, s0, w0);
                w0 = _mm256_loadu_si256((__m256i*)(W3 + k));
                Madd4<overflow>(d03, s0, w0);
            }
            if (KA < K)
            {
                size_t tail = K - KA;
                s0 = LoadTail(S0 + KA, tail);
                w0 = LoadTail(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                w0 = LoadTail(W1 + KA, tail);
                Madd4<overflow>(d01, s0, w0);
                w0 = LoadTail(W2 + KA, tail);
                Madd4<overflow>(d02, s0, w0);
                w0 = LoadTail(W3 + KA, tail);
                Madd4<overflow>(d03, s0, w0);
            }
            Save4Sums(d00, d01, d02, d03, D);
        }

        template<bool overflow> static void SynetInnerProduct8i2x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            __m256i d00 = _mm256_setzero_si256();
            __m256i d10 = _mm256_setzero_si256();
            __m256i s0, s1, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm256_loadu_si256((__m256i*)(S0 + k));
                s1 = _mm256_loadu_si256((__m256i*)(S1 + k));
                w0 = _mm256_loadu_si256((__m256i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
            }
            if (KA < K)
            {
                size_t tail = K - KA;
                s0 = LoadTail(S0 + KA, tail);
                s1 = LoadTail(S1 + KA, tail);
                w0 = LoadTail(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
            }
            D[0 * ldd] = ExtractSum<uint32_t>(d00);
            D[1 * ldd] = ExtractSum<uint32_t>(d10);
        }

        template<bool overflow> static void SynetInnerProduct8i2x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            __m256i d00 = _mm256_setzero_si256();
            __m256i d01 = _mm256_setzero_si256();
            __m256i d02 = _mm256_setzero_si256();
            __m256i d03 = _mm256_setzero_si256();
            __m256i d10 = _mm256_setzero_si256();
            __m256i d11 = _mm256_setzero_si256();
            __m256i d12 = _mm256_setzero_si256();
            __m256i d13 = _mm256_setzero_si256();
            __m256i s0, s1, w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = _mm256_loadu_si256((__m256i*)(S0 + k));
                s1 = _mm256_loadu_si256((__m256i*)(S1 + k));
                w0 = _mm256_loadu_si256((__m256i*)(W0 + k));
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                w0 = _mm256_loadu_si256((__m256i*)(W1 + k));
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                w0 = _mm256_loadu_si256((__m256i*)(W2 + k));
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                w0 = _mm256_loadu_si256((__m256i*)(W3 + k));
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
            }
            if (KA < K)
            {
                size_t tail = K - KA;
                s0 = LoadTail(S0 + KA, tail);
                s1 = LoadTail(S1 + KA, tail);
                w0 = LoadTail(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                w0 = LoadTail(W1 + KA, tail);
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                w0 = LoadTail(W2 + KA, tail);
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                w0 = LoadTail(W3 + KA, tail);
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
            }
            Save4Sums(d00, d01, d02, d03, D + 0 * ldd);
            Save4Sums(d10, d11, d12, d13, D + 1 * ldd);
        }

        template<bool overflow> void SynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst)
        {
            size_t M2 = AlignLoAny(M, 2);
            size_t N4 = AlignLoAny(N, 4);
            size_t i = 0;
            for (; i < M2; i += 2)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    SynetInnerProduct8i2x4<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                for (; j < N; j += 1)
                    SynetInnerProduct8i2x1<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                src += K * 2;
                dst += N * 2;
            }
            for (; i < M; i += 1)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    SynetInnerProduct8i1x4<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                for (; j < N; j += 1)
                    SynetInnerProduct8i1x1<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                src += K;
                dst += N;
            }
        }

        void SynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::Precise(compatibility))
                SynetInnerProduct8i<false>(M, N, K, src, weight, dst);
            else
                SynetInnerProduct8i<true>(M, N, K, src, weight, dst);
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

        void SynetSoftmaxLayerForward21(const float* src, size_t outer, float* dst)
        {
            Avx2::Exp exp;
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

        SIMD_INLINE void SynetSoftmaxLayerForward31(const Avx2::Exp& exp, __m256 buf[3])
        {
            __m256 max = _mm256_max_ps(buf[0], _mm256_max_ps(buf[1], buf[2]));
            buf[0] = exp.Exponent(_mm256_sub_ps(buf[0], max));
            buf[1] = exp.Exponent(_mm256_sub_ps(buf[1], max));
            buf[2] = exp.Exponent(_mm256_sub_ps(buf[2], max));
            __m256 sum = _mm256_add_ps(buf[0], _mm256_add_ps(buf[1], buf[2]));
            buf[0] = _mm256_div_ps(buf[0], sum);
            buf[1] = _mm256_div_ps(buf[1], sum);
            buf[2] = _mm256_div_ps(buf[2], sum);
        }

        void SynetSoftmaxLayerForward31(const float* src, size_t outer, float* dst)
        {
            Avx2::Exp exp;
            __m256 buf[3];
            size_t aligned = Simd::AlignLo(outer, F);
            for (size_t o = 0; o < aligned; o += F)
            {
                buf[0] = Avx2::Gather<3>(src + 0);
                buf[1] = Avx2::Gather<3>(src + 1);
                buf[2] = Avx2::Gather<3>(src + 2);
                SynetSoftmaxLayerForward31(exp, buf);
                Avx::Scater<3>(dst + 0, buf[0]);
                Avx::Scater<3>(dst + 1, buf[1]);
                Avx::Scater<3>(dst + 2, buf[2]);
                src += 3 * F;
                dst += 3 * F;
            }
            if (aligned < outer)
            {
                size_t tail = outer - aligned;
                buf[0] = Avx::Gather<3>(src + 0, tail);
                buf[1] = Avx::Gather<3>(src + 1, tail);
                buf[2] = Avx::Gather<3>(src + 2, tail);
                SynetSoftmaxLayerForward31(exp, buf);
                Avx::Scater<3>(dst + 0, buf[0], tail);
                Avx::Scater<3>(dst + 1, buf[1], tail);
                Avx::Scater<3>(dst + 2, buf[2], tail);
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
                Avx2::Exp exp;
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
