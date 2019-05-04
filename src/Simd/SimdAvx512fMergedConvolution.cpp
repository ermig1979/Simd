/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdMergedConvolution.h"
#include "Simd/SimdConvolution.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512f.h"
#include "Simd/SimdGemm.h"

namespace Simd
{
#if defined(SIMD_AVX512F_ENABLE) && 0 
    namespace Avx512f
    {
        typedef Simd::GemmNNcb<float, __mmask16> NhwcGemm;

        static NhwcGemm CreateNhwcGemm(size_t M, size_t N, size_t K)
        {
            const size_t L1 = 2 * 32 * 1024;
            const size_t L2 = 1024 * 1024;
            const size_t L3 = 2 * 1280 * 1024;
            NhwcGemm::Main kernelMM, kernelMT;
            NhwcGemm::Tail kernelTM, kernelTT;
            size_t microM, microN;
#if SIMD_ZMM_COUNT == 32 
            if (M == 4 || M < 8)
            {
                microM = 4;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel4x48nn;
                kernelMT = tail > DF ? Avx512f::GemmKernel4x48nn : (tail > F ? Avx512f::GemmKernel4x32nn : Avx512f::GemmKernel4x16nn);
                kernelTM = Avx512f::GemmKernelMx48nn;
                kernelTT = tail > DF ? Avx512f::GemmKernelMx48nn : (tail > F ? Avx512f::GemmKernelMx32nn : Avx512f::GemmKernelMx16nn);
            }
            else if (M == 6)
            {
                microM = 6;
                microN = 32;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel6x32nn;
                kernelMT = tail > F ? Avx512f::GemmKernel6x32nn : Avx512f::GemmKernel6x16nn;
                kernelTM = Avx512f::GemmKernelMx32nn;
                kernelTT = tail > F ? Avx512f::GemmKernelMx32nn : Avx512f::GemmKernelMx16nn;
            }
            else if (M == 12 || M == 24)
            {
                microM = 12;
                microN = 32;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel12x32nn;
                kernelMT = tail > F ? Avx512f::GemmKernel12x32nn : Avx512f::GemmKernel12x16nn;
                kernelTM = Avx512f::GemmKernelMx32nn;
                kernelTT = tail > F ? Avx512f::GemmKernelMx32nn : Avx512f::GemmKernelMx16nn;
        }
            else if (M == 8 || M == 16 || M == 32 || M < 14)
            {
                microM = 8;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel8x48nn;
                kernelMT = tail > DF ? Avx512f::GemmKernel8x48nn : (tail > F ? Avx512f::GemmKernel8x32nn : Avx512f::GemmKernel8x16nn);
                kernelTM = GemmKernelMx48nn;
                kernelTT = tail > DF ? Avx512f::GemmKernelMx48nn : (tail > F ? Avx512f::GemmKernelMx32nn : Avx512f::GemmKernelMx16nn);
            }
            else if (N <= 16)
            {
                microM = 14;
                microN = 16;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel14x16nn;
                kernelMT = Avx512f::GemmKernel14x16nn;
                kernelTM = Avx512f::GetGemmTail(M, microN);
                kernelTT = Avx512f::GetGemmTail(M, tail);
            }
            else
            {
                microM = 14;
                microN = 32;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel14x32nn;
                kernelMT = tail > F ? Avx512f::GemmKernel14x32nn : Avx512f::GemmKernel14x16nn;
                kernelTM = Avx512f::GemmKernelMx32nn;
                kernelTT = tail > F ? Avx512f::GemmKernelMx32nn : Avx512f::GemmKernelMx16nn;
            }
#elif SIMD_ZMM_COUNT == 16 
            if (M == 4 || M == 8 || M == 16)
            {
                microM = 4;
                microN = 48;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel4x48nn;
                kernelMT = tail > DF ? Avx512f::GemmKernel4x48nn : (tail > F ? Avx512f::GemmKernel4x32nn : Avx512f::GemmKernel4x16nn);
                kernelTM = Avx512f::GemmKernelMx48nn;
                kernelTT = tail > DF ? Avx512f::GemmKernelMx48nn : (tail > F ? Avx512f::GemmKernelMx32nn : Avx512f::GemmKernelMx16nn);
            }
            else
            {
                microM = 6;
                microN = 32;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx512f::GemmKernel6x32nn;
                kernelMT = tail > F ? Avx512f::GemmKernel6x32nn : Avx512f::GemmKernel6x16nn;
                kernelTM = Avx512f::GemmKernelMx32nn;
                kernelTT = tail > F ? Avx512f::GemmKernelMx32nn : Avx512f::GemmKernelMx16nn;
            }
#else
            microM = 4;
            microN = 16;
            kernelMM = Avx512f::GemmKernel4x16nn;
            kernelMT = Avx512f::GemmKernel4x16nn;
            kernelTM = Avx512f::GemmKernelMx16nn;
            kernelTT = Avx512f::GemmKernelMx16nn;
#endif
            return NhwcGemm(M, N, K, microM, microN, L1, L2, L3, F, kernelMM, kernelMT, kernelTM, kernelTT, Avx512f::GemmPackB, Avx512f::GemmScaleC, Avx512f::TailMask16);
        }

        static void NhwcRun(size_t M, size_t N, size_t K, const float * A, const float * B, float * C)
        {
            NhwcGemm nhwcGemm = CreateNhwcGemm(M, N, K);
            nhwcGemm.Run(A, K, B, C, N);
        }

        static void NhwcRun(size_t m, size_t M, size_t N, size_t K, const float * A, const float * B, float * C)
        {
            SIMD_PERF_BEG(ToStr(M) + "-" + ToStr(N) + "-" + ToStr(K));

            NhwcGemm nhwcGemm = CreateNhwcGemm(M, N, K);
            nhwcGemm.Run(m, A, K, B, C, N);
        }

        static void NhwcReorderB(size_t M, size_t N, size_t K, const float * B, float * pB)
        {
            NhwcGemm nhwcGemm = CreateNhwcGemm(M, N, K);
            nhwcGemm.ReorderB(B, N, pB);
        }

        //---------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m512 Activate(__m512 value, const float * params, size_t offset, __mmask16 tail = -1);

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationIdentity>(__m512 value, const float * params, size_t offset, __mmask16 tail)
        {
            return value;
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationRelu>(__m512 value, const float * params, size_t offset, __mmask16 tail)
        {
            return _mm512_max_ps(_mm512_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationLeakyRelu>(__m512 value, const float * params, size_t offset, __mmask16 tail)
        {
            return _mm512_fmadd_ps(_mm512_set1_ps(params[0]), _mm512_min_ps(_mm512_setzero_ps(), value), _mm512_max_ps(_mm512_setzero_ps(), value));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationRestrictRange>(__m512 value, const float * params, size_t offset, __mmask16 tail)
        {
            return _mm512_min_ps(_mm512_max_ps(_mm512_set1_ps(params[0]), value), _mm512_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationPrelu>(__m512 value, const float * params, size_t offset, __mmask16 tail)
        {
            return _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, params + offset), _mm512_min_ps(_mm512_setzero_ps(), value), _mm512_max_ps(_mm512_setzero_ps(), value));
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Edge1(const float * src, const MergConvParam & p, 
            size_t dy, size_t dx, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 sum = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    if (sy < p.srcH)
                    {
                        for (size_t kx = 0; kx < 3; ++kx)
                        {
                            size_t sx = dx * p.strideX + kx - p.padX;
                            if (sx < p.srcW)
                            {
                                const float * pw = weight + (ky * 3 + kx) * srcC;
                                const float * ps = src + (sy*p.srcW + sx) * srcC;
                                sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
                            }
                        }
                    }
                }
                _mm512_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 sum = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    if (sy < p.srcH)
                    {
                        for (size_t kx = 0; kx < 3; ++kx)
                        {
                            size_t sx = dx * p.strideX + kx - p.padX;
                            if (sx < p.srcW)
                            {
                                const float * pw = weight + (ky * 3 + kx) * srcC;
                                const float * ps = src + (sy*p.srcW + sx) * srcC;
                                sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps), _mm512_maskz_loadu_ps(tail, pw), sum);
                            }
                        }
                    }
                }
                _mm512_mask_storeu_ps(dst + c, tail, Activate<type>(sum, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Main1(const float * src, 
            size_t srcS, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 sum = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * srcC), _mm512_loadu_ps(pw + 0 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * srcC), _mm512_loadu_ps(pw + 1 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * srcC), _mm512_loadu_ps(pw + 2 * srcC), sum);
                }
                _mm512_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 sum = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps + 0 * srcC), _mm512_maskz_loadu_ps(tail, pw + 0 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps + 1 * srcC), _mm512_maskz_loadu_ps(tail, pw + 1 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps + 2 * srcC), _mm512_maskz_loadu_ps(tail, pw + 2 * srcC), sum);
                }
                _mm512_mask_storeu_ps(dst + c, tail, Activate<type>(sum, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Main2(const float * src, 
            size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            __m512 sum0, sum1, w0;
            for (; c < srcCF; c += F)
            {
                sum0 = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + 0 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + 0 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + 1 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + 1 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + 2 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + 2 * srcC), w0, sum1);
                    pw += srcC;
                }
                _mm512_storeu_ps(dst + c, Activate<type>(sum0, params, c));
                _mm512_storeu_ps(dst + c + srcC, Activate<type>(sum1, params, c));
                src += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                sum0 = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + 0 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + 0 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + 1 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + 1 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + 2 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + 2 * srcC), w0, sum1);
                    pw += srcC;
                }
                _mm512_mask_storeu_ps(dst + c, tail, Activate<type>(sum0, params, c, tail));
                _mm512_mask_storeu_ps(dst + c + srcC, tail, Activate<type>(sum1, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Main4(const float * src, 
            size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t srcCDF = AlignLo(srcC, DF);
            size_t c = 0;
            for (; c < srcCDF; c += DF)
            {
                __m512 sum00, sum01, sum10, sum11, sum20, sum21, sum30, sum31, w0, w1;
                sum00 = bias ? _mm512_loadu_ps(bias + c + 0) : _mm512_setzero_ps();
                sum01 = bias ? _mm512_loadu_ps(bias + c + F) : _mm512_setzero_ps();
                sum10 = sum00;
                sum11 = sum01;
                sum20 = sum00;
                sum21 = sum01;
                sum30 = sum00;
                sum31 = sum01;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcX;
                const float * ps1 = src + 1 * srcX;
                const float * ps2 = src + 2 * srcX;
                const float * ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_loadu_ps(pw + 0);
                    w1 = _mm512_loadu_ps(pw + F);
                    sum00 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset + 0), w0, sum00);
                    sum01 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset + F), w1, sum01);
                    sum10 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset + 0), w0, sum10);
                    sum11 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset + F), w1, sum11);
                    sum20 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset + 0), w0, sum20);
                    sum21 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset + F), w1, sum21);
                    sum30 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset + 0), w0, sum30);
                    sum31 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset + F), w1, sum31);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw + 0);
                    w1 = _mm512_loadu_ps(pw + F);
                    sum00 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset + 0), w0, sum00);
                    sum01 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset + F), w1, sum01);
                    sum10 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset + 0), w0, sum10);
                    sum11 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset + F), w1, sum11);
                    sum20 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset + 0), w0, sum20);
                    sum21 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset + F), w1, sum21);
                    sum30 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset + 0), w0, sum30);
                    sum31 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset + F), w1, sum31);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw + 0);
                    w1 = _mm512_loadu_ps(pw + F);
                    sum00 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset + 0), w0, sum00);
                    sum01 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset + F), w1, sum01);
                    sum10 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset + 0), w0, sum10);
                    sum11 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset + F), w1, sum11);
                    sum20 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset + 0), w0, sum20);
                    sum21 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset + F), w1, sum21);
                    sum30 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset + 0), w0, sum30);
                    sum31 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset + F), w1, sum31);
                    pw += srcC, offset += srcC;
                }
                _mm512_storeu_ps(dst + 0 * srcC + 0, Activate<type>(sum00, params, c + 0));
                _mm512_storeu_ps(dst + 0 * srcC + F, Activate<type>(sum01, params, c + F));
                _mm512_storeu_ps(dst + 1 * srcC + 0, Activate<type>(sum10, params, c + 0));
                _mm512_storeu_ps(dst + 1 * srcC + F, Activate<type>(sum11, params, c + F));
                _mm512_storeu_ps(dst + 2 * srcC + 0, Activate<type>(sum20, params, c + 0));
                _mm512_storeu_ps(dst + 2 * srcC + F, Activate<type>(sum21, params, c + F));
                _mm512_storeu_ps(dst + 3 * srcC + 0, Activate<type>(sum30, params, c + 0));
                _mm512_storeu_ps(dst + 3 * srcC + F, Activate<type>(sum31, params, c + F));
                src += DF;
                dst += DF;
            }
            for (; c < srcCF; c += F)
            {
                __m512 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                sum2 = sum0;
                sum3 = sum0;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcX;
                const float * ps1 = src + 1 * srcX;
                const float * ps2 = src + 2 * srcX;
                const float * ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                }
                _mm512_storeu_ps(dst + 0 * srcC, Activate<type>(sum0, params, c));
                _mm512_storeu_ps(dst + 1 * srcC, Activate<type>(sum1, params, c));
                _mm512_storeu_ps(dst + 2 * srcC, Activate<type>(sum2, params, c));
                _mm512_storeu_ps(dst + 3 * srcC, Activate<type>(sum3, params, c));
                src += F;
                dst += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                sum2 = sum0;
                sum3 = sum0;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcX;
                const float * ps1 = src + 1 * srcX;
                const float * ps2 = src + 2 * srcX;
                const float * ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                }
                _mm512_mask_storeu_ps(dst + 0 * srcC, tail, Activate<type>(sum0, params, c, tail));
                _mm512_mask_storeu_ps(dst + 1 * srcC, tail, Activate<type>(sum1, params, c, tail));
                _mm512_mask_storeu_ps(dst + 2 * srcC, tail, Activate<type>(sum2, params, c, tail));
                _mm512_mask_storeu_ps(dst + 3 * srcC, tail, Activate<type>(sum3, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3_1x1_Main4(const float * src,
            size_t srcS, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            __m512 sum0, sum1, sum2, sum3, w0, w1, w2, s0, s1, s2, s3, s4, s5;
            for (; c < srcCF; c += F)
            {
                sum0 = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                sum2 = sum0;
                sum3 = sum0;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcC;
                const float * ps1 = src + 1 * srcC;
                const float * ps2 = src + 2 * srcC;
                const float * ps3 = src + 3 * srcC;
                const float * ps4 = src + 4 * srcC;
                const float * ps5 = src + 5 * srcC;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_loadu_ps(pw); pw += srcC;
                    s0 = _mm512_loadu_ps(ps0 + offset);
                    sum0 = _mm512_fmadd_ps(s0, w0, sum0);
                    w1 = _mm512_loadu_ps(pw); pw += srcC;
                    s1 = _mm512_loadu_ps(ps1 + offset);
                    sum0 = _mm512_fmadd_ps(s1, w1, sum0);
                    sum1 = _mm512_fmadd_ps(s1, w0, sum1);
                    w2 = _mm512_loadu_ps(pw); pw += srcC;
                    s2 = _mm512_loadu_ps(ps2 + offset);
                    sum0 = _mm512_fmadd_ps(s2, w2, sum0);
                    sum1 = _mm512_fmadd_ps(s2, w1, sum1);
                    sum2 = _mm512_fmadd_ps(s2, w0, sum2);
                    s3 = _mm512_loadu_ps(ps3 + offset);
                    sum1 = _mm512_fmadd_ps(s3, w2, sum1);
                    sum2 = _mm512_fmadd_ps(s3, w1, sum2);
                    sum3 = _mm512_fmadd_ps(s3, w0, sum3);
                    s4 = _mm512_loadu_ps(ps4 + offset);
                    sum2 = _mm512_fmadd_ps(s4, w2, sum2);
                    sum3 = _mm512_fmadd_ps(s4, w1, sum3);
                    s5 = _mm512_loadu_ps(ps5 + offset);
                    sum3 = _mm512_fmadd_ps(s5, w2, sum3);
                }
                _mm512_storeu_ps(dst + 0 * srcC, Activate<type>(sum0, params, c));
                _mm512_storeu_ps(dst + 1 * srcC, Activate<type>(sum1, params, c));
                _mm512_storeu_ps(dst + 2 * srcC, Activate<type>(sum2, params, c));
                _mm512_storeu_ps(dst + 3 * srcC, Activate<type>(sum3, params, c));
                src += F;
                dst += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                sum0 = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                sum2 = sum0;
                sum3 = sum0;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcC;
                const float * ps1 = src + 1 * srcC;
                const float * ps2 = src + 2 * srcC;
                const float * ps3 = src + 3 * srcC;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w1 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w1, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w1, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w1, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w1, sum3);
                    pw += srcC, offset += srcC;
                    w2 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w2, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w2, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w2, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w2, sum3);
                    pw += srcC, offset += srcC;
                }
                _mm512_mask_storeu_ps(dst + 0 * srcC, tail, Activate<type>(sum0, params, c, tail));
                _mm512_mask_storeu_ps(dst + 1 * srcC, tail, Activate<type>(sum1, params, c, tail));
                _mm512_mask_storeu_ps(dst + 2 * srcC, tail, Activate<type>(sum2, params, c, tail));
                _mm512_mask_storeu_ps(dst + 3 * srcC, tail, Activate<type>(sum3, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> void DepthwiseConvolutionBiasActivation3x3(const float * src, const MergConvParam & p, 
            size_t yBeg, size_t yEnd, const float * weight, const float * bias, const float * params, float * dst)
        {
            SIMD_PERF_BEG(p.Info());

            size_t srcC = p.srcC;
            size_t srcS = p.srcC*p.srcW;
            size_t srcX = p.srcC*p.strideX;
            size_t dstH = p.dstH - p.padH;
            size_t dstW = p.dstW - p.padW;
            size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
            size_t dstW4 = AlignLo(dstW - p.padX, 4) + p.padX;
            size_t dy = yBeg;
            for (; dy < p.padY; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    DepthwiseConvolutionBiasActivation3x3Edge1<type>(src, p, dy, dx, weight, bias, params, dst), dst += srcC;
            for (; dy < dstH && dy < yEnd; ++dy)
            {
                size_t dx = 0;
                for (; dx < p.padX; ++dx)
                    DepthwiseConvolutionBiasActivation3x3Edge1<type>(src, p, dy, dx, weight, bias, params, dst), dst += srcC;
                size_t offset = ((dy * p.strideY - p.padY)*p.srcW + dx * p.strideX - p.padX)*srcC;
                if(srcX == srcC)
                    for (; dx < dstW4; dx += 4)
                        DepthwiseConvolutionBiasActivation3x3_1x1_Main4<type>(src + offset, srcS, srcC, weight, bias, params, dst), dst += 4 * srcC, offset += 4 * srcX;
                else
                    for (; dx < dstW4; dx += 4)
                        DepthwiseConvolutionBiasActivation3x3Main4<type>(src + offset, srcS, srcX, srcC, weight, bias, params, dst), dst += 4 * srcC, offset += 4 * srcX;
                for (; dx < dstW2; dx += 2)
                    DepthwiseConvolutionBiasActivation3x3Main2<type>(src + offset, srcS, srcX, srcC, weight, bias, params, dst), dst += 2 * srcC, offset += 2 * srcX;
                for (; dx < dstW; ++dx)
                    DepthwiseConvolutionBiasActivation3x3Main1<type>(src + offset, srcS, srcC, weight, bias, params, dst), dst += srcC, offset += srcX;
                for (; dx < p.dstW; ++dx)
                    DepthwiseConvolutionBiasActivation3x3Edge1<type>(src, p, dy, dx, weight, bias, params, dst), dst += srcC;
            }
            for (; dy < p.dstH && dy < yEnd; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    DepthwiseConvolutionBiasActivation3x3Edge1<type>(src, p, dy, dx, weight, bias, params, dst), dst += srcC;
        }

        MergedConvolution::MergedConvolution(const MergConvParam & p)
            : Avx2::MergedConvolution(p)
        {
            _merge = p.dstH*p.dstW <= 256 && p.batch > 1;
            SetSize(128 * 1024);
            switch (p.activation0)
            {
            case SimdConvolutionActivationIdentity: _depthwise = DepthwiseConvolutionBiasActivation3x3<SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: _depthwise = DepthwiseConvolutionBiasActivation3x3<SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: _depthwise = DepthwiseConvolutionBiasActivation3x3<SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: _depthwise = DepthwiseConvolutionBiasActivation3x3<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _depthwise = DepthwiseConvolutionBiasActivation3x3<SimdConvolutionActivationPrelu>; break;
            default: assert(0);
            }
            _gemm.Init(Avx512f::Gemm32fNN, "Avx512f", p.gemm, "Ext");
            NhwcGemm nhwcGemm = CreateNhwcGemm(_M, _N, _K);
            _nhwcWeight.Resize(nhwcGemm.BufferSize());
            _nhwcRun = Avx512f::NhwcRun;
            _nhwcReorderB = Avx512f::NhwcReorderB;
            _biasAndActivation = Avx512f::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        void * MergedConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW,
            SimdConvolutionActivationType activation0, SimdConvolutionActivationType activation1, SimdGemm32fNNPtr gemm)
        {
            MergConvParam param(batch, srcC, srcH, srcW, dstC, kernelY, kernelX, strideY, strideX, padY, padX, padH, padW, activation0, activation1, gemm);
            if (!param.Valid())
                return NULL;
            if(param.srcC > HF && param.dstC > HF)
                return new Avx512f::MergedConvolution(param);
            else
                return new Avx2::MergedConvolution(param);
        }
    }
 #endif//SIMD_AVX512f_ENABLE
}
