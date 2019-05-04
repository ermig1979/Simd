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
#include "Simd/SimdAvx1.h"
#include "Simd/SimdGemm.h"

namespace Simd
{
#if defined(SIMD_AVX_ENABLE) && 0 
    namespace Avx
    {
        typedef Simd::GemmNNcb<float, size_t> NhwcGemm;

        static NhwcGemm CreateNhwcGemm(size_t M, size_t N, size_t K)
        {
            const size_t L1 = 32 * 1024;
            const size_t L2 = 256 * 1024;
            const size_t L3 = 2 * 1024 * 1024;
            NhwcGemm::Main kernelMM, kernelMT;
            NhwcGemm::Tail kernelTM, kernelTT;
            size_t microM, microN;
#ifdef SIMD_X64_ENABLE
            if (M == 4 || M == 8 || /*M == 12 || */M == 16)
            {
                microM = 4;
                microN = 24;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx::GemmKernel4x24nn;
                kernelMT = tail > DF ? Avx::GemmKernel4x24nn : (tail > F ? Avx::GemmKernel4x16nn : Avx::GemmKernel4x8nn);
                kernelTM = Avx::GemmKernelMx24nn;
                kernelTT = tail > DF ? Avx::GemmKernelMx24nn : (tail > F ? Avx::GemmKernelMx16nn : Avx::GemmKernelMx8nn);
            }
            else
            {
                microM = 6;
                microN = 16;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Avx::GemmKernel6x16nn;
                kernelMT = tail > F ? Avx::GemmKernel6x16nn : Avx::GemmKernel6x8nn;
                kernelTM = Avx::GemmKernelMx16nn;
                kernelTT = tail > F ? Avx::GemmKernelMx16nn : Avx::GemmKernelMx8nn;
            }
#else
            microM = 4;
            microN = 8;
            kernelMM = Avx::GemmKernel4x8nn;
            kernelMT = Avx::GemmKernel4x8nn;
            kernelTM = Avx::GemmKernelMx8nn;
            kernelTT = Avx::GemmKernelMx8nn;
#endif
            return NhwcGemm(M, N, K, microM, microN, L1, L2, L3, F, kernelMM, kernelMT, kernelTM, kernelTT, Avx::GemmPackB, Avx::GemmScaleC, NULL);
        }

        static void NhwcRun(size_t M, size_t N, size_t K, const float * A, const float * B, float * C)
        {
            NhwcGemm nhwcGemm = CreateNhwcGemm(M, N, K);
            nhwcGemm.Run(A, K, B, C, N);
        }

        static void NhwcRun(size_t m, size_t M, size_t N, size_t K, const float * A, const float * B, float * C)
        {
            NhwcGemm nhwcGemm = CreateNhwcGemm(M, N, K);
            nhwcGemm.Run(m, A, K, B, C, N);
        }

        static void NhwcReorderB(size_t M, size_t N, size_t K, const float * B, float * pB)
        {
            NhwcGemm nhwcGemm = CreateNhwcGemm(M, N, K);
            nhwcGemm.ReorderB(B, N, pB);
        }

        //---------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m256 Activate(__m256 value, const float * params, size_t offset);

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationIdentity>(__m256 value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRelu>(__m256 value, const float * params, size_t offset)
        {
            return _mm256_max_ps(_mm256_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationLeakyRelu>(__m256 value, const float * params, size_t offset)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(_mm256_set1_ps(params[0]), _mm256_min_ps(_mm256_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRestrictRange>(__m256 value, const float * params, size_t offset)
        {
            return _mm256_min_ps(_mm256_max_ps(_mm256_set1_ps(params[0]), value), _mm256_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationPrelu>(__m256 value, const float * params, size_t offset)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(_mm256_loadu_ps(params + offset), _mm256_min_ps(_mm256_setzero_ps(), value)));
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Edge1(const float * src, const MergConvParam & p, 
            size_t dy, size_t dx, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m256 sum = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
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
                                sum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps), _mm256_loadu_ps(pw)), sum);
                            }
                        }
                    }
                }
                _mm256_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                weight -= srcCF - c;
                __m256 sum = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
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
                                sum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps), _mm256_loadu_ps(pw)), sum);
                            }
                        }
                    }
                }
                _mm256_storeu_ps(dst + c, Activate<type>(sum, params, c));
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Main1(const float * src, 
            size_t srcS, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m256 sum = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps + 0 * srcC), _mm256_loadu_ps(pw + 0 * srcC)), sum);
                    sum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps + 1 * srcC), _mm256_loadu_ps(pw + 1 * srcC)), sum);
                    sum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps + 2 * srcC), _mm256_loadu_ps(pw + 2 * srcC)), sum);
                }
                _mm256_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                weight -= srcCF - c;
                __m256 sum = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps + 0 * srcC), _mm256_loadu_ps(pw + 0 * srcC)), sum);
                    sum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps + 1 * srcC), _mm256_loadu_ps(pw + 1 * srcC)), sum);
                    sum = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps + 2 * srcC), _mm256_loadu_ps(pw + 2 * srcC)), sum);
                }
                _mm256_storeu_ps(dst + c, Activate<type>(sum, params, c));
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Main2(const float * src, 
            size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            __m256 sum0, sum1, w0;
            for (; c < srcCF; c += F)
            {
                sum0 = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + 0 * srcC), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + 0 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + 1 * srcC), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + 1 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + 2 * srcC), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + 2 * srcC), w0), sum1);
                    pw += srcC;
                }
                _mm256_storeu_ps(dst + c, Activate<type>(sum0, params, c));
                _mm256_storeu_ps(dst + c + srcC, Activate<type>(sum1, params, c));
                src += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                sum0 = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + 0 * srcC), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + 0 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + 1 * srcC), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + 1 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + 2 * srcC), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + 2 * srcC), w0), sum1);
                    pw += srcC;
                }
                _mm256_storeu_ps(dst + c, Activate<type>(sum0, params, c));
                _mm256_storeu_ps(dst + c + srcC, Activate<type>(sum1, params, c));
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Main4(const float * src, 
            size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m256 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
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
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                }
                _mm256_storeu_ps(dst + 0 * srcC, Activate<type>(sum0, params, c));
                _mm256_storeu_ps(dst + 1 * srcC, Activate<type>(sum1, params, c));
                _mm256_storeu_ps(dst + 2 * srcC, Activate<type>(sum2, params, c));
                _mm256_storeu_ps(dst + 3 * srcC, Activate<type>(sum3, params, c));
                src += F;
                dst += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                dst -= srcCF - c;
                __m256 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm256_loadu_ps(bias + c) : _mm256_setzero_ps();
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
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm256_loadu_ps(pw);
                    sum0 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                }
                _mm256_storeu_ps(dst + 0 * srcC, Activate<type>(sum0, params, c));
                _mm256_storeu_ps(dst + 1 * srcC, Activate<type>(sum1, params, c));
                _mm256_storeu_ps(dst + 2 * srcC, Activate<type>(sum2, params, c));
                _mm256_storeu_ps(dst + 3 * srcC, Activate<type>(sum3, params, c));
            }
        }

        template<::SimdConvolutionActivationType type> void DepthwiseConvolutionBiasActivation3x3(const float * src, const MergConvParam & p, 
            size_t yBeg, size_t yEnd, const float * weight, const float * bias, const float * params, float * dst)
        {
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
                size_t offset = ((dy * p.strideY - p.padY)*p.srcW + dx * p.strideX - p.padX)*p.srcC;
                for (; dx < dstW4; dx += 4)
                    DepthwiseConvolutionBiasActivation3x3Main4<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 4 * srcC, offset += 4 * srcX;
                for (; dx < dstW2; dx += 2)
                    DepthwiseConvolutionBiasActivation3x3Main2<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 2 * srcC, offset += 2 * srcX;
                for (; dx < dstW; ++dx)
                    DepthwiseConvolutionBiasActivation3x3Main1<type>(src + offset, srcS, p.srcC, weight, bias, params, dst), dst += srcC, offset += srcX;
                for (; dx < p.dstW; ++dx)
                    DepthwiseConvolutionBiasActivation3x3Edge1<type>(src, p, dy, dx, weight, bias, params, dst), dst += srcC;
            }
            for (; dy < p.dstH && dy < yEnd; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    DepthwiseConvolutionBiasActivation3x3Edge1<type>(src, p, dy, dx, weight, bias, params, dst), dst += srcC;
        }

        MergedConvolution::MergedConvolution(const MergConvParam & p)
            : Sse::MergedConvolution(p)
        {
            _merge = p.dstH*p.dstW <= 256 && p.batch > 1;
            SetSize(256 * 1024);
            switch (p.activation0)
            {
            case SimdConvolutionActivationIdentity: _depthwise = DepthwiseConvolutionBiasActivation3x3<SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: _depthwise = DepthwiseConvolutionBiasActivation3x3<SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: _depthwise = DepthwiseConvolutionBiasActivation3x3<SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: _depthwise = DepthwiseConvolutionBiasActivation3x3<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _depthwise = DepthwiseConvolutionBiasActivation3x3<SimdConvolutionActivationPrelu>; break;
            default: assert(0);
            }
            _gemm.Init(Avx::Gemm32fNN, "Avx", p.gemm, "Ext");
            NhwcGemm nhwcGemm = CreateNhwcGemm(_M, _N, _K);
            _nhwcWeight.Resize(nhwcGemm.BufferSize());
            _nhwcRun = Avx::NhwcRun;
            _nhwcReorderB = Avx::NhwcReorderB;
            _biasAndActivation = Avx::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        void * MergedConvolutionInit(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW,
            SimdConvolutionActivationType activation0, SimdConvolutionActivationType activation1, SimdGemm32fNNPtr gemm)
        {
            MergConvParam param(batch, srcC, srcH, srcW, dstC, kernelY, kernelX, strideY, strideX, padY, padX, padH, padW, activation0, activation1, gemm);
            if (!param.Valid())
                return NULL;
            if(param.srcC >= F && param.dstC >= F)
                return new Avx::MergedConvolution(param);
            else
                return new Sse::MergedConvolution(param);
        }
    }
 #endif//SIMD_AVX_ENABLE
}
