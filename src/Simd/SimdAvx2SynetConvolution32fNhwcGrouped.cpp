/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdErf.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx2
    {
        template<SimdConvolutionActivationType type> SIMD_INLINE void SaveResult(__m256 sum0, __m256 sum1, const float* bias, const float* params, size_t offset, float* dst)
        {
            sum0 = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(sum0), 0xD8));
            sum1 = _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(sum1), 0xD8));
            _mm256_storeu_ps(dst + offset + 0, Activate<type>(_mm256_add_ps(_mm256_unpacklo_ps(sum0, sum1), _mm256_loadu_ps(bias + offset + 0)), params, offset + 0));
            _mm256_storeu_ps(dst + offset + F, Activate<type>(_mm256_add_ps(_mm256_unpackhi_ps(sum0, sum1), _mm256_loadu_ps(bias + offset + F)), params, offset + F));
        }

        template<SimdConvolutionActivationType type> void ConvolutionNhwcGroupedBlock1x2Default(const float* src, const ConvParam& p, const float* weight, const float* bias, const float* params, float* dst)
        {
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t srcC2F = AlignLo(srcC, 2 * F);
            size_t srcC4F = AlignLo(srcC, 4 * F);
            size_t dW = p.kernelY * p.kernelX * p.srcC;
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t c = 0;
                    for (; c < srcC4F; c += 4 * F)
                    {
                        const float* pwc = weight + c;
                        const float* psc = src + c;
                        __m256 sum00 = _mm256_setzero_ps();
                        __m256 sum01 = _mm256_setzero_ps();
                        __m256 sum02 = _mm256_setzero_ps();
                        __m256 sum03 = _mm256_setzero_ps();
                        __m256 sum10 = _mm256_setzero_ps();
                        __m256 sum11 = _mm256_setzero_ps();
                        __m256 sum12 = _mm256_setzero_ps();
                        __m256 sum13 = _mm256_setzero_ps();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                const float* pwy = pwc + ky * p.kernelX * srcC;
                                const float* psy = psc + sy * p.srcW * srcC;
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw0 = pwy + kx * srcC, * pw1 = pw0 + dW;
                                        const float* ps0 = psy + sx * srcC;
                                        __m256 s0 = _mm256_loadu_ps(ps0 + 0 * F);
                                        sum00 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(pw0 + 0 * F), sum00);
                                        sum10 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(pw1 + 0 * F), sum10);
                                        __m256 s1 = _mm256_loadu_ps(ps0 + 1 * F);
                                        sum01 = _mm256_fmadd_ps(s1, _mm256_loadu_ps(pw0 + 1 * F), sum01);
                                        sum11 = _mm256_fmadd_ps(s1, _mm256_loadu_ps(pw1 + 1 * F), sum11);
                                        __m256 s2 = _mm256_loadu_ps(ps0 + 2 * F);
                                        sum02 = _mm256_fmadd_ps(s2, _mm256_loadu_ps(pw0 + 2 * F), sum02);
                                        sum12 = _mm256_fmadd_ps(s2, _mm256_loadu_ps(pw1 + 2 * F), sum12);
                                        __m256 s3 = _mm256_loadu_ps(ps0 + 3 * F);
                                        sum03 = _mm256_fmadd_ps(s3, _mm256_loadu_ps(pw0 + 3 * F), sum03);
                                        sum13 = _mm256_fmadd_ps(s3, _mm256_loadu_ps(pw1 + 3 * F), sum13);
                                    }
                                }
                            }
                        }
                        size_t d = 2 * c;
                        SaveResult<type>(sum00, sum10, bias, params, d + 0 * F, dst);
                        SaveResult<type>(sum01, sum11, bias, params, d + 2 * F, dst);
                        SaveResult<type>(sum02, sum12, bias, params, d + 4 * F, dst);
                        SaveResult<type>(sum03, sum13, bias, params, d + 6 * F, dst);
                    }
                    for (; c < srcC2F; c += 2 * F)
                    {
                        const float* pwc = weight + c;
                        const float* psc = src + c;
                        __m256 sum00 = _mm256_setzero_ps();
                        __m256 sum01 = _mm256_setzero_ps();
                        __m256 sum10 = _mm256_setzero_ps();
                        __m256 sum11 = _mm256_setzero_ps();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                const float* pwy = pwc + ky * p.kernelX * srcC;
                                const float* psy = psc + sy * p.srcW * srcC;
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw0 = pwy + kx * srcC, *pw1 = pw0 + dW;
                                        const float* ps0 = psy + sx * srcC;
                                        __m256 s0 = _mm256_loadu_ps(ps0 + 0 * F);
                                        sum00 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(pw0 + 0 * F), sum00);
                                        sum10 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(pw1 + 0 * F), sum10);
                                        __m256 s1 = _mm256_loadu_ps(ps0 + 1 * F);
                                        sum01 = _mm256_fmadd_ps(s1, _mm256_loadu_ps(pw0 + 1 * F), sum01);
                                        sum11 = _mm256_fmadd_ps(s1, _mm256_loadu_ps(pw1 + 1 * F), sum11);
                                    }
                                }
                            }
                        }
                        size_t d = 2 * c;
                        SaveResult<type>(sum00, sum10, bias, params, d + 0 * F, dst);
                        SaveResult<type>(sum01, sum11, bias, params, d + 2 * F, dst);
                    }
                    for (; c < srcC; c += F)
                    {
                        c = c >= srcCF ? srcC - F : c;
                        const float* pwc = weight + c;
                        const float* psc = src + c;
                        __m256 sum00 = _mm256_setzero_ps();
                        __m256 sum10 = _mm256_setzero_ps();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                const float* pwy = pwc + ky * p.kernelX * srcC;
                                const float* psy = psc + sy * p.srcW * srcC;
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw0 = pwy + kx * srcC, *pw1 = pw0 + dW;
                                        const float* ps0 = psy + sx * srcC;
                                        __m256 s0 = _mm256_loadu_ps(ps0);
                                        sum00 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(pw0 + 0 * F), sum00);
                                        sum10 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(pw1 + 0 * F), sum10);
                                    }
                                }
                            }
                        }
                        SaveResult<type>(sum00, sum10, bias, params, c * 2, dst);
                    }
                    dst += p.dstC;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SynetConvolution32fNhwcGroupedBlock1x2::ConvolutionPtr GetConvolution(const ConvParam& p)
        {
            return ConvolutionNhwcGroupedBlock1x2Default<type>;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fNhwcGroupedBlock1x2::SynetConvolution32fNhwcGroupedBlock1x2(const ConvParam& p)
            : Sse41::SynetConvolution32fNhwcGroupedBlock1x2(p)
        {
            if (p.srcC >= F)
            {
                switch (p.activation)
                {
                case ::SimdConvolutionActivationIdentity: _convolution = GetConvolution<SimdConvolutionActivationIdentity>(p); break;
                case ::SimdConvolutionActivationRelu: _convolution = GetConvolution<SimdConvolutionActivationRelu>(p); break;
                case ::SimdConvolutionActivationLeakyRelu: _convolution = GetConvolution<SimdConvolutionActivationLeakyRelu>(p); break;
                case ::SimdConvolutionActivationRestrictRange: _convolution = GetConvolution<SimdConvolutionActivationRestrictRange>(p); break;
                case ::SimdConvolutionActivationPrelu: _convolution = GetConvolution<SimdConvolutionActivationPrelu>(p); break;
                case ::SimdConvolutionActivationElu: _convolution = GetConvolution<SimdConvolutionActivationElu>(p); break;
                case ::SimdConvolutionActivationHswish: _convolution = GetConvolution<SimdConvolutionActivationHswish>(p); break;
                case ::SimdConvolutionActivationMish: _convolution = GetConvolution<SimdConvolutionActivationMish>(p); break;
                case ::SimdConvolutionActivationHardSigmoid: _convolution = GetConvolution<SimdConvolutionActivationHardSigmoid>(p); break;
                case ::SimdConvolutionActivationSwish: _convolution = GetConvolution<SimdConvolutionActivationSwish>(p); break;
                case ::SimdConvolutionActivationGelu: _convolution = GetConvolution<SimdConvolutionActivationGelu>(p); break;
                }
            }
        }
    }
#endif
}
