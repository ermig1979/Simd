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
#include "Simd/SimdUpdate.h"

namespace Simd
{
#if defined(SIMD_SSE_ENABLE)
    namespace Sse
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE float Activate(float value, const float * params, size_t offset);

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationIdentity>(float value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationLeakyRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[0] * Simd::Min(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationRestrictRange>(float value, const float * params, size_t offset)
        {
            return Simd::Min(Simd::Max(params[0], value), params[1]);
        }

        template<> SIMD_INLINE float Activate<::SimdConvolutionActivationPrelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[offset] * Simd::Min(0.0f, value);
        }

        template<::SimdConvolutionActivationType type, UpdateType update> void DirectConvolutionBiasActivation(
            const float * src, const SimdConvolutionParameters & p, size_t yBeg, size_t yEnd, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            Array32f buf(dstC);
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    if (bias)
                        memcpy(buf.data, bias, dstC * sizeof(float));
                    else
                        memset(buf.data, 0, dstC * sizeof(float));
                    for (size_t ky = 0; ky < kernelY; ++ky)
                    {
                        size_t sy = dy * strideY + ky - padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < kernelX; ++kx)
                            {
                                size_t sx = dx * strideX + kx - padX;
                                if (sx < p.srcW)
                                {
                                    const float * pw = weight + (ky*kernelX + kx)*srcC*dstC;
                                    const float * ps = src + (sy*srcW + sx)*srcC;
                                    for (size_t sc = 0; sc < srcC; ++sc)
                                    {
                                        for (size_t dc = 0; dc < dstC; ++dc)
                                            buf[dc] += ps[sc] * pw[dc];
                                        pw += dstC;
                                    }
                                }
                            }
                        }
                    }
                    for (size_t dc = 0; dc < dstC; ++dc)
                        Base::Update<update>(dst + dc, Activate<type>(buf[dc], params, dc));
                    dst += p.dstC;
                }
            }
        }

        template<::SimdConvolutionActivationType type> void DepthwiseConvolutionBiasActivation(
            const float * src, const SimdConvolutionParameters & p, size_t yBeg, size_t yEnd, const float * weight, const float * bias, const float * params, float * dst)
        {
            assert(p.group == p.srcC && p.group == p.dstC);
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    for (size_t c = 0; c < srcC; ++c)
                    {
                        float sum = bias ? bias[c] : 0;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float * pw = weight + (ky * kernelX + kx) * srcC + c;
                                        const float * ps = src + (sy * srcW + sx) * srcC + c;
                                        sum += ps[0] * pw[0];
                                    }
                                }
                            }
                        }
                        dst[c] = Activate<type>(sum, params, c);
                    }
                    dst += srcC;
                }
            }
        }

        MergedConvolution::MergedConvolution(const MergConvParam & p)
            : Base::MergedConvolution(p)
        {
            _sizeS = p.conv[0].srcH*p.conv[0].srcW*p.conv[0].srcC;
            _sizeB0 = p.conv[1].srcH*p.conv[1].srcW*p.conv[1].srcC;
            _sizeB1 = p.conv[1].dstH*p.conv[1].dstW*p.conv[1].dstC;
            _sizeD = p.conv[2].dstH*p.conv[2].dstW*p.conv[2].dstC;

            switch (p.conv[0].activation)
            {
            case SimdConvolutionActivationIdentity: _convolution[0] = DirectConvolutionBiasActivation<SimdConvolutionActivationIdentity, UpdateSet>; break;
            case SimdConvolutionActivationRelu: _convolution[0] = DirectConvolutionBiasActivation<SimdConvolutionActivationRelu, UpdateSet>; break;
            case SimdConvolutionActivationLeakyRelu: _convolution[0] = DirectConvolutionBiasActivation<SimdConvolutionActivationLeakyRelu, UpdateSet>; break;
            case SimdConvolutionActivationRestrictRange: _convolution[0] = DirectConvolutionBiasActivation<SimdConvolutionActivationRestrictRange, UpdateSet>; break;
            case SimdConvolutionActivationPrelu: _convolution[0] = DirectConvolutionBiasActivation<SimdConvolutionActivationPrelu, UpdateSet>; break;
            default: assert(0);
            }
            switch (p.conv[1].activation)
            {
            case SimdConvolutionActivationIdentity: _convolution[1] = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: _convolution[1] = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: _convolution[1] = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: _convolution[1] = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _convolution[1] = DepthwiseConvolutionBiasActivation<SimdConvolutionActivationPrelu>; break;
            default: assert(0);
            }
            if (p.add)
            {
                switch (p.conv[2].activation)
                {
                case SimdConvolutionActivationIdentity: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationIdentity, UpdateAdd>; break;
                case SimdConvolutionActivationRelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationRelu, UpdateAdd>; break;
                case SimdConvolutionActivationLeakyRelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationLeakyRelu, UpdateAdd>; break;
                case SimdConvolutionActivationRestrictRange: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationRestrictRange, UpdateAdd>; break;
                case SimdConvolutionActivationPrelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationPrelu, UpdateAdd>; break;
                default: assert(0);
                }
            }
            else
            {
                switch (p.conv[2].activation)
                {
                case SimdConvolutionActivationIdentity: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationIdentity, UpdateSet>; break;
                case SimdConvolutionActivationRelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationRelu, UpdateSet>; break;
                case SimdConvolutionActivationLeakyRelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationLeakyRelu, UpdateSet>; break;
                case SimdConvolutionActivationRestrictRange: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationRestrictRange, UpdateSet>; break;
                case SimdConvolutionActivationPrelu: _convolution[2] = DirectConvolutionBiasActivation<SimdConvolutionActivationPrelu, UpdateSet>; break;
                default: assert(0);
                }
            }
        }

        //---------------------------------------------------------------------

        void * MergedConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add)
        {
            MergConvParam param(trans, batch, convs, count, add);
            if (!param.Valid())
                return NULL;
            return new MergedConvolution(param);
        }
#if 0
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
                microN = 12;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Sse::GemmKernel4x12nn;
                kernelMT = tail > DF ? Sse::GemmKernel4x12nn : (tail > F ? Sse::GemmKernel4x8nn : Sse::GemmKernel4x4nn);
                kernelTM = Sse::GemmKernelMx12nn;
                kernelTT = tail > DF ? Sse::GemmKernelMx12nn : (tail > F ? Sse::GemmKernelMx8nn : Sse::GemmKernelMx4nn);
            }
            else
            {
                microM = 6;
                microN = 8;
                size_t tail = N - AlignLoAny(N, microN);
                kernelMM = Sse::GemmKernel6x8nn;
                kernelMT = tail > F ? Sse::GemmKernel6x8nn : Sse::GemmKernel6x4nn;
                kernelTM = Sse::GemmKernelMx8nn;
                kernelTT = tail > F ? Sse::GemmKernelMx8nn : Sse::GemmKernelMx4nn;
            }
#else
            microM = 4;
            microN = 4;
            kernelMM = Sse::GemmKernel4x4nn;
            kernelMT = Sse::GemmKernel4x4nn;
            kernelTM = Sse::GemmKernelMx4nn;
            kernelTT = Sse::GemmKernelMx4nn;
#endif
            return NhwcGemm(M, N, K, microM, microN, L1, L2, L3, F, kernelMM, kernelMT, kernelTM, kernelTT, Sse::GemmPackB, Sse::GemmScaleC, NULL);
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

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m128 Activate(__m128 value, const float * params, size_t offset);

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationIdentity>(__m128 value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRelu>(__m128 value, const float * params, size_t offset)
        {
            return _mm_max_ps(_mm_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationLeakyRelu>(__m128 value, const float * params, size_t offset)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(_mm_set1_ps(params[0]), _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRestrictRange>(__m128 value, const float * params, size_t offset)
        {
            return _mm_min_ps(_mm_max_ps(_mm_set1_ps(params[0]), value), _mm_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationPrelu>(__m128 value, const float * params, size_t offset)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(_mm_loadu_ps(params + offset), _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Edge1(const float * src, const MergConvParam & p, 
            size_t dy, size_t dx, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m128 sum = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
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
                                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                            }
                        }
                    }
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                weight -= srcCF - c;
                __m128 sum = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
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
                                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                            }
                        }
                    }
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum, params, c));
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Main1(const float * src, 
            size_t srcS, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m128 sum = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * srcC), _mm_loadu_ps(pw + 0 * srcC)), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * srcC), _mm_loadu_ps(pw + 1 * srcC)), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * srcC), _mm_loadu_ps(pw + 2 * srcC)), sum);
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                weight -= srcCF - c;
                __m128 sum = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * srcC), _mm_loadu_ps(pw + 0 * srcC)), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * srcC), _mm_loadu_ps(pw + 1 * srcC)), sum);
                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * srcC), _mm_loadu_ps(pw + 2 * srcC)), sum);
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum, params, c));
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Main2(const float * src, 
            size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            __m128 sum0, sum1, w0;
            for (; c < srcCF; c += F)
            {
                sum0 = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 0 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 0 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 1 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 1 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 2 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 2 * srcC), w0), sum1);
                    pw += srcC;
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum0, params, c));
                _mm_storeu_ps(dst + c + srcC, Activate<type>(sum1, params, c));
                src += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                sum0 = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 0 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 0 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 1 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 1 * srcC), w0), sum1);
                    pw += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + 2 * srcC), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + 2 * srcC), w0), sum1);
                    pw += srcC;
                }
                _mm_storeu_ps(dst + c, Activate<type>(sum0, params, c));
                _mm_storeu_ps(dst + c + srcC, Activate<type>(sum1, params, c));
            }
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE void DepthwiseConvolutionBiasActivation3x3Main4(const float * src, 
            size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m128 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
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
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                }
                _mm_storeu_ps(dst + 0 * srcC, Activate<type>(sum0, params, c));
                _mm_storeu_ps(dst + 1 * srcC, Activate<type>(sum1, params, c));
                _mm_storeu_ps(dst + 2 * srcC, Activate<type>(sum2, params, c));
                _mm_storeu_ps(dst + 3 * srcC, Activate<type>(sum3, params, c));
                src += F;
                dst += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                dst -= srcCF - c;
                __m128 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
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
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm_loadu_ps(pw);
                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps0 + offset), w0), sum0);
                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps1 + offset), w0), sum1);
                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps2 + offset), w0), sum2);
                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps3 + offset), w0), sum3);
                    pw += srcC, offset += srcC;
                }
                _mm_storeu_ps(dst + 0 * srcC, Activate<type>(sum0, params, c));
                _mm_storeu_ps(dst + 1 * srcC, Activate<type>(sum1, params, c));
                _mm_storeu_ps(dst + 2 * srcC, Activate<type>(sum2, params, c));
                _mm_storeu_ps(dst + 3 * srcC, Activate<type>(sum3, params, c));
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
            : Base::MergedConvolution(p)
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
            _gemm.Init(Sse::Gemm32fNN, "Sse", p.gemm, "Ext");
            NhwcGemm nhwcGemm = CreateNhwcGemm(_M, _N, _K);
            _nhwcWeight.Resize(nhwcGemm.BufferSize());
            _nhwcRun = Sse::NhwcRun;
            _nhwcReorderB = Sse::NhwcReorderB;
            _biasAndActivation = Sse::ConvolutionBiasAndActivation;
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
                return new Sse::MergedConvolution(param);
            else
                return new Base::MergedConvolution(param);
        }
#endif
    }
 #endif//SIMD_SSE_ENABLE
}
