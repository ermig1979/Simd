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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynetInnerProduct32f.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdSse41.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)      
    namespace Sse41
    {
        SynetInnerProduct32fGemm::SynetInnerProduct32fGemm(const InnerProductParam32f& p)
            : Base::SynetInnerProduct32fGemm(p)
        {
            _biasAndActivation = Sse2::ConvolutionBiasAndActivation;
            if (_param.transpose)
            {
                _gemm = Sse41::Gemm32fNT;
                if (_M == 1 && _param.activation == SimdConvolutionActivationIdentity)
                    _prod = Sse2::SynetInnerProductLayerForward;
                else
                    _prod = NULL;
            }
            else
            {
                _gemm = Sse2::Gemm32fNN;
            }
            if (_param.output > Sse2::F && _prod == NULL)
            {
                _cbRun = Sse2::Gemm32fNNcbRun;
                _cbPack = Sse2::Gemm32fNNcbReorderB;
                _cbWeight.Resize(Sse2::Gemm32fNNcbBufferSize(_M, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
            }
        }

        //---------------------------------------------------------------------

        void InnerProductKxKNr1x1(size_t K, const float *src, const float* weight0, const float* bias, float* dst, size_t tail)
        {
            __m128 d00 = _mm_loadu_ps(bias + 0 * F);
            __m128 s0, s1, s2, s3, w0, w1, w2, w3;
            size_t K2 = AlignLo(K, 2);
            size_t K4 = AlignLo(K, 4);
            size_t k = 0, off = 0;
            for (; k < K4; k += 4, off += F * 4)
            {
                s0 = _mm_set1_ps(src[k + 0]);
                s1 = _mm_set1_ps(src[k + 1]);
                s2 = _mm_set1_ps(src[k + 2]);
                s3 = _mm_set1_ps(src[k + 3]);
                w0 = _mm_loadu_ps(weight0 + off + 0 * F);
                w1 = _mm_loadu_ps(weight0 + off + 1 * F);
                w2 = _mm_loadu_ps(weight0 + off + 2 * F);
                w3 = _mm_loadu_ps(weight0 + off + 3 * F);
                d00 = _mm_add_ps(_mm_mul_ps(w0, s0), d00);
                d00 = _mm_add_ps(_mm_mul_ps(w1, s1), d00);
                d00 = _mm_add_ps(_mm_mul_ps(w2, s2), d00);
                d00 = _mm_add_ps(_mm_mul_ps(w3, s3), d00);
            }
            for (; k < K2; k += 2, off += F * 2)
            {
                s0 = _mm_set1_ps(src[k + 0]);
                s1 = _mm_set1_ps(src[k + 1]);
                w0 = _mm_loadu_ps(weight0 + off + 0 * F);
                w1 = _mm_loadu_ps(weight0 + off + 1 * F);
                d00 = _mm_add_ps(_mm_mul_ps(w0, s0), d00);
                d00 = _mm_add_ps(_mm_mul_ps(w1, s1), d00);
            }
            for (; k < K; k++, off += F)
            {
                s0 = _mm_set1_ps(src[k]);
                w0 = _mm_loadu_ps(weight0 + off);
                d00 = _mm_add_ps(_mm_mul_ps(w0, s0), d00);
            }
            Sse2::Store(dst + 0 * F, d00, tail);
        }

        void InnerProductKxKNr1x4(size_t K, const float* src, const float* weight0, const float* bias, float* dst)
        {
            __m128 d00 = _mm_loadu_ps(bias + 0 * F);
            __m128 d01 = _mm_loadu_ps(bias + 1 * F);
            __m128 d02 = _mm_loadu_ps(bias + 2 * F);
            __m128 d03 = _mm_loadu_ps(bias + 3 * F);
            __m128 s0, s1, s2, s3, w00, w01, w02, w03, w10, w11, w12, w13;
            const float* weight1 = weight0 + 1 * K * F;
            const float* weight2 = weight0 + 2 * K * F;
            const float* weight3 = weight0 + 3 * K * F;
            size_t K2 = AlignLo(K, 2);
            size_t K4 = AlignLo(K, 4);
            size_t k = 0, off = 0;
            for (; k < K4; k += 4, off += F * 4)
            {
                s0 = _mm_set1_ps(src[k + 0]);
                s1 = _mm_set1_ps(src[k + 1]);
                s2 = _mm_set1_ps(src[k + 2]);
                s3 = _mm_set1_ps(src[k + 3]);
                w00 = _mm_loadu_ps(weight0 + off + 0 * F);
                w01 = _mm_loadu_ps(weight0 + off + 1 * F);
                w02 = _mm_loadu_ps(weight0 + off + 2 * F);
                w03 = _mm_loadu_ps(weight0 + off + 3 * F);
                w10 = _mm_loadu_ps(weight1 + off + 0 * F);
                w11 = _mm_loadu_ps(weight1 + off + 1 * F);
                w12 = _mm_loadu_ps(weight1 + off + 2 * F);
                w13 = _mm_loadu_ps(weight1 + off + 3 * F);
                d00 = _mm_add_ps(_mm_mul_ps(w00, s0), d00);
                d01 = _mm_add_ps(_mm_mul_ps(w10, s0), d01);
                d00 = _mm_add_ps(_mm_mul_ps(w01, s1), d00);
                d01 = _mm_add_ps(_mm_mul_ps(w11, s1), d01);
                d00 = _mm_add_ps(_mm_mul_ps(w02, s2), d00);
                d01 = _mm_add_ps(_mm_mul_ps(w12, s2), d01);
                d00 = _mm_add_ps(_mm_mul_ps(w03, s3), d00);
                d01 = _mm_add_ps(_mm_mul_ps(w13, s3), d01);
                w00 = _mm_loadu_ps(weight2 + off + 0 * F);
                w01 = _mm_loadu_ps(weight2 + off + 1 * F);
                w02 = _mm_loadu_ps(weight2 + off + 2 * F);
                w03 = _mm_loadu_ps(weight2 + off + 3 * F);
                w10 = _mm_loadu_ps(weight3 + off + 0 * F);
                w11 = _mm_loadu_ps(weight3 + off + 1 * F);
                w12 = _mm_loadu_ps(weight3 + off + 2 * F);
                w13 = _mm_loadu_ps(weight3 + off + 3 * F);
                d02 = _mm_add_ps(_mm_mul_ps(w00, s0), d02);
                d03 = _mm_add_ps(_mm_mul_ps(w10, s0), d03);
                d02 = _mm_add_ps(_mm_mul_ps(w01, s1), d02);
                d03 = _mm_add_ps(_mm_mul_ps(w11, s1), d03);
                d02 = _mm_add_ps(_mm_mul_ps(w02, s2), d02);
                d03 = _mm_add_ps(_mm_mul_ps(w12, s2), d03);
                d02 = _mm_add_ps(_mm_mul_ps(w03, s3), d02);
                d03 = _mm_add_ps(_mm_mul_ps(w13, s3), d03);
            }
            for (; k < K2; k += 2, off += F * 2)
            {
                s0 = _mm_set1_ps(src[k + 0]);
                s1 = _mm_set1_ps(src[k + 1]);
                w00 = _mm_loadu_ps(weight0 + off + 0 * F);
                w01 = _mm_loadu_ps(weight0 + off + 1 * F);
                w10 = _mm_loadu_ps(weight1 + off + 0 * F);
                w11 = _mm_loadu_ps(weight1 + off + 1 * F);
                d00 = _mm_add_ps(_mm_mul_ps(w00, s0), d00);
                d01 = _mm_add_ps(_mm_mul_ps(w10, s0), d01);
                d00 = _mm_add_ps(_mm_mul_ps(w01, s1), d00);
                d01 = _mm_add_ps(_mm_mul_ps(w11, s1), d01);
                w00 = _mm_loadu_ps(weight2 + off + 0 * F);
                w01 = _mm_loadu_ps(weight2 + off + 1 * F);
                w10 = _mm_loadu_ps(weight3 + off + 0 * F);
                w11 = _mm_loadu_ps(weight3 + off + 1 * F);
                d02 = _mm_add_ps(_mm_mul_ps(w00, s0), d02);
                d03 = _mm_add_ps(_mm_mul_ps(w10, s0), d03);
                d02 = _mm_add_ps(_mm_mul_ps(w01, s1), d02);
                d03 = _mm_add_ps(_mm_mul_ps(w11, s1), d03);
            }
            for (; k < K; k++, off += F)
            {
                s0 = _mm_set1_ps(src[k + 0]);
                w00 = _mm_loadu_ps(weight0 + off + 0 * F);
                w10 = _mm_loadu_ps(weight1 + off + 0 * F);
                d00 = _mm_add_ps(_mm_mul_ps(w00, s0), d00);
                d01 = _mm_add_ps(_mm_mul_ps(w10, s0), d01);
                w00 = _mm_loadu_ps(weight2 + off + 0 * F);
                w10 = _mm_loadu_ps(weight3 + off + 0 * F);
                d02 = _mm_add_ps(_mm_mul_ps(w00, s0), d02);
                d03 = _mm_add_ps(_mm_mul_ps(w10, s0), d03);
            }
            _mm_storeu_ps(dst + 0 * F, d00);
            _mm_storeu_ps(dst + 1 * F, d01);
            _mm_storeu_ps(dst + 2 * F, d02);
            _mm_storeu_ps(dst + 3 * F, d03);
        }

        void InnerProductKxKNr(const float* src, const float* weight, const float* bias, size_t input, size_t output, float* dst)
        {
            size_t outputF1 = AlignLo(output, F * 1);
            size_t outputF4 = AlignLo(output, F * 4);
            size_t o = 0;
            for (; o < outputF4; o += F * 4)
                InnerProductKxKNr1x4(input, src, weight + o * input, bias + o, dst + o);
            for (; o < outputF1; o += F * 1)
                InnerProductKxKNr1x1(input, src, weight + o * input, bias + o, dst + o, F);
            if(o < output)
                InnerProductKxKNr1x1(input, src, weight + o * input, bias + o, dst + o, output - o);
        }

        SynetInnerProduct32fProd::SynetInnerProduct32fProd(const InnerProductParam32f& p)
            : Base::SynetInnerProduct32fProd(p)
        {
            if (_param.output > 1)
            {
                SetSize(Sse2::F);
                _prod = InnerProductKxKNr;
            }
        }

        //---------------------------------------------------------------------

        void* SynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation)
        {
            InnerProductParam32f param(batch, input, output, transpose, activation);
            if (!param.Valid())
                return NULL;
            if (SynetInnerProduct32fProd::Preferable(param))
                return new SynetInnerProduct32fProd(param);
            else
                return new SynetInnerProduct32fGemm(param);
        }
    }
#endif// SIMD_SSE41_ENABLE
}
