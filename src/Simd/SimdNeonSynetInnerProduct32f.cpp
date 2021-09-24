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
#include "Simd/SimdNeon.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)      
    namespace Neon
    {
        SynetInnerProduct32fGemm::SynetInnerProduct32fGemm(const InnerProductParam32f& p)
            : Base::SynetInnerProduct32fGemm(p)
        {
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
            if (_param.transpose)
            {
                _gemm = Neon::Gemm32fNT;
                if (_M == 1 && _param.activation == SimdConvolutionActivationIdentity)
                    _prod = Neon::SynetInnerProductLayerForward;
                else
                    _prod = NULL;
            }
            else
            {
                _gemm = Neon::Gemm32fNN;
            }
            if (_param.output > Neon::F && _prod == NULL)
            {
                _cbRun = Neon::Gemm32fNNcbRun;
                _cbPack = Neon::Gemm32fNNcbReorderB;
                _cbWeight.Resize(Neon::Gemm32fNNcbBufferSize(_M, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
            }
        }

        //---------------------------------------------------------------------

        void InnerProductKxKNr1x1(size_t K, const float* src, const float* weight0, const float* bias, float* dst, size_t tail)
        {
            float32x4_t d00 = Load<false>(bias + 0 * F);
            float32x4_t s0, s1, s2, s3, w0, w1, w2, w3;
            size_t K2 = AlignLo(K, 2);
            size_t K4 = AlignLo(K, 4);
            size_t k = 0, off = 0;
            for (; k < K4; k += 4, off += F * 4)
            {
                s0 = vdupq_n_f32(src[k + 0]);
                s1 = vdupq_n_f32(src[k + 1]);
                s2 = vdupq_n_f32(src[k + 2]);
                s3 = vdupq_n_f32(src[k + 3]);
                w0 = Load<false>(weight0 + off + 0 * F);
                w1 = Load<false>(weight0 + off + 1 * F);
                w2 = Load<false>(weight0 + off + 2 * F);
                w3 = Load<false>(weight0 + off + 3 * F);
                d00 = vmlaq_f32(d00, w0, s0);
                d00 = vmlaq_f32(d00, w1, s1);
                d00 = vmlaq_f32(d00, w2, s2);
                d00 = vmlaq_f32(d00, w3, s3);
            }
            for (; k < K2; k += 2, off += F * 2)
            {
                s0 = vdupq_n_f32(src[k + 0]);
                s1 = vdupq_n_f32(src[k + 1]);
                w0 = Load<false>(weight0 + off + 0 * F);
                w1 = Load<false>(weight0 + off + 1 * F);
                d00 = vmlaq_f32(d00, w0, s0);
                d00 = vmlaq_f32(d00, w1, s1);
            }
            for (; k < K; k++, off += F)
            {
                s0 = vdupq_n_f32(src[k]);
                w0 = Load<false>(weight0 + off);
                d00 = vmlaq_f32(d00, w0, s0);
            }
           Store(dst + 0 * F, d00, tail);
        }

        void InnerProductKxKNr1x4(size_t K, const float* src, const float* weight0, const float* bias, float* dst)
        {
            float32x4_t d00 = Load<false>(bias + 0 * F);
            float32x4_t d01 = Load<false>(bias + 1 * F);
            float32x4_t d02 = Load<false>(bias + 2 * F);
            float32x4_t d03 = Load<false>(bias + 3 * F);
            float32x4_t s0, s1, s2, s3, w00, w01, w02, w03, w10, w11, w12, w13;
            const float* weight1 = weight0 + 1 * K * F;
            const float* weight2 = weight0 + 2 * K * F;
            const float* weight3 = weight0 + 3 * K * F;
            size_t K2 = AlignLo(K, 2);
            size_t K4 = AlignLo(K, 4);
            size_t k = 0, off = 0;
            for (; k < K4; k += 4, off += F * 4)
            {
                s0 = vdupq_n_f32(src[k + 0]);
                s1 = vdupq_n_f32(src[k + 1]);
                s2 = vdupq_n_f32(src[k + 2]);
                s3 = vdupq_n_f32(src[k + 3]);
                w00 = Load<false>(weight0 + off + 0 * F);
                w01 = Load<false>(weight0 + off + 1 * F);
                w02 = Load<false>(weight0 + off + 2 * F);
                w03 = Load<false>(weight0 + off + 3 * F);
                w10 = Load<false>(weight1 + off + 0 * F);
                w11 = Load<false>(weight1 + off + 1 * F);
                w12 = Load<false>(weight1 + off + 2 * F);
                w13 = Load<false>(weight1 + off + 3 * F);
                d00 = vmlaq_f32(d00, w00, s0);
                d01 = vmlaq_f32(d01, w10, s0);
                d00 = vmlaq_f32(d00, w01, s1);
                d01 = vmlaq_f32(d01, w11, s1);
                d00 = vmlaq_f32(d00, w02, s2);
                d01 = vmlaq_f32(d01, w12, s2);
                d00 = vmlaq_f32(d00, w03, s3);
                d01 = vmlaq_f32(d01, w13, s3);
                w00 = Load<false>(weight2 + off + 0 * F);
                w01 = Load<false>(weight2 + off + 1 * F);
                w02 = Load<false>(weight2 + off + 2 * F);
                w03 = Load<false>(weight2 + off + 3 * F);
                w10 = Load<false>(weight3 + off + 0 * F);
                w11 = Load<false>(weight3 + off + 1 * F);
                w12 = Load<false>(weight3 + off + 2 * F);
                w13 = Load<false>(weight3 + off + 3 * F);
                d02 = vmlaq_f32(d02, w00, s0);
                d03 = vmlaq_f32(d03, w10, s0);
                d02 = vmlaq_f32(d02, w01, s1);
                d03 = vmlaq_f32(d03, w11, s1);
                d02 = vmlaq_f32(d02, w02, s2);
                d03 = vmlaq_f32(d03, w12, s2);
                d02 = vmlaq_f32(d02, w03, s3);
                d03 = vmlaq_f32(d03, w13, s3);
            }
            for (; k < K2; k += 2, off += F * 2)
            {
                s0 = vdupq_n_f32(src[k + 0]);
                s1 = vdupq_n_f32(src[k + 1]);
                w00 = Load<false>(weight0 + off + 0 * F);
                w01 = Load<false>(weight0 + off + 1 * F);
                w10 = Load<false>(weight1 + off + 0 * F);
                w11 = Load<false>(weight1 + off + 1 * F);
                d00 = vmlaq_f32(d00, w00, s0);
                d01 = vmlaq_f32(d01, w10, s0);
                d00 = vmlaq_f32(d00, w01, s1);
                d01 = vmlaq_f32(d01, w11, s1);
                w00 = Load<false>(weight2 + off + 0 * F);
                w01 = Load<false>(weight2 + off + 1 * F);
                w10 = Load<false>(weight3 + off + 0 * F);
                w11 = Load<false>(weight3 + off + 1 * F);
                d02 = vmlaq_f32(d02, w00, s0);
                d03 = vmlaq_f32(d03, w10, s0);
                d02 = vmlaq_f32(d02, w01, s1);
                d03 = vmlaq_f32(d03, w11, s1);
            }
            for (; k < K; k++, off += F)
            {
                s0 = vdupq_n_f32(src[k + 0]);
                w00 = Load<false>(weight0 + off + 0 * F);
                w10 = Load<false>(weight1 + off + 0 * F);
                d00 = vmlaq_f32(d00, w00, s0);
                d01 = vmlaq_f32(d01, w10, s0);
                w00 = Load<false>(weight2 + off + 0 * F);
                w10 = Load<false>(weight3 + off + 0 * F);
                d02 = vmlaq_f32(d02, w00, s0);
                d03 = vmlaq_f32(d03, w10, s0);
            }
            Store<false>(dst + 0 * F, d00);
            Store<false>(dst + 1 * F, d01);
            Store<false>(dst + 2 * F, d02);
            Store<false>(dst + 3 * F, d03);
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
            if (o < output)
                InnerProductKxKNr1x1(input, src, weight + o * input, bias + o, dst + o, output - o);
        }

        SynetInnerProduct32fProd::SynetInnerProduct32fProd(const InnerProductParam32f& p)
            : Base::SynetInnerProduct32fProd(p)
        {
            if (_param.output > 1)
            {
                SetSize(Neon::F);
                _prod = InnerProductKxKNr;
            }
        }

        //---------------------------------------------------------------------

        void* SynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation)
        {
            InnerProductParam32f param(batch, input, output, transpose, activation);
            if (!param.Valid())
                return NULL;
            if (SynetInnerProduct32fProd::Preferable(param) && 0)
                return new SynetInnerProduct32fProd(param);
            else
                return new SynetInnerProduct32fGemm(param);
        }
    }
#endif// SIMD_NEON_ENABLE
}
