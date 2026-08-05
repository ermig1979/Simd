/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdSve2.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)      
    namespace Sve2
    {
        SynetInnerProduct32fGemm::SynetInnerProduct32fGemm(const InnerProductParam32f& p)
            : Base::SynetInnerProduct32fGemm(p)
        {
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
            if (_param.transB)
            {
                _gemm = Sve2::Gemm32fNT;
                if (_M == 1 && _param.activation == SimdConvolutionActivationIdentity && _param.constB)
                    _prod = Sve2::SynetInnerProductLayerForward;
                else
                    _prod = NULL;
            }
            else
            {
                _gemm = Sve2::Gemm32fNN;
            }
            if (_param.N > Neon::F && _prod == NULL && _param.constB)
            {
                _cbRun = Neon::Gemm32fNNcbRun;
                _cbPack = Neon::Gemm32fNNcbReorderB;
                _cbWeight.Resize(Neon::Gemm32fNNcbBufferSize(_M, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
            }
        }

        //-----------------------------------------------------------------------------------------

        void InnerProductKxKNr1x1(size_t K, const float* src, const float* weight0, const float* bias, float* dst, const svbool_t& tail)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t d00 = svld1_f32(body, bias + 0 * F);
            svfloat32_t s0, s1, s2, s3, w0, w1, w2, w3;
            size_t K2 = AlignLo(K, 2);
            size_t K4 = AlignLo(K, 4);
            size_t k = 0, off = 0;
            for (; k < K4; k += 4, off += F * 4)
            {
                s0 = svdup_n_f32(src[k + 0]);
                s1 = svdup_n_f32(src[k + 1]);
                s2 = svdup_n_f32(src[k + 2]);
                s3 = svdup_n_f32(src[k + 3]);
                w0 = svld1_f32(body, weight0 + off + 0 * F);
                w1 = svld1_f32(body, weight0 + off + 1 * F);
                w2 = svld1_f32(body, weight0 + off + 2 * F);
                w3 = svld1_f32(body, weight0 + off + 3 * F);
                d00 = svmla_f32_x(body, d00, w0, s0);
                d00 = svmla_f32_x(body, d00, w1, s1);
                d00 = svmla_f32_x(body, d00, w2, s2);
                d00 = svmla_f32_x(body, d00, w3, s3);
            }
            for (; k < K2; k += 2, off += F * 2)
            {
                s0 = svdup_n_f32(src[k + 0]);
                s1 = svdup_n_f32(src[k + 1]);
                w0 = svld1_f32(body, weight0 + off + 0 * F);
                w1 = svld1_f32(body, weight0 + off + 1 * F);
                d00 = svmla_f32_x(body, d00, w0, s0);
                d00 = svmla_f32_x(body, d00, w1, s1);
            }
            for (; k < K; k++, off += F)
            {
                s0 = svdup_n_f32(src[k]);
                w0 = svld1_f32(body, weight0 + off);
                d00 = svmla_f32_x(body, d00, w0, s0);
            }
            svst1_f32(tail, dst + 0 * F, d00);
        }

        void InnerProductKxKNr1x4(size_t K, const float* src, const float* weight0, const float* bias, float* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t d00 = svld1_f32(body, bias + 0 * F);
            svfloat32_t d01 = svld1_f32(body, bias + 1 * F);
            svfloat32_t d02 = svld1_f32(body, bias + 2 * F);
            svfloat32_t d03 = svld1_f32(body, bias + 3 * F);
            svfloat32_t s0, s1, s2, s3, w00, w01, w02, w03, w10, w11, w12, w13;
            const float* weight1 = weight0 + 1 * K * F;
            const float* weight2 = weight0 + 2 * K * F;
            const float* weight3 = weight0 + 3 * K * F;
            size_t K2 = AlignLo(K, 2);
            size_t K4 = AlignLo(K, 4);
            size_t k = 0, off = 0;
            for (; k < K4; k += 4, off += F * 4)
            {
                s0 = svdup_n_f32(src[k + 0]);
                s1 = svdup_n_f32(src[k + 1]);
                s2 = svdup_n_f32(src[k + 2]);
                s3 = svdup_n_f32(src[k + 3]);
                w00 = svld1_f32(body, weight0 + off + 0 * F);
                w01 = svld1_f32(body, weight0 + off + 1 * F);
                w02 = svld1_f32(body, weight0 + off + 2 * F);
                w03 = svld1_f32(body, weight0 + off + 3 * F);
                w10 = svld1_f32(body, weight1 + off + 0 * F);
                w11 = svld1_f32(body, weight1 + off + 1 * F);
                w12 = svld1_f32(body, weight1 + off + 2 * F);
                w13 = svld1_f32(body, weight1 + off + 3 * F);
                d00 = svmla_f32_x(body, d00, w00, s0);
                d01 = svmla_f32_x(body, d01, w10, s0);
                d00 = svmla_f32_x(body, d00, w01, s1);
                d01 = svmla_f32_x(body, d01, w11, s1);
                d00 = svmla_f32_x(body, d00, w02, s2);
                d01 = svmla_f32_x(body, d01, w12, s2);
                d00 = svmla_f32_x(body, d00, w03, s3);
                d01 = svmla_f32_x(body, d01, w13, s3);
                w00 = svld1_f32(body, weight2 + off + 0 * F);
                w01 = svld1_f32(body, weight2 + off + 1 * F);
                w02 = svld1_f32(body, weight2 + off + 2 * F);
                w03 = svld1_f32(body, weight2 + off + 3 * F);
                w10 = svld1_f32(body, weight3 + off + 0 * F);
                w11 = svld1_f32(body, weight3 + off + 1 * F);
                w12 = svld1_f32(body, weight3 + off + 2 * F);
                w13 = svld1_f32(body, weight3 + off + 3 * F);
                d02 = svmla_f32_x(body, d02, w00, s0);
                d03 = svmla_f32_x(body, d03, w10, s0);
                d02 = svmla_f32_x(body, d02, w01, s1);
                d03 = svmla_f32_x(body, d03, w11, s1);
                d02 = svmla_f32_x(body, d02, w02, s2);
                d03 = svmla_f32_x(body, d03, w12, s2);
                d02 = svmla_f32_x(body, d02, w03, s3);
                d03 = svmla_f32_x(body, d03, w13, s3);
            }
            for (; k < K2; k += 2, off += F * 2)
            {
                s0 = svdup_n_f32(src[k + 0]);
                s1 = svdup_n_f32(src[k + 1]);
                w00 = svld1_f32(body, weight0 + off + 0 * F);
                w01 = svld1_f32(body, weight0 + off + 1 * F);
                w10 = svld1_f32(body, weight1 + off + 0 * F);
                w11 = svld1_f32(body, weight1 + off + 1 * F);
                d00 = svmla_f32_x(body, d00, w00, s0);
                d01 = svmla_f32_x(body, d01, w10, s0);
                d00 = svmla_f32_x(body, d00, w01, s1);
                d01 = svmla_f32_x(body, d01, w11, s1);
                w00 = svld1_f32(body, weight2 + off + 0 * F);
                w01 = svld1_f32(body, weight2 + off + 1 * F);
                w10 = svld1_f32(body, weight3 + off + 0 * F);
                w11 = svld1_f32(body, weight3 + off + 1 * F);
                d02 = svmla_f32_x(body, d02, w00, s0);
                d03 = svmla_f32_x(body, d03, w10, s0);
                d02 = svmla_f32_x(body, d02, w01, s1);
                d03 = svmla_f32_x(body, d03, w11, s1);
            }
            for (; k < K; k++, off += F)
            {
                s0 = svdup_n_f32(src[k + 0]);
                w00 = svld1_f32(body, weight0 + off + 0 * F);
                w10 = svld1_f32(body, weight1 + off + 0 * F);
                d00 = svmla_f32_x(body, d00, w00, s0);
                d01 = svmla_f32_x(body, d01, w10, s0);
                w00 = svld1_f32(body, weight2 + off + 0 * F);
                w10 = svld1_f32(body, weight3 + off + 0 * F);
                d02 = svmla_f32_x(body, d02, w00, s0);
                d03 = svmla_f32_x(body, d03, w10, s0);
            }
            svst1_f32(body, dst + 0 * F, d00);
            svst1_f32(body, dst + 1 * F, d01);
            svst1_f32(body, dst + 2 * F, d02);
            svst1_f32(body, dst + 3 * F, d03);
        }

        void InnerProductKxKNr1x8(size_t K, const float* src, const float* weight0, const float* bias, float* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t d00 = svld1_f32(body, bias + 0 * F);
            svfloat32_t d01 = svld1_f32(body, bias + 1 * F);
            svfloat32_t d02 = svld1_f32(body, bias + 2 * F);
            svfloat32_t d03 = svld1_f32(body, bias + 3 * F);
            svfloat32_t d04 = svld1_f32(body, bias + 4 * F);
            svfloat32_t d05 = svld1_f32(body, bias + 5 * F);
            svfloat32_t d06 = svld1_f32(body, bias + 6 * F);
            svfloat32_t d07 = svld1_f32(body, bias + 7 * F);
            svfloat32_t s0, s1, s2, s3, w00, w01, w10, w11;
            const float* weight1 = weight0 + 1 * K * F;
            const float* weight2 = weight0 + 2 * K * F;
            const float* weight3 = weight0 + 3 * K * F;
            size_t K2 = AlignLo(K, 2);
            size_t K4 = AlignLo(K, 4);
            size_t k = 0, off0 = 0, off4 = 4 * K * F;
            for (; k < K4; k += 4, off0 += F * 4, off4 += F * 4)
            {
                s0 = svdup_n_f32(src[k + 0]);
                s1 = svdup_n_f32(src[k + 1]);
                s2 = svdup_n_f32(src[k + 2]);
                s3 = svdup_n_f32(src[k + 3]);

                w00 = svld1_f32(body, weight0 + off0 + 0 * F);
                w01 = svld1_f32(body, weight0 + off0 + 1 * F);
                w10 = svld1_f32(body, weight1 + off0 + 0 * F);
                w11 = svld1_f32(body, weight1 + off0 + 1 * F);
                d00 = svmla_f32_x(body, d00, w00, s0);
                d01 = svmla_f32_x(body, d01, w10, s0);
                d00 = svmla_f32_x(body, d00, w01, s1);
                d01 = svmla_f32_x(body, d01, w11, s1);
                w00 = svld1_f32(body, weight0 + off0 + 2 * F);
                w01 = svld1_f32(body, weight0 + off0 + 3 * F);
                w10 = svld1_f32(body, weight1 + off0 + 2 * F);
                w11 = svld1_f32(body, weight1 + off0 + 3 * F);
                d00 = svmla_f32_x(body, d00, w00, s2);
                d01 = svmla_f32_x(body, d01, w10, s2);
                d00 = svmla_f32_x(body, d00, w01, s3);
                d01 = svmla_f32_x(body, d01, w11, s3);
                w00 = svld1_f32(body, weight2 + off0 + 0 * F);
                w01 = svld1_f32(body, weight2 + off0 + 1 * F);
                w10 = svld1_f32(body, weight3 + off0 + 0 * F);
                w11 = svld1_f32(body, weight3 + off0 + 1 * F);
                d02 = svmla_f32_x(body, d02, w00, s0);
                d03 = svmla_f32_x(body, d03, w10, s0);
                d02 = svmla_f32_x(body, d02, w01, s1);
                d03 = svmla_f32_x(body, d03, w11, s1);
                w00 = svld1_f32(body, weight2 + off0 + 2 * F);
                w01 = svld1_f32(body, weight2 + off0 + 3 * F);
                w10 = svld1_f32(body, weight3 + off0 + 2 * F);
                w11 = svld1_f32(body, weight3 + off0 + 3 * F);
                d02 = svmla_f32_x(body, d02, w00, s2);
                d03 = svmla_f32_x(body, d03, w10, s2);
                d02 = svmla_f32_x(body, d02, w01, s3);
                d03 = svmla_f32_x(body, d03, w11, s3);

                w00 = svld1_f32(body, weight0 + off4 + 0 * F);
                w01 = svld1_f32(body, weight0 + off4 + 1 * F);
                w10 = svld1_f32(body, weight1 + off4 + 0 * F);
                w11 = svld1_f32(body, weight1 + off4 + 1 * F);
                d04 = svmla_f32_x(body, d04, w00, s0);
                d05 = svmla_f32_x(body, d05, w10, s0);
                d04 = svmla_f32_x(body, d04, w01, s1);
                d05 = svmla_f32_x(body, d05, w11, s1);
                w00 = svld1_f32(body, weight0 + off4 + 2 * F);
                w01 = svld1_f32(body, weight0 + off4 + 3 * F);
                w10 = svld1_f32(body, weight1 + off4 + 2 * F);
                w11 = svld1_f32(body, weight1 + off4 + 3 * F);
                d04 = svmla_f32_x(body, d04, w00, s2);
                d05 = svmla_f32_x(body, d05, w10, s2);
                d04 = svmla_f32_x(body, d04, w01, s3);
                d05 = svmla_f32_x(body, d05, w11, s3);
                w00 = svld1_f32(body, weight2 + off4 + 0 * F);
                w01 = svld1_f32(body, weight2 + off4 + 1 * F);
                w10 = svld1_f32(body, weight3 + off4 + 0 * F);
                w11 = svld1_f32(body, weight3 + off4 + 1 * F);
                d06 = svmla_f32_x(body, d06, w00, s0);
                d07 = svmla_f32_x(body, d07, w10, s0);
                d06 = svmla_f32_x(body, d06, w01, s1);
                d07 = svmla_f32_x(body, d07, w11, s1);
                w00 = svld1_f32(body, weight2 + off4 + 2 * F);
                w01 = svld1_f32(body, weight2 + off4 + 3 * F);
                w10 = svld1_f32(body, weight3 + off4 + 2 * F);
                w11 = svld1_f32(body, weight3 + off4 + 3 * F);
                d06 = svmla_f32_x(body, d06, w00, s2);
                d07 = svmla_f32_x(body, d07, w10, s2);
                d06 = svmla_f32_x(body, d06, w01, s3);
                d07 = svmla_f32_x(body, d07, w11, s3);
            }
            for (; k < K2; k += 2, off0 += F * 2, off4 += F * 2)
            {
                s0 = svdup_n_f32(src[k + 0]);
                s1 = svdup_n_f32(src[k + 1]);

                w00 = svld1_f32(body, weight0 + off0 + 0 * F);
                w01 = svld1_f32(body, weight0 + off0 + 1 * F);
                w10 = svld1_f32(body, weight1 + off0 + 0 * F);
                w11 = svld1_f32(body, weight1 + off0 + 1 * F);
                d00 = svmla_f32_x(body, d00, w00, s0);
                d01 = svmla_f32_x(body, d01, w10, s0);
                d00 = svmla_f32_x(body, d00, w01, s1);
                d01 = svmla_f32_x(body, d01, w11, s1);
                w00 = svld1_f32(body, weight2 + off0 + 0 * F);
                w01 = svld1_f32(body, weight2 + off0 + 1 * F);
                w10 = svld1_f32(body, weight3 + off0 + 0 * F);
                w11 = svld1_f32(body, weight3 + off0 + 1 * F);
                d02 = svmla_f32_x(body, d02, w00, s0);
                d03 = svmla_f32_x(body, d03, w10, s0);
                d02 = svmla_f32_x(body, d02, w01, s1);
                d03 = svmla_f32_x(body, d03, w11, s1);

                w00 = svld1_f32(body, weight0 + off4 + 0 * F);
                w01 = svld1_f32(body, weight0 + off4 + 1 * F);
                w10 = svld1_f32(body, weight1 + off4 + 0 * F);
                w11 = svld1_f32(body, weight1 + off4 + 1 * F);
                d04 = svmla_f32_x(body, d04, w00, s0);
                d05 = svmla_f32_x(body, d05, w10, s0);
                d04 = svmla_f32_x(body, d04, w01, s1);
                d05 = svmla_f32_x(body, d05, w11, s1);
                w00 = svld1_f32(body, weight2 + off4 + 0 * F);
                w01 = svld1_f32(body, weight2 + off4 + 1 * F);
                w10 = svld1_f32(body, weight3 + off4 + 0 * F);
                w11 = svld1_f32(body, weight3 + off4 + 1 * F);
                d06 = svmla_f32_x(body, d06, w00, s0);
                d07 = svmla_f32_x(body, d07, w10, s0);
                d06 = svmla_f32_x(body, d06, w01, s1);
                d07 = svmla_f32_x(body, d07, w11, s1);
            }
            for (; k < K; k++, off0 += F, off4 += F)
            {
                s0 = svdup_n_f32(src[k + 0]);

                w00 = svld1_f32(body, weight0 + off0 + 0 * F);
                w10 = svld1_f32(body, weight1 + off0 + 0 * F);
                d00 = svmla_f32_x(body, d00, w00, s0);
                d01 = svmla_f32_x(body, d01, w10, s0);
                w00 = svld1_f32(body, weight2 + off0 + 0 * F);
                w10 = svld1_f32(body, weight3 + off0 + 0 * F);
                d02 = svmla_f32_x(body, d02, w00, s0);
                d03 = svmla_f32_x(body, d03, w10, s0);

                w00 = svld1_f32(body, weight0 + off4 + 0 * F);
                w10 = svld1_f32(body, weight1 + off4 + 0 * F);
                d04 = svmla_f32_x(body, d04, w00, s0);
                d05 = svmla_f32_x(body, d05, w10, s0);
                w00 = svld1_f32(body, weight2 + off4 + 0 * F);
                w10 = svld1_f32(body, weight3 + off4 + 0 * F);
                d06 = svmla_f32_x(body, d06, w00, s0);
                d07 = svmla_f32_x(body, d07, w10, s0);
            }
            svst1_f32(body, dst + 0 * F, d00);
            svst1_f32(body, dst + 1 * F, d01);
            svst1_f32(body, dst + 2 * F, d02);
            svst1_f32(body, dst + 3 * F, d03);
            svst1_f32(body, dst + 4 * F, d04);
            svst1_f32(body, dst + 5 * F, d05);
            svst1_f32(body, dst + 6 * F, d06);
            svst1_f32(body, dst + 7 * F, d07);
        }

        void InnerProductKxKNr(const float* src, const float* weight, const float* bias, size_t input, size_t output, float* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            size_t outputF1 = AlignLo(output, F * 1);
            size_t outputF4 = AlignLo(output, F * 4);
            size_t outputF8 = AlignLo(output, F * 8);
            size_t o = 0;
            for (; o < outputF8; o += F * 8)
                InnerProductKxKNr1x8(input, src, weight + o * input, bias + o, dst + o);
            for (; o < outputF4; o += F * 4)
                InnerProductKxKNr1x4(input, src, weight + o * input, bias + o, dst + o);
            for (; o < outputF1; o += F * 1)
                InnerProductKxKNr1x1(input, src, weight + o * input, bias + o, dst + o, body);
            if (o < output)
                InnerProductKxKNr1x1(input, src, weight + o * input, bias + o, dst + o, svwhilelt_b32(o, output));
        }

        SynetInnerProduct32fProd::SynetInnerProduct32fProd(const InnerProductParam32f& p)
            : Base::SynetInnerProduct32fProd(p)
        {
            if (_param.N > 1)
            {
                SetSize(svcntw());
                _prod = InnerProductKxKNr;
            }
        }

        //-----------------------------------------------------------------------------------------

        void* SynetInnerProduct32fInit(size_t M, size_t N, size_t K, SimdBool transB, SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation)
        {
            InnerProductParam32f param(M, N, K, transB, constB, bias, activation);
            if (!param.Valid())
                return NULL;
            if (SynetInnerProduct32fProd::Preferable(param))
                return new SynetInnerProduct32fProd(param);
            else
                return new SynetInnerProduct32fGemm(param);
        }
    }
#endif
}
