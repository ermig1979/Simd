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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdPrefetch.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        SynetInnerProduct32fGemm::SynetInnerProduct32fGemm(const InnerProductParam32f& p)
            : Avx::SynetInnerProduct32fGemm(p)
        {
            _biasAndActivation = Avx2::ConvolutionBiasAndActivation;
            if (_param.transpose)
            {
                if (_param.input > Sse2::F)
                {
                    _gemm = Avx2::Gemm32fNT;
                    if (_M == 1 && _param.activation == SimdConvolutionActivationIdentity)
                        _prod = Avx2::SynetInnerProductLayerForward;
                    else
                        _prod = NULL;
                }
            }
            else
            {
                if (_param.output > Sse2::F)
                    _gemm = Avx2::Gemm32fNN;
            }
            if (_param.output > Sse2::F && _prod == NULL)
            {
                _cbRun = Avx2::Gemm32fNNcbRun;
                _cbPack = Avx2::Gemm32fNNcbReorderB;
                _cbWeight.Resize(Avx2::Gemm32fNNcbBufferSize(_M, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
            }
        }

        //---------------------------------------------------------------------

        void InnerProductKxKNr1x1(size_t K, const float* src, const float* weight0, const float* bias, float* dst, size_t tail)
        {
            __m256 d00 = _mm256_loadu_ps(bias + 0 * F);
            __m256 s0, s1, s2, s3, w0, w1, w2, w3;
            size_t K2 = AlignLo(K, 2);
            size_t K4 = AlignLo(K, 4);
            size_t k = 0, off = 0;
            for (; k < K4; k += 4, off += F * 4)
            {
                s0 = _mm256_set1_ps(src[k + 0]);
                s1 = _mm256_set1_ps(src[k + 1]);
                s2 = _mm256_set1_ps(src[k + 2]);
                s3 = _mm256_set1_ps(src[k + 3]);
                w0 = _mm256_loadu_ps(weight0 + off + 0 * F);
                w1 = _mm256_loadu_ps(weight0 + off + 1 * F);
                w2 = _mm256_loadu_ps(weight0 + off + 2 * F);
                w3 = _mm256_loadu_ps(weight0 + off + 3 * F);
                d00 = _mm256_fmadd_ps(w0, s0, d00);
                d00 = _mm256_fmadd_ps(w1, s1, d00);
                d00 = _mm256_fmadd_ps(w2, s2, d00);
                d00 = _mm256_fmadd_ps(w3, s3, d00);
            }
            for (; k < K2; k += 2, off += F * 2)
            {
                s0 = _mm256_set1_ps(src[k + 0]);
                s1 = _mm256_set1_ps(src[k + 1]);
                w0 = _mm256_loadu_ps(weight0 + off + 0 * F);
                w1 = _mm256_loadu_ps(weight0 + off + 1 * F);
                d00 = _mm256_fmadd_ps(w0, s0, d00);
                d00 = _mm256_fmadd_ps(w1, s1, d00);
            }
            for (; k < K; k++, off += F)
            {
                s0 = _mm256_set1_ps(src[k]);
                w0 = _mm256_loadu_ps(weight0 + off);
                d00 = _mm256_fmadd_ps(w0, s0, d00);
            }
            Avx::Store(dst + 0 * F, d00, tail);
        }

        void InnerProductKxKNr1x4(size_t K, const float* src, const float* weight0, const float* bias, float* dst)
        {
            __m256 d00 = _mm256_loadu_ps(bias + 0 * F);
            __m256 d01 = _mm256_loadu_ps(bias + 1 * F);
            __m256 d02 = _mm256_loadu_ps(bias + 2 * F);
            __m256 d03 = _mm256_loadu_ps(bias + 3 * F);
            __m256 s0, s1, s2, s3, w00, w01, w10, w11;
            const float* weight1 = weight0 + 1 * K * F;
            const float* weight2 = weight0 + 2 * K * F;
            const float* weight3 = weight0 + 3 * K * F;
            size_t K2 = AlignLo(K, 2);
            size_t K4 = AlignLo(K, 4);
            size_t k = 0, off = 0;
            for (; k < K4; k += 4, off += F * 4)
            {
                s0 = _mm256_set1_ps(src[k + 0]);
                s1 = _mm256_set1_ps(src[k + 1]);
                s2 = _mm256_set1_ps(src[k + 2]);
                s3 = _mm256_set1_ps(src[k + 3]);
                w00 = _mm256_loadu_ps(weight0 + off + 0 * F);
                w01 = _mm256_loadu_ps(weight0 + off + 1 * F);
                w10 = _mm256_loadu_ps(weight1 + off + 0 * F);
                w11 = _mm256_loadu_ps(weight1 + off + 1 * F);
                d00 = _mm256_fmadd_ps(w00, s0, d00);
                d01 = _mm256_fmadd_ps(w10, s0, d01);
                d00 = _mm256_fmadd_ps(w01, s1, d00);
                d01 = _mm256_fmadd_ps(w11, s1, d01);
                w00 = _mm256_loadu_ps(weight0 + off + 2 * F);
                w01 = _mm256_loadu_ps(weight0 + off + 3 * F);
                w10 = _mm256_loadu_ps(weight1 + off + 2 * F);
                w11 = _mm256_loadu_ps(weight1 + off + 3 * F);
                d00 = _mm256_fmadd_ps(w00, s2, d00);
                d01 = _mm256_fmadd_ps(w10, s2, d01);
                d00 = _mm256_fmadd_ps(w01, s3, d00);
                d01 = _mm256_fmadd_ps(w11, s3, d01);
                w00 = _mm256_loadu_ps(weight2 + off + 0 * F);
                w01 = _mm256_loadu_ps(weight2 + off + 1 * F);
                w10 = _mm256_loadu_ps(weight3 + off + 0 * F);
                w11 = _mm256_loadu_ps(weight3 + off + 1 * F);
                d02 = _mm256_fmadd_ps(w00, s0, d02);
                d03 = _mm256_fmadd_ps(w10, s0, d03);
                d02 = _mm256_fmadd_ps(w01, s1, d02);
                d03 = _mm256_fmadd_ps(w11, s1, d03);
                w00 = _mm256_loadu_ps(weight2 + off + 2 * F);
                w01 = _mm256_loadu_ps(weight2 + off + 3 * F);
                w10 = _mm256_loadu_ps(weight3 + off + 2 * F);
                w11 = _mm256_loadu_ps(weight3 + off + 3 * F);
                d02 = _mm256_fmadd_ps(w00, s2, d02);
                d03 = _mm256_fmadd_ps(w10, s2, d03);
                d02 = _mm256_fmadd_ps(w01, s3, d02);
                d03 = _mm256_fmadd_ps(w11, s3, d03);
            }
            for (; k < K2; k += 2, off += F * 2)
            {
                s0 = _mm256_set1_ps(src[k + 0]);
                s1 = _mm256_set1_ps(src[k + 1]);
                w00 = _mm256_loadu_ps(weight0 + off + 0 * F);
                w01 = _mm256_loadu_ps(weight0 + off + 1 * F);
                w10 = _mm256_loadu_ps(weight1 + off + 0 * F);
                w11 = _mm256_loadu_ps(weight1 + off + 1 * F);
                d00 = _mm256_fmadd_ps(w00, s0, d00);
                d01 = _mm256_fmadd_ps(w10, s0, d01);
                d00 = _mm256_fmadd_ps(w01, s1, d00);
                d01 = _mm256_fmadd_ps(w11, s1, d01);
                w00 = _mm256_loadu_ps(weight2 + off + 0 * F);
                w01 = _mm256_loadu_ps(weight2 + off + 1 * F);
                w10 = _mm256_loadu_ps(weight3 + off + 0 * F);
                w11 = _mm256_loadu_ps(weight3 + off + 1 * F);
                d02 = _mm256_fmadd_ps(w00, s0, d02);
                d03 = _mm256_fmadd_ps(w10, s0, d03);
                d02 = _mm256_fmadd_ps(w01, s1, d02);
                d03 = _mm256_fmadd_ps(w11, s1, d03);
            }
            for (; k < K; k++, off += F)
            {
                s0 = _mm256_set1_ps(src[k + 0]);
                w00 = _mm256_loadu_ps(weight0 + off + 0 * F);
                w10 = _mm256_loadu_ps(weight1 + off + 0 * F);
                d00 = _mm256_fmadd_ps(w00, s0, d00);
                d01 = _mm256_fmadd_ps(w10, s0, d01);
                w00 = _mm256_loadu_ps(weight2 + off + 0 * F);
                w10 = _mm256_loadu_ps(weight3 + off + 0 * F);
                d02 = _mm256_fmadd_ps(w00, s0, d02);
                d03 = _mm256_fmadd_ps(w10, s0, d03);
            }
            _mm256_storeu_ps(dst + 0 * F, d00);
            _mm256_storeu_ps(dst + 1 * F, d01);
            _mm256_storeu_ps(dst + 2 * F, d02);
            _mm256_storeu_ps(dst + 3 * F, d03);
        }

        void InnerProductKxKNr1x8(size_t K, const float* src, const float* weight0, const float* bias, float* dst)
        {
            __m256 d00 = _mm256_loadu_ps(bias + 0 * F);
            __m256 d01 = _mm256_loadu_ps(bias + 1 * F);
            __m256 d02 = _mm256_loadu_ps(bias + 2 * F);
            __m256 d03 = _mm256_loadu_ps(bias + 3 * F);
            __m256 d04 = _mm256_loadu_ps(bias + 4 * F);
            __m256 d05 = _mm256_loadu_ps(bias + 5 * F);
            __m256 d06 = _mm256_loadu_ps(bias + 6 * F);
            __m256 d07 = _mm256_loadu_ps(bias + 7 * F);
            __m256 s0, s1, s2, s3, w00, w01, w10, w11;
            const float* weight1 = weight0 + 1 * K * F;
            const float* weight2 = weight0 + 2 * K * F;
            const float* weight3 = weight0 + 3 * K * F;
            size_t K2 = AlignLo(K, 2);
            size_t K4 = AlignLo(K, 4);
            size_t k = 0, off0 = 0, off4 = 4 * K * F;
            for (; k < K4; k += 4, off0 += F * 4, off4 += 4 *F)
            {
                s0 = _mm256_set1_ps(src[k + 0]);
                s1 = _mm256_set1_ps(src[k + 1]);
                s2 = _mm256_set1_ps(src[k + 2]);
                s3 = _mm256_set1_ps(src[k + 3]);

                w00 = _mm256_loadu_ps(weight0 + off0 + 0 * F);
                w01 = _mm256_loadu_ps(weight0 + off0 + 1 * F);
                w10 = _mm256_loadu_ps(weight1 + off0 + 0 * F);
                w11 = _mm256_loadu_ps(weight1 + off0 + 1 * F);
                d00 = _mm256_fmadd_ps(w00, s0, d00);
                d01 = _mm256_fmadd_ps(w10, s0, d01);
                d00 = _mm256_fmadd_ps(w01, s1, d00);
                d01 = _mm256_fmadd_ps(w11, s1, d01);
                w00 = _mm256_loadu_ps(weight0 + off0 + 2 * F);
                w01 = _mm256_loadu_ps(weight0 + off0 + 3 * F);
                w10 = _mm256_loadu_ps(weight1 + off0 + 2 * F);
                w11 = _mm256_loadu_ps(weight1 + off0 + 3 * F);
                d00 = _mm256_fmadd_ps(w00, s2, d00);
                d01 = _mm256_fmadd_ps(w10, s2, d01);
                d00 = _mm256_fmadd_ps(w01, s3, d00);
                d01 = _mm256_fmadd_ps(w11, s3, d01);
                w00 = _mm256_loadu_ps(weight2 + off0 + 0 * F);
                w01 = _mm256_loadu_ps(weight2 + off0 + 1 * F);
                w10 = _mm256_loadu_ps(weight3 + off0 + 0 * F);
                w11 = _mm256_loadu_ps(weight3 + off0 + 1 * F);
                d02 = _mm256_fmadd_ps(w00, s0, d02);
                d03 = _mm256_fmadd_ps(w10, s0, d03);
                d02 = _mm256_fmadd_ps(w01, s1, d02);
                d03 = _mm256_fmadd_ps(w11, s1, d03);
                w00 = _mm256_loadu_ps(weight2 + off0 + 2 * F);
                w01 = _mm256_loadu_ps(weight2 + off0 + 3 * F);
                w10 = _mm256_loadu_ps(weight3 + off0 + 2 * F);
                w11 = _mm256_loadu_ps(weight3 + off0 + 3 * F);
                d02 = _mm256_fmadd_ps(w00, s2, d02);
                d03 = _mm256_fmadd_ps(w10, s2, d03);
                d02 = _mm256_fmadd_ps(w01, s3, d02);
                d03 = _mm256_fmadd_ps(w11, s3, d03);

                w00 = _mm256_loadu_ps(weight0 + off4 + 0 * F);
                w01 = _mm256_loadu_ps(weight0 + off4 + 1 * F);
                w10 = _mm256_loadu_ps(weight1 + off4 + 0 * F);
                w11 = _mm256_loadu_ps(weight1 + off4 + 1 * F);
                d04 = _mm256_fmadd_ps(w00, s0, d04);
                d05 = _mm256_fmadd_ps(w10, s0, d05);
                d04 = _mm256_fmadd_ps(w01, s1, d04);
                d05 = _mm256_fmadd_ps(w11, s1, d05);
                w00 = _mm256_loadu_ps(weight0 + off4 + 2 * F);
                w01 = _mm256_loadu_ps(weight0 + off4 + 3 * F);
                w10 = _mm256_loadu_ps(weight1 + off4 + 2 * F);
                w11 = _mm256_loadu_ps(weight1 + off4 + 3 * F);
                d04 = _mm256_fmadd_ps(w00, s2, d04);
                d05 = _mm256_fmadd_ps(w10, s2, d05);
                d04 = _mm256_fmadd_ps(w01, s3, d04);
                d05 = _mm256_fmadd_ps(w11, s3, d05);
                w00 = _mm256_loadu_ps(weight2 + off4 + 0 * F);
                w01 = _mm256_loadu_ps(weight2 + off4 + 1 * F);
                w10 = _mm256_loadu_ps(weight3 + off4 + 0 * F);
                w11 = _mm256_loadu_ps(weight3 + off4 + 1 * F);
                d06 = _mm256_fmadd_ps(w00, s0, d06);
                d07 = _mm256_fmadd_ps(w10, s0, d07);
                d06 = _mm256_fmadd_ps(w01, s1, d06);
                d07 = _mm256_fmadd_ps(w11, s1, d07);
                w00 = _mm256_loadu_ps(weight2 + off4 + 2 * F);
                w01 = _mm256_loadu_ps(weight2 + off4 + 3 * F);
                w10 = _mm256_loadu_ps(weight3 + off4 + 2 * F);
                w11 = _mm256_loadu_ps(weight3 + off4 + 3 * F);
                d06 = _mm256_fmadd_ps(w00, s2, d06);
                d07 = _mm256_fmadd_ps(w10, s2, d07);
                d06 = _mm256_fmadd_ps(w01, s3, d06);
                d07 = _mm256_fmadd_ps(w11, s3, d07);
            }
            for (; k < K2; k += 2, off0 += F * 2, off4 += F * 2)
            {
                s0 = _mm256_set1_ps(src[k + 0]);
                s1 = _mm256_set1_ps(src[k + 1]);

                w00 = _mm256_loadu_ps(weight0 + off0 + 0 * F);
                w01 = _mm256_loadu_ps(weight0 + off0 + 1 * F);
                w10 = _mm256_loadu_ps(weight1 + off0 + 0 * F);
                w11 = _mm256_loadu_ps(weight1 + off0 + 1 * F);
                d00 = _mm256_fmadd_ps(w00, s0, d00);
                d01 = _mm256_fmadd_ps(w10, s0, d01);
                d00 = _mm256_fmadd_ps(w01, s1, d00);
                d01 = _mm256_fmadd_ps(w11, s1, d01);
                w00 = _mm256_loadu_ps(weight2 + off0 + 0 * F);
                w01 = _mm256_loadu_ps(weight2 + off0 + 1 * F);
                w10 = _mm256_loadu_ps(weight3 + off0 + 0 * F);
                w11 = _mm256_loadu_ps(weight3 + off0 + 1 * F);
                d02 = _mm256_fmadd_ps(w00, s0, d02);
                d03 = _mm256_fmadd_ps(w10, s0, d03);
                d02 = _mm256_fmadd_ps(w01, s1, d02);
                d03 = _mm256_fmadd_ps(w11, s1, d03);

                w00 = _mm256_loadu_ps(weight0 + off4 + 0 * F);
                w01 = _mm256_loadu_ps(weight0 + off4 + 1 * F);
                w10 = _mm256_loadu_ps(weight1 + off4 + 0 * F);
                w11 = _mm256_loadu_ps(weight1 + off4 + 1 * F);
                d04 = _mm256_fmadd_ps(w00, s0, d04);
                d05 = _mm256_fmadd_ps(w10, s0, d05);
                d04 = _mm256_fmadd_ps(w01, s1, d04);
                d05 = _mm256_fmadd_ps(w11, s1, d05);
                w00 = _mm256_loadu_ps(weight2 + off4 + 0 * F);
                w01 = _mm256_loadu_ps(weight2 + off4 + 1 * F);
                w10 = _mm256_loadu_ps(weight3 + off4 + 0 * F);
                w11 = _mm256_loadu_ps(weight3 + off4 + 1 * F);
                d06 = _mm256_fmadd_ps(w00, s0, d06);
                d07 = _mm256_fmadd_ps(w10, s0, d07);
                d06 = _mm256_fmadd_ps(w01, s1, d06);
                d07 = _mm256_fmadd_ps(w11, s1, d07);

            }
            for (; k < K; k++, off0 += F, off4 += F)
            {
                s0 = _mm256_set1_ps(src[k + 0]);

                w00 = _mm256_loadu_ps(weight0 + off0 + 0 * F);
                w10 = _mm256_loadu_ps(weight1 + off0 + 0 * F);
                d00 = _mm256_fmadd_ps(w00, s0, d00);
                d01 = _mm256_fmadd_ps(w10, s0, d01);
                w00 = _mm256_loadu_ps(weight2 + off0 + 0 * F);
                w10 = _mm256_loadu_ps(weight3 + off0 + 0 * F);
                d02 = _mm256_fmadd_ps(w00, s0, d02);
                d03 = _mm256_fmadd_ps(w10, s0, d03);

                w00 = _mm256_loadu_ps(weight0 + off4 + 0 * F);
                w10 = _mm256_loadu_ps(weight1 + off4 + 0 * F);
                d04 = _mm256_fmadd_ps(w00, s0, d04);
                d05 = _mm256_fmadd_ps(w10, s0, d05);
                w00 = _mm256_loadu_ps(weight2 + off4 + 0 * F);
                w10 = _mm256_loadu_ps(weight3 + off4 + 0 * F);
                d06 = _mm256_fmadd_ps(w00, s0, d06);
                d07 = _mm256_fmadd_ps(w10, s0, d07);
            }
            _mm256_storeu_ps(dst + 0 * F, d00);
            _mm256_storeu_ps(dst + 1 * F, d01);
            _mm256_storeu_ps(dst + 2 * F, d02);
            _mm256_storeu_ps(dst + 3 * F, d03);
            _mm256_storeu_ps(dst + 4 * F, d04);
            _mm256_storeu_ps(dst + 5 * F, d05);
            _mm256_storeu_ps(dst + 6 * F, d06);
            _mm256_storeu_ps(dst + 7 * F, d07);
        }

        void InnerProductKxKNr(const float* src, const float* weight, const float* bias, size_t input, size_t output, float* dst)
        {
            size_t outputF1 = AlignLo(output, F * 1);
            size_t outputF4 = AlignLo(output, F * 4);
            size_t outputF8 = AlignLo(output, F * 8);
            size_t o = 0;
            for (; o < outputF8; o += F * 8)
                InnerProductKxKNr1x8(input, src, weight + o * input, bias + o, dst + o);
            for (; o < outputF4; o += F * 4)
                InnerProductKxKNr1x4(input, src, weight + o * input, bias + o, dst + o);
            for (; o < outputF1; o += F * 1)
                InnerProductKxKNr1x1(input, src, weight + o * input, bias + o, dst + o, F);
            if (o < output)
                InnerProductKxKNr1x1(input, src, weight + o * input, bias + o, dst + o, output - o);
        }

        SynetInnerProduct32fProd::SynetInnerProduct32fProd(const InnerProductParam32f& p)
            : Avx::SynetInnerProduct32fProd(p)
        {
            if (_param.output > Sse2::F)
            {
                SetSize(Avx::F);
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
#endif// SIMD_AVX2_ENABLE
}
