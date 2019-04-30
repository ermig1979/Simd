/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
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
#include "Test/TestUtils.h"
#include "Test/TestPerformance.h"
#include "Test/TestData.h"
#include "Test/TestTensor.h"

#include "Simd/SimdMergedConvolution.h"

namespace Test
{
    namespace
    {
        struct Param
        {
            size_t batch, srcC, srcH, srcW, dstC, kernelY, kernelX, strideY, strideX, padY, padX, padH, padW;
            ::SimdConvolutionActivationType activation0, activation1;

            Param(size_t n, size_t sC, size_t sH, size_t sW, size_t dC, size_t kY, size_t kX, size_t sY, size_t sX, 
                size_t pY, size_t pX, size_t pH, size_t pW, ::SimdConvolutionActivationType a0, ::SimdConvolutionActivationType a1)
                : batch(n), srcC(sC), srcH(sH), srcW(sW), dstC(dC), kernelY(kY), kernelX(kX), strideY(sY), strideX(sX), 
                padY(pY), padX(pX), padH(pH), padW(pW), activation0(a0), activation1(a1)
            {}

            Param(size_t n, size_t sC, size_t sH, size_t sW, size_t dC, Size k, Size s, Size b, Size e, 
                ::SimdConvolutionActivationType a0, ::SimdConvolutionActivationType a1)
                : batch(n), srcC(sC), srcH(sH), srcW(sW), dstC(dC), kernelY(k.y), kernelX(k.x), strideY(s.y), strideX(s.x), 
                padY(b.y), padX(b.x), padH(e.y), padW(e.x), activation0(a0), activation1(a1)
            {}
        };

        struct FuncMC
        {
            typedef void*(*FuncPtr)(size_t batch, size_t srcC, size_t srcH, size_t srcW, size_t dstC,
                size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW,
                SimdConvolutionActivationType activation0, SimdConvolutionActivationType activation1, SimdGemm32fNNPtr gemm);

            FuncPtr func;
            String description;

            FuncMC(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(const Param & p)
            {
                std::stringstream ss;
                ss << description;
                ss << "[" << p.batch << "x" << p.srcC << "x" << p.srcH << "x" << p.srcW;
                ss << "-" << p.dstC << "x" << p.kernelY << "x" << p.kernelX;
                ss << "-" << p.strideX << "-" << Simd::Max(p.padX, p.padW);
                ss << "]";
                description = ss.str();
            }

            void Call(const Param & p, const Tensor32f & w0, const Tensor32f & w1, const Tensor32f & b0, const Tensor32f & b1, 
                const Tensor32f & p0, const Tensor32f & p1, const Tensor32f & src, Tensor32f & buf, Tensor32f & dst) const
            {
                void * context = func(p.batch, p.srcC, p.srcH, p.srcW, p.dstC, p.kernelY, p.kernelX, 
                    p.strideY, p.strideX, p.padY, p.padX, p.padH, p.padW, p.activation0, p.activation1, NULL);
                buf.Extend({ ::SimdMergedConvolutionExternalBufferSize(context) });
                ::SimdMergedConvolutionSetParams(context, w0.Data(), w1.Data(), NULL, b0.Data(), b1.Data(), p0.Data(), p1.Data());
                {
                    TEST_PERFORMANCE_TEST(description);
                    ::SimdMergedConvolutionForward(context, src.Data(), buf.Data(), dst.Data());
                }
                ::SimdRelease(context);
            }
        };
    }

#define FUNC_MC(function) \
    FuncMC(function, std::string(#function))

    bool MergedConvolutionForwardAutoTest(float eps, const Param & p, FuncMC f1, FuncMC f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << "].");

        Tensor32f src({p.batch, p.srcH, p.srcW, p.srcC });
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f w0({ p.kernelY, p.kernelX, 1, p.srcC });
        FillRandom(w0.Data(), w0.Size(), -1.0, 1.0f);

        Tensor32f b0({ p.srcC });
        FillRandom(b0.Data(), b0.Size(), -1.0, 1.0f);

        Tensor32f p0({ Simd::Max<size_t>(2, p.srcC) });
        FillRandom(p0.Data(), p0.Size(), 0.0f, 2.0f);
        p0.Data()[0] = 0.1f;
        p0.Data()[1] = 1.1f;

        Tensor32f w1({ 1, 1, p.srcC, p.dstC });
        FillRandom(w1.Data(), w1.Size(), -1.0, 1.0f);

        Tensor32f b1({ p.dstC });
        FillRandom(b1.Data(), b1.Size(), -1.0, 1.0f);

        Tensor32f p1({ Simd::Max<size_t>(2, p.dstC) });
        FillRandom(p1.Data(), p1.Size(), 0.0f, 2.0f);
        p1.Data()[0] = 0.2f;
        p1.Data()[1] = 1.2f;

        Tensor32f buf;

        size_t dstH = (p.srcH + p.padY + p.padH - p.kernelY) / p.strideY + 1;
        size_t dstW = (p.srcW + p.padX + p.padW - p.kernelX) / p.strideX + 1;
        Tensor32f dst1({ p.batch, dstH, dstW, p.dstC});
        Tensor32f dst2({ p.batch, dstH, dstW, p.dstC});

        float fv1 = 0.01, fv2 = 0.02;
        ::SimdFill32f(dst1.Data(), dst1.Size(), &fv1);
        ::SimdFill32f(dst2.Data(), dst2.Size(), &fv2);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(p, w0, w1, b0, b1, p0, p1, src, buf, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(p, w0, w1, b0, b1, p0, p1, src, buf, dst2));

        result = result && Compare(dst1, dst2, eps, true, 64, DifferenceBoth);

        return result;
    }

    bool MergedConvolutionForwardAutoTest(float eps, ::SimdConvolutionActivationType a0, ::SimdConvolutionActivationType a1, const FuncMC & f1, const FuncMC & f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3);
#ifdef NDEBUG
#if 1
        result = result && MergedConvolutionForwardAutoTest(eps, Param(1, 144, 96, 96, 24, _3, _1, _1, _1, a0, a1), f1, f2);
        result = result && MergedConvolutionForwardAutoTest(eps, Param(1, 96, 192, 192, 24, _3, _2, _1, _1, a0, a1), f1, f2);
        result = result && MergedConvolutionForwardAutoTest(eps, Param(1, 32, 192, 192, 16, _3, _1, _1, _1, a0, a1), f1, f2);
        result = result && MergedConvolutionForwardAutoTest(eps, Param(1, 384, 24, 24, 64, _3, _1, _1, _1, a0, a1), f1, f2);
        result = result && MergedConvolutionForwardAutoTest(eps, Param(1, 576, 24, 24, 96, _3, _1, _1, _1, a0, a1), f1, f2);
#endif
#else
        result = result && MergedConvolutionForwardAutoTest(eps, Param(1, 32, 192, 192, 16, _3, _1, _1, _1, a0, a1), f1, f2);
#endif
        return result;
    }

    bool MergedConvolutionForwardAutoTest(float eps, const FuncMC & f1, const FuncMC & f2)
    {
        bool result = true;

        result = result && MergedConvolutionForwardAutoTest(eps, ::SimdConvolutionActivationRestrictRange, ::SimdConvolutionActivationIdentity, f1, f2);

        return result;
    }

    bool MergedConvolutionForwardAutoTest()
    {
        bool result = true;

        result = result && MergedConvolutionForwardAutoTest(EPS, FUNC_MC(Simd::Base::MergedConvolutionInit), FUNC_MC(SimdMergedConvolutionInit));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && MergedConvolutionForwardAutoTest(EPS, FUNC_MC(Simd::Sse::MergedConvolutionInit), FUNC_MC(SimdMergedConvolutionInit));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && MergedConvolutionForwardAutoTest(EPS, FUNC_MC(Simd::Avx::MergedConvolutionInit), FUNC_MC(SimdMergedConvolutionInit));
#endif 

        return result;
    }
}
