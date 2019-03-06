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

#include "Simd/SimdConvolution.h"

namespace Test
{
    namespace
    {
        struct Param
        {
            size_t srcC, srcH, srcW, dstC, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group;
            SimdBool srcT, dstT;
            ::SimdConvolutionActivationType activation;

            Param(size_t sC, size_t sH, size_t sW, SimdBool sT, size_t dC, SimdBool dT, size_t kY, size_t kX, size_t dY, size_t dX, 
                size_t sY, size_t sX, size_t pY, size_t pX, size_t pH, size_t pW, size_t g, ::SimdConvolutionActivationType a)
                : srcC(sC), srcH(sH), srcW(sW), srcT(sT), dstC(dC), dstT(dT), kernelY(kY), kernelX(kX), dilationY(dY), dilationX(dX), 
                strideY(sY), strideX(sX), padY(pY), padX(pX), padH(pH), padW(pW), group(g), activation(a)
            {}

            Param(size_t sC, size_t sH, size_t sW, size_t dC, Size k, Size d, Size s, Size b, Size e, size_t g, ::SimdConvolutionActivationType a, ::SimdBool t)
                : srcC(sC), srcH(sH), srcW(sW), srcT(t), dstC(dC), dstT(t), kernelY(k.y), kernelX(k.x), dilationY(d.y), dilationX(d.x), 
                strideY(s.y), strideX(s.x), padY(b.y), padX(b.x), padH(e.y), padW(e.x), group(g), activation(a)
            {}
        };

        struct FuncC
        {
            typedef void*(*FuncPtr)(size_t srcC, size_t srcH, size_t srcW, SimdBool srcT, size_t dstC, SimdBool dstT, 
                size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, 
                size_t padY, size_t padX, size_t padH, size_t padW, size_t group, SimdConvolutionActivationType activation, SimdGemm32fNNPtr gemm);

            FuncPtr func;
            String description;

            FuncC(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(const Param & p)
            {
                std::stringstream ss;
                ss << description;
                ss << "[" << p.srcC << "x" << p.srcH << "x" << p.srcW;
                ss << "-" << p.dstC << "x" << p.kernelY << "x" << p.kernelX;
                ss << "-" << p.strideX << "-" << Simd::Max(p.padX, p.padW) << "-" << p.group << "-" << p.srcT;
                ss << "]";
                description = ss.str();
            }

            void Call(const Param & p, const Tensor32f & weight, const Tensor32f & bias, const Tensor32f & params, const Tensor32f & src, Tensor32f & buf, Tensor32f & dst) const
            {
                void * convolution = func(p.srcC, p.srcH, p.srcW, p.srcT, p.dstC, p.dstT, p.kernelY, p.kernelX, 
                    p.dilationY, p.dilationX, p.strideY, p.strideX, p.padY, p.padX, p.padH, p.padW, p.group, p.activation, NULL);
                buf.Extend({ ::SimdConvolutionBufferSize(convolution) });
                ::SimdConvolutionSetParams(convolution, weight.Data(), p.srcT, NULL, bias.Data(), params.Data());
                {
                    TEST_PERFORMANCE_TEST(description);
                    ::SimdConvolutionForward(convolution, src.Data(), buf.Data(), dst.Data());
                }
                ::SimdRelease(convolution);
            }
        };
    }

#define FUNC_C(function) \
    FuncC(function, std::string(#function))

    bool ConvolutionForwardAutoTest(const Param & p, FuncC f1, FuncC f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << "].");

        Tensor32f src({p.srcT ? p.srcH : p.srcC, p.srcT ? p.srcW : p.srcH, p.srcT ? p.srcC : p.srcW });
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f weight({ p.srcT ? p.kernelY : p.dstC, p.srcT ? p.kernelX : p.srcC / p.group, 
            p.srcT ? p.srcC / p.group : p.kernelY, p.srcT ? p.dstC : p.kernelX });
        FillRandom(weight.Data(), weight.Size(), -1.0, 1.0f);

        Tensor32f bias({ p.dstC });
        FillRandom(bias.Data(), bias.Size(), -1.0, 1.0f);

        Tensor32f params({ p.dstC });
        FillRandom(params.Data(), params.Size(), 0.0f, 2.0f);

        params.Data()[0] = 0.1f;
        params.Data()[1] = 1.1f;

        Tensor32f buf;

        size_t dstH = (p.srcH + p.padY + p.padH - (p.dilationY * (p.kernelY - 1) + 1)) / p.strideY + 1;
        size_t dstW = (p.srcW + p.padX + p.padW - (p.dilationX * (p.kernelX - 1) + 1)) / p.strideX + 1;
        Tensor32f dst1({ p.dstT ? dstH : p.dstC, p.dstT ? dstW : dstH, p.dstT ? p.dstC : dstW });
        Tensor32f dst2({ p.dstT ? dstH : p.dstC, p.dstT ? dstW : dstH, p.dstT ? p.dstC : dstW });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(p, weight, bias, params, src, buf, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(p, weight, bias, params, src, buf, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool ConvolutionForwardAutoTest(::SimdConvolutionActivationType a, ::SimdBool t, const FuncC & f1, const FuncC & f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _7(7, 7);

#ifdef NDEBUG
#if 0
        result = result && ConvolutionForwardAutoTest(Param(16, 112, 96, 32, _3, _1, _3, Size(1, 0), Size(1, 0), 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(16, 114, 96, 32, _3, _1, _3, _0, _0, 1, a, t), f1, f2);
        //result = result && ConvolutionForwardAutoTest(Param(64, 112, 96, 64, _3, _1, _3, Size(1, 0), Size(1, 0), 64, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 19, 16, 64, _3, _1, _3, _1, _1, 64, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(32, 19, 16, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 7, 7, 128, _7, _1, _1, _0, _0, 128, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(16, 56, 56, 32, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 112, 112, 16, _3, _1, _2, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 7, 6, 128, _3, _1, _2, Size(0, 1), Size(1, 1), 128, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 4, 3, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(32, 38, 32, 32, _3, _1, _2, _0, _1, 32, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(3, 112, 96, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(16, 112, 96, 32, _3, _1, _3, Size(1, 0), Size(1, 0), 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(32, 38, 32, 32, _3, _1, _2, _0, _1, 32, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(32, 19, 16, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 19, 16, 64, _3, _1, _3, _1, _1, 64, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 7, 6, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 7, 6, 128, _3, _1, _2, Size(0, 1), Size(1, 1), 128, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 4, 3, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(1024, 13, 13, 1024, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(512, 10, 10, 1024, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 10, 10, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(384, 20, 20, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(16, 150, 150, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(17, 150, 150, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(16, 150, 150, 96, _2, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(17, 150, 150, 96, _2, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(16, 150, 150, 96, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(17, 150, 150, 96, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _1, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _1, _1, _2, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _2, _1, _1, _1, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _2, _1, _2, _1, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _3, _1, _2, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _4, _1, _1, _2, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _4, _1, _2, _2, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _5, _1, _2, _2, _2, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _7, _1, _1, _3, _3, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _7, _1, _2, _3, _3, 1, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(32, 150, 150, 16, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(16, 150, 150, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(96, 75, 75, 24, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(24, 75, 75, 144, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(144, 75, 75, 24, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(144, 38, 38, 32, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(32, 38, 38, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(192, 38, 38, 32, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(192, 19, 19, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 19, 19, 384, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(384, 19, 19, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(384, 19, 19, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(96, 19, 19, 576, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(576, 19, 19, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(576, 10, 10, 160, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(160, 10, 10, 960, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(960, 10, 10, 160, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(960, 10, 10, 320, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(320, 10, 10, 1280, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(1280, 10, 10, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 5, 5, 512, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(512, 5, 5, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 3, 3, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 3, 3, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 2, 2, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 2, 2, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 2, 2, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 1, 1, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);

        result = result && ConvolutionForwardAutoTest(Param(576, 19, 19, 12, _3, _1, _1, _1, _1, 1, a, t), f1, f2);  
        result = result && ConvolutionForwardAutoTest(Param(1280, 10, 10, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(512, 5, 5, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 3, 3, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 2, 2, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 1, 1, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        
        result = result && ConvolutionForwardAutoTest(Param(3, 300, 300, 32, _3, _1, _2, _0, _1, 1, a, t), f1, f2);

        result = result && ConvolutionForwardAutoTest(Param(32, 150, 150, 32, _3, _1, _1, _1, _1, 32, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(144, 75, 75, 144, _3, _1, _1, _1, _1, 144, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(192, 38, 38, 192, _3, _1, _1, _1, _1, 192, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(384, 19, 19, 384, _3, _1, _1, _1, _1, 384, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(576, 19, 19, 576, _3, _1, _1, _1, _1, 576, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(960, 10, 10, 960, _3, _1, _1, _1, _1, 960, a, t), f1, f2);

        result = result && ConvolutionForwardAutoTest(Param(96, 150, 150, 96, _3, _1, _2, _0, _1, 96, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(144, 75, 75, 144, _3, _1, _2, _1, _1, 144, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(192, 38, 38, 192, _3, _1, _2, _0, _1, 192, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(576, 19, 19, 576, _3, _1, _2, _1, _1, 576, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 10, 10, 256, _3, _1, _2, _0, _1, 256, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 5, 5, 128, _3, _1, _2, _1, _1, 128, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 3, 3, 128, _3, _1, _2, _1, _1, 128, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 2, 2, 64, _3, _1, _2, _0, _1, 64, a, t), f1, f2);
#endif
#if 1
        result = result && ConvolutionForwardAutoTest(Param(48, 256, 256, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(96, 128, 128, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(192, 64, 64, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(384, 32, 32, 384, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(768, 16, 16, 768, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(1536, 8, 8, 1536, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3072, 4, 4, 3072, _1, _1, _1, _0, _0, 1, a, t), f1, f2);//slow

        result = result && ConvolutionForwardAutoTest(Param(16, 256, 256, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);//slow
        result = result && ConvolutionForwardAutoTest(Param(32, 128, 128, 32, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 64, 64, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 32, 32, 128, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 16, 16, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(512, 8, 8, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(1024, 4, 4, 1024, _3, _1, _1, _1, _1, 1, a, t), f1, f2);

        result = result && ConvolutionForwardAutoTest(Param(10, 256, 256, 10, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(20, 128, 128, 20, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(40, 64, 64, 40, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(80, 32, 32, 80, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(160, 16, 16, 160, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(320, 8, 8, 320, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(640, 4, 4, 640, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(3, 300, 300, 32, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 300, 300, 16, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 112, 112, 16, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 180, 320, 10, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(10, 89, 159, 16, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(16, 87, 157, 32, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 224, 224, 16, _5, _1, _2, _2, _2, 1, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(32, 150, 150, 32, _3, _1, _1, _1, _1, 32, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(144, 75, 75, 144, _3, _1, _1, _1, _1, 144, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(192, 38, 38, 192, _3, _1, _1, _1, _1, 192, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(384, 19, 19, 384, _3, _1, _1, _1, _1, 384, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(576, 19, 19, 576, _3, _1, _1, _1, _1, 576, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(960, 10, 10, 960, _3, _1, _1, _1, _1, 960, a, t), f1, f2);

        result = result && ConvolutionForwardAutoTest(Param(96, 150, 150, 96, _3, _1, _2, _0, _1, 96, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(144, 75, 75, 144, _3, _1, _2, _1, _1, 144, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(192, 38, 38, 192, _3, _1, _2, _0, _1, 192, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(576, 19, 19, 576, _3, _1, _2, _1, _1, 576, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 10, 10, 256, _3, _1, _2, _0, _1, 256, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 5, 5, 128, _3, _1, _2, _1, _1, 128, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 3, 3, 128, _3, _1, _2, _1, _1, 128, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 2, 2, 64, _3, _1, _2, _0, _1, 64, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(32, 38, 32, 32, _3, _1, _2, _0, _1, 32, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 19, 16, 64, _3, _1, _3, _1, _1, 64, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(728, 14, 14, 728, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 56, 48, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 28, 24, 128, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 14, 12, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(512, 7, 6, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(3, 300, 300, 32, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(32, 150, 150, 32, _3, _1, _1, _1, _1, 32, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(32, 150, 150, 16, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(16, 150, 150, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(96, 150, 150, 96, _3, _1, _2, _0, _1, 96, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(96, 75, 75, 24, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(3, 112, 96, 64, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 56, 48, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(64, 56, 48, 128, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 28, 24, 128, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(128, 28, 24, 256, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 14, 12, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(256, 14, 12, 512, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(512, 7, 6, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(3, 1024, 1024, 24, _7, _1, _4, _3, _3, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(48, 128, 128, 64, _5, _1, _2, _2, _2, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(116, 8, 8, 116, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(16, 160, 160, 16, _3, _1, _1, _1, _1, 16, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(48, 160, 160, 48, _3, _1, _2, _1, _0, 48, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(48, 80, 80, 48, _3, _1, _2, _1, _0, 48, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(48, 80, 80, 48, _3, _1, _1, _1, _1, 48, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(144, 20, 20, 144, _3, _1, _1, _1, _1, 144, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(192, 20, 20, 192, _3, _1, _1, _1, _1, 192, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(96, 40, 40, 96, _3, _1, _1, _1, _1, 96, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(96, 40, 40, 96, _3, _1, _2, _1, _0, 96, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(16, 160, 160, 8, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(8, 160, 160, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(16, 16, 160, 8, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(8, 16, 160, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(8, 80, 80, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(48, 80, 80, 8, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(8, 80, 80, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && ConvolutionForwardAutoTest(Param(32, 115, 63, 4, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && ConvolutionForwardAutoTest(Param(32, 115, 63, 2, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#else
        result = result && ConvolutionForwardAutoTest(Param(32, 115, 63, 4, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
        return result;
    }

    bool ConvolutionForwardAutoTest(const FuncC & f1, const FuncC & f2)
    {
        bool result = true;

        result = result && ConvolutionForwardAutoTest(::SimdConvolutionActivationPrelu, ::SimdFalse, f1, f2);
        result = result && ConvolutionForwardAutoTest(::SimdConvolutionActivationPrelu, ::SimdTrue, f1, f2);

        return result;
    }

    bool ConvolutionForwardAutoTest()
    {
        bool result = true;

        result = result && ConvolutionForwardAutoTest(FUNC_C(Simd::Base::ConvolutionInit), FUNC_C(SimdConvolutionInit));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && ConvolutionForwardAutoTest(FUNC_C(Simd::Sse::ConvolutionInit), FUNC_C(SimdConvolutionInit));
#endif 

#ifdef SIMD_SSE3_ENABLE
        if (Simd::Sse3::Enable)
            result = result && ConvolutionForwardAutoTest(FUNC_C(Simd::Sse3::ConvolutionInit), FUNC_C(SimdConvolutionInit));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && ConvolutionForwardAutoTest(FUNC_C(Simd::Avx::ConvolutionInit), FUNC_C(SimdConvolutionInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && ConvolutionForwardAutoTest(FUNC_C(Simd::Avx2::ConvolutionInit), FUNC_C(SimdConvolutionInit));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && ConvolutionForwardAutoTest(FUNC_C(Simd::Avx512f::ConvolutionInit), FUNC_C(SimdConvolutionInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && ConvolutionForwardAutoTest(FUNC_C(Simd::Neon::ConvolutionInit), FUNC_C(SimdConvolutionInit));
#endif

        return result;
    }
}
