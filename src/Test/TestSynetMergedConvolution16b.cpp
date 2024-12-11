/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestSynetConvolutionParam.h"
#include "Test/TestTensor.h"
#include "Test/TestString.h"
#include "Test/TestRandom.h"

#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        typedef MergeConvParam Param;

        struct FuncMC
        {
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * params, size_t count, SimdBool add);

            FuncPtr func;
            String desc;

            FuncMC(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Param & p)
            {
                std::stringstream ss;
                ss << desc;
                ss << "[" << p.count << ":" << p.batch << "x" << p.conv[0].srcC << "x" << p.conv[0].srcH << "x" << p.conv[0].srcW;
                for (size_t i = 0; i < p.count; ++i)
                    ss << "-" << (p.conv[i].group != 1 ? String("") : ToString(p.conv[i].dstC) + "x") << p.conv[i].kernelY << "x" << p.conv[i].strideY;
                ss << "-" << (p.conv[0].srcT == SimdTensorData32f ? "f" : "b") << (p.conv[p.count - 1].dstT == SimdTensorData32f ? "f" : "b") << (p.count == 3 ? ToString(p.add) : "") << "]";
                desc = ss.str();
            }

            void Call(void * context, const uint8_t* src, uint8_t * buf, uint8_t* dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetMergedConvolution16bForward(context, src, buf, dst);
            }
        };
    }

#define FUNC_MC(function) \
    FuncMC(function, std::string(#function))

    bool SynetMergedConvolution16bForwardAutoTest(float eps, Param p, FuncMC f1, FuncMC f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        SimdConvolutionParameters & beg = p.conv[0];
        SimdConvolutionParameters & end = p.conv[p.count - 1];

        Shape srcS = Shp(p.batch, beg.srcH, beg.srcW, beg.srcC), dstS = Shp(p.batch, end.dstH, end.dstW, end.dstC);

        Tensor32f weight[3], bias[3], params[3], buf32f;
        Tensor32f src32f(srcS, beg.srcF), dst32f1(dstS, end.dstF), dst32f2(dstS, end.dstF), dst32f3(dstS, end.dstF);
        Tensor16u src16u(srcS, beg.srcF), dst16u1(dstS, end.dstF), dst16u2(dstS, end.dstF);
        Tensor8u buf8u1, buf8u2;

        srand(0);
        FillRandom(src32f.Data(), src32f.Size(), -1.0, 1.0f);
        SimdFloat32ToBFloat16(src32f.Data(), src32f.Size(), src16u.Data());

        size_t weightSize = 0;
        for (size_t i = 0; i < p.count; ++i)
        {
            size_t dc = p.conv[i].dstC;
            weight[i].Reshape(Shp(p.conv[i].kernelY, p.conv[i].kernelX, p.conv[i].srcC / p.conv[i].group, dc));
            FillRandom(weight[i].Data(), weight[i].Size(), -0.500, 0.500f);
            p.weight[i] = weight[i].Data();

            bias[i].Reshape(Shp(dc));
            FillRandom(bias[i].Data(), bias[i].Size(), -1.0, 1.0f);
            p.bias[i] = bias[i].Data();

            params[i].Reshape(Shp(Simd::Max<size_t>(2, dc)));
            FillRandom(params[i].Data(), params[i].Size(), 0.0, 2.0f);
            if (p.conv[i].activation == ::SimdConvolutionActivationHswish)
            {
                params[i].Data()[0] = 3.0f;
                params[i].Data()[1] = 1.0f / 6.0f;
            }
            else if (p.conv[i].activation == ::SimdConvolutionActivationMish)
                params[i].Data()[0] = 20.0f;
            else if (p.conv[i].activation == ::SimdConvolutionActivationHardSigmoid)
            {
                params[i].Data()[0] = 1.0f / 6.0f;
                params[i].Data()[1] = 0.5f;
            }
            else
            {
                params[i].Data()[0] = 0.0f + 0.1f * float(i);
                params[i].Data()[1] = 1.0f + 0.1f * float(i);
            }
            p.params[i] = params[i].Data();
            weightSize += weight[i].Size();
        }

        Fill(dst32f1, 1.1f);
        Fill(dst32f2, 2.2f);
        SimdFloat32ToBFloat16(dst32f1.Data(), dst32f1.Size(), dst16u1.Data());
        SimdFloat32ToBFloat16(dst32f2.Data(), dst32f2.Size(), dst16u2.Data());

        const uint8_t* src = beg.srcT == SimdTensorData32f ? (uint8_t*)src32f.Data() : (uint8_t*)src16u.Data();
        uint8_t* dst1 = end.dstT == SimdTensorData32f ? (uint8_t*)dst32f1.Data() : (uint8_t*)dst16u1.Data();
        uint8_t* dst2 = end.dstT == SimdTensorData32f ? (uint8_t*)dst32f2.Data() : (uint8_t*)dst16u2.Data();

        void* context1 = f1.func(p.batch, p.conv, p.count, p.add);
        void* context2 = f2.func(p.batch, p.conv, p.count, p.add);

        if (context1 == NULL || context2 == NULL)
            return result;

        buf8u1.Extend({ ::SimdSynetMergedConvolution16bExternalBufferSize(context1) });
        buf8u2.Extend({ ::SimdSynetMergedConvolution16bExternalBufferSize(context2) });
        Fill(buf8u1, uint8_t(1));
        Fill(buf8u2, uint8_t(2));

        ::SimdSynetMergedConvolution16bSetParams(context1, p.weight, p.bias, p.params);
        ::SimdSynetMergedConvolution16bSetParams(context2, p.weight, p.bias, p.params);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf8u1.Data(), dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf8u2.Data(), dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        if (end.dstT == SimdTensorData16b)
            eps = eps * 9.0f;
        else if(weightSize > 1024 * 1024)
            eps = eps * 5.3f;
        else
            eps = eps * 2.0f;

        if (end.dstT == SimdTensorData16b)
        {
            SimdBFloat16ToFloat32(dst16u1.Data(), dst16u1.Size(), dst32f1.Data());
            SimdBFloat16ToFloat32(dst16u2.Data(), dst16u2.Size(), dst32f2.Data());
        }
        result = result && Compare(dst32f1, dst32f2, eps, true, 64, DifferenceBoth);

        if (1)
        {
            beg.srcT = SimdTensorData32f;
            end.dstT = SimdTensorData32f;

            void* context3 = SimdSynetMergedConvolution32fInit(p.batch, p.conv, p.count, p.add);

            Tensor32f dst32f3(dstS, end.dstF);
            Fill(dst32f3, 3.3f);
            Tensor32f buf32f(Shp(::SimdSynetMergedConvolution32fExternalBufferSize(context3)));

            ::SimdSynetMergedConvolution32fSetParams(context3, p.weight, NULL, p.bias, p.params);

            ::SimdSynetMergedConvolution32fForward(context3, src32f.Data(), buf32f.Data(), dst32f3.Data());

            ::SimdRelease(context3);

            result = result && Compare(dst32f1, dst32f3, 0.077, true, 64, DifferenceBoth, " Compare to SynetMergedConvolution32f.");
        }

        return result;
    }

    bool SynetMergedConvolution16bForwardAutoTest(float eps, const FuncMC & f1, const FuncMC & f2)
    {
        bool result = true;
        const SimdBool t = SimdTrue, f = SimdFalse;
        const SimdTensorDataType f32 = SimdTensorData32f, b16 = SimdTensorData16b;
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;
        const SimdConvolutionActivationType a0 = aSw, a1 = aSw, a2 = aSw;
#if defined(NDEBUG)
#if 0
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 555, 40, 23), Cnv(a1, 1, 1, 256), Cnv(a0, 3, 1), f32, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 555, 40, 23), Cnv(a1, 1, 1, 256), Cnv(a0, 3, 1), b16, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 555, 40, 23), Cnv(a0, 3, 2), Cnv(a1, 1, 1, 1555), f32, f32), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 1024, 8, 6), Cnv(a0, 1, 1, 1548), Cnv(a1, 3, 1), f32, f32), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 256, 10, 6), Cnv(a0, 1, 1, 64), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 256), f, f32, f32), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 64, 40, 23), Cnv(a0, 3, 2), Cnv(a1, 1, 1, 128), b16, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 1024, 8, 6), Cnv(a0, 1, 1, 1548), Cnv(a1, 3, 1), b16, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 256, 10, 6), Cnv(a0, 1, 1, 64), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 256), f, b16, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 1024, 6, 6), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 1024), f32, f32), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 512, 6, 6), Cnv(a0, 1, 1, 1024), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 512), f, f32, f32), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 64, 19, 19), Cnv(a0, 1, 1, 128), Cnv(a1, 3, 1), b16, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 224, 2, 3), Cnv(a0, 1, 1, 64), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 128), f, f32, f32), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 116, 15, 5), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 116), f32, f32), f1, f2);
#endif
#if 0
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 192, 40, 40), Cnv(a2, 1, 1, 192), Cnv(a1, 5, 1), b16, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 256, 16, 16), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 256), f32, f32), f1, f2);
        //result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 256, 16, 16), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 256), b16, b16), f1, f2);
        //result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 304, 17, 15), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 1216), b16, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 76, 64, 64), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 304), f32, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 152, 32, 32), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 608), f32, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 304, 16, 16), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 1216), f32, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 608, 8, 8), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 2432), f32, b16), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 64, 28, 24), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 192), f32, f32), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 128, 14, 12), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 386), f32, f32), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 256, 7, 6), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 768), f32, f32), f1, f2);
#endif
#if 0
        {
            Param p(Shp(1, 64, 20, 60), Cnv(a0, 1, 1, 128), Cnv(a1, 3, 1), f, b16, b16);
            p.conv[1].strideY = 2;
            Param::SetDst(p.conv + 1);
            result = result && SynetMergedConvolution16bForwardAutoTest(eps, p, f1, f2);
        }
#endif
#if 0
        {
            Param p(Shp(1, 116, 15, 5), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 116), f, f32, f32);
            p.conv[0].padX = 1; p.conv[0].padY = 1;
            Param::SetDst(p.conv + 0); Param::SetDst(p.conv + 1);
            result = result && SynetMergedConvolution16bForwardAutoTest(eps, p, f1, f2);
        }
#endif
#if 1
        {
            Param p(Shp(1, 68, 56, 56), Cnv(aHs, 1, 1, 84), Cnv(aId, 5, 2), Cnv(aHs, 1, 1, 100), f, f32, f32);
            p.conv[1].padX = 2; p.conv[1].padY = 2;
            Param::SetDst(p.conv + 1); Param::SetDst(p.conv + 2);
            result = result && SynetMergedConvolution16bForwardAutoTest(eps, p, f1, f2);
        }
#endif
#if 0
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 68, 56, 56), Cnv(aHs, 1, 1, 84), Cnv(aId, 5, 2), Cnv(aHs, 1, 1, 100), f, f32, f32), f1, f2);
        //result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(2, 128, 15, 15), Cnv(a0, 1, 1, 256), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 128), f, f32, f32), f1, f2);
        //result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(2, 128, 15, 15), Cnv(a0, 1, 1, 256), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 128), t, f32, f32), f1, f2);
        //result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(2, 128, 15, 15), Cnv(a0, 1, 1, 256), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 128), t, b16, f32), f1, f2);
        //result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(2, 128, 15, 15), Cnv(a0, 1, 1, 256), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 128), t, f32, b16), f1, f2);
        //result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(2, 128, 15, 15), Cnv(a0, 1, 1, 256), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 128), t, b16, b16), f1, f2);
        //result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(2, 512, 127, 127), Cnv(a0, 1, 1, 1024), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 512), t, f32, f32), f1, f2);
#endif
#else
        {
            Param p(Shp(1, 68, 56, 56), Cnv(aHs, 1, 1, 84), Cnv(aId, 5, 2), Cnv(aHs, 1, 1, 100), f, f32, f32);
            p.conv[1].padX = 2; p.conv[1].padY = 2;
            Param::SetDst(p.conv + 1); Param::SetDst(p.conv + 2);
            result = result && SynetMergedConvolution16bForwardAutoTest(eps, p, f1, f2);
        }
#endif
        return result;
    }

    bool SynetMergedConvolution16bForwardAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && SynetMergedConvolution16bForwardAutoTest(EPS, FUNC_MC(Simd::Base::SynetMergedConvolution16bInit), FUNC_MC(SimdSynetMergedConvolution16bInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetMergedConvolution16bForwardAutoTest(EPS, FUNC_MC(Simd::Sse41::SynetMergedConvolution16bInit), FUNC_MC(SimdSynetMergedConvolution16bInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetMergedConvolution16bForwardAutoTest(EPS, FUNC_MC(Simd::Avx2::SynetMergedConvolution16bInit), FUNC_MC(SimdSynetMergedConvolution16bInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetMergedConvolution16bForwardAutoTest(EPS, FUNC_MC(Simd::Avx512bw::SynetMergedConvolution16bInit), FUNC_MC(SimdSynetMergedConvolution16bInit));
#endif 

#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
        if (Simd::AmxBf16::Enable && TestAmxBf16())
            result = result && SynetMergedConvolution16bForwardAutoTest(EPS, FUNC_MC(Simd::AmxBf16::SynetMergedConvolution16bInit), FUNC_MC(SimdSynetMergedConvolution16bInit));
#endif

        return result;
    }
#endif
}
