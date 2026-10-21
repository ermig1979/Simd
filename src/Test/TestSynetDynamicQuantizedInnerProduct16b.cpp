/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
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
#include "Test/TestUtils.h"
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestTensor.h"
#include "Test/TestRandom.h"
#include "Test/TestOptions.h"

#include "Simd/SimdSynetDynamicQuantizedInnerProduct16b.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    typedef Simd::DynamicQuantizedInnerProduct16bParam Param;

    namespace
    {
        struct FuncDQIP16b
        {
            typedef void* (*FuncPtr)(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC,
                SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation, SimdBool avoidOverflow);

            FuncPtr func;
            String desc;

            FuncDQIP16b(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(const Param& p)
            {
                desc = desc + "[" + p.Info() + "]";
            }

            void Call(void* context, const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetDynamicQuantizedInnerProduct16bForward(context, A, B, buf, C);
            }
        };
    }

#define FUNC_DQIP16B(function) \
    FuncDQIP16b(function, std::string(#function))

    static float GetRange(const float* src, size_t size)
    {
        float min = FLT_MAX;
        float max = -FLT_MAX;
        for (size_t i = 0; i < size; ++i)
        {
            float val = src[i];
            min = Simd::Min(val, min);
            max = Simd::Max(val, max);
        }
        return max - min;
    }

    struct Dqip16bParams
    {
        Tensor32f a, b, scale, bias, params, c1, c2, c3;
        Tensor16u a16, b16, c1b, c2b;
        Tensor8i b8i;

        bool Init(const Param& p)
        {
            Shape sA = Shp(p.M, p.K), sB = Shp(p.K, p.N), sC = Shp(p.M, p.N);

            a.Reshape(sA);
            FillRandom(a, -1.0f, 1.0f);

            b.Reshape(sB);
            FillRandom(b, -1.0f, 1.0f);

            a16.Reshape(sA);
            b16.Reshape(sB);
            SimdFloat32ToBFloat16(a.Data(), a.Size(), a16.Data());
            SimdFloat32ToBFloat16(b.Data(), b.Size(), b16.Data());

            if (p.typeB == SimdTensorData8i)
            {
                if (!p.constB || !QuantizeB(b, p.overflow, b8i, scale))
                    return false;
            }

            bias.Reshape(Shp(p.N));
            FillRandom(bias, -1.0f, 1.0f);

            params.Reshape(Shp(std::max<size_t>(p.N, 2)));
            FillRandom(params, 0.0f, 1.0f);
            if (p.activation == ::SimdConvolutionActivationHswish)
            {
                params.Data()[0] = 3.0f;
                params.Data()[1] = 1.0f / 6.0f;
            }
            else if (p.activation == ::SimdConvolutionActivationMish)
                params.Data()[0] = 20.0f;
            else
            {
                params.Data()[0] = 0.1f;
                params.Data()[1] = 1.1f;
            }

            c1.Reshape(sC);
            c2.Reshape(sC);
            c3.Reshape(sC);
            c1b.Reshape(sC);
            c2b.Reshape(sC);

            return true;
        }

    private:
        static bool QuantizeB(const Tensor32f& src, SimdBool overflow, Tensor8i& dst, Tensor32f& scale)
        {
            size_t size = src.Size(), N = src.Axis(1), K = size / N;
            dst.Reshape(src.Shape());
            scale.Reshape(Shp(N));
            const float* psrc = src.Data();
            int8_t* pdst = dst.Data();
            int lo = overflow ? -64 : -128, hi = overflow ? 63 : 127;
            float q = overflow ? 63.0f : 127.0f;
            for (size_t j = 0; j < N; ++j)
            {
                float max = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t offset = k * N + j;
                    max = std::max(max, std::abs(psrc[offset]));
                }
                float range = std::max(0.000001f, max);
                float invScale = q / range;
                scale.Data()[j] = range / q;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t offset = k * N + j;
                    pdst[offset] = (int8_t)Simd::RestrictRange((int)std::nearbyint(psrc[offset] * invScale), lo, hi);
                }
            }
            return true;
        }
    };

    bool SynetDynamicQuantizedInnerProduct16bForwardAutoTest(float eps, const Param& p, FuncDQIP16b f1, FuncDQIP16b f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        Srand(0);

        Dqip16bParams dp;
        if (!dp.Init(p))
            return false;

        void* context1 = f1.func(p.M, p.N, p.K, p.typeA, p.typeB, p.typeC, p.constB, p.bias, p.activation, p.overflow);
        void* context2 = f2.func(p.M, p.N, p.K, p.typeA, p.typeB, p.typeC, p.constB, p.bias, p.activation, p.overflow);
        if (context1 == NULL || context2 == NULL)
            return false;

        const uint8_t* weight = NULL;
        const float* scale = NULL;
        if (p.constB)
        {
            if (p.typeB == SimdTensorData8i)
            {
                weight = (const uint8_t*)dp.b8i.Data();
                scale = dp.scale.Data();
            }
            else
                weight = (const uint8_t*)dp.b.Data();
        }
        const float* bias = p.bias ? dp.bias.Data() : NULL;

        ::SimdSynetDynamicQuantizedInnerProduct16bSetParams(context1, weight, scale, bias, dp.params.Data());
        ::SimdSynetDynamicQuantizedInnerProduct16bSetParams(context2, weight, scale, bias, dp.params.Data());

        Tensor8u buf;
        buf.Extend(Shp(::SimdSynetDynamicQuantizedInnerProduct16bExternalBufferSize(context1)));
        buf.Extend(Shp(::SimdSynetDynamicQuantizedInnerProduct16bExternalBufferSize(context2)));

        const uint8_t* A = p.typeA == SimdTensorData32f ? (const uint8_t*)dp.a.Data() : (const uint8_t*)dp.a16.Data();
        const uint8_t* B = NULL;
        if (!p.constB)
            B = p.typeB == SimdTensorData32f ? (const uint8_t*)dp.b.Data() : (const uint8_t*)dp.b16.Data();
        uint8_t* C1 = p.typeC == SimdTensorData32f ? (uint8_t*)dp.c1.Data() : (uint8_t*)dp.c1b.Data();
        uint8_t* C2 = p.typeC == SimdTensorData32f ? (uint8_t*)dp.c2.Data() : (uint8_t*)dp.c2b.Data();

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, A, B, buf.Data(), C1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, A, B, buf.Data(), C2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        if (p.typeC == SimdTensorData16b)
        {
            eps = eps * 7.6f;
            SimdBFloat16ToFloat32(dp.c1b.Data(), dp.c1b.Size(), dp.c1.Data());
            SimdBFloat16ToFloat32(dp.c2b.Data(), dp.c2b.Size(), dp.c2.Data());
        }
        result = result && Compare(dp.c1, dp.c2, eps, true, 64, DifferenceBoth);

        void* context3 = ::SimdSynetInnerProduct32fInit(p.M, p.N, p.K, SimdFalse, SimdTrue, p.bias, p.activation);
        if (context3 == NULL)
            return false;
        ::SimdSynetInnerProduct32fSetParams(context3, dp.b.Data(), NULL, bias, dp.params.Data());
        ::SimdSynetInnerProduct32fForward(context3, dp.a.Data(), NULL, NULL, dp.c3.Data());
        ::SimdRelease(context3);

        float range = GetRange(dp.c3.Data(), dp.c3.Size());
        float control = std::max(0.08f, range * 0.02f);
        result = result && Compare(dp.c1, dp.c3, control, true, 64, DifferenceAbsolute, " Compare to SynetInnerProduct32f.");

        return result;
    }

    bool SynetDynamicQuantizedInnerProduct16bForwardAutoTest(float eps, const FuncDQIP16b& f1, const FuncDQIP16b& f2)
    {
        bool result = true;

        const SimdBool f = SimdFalse, t = SimdTrue;
        const SimdTensorDataType f32 = SimdTensorData32f, b16 = SimdTensorData16b, i8 = SimdTensorData8i;
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;

#ifdef NDEBUG
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(128, 128, 128, f32, f32, f32, t, t, aId, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(127, 129, 131, f32, f32, f32, t, t, aId, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(96, 128, 160, b16, f32, b16, t, t, aRe, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(64, 192, 255, f32, i8, f32, t, f, aPr, f), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(48, 80, 96, f32, f32, b16, f, t, aLr, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(32, 64, 97, b16, b16, f32, f, t, aGe, f), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(40, 72, 88, b16, f32, f32, t, t, aHs, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(24, 36, 48, f32, f32, f32, t, t, aRr, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(24, 40, 50, f32, i8, b16, t, t, aEl, f), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(20, 32, 40, b16, b16, b16, f, f, aHi, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(16, 48, 64, f32, f32, f32, f, t, aSw, f), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(16, 32, 48, b16, f32, f32, t, f, aMi, f), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(1, 128, 256, f32, f32, f32, t, t, aId, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(256, 256, 256, f32, f32, f32, t, t, aId, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(128, 512, 768, f32, i8, f32, t, t, aId, f), f1, f2);
#else
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(8, 8, 3, f32, f32, f32, t, t, aId, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(5, 7, 9, b16, f32, b16, t, t, aRe, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(4, 6, 5, f32, i8, f32, t, f, aPr, f), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(3, 5, 4, b16, b16, f32, f, t, aGe, f), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(6, 4, 7, f32, f32, b16, f, t, aHs, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(2, 3, 1, f32, i8, f32, t, t, aLr, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(7, 5, 6, b16, f32, f32, t, f, aMi, f), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(4, 4, 4, f32, f32, f32, t, t, aRr, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(3, 6, 5, f32, i8, b16, t, t, aEl, f), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(5, 3, 8, b16, b16, b16, f, f, aHi, t), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(4, 5, 6, f32, f32, f32, f, t, aSw, f), f1, f2);
#endif

        return result;
    }

    bool SynetDynamicQuantizedInnerProduct16bForwardAutoTest(const Options& options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(EPS, FUNC_DQIP16B(Simd::Base::SynetDynamicQuantizedInnerProduct16bInit), FUNC_DQIP16B(SimdSynetDynamicQuantizedInnerProduct16bInit));

        return result;
    }
#endif
}
