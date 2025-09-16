/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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

#include "Simd/SimdSynetQuantizedInnerProduct.h"

#include "Simd/SimdMath.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncQIP
        {
            typedef void*(*FuncPtr)(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC, SimdBool transB, SimdBool constB, SimdBool bias);

            FuncPtr func;
            String desc;

            FuncQIP(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Simd::QuantizedInnerProductParam& p)
            {
                desc = desc + "[" + p.Info() + "]";
            }

            void Call(void * context, const uint8_t *A, const uint8_t* B, uint8_t * buf, uint8_t * C) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetQuantizedInnerProductForward(context, A, B, buf, C);
            }
        };
    }

#define FUNC_QIP(function) \
    FuncQIP(function, std::string(#function))

    struct QipParams32f
    {
        Tensor32f a, b, bias, c, c1, c2;

        bool Init(const Simd::QuantizedInnerProductParam & p)
        {
            Shape sA = Shp(p.M, p.K), sB = p.transB ? Shp(p.N, p.K) : Shp(p.K, p.N), sC = Shp(p.M, p.N);

            a.Reshape(sA);
            FillRandom(a, -0.9, 1.1f);

            b.Reshape(sB);
            FillRandom(b, -1.1, 1.0f);

            bias.Reshape(Shp(p.N));
            FillRandom(bias, -1.1, 1.2f);

            c.Reshape(sC);

            void* context = ::SimdSynetInnerProduct32fInit(p.M, p.K, p.N, p.transB, SimdConvolutionActivationIdentity);
            if (context == NULL)
                return false;

            ::SimdSynetInnerProduct32fSetParams(context, b.Data(), NULL, bias.Data(), NULL);

            ::SimdSynetInnerProduct32fForward(context, a.Data(), c.Data());

            ::SimdRelease(context);

            if (p.typeC == SimdTensorData32f)
            {
                c1.Reshape(sC);
                c2.Reshape(sC);
            }

            return true;
        }
    };

    struct QipParams8i
    {
        Tensor8u a, c, c1, c2;
        Tensor8i b;
        Tensor32i bias;
        uint8_t aZero, cZero;
        float aScale, cScale;
        Tensor32f bScale;

        bool Init(const Simd::QuantizedInnerProductParam& p, const QipParams32f & f32, SimdBool overflow)
        {
            if (!QuantizeAC(f32.a, a, aZero, aScale))
                return false;

            if (!QuantizeAC(f32.c, c, cZero, cScale))
                return false;

            if (!QuantizeB(f32.b, p.transB, overflow, b, bScale))
                return false;

            if (!QuantizeBias(f32.bias, aScale, bScale, bias))
                return false;

            if (p.typeC == SimdTensorData8u)
            {
                c1.Reshape(c.Shape());
                c2.Reshape(c.Shape());
            }

            return true;
        }

    protected:
        static bool QuantizeAC(const Tensor32f& src, Tensor8u& dst, uint8_t& zero, float& scale)
        {
            size_t batch = src.Axis(0), size = src.Size();
            size_t channels = src.Axis(1);
            dst.Reshape(src.Shape());
            float min = 0.0f, max = 0.0f;
            const float* psrc = src.Data();
            for (size_t i = 0; i < size; ++i)
            {
                min = std::min(min, psrc[i]);
                max = std::max(max, psrc[i]);
            }
            float range = std::max(0.000001f, max - min), invScale = 255.0f / range;
            scale = range / 255.0f;
            zero = -(int)std::nearbyint(min * invScale);
            uint8_t* pdst = dst.Data();
            for (size_t i = 0; i < size; ++i)
                pdst[i] = Simd::RestrictRange((int)std::nearbyint(psrc[i] * invScale) + zero, 0, 255);
            return true;
        }

        static bool QuantizeB(const Tensor32f& src, bool trans, SimdBool overflow, Tensor8i& dst, Tensor32f& scale)
        {
            size_t size = src.Size(), N = trans ? src.Axis(0) : src.Axis(1), K = size / N;
            dst.Reshape(src.Shape());
            scale.Reshape(Shp(N));
            const float* psrc = src.Data();
            int8_t* pdst = dst.Data();
            int lo = overflow ? -64 : -128, hi = overflow ? 63 : 127;
            for (size_t j = 0; j < N; ++j)
            {
                float max = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t offset = trans ? j * K + k : k * N + j;
                    max = std::max(max, std::abs(psrc[offset]));
                }
                float range = std::max(0.000001f, max);
                float _scale =  range / (overflow ? 63.0f : 127.0f), invScale = (overflow ? 63.0f : 127.0f) / range;
                scale.Data()[j] = _scale;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t offset = trans ? j * K + k : k * N + j;
                    pdst[offset] = Simd::RestrictRange((int)std::nearbyint(psrc[offset] * invScale), lo, hi);
                }
            }
            return true;
        }

        static bool QuantizeBias(const Tensor32f& src, float aScale, const Tensor32f& bScale, Tensor32i& dst)
        {
            size_t size = src.Size();
            dst.Reshape(src.Shape());
            const float* psrc = src.Data();
            const float* pbs = bScale.Data();
            int32_t* pdst = dst.Data();
            for (size_t i = 0; i < size; ++i)
            {
                float invScale = 1.0f / (aScale * pbs[i]);
                pdst[i] = (int)std::nearbyint(psrc[i] * invScale);
            }
            return true;
        }
    };

    bool SynetQuantizedInnerProductForwardAutoTest(float eps, const Simd::QuantizedInnerProductParam &p, SimdBool overflow, FuncQIP f1, FuncQIP f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        if (p.M == 1)
            overflow = SimdTrue;

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        QipParams32f p32f;
        if (!p32f.Init(p))
            return false;

        QipParams8i p8i;
        if (!p8i.Init(p, p32f, overflow))
            return false;

        void * context1 = f1.func(p.M, p.N, p.K, p.typeA, p.typeB, p.typeC, p.transB, p.constB, p.bias);
        void * context2 = f2.func(p.M, p.N, p.K, p.typeA, p.typeB, p.typeC, p.transB, p.constB, p.bias);
        if (context1 == NULL)
            return true;

        Tensor8u buf8u;
        buf8u.Extend({ ::SimdSynetQuantizedInnerProductExternalBufferSize(context1) });
        buf8u.Extend({ ::SimdSynetQuantizedInnerProductExternalBufferSize(context2) });

        ::SimdSynetQuantizedInnerProductSetParams(context1, &p8i.aScale, &p8i.aZero, p8i.b.Data(), p8i.bScale.Data(), p8i.bias.Data(), &p8i.cScale, &p8i.cZero);
        ::SimdSynetQuantizedInnerProductSetParams(context2, &p8i.aScale, &p8i.aZero, p8i.b.Data(), p8i.bScale.Data(), p8i.bias.Data(), &p8i.cScale, &p8i.cZero);

        const uint8_t* a = p.typeA == SimdTensorData32f ? (uint8_t*)p32f.a.Data() : p8i.a.Data();
        const uint8_t* b = p.typeB == SimdTensorData32f ? (uint8_t*)p32f.b.Data() : (uint8_t*)p8i.b.Data();
        uint8_t* c1 = p.typeC == SimdTensorData32f ? (uint8_t*)p32f.c1.Data() : p8i.c1.Data();
        uint8_t* c2 = p.typeC == SimdTensorData32f ? (uint8_t*)p32f.c2.Data() : p8i.c2.Data();

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, a, b, buf8u.Data(), c1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, a, b, buf8u.Data(), c2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        if (p.typeC == SimdTensorData32f)
        {
            result = result && Compare(p32f.c1, p32f.c2, eps, true, 64, DifferenceBoth);
        }
        else
        {
            int diffMax = 0;
            result = result && Compare(p8i.c1, p8i.c2, diffMax, true, 64);

            int controlDiffMax = 2;
            result = result && Compare(p8i.c1, p8i.c, controlDiffMax, true, 64, "control");
        }

        return result;
    }

    bool SynetQuantizedInnerProductForwardAutoTest(SimdBool o, const FuncQIP& f1, const FuncQIP& f2)
    {
        bool result = true;

        const float e = EPS;
        const SimdBool f = SimdFalse, t = SimdTrue;
        const SimdTensorDataType f32 = SimdTensorData32f, u8 = SimdTensorData8u, i8 = SimdTensorData8i;
        typedef Simd::QuantizedInnerProductParam Param;

#ifdef NDEBUG
#if 0
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(64, 96, 128, u8, i8, u8, f, t, f), o, f1, f2);
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(664, 2696, 2628, u8, i8, u8, f, t, f), o, f1, f2);
#endif
#if 1
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(1, 512, 1000, u8, i8, u8, f, t, f), o, f1, f2);
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(1, 512, 1000, u8, i8, u8, t, t, f), o, f1, f2);
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(1, 1000, 512, u8, i8, u8, f, t, f), o, f1, f2);
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(1, 1000, 512, u8, i8, u8, t, t, f), o, f1, f2);
#endif
#if 0
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(1, 512, 1024, u8, i8, u8, f, t, f), o, f1, f2);
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(2, 512, 1024, u8, i8, u8, f, t, f), o, f1, f2);
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(3, 512, 1024, u8, i8, u8, f, t, f), o, f1, f2);
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(4, 512, 1024, u8, i8, u8, f, t, f), o, f1, f2);
#endif
#if 1
        //result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(333, 443, 555, u8, i8, u8, f, f, f), o, f1, f2);
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(256, 512, 1024, u8, i8, u8, f, t, f), o, f1, f2);
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(333, 443, 555, u8, i8, u8, f, t, f), o, f1, f2);
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(333, 443, 555, u8, i8, u8, t, t, f), o, f1, f2);
#endif
#else
        result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(64, 96, 128, u8, i8, u8, f, t, f), o, f1, f2);
        //result = result && SynetQuantizedInnerProductForwardAutoTest(e, Param(1, 512, 1000, u8, i8, u8, f, t, f), o, f1, f2);
#endif

        return result;
    }

    bool SynetQuantizedInnerProductForwardAutoTest(const Options & options)
    {
        bool result = true;

        const SimdBool f = SimdFalse, t = SimdTrue;

        if (TestBase())
            result = result && SynetQuantizedInnerProductForwardAutoTest(t, FUNC_QIP(Simd::Base::SynetQuantizedInnerProductInit), FUNC_QIP(SimdSynetQuantizedInnerProductInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetQuantizedInnerProductForwardAutoTest(t, FUNC_QIP(Simd::Sse41::SynetQuantizedInnerProductInit), FUNC_QIP(SimdSynetQuantizedInnerProductInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetQuantizedInnerProductForwardAutoTest(t, FUNC_QIP(Simd::Avx2::SynetQuantizedInnerProductInit), FUNC_QIP(SimdSynetQuantizedInnerProductInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetQuantizedInnerProductForwardAutoTest(t, FUNC_QIP(Simd::Avx512bw::SynetQuantizedInnerProductInit), FUNC_QIP(SimdSynetQuantizedInnerProductInit));
#endif

#if defined(SIMD_AVX512VNNI_ENABLE) && !defined(SIMD_AMX_EMULATE)
        if (Simd::Avx512vnni::Enable && TestAvx512vnni())
            result = result && SynetQuantizedInnerProductForwardAutoTest(f, FUNC_QIP(Simd::Avx512vnni::SynetQuantizedInnerProductInit), FUNC_QIP(SimdSynetQuantizedInnerProductInit));
#endif

#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
        if (Simd::AmxBf16::Enable && TestAmxBf16())
            result = result && SynetQuantizedInnerProductForwardAutoTest(f, FUNC_QIP(Simd::AmxBf16::SynetQuantizedInnerProductInit), FUNC_QIP(SimdSynetQuantizedInnerProductInit));
#endif

        return result;
    }
#endif
}
