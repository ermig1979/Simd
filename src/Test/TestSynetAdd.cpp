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
#include "Test/TestString.h"
#include "Test/TestRandom.h"
#include "Test/TestOptions.h"

#include "Simd/SimdSynet.h"
#include "Simd/SimdSynetAdd16b.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncAB
        {
            typedef void(*FuncPtr)(const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncAB(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f & bias, size_t channels, size_t spatial, SimdTensorFormatType format, const Tensor32f & dstSrc, Tensor32f & dstDst) const
            {
                Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(desc);
                func(bias.Data(), channels, spatial, dstDst.Data(), format);
            }
        };
    }

#define FUNC_AB(function) FuncAB(function, #function)

    bool SynetAddBiasAutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncAB f1, FuncAB f2)
    {
        bool result = true;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f bias( ToShape(channels, format) );
        Tensor32f dstSrc(ToShape(channels, spatial, format));
        Tensor32f dstDst1(ToShape(channels, spatial, format));
        Tensor32f dstDst2(ToShape(channels, spatial, format));

        TEST_ALIGN(SIMD_ALIGN);

        FillRandom(bias.Data(), bias.Size(), -10.0, 10.0);
        FillRandom(dstSrc.Data(), dstSrc.Size(), -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(bias, channels, spatial, format, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(bias, channels, spatial, format, dstSrc, dstDst2));

        result = result && Compare(dstDst1, dstDst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetAddBiasAutoTest(const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNhwc && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            result = result && SynetAddBiasAutoTest(H, W, format, f1, f2);
            result = result && SynetAddBiasAutoTest(H - O, W + O, format, f1, f2);
        }

        return result;
    }

    bool SynetAddBiasAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Base::SynetAddBias), FUNC_AB(SimdSynetAddBias));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Sse41::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Avx2::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Avx512bw::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Neon::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif

#ifdef SIMD_SVE2_ENABLE
        if (Simd::Sve2::Enable && TestSve2(options))
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Sve2::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncA16b
        {
            typedef void* (*FuncPtr)(const size_t* aShape, size_t aCount, SimdTensorDataType aType, 
                const size_t* bShape, size_t bCount, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncA16b(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(const Shape& as, SimdTensorDataType at, const Shape& bs, SimdTensorDataType bt, SimdTensorDataType dt, SimdTensorFormatType f)
            {
                desc = desc + "[" + ToString(as) + "+" + ToString(bs) + "-" + 
                    ToChar(at) + ToChar(bt) + ToChar(dt) +  (f == SimdTensorFormatNhwc ? "-1" : "-0") + "]";
            }

            void Call(void* context, const uint8_t* A, const uint8_t* B, uint8_t*dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetAdd16bForward(context, A, B, dst);
            }
        };
    }

#define FUNC_A16B(function) FuncA16b(function, #function)

    bool SynetAdd16bAutoTest(const Shape & aShape, SimdTensorDataType aType, const Shape& bShape, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format, FuncA16b f1, FuncA16b f2)
    {
        bool result = true;

        f1.Update(aShape, aType, bShape, bType, dstType, format);
        f2.Update(aShape, aType, bShape, bType, dstType, format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        Tensor32f Af(aShape), Bf(bShape), dst1f(aShape), dst2f(aShape);
        Tensor16u Ab(aShape), Bb(bShape), dst1b(aShape), dst2b(aShape);

        FillRandom(Af.Data(), Af.Size(), -1.0, 1.0f);
        FillRandom(Bf.Data(), Bf.Size(), -1.0, 1.0f);

        SimdFloat32ToBFloat16(Af.Data(), Af.Size(), Ab.Data());
        SimdFloat32ToBFloat16(Bf.Data(), Bf.Size(), Bb.Data());

        Fill(dst1f, 1.0f);
        Fill(dst1f, 2.0f);

        Fill(dst1b.Data(), dst1b.Size(), uint16_t(1));
        Fill(dst2b.Data(), dst2b.Size(), uint16_t(2));


        const uint8_t* A = aType == SimdTensorData32f ? (uint8_t*)Af.Data() : (uint8_t*)Ab.Data();
        const uint8_t* B = bType == SimdTensorData32f ? (uint8_t*)Bf.Data() : (uint8_t*)Bb.Data();
        uint8_t* dst1 = dstType == SimdTensorData32f ? (uint8_t*)dst1f.Data() : (uint8_t*)dst1b.Data();
        uint8_t* dst2 = dstType == SimdTensorData32f ? (uint8_t*)dst2f.Data() : (uint8_t*)dst2b.Data();

        void* context1 = f1.func(aShape.data(), aShape.size(), aType, bShape.data(), bShape.size(), bType, dstType, format);
        void* context2 = f2.func(aShape.data(), aShape.size(), aType, bShape.data(), bShape.size(), bType, dstType, format);

        if (context1 == NULL)
            return true;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, A, B, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, A, B, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        float eps = EPS;
        if (dstType == SimdTensorData16b)
        {
            //eps = eps * 7.1f;
            SimdBFloat16ToFloat32(dst1b.Data(), dst1b.Size(), dst1f.Data());
            SimdBFloat16ToFloat32(dst2b.Data(), dst2b.Size(), dst2f.Data());
        }
        result = result && Compare(dst1f, dst2f, eps, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetAdd16bAutoTest(const FuncA16b& f1, const FuncA16b& f2)
    {
        bool result = true;

        const SimdTensorFormatType nchw = SimdTensorFormatNchw, nhwc = SimdTensorFormatNhwc;
        const SimdTensorDataType f32 = SimdTensorData32f, b16 = SimdTensorData16b;

#ifdef NDEBUG

#endif

#if 0
        result = result && SynetAdd16bAutoTest(Shp(1, 127, 17, 17), b16, Shp(1, 127, 17, 17), b16, b16, nhwc, f1, f2);
        result = result && SynetAdd16bAutoTest(Shp(1, 127, 17, 17), b16, Shp(1, 127, 17, 17), f32, b16, nhwc, f1, f2);
        result = result && SynetAdd16bAutoTest(Shp(1, 127, 17, 17), b16, Shp(1, 127, 17, 17), b16, f32, nhwc, f1, f2);
        result = result && SynetAdd16bAutoTest(Shp(1, 127, 17, 17), b16, Shp(1, 127, 17, 17), f32, f32, nhwc, f1, f2);
        result = result && SynetAdd16bAutoTest(Shp(1, 127, 17, 17), f32, Shp(1, 127, 17, 17), f32, b16, nhwc, f1, f2);
        result = result && SynetAdd16bAutoTest(Shp(1, 127, 17, 17), f32, Shp(1, 127, 17, 17), f32, f32, nhwc, f1, f2);
#endif


#if 1
        result = result && SynetAdd16bAutoTest(Shp(1, 128, 74, 74), b16, Shp(1, 128, 74, 74), b16, b16, nhwc, f1, f2);
        result = result && SynetAdd16bAutoTest(Shp(1, 128, 74, 74), b16, Shp(1, 128, 74, 74), f32, b16, nhwc, f1, f2);
        result = result && SynetAdd16bAutoTest(Shp(1, 128, 74, 74), b16, Shp(1, 128, 74, 74), b16, f32, nhwc, f1, f2);
        result = result && SynetAdd16bAutoTest(Shp(1, 128, 74, 74), b16, Shp(1, 128, 74, 74), f32, f32, nhwc, f1, f2);
        result = result && SynetAdd16bAutoTest(Shp(1, 128, 74, 74), f32, Shp(1, 128, 74, 74), f32, b16, nhwc, f1, f2);
        result = result && SynetAdd16bAutoTest(Shp(1, 128, 74, 74), f32, Shp(1, 128, 74, 74), f32, f32, nhwc, f1, f2);
#endif

        return result;
    }

    bool SynetAdd16bAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetAdd16bAutoTest(FUNC_A16B(Simd::Base::SynetAdd16bInit), FUNC_A16B(SimdSynetAdd16bInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetAdd16bAutoTest(FUNC_A16B(Simd::Sse41::SynetAdd16bInit), FUNC_A16B(SimdSynetAdd16bInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetAdd16bAutoTest(FUNC_A16B(Simd::Avx2::SynetAdd16bInit), FUNC_A16B(SimdSynetAdd16bInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetAdd16bAutoTest(FUNC_A16B(Simd::Avx512bw::SynetAdd16bInit), FUNC_A16B(SimdSynetAdd16bInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && SynetAdd16bAutoTest(FUNC_A16B(Simd::Neon::SynetAdd16bInit), FUNC_A16B(SimdSynetAdd16bInit));
#endif

#ifdef SIMD_SVE2_ENABLE
        if (Simd::Sve2::Enable && TestSve2(options))
            result = result && SynetAdd16bAutoTest(FUNC_A16B(Simd::Sve2::SynetAdd16bInit), FUNC_A16B(SimdSynetAdd16bInit));
#endif

        return result;
    }
#endif
}
