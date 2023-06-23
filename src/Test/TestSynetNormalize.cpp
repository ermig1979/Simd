/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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

#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncSNLF
        {
            typedef void(*FuncPtr)(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale,
                const float* eps, SimdBool acrossSpatial, SimdTensorFormatType format, float* buf, float* dst);

            FuncPtr func;
            String desc;

            FuncSNLF(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(size_t batch, size_t channels, size_t spatial, int acrossChannels, SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(batch) + "x" + ToString(channels) + "x" + ToString(spatial) + "-"
                    + ToString(acrossChannels) + "-" + ToString(format) + "]";
            }

            void Call(const Tensor32f & src, size_t batch, size_t channels, size_t spatial, const Tensor32f & scale, 
                float eps, int acrossSpatial, SimdTensorFormatType format, Tensor32f& buf, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), batch, channels, spatial, scale.Data(), &eps, (SimdBool)acrossSpatial, format, buf.Data(), dst.Data());
            }
        };
    }

#define FUNC_SNLF(function) FuncSNLF(function, #function)

    bool SynetNormalizeLayerForwardAutoTest(size_t batch, size_t channels, size_t spatial,
        int acrossSpatial, SimdTensorFormatType format, int extBuf, FuncSNLF f1, FuncSNLF f2)
    {
        bool result = true;

        f1.Update(batch, channels, spatial, acrossSpatial, format);
        f2.Update(batch, channels, spatial, acrossSpatial, format);

        const float eps = 0.0f;
        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << batch << ", " << channels << ", " << spatial << "].");

        Tensor32f src(ToShape(batch, channels, 1, spatial, format));
        Tensor32f scale(ToShape(channels));
        Tensor32f buf;
        if(extBuf)
            buf.Reshape(ToShape(spatial));
        Tensor32f dst1(ToShape(batch, channels, 1, spatial, format));
        Tensor32f dst2(ToShape(batch, channels, 1, spatial, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        FillRandom(scale.Data(), scale.Size(), -10.0, 10.0);
        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, batch, channels, spatial, scale, eps, acrossSpatial, format, buf, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, batch, channels, spatial, scale, eps, acrossSpatial, format, buf, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetNormalizeLayerForwardAutoTest(const FuncSNLF& f1, const FuncSNLF& f2)
    {
        bool result = true;

        SimdTensorFormatType formats[2] = { SimdTensorFormatNchw, SimdTensorFormatNhwc };

        //result = result && SynetNormalizeLayerForwardAutoTest(100, 1, 256, 1, formats[0], 1, f1, f2);

        for (int f = 0; f < 2; f++)
        {
            for (int acrossSpatial = 0; acrossSpatial <= 1; ++acrossSpatial)
            {
                result = result && SynetNormalizeLayerForwardAutoTest(1, C, W, acrossSpatial, formats[f], 1, f1, f2);
                result = result && SynetNormalizeLayerForwardAutoTest(8, C, W, acrossSpatial, formats[f], 1, f1, f2);
                result = result && SynetNormalizeLayerForwardAutoTest(7, C - O, W + O, acrossSpatial, formats[f], 0, f1, f2);
            }
        }

        return result;
    }

    bool SynetNormalizeLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetNormalizeLayerForwardAutoTest(FUNC_SNLF(Simd::Base::SynetNormalizeLayerForward), FUNC_SNLF(SimdSynetNormalizeLayerForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetNormalizeLayerForwardAutoTest(FUNC_SNLF(Simd::Sse41::SynetNormalizeLayerForward), FUNC_SNLF(SimdSynetNormalizeLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetNormalizeLayerForwardAutoTest(FUNC_SNLF(Simd::Avx2::SynetNormalizeLayerForward), FUNC_SNLF(SimdSynetNormalizeLayerForward));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetNormalizeLayerForwardAutoTest(FUNC_SNLF(Simd::Avx512bw::SynetNormalizeLayerForward), FUNC_SNLF(SimdSynetNormalizeLayerForward));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    struct FuncSNLF2
    {
        typedef void(*FuncPtr)(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale,
            const float* shift, const float* eps, SimdTensorFormatType format, float* buf, float* dst);

        FuncPtr func;
        String desc;

        FuncSNLF2(const FuncPtr& f, const String& d) : func(f), desc(d) {}

        void Update(size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format)
        {
            desc = desc + "[" + ToString(batch) + "x" + ToString(channels) + "x" + ToString(spatial) + "-" + ToString(format) + "]";
        }

        void Call(const Tensor32f& src, size_t batch, size_t channels, size_t spatial, const Tensor32f& scale, 
            const Tensor32f& shift, float eps, SimdTensorFormatType format, Tensor32f& buf, Tensor32f& dst) const
        {
            TEST_PERFORMANCE_TEST(desc);
            func(src.Data(), batch, channels, spatial, scale.Data(), shift.Data(), &eps, format, buf.Data(), dst.Data());
        }
    };

#define FUNC_SNLF2(function) FuncSNLF2(function, #function)

    bool SynetNormalizeLayerForwardV2AutoTest(size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, int extBuf, FuncSNLF2 f1, FuncSNLF2 f2)
    {
        bool result = true;

        f1.Update(batch, channels, spatial, format);
        f2.Update(batch, channels, spatial, format);

        const float eps = 0.000001f;
        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << batch << ", " << channels << ", " << spatial << "].");

        Tensor32f src(ToShape(batch, channels, 1, spatial, format));
        Tensor32f scale(ToShape(channels));
        Tensor32f shift(ToShape(channels));
        Tensor32f buf;
        if (extBuf)
            buf.Reshape(ToShape(Simd::Max(spatial, channels)));
        Tensor32f dst1(ToShape(batch, channels, 1, spatial, format));
        Tensor32f dst2(ToShape(batch, channels, 1, spatial, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        FillRandom(scale.Data(), scale.Size(), -10.0, 10.0);
        FillRandom(shift.Data(), shift.Size(), -10.0, 10.0);
        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, batch, channels, spatial, scale, shift, eps, format, buf, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, batch, channels, spatial, scale, shift, eps, format, buf, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetNormalizeLayerForwardV2AutoTest(const FuncSNLF2& f1, const FuncSNLF2& f2)
    {
        bool result = true;

        SimdTensorFormatType formats[2] = { SimdTensorFormatNchw, SimdTensorFormatNhwc };

        for (int f = 0; f < 2; f++)
        {
            result = result && SynetNormalizeLayerForwardV2AutoTest(1, C, W, formats[f], 1, f1, f2);
            result = result && SynetNormalizeLayerForwardV2AutoTest(8, C, W, formats[f], 1, f1, f2);
            result = result && SynetNormalizeLayerForwardV2AutoTest(7, C - O, W + O, formats[f], 0, f1, f2);
        }

        return result;
    }

    bool SynetNormalizeLayerForwardV2AutoTest()
    {
        bool result = true;

        result = result && SynetNormalizeLayerForwardV2AutoTest(FUNC_SNLF2(Simd::Base::SynetNormalizeLayerForwardV2), FUNC_SNLF2(SimdSynetNormalizeLayerForwardV2));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetNormalizeLayerForwardV2AutoTest(FUNC_SNLF2(Simd::Sse41::SynetNormalizeLayerForwardV2), FUNC_SNLF2(SimdSynetNormalizeLayerForwardV2));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetNormalizeLayerForwardV2AutoTest(FUNC_SNLF2(Simd::Avx2::SynetNormalizeLayerForwardV2), FUNC_SNLF2(SimdSynetNormalizeLayerForwardV2));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetNormalizeLayerForwardV2AutoTest(FUNC_SNLF2(Simd::Avx512bw::SynetNormalizeLayerForwardV2), FUNC_SNLF2(SimdSynetNormalizeLayerForwardV2));
#endif 

        return result;
    }

    bool SynetNormalizeLayerForwardV3AutoTest()
    {
        bool result = true;

        result = result && SynetNormalizeLayerForwardV2AutoTest(FUNC_SNLF2(Simd::Base::SynetNormalizeLayerForwardV3), FUNC_SNLF2(SimdSynetNormalizeLayerForwardV3));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetNormalizeLayerForwardV2AutoTest(FUNC_SNLF2(Simd::Sse41::SynetNormalizeLayerForwardV3), FUNC_SNLF2(SimdSynetNormalizeLayerForwardV3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetNormalizeLayerForwardV2AutoTest(FUNC_SNLF2(Simd::Avx2::SynetNormalizeLayerForwardV3), FUNC_SNLF2(SimdSynetNormalizeLayerForwardV3));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetNormalizeLayerForwardV2AutoTest(FUNC_SNLF2(Simd::Avx512bw::SynetNormalizeLayerForwardV3), FUNC_SNLF2(SimdSynetNormalizeLayerForwardV3));
#endif 

        return result;
    }

#endif
}
