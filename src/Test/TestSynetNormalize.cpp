/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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

        return result;
    }
#endif
}
