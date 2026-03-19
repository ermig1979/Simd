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

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    struct FuncSnlf16b2
    {
        typedef void(*FuncPtr)(const uint16_t* src, size_t batch, size_t channels, size_t spatial, const float* scale,
            const float* shift, const float* eps, SimdTensorFormatType format, float* buf, uint16_t* dst);

        FuncPtr func;
        String desc;

        FuncSnlf16b2(const FuncPtr& f, const String& d) : func(f), desc(d) {}

        void Update(size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format)
        {
            desc = desc + "[" + ToString(batch) + "x" + ToString(channels) + "x" + ToString(spatial) + "-" + ToString(format) + "]";
        }

        void Call(const Tensor16u& src, size_t batch, size_t channels, size_t spatial, const Tensor32f& scale,
            const Tensor32f& shift, float eps, SimdTensorFormatType format, Tensor32f& buf, Tensor16u& dst) const
        {
            TEST_PERFORMANCE_TEST(desc);
            func(src.Data(), batch, channels, spatial, scale.Data(), shift.Data(), &eps, format, buf.Data(), dst.Data());
        }
    };

#define FUNC_SNLF16B2(function) FuncSnlf16b2(function, #function)

    bool SynetNormalizeLayerForward16bV2AutoTest(size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, int extBuf, FuncSnlf16b2 f1, FuncSnlf16b2 f2)
    {
        bool result = true;

        f1.Update(batch, channels, spatial, format);
        f2.Update(batch, channels, spatial, format);

        const float eps = 0.000001f;
        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << batch << ", " << channels << ", " << spatial << "].");

        Tensor32f src32f(ToShape(batch, channels, 1, spatial, format));
        Tensor16u src16b(ToShape(batch, channels, 1, spatial, format));
        Tensor32f scale(ToShape(channels));
        Tensor32f shift(ToShape(channels));
        Tensor32f buf;
        if (extBuf)
            buf.Reshape(ToShape(Simd::Max(spatial, channels)));
        Tensor32f dst32f1(ToShape(batch, channels, 1, spatial, format));
        Tensor32f dst32f2(ToShape(batch, channels, 1, spatial, format));
        Tensor16u dst16b1(ToShape(batch, channels, 1, spatial, format));
        Tensor16u dst16b2(ToShape(batch, channels, 1, spatial, format));

        FillRandom(src32f.Data(), src32f.Size(), -10.0, 10.0);
        SimdFloat32ToBFloat16(src32f.Data(), src32f.Size(), src16b.Data());
        FillRandom(scale.Data(), scale.Size(), -10.0, 10.0);
        FillRandom(shift.Data(), shift.Size(), -10.0, 10.0);
        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src16b, batch, channels, spatial, scale, shift, eps, format, buf, dst16b1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src16b, batch, channels, spatial, scale, shift, eps, format, buf, dst16b2));

        SimdBFloat16ToFloat32(dst16b1.Data(), dst16b1.Size(), dst32f1.Data());
        SimdBFloat16ToFloat32(dst16b2.Data(), dst16b2.Size(), dst32f2.Data());

        result = result && Compare(dst32f1, dst32f2, EPS * 8.0f, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetNormalizeLayerForward16bV2AutoTest(const FuncSnlf16b2& f1, const FuncSnlf16b2& f2)
    {
        bool result = true;

        //result = result && SynetNormalizeLayerForward16bV2AutoTest(1, 8, 1, SimdTensorFormatNhwc, 1, f1, f2);

        SimdTensorFormatType formats[1] = { /*SimdTensorFormatNchw,*/ SimdTensorFormatNhwc };

        for (int f = 0; f < 1; f++)
        {
            result = result && SynetNormalizeLayerForward16bV2AutoTest(1, C, W, formats[f], 1, f1, f2);
            result = result && SynetNormalizeLayerForward16bV2AutoTest(8, C, W, formats[f], 1, f1, f2);
            result = result && SynetNormalizeLayerForward16bV2AutoTest(7, C - O, W + O, formats[f], 0, f1, f2);
        }

        return result;
    }

    bool SynetNormalizeLayerForward16bV2AutoTest(const Options& options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetNormalizeLayerForward16bV2AutoTest(FUNC_SNLF16B2(Simd::Base::SynetNormalizeLayerForward16bV2), FUNC_SNLF16B2(SimdSynetNormalizeLayerForward16bV2));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetNormalizeLayerForward16bV2AutoTest(FUNC_SNLF16B2(Simd::Sse41::SynetNormalizeLayerForward16bV2), FUNC_SNLF16B2(SimdSynetNormalizeLayerForward16bV2));
#endif 

        return result;
    }

#endif
}
