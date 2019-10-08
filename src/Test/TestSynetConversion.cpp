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

namespace Test
{
    namespace
    {
        struct FuncCT
        {
            typedef void(*FuncPtr)(size_t n, size_t c, size_t hw, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat);

            FuncPtr func;
            String desc;

            FuncCT(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(size_t n, size_t c, size_t h, size_t w, SimdTensorFormatType src, SimdTensorFormatType dst)
            {
                desc = desc + "[" + ToString(n) + "x" + ToString(c) + "x" + ToString(h) + "x" + ToString(w) + ":" + ToString(src) + "->" + ToString(dst) + "]";
            }

            void Call(size_t n, size_t c, size_t h, size_t w, const Tensor32f & src, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(n, c, h*w, src.Data(), src.Format(), dst.Data(), dst.Format());
            }
        };
    }

#define FUNC_CT(function) FuncCT(function, #function)

    bool SynetConvertTensorAutoTest(size_t n, size_t c, size_t h, size_t w, SimdTensorFormatType srcFormat, SimdTensorFormatType dstFormat, FuncCT f1, FuncCT f2)
    {
        bool result = true;

        f1.Update(n, c, h, w, srcFormat, dstFormat);
        f2.Update(n, c, h, w, srcFormat, dstFormat);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        Tensor32f src1(ToShape(n, c, h, w, srcFormat), srcFormat);
        Tensor32f dst1(ToShape(n, c, h, w, dstFormat), dstFormat);
        Tensor32f dst2(ToShape(n, c, h, w, dstFormat), dstFormat);
        TEST_ALIGN(SIMD_ALIGN);

#if 1
        FillDebug(src1);
#else
        FillRandom(src1.Data(), src1.Size(), -10.0, 10.0);
#endif

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(n, c, h, w, src1, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(n, c, h, w, src1, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceBoth, "forward");
#if 0
        if (srcFormat == SimdTensorFormatNchw || srcFormat == SimdTensorFormatNhwc || srcFormat == SimdTensorFormatOiyx || srcFormat == SimdTensorFormatYxio)
        {
            Tensor32f src2(ToShape(n, c, h, w, srcFormat), srcFormat);
            f2.Call(n, c, h, w, dst2, src2);
            result = result && Compare(src1, src2, EPS, true, 64, DifferenceBoth, "backward");
        }
#endif
        return result;
    }

    bool SynetConvertImageAutoTest(int mask, const FuncCT & f1, const FuncCT & f2)
    {
        bool result = true;

        for (SimdTensorFormatType src = SimdTensorFormatNchw; src <= SimdTensorFormatNchw16c && result; src = (SimdTensorFormatType)((int)src + 1))
        {
            for (SimdTensorFormatType dst = SimdTensorFormatNchw; dst <= SimdTensorFormatNchw16c && result; dst = (SimdTensorFormatType)((int)dst + 1))
            {
                if (src == dst || (src >= SimdTensorFormatNchw4c && dst >= SimdTensorFormatNchw4c) || 
                    ((SimdSynetTensorAlignment(src)&mask) == 0) || ((SimdSynetTensorAlignment(dst)&mask) == 0))
                    continue;
                result = result && SynetConvertTensorAutoTest(9, W / 15 + 0, W / 60, W / 30, src, dst, f1, f2);
                result = result && SynetConvertTensorAutoTest(9, W / 15 - 1, W / 58, W / 31, src, dst, f1, f2);
            }
        }

        return result;
    }

    bool SynetConvertImageAutoTest()
    {
        bool result = true;

        result = result && SynetConvertImageAutoTest(TFM_ANY, FUNC_CT(Simd::Base::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetConvertImageAutoTest(TFM_128, FUNC_CT(Simd::Sse::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetConvertImageAutoTest(TFM_256, FUNC_CT(Simd::Avx::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetConvertImageAutoTest(TFM_512, FUNC_CT(Simd::Avx512f::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetConvertImageAutoTest(TFM_128, FUNC_CT(Simd::Neon::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));
#endif

        return result;
    }

    bool SynetConvertFilterAutoTest(int mask, const FuncCT & f1, const FuncCT & f2)
    {
        bool result = true;

        for (SimdTensorFormatType src = SimdTensorFormatOiyx; src <= SimdTensorFormatOyxi16o && result; src = (SimdTensorFormatType)((int)src + 1))
        {
            for (SimdTensorFormatType dst = SimdTensorFormatOiyx; dst <= SimdTensorFormatOyxi16o && result; dst = (SimdTensorFormatType)((int)dst + 1))
            {
                if (src == dst || (src >= SimdTensorFormatOyxi4o && dst >= SimdTensorFormatOyxi4o) ||
                    ((SimdSynetTensorAlignment(src)&mask) == 0) || ((SimdSynetTensorAlignment(dst)&mask) == 0))
                    continue;
                result = result && SynetConvertTensorAutoTest(W * 9 / 30 + 0, W * 7 / 30 + 0, 3, 3, src, dst, f1, f2);
                result = result && SynetConvertTensorAutoTest(W * 9 / 10 - 1, W * 7 / 10 + 1, 1, 1, src, dst, f1, f2);
            }
        }

        return result;
    }

    bool SynetConvertFilterAutoTest()
    {
        bool result = true;

        result = result && SynetConvertFilterAutoTest(TFM_ANY, FUNC_CT(Simd::Base::SynetConvertFilter), FUNC_CT(SimdSynetConvertFilter));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetConvertFilterAutoTest(TFM_128, FUNC_CT(Simd::Sse::SynetConvertFilter), FUNC_CT(SimdSynetConvertFilter));
#endif 
        
#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetConvertFilterAutoTest(TFM_256, FUNC_CT(Simd::Avx::SynetConvertFilter), FUNC_CT(SimdSynetConvertFilter));
#endif 
        
#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetConvertFilterAutoTest(TFM_512, FUNC_CT(Simd::Avx512f::SynetConvertFilter), FUNC_CT(SimdSynetConvertFilter));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetConvertFilterAutoTest(TFM_128, FUNC_CT(Simd::Neon::SynetConvertFilter), FUNC_CT(SimdSynetConvertFilter));
#endif

        return result;
    }

    namespace
    {
        struct FuncSI
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t width, size_t height, size_t stride, SimdPixelFormatType pixelFormat,
                const float * lower, const float * upper, float * dst, size_t channels, SimdTensorFormatType dstFormat);

            FuncPtr func;
            String desc;

            FuncSI(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(size_t c, size_t h, size_t w, View::Format src, SimdTensorFormatType dst)
            {
                desc = desc + "[" + ToString(c) + "x" + ToString(h) + "x" + ToString(w) + ":" + ToString(src) + "->" + ToString(dst) + "]";
            }

            void Call(const View & src, const float * lower, const float * upper, size_t channels, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.data, src.width, src.height, src.stride, (SimdPixelFormatType)src.format, lower, upper, dst.Data(), channels, dst.Format());
            }
        };
    }

#define FUNC_SI(function) FuncSI(function, #function)

    bool SynetSetInputAutoTest(size_t c, size_t h, size_t w, View::Format srcFormat, SimdTensorFormatType dstFormat, FuncSI f1, FuncSI f2)
    {
        bool result = true;

        assert(c == 1 || c == 3);

        f1.Update(c, h, w, srcFormat, dstFormat);
        f2.Update(c, h, w, srcFormat, dstFormat);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        View src(w, h, srcFormat);
        FillRandom(src);
        src.data[0] = 107, src.data[1] = 117, src.data[2] = 127, src.data[3] = 137;
        Tensor32f dst1(ToShape(1, c, h, w, dstFormat), dstFormat);
        Tensor32f dst2(ToShape(1, c, h, w, dstFormat), dstFormat);
        TEST_ALIGN(SIMD_ALIGN);

        float lower[3] = { -0.9f, -1.0f, -1.2f };
        float upper[3] = { 0.91f, 1.01f, 1.21f };

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, lower, upper, c, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, lower, upper, c, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetSetInputAutoTest(const FuncSI & f1, const FuncSI & f2)
    {
        bool result = true;

        View::Format srcFormat[4] = {View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24};
        size_t channels[2] = { 1, 3 };
        SimdTensorFormatType dstFormat[2] = { SimdTensorFormatNchw, SimdTensorFormatNhwc };

        for (int s = 0; s < 4; ++s)
        {
            for (int c = 0; c < 2; ++c)
            {
                for (int d = 0; d < 2; ++d)
                {
                    result = result && SynetSetInputAutoTest(channels[c], H / 3, W / 5 + O, srcFormat[s], dstFormat[d], f1, f2);
                }
            }
        }

        return result;
    }

    bool SynetSetInputAutoTest()
    {
        bool result = true;

        result = result && SynetSetInputAutoTest(FUNC_SI(Simd::Base::SynetSetInput), FUNC_SI(SimdSynetSetInput));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetSetInputAutoTest(FUNC_SI(Simd::Sse41::SynetSetInput), FUNC_SI(SimdSynetSetInput));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetSetInputAutoTest(FUNC_SI(Simd::Avx2::SynetSetInput), FUNC_SI(SimdSynetSetInput));
#endif 

//#ifdef SIMD_AVX512F_ENABLE
//        if (Simd::Avx512f::Enable)
//            result = result && SynetConvertImageAutoTest(TFM_512, FUNC_CT(Simd::Avx512f::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));
//#endif 
//
//#ifdef SIMD_NEON_ENABLE
//        if (Simd::Neon::Enable)
//            result = result && SynetConvertImageAutoTest(TFM_128, FUNC_CT(Simd::Neon::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));
//#endif

        return result;
    }
}
