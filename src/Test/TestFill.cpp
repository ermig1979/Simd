/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Test/TestString.h"

namespace Test
{
    namespace
    {
        struct Func
        {
            typedef void(*FuncPtr)(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize, uint8_t value);

            FuncPtr func;
            String description;

            Func(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(View & dst, uint8_t value) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(dst.data, dst.stride, dst.width, dst.height, dst.PixelSize(), value);
            }
        };
    }

#define FUNC(function) \
    Func(function, std::string(#function))

    bool FillAutoTest(View::Format format, int width, int height, const Func & f1, const Func & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        uint8_t value = Random(256);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(d1, value));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(d2, value));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool FillAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        for (View::Format format = View::Gray8; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            if (format == View::Float || format == View::Double)
                continue;

            Func f1c = Func(f1.func, f1.description + ColorDescription(format));
            Func f2c = Func(f2.func, f2.description + ColorDescription(format));

            result = result && FillAutoTest(format, W, H, f1c, f2c);
            result = result && FillAutoTest(format, W + O, H - O, f1c, f2c);
        }

        return result;
    }

    bool FillAutoTest()
    {
        bool result = true;

        result = result && FillAutoTest(FUNC(Simd::Base::Fill), FUNC(SimdFill));

        return result;
    }

    namespace
    {
        struct FuncF
        {
            typedef void(*FuncPtr)(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize,
                size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t value);

            FuncPtr func;
            String description;

            FuncF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(View & dst, const Rect & frame, uint8_t value) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(dst.data, dst.stride, dst.width, dst.height, dst.PixelSize(),
                    frame.left, frame.top, frame.right, frame.bottom, value);
            }
        };
    }

#define FUNC_F(function) \
    FuncF(function, std::string(#function))

    bool FillFrameAutoTest(View::Format format, int width, int height, const FuncF & f1, const FuncF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        uint8_t value = Random(256);
        Rect frame(width * 1 / 15, height * 2 / 15, width * 11 / 15, height * 12 / 15);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));
        Simd::Fill(d1, 0);
        Simd::Fill(d2, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(d1, frame, value));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(d2, frame, value));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool FillFrameAutoTest(const FuncF & f1, const FuncF & f2)
    {
        bool result = true;

        for (View::Format format = View::Gray8; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            if (format == View::Float || format == View::Double)
                continue;

            FuncF f1c = FuncF(f1.func, f1.description + ColorDescription(format));
            FuncF f2c = FuncF(f2.func, f2.description + ColorDescription(format));

            result = result && FillFrameAutoTest(format, W, H, f1c, f2c);
            result = result && FillFrameAutoTest(format, W + O, H - O, f1c, f2c);
        }

        return result;
    }

    bool FillFrameAutoTest()
    {
        bool result = true;

        result = result && FillFrameAutoTest(FUNC_F(Simd::Base::FillFrame), FUNC_F(SimdFillFrame));

        return result;
    }

    namespace
    {
        struct FuncBgra
        {
            typedef void(*FuncPtr)(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha);

            FuncPtr func;
            String description;

            FuncBgra(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(View & dst, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(dst.data, dst.stride, dst.width, dst.height, blue, green, red, alpha);
            }
        };
    }

#define FUNC_BGRA(function) \
	FuncBgra(function, std::string(#function))

    bool FillBgraAutoTest(int width, int height, const FuncBgra & f1, const FuncBgra & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        uint8_t blue = Random(256);
        uint8_t green = Random(256);
        uint8_t red = Random(256);
        uint8_t alpha = Random(256);

        View d1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(d1, blue, green, red, alpha));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(d2, blue, green, red, alpha));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool FillBgraAutoTest(const FuncBgra & f1, const FuncBgra & f2)
    {
        bool result = true;

        result = result && FillBgraAutoTest(W, H, f1, f2);
        result = result && FillBgraAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    bool FillBgraAutoTest()
    {
        bool result = true;

        result = result && FillBgraAutoTest(FUNC_BGRA(Simd::Base::FillBgra), FUNC_BGRA(SimdFillBgra));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::F)
            result = result && FillBgraAutoTest(FUNC_BGRA(Simd::Sse2::FillBgra), FUNC_BGRA(SimdFillBgra));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::F)
            result = result && FillBgraAutoTest(FUNC_BGRA(Simd::Avx2::FillBgra), FUNC_BGRA(SimdFillBgra));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && FillBgraAutoTest(FUNC_BGRA(Simd::Avx512bw::FillBgra), FUNC_BGRA(SimdFillBgra));
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::F)
            result = result && FillBgraAutoTest(FUNC_BGRA(Simd::Vmx::FillBgra), FUNC_BGRA(SimdFillBgra));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::F)
            result = result && FillBgraAutoTest(FUNC_BGRA(Simd::Neon::FillBgra), FUNC_BGRA(SimdFillBgra));
#endif 

        return result;
    }

    namespace
    {
        struct FuncBgr
        {
            typedef void(*FuncPtr)(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red);

            FuncPtr func;
            String description;

            FuncBgr(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(View & dst, uint8_t blue, uint8_t green, uint8_t red) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(dst.data, dst.stride, dst.width, dst.height, blue, green, red);
            }
        };
    }

#define FUNC_BGR(function) \
    FuncBgr(function, std::string(#function))

    bool FillBgrAutoTest(int width, int height, const FuncBgr & f1, const FuncBgr & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        uint8_t blue = Random(256);
        uint8_t green = Random(256);
        uint8_t red = Random(256);

        View d1(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(d1, blue, green, red));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(d2, blue, green, red));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool FillBgrAutoTest(const FuncBgr & f1, const FuncBgr & f2)
    {
        bool result = true;

        result = result && FillBgrAutoTest(W, H, f1, f2);
        result = result && FillBgrAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    bool FillBgrAutoTest()
    {
        bool result = true;

        result = result && FillBgrAutoTest(FUNC_BGR(Simd::Base::FillBgr), FUNC_BGR(SimdFillBgr));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && FillBgrAutoTest(FUNC_BGR(Simd::Sse2::FillBgr), FUNC_BGR(SimdFillBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && FillBgrAutoTest(FUNC_BGR(Simd::Avx2::FillBgr), FUNC_BGR(SimdFillBgr));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && FillBgrAutoTest(FUNC_BGR(Simd::Avx512bw::FillBgr), FUNC_BGR(SimdFillBgr));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && FillBgrAutoTest(FUNC_BGR(Simd::Vmx::FillBgr), FUNC_BGR(SimdFillBgr));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && FillBgrAutoTest(FUNC_BGR(Simd::Neon::FillBgr), FUNC_BGR(SimdFillBgr));
#endif 

        return result;
    }

    namespace
    {
        struct FuncFP
        {
            typedef void(*FuncPtr)(uint8_t * dst, size_t stride, size_t width, size_t height, const uint8_t * pixel, size_t pixelSize);

            FuncPtr func;
            String description;

            FuncFP(const FuncPtr & f, const String & d) : func(f), description(d) {}

            FuncFP(const FuncFP & f, size_t s) : func(f.func), description(f.description + "[" + ToString(s) + "]") {}

            void Call(View & dst, const uint8_t * pixel, size_t size) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(dst.data, dst.stride, dst.width / size, dst.height, pixel, size);
            }
        };
    }

#define FUNC_FP(function) FuncFP(function, std::string(#function))

#define ARGS_FP(s, f1, f2) s, FuncFP(f1, s), FuncFP(f2, s)

    const size_t PIXEL_SIZE_MAX = 4;

    bool FillPixelAutoTest(int width, int height, size_t size, const FuncFP & f1, const FuncFP & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        uint8_t pixel[PIXEL_SIZE_MAX] = { 1, 2, 3, 4 };

        View d1(width*size, height, View::Gray8, NULL, TEST_ALIGN(width*size));
        View d2(width*size, height, View::Gray8, NULL, TEST_ALIGN(width*size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(d1, pixel, size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(d2, pixel, size));

        result = result && Compare(d1, d2, 0, true, 32);

        return result;
    }

    bool FillPixelAutoTest(const FuncFP & f1, const FuncFP & f2)
    {
        bool result = true;

        for (size_t s = 1; s <= PIXEL_SIZE_MAX; ++s)
        {
            result = result && FillPixelAutoTest(W, H, ARGS_FP(s, f1, f2));
            result = result && FillPixelAutoTest(W + O, H - O, ARGS_FP(s, f1, f2));
        }

        return result;
    }

    bool FillPixelAutoTest()
    {
        bool result = true;

        result = result && FillPixelAutoTest(FUNC_FP(Simd::Base::FillPixel), FUNC_FP(SimdFillPixel));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && FillPixelAutoTest(FUNC_FP(Simd::Sse2::FillPixel), FUNC_FP(SimdFillPixel));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && FillPixelAutoTest(FUNC_FP(Simd::Avx2::FillPixel), FUNC_FP(SimdFillPixel));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && FillPixelAutoTest(FUNC_FP(Simd::Avx512bw::FillPixel), FUNC_FP(SimdFillPixel));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && FillPixelAutoTest(FUNC_FP(Simd::Neon::FillPixel), FUNC_FP(SimdFillPixel));
#endif 

        return result;
    }

    namespace
    {
        struct FuncFill32f
        {
            typedef void(*FuncPtr)(float * dst, size_t size, const float * value);

            FuncPtr func;
            String description;

            FuncFill32f(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(View & dst, float value) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)dst.data, dst.width, &value);
            }
        };
    }

#define FUNC_32F(function) FuncFill32f(function, std::string(#function))

    bool Fill32fAutoTest(size_t size, const FuncFill32f & f1, const FuncFill32f & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        const float value = 3.5f;

        View d1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View d2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(d1, value));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(d2, value));

        result = result && Compare(d1, d2, EPS, true, 32);

        return result;
    }

    bool Fill32fAutoTest(const FuncFill32f & f1, const FuncFill32f & f2)
    {
        bool result = true;

        result = result && Fill32fAutoTest(W*H, f1, f2);
        result = result && Fill32fAutoTest(W*H + O, f1, f2);

        return result;
    }

    bool Fill32fAutoTest()
    {
        bool result = true;

        result = result && Fill32fAutoTest(FUNC_32F(Simd::Base::Fill32f), FUNC_32F(SimdFill32f));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && Fill32fAutoTest(FUNC_32F(Simd::Sse2::Fill32f), FUNC_32F(SimdFill32f));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && Fill32fAutoTest(FUNC_32F(Simd::Avx::Fill32f), FUNC_32F(SimdFill32f));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && Fill32fAutoTest(FUNC_32F(Simd::Avx512f::Fill32f), FUNC_32F(SimdFill32f));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && Fill32fAutoTest(FUNC_32F(Simd::Neon::Fill32f), FUNC_32F(SimdFill32f));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool FillDataTest(bool create, View::Format format, int width, int height, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        const uint8_t value = 0x77;

        if (create)
        {
            f.Call(d1, value);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(d1);

            f.Call(d2, value);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool FillDataTest(bool create)
    {
        bool result = true;

        Func f = FUNC(SimdFill);

        for (View::Format format = View::Gray8; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            if (format == View::Float || format == View::Double)
                continue;

            result = result && FillDataTest(create, format, DW, DH, Func(f.func, f.description + Data::Description(format)));
        }

        return result;
    }

    bool FillFrameDataTest(bool create, View::Format format, int width, int height, const FuncF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        Rect frame(width * 1 / 15, height * 2 / 15, width * 11 / 15, height * 12 / 15);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        const uint8_t value = 0x77;

        if (create)
        {
            Simd::Fill(d1, 0);

            f.Call(d1, frame, value);

            TEST_SAVE(d1);
        }
        else
        {
            Simd::Fill(d2, 0);

            TEST_LOAD(d1);

            f.Call(d2, frame, value);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool FillFrameDataTest(bool create)
    {
        bool result = true;

        FuncF f = FUNC_F(SimdFillFrame);

        for (View::Format format = View::Gray8; format <= View::BayerBggr; format = View::Format(format + 1))
        {
            if (format == View::Float || format == View::Double)
                continue;

            result = result && FillFrameDataTest(create, format, DW, DH, FuncF(f.func, f.description + Data::Description(format)));
        }

        return result;
    }

    bool FillBgrDataTest(bool create, int width, int height, const FuncBgr & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View bgr1(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        View bgr2(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

        const uint8_t blue = 0x11;
        const uint8_t green = 0xAA;
        const uint8_t red = 0x77;

        if (create)
        {
            f.Call(bgr1, blue, green, red);

            TEST_SAVE(bgr1);
        }
        else
        {
            TEST_LOAD(bgr1);

            f.Call(bgr2, blue, green, red);

            TEST_SAVE(bgr2);

            result = result && Compare(bgr1, bgr2, 0, true, 64);
        }

        return result;
    }

    bool FillBgrDataTest(bool create)
    {
        bool result = true;

        result = result && FillBgrDataTest(create, DW, DH, FUNC_BGR(SimdFillBgr));

        return result;
    }

    bool FillBgraDataTest(bool create, int width, int height, const FuncBgra & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View bgra1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View bgra2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        const uint8_t blue = 0x11;
        const uint8_t green = 0xAA;
        const uint8_t red = 0x77;
        const uint8_t alpha = 0xFF;

        if (create)
        {
            f.Call(bgra1, blue, green, red, alpha);

            TEST_SAVE(bgra1);
        }
        else
        {
            TEST_LOAD(bgra1);

            f.Call(bgra2, blue, green, red, alpha);

            TEST_SAVE(bgra2);

            result = result && Compare(bgra1, bgra2, 0, true, 64);
        }

        return result;
    }

    bool FillBgraDataTest(bool create)
    {
        bool result = true;

        result = result && FillBgraDataTest(create, DW, DH, FUNC_BGRA(SimdFillBgra));

        return result;
    }

    bool FillPixelDataTest(bool create, int width, int height, size_t size, const FuncFP & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << ", " << size << "].");

        View dst1(width*size, height, View::Bgra32, NULL, TEST_ALIGN(width*size));
        View dst2(width*size, height, View::Bgra32, NULL, TEST_ALIGN(width*size));

        uint8_t pixel[PIXEL_SIZE_MAX] = { 1, 2, 3, 4 };

        if (create)
        {
            f.Call(dst1, pixel, size);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(dst1);

            f.Call(dst2, pixel, size);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 64);
        }

        return result;
    }

    bool FillPixelDataTest(bool create)
    {
        bool result = true;

        result = result && FillPixelDataTest(create, DW, DH, 3, FUNC_FP(SimdFillPixel));

        return result;
    }
}
