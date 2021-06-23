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
* LIABILITY, WHETHER IN AN ARTION OF CONTRART, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNERTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "Test/TestUtils.h"
#include "Test/TestPerformance.h"
#include "Test/TestData.h"
#include "Test/TestTensor.h"
#include "Test/TestString.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    template<class S, class D> struct FuncCvt
    {
        typedef void(*FuncPtr)(const S * src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, D * dst, SimdSynetCompatibilityType compatibility);

        FuncPtr func;
        String desc;

        FuncCvt(const FuncPtr& f, const String& d) : func(f), desc(d) {}

        void Update(size_t n, size_t c, size_t h, size_t w, SimdTensorFormatType f, SimdSynetCompatibilityType comp)
        {
            desc = desc + "[" + ToString(n) + "x" + ToString(c) + "x" + ToString(h) + "x" + ToString(w) + ":" + ToString(f) + "-" + ToString((int)comp) + "]";
        }

        void Call(const Tensor<S> & src, const Buffer32f & scale, const Buffer32f& shift, Tensor<D> & dst, SimdSynetCompatibilityType comp) const
        {
            TEST_PERFORMANCE_TEST(desc);
            func(src.Data(), src.Batch(), src.Channels(), src.Height(), src.Width(), src.Format(), scale.data(), shift.data(), dst.Data(), comp);
        }
    };

    typedef FuncCvt<float, uint8_t> FuncCvt32fTo8u;

#define FUNC_C_32F_8U(function) FuncCvt32fTo8u(function, #function)

    bool SynetConvert32fTo8uAutoTest(size_t n, size_t c, size_t h, size_t w, SimdTensorFormatType f, SimdSynetCompatibilityType comp, FuncCvt32fTo8u f1, FuncCvt32fTo8u f2)
    {
        bool result = true;

        f1.Update(n, c, h, w, f, comp);
        f2.Update(n, c, h, w, f, comp);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        Tensor32f src(ToShape(n, c, h, w, f), f);
        FillRandom(src, -1.0f, 1.0f);

        Tensor8u dst1(ToShape(n, c, h, w, f), f);
        Tensor8u dst2(ToShape(n, c, h, w, f), f);

        Buffer32f scale(c), shift(c);
        FillRandom(scale, -100.0f, 100.0f);
        FillRandom(shift, -100.0f, 100.0f);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, scale, shift, dst1, comp));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, scale, shift, dst2, comp));

#if defined(SIMD_X64_ENABLE) || defined(SIMD_X86_ENABLE)
        int differenceMax = (comp == SimdSynetCompatibilityFmaAvoid ? 0 : 1);
#else
        int differenceMax = 1;
#endif

        result = result && Compare(dst1, dst2, differenceMax, true, 64);

        return result;
    }

    bool SynetConvert32fTo8uAutoTest(const FuncCvt32fTo8u& f1, const FuncCvt32fTo8u& f2)
    {
        bool result = true;

        SimdTensorFormatType format[2] = { SimdTensorFormatNchw, SimdTensorFormatNhwc };
        SimdSynetCompatibilityType compatibility[4] = { SimdSynetCompatibilityFmaUse, SimdSynetCompatibilityFmaNoTail, SimdSynetCompatibilityFmaAvoid, SimdSynetCompatibility8iNarrowed };

        for (int f = 0; f <= 1; ++f)
        {
            for (int c = 0; c <= 3; ++c)
            {
                result = result && SynetConvert32fTo8uAutoTest(2, C/2, (int)sqrt(H), (int)sqrt(W), format[f], compatibility[c], f1, f2);
                result = result && SynetConvert32fTo8uAutoTest(1, 3, H / 3 + O, W / 5 - O, format[f], compatibility[c], f1, f2);
            }
        }

        return result;
    }

    bool SynetConvert32fTo8uAutoTest()
    {
        bool result = true;

        result = result && SynetConvert32fTo8uAutoTest(FUNC_C_32F_8U(Simd::Base::SynetConvert32fTo8u), FUNC_C_32F_8U(SimdSynetConvert32fTo8u));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetConvert32fTo8uAutoTest(FUNC_C_32F_8U(Simd::Sse2::SynetConvert32fTo8u), FUNC_C_32F_8U(SimdSynetConvert32fTo8u));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetConvert32fTo8uAutoTest(FUNC_C_32F_8U(Simd::Avx2::SynetConvert32fTo8u), FUNC_C_32F_8U(SimdSynetConvert32fTo8u));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetConvert32fTo8uAutoTest(FUNC_C_32F_8U(Simd::Avx512bw::SynetConvert32fTo8u), FUNC_C_32F_8U(SimdSynetConvert32fTo8u));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetConvert32fTo8uAutoTest(FUNC_C_32F_8U(Simd::Neon::SynetConvert32fTo8u), FUNC_C_32F_8U(SimdSynetConvert32fTo8u));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------

    typedef FuncCvt<uint8_t, float> FuncCvt8uTo32f;

#define FUNC_C_8U_32F(function) FuncCvt8uTo32f(function, #function)

    bool SynetConvert8uTo32fAutoTest(size_t n, size_t c, size_t h, size_t w, SimdTensorFormatType f, SimdSynetCompatibilityType comp, FuncCvt8uTo32f f1, FuncCvt8uTo32f f2)
    {
        bool result = true;

        f1.Update(n, c, h, w, f, comp);
        f2.Update(n, c, h, w, f, comp);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        Tensor8u src(ToShape(n, c, h, w, f), f);
        FillRandom(src);

        Tensor32f dst1(ToShape(n, c, h, w, f), f);
        Tensor32f dst2(ToShape(n, c, h, w, f), f);

        Buffer32f scale(c), shift(c);
        FillRandom(scale, -1.0f, 1.0f);
        FillRandom(shift, -1.0f, 1.0f);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, scale, shift, dst1, comp));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, scale, shift, dst2, comp));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetConvert8uTo32fAutoTest(const FuncCvt8uTo32f& f1, const FuncCvt8uTo32f& f2)
    {
        bool result = true;

        SimdTensorFormatType format[2] = { SimdTensorFormatNchw, SimdTensorFormatNhwc };
        SimdSynetCompatibilityType compatibility[3] = { SimdSynetCompatibilityFmaUse, SimdSynetCompatibilityFmaNoTail, SimdSynetCompatibilityFmaAvoid};

        for (int f = 0; f <= 1; ++f)
        {
            for (int c = 0; c <= 2; ++c)
            {
                result = result && SynetConvert8uTo32fAutoTest(2, C / 2, (int)sqrt(H), (int)sqrt(W), format[f], compatibility[c], f1, f2);
                result = result && SynetConvert8uTo32fAutoTest(1, 3, H / 3 + O, W / 5 - O, format[f], compatibility[c], f1, f2);
            }
        }

        return result;
    }

    bool SynetConvert8uTo32fAutoTest()
    {
        bool result = true;

        result = result && SynetConvert8uTo32fAutoTest(FUNC_C_8U_32F(Simd::Base::SynetConvert8uTo32f), FUNC_C_8U_32F(SimdSynetConvert8uTo32f));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetConvert8uTo32fAutoTest(FUNC_C_8U_32F(Simd::Sse41::SynetConvert8uTo32f), FUNC_C_8U_32F(SimdSynetConvert8uTo32f));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetConvert8uTo32fAutoTest(FUNC_C_8U_32F(Simd::Avx2::SynetConvert8uTo32f), FUNC_C_8U_32F(SimdSynetConvert8uTo32f));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetConvert8uTo32fAutoTest(FUNC_C_8U_32F(Simd::Avx512bw::SynetConvert8uTo32f), FUNC_C_8U_32F(SimdSynetConvert8uTo32f));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncSI
        {
            typedef void(*FuncPtr)(const uint8_t* src, size_t width, size_t height, size_t stride, SimdPixelFormatType pixelFormat,
                const float* lower, const float* upper, float* dst, size_t channels, SimdTensorFormatType dstFormat);

            FuncPtr func;
            String desc;

            FuncSI(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t c, size_t h, size_t w, View::Format src, SimdTensorFormatType dst)
            {
                desc = desc + "[" + ToString(c) + "x" + ToString(h) + "x" + ToString(w) + ":" + ToString(src) + "->" + ToString(dst) + "]";
            }

            void Call(const View& src, const float* lower, const float* upper, size_t channels, Tensor32f& dst) const
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

    bool SynetSetInputAutoTest(const FuncSI& f1, const FuncSI& f2)
    {
        bool result = true;

        View::Format srcFormat[4] = { View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24 };
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

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetSetInputAutoTest(FUNC_SI(Simd::Avx512bw::SynetSetInput), FUNC_SI(SimdSynetSetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetSetInputAutoTest(FUNC_SI(Simd::Neon::SynetSetInput), FUNC_SI(SimdSynetSetInput));
#endif

        return result;
    }

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncRT
        {
            typedef void(*FuncPtr)(size_t n, size_t c, size_t hw, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat);

            FuncPtr func;
            String desc;

            FuncRT(const FuncPtr & f, const String & d) : func(f), desc(d) {}

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

#define FUNC_RT(function) FuncRT(function, #function)

    bool SynetReorderTensorAutoTest(size_t n, size_t c, size_t h, size_t w, SimdTensorFormatType srcFormat, SimdTensorFormatType dstFormat, FuncRT f1, FuncRT f2)
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

    bool SynetReorderImageAutoTest(int mask, const FuncRT & f1, const FuncRT & f2)
    {
        bool result = true;

        //std::cout << "Specified tensor format: " << (int)SimdSynetSpecifyTensorFormat(SimdTensorFormatNchwXc) << std::endl;
        for (SimdTensorFormatType src = SimdTensorFormatNchw; src <= SimdTensorFormatNchw16c && result; src = (SimdTensorFormatType)((int)src + 1))
        {
            for (SimdTensorFormatType dst = SimdTensorFormatNchw; dst <= SimdTensorFormatNchw16c && result; dst = (SimdTensorFormatType)((int)dst + 1))
            {
                if (src == dst || (src >= SimdTensorFormatNchw4c && dst >= SimdTensorFormatNchw4c) || 
                    ((SimdSynetTensorAlignment(src)&mask) == 0) || ((SimdSynetTensorAlignment(dst)&mask) == 0))
                    continue;
                result = result && SynetReorderTensorAutoTest(9, W / 15 + 0, W / 60, W / 30, src, dst, f1, f2);
                result = result && SynetReorderTensorAutoTest(9, W / 15 - 1, W / 58, W / 31, src, dst, f1, f2);
            }
        }

        return result;
    }

    bool SynetReorderImageAutoTest()
    {
        bool result = true;

        result = result && SynetReorderImageAutoTest(TFM_ANY, FUNC_RT(Simd::Base::SynetReorderImage), FUNC_RT(SimdSynetReorderImage));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetReorderImageAutoTest(TFM_128, FUNC_RT(Simd::Sse2::SynetReorderImage), FUNC_RT(SimdSynetReorderImage));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetReorderImageAutoTest(TFM_256, FUNC_RT(Simd::Avx::SynetReorderImage), FUNC_RT(SimdSynetReorderImage));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetReorderImageAutoTest(TFM_512, FUNC_RT(Simd::Avx512f::SynetReorderImage), FUNC_RT(SimdSynetReorderImage));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetReorderImageAutoTest(TFM_128, FUNC_RT(Simd::Neon::SynetReorderImage), FUNC_RT(SimdSynetReorderImage));
#endif

        return result;
    }

    bool SynetReorderFilterAutoTest(int mask, const FuncRT & f1, const FuncRT & f2)
    {
        bool result = true;

        for (SimdTensorFormatType src = SimdTensorFormatOiyx; src <= SimdTensorFormatOyxi16o && result; src = (SimdTensorFormatType)((int)src + 1))
        {
            for (SimdTensorFormatType dst = SimdTensorFormatOiyx; dst <= SimdTensorFormatOyxi16o && result; dst = (SimdTensorFormatType)((int)dst + 1))
            {
                if (src == dst || (src >= SimdTensorFormatOyxi4o && dst >= SimdTensorFormatOyxi4o) ||
                    ((SimdSynetTensorAlignment(src)&mask) == 0) || ((SimdSynetTensorAlignment(dst)&mask) == 0))
                    continue;
                result = result && SynetReorderTensorAutoTest(W * 9 / 30 + 0, W * 7 / 30 + 0, 3, 3, src, dst, f1, f2);
                result = result && SynetReorderTensorAutoTest(W * 9 / 10 - 1, W * 7 / 10 + 1, 1, 1, src, dst, f1, f2);
            }
        }

        return result;
    }

    bool SynetReorderFilterAutoTest()
    {
        bool result = true;

        result = result && SynetReorderFilterAutoTest(TFM_ANY, FUNC_RT(Simd::Base::SynetReorderFilter), FUNC_RT(SimdSynetReorderFilter));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetReorderFilterAutoTest(TFM_128, FUNC_RT(Simd::Sse2::SynetReorderFilter), FUNC_RT(SimdSynetReorderFilter));
#endif 
        
#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetReorderFilterAutoTest(TFM_256, FUNC_RT(Simd::Avx::SynetReorderFilter), FUNC_RT(SimdSynetReorderFilter));
#endif 
        
#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetReorderFilterAutoTest(TFM_512, FUNC_RT(Simd::Avx512f::SynetReorderFilter), FUNC_RT(SimdSynetReorderFilter));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetReorderFilterAutoTest(TFM_128, FUNC_RT(Simd::Neon::SynetReorderFilter), FUNC_RT(SimdSynetReorderFilter));
#endif

        return result;
    }
#endif
}
