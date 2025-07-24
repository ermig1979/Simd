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
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestRandom.h"
#include "Test/TestTensor.h"
#include "Test/TestOptions.h"

#include "Simd/SimdDescrInt.h"
#include "Simd/SimdParallel.hpp"

namespace Test
{
    typedef std::vector<uint8_t*> U8Ptrs;

    static void InitEncoded(const void* c, View& u8, size_t h, float lo, float hi, int gap = 0, U8Ptrs* u8p = NULL)
    {
        View f32(SimdDescrIntDecodedSize(c), h, View::Float, NULL, 1);
        FillRandom32f(f32, lo, hi);
        u8.Recreate(SimdDescrIntEncodedSize(c) + gap, h, View::Gray8, NULL, 1);
        if (u8p)
            u8p->resize(h);
        for (size_t r = 0; r < h; r++)
        {
            uint8_t* dst = u8.Row<uint8_t>(r) + Random(gap);
            ::SimdDescrIntEncode32f(c, f32.Row<float>(r), dst);
            if (u8p)
                u8p->at(r) = dst;
        }
        TEST_ALIGN(SimdDescrIntDecodedSize(c));
        TEST_ALIGN(SIMD_ALIGN);
    }

    //-------------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncDI
        {
            typedef void* (*FuncPtr)(size_t size, size_t depth);

            FuncPtr func;
            String desc;

            FuncDI(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(const String& name, size_t s, size_t d)
            {
                std::stringstream ss;
                ss << desc << "[" << name << "-" << s << "-" << d << "]";
                desc = ss.str();
            }

            void Update(const String& name, size_t n, size_t s, size_t d)
            {
                std::stringstream ss;
                ss << desc << "[" << name << "-" << n << "-" << s << "-" << d << "]";
                desc = ss.str();
            }

            void Update(const String& name, size_t n, size_t m, size_t s, size_t d)
            {
                std::stringstream ss;
                ss << desc << "[" << name << "-" << n << "-" << m << "-" << s << "-" << d << "]";
                desc = ss.str();
            }

            void Encode32f(const void* context, const View& src, View& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntEncode32f(context, (const float*)src.data, dst.data);
            }

            void Encode16f(const void* context, const View& src, View& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntEncode16f(context, (const uint16_t*)src.data, dst.data);
            }

            void Decode32f(const void* context, const View& src, View& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntDecode32f(context, src.data, (float*)dst.data);
            }

            void Decode16f(const void* context, const View& src, View& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntDecode16f(context, src.data, (uint16_t*)dst.data);
            }

            void CosineDistance(const void* context, const View& a, const View& b, float& d) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntCosineDistance(context, a.data, b.data, &d);
            }

            void CosineDistancesMxNa(const void* context, const U8Ptrs& a, const U8Ptrs& b, Tensor32f& d) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntCosineDistancesMxNa(context, a.size(), b.size(), a.data(), b.data(), d.Data());
            }

            void CosineDistancesMxNp(const void* context, const View& a, const View& b, Tensor32f & d) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntCosineDistancesMxNp(context, a.height, b.height, a.data, b.data, d.Data());
            }
        };
    }

#define FUNC_DI(function) FuncDI(function, #function)

    //-------------------------------------------------------------------------------------------------

    bool DescrIntEncode32fAutoTest(size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("Encode32f", size, depth);
        f2.Update("Encode32f", size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(src, -17.0, 13.0);

        size_t encSize = SimdDescrIntEncodedSize(context1);
        View dst1(encSize, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(encSize, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));

        Fill(dst1, 1);
        Fill(dst2, 2);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Encode32f(context1, src, dst1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Encode32f(context2, src, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

#if defined(SIMD_X64_ENABLE)
        result = result && Compare(dst1, dst2, 0, true, 64);
#endif

        return result;
    }

    bool DescrIntEncode32fAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        size_t size = Simd::Min(H * W, 128 * 256);
        for (size_t depth = 4; depth <= 8; depth++)
        {
            //result = result && DescrIntEncode32fAutoTest(256, depth, f1, f2);
            //result = result && DescrIntEncode32fAutoTest(512, depth, f1, f2);
            result = result && DescrIntEncode32fAutoTest(size, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntEncode32fAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && DescrIntEncode32fAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && DescrIntEncode32fAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && DescrIntEncode32fAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && DescrIntEncode32fAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && DescrIntEncode32fAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    bool DescrIntEncode16fAutoTest(size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("Encode16f", size, depth);
        f2.Update("Encode16f", size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View orig(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(orig, -17.0, 13.0);

        View src(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        SimdFloat32ToFloat16((float*)orig.data, size, (uint16_t*)src.data);

        size_t encSize = SimdDescrIntEncodedSize(context1);
        View dst1(encSize, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(encSize, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));

        Fill(dst1, 1);
        Fill(dst2, 2);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Encode16f(context1, src, dst1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Encode16f(context2, src, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

#if defined(SIMD_X64_ENABLE)
        result = result && Compare(dst1, dst2, 0, true, 64);
#endif

        return result;
    }

    bool DescrIntEncode16fAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        size_t size = Simd::Min(H * W, 128 * 256);
        for (size_t depth = 4; depth <= 8; depth++)
        {
            //result = result && DescrIntEncode16fAutoTest(256, depth, f1, f2);
            //result = result && DescrIntEncode16fAutoTest(512, depth, f1, f2);
            result = result && DescrIntEncode16fAutoTest(size, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntEncode16fAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && DescrIntEncode16fAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && DescrIntEncode16fAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && DescrIntEncode16fAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && DescrIntEncode16fAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && DescrIntEncode16fAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    bool DescrIntDecode32fAutoTest(size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("Decode32f", size, depth);
        f2.Update("Decode32f", size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View orig(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        const float lo = -17.0f, hi = 13.0f, eps = (hi - lo) * 0.0021f * (1 << (8 - depth));
        FillRandom32f(orig, lo, hi);

        View src(SimdDescrIntEncodedSize(context2), 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        SimdDescrIntEncode32f(context2, (float*)orig.data, src.data);

        View dst1(SimdDescrIntDecodedSize(context1), 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(SimdDescrIntDecodedSize(context2), 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(dst1, 1.0f, 1.0f);
        FillRandom32f(dst2, 2.0f, 2.0f);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Decode32f(context1, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Decode32f(context2, src, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dst1, dst2, EPS * 2.0f, true, 64, true, "dst1 & dst2");

        result = result && Compare(dst1, orig, eps, true, 64, false, "dst1 & orig");

        return result;
    }

    bool DescrIntDecode32fAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        size_t size = Simd::Min(H * W, 128 * 256);
        for (size_t depth = 4; depth <= 8; depth++)
        {
            //result = result && DescrIntDecode32fAutoTest(256, depth, f1, f2);
            //result = result && DescrIntDecode32fAutoTest(512, depth, f1, f2);
            result = result && DescrIntDecode32fAutoTest(size, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntDecode32fAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && DescrIntDecode32fAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && DescrIntDecode32fAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && DescrIntDecode32fAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && DescrIntDecode32fAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && DescrIntDecode32fAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    bool DescrIntDecode16fAutoTest(size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("Decode16f", size, depth);
        f2.Update("Decode16f", size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View orig(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(orig, -17.0, 13.0);

        View src(SimdDescrIntEncodedSize(context2), 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        SimdDescrIntEncode32f(context2, (float*)orig.data, src.data);

        View dst1(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));

        View dstF1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstF2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        Simd::Fill(dst1, 1);
        Simd::Fill(dst2, 2);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Decode16f(context1, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Decode16f(context2, src, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        SimdFloat16ToFloat32((const uint16_t*)dst1.data, size, (float*)dstF1.data);
        SimdFloat16ToFloat32((const uint16_t*)dst2.data, size, (float*)dstF2.data);

        result = result && Compare(dstF1, dstF2, EPS, true, 64, true, "dst1 & dst2");

        return result;
    }

    bool DescrIntDecode16fAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        size_t size = Simd::Min(H * W, 128 * 256);
        for (size_t depth = 4; depth <= 8; depth++)
        {
            //result = result && DescrIntDecode16fAutoTest(256, depth, f1, f2);
            //result = result && DescrIntDecode16fAutoTest(512, depth, f1, f2);
            result = result && DescrIntDecode16fAutoTest(size, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntDecode16fAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && DescrIntDecode16fAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && DescrIntDecode16fAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && DescrIntDecode16fAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && DescrIntDecode16fAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && DescrIntDecode16fAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    bool DescrIntCosineDistanceAutoTest(size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("CosineDistance", size, depth);
        f2.Update("CosineDistance", size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View oA(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View oB(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(oA, -17.0, 13.0);
        FillRandom32f(oB, -15.0, 17.0);

        View a(SimdDescrIntEncodedSize(context2), 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        View b(SimdDescrIntEncodedSize(context2), 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        SimdDescrIntEncode32f(context2, (float*)oA.data, a.data);
        SimdDescrIntEncode32f(context2, (float*)oB.data, b.data);

        float d1 = 1.0f, d2 = 2.0f, d3 = 3.0f;
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.CosineDistance(context1, a, b, d1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.CosineDistance(context2, a, b, d2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(d1, d2, EPS * 0.1f * (1 << (8 - depth)), true, DifferenceRelative, "d1 & d2");

        if (size >= 256)
        {
            ::SimdCosineDistance32f((float*)oA.data, (float*)oB.data, size, &d3);
            result = result && Compare(d2, d3, EPS * 1.0f * (1 << (8 - depth)), true, DifferenceRelative, "d2 & d3");
        }

        return result;
    }

    bool DescrIntCosineDistanceAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        size_t size = Simd::Min(H * W, 128 * 256);
        for (size_t depth = 4; depth <= 8; depth++)
        {
            //result = result && DescrIntCosineDistanceAutoTest(256, depth, f1, f2);
            //result = result && DescrIntCosineDistanceAutoTest(512, depth, f1, f2);
            result = result && DescrIntCosineDistanceAutoTest(size, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntCosineDistanceAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#if defined(SIMD_AVX512VNNI_ENABLE) && !defined(SIMD_AMX_EMULATE)
        if (Simd::Avx512vnni::Enable && TestAvx512vnni())
            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Avx512vnni::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

#if defined(SIMD_AMXBF16_ENABLE)
        if (Simd::AmxBf16::Enable && TestAmxBf16())
            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::AmxBf16::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif
        
#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    bool DescrIntCosineDistancesMxNaAutoTest(size_t M, size_t N, size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("CosineDistancesMxNa", M, N, size, depth);
        f2.Update("CosineDistancesMxNa", M, N, size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View ai, bi;
        U8Ptrs a, b;
        InitEncoded(context2, ai, M, -17.0, 13.0, 1024, &a);
        InitEncoded(context2, bi, N, -15.0, 17.0, 1024, &b);

        Tensor32f d1({ M, N, });
        Tensor32f d2({ M, N, });
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.CosineDistancesMxNa(context1, a, b, d1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.CosineDistancesMxNa(context2, a, b, d2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = Compare(d1, d2, EPS * EPS * 2, true, 32, DifferenceAbsolute);

        return result;
    }

    bool DescrIntCosineDistancesMxNaAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        for (size_t depth = 4; depth <= 8; depth++)
        {
            result = result && DescrIntCosineDistancesMxNaAutoTest(256, 128, 256, depth, f1, f2);
            result = result && DescrIntCosineDistancesMxNaAutoTest(128, 128, 512, depth, f1, f2);
#if !(defined(__GNUC__) && defined(__clang__))
            result = result && DescrIntCosineDistancesMxNaAutoTest(127, 129, 520, depth, f1, f2);
            result = result && DescrIntCosineDistancesMxNaAutoTest(31, 33, 10000, depth, f1, f2);
#endif
        }

        return result;
    }

    bool DescrIntCosineDistancesMxNaAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif
        
#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif
        
#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

#if defined(SIMD_AVX512VNNI_ENABLE) && !defined(SIMD_AMX_EMULATE)
        if (Simd::Avx512vnni::Enable && TestAvx512vnni())
            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Avx512vnni::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

#if defined(SIMD_AMXBF16_ENABLE)
        if (Simd::AmxBf16::Enable && TestAmxBf16())
            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::AmxBf16::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif
        
#if defined(SIMD_NEON_ENABLE)
        if (Simd::Neon::Enable && TestNeon())
            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    bool DescrIntCosineDistancesMxNpAutoTest(size_t M, size_t N, size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("CosineDistancesMxNp", M, N, size, depth);
        f2.Update("CosineDistancesMxNp", M, N, size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View a, b;
        InitEncoded(context2, a, M, -17.0, 13.0, 0, NULL);
        InitEncoded(context2, b, N, -15.0, 17.0, 0, NULL);

        Tensor32f d1({ M, N, });
        Tensor32f d2({ M, N, });
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.CosineDistancesMxNp(context1, a, b, d1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.CosineDistancesMxNp(context2, a, b, d2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = Compare(d1, d2, EPS*EPS * 2, true, 32, DifferenceAbsolute);

        return result;
    }

    bool DescrIntCosineDistancesMxNpAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        for (size_t depth = 7; depth <= 8; depth++)
        {
            result = result && DescrIntCosineDistancesMxNpAutoTest(256, 128, 256, depth, f1, f2);
            result = result && DescrIntCosineDistancesMxNpAutoTest(128, 128, 512, depth, f1, f2);
            //result = result && DescrIntCosineDistancesMxNpAutoTest(1, 10*1024*1024, 512, depth, f1, f2);
            //result = result && DescrIntCosineDistancesMxNpAutoTest(10, 10 * 1024 * 1024, 512, depth, f1, f2);
            //result = result && DescrIntCosineDistancesMxNpAutoTest(100, 10 * 1024 * 1024, 512, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntCosineDistancesMxNpAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

#if defined(SIMD_AVX512VNNI_ENABLE) && !defined(SIMD_AMX_EMULATE)
        if (Simd::Avx512vnni::Enable && TestAvx512vnni())
            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Avx512vnni::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

#if defined(SIMD_AMXBF16_ENABLE)
        if (Simd::AmxBf16::Enable && TestAmxBf16())
            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::AmxBf16::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

#if defined(SIMD_NEON_ENABLE)
        if (Simd::Neon::Enable && TestNeon())
            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    static inline void SetRandomDescriptor(const float* rnd, size_t size, float mainRange, int seed, float noiseRange, size_t noiseTimes, float* dst)
    {
        memset(dst, 0, size * sizeof(float));
        SimdNeuralAddVectorMultipliedByValue(rnd + seed, size, &mainRange, dst);
        for(size_t t = 0; t < noiseTimes; ++t)
            SimdNeuralAddVectorMultipliedByValue(rnd + (Rand() & INT16_MAX), size, &noiseRange, dst);
    }

    typedef std::pair<size_t, float> Pair;
    typedef std::vector<Pair> Pairs;
    typedef std::vector<Pairs> Pairss;

    inline String FuncDescr(size_t M, size_t N, size_t size, size_t depth = 0)
    {
        std::stringstream ss;
        ss << "[" << M << "x" << N << "x" << size;
        if (depth)
            ss << "-" << depth;
        ss << "]";
        return ss.str();
    }

    static void CompareDescriptors32f(size_t M, size_t beg, size_t end, size_t size, const float* const* a, const float* const* b, float threshold, Pairss &result)
    {
        result.resize(M);
        for (size_t i = beg; i < end; ++i)
        {
            for (size_t j = 0; j < M; ++j)
            {
                float distance;
                SimdCosineDistance32f(a[j], b[i], size, &distance);
                if (distance < threshold)
                    result[j].push_back(Pair(i, distance));
            }
        }
    }

    static Pairss CompareDescriptors32f(const Options& options, size_t M, size_t N, size_t size, const float* const* a, const float* const* b, float threshold)
    {
        TEST_PERFORMANCE_TEST(String(__FUNCTION__) + FuncDescr(M, N, size));
        Pairss result(M);
        if (options.testThreads)
        {
            std::vector<Pairss> buffer(options.testThreads);
            Simd::Parallel(0, N, [&](size_t thread, size_t begin, size_t end)
                {
                    CompareDescriptors32f(M, begin, end, size, a, b, threshold, buffer[thread]);
                }, options.testThreads, 1);
            for (size_t t = 0; t < options.testThreads; ++t)
                for (size_t j = 0; j < M; ++j)
                    for (size_t i = 0; i < buffer[t][j].size(); ++i)
                        result[j].push_back(buffer[t][j][i]);            
        }
        else
        {
            CompareDescriptors32f(M, 0, N, size, a, b, threshold, result);
        }
        return result;
    }

    static void CompareDescriptors16b(size_t M, size_t beg, size_t end, size_t size, const uint16_t* const* a, const uint16_t* const* b, float threshold, Pairss& result)
    {
        result.resize(M);
        const size_t step = 256;
        Tensor32f distances(Shp(M, step));
        for (size_t i = beg; i < end; i += step)
        {
            size_t curr = std::min(i + step, end) - i;
            SimdCosineDistancesMxNa16f(M, curr, size, a, b + i, distances.Data());
            for (size_t j = 0; j < M; ++j)
            {
                const float* dist = distances.Data(Shp(j, 0));
                for (size_t c = 0; c < curr; ++c)
                    if (dist[c]< threshold)
                        result[j].push_back(Pair(i + c, dist[c]));
            }
        }
    }

    static Pairss CompareDescriptors16b(const Options& options, size_t M, size_t N, size_t size, const uint16_t* const* a, const uint16_t* const* b, float threshold)
    {
        TEST_PERFORMANCE_TEST(String(__FUNCTION__) + FuncDescr(M, N, size));
        Pairss result(M);
        if (options.testThreads)
        {
            std::vector<Pairss> buffer(options.testThreads);
            Simd::Parallel(0, N, [&](size_t thread, size_t begin, size_t end)
                {
                    CompareDescriptors16b(M, begin, end, size, a, b, threshold, buffer[thread]);
                }, options.testThreads, 256);
            for (size_t t = 0; t < options.testThreads; ++t)
                for (size_t j = 0; j < M; ++j)
                    for (size_t i = 0; i < buffer[t][j].size(); ++i)
                        result[j].push_back(buffer[t][j][i]);
        }
        else
        {
            CompareDescriptors16b(M, 0, N, size, a, b, threshold, result);
        }
        return result;
    }

    static void CompareDescriptors8u(const void* context, size_t M, size_t beg, size_t end, size_t size, const uint8_t* const* a, const uint8_t* const* b, float threshold, Pairss& result)
    {
        result.resize(M);
        const size_t step = 256;
        Tensor32f distances(Shp(M, step));
        for (size_t i = beg; i < end; i += step)
        {
            size_t curr = std::min(i + step, end) - i;
            SimdDescrIntCosineDistancesMxNa(context, M, curr, a, b + i, distances.Data());
            for (size_t j = 0; j < M; ++j)
            {
                const float* dist = distances.Data(Shp(j, 0));
                for (size_t c = 0; c < curr; ++c)
                    if (dist[c] < threshold)
                        result[j].push_back(Pair(i + c, dist[c]));
            }
        }
    }

    static Pairss CompareDescriptors8u(const Options& options, size_t depth, const void *context, size_t M, size_t N, size_t size, const uint8_t* const* a, const uint8_t* const* b, float threshold)
    {
        TEST_PERFORMANCE_TEST(String(__FUNCTION__) + FuncDescr(M, N, size, depth));
        Pairss result(M);
        if (options.testThreads)
        {
            std::vector<Pairss> buffer(options.testThreads);
            Simd::Parallel(0, N, [&](size_t thread, size_t begin, size_t end)
                {
                    CompareDescriptors8u(context, M, begin, end, size, a, b, threshold, buffer[thread]);
                }, options.testThreads, 1024);
            for (size_t t = 0; t < options.testThreads; ++t)
                for (size_t j = 0; j < M; ++j)
                    for (size_t i = 0; i < buffer[t][j].size(); ++i)
                        result[j].push_back(buffer[t][j][i]);
        }
        else
        {
            CompareDescriptors8u(context, M, 0, N, size, a, b, threshold, result);
        }
        return result;
    }

    static inline bool Compare(const Pairss& a, const Pairss& b)
    {
        if (a.size() != b.size())
        {
            TEST_LOG_SS(Error, "a.size() " << a.size() << " != b.size() " << b.size() << " !");
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i)
        {
            if (a[i].size() != b[i].size())
            {
                TEST_LOG_SS(Error, "a[" << i << "].size() " << a[i].size() << " != b[" << i << "].size() " << b[i].size() << " !");
                return false;
            }
            for (size_t j = 0; j < a[i].size(); ++j)
                if (a[i][j].first != b[i][j].first)
                    return false;
        }
        return true;
    }

    bool DescrIntCosineDistancesMxNaSpecialTest(const Options& options, size_t M, size_t N, size_t size, size_t depth)
    {
        bool result = true;

        TEST_LOG_SS(Info, "DescrIntCosineDistancesMxNa " << std::max<ptrdiff_t>(options.testThreads, 1) << " threads special test [" << M << "x" << N << "x" << size << "]:");

        const float threshold = 0.5f;

        Srand(0);

        Tensor32f rnd(Shp(std::max(N, size_t(INT16_MAX)) + size));
        FillRandom(rnd.Data(), rnd.Size(), -1.0f, 1.0f);

        void* context = SimdDescrIntInit(size, depth);
        if (!context)
            return false;
        size_t encSize = SimdDescrIntEncodedSize(context);

        Tensor32f a32f(Shp(M, size)), b32f(Shp(N, size));
        Tensor16u a16b(Shp(M, size)), b16b(Shp(N, size));
        Tensor8u a8u(Shp(M, encSize)), b8u(Shp(N, encSize));
        FloatPtrs a32fp(M), b32fp(N);
        UInt16Ptrs a16bp(M), b16bp(N);
        UInt8Ptrs a8up(M), b8up(N);
        for (size_t i = 0; i < M; ++i)
        {
            a32fp[i] = a32f.Data(Shp(i, 0));
            a16bp[i] = a16b.Data(Shp(i, 0));
            a8up[i] = a8u.Data(Shp(i, 0));
            SetRandomDescriptor(rnd.Data(), size, 1.0f, Random(int(N)), 0.3f, 3, a32fp[i]);
            SimdFloat32ToFloat16(a32fp[i], size, a16bp[i]);
            SimdDescrIntEncode32f(context, a32fp[i], a8up[i]);
        }
        for (size_t i = 0; i < N; ++i)
        {
            b32fp[i] = b32f.Data(Shp(i, 0));
            b16bp[i] = b16b.Data(Shp(i, 0));
            b8up[i] = b8u.Data(Shp(i, 0));
            SetRandomDescriptor(rnd.Data(), size, 1.0f, int(i), 0.3f, 3, b32fp[i]);
            SimdFloat32ToFloat16(b32fp[i], size, b16bp[i]);
            SimdDescrIntEncode32f(context, b32fp[i], b8up[i]);
        }

        Pairss res32f, res16b, res8u;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(res32f = CompareDescriptors32f(options, M, N, size, a32fp.data(), b32fp.data(), threshold));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(res16b = CompareDescriptors16b(options, M, N, size, a16bp.data(), b16bp.data(), threshold));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(res8u = CompareDescriptors8u(options, depth, context, M, N, size, a8up.data(), b8up.data(), threshold));

        SimdRelease(context);

        //if (!Compare(res32f, res16b))
        //    return false;

        //if (!Compare(res32f, res8u))
        //    return false;

        return true;
    }

    bool DescrIntCosineDistancesMxNaSpecialTest(const Options & options)
    {
        bool result = true;

#if defined(NDEBUG)
#if defined(SIMD_AMXBF16_ENABLE)
#if 0
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 1, 10000000, 512, 7);
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 10, 10000000, 512, 7);
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 100, 10000000, 512, 7);
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 1000, 10000000, 512, 7);
#endif
#if 1
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 1, 10000000, 256, 7);
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 10, 10000000, 256, 7);
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 100, 10000000, 256, 7);
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 1000, 10000000, 256, 7);
#endif
#else
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 1, 1000000, 512, 7);
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 8, 1000000, 512, 7);
#endif
#else
        result = result && DescrIntCosineDistancesMxNaSpecialTest(options, 10, 1024, 512, 7);
#endif

#ifdef TEST_PERFORMANCE_TEST_ENABLE
        TEST_LOG_SS(Info, PerformanceMeasurerStorage::s_storage.ConsoleReport(false, true));
        PerformanceMeasurerStorage::s_storage.Clear();
#endif

        return result;
    }
}
