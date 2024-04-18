/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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

#include "Simd/SimdDescrInt.h"

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

        for (size_t depth = 4; depth <= 8; depth++)
        {
            result = result && DescrIntCosineDistancesMxNpAutoTest(256, 128, 256, depth, f1, f2);
            result = result && DescrIntCosineDistancesMxNpAutoTest(128, 128, 512, depth, f1, f2);
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
}
