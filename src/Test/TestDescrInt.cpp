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
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestRandom.h"

#include "Simd/SimdDescrInt.h"

namespace Test
{
    namespace
    {
        struct FuncDI
        {
            typedef void* (*FuncPtr)(size_t size, size_t depth);

            FuncPtr func;
            String desc;

            FuncDI(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(const String& n, size_t s, size_t d)
            {
                std::stringstream ss;
                ss << desc << "[" << n << "-" << s << "-" << d << "]";
                desc = ss.str();
            }

            void Encode(const void* context, const View& src, View& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntEncode(context, (const float*)src.data, dst.data);
            }

            void Decode(const void* context, const View& src, View& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntDecode(context, src.data, (float*)dst.data);
            }

            void CosineDistance(const void* context, const View& a, const View& b, float& d) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntCosineDistance(context, a.data, b.data, &d);
            }

            void VectorNorm(const void* context, const View& s, float& n) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntVectorNorm(context, s.data, &n);
            }
        };
    }

#define FUNC_DI(function) FuncDI(function, #function)

    //-------------------------------------------------------------------------------------------------

    bool DescrIntEncodeAutoTest(size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("Encode", size, depth);
        f2.Update("Encode", size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(src, -17.0, 13.0);

        View dst1(SimdDescrIntEncodedSize(context1), 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(SimdDescrIntEncodedSize(context2), 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom(dst1, 1, 1);
        FillRandom(dst2, 2, 2);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Encode(context1, src, dst1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Encode(context2, src, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    bool DescrIntEncodeAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        for (size_t depth = 8; depth <= 8; depth++)
        {
            result = result && DescrIntEncodeAutoTest(256, depth, f1, f2);
            result = result && DescrIntEncodeAutoTest(512, depth, f1, f2);
            result = result && DescrIntEncodeAutoTest(1024, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntEncodeAutoTest()
    {
        bool result = true;

        result = result && DescrIntEncodeAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

//#ifdef SIMD_SSE41_ENABLE
//        if (Simd::Sse41::Enable)
//            result = result && DescrIntEncodeAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
//#endif 
//
//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable)
//            result = result && DescrIntEncodeAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
//#endif 
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable)
//            result = result && DescrIntEncodeAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
//#endif 
//
//#ifdef SIMD_NEON_ENABLE
//        if (Simd::Neon::Enable)
//            result = result && DescrIntEncodeAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
//#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    bool DescrIntDecodeAutoTest(size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("Decode", size, depth);
        f2.Update("Decode", size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View orig(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(orig, -17.0, 13.0);

        View src(SimdDescrIntEncodedSize(context2), 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        SimdDescrIntEncode(context2, (float*)orig.data, src.data);

        View dst1(SimdDescrIntDecodedSize(context1), 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(SimdDescrIntDecodedSize(context2), 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(dst1, 1.0f, 1.0f);
        FillRandom32f(dst2, 2.0f, 2.0f);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Decode(context1, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Decode(context2, src, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dst1, dst2, EPS*EPS, true, 64);

        return result;
    }

    bool DescrIntDecodeAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        for (size_t depth = 8; depth <= 8; depth++)
        {
            result = result && DescrIntDecodeAutoTest(256, depth, f1, f2);
            result = result && DescrIntDecodeAutoTest(512, depth, f1, f2);
            result = result && DescrIntDecodeAutoTest(1024, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntDecodeAutoTest()
    {
        bool result = true;

        result = result && DescrIntDecodeAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

        //#ifdef SIMD_SSE41_ENABLE
        //        if (Simd::Sse41::Enable)
        //            result = result && DescrIntDecodeAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 
        //
        //#ifdef SIMD_AVX2_ENABLE
        //        if (Simd::Avx2::Enable)
        //            result = result && DescrIntDecodeAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 
        //
        //#ifdef SIMD_AVX512BW_ENABLE
        //        if (Simd::Avx512bw::Enable)
        //            result = result && DescrIntDecodeAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 
        //
        //#ifdef SIMD_NEON_ENABLE
        //        if (Simd::Neon::Enable)
        //            result = result && DescrIntDecodeAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 

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
        SimdDescrIntEncode(context2, (float*)oA.data, a.data);
        SimdDescrIntEncode(context2, (float*)oB.data, b.data);

        float d1 = 1.0f, d2 = 2.0f, d3 = 3.0f;
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.CosineDistance(context1, a, b, d1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.CosineDistance(context2, a, b, d2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = Compare(d1, d2, EPS*EPS, true, DifferenceRelative, "d1 & d2");

        ::SimdCosineDistance32f((float*)oA.data, (float*)oB.data, size, &d3);
        result = Compare(d2, d3, EPS, true, DifferenceRelative, "d2 & d3");

        return result;
    }

    bool DescrIntCosineDistanceAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        for (size_t depth = 8; depth <= 8; depth++)
        {
            result = result && DescrIntCosineDistanceAutoTest(256, depth, f1, f2);
            result = result && DescrIntCosineDistanceAutoTest(512, depth, f1, f2);
            result = result && DescrIntCosineDistanceAutoTest(1024, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntCosineDistanceAutoTest()
    {
        bool result = true;

        result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

        //#ifdef SIMD_SSE41_ENABLE
        //        if (Simd::Sse41::Enable)
        //            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 
        //
        //#ifdef SIMD_AVX2_ENABLE
        //        if (Simd::Avx2::Enable)
        //            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 
        //
        //#ifdef SIMD_AVX512BW_ENABLE
        //        if (Simd::Avx512bw::Enable)
        //            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 
        //
        //#ifdef SIMD_NEON_ENABLE
        //        if (Simd::Neon::Enable)
        //            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    bool DescrIntVectorNormAutoTest(size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("VectorNorm", size, depth);
        f2.Update("VectorNorm", size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View oA(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(oA, -17.0, 13.0);

        View a(SimdDescrIntEncodedSize(context2), 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        SimdDescrIntEncode(context2, (float*)oA.data, a.data);

        float n1 = 1.0f, n2 = 2.0f;
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.VectorNorm(context1, a, n1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.VectorNorm(context2, a, n2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = Compare(n1, n2, EPS * EPS, true, DifferenceRelative, "n1 & n2");

        return result;
    }

    bool DescrIntVectorNormAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        for (size_t depth = 8; depth <= 8; depth++)
        {
            result = result && DescrIntVectorNormAutoTest(256, depth, f1, f2);
            result = result && DescrIntVectorNormAutoTest(512, depth, f1, f2);
            result = result && DescrIntVectorNormAutoTest(1024, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntVectorNormAutoTest()
    {
        bool result = true;

        result = result && DescrIntVectorNormAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

        //#ifdef SIMD_SSE41_ENABLE
        //        if (Simd::Sse41::Enable)
        //            result = result && DescrIntVectorNormAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 
        //
        //#ifdef SIMD_AVX2_ENABLE
        //        if (Simd::Avx2::Enable)
        //            result = result && DescrIntVectorNormAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 
        //
        //#ifdef SIMD_AVX512BW_ENABLE
        //        if (Simd::Avx512bw::Enable)
        //            result = result && DescrIntVectorNormAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 
        //
        //#ifdef SIMD_NEON_ENABLE
        //        if (Simd::Neon::Enable)
        //            result = result && DescrIntVectorNormAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 

        return result;
    }
}