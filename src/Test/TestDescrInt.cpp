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
            ::SimdDescrIntEncode(c, f32.Row<float>(r), dst);
            if (u8p)
                u8p->at(r) = dst;
        }
        TEST_ALIGN(SimdDescrIntDecodedSize(c));
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

            void VectorNorm(const void* context, const View& s, float& n) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntVectorNorm(context, s.data, &n);
            }

            void VectorNormsNa(const void* context, const U8Ptrs& s, Tensor32f& d) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntVectorNormsNa(context, s.size(), s.data(), d.Data());
            }

            void VectorNormsNp(const void* context, const View& s, Tensor32f& d) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntVectorNormsNp(context, s.height, s.data, d.Data());
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

        size_t encSize = SimdDescrIntEncodedSize(context1);
        View dst1(encSize, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(encSize, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));

        Fill(dst1, 1);
        Fill(dst2, 2);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Encode(context1, src, dst1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Encode(context2, src, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

#if defined(SIMD_X64_ENABLE)
        result = result && Compare(dst1, dst2, 0, true, 64);
#endif

        return result;
    }

    bool DescrIntEncodeAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        size_t size = Simd::Min(H * W, 128 * 256);
        for (size_t depth = 6; depth <= 8; depth++)
        {
            //result = result && DescrIntEncodeAutoTest(256, depth, f1, f2);
            //result = result && DescrIntEncodeAutoTest(512, depth, f1, f2);
            result = result && DescrIntEncodeAutoTest(size, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntEncodeAutoTest()
    {
        bool result = true;

        result = result && DescrIntEncodeAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DescrIntEncodeAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && DescrIntEncodeAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DescrIntEncodeAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 

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

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool DescrIntDecodeAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        size_t size = Simd::Min(H * W, 128 * 256);
        for (size_t depth = 6; depth <= 8; depth++)
        {
            //result = result && DescrIntDecodeAutoTest(256, depth, f1, f2);
            //result = result && DescrIntDecodeAutoTest(512, depth, f1, f2);
            result = result && DescrIntDecodeAutoTest(size, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntDecodeAutoTest()
    {
        bool result = true;

        result = result && DescrIntDecodeAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DescrIntDecodeAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && DescrIntDecodeAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DescrIntDecodeAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
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

        result = result && Compare(d1, d2, EPS * 0.1f * (1 << (8 - depth)), true, DifferenceRelative, "d1 & d2");

        ::SimdCosineDistance32f((float*)oA.data, (float*)oB.data, size, &d3);
        result = result && Compare(d2, d3, EPS * (1 << (8 - depth)), true, DifferenceRelative, "d2 & d3");

        return result;
    }

    bool DescrIntCosineDistanceAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        size_t size = Simd::Min(H * W, 128 * 256);
        for (size_t depth = 6; depth <= 8; depth++)
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

        result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
        //#ifdef SIMD_NEON_ENABLE
        //        if (Simd::Neon::Enable)
        //            result = result && DescrIntCosineDistanceAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif 

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

        for (size_t depth = 6; depth <= 8; depth++)
        {
            result = result && DescrIntCosineDistancesMxNaAutoTest(256, 128, 256, depth, f1, f2);
            result = result && DescrIntCosineDistancesMxNaAutoTest(128, 128, 512, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntCosineDistancesMxNaAutoTest()
    {
        bool result = true;

        result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif
        
#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif
        
#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif
        
        //#if defined(SIMD_NEON_ENABLE)
        //        if (Simd::Neon::Enable)
        //            result = result && DescrIntCosineDistancesMxNaAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif

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

        for (size_t depth = 6; depth <= 8; depth++)
        {
            result = result && DescrIntCosineDistancesMxNpAutoTest(256, 128, 256, depth, f1, f2);
            result = result && DescrIntCosineDistancesMxNpAutoTest(128, 128, 512, depth, f1, f2);
            result = result && DescrIntCosineDistancesMxNpAutoTest(64, 128, 10240, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntCosineDistancesMxNpAutoTest()
    {
        bool result = true;

        result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif

//#if defined(SIMD_NEON_ENABLE)
//        if (Simd::Neon::Enable)
//            result = result && DescrIntCosineDistancesMxNpAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
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

        result = Compare(n1, n2, EPS * EPS * (1 << (8 - depth)), true, DifferenceRelative, "n1 & n2");

        return result;
    }

    bool DescrIntVectorNormAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        size_t size = Simd::Min(H * W, 128 * 256);
        for (size_t depth = 6; depth <= 8; depth++)
        {
            //result = result && DescrIntVectorNormAutoTest(256, depth, f1, f2);
            //result = result && DescrIntVectorNormAutoTest(512, depth, f1, f2);
            result = result && DescrIntVectorNormAutoTest(size, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntVectorNormAutoTest()
    {
        bool result = true;

        result = result && DescrIntVectorNormAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DescrIntVectorNormAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif 
        
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

    //-------------------------------------------------------------------------------------------------

    bool DescrIntVectorNormsNaAutoTest(size_t N, size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("VectorNormsNa", N, size, depth);
        f2.Update("VectorNormsNa", N, size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View ai;
        U8Ptrs a;
        InitEncoded(context2, ai, N, -17.0, 13.0, 1024, &a);

        Tensor32f d1({ N, });
        Tensor32f d2({ N, });
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.VectorNormsNa(context1, a, d1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.VectorNormsNa(context2, a, d2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = Compare(d1, d2, EPS * EPS * (1 << (8 - depth)), true, 32, DifferenceRelative);

        return result;
    }

    bool DescrIntVectorNormsNaAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        for (size_t depth = 6; depth <= 8; depth++)
        {
            result = result && DescrIntVectorNormsNaAutoTest(H, W, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntVectorNormsNaAutoTest()
    {
        bool result = true;

        result = result && DescrIntVectorNormsNaAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DescrIntVectorNormsNaAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif
        
        //#ifdef SIMD_AVX2_ENABLE
        //        if (Simd::Avx2::Enable)
        //            result = result && DescrIntVectorNormsNaAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif
        //
        //#ifdef SIMD_AVX512BW_ENABLE
        //        if (Simd::Avx512bw::Enable)
        //            result = result && DescrIntVectorNormsNaAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif
        //
        //#if defined(SIMD_NEON_ENABLE)
        //        if (Simd::Neon::Enable)
        //            result = result && DescrIntVectorNormsNaAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    bool DescrIntVectorNormsNpAutoTest(size_t N, size_t size, size_t depth, FuncDI f1, FuncDI f2)
    {
        bool result = true;

        f1.Update("VectorNormsNp", N, size, depth);
        f2.Update("VectorNormsNp", N, size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View a;
        InitEncoded(context2, a, N, -17.0, 13.0, 0, NULL);

        Tensor32f d1({ N, });
        Tensor32f d2({ N, });
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.VectorNormsNp(context1, a, d1));
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.VectorNormsNp(context2, a, d2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = Compare(d1, d2, EPS * EPS * (1 << (8 - depth)), true, 32, DifferenceRelative);

        return result;
    }

    bool DescrIntVectorNormsNpAutoTest(const FuncDI& f1, const FuncDI& f2)
    {
        bool result = true;

        for (size_t depth = 6; depth <= 8; depth++)
        {
            result = result && DescrIntVectorNormsNpAutoTest(H, W, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntVectorNormsNpAutoTest()
    {
        bool result = true;

        result = result && DescrIntVectorNormsNpAutoTest(FUNC_DI(Simd::Base::DescrIntInit), FUNC_DI(SimdDescrIntInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && DescrIntVectorNormsNpAutoTest(FUNC_DI(Simd::Sse41::DescrIntInit), FUNC_DI(SimdDescrIntInit));
#endif
        
        //#ifdef SIMD_AVX2_ENABLE
        //        if (Simd::Avx2::Enable)
        //            result = result && DescrIntVectorNormsNpAutoTest(FUNC_DI(Simd::Avx2::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif
        //
        //#ifdef SIMD_AVX512BW_ENABLE
        //        if (Simd::Avx512bw::Enable)
        //            result = result && DescrIntVectorNormsNpAutoTest(FUNC_DI(Simd::Avx512bw::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif
        //
        //#if defined(SIMD_NEON_ENABLE)
        //        if (Simd::Neon::Enable)
        //            result = result && DescrIntVectorNormsNpAutoTest(FUNC_DI(Simd::Neon::DescrIntInit), FUNC_DI(SimdDescrIntInit));
        //#endif

        return result;
    }
}
