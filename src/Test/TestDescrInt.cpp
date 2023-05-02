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
        struct FuncDIE
        {
            typedef void*(*FuncPtr)(size_t size, size_t depth);

            FuncPtr func;
            String desc;

            FuncDIE(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(size_t s, size_t d)
            {
                std::stringstream ss;
                ss << desc << "[Encode-" << s << "-" << d << "]";
                desc = ss.str();
            }

            void Call(const void* context, const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdDescrIntEncode(context, (const float*)src.data, dst.data);
            }
        };
    }

#define FUNC_DIE(function) FuncDIE(function, #function)

    bool DescrIntEncodeAutoTest(size_t size, size_t depth, FuncDIE f1, FuncDIE f2)
    {
        bool result = true;

        f1.Update(size, depth);
        f2.Update(size, depth);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(src, -17.0, 13.0);

        void* context1 = f1.func(size, depth);
        void* context2 = f2.func(size, depth);

        View dst1(SimdDescrIntEncodedSize(context1), 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(SimdDescrIntEncodedSize(context2), 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom(dst1, 1, 1);
        FillRandom(dst2, 2, 2);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    bool DescrIntEncodeAutoTest(const FuncDIE& f1, const FuncDIE& f2)
    {
        bool result = true;

        for (size_t depth = 8; depth <= 8; depth++)
        {
            result = result && DescrIntEncodeAutoTest(256, depth, f1, f2);
            result = result && DescrIntEncodeAutoTest(512, depth, f1, f2);
        }

        return result;
    }

    bool DescrIntEncodeAutoTest()
    {
        bool result = true;

        result = result && DescrIntEncodeAutoTest(FUNC_DIE(Simd::Base::DescrIntInit), FUNC_DIE(SimdDescrIntInit));

//#ifdef SIMD_SSE41_ENABLE
//        if (Simd::Sse41::Enable)
//            result = result && DescrIntEncodeAutoTest(FUNC_DIE(Simd::Sse41::DescrIntInit), FUNC_DIE(SimdDescrIntInit));
//#endif 
//
//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable)
//            result = result && DescrIntEncodeAutoTest(FUNC_DIE(Simd::Avx2::DescrIntInit), FUNC_DIE(SimdDescrIntInit));
//#endif 
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable)
//            result = result && DescrIntEncodeAutoTest(FUNC_DIE(Simd::Avx512bw::DescrIntInit), FUNC_DIE(SimdDescrIntInit));
//#endif 
//
//#ifdef SIMD_NEON_ENABLE
//        if (Simd::Neon::Enable)
//            result = result && DescrIntEncodeAutoTest(FUNC_DIE(Simd::Neon::DescrIntInit), FUNC_DIE(SimdDescrIntInit));
//#endif 

        return result;
    }
}
