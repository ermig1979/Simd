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
        struct FuncSH
        {
            typedef void(*FuncPtr)(const float * src, size_t size, uint16_t * dst);

            FuncPtr func;
            String description;

            FuncSH(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((const float*)src.data, src.width, (uint16_t*)dst.data);
            }
        };
    }

#define FUNC_SH(function) FuncSH(function, #function)

    bool Float32ToFloat16AutoTest(size_t size, const FuncSH & f1, const FuncSH & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1, dst2, 1, true, 32);

        return result;
    }

    bool Float32ToFloat16AutoTest(const FuncSH & f1, const FuncSH & f2)
    {
        bool result = true;

        result = result && Float32ToFloat16AutoTest(W*H, f1, f2);
        result = result && Float32ToFloat16AutoTest(W*H - 1, f1, f2);

        return result;
    }

    bool Float32ToFloat16AutoTest()
    {
        bool result = true;

        result = result && Float32ToFloat16AutoTest(FUNC_SH(Simd::Base::Float32ToFloat16), FUNC_SH(SimdFloat32ToFloat16));

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Float32ToFloat16AutoTest(FUNC_SH(Simd::Avx2::Float32ToFloat16), FUNC_SH(SimdFloat32ToFloat16));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && Float32ToFloat16AutoTest(FUNC_SH(Simd::Avx512bw::Float32ToFloat16), FUNC_SH(SimdFloat32ToFloat16));
#endif 

#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
        if (Simd::Neon::Enable)
            result = result && Float32ToFloat16AutoTest(FUNC_SH(Simd::Neon::Float32ToFloat16), FUNC_SH(SimdFloat32ToFloat16));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHS
        {
            typedef void(*FuncPtr)(const uint16_t * src, size_t size, float * dst);

            FuncPtr func;
            String description;

            FuncHS(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((const uint16_t*)src.data, src.width, (float*)dst.data);
            }
        };
    }

#define FUNC_HS(function) FuncHS(function, #function)

    bool Float16ToFloat32AutoTest(size_t size, const FuncHS & f1, const FuncHS & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View origin(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View src(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(origin, -10.0, 10.0);
        ::SimdFloat32ToFloat16((const float*)origin.data, size, (uint16_t*)src.data);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32);

        return result;
    }

    bool Float16ToFloat32AutoTest(const FuncHS & f1, const FuncHS & f2)
    {
        bool result = true;

        result = result && Float16ToFloat32AutoTest(W*H, f1, f2);
        result = result && Float16ToFloat32AutoTest(W*H - 1, f1, f2);

        return result;
    }

    bool Float16ToFloat32AutoTest()
    {
        bool result = true;

        result = result && Float16ToFloat32AutoTest(FUNC_HS(Simd::Base::Float16ToFloat32), FUNC_HS(SimdFloat16ToFloat32));

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Float16ToFloat32AutoTest(FUNC_HS(Simd::Avx2::Float16ToFloat32), FUNC_HS(SimdFloat16ToFloat32));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && Float16ToFloat32AutoTest(FUNC_HS(Simd::Avx512bw::Float16ToFloat32), FUNC_HS(SimdFloat16ToFloat32));
#endif 

#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
        if (Simd::Neon::Enable)
            result = result && Float16ToFloat32AutoTest(FUNC_HS(Simd::Neon::Float16ToFloat32), FUNC_HS(SimdFloat16ToFloat32));
#endif 

        return result;
    }

    struct FuncS
    {
        typedef void(*FuncPtr)(const uint16_t * a, const uint16_t * b, size_t size, float * sum);

        FuncPtr func;
        String description;

        FuncS(const FuncPtr & f, const String & d) : func(f), description(d) {}

        void Call(const View & a, const View & b, float * sum) const
        {
            TEST_PERFORMANCE_TEST(description);
            func((const uint16_t*)a.data, (const uint16_t*)b.data, a.width, sum);
        }
    };

    typedef void(*CheckPtr)(const float * a, const float * b, size_t size, float * sum);

#define FUNC_S(function) FuncS(function, #function)

    bool DifferenceSum16fAutoTest(int size, float eps, const FuncS & f1, const FuncS & f2, CheckPtr check)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View aOrigin(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(aOrigin, -10.0, 10.0);
        View a(size, 1, View::Int16, NULL, TEST_ALIGN(size));
        ::SimdFloat32ToFloat16((float*)aOrigin.data, a.width, (uint16_t*)a.data);

        View bOrigin(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(bOrigin, -10.0, 10.0);
        View b(size, 1, View::Int16, NULL, TEST_ALIGN(size));
        ::SimdFloat32ToFloat16((float*)bOrigin.data, b.width, (uint16_t*)b.data);

        float s1, s2, s3;
        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, &s1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, &s2));

        check((float*)aOrigin.data, (float*)bOrigin.data, a.width, &s3);

        result = Compare(s1, s2, eps, true, DifferenceRelative, "s1 & s2");
        result = Compare(s2, s3, eps*2, true, DifferenceRelative, "s2 & s3");

        return result;
    }

    bool DifferenceSum16fAutoTest(float eps, const FuncS & f1, const FuncS & f2, CheckPtr check)
    {
        bool result = true;

        result = result && DifferenceSum16fAutoTest(W*H, eps, f1, f2, check);
        result = result && DifferenceSum16fAutoTest(W*H - O, eps, f1, f2, check);

        return result;
    }

    bool SquaredDifferenceSum16fAutoTest()
    {
        bool result = true;

        CheckPtr check = ::SimdSquaredDifferenceSum32f;

        result = result && DifferenceSum16fAutoTest(EPS, FUNC_S(Simd::Base::SquaredDifferenceSum16f), FUNC_S(SimdSquaredDifferenceSum16f), check);

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && DifferenceSum16fAutoTest(EPS, FUNC_S(Simd::Avx2::SquaredDifferenceSum16f), FUNC_S(SimdSquaredDifferenceSum16f), check);
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSum16fAutoTest(EPS, FUNC_S(Simd::Avx512bw::SquaredDifferenceSum16f), FUNC_S(SimdSquaredDifferenceSum16f), check);
#endif

#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
        if (Simd::Neon::Enable)
            result = result && DifferenceSum16fAutoTest(EPS, FUNC_S(Simd::Neon::SquaredDifferenceSum16f), FUNC_S(SimdSquaredDifferenceSum16f), check);
#endif 

        return result;
    }

    bool CosineDistance16fAutoTest()
    {
        bool result = true;

        CheckPtr check = ::SimdCosineDistance32f;

        result = result && DifferenceSum16fAutoTest(EPS, FUNC_S(Simd::Base::CosineDistance16f), FUNC_S(SimdCosineDistance16f), check);

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && DifferenceSum16fAutoTest(EPS, FUNC_S(Simd::Avx2::CosineDistance16f), FUNC_S(SimdCosineDistance16f), check);
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DifferenceSum16fAutoTest(EPS, FUNC_S(Simd::Avx512bw::CosineDistance16f), FUNC_S(SimdCosineDistance16f), check);
#endif

#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
        if (Simd::Neon::Enable)
            result = result && DifferenceSum16fAutoTest(EPS, FUNC_S(Simd::Neon::CosineDistance16f), FUNC_S(SimdCosineDistance16f), check);
#endif

        return result;
    }

    typedef std::vector<uint16_t*> F16Ptrs;

    struct FuncCDA
    {
        typedef void(*FuncPtr)(size_t M, size_t N, size_t K, const uint16_t * const * A, const uint16_t * const * B, float * distances);

        FuncPtr func;
        String desc;

        FuncCDA(const FuncPtr & f, const String & d) : func(f), desc(d) {}

        void Update(size_t M, size_t N, size_t K)
        {
            desc = desc + "[" + ToString(M) + "-" + ToString(N) + "-" + ToString(K) + "]";
        }

        void Call(size_t K, const F16Ptrs & A, const F16Ptrs & B, Tensor32f & D) const
        {
            TEST_PERFORMANCE_TEST(desc);
            func(A.size(), B.size(), K, A.data(), B.data(), D.Data());
        }
    };

#define FUNC_CDA(function) FuncCDA(function, #function)

    bool CosineDistancesMxNa16fAutoTest(size_t M, size_t N, size_t K, float eps, FuncCDA f1, FuncCDA f2)
    {
        bool result = true;

        f1.Update(M, N, K);
        f2.Update(M, N, K);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        View Af(K, M, View::Float, NULL, TEST_ALIGN(K));
        FillRandom32f(Af, -1.0, 1.0);
        View Ai(K*2, M, View::Int16, NULL, TEST_ALIGN(K));
        F16Ptrs A(M);
        for (size_t i = 0; i < M; i++)
        {
            ::SimdFloat32ToFloat16(Af.Row<float>(i), K, Ai.Row<uint16_t>(i));
            A[i] = Ai.Row<uint16_t>(i);
        }

        View Bf(K, N, View::Float, NULL, TEST_ALIGN(K));
        FillRandom32f(Bf, -1.0, 1.0);
        View Bi(K * 2, N, View::Int16, NULL, TEST_ALIGN(K));
        F16Ptrs B(N);
        for (size_t j = 0; j < N; j++)
        {
            ::SimdFloat32ToFloat16(Bf.Row<float>(j), K, Bi.Row<uint16_t>(j));
            B[j] = Bi.Row<uint16_t>(j);
        }

        Tensor32f D1({ M, N, });
        Tensor32f D2({ M, N, });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(K, A, B, D1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(K, A, B, D2));

        result = Compare(D1, D2, eps, true, 32, DifferenceAbsolute);

        return result;
    }

    bool CosineDistancesMxNa16fAutoTest(float eps, const FuncCDA & f1, const FuncCDA & f2)
    {
        bool result = true;

        result = result && CosineDistancesMxNa16fAutoTest(1024, 128, 1024, eps, f1, f2);
        result = result && CosineDistancesMxNa16fAutoTest(1024, 129, 1024, eps, f1, f2);
        result = result && CosineDistancesMxNa16fAutoTest(1023, 128, 1024, eps, f1, f2);
        result = result && CosineDistancesMxNa16fAutoTest(1023, 129, 1023, eps, f1, f2);

#if !defined(SIMD_NEON_ENABLE) && 0
        result = result && CosineDistancesMxNa16fAutoTest(10*1024, 128, 1024, eps, f1, f2);
        result = result && CosineDistancesMxNa16fAutoTest(1024, 10*128, 1024, eps, f1, f2);
#endif

        return result;
    }

    bool CosineDistancesMxNa16fAutoTest()
    {
        bool result = true;

        result = result && CosineDistancesMxNa16fAutoTest(EPS, FUNC_CDA(Simd::Base::CosineDistancesMxNa16f), FUNC_CDA(SimdCosineDistancesMxNa16f));

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && CosineDistancesMxNa16fAutoTest(EPS, FUNC_CDA(Simd::Avx2::CosineDistancesMxNa16f), FUNC_CDA(SimdCosineDistancesMxNa16f));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && CosineDistancesMxNa16fAutoTest(EPS, FUNC_CDA(Simd::Avx512bw::CosineDistancesMxNa16f), FUNC_CDA(SimdCosineDistancesMxNa16f));
#endif

#if defined(SIMD_NEON_ENABLE) && defined(SIMD_NEON_FP16_ENABLE)
        if (Simd::Neon::Enable)
            result = result && CosineDistancesMxNa16fAutoTest(EPS, FUNC_CDA(Simd::Neon::CosineDistancesMxNa16f), FUNC_CDA(SimdCosineDistancesMxNa16f));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool Float32ToFloat16DataTest(bool create, size_t size, const FuncSH & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(src, -10.0, 10.0);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 1, true, 32);
        }

        return result;
    }

    bool Float32ToFloat16DataTest(bool create)
    {
        bool result = true;

        result = result && Float32ToFloat16DataTest(create, DH, FUNC_SH(SimdFloat32ToFloat16));

        return result;
    }

    bool Float16ToFloat32DataTest(bool create, size_t size, const FuncHS & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View src(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            View origin(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
            FillRandom32f(origin, -10.0, 10.0);
            ::SimdFloat32ToFloat16((const float*)origin.data, size, (uint16_t*)src.data);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 32);
        }

        return result;
    }

    bool Float16ToFloat32DataTest(bool create)
    {
        bool result = true;

        result = result && Float16ToFloat32DataTest(create, DH, FUNC_HS(SimdFloat16ToFloat32));

        return result;
    }

    bool DifferenceSum16fDataTest(bool create, int size, float eps, const FuncS & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View a(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View b(size, 1, View::Float, NULL, TEST_ALIGN(size));

        float s1, s2;

        if (create)
        {
            View aOrigin(size, 1, View::Float, NULL, TEST_ALIGN(size));
            FillRandom32f(aOrigin, -10.0, 10.0);
            ::SimdFloat32ToFloat16((float*)aOrigin.data, a.width, (uint16_t*)a.data);

            View bOrigin(size, 1, View::Float, NULL, TEST_ALIGN(size));
            FillRandom32f(bOrigin, -10.0, 10.0);
            ::SimdFloat32ToFloat16((float*)bOrigin.data, b.width, (uint16_t*)b.data);

            TEST_SAVE(a);
            TEST_SAVE(b);

            f.Call(a, b, &s1);

            TEST_SAVE(s1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);

            TEST_LOAD(s1);

            f.Call(a, b, &s2);

            TEST_SAVE(s2);

            result = result && Compare(s1, s2, eps, true);
        }

        return result;
    }

    bool SquaredDifferenceSum16fDataTest(bool create)
    {
        return DifferenceSum16fDataTest(create, DH, EPS * 2, FUNC_S(SimdSquaredDifferenceSum16f));
    }

    bool CosineDistance16fDataTest(bool create)
    {
        return DifferenceSum16fDataTest(create, DH, EPS * 2, FUNC_S(SimdCosineDistance16f));
    }
}
