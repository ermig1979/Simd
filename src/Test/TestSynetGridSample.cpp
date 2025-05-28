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
#include "Test/TestUtils.h"
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestTensor.h"
#include "Test/TestString.h"
#include "Test/TestRandom.h"

#include "Simd/SimdSynetGridSample.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)

    inline String ToString(SimdGridSampleInterpType interp)
    {
        switch (interp)
        {
        case SimdGridSampleInterpBilinear: return "Bl";
        case SimdGridSampleInterpNearest: return "Nr";
        case SimdGridSampleInterpBicubic: return "Bc";
        default: assert(0); return "Assert";
        }
    }

    inline String ToString(SimdGridSamplePaddingType padding)
    {
        switch (padding)
        {
        case SimdGridSamplePaddingZeros: return "Z";
        case SimdGridSamplePaddingBorder: return "B";
        case SimdGridSamplePaddingReflect: return "R";
        default: assert(0); return "Assert";
        }
    }

    namespace
    {
        struct FuncGS2D
        {
            typedef void*(*FuncPtr)(size_t batch, size_t channels, size_t srcH, size_t srcW, size_t dstH, size_t dstW,
                SimdTensorDataType type, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align);

            FuncPtr func;
            String desc;

            FuncGS2D(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Shape& src, const Shape& grd, SimdTensorDataType type, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align)
            {
                std::stringstream ss;
                ss << desc << "[";
                for (size_t i = 0; i < src.size(); ++i)
                    ss << (i ? "x" : "") << src[i];
                ss << "-" << grd[1] << "x" << grd[2];
                ss << "-" << ToString(type) << "-" << ToString(interp);
                ss << "-" << ToString(padding) << "-" << ToString(align) << "]";
                desc = ss.str();
            }

            void Call(void * context, const uint8_t * src, const uint8_t* grd, uint8_t * dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdSynetGridSample2dForward(context, src, grd, dst);
            }
        };
    }

#define FUNC_GS2D(function) FuncGS2D(function, #function)

    template<class T> void Fill(Tensor<T>& tensor, int grid);

    template<> void Fill<float>(Tensor<float>& tensor, int grid)
    {
        if(grid)
        {
            const Shape& shape = tensor.Shape();
            for (size_t b = 0; b < shape[0]; ++b)
            {
                for (size_t y = 0; y < shape[1]; ++y)
                {
                    for (size_t x = 0; x < shape[2]; ++x)
                    {
                        tensor.Data(Shp(b, y, x, 0))[0] = float(y) / shape[1] + 0.1 * (float)Random();
                        tensor.Data(Shp(b, y, x, 1))[0] = float(x) / shape[2] + 0.1 * (float)Random();
                    }
                }
            }
        }
        else
            FillRandom(tensor, -1.1f, 1.1f);
    }

    template <class T > bool SynetGridSample2dAutoTest(const Shape& srcShape, const Shape& grdShape,
        SimdTensorDataType type, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align, FuncGS2D f1, FuncGS2D f2)
    {
        bool result = true;

        Shape dstShape = Shp(srcShape[0], srcShape[1], grdShape[1], grdShape[2]);

        f1.Update(srcShape, grdShape, type, interp, padding, align);
        f2.Update(srcShape, grdShape, type, interp, padding, align);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " .");

        Tensor<T> src(srcShape);
        Tensor<T> grd(grdShape);
        Tensor<T> dst1(dstShape);
        Tensor<T> dst2(dstShape);

        Fill(src, 0);
        Fill(grd, 1);
        memset(dst1.Data(), 1, dst1.Size() * sizeof(T));
        memset(dst2.Data(), 2, dst2.Size() * sizeof(T));

        void* context1 = f1.func(srcShape[0], srcShape[1], srcShape[2], srcShape[3], grdShape[1], grdShape[2], type, interp, padding, align);
        void* context2 = f2.func(srcShape[0], srcShape[1], srcShape[2], srcShape[3], grdShape[1], grdShape[2], type, interp, padding, align);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, (uint8_t*)src.Data(), (uint8_t*)grd.Data(), (uint8_t*)dst1.Data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, (uint8_t*)src.Data(), (uint8_t*)grd.Data(), (uint8_t*)dst2.Data()));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetGridSample2dAutoTest(const Shape& srcShape, const Shape& grdShape, const FuncGS2D& f1, const FuncGS2D& f2)
    {
        bool result = true;

        SimdBool t = SimdTrue, f = SimdFalse;
        for (int i = 0; i < 3; ++i)
        {
            for (int p = 0; p < 3; ++p)
            {
                result = result && SynetGridSample2dAutoTest<float>(srcShape, grdShape, SimdTensorData32f, (SimdGridSampleInterpType)i, (SimdGridSamplePaddingType)p, f, f1, f2);
                result = result && SynetGridSample2dAutoTest<float>(srcShape, grdShape, SimdTensorData32f, (SimdGridSampleInterpType)i, (SimdGridSamplePaddingType)p, t, f1, f2);
            }
        }

        return result;
    }

    bool SynetGridSample2dAutoTest(const FuncGS2D& f1, const FuncGS2D& f2)
    {
        bool result = true;

        SimdBool t = SimdTrue, f = SimdFalse;
        SimdTensorDataType f32 = SimdTensorData32f;
        SimdGridSampleInterpType Bl = SimdGridSampleInterpBilinear;
        SimdGridSamplePaddingType Z = SimdGridSamplePaddingZeros;


#ifdef NDEBUG
#if 1
        result = result && SynetGridSample2dAutoTest<float>(Shp(5188, 1, 54, 96), Shp(5188, 7, 7, 2), f32, Bl, Z, t, f1, f2);
        result = result && SynetGridSample2dAutoTest<float>(Shp(5188, 1, 27, 48), Shp(5188, 7, 7, 2), f32, Bl, Z, t, f1, f2);
        result = result && SynetGridSample2dAutoTest<float>(Shp(5188, 1, 13, 24), Shp(5188, 7, 7, 2), f32, Bl, Z, t, f1, f2);
        result = result && SynetGridSample2dAutoTest<float>(Shp(5188, 1, 6, 12), Shp(5188, 7, 7, 2), f32, Bl, Z, t, f1, f2);
#endif
#if 0
        result = result && SynetGridSample2dAutoTest(Shp(8, 32, 20, 20), Shp(8, 300, 4, 2), f1, f2);
        result = result && SynetGridSample2dAutoTest(Shp(8, 32, 40, 40), Shp(8, 300, 4, 2), f1, f2);
        result = result && SynetGridSample2dAutoTest(Shp(8, 32, 80, 80), Shp(8, 300, 4, 2), f1, f2);
#endif
#else
        result = result && SynetGridSample2dAutoTest<float>(Shp(5188, 1, 54, 96), Shp(5188, 7, 7, 2), f32, Bl, Z, t, f1, f2);
        result = result && SynetGridSample2dAutoTest<float>(Shp(5188, 1, 13, 24), Shp(5188, 7, 7, 2), f32, Bl, Z, t, f1, f2);
#endif

        return result;
    }

    bool SynetGridSample2dAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && SynetGridSample2dAutoTest(FUNC_GS2D(Simd::Base::SynetGridSample2dInit), FUNC_GS2D(SimdSynetGridSample2dInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetGridSample2dAutoTest(FUNC_GS2D(Simd::Sse41::SynetGridSample2dInit), FUNC_GS2D(SimdSynetGridSample2dInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetGridSample2dAutoTest(FUNC_GS2D(Simd::Avx2::SynetGridSample2dInit), FUNC_GS2D(SimdSynetGridSample2dInit));
#endif 

        return result;
    }
#endif
}
