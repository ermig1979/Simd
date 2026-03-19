/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Test/TestOptions.h"

#include "Simd/SimdSynetGatherElements.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)

    namespace
    {
        struct FuncGE
        {
            typedef void*(*FuncPtr)(SimdTensorDataType dataType, SimdTensorDataType indexType, SimdBool indexConst, size_t indexUsers, const size_t* outer, size_t outerSize, size_t srcCount, size_t inner, size_t idxCount);

            FuncPtr func;
            String desc;

            FuncGE(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorDataType dt, SimdTensorDataType it, SimdBool iC, SimdBool iN, size_t b, size_t o, size_t sc, size_t i, size_t ic)
            {
                std::stringstream ss;
                ss << desc << "[";
                ss << ToString(dt) << "-" << ToString(it) << "-" << iC << iN << "-";
                ss << b << "x" << o << "x" << sc << "x" << i << "-" << ic;
                ss << "]";
                desc = ss.str();
            }

            void Call(void * context, const uint8_t * src, const uint8_t* idx, uint8_t * dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdSynetGatherElementsForward(context, src, idx, dst);
            }
        };
    }

#define FUNC_GE(function) FuncGE(function, #function)

    template<class T> void Fill(T * data, size_t batch, size_t size, int lo, int hi)
    {
        for (size_t i = 0; i < size; ++i)
            data[i] = T(lo + Random(hi - lo));
        for (size_t b = 1; b < batch; b++)
            memcpy(data + b * size, data, size * sizeof(T));
    }

    template<class D, class I> bool SynetGatherElementsAutoTest(SimdBool indexConst, SimdBool indexNeg, size_t indexUsers, size_t batch, size_t outer, size_t srcCount, size_t inner, size_t idxCount, FuncGE f1, FuncGE f2)
    {
        bool result = true;

        Shape srcShape = Shp(batch, outer, srcCount, inner);
        Shape idxShape = Shp(batch, outer, idxCount, inner);

        f1.Update(DataType<D>(), DataType<I>(), indexConst, indexNeg, batch, outer, srcCount, inner, idxCount);
        f2.Update(DataType<D>(), DataType<I>(), indexConst, indexNeg, batch, outer, srcCount, inner, idxCount);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " .");

        Tensor<D> src(srcShape);
        Tensor<I> idx(idxShape);
        Tensor<D> dst1(idxShape);
        Tensor<D> dst2(idxShape);

        Fill(src.Data(), 1, src.Size(), 0, 255);
        Fill(idx.Data(), batch, outer * idxCount * inner, indexNeg ? -(int)srcCount : 0, (int)srcCount);
        memset(dst1.Data(), 1, dst1.Size() * sizeof(D));
        memset(dst2.Data(), 2, dst2.Size() * sizeof(D));

        void* context1 = f1.func(DataType<D>(), DataType<I>(), indexConst, indexUsers, srcShape.data(), 2, srcCount, inner, idxCount);
        void* context2 = f2.func(DataType<D>(), DataType<I>(), indexConst, indexUsers, srcShape.data(), 2, srcCount, inner, idxCount);

        if (indexConst)
        {
            SimdSynetGatherElementsSetIndex(context1, (uint8_t*)idx.Data());
            SimdSynetGatherElementsSetIndex(context2, (uint8_t*)idx.Data());
        }

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, (uint8_t*)src.Data(), (uint8_t*)idx.Data(), (uint8_t*)dst1.Data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, (uint8_t*)src.Data(), (uint8_t*)idx.Data(), (uint8_t*)dst2.Data()));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    bool SynetGatherElementsAutoTest(const FuncGE& f1, const FuncGE& f2)
    {
        bool result = true;

        SimdBool t = SimdTrue, f = SimdFalse;
        SimdTensorDataType f32 = SimdTensorData32f;

#ifdef NDEBUG
#if 1
        result = result && SynetGatherElementsAutoTest<float, int64_t>(f, t, 28, 16, 196, 49, 1, 196, f1, f2);
        result = result && SynetGatherElementsAutoTest<float, int32_t>(f, t, 28, 16, 196, 49, 1, 196, f1, f2);
        result = result && SynetGatherElementsAutoTest<uint16_t, int32_t>(f, t, 28, 16, 196, 49, 1, 196, f1, f2);
        result = result && SynetGatherElementsAutoTest<uint16_t, int32_t>(f, f, 28, 16, 196, 49, 1, 196, f1, f2);
        result = result && SynetGatherElementsAutoTest<uint16_t, int32_t>(t, t, 28, 16, 196, 49, 1, 196, f1, f2);
        result = result && SynetGatherElementsAutoTest<uint16_t, int32_t>(t, f, 28, 16, 196, 49, 1, 196, f1, f2);
        result = result && SynetGatherElementsAutoTest<uint16_t, int32_t>(t, f, 28, 1, 16 * 196, 49, 1, 196, f1, f2);
#endif
#else
        result = result && SynetGatherElementsAutoTest<uint16_t, int32_t>(t, f, 28, 16, 196, 49, 1, 196, f1, f2);
        result = result && SynetGatherElementsAutoTest<uint16_t, int32_t>(t, f, 28, 1, 16 * 196, 49, 1, 196, f1, f2);
#endif

        return result;
    }

    bool SynetGatherElementsAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetGatherElementsAutoTest(FUNC_GE(Simd::Base::SynetGatherElementsInit), FUNC_GE(SimdSynetGatherElementsInit));

        return result;
    }
#endif
}
