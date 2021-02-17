/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

#include "Simd/SimdImageLoad.h"
#include "Simd/SimdImageSave.h"

namespace Test
{
    namespace
    {
        struct FuncSM
        {
            typedef Simd::ImageSaveToMemoryPtr FuncPtr;

            FuncPtr func;
            String desc;

            FuncSM(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Call(const View& src, View& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                //func(src.data, src.stride, src.width, src.height, src.PixelSize(), dst.data, dst.stride);
            }
        };
    }

#define FUNC_SM(function) \
    Func(function, std::string(#function))

    bool ImageLoadFromMemoryAutoTest()
    {
        bool result = true;

        //result = result && CopyFrameAutoTest(FUNC_F(Simd::Base::CopyFrame), FUNC_F(SimdCopyFrame));

        return result;
    }

    //-----------------------------------------------------------------------

    bool ImageSaveToMemoryAutoTest()
    {
        bool result = true;

        //result = result && CopyFrameAutoTest(FUNC_F(Simd::Base::CopyFrame), FUNC_F(SimdCopyFrame));

        return result;
    }
}
