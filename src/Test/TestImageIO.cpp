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

            void Update(View::Format format, SimdImageFileType file)
            {
                desc = desc + "[" + ToString(format) + "-" + ToString(file) + "]";
            }

            void Call(const View& src, SimdImageFileType file, int quality, uint8_t** data, size_t* size) const
            {
                TEST_PERFORMANCE_TEST(desc);
                *data = func(src.data, src.stride, src.width, src.height, (SimdPixelFormatType)src.format, file, quality, size);
            }
        };
    }

#define FUNC_SM(func) \
    FuncSM(func, std::string(#func))

    bool ImageSaveToMemoryAutoTest(size_t width, size_t height, View::Format format, SimdImageFileType file, int quality, FuncSM f1, FuncSM f2)
    {
        bool result = true;

        f1.Update(format, file);
        f2.Update(format, file);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << width << ", " << height << "].");

        View src(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(src);

        uint8_t* data1 = NULL, * data2 = NULL;
        size_t size1 = 0, size2 = 0;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (data1) Simd::Free(data1); f1.Call(src, file, quality, &data1, &size1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (data2) SimdFree(data2); f2.Call(src, file, quality, &data2, &size2));

        result = result && Compare(data1, size1, data2, size2, 0, true, 64);

        if (data1)
            Simd::Free(data1);
        if (data2)
            SimdFree(data2);

        //src.Save((ToString(file) + ".txt").c_str(), file, 100);

        return result;
    }

    bool ImageSaveToMemoryAutoTest(const FuncSM & f1, const FuncSM& f2)
    {
        bool result = true;

        View::Format formats[4] = { View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24 };
        for (int format = 0; format < 4; format++)
        {
            for (int file = (int)SimdImageFilePgmTxt; file <= (int)SimdImageFilePpmBin; file++)
            {
                result = result && ImageSaveToMemoryAutoTest(W, H, formats[format], (SimdImageFileType)file, 100, f1, f2);
                result = result && ImageSaveToMemoryAutoTest(W + O, H - O, formats[format], (SimdImageFileType)file, 100, f1, f2);
            }
        }

        return result;
    }

    bool ImageSaveToMemoryAutoTest()
    {
        bool result = true;

        result = result && ImageSaveToMemoryAutoTest(FUNC_SM(Simd::Base::ImageSaveToMemory), FUNC_SM(SimdImageSaveToMemory));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && ImageSaveToMemoryAutoTest(FUNC_SM(Simd::Sse41::ImageSaveToMemory), FUNC_SM(SimdImageSaveToMemory));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && ImageSaveToMemoryAutoTest(FUNC_SM(Simd::Avx2::ImageSaveToMemory), FUNC_SM(SimdImageSaveToMemory));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && ImageSaveToMemoryAutoTest(FUNC_SM(Simd::Avx512bw::ImageSaveToMemory), FUNC_SM(SimdImageSaveToMemory));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && ImageSaveToMemoryAutoTest(FUNC_SM(Simd::Neon::ImageSaveToMemory), FUNC_SM(SimdImageSaveToMemory));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    namespace
    {
        struct FuncLM
        {
            typedef Simd::ImageLoadFromMemoryPtr FuncPtr;

            FuncPtr func;
            String desc;

            FuncLM(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(View::Format format, SimdImageFileType file)
            {
                desc = desc + "[" + ToString(format) + "-" + ToString(file) + "]";
            }

            void Call(const uint8_t* data, size_t size, View::Format format, View& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ((View::Format&)dst.format) = format;
                *(uint8_t**)&dst.data = func(data, size, (size_t*)&dst.stride, (size_t*)&dst.width, (size_t*)&dst.height, (SimdPixelFormatType*)&dst.format);
            }
        };
    }

#define FUNC_LM(func) \
    FuncLM(func, std::string(#func))

    bool SaveLoadCompatible(View::Format format, SimdImageFileType file, int quality)
    {
        if (file == SimdImageFilePgmTxt || file == SimdImageFilePgmBin)
            return format == View::Gray8;
        if (file == SimdImageFilePpmTxt || file == SimdImageFilePpmBin)
            return format != View::Bgra32;
        return false;
    }

    bool ImageLoadFromMemoryAutoTest(size_t width, size_t height, View::Format format, SimdImageFileType file, int quality, FuncLM f1, FuncLM f2)
    {
        bool result = true;

        f1.Update(format, file);
        f2.Update(format, file);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << width << ", " << height << "].");

        View src(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(src);

        size_t size = 0;
        uint8_t* data = SimdImageSaveToMemory(src.data, src.stride, src.width, src.height, (SimdPixelFormatType)src.format, file, quality, &size);

        View dst1, dst2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (dst1.data) Simd::Free(dst1.data); f1.Call(data, size, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (dst2.data) SimdFree(dst2.data); f2.Call(data, size, format, dst2));

        if(dst1.data && dst2.data)
            result = result && Compare(dst1, dst2, 0, true, 64, 0, "dst1 & dst2");
        if(dst1.data && SaveLoadCompatible(format, file, quality))
            result = result && Compare(dst1, src, 0, true, 64, 0, "dst1 & src");

        if (dst1.data)
            Simd::Free(dst1.data);
        if (dst2.data)
            SimdFree(dst2.data);
        SimdFree(data);

        return result;
    }

    bool ImageLoadFromMemoryAutoTest(const FuncLM& f1, const FuncLM& f2)
    {
        bool result = true;

        View::Format formats[4] = { View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24 };
        for (int format = 0; format < 4; format++)
        {
            for (int file = (int)SimdImageFilePgmTxt; file <= (int)SimdImageFilePpmBin; file++)
            {
                result = result && ImageLoadFromMemoryAutoTest(W, H, formats[format], (SimdImageFileType)file, 100, f1, f2);
                result = result && ImageLoadFromMemoryAutoTest(W + O, H - O, formats[format], (SimdImageFileType)file, 100, f1, f2);
            }
        }

        return result;
    }

    bool ImageLoadFromMemoryAutoTest()
    {
        bool result = true;

        result = result && ImageLoadFromMemoryAutoTest(FUNC_LM(Simd::Base::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && ImageLoadFromMemoryAutoTest(FUNC_LM(Simd::Sse41::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));
#endif 

        return result;
    }
}
