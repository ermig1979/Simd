/*
* Tests for Simd Library (http://simd.sourceforge.net).
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

//-----------------------------------------------------------------------------

#ifdef TEST_PERFORMANCE_TEST_ENABLE
#define SIMD_CHECK_PERFORMANCE() TEST_PERFORMANCE_TEST_(__FUNCTION__)
#endif

#include "Test/TestVideo.h"
#include "Simd/SimdMotion.hpp"
#include "Simd/SimdDrawing.hpp"
#include "Simd/SimdPixel.hpp"

namespace Test
{
    typedef Video::Frame Frame;

    struct Filter : public Video::Filter 
    {
        Filter()
        {

        }

        virtual bool Process(const Frame & input, Frame & output)
        {
            Simd::Motion::Metadata metadata;
            _detector.NextFrame(input, metadata, &output);

            Simd::Pixel::Bgr24 yellow(0, 255, 255);
            size_t width = 2;

            for (size_t i = 0; i < metadata.objects.size(); ++i)
            {
                const Simd::Motion::Object & object = metadata.objects[i];
                Simd::DrawRectangle(output.planes[0], object.current.rect, yellow, width);
            }

            return true;
        }

    private:
        Simd::Motion::Detector _detector;
    };

    bool MotionSpecialTest()
    {
        Video video;

        if (SOURCE.length() == 0)
        {
            TEST_LOG_SS(Error, "Video source is undefined (-s parameter)!");
            return false;
        }
        if(!video.SetSource(SOURCE))
        {
            TEST_LOG_SS(Error, "Can't open video file '" << SOURCE << "'!");
            return false;
        }

        Filter filter;

        video.SetFilter(&filter);

        video.Start();

#ifdef TEST_PERFORMANCE_TEST_ENABLE
        TEST_LOG_SS(Info, PerformanceMeasurerStorage::s_storage.Report(false, true));
        PerformanceMeasurerStorage::s_storage.Clear();
#endif

        return true;
    }
}