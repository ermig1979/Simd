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
#include "Test/TestVideo.h"

#ifdef SIMD_OPENCV_ENABLE
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#endif

namespace Test
{
    const String SIMD_DEBUG_WINDOW_NAME = "Simd::Output";

#ifdef SIMD_OPENCV_ENABLE
    struct Video::Native
    {
        Native()
        {
            cv::namedWindow(SIMD_DEBUG_WINDOW_NAME.c_str(), cv::WINDOW_AUTOSIZE);
        }

        ~Native()
        {
            cv::destroyWindow(SIMD_DEBUG_WINDOW_NAME.c_str());
        }

        bool SetSource(const String & source)
        {
            if (source == "0")
                _capture.open(0);
            else
                _capture.open(source);
            return _capture.isOpened();
        }

        bool SetFilter(Filter * filter)
        {
            _filter = filter;
            return true;
        }

        bool Start()
        {
            _time = 0;
            while (1)
            {
                cv::Mat frame;
                if (!_capture.read(frame))
                    break;

                if (_filter)
                {
                    _filter->Process(Convert(frame), Convert(frame).Ref());
                }

                cv::imshow(SIMD_DEBUG_WINDOW_NAME, frame);

                char c = cv::waitKey(1);
                if (c == 27) break;

                _time += 0.040;
            }
            return true;
        }

    private:
        Filter * _filter;
        cv::VideoCapture _capture;
        double _time;

        Frame Convert(const cv::Mat & frame)
        {
            if (frame.channels() == 3)
                return Frame(View(frame), false, _time);
            return Frame();
        }
    };
#else
    struct Video::Native
    {
        bool SetSource(const String & source) { return false; }
        bool SetFilter(Filter * filter) { return false; }
        bool Start() { return false; }
    };
#endif

    Video::Video()
    {
        _native = new Native();
    }

    Video::~Video()
    {
        delete _native;
    }

    bool Video::SetSource(const String & source)
    {
        return _native->SetSource(source);
    }

    bool Video::SetFilter(Filter * filter)
    {
        return _native->SetFilter(filter);
    }

    bool Video::Start()
    {
        return _native->Start();
    }
}
