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
#include "Test/TestVideo.h"

#ifdef SIMD_OPENCV_ENABLE
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

namespace Test
{
    const String SIMD_DEBUG_WINDOW_NAME = "Simd::Output";

#ifdef SIMD_OPENCV_ENABLE
    struct Video::Native
    {
        Native()
        {
            ::cvNamedWindow(SIMD_DEBUG_WINDOW_NAME.c_str(), CV_WINDOW_AUTOSIZE);
        }

        ~Native()
        {
            ::cvReleaseCapture(&capture);
            ::cvDestroyWindow(SIMD_DEBUG_WINDOW_NAME.c_str());
        }

        bool SetSource(const String & source) 
        { 
            capture = ::cvCreateFileCapture(source.c_str());
            return capture != NULL; 
        }

        bool SetFilter(Filter * filter)
        {
            this->filter = filter;
            return true;
        }

        bool Start()
        {
            time = 0;
            while (1) 
            {
                ::IplImage * frame = ::cvQueryFrame(capture);
                if (!frame)
                    break;

                if (filter)
                {
                    filter->Process(Convert(frame), Convert(frame).Ref());
                }

                ::cvShowImage(SIMD_DEBUG_WINDOW_NAME.c_str(), frame);

                char c = cvWaitKey(1);
                if (c == 27) break;

                time += 0.040;
            }
            return true;
        }

    private:
        Filter * filter;
        ::CvCapture * capture;
        double time;

        Frame Convert(::IplImage * frame)
        {
            if (::memcmp(frame->colorModel, "RGB", 3) == 0 && ::memcmp(frame->channelSeq, "BGR", 3) == 0)
            {
                View view(frame->width, frame->height, frame->widthStep, View::Bgr24, frame->imageData);
                return Frame(view, false, time);
            }
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