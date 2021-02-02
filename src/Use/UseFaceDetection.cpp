/*
*  The use examples of Simd Library (http://ermig1979.github.io/Simd).
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
*
* In order to be enable of this example for Visual Studio 2015/2019 you have to rename file 'Ocv.prop.default' into 'Ocv.prop' and set there correct paths to OpenCV.
*/
#if !(defined(SIMD_USE_INSIDE) && !defined(SIMD_OPENCV_ENABLE))
#ifdef SIMD_USE_INSIDE
#define main UseFaceDetection
#endif

#include <iostream>
#include <string>

#include "opencv2/opencv.hpp"
#include "opencv2/core/utils/logger.hpp"
#ifndef SIMD_OPENCV_ENABLE
#define SIMD_OPENCV_ENABLE
#endif
#include "Simd/SimdDetection.hpp"
#include "Simd/SimdDrawing.hpp"

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        std::cout << "You have to set video source! It can be 0 for camera or video file name." << std::endl;
        return 1;
    }
    std::string source = argv[1], output = argc > 2 ? argv[2] : "";

    cv::VideoCapture capture;
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    if (source == "0")
        capture.open(0);
    else
        capture.open(source);
    if (!capture.isOpened())
    {
        std::cout << "Can't capture '" << source << "' !" << std::endl;
        return 1;
    }
    int W = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int H = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::VideoWriter writer;
    if (output.size())
    {
        writer.open(output, cv::VideoWriter::fourcc('F', 'M', 'P', '4'), capture.get(cv::CAP_PROP_FPS), cv::Size(W, H));
        if (!writer.isOpened())
        {
            std::cout << "Can't open output file '" << output << "' !" << std::endl;
            return 1;
        }
    }

    typedef Simd::Detection<Simd::Allocator> Detection;
    Detection detection;
    detection.Load("../../data/cascade/haar_face_0.xml");
    detection.Init(Detection::Size(W, H), 1.2, Detection::Size(W, H) / 20);

    const char * WINDOW_NAME = "FaceDetection";
    cv::namedWindow(WINDOW_NAME, 1);
    for (;;)
    {
        cv::Mat frame;
        if (!capture.read(frame))
            break;
        Detection::View image = frame;

        Detection::Objects objects;
        detection.Detect(image, objects);

        for (size_t i = 0; i < objects.size(); ++i)
            Simd::DrawRectangle(image, objects[i].rect, Simd::Pixel::Bgr24(0, 255, 255));

        cv::imshow(WINDOW_NAME, frame);
        if (writer.isOpened())
            writer.write(frame);
        if (cv::waitKey(1) == 27)// "press 'Esc' to break video";
            break;
    }
    return 0;
}

#endif

