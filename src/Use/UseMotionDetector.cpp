/*
*  The use examples of Simd Library (http://ermig1979.github.io/Simd).
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
*
* In order to be enable of this example for Visual Studio 2015 you have to rename file 'Ocv.prop.default' into 'Ocv.prop' and set there correct paths to OpenCV.
*/
#if !(defined(SIMD_USE_INSIDE) && !defined(SIMD_OPENCV_ENABLE))
#ifdef SIMD_USE_INSIDE
#define main UseMotionDetector
#endif

#include <iostream>
#include <string>
#include <list>

#include "opencv2/opencv.hpp"
#ifndef SIMD_OPENCV_ENABLE
#define SIMD_OPENCV_ENABLE
#endif
#include "Simd/SimdMotion.hpp"

using namespace Simd::Motion;
typedef std::list<Event> EventList;
typedef Simd::Pixel::Bgr24 Color;

const Color Red(0, 0, 255), Yellow(0, 255, 255), White(0, 255, 255);

void Annotate(const Metadata & metadata, const Simd::Font & font, EventList & events, View & image)
{
    for (size_t i = 0; i < metadata.objects.size(); i++)
    {
        const Object & object = metadata.objects[i];
        bool alarmed = false;
        for (size_t j = 0; j < metadata.events.size(); ++j)
        {
            const Event & event = metadata.events[j];
            if (event.objectId == object.id)
            {
                alarmed = true;
                break;
            }
        }
        Color color = alarmed ? Red : Yellow;
        int width = alarmed ? 2 : 1;
        Simd::DrawRectangle(image, object.rect, color, width);
        font.Draw(image, ToString(object.id), Point(object.rect.left, object.rect.top - font.Height()), color);
        for (size_t j = 1; j < object.trajectory.size(); ++j)
            Simd::DrawLine(image, object.trajectory[j - 1].point, object.trajectory[j].point, color, width);
    }
    for (size_t i = 0; i < metadata.events.size(); ++i)
    {
        events.push_front(metadata.events[i]);
        if (events.size()*font.Height() > image.height)
            events.pop_back();
    }
    Point location;
    for (EventList::const_iterator it = events.begin(); it != events.end(); ++it)
    {
        std::stringstream ss;
        Color color = White;
        switch (it->type)
        {
        case Event::ObjectIn:
            ss << "in " << it->objectId;
            break;
        case Event::ObjectOut:
            ss << "out " << it->objectId;
            break;
        case Event::SabotageOn:
            ss << "SABOTAGE ON";
            color = Red;
            break;
        case Event::SabotageOff:
            ss << "SABOTAGE OFF";
            color = Red;
            break;
        };
        font.Draw(image, ss.str(), location, color);
        location.y += font.Height();
    }
}

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        std::cout << "You have to set video source! It can be 0 for camera or video file name." << std::endl;
        return 1;
    }
    std::string source = argv[1];

    cv::VideoCapture capture;
    if (source == "0")
        capture.open(0);
    else
        capture.open(source);
    if (!capture.isOpened())
    {
        std::cout << "Can't capture '" << source << "' !" << std::endl;
        return 1;
    }

    EventList events;
    Detector detector;
    Simd::Font font;

    const char * WINDOW_NAME = "MotionDetector";
    cv::namedWindow(WINDOW_NAME, 1);
    double time = 0;
    for (;;)
    {
        cv::Mat frame;
        capture >> frame;

        View image = frame;
        Frame input(image, false, time);
        Metadata metadata;

        detector.NextFrame(input, metadata);

        font.Resize(image.height / 32);

        Annotate(metadata, font, events, image);

        cv::imshow(WINDOW_NAME, frame);
        if (cvWaitKey(1) == 27)// "press 'Esc' to break video";
            break;
        time += 0.040;
    }
    return 0;
}

#endif

