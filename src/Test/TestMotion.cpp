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
#if 1 
            Simd::Motion::Model model;
            model.mask.Recreate(20, 20, View::Gray8);
            for (size_t y = 0; y < model.mask.height; ++y)
                for (size_t x = 0; x < model.mask.width; ++x)
                    model.mask.At<uint8_t>(x, y) = ((x < model.mask.width / 2) ? 255 : 0);
            _detector.SetModel(model);
#endif
        }

        virtual bool Process(const Frame & input, Frame & output)
        {
            Simd::Motion::Metadata metadata;
            _detector.NextFrame(input, metadata, &output);
            AnnotateMetadata(metadata, output.planes[0]);
            return true;
        }

    private:
        typedef Simd::Pixel::Bgr24 Color;
        typedef std::list<Simd::Motion::Event> Events;
        Events _events;
        Simd::Motion::Detector _detector;
        Simd::Font _font;

        void AnnotateMetadata(const Simd::Motion::Metadata & metadata, View & canvas)
        {
            _font.Resize(canvas.height / 32);
            for (size_t i = 0; i < metadata.objects.size(); i++)
            {
                const Simd::Motion::Object & object = metadata.objects[i];
                bool alarmed = false;
                for (size_t j = 0; j < metadata.events.size(); ++j)
                {
                    const Simd::Motion::Event & event = metadata.events[j];
                    if (event.objectId == object.id)
                    {
                        alarmed = true;
                        break;
                    }
                }
                Color color = alarmed ? Color(0, 0, 255) : Color(0, 255, 255);
                int width = alarmed ? 2 : 1;
                Simd::DrawRectangle(canvas, object.rect, color, width);
                _font.Draw(canvas, Simd::Motion::ToString(object.id), Point(object.rect.left, object.rect.top - _font.Height()), color);
                for (size_t j = 1; j < object.trajectory.size(); ++j)
                    Simd::DrawLine(canvas, object.trajectory[j - 1].point, object.trajectory[j].point, color, width);
            }
            for (size_t i = 0; i < metadata.events.size(); ++i)
            {
                _events.push_front(metadata.events[i]);
                if (_events.size()*_font.Height() > canvas.height)
                    _events.pop_back();
            }
            Point location;
            for (Events::const_iterator it = _events.begin(); it != _events.end(); ++it)
            {
                std::stringstream ss;
                Color color = Color(255, 255, 255);
                switch (it->type)
                {
                case Simd::Motion::Event::ObjectIn:
                    ss << "in " << it->objectId;
                    color = Color(255, 255, 255);
                    break;
                case Simd::Motion::Event::ObjectOut:
                    ss << "out " << it->objectId;
                    color = Color(255, 255, 255);
                    break;
                case Simd::Motion::Event::SabotageOn:
                    ss << "SABOTAGE ON";
                    color = Color(0, 0, 255);
                    break;
                case Simd::Motion::Event::SabotageOff:
                    ss << "SABOTAGE OFF";
                    color = Color(0, 0, 255);
                    break;
                };
                _font.Draw(canvas, ss.str(), location, color);
                location.y += _font.Height();
            }
        }
    };

    bool MotionSpecialTest()
    {
        Video video(true);

        if (SOURCE.length() == 0)
        {
            TEST_LOG_SS(Error, "Video source is undefined (-s parameter)!");
            return false;
        }
        if (!video.SetSource(SOURCE))
        {
            TEST_LOG_SS(Error, "Can't open source video file '" << SOURCE << "'!");
            return false;
        }

        if (OUTPUT.length() != 0 && !video.SetOutput(OUTPUT))
        {
            TEST_LOG_SS(Error, "Can't open output video file '" << OUTPUT << "'!");
            return false;
        }

        Filter filter;

        video.SetFilter(&filter);

        video.Start();

#ifdef TEST_PERFORMANCE_TEST_ENABLE
        TEST_LOG_SS(Info, PerformanceMeasurerStorage::s_storage.ConsoleReport(false, true));
        PerformanceMeasurerStorage::s_storage.Clear();
#endif

        return true;
    }
}
