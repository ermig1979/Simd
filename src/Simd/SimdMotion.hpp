/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __SimdMotion_hpp__
#define __SimdMotion_hpp__

#include "Simd/SimdPoint.hpp"
#include "Simd/SimdRectangle.hpp"
#include "Simd/SimdFrame.hpp"
#include "Simd/SimdDrawing.hpp"
#include "Simd/SimdFont.hpp"

#include <vector>
#include <stack>
#include <sstream>

#ifndef SIMD_CHECK_PERFORMANCE
#define SIMD_CHECK_PERFORMANCE()
#endif

namespace Simd
{
    /*! @ingroup cpp_motion

        \short Contains a C++ framework for motion detection.

        This is a wrapper around the low-level \ref motion_detection API.

        Typical usage creates a Detector, optionally calls SetModel and SetOptions
        once, then for every video frame wraps a View as Frame with a timestamp
        in seconds and calls NextFrame. NextFrame fills Metadata with classified
        moving objects and events (Event::ObjectIn, Event::ObjectOut,
        Event::SabotageOn, Event::SabotageOff).

        The example annotates Metadata on the input image: DrawRectangle around
        Object::rect, ToString(Object::id) above the box, DrawLine along
        Object::trajectory, and a scrolling list of Event::type /
        Event::objectId. Red marks objects that have an event in the same frame.

        SetModel and SetOptions are typically called once before the video loop.
        Calibration (pyramid scale and ROI mask) runs on the first NextFrame and
        when the input size changes. The shooting-star example (disabled by #if 0)
        uses a small Model::size and lowered ClassificationShiftMin /
        ClassificationTimeMin. A Gray8 Model::mask can restrict ROI (non-zero
        pixels are inside). Debug annotation is drawn on an optional Bgr24 output
        Frame of the same size as input.

        ONVIF coordinates place the origin at the screen center: X in [-1, 1]
        (left to right), Y in [-1, 1] with +Y up. Screen coordinates are pixels
        with (0, 0) at the top-left.

        Using example (motion detection in the video captured by OpenCV):
        \code
        #include <iostream>
        #include <string>
        #include <list>

        #include "opencv2/opencv.hpp"
        #include "opencv2/core/utils/logger.hpp"
        #ifndef SIMD_OPENCV_ENABLE
        #define SIMD_OPENCV_ENABLE
        #endif
        #include "Simd/SimdMotion.hpp"

        using namespace Simd::Motion;
        typedef std::list<Event> EventList;
        typedef Simd::Pixel::Bgr24 Color;

        const Color Red(0, 0, 255), Yellow(0, 255, 255), White(255, 255, 255);

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

            cv::VideoWriter writer;
            if (output.size())
            {
                writer.open(output, cv::VideoWriter::fourcc('F','M','P','4'), capture.get(cv::CAP_PROP_FPS),
                    cv::Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH), (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
                if (!writer.isOpened())
                {
                    std::cout << "Can't open output file '" << output << "' !" << std::endl;
                    return 1;
                }
            }

            EventList events;
            Detector detector;
            Simd::Font font((int)capture.get(cv::CAP_PROP_FRAME_HEIGHT) / 32);

        #if 0
            // There is an example of change of parameters to detect shooting star in the night sky:
            Model model;
            model.size = FSize(0.01, 0.01);
            detector.SetModel(model);

            Options options;
            options.TrackingAdditionalLinking = 5;
            options.ClassificationShiftMin = 0.01;
            options.ClassificationTimeMin = 0.01;
            options.DifferenceDxFeatureWeight = 0;
            options.DifferenceDyFeatureWeight = 0;
            options.BackgroundStatUpdateTime = 0.2;
            detector.SetOptions(options);
        #endif

            const char * WINDOW_NAME = "MotionDetector";
            cv::namedWindow(WINDOW_NAME, 1);
            for (;;)
            {
                cv::Mat frame;
                if (!capture.read(frame))
                    break;
                View image = frame;
                Frame input(image, false, capture.get(cv::CAP_PROP_POS_MSEC) * 0.001);
                Metadata metadata;

                detector.NextFrame(input, metadata);

                Annotate(metadata, font, events, image);

                cv::imshow(WINDOW_NAME, frame);
                if (writer.isOpened())
                    writer.write(frame);
                if (cv::waitKey(1) == 27)// "press 'Esc' to break video";
                    break;
            }
            return 0;
        }
        \endcode
    */
    namespace Motion
    {
        typedef double Time; /*!< \brief Time in seconds. Typical usage copies Frame::timestamp (OpenCV CAP_PROP_POS_MSEC * 0.001). */
        typedef int Id; /*!< \brief Object identifier. Object::id and Event::objectId; -1 if the event is not linked to an object. */
        typedef std::string String; /*!< \brief Text type. Event::text and the result of ToString(). */
        typedef Simd::Point<ptrdiff_t> Size; /*!< \brief Screen size in pixels (width and height). */
        typedef Simd::Point<ptrdiff_t> Point; /*!< \brief Screen point in pixels (x and y). Origin is the top-left corner. Typical usage is Object::trajectory[].point and DrawLine. */
        typedef std::vector<Point> Points; /*!< \brief Vector of screen points. */
        typedef Simd::Rectangle<ptrdiff_t> Rect; /*!< \brief Screen rectangle in pixels. Typical usage is Object::rect for DrawRectangle. */
        typedef Simd::Point<double> FSize; /*!< \brief ONVIF size (width and height) in range [0, 2]. Model::size; default (0.1, 0.1) is about 0.25% of the screen area. */
        typedef Simd::Point<double> FPoint; /*!< \brief ONVIF point (x and y) in range [-1, 1]. Origin is the screen center; X grows right, Y grows up. Used by Model::roi. */
        typedef std::vector<FPoint> FPoints; /*!< \brief Vector of ONVIF points. Model::roi polygon. */
        typedef Simd::View<Simd::Allocator> View; /*!< \brief Image type. Typical usage wraps OpenCV cv::Mat as Frame input and annotates Object::rect on it. */
        typedef Simd::Frame<Simd::Allocator> Frame; /*!< \brief Video frame. Typical usage is Frame(image, false, timestampInSeconds) as NextFrame input. */

        /*! @ingroup cpp_motion

            \short Converts a screen X-coordinate to an ONVIF X-coordinate.

            Maps pixel x in [0, screenWidth] to [-1, 1]. Left edge is -1, right edge is 1.
            Typical usage is ScreenToOnvif() when filling Model::roi from pixel points.

            \param [in] x - a screen X-coordinate in pixels.
            \param [in] screenWidth - a screen width in pixels.
            \return ONVIF X-coordinate in range [-1, 1].
        */
        SIMD_INLINE double ScreenToOnvifX(ptrdiff_t x, ptrdiff_t screenWidth)
        {
            return double(2 * x - screenWidth) / screenWidth;
        }

        /*! @ingroup cpp_motion

            \short Converts a screen Y-coordinate to an ONVIF Y-coordinate.

            Maps pixel y in [0, screenHeight] to [1, -1]. Top edge is 1, bottom edge is -1
            (ONVIF Y grows up). Typical usage is ScreenToOnvif() when filling Model::roi.

            \param [in] y - a screen Y-coordinate in pixels.
            \param [in] screenHeight - a screen height in pixels.
            \return ONVIF Y-coordinate in range [-1, 1].
        */
        SIMD_INLINE double ScreenToOnvifY(ptrdiff_t y, ptrdiff_t screenHeight)
        {
            return double(screenHeight - 2 * y) / screenHeight;
        }

        /*! @ingroup cpp_motion

            \short Converts screen 2D-coordinates to ONVIF 2D-coordinates.

            Combines ScreenToOnvifX and ScreenToOnvifY. Typical usage converts a pixel
            polygon to Model::roi. Detector uses the inverse OnvifToScreen when it
            applies the model.

            \param [in] point - a screen point in pixels.
            \param [in] screenSize - a screen size in pixels (width and height).
            \return ONVIF point in range [-1, 1].
        */
        SIMD_INLINE FPoint ScreenToOnvif(const Point & point, const Point & screenSize)
        {
            return FPoint(ScreenToOnvifX(point.x, screenSize.x), ScreenToOnvifY(point.y, screenSize.y));
        }

        /*! @ingroup cpp_motion

            \short Converts a screen 2D-size to an ONVIF 2D-size.

            Full screen maps to (2, 2). Typical usage converts a pixel object size
            to Model::size. Detector uses the inverse OnvifToScreenSize for Model::size.

            \param [in] size - a screen size in pixels (width and height).
            \param [in] screenSize - a screen size in pixels (width and height).
            \return ONVIF size in range [0, 2].
        */
        SIMD_INLINE FSize ScreenToOnvifSize(const Size & size, const Point & screenSize)
        {
            return FSize(double(size.x * 2 / screenSize.x), double(size.y * 2 / screenSize.y));
        }

        /*! @ingroup cpp_motion

            \short Converts an ONVIF X-coordinate to a screen X-coordinate.

            Maps x in [-1, 1] to pixels and clamps the result to [0, screenWidth - 1].
            Detector uses this through OnvifToScreen when it applies Model::roi.

            \param [in] x - an ONVIF X-coordinate in range [-1, 1].
            \param [in] screenWidth - a screen width in pixels.
            \return screen X-coordinate in pixels.
        */
        SIMD_INLINE ptrdiff_t OnvifToScreenX(double x, ptrdiff_t screenWidth)
        {
            return std::max(ptrdiff_t(0), std::min(screenWidth - 1, (ptrdiff_t)Simd::Round((1.0 + x)*screenWidth / 2.0)));
        }

        /*! @ingroup cpp_motion

            \short Converts an ONVIF Y-coordinate to a screen Y-coordinate.

            Maps y in [-1, 1] to pixels (ONVIF +Y is up) and clamps the result to
            [0, screenHeight - 1]. Detector uses this through OnvifToScreen when it
            applies Model::roi.

            \param [in] y - an ONVIF Y-coordinate in range [-1, 1].
            \param [in] screenHeight - a screen height in pixels.
            \return screen Y-coordinate in pixels.
        */
        SIMD_INLINE ptrdiff_t OnvifToScreenY(double y, ptrdiff_t screenHeight)
        {
            return std::max(ptrdiff_t(0), std::min(screenHeight - 1, (ptrdiff_t)Simd::Round((1.0 - y)*screenHeight / 2.0)));
        }

        /*! @ingroup cpp_motion

            \short Converts ONVIF 2D-coordinates to screen 2D-coordinates.

            Combines OnvifToScreenX and OnvifToScreenY. Detector uses this to convert
            Model::roi vertices to pixels.

            \param [in] point - an ONVIF point in range [-1, 1].
            \param [in] screenSize - a screen size in pixels (width and height).
            \return screen point in pixels.
        */
        SIMD_INLINE Point OnvifToScreen(const FPoint & point, const Point & screenSize)
        {
            return Point(OnvifToScreenX(point.x, screenSize.x), OnvifToScreenY(point.y, screenSize.y));
        }

        /*! @ingroup cpp_motion

            \short Converts an ONVIF 2D-size to a screen 2D-size.

            Detector uses this to convert Model::size to a pixel object size that
            sets the minimum motion-region area. Debug annotation of the model draws
            that minimum rectangle.

            \param [in] size - an ONVIF size in range [0, 2].
            \param [in] screenSize - a screen size in pixels (width and height).
            \return screen size in pixels.
        */
        SIMD_INLINE Size OnvifToScreenSize(const FSize & size, const Point & screenSize)
        {
            return Size(Round(size.x*screenSize.x / 2.0), Round(size.y*screenSize.y / 2.0));
        }

        /*! @ingroup cpp_motion

            \short Converts an object ID to a string.

            Typical usage draws ToString(object.id) above Object::rect with Simd::Font.

            \param [in] id - an object ID (Object::id or Event::objectId).
            \return decimal string representation of the ID.
        */
        SIMD_INLINE String ToString(Id id)
        {
            std::stringstream ss;
            ss << id;
            return ss.str();
        }

        /*! @ingroup cpp_motion

            \short Position of a detected object at one timestamp.

            Describes a screen 2D-point and time. Typical usage draws a polyline
            through Object::trajectory[j].point. time is copied from Frame::timestamp.
        */
        struct Position
        {
            Point point; /*!< \brief Screen point in pixels. Typical usage is DrawLine between consecutive trajectory points. */
            Time time; /*!< \brief Timestamp in seconds from Frame::timestamp of the corresponding frame. */
        };
        typedef std::vector<Position> Positions; /*!< \brief Object::trajectory. Typical usage iterates from 1 to size()-1 and draws lines. */

        /*! @ingroup cpp_motion

            \short A classified moving object on the current frame.

            Detector puts an Object into Metadata::objects after the motion region
            has lived at least Options::ClassificationTimeMin seconds and moved at
            least Options::ClassificationShiftMin of the screen diagonal. Static
            (not yet classified) tracks are not reported.

            Typical usage draws Object::rect, ToString(Object::id) above the box,
            and a polyline along Object::trajectory. The example paints the object
            red when some Event::objectId equals Object::id in the same frame.
        */
        struct Object
        {
            Id id; /*!< \brief Classification ID. Event::objectId of ObjectIn/ObjectOut uses the same value. Typical usage is ToString(id). */
            Rect rect; /*!< \brief Bounding box in input-frame pixels. Typical usage is DrawRectangle. */
            Positions trajectory; /*!< \brief Smoothed history of centers. Typical usage draws DrawLine between consecutive points. */
        };
        typedef std::vector<Object> Objects; /*!< \brief Metadata::objects: classified moving objects, including those that disappeared on this frame. */

        /*! @ingroup cpp_motion

            \short An event generated by Simd::Motion::Detector on the current frame.

            NextFrame clears Metadata::events and then appends events of this frame.
            Typical usage switches on Type to print "in N" / "out N" / "SABOTAGE ON" /
            "SABOTAGE OFF" and keeps a scrolling list of recent events.

            ObjectIn is emitted when a track is classified as moving.
            ObjectOut is emitted when a classified object is removed (absent longer
            than Options::TrackingRemoveTime, or dropped during background init).
            SabotageOn / SabotageOff are emitted when the total motion area crosses
            Options::StabilityRegionAreaMax. Sabotage events use objectId -1.
        */
        struct Event
        {
            /*!
                \enum Type

                Type of event generated by Detector::NextFrame.
            */
            enum Type
            {
                ObjectIn, /*!< \brief A new object was classified as moving. text is "ObjectIn"; objectId is Object::id. */
                ObjectOut, /*!< \brief A classified object disappeared. text is "ObjectOut"; objectId is Object::id. */
                SabotageOn, /*!< \brief Motion area exceeded Options::StabilityRegionAreaMax. text is "SabotageOn"; objectId is -1. */
                SabotageOff, /*!< \brief Motion area fell back below Options::StabilityRegionAreaMax. text is "SabotageOff"; objectId is -1. */
            } type; /*!< \brief Event type. Typical usage is a switch for annotation text. */

            String text; /*!< \brief Event text. Detector sets "ObjectIn", "ObjectOut", "SabotageOn" or "SabotageOff". */
            Id objectId; /*!< \brief Object::id for ObjectIn/ObjectOut, or -1 for sabotage events. */

            /*!
                Constructs Event.

                Detector constructs events internally. Typical usage reads type, text
                and objectId from Metadata::events.

                \param [in] type_ - a type of a new event.
                \param [in] text_ - a text description of the event. It is equal to empty string by default.
                \param [in] objectId_ - an ID of object concerned with this event. It is equal to -1 by default.
            */
            Event(Type type_, const String & text_ = String(), Id objectId_ = -1)
                : type(type_)
                , text(text_)
                , objectId(objectId_)
            {
            }
        };
        typedef std::vector<Event> Events; /*!< \brief Metadata::events of the current frame. Typical usage iterates them after objects. */

        /*! @ingroup cpp_motion

            \short Result of Detector::NextFrame for the current frame.

            objects are classified moving objects (including those just deleted on
            this frame). events are generated on this frame only: NextFrame clears
            the list first. Typical usage annotates objects (rect, id, trajectory)
            and then events (type and objectId).
        */
        struct Metadata
        {
            Objects objects; /*!< \brief Classified moving objects at the current frame. Typical usage draws rect, id and trajectory. */
            Events events; /*!< \brief Events generated at the current frame. NextFrame clears this list before filling it. */
        };

        /*! @ingroup cpp_motion

            \short Scene model used to calibrate Simd::Motion::Detector.

            Typical usage constructs Model (or uses the default), optionally sets
            size / roi / mask, and passes it to Detector::SetModel before NextFrame.

            size is the minimum object size in ONVIF units. Default (0.1, 0.1) is
            about 0.25% of the screen area. The shooting-star example uses (0.01, 0.01).

            ROI is either a polygon (roi, at least 3 ONVIF vertices) or a Gray8 mask.
            If mask.format is View::Gray8, Detector resizes it to the frame and treats
            non-zero pixels as inside ROI (EMPTY = 0, ROI = 255). Otherwise the polygon
            is used; fewer than 3 vertices mean the full screen.
        */
        struct Model
        {
            static const uint8_t EMPTY = 0; /*!< \brief Mask value outside ROI. Detector fills the polygon mask with this value. */
            static const uint8_t ROI = 255; /*!< \brief Mask value inside ROI. Typical usage writes ROI (or any non-zero) into Model::mask. */

            FSize size; /*!< \brief Minimum object size in ONVIF units [0, 2]. Default (0.1, 0.1). Shooting-star example uses (0.01, 0.01). */
            FPoints roi; /*!< \brief ROI polygon in ONVIF coordinates [-1, 1]. Used when mask is not Gray8. Empty (fewer than 3 points) means the full screen. */
            View mask; /*!< \brief ROI mask. If format is View::Gray8, Detector uses it instead of roi (non-zero pixels are inside). */

            /*!
                Copy constructor of Model.

                Copies size, roi and, if it is Gray8, a deep copy of mask.

                \param [in] model - other model.
            */
            Model(const Model & model)
                : size(model.size)
                , roi(model.roi)
            {
                if (model.mask.format == View::Gray8)
                {
                    mask.Recreate(model.mask.Size(), View::Gray8);
                    Copy(model.mask, mask);
                }
            }

            /*!
                Constructs Model from a minimum object size and an ROI polygon.

                If roi_ has fewer than 3 points, roi is set to the full-screen
                rectangle (-1, 1), (1, 1), (1, -1), (-1, -1). mask is left empty,
                so Detector uses the polygon.

                \param [in] size_ - a minimum object size in ONVIF units. Default is (0.1, 0.1) ~ 0.25% of screen area.
                \param [in] roi_ - a ROI polygon in ONVIF coordinates. Empty by default (full screen).
            */
            Model(const FSize & size_ = FSize(0.1, 0.1), const FPoints & roi_ = FPoints())
                : size(size_)
                , roi(roi_)
            {
                if (roi.size() < 3)
                {
                    roi.clear();
                    roi.push_back(FPoint(-1.0, 1.0));
                    roi.push_back(FPoint(1.0, 1.0));
                    roi.push_back(FPoint(1.0, -1.0));
                    roi.push_back(FPoint(-1.0, -1.0));
                }
            }

            /*!
                Constructs Model from a minimum object size and an ROI mask.

                If mask_ is Gray8, it is deep-copied to mask. Otherwise roi is set
                to the full screen. Typical usage creates a small Gray8 mask (for
                example 20x20, left half filled with Model::ROI) and calls SetModel.

                \param [in] size_ - a minimum object size in ONVIF units.
                \param [in] mask_ - a ROI mask. It must be 8-bit gray image (View::Gray8).
            */
            Model(const FSize& size_, const View & mask_)
                : size(size_)
            {
                if (mask_.format == View::Gray8)
                {
                    mask.Recreate(mask_.Size(), View::Gray8);
                    Copy(mask_, mask);
                }
                else
                {
                    roi.push_back(FPoint(-1.0, 1.0));
                    roi.push_back(FPoint(1.0, 1.0));
                    roi.push_back(FPoint(1.0, -1.0));
                    roi.push_back(FPoint(-1.0, -1.0));
                }
            }

            /*!
                Copy assignment of Model.

                Copies size, roi and, if it is Gray8, a deep copy of mask.

                \param [in] model - other model.
            */
            Model& operator=(const Model& model)
            {
                size = model.size;
                roi = model.roi;
                if (model.mask.format == View::Gray8)
                {
                    mask.Recreate(model.mask.Size(), View::Gray8);
                    Copy(model.mask, mask);
                }
                return *this;
            }
        };

        /*! @ingroup cpp_motion

            \short Options used by Simd::Motion::Detector.

            Typical usage constructs Options (defaults), changes a few fields and
            calls Detector::SetOptions before NextFrame. The shooting-star example
            sets TrackingAdditionalLinking = 5, ClassificationShiftMin = 0.01,
            ClassificationTimeMin = 0.01, DifferenceDxFeatureWeight = 0,
            DifferenceDyFeatureWeight = 0 and BackgroundStatUpdateTime = 0.2.

            Weight 0 disables the corresponding difference feature.
            Debug* fields apply only when NextFrame is given a Bgr24 output Frame
            of the same size as input.
        */
        struct Options
        {
            int CalibrationScaleLevelMax;  /*!< \brief Maximum downscale of the input (pyramid levels). Default 3 means up to 8 times. Applied on calibration (first NextFrame or input size change). */

            int DifferenceGrayFeatureWeight; /*!< \brief Weight of the gray feature in difference estimation. Default 18. 0 disables the gray feature. */
            int DifferenceDxFeatureWeight; /*!< \brief Weight of the X-gradient feature in difference estimation. Default 18. 0 disables it (shooting-star example). */
            int DifferenceDyFeatureWeight; /*!< \brief Weight of the Y-gradient feature in difference estimation. Default 18. 0 disables it (shooting-star example). */
            bool DifferencePropagateForward; /*!< \brief Boost difference by taking the max with the reduced previous pyramid level. Default true. */
            bool DifferenceRoiMaskEnable; /*!< \brief Restrict difference by the ROI mask. Default true. */

            double BackgroundGrowTime; /*!< \brief Duration (seconds) of fast background grow after init. Default 1 second. */
            double BackgroundStatUpdateTime; /*!< \brief Interval (seconds) of background statistics updates in normal mode. Default 0.04. Shooting-star example uses 0.2. */
            double BackgroundUpdateTime; /*!< \brief Interval (seconds) between background range adjustments in normal mode. Default 1 second. */
            int BackgroundSabotageCountMax; /*!< \brief Consecutive sabotage frames allowed before background reinitialization. Default 3. */

            double SegmentationCreateThreshold; /*!< \brief Threshold in [0, 1] to seed a motion region. Default 0.5. */
            double SegmentationExpandCoefficient; /*!< \brief Expansion of a motion region relative to the create threshold, in [0, 1]. Default 0.75. */

            double StabilityRegionAreaMax; /*!< \brief Maximum fraction of the frame that may be in motion; above this Detector emits SabotageOn. Range [0, 1]. Default 0.5. */

            int TrackingTrajectoryMax; /*!< \brief Maximum length of Object::trajectory. Default 1024. */
            double TrackingRemoveTime; /*!< \brief Seconds without an update after which an object is removed (ObjectOut if it was classified). Default 1 second. */
            double TrackingAdditionalLinking; /*!< \brief Extra border (fraction of region size) when linking a region to a track. Default 0. Shooting-star example uses 5. */
            int TrackingAveragingHalfRange; /*!< \brief Half-window (in trajectory points) used to smooth Object::rect and Object::trajectory. Default 12. */

            double ClassificationShiftMin; /*!< \brief Minimum displacement (in screen diagonals) to classify a track as a moving object. Default 0.075. Shooting-star example uses 0.01. */
            double ClassificationTimeMin; /*!< \brief Minimum lifetime (seconds) to classify a track as a moving object. Default 1 second. Shooting-star example uses 0.01. */

            int DebugDrawLevel; /*!< \brief Pyramid level of the debug inset. Default 1. Used with DebugDrawBottomRight when output is Bgr24. */
            int DebugDrawBottomRight; /*!< \brief Debug inset in the bottom-right corner: 0 empty; 1 difference; 2 gray; 3 dx; 4 dy. Default 0. Requires Bgr24 output. */
            bool DebugAnnotateModel; /*!< \brief Draw ROI polygon and minimum object size on Bgr24 output. Default false. */
            bool DebugAnnotateMovingRegions; /*!< \brief Draw raw motion regions on Bgr24 output. Default false. */
            bool DebugAnnotateTrackingObjects; /*!< \brief Draw all tracks (including not yet classified) on Bgr24 output. Default false. */

            /*!
                Default constructor of Options.

                Fills every field with the defaults listed above. Typical usage
                changes a few fields and calls Detector::SetOptions.
            */
            Options()
            {
                CalibrationScaleLevelMax = 3;

                DifferenceGrayFeatureWeight = 18;
                DifferenceDxFeatureWeight = 18;
                DifferenceDyFeatureWeight = 18;
                DifferencePropagateForward = true;
                DifferenceRoiMaskEnable = true;

                BackgroundGrowTime = 1.0;
                BackgroundStatUpdateTime = 0.04;
                BackgroundUpdateTime = 1.0;
                BackgroundSabotageCountMax = 3;

                SegmentationCreateThreshold = 0.5;
                SegmentationExpandCoefficient = 0.75;

                StabilityRegionAreaMax = 0.5;

                TrackingTrajectoryMax = 1024;
                TrackingRemoveTime = 1.0;
                TrackingAdditionalLinking = 0.0;
                TrackingAveragingHalfRange = 12;

                ClassificationShiftMin = 0.075;
                ClassificationTimeMin = 1.0;

                DebugDrawLevel = 1;
                DebugDrawBottomRight = 0;
                DebugAnnotateModel = false;
                DebugAnnotateMovingRegions = false;
                DebugAnnotateTrackingObjects = false;
            }
        };

        /*! @ingroup cpp_motion

            \short Motion detector.

            Processes a sequence of frames and reports classified moving objects
            and events through Metadata.

            Typical usage:
            - construct Detector (default Model and Options);
            - optionally SetModel and SetOptions before the video loop;
            - for each frame: Frame input(view, false, timestampInSeconds),
              NextFrame(input, metadata), then annotate metadata.objects and
              metadata.events on the image.

            NextFrame must be called for every frame in order. Timestamp is used
            for background update, tracking timeout and classification lifetime.
            Recalibration runs when the input size changes.

            The optional output Frame, if given, must have the same size as input.
            Debug drawing (Options::Debug*) is applied only when output is Bgr24.
        */
        class Detector
        {
        public:

            /*!
                Default constructor of Detector.

                Uses default Model and Options. Typical usage then optionally calls
                SetModel / SetOptions and starts the NextFrame loop.
            */
            Detector()
            {
            }

            /*!
                Destructor of Detector.
            */
            virtual ~Detector()
            {
            }

            /*!
                Sets options of the motion detector.

                Typical usage changes a few Options fields (see the shooting-star
                example) and calls this once before NextFrame. Pyramid scale and
                difference-feature allocation are applied on the next calibration
                (first NextFrame or input size change). Other fields are read every
                frame.

                \param [in] options - options of motion detector.
                \return true.
            */
            bool SetOptions(const Simd::Motion::Options & options)
            {
                *(Simd::Motion::Options*)(&_options) = options;
                return true;
            }

            /*!
                Sets the scene model of the motion detector.

                Typical usage sets Model::size and optionally Model::mask or
                Model::roi, then calls this once before NextFrame. The model is
                applied on the next calibration (first NextFrame or input size change).

                \param [in] model - a model of the scene.
                \return true.
            */
            bool SetModel(const Model & model)
            {
                _model = model;
                return true;
            }

            /*!
                Processes the next video frame.

                Call this for every frame in order. input is typically
                Frame(image, false, capture.get(cv::CAP_PROP_POS_MSEC) * 0.001).
                metadata.events is cleared, then filled with events of this frame.
                metadata.objects receives classified moving objects (including those
                that disappeared on this frame).

                output may be NULL. If it is not NULL it must have the same size as
                input; otherwise the function returns false. Debug annotation from
                Options is drawn only when output is Bgr24. Typical usage either
                annotates the input View (as in the example) or passes an output
                Frame and annotates output->planes[0].

                \param [in] input - a current input frame (image plus timestamp in seconds).
                \param [out] metadata - detected objects and generated events of this frame.
                \param [out] output - optional frame for debug annotation. Can be NULL.
                \return true on success; false if output is not NULL and its size differs from input.
            */
            bool NextFrame(const Frame & input, Metadata & metadata, Frame * output = NULL)
            {
                SIMD_CHECK_PERFORMANCE();

                if (output && output->Size() != input.Size())
                    return false;

                if (!Calibrate(input.Size()))
                    return false;

                _scene.metadata = &metadata;
                _scene.metadata->events.clear();

                SetFrame(input, output);

                EstimateTextures();

                EstimateDifference();

                PerformSegmentation();

                VerifyStability();

                TrackObjects();

                ClassifyObjects();

                UpdateBackground();

                SetMetadata();

                DebugAnnotation();

                return true;
            }

        private:
            Simd::Motion::Model _model;

            struct Options : public Simd::Motion::Options
            {
                int CalibrationLevelCountMin;
                int CalibrationTopLevelSizeMin;
                int CalibrationObjectAreaMin;

                int TextureGradientSaturation;
                int TextureGradientBoost;


                Options()
                    : Simd::Motion::Options()
                {
                    CalibrationLevelCountMin = 3;
                    CalibrationTopLevelSizeMin = 32;
                    CalibrationObjectAreaMin = 16;

                    TextureGradientSaturation = 16;
                    TextureGradientBoost = 4;
                }
            } _options;

            typedef std::pair<size_t, size_t> Scanline;
            typedef std::vector<Scanline> Scanlines;
            typedef std::vector<Rect> Rects;
            typedef Simd::Rectangle<double> FRect;
            typedef Simd::Pyramid<Simd::Allocator> Pyramid;

            struct SearchRegion
            {
                Rect rect; // rectangle on corresponding pyramid level (scale)
                int scale; // pyramid level
                Scanlines scanlines;

                SearchRegion()
                    : scale(0)
                {
                }

                SearchRegion(const Rect & rect_, const int & scale_)
                    : rect(rect_)
                    , scale(scale_)
                {
                }
            };
            typedef std::vector<SearchRegion> SearchRegions;

            struct Model
            {
                Size originalFrameSize;

                Size frameSize;
                size_t scale;
                size_t scaleLevel;

                size_t levelCount;
                int areaRegionMinEstimated;

                Points roi;
                Pyramid roiMask;
                SearchRegions searchRegions;
            };

            struct Object;

            struct MovingRegion
            {
                Rects rects;

                uint8_t index;
                Rect rect;
                int level;
                Time time;
                Point point;
                Detector::Object * object, * nearest;

                MovingRegion(const uint8_t & index_, const Rect & rect_, int level_, const Time & time_)
                    : index(index_)
                    , rect(rect_)
                    , level(level_)
                    , time(time_)
                    , object(NULL)
                    , nearest(NULL)
                {
                    rects.resize(level + 1);
                }
            };
            typedef std::shared_ptr<MovingRegion> MovingRegionPtr;
            typedef std::vector<MovingRegionPtr> MovingRegionPtrs;

            struct Texture
            {
                struct Bound
                {
                    Pyramid value;
                    Pyramid count;

                    void Create(const Size & size, size_t levelCount)
                    {
                        value.Recreate(size, levelCount);
                        count.Recreate(size, levelCount);
                    }
                };

                struct Feature
                {
                    Pyramid value;
                    Bound lo;
                    Bound hi;
                    uint16_t weight;

                    void Create(const Size & size, size_t levelCount, int weight_)
                    {
                        value.Recreate(size, levelCount);
                        lo.Create(size, levelCount);
                        hi.Create(size, levelCount);
                        weight = uint16_t(weight_ * 256);
                    }
                };

                enum FeatureType
                {
                    FeatureGray,
                    FeatureDx,
                    FeatureDy,
                };

                Feature gray;
                Feature dx;
                Feature dy;

                typedef std::vector<Feature *> Features;
                Features features;

                void Create(const Size & size, size_t levelCount, const Options & options)
                {
                    gray.Create(size, levelCount, options.DifferenceGrayFeatureWeight);
                    if (options.DifferenceDxFeatureWeight || options.DifferenceDyFeatureWeight)
                    {
                        dx.Create(size, levelCount, options.DifferenceDxFeatureWeight);
                        dy.Create(size, levelCount, options.DifferenceDyFeatureWeight);
                    }

                    features.clear();
                    if (options.DifferenceGrayFeatureWeight)
                        features.push_back(&gray);
                    if(options.DifferenceDxFeatureWeight)
                        features.push_back(&dx);
                    if (options.DifferenceDyFeatureWeight)
                        features.push_back(&dy);
                }
            };

            struct Background
            {
                enum State
                {
                    Init,
                    Grow,
                    Update
                };

                State state;
                int updateCounter;
                int sabotageCounter;
                Time growEndTime;
                Time lastFrameTime;
                Time statUpdateTime;
                Time updateTime;

                Background()
                    : state(Init)
                {
                }
            };

            struct Stability
            {
                enum State
                {
                    Stable,
                    Sabotage
                } state;

                Stability()
                    : state(Stable)
                {
                }
            };

            struct Segmentation
            {
                enum MaskIndices
                {
                    MaskNotVisited = 0,
                    MaskSeed = 1,
                    MaskInvalid = 2,
                    MaskIndexSize,
                };

                Pyramid mask;

                int differenceCreationMin;
                int differenceExpansionMin;

                MovingRegionPtrs movingRegions;
            };

            struct Object
            {
                Id trackingId, classificationId;
                Point center; 
                Rect rect; 
                MovingRegionPtrs trajectory; 

                enum Type
                {
                    Static,
                    Moving,
                } type;

                Point pointStart;
                Time timeStart;

                Object(const Id trackingId_, const MovingRegionPtr & region)
                    : trackingId(trackingId_)
                    , classificationId(-1)
                    , center(region->rect.Center())
                    , rect(region->rect)
                    , type(Static)
                    , pointStart(region->rect.Center())
                    , timeStart(region->time)
                {
                    trajectory.push_back(region);
                }
            };
            typedef std::shared_ptr<Object> ObjectPtr;
            typedef std::vector<ObjectPtr> ObjectPtrs;

            struct Tracking
            {
                ObjectPtrs objects;
                ObjectPtrs justDeletedObjects;
                Id id; 

                Tracking() 
                    : id(0)
                {
                }
            };

            struct Classification
            {
                ptrdiff_t squareShiftMin;
                Id id;

                Classification()
                    : id(0)
                {
                }
            };

            struct Scene
            {
                Frame input, * output;
                Pyramid scaled;
                Metadata * metadata;

                Font font;
                Pyramid buffer;
                Detector::Model model;

                Texture texture;

                Background background;

                Stability stability;

                Pyramid difference;

                Segmentation segmentation;

                Tracking tracking;

                Classification classification;

                void Create(const Options & options)
                {
                    scaled.Recreate(model.originalFrameSize, model.scaleLevel + 1);
                    font.Resize(model.originalFrameSize.y / 32);
                    buffer.Recreate(model.frameSize, model.levelCount);

                    texture.Create(model.frameSize, model.levelCount, options);
                    difference.Recreate(model.frameSize, model.levelCount);

                    segmentation.mask.Recreate(model.frameSize, model.levelCount);
                    segmentation.differenceCreationMin = int(255 * options.SegmentationCreateThreshold);
                    segmentation.differenceExpansionMin = int(255 * options.SegmentationExpandCoefficient*options.SegmentationCreateThreshold);

                    classification.squareShiftMin = ptrdiff_t(Simd::SquaredDistance(model.frameSize, Point())*
                        options.ClassificationShiftMin*options.ClassificationShiftMin);
                }
            };
            Scene _scene;

            void SetFrame(const Frame & input, Frame * output)
            {
                SIMD_CHECK_PERFORMANCE();

                _scene.input = input;
                _scene.output = output;
                Simd::Convert(input, Frame(_scene.scaled[0]).Ref());
                Simd::Build(_scene.scaled, SimdReduce2x2);
            }

            bool Calibrate(const Size & frameSize)
            {
                Model & model = _scene.model;

                if (model.originalFrameSize == frameSize)
                    return true;

                SIMD_CHECK_PERFORMANCE();

                model.originalFrameSize = frameSize;

                EstimateModelParameters(model);
                SetScreenRoi(model);
                GenerateSearchRegion(model);
                GenerateSearchRegionScanlines(model);

                _scene.Create(_options);

                return true;
            }

            void EstimateModelParameters(Model & model)
            {
                Size objectSize = OnvifToScreenSize(_model.size, model.originalFrameSize);
                Size size = model.originalFrameSize;
                model.areaRegionMinEstimated = int(objectSize.x*objectSize.y);
                int levelCount = 1;
                while (size.x >= _options.CalibrationTopLevelSizeMin && size.y >= _options.CalibrationTopLevelSizeMin && model.areaRegionMinEstimated > _options.CalibrationObjectAreaMin)
                {
                    size = Simd::Scale(size);
                    ++levelCount;
                    model.areaRegionMinEstimated /= 4;
                }
                model.areaRegionMinEstimated = std::max(model.areaRegionMinEstimated, _options.CalibrationObjectAreaMin / 4 + 1);
                model.scaleLevel = std::min(std::max(levelCount - _options.CalibrationLevelCountMin, 0), _options.CalibrationScaleLevelMax);
                model.levelCount = levelCount - model.scaleLevel;
                model.scale = size_t(1) << model.scaleLevel;
                model.frameSize = model.originalFrameSize;
                for (size_t level = 0; level < model.scaleLevel; ++level)
                    model.frameSize = Simd::Scale(model.frameSize);
            }

            void SetScreenRoi(Model & model)
            {
                if (_model.roi.size() > 2)
                {
                    model.roi.resize(_model.roi.size());
                    for (size_t i = 0; i < _model.roi.size(); ++i)
                        model.roi[i] = OnvifToScreen(_model.roi[i], model.frameSize);
                }
                else
                {
                    model.roi.clear();
                    model.roi.push_back(Point(0, 0));
                    model.roi.push_back(Point(model.frameSize.x, 0));
                    model.roi.push_back(Point(model.frameSize.x, model.frameSize.y));
                    model.roi.push_back(Point(0, model.frameSize.y));
                }
            }

            void GenerateSearchRegion(Model & model)
            {
                model.searchRegions.clear();

                Size size(model.frameSize);
                for (size_t level = 1; level < model.levelCount; level++)
                    size = Simd::Scale(size);

                int level = (int)model.levelCount - 1;
                const Rect rect(1, 1, size.x - 1, size.y - 1);
                model.searchRegions.push_back(SearchRegion(rect, level));
            }

            void GenerateSearchRegionScanlines(Model & model)
            {
                model.roiMask.Recreate(model.frameSize, model.levelCount);
                if (_model.mask.format == View::Gray8)
                    Simd::Resize(_model.mask, model.roiMask[0]);
                else
                {
                    Simd::Fill(model.roiMask, Motion::Model::EMPTY);
                    DrawFilledPolygon(model.roiMask[0], model.roi, (uint8_t)Motion::Model::ROI);
                }
                Simd::Build(model.roiMask, SimdReduce4x4);

                for (size_t i = 0; i < model.searchRegions.size(); ++i)
                {
                    SearchRegion & region = model.searchRegions[i];
                    assert(region.scale < (int)model.roiMask.Size());

                    const View & view = model.roiMask[region.scale];
                    const Rect & rect = region.rect;
                    for (ptrdiff_t row = rect.Top(); row < rect.Bottom(); ++row)
                    {
                        ptrdiff_t offset = row * view.stride + rect.Left();
                        ptrdiff_t end = offset + rect.Width();
                        for (; offset < end;)
                        {
                            if (view.data[offset])
                            {
                                Scanline scanline;
                                scanline.first = offset;
                                while (++offset < end && view.data[offset]);
                                scanline.second = offset;
                                region.scanlines.push_back(scanline);
                            }
                            else
                                ++offset;
                        }
                    }
                }
            }

            void EstimateTextures()
            {
                SIMD_CHECK_PERFORMANCE();

                Texture & texture = _scene.texture;
                Simd::Copy(_scene.scaled.Top(), texture.gray.value[0]);
                Simd::Build(texture.gray.value, SimdReduce4x4);
                if (_options.DifferenceDxFeatureWeight || _options.DifferenceDyFeatureWeight)
                {
                    for (size_t i = 0; i < texture.gray.value.Size(); ++i)
                    {
                        Simd::TextureBoostedSaturatedGradient(texture.gray.value[i],
                            _options.TextureGradientSaturation, _options.TextureGradientBoost,
                            texture.dx.value[i], texture.dy.value[i]);
                    }
                }
            }

            void EstimateDifference()
            {
                SIMD_CHECK_PERFORMANCE();

                const Texture & texture = _scene.texture;
                Pyramid & difference = _scene.difference;
                Pyramid & buffer = _scene.buffer;
                for (size_t i = 0; i < difference.Size(); ++i)
                {
                    Simd::Fill(difference[i], 0);
                    for (size_t j = 0; j < texture.features.size(); ++j)
                    {
                        const Texture::Feature & feature = *texture.features[j];
                        Simd::AddFeatureDifference(feature.value[i], feature.lo.value[i], feature.hi.value[i], feature.weight, difference[i]);
                    }
                }
                if (_options.DifferencePropagateForward)
                {
                    for (size_t i = 1; i < difference.Size(); ++i)
                    {
                        Simd::ReduceGray4x4(difference[i - 1], buffer[i]);
                        Simd::OperationBinary8u(difference[i], buffer[i], difference[i], SimdOperationBinary8uMaximum);
                    }
                }
                if (_options.DifferenceRoiMaskEnable)
                {
                    for (size_t i = 0; i < difference.Size(); ++i)
                        Simd::OperationBinary8u(difference[i], _scene.model.roiMask[i], difference[i], SimdOperationBinary8uAnd);
                }
            }

            void PerformSegmentation()
            {
                SIMD_CHECK_PERFORMANCE();

                Point neighbours[4];
                neighbours[0] = Point(-1, 0);
                neighbours[1] = Point(0, -1);
                neighbours[2] = Point(1, 0);
                neighbours[3] = Point(0, 1);

                Segmentation & segmentation = _scene.segmentation;
                const Model & model = _scene.model;
                const Time & time = _scene.input.timestamp;

                segmentation.movingRegions.clear();

                Simd::Fill(segmentation.mask, Segmentation::MaskNotVisited);
                for (size_t i = 0; i < model.searchRegions.size(); ++i)
                {
                    View & mask = segmentation.mask.At(model.searchRegions[i].scale);
                    Simd::FillFrame(mask, Rect(1, 1, mask.width - 1, mask.height - 1), Segmentation::MaskInvalid);
                }

                for (size_t i = 0; i < model.searchRegions.size(); ++i)
                {
                    const SearchRegion & searchRegion = model.searchRegions[i];
                    int level = searchRegion.scale;
                    const View & difference = _scene.difference.At(level);
                    View & mask = segmentation.mask.At(level);
                    Rect roi = searchRegion.rect;

                    for (size_t i = 0; i < searchRegion.scanlines.size(); ++i)
                    {
                        const Scanline & scanline = searchRegion.scanlines[i];
                        for (size_t offset = scanline.first; offset < scanline.second; ++offset)
                        {
                            if (difference.data[offset] > segmentation.differenceCreationMin && mask.data[offset] == Segmentation::MaskNotVisited)
                                mask.data[offset] = Segmentation::MaskSeed;
                        }
                    }

                    ShrinkRoi(mask, roi, Segmentation::MaskSeed);
                    roi &= searchRegion.rect;

                    for (ptrdiff_t y = roi.top; y < roi.bottom; ++y)
                    {
                        for (ptrdiff_t x = roi.left; x < roi.right; ++x)
                        {
                            if (mask.At<uint8_t>(x, y) == Segmentation::MaskSeed)
                            {
                                std::stack<Point> stack;
                                stack.push(Point(x, y));
                                if (segmentation.movingRegions.size() + Segmentation::MaskIndexSize > UINT8_MAX)
                                    return;
                                MovingRegionPtr region(new MovingRegion(uint8_t(segmentation.movingRegions.size() + Segmentation::MaskIndexSize), Rect(), level, time));
                                while (!stack.empty())
                                {
                                    Point current = stack.top();
                                    stack.pop();
                                    mask.At<uint8_t>(current) = region->index;
                                    region->rect |= current;
                                    for (size_t n = 0; n < 4; ++n)
                                    {
                                        Point neighbour = current + neighbours[n];
                                        if (difference.At<uint8_t>(neighbour) > segmentation.differenceExpansionMin &&	mask.At<uint8_t>(neighbour) <= Segmentation::MaskSeed)
                                            stack.push(neighbour);
                                    }
                                }

                                if (region->rect.Area() <= model.areaRegionMinEstimated)
                                    Simd::SegmentationChangeIndex(segmentation.mask[region->level].Region(region->rect).Ref(), region->index, Segmentation::MaskInvalid);
                                else
                                {
                                    ComputeIndex(segmentation, *region);
                                    if (!region->rect.Empty())
                                    {
                                        region->level = searchRegion.scale;
                                        region->point = region->rect.Center();
                                        segmentation.movingRegions.push_back(region);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            SIMD_INLINE void ShrinkRoi(const View & mask, Rect & roi, uint8_t index)
            {
                Simd::SegmentationShrinkRegion(mask, index, roi);
                if (!roi.Empty())
                    roi.AddBorder(1);
            }

            SIMD_INLINE void ExpandRoi(const Rect & roiParent, const Rect & rectChild, Rect & roiChild)
            {
                roiChild.SetTopLeft(roiParent.TopLeft() * 2 - Point(1, 1));
                roiChild.SetBottomRight(roiParent.BottomRight() * 2 + Point(1, 1));
                roiChild.AddBorder(1);
                roiChild &= rectChild;
            }

            void ComputeIndex(const View & parentMask, View & childMask, const View & difference, MovingRegion & region, int differenceExpansionMin)
            {
                Rect rect = region.rect;
                rect.right++;
                rect.bottom++;
                Simd::SegmentationPropagate2x2(parentMask.Region(rect), childMask.Region(2 * rect).Ref(), difference.Region(2 * rect),
                    region.index, Segmentation::MaskInvalid, Segmentation::MaskNotVisited, differenceExpansionMin);

                Rect rectChild(childMask.Size());
                rectChild.AddBorder(-1);

                ExpandRoi(region.rect, rectChild, region.rect);
                ShrinkRoi(childMask, region.rect, region.index);
                region.rect &= rectChild;
            }

            void ComputeIndex(Segmentation & segmentation, MovingRegion & region)
            {
                region.rects[region.level] = region.rect;

                int level = region.level;
                std::stack<Rect> rects;
                for (; region.level > 0; --region.level)
                {
                    const int levelChild = region.level - 1;

                    rects.push(region.rect);
                    ComputeIndex(segmentation.mask[region.level], segmentation.mask[levelChild], _scene.difference[levelChild], region, segmentation.differenceExpansionMin);

                    region.rects[region.level - 1] = region.rect;

                    if (region.rect.Empty())
                    {
                        for (; region.level <= level; region.level++)
                        {
                            region.rect = rects.top();
                            rects.pop();
                            Simd::SegmentationChangeIndex(segmentation.mask[region.level].Region(region.rect).Ref(), region.index, Segmentation::MaskInvalid);
                        }
                        region.rect = Rect();
                        return;
                    }
                }
            }

            void VerifyStability()
            {
                SIMD_CHECK_PERFORMANCE();

                if (_scene.background.state == Background::Init)
                    return;
                View mask = _scene.segmentation.mask[0];
                uint32_t count;
                Simd::ConditionalCount8u(mask, Segmentation::MaskIndexSize, SimdCompareGreaterOrEqual, count);
                bool sabotage = count >= mask.Area()*_options.StabilityRegionAreaMax;
                if (sabotage)
                {
                    if (_scene.stability.state != Stability::Sabotage)
                        _scene.metadata->events.push_back(Event(Event::SabotageOn, "SabotageOn"));
                    _scene.stability.state = Stability::Sabotage;
                }
                else
                {
                    if (_scene.stability.state == Stability::Sabotage)
                        _scene.metadata->events.push_back(Event(Event::SabotageOff, "SabotageOff"));
                    _scene.stability.state = Stability::Stable;
                }
            }

            void TrackObjects()
            {
                SIMD_CHECK_PERFORMANCE();

                if (_scene.background.state != Background::Update)
                {
                    RemoveAllObjects();
                    return;
                }

                RefreshObjectsTrajectory();

                DeleteOldObjects();

                SetNearestObjects();

                LinkObjects();

                AddNewObjects();
            }

            void RemoveAllObjects()
            {
                _scene.tracking.justDeletedObjects.clear();
                if (_scene.tracking.objects.size())
                {
                    for (size_t i = 0; i < _scene.tracking.objects.size(); ++i)
                    {
                        ObjectPtr & object = _scene.tracking.objects[i];
                        if (object->type == Object::Moving)
                            _scene.metadata->events.push_back(Event(Event::ObjectOut, "ObjectOut", object->classificationId));
                        _scene.tracking.justDeletedObjects.push_back(object);
                    }
                    _scene.tracking.objects.clear();
                }
            }

            void RefreshObjectsTrajectory()
            {
                ObjectPtrs & objects = _scene.tracking.objects;
                for (size_t j = 0; j < objects.size(); ++j)
                {
                    ObjectPtr & object = objects[j];
                    if (object->trajectory.size() > (size_t)_options.TrackingTrajectoryMax)
                        object->trajectory.erase(object->trajectory.begin());
                }
            }

            void DeleteOldObjects()
            {
                Time current = _scene.input.timestamp;
                Tracking & tracking = _scene.tracking;
                tracking.justDeletedObjects.clear();
                ObjectPtrs buffer;
                for (size_t i = 0; i < tracking.objects.size(); ++i)
                {
                    const ObjectPtr & object = tracking.objects[i];
                    if (current - object->trajectory.back()->time < _options.TrackingRemoveTime)
                        buffer.push_back(object);
                    else
                    {
                        tracking.justDeletedObjects.push_back(object);
                        if (object->type == Object::Moving)
                            _scene.metadata->events.push_back(Event(Event::ObjectOut, "ObjectOut", object->classificationId));
                    }
                }
                tracking.objects.swap(buffer);
            }

            void SetNearestObjects()
            {
                for (size_t i = 0; i < _scene.segmentation.movingRegions.size(); ++i)
                {
                    MovingRegion & region = *_scene.segmentation.movingRegions[i];
                    region.nearest = NULL;
                    ptrdiff_t minDifferenceSquared = std::numeric_limits<ptrdiff_t>::max();
                    for (size_t j = 0; j < _scene.tracking.objects.size(); ++j)
                    {
                        Detector::Object * object = _scene.tracking.objects[j].get();
                        const ptrdiff_t differenceSquared = Simd::SquaredDistance(object->center, region.rect.Center());
                        if (differenceSquared < minDifferenceSquared)
                        {
                            minDifferenceSquared = differenceSquared;
                            region.nearest = object;
                        }
                    }
                }
            }

            void LinkObjects()
            {
                for (size_t i = 0; i < _scene.tracking.objects.size(); ++i)
                {
                    ObjectPtr & object = _scene.tracking.objects[i];
                    MovingRegionPtr nearest;
                    ptrdiff_t minDifferenceSquared = std::numeric_limits<ptrdiff_t>::max();
                    for (size_t j = 0; j < _scene.segmentation.movingRegions.size(); ++j)
                    {
                        MovingRegionPtr & region = _scene.segmentation.movingRegions[j];
                        if (region->object != NULL)
                            continue;
                        if (object.get() != region->nearest)
                            continue;
                        Rect regionRect = Enlarged(region->rect);
                        Rect objectRect = Enlarged(object->rect);
                        ptrdiff_t differenceSquared = Simd::SquaredDistance(object->center, region->rect.Center());
                        if (regionRect.Contains(object->center) || objectRect.Contains(region->rect.Center()))
                        {
                            if (differenceSquared < minDifferenceSquared)
                            {
                                minDifferenceSquared = differenceSquared;
                                nearest = region;
                            }
                        }
                    }
                    if (nearest)
                    {
                        nearest->object = object.get();
                        object->trajectory.push_back(nearest);
                        Rect sum;
                        size_t end = object->trajectory.size(), start = std::max<ptrdiff_t>(0, end - _options.TrackingAveragingHalfRange);
                        for (size_t j = start; j < end; ++j)
                            sum += object->trajectory[j]->rect;
                        object->rect = sum / (end - start);
                        object->rect.Shift(nearest->rect.Center() - object->rect.Center());
                        object->rect &= Rect(_scene.model.frameSize);
                        object->center = nearest->rect.Center();
                    }
                }
            }

            SIMD_INLINE Rect Enlarged(Rect rect)
            {
                ptrdiff_t size = (rect.Width() + rect.Height()) / 2;
                ptrdiff_t border = ptrdiff_t(::ceil(size*_options.TrackingAdditionalLinking));
                rect.AddBorder(border);
                return rect;
            }

            void AddNewObjects()
            {
                for (size_t j = 0; j < _scene.segmentation.movingRegions.size(); ++j)
                {
                    const MovingRegionPtr & region = _scene.segmentation.movingRegions[j];
                    if (region->object != NULL)
                        continue;
                    bool contained = false;
                    for (size_t i = 0; i < _scene.tracking.objects.size(); ++i)
                    {
                        const ObjectPtr & object = _scene.tracking.objects[i];
                        if (object->rect.Contains(region->rect.Center()))
                        {
                            contained = true;
                            break;
                        }
                    }
                    if (!contained)
                    {
                        ObjectPtr object(new Object(_scene.tracking.id++, region));
                        region->object = object.get();
                        _scene.tracking.objects.push_back(object);
                    }
                }
            }

            void ClassifyObjects()
            {
                for (size_t i = 0; i < _scene.tracking.objects.size(); ++i)
                {
                    Object & object = *_scene.tracking.objects[i];
                    if (object.type == Object::Static)
                    {
                        Time time = _scene.input.timestamp - object.timeStart;
                        ptrdiff_t squareShift = Simd::SquaredDistance(object.trajectory.back()->point, object.pointStart);
                        if (time >= _options.ClassificationTimeMin && squareShift >= _scene.classification.squareShiftMin)
                        {
                            object.type = Object::Moving;
                            object.classificationId = _scene.classification.id++;
                            _scene.metadata->events.push_back(Event(Event::ObjectIn, "ObjectIn", object.classificationId));
                        }
                    }
                }
            }

            struct InitUpdater
            {
                void operator()(View & value, View & loValue, View & loCount, View & hiValue, View & hiCount) const
                {
                    Simd::Copy(value, loValue);
                    Simd::Copy(value, hiValue);
                    Simd::Fill(loCount, 0);
                    Simd::Fill(hiCount, 0);
                }
            };

            struct GrowRangeUpdater
            {
                void operator()(View & value, View & loValue, View & loCount, View & hiValue, View & hiCount) const
                {
                    Simd::BackgroundGrowRangeFast(value, loValue, hiValue);
                }
            };

            struct IncrementCountUpdater
            {
                void operator()(View & value, View & loValue, View & loCount, View & hiValue, View & hiCount) const
                {
                    Simd::BackgroundIncrementCount(value, loValue, hiValue, loCount, hiCount);
                }
            };

            struct AdjustRangeUpdater
            {
                void operator()(View & value, View & loValue, View & loCount, View & hiValue, View & hiCount) const
                {
                    Simd::BackgroundAdjustRange(loCount, loValue, hiCount, hiValue, 1);
                }
            };

            template <typename Updater> void Apply(Texture::Features & features, const Updater & updater)
            {
                for (size_t i = 0; i < features.size(); ++i)
                {
                    Texture::Feature & feature = *features[i];
                    for (size_t j = 0; j < feature.value.Size(); ++j)
                    {
                        updater(feature.value[j], feature.lo.value[j], feature.lo.count[j], feature.hi.value[j], feature.hi.count[j]);
                    }
                }
            }

            void UpdateBackground()
            {
                SIMD_CHECK_PERFORMANCE();

                Background & background = _scene.background;
                const Stability::State & stability = _scene.stability.state;
                const Time & time = _scene.input.timestamp;
                switch (background.state)
                {
                case Background::Update:
                    switch (stability)
                    {
                    case Stability::Stable:
                        background.statUpdateTime += time - background.lastFrameTime;
                        background.updateTime += time - background.lastFrameTime;
                        if (background.statUpdateTime > _options.BackgroundStatUpdateTime)
                        {
                            Apply(_scene.texture.features, IncrementCountUpdater());
                            background.statUpdateTime = 0;
                            ++background.updateCounter;
                            if (background.updateCounter >= CHAR_MAX || (background.updateTime > _options.BackgroundUpdateTime && background.updateCounter >= 8))
                            {
                                Apply(_scene.texture.features, AdjustRangeUpdater());
                                background.updateTime = 0;
                                background.updateCounter = 0;
                            }
                        }
                        break;
                    case Stability::Sabotage:
                        background.sabotageCounter++;
                        if (background.sabotageCounter > _options.BackgroundSabotageCountMax)
                            InitBackground();
                        break;
                    default:
                        assert(0);
                    }
                    if (stability != Stability::Sabotage)
                        background.sabotageCounter = 0;
                    break;
                case Background::Grow:
                    if (stability == Stability::Sabotage)
                        InitBackground();
                    else
                    {
                        Apply(_scene.texture.features, GrowRangeUpdater());
                        if (stability != Stability::Stable)
                            background.growEndTime = time + _options.BackgroundGrowTime;
                        if (background.growEndTime < time)
                        {
                            background.state = Background::Update;
                            background.updateCounter = 0;
                        }
                    }
                    break;
                case Background::Init:
                    InitBackground();
                    break;
                default:
                    assert(0);
                }
                background.lastFrameTime = time;
            }

            void InitBackground()
            {
                Background & background = _scene.background;
                Apply(_scene.texture.features, InitUpdater());
                background.growEndTime = _scene.input.timestamp + _options.BackgroundGrowTime;
                background.state = Background::Grow;
                background.updateCounter = 0;
                background.statUpdateTime = 0;
                background.updateTime = 0;
            }

            void SetMetadata()
            {
                _scene.metadata->objects.clear();
                AddToMetadata(_scene.tracking.objects);
                AddToMetadata(_scene.tracking.justDeletedObjects);
            }

            void AddToMetadata(const ObjectPtrs & objects)
            {
                size_t scale = _scene.model.scale;
                for (size_t i = 0; i < objects.size(); ++i)
                {
                    Object & srcObject = *objects[i];
                    if (srcObject.type == Object::Moving)
                    {
                        Motion::Object dstObject;
                        dstObject.id = srcObject.classificationId;
                        dstObject.rect = srcObject.rect*scale;
                        for (size_t j = 0, n = srcObject.trajectory.size(); j < n; ++j)
                        {
                            ptrdiff_t half = std::min<size_t>(_options.TrackingAveragingHalfRange, std::min(n - 1 - j, j));
                            ptrdiff_t beg = std::max<ptrdiff_t>(0, j - half);
                            ptrdiff_t end = std::min<ptrdiff_t>(n, j + half + 1);
                            Point sum;
                            for (ptrdiff_t l = beg; l < end; ++l)
                                sum += srcObject.trajectory[l]->point*scale;
                            Motion::Position position;
                            position.time = srcObject.trajectory[j]->time;
                            position.point = sum / (end - beg);
                            dstObject.trajectory.push_back(position);
                        }
                        _scene.metadata->objects.push_back(dstObject);
                    }
                }
            }

            void DebugAnnotation()
            {
                SIMD_CHECK_PERFORMANCE();

                Frame * output = _scene.output;
                size_t scale = _scene.model.scale;

                if (output && output->format == Frame::Bgr24)
                {
                    View & canvas = output->planes[0];

                    if (_options.DebugDrawBottomRight)
                    {
                        View src;
                        bool grad = _options.DifferenceDxFeatureWeight || _options.DifferenceDyFeatureWeight;
                        switch (_options.DebugDrawBottomRight)
                        {
                        case 1: src = _scene.difference[_options.DebugDrawLevel]; break;
                        case 2: src = _scene.texture.gray.value[_options.DebugDrawLevel]; break;
                        case 3: if(grad) src = _scene.texture.dx.value[_options.DebugDrawLevel]; break;
                        case 4: if(grad) src = _scene.texture.dy.value[_options.DebugDrawLevel]; break;
                        }
                        if(src.data)
                            Simd::GrayToBgr(src, canvas.Region(src.Size(), View::BottomRight).Ref());
                    }

                    if (_options.DebugAnnotateModel)
                    {
                        Simd::Pixel::Bgr24 color(0, 255, 255);
                        for (size_t i = 0; i < _scene.model.roi.size(); ++i)
                        {
                            Point p0 = i ? _scene.model.roi[i - 1] : _scene.model.roi.back(), p1 = _scene.model.roi[i];
                            Simd::DrawLine(canvas, p0*scale, p1*scale, color);
                        }
                        Rect objectMin(OnvifToScreenSize(_model.size, _scene.model.originalFrameSize));
                        objectMin.Shift(Point(_scene.model.originalFrameSize.x - objectMin.right - 2*scale, scale));
                        Simd::DrawRectangle(canvas, objectMin, color);
                    }

                    if (_options.DebugAnnotateMovingRegions)
                    {
                        Simd::Pixel::Bgr24 color(0, 255, 0);
                        for (size_t i = 0; i < _scene.segmentation.movingRegions.size(); ++i)
                        {
                            const MovingRegion & region = *_scene.segmentation.movingRegions[i];
                            Simd::DrawRectangle(canvas, region.rect*scale, color, 1);
                        }
                    }

                    if (_options.DebugAnnotateTrackingObjects)
                    {
                        Simd::Pixel::Bgr24 color(0, 255, 255);
                        for (size_t i = 0; i < _scene.tracking.objects.size(); ++i)
                        {
                            const Object & object = *_scene.tracking.objects[i];
                            Simd::DrawRectangle(canvas, object.rect*scale, color, 1);
                            _scene.font.Draw(canvas, ToString(object.trackingId), Point(object.rect.Center().x*scale, object.rect.top*scale - _scene.font.Height()), color);
                            const MovingRegionPtrs & regions = object.trajectory;
                            for (size_t j = 1; j < regions.size(); ++j)
                                Simd::DrawLine(canvas, regions[j]->point*scale, regions[j - 1]->point*scale, color, 1);
                        }
                    }
                }
            }
        };
    }
}

#endif//__SimdMotion_hpp__
