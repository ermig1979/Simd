/*
* Simd Library (http://ermig1979.github.io/Simd).
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

        \short Contains Framework for motion detection.

        \note This is wrapper around low-level \ref motion_detection API.

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
        typedef double Time; /*!< \brief Time type. */
        typedef int Id; /*!< \brief ID type. */
        typedef std::string String; /*!< \brief String type. */
        typedef Simd::Point<ptrdiff_t> Size; /*!< \brief screen 2D-size (width and height). */
        typedef Simd::Point<ptrdiff_t> Point; /*!< \brief screen point (x and y). */
        typedef std::vector<Point> Points; /*!< \brief Vector of screen 2D-points. */
        typedef Simd::Rectangle<ptrdiff_t> Rect; /*!< \brief Screen rectangle. */
        typedef Simd::Point<double> FSize; /*!< \brief ONVIF 2D-size (width and height). ONVIF size is restricted by range [0, 2]. */
        typedef Simd::Point<double> FPoint; /*!< \brief ONVIF 2D-point (x and y). ONVIF coordinates are restricted by range [-1, 1]. */
        typedef std::vector<FPoint> FPoints; /*!< \brief Vector of ONVIF 2D-points. */
        typedef Simd::View<Simd::Allocator> View; /*!< \brief Image type. */
        typedef Simd::Frame<Simd::Allocator> Frame; /*!< \brief Frame type. */

        /*! @ingroup cpp_motion

            \short Converts screen X-coordinate to ONVIF X-coordinate.

            \param [in] x - a screen X-coordinate.
            \param [in] screenWidth - a screen width.
            \return ONVIF X-coordinate.
        */
        SIMD_INLINE double ScreenToOnvifX(ptrdiff_t x, ptrdiff_t screenWidth)
        {
            return double(2 * x - screenWidth) / screenWidth;
        }

        /*! @ingroup cpp_motion

            \short Converts screen Y-coordinate to ONVIF Y-coordinate.

            \param [in] y - a screen Y-coordinate.
            \param [in] screenHeight - a screen height.
            \return ONVIF Y-coordinate.
        */
        SIMD_INLINE double ScreenToOnvifY(ptrdiff_t y, ptrdiff_t screenHeight)
        {
            return double(screenHeight - 2 * y) / screenHeight;
        }

        /*! @ingroup cpp_motion

            \short Converts screen 2D-coordinates to ONVIF 2D-coordinates.

            \param [in] point - a screen 2D-coordinates.
            \param [in] screenSize - a screen size (width and height).
            \return ONVIF 2D-coordinate.
        */
        SIMD_INLINE FPoint ScreenToOnvif(const Point & point, const Point & screenSize)
        {
            return FPoint(ScreenToOnvifX(point.x, screenSize.x), ScreenToOnvifY(point.y, screenSize.y));
        }

        /*! @ingroup cpp_motion

            \short Converts screen 2D-size to ONVIF 2D-size.

            \param [in] size - a screen 2D-size.
            \param [in] screenSize - a screen size (width and height).
            \return ONVIF 2D-size.
        */
        SIMD_INLINE FSize ScreenToOnvifSize(const Size & size, const Point & screenSize)
        {
            return FSize(double(size.x * 2 / screenSize.x), double(size.y * 2 / screenSize.y));
        }

        /*! @ingroup cpp_motion

            \short Converts ONVIF X-coordinate to screen X-coordinate.

            \param [in] x - a ONVIF X-coordinate. ONVIF coordinates are restricted by range [-1, 1].
            \param [in] screenWidth - a screen width.
            \return screen X-coordinate.
        */
        SIMD_INLINE ptrdiff_t OnvifToScreenX(double x, ptrdiff_t screenWidth)
        {
            return std::max(ptrdiff_t(0), std::min(screenWidth - 1, (ptrdiff_t)Simd::Round((1.0 + x)*screenWidth / 2.0)));
        }

        /*! @ingroup cpp_motion

            \short Converts ONVIF Y-coordinate to screen Y-coordinate.

            \param [in] y - a ONVIF Y-coordinate. ONVIF coordinates are restricted by range [-1, 1].
            \param [in] screenHeight - a screen height.
            \return screen Y-coordinate.
        */
        SIMD_INLINE ptrdiff_t OnvifToScreenY(double y, ptrdiff_t screenHeight)
        {
            return std::max(ptrdiff_t(0), std::min(screenHeight - 1, (ptrdiff_t)Simd::Round((1.0 - y)*screenHeight / 2.0)));
        }

        /*! @ingroup cpp_motion

            \short Converts ONVIF 2D-coordinates to screen 2D-coordinates.

            \param [in] point - a ONVIF 2D-coordinates. ONVIF coordinates are restricted by range [-1, 1].
            \param [in] screenSize - a screen size (width and height).
            \return screen 2D-coordinate.
        */
        SIMD_INLINE Point OnvifToScreen(const FPoint & point, const Point & screenSize)
        {
            return Point(OnvifToScreenX(point.x, screenSize.x), OnvifToScreenY(point.y, screenSize.y));
        }

        /*! @ingroup cpp_motion

            \short Converts ONVIF 2D-size to screen 2D-size.

            \param [in] size - a ONVIF 2D-size. ONVIF size is restricted by range [0, 2].
            \param [in] screenSize - a screen size (width and height).
            \return screen 2D-size.
        */
        SIMD_INLINE Size OnvifToScreenSize(const FSize & size, const Point & screenSize)
        {
            return Size(Round(size.x*screenSize.x / 2.0), Round(size.y*screenSize.y / 2.0));
        }

        /*! @ingroup cpp_motion

            \short Converts ID to string.

            \param [in] id - an ID.
            \return string representation of ID.
        */
        SIMD_INLINE String ToString(Id id)
        {
            std::stringstream ss;
            ss << id;
            return ss.str();
        }

        /*! @ingroup cpp_motion

            \short Position structure.

            Describes position (2D-point and time) of detected object.
        */
        struct Position
        {
            Point point; /*!< \brief Screen 2D-point. */
            Time time; /*!< \brief A timestamp. */
        };
        typedef std::vector<Position> Positions; /*!< \brief Vector of object positions. */

        /*! @ingroup cpp_motion

            \short Object structure.

            Describes object detected at screen by Simd::Motion::Detector.
        */
        struct Object
        {
            Id id; /*!< \brief An object ID. */
            Rect rect; /*!< \brief A bounding box around the object. */
            Positions trajectory; /*!< \brief A trajectory of the object. */
        };
        typedef std::vector<Object> Objects; /*!< \brief Vector of objects. */

        /*! @ingroup cpp_motion

            \short Event structure.

            Describes event generated by Simd::Motion::Detector.
        */
        struct Event
        {
            /*!
                \enum Type

                Describes types of event.
            */
            enum Type
            {
                ObjectIn, /*!< \brief An appearing of new object. */
                ObjectOut, /*!< \brief A disappearing of object */
                SabotageOn, /*!< \brief An appearing of too big motion on the screen. */
                SabotageOff, /*!< \brief A disappearing of too big motion on the screen. */
            } type; /*!< \brief A type of event. */

            String text; /*!< \brief Event text description. */
            Id objectId; /*!< \brief ID of object concerned with this event or -1. */

            /*!
                Constructs Event structure.

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
        typedef std::vector<Event> Events; /*!< \brief Vector of events. */

        /*! @ingroup cpp_motion

            \short Metadata structure.

            Contains lists of detected objects and events generated by Simd::Motion::Detector at current frame.
        */
        struct Metadata
        {
            Objects objects; /*!< \brief A list of objects detected by Simd::Motion::Detector at current frame. */
            Events events; /*!< \brief A list of events generated by Simd::Motion::Detector at current frame. */
        };

        /*! @ingroup cpp_motion

            \short Model structure.

            Describes screen scene. It is used by Simd::Motion::Detector for algorithm calibration.
        */
        struct Model
        {
            static const uint8_t EMPTY = 0;
            static const uint8_t ROI = 255;

            FSize size; /*!< \brief A minimal size of object to detect. ONVIF size is restricted by range [0, 2]. */ 
            FPoints roi; /*!< \brief A ROI (region of interest). ONVIF coordinates is restricted by range [-1, 1]. */ 
            View mask; /*!< \brief A ROI (region of interest) mask. It must be 8-bit gray image. */

            /*!
                Copy constructor of Model.

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
                Constructs Model structure on the base of detected object size and ROI polygon.

                \param [in] size_ - a minimal size of detected object. It is default value is (0.1, 0.1) ~ 0.25% of screen area. 
                \param [in] roi_ - a ROI (region of interest). It is empty by default (all screen).
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
                Constructs Model structure on the base of detected object size and ROI mask.

                \param [in] size_ - a minimal size of detected object.
                \param [in] mask_ - a ROI (region of interest) mask. It must be 8-bit gray image.
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
                Copy operator.

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

            \short Options structure.

            Describes options used by Simd::Motion::Detector.
        */
        struct Options
        {
            int CalibrationScaleLevelMax;  /*!< \brief A maximum scale of input frame. By default it is equal to 3 (maximum scale in 8 times). */ 

            int DifferenceGrayFeatureWeight; /*!< \brief A weight of gray feature for difference estimation. By default it is equal to 18. */ 
            int DifferenceDxFeatureWeight; /*!< \brief A weight of X-gradient feature for difference estimation. By default it is equal to 18. */ 
            int DifferenceDyFeatureWeight; /*!< \brief A weight of Y-gradient feature for difference estimation. By default it is equal to 18. */ 
            bool DifferencePropagateForward; /*!< \brief An additional boosting of estimated difference. By default it is true. */ 
            bool DifferenceRoiMaskEnable; /*!< \brief A flag to restrict difference estimation by ROI. By default it is true. */ 

            double BackgroundGrowTime; /*!< \brief Initial time (in seconds) of updated background in fast mode. By default it is equal to 1 second. */ 
            double BackgroundStatUpdateTime; /*!< \brief Collect background statistics update interval (in seconds) in normal mode. By default it is equal to 0.04 second. */
            double BackgroundUpdateTime; /*!< \brief Background update speed (in seconds) in normal mode. By default it is equal to 1 second. */
            int BackgroundSabotageCountMax; /*!< \brief Maximal count of frame with sabotage without scene reinitialization. By default it is equal to 3. */

            double SegmentationCreateThreshold; /*!< \brief Threshold of segmentation to create motion region. It is restricted by range [0, 1]. By default it is equal to 0.5. */
            double SegmentationExpandCoefficient; /*!< \brief Segmentation coefficient of area expansion of motion region. It is restricted by range [0, 1]. By default it is equal to 0.75. */

            double StabilityRegionAreaMax; /*!< \brief Defines maximal total area of motion regions othervise sabotage event is generated. It is restricted by range [0, 1]. By default it is equal to 0.5. */

            int TrackingTrajectoryMax; /*!< \brief Maximal length of object trajectory. By default it is equal to 1024. */
            double TrackingRemoveTime; /*!< \brief A time (in seconds) to remove absent object. By default it is equal to 1 second. */
            double TrackingAdditionalLinking; /*!< \brief A coefficient to boost trajectory linking. By default it is equal to 0. */
            int TrackingAveragingHalfRange; /*!< \brief A half range parameter used to average object trajectory. By default it is equal to 12. */

            double ClassificationShiftMin; /*!< \brief A minimal shift (in screen diagonals) of motion region to detect object. By default it is equal to 0.075. */
            double ClassificationTimeMin; /*!< \brief A minimal life time (in seconds) of motion region to detect object. By default it is equal to 1 second. */

            int DebugDrawLevel; /*!< \brief A pyramid level used for debug annotation. By default it is equal to 1. */
            int DebugDrawBottomRight; /*!< \brief A type of debug annotation in right bottom corner (0 - empty; 1 = difference; 2 - texture.gray.value; 3 - texture.dx.value; 4 - texture.dy.value). By default it is equal to 0. */
            bool DebugAnnotateModel; /*!< \brief Debug annotation of model. By default it is equal to false. */
            bool DebugAnnotateMovingRegions; /*!< \brief Debug annotation of moving region. By default it is equal to false. */
            bool DebugAnnotateTrackingObjects; /*!< \brief Debug annotation of tracked objects. By default it is equal to false. */

            /*!
                Default constructor of Options.
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

            \short Class Detector.

            Performs motion detection.
        */
        class Detector
        {
        public:

            /*!
                Default constructor of Detector.
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
                Sets options of motion detector.

                \param [in] options - options of motion detector.
                \return a result of the operation.
            */
            bool SetOptions(const Simd::Motion::Options & options)
            {
                *(Simd::Motion::Options*)(&_options) = options;
                return true;
            }

            /*!
                Sets model of scene of motion detector.

                \param [in] model - a model of scene.
                \return a result of the operation.
            */
            bool SetModel(const Model & model)
            {
                _model = model;
                return true;
            }

            /*!
                Processes next frame. You have to successively process all frame of a movie with using of this function.

                \param [in] input - a current input frame.
                \param [out] metadata - a metadata (sets of detected objects and generated events). It is a result of processing of current frame.
                \param [out] output - a pointer to output frame with debug annotation. Can be NULL.
                \return a result of the operation.
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
                    Simd::ResizeBilinear(_model.mask, model.roiMask[0]);
                else
                {
                    Simd::Fill(model.roiMask, Motion::Model::EMPTY);
                    DrawFilledPolygon(model.roiMask[0], model.roi, Motion::Model::ROI);
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
