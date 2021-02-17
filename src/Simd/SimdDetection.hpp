/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar,
*               2019-2019 Facundo Galan.
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
#ifndef __SimdDetection_hpp__
#define __SimdDetection_hpp__

#include "Simd/SimdLib.hpp"
#include "Simd/SimdParallel.hpp"

#include <vector>
#include <map>
#include <memory>

#include <limits.h>

#ifndef SIMD_CHECK_PERFORMANCE
#define SIMD_CHECK_PERFORMANCE()
#endif

namespace Simd
{
    /*! @ingroup cpp_detection

        \short The Detection structure provides object detection with using of HAAR and LBP cascade classifiers.

        Using example (face detection in the image):
        \code
        #include "Simd/SimdDetection.hpp"
        #include "Simd/SimdDrawing.hpp"

        int main()
        {
            typedef Simd::Detection<Simd::Allocator> Detection;

            Detection::View image;
            image.Load("../../data/image/face/lena.pgm");

            Detection detection;

            detection.Load("../../data/cascade/haar_face_0.xml");

            detection.Init(image.Size());

            Detection::Objects objects;
            detection.Detect(image, objects);

            for (size_t i = 0; i < objects.size(); ++i)
                Simd::DrawRectangle(image, objects[i].rect, uint8_t(255));

            image.Save("result.pgm");

            return 0;
        }
        \endcode

        Using example (face detection in the video captured by OpenCV):
        \code
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
        \endcode

        \note This is wrapper around low-level \ref object_detection API.
    */
    template <template<class> class A>
    struct Detection
    {
        typedef A<uint8_t> Allocator; /*!< Allocator type definition. */
        typedef Simd::View<A> View; /*!< An image type definition. */
        typedef Simd::Point<ptrdiff_t> Size; /*!< An image size type definition. */
        typedef std::vector<Size> Sizes; /*!< A vector of image sizes type definition. */
        typedef Simd::Rectangle<ptrdiff_t> Rect; /*!< A rectangle type definition. */
        typedef std::vector<Rect> Rects; /*!< A vector of rectangles type definition. */
        typedef int Tag; /*!< A tag type definition. */

        static const Tag UNDEFINED_OBJECT_TAG = -1; /*!< The undefined object tag. */

        /*!
            \short The Object structure describes detected object.

        */
        struct Object
        {
            Rect rect; /*!< \brief A bounding box around of detected object. */
            int weight; /*!< \brief An object weight (number of elementary detections). */
            Tag tag; /*!< \brief An object tag. It's useful if more than one detector works. */

            /*!
                Creates a new Object structure.

                \param [in] r - initial bounding box.
                \param [in] w - initial weight.
                \param [in] t - initial tag.
            */
            Object(const Rect & r = Rect(), int w = 0, Tag t = UNDEFINED_OBJECT_TAG)
                : rect(r)
                , weight(w)
                , tag(t)
            {

            }

            /*!
                Creates a new Object structure on the base of another object.

                \param [in] o - another object.
            */
            Object(const Object & o)
                : rect(o.rect)
                , weight(o.weight)
                , tag(o.tag)
            {
            }
        };
        typedef std::vector<Object> Objects; /*!< A vector of objects type defenition. */

        /*!
            Creates a new empty Detection structure.
        */
        Detection()
        {
        }

        /*!
            A Detection destructor.
        */
        ~Detection()
        {
            for (size_t i = 0; i < _data.size(); ++i)
                ::SimdRelease(_data[i].handle);
        }

        /*!
            Loads from file classifier cascade. Supports OpenCV HAAR and LBP cascades type.
            You can call this function more than once if you want to use several object detectors at the same time.

            \note Tree based cascades and old cascade formats are not supported!

            \param [in] xml - a string containing XML with cascade.
            \param [in] tag - an user defined tag. This tag will be inserted in output Object structure.
            \return a result of this operation.
        */
        bool LoadStringXml(const std::string & xml, Tag tag = UNDEFINED_OBJECT_TAG)
        {
            // Copy the received string to a non const char pointer.
            char * xmlTmp = new char[xml.size() + 1];
            std::copy(xml.begin(), xml.end(), xmlTmp);
            xmlTmp[xml.size()] = '\0';

            Handle handle = ::SimdDetectionLoadStringXml(xmlTmp);
            if (handle)
            {
                Data data;
                data.handle = handle;
                data.tag = tag;
                ::SimdDetectionInfo(handle, (size_t*)&data.size.x, (size_t*)&data.size.y, &data.flags);
                _data.push_back(data);
            }
            return handle != NULL;
        }

        /*!
            Loads from file classifier cascade. Supports OpenCV HAAR and LBP cascades type.
            You can call this function more than once if you want to use several object detectors at the same time.

            \note Tree based cascades and old cascade formats are not supported!

            \param [in] path - a path to cascade.
            \param [in] tag - an user defined tag. This tag will be inserted in output Object structure.
            \return a result of this operation.
        */
        bool Load(const std::string & path, Tag tag = UNDEFINED_OBJECT_TAG)
        {
            Handle handle = ::SimdDetectionLoadA(path.c_str());
            if (handle)
            {
                Data data;
                data.handle = handle;
                data.tag = tag;
                ::SimdDetectionInfo(handle, (size_t*)&data.size.x, (size_t*)&data.size.y, &data.flags);
                _data.push_back(data);
            }
            return handle != NULL;
        }

        /*!
            Prepares Detection structure to work with image of given size.

            \param [in] imageSize - a size of input image.
            \param [in] scaleFactor - a scale factor. To detect objects of different sizes the algorithm uses many scaled image.
                                      This parameter defines size difference between neighboring images. This parameter strongly affects to performance.
            \param [in] sizeMin - a minimal size of detected objects. This parameter strongly affects to performance.
            \param [in] sizeMax - a maximal size of detected objects.
            \param [in] roi - a 8-bit image mask which defines Region Of Interest. User can restricts detection region with using this mask.
                              The mask affects to the center of detected object.
            \param [in] threadNumber - a number of work threads. It useful for multi core CPU. Use value -1 to auto choose of thread number.
            \return a result of this operation.
        */
        bool Init(const Size & imageSize, double scaleFactor = 1.1, const Size & sizeMin = Size(0, 0),
            const Size & sizeMax = Size(INT_MAX, INT_MAX), const View & roi = View(), ptrdiff_t threadNumber = -1)
        {
            if (_data.empty())
                return false;
            _imageSize = imageSize;
            ptrdiff_t threadNumberMax = std::thread::hardware_concurrency();
            _threadNumber = (threadNumber <= 0 || threadNumber > threadNumberMax) ? threadNumberMax : threadNumber;
            return InitLevels(scaleFactor, sizeMin, sizeMax, roi);
        }

        /*!
            Detects objects at given image.

            \param [in] src - a input image.
            \param [out] objects - detected objects.
            \param [in] groupSizeMin - a minimal weight (number of elementary detections) of detected image.
            \param [in] sizeDifferenceMax - a parameter to group elementary detections.
            \param [in] motionMask - an using of motion detection flag. Useful for dynamical restriction of detection region to addition to ROI.
            \param [in] motionRegions - a set of rectangles (motion regions) to restrict detection region to addition to ROI.
                                        The regions affect to the center of detected object.
            \return a result of this operation.
        */
        bool Detect(const View & src, Objects & objects, int groupSizeMin = 3, double sizeDifferenceMax = 0.2,
            bool motionMask = false, const Rects & motionRegions = Rects())
        {
            SIMD_CHECK_PERFORMANCE();

            if (_levels.empty() || src.Size() != _imageSize)
                return false;

            FillLevels(src);

            typedef std::map<Tag, Objects> Candidates;
            Candidates candidates;

            for (size_t i = 0; i < _levels.size(); ++i)
            {
                Level & level = *_levels[i];
                View mask = level.roi;
                Rect rect = level.rect;
                if (motionMask)
                {
                    FillMotionMask(motionRegions, level, rect);
                    mask = level.mask;
                }
                if (rect.Empty())
                    continue;
                for (size_t j = 0; j < level.hids.size(); ++j)
                {
                    Hid & hid = level.hids[j];

                    hid.Detect(mask, rect, level.dst, _threadNumber, level.throughColumn);

                    AddObjects(candidates[hid.data->tag], level.dst, rect, hid.data->size, level.scale,
                        level.throughColumn ? 2 : 1, hid.data->tag);
                }
            }

            objects.clear();
            for (typename Candidates::iterator it = candidates.begin(); it != candidates.end(); ++it)
                GroupObjects(objects, it->second, groupSizeMin, sizeDifferenceMax);

            return true;
        }

    private:

        typedef void * Handle;

        struct Data
        {
            Handle handle;
            Tag tag;
            Size size;
            ::SimdDetectionInfoFlags flags;

            bool Haar() const { return (flags&::SimdDetectionInfoFeatureMask) == ::SimdDetectionInfoFeatureHaar; }
            bool Tilted() const { return (flags&::SimdDetectionInfoHasTilted) != 0; }
            bool Int16() const { return (flags&::SimdDetectionInfoCanInt16) != 0; }
        };

        typedef void(*DetectPtr)(const void * hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

        struct Worker;
        typedef std::shared_ptr<Worker> WorkerPtr;
        typedef std::vector<WorkerPtr> WorkerPtrs;

        struct Hid
        {
            Handle handle;
            Data * data;
            DetectPtr detect;

            void Detect(const View & mask, const Rect & rect, View & dst, size_t threadNumber, bool throughColumn)
            {
                SIMD_CHECK_PERFORMANCE();

                Size s = dst.Size() - data->size;
                View m = mask.Region(s, View::MiddleCenter);
                Rect r = rect.Shifted(-data->size / 2).Intersection(Rect(s));
                Simd::Fill(dst, 0);
                ::SimdDetectionPrepare(handle);

                Parallel(r.top, r.bottom, [&](size_t thread, size_t begin, size_t end)
                {
                    detect(handle, m.data, m.stride, r.left, begin, r.right, end, dst.data, dst.stride);
                }, rect.Area() >= (data->Haar() ? 10000 : 30000) ? threadNumber : 1, throughColumn ? 2 : 1);
            }
        };
        typedef std::vector<Hid> Hids;

        struct Level
        {
            Hids hids;
            double scale;

            View src;
            View roi;
            View mask;

            Rect rect;

            View sum;
            View sqsum;
            View tilted;

            View dst;

            bool throughColumn;
            bool needSqsum;
            bool needTilted;

            ~Level()
            {
                for (size_t i = 0; i < hids.size(); ++i)
                    ::SimdRelease(hids[i].handle);
            }
        };
        typedef std::shared_ptr<Level> LevelPtr;
        typedef std::vector<LevelPtr> LevelPtrs;

        std::vector<Data> _data;
        Size _imageSize;
        bool _needNormalization;
        ptrdiff_t _threadNumber;
        LevelPtrs _levels;

        bool InitLevels(double scaleFactor, const Size & sizeMin, const Size & sizeMax, const View & roi)
        {
            _needNormalization = false;
            _levels.clear();
            _levels.reserve(100);
            double scale = 1.0;
            do
            {
                std::vector<bool> inserts(_data.size(), false);
                bool exit = true, insert = false;
                for (size_t i = 0; i < _data.size(); ++i)
                {
                    Size windowSize = _data[i].size * scale;
                    if (windowSize.x <= sizeMax.x && windowSize.y <= sizeMax.y &&
                        windowSize.x <= _imageSize.x && windowSize.y <= _imageSize.y)
                    {
                        if (windowSize.x >= sizeMin.x && windowSize.y >= sizeMin.y)
                            insert = inserts[i] = true;
                        exit = false;
                    }
                }
                if (exit)
                    break;

                if (insert)
                {
                    _levels.push_back(LevelPtr(new Level()));
                    Level & level = *_levels.back();

                    level.scale = scale;
                    level.throughColumn = scale <= 2.0;
                    Size scaledSize(_imageSize / scale);

                    level.src.Recreate(scaledSize, View::Gray8);
                    level.roi.Recreate(scaledSize, View::Gray8);
                    level.mask.Recreate(scaledSize, View::Gray8);

                    level.sum.Recreate(scaledSize + Size(1, 1), View::Int32);
                    level.sqsum.Recreate(scaledSize + Size(1, 1), View::Int32);
                    level.tilted.Recreate(scaledSize + Size(1, 1), View::Int32);

                    level.dst.Recreate(scaledSize, View::Gray8);

                    level.needSqsum = false, level.needTilted = false;
                    for (size_t i = 0; i < _data.size(); ++i)
                    {
                        if (!inserts[i])
                            continue;
                        Handle handle = ::SimdDetectionInit(_data[i].handle, level.sum.data, level.sum.stride, level.sum.width, level.sum.height,
                            level.sqsum.data, level.sqsum.stride, level.tilted.data, level.tilted.stride, level.throughColumn, _data[i].Int16());
                        if (handle)
                        {
                            Hid hid;
                            hid.handle = handle;
                            hid.data = &_data[i];
                            if (_data[i].Haar())
                                hid.detect = level.throughColumn ? ::SimdDetectionHaarDetect32fi : ::SimdDetectionHaarDetect32fp;
                            else
                            {
                                if (_data[i].Int16())
                                    hid.detect = level.throughColumn ? ::SimdDetectionLbpDetect16ii : ::SimdDetectionLbpDetect16ip;
                                else
                                    hid.detect = level.throughColumn ? ::SimdDetectionLbpDetect32fi : ::SimdDetectionLbpDetect32fp;
                            }
                            level.hids.push_back(hid);
                        }
                        else
                            return false;
                        level.needSqsum = level.needSqsum | _data[i].Haar();
                        level.needTilted = level.needTilted | _data[i].Tilted();
                        _needNormalization = _needNormalization | _data[i].Haar();
                    }

                    level.rect = Rect(level.roi.Size());
                    if (roi.format == View::None)
                        Simd::Fill(level.roi, 255);
                    else
                    {
                        Simd::ResizeBilinear(roi, level.roi);
                        Simd::Binarization(level.roi, 0, 255, 0, level.roi, SimdCompareGreater);
                        Simd::SegmentationShrinkRegion(level.roi, 255, level.rect);
                    }
                }
                scale *= scaleFactor;
            } while (true);
            return !_levels.empty();
        }

        void FillLevels(View src)
        {
            View gray;
            if (src.format != View::Gray8)
            {
                gray.Recreate(src.Size(), View::Gray8);
                Convert(src, gray);
                src = gray;
            }

            Simd::ResizeBilinear(src, _levels[0]->src);
            if (_needNormalization)
                Simd::NormalizeHistogram(_levels[0]->src, _levels[0]->src);
            EstimateIntegral(*_levels[0]);
            for (size_t i = 1; i < _levels.size(); ++i)
            {
                Simd::ResizeBilinear(_levels[0]->src, _levels[i]->src);
                EstimateIntegral(*_levels[i]);
            }
        }

        void EstimateIntegral(Level & level)
        {
            if (level.needSqsum)
            {
                if (level.needTilted)
                    Simd::Integral(level.src, level.sum, level.sqsum, level.tilted);
                else
                    Simd::Integral(level.src, level.sum, level.sqsum);
            }
            else
                Simd::Integral(level.src, level.sum);
        }

        void FillMotionMask(const Rects & rects, Level & level, Rect & rect) const
        {
            Simd::Fill(level.mask, 0);
            rect = Rect();
            for (size_t i = 0; i < rects.size(); i++)
            {
                Rect r = rects[i] / level.scale;
                rect |= r;
                Simd::Fill(level.mask.Region(r).Ref(), 0xFF);
            }
            rect &= level.rect;
            Simd::OperationBinary8u(level.mask, level.roi, level.mask, SimdOperationBinary8uAnd);
        }

        void AddObjects(Objects & objects, const View & dst, const Rect & rect, const Size & size, double scale, size_t step, Tag tag)
        {
            Size s = dst.Size() - size;
            Rect r = rect.Shifted(-size / 2).Intersection(Rect(s));
            for (ptrdiff_t row = r.top; row < r.bottom; row += step)
            {
                const uint8_t * mask = dst.data + row*dst.stride;
                for (ptrdiff_t col = r.left; col < r.right; col += step)
                {
                    if (mask[col] != 0)
                        objects.push_back(Object(Rect(col, row, col + size.x, row + size.y)*scale, 1, tag));
                }
            }
        }

        struct Similar
        {
            Similar(double sizeDifferenceMax)
                : _sizeDifferenceMax(sizeDifferenceMax)
            {}

            SIMD_INLINE bool operator() (const Object & o1, const Object & o2) const
            {
                const Rect & r1 = o1.rect;
                const Rect & r2 = o2.rect;
                double delta = _sizeDifferenceMax*(std::min(r1.Width(), r2.Width()) + std::min(r1.Height(), r2.Height()))*0.5;
                return
                    std::abs(r1.left - r2.left) <= delta && std::abs(r1.top - r2.top) <= delta &&
                    std::abs(r1.right - r2.right) <= delta && std::abs(r1.bottom - r2.bottom) <= delta;
            }

        private:
            double _sizeDifferenceMax;
        };

        template<typename T> int Partition(const std::vector<T> & vec, std::vector<int> & labels, double sizeDifferenceMax)
        {
            Similar similar(sizeDifferenceMax);
            int i, j, N = (int)vec.size();
            const int PARENT = 0;
            const int RANK = 1;

            std::vector<int> _nodes(N * 2);
            int(*nodes)[2] = (int(*)[2])&_nodes[0];

            for (i = 0; i < N; i++)
            {
                nodes[i][PARENT] = -1;
                nodes[i][RANK] = 0;
            }

            for (i = 0; i < N; i++)
            {
                int root = i;
                while (nodes[root][PARENT] >= 0)
                    root = nodes[root][PARENT];

                for (j = 0; j < N; j++)
                {
                    if (i == j || !similar(vec[i], vec[j]))
                        continue;
                    int root2 = j;

                    while (nodes[root2][PARENT] >= 0)
                        root2 = nodes[root2][PARENT];

                    if (root2 != root)
                    {
                        int rank = nodes[root][RANK], rank2 = nodes[root2][RANK];
                        if (rank > rank2)
                            nodes[root2][PARENT] = root;
                        else
                        {
                            nodes[root][PARENT] = root2;
                            nodes[root2][RANK] += rank == rank2;
                            root = root2;
                        }
                        assert(nodes[root][PARENT] < 0);

                        int k = j, parent;
                        while ((parent = nodes[k][PARENT]) >= 0)
                        {
                            nodes[k][PARENT] = root;
                            k = parent;
                        }

                        k = i;
                        while ((parent = nodes[k][PARENT]) >= 0)
                        {
                            nodes[k][PARENT] = root;
                            k = parent;
                        }
                    }
                }
            }

            labels.resize(N);
            int nclasses = 0;

            for (i = 0; i < N; i++)
            {
                int root = i;
                while (nodes[root][PARENT] >= 0)
                    root = nodes[root][PARENT];
                if (nodes[root][RANK] >= 0)
                    nodes[root][RANK] = ~nclasses++;
                labels[i] = ~nodes[root][RANK];
            }

            return nclasses;
        }

        void GroupObjects(Objects & dst, const Objects & src, size_t groupSizeMin, double sizeDifferenceMax)
        {
            if (groupSizeMin == 0 || src.size() < groupSizeMin)
                return;

            std::vector<int> labels;
            int nclasses = Partition(src, labels, sizeDifferenceMax);

            Objects buffer;
            buffer.resize(nclasses);
            for (size_t i = 0; i < labels.size(); ++i)
            {
                int cls = labels[i];
                buffer[cls].rect += src[i].rect;
                buffer[cls].weight++;
                buffer[cls].tag = src[i].tag;
            }

            for (size_t i = 0; i < buffer.size(); i++)
                buffer[i].rect = buffer[i].rect / double(buffer[i].weight);

            for (size_t i = 0; i < buffer.size(); i++)
            {
                Rect r1 = buffer[i].rect;
                int n1 = buffer[i].weight;
                if (n1 < (int)groupSizeMin)
                    continue;

                size_t j;
                for (j = 0; j < buffer.size(); j++)
                {
                    int n2 = buffer[j].weight;

                    if (j == i || n2 < (int)groupSizeMin)
                        continue;

                    Rect r2 = buffer[j].rect;

                    int dx = Simd::Round(r2.Width() * sizeDifferenceMax);
                    int dy = Simd::Round(r2.Height() * sizeDifferenceMax);

                    if (i != j && (n2 > std::max(3, n1) || n1 < 3) &&
                        r1.left >= r2.left - dx && r1.top >= r2.top - dy &&
                        r1.right <= r2.right + dx && r1.bottom <= r2.bottom + dy)
                        break;
                }

                if (j == buffer.size())
                    dst.push_back(buffer[i]);
            }
        }
    };
}

#endif//__SimdDetection_hpp__
