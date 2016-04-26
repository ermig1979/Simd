/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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

#include <vector>
#include <map>

namespace Simd
{
	/*! @ingroup cpp_detection

		\short The Detection structure provides object detection with using HAAR and LBP cascade classifiers.

		\ref cpp_detection_functions.
	*/
	template <class A>
	struct Detection
	{
		typedef A Allocator; /*!< Allocator type definition. */
        typedef Simd::Point<ptrdiff_t> Size;
        typedef std::vector<Size> Sizes;
        typedef Simd::Rectangle<ptrdiff_t> Rect;
        typedef std::vector<Rect> Rects;
        typedef Simd::View<Simd::Allocator> View;
        typedef int Tag;

        static const Tag UNDEFINED_OBJECT_TAG = -1;

        struct Object
        {
            Rect rect; /*!< \brief A width of the frame. */
            int weight;
            Tag tag;

            Object(const Rect & r = Rect(), int w = 0, Tag t = UNDEFINED_OBJECT_TAG)
                : rect(r)
                , weight(w)
                , tag(t)
            {

            }

            Object(const Object & o)
                : rect(o.rect)
                , weight(o.weight)
                , tag(o.tag)
            {
            }
        };
        typedef std::vector<Object> Objects;

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
                ::SimdDetectionFree(_data[i].handle);
        }

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

        bool Init(const Size & frameSize, double scaleFactor = 1.1, const Size & sizeMin = Size(0, 0),
            const Size & sizeMax = Size(INT_MAX, INT_MAX), const View & roi = View())
        {
            if (_data.empty())
                return false;
            _frameSize = frameSize;
            _scaleFactor = scaleFactor;
            _sizeMin = sizeMin;
            _sizeMax = sizeMax;
            _needNormalization = NeedNormalization();
            _roi.Recreate(frameSize, View::Gray8);
            if (roi.format == View::Gray8)
                Simd::Copy(roi, _roi);
            else
                Simd::Fill(_roi, 0xFF);
            return InitLevels();
        }

        bool Ready() const
        {
            return !(_data.empty() || _levels.empty());
        }

        bool Detect(const View & src, Objects & objects, int groupSizeMin = 3, double sizeDifferenceMax = 0.2,
            bool motionMask = false, const Rects & motionRegions = Rects())
        {
            if (!Ready() || src.Size() != _frameSize)
                return false;

            BuildPyramid(src);

            typedef std::map<Tag, Objects> Candidates;
            Candidates candidates;

            for (size_t i = 0; i < _levels.size(); ++i)
            {
                Level & level = _levels[i];
                View mask = level.roi;
                if (motionMask)
                {
                    FillMotionMask(motionRegions, level);
                    mask = level.mask;
                }
                for (size_t j = 0; j < level.hids.size(); ++j)
                {
                    Hid & hid = level.hids[j];

                    hid.Detect(mask, level.rect, level.dst);

                    AddObjects(candidates[hid.data->tag], level.dst, level.rect, hid.data->size, level.scale,
                        level.throughColumn ? 2 : 1, hid.data->tag);
                }
            }

            objects.clear();
            for (Candidates::iterator it = candidates.begin(); it != candidates.end(); ++it)
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
            bool Int16() const {return (flags&::SimdDetectionInfoCanInt16) != 0; }
        };

        typedef void(*DetectPtr)(const void * hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

        struct Hid
        {
            Handle handle;
            Data * data;
            DetectPtr detect;

            void Detect(const View & mask, const Rect & rect, View & dst)
            {
                Size s = dst.Size() - data->size;
                View m = mask.Region(s, View::MiddleCenter);
                Rect r = rect.Shifted(-data->size / 2).Intersection(Rect(s));
                Simd::Fill(dst, 0);
                ::SimdDetectionPrepare(handle);
                detect(handle, m.data, m.stride, r.left, r.top, r.right, r.bottom, dst.data, dst.stride);
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
                    ::SimdDetectionFree(hids[i].handle);
            }
        };
        typedef std::vector<Level> Levels;

        std::vector<Data> _data;

        Size _frameSize;
        double _scaleFactor;
        Size _sizeMin;
        Size _sizeMax;
        bool _needNormalization;

        View _roi;
        Levels _levels;

        bool NeedNormalization()
        {
            for (size_t i = 0; i < _data.size(); ++i)
            {
                if(_data[i].Haar())
                    return true;
            }
            return false;
        }

        bool InitLevels()
        {
            _levels.clear();
            _levels.reserve(100);
            double scale = 1.0;
            do
            {
                std::vector<bool> inserts(_data.size(), false);
                bool exit = true, insert = false;
                for (size_t i = 0; i < _data.size(); ++i)
                {
                    Size imageSize = _data[i].size * scale;
                    if (imageSize.x <= _sizeMax.x && imageSize.y <= _sizeMax.y &&
                        imageSize.x <= _frameSize.x && imageSize.y <= _frameSize.y)
                    {
                        if (imageSize.x >= _sizeMin.x && imageSize.y >= _sizeMin.y)
                            insert = inserts[i] = true;
                        exit = false;
                    }
                }
                if (exit)
                    break;

                if (insert)
                {
                    _levels.push_back(Level());
                    Level & level = _levels.back();

                    level.scale = scale;
                    level.throughColumn = scale <= 2.0;
                    Size scaledSize(_frameSize / scale);

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
                                if(_data[i].Int16())
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
                    }

                    Simd::ResizeBilinear(_roi, level.roi);
                    Simd::Binarization(level.roi, 0, 255, 0, level.roi, SimdCompareGreater);
                    level.rect = Rect(level.roi.Size());
                    Simd::SegmentationShrinkRegion(level.roi, 255, level.rect);
                }
                scale *= _scaleFactor;
            } while (true);
            return !_levels.empty();
        }

        void BuildPyramid(View src)
        {
            View gray;
            if (src.format != View::Gray8)
            {
                gray.Recreate(src.Size(), View::Gray8);
                Convert(src, gray);
                src = gray;
            }

            Simd::ResizeBilinear(src, _levels[0].src);
            if (_needNormalization)
                Simd::NormalizeHistogram(_levels[0].src, _levels[0].src);
            EstimateIntegral(_levels[0]);
            for (size_t i = 1; i < _levels.size(); ++i)
            {
                Simd::ResizeBilinear(_levels[0].src, _levels[i].src);
                EstimateIntegral(_levels[i]);
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

        void FillMotionMask(const Rects & rects, Level & level) const
        {
            Simd::Fill(level.mask, 0);
            for (size_t i = 0; i < rects.size(); i++)
                Simd::Fill(level.mask.Region(rects[i]/level.scale).Ref(), 0xFF);
            Simd::OperationBinary8u(level.mask, level.roi, level.mask, SimdOperationBinary8uAnd);
        }

        void AddObjects(Objects & objects, const View & dst, const Rect & rect, const Size & size, double scale, size_t step, Tag tag)
        {
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += step)
            {
                const uint8_t * mask = dst.data + row*dst.stride;
                for (ptrdiff_t col = rect.left; col < rect.right; col += step)
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

        template<typename T> int Partition(const std::vector<T> & _vec, std::vector<int> & labels, double sizeDifferenceMax)
        {
            Similar similar(sizeDifferenceMax);

            int i, j, N = (int)_vec.size();
            const T* vec = &_vec[0];

            const int PARENT = 0;
            const int RANK = 1;

            std::vector<int> _nodes(N * 2);
            int(*nodes)[2] = (int(*)[2])&_nodes[0];

            // The first O(N) pass: create N single-vertex trees
            for (i = 0; i < N; i++)
            {
                nodes[i][PARENT] = -1;
                nodes[i][RANK] = 0;
            }

            // The main O(N^2) pass: merge connected components
            for (i = 0; i < N; i++)
            {
                int root = i;

                // find root
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
                        // unite both trees
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

                        // compress the path from node2 to root
                        while ((parent = nodes[k][PARENT]) >= 0)
                        {
                            nodes[k][PARENT] = root;
                            k = parent;
                        }

                        // compress the path from node to root
                        k = i;
                        while ((parent = nodes[k][PARENT]) >= 0)
                        {
                            nodes[k][PARENT] = root;
                            k = parent;
                        }
                    }
                }
            }

            // Final O(N) pass: enumerate classes
            labels.resize(N);
            int nclasses = 0;

            for (i = 0; i < N; i++)
            {
                int root = i;
                while (nodes[root][PARENT] >= 0)
                    root = nodes[root][PARENT];
                // re-use the rank as the class label
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
                buffer[i].rect = buffer[i].rect / buffer[i].weight;

            for (size_t i = 0; i < buffer.size(); i++)
            {
                Rect r1 = buffer[i].rect;
                int n1 = buffer[i].weight;
                if (n1 < groupSizeMin)
                    continue;

                // filter out small object rectangles inside large rectangles	
                size_t j;
                for (j = 0; j < buffer.size(); j++)
                {
                    int n2 = buffer[j].weight;

                    if (j == i || n2 < groupSizeMin)
                        continue;

                    Rect r2 = buffer[j].rect;

                    int dx = int(r2.Width() * sizeDifferenceMax);
                    int dy = int(r2.Height() * sizeDifferenceMax);

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
