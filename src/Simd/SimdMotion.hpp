/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __SimdMotion_hpp__
#define __SimdMotion_hpp__

#include "Simd/SimdPoint.hpp"
#include "Simd/SimdRectangle.hpp"
#include "Simd/SimdFrame.hpp"
#include "Simd/SimdDrawing.hpp"

#include <vector>
#include <stack>

#ifndef SIMD_CHECK_PERFORMANCE
#define SIMD_CHECK_PERFORMANCE()
#endif

namespace Simd
{
    namespace Motion
    {
        typedef double Time;
        typedef Simd::Point<ptrdiff_t> Size;
        typedef Simd::Point<ptrdiff_t> Point;
        typedef std::vector<Point> Points;
        typedef Simd::Rectangle<ptrdiff_t> Rect;
        typedef std::vector<Rect> Rects;
        typedef Simd::View<Simd::Allocator> View; 
        typedef Simd::Frame<Simd::Allocator> Frame;
        typedef Simd::Pyramid<Simd::Allocator> Pyramid;

        struct Position
        {
            Point point;
            Time time;
            Rect rect;
        };
        typedef std::vector<Position> Positions;

        struct Object
        {
            Positions trajectory;
            Position current;
        };
        typedef std::vector<Object> Objects;

        struct Metadata
        {
            Objects objects;
        };

        struct Model
        {
            Size size;
            Points roi;

            Model(const Model & m)
                : size(m.size)
                , roi(m.roi)
            {
            }

            Model(const Size & s = Size(), const Points & r = Points())
                : size(s)
                , roi(r)
            {
                if (roi.size() < 3)
                {
                    roi.clear();
                    roi.push_back(Point(0, 0));
                    roi.push_back(Point(size.x, 0));
                    roi.push_back(Point(size.x, size.y));
                    roi.push_back(Point(0, size.y));
                }
            }
        };

        struct Options
        {
            int TextureGradientSaturation;
            int TextureGradientBoost;

            int DifferenceGrayFeatureWeight;
            int DifferenceDxFeatureWeight;
            int DifferenceDyFeatureWeight;
            bool DifferencePropagateForward;
            bool DifferenceRoiMaskEnable;

            double BackgroundGrowTime;
            double BackgroundIncrementTime;

            bool DebugAnnotateDifference;
            bool DebugAnnotateMovingRegions;

            Options()
            {
                TextureGradientSaturation = 16;
                TextureGradientBoost = 4;

                DifferenceGrayFeatureWeight = 18;
                DifferenceDxFeatureWeight = 18;
                DifferenceDyFeatureWeight = 18;
                DifferencePropagateForward = true;
                DifferenceRoiMaskEnable = true;

                BackgroundGrowTime = 1.0;
                BackgroundIncrementTime = 1.0;

                DebugAnnotateDifference = false;
                DebugAnnotateMovingRegions = true;
            }
        };

        struct Detector
        {
            Detector()
            {
            }

            ~Detector()
            {
            }

            bool SetOptions(const Options & options)
            {
                _options = options;
                return true;
            }

            bool SetModel(const Model & model)
            {
                _model = model;
                return true;
            }

            bool NextFrame(const Frame & input, Metadata & metadata, Frame * output = NULL)
            {
                SIMD_CHECK_PERFORMANCE();

                if (output && output->Size() != input.Size())
                    return false;

                if (!Calibrate(input.Size()))
                    return false;

                SetFrame(input, output);

                EstimateTextures();

                EstimateDifference();

                PerformSegmentation();

                _scene.stability.sceneState = Stability::Stable;

                UpdateBackground();

                DebugAnnotation();

                return true;
            }

        private:
            Options _options;
            Simd::Motion::Model _model;

            typedef std::pair<size_t, size_t> Scanline;
            typedef std::vector<Scanline> Scanlines;

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
                Size frameSize;
                Points roi;

                size_t levelCount;

                Pyramid roiMask;
                SearchRegions searchRegions;
            };

            struct MovingRegion
            {
                Rects rects;

                uint8_t index;
                Rect rect;
                int level;
                Time time;
                Point point;

                MovingRegion(const uint8_t & index_, const Rect & rect_, int level_, const Time & time_)
                    : index(index_)
                    , rect(rect_)
                    , level(level_)
                    , time(time_)
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
                    dx.Create(size, levelCount, options.DifferenceDxFeatureWeight);
                    dy.Create(size, levelCount, options.DifferenceDyFeatureWeight);

                    features.clear();
                    features.push_back(&gray);
                    features.push_back(&dx);
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
                int count;
                int sabotageCounter;
                Time expandEndTime;
                Time lastFrameTime;
                Time incrementCounterTime;

                Background()
                    : state(Init)
                {
                }
            };

            struct Stability
            {
                enum State
                {
                    Empty,
                    Ready
                };

                State state;

                enum SceneState
                {
                    Unknown,
                    Stable,
                    Sabotage
                } sceneState;

                bool sabotage;
                int kMovingRegionScore;
                int kMovingRegionThreshold;
                int kMovingRegionMinShortage;

                Stability()
                    : state(Empty)
                    , sceneState(Unknown)
                    , sabotage(false)
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
                int movingRegionAreaMin;

                MovingRegionPtrs movingRegions;
            };

            struct Scene
            {
                Frame input, *output;
                View gray;

                Pyramid buffer;

                Detector::Model model;

                Texture texture;

                Background background;

                Stability stability;

                Pyramid difference;

                Segmentation segmentation;

                void Create(const Options & options)
                {
                    gray.Recreate(model.frameSize, View::Gray8);
                    buffer.Recreate(model.frameSize, model.levelCount);
                    texture.Create(model.frameSize, model.levelCount, options);
                    difference.Recreate(model.frameSize, model.levelCount);
                    segmentation.mask.Recreate(model.frameSize, model.levelCount);
                }
            };

            bool SetFrame(const Frame & input, Frame * output)
            {
                SIMD_CHECK_PERFORMANCE();

                _scene.input = input;
                _scene.output = output;
                Simd::Convert(input, Frame(_scene.gray).Ref());

                return true;
            }

            bool Calibrate(const Size & frameSize)
            {
                Model & model = _scene.model;

                if (model.frameSize == frameSize)
                    return true;

                model.frameSize = frameSize;
                model.roi = _model.roi;
                model.levelCount = 3;

                GenerateSearchRegion(model);
                GenerateSearchRegionScanlines(model);

                _scene.Create(_options);

                return true;
            }

            void GenerateSearchRegion(Model & model)
            {
                model.searchRegions.clear();

                Size size(model.frameSize);
                for (int level = 1; level < model.levelCount; level++)
                    size = Simd::Scale(size);

                int level = (int)model.levelCount - 1;
                const Rect rect(1, 1, size.x - 1, size.y - 1);
                model.searchRegions.push_back(SearchRegion(rect, level));
            }

            void GenerateSearchRegionScanlines(Model & model)
            {
                //static const int ROI_EMPTY = 0;
                static const int ROI_NON_EMPTY = 255;

                model.roiMask.Recreate(model.frameSize, model.levelCount);

                //model.roiMask.Fill(ROI_EMPTY);
                //Alg::DrawFilledPolygon<Channel> drawFilledPolygon(model.roiMask[0], ROI_NON_EMPTY);
                //drawFilledPolygon(model.roi);
                //Simd::Build(model.roiMask, SimdReduce4x4);
                Simd::Fill(model.roiMask, ROI_NON_EMPTY);

                for (size_t i = 0; i < model.searchRegions.size(); ++i)
                {
                    SearchRegion & region = model.searchRegions[i];
                    int scale = region.scale;

                    assert(scale < (int)model.roiMask.Size());

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

            bool EstimateTextures()
            {
                SIMD_CHECK_PERFORMANCE();

                Texture & texture = _scene.texture;

                Simd::Copy(_scene.gray, texture.gray.value[0]);
                Simd::Build(texture.gray.value, SimdReduce4x4);

                for (size_t i = 0; i < texture.gray.value.Size(); ++i)
                {
                    Simd::TextureBoostedSaturatedGradient(texture.gray.value[i],
                        _options.TextureGradientSaturation, _options.TextureGradientBoost, 
                        texture.dx.value[i], texture.dy.value[i]);
                }

                return true;
            }

            bool EstimateDifference()
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

                return true;
            }

            bool PerformSegmentation()
            {
                Point neighbours[4];
                neighbours[0] = Point(-1, 0);
                neighbours[1] = Point(0, -1);
                neighbours[2] = Point(1, 0);
                neighbours[3] = Point(0, 1);

                Segmentation & segmentation = _scene.segmentation;
                const Model & model = _scene.model;
                const Time & time = _scene.input.timestamp;

                segmentation.differenceCreationMin = 128;
                segmentation.differenceExpansionMin = 96;
                segmentation.movingRegionAreaMin = 16;
                    
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
                                    return false;
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

                                if (region->rect.Area() <= segmentation.movingRegionAreaMin)
                                    Simd::SegmentationChangeIndex(segmentation.mask[region->level].Region(region->rect).Ref(), region->index, Segmentation::MaskInvalid);
                                else
                                {
                                    ComputeIndex(segmentation,*region);
                                    if (!region->rect.Empty())
                                    {
                                        region->level = searchRegion.scale;
                                        //if (motion.detectLevel)
                                        //    region->rect = region->rect * (1 << motion.detectLevel);
                                        region->point = region->rect.Center();
                                        //region->inRoi = model.roi.HasPoint(region->point);
                                        segmentation.movingRegions.push_back(region);
                                    }
                                }
                            }
                        }
                    }
                }
                return true;
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

            bool UpdateBackground()
            {
                SIMD_CHECK_PERFORMANCE();

                Background & background = _scene.background;
                const Stability::SceneState & stabilityState = _scene.stability.sceneState;
                const Time & time = _scene.input.timestamp;

                switch (background.state)
                {
                case Background::Update:
                    switch (stabilityState)
                    {
                    case Stability::Stable:
                        Apply(_scene.texture.features, IncrementCountUpdater());
                        ++background.count;
                        background.incrementCounterTime += time - background.lastFrameTime;

                        if (background.count >= 127 || (background.incrementCounterTime > _options.BackgroundIncrementTime && background.count >= 8))
                        {
                            Apply(_scene.texture.features, AdjustRangeUpdater());

                            background.incrementCounterTime = 0;
                            background.count = 0;
                        }
                        break;
                    case Stability::Sabotage:
                        background.sabotageCounter++;
                        if (background.sabotageCounter > 0)//_sabotageCounterMax)
                        {
                            InitBackground();
                        }
                        break;

                    default:
                        assert(0);
                    }

                    if (stabilityState != Stability::Sabotage)
                    {
                        background.sabotageCounter = 0;
                    }
                    break;

                case Background::Grow:

                    if (stabilityState == Stability::Sabotage)
                    {
                        InitBackground();
                    }
                    else
                    {
                        Apply(_scene.texture.features, GrowRangeUpdater());

                        if (stabilityState != Stability::Stable && _scene.stability.state != Stability::Empty)
                        {
                            background.expandEndTime = time + _options.BackgroundGrowTime;
                        }

                        if (background.expandEndTime < time)
                        {
                            background.state = Background::Update;
                            background.count = 0;
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
                return true;
            }

            void InitBackground()
            {
                Background & background = _scene.background;

                Apply(_scene.texture.features, InitUpdater());

                background.expandEndTime = _scene.input.timestamp + _options.BackgroundGrowTime;
                background.state = Background::Grow;
                background.count = 0;
                background.incrementCounterTime = 0;
            }

            bool DebugAnnotation()
            {
                Frame * output = _scene.output;

                if (output && output->format == Frame::Bgr24)
                {
                    if (_options.DebugAnnotateDifference)
                    {
                        const View & src = _scene.difference[1];
                        Simd::GrayToBgr(src, output->planes[0].Region(src.Size(), View::BottomRight).Ref());
                    }

                    if (_options.DebugAnnotateMovingRegions)
                    {
                        Simd::Pixel::Bgr24 color(0, 255, 0);
                        for (size_t i = 0; i < _scene.segmentation.movingRegions.size(); ++i)
                        {
                            const MovingRegion & region = *_scene.segmentation.movingRegions[i];
                            Simd::DrawRectangle(output->planes[0], region.rect, color, 1);
                        }
                    }
                }

                return true;
            }

            Scene _scene;
        };
    }
}

#endif//__SimdMotion_hpp__
