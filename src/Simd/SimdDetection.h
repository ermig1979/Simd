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
#ifndef __SimdDetection_h__
#define __SimdDetection_h__

#include "Simd/SimdConst.h"

#ifdef SIMD_DETECTION_ENABLE

#include "Simd/SimdView.hpp"

#include <vector>

namespace Simd
{
    namespace Detection
    {
        typedef Simd::View<Simd::Allocator> View;
        typedef Simd::Point<ptrdiff_t> Size;
        typedef Simd::Rectangle<ptrdiff_t> Rect;

        struct Data
        {
            enum FeatureType
            {
                HAAR = 0,
                LBP = 1,
                HOG = 2
            };

            struct DTreeNode
            {
                int featureIdx;
                float threshold; // for ordered features only
                int left;
                int right;
            };

            struct DTree
            {
                int nodeCount;
            };

            struct Stage
            {
                int first;
                int ntrees;
                float threshold;
            };

            struct Rect
            {
                int x, y, width, height;
                Rect() : x(0), y(0), width(0), height(0) {}
            };

            struct WeightedRect
            {
                Rect r;
                float weight;
                WeightedRect() : weight(0) {}
            };

            struct HaarFeature
            {
                bool tilted;
                enum { RECT_NUM = 3 };
                WeightedRect rect[RECT_NUM];
            };

            struct LbpFeature
            {
                Rect rect;
            };

            bool isStumpBased;

            int stageType;
            FeatureType featureType;
            int ncategories;
            Size origWinSize;

            std::vector<Stage> stages;
            std::vector<DTree> classifiers;
            std::vector<DTreeNode> nodes;
            std::vector<float> leaves;
            std::vector<int> subsets;

            std::vector<HaarFeature> haarFeatures;
            std::vector<LbpFeature> lbpFeatures;
        };

        //---------------------------------------------------------------------

        struct WeightedRect
        {
            uint32_t *p0, *p1, *p2, *p3;
            float weight;
        };

        struct HidHaarFeature
        {
            WeightedRect rect[Data::HaarFeature::RECT_NUM];
        };

        struct HidHaarStage
        {
            int first;
            int ntrees;
            float threshold;
            bool hasThree;
            bool canSkip;
        };

        struct HidHaarNode
        {
            int featureIdx;
            int left;
            int right;
            float threshold;
        };

        struct HidHaarCascade
        {
            Size origWinSize;
            bool isStumpBased;
            bool isThroughColumn;
            bool hasTilted;

            typedef HidHaarNode Node;
            typedef std::vector<Node> Nodes;

            struct Tree
            {
                int nodeCount;
            };
            typedef std::vector<Tree> Trees;

            typedef HidHaarFeature Feature;
            typedef std::vector<Feature> Features;

            typedef HidHaarStage Stage;
            typedef std::vector<Stage> Stages;

            typedef float Leave;
            typedef std::vector<Leave> Leaves;

            typedef int ILeave;
            typedef std::vector<ILeave> ILeaves;

            Nodes nodes;
            Trees trees;
            Stages stages;
            Leaves leaves;
            Features features;

            float windowArea;
            float invWinArea;
            uint32_t *pq[4];
            uint32_t *p[4];

            View sum, sqsum, tilted;
            View isum, itilted;
        };
    }

    namespace Base
    {
        using namespace Detection;

        SIMD_INLINE uint32_t Sum32i(uint32_t * const ptr[4], size_t offset)
        {
            return ptr[0][offset] - ptr[1][offset] - ptr[2][offset] + ptr[3][offset];
        }

        SIMD_INLINE float Norm32f(const HidHaarCascade & hid, size_t offset)
        {
            float sum = float(Sum32i(hid.p, offset));
            float sqsum = float(Sum32i(hid.pq, offset));
            float q = sqsum*hid.windowArea - sum *sum;
            return q < 0.0f ? 1.0f : sqrtf(q);
        }

        SIMD_INLINE int Norm16i(const HidHaarCascade & hid, size_t offset)
        {
            return Simd::Round(Norm32f(hid, offset)*hid.invWinArea);
        }

        SIMD_INLINE float WeightedSum32f(const WeightedRect & rect, size_t offset)
        {
            uint32_t sum = rect.p0[offset] - rect.p1[offset] - rect.p2[offset] + rect.p3[offset];
            return rect.weight*sum;
        }

        int DetectionHaarDetect32f(const struct HidHaarCascade & hid, size_t offset, int startStage, float norm);
    }
}

#endif

#endif//__SimdDetection_h__
