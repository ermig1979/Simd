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
        typedef Simd::Point<ptrdiff_t> Point;
        typedef Point Size;

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
    }
}

#endif

#endif//__SimdDetection_h__
