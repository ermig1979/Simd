/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdAlignment.h"

#include "Simd/SimdView.hpp"

#include <vector>

namespace Simd
{
    namespace Detection
    {
        typedef Simd::View<Simd::Allocator> Image;
        typedef Simd::Point<ptrdiff_t> Size;
        typedef Simd::Rectangle<ptrdiff_t> Rect;

        struct Data : public Deletable
        {
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
            bool hasTilted;
            bool canInt16;

            int stageType;
            SimdDetectionInfoFlags featureType;
            int ncategories;
            Size origWinSize;

            std::vector<Stage> stages;
            std::vector<DTree> classifiers;
            std::vector<DTreeNode> nodes;
            std::vector<float> leaves;
            std::vector<int> subsets;

            std::vector<HaarFeature> haarFeatures;
            std::vector<LbpFeature> lbpFeatures;

            virtual ~Data() {}
        };

        struct HidBase : public Deletable
        {
            SimdDetectionInfoFlags featureType;
            Size origWinSize;
            bool isStumpBased;
            bool isThroughColumn;
            bool hasTilted;
            bool isInt16;
            int ncategories;

            virtual ~HidBase() {}
        };

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

        struct HidHaarCascade : public HidBase
        {
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

            Image sum, sqsum, tilted;
            Image isum, itilted;

            virtual ~HidHaarCascade()
            {
            }
        };

        template<class TSum> struct HidLbpFeature
        {
            Rect rect;
            const TSum * p[16];
        };

        template <class TWeight> struct HidLbpStage
        {
            int first;
            int ntrees;
            TWeight threshold;
        };

        template<class TWeight, class TSum> struct HidLbpCascade : public HidBase
        {
            struct Node
            {
                int featureIdx;
                int left;
                int right;
            };
            typedef std::vector<Node> Nodes;

            struct Tree
            {
                int nodeCount;
            };
            typedef std::vector<Tree> Trees;

            typedef HidLbpStage<TWeight> Stage;
            typedef std::vector<Stage> Stages;

            typedef TWeight Leave;
            typedef std::vector<Leave> Leaves;

            typedef int Subset;
            typedef std::vector<Subset> Subsets;

            typedef HidLbpFeature<TSum> Feature;
            typedef std::vector<Feature> Features;

            Nodes nodes;
            Trees trees;
            Stages stages;
            Leaves leaves;
            Subsets subsets;
            Features features;

            Image sum;
            Image isum;

            virtual ~HidLbpCascade() {}
        };

        template <class T> struct Buffer
        {
            Buffer(size_t size)
            {
                _p = Allocate(2 * size * sizeof(T));
                m = (T*)_p;
                d = m + size;
            }

            ~Buffer()
            {
                Free(_p);
            }

            T *m, *d;
        private:
            void *_p;
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

        int Detect32f(const struct HidHaarCascade & hid, size_t offset, int startStage, float norm);

        template< class T> SIMD_INLINE T IntegralSum(const T * p0, const T * p1, const T * p2, const T * p3, ptrdiff_t offset)
        {
            return p0[offset] - p1[offset] - p2[offset] + p3[offset];
        }

        template< class T> SIMD_INLINE int Calculate(const HidLbpFeature<T> & feature, ptrdiff_t offset)
        {
            T central = IntegralSum(feature.p[5], feature.p[6], feature.p[9], feature.p[10], offset);

            return
                (IntegralSum(feature.p[0], feature.p[1], feature.p[4], feature.p[5], offset) >= central ? 128 : 0) |
                (IntegralSum(feature.p[1], feature.p[2], feature.p[5], feature.p[6], offset) >= central ? 64 : 0) |
                (IntegralSum(feature.p[2], feature.p[3], feature.p[6], feature.p[7], offset) >= central ? 32 : 0) |
                (IntegralSum(feature.p[6], feature.p[7], feature.p[10], feature.p[11], offset) >= central ? 16 : 0) |
                (IntegralSum(feature.p[10], feature.p[11], feature.p[14], feature.p[15], offset) >= central ? 8 : 0) |
                (IntegralSum(feature.p[9], feature.p[10], feature.p[13], feature.p[14], offset) >= central ? 4 : 0) |
                (IntegralSum(feature.p[8], feature.p[9], feature.p[12], feature.p[13], offset) >= central ? 2 : 0) |
                (IntegralSum(feature.p[4], feature.p[5], feature.p[8], feature.p[9], offset) >= central ? 1 : 0);
        }

        template<class TWeight, class TSum> inline int Detect(const HidLbpCascade<TWeight, TSum> & hid, size_t offset, int startStage)
        {
            typedef HidLbpCascade<TWeight, TSum> Hid;

            size_t subsetSize = (hid.ncategories + 31) / 32;
            const int * subsets = hid.subsets.data();
            const typename Hid::Leave * leaves = hid.leaves.data();
            const typename Hid::Node * nodes = hid.nodes.data();
            const typename Hid::Stage * stages = hid.stages.data();
            if (startStage >= (int)hid.stages.size())
                return 1;
            int nodeOffset = stages[startStage].first;
            int leafOffset = 2 * nodeOffset;
            for (int i_stage = startStage, n_stages = (int)hid.stages.size(); i_stage < n_stages; i_stage++)
            {
                const typename Hid::Stage & stage = stages[i_stage];
                TWeight sum = 0;
                for (int i_tree = 0, n_trees = stage.ntrees; i_tree < n_trees; i_tree++)
                {
                    const typename Hid::Node & node = nodes[nodeOffset];
                    int c = Calculate(hid.features[node.featureIdx], offset);
                    const int * subset = subsets + nodeOffset*subsetSize;
                    sum += leaves[subset[c >> 5] & (1 << (c & 31)) ? leafOffset : leafOffset + 1];
                    nodeOffset++;
                    leafOffset += 2;
                }
                if (sum < stage.threshold)
                    return -i_stage;
            }
            return 1;
        }
    }
}

#endif//__SimdDetection_h__
