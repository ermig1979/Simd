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
#include "Simd/SimdMemory.h"
#include "Simd/SimdBase.h"

#ifdef SIMD_DETECTION_ENABLE

#include "Simd/SimdDetection.h"
#include "Simd/SimdBase_tinyxml2.h"

#include <sstream>
#include <exception>
#include <iostream>

#define SIMD_EX(message) \
{ \
	std::stringstream __ss; \
	__ss << message; \
    std::cerr << __ss.str().c_str() << std::endl; \
    throw std::runtime_error(__ss.str().c_str()); \
}

namespace Simd
{
    namespace Base
    {
        namespace Xml
        {
            template <class T> T FromString(const std::string & s)
            {
                T t;
                std::stringstream(s) >> t;
                return t;
            }

            template<class T> inline T GetValue(tinyxml2::XMLNode * parent)
            {
                if (parent == NULL)
                    SIMD_EX("Invalid element!");
                tinyxml2::XMLNode * child = parent->FirstChild();
                if (child == NULL)
                    SIMD_EX("Invalid node!");
                return FromString<T>(child->Value());
            }

            template<class T> inline T GetValue(tinyxml2::XMLNode * parent, const char * name)
            {
                if (parent == NULL)
                    SIMD_EX("Invalid element!");
                return GetValue<T>(parent->FirstChildElement(name));
            }

            template<class T> inline std::vector<T> GetValues(tinyxml2::XMLNode * parent)
            {
                if (parent == NULL)
                    SIMD_EX("Invalid element!");
                tinyxml2::XMLNode * child = parent->FirstChild();
                if (child == NULL)
                    SIMD_EX("Invalid node!");
                std::stringstream ss(tinyxml2::XMLUtil::SkipWhiteSpace(child->Value()));
                std::vector<T> values;
                while (!ss.eof())
                {
                    T value;
                    ss >> value;
                    values.push_back(value);
                }
                return values;
            }

            template<class T> inline std::vector<T> GetValues(tinyxml2::XMLNode * parent, const char * name)
            {
                if (parent == NULL)
                    SIMD_EX("Invalid element!");
                return GetValues<T>(parent->FirstChildElement(name));
            }

            inline size_t GetSize(tinyxml2::XMLNode * parent)
            {
                size_t count = 0;
                for (tinyxml2::XMLNode * node = parent->FirstChildElement(); node != NULL; node = node->NextSiblingElement())
                    count++;
                return count;
            }
        }

        namespace Names
        {
            const char * cascade = "cascade";
            const char * BOOST = "BOOST";
            const char * stageType = "stageType";
            const char * featureType = "featureType";
            const char * HAAR = "HAAR";
            const char * LBP = "LBP";
            const char * HOG = "HOG";
            const char * width = "width";
            const char * height = "height";
            const char * stageParams = "stageParams";
            const char * maxDepth = "maxDepth";
            const char * featureParams = "featureParams";
            const char * maxCatCount = "maxCatCount";
            const char * stages = "stages";
            const char * stageThreshold = "stageThreshold";
            const char * weakClassifiers = "weakClassifiers";
            const char * internalNodes = "internalNodes";
            const char * leafValues = "leafValues";
            const char * features = "features";
            const char * rects = "rects";
            const char * tilted = "tilted";
            const char * rect = "rect";
        }

        void * DetectionDataLoadA(const char * path)
        {
            static const float THRESHOLD_EPS = 1e-5f;

            Data * data = NULL; 
            try
            {
                tinyxml2::XMLDocument xml;
                if (xml.LoadFile(path) != tinyxml2::XML_SUCCESS)
                    SIMD_EX("Can't load XML file '" << path << "'!");

                tinyxml2::XMLElement * root = xml.RootElement();
                if (root == NULL)
                    SIMD_EX("Invalid format of XML file '" << path << "'!");

                tinyxml2::XMLElement * cascade = root->FirstChildElement(Names::cascade);
                if (cascade == NULL)
                    return data;

                data = new Data();

                if (Xml::GetValue<std::string>(cascade, Names::stageType) != Names::BOOST)
                    SIMD_EX("Invalid cascade stage type!");
                data->stageType = 0;

                std::string featureType = Xml::GetValue<std::string>(cascade, Names::featureType);
                if (featureType == Names::HAAR)
                    data->featureType = Data::HAAR;
                else if (featureType == Names::LBP)
                    data->featureType = Data::LBP;
                else if (featureType == Names::HOG)
                    data->featureType = Data::HOG;
                else
                    SIMD_EX("Invalid cascade feature type!")

                    data->origWinSize.x = Xml::GetValue<int>(cascade, Names::width);
                data->origWinSize.y = Xml::GetValue<int>(cascade, Names::height);
                if (data->origWinSize.x <= 0 || data->origWinSize.y <= 0)
                    SIMD_EX("Invalid cascade width or height!")

                    tinyxml2::XMLElement * stageParams = cascade->FirstChildElement(Names::stageParams);
                if (stageParams && stageParams->FirstChildElement(Names::maxDepth))
                    data->isStumpBased = Xml::GetValue<int>(stageParams, Names::maxDepth) == 1 ? true : false;
                else
                    data->isStumpBased = true;

                tinyxml2::XMLElement * featureParams = cascade->FirstChildElement(Names::featureParams);
                data->ncategories = Xml::GetValue<int>(featureParams, Names::maxCatCount);
                int subsetSize = (data->ncategories + 31) / 32;
                int nodeStep = 3 + (data->ncategories > 0 ? subsetSize : 1);

                tinyxml2::XMLElement * stages = cascade->FirstChildElement(Names::stages);
                if (stages == NULL)
                    SIMD_EX("Invalid stages count!");
                data->stages.reserve(Xml::GetSize(stages));
                int stageIndex = 0;
                for (tinyxml2::XMLNode * stageNode = stages->FirstChildElement(); stageNode != NULL; stageNode = stageNode->NextSiblingElement(), ++stageIndex)
                {
                    Data::Stage stage;
                    stage.threshold = Xml::GetValue<float>(stageNode, Names::stageThreshold) - THRESHOLD_EPS;

                    tinyxml2::XMLNode * weakClassifiers = stageNode->FirstChildElement(Names::weakClassifiers);
                    if (weakClassifiers == NULL)
                        SIMD_EX("Invalid weak classifiers count!");
                    stage.ntrees = (int)Xml::GetSize(weakClassifiers);
                    stage.first = (int)data->classifiers.size();
                    data->stages.push_back(stage);
                    data->classifiers.reserve(data->stages[stageIndex].first + data->stages[stageIndex].ntrees);

                    for (tinyxml2::XMLNode * weakClassifier = weakClassifiers->FirstChildElement(); weakClassifier != NULL; weakClassifier = weakClassifier->NextSiblingElement())
                    {
                        std::vector<double> internalNodes = Xml::GetValues<double>(weakClassifier, Names::internalNodes);
                        std::vector<float> leafValues = Xml::GetValues<float>(weakClassifier, Names::leafValues);

                        Data::DTree tree;
                        tree.nodeCount = (int)internalNodes.size() / nodeStep;
                        if (tree.nodeCount > 1)
                            data->isStumpBased = false;
                        data->classifiers.push_back(tree);

                        data->nodes.reserve(data->nodes.size() + tree.nodeCount);
                        data->leaves.reserve(data->leaves.size() + leafValues.size());
                        if (subsetSize)
                            data->subsets.reserve(data->subsets.size() + tree.nodeCount*subsetSize);

                        for (int n = 0; n < tree.nodeCount; ++n)
                        {
                            Data::DTreeNode node;
                            node.left = (int)internalNodes[n*nodeStep + 0];
                            node.right = (int)internalNodes[n*nodeStep + 1];
                            node.featureIdx = (int)internalNodes[n*nodeStep + 2];
                            if (subsetSize)
                            {
                                for (int j = 0; j < subsetSize; j++)
                                    data->subsets.push_back((int)internalNodes[n*nodeStep + 3 + j]);
                                node.threshold = 0.f;
                            }
                            else
                            {
                                node.threshold = (float)internalNodes[n*nodeStep + 3];
                            }
                            data->nodes.push_back(node);
                        }

                        for (size_t i = 0; i < leafValues.size(); ++i)
                            data->leaves.push_back(leafValues[i]);
                    }
                }

                tinyxml2::XMLNode * featureNodes = cascade->FirstChildElement(Names::features);
                if (data->featureType == Data::HAAR)
                {
                    data->haarFeatures.reserve(Xml::GetSize(featureNodes));
                    for (tinyxml2::XMLNode * featureNode = featureNodes->FirstChildElement(); featureNode != NULL; featureNode = featureNode->NextSiblingElement())
                    {
                        Data::HaarFeature feature;
                        int rectIndex = 0;
                        tinyxml2::XMLNode * rectsNode = featureNode->FirstChildElement(Names::rects);
                        for (tinyxml2::XMLNode * rectNode = rectsNode->FirstChildElement(); rectNode != NULL; rectNode = rectNode->NextSiblingElement(), rectIndex++)
                        {
                            std::vector<double> values = Xml::GetValues<double>(rectNode);
                            feature.rect[rectIndex].r.x = (int)values[0];
                            feature.rect[rectIndex].r.y = (int)values[1];
                            feature.rect[rectIndex].r.width = (int)values[2];
                            feature.rect[rectIndex].r.height = (int)values[3];
                            feature.rect[rectIndex].weight = (float)values[4];
                        }
                        feature.tilted = featureNode->FirstChildElement(Names::tilted) && Xml::GetValue<int>(featureNode, Names::tilted) != 0;
                        data->haarFeatures.push_back(feature);
                    }
                }

                if (data->featureType == Data::LBP)
                {
                    data->lbpFeatures.reserve(Xml::GetSize(featureNodes));
                    for (tinyxml2::XMLNode * featureNode = featureNodes->FirstChildElement(); featureNode != NULL; featureNode = featureNode->NextSiblingElement())
                    {
                        Data::LbpFeature feature;
                        std::vector<int> values = Xml::GetValues<int>(featureNode, Names::rect);
                        feature.rect.x = values[0];
                        feature.rect.y = values[1];
                        feature.rect.width = values[2];
                        feature.rect.height = values[3];
                        data->lbpFeatures.push_back(feature);
                    }
                }
            }
            catch (...)
            {
                delete data;
                data = NULL;
            }
            return data;
        }

        void DetectionDataFree(void * data)
        {
            delete (Data*)data;
        }

        size_t DetectionWindowWidth(const void * data)
        {
            if (data)
                return ((Data*)data)->origWinSize.x;
            else
                return 0;
        }

        size_t DetectionWindowHeight(const void * data)
        {
            if (data)
                return ((Data*)data)->origWinSize.y;
            else
                return 0;
        }

        HidHaarCascade * CreateHid(const Data & data)
        {
            if (data.featureType != Data::HAAR)
                SIMD_EX("It is not HAAR cascade!");

            HidHaarCascade * hid = new HidHaarCascade();

            hid->isThroughColumn = false;
            hid->isStumpBased = data.isStumpBased;
            hid->origWinSize = data.origWinSize;

            hid->trees.resize(data.classifiers.size());
            for (size_t i = 0; i < data.classifiers.size(); ++i)
                hid->trees[i].nodeCount = data.classifiers[i].nodeCount;

            hid->nodes.resize(data.nodes.size());
            for (size_t i = 0; i < data.nodes.size(); ++i)
            {
                hid->nodes[i].featureIdx = data.nodes[i].featureIdx;
                hid->nodes[i].left = data.nodes[i].left;
                hid->nodes[i].right = data.nodes[i].right;
                hid->nodes[i].threshold = data.nodes[i].threshold;
            }

            hid->stages.resize(data.stages.size());
            hid->leaves.resize(data.leaves.size());
            for (size_t i = 0; i < data.stages.size(); ++i)
            {
                hid->stages[i].first = data.stages[i].first;
                hid->stages[i].ntrees = data.stages[i].ntrees;
                hid->stages[i].threshold = data.stages[i].threshold;
                hid->stages[i].hasThree = false;
                for (int j = data.stages[i].first, n = data.stages[i].first + data.stages[i].ntrees; j < n; ++j)
                {
                    hid->leaves[2 * j + 0] = data.leaves[2 * j + 0];
                    hid->leaves[2 * j + 1] = data.leaves[2 * j + 1];
                    if (data.haarFeatures[data.nodes[j].featureIdx].rect[2].weight != 0)
                        hid->stages[i].hasThree = true;
                }
            }

            hid->features.resize(data.haarFeatures.size());
            for (size_t i = 0; i < hid->features.size(); ++i)
            {
                for (int j = 0; j < Data::HaarFeature::RECT_NUM; ++j)
                    hid->features[i].rect[j].weight = data.haarFeatures[i].rect[j].weight;
                if (data.haarFeatures[i].tilted)
                    hid->hasTilted = true;
            }

            return hid;
        }

        typedef Detection::View View;

        template <class T> SIMD_INLINE T * SumElemPtr(const View & view, ptrdiff_t row, ptrdiff_t col, bool throughColumn)
        {
            assert(view.ChannelCount() == 1 && view.ChannelSize() == sizeof(T));
            assert(row >= 0 && col >= 0 && col < (ptrdiff_t)view.width && row < (ptrdiff_t)view.height);

            if (throughColumn)
            {
                if (col & 1)
                    return (T*)& view.At<T>(col / 2 + (view.width + 1) / 2, row);
                else
                    return (T*)& view.At<T>(col / 2, row);
            }
            else
                return (T*)& view.At<T>(col, row);
        }

        static void InitBase(HidHaarCascade * hid, const View & sum, const View & sqsum, const View & tilted)
        {
            Rect rect(1, 1, hid->origWinSize.x - 1, hid->origWinSize.y - 1);
            hid->windowArea = (float)rect.Area();

            hid->p[0] = SumElemPtr<uint32_t>(sum, rect.top, rect.left, false);
            hid->p[1] = SumElemPtr<uint32_t>(sum, rect.top, rect.right, false);
            hid->p[2] = SumElemPtr<uint32_t>(sum, rect.bottom, rect.left, false);
            hid->p[3] = SumElemPtr<uint32_t>(sum, rect.bottom, rect.right, false);

            hid->pq[0] = SumElemPtr<uint32_t>(sqsum, rect.top, rect.left, false);
            hid->pq[1] = SumElemPtr<uint32_t>(sqsum, rect.top, rect.right, false);
            hid->pq[2] = SumElemPtr<uint32_t>(sqsum, rect.bottom, rect.left, false);
            hid->pq[3] = SumElemPtr<uint32_t>(sqsum, rect.bottom, rect.right, false);

            hid->sum = sum;
            hid->sqsum = sum;
            hid->tilted = tilted;
        }

        template<class T> SIMD_INLINE void UpdateFeaturePtrs(HidHaarCascade * hid, const Data & data)
        {
            View sum = hid->isThroughColumn ? hid->isum : hid->sum;
            View tilted = hid->isThroughColumn ? hid->itilted : hid->tilted;
            for (size_t i = 0; i < hid->features.size(); i++)
            {
                const Data::HaarFeature & df = data.haarFeatures[i];
                HidHaarCascade::Feature & hf = hid->features[i];
                for (int j = 0; j < Data::HaarFeature::RECT_NUM; ++j)
                {
                    const Data::Rect & dr = df.rect[j].r;
                    WeightedRect & hr = hf.rect[j];
                    if (hr.weight != 0.0)
                    {
                        if (df.tilted)
                        {
                            hr.p0 = SumElemPtr<T>(tilted, dr.y, dr.x, hid->isThroughColumn);
                            hr.p1 = SumElemPtr<T>(tilted, dr.y + dr.height, dr.x - dr.height, hid->isThroughColumn);
                            hr.p2 = SumElemPtr<T>(tilted, dr.y + dr.width, dr.x + dr.width, hid->isThroughColumn);
                            hr.p3 = SumElemPtr<T>(tilted, dr.y + dr.width + dr.height, dr.x + dr.width - dr.height, hid->isThroughColumn);
                        }
                        else
                        {
                            hr.p0 = SumElemPtr<T>(sum, dr.y, dr.x, hid->isThroughColumn);
                            hr.p1 = SumElemPtr<T>(sum, dr.y, dr.x + dr.width, hid->isThroughColumn);
                            hr.p2 = SumElemPtr<T>(sum, dr.y + dr.height, dr.x, hid->isThroughColumn);
                            hr.p3 = SumElemPtr<T>(sum, dr.y + dr.height, dr.x + dr.width, hid->isThroughColumn);
                        }
                    }
                    else
                    {
                        hr.p0 = NULL;
                        hr.p1 = NULL;
                        hr.p2 = NULL;
                        hr.p3 = NULL;
                    }
                }
            }
        }

        HidHaarCascade * InitVector(const Data & data, const View & sum, const View & sqsum, const View & tilted, bool throughColumn)
        {
            if (!data.isStumpBased)
                SIMD_EX("Can't use tree classfier for vector haar classifier!");

            HidHaarCascade * hid = CreateHid(data);
            InitBase(hid, sum, sqsum, tilted);
            if (throughColumn)
            {
                hid->isThroughColumn = true;
                hid->isum.Recreate(sum.width, sum.height, View::Int32, NULL, View::PixelSize(View::Int32));
                if (hid->hasTilted)
                    hid->itilted.Recreate(tilted.width, tilted.height, View::Int32, NULL, View::PixelSize(View::Int32));
            }
            UpdateFeaturePtrs<uint32_t>(hid, data);
            return hid;
        }

        void * DetectionHaarInit(const void * data, uint8_t * sum, size_t sumStride, size_t width, size_t height,
            uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride, int throughColumn)
        {
            return InitVector(*(Data*)data, 
                View(width, height, sumStride, View::Int32, sum),
                View(width, height, sqsumStride, View::Int32, sqsum), 
                View(width, height, tiltedStride, View::Int32, tilted),
                throughColumn != 0);
        }

        void DetectionHaarFree(void * hid)
        {
            delete (HidHaarCascade*)hid;
        }
        
        int DetectionHaarHasTilted(const void * hid)
        {
            return ((HidHaarCascade*)hid)->hasTilted;
        }

        void PrepareThroughColumn32i(const View & src, View & dst)
        {
            assert(Simd::Compatible(src, dst) && src.format == View::Int32);

            for (size_t row = 0; row < src.height; ++row)
            {
                const uint32_t * s = &src.At<uint32_t>(0, row);

                uint32_t * evenDst = &dst.At<uint32_t>(0, row);
                for (size_t col = 0; col < src.width; col += 2)
                    evenDst[col >> 1] = s[col];

                uint32_t * oddDst = &dst.At<uint32_t>((dst.width + 1) >> 1, row);
                for (size_t col = 1; col < src.width; col += 2)
                    oddDst[col >> 1] = s[col];
            }
        }

        void DetectionHaarPrepare(void * _hid)
        {
            HidHaarCascade * hid = (HidHaarCascade*)_hid;
            if (hid->isThroughColumn)
            {
                PrepareThroughColumn32i(hid->sum, hid->isum);
                if (hid->hasTilted)
                    PrepareThroughColumn32i(hid->tilted, hid->itilted);
            }
        }

        int DetectionHaarDetect32f(const HidHaarCascade & hid, size_t offset, int startStage, float norm)
        {
            typedef HidHaarCascade Hid;
            const Hid::Stage * stages = hid.stages.data();
            const Hid::Node * node = hid.nodes.data() + stages[startStage].first;
            const float * leaves = hid.leaves.data() + stages[startStage].first * 2;
            for (int i = startStage, n = (int)hid.stages.size(); i < n; ++i)
            {
                const Hid::Stage & stage = stages[i];
                if (stage.canSkip)
                    continue;
                const Hid::Node * end = node + stage.ntrees;
                float stageSum = 0.0;
                if (stage.hasThree)
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature & feature = hid.features[node->featureIdx];
                        float sum = WeightedSum32f(feature.rect[0], offset) + WeightedSum32f(feature.rect[1], offset);
                        if (feature.rect[2].p0)
                            sum += WeightedSum32f(feature.rect[2], offset);
                        stageSum += leaves[sum >= node->threshold*norm];
                    }
                }
                else
                {
                    for (; node < end; ++node, leaves += 2)
                    {
                        const Hid::Feature & feature = hid.features[node->featureIdx];
                        float sum = WeightedSum32f(feature.rect[0], offset) + WeightedSum32f(feature.rect[1], offset);
                        stageSum += leaves[sum >= node->threshold*norm];
                    }
                }
                if (stageSum < stage.threshold)
                    return -i;
            }
            return 1;
        }

        void DetectionHaarDetect32fp(const HidHaarCascade & hid, const View & mask, const Rect & rect, View & dst)
        {
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t p_offset = row * hid.sum.stride / sizeof(uint32_t);
                size_t pq_offset = row * hid.sqsum.stride / sizeof(uint32_t);
                for (ptrdiff_t col = rect.left; col < rect.right; col += 1)
                {
                    if (mask.At<uint8_t>(col, row) == 0)
                        continue;
                    float norm = Norm32f(hid, pq_offset + col);
                    if (DetectionHaarDetect32f(hid, p_offset + col, 0, norm) > 0)
                        dst.At<uint8_t>(col, row) = 1;
                }
            }
        }

        void DetectionHaarDetect32fp(const void * _hid, const uint8_t * mask, size_t maskStride, 
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidHaarCascade & hid = *(HidHaarCascade*)_hid;
            return DetectionHaarDetect32fp(hid,
                View(hid.isum.width - 1, hid.isum.height - 1, maskStride, View::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                View(hid.isum.width - 1, hid.isum.height - 1, dstStride, View::Gray8, dst));
        }

        void DetectionHaarDetect32fi(const HidHaarCascade & hid, const View & mask, const Rect & rect, View & dst)
        {
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 2)
            {
                size_t p_offset = row * hid.isum.stride / sizeof(uint32_t);
                size_t pq_offset = row * hid.sqsum.stride / sizeof(uint32_t);
                for (ptrdiff_t col = rect.left; col < rect.right; col += 2)
                {
                    if (mask.At<uint8_t>(col, row) == 0)
                        continue;
                    float norm = Norm32f(hid, pq_offset + col);
                    if (DetectionHaarDetect32f(hid, p_offset + col / 2, 0, norm) > 0)
                        dst.At<uint8_t>(col, row) = 1;
                }
            }
        }

        void DetectionHaarDetect32fi(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidHaarCascade & hid = *(HidHaarCascade*)_hid;
            return DetectionHaarDetect32fi(hid,
                View(hid.isum.width - 1, hid.isum.height - 1, maskStride, View::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                View(hid.isum.width - 1, hid.isum.height - 1, dstStride, View::Gray8, dst));
        }
    }
}

#endif