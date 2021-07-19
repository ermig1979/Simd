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
#include "Simd/SimdMemory.h"
#include "Simd/SimdDetection.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdXml.hpp"

#include <exception>
#include <stdexcept>
#include <iostream>
#include <sstream>

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
            typedef Simd::Xml::File<char> File;
            typedef Simd::Xml::Document<char> Document;
            typedef Simd::Xml::Node<char> Node;

            template <class T> T FromString(const std::string & s)
            {
                T t;
                std::stringstream(s) >> t;
                return t;
            }

            template<class InputIterator> inline  InputIterator FindNotSpace(InputIterator first, InputIterator last)
            {
                while (first != last)
                {
                    if (!isspace(*first))
                        return first;
                    ++first;
                }
                return last;
            }

            template <> inline std::string FromString(const std::string & s)
            {
                std::string str(s);
                str.erase(str.begin(), FindNotSpace(str.begin(), str.end()));
                str.erase(FindNotSpace(str.rbegin(), str.rend()).base(), str.end());
                return str;
            }

            template<class T> inline T GetValue(Node * parent)
            {
                if (parent == NULL)
                    SIMD_EX("Invalid element!");
                Node * child = parent->FirstNode();
                if (child == NULL)
                    SIMD_EX("Invalid node!");
                return FromString<T>(child->Value());
            }

            template<class T> inline T GetValue(Node * parent, const char * name)
            {
                if (parent == NULL)
                    SIMD_EX("Invalid element!");
                return GetValue<T>(parent->FirstNode(name));
            }

            template<class T> inline std::vector<T> GetValues(Node * parent)
            {
                if (parent == NULL)
                    SIMD_EX("Invalid element!");
                Node * child = parent->FirstNode();
                if (child == NULL)
                    SIMD_EX("Invalid node!");
                std::stringstream ss(child->Value());
                std::vector<T> values;
                while (!ss.eof())
                {
                    T value;
                    ss >> value;
                    values.push_back(value);
                }
                return values;
            }

            template<class T> inline std::vector<T> GetValues(Node * parent, const char * name)
            {
                if (parent == NULL)
                    SIMD_EX("Invalid element!");
                return GetValues<T>(parent->FirstNode(name));
            }

            inline size_t GetSize(Node * parent)
            {
                return Simd::Xml::CountChildren(parent);
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

        void * DetectionLoadStringXml(char * xml, const char * path)
        {
            static const float THRESHOLD_EPS = 1e-5f;

            Data * data = NULL;
            try
            {
                Xml::Document doc;
                doc.Parse<0>(xml);

                Xml::Node * root = doc.FirstNode();
                if (root == NULL) {
                    if (path == NULL) {
                        SIMD_EX("Invalid format of XML string!");
                    }
                    else {
                        SIMD_EX("Invalid format of XML file '" << path << "'!");
                    }
                }

                Xml::Node * cascade = root->FirstNode(Names::cascade);
                if (cascade == NULL)
                    return data;

                data = new Data();

                if (Xml::GetValue<std::string>(cascade, Names::stageType) != Names::BOOST)
                    SIMD_EX("Invalid cascade stage type!");
                data->stageType = 0;

                std::string featureType = Xml::GetValue<std::string>(cascade, Names::featureType);
                if (featureType == Names::HAAR)
                    data->featureType = SimdDetectionInfoFeatureHaar;
                else if (featureType == Names::LBP)
                    data->featureType = SimdDetectionInfoFeatureLbp;
                else if (featureType == Names::HOG)
                    SIMD_EX("HOG feature type is not supported!")
                else
                    SIMD_EX("Invalid cascade feature type!");

                data->origWinSize.x = Xml::GetValue<int>(cascade, Names::width);
                data->origWinSize.y = Xml::GetValue<int>(cascade, Names::height);
                if (data->origWinSize.x <= 0 || data->origWinSize.y <= 0)
                    SIMD_EX("Invalid cascade width or height!");

                Xml::Node * stageParams = cascade->FirstNode(Names::stageParams);
                if (stageParams && stageParams->FirstNode(Names::maxDepth))
                    data->isStumpBased = Xml::GetValue<int>(stageParams, Names::maxDepth) == 1 ? true : false;
                else
                    data->isStumpBased = true;

                if (!data->isStumpBased)
                    SIMD_EX("Tree classifier cascades are not supported!");

                Xml::Node * featureParams = cascade->FirstNode(Names::featureParams);
                data->ncategories = Xml::GetValue<int>(featureParams, Names::maxCatCount);
                int subsetSize = (data->ncategories + 31) / 32;
                int nodeStep = 3 + (data->ncategories > 0 ? subsetSize : 1);

                Xml::Node * stages = cascade->FirstNode(Names::stages);
                if (stages == NULL)
                    SIMD_EX("Invalid stages count!");
                data->stages.reserve(Xml::GetSize(stages));
                int stageIndex = 0;
                for (Xml::Node * stageNode = stages->FirstNode(); stageNode != NULL; stageNode = stageNode->NextSibling(), ++stageIndex)
                {
                    Data::Stage stage;
                    stage.threshold = Xml::GetValue<float>(stageNode, Names::stageThreshold) - THRESHOLD_EPS;

                    Xml::Node * weakClassifiers = stageNode->FirstNode(Names::weakClassifiers);
                    if (weakClassifiers == NULL)
                        SIMD_EX("Invalid weak classifiers count!");
                    stage.ntrees = (int)Xml::GetSize(weakClassifiers);
                    stage.first = (int)data->classifiers.size();
                    data->stages.push_back(stage);
                    data->classifiers.reserve(data->stages[stageIndex].first + data->stages[stageIndex].ntrees);

                    for (Xml::Node * weakClassifier = weakClassifiers->FirstNode(); weakClassifier != NULL; weakClassifier = weakClassifier->NextSibling())
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

                Xml::Node * featureNodes = cascade->FirstNode(Names::features);
                if (data->featureType == SimdDetectionInfoFeatureHaar)
                {
                    data->hasTilted = false;
                    data->haarFeatures.reserve(Xml::GetSize(featureNodes));
                    for (Xml::Node * featureNode = featureNodes->FirstNode(); featureNode != NULL; featureNode = featureNode->NextSibling())
                    {
                        Data::HaarFeature feature;
                        int rectIndex = 0;
                        Xml::Node * rectsNode = featureNode->FirstNode(Names::rects);
                        for (Xml::Node * rectNode = rectsNode->FirstNode(); rectNode != NULL; rectNode = rectNode->NextSibling(), rectIndex++)
                        {
                            std::vector<double> values = Xml::GetValues<double>(rectNode);
                            feature.rect[rectIndex].r.x = (int)values[0];
                            feature.rect[rectIndex].r.y = (int)values[1];
                            feature.rect[rectIndex].r.width = (int)values[2];
                            feature.rect[rectIndex].r.height = (int)values[3];
                            feature.rect[rectIndex].weight = (float)values[4];
                        }
                        feature.tilted = featureNode->FirstNode(Names::tilted) && Xml::GetValue<int>(featureNode, Names::tilted) != 0;
                        if (feature.tilted)
                            data->hasTilted = true;
                        data->haarFeatures.push_back(feature);
                    }
                }

                if (data->featureType == SimdDetectionInfoFeatureLbp)
                {
                    data->canInt16 = true;
                    data->lbpFeatures.reserve(Xml::GetSize(featureNodes));
                    for (Xml::Node * featureNode = featureNodes->FirstNode(); featureNode != NULL; featureNode = featureNode->NextSibling())
                    {
                        Data::LbpFeature feature;
                        std::vector<int> values = Xml::GetValues<int>(featureNode, Names::rect);
                        feature.rect.x = values[0];
                        feature.rect.y = values[1];
                        feature.rect.width = values[2];
                        feature.rect.height = values[3];
                        if (feature.rect.width*feature.rect.height > 256)
                            data->canInt16 = false;
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

        void * DetectionLoadA(const char * path)
        {
            Xml::File file;
            if (!file.Open(path))
            {
                SIMD_LOG_ERROR("Can't load XML file '" << path << "'!");
                return NULL;
            }

            return DetectionLoadStringXml(file.Data(), path);
        }

        void DetectionInfo(const void * _data, size_t * width, size_t * height, SimdDetectionInfoFlags * flags)
        {
            Data * data = (Data*)_data;
            if (data)
            {
                if (width)
                    *width = data->origWinSize.x;
                if (height)
                    *height = data->origWinSize.y;
                if (flags)
                    *flags = SimdDetectionInfoFlags(data->featureType |
                    (data->hasTilted ? SimdDetectionInfoHasTilted : 0) |
                        (data->canInt16 ? SimdDetectionInfoCanInt16 : 0));
            }
        }

        HidHaarCascade * CreateHidHaar(const Data & data)
        {
            if (data.featureType != SimdDetectionInfoFeatureHaar)
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

        template <class T> SIMD_INLINE T * SumElemPtr(const Image & view, ptrdiff_t row, ptrdiff_t col, bool throughColumn)
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

        static void InitBase(HidHaarCascade * hid, const Image & sum, const Image & sqsum, const Image & tilted)
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
            Image sum = hid->isThroughColumn ? hid->isum : hid->sum;
            Image tilted = hid->isThroughColumn ? hid->itilted : hid->tilted;
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

        HidHaarCascade * InitHaar(const Data & data, const Image & sum, const Image & sqsum, const Image & tilted, bool throughColumn)
        {
            if (!data.isStumpBased)
                SIMD_EX("Can't use tree classfier for vector haar classifier!");

            HidHaarCascade * hid = CreateHidHaar(data);
            InitBase(hid, sum, sqsum, tilted);
            if (throughColumn)
            {
                hid->isThroughColumn = true;
                hid->isum.Recreate(sum.width, sum.height, Image::Int32, NULL, Image::PixelSize(Image::Int32));
                if (hid->hasTilted)
                    hid->itilted.Recreate(tilted.width, tilted.height, Image::Int32, NULL, Image::PixelSize(Image::Int32));
            }
            UpdateFeaturePtrs<uint32_t>(hid, data);
            return hid;
        }

        template<class T> void InitLbp(const Data & data, size_t index, HidLbpStage<T> * stages, T * leaves);

        template<> void InitLbp<float>(const Data & data, size_t index, HidLbpStage<float> * stages, float * leaves)
        {
            stages[index].first = data.stages[index].first;
            stages[index].ntrees = data.stages[index].ntrees;
            stages[index].threshold = data.stages[index].threshold;
            for (int i = stages[index].first * 2, n = (stages[index].first + stages[index].ntrees) * 2; i < n; ++i)
                leaves[i] = data.leaves[i];
        }

        template<> void InitLbp<int>(const Data & data, size_t index, HidLbpStage<int> * stages, int * leaves)
        {
            float min = 0, max = 0;
            for (int i = 0; i < data.stages[index].ntrees; ++i)
            {
                const float * leave = data.leaves.data() + (data.stages[index].first + i) * 2;
                min += std::min(leave[0], leave[1]);
                max += std::max(leave[0], leave[1]);
            }
            float k = float(SHRT_MAX)*0.9f / Simd::Max(Simd::Abs(min), Simd::Abs(max));

            stages[index].first = data.stages[index].first;
            stages[index].ntrees = data.stages[index].ntrees;
            stages[index].threshold = Simd::Round(data.stages[index].threshold*k);
            for (int i = stages[index].first * 2, n = (stages[index].first + stages[index].ntrees) * 2; i < n; ++i)
                leaves[i] = Simd::Round(data.leaves[i] * k);
#if 0
            std::cout
                << "stage = " << index
                << "; ntrees = " << data.stages[index].ntrees
                << "; threshold = " << data.stages[index].threshold
                << "; min = " << min
                << "; max = " << max
                << "; k = " << k
                << "." << std::endl;
#endif
        }

        template<class TWeight, class TSum> HidLbpCascade<TWeight, TSum> * CreateHidLbp(const Data & data)
        {
            HidLbpCascade<TWeight, TSum> * hid = new HidLbpCascade<TWeight, TSum>();

            hid->isInt16 = (sizeof(TSum) == 2);
            hid->isThroughColumn = false;

            hid->isStumpBased = data.isStumpBased;
            //hid->stageType = data.stageType;
            hid->featureType = data.featureType;
            hid->ncategories = data.ncategories;
            hid->origWinSize = data.origWinSize;

            hid->trees.resize(data.classifiers.size());
            for (size_t i = 0; i < data.classifiers.size(); ++i)
            {
                hid->trees[i].nodeCount = data.classifiers[i].nodeCount;
            }

            hid->nodes.resize(data.nodes.size());
            for (size_t i = 0; i < data.nodes.size(); ++i)
            {
                hid->nodes[i].featureIdx = data.nodes[i].featureIdx;
                hid->nodes[i].left = data.nodes[i].left;
                hid->nodes[i].right = data.nodes[i].right;
            }

            hid->stages.resize(data.stages.size());
            hid->leaves.resize(data.leaves.size());
            for (size_t i = 0; i < data.stages.size(); ++i)
            {
                InitLbp(data, i, hid->stages.data(), hid->leaves.data());
            }

            hid->subsets.resize(data.subsets.size());
            for (size_t i = 0; i < data.subsets.size(); ++i)
            {
                hid->subsets[i] = data.subsets[i];
            }

            hid->features.resize(data.lbpFeatures.size());
            for (size_t i = 0; i < hid->features.size(); ++i)
            {
                hid->features[i].rect.left = data.lbpFeatures[i].rect.x;
                hid->features[i].rect.top = data.lbpFeatures[i].rect.y;
                hid->features[i].rect.right = data.lbpFeatures[i].rect.x + data.lbpFeatures[i].rect.width;
                hid->features[i].rect.bottom = data.lbpFeatures[i].rect.y + data.lbpFeatures[i].rect.height;
            }

            return hid;
        }

        template<class TLeave, class TSum> SIMD_INLINE void UpdateFeaturePtrs(HidLbpCascade<TLeave, TSum> * hid)
        {
            Image sum = (hid->isThroughColumn || hid->isInt16) ? hid->isum : hid->sum;
            for (size_t i = 0; i < hid->features.size(); i++)
            {
                typename HidLbpCascade<TLeave, TSum>::Feature& feature = hid->features[i];
                for (size_t row = 0; row < 4; ++row)
                {
                    for (size_t col = 0; col < 4; ++col)
                    {
                        feature.p[row * 4 + col] = SumElemPtr<TSum>(sum,
                            feature.rect.top + feature.rect.Height()*row,
                            feature.rect.left + feature.rect.Width()*col, hid->isThroughColumn);
                    }
                }
            }
        }

        HidBase * InitLbp(const Data & data, const Image & sum, bool throughColumn, bool int16)
        {
            assert(sum.format == Image::Int32);
            if (int16 && data.canInt16)
            {
                HidLbpCascade<int, short> * hid = CreateHidLbp<int, short>(data);
                hid->isThroughColumn = throughColumn;
                hid->sum = sum;
                hid->isum.Recreate(sum.Size(), Image::Int16);
                UpdateFeaturePtrs(hid);
                return hid;
            }
            else
            {
                HidLbpCascade<float, int> * hid = CreateHidLbp<float, int>(data);
                hid->isThroughColumn = throughColumn;
                hid->sum = sum;
                if (throughColumn)
                    hid->isum.Recreate(sum.Size(), Image::Int32);
                UpdateFeaturePtrs(hid);
                return hid;
            }
        }

        void * DetectionInit(const void * _data, uint8_t * sum, size_t sumStride, size_t width, size_t height,
            uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride, int throughColumn, int int16)
        {
            Data & data = *(Data*)_data;
            switch (data.featureType)
            {
            case SimdDetectionInfoFeatureHaar:
                return InitHaar(data,
                    Image(width, height, sumStride, Image::Int32, sum),
                    Image(width, height, sqsumStride, Image::Int32, sqsum),
                    Image(width, height, tiltedStride, Image::Int32, tilted),
                    throughColumn != 0);
            case SimdDetectionInfoFeatureLbp:
                return InitLbp(data,
                    Image(width, height, sumStride, Image::Int32, sum),
                    throughColumn != 0,
                    int16 != 0);
            default:
                return NULL;
            }
        }

        void PrepareThroughColumn32i(const Image & src, Image & dst)
        {
            assert(Simd::Compatible(src, dst) && src.format == Image::Int32);

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

        void Prepare16i(const Image & src, bool throughColumn, Image & dst)
        {
            assert(Simd::EqualSize(src, dst) && src.format == Image::Int32 && dst.format == Image::Int16);

            if (throughColumn)
            {
                for (size_t row = 0; row < src.height; ++row)
                {
                    const uint32_t * s = &src.At<uint32_t>(0, row);

                    uint16_t * evenDst = &dst.At<uint16_t>(0, row);
                    for (size_t col = 0; col < src.width; col += 2)
                        evenDst[col >> 1] = (uint16_t)s[col];

                    uint16_t * oddDst = &dst.At<uint16_t>((dst.width + 1) >> 1, row);
                    for (size_t col = 1; col < src.width; col += 2)
                        oddDst[col >> 1] = (uint16_t)s[col];
                }
            }
            else
            {
                for (size_t row = 0; row < src.height; ++row)
                {
                    const uint32_t * s = &src.At<uint32_t>(0, row);
                    uint16_t * d = &dst.At<uint16_t>(0, row);
                    for (size_t col = 0; col < src.width; ++col)
                        d[col] = (uint16_t)s[col];
                }
            }
        }

        void DetectionPrepare(void * _hid)
        {
            HidBase * hidBase = (HidBase*)_hid;
            if (hidBase->featureType == SimdDetectionInfoFeatureHaar && hidBase->isThroughColumn)
            {
                HidHaarCascade * hid = (HidHaarCascade*)hidBase;
                PrepareThroughColumn32i(hid->sum, hid->isum);
                if (hid->hasTilted)
                    PrepareThroughColumn32i(hid->tilted, hid->itilted);
            }
            else if (hidBase->featureType == SimdDetectionInfoFeatureLbp)
            {
                if (hidBase->isInt16)
                {
                    HidLbpCascade<int, short> * hid = (HidLbpCascade<int, short>*)hidBase;
                    Prepare16i(hid->sum, hid->isThroughColumn, hid->isum);
                }
                else if (hidBase->isThroughColumn)
                {
                    HidLbpCascade<float, int> * hid = (HidLbpCascade<float, int>*)hidBase;
                    PrepareThroughColumn32i(hid->sum, hid->isum);
                }
            }
        }

        int Detect32f(const HidHaarCascade & hid, size_t offset, int startStage, float norm)
        {
            typedef HidHaarCascade Hid;
            const Hid::Stage * stages = hid.stages.data();
            if (startStage >= (int)hid.stages.size())
                return 1;
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

        void DetectionHaarDetect32fp(const HidHaarCascade & hid, const Image & mask, const Rect & rect, Image & dst)
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
                    if (Detect32f(hid, p_offset + col, 0, norm) > 0)
                        dst.At<uint8_t>(col, row) = 1;
                }
            }
        }

        void DetectionHaarDetect32fp(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidHaarCascade & hid = *(HidHaarCascade*)_hid;
            return DetectionHaarDetect32fp(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        void DetectionHaarDetect32fi(const HidHaarCascade & hid, const Image & mask, const Rect & rect, Image & dst)
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
                    if (Detect32f(hid, p_offset + col / 2, 0, norm) > 0)
                        dst.At<uint8_t>(col, row) = 1;
                }
            }
        }

        void DetectionHaarDetect32fi(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidHaarCascade & hid = *(HidHaarCascade*)_hid;
            return DetectionHaarDetect32fi(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        void DetectionLbpDetect32fp(const HidLbpCascade<float, uint32_t> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t offset = row * hid.sum.stride / sizeof(int);
                for (ptrdiff_t col = rect.left; col < rect.right; col += 1)
                {
                    if (mask.At<uint8_t>(col, row) == 0)
                        continue;
                    if (Detect(hid, offset + col, 0) > 0)
                        dst.At<uint8_t>(col, row) = 1;
                }
            }
        }

        void DetectionLbpDetect32fp(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidLbpCascade<float, uint32_t> & hid = *(HidLbpCascade<float, uint32_t>*)_hid;
            return DetectionLbpDetect32fp(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        void DetectionLbpDetect32fi(const HidLbpCascade<float, uint32_t> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 2)
            {
                size_t offset = row * hid.isum.stride / sizeof(int);
                for (ptrdiff_t col = rect.left; col < rect.right; col += 2)
                {
                    if (mask.At<uint8_t>(col, row) == 0)
                        continue;
                    if (Detect(hid, offset + col / 2, 0) > 0)
                        dst.At<uint8_t>(col, row) = 1;
                }
            }
        }

        void DetectionLbpDetect32fi(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidLbpCascade<float, uint32_t> & hid = *(HidLbpCascade<float, uint32_t>*)_hid;
            return DetectionLbpDetect32fi(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        void DetectionLbpDetect16ip(const HidLbpCascade<int, uint16_t> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 1)
            {
                size_t offset = row * hid.isum.stride / sizeof(uint16_t);
                for (ptrdiff_t col = rect.left; col < rect.right; col += 1)
                {
                    if (mask.At<uint8_t>(col, row) == 0)
                        continue;
                    if (Detect(hid, offset + col, 0) > 0)
                        dst.At<uint8_t>(col, row) = 1;
                }
            }
        }

        void DetectionLbpDetect16ip(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidLbpCascade<int, uint16_t> & hid = *(HidLbpCascade<int, uint16_t>*)_hid;
            return DetectionLbpDetect16ip(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }

        void DetectionLbpDetect16ii(const HidLbpCascade<int, uint16_t> & hid, const Image & mask, const Rect & rect, Image & dst)
        {
            for (ptrdiff_t row = rect.top; row < rect.bottom; row += 2)
            {
                size_t offset = row * hid.isum.stride / sizeof(uint16_t);
                for (ptrdiff_t col = rect.left; col < rect.right; col += 2)
                {
                    if (mask.At<uint8_t>(col, row) == 0)
                        continue;
                    if (Detect(hid, offset + col / 2, 0) > 0)
                        dst.At<uint8_t>(col, row) = 1;
                }
            }
        }

        void DetectionLbpDetect16ii(const void * _hid, const uint8_t * mask, size_t maskStride,
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {
            const HidLbpCascade<int, uint16_t> & hid = *(HidLbpCascade<int, uint16_t>*)_hid;
            return DetectionLbpDetect16ii(hid,
                Image(hid.sum.width - 1, hid.sum.height - 1, maskStride, Image::Gray8, (uint8_t*)mask),
                Rect(left, top, right, bottom),
                Image(hid.sum.width - 1, hid.sum.height - 1, dstStride, Image::Gray8, dst).Ref());
        }
    }
}
