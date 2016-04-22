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
        using namespace Detection;

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

        void * DetectionHaarInit(const void * data, uint8_t * sum, size_t sumStride, size_t width, size_t height,
            uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride, int throughColumn)
        {
            return NULL;
        }

        void DetectionHaarFree(void * hid)
        {

        }
        
        int DetectionHaarHasTilted(const void * hid)
        {
            return 0;
        }

        void DetectionHaarPrepare(void * hid)
        {

        }

        void DetectionHaarDetect32fp(const void * hid, const uint8_t * mask, size_t maskStride, 
            ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride)
        {

        }
    }
}

#endif