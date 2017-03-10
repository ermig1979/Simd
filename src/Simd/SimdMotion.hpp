/*
* Simd Library (http://simd.sourceforge.net).
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

#include <vector>

namespace Simd
{
    namespace Motion
    {
        typedef Simd::Point<ptrdiff_t> Size;
        typedef Simd::Point<ptrdiff_t> Point;
        typedef std::vector<Point> Points;
        typedef Simd::Rectangle<ptrdiff_t> Rect;
        typedef Simd::View<Simd::Allocator> View; 
        typedef Simd::Frame<Simd::Allocator> Frame;
        typedef Simd::Pyramid<Simd::Allocator> Pyramid;

        struct Position
        {
            Point point;
            double time;
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

            Options()
            {
                TextureGradientSaturation = 16;
                TextureGradientBoost = 4;
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
                if (output && output->Size() != input.Size())
                    return false;

                if (_scene.model.size != input.Size())
                {
                    _model = Model(input.Size());
                    if(!Calibrate(_model))
                        return false;
                }

                SetFrame(input, output);

                EstimateTextures();

                return true;
            }

        private:
            Options _options;
            Model _model;

            struct Texture
            {
                struct Bound 
                {
                    Pyramid value; 
                    Pyramid count; 

                    void Create(const Size & size, int levelCount)
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

                    void Create(const Size & size, int levelCount)
                    {
                        value.Recreate(size, levelCount);
                        lo.Create(size, levelCount);
                        hi.Create(size, levelCount);
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

                void Create(const Size & size, int levelCount)
                {
                    features.clear();
                    gray.Create(size, levelCount);
                    features.push_back(&gray);
                    dx.Create(size, levelCount);
                    features.push_back(&dx);
                    dy.Create(size, levelCount);
                    features.push_back(&dy);
                }
            };

            struct Scene
            {
                Frame input, *output;
                View gray;

                Model model;

                Texture texture;

                void Create(const Model & m)
                {
                    model = m;
                    gray.Recreate(model.size, View::Gray8);
                    texture.Create(model.size, 3);
                }
            };

            bool SetFrame(const Frame & input, Frame * output)
            {
                _scene.input = input;
                _scene.output = output;
                Simd::Convert(input, Frame(_scene.gray).Ref());

                return true;
            }

            bool Calibrate(const Model & model)
            {
                _scene.Create(model);

                return true;
            }

            bool EstimateTextures()
            {
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

            Scene _scene;
        };
    }
}

#endif//__SimdMotion_hpp__
