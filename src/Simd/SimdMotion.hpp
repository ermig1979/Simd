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
                return true;
            }

            bool NextFrame(const Frame & input, Metadata & metadata, Frame * output = NULL)
            {
                return true;
            }

        private:
            Options _options;
            Model _model;

            struct Scene
            {
                Model model;
            };

            Scene _scene;
        };
    }
}

#endif//__SimdMotion_hpp__
