/*
* Tests for Simd Library (http://simd.sourceforge.net).
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

/*                                                                                       
* File name   : SimdCheckCpp.cpp                                                       
* Description : This file is needed to verify the C++ API of Simd Library.               
*/

#include "Simd/SimdLib.hpp"
#include "Simd/SimdFrame.hpp"
#include "Simd/SimdPyramid.hpp"
#include "Simd/SimdDetection.hpp"

#include "Test/TestUtils.h"

namespace Test
{
    typedef Simd::View<Simd::Allocator> View;
    typedef Simd::Frame<Simd::Allocator> Frame;
    typedef Simd::Pyramid<Simd::Allocator> Pyramid;
    typedef Simd::Detection<Simd::Allocator> Detection;
    typedef Detection::Size Size;
    typedef Detection::Objects Objects;

    static void TestView()
    {
        View vs(6, 6, View::Bgra32);
        View vd(6, 6, View::Gray8);
        Simd::Convert(vs, vd);
    }

    static void TestFrame()
    {
        Frame fs(2, 2, Frame::Yuv420p);
        Frame fd(2, 2, Frame::Bgr24);
        Simd::Convert(fs, fd);
    }

    static void TestPyramid()
    {
        Pyramid p(16, 16, 3);
        p.Fill(1);
        p.Build(Pyramid::ReduceGray2x2);
    }

    View GetSample(const Size & size, bool large);

    static void TestDetection()
    {
        Detection detection;

        detection.Load("../../data/cascade/haar_face_0.xml", 0);
        detection.Load("../../data/cascade/haar_face_1.xml", 0);
        detection.Load("../../data/cascade/lbp_face.xml", 0);

        View src = GetSample(Size(W, H), true);
        detection.Init(src.Size(), 1.2);

        Objects objects;
        detection.Detect(src, objects);

        for (size_t i = 0; i < objects.size(); ++i)
        {
            Size s = objects[i].rect.Size();
            Simd::FillFrame(src.Region(objects[i].rect).Ref(), Rect(1, 1, s.x - 1, s.y - 1), 255);
        }
        Save(src, "dst.pgm");
    }

    void CheckCpp()
    {
        TestView();

        TestFrame();

        TestPyramid();

        TestDetection();

        exit(0);
	}
}


