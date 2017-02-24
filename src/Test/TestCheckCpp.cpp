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

//#define SIMD_OPENCV_ENABLE

#include "Simd/SimdLib.hpp"
#include "Simd/SimdFrame.hpp"
#include "Simd/SimdPyramid.hpp"

namespace Test
{
    static void TestPoint()
    {
        typedef Simd::Point<ptrdiff_t> Point;
        typedef Simd::Point<double> FPoint;

        Point p(1.4, 2.6);
        FPoint fp(1.4, 3.6);
    }

    static void TestView()
    {
        typedef Simd::View<Simd::Allocator> View;
        View vs(6, 6, View::Bgra32);
        View vd(6, 6, View::Gray8);
        Simd::Convert(vs, vd);

        View sv;
        sv = vs;
#ifdef SIMD_OPENCV_ENABLE
        cv::Mat cm;
        sv = cm;
        cm = sv;
#endif
        sv.Swap(vs);
    }

    static void TestFrame()
    {
        typedef Simd::Frame<Simd::Allocator> Frame;
        Frame fs(2, 2, Frame::Yuv420p);
        Frame fd(2, 2, Frame::Bgr24);
        Simd::Convert(fs, fd);
    }

    static void TestPyramid()
    {
        typedef Simd::Pyramid<Simd::Allocator> Pyramid;
        Pyramid p(16, 16, 3);
        Fill(p, 1);
        Build(p, ::SimdReduce2x2);
    }

    static void TestStdVector()
    {
        typedef std::vector<float, Simd::Allocator<float> > Vector;
        Vector v(16, 1.0f);
        v[15] = 0.0f;
    }

    static void CheckCpp()
    {
        TestPoint();

        TestView();

        TestFrame();

        TestPyramid();

        TestStdVector();
	}
}


