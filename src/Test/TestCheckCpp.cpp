/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
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

/*
* File name   : SimdCheckCpp.cpp
* Description : This file is needed to verify the C++ API of Simd Library.
*/

//#define SIMD_OPENCV_ENABLE

#include "Simd/SimdLib.hpp"
#include "Simd/SimdFrame.hpp"
#include "Simd/SimdPyramid.hpp"

#include <iostream>

namespace Test
{
    static void TestCpuInfo()
    {
        int info = ::SimdCpuInfo();
        std::cout << "SSE: " << (info&(1 << SimdCpuInfoSse) ? "Yes" : "No") << std::endl;
        std::cout << "SSE2: " << (info&(1 << SimdCpuInfoSse2) ? "Yes" : "No") << std::endl;
        std::cout << "SSE3: " << (info&(1 << SimdCpuInfoSse3) ? "Yes" : "No") << std::endl;
        std::cout << "SSSE3: " << (info&(1 << SimdCpuInfoSsse3) ? "Yes" : "No") << std::endl;
        std::cout << "SSE4.1: " << (info&(1 << SimdCpuInfoSse41) ? "Yes" : "No") << std::endl;
        std::cout << "SSE4.2: " << (info&(1 << SimdCpuInfoSse42) ? "Yes" : "No") << std::endl;
        std::cout << "AVX: " << (info&(1 << SimdCpuInfoAvx) ? "Yes" : "No") << std::endl;
        std::cout << "AVX2: " << (info&(1 << SimdCpuInfoAvx2) ? "Yes" : "No") << std::endl;
        std::cout << "AVX-512F: " << (info&(1 << SimdCpuInfoAvx512f) ? "Yes" : "No") << std::endl;
        std::cout << "AVX-512BW: " << (info&(1 << SimdCpuInfoAvx512bw) ? "Yes" : "No") << std::endl;
        std::cout << "PowerPC-Altivec: " << (info&(1 << SimdCpuInfoVmx) ? "Yes" : "No") << std::endl;
        std::cout << "PowerPC-VSX: " << (info&(1 << SimdCpuInfoVsx) ? "Yes" : "No") << std::endl;
        std::cout << "ARM-NEON: " << (info&(1 << SimdCpuInfoNeon) ? "Yes" : "No") << std::endl;
        std::cout << "MIPS-MSA: " << (info&(1 << SimdCpuInfoMsa) ? "Yes" : "No") << std::endl;
    }

    static void TestPoint()
    {
        typedef Simd::Point<ptrdiff_t> Point;
        typedef Simd::Point<double> FPoint;

        Point p(1.4, 2.6);
        FPoint fp(1.4, 3.6);
    }

    static void TestRectangle()
    {
        typedef Simd::Point<ptrdiff_t> Point;
        typedef Simd::Rectangle<ptrdiff_t> Rect;

        Rect r1(0, 0, 100, 100), r2(10, 10, 90, 90);
        Point p(50, 50);
        r1 &= r2;
        r1 &= p;
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
        Pyramid p1(16, 16, 3), p2(16, 16, 3);
        Fill(p1, 1);
        Build(p1, ::SimdReduce2x2);
        Simd::Copy(p1, p2);
    }

    static void TestStdVector()
    {
        typedef std::vector<float, Simd::Allocator<float> > Vector;
        Vector v(16, 1.0f);
        v[15] = 0.0f;
    }

    void CheckCpp()
    {
        //TestCpuInfo();

        TestPoint();

        TestRectangle();

        TestView();

        TestFrame();

        TestPyramid();

        TestStdVector();
    }
}


