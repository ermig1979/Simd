/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
        std::cout << "Simd Library : " << SimdVersion() << std::endl;
        std::cout << "Sockets : " << SimdCpuInfo(SimdCpuInfoSockets) << std::endl;
        std::cout << "Cores : " << SimdCpuInfo(SimdCpuInfoCores) << std::endl;
        std::cout << "Threads : " << SimdCpuInfo(SimdCpuInfoThreads) << std::endl;
        std::cout << "L1D Cache : " << SimdCpuInfo(SimdCpuInfoCacheL1) / 1024 << " KB" << std::endl;
        std::cout << "L2 Cache : " << SimdCpuInfo(SimdCpuInfoCacheL2) / 1024 << " KB" << std::endl;
        std::cout << "L3 Cache : " << SimdCpuInfo(SimdCpuInfoCacheL3) / 1024 << " KB" << std::endl;
        std::cout << "SSE2: " << (SimdCpuInfo(SimdCpuInfoSse2) ? "Yes" : "No") << std::endl;
        std::cout << "SSE4.1: " << (SimdCpuInfo(SimdCpuInfoSse41) ? "Yes" : "No") << std::endl;
        std::cout << "AVX: " << (SimdCpuInfo(SimdCpuInfoAvx) ? "Yes" : "No") << std::endl;
        std::cout << "AVX2: " << (SimdCpuInfo(SimdCpuInfoAvx2) ? "Yes" : "No") << std::endl;
        std::cout << "AVX-512F: " << (SimdCpuInfo(SimdCpuInfoAvx512f) ? "Yes" : "No") << std::endl;
        std::cout << "AVX-512BW: " << (SimdCpuInfo(SimdCpuInfoAvx512bw) ? "Yes" : "No") << std::endl;
        std::cout << "AVX-512VNNI: " << (SimdCpuInfo(SimdCpuInfoAvx512vnni) ? "Yes" : "No") << std::endl;
        std::cout << "PowerPC-Altivec: " << (SimdCpuInfo(SimdCpuInfoVmx) ? "Yes" : "No") << std::endl;
        std::cout << "PowerPC-VSX: " << (SimdCpuInfo(SimdCpuInfoVsx) ? "Yes" : "No") << std::endl;
        std::cout << "ARM-NEON: " << (SimdCpuInfo(SimdCpuInfoNeon) ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
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

    static void TestResize()
    {
        typedef Simd::View<Simd::Allocator> View;
        typedef Simd::Point<ptrdiff_t> Size;

        View src(128, 96, View::Bgr24), dst(40, 30, View::Bgr24);
        Simd::Resize(src, dst, SimdResizeMethodArea);
        Simd::Resize(dst, dst, Size(80, 60), SimdResizeMethodArea);
    }

    static void TestViewVector()
    {
        typedef Simd::View<Simd::Allocator> View;
        typedef std::vector<View> Views;

        Views views;
        for (size_t i = 0; i < 10; ++i)
        {
            views.push_back(View(128 + i, 96 + i, View::Gray8));
            views[i].data[i] = uint8_t(i);
        }
    }

    static void TestViewMove()
    {
        typedef Simd::View<Simd::Allocator> View;

        View a = View(128, 96, View::Gray8), b(40, 30, View::Bgr24);

        b = std::move(a);
    }

    static void TestFrameVector()
    {
        typedef Simd::View<Simd::Allocator> View;
        typedef Simd::Frame<Simd::Allocator> Frame;
        typedef std::vector<Frame> Frames;

        Frames frames;
        for (size_t i = 0; i < 10; ++i)
            frames.push_back(Frame(View(128 + i, 96 + i, View::Gray8), false, i * 0.040));
    }

    static void TestFrameMove()
    {
        typedef Simd::View<Simd::Allocator> View;
        typedef Simd::Frame<Simd::Allocator> Frame;

        Frame a = Frame(View(128, 96, View::Gray8)), b(View(40, 30, View::Bgr24));

        b = std::move(a);
    }

    void CheckCpp()
    {
        TestCpuInfo();

        TestPoint();

        TestRectangle();

        TestView();

        TestFrame();

        TestPyramid();

        TestStdVector();

        TestResize();

        TestViewVector();

        TestViewMove();

        TestFrameVector();

        TestFrameMove();
    }
}


