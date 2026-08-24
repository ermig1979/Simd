/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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

#if defined(__GNUC__) && defined(__MINGW32__) && !defined(SIMD_STATIC)
#define SIMD_STATIC
#endif

#include "Simd/SimdLib.hpp"
#include "Simd/SimdFrame.hpp"
#include "Simd/SimdPyramid.hpp"
#include "Simd/SimdSynet.hpp"

#include <iostream>
#include <vector>

namespace Test
{
    static void TestCpuInfo()
    {
        std::cout << "Simd Library : " << SimdVersion() << std::endl;
        std::cout << "CPU : " << SimdCpuDesc(SimdCpuDescModel) << std::endl;
        std::cout << "Sockets : " << SimdCpuInfo(SimdCpuInfoSockets) << std::endl;
        std::cout << "Cores : " << SimdCpuInfo(SimdCpuInfoCores) << std::endl;
        std::cout << "Threads : " << SimdCpuInfo(SimdCpuInfoThreads) << std::endl;
        std::cout << "L1D Cache : " << SimdCpuInfo(SimdCpuInfoCacheL1) / 1024 << " KB" << std::endl;
        std::cout << "L2 Cache : " << SimdCpuInfo(SimdCpuInfoCacheL2) / 1024 << " KB" << std::endl;
        std::cout << "L3 Cache : " << SimdCpuInfo(SimdCpuInfoCacheL3) / 1024 << " KB" << std::endl;
        std::cout << "RAM : " << SimdCpuInfo(SimdCpuInfoRam) / 1024 / 1024  << " MB" << std::endl;
        std::cout << "SSE4.1: " << (SimdCpuInfo(SimdCpuInfoSse41) ? "Yes" : "No") << std::endl;
        std::cout << "AVX2: " << (SimdCpuInfo(SimdCpuInfoAvx2) ? "Yes" : "No") << std::endl;
        std::cout << "AVX-512BW: " << (SimdCpuInfo(SimdCpuInfoAvx512bw) ? "Yes" : "No") << std::endl;
        std::cout << "AVX-512VNNI: " << (SimdCpuInfo(SimdCpuInfoAvx512vnni) ? "Yes" : "No") << std::endl;
        std::cout << "AMX-BF16: " << (SimdCpuInfo(SimdCpuInfoAmxBf16) ? "Yes" : "No") << std::endl;
        std::cout << "ARM-NEON: " << (SimdCpuInfo(SimdCpuInfoNeon) ? "Yes" : "No") << std::endl;
        std::cout << "ARM-SVE size: " << SimdCpuInfo(SimdCpuInfoSveSize) * 8 << " bit" << std::endl;
        std::cout << "ARM-SVE2: " << (SimdCpuInfo(SimdCpuInfoSve2) ? "Yes" : "No") << std::endl;
        std::cout << "Hexagon-HVX: " << (SimdCpuInfo(SimdCpuInfoHvx) ? "Yes" : "No") << std::endl;
        std::cout << "Current Frequency: " << SimdCpuInfo(SimdCpuInfoCurrentFrequency) / 1000 / 1000 << " MHz" << std::endl;
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

        View cp = sv;
        cp.Capture();
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

    static void TestImageResize()
    {
        typedef Simd::View<Simd::Allocator> View;
        typedef Simd::Point<ptrdiff_t> Size;

        View src(128, 96, View::Bgr24), dst(40, 30, View::Bgr24);
        Simd::Resize(src, dst, SimdResizeMethodArea);
        Simd::Resize(dst, dst, Size(80, 60), SimdResizeMethodArea);
    }

    static void TestFrameResize()
    {
        typedef Simd::Frame<Simd::Allocator> Frame;
        typedef Simd::Point<ptrdiff_t> Size;

        Frame src(128, 96, Frame::Yuv420p), dst(40, 30, Frame::Yuv420p);
        Simd::Resize(src, dst, SimdResizeMethodArea);
        Simd::Resize(dst, dst, Size(80, 60), SimdResizeMethodBilinear);
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

    static void TestFrameVector()
    {
        typedef Simd::View<Simd::Allocator> View;
        typedef Simd::Frame<Simd::Allocator> Frame;
        typedef std::vector<Frame> Frames;

        Frames frames;
        for (size_t i = 0; i < 10; ++i)
            frames.push_back(Frame(View(128 + i, 96 + i, View::Gray8), false, i * 0.040));
    }

#if defined(SIMD_CPP_2011_ENABLE)
    static void TestViewMove()
    {
        typedef Simd::View<Simd::Allocator> View;

        View a = View(128, 96, View::Gray8), b(40, 30, View::Bgr24);

        b = std::move(a);
    }

    static void TestFrameMove()
    {
        typedef Simd::View<Simd::Allocator> View;
        typedef Simd::Frame<Simd::Allocator> Frame;

        Frame a = Frame(View(128, 96, View::Gray8)), b(View(40, 30, View::Bgr24));

        b = std::move(a);
    }
#endif

#if defined(SIMD_SYNET_ENABLE)
    static void TestSynetAdd16b()
    {
        const size_t n = 64;
        std::vector<float> a(n, 1.0f), b(n, 2.0f), dst1(n, 0.0f), dst2(n, 0.0f);
        Simd::Shape dims = Simd::Shape({ n });

        Simd::SynetAdd16b add;
        add.Init(dims, SimdTensorData32f, dims, SimdTensorData32f, SimdTensorData32f, SimdTensorFormatNhwc);
        if (add.Enable())
            add.Forward((const uint8_t*)a.data(), (const uint8_t*)b.data(), (uint8_t*)dst1.data());

        void* context = SimdSynetAdd16bInit(dims.data(), dims.size(), SimdTensorData32f, 
            dims.data(), dims.size(), SimdTensorData32f, SimdTensorData32f, SimdTensorFormatNhwc);
        if (context)
        {
            SimdSynetAdd16bForward(context, (const uint8_t*)a.data(), (const uint8_t*)b.data(), (uint8_t*)dst2.data());
            SimdRelease(context);
        }

        for (size_t i = 0; i < n; ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetAdd16b is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }

    static void TestSynetQuantizedAdd()
    {
        const size_t n = 64;
        std::vector<uint8_t> a(n, 50), b(n, 80), dst1(n, 0), dst2(n, 0);
        Simd::Shape dims = Simd::Shape({ n });
        float aScale = 0.010f, bScale = 0.020f, dstScale = 0.015f;
        int32_t aZero = 47, bZero = 30, dstZero = 38;
        float actParams[2] = { 0.0f, 6.0f };

        Simd::SynetQuantizedAdd add;
        add.Init(dims, SimdTensorData8u, aScale, aZero, dims, SimdTensorData8u, bScale, bZero,
            SimdConvolutionActivationIdentity, actParams, SimdTensorData8u, dstScale, dstZero);
        if (add.Enable())
            add.Forward(a.data(), b.data(), dst1.data());

        void* context = SimdSynetQuantizedAddInit(dims.data(), dims.size(), SimdTensorData8u, &aScale, aZero,
            dims.data(), dims.size(), SimdTensorData8u, &bScale, bZero,
            SimdConvolutionActivationIdentity, actParams, SimdTensorData8u, &dstScale, dstZero);
        if (context)
        {
            SimdSynetQuantizedAddForward(context, a.data(), b.data(), dst2.data());
            SimdRelease(context);
        }

        for (size_t i = 0; i < n; ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetQuantizedAdd is failed at " << i << " : " << (int)dst1[i] << " != " << (int)dst2[i] << std::endl;
        }
    }

    static void TestSynetQuantizedMul()
    {
        const size_t n = 64;
        std::vector<uint8_t> a(n, 50), b(n, 80), dst1(n, 0), dst2(n, 0);
        Simd::Shape dims = Simd::Shape({ n });
        float aScale = 0.010f, bScale = 0.020f, dstScale = 0.015f;
        int32_t aZero = 47, bZero = 30, dstZero = 38;

        Simd::SynetQuantizedMul mul;
        mul.Init(dims, SimdTensorData8u, aScale, aZero, dims, SimdTensorData8u, bScale, bZero,
            SimdTensorData8u, dstScale, dstZero);
        if (mul.Enable())
            mul.Forward(a.data(), b.data(), dst1.data());

        void* context = SimdSynetQuantizedMulInit(dims.data(), dims.size(), SimdTensorData8u, &aScale, aZero,
            dims.data(), dims.size(), SimdTensorData8u, &bScale, bZero,
            SimdTensorData8u, &dstScale, dstZero);
        if (context)
        {
            SimdSynetQuantizedMulForward(context, a.data(), b.data(), dst2.data());
            SimdRelease(context);
        }

        for (size_t i = 0; i < n; ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetQuantizedMul is failed at " << i << " : " << (int)dst1[i] << " != " << (int)dst2[i] << std::endl;
        }
    }

    static void TestSynetGatherElements()
    {
        const size_t srcCount = 4, inner = 1, idxCount = 3;
        Simd::Shape outer = Simd::Shape({ 2 });
        std::vector<float> src(outer[0] * srcCount * inner, 0.0f), dst1(outer[0] * idxCount * inner, 0.0f), dst2(outer[0] * idxCount * inner, 0.0f);
        std::vector<int32_t> idx(outer[0] * idxCount * inner, 0);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i);
        idx[0] = 0; idx[1] = 2; idx[2] = 1;
        idx[3] = 3; idx[4] = 1; idx[5] = 0;

        Simd::SynetGatherElements gather;
        gather.Init(SimdTensorData32f, SimdTensorData32i, SimdTrue, 1, outer, srcCount, inner, idxCount);
        if (gather.Enable())
        {
            gather.SetIndex((const uint8_t*)idx.data());
            gather.Forward((const uint8_t*)src.data(), (const uint8_t*)idx.data(), (uint8_t*)dst1.data());
        }

        void* context = SimdSynetGatherElementsInit(SimdTensorData32f, SimdTensorData32i, SimdTrue, 1,
            outer.data(), outer.size(), srcCount, inner, idxCount);
        if (context)
        {
            SimdSynetGatherElementsSetIndex(context, (const uint8_t*)idx.data());
            SimdSynetGatherElementsForward(context, (const uint8_t*)src.data(), (const uint8_t*)idx.data(), (uint8_t*)dst2.data());
            if (gather.InternalBufferSize() != SimdSynetGatherElementsInternalBufferSize(context))
                std::cout << "TestSynetGatherElements is failed : InternalBufferSize mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetGatherElements is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }
#endif

    void CheckCpp()
    {
        TestCpuInfo();

        TestPoint();

        TestRectangle();

        TestView();

        TestFrame();

        TestPyramid();

        TestStdVector();

        TestImageResize();

        TestFrameResize();

        TestViewVector();

        TestFrameVector();

#if defined(SIMD_CPP_2011_ENABLE)
        TestViewMove();

        TestFrameMove();
#endif

#if defined(SIMD_SYNET_ENABLE)
        TestSynetAdd16b();
        TestSynetQuantizedAdd();
        TestSynetQuantizedMul();
        TestSynetGatherElements();
#endif
    }
}


