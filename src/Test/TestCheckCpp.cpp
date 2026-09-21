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
#include "Simd/SimdDefs.h"
#include "Simd/SimdSynet.hpp"

#include <cstring>
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

    static void TestSynetPermute()
    {
        const size_t n = 4, m = 8;
        Simd::Shape shape = Simd::Shape({ n, m });
        Simd::Shape order = Simd::Shape({ 1, 0 });
        std::vector<float> src(n * m, 0.0f), dst1(n * m, 0.0f), dst2(n * m, 0.0f);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i);

        Simd::SynetPermute permute;
        permute.Init(shape, order, SimdTensorData32f);
        if (permute.Enable())
            permute.Forward((const uint8_t*)src.data(), (uint8_t*)dst1.data());

        void* context = SimdSynetPermuteInit(shape.data(), order.data(), shape.size(), SimdTensorData32f);
        if (context)
        {
            SimdSynetPermuteForward(context, (const uint8_t*)src.data(), (uint8_t*)dst2.data());
            if (permute.InternalBufferSize() != SimdSynetPermuteInternalBufferSize(context))
                std::cout << "TestSynetPermute is failed : InternalBufferSize mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetPermute is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }

    static void TestSynetInnerProduct32f()
    {
        const size_t M = 4, N = 8, K = 16;
        std::vector<float> A(M * K), B(K * N), C1(M * N, 0.0f), C2(M * N, 0.0f), bias(N, 0.1f);
        for (size_t i = 0; i < A.size(); ++i)
            A[i] = float(i) * 0.01f;
        for (size_t i = 0; i < B.size(); ++i)
            B[i] = float(i) * 0.02f;

        Simd::SynetInnerProduct32f innerProduct;
        innerProduct.Init(M, N, K, SimdFalse, SimdTrue, SimdTrue, SimdConvolutionActivationIdentity);
        if (innerProduct.Enable())
        {
            innerProduct.SetParams(B.data(), NULL, bias.data(), NULL);
            innerProduct.Forward(A.data(), NULL, NULL, C1.data());
        }

        void* context = SimdSynetInnerProduct32fInit(M, N, K, SimdFalse, SimdTrue, SimdTrue, SimdConvolutionActivationIdentity);
        if (context)
        {
            SimdSynetInnerProduct32fSetParams(context, B.data(), NULL, bias.data(), NULL);
            SimdSynetInnerProduct32fForward(context, A.data(), NULL, NULL, C2.data());
            if (innerProduct.InternalBufferSize() != SimdSynetInnerProduct32fInternalBufferSize(context))
                std::cout << "TestSynetInnerProduct32f is failed : InternalBufferSize mismatch" << std::endl;
            if (innerProduct.ExternalBufferSize() != SimdSynetInnerProduct32fExternalBufferSize(context))
                std::cout << "TestSynetInnerProduct32f is failed : ExternalBufferSize mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < C1.size(); ++i)
        {
            if (C1[i] != C2[i])
                std::cout << "TestSynetInnerProduct32f is failed at " << i << " : " << C1[i] << " != " << C2[i] << std::endl;
        }
    }

    static void TestSynetInnerProduct16b()
    {
        const size_t M = 4, N = 8, K = 16;
        std::vector<float> A(M * K), B(K * N), C1(M * N, 0.0f), C2(M * N, 0.0f), bias(N, 0.1f);
        for (size_t i = 0; i < A.size(); ++i)
            A[i] = float(i) * 0.01f;
        for (size_t i = 0; i < B.size(); ++i)
            B[i] = float(i) * 0.02f;

        Simd::SynetInnerProduct16b innerProduct;
        innerProduct.Init(M, N, K, SimdTensorData32f, SimdTensorData32f, SimdTensorData32f,
            SimdFalse, SimdTrue, SimdTrue, SimdConvolutionActivationIdentity);
        if (innerProduct.Enable())
        {
            innerProduct.SetParams(B.data(), bias.data(), NULL);
            innerProduct.Forward((const uint8_t*)A.data(), NULL, NULL, (uint8_t*)C1.data());
        }

        void* context = SimdSynetInnerProduct16bInit(M, N, K, SimdTensorData32f, SimdTensorData32f, SimdTensorData32f,
            SimdFalse, SimdTrue, SimdTrue, SimdConvolutionActivationIdentity);
        if (context)
        {
            SimdSynetInnerProduct16bSetParams(context, B.data(), bias.data(), NULL);
            SimdSynetInnerProduct16bForward(context, (const uint8_t*)A.data(), NULL, NULL, (uint8_t*)C2.data());
            if (innerProduct.InternalBufferSize() != SimdSynetInnerProduct16bInternalBufferSize(context))
                std::cout << "TestSynetInnerProduct16b is failed : InternalBufferSize mismatch" << std::endl;
            if (innerProduct.ExternalBufferSize() != SimdSynetInnerProduct16bExternalBufferSize(context))
                std::cout << "TestSynetInnerProduct16b is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = innerProduct.Info();
            const char* info2 = SimdSynetInnerProduct16bInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetInnerProduct16b is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < C1.size(); ++i)
        {
            if (C1[i] != C2[i])
                std::cout << "TestSynetInnerProduct16b is failed at " << i << " : " << C1[i] << " != " << C2[i] << std::endl;
        }
    }

    static void TestSynetQuantizedInnerProduct()
    {
        const size_t M = 4, N = 8, K = 16;
        std::vector<uint8_t> A(M * K), C1(M * N, 0), C2(M * N, 0);
        std::vector<int8_t> B(K * N);
        std::vector<float> bScale(N, 0.02f);
        std::vector<int32_t> bias(N, 10);
        float aScale = 0.01f, cScale = 0.015f;
        uint8_t aZero = 47, cZero = 38;
        for (size_t i = 0; i < A.size(); ++i)
            A[i] = uint8_t(50 + i % 20);
        for (size_t i = 0; i < B.size(); ++i)
            B[i] = int8_t((int)i % 17 - 8);
        for (size_t i = 0; i < N; ++i)
        {
            bScale[i] = 0.02f + 0.001f * float(i);
            bias[i] = 10 + int32_t(i);
        }

        Simd::SynetQuantizedInnerProduct innerProduct;
        innerProduct.Init(M, N, K, SimdTensorData8u, SimdTensorData8i, SimdTensorData8u,
            SimdFalse, SimdTrue, SimdTrue);
        if (innerProduct.Enable())
        {
            innerProduct.SetParams(&aScale, &aZero, B.data(), bScale.data(), bias.data(), &cScale, &cZero);
            innerProduct.Forward(A.data(), NULL, NULL, C1.data());
        }

        void* context = SimdSynetQuantizedInnerProductInit(M, N, K, SimdTensorData8u, SimdTensorData8i, SimdTensorData8u,
            SimdFalse, SimdTrue, SimdTrue);
        if (context)
        {
            SimdSynetQuantizedInnerProductSetParams(context, &aScale, &aZero, B.data(), bScale.data(), bias.data(), &cScale, &cZero);
            SimdSynetQuantizedInnerProductForward(context, A.data(), NULL, NULL, C2.data());
            if (innerProduct.InternalBufferSize() != SimdSynetQuantizedInnerProductInternalBufferSize(context))
                std::cout << "TestSynetQuantizedInnerProduct is failed : InternalBufferSize mismatch" << std::endl;
            if (innerProduct.ExternalBufferSize() != SimdSynetQuantizedInnerProductExternalBufferSize(context))
                std::cout << "TestSynetQuantizedInnerProduct is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = innerProduct.Info();
            const char* info2 = SimdSynetQuantizedInnerProductInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetQuantizedInnerProduct is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < C1.size(); ++i)
        {
            if (C1[i] != C2[i])
                std::cout << "TestSynetQuantizedInnerProduct is failed at " << i << " : " << (int)C1[i] << " != " << (int)C2[i] << std::endl;
        }
    }

    static void TestSynetConvolution32f()
    {
        const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, dstC = 8;
        SimdConvolutionParameters conv = {};
        conv.srcC = srcC;
        conv.srcH = srcH;
        conv.srcW = srcW;
        conv.srcT = SimdTensorData32f;
        conv.srcF = SimdTensorFormatNhwc;
        conv.dstC = dstC;
        conv.kernelY = 3;
        conv.kernelX = 3;
        conv.dilationY = 1;
        conv.dilationX = 1;
        conv.strideY = 1;
        conv.strideX = 1;
        conv.padY = 1;
        conv.padX = 1;
        conv.padH = 1;
        conv.padW = 1;
        conv.group = 1;
        conv.activation = SimdConvolutionActivationIdentity;
        conv.dstH = (conv.srcH + conv.padY + conv.padH - (conv.dilationY * (conv.kernelY - 1) + 1)) / conv.strideY + 1;
        conv.dstW = (conv.srcW + conv.padX + conv.padW - (conv.dilationX * (conv.kernelX - 1) + 1)) / conv.strideX + 1;
        conv.dstT = SimdTensorData32f;
        conv.dstF = SimdTensorFormatNhwc;

        const size_t srcSize = batch * srcH * srcW * srcC;
        const size_t weightSize = conv.kernelY * conv.kernelX * srcC * dstC / conv.group;
        const size_t dstSize = batch * conv.dstH * conv.dstW * dstC;
        std::vector<float> src(srcSize), weight(weightSize), bias(dstC, 0.1f);
        std::vector<float> dst1(dstSize, 0.0f), dst2(dstSize, 0.0f);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i) * 0.01f;
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] = float(i) * 0.02f;

        Simd::SynetConvolution32f convolution;
        convolution.Init(batch, &conv);
        if (convolution.Enable())
        {
            convolution.SetParams(weight.data(), NULL, bias.data(), NULL);
            convolution.Forward(src.data(), NULL, dst1.data());
        }

        void* context = SimdSynetConvolution32fInit(batch, &conv);
        if (context)
        {
            SimdSynetConvolution32fSetParams(context, weight.data(), NULL, bias.data(), NULL);
            SimdSynetConvolution32fForward(context, src.data(), NULL, dst2.data());
            if (convolution.InternalBufferSize() != SimdSynetConvolution32fInternalBufferSize(context))
                std::cout << "TestSynetConvolution32f is failed : InternalBufferSize mismatch" << std::endl;
            if (convolution.ExternalBufferSize() != SimdSynetConvolution32fExternalBufferSize(context))
                std::cout << "TestSynetConvolution32f is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = convolution.Info();
            const char* info2 = SimdSynetConvolution32fInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetConvolution32f is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetConvolution32f is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }

    static void TestSynetConvolution16b()
    {
        const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, dstC = 8;
        SimdConvolutionParameters conv = {};
        conv.srcC = srcC;
        conv.srcH = srcH;
        conv.srcW = srcW;
        conv.srcT = SimdTensorData32f;
        conv.srcF = SimdTensorFormatNhwc;
        conv.dstC = dstC;
        conv.kernelY = 3;
        conv.kernelX = 3;
        conv.dilationY = 1;
        conv.dilationX = 1;
        conv.strideY = 1;
        conv.strideX = 1;
        conv.padY = 1;
        conv.padX = 1;
        conv.padH = 1;
        conv.padW = 1;
        conv.group = 1;
        conv.activation = SimdConvolutionActivationIdentity;
        conv.dstH = (conv.srcH + conv.padY + conv.padH - (conv.dilationY * (conv.kernelY - 1) + 1)) / conv.strideY + 1;
        conv.dstW = (conv.srcW + conv.padX + conv.padW - (conv.dilationX * (conv.kernelX - 1) + 1)) / conv.strideX + 1;
        conv.dstT = SimdTensorData32f;
        conv.dstF = SimdTensorFormatNhwc;

        const size_t srcSize = batch * srcH * srcW * srcC;
        const size_t weightSize = conv.kernelY * conv.kernelX * srcC * dstC / conv.group;
        const size_t dstSize = batch * conv.dstH * conv.dstW * dstC;
        std::vector<float> src(srcSize), weight(weightSize), bias(dstC, 0.1f);
        std::vector<float> dst1(dstSize, 0.0f), dst2(dstSize, 0.0f);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i) * 0.01f;
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] = float(i) * 0.02f;

        Simd::SynetConvolution16b convolution;
        convolution.Init(batch, &conv);
        if (convolution.Enable())
        {
            convolution.SetParams(weight.data(), bias.data(), NULL);
            convolution.Forward((const uint8_t*)src.data(), NULL, (uint8_t*)dst1.data());
        }

        void* context = SimdSynetConvolution16bInit(batch, &conv, SimdSynetCompatibilityDefault);
        if (context)
        {
            SimdSynetConvolution16bSetParams(context, weight.data(), bias.data(), NULL);
            SimdSynetConvolution16bForward(context, (const uint8_t*)src.data(), NULL, (uint8_t*)dst2.data());
            if (convolution.InternalBufferSize() != SimdSynetConvolution16bInternalBufferSize(context))
                std::cout << "TestSynetConvolution16b is failed : InternalBufferSize mismatch" << std::endl;
            if (convolution.ExternalBufferSize() != SimdSynetConvolution16bExternalBufferSize(context))
                std::cout << "TestSynetConvolution16b is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = convolution.Info();
            const char* info2 = SimdSynetConvolution16bInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetConvolution16b is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetConvolution16b is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }

    static void TestSynetConvolution8i()
    {
        const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, dstC = 8;
        const SimdSynetCompatibilityType compatibility = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaUse);
        SimdConvolutionParameters conv = {};
        conv.srcC = srcC;
        conv.srcH = srcH;
        conv.srcW = srcW;
        conv.srcT = SimdTensorData32f;
        conv.srcF = SimdTensorFormatNhwc;
        conv.dstC = dstC;
        conv.kernelY = 3;
        conv.kernelX = 3;
        conv.dilationY = 1;
        conv.dilationX = 1;
        conv.strideY = 1;
        conv.strideX = 1;
        conv.padY = 1;
        conv.padX = 1;
        conv.padH = 1;
        conv.padW = 1;
        conv.group = 1;
        conv.activation = SimdConvolutionActivationIdentity;
        conv.dstH = (conv.srcH + conv.padY + conv.padH - (conv.dilationY * (conv.kernelY - 1) + 1)) / conv.strideY + 1;
        conv.dstW = (conv.srcW + conv.padX + conv.padW - (conv.dilationX * (conv.kernelX - 1) + 1)) / conv.strideX + 1;
        conv.dstT = SimdTensorData32f;
        conv.dstF = SimdTensorFormatNhwc;

        const size_t srcSize = batch * srcH * srcW * srcC;
        const size_t weightSize = conv.kernelY * conv.kernelX * srcC * dstC / conv.group;
        const size_t dstSize = batch * conv.dstH * conv.dstW * dstC;
        std::vector<float> src(srcSize), weight(weightSize), bias(dstC, 0.1f);
        std::vector<float> srcMin(srcC, -1.0f), srcMax(srcC, 1.0f);
        std::vector<float> dstMin(dstC, -1.0f), dstMax(dstC, 1.0f);
        std::vector<float> dst1(dstSize, 0.0f), dst2(dstSize, 0.0f);
        const float* stats[4] = { srcMin.data(), srcMax.data(), dstMin.data(), dstMax.data() };
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i) * 0.01f;
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] = float(i) * 0.02f;

        Simd::SynetConvolution8i convolution;
        convolution.Init(batch, &conv, compatibility);
        if (convolution.Enable())
        {
            convolution.SetParams(weight.data(), bias.data(), NULL, stats);
            convolution.Forward((const uint8_t*)src.data(), NULL, (uint8_t*)dst1.data());
        }

        void* context = SimdSynetConvolution8iInit(batch, &conv, compatibility);
        if (context)
        {
            SimdSynetConvolution8iSetParams(context, weight.data(), bias.data(), NULL, stats);
            SimdSynetConvolution8iForward(context, (const uint8_t*)src.data(), NULL, (uint8_t*)dst2.data());
            if (convolution.InternalBufferSize() != SimdSynetConvolution8iInternalBufferSize(context))
                std::cout << "TestSynetConvolution8i is failed : InternalBufferSize mismatch" << std::endl;
            if (convolution.ExternalBufferSize() != SimdSynetConvolution8iExternalBufferSize(context))
                std::cout << "TestSynetConvolution8i is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = convolution.Info();
            const char* info2 = SimdSynetConvolution8iInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetConvolution8i is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetConvolution8i is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }

    static void TestSynetQuantizedConvolution()
    {
        const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, dstC = 8;
        SimdConvolutionParameters conv = {};
        conv.srcC = srcC;
        conv.srcH = srcH;
        conv.srcW = srcW;
        conv.srcT = SimdTensorData8u;
        conv.srcF = SimdTensorFormatNhwc;
        conv.dstC = dstC;
        conv.kernelY = 3;
        conv.kernelX = 3;
        conv.dilationY = 1;
        conv.dilationX = 1;
        conv.strideY = 1;
        conv.strideX = 1;
        conv.padY = 1;
        conv.padX = 1;
        conv.padH = 1;
        conv.padW = 1;
        conv.group = 1;
        conv.activation = SimdConvolutionActivationIdentity;
        conv.dstH = (conv.srcH + conv.padY + conv.padH - (conv.dilationY * (conv.kernelY - 1) + 1)) / conv.strideY + 1;
        conv.dstW = (conv.srcW + conv.padX + conv.padW - (conv.dilationX * (conv.kernelX - 1) + 1)) / conv.strideX + 1;
        conv.dstT = SimdTensorData8u;
        conv.dstF = SimdTensorFormatNhwc;

        const size_t srcSize = batch * srcH * srcW * srcC;
        const size_t weightSize = conv.kernelY * conv.kernelX * srcC * dstC / conv.group;
        const size_t dstSize = batch * conv.dstH * conv.dstW * dstC;
        std::vector<uint8_t> src(srcSize);
        std::vector<int8_t> weight(weightSize);
        std::vector<float> weightScale(dstC, 0.02f);
        std::vector<int32_t> bias(dstC, 1);
        float ioScale[3] = { 0.01f, 0.015f, 0.02f };
        uint8_t ioZero[3] = { 128, 127, 126 };
        std::vector<uint8_t> dst1(dstSize, 0), dst2(dstSize, 0);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = uint8_t(i);
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] = int8_t(i);
        for (size_t i = 0; i < dstC; ++i)
        {
            weightScale[i] = 0.02f + 0.001f * float(i);
            bias[i] = 1 + int32_t(i);
        }

        Simd::SynetQuantizedConvolution convolution;
        convolution.Init(batch, &conv);
        if (convolution.Enable())
        {
            convolution.SetParams(ioScale, ioZero, weight.data(), weightScale.data(), bias.data(), NULL);
            convolution.Forward(src.data(), NULL, dst1.data());
        }

        void* context = SimdSynetQuantizedConvolutionInit(batch, &conv);
        if (context)
        {
            SimdSynetQuantizedConvolutionSetParams(context, ioScale, ioZero, weight.data(), weightScale.data(), bias.data(), NULL);
            SimdSynetQuantizedConvolutionForward(context, src.data(), NULL, dst2.data());
            if (convolution.InternalBufferSize() != SimdSynetQuantizedConvolutionInternalBufferSize(context))
                std::cout << "TestSynetQuantizedConvolution is failed : InternalBufferSize mismatch" << std::endl;
            if (convolution.ExternalBufferSize() != SimdSynetQuantizedConvolutionExternalBufferSize(context))
                std::cout << "TestSynetQuantizedConvolution is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = convolution.Info();
            const char* info2 = SimdSynetQuantizedConvolutionInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetQuantizedConvolution is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetQuantizedConvolution is failed at " << i << " : " << (int)dst1[i] << " != " << (int)dst2[i] << std::endl;
        }
    }

    static void TestSynetDeconvolution32f()
    {
        const size_t batch = 1, srcC = 4, srcH = 3, srcW = 3, dstC = 4;
        SimdConvolutionParameters conv = {};
        conv.srcC = srcC;
        conv.srcH = srcH;
        conv.srcW = srcW;
        conv.srcT = SimdTensorData32f;
        conv.srcF = SimdTensorFormatNhwc;
        conv.dstC = dstC;
        conv.kernelY = 2;
        conv.kernelX = 2;
        conv.dilationY = 1;
        conv.dilationX = 1;
        conv.strideY = 2;
        conv.strideX = 2;
        conv.padY = 0;
        conv.padX = 0;
        conv.padH = 0;
        conv.padW = 0;
        conv.group = 1;
        conv.activation = SimdConvolutionActivationIdentity;
        conv.dstH = conv.strideY * (conv.srcH - 1) + conv.dilationY * (conv.kernelY - 1) + 1 - conv.padY - conv.padH;
        conv.dstW = conv.strideX * (conv.srcW - 1) + conv.dilationX * (conv.kernelX - 1) + 1 - conv.padX - conv.padW;
        conv.dstT = SimdTensorData32f;
        conv.dstF = SimdTensorFormatNhwc;

        const size_t srcSize = batch * srcH * srcW * srcC;
        const size_t weightSize = conv.kernelY * conv.kernelX * srcC * dstC / conv.group;
        const size_t dstSize = batch * conv.dstH * conv.dstW * dstC;
        std::vector<float> src(srcSize), weight(weightSize), bias(dstC, 0.1f);
        std::vector<float> dst1(dstSize, 0.0f), dst2(dstSize, 0.0f);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i) * 0.01f;
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] = float(i) * 0.02f;

        Simd::SynetDeconvolution32f deconvolution;
        deconvolution.Init(batch, &conv);
        if (deconvolution.Enable())
        {
            deconvolution.SetParams(weight.data(), NULL, bias.data(), NULL);
            deconvolution.Forward(src.data(), NULL, dst1.data());
        }

        void* context = SimdSynetDeconvolution32fInit(batch, &conv, SimdSynetCompatibilityDefault);
        if (context)
        {
            SimdSynetDeconvolution32fSetParams(context, weight.data(), NULL, bias.data(), NULL);
            SimdSynetDeconvolution32fForward(context, src.data(), NULL, dst2.data());
            if (deconvolution.InternalBufferSize() != SimdSynetDeconvolution32fInternalBufferSize(context))
                std::cout << "TestSynetDeconvolution32f is failed : InternalBufferSize mismatch" << std::endl;
            if (deconvolution.ExternalBufferSize() != SimdSynetDeconvolution32fExternalBufferSize(context))
                std::cout << "TestSynetDeconvolution32f is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = deconvolution.Info();
            const char* info2 = SimdSynetDeconvolution32fInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetDeconvolution32f is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetDeconvolution32f is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }

    static void TestSynetDeconvolution16b()
    {
        const size_t batch = 1, srcC = 4, srcH = 3, srcW = 3, dstC = 4;
        SimdConvolutionParameters conv = {};
        conv.srcC = srcC;
        conv.srcH = srcH;
        conv.srcW = srcW;
        conv.srcT = SimdTensorData32f;
        conv.srcF = SimdTensorFormatNhwc;
        conv.dstC = dstC;
        conv.kernelY = 2;
        conv.kernelX = 2;
        conv.dilationY = 1;
        conv.dilationX = 1;
        conv.strideY = 2;
        conv.strideX = 2;
        conv.padY = 0;
        conv.padX = 0;
        conv.padH = 0;
        conv.padW = 0;
        conv.group = 1;
        conv.activation = SimdConvolutionActivationIdentity;
        conv.dstH = conv.strideY * (conv.srcH - 1) + conv.dilationY * (conv.kernelY - 1) + 1 - conv.padY - conv.padH;
        conv.dstW = conv.strideX * (conv.srcW - 1) + conv.dilationX * (conv.kernelX - 1) + 1 - conv.padX - conv.padW;
        conv.dstT = SimdTensorData32f;
        conv.dstF = SimdTensorFormatNhwc;

        const size_t srcSize = batch * srcH * srcW * srcC;
        const size_t weightSize = conv.kernelY * conv.kernelX * srcC * dstC / conv.group;
        const size_t dstSize = batch * conv.dstH * conv.dstW * dstC;
        std::vector<float> src(srcSize), weight(weightSize), bias(dstC, 0.1f);
        std::vector<float> dst1(dstSize, 0.0f), dst2(dstSize, 0.0f);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i) * 0.01f;
        for (size_t i = 0; i < weight.size(); ++i)
            weight[i] = float(i) * 0.02f;

        Simd::SynetDeconvolution16b deconvolution;
        deconvolution.Init(batch, &conv);
        if (deconvolution.Enable())
        {
            deconvolution.SetParams(weight.data(), bias.data(), NULL);
            deconvolution.Forward((const uint8_t*)src.data(), NULL, (uint8_t*)dst1.data());
        }

        void* context = SimdSynetDeconvolution16bInit(batch, &conv, SimdSynetCompatibilityDefault);
        if (context)
        {
            SimdSynetDeconvolution16bSetParams(context, weight.data(), bias.data(), NULL);
            SimdSynetDeconvolution16bForward(context, (const uint8_t*)src.data(), NULL, (uint8_t*)dst2.data());
            if (deconvolution.InternalBufferSize() != SimdSynetDeconvolution16bInternalBufferSize(context))
                std::cout << "TestSynetDeconvolution16b is failed : InternalBufferSize mismatch" << std::endl;
            if (deconvolution.ExternalBufferSize() != SimdSynetDeconvolution16bExternalBufferSize(context))
                std::cout << "TestSynetDeconvolution16b is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = deconvolution.Info();
            const char* info2 = SimdSynetDeconvolution16bInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetDeconvolution16b is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetDeconvolution16b is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }

    static void TestSynetMergedConvolution32f()
    {
        const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, midC = 8, count = 2;
        SimdConvolutionParameters convs[2] = {};

        convs[0].srcC = srcC;
        convs[0].srcH = srcH;
        convs[0].srcW = srcW;
        convs[0].srcT = SimdTensorData32f;
        convs[0].srcF = SimdTensorFormatNhwc;
        convs[0].dstC = midC;
        convs[0].kernelY = 1;
        convs[0].kernelX = 1;
        convs[0].dilationY = 1;
        convs[0].dilationX = 1;
        convs[0].strideY = 1;
        convs[0].strideX = 1;
        convs[0].padY = 0;
        convs[0].padX = 0;
        convs[0].padH = 0;
        convs[0].padW = 0;
        convs[0].group = 1;
        convs[0].activation = SimdConvolutionActivationIdentity;
        convs[0].dstH = srcH;
        convs[0].dstW = srcW;
        convs[0].dstT = SimdTensorData32f;
        convs[0].dstF = SimdTensorFormatNhwc;

        convs[1].srcC = midC;
        convs[1].srcH = convs[0].dstH;
        convs[1].srcW = convs[0].dstW;
        convs[1].srcT = SimdTensorData32f;
        convs[1].srcF = SimdTensorFormatNhwc;
        convs[1].dstC = midC;
        convs[1].kernelY = 3;
        convs[1].kernelX = 3;
        convs[1].dilationY = 1;
        convs[1].dilationX = 1;
        convs[1].strideY = 1;
        convs[1].strideX = 1;
        convs[1].padY = 1;
        convs[1].padX = 1;
        convs[1].padH = 1;
        convs[1].padW = 1;
        convs[1].group = midC;
        convs[1].activation = SimdConvolutionActivationIdentity;
        convs[1].dstH = convs[1].srcH;
        convs[1].dstW = convs[1].srcW;
        convs[1].dstT = SimdTensorData32f;
        convs[1].dstF = SimdTensorFormatNhwc;

        const size_t srcSize = batch * srcH * srcW * srcC;
        const size_t weight0Size = convs[0].kernelY * convs[0].kernelX * convs[0].srcC * convs[0].dstC / convs[0].group;
        const size_t weight1Size = convs[1].kernelY * convs[1].kernelX * convs[1].srcC * convs[1].dstC / convs[1].group;
        const size_t dstSize = batch * convs[1].dstH * convs[1].dstW * convs[1].dstC;
        std::vector<float> src(srcSize), weight0(weight0Size), weight1(weight1Size);
        std::vector<float> bias0(convs[0].dstC, 0.1f), bias1(convs[1].dstC, 0.2f);
        std::vector<float> dst1(dstSize, 0.0f), dst2(dstSize, 0.0f);
        const float* weight[2] = { weight0.data(), weight1.data() };
        const float* bias[2] = { bias0.data(), bias1.data() };
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i) * 0.01f;
        for (size_t i = 0; i < weight0.size(); ++i)
            weight0[i] = float(i) * 0.02f;
        for (size_t i = 0; i < weight1.size(); ++i)
            weight1[i] = float(i) * 0.03f;

        Simd::SynetMergedConvolution32f mergedConvolution;
        mergedConvolution.Init(batch, convs, count, SimdFalse);
        if (mergedConvolution.Enable())
        {
            mergedConvolution.SetParams(weight, NULL, bias, NULL);
            mergedConvolution.Forward(src.data(), NULL, dst1.data());
        }

        void* context = SimdSynetMergedConvolution32fInit(batch, convs, count, SimdFalse);
        if (context)
        {
            SimdSynetMergedConvolution32fSetParams(context, weight, NULL, bias, NULL);
            SimdSynetMergedConvolution32fForward(context, src.data(), NULL, dst2.data());
            if (mergedConvolution.InternalBufferSize() != SimdSynetMergedConvolution32fInternalBufferSize(context))
                std::cout << "TestSynetMergedConvolution32f is failed : InternalBufferSize mismatch" << std::endl;
            if (mergedConvolution.ExternalBufferSize() != SimdSynetMergedConvolution32fExternalBufferSize(context))
                std::cout << "TestSynetMergedConvolution32f is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = mergedConvolution.Info();
            const char* info2 = SimdSynetMergedConvolution32fInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetMergedConvolution32f is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetMergedConvolution32f is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }

    static void TestSynetMergedConvolution16b()
    {
        const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, midC = 8, count = 2;
        SimdConvolutionParameters convs[2] = {};

        convs[0].srcC = srcC;
        convs[0].srcH = srcH;
        convs[0].srcW = srcW;
        convs[0].srcT = SimdTensorData32f;
        convs[0].srcF = SimdTensorFormatNhwc;
        convs[0].dstC = midC;
        convs[0].kernelY = 1;
        convs[0].kernelX = 1;
        convs[0].dilationY = 1;
        convs[0].dilationX = 1;
        convs[0].strideY = 1;
        convs[0].strideX = 1;
        convs[0].padY = 0;
        convs[0].padX = 0;
        convs[0].padH = 0;
        convs[0].padW = 0;
        convs[0].group = 1;
        convs[0].activation = SimdConvolutionActivationIdentity;
        convs[0].dstH = srcH;
        convs[0].dstW = srcW;
        convs[0].dstT = SimdTensorData32f;
        convs[0].dstF = SimdTensorFormatNhwc;

        convs[1].srcC = midC;
        convs[1].srcH = convs[0].dstH;
        convs[1].srcW = convs[0].dstW;
        convs[1].srcT = SimdTensorData32f;
        convs[1].srcF = SimdTensorFormatNhwc;
        convs[1].dstC = midC;
        convs[1].kernelY = 3;
        convs[1].kernelX = 3;
        convs[1].dilationY = 1;
        convs[1].dilationX = 1;
        convs[1].strideY = 1;
        convs[1].strideX = 1;
        convs[1].padY = 1;
        convs[1].padX = 1;
        convs[1].padH = 1;
        convs[1].padW = 1;
        convs[1].group = midC;
        convs[1].activation = SimdConvolutionActivationIdentity;
        convs[1].dstH = convs[1].srcH;
        convs[1].dstW = convs[1].srcW;
        convs[1].dstT = SimdTensorData32f;
        convs[1].dstF = SimdTensorFormatNhwc;

        const size_t srcSize = batch * srcH * srcW * srcC;
        const size_t weight0Size = convs[0].kernelY * convs[0].kernelX * convs[0].srcC * convs[0].dstC / convs[0].group;
        const size_t weight1Size = convs[1].kernelY * convs[1].kernelX * convs[1].srcC * convs[1].dstC / convs[1].group;
        const size_t dstSize = batch * convs[1].dstH * convs[1].dstW * convs[1].dstC;
        std::vector<float> src(srcSize), weight0(weight0Size), weight1(weight1Size);
        std::vector<float> bias0(convs[0].dstC, 0.1f), bias1(convs[1].dstC, 0.2f);
        std::vector<float> dst1(dstSize, 0.0f), dst2(dstSize, 0.0f);
        const float* weight[2] = { weight0.data(), weight1.data() };
        const float* bias[2] = { bias0.data(), bias1.data() };
        const float* params[2] = { NULL, NULL };
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i) * 0.01f;
        for (size_t i = 0; i < weight0.size(); ++i)
            weight0[i] = float(i) * 0.02f;
        for (size_t i = 0; i < weight1.size(); ++i)
            weight1[i] = float(i) * 0.03f;

        Simd::SynetMergedConvolution16b mergedConvolution;
        mergedConvolution.Init(batch, convs, count, SimdFalse);
        if (mergedConvolution.Enable())
        {
            mergedConvolution.SetParams(weight, bias, params);
            mergedConvolution.Forward((const uint8_t*)src.data(), NULL, (uint8_t*)dst1.data());
        }

        void* context = SimdSynetMergedConvolution16bInit(batch, convs, count, SimdFalse);
        if (context)
        {
            SimdSynetMergedConvolution16bSetParams(context, weight, bias, params);
            SimdSynetMergedConvolution16bForward(context, (const uint8_t*)src.data(), NULL, (uint8_t*)dst2.data());
            if (mergedConvolution.InternalBufferSize() != SimdSynetMergedConvolution16bInternalBufferSize(context))
                std::cout << "TestSynetMergedConvolution16b is failed : InternalBufferSize mismatch" << std::endl;
            if (mergedConvolution.ExternalBufferSize() != SimdSynetMergedConvolution16bExternalBufferSize(context))
                std::cout << "TestSynetMergedConvolution16b is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = mergedConvolution.Info();
            const char* info2 = SimdSynetMergedConvolution16bInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetMergedConvolution16b is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetMergedConvolution16b is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }

    static void TestSynetMergedConvolution8i()
    {
        const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, midC = 8, count = 2;
        const SimdSynetCompatibilityType compatibility = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaUse);
        SimdConvolutionParameters convs[2] = {};

        convs[0].srcC = srcC;
        convs[0].srcH = srcH;
        convs[0].srcW = srcW;
        convs[0].srcT = SimdTensorData32f;
        convs[0].srcF = SimdTensorFormatNhwc;
        convs[0].dstC = midC;
        convs[0].kernelY = 1;
        convs[0].kernelX = 1;
        convs[0].dilationY = 1;
        convs[0].dilationX = 1;
        convs[0].strideY = 1;
        convs[0].strideX = 1;
        convs[0].padY = 0;
        convs[0].padX = 0;
        convs[0].padH = 0;
        convs[0].padW = 0;
        convs[0].group = 1;
        convs[0].activation = SimdConvolutionActivationIdentity;
        convs[0].dstH = srcH;
        convs[0].dstW = srcW;
        convs[0].dstT = SimdTensorData32f;
        convs[0].dstF = SimdTensorFormatNhwc;

        convs[1].srcC = midC;
        convs[1].srcH = convs[0].dstH;
        convs[1].srcW = convs[0].dstW;
        convs[1].srcT = SimdTensorData32f;
        convs[1].srcF = SimdTensorFormatNhwc;
        convs[1].dstC = midC;
        convs[1].kernelY = 3;
        convs[1].kernelX = 3;
        convs[1].dilationY = 1;
        convs[1].dilationX = 1;
        convs[1].strideY = 1;
        convs[1].strideX = 1;
        convs[1].padY = 1;
        convs[1].padX = 1;
        convs[1].padH = 1;
        convs[1].padW = 1;
        convs[1].group = midC;
        convs[1].activation = SimdConvolutionActivationIdentity;
        convs[1].dstH = convs[1].srcH;
        convs[1].dstW = convs[1].srcW;
        convs[1].dstT = SimdTensorData32f;
        convs[1].dstF = SimdTensorFormatNhwc;

        const size_t srcSize = batch * srcH * srcW * srcC;
        const size_t weight0Size = convs[0].kernelY * convs[0].kernelX * convs[0].srcC * convs[0].dstC / convs[0].group;
        const size_t weight1Size = convs[1].kernelY * convs[1].kernelX * convs[1].srcC * convs[1].dstC / convs[1].group;
        const size_t dstSize = batch * convs[1].dstH * convs[1].dstW * convs[1].dstC;
        std::vector<float> src(srcSize), weight0(weight0Size), weight1(weight1Size);
        std::vector<float> bias0(convs[0].dstC, 0.1f), bias1(convs[1].dstC, 0.2f);
        std::vector<float> srcMin(srcC, -1.0f), srcMax(srcC, 1.0f);
        std::vector<float> midMin(midC, -1.0f), midMax(midC, 1.0f);
        std::vector<float> dstMin(midC, -1.0f), dstMax(midC, 1.0f);
        std::vector<float> dst1(dstSize, 0.0f), dst2(dstSize, 0.0f);
        const float* weight[2] = { weight0.data(), weight1.data() };
        const float* bias[2] = { bias0.data(), bias1.data() };
        const float* params[2] = { NULL, NULL };
        const float* stats[6] = { srcMin.data(), srcMax.data(), midMin.data(), midMax.data(), dstMin.data(), dstMax.data() };
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i) * 0.01f;
        for (size_t i = 0; i < weight0.size(); ++i)
            weight0[i] = float(i) * 0.02f;
        for (size_t i = 0; i < weight1.size(); ++i)
            weight1[i] = float(i) * 0.03f;

        Simd::SynetMergedConvolution8i mergedConvolution;
        mergedConvolution.Init(batch, convs, count, compatibility);
        if (mergedConvolution.Enable())
        {
            mergedConvolution.SetParams(weight, NULL, bias, params, stats);
            mergedConvolution.Forward((const uint8_t*)src.data(), NULL, (uint8_t*)dst1.data());
        }

        void* context = SimdSynetMergedConvolution8iInit(batch, convs, count, compatibility);
        if (context)
        {
            SimdSynetMergedConvolution8iSetParams(context, weight, NULL, bias, params, stats);
            SimdSynetMergedConvolution8iForward(context, (const uint8_t*)src.data(), NULL, (uint8_t*)dst2.data());
            if (mergedConvolution.InternalBufferSize() != SimdSynetMergedConvolution8iInternalBufferSize(context))
                std::cout << "TestSynetMergedConvolution8i is failed : InternalBufferSize mismatch" << std::endl;
            if (mergedConvolution.ExternalBufferSize() != SimdSynetMergedConvolution8iExternalBufferSize(context))
                std::cout << "TestSynetMergedConvolution8i is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = mergedConvolution.Info();
            const char* info2 = SimdSynetMergedConvolution8iInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetMergedConvolution8i is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetMergedConvolution8i is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
        }
    }

    static void TestSynetQuantizedMergedConvolution()
    {
        const size_t batch = 1, srcC = 4, srcH = 8, srcW = 8, midC = 8, count = 2, add = 0;
        SimdConvolutionParameters convs[2] = {};

        convs[0].srcC = srcC;
        convs[0].srcH = srcH;
        convs[0].srcW = srcW;
        convs[0].srcT = SimdTensorData8u;
        convs[0].srcF = SimdTensorFormatNhwc;
        convs[0].dstC = midC;
        convs[0].kernelY = 1;
        convs[0].kernelX = 1;
        convs[0].dilationY = 1;
        convs[0].dilationX = 1;
        convs[0].strideY = 1;
        convs[0].strideX = 1;
        convs[0].padY = 0;
        convs[0].padX = 0;
        convs[0].padH = 0;
        convs[0].padW = 0;
        convs[0].group = 1;
        convs[0].activation = SimdConvolutionActivationIdentity;
        convs[0].dstH = srcH;
        convs[0].dstW = srcW;
        convs[0].dstT = SimdTensorData8u;
        convs[0].dstF = SimdTensorFormatNhwc;

        convs[1].srcC = midC;
        convs[1].srcH = convs[0].dstH;
        convs[1].srcW = convs[0].dstW;
        convs[1].srcT = SimdTensorData8u;
        convs[1].srcF = SimdTensorFormatNhwc;
        convs[1].dstC = midC;
        convs[1].kernelY = 3;
        convs[1].kernelX = 3;
        convs[1].dilationY = 1;
        convs[1].dilationX = 1;
        convs[1].strideY = 1;
        convs[1].strideX = 1;
        convs[1].padY = 1;
        convs[1].padX = 1;
        convs[1].padH = 1;
        convs[1].padW = 1;
        convs[1].group = midC;
        convs[1].activation = SimdConvolutionActivationIdentity;
        convs[1].dstH = convs[1].srcH;
        convs[1].dstW = convs[1].srcW;
        convs[1].dstT = SimdTensorData8u;
        convs[1].dstF = SimdTensorFormatNhwc;

        const size_t srcSize = batch * srcH * srcW * srcC;
        const size_t weight0Size = convs[0].kernelY * convs[0].kernelX * convs[0].srcC * convs[0].dstC / convs[0].group;
        const size_t weight1Size = convs[1].kernelY * convs[1].kernelX * convs[1].srcC * convs[1].dstC / convs[1].group;
        const size_t dstSize = batch * convs[1].dstH * convs[1].dstW * convs[1].dstC;
        std::vector<uint8_t> src(srcSize);
        std::vector<int8_t> weight0(weight0Size), weight1(weight1Size);
        std::vector<float> weightScale0(convs[0].dstC, 0.02f), weightScale1(convs[1].dstC, 0.03f);
        std::vector<int32_t> bias0(convs[0].dstC, 1), bias1(convs[1].dstC, 2);
        float ioScale[3] = { 0.01f, 0.015f, 0.02f };
        uint8_t ioZero[3] = { 128, 127, 126 };
        std::vector<uint8_t> dst1(dstSize, 0), dst2(dstSize, 0);
        const int8_t* weight[2] = { weight0.data(), weight1.data() };
        const float* weightScale[2] = { weightScale0.data(), weightScale1.data() };
        const int32_t* bias[2] = { bias0.data(), bias1.data() };
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = uint8_t(i);
        for (size_t i = 0; i < weight0.size(); ++i)
            weight0[i] = int8_t(i);
        for (size_t i = 0; i < weight1.size(); ++i)
            weight1[i] = int8_t(i);

        Simd::SynetQuantizedMergedConvolution mergedConvolution;
        mergedConvolution.Init(batch, convs, count, add);
        if (mergedConvolution.Enable())
        {
            mergedConvolution.SetParams(ioScale, ioZero, weight, weightScale, bias);
            mergedConvolution.Forward(src.data(), NULL, dst1.data());
        }

        void* context = SimdSynetQuantizedMergedConvolutionInit(batch, convs, count, add);
        if (context)
        {
            SimdSynetQuantizedMergedConvolutionSetParams(context, ioScale, ioZero, weight, weightScale, bias);
            SimdSynetQuantizedMergedConvolutionForward(context, src.data(), NULL, dst2.data());
            if (mergedConvolution.InternalBufferSize() != SimdSynetQuantizedMergedConvolutionInternalBufferSize(context))
                std::cout << "TestSynetQuantizedMergedConvolution is failed : InternalBufferSize mismatch" << std::endl;
            if (mergedConvolution.ExternalBufferSize() != SimdSynetQuantizedMergedConvolutionExternalBufferSize(context))
                std::cout << "TestSynetQuantizedMergedConvolution is failed : ExternalBufferSize mismatch" << std::endl;
            const char* info1 = mergedConvolution.Info();
            const char* info2 = SimdSynetQuantizedMergedConvolutionInfo(context);
            if ((info1 == NULL) != (info2 == NULL) || (info1 && info2 && std::strcmp(info1, info2) != 0))
                std::cout << "TestSynetQuantizedMergedConvolution is failed : Info mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetQuantizedMergedConvolution is failed at " << i << " : " << (int)dst1[i] << " != " << (int)dst2[i] << std::endl;
        }
    }

    static void TestSynetScale8i()
    {
        const size_t batch = 1, channels = 4, spatial = 16;
        const SimdSynetCompatibilityType compatibility = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaUse);
        const size_t size = batch * channels * spatial;
        std::vector<uint8_t> src(size), dst1(size, 0), dst2(size, 0);
        std::vector<float> scale(channels), bias(channels);
        std::vector<float> srcMin(channels, 0.0f), srcMax(channels, 1.0f);
        std::vector<float> dstMin(channels, 0.0f), dstMax(channels, 1.0f);
        const float* stats[4] = { srcMin.data(), srcMax.data(), dstMin.data(), dstMax.data() };
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = uint8_t(40 + i % 80);
        for (size_t i = 0; i < channels; ++i)
        {
            scale[i] = 0.5f + 0.1f * float(i);
            bias[i] = 0.1f * float(i);
        }

        Simd::SynetScale8i scale8i;
        scale8i.Init(batch, channels, spatial, SimdTensorData8u, SimdTensorData8u, SimdTensorFormatNhwc, compatibility);
        if (scale8i.Enable())
        {
            scale8i.SetParams(scale.data(), bias.data(), stats);
            scale8i.Forward(src.data(), dst1.data());
        }

        void* context = SimdSynetScale8iInit(batch, channels, spatial, SimdTensorData8u, SimdTensorData8u, SimdTensorFormatNhwc, compatibility);
        if (context)
        {
            SimdSynetScale8iSetParams(context, scale.data(), bias.data(), stats);
            SimdSynetScale8iForward(context, src.data(), dst2.data());
            if (scale8i.InternalBufferSize() != SimdSynetScale8iInternalBufferSize(context))
                std::cout << "TestSynetScale8i is failed : InternalBufferSize mismatch" << std::endl;
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetScale8i is failed at " << i << " : " << (int)dst1[i] << " != " << (int)dst2[i] << std::endl;
        }
    }

    static void TestSynetScale16b()
    {
        const size_t channels = 4, spatial = 16;
        const size_t size = channels * spatial;
        std::vector<float> src(size), dst1(size, 0.0f), dst2(size, 0.0f);
        std::vector<float> norm(channels), bias(channels);
        for (size_t i = 0; i < src.size(); ++i)
            src[i] = float(i) * 0.01f;
        for (size_t i = 0; i < channels; ++i)
        {
            norm[i] = 0.5f + 0.1f * float(i);
            bias[i] = 0.1f * float(i);
        }

        Simd::SynetScale16b scale16b;
        scale16b.Init(channels, spatial, SimdTensorData32f, SimdTensorData32f, SimdTensorFormatNhwc, SimdTrue, SimdTrue);
        if (scale16b.Enable())
            scale16b.Forward((const uint8_t*)src.data(), norm.data(), bias.data(), (uint8_t*)dst1.data());

        void* context = SimdSynetScale16bInit(channels, spatial, SimdTensorData32f, SimdTensorData32f, SimdTensorFormatNhwc, SimdTrue, SimdTrue);
        if (context)
        {
            SimdSynetScale16bForward(context, (const uint8_t*)src.data(), norm.data(), bias.data(), (uint8_t*)dst2.data());
            SimdRelease(context);
        }

        for (size_t i = 0; i < dst1.size(); ++i)
        {
            if (dst1[i] != dst2[i])
                std::cout << "TestSynetScale16b is failed at " << i << " : " << dst1[i] << " != " << dst2[i] << std::endl;
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
        TestSynetPermute();
        TestSynetInnerProduct32f();
        TestSynetInnerProduct16b();
        TestSynetQuantizedInnerProduct();
        TestSynetConvolution32f();
        TestSynetConvolution16b();
        TestSynetConvolution8i();
        TestSynetQuantizedConvolution();
        TestSynetDeconvolution32f();
        TestSynetDeconvolution16b();
        TestSynetMergedConvolution32f();
        TestSynetMergedConvolution16b();
        TestSynetMergedConvolution8i();
        TestSynetQuantizedMergedConvolution();
        TestSynetScale8i();
        TestSynetScale16b();
#endif
    }
}


