/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Test/TestUtils.h"
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestTensor.h"
#include "Test/TestSynetConvolutionParam.h"
#include "Test/TestRandom.h"
#include "Test/TestString.h"

#include "Simd/SimdSynetQuantizedMergedConvolution.h"
#include "Simd/SimdSynet.h"

#include "Simd/SimdMath.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        typedef MergeConvParam Param;

        struct FuncQMC
        {
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters* params, size_t count, SimdBool add);

            FuncPtr func;
            String desc;

            FuncQMC(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Param & p)
            {
                std::stringstream ss;
                ss << desc;
                ss << "[" << p.count << ":" << p.batch << "x" << p.conv[0].srcC << "x" << p.conv[0].srcH << "x" << p.conv[0].srcW;
                for (size_t i = 0; i < p.count; ++i)
                    ss << "-" << (p.conv[i].group != 1 ? String("") : ToString(p.conv[i].dstC) + "x") << p.conv[i].kernelY << "x" << p.conv[i].strideY;
                ss << "-" << (p.count == 3 ? ToString(p.add) : "") << "]";
                desc = ss.str();
            }

            void Call(void * context, const uint8_t * src, uint8_t * buf, uint8_t * dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdSynetQuantizedMergedConvolutionForward(context, src, buf, dst);
            }
        };
    }

#define FUNC_QMC(function) \
    FuncQMC(function, std::string(#function))

    struct QmcParams32f
    {
        Tensor32f image[5], weight[3], bias[3], params[3];

        bool Init(Param p)
        {
            image[0].Reshape(p.SrcShape(0));
            FillRandom(image[0], -0.9, 1.1f);
            for (size_t c = 0; c < p.count; ++c)
            {
                p.conv[c].srcT = SimdTensorData32f;
                p.conv[c].dstT = SimdTensorData32f;
                image[c + 1].Reshape(p.DstShape(c));

                weight[c].Reshape(p.WeightShape(c));
                FillRandom(weight[c], -1.1, 1.0f);

                bias[c].Reshape(Shp(p.conv[c].dstC));
                FillRandom(bias[c], -1.1, 1.2f);

                params[c].Reshape(Shp(p.conv[c].dstC));
                FillRandom(params[c], 0.0f, 2.0f);
                if (p.conv[c].activation == ::SimdConvolutionActivationHswish)
                {
                    params[c].Data()[0] = 3.0f;
                    params[c].Data()[1] = 1.0f / 6.0f;
                }
                else if (p.conv[c].activation == ::SimdConvolutionActivationMish)
                    params[c].Data()[0] = 20.0f;
                else if (p.conv[c].activation == ::SimdConvolutionActivationHardSigmoid)
                {
                    params[c].Data()[0] = 1.0f / 6.0f;
                    params[c].Data()[1] = 0.5f;
                }
                else
                {
                    params[c].Data()[0] = 0.1f;
                    params[c].Data()[1] = 1.1f;
                }

                void* context = SimdSynetConvolution32fInit(p.batch, &p.conv[c]);
                if (context == NULL)
                    return false;

                Tensor32f buf;
                buf.Extend({ SimdSynetConvolution32fExternalBufferSize(context) });

                SimdSynetConvolution32fSetParams(context, weight[c].Data(), NULL, bias[c].Data(), params[c].Data());

                SimdSynetConvolution32fForward(context, image[c + 0].Data(), buf.Data(), image[c + 1].Data());

                SimdRelease(context);
            }
            if (p.add)
            {
                image[4].Clone(image[3]);
                SimdNeuralAddVector(image[0].Data(), image[0].Size(), image[4].Data());
            }
            return true;
        }
    };

    struct QmcParams8i
    {
        Tensor8u image[5], dst1, dst2;
        Tensor8i weight[3];
        Tensor32i bias[3];
        uint8_t imageZero[5];
        float imageScale[5], *ptrWS[3];
        Tensor32f weightScale[3];
        size_t last;
        int8_t* ptrW[3];
        int32_t* ptrB[3];

        bool Init(const Param & p, const QmcParams32f & f32, SimdBool overflow)
        {
            last = p.count + (p.add ? 1 : 0);
            for(size_t i = 0; i <= last; ++i)
                if (!QuantizeImage(f32.image[i], image[i], imageZero[i], imageScale[i]))
                    return false;

            for (size_t c = 0; c < p.count; ++c)
            {
                if (!QuantizeWeight(f32.weight[c], overflow, weight[c], weightScale[c]))
                    return false;
                if (!QuantizeBias(f32.bias[c], imageScale[c], weightScale[c], bias[c]))
                    return false;
                ptrW[c] = weight[c].Data();
                ptrB[c] = bias[c].Data();
                ptrWS[c] = weightScale[c].Data();
            }

            dst1.Reshape(p.DstShape(p.count - 1), p.conv[p.count - 1].dstF);
            dst2.Reshape(p.DstShape(p.count - 1), p.conv[p.count - 1].dstF);

            return true;
        }

    protected:
        static bool QuantizeImage(const Tensor32f& src, Tensor8u& dst, uint8_t& zero, float& scale)
        {
            size_t batch = src.Axis(0), size = src.Size();
            size_t channels = src.Axis(3);
            size_t spatial = src.Size(1, 3);
            dst.Reshape(src.Shape());
            float min = 0.0f, max = 0.0f;
            const float* psrc = src.Data();
            for (size_t i = 0; i < size; ++i)
            {
                min = std::min(min, psrc[i]);
                max = std::max(max, psrc[i]);
            }
            float range = std::max(0.000001f, max - min), invScale = 255.0f / range;
            scale = range / 255.0f;
            zero = -(int)std::nearbyint(min * invScale);
            uint8_t* pdst = dst.Data();
            for (size_t i = 0; i < size; ++i)
                pdst[i] = Simd::RestrictRange((int)std::nearbyint(psrc[i] * invScale) + zero, 0, 255);
            return true;
        }

        static bool QuantizeWeight(const Tensor32f& src, SimdBool overflow, Tensor8i& dst, Tensor32f& scale)
        {
            size_t size = src.Size(), D = src.Axis(3), CK = size / D;
            dst.Reshape(src.Shape());
            scale.Reshape(Shp(D));
            const float* psrc = src.Data();
            int8_t* pdst = dst.Data();
            int lo = overflow ? -64 : -128, hi = overflow ? 63 : 127;
            for (size_t d = 0; d < D; ++d)
            {
                float max = 0;
                for (size_t ck = 0; ck < CK; ++ck)
                {
                    size_t offset = ck * D + d;
                    max = std::max(max, std::abs(psrc[offset]));
                }
                float range = std::max(0.000001f, max);
                float _scale =  range / (overflow ? 63.0f : 127.0f), invScale = (overflow ? 63.0f : 127.0f) / range;
                scale.Data()[d] = _scale;
                for (size_t ck = 0; ck < CK; ++ck)
                {
                    size_t offset = ck * D + d;
                    pdst[offset] = Simd::RestrictRange((int)std::nearbyint(psrc[offset] * invScale), lo, hi);
                }
            }
            return true;
        }

        static bool QuantizeBias(const Tensor32f& src, float srcScale, const Tensor32f& weightScale, Tensor32i& dst)
        {
            size_t size = src.Size();
            dst.Reshape(src.Shape());
            const float* psrc = src.Data();
            const float* pws = weightScale.Data();
            int32_t* pdst = dst.Data();
            for (size_t i = 0; i < size; ++i)
            {
                float invScale = 1.0f / (srcScale * pws[i]);
                pdst[i] = (int)std::nearbyint(psrc[i] * invScale);
            }
            return true;
        }
    };

    bool SynetQuantizedMergedConvolutionForwardAutoTest(float eps, Param p, SimdBool overflow, FuncQMC f1, FuncQMC f2)
    {
        bool result = true;

        for (size_t c = 0; c < p.count; ++c)
        {
            p.conv[c].srcT = SimdTensorData8u;
            p.conv[c].dstT = SimdTensorData8u;
        }

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        QmcParams32f p32f;
        if (!p32f.Init(p))
            return false;

        QmcParams8i p8i;
        if (!p8i.Init(p, p32f, overflow))
            return false;

        void * context1 = f1.func(p.batch, p.conv, p.count, p.add);
        void * context2 = f2.func(p.batch, p.conv, p.count, p.add);

        Tensor8u buf8u;
        buf8u.Extend({ ::SimdSynetQuantizedMergedConvolutionExternalBufferSize(context1) });
        buf8u.Extend({ ::SimdSynetQuantizedMergedConvolutionExternalBufferSize(context2) });

        ::SimdSynetQuantizedMergedConvolutionSetParams(context1, p8i.imageScale, p8i.imageZero, p8i.ptrW, p8i.ptrWS, p8i.ptrB);
        ::SimdSynetQuantizedMergedConvolutionSetParams(context2, p8i.imageScale, p8i.imageZero, p8i.ptrW, p8i.ptrWS, p8i.ptrB);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, p8i.image[0].Data(), buf8u.Data(), p8i.dst1.Data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, p8i.image[0].Data(), buf8u.Data(), p8i.dst2.Data()));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        int diffMax = 0;
        result = result && Compare(p8i.dst1, p8i.dst2, diffMax, true, 64);

        int controlDiffMax = 4;
        result = result && Compare(p8i.dst1, p8i.image[p8i.last], controlDiffMax, true, 64, "control");

        return result;
    }

    bool SynetQuantizedMergedConvolutionForwardAutoTest(SimdBool o, const FuncQMC& f1, const FuncQMC& f2)
    {
        bool result = true;

        const Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _7(7, 7);
        const float e = EPS;
        const SimdBool f = SimdFalse, t = SimdTrue;
        const SimdTensorDataType f32 = SimdTensorData32f, u8 = SimdTensorData8u;
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu, 
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu, 
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;

#ifdef NDEBUG
#if 1
        result = result && SynetQuantizedMergedConvolutionForwardAutoTest(e, Param(Shp(1, 16, 112, 112), Cnv(aId, 1, 1, 96), Cnv(aId, 3, 2), Cnv(aId, 1, 1, 24), f, u8, u8), o, f1, f2);
#endif
#else
        result = result && SynetQuantizedMergedConvolutionForwardAutoTest(e, Param(Shp(1, 16, 112, 112), Cnv(aId, 1, 1, 96), Cnv(aId, 3, 2), Cnv(aId, 1, 1, 24), f, u8, u8), o, f1, f2);
#endif

        return result;
    }

    bool SynetQuantizedMergedConvolutionForwardAutoTest()
    {
        bool result = true;

        const SimdBool f = SimdFalse, t = SimdTrue;

        if (TestBase())
            result = result && SynetQuantizedMergedConvolutionForwardAutoTest(t, FUNC_QMC(Simd::Base::SynetQuantizedMergedConvolutionInit), FUNC_QMC(SimdSynetQuantizedMergedConvolutionInit));

//#ifdef SIMD_SSE41_ENABLE
//        if (Simd::Sse41::Enable && TestSse41())
//            result = result && SynetQuantizedMergedConvolutionForwardAutoTest(t, FUNC_QMC(Simd::Sse41::SynetQuantizedMergedConvolutionInit), FUNC_QMC(SimdSynetQuantizedMergedConvolutionInit));
//#endif 
//
//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable && TestAvx2())
//            result = result && SynetQuantizedMergedConvolutionForwardAutoTest(t, FUNC_QMC(Simd::Avx2::SynetQuantizedMergedConvolutionInit), FUNC_QMC(SimdSynetQuantizedMergedConvolutionInit));
//#endif
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable && TestAvx512bw())
//            result = result && SynetQuantizedMergedConvolutionForwardAutoTest(t, FUNC_QMC(Simd::Avx512bw::SynetQuantizedMergedConvolutionInit), FUNC_QMC(SimdSynetQuantizedMergedConvolutionInit));
//#endif
//
//#if defined(SIMD_AVX512VNNI_ENABLE) && !defined(SIMD_AMX_EMULATE)
//        if (Simd::Avx512vnni::Enable && TestAvx512vnni())
//            result = result && SynetQuantizedMergedConvolutionForwardAutoTest(f, FUNC_QMC(Simd::Avx512vnni::SynetQuantizedMergedConvolutionInit), FUNC_QMC(SimdSynetQuantizedMergedConvolutionInit));
//#endif
//
//#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
//        if (Simd::AmxBf16::Enable && TestAmxBf16())
//            result = result && SynetQuantizedMergedConvolutionForwardAutoTest(f, FUNC_QMC(Simd::AmxBf16::SynetQuantizedMergedConvolutionInit), FUNC_QMC(SimdSynetQuantizedMergedConvolutionInit));
//#endif

        return result;
    }
#endif
}
