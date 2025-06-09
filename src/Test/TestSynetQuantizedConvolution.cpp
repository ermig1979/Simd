/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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

#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        typedef Test::SynetConvolutionParam<false> Param;

        struct FuncQC
        {
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * conv);

            FuncPtr func;
            String desc;

            FuncQC(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Param & p)
            {
                const char* afs[] = { "-id", "-re", "-lr", "-rr", "-pr", "-el", "-hs", "-mi", "-hi", "-sw", "-ge" };
                std::stringstream extra;
                extra << (p.conv.srcT == SimdTensorData32f ? "-f" : "-u");
                extra << (p.conv.dstT == SimdTensorData32f ? "f" : "u");
                extra << afs[p.conv.activation];
                desc = desc + p.Decription(extra.str());
            }

            void Call(void * context, const uint8_t * src, uint8_t * buf, uint8_t * dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetQuantizedConvolutionForward(context, src, buf, dst);
            }
        };
    }

#define FUNC_QC(function) \
    FuncQC(function, std::string(#function))

    struct QcParams32f
    {
        Tensor32f src, weight, bias, params, dst1, dst2;

        bool Init(Param p)
        {
            p.conv.srcT = SimdTensorData32f;
            p.conv.dstT = SimdTensorData32f;

            src.Reshape(p.SrcShape());
            FillRandom(src, -0.9, 1.1f);

            weight.Reshape(p.WeightShape());
            FillRandom(weight, -1.1, 1.0f);

            bias.Reshape(Shp(p.conv.dstC));
            FillRandom(bias, -1.1, 1.2f);

            params.Reshape(Shp(p.conv.dstC));
            FillRandom(params, 0.0f, 2.0f);
            if (p.conv.activation == ::SimdConvolutionActivationHswish)
            {
                params.Data()[0] = 3.0f;
                params.Data()[1] = 1.0f / 6.0f;
            }
            else if (p.conv.activation == ::SimdConvolutionActivationMish)
                params.Data()[0] = 20.0f;
            else if (p.conv.activation == ::SimdConvolutionActivationHardSigmoid)
            {
                params.Data()[0] = 1.0f / 6.0f;
                params.Data()[1] = 0.5f;
            }
            else
            {
                params.Data()[0] = 0.1f;
                params.Data()[1] = 1.1f;
            }

            dst1.Reshape(p.DstShape());

            void* context = ::SimdSynetConvolution32fInit(p.batch, &p.conv);
            if (context == NULL)
                return false;

            Tensor32f buf;
            buf.Extend({ ::SimdSynetConvolution32fExternalBufferSize(context) });

            ::SimdSynetConvolution32fSetParams(context, weight.Data(), NULL, bias.Data(), params.Data());

            ::SimdSynetConvolution32fForward(context, src.Data(), buf.Data(), dst1.Data());

            ::SimdRelease(context);

            dst2.Reshape(p.DstShape());

            return true;
        }
    };

    struct QcParams8i
    {
        Tensor8u src, dst1, dst2, srcZero, dstZero;
        Tensor8i weight;
        Tensor32i bias;
        Tensor32f weightScale, srcScale, dstScale, norm;

        bool Init(const Param & p, const QcParams32f & f32, SimdBool overflow, SimdBool uniform)
        {
            bool trans = p.conv.srcF == SimdTensorFormatNhwc;

            weight.Reshape(p.WeightShape());

            bias.Reshape(Shp(p.conv.dstC));
            norm.Reshape(Shp(p.conv.dstC));

            if (!QuantizeSrcDst(f32.src, trans, uniform, src, srcZero, srcScale))
                return false;
            if (!QuantizeSrcDst(f32.dst1, trans, uniform, dst1, dstZero, dstScale))
                return false;

            if (!QuantizeWeight(f32.weight, trans, overflow, weight, weightScale))
                return false;

            Tensor32f scaleBias;
            Tensor32i iBias;
            if (!QuantizeBias(f32.bias, srcScale.Data()[0], weightScale, iBias, scaleBias))
                return false;

            if (!SetNorm(srcScale.Data()[0], weightScale, dstScale.Data()[0], norm))
                return false;

            if (!SetBias(weight, iBias, srcZero.Data()[0], trans, bias))
                return false;

            dst2.Reshape(p.DstShape(), p.conv.dstF);
            //Copy(dst1, dst2);

            return true;
        }

    protected:
        static bool QuantizeSrcDst(const Tensor32f& src, bool trans, SimdBool uniform, Tensor8u& dst, Tensor8u& zero, Tensor32f& scale)
        {
            size_t batch = src.Axis(0), size = src.Size();
            size_t channels = trans ? src.Axis(3) : src.Axis(1);
            size_t spatial = trans ? src.Size(1, 3) : src.Size(2, 4);
            dst.Reshape(src.Shape());
            zero.Reshape(Shp(channels));
            scale.Reshape(Shp(channels));
            if (uniform)
            {
                float min = FLT_MAX, max = -FLT_MAX;
                const float* psrc = src.Data();
                for (size_t i = 0; i < size; ++i)
                {
                    min = std::min(min, psrc[i]);
                    max = std::max(max, psrc[i]);
                }
                float range = std::max(0.000001f, max - min);
                float _scale = range / 255.0f, invScale = 255.0f / range;
                Fill(scale, _scale);
                uint8_t _zero = (int)std::nearbyint(min * invScale);
                Fill(zero, _zero);
                uint8_t* pdst = dst.Data();
                for (size_t i = 0; i < size; ++i)
                    pdst[i] = Simd::RestrictRange((int)std::nearbyint(psrc[i] * invScale) + _zero, 0, 255);
            }
            else
                return false;
            return true;
        }

        static bool QuantizeWeight(const Tensor32f& src, bool trans, SimdBool overflow, Tensor8i& dst, Tensor32f& scale)
        {
            size_t size = src.Size(), D = trans ? src.Axis(3) : src.Axis(0), CK = size / D;
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
                    size_t offset = trans ? ck * D + d : d * CK + ck;
                    max = std::max(max, std::abs(psrc[offset]));
                }
                float range = std::max(0.000001f, max);
                float _scale =  range / (overflow ? 63.0f : 127.0f), invScale = (overflow ? 63.0f : 127.0f) / range;
                scale.Data()[d] = _scale;
                for (size_t ck = 0; ck < CK; ++ck)
                {
                    size_t offset = trans ? ck * D + d : d * CK + ck;
                    pdst[offset] = Simd::RestrictRange((int)std::nearbyint(psrc[offset] * invScale), lo, hi);
                }
            }
            return true;
        }

        static bool QuantizeBias(const Tensor32f& src, float sSrc, const Tensor32f& weightScale, Tensor32i& dst, Tensor32f& scale)
        {
            size_t size = src.Size();
            dst.Reshape(src.Shape());
            scale.Reshape(src.Shape());
            const float* pws = weightScale.Data();
            const float* psrc = src.Data();
            int32_t* pdst = dst.Data();
            for (size_t i = 0; i < size; ++i)
            {
                float _scale = sSrc * pws[i], invScale = 1.0f / _scale;
                scale.Data()[i] = _scale;
                pdst[i] = (int)std::nearbyint(psrc[i] * invScale);
            }
            return true;
        }

        static bool SetNorm(float sSrc, const Tensor32f& sWeight, float sDst, Tensor32f& norm)
        {
            size_t size = sWeight.Size();
            norm.Reshape(sWeight.Shape());
            const float* psw = sWeight.Data();
            float* pn = norm.Data();
            for (size_t i = 0; i < size; ++i)
                pn[i] = sSrc * psw[i] / sDst;
            return true;
        }

        static bool SetBias(const Tensor8i& weight, const Tensor32i& bias, int srcZero, bool trans, Tensor32i& dst)
        {
            size_t size = weight.Size(), D = trans ? weight.Axis(3) : weight.Axis(0), CK = size / D;
            dst.Reshape(Shp(D));
            const int8_t* pw = weight.Data();
            const int32_t* pb = bias.Data();
            int32_t* pdst = dst.Data();
            for (size_t d = 0; d < D; ++d)
            {
                pdst[d] = pb[d];
                for (size_t ck = 0; ck < CK; ++ck)
                {
                    size_t offset = trans ? ck * D + d : d * CK + ck;
                    pdst[d] -= pw[offset] * srcZero;
                }
            }
            return true;
        }
    };

    bool SynetQuantizedConvolutionForwardAutoTest(float eps, Param p, SimdBool uniform, SimdBool overflow, FuncQC f1, FuncQC f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        QcParams32f p32f;
        if (!p32f.Init(p))
            return false;

        QcParams8i p8i;
        if (!p8i.Init(p, p32f, overflow, uniform))
            return false;

        void * context1 = f1.func(p.batch, &p.conv);
        void * context2 = f2.func(p.batch, &p.conv);

        Tensor8u buf8u;
        buf8u.Extend({ ::SimdSynetQuantizedConvolutionExternalBufferSize(context1) });
        buf8u.Extend({ ::SimdSynetQuantizedConvolutionExternalBufferSize(context2) });

        ::SimdSynetQuantizedConvolutionSetParams(context1, p8i.weight.Data(), p8i.bias.Data(), p8i.norm.Data(), p8i.srcZero.Data(), p8i.dstZero.Data());
        ::SimdSynetQuantizedConvolutionSetParams(context2, p8i.weight.Data(), p8i.bias.Data(), p8i.norm.Data(), p8i.srcZero.Data(), p8i.dstZero.Data());

        const uint8_t * src = p.conv.srcT == SimdTensorData32f ? (uint8_t*)p32f.src.Data() : p8i.src.Data();
        uint8_t* dst1 = p.conv.dstT == SimdTensorData32f ? (uint8_t*)p32f.dst1.Data() : p8i.dst1.Data();
        uint8_t* dst2 = p.conv.dstT == SimdTensorData32f ? (uint8_t*)p32f.dst2.Data() : p8i.dst2.Data();

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf8u.Data(), dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf8u.Data(), dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        int differenceMax = 0;

        if(p.conv.dstT == SimdTensorData32f)
            result = result && Compare(p32f.dst1, p32f.dst2, eps, true, 64, DifferenceBoth);
        else
            result = result && Compare(p8i.dst1, p8i.dst2, differenceMax, true, 64);

        return result;
    }

    bool SynetQuantizedConvolutionForwardAutoTest(SimdBool o, const FuncQC& f1, const FuncQC& f2)
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
        result = result && SynetQuantizedConvolutionForwardAutoTest(e, Param(1, 177, 31, 41, 155, _3, _1, _1, _1, _1, 1, aId, f, u8, u8), t, o, f1, f2);
#endif
#else
        result = result && SynetQuantizedConvolutionForwardAutoTest(e, Param(1, 160, 32, 42, 156, _3, _1, _1, _1, _1, 1, aId, f, u8, u8), t, o, f1, f2);
#endif

        return result;
    }

    bool SynetQuantizedConvolutionForwardAutoTest()
    {
        bool result = true;

        const SimdBool f = SimdFalse, t = SimdTrue;

        if (TestBase())
            result = result && SynetQuantizedConvolutionForwardAutoTest(t, FUNC_QC(Simd::Base::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));

//#ifdef SIMD_SSE41_ENABLE
//        if (Simd::Sse41::Enable && TestSse41())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Sse41::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif 
//
//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable && TestAvx2())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Avx2::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable && TestAvx512bw())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Avx512bw::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif
//
//#if defined(SIMD_AVX512VNNI_ENABLE) && !defined(SIMD_AMX_EMULATE)
//        if (Simd::Avx512vnni::Enable && TestAvx512vnni())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Avx512vnni::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif
//
//#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
//        if (Simd::AmxBf16::Enable && TestAmxBf16())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::AmxBf16::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif
//
//#ifdef SIMD_NEON_ENABLE
//        if (Simd::Neon::Enable && TestNeon())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Neon::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif 

        return result;
    }
#endif
}
