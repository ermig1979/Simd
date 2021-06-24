/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        using Convert8uTo32fPtr = Base::SynetMergedConvolution8i::Convert8uTo32fPtr;
        using Convert32fTo8uPtr = Base::SynetMergedConvolution8i::Convert32fTo8uPtr;

        //---------------------------------------------------------------------

        SIMD_INLINE void Cvt8uTo32f(const uint8_t* src, const float* scale, const float* shift, float * dst)
        {
            __m128 f32 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)src)));
            _mm_storeu_ps(dst, _mm_add_ps(_mm_mul_ps(f32, _mm_loadu_ps(scale)), _mm_loadu_ps(shift)));
        }

        void Convert8uTo32f(const uint8_t* src, size_t maC, size_t yBeg, size_t yEnd, size_t width, size_t channels,
            const float* scale, const float* shift, float* dst, size_t bufH, SimdSynetCompatibilityType compatibility)
        {
            size_t dM = bufH - 1, cD = width* bufH;
            src += yBeg * width * channels;
            for (size_t y = yBeg; y < yEnd; ++y)
            {
                float* pd = dst + (y & dM) * width * F;
                for (size_t x = 0; x < width; ++x)
                {
                    for (size_t c = 0; c < maC; c += F)
                        Cvt8uTo32f(src + c, scale + c, shift + c, pd + c * cD);
                    src += channels;
                    pd += F;
                }
            }
        }

        //---------------------------------------------------------------------

        void Convert32fTo8u(const float* src, size_t yBeg, size_t yEnd, size_t width, size_t channels,
            const float* scale, const float* shift, uint8_t* dst, size_t bufH, SimdSynetCompatibilityType compatibility)
        {
            size_t size = width * channels, mask = bufH - 1;
            size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, bufH));
            if (yInt > yBeg)
                Sse2::SynetConvert32fTo8u(src + yBeg * size, 1, channels, yInt - yBeg, width, SimdTensorFormatNhwc, scale, shift, dst + (yBeg & mask) * size, compatibility);
            if (yEnd > yInt)
                Sse2::SynetConvert32fTo8u(src + yInt * size, 1, channels, yEnd - yInt, width, SimdTensorFormatNhwc, scale, shift, dst + (yInt & mask) * size, compatibility);
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iCdc::SynetMergedConvolution8iCdc(const MergConvParam8i& p)
            : Base::SynetMergedConvolution8iCdc(p)
        {
            SetSize(Sse2::F);
            _cvt32fTo8u = _s8u ? NULL : Convert32fTo8u;
            SetInput(_param.conv[0], _input);
            SetDepthwise(_param.conv[1], _depthwise);
            SetOutput(_param.conv[2], _output);
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iCd::SynetMergedConvolution8iCd(const MergConvParam8i& p)
            : Base::SynetMergedConvolution8iCd(p)
        {
            SetSize(Sse2::F);
            _cvt32fTo8u = _s8u ? NULL : Convert32fTo8u;
            SetInput(_param.conv[0], _input);
            SetDepthwise(_param.conv[1], _depthwise);
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iDc::SynetMergedConvolution8iDc(const MergConvParam8i& p)
            : Base::SynetMergedConvolution8iDc(p)
        {
            SetSize(Sse2::F);
            _cvt8uTo32f = _s8u ? Convert8uTo32f : NULL;
            SetDepthwise(_param.conv[0], _depthwise);
            SetOutput(_param.conv[1], _output);
        }

        //---------------------------------------------------------------------

        void* SynetMergedConvolution8iInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdSynetCompatibilityType compatibility)
        {
            MergConvParam8i param(batch, convs, count, compatibility);
            if (!param.Valid())
                return NULL;
            if (SynetMergedConvolution8iCdc::Preferable(param))
                return new Sse41::SynetMergedConvolution8iCdc(param);
            else if (SynetMergedConvolution8iCd::Preferable(param))
                return new Sse41::SynetMergedConvolution8iCd(param);
            else if (SynetMergedConvolution8iDc::Preferable(param))
                return new Sse41::SynetMergedConvolution8iDc(param);
            else
                return new Base::SynetMergedConvolution8i(param);
        }
    }
#endif
}
