/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#if defined(SIMD_SSE41_ENABLE)
    namespace Sse41
    {
        using AlgParam = Base::SynetMergedConvolution8i::AlgParam;
        using Convert8uTo32fPtr = Base::SynetMergedConvolution8i::Convert8uTo32fPtr;
        using Convert32fTo8uPtr = Base::SynetMergedConvolution8i::Convert32fTo8uPtr;
        using InputConvolutionPtr = Base::SynetMergedConvolution8i::InputConvolutionPtr;
        using DepthwiseConvolutionPtr = Base::SynetMergedConvolution8i::DepthwiseConvolutionPtr;
        using OutputConvolutionPtr = Base::SynetMergedConvolution8i::OutputConvolutionPtr;

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

        template<SimdConvolutionActivationType type>
        SIMD_INLINE void SaveInput1(float* dst, __m128i sum, const __m128* norm, const __m128* bias, const __m128* params)
        {
            _mm_storeu_ps((float*)dst, Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), norm[0]), bias[0]), params, 0));
        }

        template<SimdConvolutionActivationType type>
        SIMD_INLINE void SaveInput2(float* dst0, float* dst1, __m128i sum0, __m128i sum1, const __m128* norm, const __m128* bias, const __m128* params)
        {
            _mm_storeu_ps(dst0, Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum0), norm[0]), bias[0]), params, 0));
            _mm_storeu_ps(dst1, Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum1), norm[1]), bias[1]), params, 1));
        }

        template<bool overflow, SimdConvolutionActivationType type> void InputConvolution_2x1(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t dstC, const int8_t* weight,
            const __m128* norm, const __m128* bias, const __m128* params, float* dst0, float* dst1)
        {
            __m128i d00, d01, s0, w0, w1;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dWz = DivHi(p.srcC, 4) * DA, sM = a.bufH[0] - 1;
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                d00 = _mm_setzero_si128(), d01 = _mm_setzero_si128();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                w1 = _mm_loadu_si128((__m128i*)weight + 1);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                Madd4<overflow>(d01, s0, w1);
                                weight += DA;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = _mm_set1_epi32(a.zero);
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    w1 = _mm_loadu_si128((__m128i*)weight + 1);
                                    Madd4<overflow>(d00, s0, w0);
                                    Madd4<overflow>(d01, s0, w1);
                                    weight += DA;
                                }
                            }
                            else
                                weight += dWz;
                        }
                    }
                }
                SaveInput2<type>(dst0, dst1, d00, d01, norm, bias, params);
            }
            else
            {
                d00 = _mm_setzero_si128();
                for (size_t ky = 0; ky < kY; ky += p.dilationY)
                {
                    for (size_t kx = 0; kx < kX; kx += p.dilationX)
                    {
                        if (sy + ky < p.srcH && sx + kx < p.srcW)
                        {
                            size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                            for (; offs < end; offs += 4)
                            {
                                w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                s0 = Set4(src0 + offs);
                                Madd4<overflow>(d00, s0, w0);
                                weight += DA;
                            }
                        }
                        else
                        {
                            if (a.zero)
                            {
                                s0 = _mm_set1_epi32(a.zero);
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    Madd4<overflow>(d00, s0, w0);
                                    weight += DA;
                                }
                            }
                            else
                                weight += dWz;
                        }
                    }
                }
                SaveInput1<type>(dst0, d00, norm, bias, params);
            }
        }

        typedef void(*InputConvolution_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx,
            size_t dstC, const int8_t* weight, const __m128* norm, const __m128* bias, const __m128* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution_2xM_Ptr GetInputConvolution_2x1(const ConvParam8i& p)
        {
            if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                return InputConvolution_2x1<true, type>;
            else
                return InputConvolution_2x1<false, type>;
        }

        template<SimdConvolutionActivationType type, int M> void InputConvolution_2xM(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, size_t dstC, const int8_t* weight,
            const __m128* norm, const __m128* bias, const __m128* params, float* dst0, float* dst1)
        {
            __m128i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dY = p.srcW * p.srcC, dX = p.srcC, dS = p.srcC * p.strideX, dD = p.dstC * a.size, dWz = DivHi(p.srcC, 4) * DA * p.kernelX, sM = a.bufH[0] - 1;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            __m128i upper = _mm_set1_epi32(a.upper);
            size_t sy = dy * p.strideY - p.padY;
            size_t sx = dx * p.strideX - p.padX;
            size_t kY = p.kernelY * p.dilationY;
            size_t kX = p.kernelX * p.dilationX;
            if (dstC > F)
            {
                if (M > 0) d00 = _mm_setzero_si128(), d01 = _mm_setzero_si128();
                if (M > 1) d10 = _mm_setzero_si128(), d11 = _mm_setzero_si128();
                if (M > 2) d20 = _mm_setzero_si128(), d21 = _mm_setzero_si128();
                if (M > 3) d30 = _mm_setzero_si128(), d31 = _mm_setzero_si128();
                if (M > 4) d40 = _mm_setzero_si128(), d41 = _mm_setzero_si128();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                                for (; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    w1 = _mm_loadu_si128((__m128i*)weight + 1);
                                    if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                                    if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                                    if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                                    if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                                    if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                                    weight += DA;
                                }
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    w1 = _mm_loadu_si128((__m128i*)weight + 1);
                                    if (M > 0) Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                                    if (M > 1) Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                                    if (M > 2) Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                                    if (M > 3) Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                                    if (M > 4) Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                                    weight += DA;
                                }
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                                for (; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    w1 = _mm_loadu_si128((__m128i*)weight + 1);
                                    if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                                    if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                                    if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                                    if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                                    if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                                    weight += DA;
                                }
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    w1 = _mm_loadu_si128((__m128i*)weight + 1);
                                    if (M > 0) Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                                    if (M > 1) Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                                    if (M > 2) Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                                    if (M > 3) Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                                    if (M > 4) Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                                    weight += DA;
                                }
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                if (M > 0) SaveInput2<type>(dst0 + 0 * F, dst1 + 0 * F, d00, d01, norm, bias, params);
                if (M > 1) SaveInput2<type>(dst0 + 1 * F, dst1 + 1 * F, d10, d11, norm, bias, params);
                if (M > 2) SaveInput2<type>(dst0 + 2 * F, dst1 + 2 * F, d20, d21, norm, bias, params);
                if (M > 3) SaveInput2<type>(dst0 + 3 * F, dst1 + 3 * F, d30, d31, norm, bias, params);
                if (M > 4) SaveInput2<type>(dst0 + 4 * F, dst1 + 4 * F, d40, d41, norm, bias, params);
            }
            else
            {
                if (M > 0) d00 = _mm_setzero_si128();
                if (M > 1) d10 = _mm_setzero_si128();
                if (M > 2) d20 = _mm_setzero_si128();
                if (M > 3) d30 = _mm_setzero_si128();
                if (M > 4) d40 = _mm_setzero_si128();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                                for (; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0);
                                    if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0);
                                    if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0);
                                    if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0);
                                    if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0);
                                    weight += DA;
                                }
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    if (M > 0) Madd4<true>(d00, s0, w0);
                                    if (M > 1) Madd4<true>(d10, s0, w0);
                                    if (M > 2) Madd4<true>(d20, s0, w0);
                                    if (M > 3) Madd4<true>(d30, s0, w0);
                                    if (M > 4) Madd4<true>(d40, s0, w0);
                                    weight += DA;
                                }
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                else
                {
                    for (size_t ky = 0; ky < kY; ky += p.dilationY)
                    {
                        if (sy + ky < p.srcH)
                        {
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                assert(sx + kx < p.srcW&& sx + kx + M <= p.srcW);
                                size_t offs = (sM & (sy + ky)) * dY + (sx + kx) * dX, end = offs + p.srcC;
                                for (; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0);
                                    if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0);
                                    if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0);
                                    if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0);
                                    if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0);
                                    weight += DA;
                                }
                            }
                        }
                        else if (a.zero)
                        {
                            s0 = _mm_set1_epi32(a.zero);
                            for (size_t kx = 0; kx < kX; kx += p.dilationX)
                            {
                                for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                                {
                                    w0 = _mm_loadu_si128((__m128i*)weight + 0);
                                    if (M > 0) Madd4<false>(d00, s0, w0);
                                    if (M > 1) Madd4<false>(d10, s0, w0);
                                    if (M > 2) Madd4<false>(d20, s0, w0);
                                    if (M > 3) Madd4<false>(d30, s0, w0);
                                    if (M > 4) Madd4<false>(d40, s0, w0);
                                    weight += DA;
                                }
                            }
                        }
                        else
                            weight += dWz;
                    }
                }
                if (M > 0) SaveInput1<type>(dst0 + 0 * F, d00, norm, bias, params);
                if (M > 1) SaveInput1<type>(dst0 + 1 * F, d10, norm, bias, params);
                if (M > 2) SaveInput1<type>(dst0 + 2 * F, d20, norm, bias, params);
                if (M > 3) SaveInput1<type>(dst0 + 3 * F, d30, norm, bias, params);
                if (M > 4) SaveInput1<type>(dst0 + 4 * F, d40, norm, bias, params);
            }
        }

        template<SimdConvolutionActivationType type> InputConvolution_2xM_Ptr GetInputConvolution_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return InputConvolution_2xM<type, 1>;
            case 2: return InputConvolution_2xM<type, 2>;
            case 3: return InputConvolution_2xM<type, 3>;
            case 4: return InputConvolution_2xM<type, 4>;
            case 5: return InputConvolution_2xM<type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution_2(const uint8_t* src, const ConvParam8i& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const int8_t* weight, const float* norm, const float* bias, const float* params, float* dst)
        {
            size_t noseW = p.NoseW(), bodyW = p.BodyW(), tailW = p.dstW;
            size_t n = 5, bodyWn = AlignLoAny(bodyW - noseW, n) + noseW, m = bodyW - bodyWn;
            size_t dstM = (a.bufH[1] - 1), dstS = a.bufH[1] * p.dstW * F;
            InputConvolution_2xM_Ptr inputConvolution_2x1 = GetInputConvolution_2x1<type>(p);
            InputConvolution_2xM_Ptr inputConvolution_2xN = GetInputConvolution_2xM<type>(n);
            InputConvolution_2xM_Ptr inputConvolution_2xM = GetInputConvolution_2xM<type>(m);
            __m128 _bias[2], _norm[2], _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            _params[1] = _mm_set1_ps(params[1]);
            for (size_t dc = 0; dc < maC; dc += DF)
            {
                size_t dC = Simd::Min(DF, maC - dc);
                _norm[0] = _mm_loadu_ps(norm + dc + 0);
                _norm[1] = _mm_loadu_ps(norm + dc + F);
                _bias[0] = _mm_loadu_ps(bias + dc + 0);
                _bias[1] = _mm_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm_loadu_ps(params + dc + 0);
                    _params[1] = _mm_loadu_ps(params + dc + F);
                }
                for (size_t dy = yBeg; dy < yEnd; dy++)
                {
                    float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                    size_t dx = 0;
                    for (; dx < noseW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                    for (; dx < bodyWn; dx += n, dst0 += F * n, dst1 += F * n)
                        inputConvolution_2xN(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                    for (; dx < bodyW; dx += m, dst0 += F * m, dst1 += F * m)
                        inputConvolution_2xM(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                    for (; dx < tailW; dx += 1, dst0 += F, dst1 += F)
                        inputConvolution_2x1(src, p, a, dy, dx, dC, weight, _norm, _bias, _params, dst0, dst1);
                }
                dst += a.bufH[1] * p.dstW * DF;
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int M> void InputConvolution1x1_2xM(const uint8_t* src0, const ConvParam8i& p,
            const AlgParam& a, size_t dstC, const int8_t* weight, const __m128* norm, const __m128* bias, const __m128* params, float* dst0, float* dst1)
        {
            __m128i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            const uint8_t* src1 = src0 + 1 * p.srcC;
            const uint8_t* src2 = src0 + 2 * p.srcC;
            const uint8_t* src3 = src0 + 3 * p.srcC;
            const uint8_t* src4 = src0 + 4 * p.srcC;
            __m128i upper = _mm_set1_epi32(a.upper);
            if (dstC > F)
            {
                if (M > 0) d00 = _mm_setzero_si128(), d01 = _mm_setzero_si128();
                if (M > 1) d10 = _mm_setzero_si128(), d11 = _mm_setzero_si128();
                if (M > 2) d20 = _mm_setzero_si128(), d21 = _mm_setzero_si128();
                if (M > 3) d30 = _mm_setzero_si128(), d31 = _mm_setzero_si128();
                if (M > 4) d40 = _mm_setzero_si128(), d41 = _mm_setzero_si128();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        w1 = _mm_loadu_si128((__m128i*)weight + 1);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                        weight += DA;
                    }
                }
                else
                {
                    for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        w1 = _mm_loadu_si128((__m128i*)weight + 1);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                        weight += DA;
                    }
                }
                if (M > 0) SaveInput2<type>(dst0 + 0 * F, dst1 + 0 * F, d00, d01, norm, bias, params);
                if (M > 1) SaveInput2<type>(dst0 + 1 * F, dst1 + 1 * F, d10, d11, norm, bias, params);
                if (M > 2) SaveInput2<type>(dst0 + 2 * F, dst1 + 2 * F, d20, d21, norm, bias, params);
                if (M > 3) SaveInput2<type>(dst0 + 3 * F, dst1 + 3 * F, d30, d31, norm, bias, params);
                if (M > 4) SaveInput2<type>(dst0 + 4 * F, dst1 + 4 * F, d40, d41, norm, bias, params);
            }
            else
            {
                if (M > 0) d00 = _mm_setzero_si128();
                if (M > 1) d10 = _mm_setzero_si128();
                if (M > 2) d20 = _mm_setzero_si128();
                if (M > 3) d30 = _mm_setzero_si128();
                if (M > 4) d40 = _mm_setzero_si128();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0);
                        weight += DA;
                    }
                }
                else
                {
                    for (size_t offs = 0, end = p.srcC; offs < end; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0);
                        weight += DA;
                    }
                }
                if (M > 0) SaveInput1<type>(dst0 + 0 * F, d00, norm, bias, params);
                if (M > 1) SaveInput1<type>(dst0 + 1 * F, d10, norm, bias, params);
                if (M > 2) SaveInput1<type>(dst0 + 2 * F, d20, norm, bias, params);
                if (M > 3) SaveInput1<type>(dst0 + 3 * F, d30, norm, bias, params);
                if (M > 4) SaveInput1<type>(dst0 + 4 * F, d40, norm, bias, params);
            }
        }

        typedef void(*InputConvolution1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t dstC,
            const int8_t* weight, const __m128* norm, const __m128* bias, const __m128* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> InputConvolution1x1_2xM_Ptr GetInputConvolution1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return InputConvolution1x1_2xM<type, 1>;
            case 2: return InputConvolution1x1_2xM<type, 2>;
            case 3: return InputConvolution1x1_2xM<type, 3>;
            case 4: return InputConvolution1x1_2xM<type, 4>;
            case 5: return InputConvolution1x1_2xM<type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2(const uint8_t* src, const ConvParam8i& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const int8_t* weight, const float* norm, const float* bias, const float* params, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcM = a.bufH[0] - 1;
            __m128 _bias[2], _norm[2], _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            _params[1] = _mm_set1_ps(params[1]);
            if (a.bufH[0] == 0)
            {
                size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 5;
                size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLoAny(i1, n), i = i1 - in;
                size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLoAny(e1, n), e = e1 - en;
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xI = GetInputConvolution1x1_2xM<type>(i);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xE = GetInputConvolution1x1_2xM<type>(e);
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _norm[0] = _mm_loadu_ps(norm + dc + 0);
                    _norm[1] = _mm_loadu_ps(norm + dc + F);
                    _bias[0] = _mm_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm_loadu_ps(params + dc + 0);
                        _params[1] = _mm_loadu_ps(params + dc + F);
                    }
                    if (yInt > yBeg)
                    {
                        const uint8_t* src0 = src + yBeg * p.srcW * p.srcC;
                        float* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        for (size_t j = 0; j < in; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                        if (in < i1)
                            inputConvolution1x1_2xI(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                    }
                    if (yEnd > yInt)
                    {
                        const uint8_t* src0 = src + yInt * p.srcW * p.srcC;
                        float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        for (size_t j = 0; j < en; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                        if (en < e1)
                            inputConvolution1x1_2xE(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += DivHi(p.srcC, 4) * DA;
                }
            }
            else
            {
                size_t n = 5, bodyW = p.dstW, bodyWn = AlignLoAny(bodyW, n), m = bodyW - bodyWn;
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xN = GetInputConvolution1x1_2xM<type>(n);
                InputConvolution1x1_2xM_Ptr inputConvolution1x1_2xM = GetInputConvolution1x1_2xM<type>(m);
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _norm[0] = _mm_loadu_ps(norm + dc + 0);
                    _norm[1] = _mm_loadu_ps(norm + dc + F);
                    _bias[0] = _mm_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm_loadu_ps(params + dc + 0);
                        _params[1] = _mm_loadu_ps(params + dc + F);
                    }
                    for (size_t dy = yBeg; dy < yEnd; dy++)
                    {
                        const uint8_t* src0 = src + (dy & srcM) * p.srcW * p.srcC;
                        float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        size_t dx = 0;
                        for (; dx < bodyWn; dx += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            inputConvolution1x1_2xN(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                        if (dx < bodyW)
                            inputConvolution1x1_2xM(src0, p, a, dC, weight, _norm, _bias, _params, dst0, dst1);
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += DivHi(p.srcC, 4) * DA;
                }
            }
        }

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const ConvParam8i& p, const AlgParam& a, size_t dstC,
            size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC : p.dstC * a.size), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F : F * a.size;
            size_t wD = p.kernelY * p.kernelX * F, ssX =  strideX * sX;
            size_t noseY = p.NoseH(), bodyY = p.BodyH(), noseX = p.NoseW(), bodyX = p.BodyW();
            size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
            size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
            size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;
            size_t dstCF = AlignLo(dstC, F);

            __m128i _upper = _mm_set1_epi32(a.upper);
            __m128 _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                _params[1] = _mm_set1_ps(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                __m128 _bias = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm_loadu_ps(params + c);
                __m128 _scale = _mm_loadu_ps(scale + c);
                __m128 _shift = _mm_loadu_ps(shift + c);

                if (c == dstCF)
                {
                    size_t tail = dstC - dstCF;
                    for (size_t dy = yBeg; dy < yEnd; ++dy)
                    {
                        uint8_t* pd = dst + (dy - dy0) * dY;
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                        {
                            __m128 sum = _bias;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                if (sy < p.srcH)
                                {
                                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                                    {
                                        size_t sx = dx * strideX + kx - padX;
                                        if (sx < p.srcW)
                                        {
                                            const float* pw = weight + (ky * p.kernelX + kx) * F;
                                            const float* ps = src + (sy & sM) * sY + sx * sX;
                                            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                                        }
                                    }
                                }
                            }
                            Save1<term, type>(pd, sum, _params, _scale, _shift, _upper, tail);
                        }
                    }
                    return;
                }
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    if (dy >= noseY && dy < bodyY)
                    {
                        size_t dx = 0;
                        for (; dx < noseX; dx += 1, pd += dX)
                        {
                            __m128 sum = _bias;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * p.strideY + ky - padY;
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx - padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw = weight + (ky * p.kernelX + kx) * F;
                                        const float* ps = src + (sy & sM) * sY + sx * sX;
                                        sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                                    }
                                }
                            }
                            Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
                        }
                        for (; dx < bodyX8; dx += 8, pd += 8 * dX)
                        {
                            __m128 sum0 = _bias;
                            __m128 sum1 = _bias;
                            __m128 sum2 = _bias;
                            __m128 sum3 = _bias;
                            __m128 sum4 = _bias;
                            __m128 sum5 = _bias;
                            __m128 sum6 = _bias;
                            __m128 sum7 = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m128 w0 = _mm_loadu_ps(pw);
                                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * ssX), w0), sum0);
                                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * ssX), w0), sum1);
                                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * ssX), w0), sum2);
                                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 3 * ssX), w0), sum3);
                                    sum4 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 4 * ssX), w0), sum4);
                                    sum5 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 5 * ssX), w0), sum5);
                                    sum6 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 6 * ssX), w0), sum6);
                                    sum7 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 7 * ssX), w0), sum7);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 2 * dX, sum2, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 3 * dX, sum3, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 4 * dX, sum4, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 5 * dX, sum5, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 6 * dX, sum6, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 7 * dX, sum7, _params, _scale, _shift, _upper);
                        }
                        for (; dx < bodyX4; dx += 4, pd += 4 * dX)
                        {
                            __m128 sum0 = _bias;
                            __m128 sum1 = _bias;
                            __m128 sum2 = _bias;
                            __m128 sum3 = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m128 w0 = _mm_loadu_ps(pw);
                                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * ssX), w0), sum0);
                                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * ssX), w0), sum1);
                                    sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 2 * ssX), w0), sum2);
                                    sum3 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 3 * ssX), w0), sum3);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 2 * dX, sum2, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 3 * dX, sum3, _params, _scale, _shift, _upper);
                        }
                        for (; dx < bodyX2; dx += 2, pd += 2 * dX)
                        {
                            __m128 sum0 = _bias;
                            __m128 sum1 = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m128 w0 = _mm_loadu_ps(pw);
                                    sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 0 * ssX), w0), sum0);
                                    sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps + 1 * ssX), w0), sum1);
                                }
                            }
                            Save1<term, type>(pd + 0 * dX, sum0, _params, _scale, _shift, _upper);
                            Save1<term, type>(pd + 1 * dX, sum1, _params, _scale, _shift, _upper);
                        }
                        for (; dx < bodyX; dx += 1, pd += dX)
                        {
                            __m128 sum = _bias;
                            const float* pw = weight;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                const float* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < p.kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m128 w0 = _mm_loadu_ps(pw);
                                    sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), w0), sum);
                                }
                            }
                            Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
                        }
                        for (; dx < p.dstW; dx += 1, pd += dX)
                        {
                            __m128 sum = _bias;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < p.srcW)
                                    {
                                        const float* pw = weight + (ky * p.kernelX + kx) * F;
                                        const float* ps = src + (sy & sM) * sY + sx * sX;
                                        sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                                    }
                                }
                            }
                            Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
                        }
                    }
                    else
                    {
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                        {
                            __m128 sum = _bias;
                            for (size_t ky = 0; ky < p.kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                if (sy < p.srcH)
                                {
                                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                                    {
                                        size_t sx = dx * strideX + kx - padX;
                                        if (sx < p.srcW)
                                        {
                                            const float* pw = weight + (ky * p.kernelX + kx) * F;
                                            const float* ps = src + (sy & sM) * sY + sx * sX;
                                            sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(ps), _mm_loadu_ps(pw)), sum);
                                        }
                                    }
                                }
                            }
                            Save1<term, type>(pd, sum, _params, _scale, _shift, _upper);
                        }
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x2(const float* src0, const float* src1, 
            size_t sX, const __m128* weight, const __m128& bias, const __m128* params, const __m128& scale, const __m128& shift, const __m128i& upper, uint8_t* dst)
        {
            if (nofma)
            {
                __m128 sum = bias;
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * sX), weight[0]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * sX), weight[1]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * sX), weight[3]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * sX), weight[4]), sum);
                Save1<term, type>(dst, sum, params, scale, shift, upper);
            }
            else
            {
                __m128 sum0 = bias, sum1 = _mm_setzero_ps();
                sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * sX), weight[0]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * sX), weight[1]), sum1);
                sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * sX), weight[3]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * sX), weight[4]), sum1);
                Save1<term, type>(dst, _mm_add_ps(sum0, sum1), params, scale, shift, upper);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x3(const float* src0, const float* src1,
            size_t sX, const __m128* weight, const __m128& bias, const __m128* params, const __m128& scale, const __m128& shift, const __m128i& upper, uint8_t* dst)
        {
            if (nofma)
            {
                __m128 sum = bias;
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * sX), weight[0]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * sX), weight[1]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * sX), weight[2]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * sX), weight[3]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * sX), weight[4]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * sX), weight[5]), sum);
                Save1<term, type>(dst, sum, params, scale, shift, upper);
            }
            else
            {
                __m128 sum0 = bias, sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps();
                sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * sX), weight[0]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * sX), weight[1]), sum1);
                sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * sX), weight[2]), sum2);
                sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * sX), weight[3]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * sX), weight[4]), sum1);
                sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * sX), weight[5]), sum2);
                Save1<term, type>(dst, _mm_add_ps(_mm_add_ps(sum0, sum1), sum2), params, scale, shift, upper);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge3x2(const float* src0, const float* src1, const float* src2, 
            size_t sX, const __m128* weight, const __m128& bias, const __m128* params, const __m128& scale, const __m128& shift, const __m128i& upper, uint8_t* dst)
        {
            if (nofma)
            {
                __m128 sum = bias;
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * sX), weight[0]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * sX), weight[1]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * sX), weight[3]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * sX), weight[4]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * sX), weight[6]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * sX), weight[7]), sum);
                Save1<term, type>(dst, sum, params, scale, shift, upper);
            }
            else
            {
                __m128 sum0 = bias, sum1 = _mm_setzero_ps();
                sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * sX), weight[0]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * sX), weight[1]), sum1);
                sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * sX), weight[3]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * sX), weight[4]), sum1);
                sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * sX), weight[6]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * sX), weight[7]), sum1);
                Save1<term, type>(dst, _mm_add_ps(sum0, sum1), params, scale, shift, upper);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x1(const float* src0, const float* src1, const float* src2,
            size_t sX, const __m128* weight, const __m128& bias, const __m128* params, const __m128& scale, const __m128& shift, const __m128i& upper, uint8_t* dst)
        {
            if (nofma)
            {
                __m128 sum = bias;
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * sX), weight[0]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * sX), weight[1]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * sX), weight[2]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * sX), weight[3]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * sX), weight[4]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * sX), weight[5]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * sX), weight[6]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * sX), weight[7]), sum);
                sum = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 2 * sX), weight[8]), sum);
                Save1<term, type>(dst, sum, params, scale, shift, upper);
            }
            else
            {
                __m128 sum0 = bias, sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps();
                sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * sX), weight[0]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * sX), weight[1]), sum1);
                sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * sX), weight[2]), sum2);
                sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * sX), weight[3]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * sX), weight[4]), sum1);
                sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * sX), weight[5]), sum2);
                sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * sX), weight[6]), sum0);
                sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * sX), weight[7]), sum1);
                sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 2 * sX), weight[8]), sum2);
                Save1<term, type>(dst, _mm_add_ps(_mm_add_ps(sum0, sum1), sum2), params, scale, shift, upper);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution3x3(const float* src, const ConvParam8i& p, const AlgParam& a,
            size_t dstC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC : p.dstC * a.size), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F : F * a.size;
            size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX)*sX;
            size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

            __m128i _upper = _mm_set1_epi32(a.upper);
            __m128 _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                _params[1] = _mm_set1_ps(params[1]);
            for (size_t c = 0; c < dstC; c += F)
            {
                __m128 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm_loadu_ps(weight + i * F);
                __m128 _bias = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm_loadu_ps(params + c);
                __m128 _scale = _mm_loadu_ps(scale + c);
                __m128 _shift = _mm_loadu_ps(shift + c);

                size_t dy = yBeg;
                if (yBeg == 0 && padY)
                {
                    size_t sy = 0, dx = 0;
                    const float* src0 = src + ((sy + 0) & sM) * sY;
                    const float* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 4, _bias, _params, _scale, _shift, _upper, pDst),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, _scale, _shift, _upper, pDst);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, _scale, _shift, _upper, pDst);
                    dy++;
                }
                for (; dy < yMainEnd; ++dy)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const float* src0 = src + ((sy + 0) & sM) * sY;
                    const float* src1 = src + ((sy + 1) & sM) * sY;
                    const float* src2 = src + ((sy + 2) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge3x2<term, type, nofma>(src0, src1, src2, sX, _weight + 1, _bias, _params, _scale, _shift, _upper, pDst),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0, src2 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX, src2 += ssX)
                        DepthwiseConvolution3x3Main1x1<term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                    if (padW)
                        DepthwiseConvolution3x3Edge3x2<term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                }
                if (dy < yEnd)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const float* src0 = src + ((sy + 0) & sM) * sY;
                    const float* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 1, _bias, _params, _scale, _shift, _upper, pDst),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, _scale, _shift, _upper, pDst);
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //---------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int M> void OutputConvolution1x1_2xM(
            const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC, const int8_t* weight,
            const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, int32_t* buf, uint8_t* dst)
        {
            __m128i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dS = a.maC * p.strideX, dD = p.dstC * a.size, dB = p.dstC;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            __m128i upper = _mm_set1_epi32(a.upper);
            if (dstC > F)
            {
                if (M > 0) d00 = _mm_setzero_si128(), d01 = _mm_setzero_si128();
                if (M > 1) d10 = _mm_setzero_si128(), d11 = _mm_setzero_si128();
                if (M > 2) d20 = _mm_setzero_si128(), d21 = _mm_setzero_si128();
                if (M > 3) d30 = _mm_setzero_si128(), d31 = _mm_setzero_si128();
                if (M > 4) d40 = _mm_setzero_si128(), d41 = _mm_setzero_si128();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        w1 = _mm_loadu_si128((__m128i*)weight + 1);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                        weight += DA;
                    }
                }
                else
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        w1 = _mm_loadu_si128((__m128i*)weight + 1);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s0, w1);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s0, w1);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s0, w1);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s0, w1);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s0, w1);
                        weight += DA;
                    }
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, norm, bias, params, scale, shift, upper, dstC - F), dst += dD, buf += dB;
                }
            }
            else
            {
                if (M > 0) d00 = _mm_setzero_si128();
                if (M > 1) d10 = _mm_setzero_si128();
                if (M > 2) d20 = _mm_setzero_si128();
                if (M > 3) d30 = _mm_setzero_si128();
                if (M > 4) d40 = _mm_setzero_si128();
                if (Base::Overflow(p.compatibility) || Base::Narrowed(p.compatibility))
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0);
                        weight += DA;
                    }
                }
                else
                {
                    for (size_t offs = 0; offs < srcC; offs += 4)
                    {
                        w0 = _mm_loadu_si128((__m128i*)weight + 0);
                        if (M > 0) s0 = Set4(src0 + offs), Madd4<false>(d00, s0, w0);
                        if (M > 1) s0 = Set4(src1 + offs), Madd4<false>(d10, s0, w0);
                        if (M > 2) s0 = Set4(src2 + offs), Madd4<false>(d20, s0, w0);
                        if (M > 3) s0 = Set4(src3 + offs), Madd4<false>(d30, s0, w0);
                        if (M > 4) s0 = Set4(src4 + offs), Madd4<false>(d40, s0, w0);
                        weight += DA;
                    }
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm, bias, params, scale, shift, upper), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, norm, bias, params, scale, shift, upper, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*OutputConvolution1x1_2xM_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstC,
            const int8_t* weight0, const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> OutputConvolution1x1_2xM_Ptr GetOutputConvolution1x1_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return OutputConvolution1x1_2xM<term, type, 1>;
            case 2: return OutputConvolution1x1_2xM<term, type, 2>;
            case 3: return OutputConvolution1x1_2xM<term, type, 3>;
            case 4: return OutputConvolution1x1_2xM<term, type, 4>;
            case 5: return OutputConvolution1x1_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term, SimdConvolutionActivationType type> void OutputConvolution1x1_2(const uint8_t* src,
            const ConvParam8i& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const int8_t* weight,
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst)
        {
            size_t n = 5, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xN = GetOutputConvolution1x1_2xM<term, type>(n);
            OutputConvolution1x1_2xM_Ptr outputConvolution1x1_2xM = GetOutputConvolution1x1_2xM<term, type>(m);
            __m128 _norm[2], _bias[2], _params[2], _scale[2], _shift[2];
            _params[0] = _mm_set1_ps(params[0]);
            _params[1] = _mm_set1_ps(params[1]);
            for (size_t dc = 0; dc < p.dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, p.dstC - dc);
                _norm[0] = _mm_loadu_ps(norm + dc + 0);
                _norm[1] = _mm_loadu_ps(norm + dc + F);
                _bias[0] = _mm_loadu_ps(bias + dc + 0);
                _bias[1] = _mm_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm_loadu_ps(params + dc + 0);
                    _params[1] = _mm_loadu_ps(params + dc + F);
                }
                _scale[0] = _mm_loadu_ps(scale + dc + 0);
                _scale[1] = _mm_loadu_ps(scale + dc + F);
                _shift[0] = _mm_loadu_ps(shift + dc + 0);
                _shift[1] = _mm_loadu_ps(shift + dc + F);
                const uint8_t* s = src;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                for (; i < nn; i += n, s += a.maC * n, b += p.dstC * n, d += p.dstC * a.size * n)
                    outputConvolution1x1_2xN(s, p, a, maC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                for (; i < n1; i += m, s += a.maC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                    outputConvolution1x1_2xM(s, p, a, maC, dC, weight, _norm, _bias, _params, _scale, _shift, b, d);
                weight += DivHi(maC, 4) * DA;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetInput(const ConvParam8i& p, InputConvolutionPtr& input)
        {
            if (p.Is1x1())
                input = InputConvolution1x1_2<type>;
            else
                input = InputConvolution_2<type>;
        }

        void SetInput(const ConvParam8i& p, InputConvolutionPtr& input)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationRelu: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationLeakyRelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationRestrictRange: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationPrelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationElu: SetInput<SimdConvolutionActivationElu>(p, input); break;
            case SimdConvolutionActivationHswish: SetInput<SimdConvolutionActivationHswish>(p, input); break;
            case SimdConvolutionActivationMish: SetInput<SimdConvolutionActivationMish>(p, input); break;
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam8i& p, DepthwiseConvolutionPtr& depthwise)
        {
            if (p.IsKernel(3) && p.IsDilation(1) && Aligned(p.dstC, F))
            {
                if (Base::FmaAvoid(p.compatibility))
                    depthwise = DepthwiseConvolution3x3<term, type, true>;
                else
                    depthwise = DepthwiseConvolution3x3<term, type, false>;
            }
            else
                depthwise = DepthwiseConvolution<term, type>;
        }

        template<SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam8i& p, DepthwiseConvolutionPtr& depthwise)
        {
            if (p.dstT == SimdTensorData32f)
                SetDepthwise<Term8iSingle32f, type>(p, depthwise);
            else
                SetDepthwise<Term8iSingle8u, type>(p, depthwise);
        }

        void SetDepthwise(const ConvParam8i& p, DepthwiseConvolutionPtr& depthwise)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, depthwise); break;
            case SimdConvolutionActivationRelu: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, depthwise); break;
            case SimdConvolutionActivationLeakyRelu: SetDepthwise<SimdConvolutionActivationPrelu>(p, depthwise); break;
            case SimdConvolutionActivationRestrictRange: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, depthwise); break;
            case SimdConvolutionActivationPrelu: SetDepthwise<SimdConvolutionActivationPrelu>(p, depthwise); break;
            case SimdConvolutionActivationElu: SetDepthwise<SimdConvolutionActivationElu>(p, depthwise); break;
            case SimdConvolutionActivationHswish: SetDepthwise<SimdConvolutionActivationHswish>(p, depthwise); break;
            case SimdConvolutionActivationMish: SetDepthwise<SimdConvolutionActivationMish>(p, depthwise); break;
            }
        }

        template<SimdConvolutionActivationType type> static void SetOutput(const ConvParam8i& p, OutputConvolutionPtr* output)
        {
            output[0] = p.dstT == SimdTensorData32f ? OutputConvolution1x1_2<Term8iSingle32f, type> : OutputConvolution1x1_2<Term8iSingle8u, type>;
            output[1] = OutputConvolution1x1_2<Term8iFirst, SimdConvolutionActivationIdentity>;
            output[2] = OutputConvolution1x1_2<Term8iIterim, SimdConvolutionActivationIdentity>;
            output[3] = p.dstT == SimdTensorData32f ? OutputConvolution1x1_2<Term8iLast32f, type> : OutputConvolution1x1_2<Term8iLast8u, type>;
        }

        void SetOutput(const ConvParam8i& p, OutputConvolutionPtr* output)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationRelu: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationLeakyRelu: SetOutput<SimdConvolutionActivationPrelu>(p, output); break;
            case SimdConvolutionActivationRestrictRange: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationPrelu: SetOutput<SimdConvolutionActivationPrelu>(p, output); break;
            case SimdConvolutionActivationElu: SetOutput<SimdConvolutionActivationElu>(p, output); break;
            case SimdConvolutionActivationHswish: SetOutput<SimdConvolutionActivationHswish>(p, output); break;
            case SimdConvolutionActivationMish: SetOutput<SimdConvolutionActivationMish>(p, output); break;
            }
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iCdc::SynetMergedConvolution8iCdc(const MergConvParam8i& p)
            : Base::SynetMergedConvolution8iCdc(p)
        {
            SetSize(Sse::F);
            _cvt32fTo8u = _s8u ? NULL : Convert32fTo8u;
            SetInput(_param.conv[0], _input);
            SetDepthwise(_param.conv[1], _depthwise);
            SetOutput(_param.conv[2], _output);
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iCd::SynetMergedConvolution8iCd(const MergConvParam8i& p)
            : Base::SynetMergedConvolution8iCd(p)
        {
            SetSize(Sse::F);
            _cvt32fTo8u = _s8u ? NULL : Convert32fTo8u;
            SetInput(_param.conv[0], _input);
            SetDepthwise(_param.conv[1], _depthwise);
        }

        //---------------------------------------------------------------------

        SynetMergedConvolution8iDc::SynetMergedConvolution8iDc(const MergConvParam8i& p)
            : Base::SynetMergedConvolution8iDc(p)
        {
            SetSize(Sse::F);
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
