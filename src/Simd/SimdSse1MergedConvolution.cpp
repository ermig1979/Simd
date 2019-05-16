/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdMergedConvolution.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#if defined(SIMD_SSE_ENABLE)
    namespace Sse
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE __m128 Activate(__m128 value, const float * params, size_t offset);

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationIdentity>(__m128 value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRelu>(__m128 value, const float * params, size_t offset)
        {
            return _mm_max_ps(_mm_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationLeakyRelu>(__m128 value, const float * params, size_t offset)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(_mm_set1_ps(params[0]), _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRestrictRange>(__m128 value, const float * params, size_t offset)
        {
            return _mm_min_ps(_mm_max_ps(_mm_set1_ps(params[0]), value), _mm_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationPrelu>(__m128 value, const float * params, size_t offset)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(_mm_loadu_ps(params + offset), _mm_min_ps(_mm_setzero_ps(), value)));
        }

        //---------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m128 Activate(__m128 value, const __m128 * params);

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationIdentity>(__m128 value, const __m128 * params)
        {
            return value;
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRelu>(__m128 value, const __m128 * params)
        {
            return _mm_max_ps(_mm_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationLeakyRelu>(__m128 value, const __m128 * params)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(params[0], _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRestrictRange>(__m128 value, const __m128 * params)
        {
            return _mm_min_ps(_mm_max_ps(params[0], value), params[1]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationPrelu>(__m128 value, const __m128 * params)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(params[0], _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x2(
            const float * src0, const float * src1, const __m128 * weight, const __m128 & bias, const __m128 * params, float * dst)
        {
            __m128 sum0 = bias, sum1 = _mm_setzero_ps();
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
            _mm_storeu_ps(dst, Activate<type>(_mm_add_ps(sum0, sum1), params));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge2x3(
            const float * src0, const float * src1, const __m128 * weight, const __m128 & bias, const __m128 * params, float * dst)
        {
            __m128 sum0 = bias, sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps();
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * F), weight[2]), sum2);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * F), weight[5]), sum2);
            _mm_storeu_ps(dst, Activate<type>(_mm_add_ps(_mm_add_ps(sum0, sum1), sum2), params));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Edge3x2(
            const float * src0, const float * src1, const float * src2, const __m128 * weight, const __m128 & bias, const __m128 * params, float * dst)
        {
            __m128 sum0 = bias, sum1 = _mm_setzero_ps();
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * F), weight[6]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * F), weight[7]), sum1);
            _mm_storeu_ps(dst, Activate<type>(_mm_add_ps(sum0, sum1), params));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void ConvolutionDepthwise3x3Main1x1(
            const float * src0, const float * src1, const float * src2, const __m128 * weight, const __m128 & bias, const __m128 * params, float * dst)
        {
            __m128 sum0 = bias, sum1 = _mm_setzero_ps(), sum2 = _mm_setzero_ps();
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 0 * F), weight[0]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 1 * F), weight[1]), sum1);
            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src0 + 2 * F), weight[2]), sum2);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 0 * F), weight[3]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 1 * F), weight[4]), sum1);
            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src1 + 2 * F), weight[5]), sum2);
            sum0 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 0 * F), weight[6]), sum0);
            sum1 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 1 * F), weight[7]), sum1);
            sum2 = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src2 + 2 * F), weight[8]), sum2);
            _mm_storeu_ps(dst, Activate<type>(_mm_add_ps(_mm_add_ps(sum0, sum1), sum2), params));
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolution3x3(const float * src, const SimdConvolutionParameters & p,
            size_t yBeg, size_t yEnd, const size_t bufH[2], const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t srcC = p.srcC, srcW = p.srcW * F, dstW = p.dstW * F, weightS = p.kernelY * p.kernelX * F;
            size_t srcM = (bufH[0] - 1), dstM = (bufH[1] - 1), srcS = bufH[0] * srcW, dstS = bufH[1] * dstW;
            size_t xStep = F * p.strideX, xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;

            __m128 _params[2];
            _params[0] = _mm_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm_set1_ps(params[1]);
            for (size_t c = 0; c < srcC; c += F)
            {
                __m128 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm_loadu_ps(weight + i * F);
                __m128 _bias = bias ? _mm_loadu_ps(bias + c) : _mm_setzero_ps();
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm_loadu_ps(params + c);

                size_t dy = yBeg;
                if (yBeg == 0 && padY)
                {
                    size_t sy = 0, dx = 0;
                    const float * src0 = src + ((sy + 0)&srcM)*srcW;
                    const float * src1 = src + ((sy + 1)&srcM)*srcW;
                    float * pDst = dst + (dy&dstM)*dstW;
                    if (padX)
                        ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 4, _bias, _params, pDst), pDst += F, dx++;
                    for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep)
                        ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, _weight + 3, _bias, _params, pDst);
                    if (padW)
                        ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 3, _bias, _params, pDst);
                    dy++;
                }
                for (; dy < yMainEnd; ++dy)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const float * src0 = src + ((sy + 0)&srcM)*srcW;
                    const float * src1 = src + ((sy + 1)&srcM)*srcW;
                    const float * src2 = src + ((sy + 2)&srcM)*srcW;
                    float * pDst = dst + (dy&dstM)*dstW;
                    if (padX)
                        ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 1, _bias, _params, pDst), pDst += F, dx++;
                    for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep, src2 += xStep)
                        ConvolutionDepthwise3x3Main1x1<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst);
                    if (padW)
                        ConvolutionDepthwise3x3Edge3x2<type>(src0, src1, src2, _weight + 0, _bias, _params, pDst);
                }
                if (dy < yEnd)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const float * src0 = src + ((sy + 0)&srcM)*srcW;
                    const float * src1 = src + ((sy + 1)&srcM)*srcW;
                    float * pDst = dst + (dy&dstM)*dstW;
                    if (padX)
                        ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 1, _bias, _params, pDst), pDst += F, dx++;
                    for (; dx < xMainEnd; dx++, pDst += F, src0 += xStep, src1 += xStep)
                        ConvolutionDepthwise3x3Edge2x3<type>(src0, src1, _weight + 0, _bias, _params, pDst);
                    if (padW)
                        ConvolutionDepthwise3x3Edge2x2<type>(src0, src1, _weight + 0, _bias, _params, pDst);
                }
                src += srcS;
                dst += dstS;
                weight += weightS;
            }
        }

        void DepthwiseReorder(const float * src, const SimdConvolutionParameters & p, float * dst)
        {
            size_t channels = p.dstC, kernel = p.kernelY*p.kernelX;
            for (size_t c = 0; c < channels; c += F)
            {
                size_t n = Simd::Min(F, channels - c);
                for (size_t k = 0; k < kernel; k++)
                {
                    size_t i = 0;
                    for (; i < n; ++i)
                        dst[i] = src[k*channels + c + i];
                    for (; i < F; ++i)
                        dst[i] = 0;
                    dst += F;
                }
            }
        }

        template <SimdConvolutionActivationType type> void SetConvolutionPtr(const MergConvParam & p, size_t index, MergedConvolution::ConvolutionPtr convolution[3])
        {
            switch (index)
            {
            case 0:
                //convolution[0] = old ? DirectConvolutionOld<type, UpdateSet> : InputConvolution<type>;
                break;
            case 1:
                convolution[1] = DepthwiseConvolution3x3<type>;
                break;
            case 2:
                //if (p.add)
                //    convolution[2] = old ? DirectConvolutionOld<type, UpdateAdd> : OutputConvolution<type, UpdateAdd>;
                //else
                //    convolution[2] = old ? DirectConvolutionOld<type, UpdateSet> : OutputConvolution<type, UpdateSet>;
                break;
            default:
                assert(0);
            }
        }

        MergedConvolution::MergedConvolution(const MergConvParam & p)
            : Base::MergedConvolution(p, false)
        {
            SetSize(256 * 1024, Sse::F);
            for (size_t i = 0; i < _param.count; ++i)
            {
                _reorder[i] = NULL;
                switch (p.conv[i].activation)
                {
                case SimdConvolutionActivationIdentity: SetConvolutionPtr<SimdConvolutionActivationIdentity>(_param, i, _convolution); break;
                case SimdConvolutionActivationRelu: SetConvolutionPtr<SimdConvolutionActivationRelu>(_param, i, _convolution); break;
                case SimdConvolutionActivationLeakyRelu: SetConvolutionPtr<SimdConvolutionActivationLeakyRelu>(_param, i, _convolution); break;
                case SimdConvolutionActivationRestrictRange: SetConvolutionPtr<SimdConvolutionActivationRestrictRange>(_param, i, _convolution); break;
                case SimdConvolutionActivationPrelu: SetConvolutionPtr<SimdConvolutionActivationPrelu>(_param, i, _convolution); break;
                default: assert(0);
                }
            }
            _rWeight[1].Resize(AlignHi(p.conv[1].dstC, F)*p.conv[1].kernelY*p.conv[1].kernelX);
            _reorder[1] = DepthwiseReorder;
        }

        //---------------------------------------------------------------------

        void * MergedConvolutionInit(SimdBool trans, size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add)
        {
            MergConvParam param(trans, batch, convs, count, add);
            if (!param.Valid())
                return NULL;
            return new Sse::MergedConvolution(param);
        }
    }
 #endif//SIMD_SSE_ENABLE
}
