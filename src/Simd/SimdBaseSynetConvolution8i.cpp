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
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
    SynetConvolution8i::SynetConvolution8i(const ConvParam8i& p)
        : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC)
        , _perf(NULL)
#endif
    {
        _sizeS = p.srcC * p.srcH * p.srcW;
        _sizeD = p.dstC * p.dstH * p.dstW;
        _merge = 1;
        _src8u = p.srcT == SimdTensorData8u;
        _dst8u = p.dstT == SimdTensorData8u;
        _overflow16i = (p.compatibility & SimdSynetCompatibilityOverflow16i) != 0;
        _weight8i.Resize(p.kernelY * p.kernelX * p.srcC / p.group * p.dstC);
        _norm32i.Resize(2 * p.dstC);
        _norm32f.Resize(2 * p.dstC);
        _convertSrc = Base::SynetConvert32fTo8u;
    }

    size_t SynetConvolution8i::ExternalBufferSize() const
    {
        size_t size = SIMD_ALIGN;
        if (!_src8u)
            size += AlignHi(_sizeS * _merge * sizeof(uint8_t), SIMD_ALIGN);
        return size;
    }

    size_t SynetConvolution8i::InternalBufferSize() const
    {
        return _buffer.size * sizeof(uint8_t) + _srcCvt.Size() + _dstCvt.Size() +
            _weight8i.size * sizeof(int8_t) + _norm32i.size * sizeof(int32_t) + _norm32f.size * sizeof(float);
    }

    void SynetConvolution8i::SetParams(const float* weight, const float* bias, const float* params, const float* const* stats)
    {
        const ConvParam8i& p = _param;
        _srcCvt.Init(stats[0], stats[1], p.srcC);
        _dstCvt.Init(stats[2], stats[3], p.dstC);
        size_t G = p.group, D = p.dstC / G, C = p.srcC / G, K = p.kernelY * p.kernelX, CK = C * K, GD = G * D;
        Array32f normW(CK);
        const float* pSrcW = weight;
        const float* pSrcB = bias;
        const float* pSrcScaleInv = _srcCvt.iScale.data;
        const float* pSrcScale = _srcCvt.scale.data;
        const float* pSrcShift = _srcCvt.shift.data;
        const float* pDstScale = _dstCvt.iScale.data;
        const float* pDstScaleInv = _dstCvt.scale.data;
        const float* pDstShift = _dstCvt.iShift.data;
        float* pNormW = normW.data;
        int8_t* pDstW = _weight8i.data;
        int32_t* pDstS = _norm32i.data;
        int32_t* pDstB = pDstS + p.dstC;
        float* pNormScale = _norm32f.data;
        float* pNormShift = pNormScale + p.dstC;
        for (size_t g = 0; g < G; ++g)
        {
            for (size_t d = 0; d < D; ++d)
            {
                float normB = 0, minW = FLT_MAX, maxW = -FLT_MAX, scale = 1.0f;
                if (p.trans)
                {
                    for (size_t k = 0, kc = 0; k < K; ++k)
                        for (size_t c = 0; c < C; ++c, ++kc)
                        {
                            pNormW[kc] = pSrcW[kc * GD + d] * pSrcScaleInv[c];
                            minW = Simd::Min(minW, pNormW[kc]);
                            maxW = Simd::Max(maxW, pNormW[kc]);
                        }
                    float abs = Simd::Max(::abs(maxW), ::abs(minW));
                    if(pSrcB)
                        abs = Simd::Max(abs, ::abs(pSrcB[d]) / float(128 * 256 * 256));
                    scale = 127.0f / abs;
                    for (size_t k = 0, kc = 0; k < K; ++k)
                        for (size_t c = 0; c < C; ++c, ++kc)
                            if (_srcCvt.neg && (p.compatibility & SimdSynetCompatibilityOverflow16i))
                            {
                                int w = Base::SynetConvert32fTo8i(pNormW[kc], scale, 0.0f);
                                if (w & 1)
                                    w = Round(w * 0.25f) * 4;
                                pDstW[kc * GD + d] = w / 2;
                                normB -= w * pSrcShift[c];
                            }
                            else
                            {
                                pDstW[kc * GD + d] = Base::SynetConvert32fTo8i(pNormW[kc], scale, 0.0f);
                                normB -= pDstW[kc * GD + d] * pSrcShift[c];
                            }
                }
                else
                {
                    for (size_t c = 0, ck = 0; c < C; ++c)
                        for (size_t k = 0; k < K; ++k, ++ck)
                        {
                            pNormW[ck] = pSrcW[d * CK + ck] * pSrcScaleInv[c];
                            minW = Simd::Min(minW, pNormW[ck]);
                            maxW = Simd::Max(maxW, pNormW[ck]);
                        }
                    float abs = Simd::Max(::abs(maxW), ::abs(minW));
                    if (pSrcB)
                        abs = Simd::Max(abs, ::abs(pSrcB[d]) / float(128 * 256 * 256));
                    scale = 127.0f / abs;
                    for (size_t c = 0, ck = 0; c < C; ++c)
                        for (size_t k = 0; k < K; ++k, ++ck)
                            if (_srcCvt.neg && (p.compatibility & SimdSynetCompatibilityOverflow16i))
                            {
                                int w = Base::SynetConvert32fTo8i(pNormW[ck], scale, 0.0f);
                                if (w & 1)
                                    w = Round(w * 0.25f) * 4;
                                pDstW[d * CK + ck] = w / 2;
                                normB -= w * pSrcShift[c];
                            }
                            else
                            {
                                pDstW[d * CK + ck] = Base::SynetConvert32fTo8i(pNormW[ck], scale, 0.0f);
                                normB -= pDstW[d * CK + ck] * pSrcShift[c];
                            }
                }
                pDstS[d] = _srcCvt.neg && (p.compatibility & SimdSynetCompatibilityOverflow16i) ? 2 : 1;
                if (pSrcB)
                    normB += pSrcB[d] * scale;
                pDstB[d] = Round(normB);
                if (_dst8u)
                {
                    pNormScale[d] = (1.0f / scale) * pDstScaleInv[d];
                    pNormShift[d] = -pDstShift[d] / pDstScale[d];
                }
                else
                {
                    pNormScale[d] = 1.0f / scale;
                    pNormShift[d] = 0;
                }
            }
            if (p.trans)
            {
                pSrcW += D;
                pDstW += D;
            }
            else
            {
                pSrcW += CK * D;
                pDstW += CK * D;
            }
            if(pSrcB)
                pSrcB += D;
            pDstB += D;
            pDstS += D;
            pSrcScale += C;
            pSrcScaleInv += C;
            pSrcShift += C;
            pDstScale += D;
            pDstScaleInv += D;
            pDstShift += D;
            pNormScale += D;
            pNormShift += D;
        }
    }

    void SynetConvolution8i::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
    {
        if (buf == NULL)
        {
            _buffer.Resize(ExternalBufferSize());
            buf = _buffer.data;
        }
        const ConvParam8i& p = _param;
        uint8_t* src8u = _src8u ? NULL : Allocate<uint8_t>(buf, _sizeS * _merge);
        for (size_t b = 0; b < p.batch; b += _merge)
        {
            if (!_src8u)
                _convertSrc((float*)src + b * _sizeS, _merge, p.srcC, p.srcH, p.srcW, p.srcF, _srcCvt.scale.data, _srcCvt.shift.data, src8u, p.compatibility);
            Forward8u(_src8u ? src + b * _sizeS : src8u, buf, dst + b * (_dst8u ? sizeof(uint8_t) : sizeof(float)));
        }
    }

#if defined(SIMD_PERFORMANCE_STATISTIC)
    Base::PerformanceMeasurer * SynetConvolution8i::Perf(const String& func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    //-------------------------------------------------------------------------

    namespace Base
    {
        template<class S, class D, class F> SIMD_INLINE D Convert(S value, F scale, F shift)
        {
            return (D)(F(value) * scale + shift);
        }

        template<> SIMD_INLINE uint8_t Convert<int32_t, uint8_t, float>(int32_t value, float scale, float shift)
        {
            return (uint8_t)Simd::RestrictRange(Round(float(value) * scale + shift), 0, 255);
        }

        template<> SIMD_INLINE uint8_t Convert<float, uint8_t, float>(float value, float scale, float shift)
        {
            return (uint8_t)Simd::RestrictRange(Round(value * scale + shift), 0, 255);
        }

        template<> SIMD_INLINE int8_t Convert<float, int8_t, float>(float value, float scale, float shift)
        {
            return (int8_t)Simd::RestrictRange(Round(value * scale + shift), -128, 127);
        }

        template<class S, class D, class F> void Convert(const S * src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const F* scale, const F* shift, D * dst)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == SimdTensorFormatNchw)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        F _scale = scale[c];
                        F _shift = shift[c];
                        for (size_t h = 0; h < height; ++h)
                        {
                            for (size_t w = 0; w < width; ++w)
                                dst[w] = Convert<S, D, F>(src[w], _scale, _shift);
                            src += width;
                            dst += width;
                        }
                    }
                }
                else if (format == SimdTensorFormatNhwc)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = Convert<S, D, F>(src[c], scale[c], shift[c]);
                            src += channels;
                            dst += channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        SynetConvolution8iGemmNN::SynetConvolution8iGemmNN(const ConvParam8i& p)
            : SynetConvolution8i(p)
        {
            if (p.IsDilation(1) && p.IsStride(1) && p.IsPad(0))
            {
                _skipConv = p.IsKernel(1) || (p.srcH == p.kernelY && p.srcW == p.kernelX);
            }
            else
                _skipConv = false;
            _sizeB = p.srcC * p.kernelY * p.kernelX * p.dstH * p.dstW;
            if (p.trans)
            {
                _ldS = p.srcC * p.kernelY * p.kernelX / p.group * (_skipConv ? p.group : 1);
                _ldW = p.dstC;
                _ldD = p.dstC;
                _grW = p.dstC / p.group;
                _grS = p.srcC * p.kernelY * p.kernelX / p.group * (_skipConv ? 1 : p.dstH * p.dstW);
                _grD = p.dstC / p.group;
            }
            else
            {
                _ldW = p.srcC * p.kernelY * p.kernelX / p.group;
                _ldS = p.dstH * p.dstW;
                _ldD = p.dstH * p.dstW;
                _grW = p.dstC / p.group * p.srcC * p.kernelY * p.kernelX / p.group;
                _grS = p.srcC * p.kernelY * p.kernelX / p.group * p.dstH * p.dstW;
                _grD = p.dstH * p.dstW *p.dstC / p.group;
            }
            _siK = p.kernelY * p.kernelX;
            _siC = p.srcC / p.group;
            _siD = p.dstC / p.group;
            _siS = p.dstH * p.dstW;
        }

        size_t SynetConvolution8iGemmNN::ExternalBufferSize() const
        {
            size_t size = SynetConvolution8i::ExternalBufferSize();
            size += AlignHi(_sizeD * _merge * sizeof(int32_t), SIMD_ALIGN);
            if(!_skipConv)
                size += AlignHi(_sizeB * _merge * sizeof(uint8_t), SIMD_ALIGN);
            return size;
        }

        template <typename T> void Relu(T* data, size_t size)
        {
            for (size_t i = 0; i < size; ++i)
                data[i] = Simd::Max(data[i], T(0));
        }

        void SynetConvolution8iGemmNN::Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const ConvParam8i& p = _param;
            const int8_t * weight = _weight8i.data;
            int32_t * sum = Allocate<int32_t>(buf, _sizeD * _merge);
            if (!_skipConv)
            {
                if(p.trans)
                    for (size_t m = 0; m < _merge; ++m)
                        ImgToRow(src + m * _sizeS, buf + m * _sizeB);
                else
                    for (size_t m = 0; m < _merge; ++m)
                        ImgToCol(src + m * _sizeS, buf + m * _sizeB);
                src = buf;
            }
            if (_merge > 1)
            {
                assert(0);
            }
            else
            {
                for (size_t g = 0; g < p.group; ++g)
                {
                    if (p.trans)
                        GemmNhwc(_siS, _siD, _siK, _siC, src + _grS * g, _ldS, weight + _grW * g, _ldW, sum + _grD * g, _ldD);
                    else
                        GemmNchw(_siD, _siS, _siC, _siK, weight + _grW * g, _ldW, src + _grS * g, _ldS, sum + _grD * g, _ldD);
                }
            }
            Convert<int32_t, int32_t, int32_t>(sum, _merge, p.dstC, p.dstH, p.dstW, p.dstF, _norm32i.data, _norm32i.data + p.dstC, sum);
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity:
                break;
            case SimdConvolutionActivationRelu:
                Relu(sum, _sizeD * _merge);
                break;
            default:
                assert(0);
            }
            if (_dst8u)
                Convert<int32_t, uint8_t, float>(sum, _merge, p.dstC, p.dstH, p.dstW, p.dstF, _norm32f.data, _norm32f.data + p.dstC, dst);
            else
                Convert<int32_t, float, float>(sum, _merge, p.dstC, p.dstH, p.dstW, p.dstF, _norm32f.data, _norm32f.data + p.dstC, (float*)dst);
        }

        void SynetConvolution8iGemmNN::ImgToCol(const uint8_t* src, uint8_t* dst)
        {
            const ConvParam8i& p = _param;
            assert(!p.trans);
            size_t srcSize = p.srcW * p.srcH;
            const uint8_t* zero = _srcCvt.zero.data;
            if (p.IsDilation(1) && p.IsStride(2) && p.IsPad(0) && p.IsKernel(1))
            {
                for (size_t channel = 0; channel < p.srcC; ++channel)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const uint8_t * psrc = src + 2 * dy * p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += srcSize;
                }
            }
            else if (p.IsDilation(1) && p.IsStride(1))
            {
                const ptrdiff_t bodySize = p.dstW - p.padX - p.padW;
                for (size_t channel = 0; channel < p.srcC; ++channel)
                {
                    for (size_t ky = 0; ky < p.kernelY; ++ky)
                    {
                        for (size_t kx = 0; kx < p.kernelX; ++kx)
                        {
                            size_t sy = ky - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy, ++sy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t sx = kx - p.padX, dx = 0;
                                    const uint8_t * psrc = src + sy * p.srcW;
                                    for (; dx < p.padX; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = psrc[sx];
                                        else
                                            *(dst++) = zero[channel];
                                    }
                                    if (bodySize > 0)
                                    {
                                        memcpy(dst, psrc + sx, bodySize * sizeof(uint8_t));
                                        dst += bodySize;
                                        dx += bodySize;
                                        sx += bodySize;
                                    }
                                    for (; dx < p.dstW; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = psrc[sx];
                                        else
                                            *(dst++) = zero[channel];
                                    }
                                }
                                else
                                {
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                        *(dst++) = zero[channel];
                                }
                            }
                        }
                    }
                    src += srcSize;
                }
            }
            else
            {
                for (size_t channel = 0; channel < p.srcC; ++channel)
                {
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        for (size_t kx = 0; kx < p.kernelX; kx++)
                        {
                            size_t sy = ky * p.dilationY - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t sx = kx * p.dilationX - p.padX;
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = src[sy * p.srcW + sx];
                                        else
                                            *(dst++) = zero[channel];
                                        sx += p.strideX;
                                    }
                                }
                                else
                                {
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                        *(dst++) = zero[channel];
                                }
                                sy += p.strideY;
                            }
                        }
                    }
                    src += srcSize;
                }
            }
        }

        void SynetConvolution8iGemmNN::ImgToRow(const uint8_t* src, uint8_t* dst)
        {
            const ConvParam8i& p = _param;
            assert(p.trans);
            size_t size = p.srcC / p.group;
            const uint8_t* zero = _srcCvt.zero.data;
            for (size_t g = 0; g < p.group; ++g)
            {
                for (size_t dy = 0; dy < p.dstH; ++dy)
                {
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                    {
                        for (size_t ky = 0; ky < p.kernelY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; kx++)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        memcpy(dst, src + (sy * p.srcW + sx) * p.srcC, size * sizeof(uint8_t));
                                        dst += size;
                                    }
                                    else
                                    {
                                        memcpy(dst, zero, size * sizeof(uint8_t));
                                        dst += size;
                                    }
                                }
                            }
                            else
                            {
                                for (size_t kx = 0; kx < p.kernelX; kx++)
                                {
                                    memcpy(dst, zero, size * sizeof(uint8_t));
                                    dst += size;
                                }
                            }
                        }
                    }
                }
                src += size;
                zero += size;
            }
        }

        void SynetConvolution8iGemmNN::GemmNchw(size_t D, size_t S, size_t C, size_t K, const int8_t* wgt, size_t ldw, const uint8_t* src, size_t lds, int32_t* dst, size_t ldd)
        {
            const size_t C2 = _overflow16i ? AlignLo(C, 2) : 0;
            for (size_t i = 0; i < D; ++i)
            {
                for (size_t j = 0; j < S; ++j)
                    dst[j] = 0;
                size_t c = 0;
                for (; c < C2; c += 2)
                {
                    for (size_t k = 0; k < K; k++)
                    {
                        int32_t w0 = wgt[(c + 0) * K + k];
                        int32_t w1 = wgt[(c + 1) * K + k];
                        const uint8_t* s0 = src + ((c + 0) * K + k) * lds;
                        const uint8_t* s1 = src + ((c + 1) * K + k) * lds;
                        for (size_t j = 0; j < S; ++j)
                            dst[j] += Simd::RestrictRange(s0[j] * w0 + s1[j] * w1, SHRT_MIN, SHRT_MAX);
                    }
                }
                for (; c < C; ++c)
                {
                    for (size_t k = 0; k < K; k++)
                    {
                        int32_t w0 = wgt[(c + 0) * K + k];
                        const uint8_t* s0 = src + ((c + 0) * K + k) * lds;
                        for (size_t j = 0; j < S; ++j)
                            dst[j] += s0[j] * w0;
                    }
                }
                wgt += ldw;
                dst += ldd;
            }
        }

        void SynetConvolution8iGemmNN::GemmNhwc(size_t S, size_t D, size_t K, size_t C, const uint8_t* src, size_t lds, const int8_t* wgt, size_t ldw, int32_t* dst, size_t ldd)
        {
            const size_t C2 = _overflow16i ? AlignLo(C, 2) : 0;
            for (size_t i = 0; i < S; ++i)
            {
                for (size_t j = 0; j < D; ++j)
                    dst[j] = 0;
                for (size_t k = 0, o = 0; k < K; k++)
                {
                    size_t c = 0;
                    for (; c < C2; c += 2, o += 2)
                    {
                        int32_t s0 = src[o + 0];
                        int32_t s1 = src[o + 1];
                        const int8_t* w0 = wgt + (o + 0) * ldw;
                        const int8_t* w1 = wgt + (o + 1) * ldw;
                        for (size_t j = 0; j < D; ++j)
                            dst[j] += Simd::RestrictRange(s0 * w0[j] + s1 * w1[j], SHRT_MIN, SHRT_MAX);
                    }
                    for (; c < C; ++c, ++o)
                    {
                        int32_t s0 = src[o];
                        const int8_t* w0 = wgt + o * ldw;
                        for (size_t j = 0; j < D; ++j)
                            dst[j] += s0 * w0[j];
                    }
                }
                src += lds;
                dst += ldd;
            }
        }

        //---------------------------------------------------------------------

        SynetConvolution8iNhwcDirect::SynetConvolution8iNhwcDirect(const ConvParam8i& p)
            : SynetConvolution8i(p)
        {
            for (size_t i = 0; i < Term8iSize; ++i)
                _convolutions[i] = NULL;
        }

        String SynetConvolution8iNhwcDirect::Desc() const 
        { 
            return Ext() + "::NhwcDirect" + ((_param.compatibility& SimdSynetCompatibilityOverflow16i) ? "-o" : "-e");
        }

        size_t SynetConvolution8iNhwcDirect::InternalBufferSize() const
        {
            size_t size = SynetConvolution8i::InternalBufferSize();
            return size;
        }
        
        size_t SynetConvolution8iNhwcDirect::ExternalBufferSize() const
        {
            const ConvParam8i& p = _param;
            size_t size = SynetConvolution8i::ExternalBufferSize();
            if (_alg.macroC < p.srcC)
                size += AlignHi(_sizeD*sizeof(int32_t), SIMD_ALIGN);
            return size;
        }

        void SynetConvolution8iNhwcDirect::SetParams(const float* weight, const float* bias, const float* params, const float* const* stats)
        {
            SynetConvolution8i::SetParams(weight, bias, params, stats);
            ReorderWeight();
            _alg.norm = _srcCvt.neg && (_param.compatibility & SimdSynetCompatibilityOverflow16i) ? 2 : 1;
            _alg.zero = _srcCvt.neg ? 0x80808080 : 0;
        }

        bool SynetConvolution8iNhwcDirect::Preferable(const ConvParam8i& p)
        {
            return false;
        }

        void SynetConvolution8iNhwcDirect::SetAlgParam(size_t F, size_t microD, size_t L1, size_t L2, size_t L3)
        {
            const ConvParam8i& p = _param;
            _alg.F = F;
            _alg.microD = microD;
            _alg.macroC = Simd::Min(AlignLoAny(L1 / p.kernelY / p.kernelX / microD, 4), p.srcC);
            for (size_t macroH = p.dstH; macroH >= 1; macroH--)
            {
                _alg.macroH = macroH;
                if (_alg.macroC * p.srcW * (_alg.macroH * p.strideY + p.kernelY * p.dilationY - 1) <= L2)
                    break;
            }
            _alg.macroD = Simd::Min(AlignLoAny(L3 / p.kernelY / p.kernelX / _alg.macroC, _alg.microD), AlignHiAny(p.dstC, _alg.microD));
            _alg.size = (p.dstT == SimdTensorData32f ? 4 : 1);
        }

        void SynetConvolution8iNhwcDirect::ReorderWeight()
        {
            const ConvParam8i& p = _param;
            size_t C = DivHi(p.srcC, 4), D = DivHi(p.dstC, _alg.F);
            Array8i weight8i(p.kernelY * p.kernelX * C * D * _alg.F * 4);
            int8_t* dst = weight8i.data;
            for (size_t d = 0; d < D; d++)
            {
                for (size_t ky = 0; ky < p.kernelY; ++ky)
                {
                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                    {
                        for (size_t c = 0; c < C; ++c)
                        {
                            const int8_t* src = _weight8i.data + ((ky*p.kernelX + kx)*p.srcC + c*4)*p.dstC + d*_alg.F;
                            for (size_t f = 0; f < _alg.F; ++f)
                            {
                                for (size_t i = 0; i < 4; ++i)
                                {
                                    if (d * _alg.F + f < p.dstC && c * 4 + i < p.srcC)
                                        *(dst++) = src[i * p.dstC];
                                    else
                                        *(dst++) = 0;
                                }
                                src++;
                            }
                        }
                    }
                }
            }
            _weight8i.Swap(weight8i);
        }

        void SynetConvolution8iNhwcDirect::Forward8u(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const ConvParam8i& p = _param;
            int32_t* sum = _alg.macroC < p.srcC ? Allocate<int32_t>(buf, _sizeD) : NULL;
            for (size_t m = 0; m < _merge; ++m)
            {
                Forward8u(src, sum, dst);
                src += _sizeS;
                dst += _sizeD*(_dst8u ? sizeof(uint8_t) : sizeof(float));
            }
        }

        void SynetConvolution8iNhwcDirect::Forward8u(const uint8_t* src, int32_t* buf, uint8_t* dst)
        {
            const ConvParam8i& p = _param;
            const int8_t* weight = _weight8i.data;
            const int32_t* bias = _norm32i.data + p.dstC;
            const int32_t* params = NULL;
            const float* scale = _norm32f.data;
            const float* shift = _norm32f.data + p.dstC;
            for (size_t dc = 0; dc < p.dstC; dc += _alg.macroD)
            {
                size_t macroD = Simd::Min(p.dstC, dc + _alg.macroD) - dc;
                for (size_t sc = 0; sc < p.srcC; sc += _alg.macroC)
                {
                    size_t macroC = Simd::Min(p.srcC, sc + _alg.macroC) - sc;
                    for (size_t yBeg = 0; yBeg < p.dstH;)
                    {
                        size_t yEnd = Simd::Min(yBeg + _alg.macroH, p.dstH);
                        if (_alg.macroC == p.srcC)
                        {
                            if (_alg.size == 1)
                                _convolutions[Term8iSingle8u](src + sc, p, _alg, macroD, yBeg, yEnd, macroC, weight, bias, params, scale, shift, buf, dst);
                            else
                                _convolutions[Term8iSingle32f](src + sc, p, _alg, macroD, yBeg, yEnd, macroC, weight, bias, params, scale, shift, buf, dst);
                        }
                        else if (sc == 0)
                            _convolutions[Term8iFirst](src + sc, p, _alg, macroD, yBeg, yEnd, macroC, weight, bias, params, scale, shift, buf, dst);
                        else if (sc + macroC == p.srcC)
                        {
                            if (_alg.size == 1)
                                _convolutions[Term8iLast8u](src + sc, p, _alg, macroD, yBeg, yEnd, macroC, weight, bias, params, scale, shift, buf, dst);
                            else
                                _convolutions[Term8iLast32f](src + sc, p, _alg, macroD, yBeg, yEnd, macroC, weight, bias, params, scale, shift, buf, dst);
                        }
                        else
                            _convolutions[Term8iIterim](src + sc, p, _alg, macroD, yBeg, yEnd, macroC, weight, bias, params, scale, shift, buf, dst);
                        yBeg = yEnd;
                    }
                    weight += DivHi(macroC, 4) * _alg.F * 4;
                }
                weight += p.kernelY * p.kernelX * DivHi(p.srcC, 4) * macroD * 4 - DivHi(p.srcC, 4) * _alg.F * 4;
                bias += _alg.macroD;
                //if (type == ::SimdConvolutionActivationPrelu)
                //    params += macroD;
                shift += _alg.macroD;
                scale += _alg.macroD;
                if (buf)
                    buf += _alg.macroD;
                dst += _alg.macroD * _alg.size;
            }
        }

        //---------------------------------------------------------------------

//#define SIMD_BASE_ONLY_GEMM_NN

        void * SynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam8i param(batch, conv, compatibility);
            if (!param.Valid())
                return NULL;
#if !defined(SIMD_BASE_ONLY_GEMM_NN)
            else if (SynetConvolution8iNhwcDirect::Preferable(param))
                return new SynetConvolution8iNhwcDirect(param);
#endif
            else
                return new SynetConvolution8iGemmNN(param);
        }
    }
}
