/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynetActivation.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        namespace
        {
            SIMD_INLINE svuint32_t Float32ToBFloat16(svfloat32_t value, const svbool_t& mask)
            {
                svuint32_t bits = svreinterpret_u32_f32(value);
                svuint32_t round = svadd_n_u32_x(mask, svand_n_u32_x(mask, svlsr_n_u32_x(mask, bits, Base::Bf16::SHIFT), 1), Base::Bf16::ROUND);
                return svlsr_n_u32_x(mask, svadd_u32_x(mask, bits, round), Base::Bf16::SHIFT);
            }

            template<SimdConvolutionActivationType type> SIMD_INLINE svfloat32_t Activate(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                return Sve2::Activate<type>(value, svdup_n_f32(params[0]), svdup_n_f32(params[1]), 0, mask);
            }

            template<> SIMD_INLINE svfloat32_t Activate<SimdConvolutionActivationPrelu>(svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                return Sve2::Activate<SimdConvolutionActivationPrelu>(value, svld1_f32(mask, params + offset), svdup_n_f32(0.0f), 0, mask);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term> struct DepthwiseTerm16b
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, svfloat32_t value, const float* params, size_t offset, const svbool_t& mask);
        };

        template <> struct DepthwiseTerm16b<Term16bLast16b>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                svfloat32_t f32 = Activate<type>(value, params, offset, mask);
                svst1h_u32(mask, (uint16_t*)(ptr + offset * 2), Float32ToBFloat16(f32, mask));
            }
        };

        template <> struct DepthwiseTerm16b<Term16bLast32f>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, svfloat32_t value, const float* params, size_t offset, const svbool_t& mask)
            {
                svfloat32_t f32 = Activate<type>(value, params, offset, mask);
                svst1_f32(mask, (float*)ptr + offset, f32);
            }
        };

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, svfloat32_t val0, const float* params, size_t offset, const svbool_t& mask)
        {
            DepthwiseTerm16b<term>::template Save<type>(ptr, val0, params, offset, mask);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename T, Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcDepthwiseDefault(const uint8_t* src8, const ConvParam& p, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.trans && p.IsDepthwise());
            const T* src = (T*)src8;
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            size_t kX = p.kernelX, kY = p.kernelY, dX = p.dilationX, dY = p.dilationY, srcH = p.srcH, srcW = p.srcW;
            size_t size = p.group, elem = (term == Term16bLast16b ? 2 : 4), sdS = size * dX;
            size_t size2F = AlignLo(size, 2 * F), size4F = AlignLo(size, 4 * F);

            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                size_t sy0 = dy * p.strideY - p.padY;
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t sx0 = dx * p.strideX - p.padX;
                    size_t c = 0;
                    for (; c < size4F; c += 4 * F)
                    {
                        svfloat32_t d00 = svld1_f32(body, bias + c + 0 * F);
                        svfloat32_t d01 = svld1_f32(body, bias + c + 1 * F);
                        svfloat32_t d02 = svld1_f32(body, bias + c + 2 * F);
                        svfloat32_t d03 = svld1_f32(body, bias + c + 3 * F);
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pw = weight + ky * kX * size + c;
                                const T* ps = src + (sy * srcW + sx0) * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    if (sx < srcW)
                                    {
                                        d00 = svmla_f32_x(body, d00, LoadSrc(ps + 0 * F, body), svld1_f32(body, pw + 0 * F));
                                        d01 = svmla_f32_x(body, d01, LoadSrc(ps + 1 * F, body), svld1_f32(body, pw + 1 * F));
                                        d02 = svmla_f32_x(body, d02, LoadSrc(ps + 2 * F, body), svld1_f32(body, pw + 2 * F));
                                        d03 = svmla_f32_x(body, d03, LoadSrc(ps + 3 * F, body), svld1_f32(body, pw + 3 * F));
                                    }
                                    pw += size, ps += sdS;
                                }
                            }
                        }
                        Save1<term, type>(dst, d00, params, c + 0 * F, body);
                        Save1<term, type>(dst, d01, params, c + 1 * F, body);
                        Save1<term, type>(dst, d02, params, c + 2 * F, body);
                        Save1<term, type>(dst, d03, params, c + 3 * F, body);
                    }
                    for (; c < size2F; c += 2 * F)
                    {
                        svfloat32_t d00 = svld1_f32(body, bias + c + 0 * F);
                        svfloat32_t d01 = svld1_f32(body, bias + c + 1 * F);
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pw = weight + ky * kX * size + c;
                                const T* ps = src + (sy * srcW + sx0) * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    if (sx < srcW)
                                    {
                                        d00 = svmla_f32_x(body, d00, LoadSrc(ps + 0 * F, body), svld1_f32(body, pw + 0 * F));
                                        d01 = svmla_f32_x(body, d01, LoadSrc(ps + 1 * F, body), svld1_f32(body, pw + 1 * F));
                                    }
                                    pw += size, ps += sdS;
                                }
                            }
                        }
                        Save1<term, type>(dst, d00, params, c + 0 * F, body);
                        Save1<term, type>(dst, d01, params, c + 1 * F, body);
                    }
                    for (; c < size; c += F)
                    {
                        svbool_t mask = svwhilelt_b32((uint32_t)c, (uint32_t)size);
                        svfloat32_t d00 = svld1_f32(mask, bias + c);
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pw = weight + ky * kX * size + c;
                                const T* ps = src + (sy * srcW + sx0) * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    if (sx < srcW)
                                    {
                                        d00 = svmla_f32_x(mask, d00, LoadSrc(ps, mask), svld1_f32(mask, pw));
                                    }
                                    pw += size, ps += sdS;
                                }
                            }
                        }
                        Save1<term, type>(dst, d00, params, c, mask);
                    }
                    dst += size * elem;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution16bNhwcDepthwise3x3Edge(const T* src, const ConvParam& p, size_t dy, size_t dx, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const size_t F = svcntw();
            size_t srcC = p.srcC;
            for (size_t c = 0; c < srcC; c += F)
            {
                svbool_t mask = svwhilelt_b32((uint32_t)c, (uint32_t)srcC);
                svfloat32_t d00 = svld1_f32(mask, bias + c);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    if (sy < p.srcH)
                    {
                        for (size_t kx = 0; kx < 3; ++kx)
                        {
                            size_t sx = dx * p.strideX + kx - p.padX;
                            if (sx < p.srcW)
                            {
                                const float* pw = weight + (ky * 3 + kx) * srcC + c;
                                const T* ps = src + (sy * p.srcW + sx) * srcC + c;
                                d00 = svmla_f32_x(mask, d00, LoadSrc(ps, mask), svld1_f32(mask, pw));
                            }
                        }
                    }
                }
                Save1<term, type>(dst, d00, params, c, mask);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution16bNhwcDepthwise3x3Main1(const T* src, size_t srcS, size_t srcC, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const size_t F = svcntw();
            for (size_t c = 0; c < srcC; c += F)
            {
                svbool_t mask = svwhilelt_b32((uint32_t)c, (uint32_t)srcC);
                svfloat32_t d00 = svld1_f32(mask, bias + c);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const T* ps = src + ky * srcS + c;
                    const float* pw = weight + ky * 3 * srcC + c;
                    d00 = svmla_f32_x(mask, d00, LoadSrc(ps + 0 * srcC, mask), svld1_f32(mask, pw + 0 * srcC));
                    d00 = svmla_f32_x(mask, d00, LoadSrc(ps + 1 * srcC, mask), svld1_f32(mask, pw + 1 * srcC));
                    d00 = svmla_f32_x(mask, d00, LoadSrc(ps + 2 * srcC, mask), svld1_f32(mask, pw + 2 * srcC));
                }
                Save1<term, type>(dst, d00, params, c, mask);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution16bNhwcDepthwise3x3Main2(const T* src, size_t srcS, size_t srcX, size_t srcC, size_t dstC, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const size_t F = svcntw();
            for (size_t c = 0; c < srcC; c += F)
            {
                svbool_t mask = svwhilelt_b32((uint32_t)c, (uint32_t)srcC);
                svfloat32_t d00 = svld1_f32(mask, bias + c);
                svfloat32_t d01 = d00;
                const float* pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const T* ps0 = src + ky * srcS + c;
                    const T* ps1 = ps0 + srcX;
                    svfloat32_t w0 = svld1_f32(mask, pw);
                    d00 = svmla_f32_x(mask, d00, LoadSrc(ps0 + 0 * srcC, mask), w0);
                    d01 = svmla_f32_x(mask, d01, LoadSrc(ps1 + 0 * srcC, mask), w0);
                    pw += srcC;
                    w0 = svld1_f32(mask, pw);
                    d00 = svmla_f32_x(mask, d00, LoadSrc(ps0 + 1 * srcC, mask), w0);
                    d01 = svmla_f32_x(mask, d01, LoadSrc(ps1 + 1 * srcC, mask), w0);
                    pw += srcC;
                    w0 = svld1_f32(mask, pw);
                    d00 = svmla_f32_x(mask, d00, LoadSrc(ps0 + 2 * srcC, mask), w0);
                    d01 = svmla_f32_x(mask, d01, LoadSrc(ps1 + 2 * srcC, mask), w0);
                    pw += srcC;
                }
                Save1<term, type>(dst + 0 * dstC, d00, params, c, mask);
                Save1<term, type>(dst + 1 * dstC, d01, params, c, mask);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution16bNhwcDepthwise3x3Main4(const T* src, size_t srcS, size_t srcX, size_t srcC, size_t dstC, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const size_t F = svcntw();
            for (size_t c = 0; c < srcC; c += F)
            {
                svbool_t mask = svwhilelt_b32((uint32_t)c, (uint32_t)srcC);
                svfloat32_t d00 = svld1_f32(mask, bias + c);
                svfloat32_t d01 = d00;
                svfloat32_t d02 = d00;
                svfloat32_t d03 = d00;
                const float* pw = weight + c;
                const T* ps0 = src + 0 * srcX + c;
                const T* ps1 = src + 1 * srcX + c;
                const T* ps2 = src + 2 * srcX + c;
                const T* ps3 = src + 3 * srcX + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    svfloat32_t w0 = svld1_f32(mask, pw);
                    d00 = svmla_f32_x(mask, d00, LoadSrc(ps0 + offset, mask), w0);
                    d01 = svmla_f32_x(mask, d01, LoadSrc(ps1 + offset, mask), w0);
                    d02 = svmla_f32_x(mask, d02, LoadSrc(ps2 + offset, mask), w0);
                    d03 = svmla_f32_x(mask, d03, LoadSrc(ps3 + offset, mask), w0);
                    pw += srcC, offset += srcC;
                    w0 = svld1_f32(mask, pw);
                    d00 = svmla_f32_x(mask, d00, LoadSrc(ps0 + offset, mask), w0);
                    d01 = svmla_f32_x(mask, d01, LoadSrc(ps1 + offset, mask), w0);
                    d02 = svmla_f32_x(mask, d02, LoadSrc(ps2 + offset, mask), w0);
                    d03 = svmla_f32_x(mask, d03, LoadSrc(ps3 + offset, mask), w0);
                    pw += srcC, offset += srcC;
                    w0 = svld1_f32(mask, pw);
                    d00 = svmla_f32_x(mask, d00, LoadSrc(ps0 + offset, mask), w0);
                    d01 = svmla_f32_x(mask, d01, LoadSrc(ps1 + offset, mask), w0);
                    d02 = svmla_f32_x(mask, d02, LoadSrc(ps2 + offset, mask), w0);
                    d03 = svmla_f32_x(mask, d03, LoadSrc(ps3 + offset, mask), w0);
                    pw += srcC;
                }
                Save1<term, type>(dst + 0 * dstC, d00, params, c, mask);
                Save1<term, type>(dst + 1 * dstC, d01, params, c, mask);
                Save1<term, type>(dst + 2 * dstC, d02, params, c, mask);
                Save1<term, type>(dst + 3 * dstC, d03, params, c, mask);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcDepthwise3x3(const uint8_t* src8, const ConvParam& p, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t srcS = p.srcC * p.srcW;
            size_t srcX = p.srcC * p.strideX;
            size_t dstH = p.dstH - p.padH;
            size_t dstW = p.dstW - p.padW;
            size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
            size_t dstW4 = AlignLo(dstW - p.padX, 4) + p.padX;
            size_t dstC = p.dstC * (term == Term16bLast16b ? 2 : 4);
            size_t dy = 0;
            for (; dy < p.padY; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    Convolution16bNhwcDepthwise3x3Edge<T, term, type>(src, p, dy, dx, weight, bias, params, dst), dst += dstC;
            for (; dy < dstH; ++dy)
            {
                size_t dx = 0;
                for (; dx < p.padX; ++dx)
                    Convolution16bNhwcDepthwise3x3Edge<T, term, type>(src, p, dy, dx, weight, bias, params, dst), dst += dstC;
                size_t offset = ((dy * p.strideY - p.padY) * p.srcW + dx * p.strideX - p.padX) * p.srcC;
                for (; dx < dstW4; dx += 4)
                    Convolution16bNhwcDepthwise3x3Main4<T, term, type>(src + offset, srcS, srcX, p.srcC, dstC, weight, bias, params, dst), dst += 4 * dstC, offset += 4 * srcX;
                for (; dx < dstW2; dx += 2)
                    Convolution16bNhwcDepthwise3x3Main2<T, term, type>(src + offset, srcS, srcX, p.srcC, dstC, weight, bias, params, dst), dst += 2 * dstC, offset += 2 * srcX;
                for (; dx < dstW; ++dx)
                    Convolution16bNhwcDepthwise3x3Main1<T, term, type>(src + offset, srcS, p.srcC, weight, bias, params, dst), dst += dstC, offset += srcX;
                for (; dx < p.dstW; ++dx)
                    Convolution16bNhwcDepthwise3x3Edge<T, term, type>(src, p, dy, dx, weight, bias, params, dst), dst += dstC;
            }
            for (; dy < p.dstH; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    Convolution16bNhwcDepthwise3x3Edge<T, term, type>(src, p, dy, dx, weight, bias, params, dst), dst += dstC;
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void SetConvolution(const ConvParam& p, SynetConvolution16bNhwcDepthwise::ConvolutionPtr& convolution)
        {
            if (p.IsKernel(3) && p.IsDilation(1) && p.dstH >= p.padY + p.padH && p.dstW >= p.padX + p.padW)
                convolution = Convolution16bNhwcDepthwise3x3<T, term, type>;
            else
                convolution = Convolution16bNhwcDepthwiseDefault<T, term, type>;
        }

        template<typename T, SimdConvolutionActivationType type> static void SetConvolution(const ConvParam& p, SynetConvolution16bNhwcDepthwise::ConvolutionPtr& convolution)
        {
            if (p.dstT == SimdTensorData32f)
                SetConvolution<T, Term16bLast32f, type>(p, convolution);
            else
                SetConvolution<T, Term16bLast16b, type>(p, convolution);
        }

        template<SimdConvolutionActivationType type> static void SetConvolution(const ConvParam& p, SynetConvolution16bNhwcDepthwise::ConvolutionPtr& convolution)
        {
            if (p.srcT == SimdTensorData16b)
                SetConvolution<uint16_t, type>(p, convolution);
            else
                SetConvolution<float, type>(p, convolution);
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution16bNhwcDepthwise::SynetConvolution16bNhwcDepthwise(const ConvParam& p)
            : Base::SynetConvolution16bNhwcDepthwise(p)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationRelu: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationLeakyRelu: SetConvolution<SimdConvolutionActivationPrelu>(p, _convolution); break;
            case SimdConvolutionActivationRestrictRange: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationPrelu: SetConvolution<SimdConvolutionActivationPrelu>(p, _convolution); break;
            case SimdConvolutionActivationElu: SetConvolution<SimdConvolutionActivationElu>(p, _convolution); break;
            case SimdConvolutionActivationHswish: SetConvolution<SimdConvolutionActivationHswish>(p, _convolution); break;
            case SimdConvolutionActivationMish: SetConvolution<SimdConvolutionActivationMish>(p, _convolution); break;
            case SimdConvolutionActivationHardSigmoid: SetConvolution<SimdConvolutionActivationHardSigmoid>(p, _convolution); break;
            case SimdConvolutionActivationSwish: SetConvolution<SimdConvolutionActivationSwish>(p, _convolution); break;
            case SimdConvolutionActivationGelu: SetConvolution<SimdConvolutionActivationGelu>(p, _convolution); break;
            }
        }
    }
#endif
}
