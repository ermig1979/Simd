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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SynetConvolution32fGemmNN::SynetConvolution32fGemmNN(const ConvParam & p)
            : Base::SynetConvolution32fGemmNN(p)
        {
            const size_t F = svcntw();
            _index.Resize(F);
            for (size_t i = 0; i < F; ++i)
                _index[i] = int(i * p.strideX);
            _nose.Resize(p.kernelX);
            _tail.Resize(p.kernelX);
            _start.Resize(p.kernelX);
            for (size_t kx = 0; kx < p.kernelX; ++kx)
            {
                _nose[kx] = 0;
                _tail[kx] = int(p.dstW);
                ptrdiff_t sx = kx * p.dilationX - p.padX;
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    if (sx < 0)
                        _nose[kx]++;
                    if (sx >= ptrdiff_t(p.srcW))
                        _tail[kx]--;
                    sx += p.strideX;
                }
                _start[kx] = int(kx * p.dilationX - p.padX + _nose[kx] * p.strideX);
            }
            _gemm.Init(InitGemmFuncs(Sve2::Gemm32fNN, "Sve2"));
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
        }

        //-------------------------------------------------------------------------------------------------

        void SynetConvolution32fGemmNN::ImgToCol(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            size_t srcSize = p.srcW * p.srcH;
            if (p.dilationX == 1 && p.dilationY == 1 && p.strideX == 2 && p.strideY == 2 && p.padX == 0 && p.padY == 0 && p.padW == 0 && p.padH == 0 && p.kernelX == 1 && p.kernelY == 1)
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const float * psrc = src + 2 * dy * p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += srcSize;
                }
            }
            else if (p.dilationX * p.dilationY * p.strideX * p.strideY != 1)
            {
                const size_t F = svcntw();
                svbool_t all = svptrue_b32();
                svuint32_t index = svld1_u32(all, (const uint32_t*)_index.data);
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        for (size_t kx = 0; kx < p.kernelX; kx++)
                        {
                            size_t noseDx = _nose[kx];
                            size_t tailDx = _tail[kx];
                            size_t bodyDx = AlignLo(tailDx - noseDx, F) + noseDx;
                            size_t sx0 = _start[kx];
                            size_t sy = ky * p.dilationY - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t dx = 0, sx = sx0 + sy * p.srcW;
                                    for (; dx < noseDx; ++dx)
                                        *(dst++) = 0;
                                    for (; dx < bodyDx; dx += F, sx += p.strideX * F, dst += F)
                                        svst1_f32(all, dst, svld1_gather_u32index_f32(all, src + sx, index));
                                    for (; dx < tailDx; ++dx, sx += p.strideX)
                                        *(dst++) = src[sx];
                                    for (; dx < p.dstW; ++dx)
                                        *(dst++) = 0;
                                }
                                else
                                {
                                    memset(dst, 0, p.dstW * sizeof(float));
                                    dst += p.dstW;
                                }
                                sy += p.strideY;
                            }
                        }
                    }
                    src += srcSize;
                }
            }
            else
            {
                Base::SynetConvolution32fGemmNN::ImgToCol(src, dst);
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SynetConvolution32fGemmNN::ImgToRow(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            assert(p.trans);
            const size_t F = svcntw();
            size_t size = p.srcC / p.group;
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
                                        const float * s = src + (sy * p.srcW + sx) * p.srcC;
                                        for (size_t i = 0; i < size; i += F)
                                        {
                                            svbool_t mask = svwhilelt_b32(i, size);
                                            svst1_f32(mask, dst + i, svld1_f32(mask, s + i));
                                        }
                                        dst += size;
                                    }
                                    else
                                    {
                                        for (size_t i = 0; i < size; i += F)
                                        {
                                            svbool_t mask = svwhilelt_b32(i, size);
                                            svst1_f32(mask, dst + i, svdup_n_f32(0.0f));
                                        }
                                        dst += size;
                                    }
                                }
                            }
                            else
                            {
                                size_t n = p.kernelX * size;
                                for (size_t i = 0; i < n; i += F)
                                {
                                    svbool_t mask = svwhilelt_b32(i, n);
                                    svst1_f32(mask, dst + i, svdup_n_f32(0.0f));
                                }
                                dst += n;
                            }
                        }
                    }
                }
                src += size;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fGemmNT::SynetConvolution32fGemmNT(const ConvParam & p)
            : Base::SynetConvolution32fGemmNT(p)
        {
            _gemm.Init(InitGemmFuncs(Sve2::Gemm32fNT, "Sve2"));
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
        }

        bool SynetConvolution32fGemmNT::Preferable(const ConvParam & p)
        {
            if (p.group != 1)
                return false;
            if (p.trans)
                return p.Is1x1() && p.dstC == 1;
            else
                return p.srcH < 4 && p.srcW < 4;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fWinograd::SynetConvolution32fWinograd(const ConvParam & p)
            : Base::SynetConvolution32fWinograd(p)
        {
            if (p.kernelY == 1 && p.kernelX == 3)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Sve2::WinogradKernel1x3Block1x4SetFilter;
                    _setInput = Sve2::WinogradKernel1x3Block1x4SetInput;
                    _setOutput = Sve2::WinogradKernel1x3Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 1 && p.kernelX == 5)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Sve2::WinogradKernel1x5Block1x4SetFilter;
                    _setInput = Sve2::WinogradKernel1x5Block1x4SetInput;
                    _setOutput = Sve2::WinogradKernel1x5Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 2 && p.kernelX == 2)
            {
                if (p.trans && p.srcH >= 8 && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 144)
                {
                    SetBlock(4, 4);
                    _setFilter = Sve2::WinogradKernel2x2Block4x4SetFilter;
                    _setInput = Sve2::WinogradKernel2x2Block4x4SetInput;
                    _setOutput = Sve2::WinogradKernel2x2Block4x4SetOutput;
                }
                else
                {
                    SetBlock(2, 2);
                    _setFilter = Sve2::WinogradKernel2x2Block2x2SetFilter;
                    _setInput = Sve2::WinogradKernel2x2Block2x2SetInput;
                    _setOutput = Sve2::WinogradKernel2x2Block2x2SetOutput;
                }
            }
            else if (p.kernelY == 3 && p.kernelX == 3)
            {
                if (p.trans && p.srcH >= 8 && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 144)
                {
                    SetBlock(4, 4);
                    _setFilter = Sve2::WinogradKernel3x3Block4x4SetFilter;
                    _setInput = Sve2::WinogradKernel3x3Block4x4SetInput;
                    _setOutput = Sve2::WinogradKernel3x3Block4x4SetOutput;
                }
                else if (p.trans && p.srcH >= 6 && p.srcW >= 6 && p.srcH * p.srcW * p.batch >= 81 && p.dstH % 3 == 0 && p.dstW % 3 == 0)
                {
                    SetBlock(3, 3);
                    _setFilter = Sve2::WinogradKernel3x3Block3x3SetFilter;
                    _setInput = Sve2::WinogradKernel3x3Block3x3SetInput;
                    _setOutput = Sve2::WinogradKernel3x3Block3x3SetOutput;
                }
                else
                {
                    SetBlock(2, 2);
                    _setFilter = Sve2::WinogradKernel3x3Block2x2SetFilter;
                    _setInput = Sve2::WinogradKernel3x3Block2x2SetInput;
                    _setOutput = Sve2::WinogradKernel3x3Block2x2SetOutput;
                }
            }
            else
                assert(0);
            _gemm.Init(InitGemmFuncs(Sve2::Gemm32fNN, "Sve2"));
            if (_param.trans)
            {
                if (NHWC_GEMM_RUNTIME)
                {
#if defined(SIMD_ARM64_ENABLE)
                    _gemmCb.Init(InitGemmCbFuncs(Neon::Gemm32fNNcbBufferSize, Neon::Gemm32fNNcbReorderB, Neon::Gemm32fNNcbRun, "Neon", GemmKernelF2, GemmKernelF4));
#else
                    _gemmCb.Init(InitGemmCbFuncs(Neon::Gemm32fNNcbBufferSize, Neon::Gemm32fNNcbReorderB, Neon::Gemm32fNNcbRun, "Neon", GemmKernelF2, GemmKernelF3));
#endif
                    _nhwcStrideW = _gemmCb.At(0).BufferSize(_M*_merge, _N, _K);
                }
                else
                    _nhwcStrideW = Neon::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                _nhwcWeight.Resize(_nhwcStrideW*_count);
                _nhwcRun = Neon::Gemm32fNNcbRun;
                _nhwcReorderB = Neon::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
        }

        bool SynetConvolution32fWinograd::Preferable(const ConvParam & p)
        {
            if (!p.IsDilation(1) || !p.IsStride(1) || p.group != 1 || p.srcC < 10)
                return false;
            if (p.IsKernel(1, 3))
            {
                if (!(p.IsPad(0) || (p.padX == 1 && p.padW == 1)))
                    return false;
                if (p.srcC <= 32)
                    return false;
                return p.trans && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 36;
            }
            else if (p.IsKernel(1, 5))
            {
                if (!(p.IsPad(0) || (p.padX == 2 && p.padW == 2)))
                    return false;
                return p.trans && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 36;
            }
            else if (p.IsKernel(2))
            {
                if (!(p.IsPad(0) || (p.padY + p.padH == 1 && p.padX + p.padW == 1)))
                    return false;
                return p.trans && p.srcH >= 4 && p.srcW >= 4 && p.srcH * p.srcW * p.batch >= 36;
            }
            else if (p.IsKernel(3))
            {
                if (!(p.IsPad(0) || p.IsPad(1)))
                    return false;
                if (p.trans)
                    return p.srcH >= 4 && p.srcW >= 4 && p.srcH * p.srcW * p.batch >= 36;
                else
                    return p.srcH >= 6 && p.srcW >= 6;
            }
            return false;
        }
    }
#endif
}
