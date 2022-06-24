/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdAvx512f.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)  
    namespace Avx512bw
    {
        SynetConvolution32fGemmNN::SynetConvolution32fGemmNN(const ConvParam32f& p)
            : Avx2::SynetConvolution32fGemmNN(p)
        {
            _index.Resize(F);
            for (size_t i = 0; i < F; ++i)
                _index[i] = int(i * p.strideX);
            _nose.Resize(p.kernelX);
            _tail.Resize(p.kernelX);
            ptrdiff_t aligned = AlignHi(p.dstW, F) - F;
            for (size_t kx = 0; kx < p.kernelX; ++kx)
            {
                _nose[kx] = 0;
                _tail[kx] = 0;
                ptrdiff_t sx = kx * p.dilationX - p.padX;
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    if (sx >= 0 && sx < ptrdiff_t(p.srcW) && dx < F)
                        _nose[kx] |= 1 << dx;
                    if (sx < ptrdiff_t(p.srcW) && ptrdiff_t(dx) >= aligned)
                        _tail[kx] |= 1 << (dx - aligned);
                    sx += p.strideX;
                }
            }
            if (p.dstC == 8)
                return;
            _gemm.Init(InitGemmFuncs(Avx512f::Gemm32fNN, "Avx512f"));
            if (_param.trans && _param.group == 1)
            {
                if (GemmRuntime())
                {
                    _gemmCb.Init(InitGemmCbFuncs(Avx512f::Gemm32fNNcbBufferSize, Avx512f::Gemm32fNNcbReorderB, Avx512f::Gemm32fNNcbRun, "Avx512f", GemmKernelF2, GemmKernelF3));
                    _nhwcWeight.Resize(_gemmCb.At(0).BufferSize(_M * _merge, _N, _K));
                }
                else
                    _nhwcWeight.Resize(Avx512f::Gemm32fNNcbBufferSize(_M * _merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
                _nhwcRun = Avx512f::Gemm32fNNcbRun;
                _nhwcReorderB = Avx512f::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = _N > Avx::F ? Avx512f::ConvolutionBiasAndActivation : Avx::ConvolutionBiasAndActivation;
        }

        void SynetConvolution32fGemmNN::ImgToCol(const float* src, float* dst)
        {
            const ConvParam32f& p = _param;
            size_t srcSize = p.srcW * p.srcH;
            if (p.dilationX == 1 && p.dilationY == 1 && p.strideX == 2 && p.strideY == 2 && p.padX == 0 && p.padY == 0 && p.padW == 0 && p.padH == 0 && p.kernelX == 1 && p.kernelY == 1)
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const float* psrc = src + 2 * dy * p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += srcSize;
                }
            }
            else if (p.dilationX * p.dilationY * p.strideX * p.strideY != 1)
            {
                __m512 _0 = _mm512_setzero_ps();
                __m512i index = _mm512_loadu_si512(_index.data);
                size_t aligned = AlignHi(p.dstW, F) - F;
                __mmask16 storeTail = TailMask16(p.dstW - aligned);
                __mmask16 storeNose = aligned ? __mmask16(-1) : storeTail;
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        for (size_t kx = 0; kx < p.kernelX; kx++)
                        {
                            __mmask16 nose = _nose[kx];
                            __mmask16 tail = _tail[kx];
                            size_t sx0 = kx * p.dilationX - p.padX;
                            size_t sy = ky * p.dilationY - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t dx = 0, sx = sx0 + sy * p.srcW;
                                    _mm512_mask_storeu_ps(dst + dx, storeNose, _mm512_mask_i32gather_ps(_0, nose, index, (src + sx), 4));
                                    dx += F, sx += p.strideX * F;
                                    //if (p.strideX == 3)
                                    //{
                                    //    for (; dx < aligned; dx += F, sx += p.strideX*F)
                                    //        _mm512_storeu_ps(dst + dx, Avx512f::Gather<3>(src + sx));
                                    //}
                                    //else
                                    //{
                                    for (; dx < aligned; dx += F, sx += p.strideX * F)
                                        _mm512_storeu_ps(dst + dx, _mm512_i32gather_ps(index, (src + sx), 4));
                                    //}
                                    if (aligned)
                                        _mm512_mask_storeu_ps(dst + dx, storeTail, _mm512_mask_i32gather_ps(_0, tail, index, (src + sx), 4));
                                }
                                else
                                {
                                    memset(dst, 0, p.dstW * sizeof(float));
                                }
                                dst += p.dstW;
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

        //-----------------------------------------------------------------------------------------

        SynetConvolution32fGemmNT::SynetConvolution32fGemmNT(const ConvParam32f& p)
            : Avx2::SynetConvolution32fGemmNT(p)
        {
            _gemm.Init(InitGemmFuncs(Avx512f::Gemm32fNT, "Avx512f"));
            _biasAndActivation = Avx512f::ConvolutionBiasAndActivation;
        }

        //-----------------------------------------------------------------------------------------

        SynetConvolution32fWinograd::SynetConvolution32fWinograd(const ConvParam32f& p)
            : Avx2::SynetConvolution32fWinograd(p)
        {
            if (p.dstC == 8)
                return;
            if (p.kernelY == 1 && p.kernelX == 3)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Avx512f::WinogradKernel1x3Block1x4SetFilter;
                    _setInput = Avx512f::WinogradKernel1x3Block1x4SetInput;
                    _setOutput = Avx512f::WinogradKernel1x3Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 1 && p.kernelX == 5)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Avx512f::WinogradKernel1x5Block1x4SetFilter;
                    _setInput = Avx512f::WinogradKernel1x5Block1x4SetInput;
                    _setOutput = Avx512f::WinogradKernel1x5Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 2 && p.kernelX == 2)
            {
                if (_blockY == 4 && _blockX == 4)
                {
                    SetBlock(4, 4);
                    _setFilter = Avx512f::WinogradKernel2x2Block4x4SetFilter;
                    _setInput = Avx512f::WinogradKernel2x2Block4x4SetInput;
                    _setOutput = Avx512f::WinogradKernel2x2Block4x4SetOutput;
                }
                else if (_blockY == 2 && _blockX == 2)
                {
                    SetBlock(2, 2);
                    _setFilter = Avx512f::WinogradKernel2x2Block2x2SetFilter;
                    _setInput = Avx512f::WinogradKernel2x2Block2x2SetInput;
                    _setOutput = Avx512f::WinogradKernel2x2Block2x2SetOutput;
                }
                else
                    assert(0);
            }
            else if (p.kernelY == 3 && p.kernelX == 3)
            {
                if (_blockY == 4 && _blockX == 4)
                {
                    _setFilter = Avx512f::WinogradKernel3x3Block4x4SetFilter;
                    _setInput = Avx512f::WinogradKernel3x3Block4x4SetInput;
                    _setOutput = Avx512f::WinogradKernel3x3Block4x4SetOutput;
                }
                else if (_blockY == 3 && _blockX == 3)
                {
                    _setFilter = Avx512f::WinogradKernel3x3Block3x3SetFilter;
                    _setInput = Avx512f::WinogradKernel3x3Block3x3SetInput;
                    _setOutput = Avx512f::WinogradKernel3x3Block3x3SetOutput;
                }
                else if (_blockY == 2 && _blockX == 2)
                {
                    _setFilter = Avx512f::WinogradKernel3x3Block2x2SetFilter;
                    _setInput = Avx512f::WinogradKernel3x3Block2x2SetInput;
                    _setOutput = Avx512f::WinogradKernel3x3Block2x2SetOutput;
                }
                else
                    assert(0);
            }
            else
                assert(0);
            _gemm.Init(InitGemmFuncs(Avx512f::Gemm32fNN, "Avx512f"));
            if (_param.trans)
            {
                if (NHWC_GEMM_RUNTIME)
                {
                    _gemmCb.Init(InitGemmCbFuncs(Avx512f::Gemm32fNNcbBufferSize, Avx512f::Gemm32fNNcbReorderB, Avx512f::Gemm32fNNcbRun, "Avx512f", GemmKernelF2, GemmKernelF3));
                    _nhwcStrideW = _gemmCb.At(0).BufferSize(_M * _merge, _N, _K);
                }
                else
                    _nhwcStrideW = Avx512f::Gemm32fNNcbBufferSize(_M * _merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                _nhwcWeight.Resize(_nhwcStrideW * _count);
                _nhwcRun = Avx512f::Gemm32fNNcbRun;
                _nhwcReorderB = Avx512f::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Avx512f::ConvolutionBiasAndActivation;
        }

        //-----------------------------------------------------------------------------------------

        void * SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam32f param(batch, conv, compatibility);
            if (!param.Valid())
                return NULL;
            else if (Base::Bf16Soft(compatibility))
            {
                if (Base::SynetConvolution32fBf16Nhwc::Preferable(param))
                    return new Avx512bw::SynetConvolution32fBf16Nhwc(param);
                else
                    return new Base::SynetConvolution32fBf16Gemm(param);
            }
            else if (Avx::SynetConvolution32fDepthwiseDotProduct::Preferable(param))
                return new Avx::SynetConvolution32fDepthwiseDotProduct(param);
            else if (SynetConvolution32fWinograd::Preferable(param))
                return new SynetConvolution32fWinograd(param);
            else if (SynetConvolution32fGemmNT::Preferable(param))
                return new SynetConvolution32fGemmNT(param);
            else if (SynetConvolution32fDirectNchw::Preferable(param))
                return new Avx512f::SynetConvolution32fDirectNchw(param);
            else if (SynetConvolution32fNhwcDirect::Preferable(param))
                return new Avx512f::SynetConvolution32fNhwcDirect(param);
            else if (SynetConvolution32fDirectNhwc::Preferable(param))
                return new Avx512f::SynetConvolution32fDirectNhwc(param);
            else
                return new SynetConvolution32fGemmNN(param);
        }
    }
#endif//SIMD_AVX2_ENABLE
}
