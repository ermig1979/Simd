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
#include "Simd/SimdSynetDeconvolution32f.h"
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
        SynetDeconvolution32fGemmNN::SynetDeconvolution32fGemmNN(const DeconvParam& p)
            : Base::SynetDeconvolution32fGemmNN(p)
        {
            _gemm.Init(InitGemmFuncs(Sve2::Gemm32fNN, "Sve2"));
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
        }

        //-------------------------------------------------------------------------------------------------

        void SynetDeconvolution32fGemmNN::RowToImg(const float* src, float* dst)
        {
            const DeconvParam& p = _param;
            assert(p.trans && p.group == 1);
            if (p.IsPad(0) && p.IsDilation(1) && p.kernelY == p.strideX && p.kernelX == p.strideX)
            {
                Base::SynetDeconvolution32fGemmNN::RowToImg(src, dst);
                return;
            }
            else
            {
                const size_t F = svcntw();
                for (size_t dy = 0; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        memset(dst + (dy * p.dstW + dx) * p.dstC, 0, p.dstC * sizeof(float));
                for (size_t sy = 0; sy < p.srcH; ++sy)
                {
                    for (size_t sx = 0; sx < p.srcW; ++sx)
                    {
                        size_t dy = sy * p.strideY - p.padY;
                        for (size_t ky = 0; ky < p.kernelY; ky++, dy += p.dilationY)
                        {
                            if (dy < p.dstH)
                            {
                                size_t dx = sx * p.strideX - p.padX;
                                for (size_t kx = 0; kx < p.kernelX; kx++, dx += p.dilationX)
                                {
                                    if (dx < p.dstW)
                                    {
                                        float* d = dst + (dy * p.dstW + dx) * p.dstC;
                                        for (size_t dc = 0; dc < p.dstC; dc += F)
                                        {
                                            svbool_t mask = svwhilelt_b32(dc, p.dstC);
                                            svst1_f32(mask, d + dc, svadd_f32_x(mask, svld1_f32(mask, d + dc), svld1_f32(mask, src + dc)));
                                        }
                                    }
                                    src += p.dstC;
                                }
                            }
                            else
                                src += p.kernelX * p.dstC;
                        }
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
        {
            DeconvParam param(batch, conv, compatibility);
            if (!param.Valid(SimdTensorData32f))
                return NULL;
            return new SynetDeconvolution32fGemmNN(param);
        }
    }
#endif
}
