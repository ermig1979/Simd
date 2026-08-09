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

#include "Simd/SimdSynetGridSample.h"
#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        template<int align, int range> void IndexCoeffs32fBlZ(const float* grd, size_t dstS, int srcH, int srcW, int padW, uint32_t* idx, float* dy, float* dx, int& yMin, int& yMax)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            const svuint32_t offsets = svlsl_n_u32_x(body, svindex_u32(0, 1), 1);
            const svfloat32_t ax = svdup_n_f32((srcW - align) / 2.0f);
            const svfloat32_t ay = svdup_n_f32((srcH - align) / 2.0f);
            const svfloat32_t bx = svdup_n_f32((srcW - 1) / 2.0f);
            const svfloat32_t by = svdup_n_f32((srcH - 1) / 2.0f);
            const svint32_t _0 = svdup_n_s32(0);
            const svint32_t _2 = svdup_n_s32(2);
            const svint32_t _srcH = svdup_n_s32(srcH + 2);
            const svint32_t _srcW = svdup_n_s32(srcW + 2);
            const svint32_t _padW = svdup_n_s32(padW);
            for (size_t d = 0; d < dstS; d += F)
            {
                svbool_t mask = svwhilelt_b32(d, dstS);
                svfloat32_t x = svmla_f32_x(mask, bx, svld1_gather_u32index_f32(mask, grd + 0, offsets), ax);
                svfloat32_t y = svmla_f32_x(mask, by, svld1_gather_u32index_f32(mask, grd + 1, offsets), ay);
                svfloat32_t xf = svrintm_f32_x(mask, x);
                svfloat32_t yf = svrintm_f32_x(mask, y);
                svst1_f32(mask, dy + d, svsub_f32_x(mask, y, yf));
                svst1_f32(mask, dx + d, svsub_f32_x(mask, x, xf));
                svint32_t xi = svmin_s32_x(mask, svmax_s32_x(mask, svadd_s32_x(mask, svcvt_s32_f32_x(mask, xf), _2), _0), _srcW);
                svint32_t yi = svmin_s32_x(mask, svmax_s32_x(mask, svadd_s32_x(mask, svcvt_s32_f32_x(mask, yf), _2), _0), _srcH);
                svst1_u32(mask, idx + d, svreinterpret_u32_s32(svmla_s32_x(mask, xi, yi, _padW)));
                if (range)
                {
                    yMin = Min(yMin, svminv_s32(mask, yi));
                    yMax = Max(yMax, svmaxv_s32(mask, yi));
                }
                grd += 2 * F;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void BilinearInterp32fBlZ(const float* pad0, size_t dstS, int padW, uint32_t* idx, float* dy, float* dx, float* dst)
        {
            const size_t F = svcntw();
            const float* pad1 = pad0 + padW;
            const svfloat32_t _1 = svdup_n_f32(1.0f);
            for (size_t d = 0; d < dstS; d += F)
            {
                svbool_t mask = svwhilelt_b32(d, dstS);
                svuint32_t offs = svld1_u32(mask, idx + d);
                svfloat32_t p00 = svld1_gather_u32index_f32(mask, pad0 + 0, offs);
                svfloat32_t p01 = svld1_gather_u32index_f32(mask, pad0 + 1, offs);
                svfloat32_t p10 = svld1_gather_u32index_f32(mask, pad1 + 0, offs);
                svfloat32_t p11 = svld1_gather_u32index_f32(mask, pad1 + 1, offs);
                svfloat32_t dy1 = svld1_f32(mask, dy + d);
                svfloat32_t dy0 = svsub_f32_x(mask, _1, dy1);
                svfloat32_t dx1 = svld1_f32(mask, dx + d);
                svfloat32_t dx0 = svsub_f32_x(mask, _1, dx1);
                svfloat32_t d0 = svmla_f32_x(mask, svmul_f32_x(mask, dx0, p00), dx1, p01);
                svfloat32_t d1 = svmla_f32_x(mask, svmul_f32_x(mask, dx0, p10), dx1, p11);
                svst1_f32(mask, dst + d, svmla_f32_x(mask, svmul_f32_x(mask, dy0, d0), dy1, d1));
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetGridSample2d32fBlZ::SynetGridSample2d32fBlZ(const GridSample2dParam& param)
            : Base::SynetGridSample2d32fBlZ(param)
        {
            if (_sparse)
                _indexCoeffs = _param.align ? IndexCoeffs32fBlZ<1, 1> : IndexCoeffs32fBlZ<0, 1>;
            else
                _indexCoeffs = _param.align ? IndexCoeffs32fBlZ<1, 0> : IndexCoeffs32fBlZ<0, 0>;
            _bilinearInterp = BilinearInterp32fBlZ;
        }

        void* SynetGridSample2dInit(size_t batch, size_t channels, size_t srcH, size_t srcW, size_t dstH, size_t dstW,
            SimdTensorDataType type, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align)
        {
            GridSample2dParam param(batch, channels, srcH, srcW, dstH, dstW, type, interp, padding, align);
            if (!param.Valid())
                return NULL;
            if (param.Is32fBlZ())
                return new Sve2::SynetGridSample2d32fBlZ(param);
            else
                return new Base::SynetGridSample2dRef(param);
        }
    }
#endif
}
