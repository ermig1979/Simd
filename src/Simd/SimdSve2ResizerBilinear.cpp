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
#include "Simd/SimdMemory.h"
#include "Simd/SimdResizer.h"
#include "Simd/SimdResizerCommon.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        ResizerByteBilinear::ResizerByteBilinear(const ResParam& param)
            : Base::ResizerByteBilinear(param)
            , _blocks(0)
        {
        }

        size_t ResizerByteBilinear::BlockCountMax(size_t align)
        {
            return (size_t)Simd::Max(::ceil(float(_param.srcW) / (align - 1)), ::ceil(float(_param.dstW) * 2.0f / align));
        }

        void ResizerByteBilinear::EstimateParams()
        {
            if (_ax.data)
                return;
            const size_t A = svcntb();
            if (_param.channels == 1 && _param.srcW < 4 * _param.dstW)
                _blocks = BlockCountMax(A);
            float scale = (float)_param.srcW / _param.dstW;
            size_t size = _param.dstW * _param.channels * 2;
            _ax.Resize(AlignHi(size, A), false, _param.align);
            uint8_t* alphas = _ax.data;
            if (_blocks)
            {
                _ixg.Resize(_blocks);
                int block = 0;
                _ixg[0].src = 0;
                _ixg[0].dst = 0;
                for (int dstIndex = 0; dstIndex < (int)_param.dstW; ++dstIndex)
                {
                    float alpha = (float)((dstIndex + 0.5) * scale - 0.5);
                    int srcIndex = (int)::floor(alpha);
                    alpha -= srcIndex;

                    if (srcIndex < 0)
                    {
                        srcIndex = 0;
                        alpha = 0;
                    }

                    if (srcIndex >(int)_param.srcW - 2)
                    {
                        srcIndex = (int)_param.srcW - 2;
                        alpha = 1;
                    }

                    int dst = 2 * dstIndex - _ixg[block].dst;
                    int src = srcIndex - _ixg[block].src;
                    if (src >= (int)A - 1 || dst >= (int)A)
                    {
                        block++;
                        _ixg[block].src = Simd::Min(srcIndex, int(_param.srcW - A));
                        _ixg[block].dst = 2 * dstIndex;
                        dst = 0;
                        src = srcIndex - _ixg[block].src;
                    }
                    _ixg[block].shuffle[dst] = (uint8_t)src;
                    _ixg[block].shuffle[dst + 1] = (uint8_t)src + 1;

                    alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                    alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                    alphas += 2;
                }
                _blocks = block + 1;
            }
            else
            {
                _ix.Resize(_param.dstW);
                for (size_t i = 0; i < _param.dstW; ++i)
                {
                    float alpha = (float)((i + 0.5) * scale - 0.5);
                    ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                    alpha -= index;

                    if (index < 0)
                    {
                        index = 0;
                        alpha = 0;
                    }

                    if (index > (ptrdiff_t)_param.srcW - 2)
                    {
                        index = _param.srcW - 2;
                        alpha = 1;
                    }

                    _ix[i] = (int)index;
                    alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                    alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                    for (size_t channel = 1; channel < _param.channels; channel++)
                        ((uint16_t*)alphas)[channel] = *(uint16_t*)alphas;
                    alphas += 2 * _param.channels;
                }
            }
            _bx[0].Resize(AlignHi(size, A), false, _param.align);
            _bx[1].Resize(AlignHi(size, A), false, _param.align);
        }

        SIMD_INLINE void ResizerByteBilinearInterpolateX(const uint8_t* alpha, uint8_t* buffer)
        {
            const svbool_t mask8 = svptrue_b8(), mask16 = svptrue_b16();
            svuint8_t _src = svld1_u8(mask8, buffer);
            svuint8_t _alpha = svld1_u8(mask8, alpha);
            svuint16_t lo = svmul_u16_x(mask16, svunpklo_u16(svuzp1_u8(_src, _src)), svunpklo_u16(svuzp1_u8(_alpha, _alpha)));
            lo = svmla_u16_x(mask16, lo, svunpklo_u16(svuzp2_u8(_src, _src)), svunpklo_u16(svuzp2_u8(_alpha, _alpha)));
            svst1_u16(mask16, (uint16_t*)buffer, lo);
        }

        SIMD_INLINE svuint8_t PackU16ToU8(const svuint16_t& lo, const svuint16_t& hi)
        {
            return svuzp1_u8(svqxtnb_u16(lo), svqxtnb_u16(hi));
        }

        SIMD_INLINE svuint16_t ResizerByteBilinearInterpolateY(const uint16_t* pbx0, const uint16_t* pbx1, const svuint16_t alpha[2], const svbool_t& mask)
        {
            svuint16_t sum = svmul_u16_x(mask, svld1_u16(mask, pbx0), alpha[0]);
            sum = svmla_u16_x(mask, sum, svld1_u16(mask, pbx1), alpha[1]);
            return svlsr_n_u16_x(mask, svadd_n_u16_x(mask, sum, Base::BILINEAR_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        SIMD_INLINE void ResizerByteBilinearInterpolateY(const uint8_t* bx0, const uint8_t* bx1, const svuint16_t alpha[2],
            uint8_t* dst, const svbool_t& mask8, const svbool_t& maskLo, const svbool_t& maskHi, size_t half)
        {
            svuint16_t lo = ResizerByteBilinearInterpolateY((uint16_t*)bx0, (uint16_t*)bx1, alpha, maskLo);
            svuint16_t hi = ResizerByteBilinearInterpolateY((uint16_t*)(bx0 + half * 2), (uint16_t*)(bx1 + half * 2), alpha, maskHi);
            svst1_u8(mask8, dst, PackU16ToU8(lo, hi));
        }

        template<size_t N> void ResizerByteBilinear::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t size = 2 * _param.dstW * N;
            const size_t A = svcntb(), HA = svcnth(), DA = 2 * A;
            size_t aligned = AlignHi(size, DA) - DA;
            ptrdiff_t previous = -2;
            svuint16_t a[2];
            uint8_t* bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t* ax = _ax.data;
            const int32_t* ix = _ix.data;
            size_t dstW = _param.dstW;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = svdup_n_u16((uint16_t)(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = svdup_n_u16((uint16_t)_ay[yDst]);

                ptrdiff_t sy = _iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(bx[0], bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    uint8_t* pb = bx[k];
                    const uint8_t* psrc = src + (sy + k) * srcStride;
                    for (size_t x = 0; x < dstW; x++)
                    {
                        const uint8_t* ps = psrc + ix[x] * N;
                        uint8_t* pd = pb + 2 * x * N;
                        for (size_t c = 0; c < N; c++)
                        {
                            pd[2 * c + 0] = ps[c];
                            pd[2 * c + 1] = ps[c + N];
                        }
                    }

                    for (size_t i = 0; i < size; i += A)
                        ResizerByteBilinearInterpolateX(ax + i, pb + i);
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY(bx[0] + ib, bx[1] + ib, a, dst + id, svptrue_b8(), svptrue_b16(), svptrue_b16(), HA);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY(bx[0] + i, bx[1] + i, a, dst + i / 2, svwhilelt_b8(i / 2, size / 2),
                    svwhilelt_b16(i / 2, size / 2), svwhilelt_b16(i / 2 + HA, size / 2), HA);
            }
        }

        template <class Idx> SIMD_INLINE void ResizerByteBilinearLoadGrayInterpolated(const uint8_t* src, const Idx& index, const uint8_t* alpha, uint8_t* dst)
        {
            const svbool_t mask8 = svptrue_b8();
            svuint8_t _src = svld1_u8(mask8, src + index.src);
            svuint8_t _shuffle = svld1_u8(mask8, index.shuffle);
            svuint8_t _buffer = svtbl_u8(_src, _shuffle);
            svst1_u8(mask8, dst + index.dst, _buffer);
            ResizerByteBilinearInterpolateX(alpha + index.dst, dst + index.dst);
        }

        void ResizerByteBilinear::RunG(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t size = 2 * _param.dstW;
            const size_t A = svcntb(), HA = svcnth(), DA = 2 * A;
            size_t aligned = AlignHi(size, DA) - DA;
            size_t blocks = _blocks;
            ptrdiff_t previous = -2;
            svuint16_t a[2];
            uint8_t* bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t* ax = _ax.data;
            const Idx* ixg = _ixg.data;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                a[0] = svdup_n_u16((uint16_t)(Base::FRACTION_RANGE - _ay[yDst]));
                a[1] = svdup_n_u16((uint16_t)_ay[yDst]);

                ptrdiff_t sy = _iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(bx[0], bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    const uint8_t* psrc = src + (sy + k) * srcStride;
                    uint8_t* pdst = bx[k];
                    for (size_t i = 0; i < blocks; ++i)
                        ResizerByteBilinearLoadGrayInterpolated(psrc, ixg[i], ax, pdst);
                }

                for (size_t ib = 0, id = 0; ib < aligned; ib += DA, id += A)
                    ResizerByteBilinearInterpolateY(bx[0] + ib, bx[1] + ib, a, dst + id, svptrue_b8(), svptrue_b16(), svptrue_b16(), HA);
                size_t i = size - DA;
                ResizerByteBilinearInterpolateY(bx[0] + i, bx[1] + i, a, dst + i / 2, svwhilelt_b8(i / 2, size / 2),
                    svwhilelt_b16(i / 2, size / 2), svwhilelt_b16(i / 2 + HA, size / 2), HA);
            }
        }

        void ResizerByteBilinear::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            assert(_param.dstW >= svcntb());

            EstimateParams();
            switch (_param.channels)
            {
            case 1:
                if (_blocks)
                    RunG(src, srcStride, dst, dstStride);
                else
                    Run<1>(src, srcStride, dst, dstStride);
                break;
            case 2: Run<2>(src, srcStride, dst, dstStride); break;
            case 3: Run<3>(src, srcStride, dst, dstStride); break;
            case 4: Run<4>(src, srcStride, dst, dstStride); break;
            default:
                assert(0);
            }
        }
    }
#endif
}

