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
#include "Simd/SimdBFloat16.h"

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
            _ax.Resize(AlignHi(_param.dstW, A) * _param.channels * 2, false, _param.align);
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
            size_t size = AlignHi(_param.dstW, _param.align) * _param.channels * 2 + SIMD_ALIGN;
            _bx[0].Resize(size, false, _param.align);
            _bx[1].Resize(size, false, _param.align);
        }

        SIMD_INLINE svuint16_t ResizerByteBilinearMaddubs(const svuint8_t& src, const svuint8_t& alpha)
        {
            return svmlalt_u16(svmlalb_u16(svdup_n_u16(0), src, alpha), src, alpha);
        }

        SIMD_INLINE svuint8_t ResizerByteBilinearShuffleX2()
        {
            const svbool_t all = svptrue_b8();
            svuint8_t i = svindex_u8(0, 1);
            svuint8_t base = svand_n_u8_x(all, i, 0xFC);
            svuint8_t even = svlsl_n_u8_x(all, svand_n_u8_x(all, i, 1), 1);
            svuint8_t odd = svlsr_n_u8_x(all, svand_n_u8_x(all, i, 2), 1);
            return svadd_u8_x(all, base, svadd_u8_x(all, even, odd));
        }

        SIMD_INLINE svuint8_t ResizerByteBilinearShuffleX4()
        {
            const svbool_t all = svptrue_b8();
            svuint8_t i = svindex_u8(0, 1);
            svuint8_t base = svand_n_u8_x(all, i, 0xF8);
            svuint8_t p = svand_n_u8_x(all, i, 7);
            svuint8_t lo = svlsl_n_u8_x(all, svand_n_u8_x(all, p, 1), 2);
            svuint8_t hi = svlsr_n_u8_x(all, p, 1);
            return svadd_u8_x(all, base, svadd_u8_x(all, lo, hi));
        }

        template<size_t N> SIMD_INLINE svuint8_t ResizerByteBilinearShuffleX(const svuint8_t& src, const svuint8_t& index)
        {
            if (N == 1)
                return src;
            else
                return svtbl_u8(src, index);
        }

        template<size_t N> SIMD_INLINE void ResizerByteBilinearInterpolateX(const uint8_t* alpha, uint8_t* buffer, const svuint8_t& index)
        {
            const svbool_t mask8 = svptrue_b8();
            const svbool_t mask16 = svptrue_b16();
            svuint8_t src = ResizerByteBilinearShuffleX<N>(svld1_u8(mask8, buffer), index);
            svst1_u16(mask16, (uint16_t*)buffer, ResizerByteBilinearMaddubs(src, svld1_u8(mask8, alpha)));
        }

        SIMD_INLINE void ResizerByteBilinearInitShuffleX3(uint8_t* idx, size_t A)
        {
            size_t step = AlignLo(2 * A, size_t(6));
            for (size_t i = 0; i < 2 * A; ++i)
            {
                if (i < step)
                {
                    size_t pair = i / 2;
                    idx[i] = (uint8_t)((pair / 3) * 6 + (i & 1) * 3 + (pair % 3));
                }
                else
                    idx[i] = 0;
            }
        }

        SIMD_INLINE void ResizerByteBilinearInterpolateX3(const uint8_t* alpha, uint8_t* buffer, size_t n,
            const svuint8_t& idx0, const svuint8_t& idx1)
        {
            const size_t A = svcntb();
            const size_t HA = svcnth();
            svbool_t mask0 = svwhilelt_b8((size_t)0, n);
            svbool_t mask1 = svwhilelt_b8(A, n);
            svuint8x2_t src = svcreate2_u8(svld1_u8(mask0, buffer), svld1_u8(mask1, buffer + A));
            svst1_u16(svwhilelt_b16((size_t)0, n / 2), (uint16_t*)buffer,
                ResizerByteBilinearMaddubs(svtbl2_u8(src, idx0), svld1_u8(mask0, alpha)));
            svst1_u16(svwhilelt_b16(HA, n / 2), (uint16_t*)buffer + HA,
                ResizerByteBilinearMaddubs(svtbl2_u8(src, idx1), svld1_u8(mask1, alpha + A)));
        }

        SIMD_INLINE svuint8_t PackU16ToU8(const svuint16_t& lo, const svuint16_t& hi)
        {
            return svuzp1_u8(svqxtnb_u16(lo), svqxtnb_u16(hi));
        }

        SIMD_INLINE void ResizerByteBilinearInterpolateY(const uint16_t* bx0, const uint16_t* bx1,
            const svuint16_t& alpha0, const svuint16_t& alpha1, uint8_t* dst, const svbool_t& mask)
        {
            svuint16_t sum = svmul_u16_x(mask, svld1_u16(mask, bx0), alpha0);
            sum = svmla_u16_x(mask, sum, svld1_u16(mask, bx1), alpha1);
            svst1b_u16(mask, dst, svrshr_n_u16_x(mask, sum, Base::BILINEAR_SHIFT));
        }

        SIMD_INLINE void ResizerByteBilinearInterpolateYRow(const uint8_t* bx0, const uint8_t* bx1,
            const svuint16_t& alpha0, const svuint16_t& alpha1, uint8_t* dst, size_t rs)
        {
            const size_t HA = svcnth();
            const svbool_t mask16 = svptrue_b16();
            size_t i = 0;
            for (; i + 2 * HA <= rs; i += 2 * HA)
            {
                ResizerByteBilinearInterpolateY((const uint16_t*)(bx0 + 2 * i), (const uint16_t*)(bx1 + 2 * i), alpha0, alpha1, dst + i, mask16);
                ResizerByteBilinearInterpolateY((const uint16_t*)(bx0 + 2 * (i + HA)), (const uint16_t*)(bx1 + 2 * (i + HA)), alpha0, alpha1, dst + i + HA, mask16);
            }
            for (; i + HA <= rs; i += HA)
                ResizerByteBilinearInterpolateY((const uint16_t*)(bx0 + 2 * i), (const uint16_t*)(bx1 + 2 * i), alpha0, alpha1, dst + i, mask16);
            if (i < rs)
                ResizerByteBilinearInterpolateY((const uint16_t*)(bx0 + 2 * i), (const uint16_t*)(bx1 + 2 * i), alpha0, alpha1, dst + i, svwhilelt_b16(i, rs));
        }

        template<size_t N> void ResizerByteBilinear::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            struct One { uint8_t val[N]; };
            struct Two { uint8_t val[N * 2]; };

            size_t size = 2 * _param.dstW * N;
            size_t rs = _param.dstW * N;
            const size_t A = svcntb();
            ptrdiff_t previous = -2;
            uint8_t* bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t* ax = _ax.data;
            const int32_t* ix = _ix.data;
            size_t dstW = _param.dstW;

            svuint8_t shuffle = svindex_u8(0, 1);
            if (N == 2)
                shuffle = ResizerByteBilinearShuffleX2();
            else if (N == 4)
                shuffle = ResizerByteBilinearShuffleX4();

            uint8_t idx3[2 * SIMD_SVE2_VECTOR_SIZE_MAX];
            svuint8_t idx30 = shuffle, idx31 = shuffle;
            size_t step3 = AlignLo(2 * A, size_t(6));
            if (N == 3)
            {
                ResizerByteBilinearInitShuffleX3(idx3, A);
                idx30 = svld1_u8(svptrue_b8(), idx3);
                idx31 = svld1_u8(svptrue_b8(), idx3 + A);
            }

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                svuint16_t a0 = svdup_n_u16((uint16_t)(Base::FRACTION_RANGE - _ay[yDst]));
                svuint16_t a1 = svdup_n_u16((uint16_t)_ay[yDst]);

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
                    Two* pb = (Two*)bx[k];
                    const One* psrc = (const One*)(src + (sy + k) * srcStride);
                    for (size_t x = 0; x < dstW; x++)
                        pb[x] = *(Two*)(psrc + ix[x]);

                    uint8_t* pbx = bx[k];
                    if (N == 3)
                    {
                        size_t i = 0;
                        for (; i + step3 <= size; i += step3)
                            ResizerByteBilinearInterpolateX3(ax + i, pbx + i, step3, idx30, idx31);
                        if (i < size)
                            ResizerByteBilinearInterpolateX3(ax + i, pbx + i, size - i, idx30, idx31);
                    }
                    else
                    {
                        size_t aligned = AlignLo(size, 2 * A);
                        size_t i = 0;
                        for (; i < aligned; i += 2 * A)
                        {
                            ResizerByteBilinearInterpolateX<N>(ax + i, pbx + i, shuffle);
                            ResizerByteBilinearInterpolateX<N>(ax + i + A, pbx + i + A, shuffle);
                        }
                        for (; i < size; i += A)
                            ResizerByteBilinearInterpolateX<N>(ax + i, pbx + i, shuffle);
                    }
                }

                ResizerByteBilinearInterpolateYRow(bx[0], bx[1], a0, a1, dst, rs);
            }
        }

        template <class Idx> SIMD_INLINE void ResizerByteBilinearLoadGrayInterpolated(const uint8_t* src, const Idx& index, const uint8_t* alpha, uint8_t* dst)
        {
            const svbool_t mask8 = svptrue_b8();
            const svbool_t mask16 = svptrue_b16();
            svuint8_t gathered = svtbl_u8(svld1_u8(mask8, src + index.src), svld1_u8(mask8, index.shuffle));
            svst1_u16(mask16, (uint16_t*)(dst + index.dst), ResizerByteBilinearMaddubs(gathered, svld1_u8(mask8, alpha + index.dst)));
        }

        void ResizerByteBilinear::RunG(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW;
            size_t blocks = _blocks;
            ptrdiff_t previous = -2;
            uint8_t* bx[2] = { _bx[0].data, _bx[1].data };
            const uint8_t* ax = _ax.data;
            const Idx* ixg = _ixg.data;

            for (size_t yDst = 0; yDst < _param.dstH; yDst++, dst += dstStride)
            {
                svuint16_t a0 = svdup_n_u16((uint16_t)(Base::FRACTION_RANGE - _ay[yDst]));
                svuint16_t a1 = svdup_n_u16((uint16_t)_ay[yDst]);

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

                ResizerByteBilinearInterpolateYRow(bx[0], bx[1], a0, a1, dst, rs);
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

        //-------------------------------------------------------------------------------------------------

        ResizerByteBilinearOpenCv::ResizerByteBilinearOpenCv(const ResParam& param)
            : Base::ResizerByteBilinearOpenCv(param)
        {
        }

        void ResizerByteBilinearOpenCv::EstimateParams()
        {
            if (_ax.data)
                return;
            const size_t A = svcntb();
            size_t rs = _param.dstW * _param.channels;
            size_t size = 2 * rs;
            _ix.Resize(rs);
            _ax.Resize(AlignHi(size, A), false, _param.align);
            EstimateIndexAlpha(_param.srcW, _param.dstW, _param.channels, _ix.data, _ax.data, Base::LINEAR_X_RANGE);
            _sx.Resize(AlignHi(size, A), false, _param.align);
            _bx[0].Resize(AlignHi(rs, A), false, _param.align);
            _bx[1].Resize(AlignHi(rs, A), false, _param.align);
        }

        SIMD_INLINE void ResizerByteBilinearOpenCvInterpolateX(const uint8_t* src, const int16_t* alpha, int16_t* dst)
        {
            const svbool_t mask8 = svptrue_b8(), mask16 = svptrue_b16(), mask32 = svptrue_b32();
            const size_t HA = svcnth();
            svuint8_t _src = svld1_u8(mask8, src);
            svuint16_t src0 = svunpklo_u16(svuzp1_u8(_src, _src));
            svuint16_t src1 = svunpklo_u16(svuzp2_u8(_src, _src));
            svint16_t alphaLo = svld1_s16(mask16, alpha);
            svint16_t alphaHi = svld1_s16(mask16, alpha + HA);
            svuint16_t alpha0 = svreinterpret_u16_s16(svuzp1_s16(alphaLo, alphaHi));
            svuint16_t alpha1 = svreinterpret_u16_s16(svuzp2_s16(alphaLo, alphaHi));
            svuint32_t lo = svmul_u32_x(mask32, svunpklo_u32(src0), svunpklo_u32(alpha0));
            svuint32_t hi = svmul_u32_x(mask32, svunpkhi_u32(src0), svunpkhi_u32(alpha0));
            lo = svmla_u32_x(mask32, lo, svunpklo_u32(src1), svunpklo_u32(alpha1));
            hi = svmla_u32_x(mask32, hi, svunpkhi_u32(src1), svunpkhi_u32(alpha1));
            lo = svlsr_n_u32_x(mask32, lo, Base::LINEAR_X_RSHIFT);
            hi = svlsr_n_u32_x(mask32, hi, Base::LINEAR_X_RSHIFT);
            svst1_s16(mask16, dst, svreinterpret_s16_u16(svuzp1_u16(svreinterpret_u16_u32(lo), svreinterpret_u16_u32(hi))));
        }

        SIMD_INLINE svuint16_t ResizerByteBilinearOpenCvInterpolateY(const int16_t* bx0, const int16_t* bx1,
            const svuint16_t& alpha0, const svuint16_t& alpha1, const svbool_t& mask)
        {
            svuint16_t sum = svmulh_u16_x(mask, svreinterpret_u16_s16(svld1_s16(mask, bx0)), alpha0);
            sum = svadd_u16_x(mask, sum, svmulh_u16_x(mask, svreinterpret_u16_s16(svld1_s16(mask, bx1)), alpha1));
            return svlsr_n_u16_x(mask, svadd_n_u16_x(mask, sum, Base::LINEAR_Y_ROUND), Base::LINEAR_Y_RSHIFT);
        }

        SIMD_INLINE void ResizerByteBilinearOpenCvInterpolateY(const int16_t* bx0, const int16_t* bx1,
            const svuint16_t& alpha0, const svuint16_t& alpha1, uint8_t* dst,
            const svbool_t& mask8, const svbool_t& maskLo, const svbool_t& maskHi, size_t half)
        {
            svuint16_t lo = ResizerByteBilinearOpenCvInterpolateY(bx0, bx1, alpha0, alpha1, maskLo);
            svuint16_t hi = ResizerByteBilinearOpenCvInterpolateY(bx0 + half, bx1 + half, alpha0, alpha1, maskHi);
            svst1_u8(mask8, dst, PackU16ToU8(lo, hi));
        }

        template<size_t N> void ResizerByteBilinearOpenCv::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            size_t rs = _param.dstW * N, size = 2 * rs;
            const size_t A = svcntb(), HA = svcnth();
            size_t sizeA = AlignHi(size, A);
            size_t aligned = AlignHi(rs, A) - A;
            ptrdiff_t previous = -2;
            int16_t* bx[2] = { _bx[0].data, _bx[1].data };
            const int16_t* ax = _ax.data;
            const int32_t* ix = _ix.data;
            uint8_t* sx = _sx.data;

            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                svuint16_t a0 = svdup_n_u16((uint16_t)_ay[dy * 2 + 0]);
                svuint16_t a1 = svdup_n_u16((uint16_t)_ay[dy * 2 + 1]);

                ptrdiff_t sy = _iy[dy];
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
                    for (size_t dx = 0; dx < rs; dx++)
                    {
                        size_t sx0 = ix[dx];
                        sx[2 * dx + 0] = psrc[sx0];
                        sx[2 * dx + 1] = psrc[sx0 + N];
                    }
                    for (size_t i = 0; i < sizeA; i += A)
                        ResizerByteBilinearOpenCvInterpolateX(sx + i, ax + i, bx[k] + i / 2);
                }

                for (size_t i = 0; i < aligned; i += A)
                    ResizerByteBilinearOpenCvInterpolateY(bx[0] + i, bx[1] + i, a0, a1, dst + i, svptrue_b8(), svptrue_b16(), svptrue_b16(), HA);
                size_t i = rs - A;
                ResizerByteBilinearOpenCvInterpolateY(bx[0] + i, bx[1] + i, a0, a1, dst + i, svwhilelt_b8(i, rs),
                    svwhilelt_b16(i, rs), svwhilelt_b16(i + HA, rs), HA);
            }
        }

        void ResizerByteBilinearOpenCv::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            assert(_param.dstW >= svcntb());

            EstimateParams();
            switch (_param.channels)
            {
            case 1: Run<1>(src, srcStride, dst, dstStride); break;
            case 2: Run<2>(src, srcStride, dst, dstStride); break;
            case 3: Run<3>(src, srcStride, dst, dstStride); break;
            case 4: Run<4>(src, srcStride, dst, dstStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        ResizerFloatBilinear::ResizerFloatBilinear(const ResParam& param)
            : Base::ResizerFloatBilinear(param)
        {
        }

        SIMD_INLINE void ResizerFloatBilinearInterpolateX(const float* src, size_t channels, const int32_t* ix, const float* ax, float* dst, size_t offset, size_t size)
        {
            svbool_t pg = svwhilelt_b32(offset, size);
            svuint32_t idx = svreinterpret_u32_s32(svld1_s32(pg, ix + offset));
            svfloat32_t fx1 = svld1_f32(pg, ax + offset);
            svfloat32_t fx0 = svsub_f32_x(pg, svdup_n_f32(1.0f), fx1);
            svfloat32_t s0 = svld1_gather_u32index_f32(pg, src, idx);
            svfloat32_t s1 = svld1_gather_u32index_f32(pg, src + channels, idx);
            svst1_f32(pg, dst + offset, svmla_f32_x(pg, svmul_f32_x(pg, s0, fx0), s1, fx1));
        }

        SIMD_INLINE void ResizerFloatBilinearInterpolateY(const float* bx0, const float* bx1, const svfloat32_t& fy0, const svfloat32_t& fy1, float* dst, size_t offset, size_t size)
        {
            svbool_t pg = svwhilelt_b32(offset, size);
            svfloat32_t b0 = svld1_f32(pg, bx0 + offset);
            svfloat32_t b1 = svld1_f32(pg, bx1 + offset);
            svst1_f32(pg, dst + offset, svmla_f32_x(pg, svmul_f32_x(pg, b0, fy0), b1, fy1));
        }

        void ResizerFloatBilinear::Run(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
            assert(_rowBuf);

            size_t cn = _param.channels;
            size_t rs = _param.dstW * cn;
            size_t F = svcntw(), rsF = AlignLo(rs, F);
            float* pbx[2] = { _bx[0].data, _bx[1].data };
            int32_t prev = -2;
            for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
            {
                svfloat32_t fy1 = svdup_n_f32(_ay[dy]);
                svfloat32_t fy0 = svdup_n_f32(1.0f - _ay[dy]);
                int32_t sy = _iy[dy];
                int32_t k = 0;

                if (sy == prev)
                    k = 2;
                else if (sy == prev + 1)
                {
                    Swap(pbx[0], pbx[1]);
                    k = 1;
                }

                prev = sy;

                for (; k < 2; k++)
                {
                    float* pb = pbx[k];
                    const float* ps = src + (sy + k) * srcStride;
                    size_t dx = 0;
                    for (; dx < rsF; dx += F)
                        ResizerFloatBilinearInterpolateX(ps, cn, _ix.data, _ax.data, pb, dx, rs);
                    if (dx < rs)
                        ResizerFloatBilinearInterpolateX(ps, cn, _ix.data, _ax.data, pb, dx, rs);
                }

                size_t dx = 0;
                for (; dx < rsF; dx += F)
                    ResizerFloatBilinearInterpolateY(pbx[0], pbx[1], fy0, fy1, dst, dx, rs);
                if (dx < rs)
                    ResizerFloatBilinearInterpolateY(pbx[0], pbx[1], fy0, fy1, dst, dx, rs);
            }
        }

        //-------------------------------------------------------------------------------------------------

        ResizerBf16Bilinear::ResizerBf16Bilinear(const ResParam& param)
            : Base::ResizerBf16Bilinear(param)
        {
        }

        SIMD_INLINE svuint32_t Float32ToBFloat16(svfloat32_t value, const svbool_t& mask)
        {
            svuint32_t bits = svreinterpret_u32_f32(value);
            svuint32_t round = svadd_n_u32_x(mask, svand_n_u32_x(mask, svlsr_n_u32_x(mask, bits, Base::Bf16::SHIFT), 1), Base::Bf16::ROUND);
            return svlsr_n_u32_x(mask, svadd_u32_x(mask, bits, round), Base::Bf16::SHIFT);
        }

        SIMD_INLINE svfloat32_t BFloat16ToFloat32(svuint32_t value, const svbool_t& mask)
        {
            return svreinterpret_f32_u32(svlsl_n_u32_x(mask, value, Base::Bf16::SHIFT));
        }

        SIMD_INLINE svfloat32_t LoadBf16(const uint16_t* src, const svbool_t& mask)
        {
            return BFloat16ToFloat32(svld1uh_u32(mask, src), mask);
        }

        SIMD_INLINE svfloat32_t GatherBf16(const uint16_t* src, const svuint32_t& index, const svbool_t& mask)
        {
            return BFloat16ToFloat32(svld1uh_gather_u32index_u32(mask, src, index), mask);
        }

        SIMD_INLINE void ResizerBf16BilinearInterpolateX(const uint16_t* src, size_t channels, const int32_t* ix, const float* ax, float* dst, size_t offset, size_t size)
        {
            svbool_t pg = svwhilelt_b32(offset, size);
            svuint32_t idx = svreinterpret_u32_s32(svld1_s32(pg, ix + offset));
            svfloat32_t fx1 = svld1_f32(pg, ax + offset);
            svfloat32_t fx0 = svsub_f32_x(pg, svdup_n_f32(1.0f), fx1);
            svfloat32_t s0 = GatherBf16(src, idx, pg);
            svfloat32_t s1 = GatherBf16(src + channels, idx, pg);
            svst1_f32(pg, dst + offset, svmla_f32_x(pg, svmul_f32_x(pg, s0, fx0), s1, fx1));
        }

        SIMD_INLINE void ResizerBf16BilinearInterpolateY(const float* bx0, const float* bx1, const svfloat32_t& fy0, const svfloat32_t& fy1, uint16_t* dst, size_t offset, size_t size)
        {
            svbool_t pg = svwhilelt_b32(offset, size);
            svfloat32_t b0 = svld1_f32(pg, bx0 + offset);
            svfloat32_t b1 = svld1_f32(pg, bx1 + offset);
            svst1h_u32(pg, dst + offset, Float32ToBFloat16(svmla_f32_x(pg, svmul_f32_x(pg, b0, fy0), b1, fy1), pg));
        }

        SIMD_INLINE void ResizerBf16BilinearInterpolate(const uint16_t* src0, const uint16_t* src1, size_t channels, const svfloat32_t& fy0, const svfloat32_t& fy1,
            const svfloat32_t& fx0, const svfloat32_t& fx1, uint16_t* dst, size_t offset, size_t size)
        {
            svbool_t pg = svwhilelt_b32(offset, size);
            svfloat32_t s00 = LoadBf16(src0 + offset, pg);
            svfloat32_t s01 = LoadBf16(src0 + channels + offset, pg);
            svfloat32_t s10 = LoadBf16(src1 + offset, pg);
            svfloat32_t s11 = LoadBf16(src1 + channels + offset, pg);
            svfloat32_t r0 = svmla_f32_x(pg, svmul_f32_x(pg, s00, fx0), s01, fx1);
            svfloat32_t r1 = svmla_f32_x(pg, svmul_f32_x(pg, s10, fx0), s11, fx1);
            svst1h_u32(pg, dst + offset, Float32ToBFloat16(svmla_f32_x(pg, svmul_f32_x(pg, r0, fy0), r1, fy1), pg));
        }

        void ResizerBf16Bilinear::Run(const uint16_t* src, size_t srcStride, uint16_t* dst, size_t dstStride)
        {
            size_t cn = _param.channels;
            size_t F = svcntw();
            if (_rowBuf)
            {
                size_t rs = _param.dstW * cn, rsF = AlignLo(rs, F);
                float* pbx[2] = { _bx[0].data, _bx[1].data };
                int32_t prev = -2;
                for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                {
                    svfloat32_t fy1 = svdup_n_f32(_ay[dy]);
                    svfloat32_t fy0 = svdup_n_f32(1.0f - _ay[dy]);
                    int32_t sy = _iy[dy];
                    int32_t k = 0;

                    if (sy == prev)
                        k = 2;
                    else if (sy == prev + 1)
                    {
                        Swap(pbx[0], pbx[1]);
                        k = 1;
                    }

                    prev = sy;

                    for (; k < 2; k++)
                    {
                        float* pb = pbx[k];
                        const uint16_t* ps = src + (sy + k) * srcStride;
                        size_t dx = 0;
                        for (; dx < rsF; dx += F)
                            ResizerBf16BilinearInterpolateX(ps, cn, _ix.data, _ax.data, pb, dx, rs);
                        if (dx < rs)
                            ResizerBf16BilinearInterpolateX(ps, cn, _ix.data, _ax.data, pb, dx, rs);
                    }

                    size_t dx = 0;
                    for (; dx < rsF; dx += F)
                        ResizerBf16BilinearInterpolateY(pbx[0], pbx[1], fy0, fy1, dst, dx, rs);
                    if (dx < rs)
                        ResizerBf16BilinearInterpolateY(pbx[0], pbx[1], fy0, fy1, dst, dx, rs);
                }
            }
            else
            {
                size_t cnF = AlignLo(cn, F);
                for (size_t dy = 0; dy < _param.dstH; dy++, dst += dstStride)
                {
                    svfloat32_t fy1 = svdup_n_f32(_ay[dy]);
                    svfloat32_t fy0 = svdup_n_f32(1.0f - _ay[dy]);
                    const uint16_t* src0 = src + _iy[dy] * srcStride;
                    const uint16_t* src1 = src0 + srcStride;
                    for (size_t dx = 0; dx < _param.dstW; dx++)
                    {
                        const uint16_t* ps0 = src0 + _ix[dx];
                        const uint16_t* ps1 = src1 + _ix[dx];
                        uint16_t* pd = dst + dx * cn;
                        svfloat32_t fx1 = svdup_n_f32(_ax[dx]);
                        svfloat32_t fx0 = svsub_f32_x(svptrue_b32(), svdup_n_f32(1.0f), fx1);
                        size_t c = 0;
                        for (; c < cnF; c += F)
                            ResizerBf16BilinearInterpolate(ps0, ps1, cn, fy0, fy1, fx0, fx1, pd, c, cn);
                        if (c < cn)
                            ResizerBf16BilinearInterpolate(ps0, ps1, cn, fy0, fy1, fx0, fx1, pd, c, cn);
                    }
                }
            }
        }
    }
#endif
}

