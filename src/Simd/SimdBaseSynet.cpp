/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAlignment.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SimdTensorFormatType SynetSpecifyTensorFormat(SimdTensorFormatType format)
        {
            const size_t alignment = Alignment();
            if (format == SimdTensorFormatNchwXc)
            {
                switch (alignment)
                {
                case 16: return SimdTensorFormatNchw4c;
                case 32: return SimdTensorFormatNchw8c;
                case 64: return SimdTensorFormatNchw16c;
                }
            }
            if (format == SimdTensorFormatOyxiXo)
            {
                switch (alignment)
                {
                case 16: return SimdTensorFormatOyxi4o;
                case 32: return SimdTensorFormatOyxi8o;
                case 64: return SimdTensorFormatOyxi16o;
                }
            }
            return SimdTensorFormatUnknown;
        }

        size_t SynetTensorAlignment(SimdTensorFormatType format)
        {
            switch (format)
            {
            case SimdTensorFormatNchw: return 1;
            case SimdTensorFormatNhwc: return 1;
            case SimdTensorFormatNchw4c: return 4;
            case SimdTensorFormatNchw8c: return 8;
            case SimdTensorFormatNchw16c: return 16;
            case SimdTensorFormatOiyx: return 1;
            case SimdTensorFormatYxio: return 1;
            case SimdTensorFormatOyxi4o: return 4;
            case SimdTensorFormatOyxi8o: return 8;
            case SimdTensorFormatOyxi16o: return 16;
            default: 
                assert(0); 
                return 0;
            }
        }

        void SynetAddBiasNchw(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; ++c)
            {
                float value = bias[c];
                size_t s = 0;
                for (; s < aligned; s += 4)
                {
                    dst[s + 0] += value;
                    dst[s + 1] += value;
                    dst[s + 2] += value;
                    dst[s + 3] += value;
                }
                for (; s < spatial; ++s)
                    dst[s] += value;
                dst += spatial;
            }
        }

        void SynetAddBiasNhwc(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(channels, 4);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += 4)
                {
                    dst[c + 0] += bias[c + 0];
                    dst[c + 1] += bias[c + 1];
                    dst[c + 2] += bias[c + 2];
                    dst[c + 3] += bias[c + 3];
                }
                for (; c < channels; ++c)
                    dst[c] += bias[c];
                dst += channels;
            }
        }

        template<int N> void SynetAddBiasNchwXc(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            for (size_t c = 0; c < channels; c += N)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t i = 0; i < N; ++i)
                        dst[i] += bias[i];
                    dst += N;
                }
                bias += N;
            }
        }

        void SynetAddBias(const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetAddBiasNchw(bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetAddBiasNhwc(bias, channels, spatial, dst);
            else if(format == SimdTensorFormatNchw4c)
                SynetAddBiasNchwXc<4>(bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetAddBiasNchwXc<8>(bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetAddBiasNchwXc<16>(bias, channels, spatial, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        void SynetAdd8i(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            int lower, upper;
            if (Base::Narrowed(compatibility))
                lower = Base::U8_NARROWED_MIN, upper = Base::U8_NARROWED_MAX;
            else
                lower = Base::U8_PRECISE_MIN, upper = Base::U8_PRECISE_MAX;
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == SimdTensorFormatNchw)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        for (size_t s = 0; s < spatial; ++s)
                        {
                            float a = float(aData[s]) * aScale[c] + aShift[c];
                            float b = float(bData[s]) * bScale[c] + bShift[c];
                            cData[s] = Base::SynetConvert32fTo8u(a + b, cScale[c], cShift[c], lower, upper);
                        }
                        aData += spatial, bData += spatial, cData += spatial;
                    }
                }
                else if (format == SimdTensorFormatNhwc)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t c = 0; c < channels; ++c)
                        {
                            float a = float(aData[c]) * aScale[c] + aShift[c];
                            float b = float(bData[c]) * bScale[c] + bShift[c];
                            cData[c] = Base::SynetConvert32fTo8u(a + b, cScale[c], cShift[c], lower, upper);
                        }
                        aData += channels, bData += channels, cData += channels;
                    }
                }
                else
                    assert(0);
            }
        }

        //---------------------------------------------------------------------

        template <SimdSynetEltwiseOperationType type> void SynetEltwiseLayerForward(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            for (; j < aligned; j += 4)
            {
                dst[j + 0] = SynetEltwiseLayerForward<type>(src0[j + 0], src1[j + 0]);
                dst[j + 1] = SynetEltwiseLayerForward<type>(src0[j + 1], src1[j + 1]);
                dst[j + 2] = SynetEltwiseLayerForward<type>(src0[j + 2], src1[j + 2]);
                dst[j + 3] = SynetEltwiseLayerForward<type>(src0[j + 3], src1[j + 3]);
            }
            for (; j < size; ++j)
                dst[j] = SynetEltwiseLayerForward<type>(src0[j], src1[j]);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                for (j = 0; j < aligned; j += 4)
                {
                    dst[j + 0] = SynetEltwiseLayerForward<type>(dst[j + 0], srci[j + 0]);
                    dst[j + 1] = SynetEltwiseLayerForward<type>(dst[j + 1], srci[j + 1]);
                    dst[j + 2] = SynetEltwiseLayerForward<type>(dst[j + 2], srci[j + 2]);
                    dst[j + 3] = SynetEltwiseLayerForward<type>(dst[j + 3], srci[j + 3]);
                }
                for (; j < size; ++j)
                    dst[j] = SynetEltwiseLayerForward<type>(dst[j], srci[j]);
            }
        }

        void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            const float * src0 = src[0];
            const float * src1 = src[1];
            float weight0 = weight[0], weight1 = weight[1];
            size_t j = 0;
            for (; j < aligned; j += 4)
            {
                dst[j + 0] = src0[j + 0] * weight0 + src1[j + 0] * weight1;
                dst[j + 1] = src0[j + 1] * weight0 + src1[j + 1] * weight1;
                dst[j + 2] = src0[j + 2] * weight0 + src1[j + 2] * weight1;
                dst[j + 3] = src0[j + 3] * weight0 + src1[j + 3] * weight1;
            }
            for (; j < size; ++j)
                dst[j] = src0[j] * weight0 + src1[j] * weight1;
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                float weighti = weight[i];
                for (j = 0; j < aligned; j += 4)
                {
                    dst[j + 0] += srci[j + 0] * weighti;
                    dst[j + 1] += srci[j + 1] * weighti;
                    dst[j + 2] += srci[j + 2] * weighti;
                    dst[j + 3] += srci[j + 3] * weighti;
                }
                for (; j < size; ++j)
                    dst[j] += srci[j] * weighti;
            }
        }

        void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            switch (type)
            {
            case SimdSynetEltwiseOperationProduct:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationSum:
                SynetEltwiseLayerForwardSum(src, weight, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMax:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMin:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(src, count, size, dst);
                break;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        void SynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst, SimdSynetCompatibilityType compatibility)
        {
            const size_t K2 = Base::Precise(compatibility) ? 0 : K / 2 * 2;
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    const int8_t* w = weight + j * K;
                    size_t k = 0;
                    int32_t sum = 0;
                    for (; k < K2; k += 2)
                        sum += RestrictRange(int(src[k + 0]) * int(w[k + 0]) + int(src[k + 1]) * int(w[k + 1]), SHRT_MIN, SHRT_MAX);
                    for (; k < K; ++k)
                        sum += int(src[k + 0]) * int(w[k + 0]);
                    dst[j] = sum;
                }
                src += K;
                dst += N;
            }
        }

        //---------------------------------------------------------------------

        void SynetLrnLayerCrossChannelsNchw(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            float k0 = k[0], k1 = k[1], k2 = k[2];
            Array32f sum(spatial, true), zero(spatial, true);
            for (size_t c = 0; c < half; ++c)
            {
                const float * pos = src + c * spatial;
                for (size_t s = 0; s < spatial; ++s)
                    sum[s] += Simd::Square(pos[s]);
            }
            for (size_t c = 0; c < channels; ++c)
            {
                const float * pos = (c < channels - half) ? src + half * spatial : zero.data;
                const float * neg = (c > half) ? src - (half + 1) * spatial : zero.data;
                for (size_t s = 0; s < spatial; ++s)
                {
                    sum[s] += Simd::Square(pos[s]);
                    sum[s] -= Simd::Square(neg[s]);
                    dst[s] = src[s] * Pow(k0 + k1 * sum[s], k2);
                }
                src += spatial;
                dst += spatial;
            }
        }

        void SynetLrnLayerCrossChannelsNhwc(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            float k0 = k[0], k1 = k[1], k2 = k[2];
            size_t beg = half + 1;
            size_t end = channels - half;
            for (size_t s = 0; s < spatial; ++s)
            {
                float sum = 0;
                for (size_t c = 0; c < half; ++c)
                    sum += Simd::Square(src[c]);
                for (size_t c = 0; c < beg; ++c)
                {
                    sum += Simd::Square(src[c + half]);
                    dst[c] = src[c] * Pow(k0 + k1 * sum, k2);
                }
                for (size_t c = beg; c < end; ++c)
                {
                    sum += Simd::Square(src[c + half]);
                    sum -= Simd::Square(src[c - half - 1]);
                    dst[c] = src[c] * Pow(k0 + k1 * sum, k2);
                }
                for (size_t c = end; c < channels; ++c)
                {
                    sum -= Simd::Square(src[c - half - 1]);
                    dst[c] = src[c] * Pow(k0 + k1 * sum, k2);
                }
                src += channels;
                dst += channels;
            }
        }

        template<int N> void SynetLrnLayerCrossChannelsNchwXc(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            assert(0);
        }

        void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNchw)
                SynetLrnLayerCrossChannelsNchw(src, half, channels, spatial, k, dst);
            else if (format == SimdTensorFormatNhwc)
                SynetLrnLayerCrossChannelsNhwc(src, half, channels, spatial, k, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetLrnLayerCrossChannelsNchwXc<4>(src, half, channels, spatial, k, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetLrnLayerCrossChannelsNchwXc<8>(src, half, channels, spatial, k, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetLrnLayerCrossChannelsNchwXc<16>(src, half, channels, spatial, k, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        void SynetShuffleLayerForward(const float* src0, const float* src1, size_t channels0, size_t channels1, size_t spatial, float* dst0, float* dst1, SimdTensorFormatType format, int type)
        {
            size_t channels = (channels0 + channels1) / 2, size = sizeof(float) * spatial;
            switch (type)
            {
            case 0:
                if (format == SimdTensorFormatNchw)
                {
                    size_t cd = 0;
                    for (size_t cs = 0; cs < channels0; cs += 2, cd += 1)
                    {
                        memcpy(dst0, src0 + 0 * spatial, size);
                        memcpy(dst1, src0 + 1 * spatial, size);
                        src0 += 2 * spatial;
                        dst0 += spatial;
                        dst1 += spatial;
                    }
                    for (size_t cs = 0; cs < channels1; cs += 2, cd += 1)
                    {
                        memcpy(dst0, src1 + 0 * spatial, size);
                        memcpy(dst1, src1 + 1 * spatial, size);
                        src1 += 2 * spatial;
                        dst0 += spatial;
                        dst1 += spatial;
                    }
                }
                else if (format == SimdTensorFormatNhwc)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cd = 0;
                        for (size_t cs = 0; cs < channels0; cs += 2, cd += 1)
                        {
                            dst0[cd] = src0[cs + 0];
                            dst1[cd] = src0[cs + 1];
                        }
                        for (size_t cs = 0; cs < channels1; cs += 2, cd += 1)
                        {
                            dst0[cd] = src1[cs + 0];
                            dst1[cd] = src1[cs + 1];
                        }
                        src0 += channels0;
                        src1 += channels1;
                        dst0 += channels;
                        dst1 += channels;
                    }
                }
                else
                    assert(0);                
                break;
            case 1:
                if (format == SimdTensorFormatNchw)
                {
                    size_t cs = 0;
                    for (size_t cd = 0; cd < channels0; cs += 1, cd += 2)
                    {
                        memcpy(dst0 + 0 * spatial, src0, size);
                        memcpy(dst0 + 1 * spatial, src1, size);
                        src0 += spatial;
                        src1 += spatial;
                        dst0 += 2 * spatial;
                    }
                    for (size_t cd = 0; cd < channels1; cs += 1, cd += 2)
                    {
                        memcpy(dst1 + 0 * spatial, src0, size);
                        memcpy(dst1 + 1 * spatial, src1, size);
                        src0 += spatial;
                        src1 += spatial;
                        dst1 += 2 * spatial;
                    }
                }                
                else if (format == SimdTensorFormatNhwc)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cs = 0;
                        for (size_t cd = 0; cd < channels0; cd += 2, cs += 1)
                        {
                            dst0[cd + 0] = src0[cs];
                            dst0[cd + 1] = src1[cs];
                        }
                        for (size_t cd = 0; cd < channels1; cd += 2, cs += 1)
                        {
                            dst1[cd + 0] = src0[cs];
                            dst1[cd + 1] = src1[cs];
                        }
                        src0 += channels;
                        src1 += channels;
                        dst0 += channels0;
                        dst1 += channels1;
                    }
                }
                else
                    assert(0);
                break;
            default:
                assert(0);
            }

        }

        //---------------------------------------------------------------------

        void SynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst)
        {
            if (inner == 1 && count == 2)
            {
                for (size_t o = 0; o < outer; ++o)
                {
                    float max = Simd::Max(src[0], src[1]);
                    float exp0 = ::exp(src[0] - max);
                    float exp1 = ::exp(src[1] - max);
                    float sum = exp0 + exp1;
                    dst[0] = exp0 / sum;
                    dst[1] = exp1 / sum;
                    src += 2;
                    dst += 2;
                }
            }
            else
            {
                Array32f tmp(inner * 2);
                const float * s;
                float * max = tmp.data, *sum = tmp.data + inner, *d;
                for (size_t o = 0; o < outer; ++o)
                {
                    for (size_t i = 0; i < inner; ++i)
                        max[i] = src[i];
                    s = src + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                            max[i] = Simd::Max(max[i], s[i]);
                        s += inner;
                    }

                    s = src;
                    d = dst;
                    for (size_t i = 0; i < inner; ++i)
                        sum[i] = 0;
                    for (size_t c = 0; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                        {
                            d[i] = ::exp(s[i] - max[i]);
                            sum[i] += d[i];
                        }
                        s += inner;
                        d += inner;
                    }

                    d = dst;
                    for (size_t c = 0; c < count; ++c)
                    {
                        for (size_t i = 0; i < inner; ++i)
                            d[i] /= sum[i];
                        d += inner;
                    }
                    src += count * inner;
                    dst += count * inner;
                }
            }
        }

        //---------------------------------------------------------------------

        template<SimdSynetUnaryOperation32fType type> void SynetUnaryOperation32fLayerForward(const float* src, size_t size, float* dst)
        {
            size_t size4 = AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4)
            {
                dst[i + 0] = SynetUnaryOperation32f<type>(src[i + 0]);
                dst[i + 1] = SynetUnaryOperation32f<type>(src[i + 1]);
                dst[i + 2] = SynetUnaryOperation32f<type>(src[i + 2]);
                dst[i + 3] = SynetUnaryOperation32f<type>(src[i + 3]);
            }
            for (; i < size; ++i)
                dst[i] = SynetUnaryOperation32f<type>(src[i]);
        }

        void SynetUnaryOperation32fLayerForward(const float * src, size_t size, SimdSynetUnaryOperation32fType type, float * dst)
        {
            switch (type)
            {
            case SimdSynetUnaryOperation32fAbs: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fAbs>(src, size, dst); break;
            case SimdSynetUnaryOperation32fExp: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fExp>(src, size, dst); break;
            case SimdSynetUnaryOperation32fLog: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fLog>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNeg: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fNeg>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRsqrt: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fRsqrt>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSqrt: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fSqrt>(src, size, dst); break;
            case SimdSynetUnaryOperation32fTanh: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fTanh>(src, size, dst); break;
            case SimdSynetUnaryOperation32fZero: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fZero>(src, size, dst); break;
            default:
                assert(0);
            }
        }
    }
#endif
}
