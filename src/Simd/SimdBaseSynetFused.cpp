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
#include "Simd/SimdMemory.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void SynetFusedLayerForward0Nchw(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; ++c)
            {
                float _bias = bias[c];
                float _scale = scale[c];
                size_t s = 0;
                for (; s < aligned; s += 4)
                {
                    dst[s + 0] = SynetFusedLayerForward0(src[s + 0] + _bias, _scale);
                    dst[s + 1] = SynetFusedLayerForward0(src[s + 1] + _bias, _scale);
                    dst[s + 2] = SynetFusedLayerForward0(src[s + 2] + _bias, _scale);
                    dst[s + 3] = SynetFusedLayerForward0(src[s + 3] + _bias, _scale);
                }
                for (; s < spatial; ++s)
                    dst[s] = SynetFusedLayerForward0(src[s] + _bias, _scale);
                src += spatial;
                dst += spatial;
            }
        }

        void SynetFusedLayerForward0Nhwc(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(channels, 4);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += 4)
                {
                    dst[c + 0] = SynetFusedLayerForward0(src[c + 0] + bias[c + 0], scale[c + 0]);
                    dst[c + 1] = SynetFusedLayerForward0(src[c + 1] + bias[c + 1], scale[c + 1]);
                    dst[c + 2] = SynetFusedLayerForward0(src[c + 2] + bias[c + 2], scale[c + 2]);
                    dst[c + 3] = SynetFusedLayerForward0(src[c + 3] + bias[c + 3], scale[c + 3]);
                }
                for (; c < channels; ++c)
                    dst[c] = SynetFusedLayerForward0(src[c] + bias[c], scale[c]);
                src += channels;
                dst += channels;
            }
        }

        template<int N> void SynetFusedLayerForward0NchwXc(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            for (size_t c = 0; c < channels; c += N)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t i = 0; i < N; ++i)
                        dst[i] = SynetFusedLayerForward0(src[i] + bias[i], scale[i]);
                    src += N;
                    dst += N;
                }
                bias += N;
                scale += N;
            }
        }

        void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward0Nchw(src, bias, scale, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward0Nhwc(src, bias, scale, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward0NchwXc<4>(src, bias, scale, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward0NchwXc<8>(src, bias, scale, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetFusedLayerForward0NchwXc<16>(src, bias, scale, channels, spatial, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        void SynetFusedLayerForward1Nchw(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; ++c)
            {
                float _bias0 = bias0[c];
                float _scale1 = scale1[c];
                float _bias1 = bias1[c];
                size_t s = 0;
                for (; s < aligned; s += 4)
                {
                    dst[s + 0] = SynetFusedLayerForward1(src[s + 0] + _bias0, _scale1, _bias1);
                    dst[s + 1] = SynetFusedLayerForward1(src[s + 1] + _bias0, _scale1, _bias1);
                    dst[s + 2] = SynetFusedLayerForward1(src[s + 2] + _bias0, _scale1, _bias1);
                    dst[s + 3] = SynetFusedLayerForward1(src[s + 3] + _bias0, _scale1, _bias1);
                }
                for (; s < spatial; ++s)
                    dst[s] = SynetFusedLayerForward1(src[s] + _bias0, _scale1, _bias1);
                src += spatial;
                dst += spatial;
            }
        }

        void SynetFusedLayerForward1Nhwc(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(channels, 4);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += 4)
                {
                    dst[c + 0] = SynetFusedLayerForward1(src[c + 0] + bias0[c + 0], scale1[c + 0], bias1[c + 0]);
                    dst[c + 1] = SynetFusedLayerForward1(src[c + 1] + bias0[c + 1], scale1[c + 1], bias1[c + 1]);
                    dst[c + 2] = SynetFusedLayerForward1(src[c + 2] + bias0[c + 2], scale1[c + 2], bias1[c + 2]);
                    dst[c + 3] = SynetFusedLayerForward1(src[c + 3] + bias0[c + 3], scale1[c + 3], bias1[c + 3]);
                }
                for (; c < channels; ++c)
                    dst[c] = SynetFusedLayerForward1(src[c] + bias0[c], scale1[c], bias1[c]);
                src += channels;
                dst += channels;
            }
        }

        template<int N> void SynetFusedLayerForward1NchwXc(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            for (size_t c = 0; c < channels; c += N)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t i = 0; i < N; ++i)
                        dst[i] = SynetFusedLayerForward1(src[i] + bias0[i], scale1[i], bias1[i]);
                    src += N;
                    dst += N;
                }
                bias0 += N;
                scale1 += N;
                bias1 += N;
            }
        }

        void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward1Nchw(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward1Nhwc(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward1NchwXc<4>(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward1NchwXc<8>(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetFusedLayerForward1NchwXc<16>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        void SynetFusedLayerForward2Nchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            float _slope = slope[0];
            size_t aligned = Simd::AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; ++c)
            {
                float _scale = scale[c];
                float _bias = bias[c];
                size_t s = 0;
                for (; s < aligned; s += 4)
                {
                    dst[s + 0] = SynetFusedLayerForward2(src[s + 0], _scale, _bias, _slope);
                    dst[s + 1] = SynetFusedLayerForward2(src[s + 1], _scale, _bias, _slope);
                    dst[s + 2] = SynetFusedLayerForward2(src[s + 2], _scale, _bias, _slope);
                    dst[s + 3] = SynetFusedLayerForward2(src[s + 3], _scale, _bias, _slope);
                }
                for (; s < spatial; ++s)
                    dst[s] = SynetFusedLayerForward2(src[s], _scale, _bias, _slope);
                src += spatial;
                dst += spatial;
            }
        }

        void SynetFusedLayerForward2Nhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            float _slope = slope[0];
            size_t aligned = Simd::AlignLo(channels, 4);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += 4)
                {
                    dst[c + 0] = SynetFusedLayerForward2(src[c + 0], scale[c + 0], bias[c + 0], _slope);
                    dst[c + 1] = SynetFusedLayerForward2(src[c + 1], scale[c + 1], bias[c + 1], _slope);
                    dst[c + 2] = SynetFusedLayerForward2(src[c + 2], scale[c + 2], bias[c + 2], _slope);
                    dst[c + 3] = SynetFusedLayerForward2(src[c + 3], scale[c + 3], bias[c + 3], _slope);
                }
                for (; c < channels; ++c)
                    dst[c] = SynetFusedLayerForward2(src[c], scale[c], bias[c], _slope);
                src += channels;
                dst += channels;
            }
        }

        template<int N> void SynetFusedLayerForward2NchwXc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst)
        {
            float _slope = slope[0];
            for (size_t c = 0; c < channels; c += N)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t i = 0; i < N; ++i)
                        dst[i] = SynetFusedLayerForward2(src[i], scale[i], bias[i], _slope);
                    src += N;
                    dst += N;
                }
                scale += N;
                bias += N;
            }
        }

        void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward2Nchw(src, scale, bias, channels, spatial, slope, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward2Nhwc(src, scale, bias, channels, spatial, slope, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward2NchwXc<4>(src, scale, bias, channels, spatial, slope, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward2NchwXc<8>(src, scale, bias, channels, spatial, slope, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetFusedLayerForward2NchwXc<16>(src, scale, bias, channels, spatial, slope, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        void SynetFusedLayerForward3Nchw(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; ++c)
            {
                float _bias = bias[c];
                float _scale = scale[c];
                size_t s = 0;
                for (; s < aligned; s += 4)
                {
                    dst[s + 0] = SynetFusedLayerForward3(src[s + 0] + _bias, _scale);
                    dst[s + 1] = SynetFusedLayerForward3(src[s + 1] + _bias, _scale);
                    dst[s + 2] = SynetFusedLayerForward3(src[s + 2] + _bias, _scale);
                    dst[s + 3] = SynetFusedLayerForward3(src[s + 3] + _bias, _scale);
                }
                for (; s < spatial; ++s)
                    dst[s] = SynetFusedLayerForward3(src[s] + _bias, _scale);
                src += spatial;
                dst += spatial;
            }
        }

        void SynetFusedLayerForward3Nhwc(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
             size_t aligned = Simd::AlignLo(channels, 4);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += 4)
                {
                    dst[c + 0] = SynetFusedLayerForward3(src[c + 0] + bias[c + 0], scale[c + 0]);
                    dst[c + 1] = SynetFusedLayerForward3(src[c + 1] + bias[c + 1], scale[c + 1]);
                    dst[c + 2] = SynetFusedLayerForward3(src[c + 2] + bias[c + 2], scale[c + 2]);
                    dst[c + 3] = SynetFusedLayerForward3(src[c + 3] + bias[c + 3], scale[c + 3]);
                }
                for (; c < channels; ++c)
                    dst[c] = SynetFusedLayerForward3(src[c] + bias[c], scale[c]);
                src += channels;
                dst += channels;
            }
        }

        template<int N> void SynetFusedLayerForward3NchwXc(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst)
        {
            for (size_t c = 0; c < channels; c += N)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t i = 0; i < N; ++i)
                        dst[i] = SynetFusedLayerForward3(src[i] + bias[i], scale[i]);
                    src += N;
                    dst += N;
                }
                bias += N;
                scale += N;
            }
        }

        void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward3Nchw(src, bias, scale, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward3Nhwc(src, bias, scale, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward3NchwXc<4>(src, bias, scale, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward3NchwXc<8>(src, bias, scale, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetFusedLayerForward3NchwXc<16>(src, bias, scale, channels, spatial, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        void SynetFusedLayerForward4Nchw(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            float _scale1 = scale1[0];
            float _bias1 = bias1[0];
            float * dst1 = dst0 + channels * spatial;
            size_t aligned = Simd::AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; ++c)
            {
                float _bias0 = bias0[c];
                size_t s = 0;
                for (; s < aligned; s += 4)
                {
                    SynetFusedLayerForward4(src[s + 0], _bias0, _scale1, _bias1, dst0 + s + 0, dst1 + s + 0);
                    SynetFusedLayerForward4(src[s + 1], _bias0, _scale1, _bias1, dst0 + s + 1, dst1 + s + 1);
                    SynetFusedLayerForward4(src[s + 2], _bias0, _scale1, _bias1, dst0 + s + 2, dst1 + s + 2);
                    SynetFusedLayerForward4(src[s + 3], _bias0, _scale1, _bias1, dst0 + s + 3, dst1 + s + 3);
                }
                for (; s < spatial; ++s)
                    SynetFusedLayerForward4(src[s], _bias0, _scale1, _bias1, dst0 + s, dst1 + s);
                src += spatial;
                dst0 += spatial;
                dst1 += spatial;
            }
        }

        void SynetFusedLayerForward4Nhwc(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            float _scale1 = scale1[0];
            float _bias1 = bias1[0];
            float * dst1 = dst0 + channels;
            size_t aligned = Simd::AlignLo(channels, 4);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += 4)
                {
                    SynetFusedLayerForward4(src[c + 0], bias0[c + 0], _scale1, _bias1, dst0 + c + 0, dst1 + c + 0);
                    SynetFusedLayerForward4(src[c + 1], bias0[c + 1], _scale1, _bias1, dst0 + c + 1, dst1 + c + 1);
                    SynetFusedLayerForward4(src[c + 2], bias0[c + 2], _scale1, _bias1, dst0 + c + 2, dst1 + c + 2);
                    SynetFusedLayerForward4(src[c + 3], bias0[c + 3], _scale1, _bias1, dst0 + c + 3, dst1 + c + 3);
                }
                for (; c < channels; ++c)
                    SynetFusedLayerForward4(src[c], bias0[c], _scale1, _bias1, dst0 + c, dst1 + c);
                src += channels;
                dst0 += 2 * channels;
                dst1 += 2 * channels;
            }
        }

        template<int N> void SynetFusedLayerForward4NchwXcA(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            assert(Aligned(channels, N));
            float _scale1 = scale1[0];
            float _bias1 = bias1[0];
            float * dst1 = dst0 + channels * spatial;
            for (size_t c = 0; c < channels; c += N)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t i = 0; i < N; ++i)
                        SynetFusedLayerForward4(src[i], bias0[i], _scale1, _bias1, dst0 + i, dst1 + i);
                    src += N;
                    dst0 += N;
                    dst1 += N;
                }
                bias0 += N;
            }
        }

        template<int N> void SynetFusedLayerForward4NchwXcU(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst0)
        {
            assert(0);
        }

        template<int N> void SynetFusedLayerForward4NchwXc(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(channels, N))
                SynetFusedLayerForward4NchwXcA<N>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                SynetFusedLayerForward4NchwXcU<N>(src, bias0, scale1, bias1, channels, spatial, dst);
        }

        void SynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward4Nchw(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward4Nhwc(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward4NchwXc<4>(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward4NchwXc<8>(src, bias0, scale1, bias1, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetFusedLayerForward4NchwXc<16>(src, bias0, scale1, bias1, channels, spatial, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        void SynetFusedLayerForward8Nchw(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; ++c)
            {
                float _src2 = src2[c];
                size_t s = 0;
                for (; s < aligned; s += 4)
                {
                    dst[s + 0] = SynetFusedLayerForward8(src0[s + 0], src1[s + 0], _src2);
                    dst[s + 1] = SynetFusedLayerForward8(src0[s + 1], src1[s + 1], _src2);
                    dst[s + 2] = SynetFusedLayerForward8(src0[s + 2], src1[s + 2], _src2);
                    dst[s + 3] = SynetFusedLayerForward8(src0[s + 3], src1[s + 3], _src2);
                }
                for (; s < spatial; ++s)
                    dst[s] = SynetFusedLayerForward8(src0[s], src1[s], _src2);
                src0 += spatial;
                src1 += spatial;
                dst += spatial;
            }
        }

        void SynetFusedLayerForward8Nhwc(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            size_t aligned = Simd::AlignLo(channels, 4);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += 4)
                {
                    dst[c + 0] = SynetFusedLayerForward8(src0[c + 0], src1[c + 0], src2[c + 0]);
                    dst[c + 1] = SynetFusedLayerForward8(src0[c + 1], src1[c + 1], src2[c + 1]);
                    dst[c + 2] = SynetFusedLayerForward8(src0[c + 2], src1[c + 2], src2[c + 2]);
                    dst[c + 3] = SynetFusedLayerForward8(src0[c + 3], src1[c + 3], src2[c + 3]);
                }
                for (; c < channels; ++c)
                    dst[c] = SynetFusedLayerForward8(src0[c], src1[c], src2[c]);
                src0 += channels;
                src1 += channels;
                dst += channels;
            }
        }

        template<int N> void SynetFusedLayerForward8NchwXc(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst)
        {
            for (size_t c = 0; c < channels; c += N)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    for (size_t i = 0; i < N; ++i)
                        dst[i] = SynetFusedLayerForward8(src0[i], src1[i], src2[i]);
                    src0 += N;
                    src1 += N;
                    dst += N;
                }
                src2 += N;
            }
        }

        void SynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetFusedLayerForward8Nchw(src0, src1, src2, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetFusedLayerForward8Nhwc(src0, src1, src2, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward8NchwXc<4>(src0, src1, src2, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward8NchwXc<8>(src0, src1, src2, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw16c)
                SynetFusedLayerForward8NchwXc<16>(src0, src1, src2, channels, spatial, dst);
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        void SynetFusedLayerForward9Nchw(const float * src0, const float * src1, const float * scale0, const float * bias0, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            const float * scale1 = scale0 + channels0;
            const float * bias1 = bias0 + channels0;
            size_t aligned = Simd::AlignLo(spatial, 4);
            if (dst1)
            {
                for (size_t c = 0; c < channels0; ++c)
                {
                    float _scale0 = scale0[c];
                    float _bias0 = bias0[c];
                    size_t s = 0;
                    for (; s < aligned; ++s)
                    {
                        dst0[s + 0] = SynetFusedLayerForward9(src0[s + 0], _scale0, _bias0), dst1[s + 0] = src0[s + 0];
                        dst0[s + 1] = SynetFusedLayerForward9(src0[s + 1], _scale0, _bias0), dst1[s + 1] = src0[s + 1];
                        dst0[s + 2] = SynetFusedLayerForward9(src0[s + 2], _scale0, _bias0), dst1[s + 2] = src0[s + 2];
                        dst0[s + 3] = SynetFusedLayerForward9(src0[s + 3], _scale0, _bias0), dst1[s + 3] = src0[s + 3];
                    }
                    for (; s < spatial; ++s)
                        dst0[s] = SynetFusedLayerForward9(src0[s], _scale0, _bias0), dst1[s] = src0[s];
                    src0 += spatial;
                    dst0 += spatial;
                    dst1 += spatial;
                }
                for (size_t c = 0; c < channels1; ++c)
                {
                    float _scale1 = scale1[c];
                    float _bias1 = bias1[c];
                    size_t s = 0;
                    for (; s < aligned; ++s)
                    {
                        dst0[s + 0] = SynetFusedLayerForward9(src1[s + 0], _scale1, _bias1), dst1[s + 0] = src1[s + 0];
                        dst0[s + 1] = SynetFusedLayerForward9(src1[s + 1], _scale1, _bias1), dst1[s + 1] = src1[s + 1];
                        dst0[s + 2] = SynetFusedLayerForward9(src1[s + 2], _scale1, _bias1), dst1[s + 2] = src1[s + 2];
                        dst0[s + 3] = SynetFusedLayerForward9(src1[s + 3], _scale1, _bias1), dst1[s + 3] = src1[s + 3];
                    }
                    for (; s < spatial; ++s)
                        dst0[s] = SynetFusedLayerForward9(src1[s], _scale1, _bias1), dst1[s] = src1[s];
                    src1 += spatial;
                    dst0 += spatial;
                    dst1 += spatial;
                }
            }
            else
            {
                for (size_t c = 0; c < channels0; ++c)
                {
                    float _scale0 = scale0[c];
                    float _bias0 = bias0[c];
                    size_t s = 0;
                    for (; s < aligned; ++s)
                    {
                        dst0[s + 0] = SynetFusedLayerForward9(src0[s + 0], _scale0, _bias0);
                        dst0[s + 1] = SynetFusedLayerForward9(src0[s + 1], _scale0, _bias0);
                        dst0[s + 2] = SynetFusedLayerForward9(src0[s + 2], _scale0, _bias0);
                        dst0[s + 3] = SynetFusedLayerForward9(src0[s + 3], _scale0, _bias0);
                    }
                    for (; s < spatial; ++s)
                        dst0[s] = SynetFusedLayerForward9(src0[s], _scale0, _bias0);
                    src0 += spatial;
                    dst0 += spatial;
                }
                for (size_t c = 0; c < channels1; ++c)
                {
                    float _scale1 = scale1[c];
                    float _bias1 = bias1[c];
                    size_t s = 0;
                    for (; s < aligned; ++s)
                    {
                        dst0[s + 0] = SynetFusedLayerForward9(src1[s + 0], _scale1, _bias1);
                        dst0[s + 1] = SynetFusedLayerForward9(src1[s + 1], _scale1, _bias1);
                        dst0[s + 2] = SynetFusedLayerForward9(src1[s + 2], _scale1, _bias1);
                        dst0[s + 3] = SynetFusedLayerForward9(src1[s + 3], _scale1, _bias1);
                    }
                    for (; s < spatial; ++s)
                        dst0[s] = SynetFusedLayerForward9(src1[s], _scale1, _bias1);
                    src1 += spatial;
                    dst0 += spatial;
                }
            }
        }

        void SynetFusedLayerForward9Nhwc(const float * src0, const float * src1, const float * scale0, const float * bias0, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            const float * scale1 = scale0 + channels0;
            const float * bias1 = bias0 + channels0;
            size_t aligned0 = Simd::AlignLo(channels0, 4);
            size_t aligned1 = Simd::AlignLo(channels1, 4);
            if (dst1)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c;
                    for (c = 0; c < channels0; c += 4)
                    {
                        dst0[c + 0] = SynetFusedLayerForward9(src0[c + 0], scale0[c + 0], bias0[c + 0]), dst1[c + 0] = src0[c + 0];
                        dst0[c + 1] = SynetFusedLayerForward9(src0[c + 1], scale0[c + 1], bias0[c + 1]), dst1[c + 1] = src0[c + 1];
                        dst0[c + 2] = SynetFusedLayerForward9(src0[c + 2], scale0[c + 2], bias0[c + 2]), dst1[c + 2] = src0[c + 2];
                        dst0[c + 3] = SynetFusedLayerForward9(src0[c + 3], scale0[c + 3], bias0[c + 3]), dst1[c + 3] = src0[c + 3];
                    }
                    for (; c < channels0; ++c)
                        dst0[c] = SynetFusedLayerForward9(src0[c], scale0[c], bias0[c]), dst1[c] = src0[c];
                    src0 += channels0;
                    dst0 += channels0;
                    dst1 += channels0;
                    for (c = 0; c < channels1; c += 4)
                    {
                        dst0[c + 0] = SynetFusedLayerForward9(src1[c + 0], scale1[c + 0], bias1[c + 0]), dst1[c + 0] = src1[c + 0];
                        dst0[c + 1] = SynetFusedLayerForward9(src1[c + 1], scale1[c + 1], bias1[c + 1]), dst1[c + 1] = src1[c + 1];
                        dst0[c + 2] = SynetFusedLayerForward9(src1[c + 2], scale1[c + 2], bias1[c + 2]), dst1[c + 2] = src1[c + 2];
                        dst0[c + 3] = SynetFusedLayerForward9(src1[c + 3], scale1[c + 3], bias1[c + 3]), dst1[c + 3] = src1[c + 3];
                    }
                    for (; c < channels1; ++c)
                        dst0[c] = SynetFusedLayerForward9(src1[c], scale1[c], bias1[c]), dst1[c] = src1[c];
                    src1 += channels1;
                    dst0 += channels1;
                    dst1 += channels1;
                }
            }
            else
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c;
                    for (c = 0; c < channels0; c += 4)
                    {
                        dst0[c + 0] = SynetFusedLayerForward9(src0[c + 0], scale0[c + 0], bias0[c + 0]);
                        dst0[c + 1] = SynetFusedLayerForward9(src0[c + 1], scale0[c + 1], bias0[c + 1]);
                        dst0[c + 2] = SynetFusedLayerForward9(src0[c + 2], scale0[c + 2], bias0[c + 2]);
                        dst0[c + 3] = SynetFusedLayerForward9(src0[c + 3], scale0[c + 3], bias0[c + 3]);
                    }
                    for (; c < channels0; ++c)
                        dst0[c] = SynetFusedLayerForward9(src0[c], scale0[c], bias0[c]);
                    src0 += channels0;
                    dst0 += channels0;
                    for (c = 0; c < channels1; c += 4)
                    {
                        dst0[c + 0] = SynetFusedLayerForward9(src1[c + 0], scale1[c + 0], bias1[c + 0]);
                        dst0[c + 1] = SynetFusedLayerForward9(src1[c + 1], scale1[c + 1], bias1[c + 1]);
                        dst0[c + 2] = SynetFusedLayerForward9(src1[c + 2], scale1[c + 2], bias1[c + 2]);
                        dst0[c + 3] = SynetFusedLayerForward9(src1[c + 3], scale1[c + 3], bias1[c + 3]);
                    }
                    for (; c < channels1; ++c)
                        dst0[c] = SynetFusedLayerForward9(src1[c], scale1[c], bias1[c]);
                    src1 += channels1;
                    dst0 += channels1;
                }
            }
        }

        template<int N> void SynetFusedLayerForward9NchwXcA(const float * src0, const float * src1, const float * scale0, const float * bias0, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            assert(Aligned(channels0, N));
            const float * scale1 = scale0 + channels0;
            const float * bias1 = bias0 + channels0;
            if (dst1)
            {
                for (size_t c = 0; c < channels0; c += N)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t i = 0; i < N; ++i)
                            dst0[i] = SynetFusedLayerForward9(src0[i], scale0[i], bias0[i]), dst1[i] = src0[i];
                        src0 += N;
                        dst0 += N;
                        dst1 += N;
                    }
                    scale0 += N;
                    bias0 += N;
                }
                for (size_t c = 0; c < channels1; c += N)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t i = 0; i < N; ++i)
                            dst0[i] = SynetFusedLayerForward9(src1[i], scale1[i], bias1[i]), dst1[i] = src1[i];
                        src1 += N;
                        dst0 += N;
                        dst1 += N;
                    }
                    scale1 += N;
                    bias1 += N;
                }
            }
            else
            {
                for (size_t c = 0; c < channels0; c += N)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t i = 0; i < N; ++i)
                            dst0[i] = SynetFusedLayerForward9(src0[i], scale0[i], bias0[i]);
                        src0 += N;
                        dst0 += N;
                    }
                    scale0 += N;
                    bias0 += N;
                }
                for (size_t c = 0; c < channels1; c += N)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        for (size_t i = 0; i < N; ++i)
                            dst0[i] = SynetFusedLayerForward9(src1[i], scale1[i], bias1[i]);
                        src1 += N;
                        dst0 += N;
                    }
                    scale1 += N;
                    bias1 += N;
                }
            }
        }

        template<int N> void SynetFusedLayerForward9NchwXcU(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            assert(0);
        }

        template<int N> void SynetFusedLayerForward9NchwXc(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1)
        {
            if (Aligned(channels0, N))
                SynetFusedLayerForward9NchwXcA<N>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else
                SynetFusedLayerForward9NchwXcU<N>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
        }

        void SynetFusedLayerForward9(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels0 + channels1, spatial, format))
                SynetFusedLayerForward9Nchw(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else if (Base::NhwcCompatible(channels0 + channels1, spatial, format))
                SynetFusedLayerForward9Nhwc(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else if (format == SimdTensorFormatNchw4c)
                SynetFusedLayerForward9NchwXc<4>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else if (format == SimdTensorFormatNchw8c)
                SynetFusedLayerForward9NchwXc<8>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else if (format == SimdTensorFormatNchw16c)
                SynetFusedLayerForward9NchwXc<16>(src0, src1, scale, bias, channels0, channels1, spatial, dst0, dst1);
            else
                assert(0);
        }
    }
#endif
}
