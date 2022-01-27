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
#ifndef __SimdImageSaveJpeg_h__
#define __SimdImageSaveJpeg_h__

#include "Simd/SimdImageSave.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdSet.h"

#define SIMD_JPEG_CALC_BITS_TABLE

namespace Simd
{
    namespace Base
    {
        struct BitBuf
        {
            static const uint32_t capacity = 2048;
            uint32_t size;
            uint16_t data[capacity][2];

            SIMD_INLINE BitBuf()
                : size(0) 
            {
            }

            SIMD_INLINE void Push(const uint16_t* bits)
            {
                ((uint32_t*)data)[size++] = ((uint32_t*)bits)[0];
            }

            SIMD_INLINE bool Full(uint32_t tail = capacity / 2) const
            {
                assert(size <= capacity);
                return size + tail >= capacity;
            }

            SIMD_INLINE uint32_t Capacity() const 
            {
                return capacity;
            }

            SIMD_INLINE void Clear()
            {
                size = 0;
            }
        }; 

        extern const uint8_t JpegZigZagD[64];
        extern const uint8_t JpegZigZagT[64];

        extern const uint16_t HuffmanYdc[256][2];
        extern const uint16_t HuffmanUVdc[256][2];
        extern const uint16_t HuffmanYac[256][2];
        extern const uint16_t HuffmanUVac[256][2];

#if defined(SIMD_JPEG_CALC_BITS_TABLE)
        const int JpegCalcBitsRange = 2048;
        extern uint16_t JpegCalcBitsTable[JpegCalcBitsRange * 2][2];
        SIMD_INLINE void JpegCalcBits(int val, uint16_t bits[2])
        {
            assert(val >= -JpegCalcBitsRange && val < JpegCalcBitsRange);
            ((uint32_t*)bits)[0] = ((uint32_t*)JpegCalcBitsTable)[val + JpegCalcBitsRange];
        }
#else
        SIMD_INLINE void JpegCalcBits(int val, uint16_t bits[2])
        {
            int tmp = val < 0 ? -val : val;
            val = val < 0 ? val - 1 : val;
            bits[1] = 1;
            while (tmp >>= 1)
                ++bits[1];
            bits[0] = val & ((1 << bits[1]) - 1);
        }
#endif

        SIMD_INLINE void RgbToYuv(const uint8_t* r, const uint8_t* g, const uint8_t* b, int stride, int height, int width, float* y, float* u, float* v, int size)
        {
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += 1)
                {
                    int offs = (col < width ? col : width - 1);
                    float _r = r[offs], _g = g[offs], _b = b[offs];
                    y[col] = +0.29900f * _r + 0.58700f * _g + 0.11400f * _b - 128.000f;
                    u[col] = -0.16874f * _r - 0.33126f * _g + 0.50000f * _b;
                    v[col] = +0.50000f * _r - 0.41869f * _g - 0.08131f * _b;
                }
                if (++row < height)
                    r += stride, g += stride, b += stride;
                y += size, u += size, v += size;
            }
        }

        SIMD_INLINE void GrayToY(const uint8_t* g, int stride, int height, int width, float* y, int size)
        {
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += 1)
                {
                    int offs = (col < width ? col : width - 1);
                    y[col] = g[offs] - 128.000f;
                }
                if (++row < height)
                    g += stride;
                y += size;
            }
        }

        SIMD_INLINE int UvSize(int ySize)
        {
            return (ySize + 1) >> 1;
        }

        SIMD_INLINE void Nv12ToUv(const uint8_t* uvSrc, int uvStride, int height, int width, float* u, float* v)
        {
            for (int row = 0; row < 8;)
            {
                for (int col = 0; col < 8; col += 1)
                {
                    int offs = (col < width ? col : width - 1) << 1;
                    u[col] = uvSrc[offs + 0] - 128.000f;
                    v[col] = uvSrc[offs + 1] - 128.000f;
                }
                if (++row < height)
                    uvSrc += uvStride;
                u += 8, v += 8;
            }
        }

        SIMD_INLINE void JpegProcessDuGrayUv(BitBuf & bitBuf)
        {
            bitBuf.Push(Base::HuffmanUVdc[0]);
            bitBuf.Push(Base::HuffmanUVac[0]);
            bitBuf.Push(Base::HuffmanUVdc[0]);
            bitBuf.Push(Base::HuffmanUVac[0]);
        }

        SIMD_INLINE void WriteBits(OutputMemoryStream & stream, const uint16_t bits[2])
        {
            stream.BitCount() += bits[1];
#if defined(SIMD_X64_ENABLE) || defined(SIMD_ARM64_ENABLE)
            stream.BitBuffer() |= uint64_t(bits[0]) << (64 - stream.BitCount());
            while (stream.BitCount() >= 8)
            {
                uint8_t byte = stream.BitBuffer() >> 56;
                stream.Write8u(byte);
                if (byte == 255)
                    stream.Write8u(0);
                stream.BitBuffer() <<= 8;
                stream.BitCount() -= 8;
            }
#else
            stream.BitBuffer() |= uint32_t(bits[0]) << (32 - stream.BitCount());
            while (stream.BitCount() >= 8)
            {
                uint8_t byte = stream.BitBuffer() >> 24;
                stream.Write8u(byte);
                if (byte == 255)
                    stream.Write8u(0);
                stream.BitBuffer() <<= 8;
                stream.BitCount() -= 8;
            }
#endif
        }

        SIMD_INLINE void WriteBits(OutputMemoryStream& stream, const uint16_t bits[][2], size_t size)
        {
            size_t pos = stream.Pos();
            stream.Reserve(pos + size * 2);
            uint8_t* data = stream.Data();
            size_t & bitCount = stream.BitCount();
            size_t i = 0;
#if defined(SIMD_X64_ENABLE) || defined(SIMD_ARM64_ENABLE)
            uint64_t &bitBuffer = stream.BitBuffer();
            for (size_t size3 = AlignLoAny(size, 3); i < size3; i += 3, bits += 3)
            {
                bitCount += bits[0][1];
                bitBuffer |= uint64_t(bits[0][0]) << (64 - bitCount);
                bitCount += bits[1][1];
                bitBuffer |= uint64_t(bits[1][0]) << (64 - bitCount);
                bitCount += bits[2][1];
                bitBuffer |= uint64_t(bits[2][0]) << (64 - bitCount);
                assert(bitCount <= 64);
                while (bitCount >= 16)
                {
                    uint8_t byte = uint8_t(bitBuffer >> 56);
                    data[pos++] = byte;
                    if (byte == 255)
                        data[pos++] = 0;
                    byte = uint8_t(bitBuffer >> 48);
                    data[pos++] = byte;
                    if (byte == 255)
                        data[pos++] = 0;
                    bitBuffer <<= 16;
                    bitCount -= 16;
                }
            }
            if(bitCount >= 8)
            {
                assert(bitCount < 16);
                uint8_t byte = uint8_t(bitBuffer >> 56);
                data[pos++] = byte;
                if (byte == 255)
                    data[pos++] = 0;
                bitBuffer <<= 8;
                bitCount -= 8;
            }
            for (; i < size; ++i, ++bits)
            {
                bitCount += bits[0][1];
                bitBuffer |= uint64_t(bits[0][0]) << (64 - bitCount);
                while (bitCount >= 8)
                {
                    uint8_t byte = uint8_t(bitBuffer >> 56);
                    data[pos++] = byte;
                    if (byte == 255)
                        data[pos++] = 0;
                    bitBuffer <<= 8;
                    bitCount -= 8;
                }
            }
#else
            uint32_t &bitBuffer = stream.BitBuffer();
            for (; i < size; ++i, ++bits)
            {
                bitCount += bits[0][1];
                bitBuffer |= uint32_t(bits[0][0]) << (32 - bitCount);
                while (bitCount >= 8)
                {
                    uint8_t byte = uint8_t(bitBuffer >> 24);
                    data[pos++] = byte;
                    if (byte == 255)
                        data[pos++] = 0;
                    bitBuffer <<= 8;
                    bitCount -= 8;
                }
            }
#endif
            stream.Seek(pos);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
    }
#endif// SIMD_SSE41_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        const __m256i K8_SHUFFLE_UV_U = SIMD_MM256_SETR_EPI8(
            0x0, -1, -1, -1, 0x2, -1, -1, -1, 0x4, -1, -1, -1, 0x6, -1, -1, -1,
            0x8, -1, -1, -1, 0xA, -1, -1, -1, 0xC, -1, -1, -1, 0xE, -1, -1, -1);
        const __m256i K8_SHUFFLE_UV_V = SIMD_MM256_SETR_EPI8(
            0x1, -1, -1, -1, 0x3, -1, -1, -1, 0x5, -1, -1, -1, 0x7, -1, -1, -1,
            0x9, -1, -1, -1, 0xB, -1, -1, -1, 0xD, -1, -1, -1, 0xF, -1, -1, -1);

        SIMD_INLINE void Nv12ToUv(const uint8_t* uvSrc, int uvStride, int height, float* u, float* v)
        {
            __m256 k = _mm256_set1_ps(-128.000f);
            for (int row = 0; row < 8;)
            {
                __m256i _uv = Set(_mm_loadu_si128((__m128i*)uvSrc));
                _mm256_storeu_ps(u, _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_shuffle_epi8(_uv, K8_SHUFFLE_UV_U)), k));
                _mm256_storeu_ps(v, _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_shuffle_epi8(_uv, K8_SHUFFLE_UV_V)), k));
                if (++row < height)
                    uvSrc += uvStride;
                u += 8, v += 8;
            }
        }

        extern const uint32_t JpegZigZagTi32[64];

        SIMD_INLINE void JpegDctV(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
            static const __m256 _0_707106781 = _mm256_set1_ps(0.707106781f);
            static const __m256 _0_382683433 = _mm256_set1_ps(0.382683433f);
            static const __m256 _0_541196100 = _mm256_set1_ps(0.541196100f);
            static const __m256 _1_306562965 = _mm256_set1_ps(1.306562965f);

            __m256 d0 = _mm256_loadu_ps(src + 0 * srcStride);
            __m256 d1 = _mm256_loadu_ps(src + 1 * srcStride);
            __m256 d2 = _mm256_loadu_ps(src + 2 * srcStride);
            __m256 d3 = _mm256_loadu_ps(src + 3 * srcStride);
            __m256 d4 = _mm256_loadu_ps(src + 4 * srcStride);
            __m256 d5 = _mm256_loadu_ps(src + 5 * srcStride);
            __m256 d6 = _mm256_loadu_ps(src + 6 * srcStride);
            __m256 d7 = _mm256_loadu_ps(src + 7 * srcStride);

            __m256 tmp0 = _mm256_add_ps(d0, d7);
            __m256 tmp7 = _mm256_sub_ps(d0, d7);
            __m256 tmp1 = _mm256_add_ps(d1, d6);
            __m256 tmp6 = _mm256_sub_ps(d1, d6);
            __m256 tmp2 = _mm256_add_ps(d2, d5);
            __m256 tmp5 = _mm256_sub_ps(d2, d5);
            __m256 tmp3 = _mm256_add_ps(d3, d4);
            __m256 tmp4 = _mm256_sub_ps(d3, d4);

            __m256 tmp10 = _mm256_add_ps(tmp0, tmp3);
            __m256 tmp13 = _mm256_sub_ps(tmp0, tmp3);
            __m256 tmp11 = _mm256_add_ps(tmp1, tmp2);
            __m256 tmp12 = _mm256_sub_ps(tmp1, tmp2);

            d0 = _mm256_add_ps(tmp10, tmp11);
            d4 = _mm256_sub_ps(tmp10, tmp11);

            __m256 z1 = _mm256_mul_ps(_mm256_add_ps(tmp12, tmp13), _0_707106781);
            d2 = _mm256_add_ps(tmp13, z1);
            d6 = _mm256_sub_ps(tmp13, z1);

            tmp10 = _mm256_add_ps(tmp4, tmp5);
            tmp11 = _mm256_add_ps(tmp5, tmp6);
            tmp12 = _mm256_add_ps(tmp6, tmp7);

            __m256 z5 = _mm256_mul_ps(_mm256_sub_ps(tmp10, tmp12), _0_382683433);
            __m256 z2 = _mm256_add_ps(_mm256_mul_ps(tmp10, _0_541196100), z5);
            __m256 z4 = _mm256_add_ps(_mm256_mul_ps(tmp12, _1_306562965), z5);
            __m256 z3 = _mm256_mul_ps(tmp11, _0_707106781);

            __m256 z11 = _mm256_add_ps(tmp7, z3);
            __m256 z13 = _mm256_sub_ps(tmp7, z3);

            _mm256_storeu_ps(dst + 0 * dstStride, d0);
            _mm256_storeu_ps(dst + 1 * dstStride, _mm256_add_ps(z11, z4));
            _mm256_storeu_ps(dst + 2 * dstStride, d2);
            _mm256_storeu_ps(dst + 3 * dstStride, _mm256_sub_ps(z13, z2));
            _mm256_storeu_ps(dst + 4 * dstStride, d4);
            _mm256_storeu_ps(dst + 5 * dstStride, _mm256_add_ps(z13, z2));
            _mm256_storeu_ps(dst + 6 * dstStride, d6);
            _mm256_storeu_ps(dst + 7 * dstStride, _mm256_sub_ps(z11, z4));
        }

        SIMD_INLINE void JpegDct(const float* src, size_t stride, const float* fdt, int* dst)
        {
            static const __m256 _0_707106781 = _mm256_set1_ps(0.707106781f);
            static const __m256 _0_382683433 = _mm256_set1_ps(0.382683433f);
            static const __m256 _0_541196100 = _mm256_set1_ps(0.541196100f);
            static const __m256 _1_306562965 = _mm256_set1_ps(1.306562965f);

            __m256 d0 = _mm256_loadu_ps(src + 0 * stride);
            __m256 d1 = _mm256_loadu_ps(src + 1 * stride);
            __m256 d2 = _mm256_loadu_ps(src + 2 * stride);
            __m256 d3 = _mm256_loadu_ps(src + 3 * stride);
            __m256 d4 = _mm256_loadu_ps(src + 4 * stride);
            __m256 d5 = _mm256_loadu_ps(src + 5 * stride);
            __m256 d6 = _mm256_loadu_ps(src + 6 * stride);
            __m256 d7 = _mm256_loadu_ps(src + 7 * stride);

            __m256 tmp0 = _mm256_add_ps(d0, d7);
            __m256 tmp7 = _mm256_sub_ps(d0, d7);
            __m256 tmp1 = _mm256_add_ps(d1, d6);
            __m256 tmp6 = _mm256_sub_ps(d1, d6);
            __m256 tmp2 = _mm256_add_ps(d2, d5);
            __m256 tmp5 = _mm256_sub_ps(d2, d5);
            __m256 tmp3 = _mm256_add_ps(d3, d4);
            __m256 tmp4 = _mm256_sub_ps(d3, d4);

            __m256 tmp10 = _mm256_add_ps(tmp0, tmp3);
            __m256 tmp13 = _mm256_sub_ps(tmp0, tmp3);
            __m256 tmp11 = _mm256_add_ps(tmp1, tmp2);
            __m256 tmp12 = _mm256_sub_ps(tmp1, tmp2);

            d0 = _mm256_add_ps(tmp10, tmp11);
            d4 = _mm256_sub_ps(tmp10, tmp11);

            __m256 z1 = _mm256_mul_ps(_mm256_add_ps(tmp12, tmp13), _0_707106781);
            d2 = _mm256_add_ps(tmp13, z1);
            d6 = _mm256_sub_ps(tmp13, z1);

            tmp10 = _mm256_add_ps(tmp4, tmp5);
            tmp11 = _mm256_add_ps(tmp5, tmp6);
            tmp12 = _mm256_add_ps(tmp6, tmp7);

            __m256 z5 = _mm256_mul_ps(_mm256_sub_ps(tmp10, tmp12), _0_382683433);
            __m256 z2 = _mm256_add_ps(_mm256_mul_ps(tmp10, _0_541196100), z5);
            __m256 z4 = _mm256_add_ps(_mm256_mul_ps(tmp12, _1_306562965), z5);
            __m256 z3 = _mm256_mul_ps(tmp11, _0_707106781);

            __m256 z11 = _mm256_add_ps(tmp7, z3);
            __m256 z13 = _mm256_sub_ps(tmp7, z3);

            d1 = _mm256_add_ps(z11, z4);
            d3 = _mm256_sub_ps(z13, z2);
            d5 = _mm256_add_ps(z13, z2);
            d7 = _mm256_sub_ps(z11, z4);

            tmp10 = _mm256_permute2f128_ps(d0, d4, 0x20);
            tmp11 = _mm256_permute2f128_ps(d1, d5, 0x20);
            tmp12 = _mm256_permute2f128_ps(d2, d6, 0x20);
            tmp13 = _mm256_permute2f128_ps(d3, d7, 0x20);
            d4 = _mm256_permute2f128_ps(d0, d4, 0x31);
            d5 = _mm256_permute2f128_ps(d1, d5, 0x31);
            d6 = _mm256_permute2f128_ps(d2, d6, 0x31);
            d7 = _mm256_permute2f128_ps(d3, d7, 0x31);

            tmp0 = _mm256_unpacklo_ps(tmp10, tmp12);
            tmp1 = _mm256_unpackhi_ps(tmp10, tmp12);
            tmp2 = _mm256_unpacklo_ps(tmp11, tmp13);
            tmp3 = _mm256_unpackhi_ps(tmp11, tmp13);
            d0 = _mm256_unpacklo_ps(tmp0, tmp2);
            d1 = _mm256_unpackhi_ps(tmp0, tmp2);
            d2 = _mm256_unpacklo_ps(tmp1, tmp3);
            d3 = _mm256_unpackhi_ps(tmp1, tmp3);

            tmp0 = _mm256_unpacklo_ps(d4, d6);
            tmp1 = _mm256_unpackhi_ps(d4, d6);
            tmp2 = _mm256_unpacklo_ps(d5, d7);
            tmp3 = _mm256_unpackhi_ps(d5, d7);
            d4 = _mm256_unpacklo_ps(tmp0, tmp2);
            d5 = _mm256_unpackhi_ps(tmp0, tmp2);
            d6 = _mm256_unpacklo_ps(tmp1, tmp3);
            d7 = _mm256_unpackhi_ps(tmp1, tmp3);

            tmp0 = _mm256_add_ps(d0, d7);
            tmp1 = _mm256_add_ps(d1, d6);
            tmp2 = _mm256_add_ps(d2, d5);
            tmp3 = _mm256_add_ps(d3, d4);
            tmp7 = _mm256_sub_ps(d0, d7);
            tmp6 = _mm256_sub_ps(d1, d6);
            tmp5 = _mm256_sub_ps(d2, d5);
            tmp4 = _mm256_sub_ps(d3, d4);

            tmp10 = _mm256_add_ps(tmp0, tmp3);
            tmp13 = _mm256_sub_ps(tmp0, tmp3);
            tmp11 = _mm256_add_ps(tmp1, tmp2);
            tmp12 = _mm256_sub_ps(tmp1, tmp2);

            d0 = _mm256_add_ps(tmp10, tmp11);
            d4 = _mm256_sub_ps(tmp10, tmp11);

            z1 = _mm256_mul_ps(_mm256_add_ps(tmp12, tmp13), _0_707106781);
            d2 = _mm256_add_ps(tmp13, z1);
            d6 = _mm256_sub_ps(tmp13, z1);

            tmp10 = _mm256_add_ps(tmp4, tmp5);
            tmp11 = _mm256_add_ps(tmp5, tmp6);
            tmp12 = _mm256_add_ps(tmp6, tmp7);

            z5 = _mm256_mul_ps(_mm256_sub_ps(tmp10, tmp12), _0_382683433);
            z2 = _mm256_add_ps(_mm256_mul_ps(tmp10, _0_541196100), z5);
            z4 = _mm256_add_ps(_mm256_mul_ps(tmp12, _1_306562965), z5);
            z3 = _mm256_mul_ps(tmp11, _0_707106781);

            z11 = _mm256_add_ps(tmp7, z3);
            z13 = _mm256_sub_ps(tmp7, z3);

            d1 = _mm256_add_ps(z11, z4);
            d3 = _mm256_sub_ps(z13, z2);
            d5 = _mm256_add_ps(z13, z2);
            d7 = _mm256_sub_ps(z11, z4);

            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 0), d0)));
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 1), d1)));
            _mm256_storeu_si256((__m256i*)dst + 2, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 2), d2)));
            _mm256_storeu_si256((__m256i*)dst + 3, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 3), d3)));
            _mm256_storeu_si256((__m256i*)dst + 4, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 4), d4)));
            _mm256_storeu_si256((__m256i*)dst + 5, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 5), d5)));
            _mm256_storeu_si256((__m256i*)dst + 6, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 6), d6)));
            _mm256_storeu_si256((__m256i*)dst + 7, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 7), d7)));
        }

        const __m256i K32_PERM_LD = SIMD_MM256_SETR_EPI32(0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1);

        const __m256i K8_SHFL_VS = SIMD_MM256_SETR_EPI8(
            0x8, 0x9, 0x4, 0x5, 0x0, 0x1, -1, -1, 0xA, 0xB, 0x6, 0x7, 0x2, 0x3, -1, -1,
            0x8, 0x9, 0x4, 0x5, 0x0, 0x1, -1, -1, 0xA, 0xB, 0x6, 0x7, 0x2, 0x3, -1, -1);

        const __m256i K8_SHFL_SH = SIMD_MM256_SETR_EPI8(
            0x2, 0x3, -1, -1, 0x6, 0x7, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1,
            0x2, 0x3, -1, -1, 0x6, 0x7, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1);

        const __m256i K32_32 = SIMD_MM256_SET1_EPI32(32);

#if defined(SIMD_X64_ENABLE)
        SIMD_INLINE void WriteBits(uint8_t* data, size_t & pos, uint64_t & bitBuffer, size_t &bitCount, uint64_t shift, uint64_t value, uint64_t mask)
        {
            bitCount += shift;
            assert(bitCount <= 64);
            bitBuffer |= _pext_u64(value, mask) << (64 - bitCount);
            while (bitCount >= 16)
            {
                uint8_t byte = uint8_t(bitBuffer >> 56);
                data[pos++] = byte;
                if (byte == 255)
                    data[pos++] = 0;
                byte = uint8_t(bitBuffer >> 48);
                data[pos++] = byte;
                if (byte == 255)
                    data[pos++] = 0;
                bitBuffer <<= 16;
                bitCount -= 16;
            }
        }
#endif

        SIMD_INLINE void WriteBits(OutputMemoryStream& stream, const uint16_t bits[][2], size_t size)
        {
            size_t pos = stream.Pos();
            stream.Reserve(pos + size * 2);
            uint8_t* data = stream.Data();
            size_t& bitCount = stream.BitCount();
            size_t i = 0;
#if defined(SIMD_X64_ENABLE)
            uint64_t &bitBuffer = stream.BitBuffer();
            size_t size12 = AlignLoAny(size, 12);
            for (; i < size12; i += 12, bits += 12)
            {
                __m256i b0 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)(bits + 0)), K32_PERM_LD);
                __m256i b1 = _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)(bits + 6)), K32_PERM_LD);
                __m256i vs0 = _mm256_shuffle_epi8(b0, K8_SHFL_VS);
                __m256i vs1 = _mm256_shuffle_epi8(b1, K8_SHFL_VS);
                __m256i vv = Shuffle64i<0x0>(vs0, vs1);
                __m256i ss = Shuffle64i<0xF>(vs0, vs1);
                SIMD_ALIGNED(32) uint64_t value[4], mask[4], shift[4];
                _mm256_storeu_si256((__m256i*)value, vv);
                _mm256_storeu_si256((__m256i*)shift, _mm256_sad_epu8(ss, K_ZERO));
                __m256i s0 = _mm256_sub_epi32(K32_32, _mm256_shuffle_epi8(b0, K8_SHFL_SH));
                __m256i m0 = _mm256_srlv_epi32(K_INV_ZERO, s0);
                __m256i s1 = _mm256_sub_epi32(K32_32, _mm256_shuffle_epi8(b1, K8_SHFL_SH));
                __m256i m1 = _mm256_srlv_epi32(K_INV_ZERO, s1);
                __m256i ms0 = _mm256_shuffle_epi8(m0, K8_SHFL_VS);
                __m256i ms1 = _mm256_shuffle_epi8(m1, K8_SHFL_VS);
                _mm256_storeu_si256((__m256i*)mask, Shuffle64i<0x0>(ms0, ms1));
                WriteBits(data, pos, bitBuffer, bitCount, shift[0], value[0], mask[0]);
                WriteBits(data, pos, bitBuffer, bitCount, shift[2], value[2], mask[2]);
                WriteBits(data, pos, bitBuffer, bitCount, shift[1], value[1], mask[1]);
                WriteBits(data, pos, bitBuffer, bitCount, shift[3], value[3], mask[3]);
            }
            if (bitCount >= 8)
            {
                assert(bitCount < 16);
                uint8_t byte = uint8_t(bitBuffer >> 56);
                data[pos++] = byte;
                if (byte == 255)
                    data[pos++] = 0;
                bitBuffer <<= 8;
                bitCount -= 8;
            }
            for (; i < size; ++i, ++bits)
            {
                bitCount += bits[0][1];
                bitBuffer |= uint64_t(bits[0][0]) << (64 - bitCount);
                while (bitCount >= 8)
                {
                    uint8_t byte = uint8_t(bitBuffer >> 56);
                    data[pos++] = byte;
                    if (byte == 255)
                        data[pos++] = 0;
                    bitBuffer <<= 8;
                    bitCount -= 8;
                }
            }
#else
            uint32_t& bitBuffer = stream.BitBuffer();
            for (; i < size; ++i, ++bits)
            {
                bitCount += bits[0][1];
                bitBuffer |= uint32_t(bits[0][0]) << (32 - bitCount);
                while (bitCount >= 8)
                {
                    uint8_t byte = uint8_t(bitBuffer >> 24);
                    data[pos++] = byte;
                    if (byte == 255)
                        data[pos++] = 0;
                    bitBuffer <<= 8;
                    bitCount -= 8;
                }
            }
#endif
            stream.Seek(pos);
        }
    }
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i K32_PERM_LD = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);

        const __m512i K8_SHFL_VS = SIMD_MM512_SETR_EPI8(
            0x8, 0x9, 0x4, 0x5, 0x0, 0x1, -1, -1, 0xA, 0xB, 0x6, 0x7, 0x2, 0x3, -1, -1,
            0x8, 0x9, 0x4, 0x5, 0x0, 0x1, -1, -1, 0xA, 0xB, 0x6, 0x7, 0x2, 0x3, -1, -1,
            0x8, 0x9, 0x4, 0x5, 0x0, 0x1, -1, -1, 0xA, 0xB, 0x6, 0x7, 0x2, 0x3, -1, -1,
            0x8, 0x9, 0x4, 0x5, 0x0, 0x1, -1, -1, 0xA, 0xB, 0x6, 0x7, 0x2, 0x3, -1, -1);

        SIMD_INLINE void WriteBits(OutputMemoryStream& stream, const uint16_t bits[][2], size_t size)
        {
            size_t pos = stream.Pos();
            stream.Reserve(pos + size * 2);
            uint8_t* data = stream.Data();
            size_t& bitCount = stream.BitCount();
            size_t i = 0;
#if defined(SIMD_X64_ENABLE)
            uint64_t &bitBuffer = stream.BitBuffer();
            size_t size24 = AlignLoAny(size, 24);
            for (; i < size24; i += 24, bits += 24)
            {
                __m512i b0 = _mm512_permutexvar_epi32(K32_PERM_LD, _mm512_loadu_si512((__m512i*)(bits + 00)));
                __m512i b1 = _mm512_permutexvar_epi32(K32_PERM_LD, _mm512_loadu_si512((__m512i*)(bits + 12)));
                __m512i vs0 = _mm512_shuffle_epi8(b0, K8_SHFL_VS);
                __m512i vs1 = _mm512_shuffle_epi8(b1, K8_SHFL_VS);
                __m512i vv = Shuffle64i<0x00>(vs0, vs1);
                __m512i ss = Shuffle64i<0xFF>(vs0, vs1);
                SIMD_ALIGNED(64) uint64_t value[8], mask[8], shift[8];
                _mm512_storeu_si512((__m512i*)value, vv);
                _mm512_storeu_si512((__m512i*)shift, _mm512_sad_epu8(ss, K_ZERO));
                _mm512_storeu_si512((__m512i*)mask, _mm512_srlv_epi16(K_INV_ZERO, _mm512_sub_epi16(K16_0010, ss)));
                Avx2::WriteBits(data, pos, bitBuffer, bitCount, shift[0], value[0], mask[0]);
                Avx2::WriteBits(data, pos, bitBuffer, bitCount, shift[2], value[2], mask[2]);
                Avx2::WriteBits(data, pos, bitBuffer, bitCount, shift[4], value[4], mask[4]);
                Avx2::WriteBits(data, pos, bitBuffer, bitCount, shift[6], value[6], mask[6]);
                Avx2::WriteBits(data, pos, bitBuffer, bitCount, shift[1], value[1], mask[1]);
                Avx2::WriteBits(data, pos, bitBuffer, bitCount, shift[3], value[3], mask[3]);
                Avx2::WriteBits(data, pos, bitBuffer, bitCount, shift[5], value[5], mask[5]);
                Avx2::WriteBits(data, pos, bitBuffer, bitCount, shift[7], value[7], mask[7]);
            }
            if (bitCount >= 8)
            {
                assert(bitCount < 16);
                uint8_t byte = uint8_t(bitBuffer >> 56);
                data[pos++] = byte;
                if (byte == 255)
                    data[pos++] = 0;
                bitBuffer <<= 8;
                bitCount -= 8;
            }
            for (; i < size; ++i, ++bits)
            {
                bitCount += bits[0][1];
                bitBuffer |= uint64_t(bits[0][0]) << (64 - bitCount);
                while (bitCount >= 8)
                {
                    uint8_t byte = uint8_t(bitBuffer >> 56);
                    data[pos++] = byte;
                    if (byte == 255)
                        data[pos++] = 0;
                    bitBuffer <<= 8;
                    bitCount -= 8;
                }
            }
#else
            uint32_t& bitBuffer = stream.BitBuffer();
            for (; i < size; ++i, ++bits)
            {
                bitCount += bits[0][1];
                bitBuffer |= uint32_t(bits[0][0]) << (32 - bitCount);
                while (bitCount >= 8)
                {
                    uint8_t byte = uint8_t(bitBuffer >> 24);
                    data[pos++] = byte;
                    if (byte == 255)
                        data[pos++] = 0;
                    bitBuffer <<= 8;
                    bitCount -= 8;
                }
            }
#endif
            stream.Seek(pos);
        }
    }
#endif// SIMD_AVX512BW_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
    }
#endif// SIMD_NEON_ENABLE
}

#endif//__SimdImageSaveJpeg_h__
