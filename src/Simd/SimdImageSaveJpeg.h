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
#ifndef __SimdImageSaveJpeg_h__
#define __SimdImageSaveJpeg_h__

#include "Simd/SimdImageSave.h"

#define SIMD_JPEG_CALC_BITS_TABLE

namespace Simd
{
    namespace Base
    {
        struct BitBuf
        {
            static const uint32_t capacity = 1024;
            uint32_t size;
            uint16_t data[1024][2];

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
            stream.BitBuffer() |= bits[0] << (32 - stream.BitCount());
            while (stream.BitCount() >= 8)
            {
                uint8_t byte = stream.BitBuffer() >> 24;
                stream.Write8u(byte);
                if (byte == 255)
                    stream.Write8u(0);
                stream.BitBuffer() <<= 8;
                stream.BitCount() -= 8;
            }
        }

        SIMD_INLINE void WriteBits(OutputMemoryStream& stream, const uint16_t bits[][2], size_t size)
        {
            size_t pos = stream.Pos();
            stream.Reserve(pos + size * 2);
            uint8_t* data = stream.Data();
            size_t i = 0;
#if defined(SIMD_X64_ENABLE)
            uint64_t bitBuffer = uint64_t(stream.BitBuffer()) << 32;
            size_t & bitCount = stream.BitCount();
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
            stream.BitBuffer() = uint32_t(bitBuffer >> 32);
            while (bitCount >= 8)
            {
                uint8_t byte = uint8_t(stream.BitBuffer() >> 24);
                data[pos++] = byte;
                if (byte == 255)
                    data[pos++] = 0;
                stream.BitBuffer() <<= 8;
                bitCount -= 8;
            }
#endif
            for (; i < size; ++i, ++bits)
            {
                bitCount += bits[0][1];
                stream.BitBuffer() |= bits[0][0] << (32 - bitCount);
                while (bitCount >= 8)
                {
                    uint8_t byte = uint8_t(stream.BitBuffer() >> 24);
                    data[pos++] = byte;
                    if (byte == 255)
                        data[pos++] = 0;
                    stream.BitBuffer() <<= 8;
                    bitCount -= 8;
                }
            }
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
        extern const uint32_t JpegZigZagTi32[64];

        SIMD_INLINE void JpegDctV(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
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

            __m256 z1 = _mm256_mul_ps(_mm256_add_ps(tmp12, tmp13), _mm256_set1_ps(0.707106781f));
            d2 = _mm256_add_ps(tmp13, z1);
            d6 = _mm256_sub_ps(tmp13, z1);

            tmp10 = _mm256_add_ps(tmp4, tmp5);
            tmp11 = _mm256_add_ps(tmp5, tmp6);
            tmp12 = _mm256_add_ps(tmp6, tmp7);

            __m256 z5 = _mm256_mul_ps(_mm256_sub_ps(tmp10, tmp12), _mm256_set1_ps(0.382683433f));
            __m256 z2 = _mm256_add_ps(_mm256_mul_ps(tmp10, _mm256_set1_ps(0.541196100f)), z5);
            __m256 z4 = _mm256_add_ps(_mm256_mul_ps(tmp12, _mm256_set1_ps(1.306562965f)), z5);
            __m256 z3 = _mm256_mul_ps(tmp11, _mm256_set1_ps(0.707106781f));

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
    }
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
    }
#endif// SIMD_AVX512BW_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
    }
#endif// SIMD_NEON_ENABLE
}

#endif//__SimdImageSaveJpeg_h__
