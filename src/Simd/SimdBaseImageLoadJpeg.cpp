/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdImageLoad.h"
#include "Simd/SimdImageLoadJpeg.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
    namespace Base
    {
        SIMD_ALIGNED(16) const uint8_t JpegDeZigZag[80] =
        {
            0,  1,  8, 16,  9,  2,  3, 10,
           17, 24, 32, 25, 18, 11,  4,  5,
           12, 19, 26, 33, 40, 48, 41, 34,
           27, 20, 13,  6,  7, 14, 21, 28,
           35, 42, 49, 56, 57, 50, 43, 36,
           29, 22, 15, 23, 30, 37, 44, 51,
           58, 59, 52, 45, 38, 31, 39, 46,
           53, 60, 61, 54, 47, 55, 62, 63,
           63, 63, 63, 63, 63, 63, 63, 63,
           63, 63, 63, 63, 63, 63, 63, 63
        };

        //-------------------------------------------------------------------------------------------------

        int JpegHuffman::Build(const int* count)
        {
            int i, j, k = 0;
            for (i = 0; i < 16; ++i)
                for (j = 0; j < count[i]; ++j)
                    size[k++] = (uint8_t)(i + 1);
            size[k] = 0;
            unsigned int c = 0;
            for (j = 1, k = 0; j <= 16; ++j) 
            {
                delta[j] = k - c;
                if (size[k] == j) 
                {
                    while (size[k] == j)
                        code[k++] = (uint16_t)(c++);
                    if (c - 1 >= (1u << j)) 
                        return JpegLoadError("bad code lengths", "Corrupt JPEG");
                }
                maxcode[j] = c << (16 - j);
                c <<= 1;
            }
            maxcode[j] = 0xffffffff;
            memset(fast, 255, 1 << JpegFastBits);
            for (i = 0; i < k; ++i) 
            {
                int s = size[i];
                if (s <= JpegFastBits) 
                {
                    int c = code[i] << (JpegFastBits - s);
                    int m = 1 << (JpegFastBits - s);
                    for (j = 0; j < m; ++j)
                        fast[c + j] = (uint8_t)i;
                }
            }
            return 1;
        }

        void JpegHuffman::BuildFastAc()
        {
            for (int i = 0; i < (1 << JpegFastBits); ++i)
            {
                uint8_t f = fast[i];
                fast_ac[i] = 0;
                if (f < 255)
                {
                    int rs = values[f];
                    int run = (rs >> 4) & 15;
                    int magbits = rs & 15;
                    int len = size[f];
                    if (magbits && len + magbits <= JpegFastBits)
                    {
                        int k = ((i << len) & ((1 << JpegFastBits) - 1)) >> (JpegFastBits - magbits);
                        int m = 1 << (magbits - 1);
                        if (k < m)
                            k += (~0U << magbits) + 1;
                        if (k >= -128 && k <= 127)
                            fast_ac[i] = (int16_t)((k * 256) + (run * 16) + (len + magbits));
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        JpegContext::JpegContext(InputMemoryStream* s)
            : stream(s)
            , img_n(0)
        {
        }

        void JpegContext::Reset()
        {
            code_bits = 0;
            code_buffer = 0;
            nomore = 0;
            for(int i = 0; i < 4; ++i)
                img_comp[i].dc_pred = 0;
            marker = JpegMarkerNone;
            todo = restart_interval ? restart_interval : 0x7fffffff;
            eob_run = 0;
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE static void JpegGrowBufferUnsafe(JpegContext* j)
        {
            do 
            {
                unsigned int b = j->nomore ? 0 : j->stream->Get8u();
                if (b == 0xff) 
                {
                    int c = j->stream->Get8u();
                    while (c == 0xff) 
                        c = j->stream->Get8u();
                    if (c != 0) 
                    {
                        j->marker = (unsigned char)c;
                        j->nomore = 1;
                        return;
                    }
                }
                j->code_buffer |= b << (24 - j->code_bits);
                j->code_bits += 8;
            } while (j->code_bits <= 24);
        }

        static const uint32_t JpegBmask[17] = { 0,1,3,7,15,31,63,127,255,511,1023,2047,4095,8191,16383,32767,65535 }; // (1 << n) - 1

        SIMD_INLINE static int JpegHuffmanDecode(JpegContext* j, const JpegHuffman* h)
        {
            if (j->code_bits < 16) 
                JpegGrowBufferUnsafe(j);
            int c = (j->code_buffer >> (32 - JpegFastBits)) & ((1 << JpegFastBits) - 1);
            int k = h->fast[c];
            if (k < 255) 
            {
                int s = h->size[k];
                if (s > j->code_bits)
                    return -1;
                j->code_buffer <<= s;
                j->code_bits -= s;
                return h->values[k];
            }
            unsigned int temp = j->code_buffer >> 16;
            for (k = JpegFastBits + 1; ; ++k)
                if (temp < h->maxcode[k])
                    break;
            if (k == 17) 
            {
                j->code_bits -= 16;
                return -1;
            }
            if (k > j->code_bits)
                return -1;
            c = ((j->code_buffer >> (32 - k)) & JpegBmask[k]) + h->delta[k];
            assert((((j->code_buffer) >> (32 - h->size[c])) & JpegBmask[h->size[c]]) == h->code[c]);
            j->code_bits -= k;
            j->code_buffer <<= k;
            return h->values[c];
        }

        SIMD_INLINE uint32_t JpegLrot(uint32_t value, int shift)
        {
            return (value << shift) | (value >> (32 - shift));
        }

        SIMD_INLINE int JpegExtendReceive(JpegContext* j, int n)
        {
            static const int _jbias[16] = { 0,-1,-3,-7,-15,-31,-63,-127,-255,-511,-1023,-2047,-4095,-8191,-16383,-32767 };// bias[n] = (-1<<n) + 1
            if (j->code_bits < n) 
                JpegGrowBufferUnsafe(j);
            int sgn = (int32_t)j->code_buffer >> 31;
            unsigned int k = JpegLrot(j->code_buffer, n);
            if (n < 0 || n >= (int)(sizeof(JpegBmask) / sizeof(*JpegBmask))) 
                return 0;
            j->code_buffer = k & ~JpegBmask[n];
            k &= JpegBmask[n];
            j->code_bits -= n;
            return k + (_jbias[n] & ~sgn);
        }

        SIMD_INLINE int JpegGetBits(JpegContext* j, int n)
        {
            if (j->code_bits < n) 
                JpegGrowBufferUnsafe(j);
            unsigned int k = JpegLrot(j->code_buffer, n);
            j->code_buffer = k & ~JpegBmask[n];
            k &= JpegBmask[n];
            j->code_bits -= n;
            return k;
        }

        SIMD_INLINE int JpegGetBit(JpegContext* j)
        {
            if (j->code_bits < 1) 
                JpegGrowBufferUnsafe(j);
            unsigned int k = j->code_buffer;
            j->code_buffer <<= 1;
            --j->code_bits;
            return k & 0x80000000;
        }

        static int JpegDecodeBlock(JpegContext* j, short data[64], JpegHuffman* hdc, JpegHuffman* hac, int16_t* fac, int b, uint16_t* dequant)
        {
            if (j->code_bits < 16) 
                JpegGrowBufferUnsafe(j);
            int t = JpegHuffmanDecode(j, hdc);
            if (t < 0) 
                return JpegLoadError("bad huffman code", "Corrupt JPEG");
            memset(data, 0, 64 * sizeof(data[0]));
            int diff = t ? JpegExtendReceive(j, t) : 0;
            int dc = j->img_comp[b].dc_pred + diff;
            j->img_comp[b].dc_pred = dc;
            data[0] = (short)(dc * dequant[0]);
            int k = 1;
            do 
            {
                unsigned int zig;
                int c, r, s;
                if (j->code_bits < 16) 
                    JpegGrowBufferUnsafe(j);
                c = (j->code_buffer >> (32 - JpegFastBits)) & ((1 << JpegFastBits) - 1);
                r = fac[c];
                if (r)
                {
                    k += (r >> 4) & 15;
                    s = r & 15;
                    j->code_buffer <<= s;
                    j->code_bits -= s;
                    zig = Base::JpegDeZigZag[k++];
                    data[zig] = (short)((r >> 8) * dequant[zig]);
                }
                else 
                {
                    int rs = JpegHuffmanDecode(j, hac);
                    if (rs < 0) 
                        return JpegLoadError("bad huffman code", "Corrupt JPEG");
                    s = rs & 15;
                    r = rs >> 4;
                    if (s == 0) 
                    {
                        if (rs != 0xf0) 
                            break;
                        k += 16;
                    }
                    else 
                    {
                        k += r;
                        zig = Base::JpegDeZigZag[k++];
                        data[zig] = (short)(JpegExtendReceive(j, s) * dequant[zig]);
                    }
                }
            } while (k < 64);
            return 1;
        }

        static int JpegDecodeBlockProgDc(JpegContext* j, short data[64], JpegHuffman* hdc, int b)
        {
            if (j->spec_end != 0) 
                return JpegLoadError("can't merge dc and ac", "Corrupt JPEG");
            if (j->code_bits < 16) 
                JpegGrowBufferUnsafe(j);
            if (j->succ_high == 0) 
            {
                memset(data, 0, 64 * sizeof(data[0]));
                int t = JpegHuffmanDecode(j, hdc);
                if (t == -1) 
                    return JpegLoadError("can't merge dc and ac", "Corrupt JPEG");
                int diff = t ? JpegExtendReceive(j, t) : 0;
                int dc = j->img_comp[b].dc_pred + diff;
                j->img_comp[b].dc_pred = dc;
                data[0] = (short)(dc << j->succ_low);
            }
            else 
            {
                if (JpegGetBit(j))
                    data[0] += (short)(1 << j->succ_low);
            }
            return 1;
        }

        static int JpegDecodeBlockProgAc(JpegContext* j, short data[64], JpegHuffman* hac, int16_t* fac)
        {
            if (j->spec_start == 0) 
                return JpegLoadError("can't merge dc and ac", "Corrupt JPEG");
            if (j->succ_high == 0) 
            {
                int shift = j->succ_low;
                if (j->eob_run) 
                {
                    --j->eob_run;
                    return 1;
                }
                int k = j->spec_start;
                do 
                {
                    unsigned int zig;
                    int c, r, s;
                    if (j->code_bits < 16) 
                        JpegGrowBufferUnsafe(j);
                    c = (j->code_buffer >> (32 - JpegFastBits)) & ((1 << JpegFastBits) - 1);
                    r = fac[c];
                    if (r) 
                    { 
                        k += (r >> 4) & 15;
                        s = r & 15;
                        j->code_buffer <<= s;
                        j->code_bits -= s;
                        zig = Base::JpegDeZigZag[k++];
                        data[zig] = (short)((r >> 8) << shift);
                    }
                    else 
                    {
                        int rs = JpegHuffmanDecode(j, hac);
                        if (rs < 0) 
                            return JpegLoadError("bad huffman code", "Corrupt JPEG");
                        s = rs & 15;
                        r = rs >> 4;
                        if (s == 0) 
                        {
                            if (r < 15) 
                            {
                                j->eob_run = (1 << r);
                                if (r)
                                    j->eob_run += JpegGetBits(j, r);
                                --j->eob_run;
                                break;
                            }
                            k += 16;
                        }
                        else 
                        {
                            k += r;
                            zig = Base::JpegDeZigZag[k++];
                            data[zig] = (short)(JpegExtendReceive(j, s) << shift);
                        }
                    }
                } while (k <= j->spec_end);
            }
            else 
            {
                short bit = (short)(1 << j->succ_low);
                if (j->eob_run) 
                {
                    --j->eob_run;
                    for (int k = j->spec_start; k <= j->spec_end; ++k)
                    {
                        short* p = &data[Base::JpegDeZigZag[k]];
                        if (*p != 0 && JpegGetBit(j) && (*p & bit) == 0)
                        {
                            if (*p > 0)
                                *p += bit;
                            else
                                *p -= bit;
                        }
                    }
                }
                else 
                {
                    int k = j->spec_start;
                    do 
                    {
                        int r, s;
                        int rs = JpegHuffmanDecode(j, hac);
                        if (rs < 0) 
                            return JpegLoadError("bad huffman code", "Corrupt JPEG");
                        s = rs & 15;
                        r = rs >> 4;
                        if (s == 0) 
                        {
                            if (r < 15) 
                            {
                                j->eob_run = (1 << r) - 1;
                                if (r)
                                    j->eob_run += JpegGetBits(j, r);
                                r = 64;
                            }
                        }
                        else 
                        {
                            if (s != 1) 
                                return JpegLoadError("bad huffman code", "Corrupt JPEG");
                            if (JpegGetBit(j))
                                s = bit;
                            else
                                s = -bit;
                        }
                        while (k <= j->spec_end) 
                        {
                            short* p = &data[Base::JpegDeZigZag[k++]];
                            if (*p != 0) 
                            {
                                if (JpegGetBit(j) && (*p & bit) == 0) 
                                {
                                    if (*p > 0)
                                        *p += bit;
                                    else
                                        *p -= bit;
                                }
                            }
                            else
                            {
                                if (r == 0) 
                                {
                                    *p = (short)s;
                                    break;
                                }
                                --r;
                            }
                        }
                    } while (k <= j->spec_end);
                }
            }
            return 1;
        }

        template<class S, class D, int s> SIMD_INLINE void JpegIdct(const S* src, D* dst)
        {
            if (s == 8 && src[8] == 0 && src[16] == 0 && src[24] == 0 && src[32] == 0 && src[40] == 0 && src[48] == 0 && src[56] == 0)
            {
                int val = src[0] * 4;
                dst[0] = dst[8] = dst[16] = dst[24] = dst[32] = dst[40] = dst[48] = dst[56] = val;
            }
            else
            {
                int t0, t1, t2, t3, p1, p2, p3, p4, p5, x0, x1, x2, x3;
                p2 = src[s * 2];
                p3 = src[s * 6];
                p1 = (p2 + p3) * JpegIdctK00;
                t2 = p1 + p3 * JpegIdctK01;
                t3 = p1 + p2 * JpegIdctK02;
                p2 = src[s * 0];
                p3 = src[s * 4];
                t0 = (p2 + p3) * 4096;
                t1 = (p2 - p3) * 4096;
                int r = (s == 8 ? 512 : 65536 + (128 << 17));
                x0 = t0 + t3 + r;
                x3 = t0 - t3 + r;
                x1 = t1 + t2 + r;
                x2 = t1 - t2 + r;
                t0 = src[s * 7];
                t1 = src[s * 5];
                t2 = src[s * 3];
                t3 = src[s * 1];
                p3 = t0 + t2;
                p4 = t1 + t3;
                p1 = t0 + t3;
                p2 = t1 + t2;
                p5 = (p3 + p4) * JpegIdctK03;
                t0 = t0 * JpegIdctK04;
                t1 = t1 * JpegIdctK05;
                t2 = t2 * JpegIdctK06;
                t3 = t3 * JpegIdctK07;
                p1 = p5 + p1 * JpegIdctK08;
                p2 = p5 + p2 * JpegIdctK09;
                p3 = p3 * JpegIdctK10;
                p4 = p4 * JpegIdctK11;
                t3 += p1 + p4;
                t2 += p2 + p3;
                t1 += p2 + p4;
                t0 += p1 + p3;
                if (s == 8)
                {
                    dst[0] = (x0 + t3) >> 10;
                    dst[56] = (x0 - t3) >> 10;
                    dst[8] = (x1 + t2) >> 10;
                    dst[48] = (x1 - t2) >> 10;
                    dst[16] = (x2 + t1) >> 10;
                    dst[40] = (x2 - t1) >> 10;
                    dst[24] = (x3 + t0) >> 10;
                    dst[32] = (x3 - t0) >> 10;
                }
                else
                {
                    dst[0] = RestrictRange((x0 + t3) >> 17);
                    dst[7] = RestrictRange((x0 - t3) >> 17);
                    dst[1] = RestrictRange((x1 + t2) >> 17);
                    dst[6] = RestrictRange((x1 - t2) >> 17);
                    dst[2] = RestrictRange((x2 + t1) >> 17);
                    dst[5] = RestrictRange((x2 - t1) >> 17);
                    dst[3] = RestrictRange((x3 + t0) >> 17);
                    dst[4] = RestrictRange((x3 - t0) >> 17);
                }
            }
        }

        static void JpegIdctBlock(const int16_t* src, uint8_t* dst, int stride)
        {
            int buf[64];
            for (int i = 0; i < 8; ++i)
                JpegIdct<short, int, 8>(src + i, buf + i);
            for (int i = 0; i < 8; ++i, dst += stride) 
                JpegIdct<int, uint8_t, 1>(buf + 8 * i, dst);
        }

        static uint8_t JpegGetMarker(JpegContext* j)
        {
            uint8_t x;
            if (j->marker != JpegMarkerNone) 
            { 
                x = j->marker; 
                j->marker = JpegMarkerNone; 
                return x; 
            }
            x = j->stream->Get8u();
            if (x != 0xff) 
                return JpegMarkerNone;
            while (x == 0xff)
                x = j->stream->Get8u();
            return x;
        }

        static int JpegParseEntropyCodedData(JpegContext* z)
        {
            z->Reset();
            if (!z->progressive)
            {
                SIMD_ALIGNED(16) short data[64];
                if (z->scan_n == 1) 
                {
                    int n = z->order[0];
                    int w = (z->img_comp[n].x + 7) >> 3;
                    int h = (z->img_comp[n].y + 7) >> 3;
                    for (int j = 0; j < h; ++j) 
                    {
                        for (int i = 0; i < w; ++i) 
                        {
                            int ha = z->img_comp[n].ha;
                            if (!JpegDecodeBlock(z, data, z->huff_dc + z->img_comp[n].hd, z->huff_ac + ha, z->huff_ac[ha].fast_ac, n, z->dequant[z->img_comp[n].tq])) 
                                return 0;
                            z->idctBlock(data, z->img_comp[n].data + z->img_comp[n].w2 * j * 8 + i * 8, z->img_comp[n].w2);
                            if (--z->todo <= 0) 
                            {
                                if (z->code_bits < 24) 
                                    JpegGrowBufferUnsafe(z);
                                if (!z->NeedRestart()) 
                                    return 1;
                                z->Reset();
                            }
                        }
                    }
                    return 1;
                }
                else 
                {
                    for (int j = 0; j < z->img_mcu_y; ++j) 
                    {
                        for (int i = 0; i < z->img_mcu_x; ++i)
                        {
                            for (int k = 0; k < z->scan_n; ++k)
                            {
                                int n = z->order[k];
                                for (int y = 0; y < z->img_comp[n].v; ++y)
                                {
                                    for (int x = 0; x < z->img_comp[n].h; ++x)
                                    {
                                        int x2 = (i * z->img_comp[n].h + x) * 8;
                                        int y2 = (j * z->img_comp[n].v + y) * 8;
                                        int ha = z->img_comp[n].ha;
                                        if (!JpegDecodeBlock(z, data, z->huff_dc + z->img_comp[n].hd, z->huff_ac + ha, z->huff_ac[ha].fast_ac, n, z->dequant[z->img_comp[n].tq])) 
                                            return 0;
                                        z->idctBlock(data, z->img_comp[n].data + z->img_comp[n].w2 * y2 + x2, z->img_comp[n].w2);
                                    }
                                }
                            }
                            if (--z->todo <= 0) 
                            {
                                if (z->code_bits < 24) 
                                    JpegGrowBufferUnsafe(z);
                                if (!z->NeedRestart())
                                    return 1;
                                z->Reset();
                            }
                        }
                    }
                    return 1;
                }
            }
            else 
            {
                if (z->scan_n == 1) 
                {
                    int n = z->order[0];
                    int w = (z->img_comp[n].x + 7) >> 3;
                    int h = (z->img_comp[n].y + 7) >> 3;
                    for (int j = 0; j < h; ++j) 
                    {
                        for (int i = 0; i < w; ++i) 
                        {
                            short* data = z->img_comp[n].coeff + 64 * (i + j * z->img_comp[n].coeffW);
                            if (z->spec_start == 0) 
                            {
                                if (!JpegDecodeBlockProgDc(z, data, &z->huff_dc[z->img_comp[n].hd], n))
                                    return 0;
                            }
                            else 
                            {
                                int ha = z->img_comp[n].ha;
                                if (!JpegDecodeBlockProgAc(z, data, &z->huff_ac[ha], z->huff_ac[ha].fast_ac))
                                    return 0;
                            }
                            if (--z->todo <= 0) 
                            {
                                if (z->code_bits < 24) 
                                    JpegGrowBufferUnsafe(z);
                                if (!z->NeedRestart())
                                    return 1;
                                z->Reset();
                            }
                        }
                    }
                    return 1;
                }
                else 
                {
                    for (int j = 0; j < z->img_mcu_y; ++j)
                    {
                        for (int i = 0; i < z->img_mcu_x; ++i)
                        {
                            for (int k = 0; k < z->scan_n; ++k)
                            {
                                int n = z->order[k];
                                for (int y = 0; y < z->img_comp[n].v; ++y)
                                {
                                    for (int x = 0; x < z->img_comp[n].h; ++x)
                                    {
                                        int x2 = (i * z->img_comp[n].h + x);
                                        int y2 = (j * z->img_comp[n].v + y);
                                        short* data = z->img_comp[n].coeff + 64 * (x2 + y2 * z->img_comp[n].coeffW);
                                        if (!JpegDecodeBlockProgDc(z, data, &z->huff_dc[z->img_comp[n].hd], n))
                                            return 0;
                                    }
                                }
                            }
                            if (--z->todo <= 0) 
                            {
                                if (z->code_bits < 24) 
                                    JpegGrowBufferUnsafe(z);
                                if (!z->NeedRestart())
                                    return 1;
                                z->Reset();
                            }
                        }
                    }
                    return 1;
                }
            }
        }

        static void JpegFinish(JpegContext* z)
        {
            for (int n = 0; n < z->img_n; ++n) 
            {
                int w = (z->img_comp[n].x + 7) >> 3;
                int h = (z->img_comp[n].y + 7) >> 3;
                for (int j = 0; j < h; ++j) 
                {
                    for (int i = 0; i < w; ++i) 
                    {
                        short* data = z->img_comp[n].coeff + 64 * (i + j * z->img_comp[n].coeffW);
                        const uint16_t* dequant = z->dequant[z->img_comp[n].tq];
                        for (int k = 0; k < 64; ++k)
                            data[k] *= dequant[k];
                        z->idctBlock(data, z->img_comp[n].data + z->img_comp[n].w2 * j * 8 + i * 8, z->img_comp[n].w2);
                    }
                }
            }
        }

        static int JpegProcessMarker(JpegContext* z, int m)
        {
            int L;
            switch (m) 
            {
            case JpegMarkerNone:
                return JpegLoadError("expected marker", "Corrupt JPEG");

            case 0xDD: 
                if (z->stream->GetBe16u() != 4) 
                    return JpegLoadError("bad DRI len", "Corrupt JPEG");
                z->restart_interval = z->stream->GetBe16u();
                return 1;
            case 0xDB: 
                L = z->stream->GetBe16u() - 2;
                while (L > 0) 
                {
                    int q = z->stream->Get8u();
                    int p = q >> 4, sixteen = (p != 0);
                    int t = q & 15, i;
                    if (p != 0 && p != 1) 
                        return JpegLoadError("bad DQT type", "Corrupt JPEG");
                    if (t > 3) 
                        return JpegLoadError("bad DQT table", "Corrupt JPEG");
                    for (i = 0; i < 64; ++i)
                        z->dequant[t][Base::JpegDeZigZag[i]] = (uint16_t)(sixteen ? z->stream->GetBe16u() : z->stream->Get8u());
                    L -= (sixteen ? 129 : 65);
                }
                return L == 0;
            case 0xC4: 
                L = z->stream->GetBe16u() - 2;
                while (L > 0) {
                    uint8_t* v;
                    int sizes[16], i, n = 0;
                    int q = z->stream->Get8u();
                    int tc = q >> 4;
                    int th = q & 15;
                    if (tc > 1 || th > 3) 
                        return JpegLoadError("bad DHT header", "Corrupt JPEG");
                    for (i = 0; i < 16; ++i) 
                    {
                        sizes[i] = z->stream->Get8u();
                        n += sizes[i];
                    }
                    L -= 17;
                    if (tc == 0) 
                    {
                        if (!z->huff_dc[th].Build(sizes)) 
                            return 0;
                        v = z->huff_dc[th].values;
                    }
                    else 
                    {
                        if (!z->huff_ac[th].Build(sizes)) 
                            return 0;
                        v = z->huff_ac[th].values;
                    }
                    for (i = 0; i < n; ++i)
                        v[i] = z->stream->Get8u();
                    if (tc != 0)
                        z->huff_ac[th].BuildFastAc();
                    L -= n;
                }
                return L == 0;
            }
            if ((m >= 0xE0 && m <= 0xEF) || m == 0xFE) 
            {
                L = z->stream->GetBe16u();
                if (L < 2) 
                {
                    if (m == 0xFE)
                        return JpegLoadError("bad COM len", "Corrupt JPEG");
                    else
                        return JpegLoadError("bad APP len", "Corrupt JPEG");
                }
                L -= 2;
                if (m == 0xE0 && L >= 5) 
                { 
                    static const unsigned char tag[5] = { 'J','F','I','F','\0' };
                    int ok = 1;
                    for (int i = 0; i < 5; ++i)
                        if (z->stream->Get8u() != tag[i])
                            ok = 0;
                    L -= 5;
                    if (ok)
                        z->jfif = 1;
                }
                else if (m == 0xEE && L >= 12) 
                {
                    static const unsigned char tag[6] = { 'A','d','o','b','e','\0' };
                    int ok = 1;
                    for (int i = 0; i < 6; ++i)
                        if (z->stream->Get8u() != tag[i])
                            ok = 0;
                    L -= 6;
                    if (ok)
                    {
                        z->stream->Get8u();
                        z->stream->GetBe16u();
                        z->stream->GetBe16u();
                        z->app14_color_transform = z->stream->Get8u();
                        L -= 6;
                    }
                }
                if (L > 0)
                    z->stream->Skip(L);
                return 1;
            }
            return JpegLoadError("unknown marker", "Corrupt JPEG");
        }

        static int JpegProcessScanHeader(JpegContext* z)
        {
            int Ls = z->stream->GetBe16u();
            z->scan_n = z->stream->Get8u();
            if (z->scan_n < 1 || z->scan_n > 4 || z->scan_n > (int)z->img_n) 
                return JpegLoadError("bad SOS component count", "Corrupt JPEG");
            if (Ls != 6 + 2 * z->scan_n) 
                return JpegLoadError("bad SOS len", "Corrupt JPEG");
            for (int i = 0; i < z->scan_n; ++i) 
            {
                int id = z->stream->Get8u(), which;
                int q = z->stream->Get8u();
                for (which = 0; which < z->img_n; ++which)
                    if (z->img_comp[which].id == id)
                        break;
                if (which == z->img_n) 
                    return 0;
                z->img_comp[which].hd = q >> 4;   
                if (z->img_comp[which].hd > 3) 
                    return JpegLoadError("bad DC huff", "Corrupt JPEG");
                z->img_comp[which].ha = q & 15;   
                if (z->img_comp[which].ha > 3) 
                    return JpegLoadError("bad AC huff", "Corrupt JPEG");
                z->order[i] = which;
            }
            z->spec_start = z->stream->Get8u();
            z->spec_end = z->stream->Get8u();
            int aa = z->stream->Get8u();
            z->succ_high = (aa >> 4);
            z->succ_low = (aa & 15);
            if (z->progressive) 
            {
                if (z->spec_start > 63 || z->spec_end > 63 || z->spec_start > z->spec_end || z->succ_high > 13 || z->succ_low > 13)
                    return JpegLoadError("bad SOS", "Corrupt JPEG");
            }
            else 
            {
                if (z->spec_start != 0) 
                    return JpegLoadError("bad SOS", "Corrupt JPEG");
                if (z->succ_high != 0 || z->succ_low != 0) 
                    return JpegLoadError("bad SOS", "Corrupt JPEG");
                z->spec_end = 63;
            }
            return 1;
        }

        static int JpegProcessFrameHeader(JpegContext* z, int scan)
        {
            int Lf = z->stream->GetBe16u();         
            if (Lf < 11) 
                return JpegLoadError("bad SOF len", "Corrupt JPEG");
            int p = z->stream->Get8u();            
            if (p != 8) 
                return JpegLoadError("only 8-bit", "JPEG format not supported: 8-bit only");
            z->img_y = z->stream->GetBe16u();   
            if (z->img_y == 0) 
                return JpegLoadError("no header height", "JPEG format not supported: delayed height"); 
            z->img_x = z->stream->GetBe16u();   
            if (z->img_x == 0) 
                return JpegLoadError("0 width", "Corrupt JPEG"); 
            if (z->img_y > JpegMaxDimensions || z->img_x > JpegMaxDimensions) 
                return JpegLoadError("too large", "Very large image (corrupt?)");
            int c = z->stream->Get8u();
            if (c != 3 && c != 1 && c != 4) 
                return JpegLoadError("bad component count", "Corrupt JPEG");
            z->img_n = c;
            for (int i = 0; i < c; ++i)
                z->img_comp[i].data = NULL;
            if (Lf != 8 + 3 * z->img_n) 
                return JpegLoadError("bad SOF len", "Corrupt JPEG");
            z->rgb = 0;
            for (int i = 0; i < z->img_n; ++i) 
            {
                static const unsigned char rgb[3] = { 'R', 'G', 'B' };
                z->img_comp[i].id = z->stream->Get8u();
                if (z->img_n == 3 && z->img_comp[i].id == rgb[i])
                    ++z->rgb;
                int q = z->stream->Get8u();
                z->img_comp[i].h = (q >> 4);  
                if (!z->img_comp[i].h || z->img_comp[i].h > 4) 
                    return JpegLoadError("bad H", "Corrupt JPEG");
                z->img_comp[i].v = q & 15;    
                if (!z->img_comp[i].v || z->img_comp[i].v > 4) 
                    return JpegLoadError("bad V", "Corrupt JPEG");
                z->img_comp[i].tq = z->stream->Get8u();  
                if (z->img_comp[i].tq > 3) 
                    return JpegLoadError("bad TQ", "Corrupt JPEG");
            }
            if (scan) 
                return 1;
            if (z->img_x* z->img_y * z->img_n > INT_MAX) 
                return JpegLoadError("too large", "Image too large to decode");
            int h_max = 1, v_max = 1;
            for (int i = 0; i < z->img_n; ++i) 
            {
                h_max = Max(z->img_comp[i].h, h_max);
                v_max = Max(z->img_comp[i].v, v_max);
            }
            z->img_h_max = h_max;
            z->img_v_max = v_max;
            z->img_mcu_w = h_max * 8;
            z->img_mcu_h = v_max * 8;
            z->img_mcu_x = (z->img_x + z->img_mcu_w - 1) / z->img_mcu_w;
            z->img_mcu_y = (z->img_y + z->img_mcu_h - 1) / z->img_mcu_h;
            for (int i = 0; i < z->img_n; ++i) 
            {
                z->img_comp[i].x = (z->img_x * z->img_comp[i].h + h_max - 1) / h_max;
                z->img_comp[i].y = (z->img_y * z->img_comp[i].v + v_max - 1) / v_max;
                z->img_comp[i].w2 = z->img_mcu_x * z->img_comp[i].h * 8;
                z->img_comp[i].h2 = z->img_mcu_y * z->img_comp[i].v * 8;
                z->img_comp[i].coeff = 0;
                z->img_comp[i].bufD.Resize(z->img_comp[i].w2 * z->img_comp[i].h2);
                if (z->img_comp[i].bufD.Empty())
                    return JpegLoadError("outofmem", "Out of memory");
                z->img_comp[i].data = z->img_comp[i].bufD.data;
                if (z->progressive) 
                {
                    z->img_comp[i].coeffW = z->img_comp[i].w2 / 8;
                    z->img_comp[i].coeffH = z->img_comp[i].h2 / 8;
                    z->img_comp[i].bufC.Resize(z->img_comp[i].w2 * z->img_comp[i].h2);
                    if (z->img_comp[i].bufC.Empty())
                        return JpegLoadError("outofmem", "Out of memory");
                    z->img_comp[i].coeff = z->img_comp[i].bufC.data;
                }
            }
            return 1;
        }

        static int DecodeJpegHeader(JpegContext* z, int scan)
        {
            z->jfif = 0;
            z->app14_color_transform = -1;
            z->marker = JpegMarkerNone;
            int m = JpegGetMarker(z);
            if (m != JpegMarkerSoi)
                return JpegLoadError("no SOI", "Corrupt JPEG");
            if (scan) 
                return 1;
            m = JpegGetMarker(z);
            while (!(m == 0xC0 || m == 0xC1 || m == 0xC2))
            {
                if (!JpegProcessMarker(z, m)) 
                    return 0;
                m = JpegGetMarker(z);
                while (m == JpegMarkerNone) 
                {
                    if (z->stream->Eof()) 
                        return JpegLoadError("no SOF", "Corrupt JPEG");
                    m = JpegGetMarker(z);
                }
            }
            z->progressive = (m == 0xC2);
            if (!JpegProcessFrameHeader(z, scan)) 
                return 0;
            return 1;
        }

        static int JpegDecode(JpegContext* j)
        {
            j->restart_interval = 0;
            if (!DecodeJpegHeader(j, 0)) 
                return 0;
            int m = JpegGetMarker(j);
            while (m != JpegMarkerEoi)
            {
                if (m == JpegMarkerSos)
                {
                    if (!JpegProcessScanHeader(j)) 
                        return 0;
                    if (!JpegParseEntropyCodedData(j))
                        return 0;
                    if (j->marker == JpegMarkerNone) 
                    {
                        while (!j->stream->Eof()) 
                        {
                            int x = j->stream->Get8u();
                            if (x == 0xFF) 
                            {
                                j->marker = j->stream->Get8u();
                                break;
                            }
                        }
                    }
                }
                else if (m == JpegMarkerDnl)
                {
                    int Ld = j->stream->GetBe16u();
                    uint32_t NL = j->stream->GetBe16u();
                    if (Ld != 4) 
                        return JpegLoadError("bad DNL len", "Corrupt JPEG");
                    if (NL != j->img_y) 
                        return JpegLoadError("bad DNL height", "Corrupt JPEG");
                }
                else 
                {
                    if (!JpegProcessMarker(j, m)) 
                        return 0;
                }
                m = JpegGetMarker(j);
            }
            if (j->progressive)
                JpegFinish(j);
            return 1;
        }

        //-------------------------------------------------------------------------------------------------

#define jpeg__div4(x) ((uint8_t) ((x) >> 2))
#define jpeg__div16(x) ((uint8_t) ((x) >> 4))

        static uint8_t* JpegResampleRow1(uint8_t* , const uint8_t* in_near, const uint8_t* , int , int )
        {
            return (uint8_t*)in_near;
        }

        static uint8_t* JpegResampleRowV2(uint8_t* out, const uint8_t* in_near, const uint8_t* in_far, int w, int)
        {
            for (int i = 0; i < w; ++i)
                out[i] = jpeg__div4(3 * in_near[i] + in_far[i] + 2);
            return out;
        }

        static uint8_t* JpegResampleRowH2(uint8_t* out, const uint8_t* in_near, const uint8_t* , int w, int )
        {
            int i;
            const uint8_t* input = in_near;
            if (w == 1) 
            {
                out[0] = out[1] = input[0];
                return out;
            }
            out[0] = input[0];
            out[1] = jpeg__div4(input[0] * 3 + input[1] + 2);
            for (i = 1; i < w - 1; ++i) 
            {
                int n = 3 * input[i] + 2;
                out[i * 2 + 0] = jpeg__div4(n + input[i - 1]);
                out[i * 2 + 1] = jpeg__div4(n + input[i + 1]);
            }
            out[i * 2 + 0] = jpeg__div4(input[w - 2] * 3 + input[w - 1] + 2);
            out[i * 2 + 1] = input[w - 1];
            return out;
        }

        static uint8_t* JpegResampleRowHv2(uint8_t* out, const uint8_t* in_near, const uint8_t* in_far, int w, int)
        {
            int i, t0, t1;
            if (w == 1) 
            {
                out[0] = out[1] = jpeg__div4(3 * in_near[0] + in_far[0] + 2);
                return out;
            }
            t1 = 3 * in_near[0] + in_far[0];
            out[0] = jpeg__div4(t1 + 2);
            for (i = 1; i < w; ++i) 
            {
                t0 = t1;
                t1 = 3 * in_near[i] + in_far[i];
                out[i * 2 - 1] = jpeg__div16(3 * t0 + t1 + 8);
                out[i * 2] = jpeg__div16(3 * t1 + t0 + 8);
            }
            out[w * 2 - 1] = jpeg__div4(t1 + 2);
            return out;
        }

        static uint8_t* JpegResampleRowGeneric(uint8_t* out, const uint8_t* in_near, const uint8_t* , int w, int hs)
        {
            for (int i = 0; i < w; ++i)
                for (int j = 0; j < hs; ++j)
                    out[i * hs + j] = in_near[i];
            return out;
        }

        static void JpegYuvToRgbRow(uint8_t* out, const uint8_t* y, const uint8_t* pcb, const uint8_t* pcr, int count, int step)
        {
            for (int i = 0; i < count; ++i) 
            {
                YuvToRgb<Trect871>(y[i], pcb[i], pcr[i], out);
                out[3] = 255;
                out += step;
            }
        }

        struct JpegResample
        {
            ResampleRowPtr resample;
            uint8_t* line0, * line1;
            int hs, vs, w_lores, ystep, ypos;
        };

        SIMD_INLINE uint8_t JpegBlinn(uint8_t x, uint8_t y)
        {
            unsigned int t = x * y + 128;
            return (uint8_t)((t + (t >> 8)) >> 8);
        }

        static int JpegToRgba(JpegContext* z)
        {
            const int n = 4;
            int is_rgb = z->img_n == 3 && (z->rgb == 3 || (z->app14_color_transform == 0 && !z->jfif));
            unsigned int i;
            uint8_t* coutput[4] = { NULL, NULL, NULL, NULL };
            JpegResample res_comp[4];
            for (int k = 0; k < z->img_n; ++k)
            {
                JpegResample* r = &res_comp[k];

                z->img_comp[k].bufL.Resize(z->img_x + 3);
                if (z->img_comp[k].bufL.Empty()) 
                    return JpegLoadError("outofmem", "Out of memory");

                r->hs = z->img_h_max / z->img_comp[k].h;
                r->vs = z->img_v_max / z->img_comp[k].v;
                r->ystep = r->vs >> 1;
                r->w_lores = (z->img_x + r->hs - 1) / r->hs;
                r->ypos = 0;
                r->line0 = r->line1 = z->img_comp[k].data;

                if (r->hs == 1 && r->vs == 1) 
                    r->resample = JpegResampleRow1;
                else if (r->hs == 1 && r->vs == 2) 
                    r->resample = JpegResampleRowV2;
                else if (r->hs == 2 && r->vs == 1) 
                    r->resample = JpegResampleRowH2;
                else if (r->hs == 2 && r->vs == 2) 
                    r->resample = z->resampleRowHv2;
                else                               
                    r->resample = JpegResampleRowGeneric;
            }
            z->out.Resize(n * z->img_x * z->img_y + 1);
            if (z->out.Empty()) 
                return JpegLoadError("outofmem", "Out of memory");
            for (unsigned int j = 0; j < z->img_y; ++j) 
            {
                uint8_t* out = z->out.data + n * z->img_x * j;
                for (int k = 0; k < z->img_n; ++k)
                {
                    JpegResample* r = &res_comp[k];
                    int y_bot = r->ystep >= (r->vs >> 1);
                    coutput[k] = r->resample(z->img_comp[k].bufL.data,
                        y_bot ? r->line1 : r->line0,
                        y_bot ? r->line0 : r->line1,
                        r->w_lores, r->hs);
                    if (++r->ystep >= r->vs)
                    {
                        r->ystep = 0;
                        r->line0 = r->line1;
                        if (++r->ypos < z->img_comp[k].y)
                            r->line1 += z->img_comp[k].w2;
                    }
                }
                uint8_t* y = coutput[0];
                if (z->img_n == 3) 
                {
                    if (is_rgb) 
                    {
                        for (i = 0; i < z->img_x; ++i) 
                        {
                            out[0] = y[i];
                            out[1] = coutput[1][i];
                            out[2] = coutput[2][i];
                            out[3] = 255;
                            out += n;
                        }
                    }
                    else 
                        z->yuvToRgbRow(out, y, coutput[1], coutput[2], z->img_x, n);
                }
                else if (z->img_n == 4) 
                {
                    if (z->app14_color_transform == 0) 
                    {
                        for (i = 0; i < z->img_x; ++i) 
                        {
                            uint8_t m = coutput[3][i];
                            out[0] = JpegBlinn(coutput[0][i], m);
                            out[1] = JpegBlinn(coutput[1][i], m);
                            out[2] = JpegBlinn(coutput[2][i], m);
                            out[3] = 255;
                            out += n;
                        }
                    }
                    else if (z->app14_color_transform == 2) 
                    {
                        z->yuvToRgbRow(out, y, coutput[1], coutput[2], z->img_x, n);
                        for (i = 0; i < z->img_x; ++i) 
                        {
                            uint8_t m = coutput[3][i];
                            out[0] = JpegBlinn(255 - out[0], m);
                            out[1] = JpegBlinn(255 - out[1], m);
                            out[2] = JpegBlinn(255 - out[2], m);
                            out += n;
                        }
                    }
                    else 
                        z->yuvToRgbRow(out, y, coutput[1], coutput[2], z->img_x, n);
                }
                else
                {
                    for (i = 0; i < z->img_x; ++i)
                    {
                        out[0] = out[1] = out[2] = y[i];
                        out[3] = 255;
                        out += n;
                    }
                }
            }
            return 1;
           
        }

        //-------------------------------------------------------------------------------------------------

        void JpegYuv420pToBgr(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType)
        {
            size_t hL = height - 1, w2 = (width + 1) / 2;
            Array8u buf(width * 2 + 6);
            uint8_t* bu = buf.data, * bv = buf.data + width + 3;
            for (size_t row = 0; row < height; row += 1)
            {
                int odd = row & 1;
                JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
                JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
                for(size_t col = 0; col < width; ++col)
                    YuvToBgr<Trect871>(y[col], bu[col], bv[col], bgr + col * 3);
                y += yStride;
                bgr += bgrStride;
                if (odd)
                {
                    u += uStride;
                    v += vStride;
                }
            }
        }

        void JpegYuv420pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType)
        {
            size_t hL = height - 1, w2 = (width + 1) / 2;
            Array8u buf(width * 2 + 6);
            uint8_t* bu = buf.data, * bv = buf.data + width + 3;
            for (size_t row = 0; row < height; row += 1)
            {
                int odd = row & 1;
                JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
                JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
                for (size_t col = 0; col < width; ++col)
                    YuvToRgb<Trect871>(y[col], bu[col], bv[col], rgb + col * 3);
                y += yStride;
                rgb += rgbStride;
                if (odd)
                {
                    u += uStride;
                    v += vStride;
                }
            }
        }

        void JpegYuv420pToBgra(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType)
        {
            size_t hL = height - 1, w2 = (width + 1) / 2;
            Array8u buf(width * 2 + 6);
            uint8_t* bu = buf.data, * bv = buf.data + width + 3;
            for (size_t row = 0; row < height; row += 1)
            {
                int odd = row & 1;
                JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
                JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
                for (size_t col = 0; col < width; ++col)
                    YuvToBgra<Trect871>(y[col], bu[col], bv[col], alpha, bgra + col * 4);
                y += yStride;
                bgra += bgraStride;
                if (odd)
                {
                    u += uStride;
                    v += vStride;
                }
            }
        }

        void JpegYuv420pToRgba(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha, SimdYuvType)
        {
            size_t hL = height - 1, w2 = (width + 1) / 2;
            Array8u buf(width * 2 + 6);
            uint8_t* bu = buf.data, * bv = buf.data + width + 3;
            for (size_t row = 0; row < height; row += 1)
            {
                int odd = row & 1;
                JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
                JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
                for (size_t col = 0; col < width; ++col)
                    YuvToRgba<Trect871>(y[col], bu[col], bv[col], alpha, rgba + col * 4);
                y += yStride;
                rgba += rgbaStride;
                if (odd)
                {
                    u += uStride;
                    v += vStride;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE bool CanCopyGray(const JpegContext& jc)
        {
            return !(jc.img_n == 3 && (jc.rgb == 3 || (jc.app14_color_transform == 0 && !jc.jfif)));
        }

        SIMD_INLINE bool IsYuv444(const JpegContext & jc)
        {
            return jc.img_n == 3 && jc.rgb != 3 && 
                jc.img_comp[0].h2 == jc.img_comp[1].h2 && jc.img_comp[0].w2 == jc.img_comp[1].w2 &&
                jc.img_comp[0].h2 == jc.img_comp[2].h2 && jc.img_comp[0].w2 == jc.img_comp[2].w2;
        }

        SIMD_INLINE bool IsYuv420(const JpegContext& jc)
        {
            return jc.img_n == 3 && jc.rgb != 3 &&
                (jc.img_comp[0].h2 + 1) / 2 == jc.img_comp[1].h2 && (jc.img_comp[0].w2 + 1)/ 2 == jc.img_comp[1].w2 &&
                (jc.img_comp[0].h2 + 1) / 2 == jc.img_comp[2].h2 && (jc.img_comp[0].w2 + 1) / 2 == jc.img_comp[2].w2;
        }

        //-------------------------------------------------------------------------------------------------

        ImageJpegLoader::ImageJpegLoader(const ImageLoaderParam& param)
            : ImageLoader(param)
            , _context(new JpegContext(&_stream))
        {
            _context->idctBlock = JpegIdctBlock;
            _context->resampleRowHv2 = JpegResampleRowHv2;
            _context->yuvToRgbRow = JpegYuvToRgbRow;
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatRgb24;
            if (_param.format == SimdPixelFormatGray8)
                _context->rgbaToAny = Base::RgbaToGray;
            if (_param.format == SimdPixelFormatBgr24)
            {
                _context->yuv444pToBgr = Base::Yuv444pToBgrV2;
                _context->yuv420pToBgr = Base::JpegYuv420pToBgr;
                _context->rgbaToAny = Base::BgraToRgb;
            }
            if (_param.format == SimdPixelFormatBgra32)
            {
                _context->yuv444pToBgra = Base::Yuv444pToBgraV2;
                _context->yuv420pToBgra = Base::JpegYuv420pToBgra;
                _context->rgbaToAny = Base::BgraToRgba;
            }
            if (_param.format == SimdPixelFormatRgb24)
            {
                _context->yuv444pToBgr = Base::Yuv444pToRgbV2;
                _context->yuv420pToBgr = Base::JpegYuv420pToRgb;
                _context->rgbaToAny = Base::BgraToBgr;
            }
            if (_param.format == SimdPixelFormatRgba32)
            {
                _context->yuv444pToBgra = Base::Yuv444pToRgbaV2;
                _context->yuv420pToBgra = Base::JpegYuv420pToRgba;
            }
        }

        ImageJpegLoader::~ImageJpegLoader()
        {
            if (_context)
                delete _context;
        }

        bool ImageJpegLoader::FromStream()
        {
            if (!JpegDecode(_context))
                return false;
            _image.Recreate(_context->img_x, _context->img_y, (Image::Format)_param.format);
            if (CanCopyGray(*_context) && _param.format == SimdPixelFormatGray8)
            {
                Base::Copy(_context->img_comp[0].data, _context->img_comp[0].w2, _context->img_x, _context->img_y, 1, _image.data, _image.stride);
                return true;
            }
            if (IsYuv420(*_context))
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24:
                case SimdPixelFormatRgb24:
                    _context->yuv420pToBgr(_context->img_comp[0].data, _context->img_comp[0].w2, _context->img_comp[1].data, _context->img_comp[1].w2,
                        _context->img_comp[2].data, _context->img_comp[2].w2, _context->img_x, _context->img_y, _image.data, _image.stride, SimdYuvTrect871);
                    return true;
                case SimdPixelFormatBgra32:
                case SimdPixelFormatRgba32:
                    _context->yuv420pToBgra(_context->img_comp[0].data, _context->img_comp[0].w2, _context->img_comp[1].data, _context->img_comp[1].w2,
                        _context->img_comp[2].data, _context->img_comp[2].w2, _context->img_x, _context->img_y, _image.data, _image.stride, 0xFF, SimdYuvTrect871);
                    return true;
                }
            }
            if (IsYuv444(*_context))
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24:
                case SimdPixelFormatRgb24:
                    _context->yuv444pToBgr(_context->img_comp[0].data, _context->img_comp[0].w2, _context->img_comp[1].data, _context->img_comp[1].w2,
                        _context->img_comp[2].data, _context->img_comp[2].w2, _context->img_x, _context->img_y, _image.data, _image.stride, SimdYuvTrect871);
                    return true;
                case SimdPixelFormatBgra32:
                case SimdPixelFormatRgba32:
                    _context->yuv444pToBgra(_context->img_comp[0].data, _context->img_comp[0].w2, _context->img_comp[1].data, _context->img_comp[1].w2,
                        _context->img_comp[2].data, _context->img_comp[2].w2, _context->img_x, _context->img_y, _image.data, _image.stride, 0xFF, SimdYuvTrect871);
                    return true;
                }
            }
            if (JpegToRgba(_context))
            {
                size_t stride = 4 * _context->img_x;
                if (_param.format == SimdPixelFormatRgba32)
                    Base::Copy(_context->out.data, stride, _context->img_x, _context->img_y, 4, _image.data, _image.stride);
                else if (_param.format == SimdPixelFormatGray8 || _param.format == SimdPixelFormatBgr24 ||
                    _param.format == SimdPixelFormatBgra32 || _param.format == SimdPixelFormatRgb24)
                    _context->rgbaToAny(_context->out.data, _context->img_x, _context->img_y, stride, _image.data, _image.stride);
                else
                    return false;
                return true;
            }
            return false;
        }
    }
}
