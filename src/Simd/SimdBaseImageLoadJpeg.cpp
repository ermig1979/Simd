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

        const int JpegMarkerNone = 0xFF;
        const int JpegMaxDimensions = 1 << 24;
        const int JpegFastBits = 9;

#ifdef _MSC_VER
#define JPEG_NOTUSED(v)  (void)(v)
#else
#define JPEG_NOTUSED(v)  (void)sizeof(v)
#endif

#define jpeg__errpuc(x,y)  ((unsigned char *)(size_t) (JpegLoadError(x,y)?NULL:NULL))

#define jpeg_lrot(x,y)  (((x) << (y)) | ((x) >> (32 - (y))))

#define JPEG_SIMD_ALIGN(type, name) SIMD_ALIGNED(16) type name

        static uint8_t jpeg__compute_y(int r, int g, int b)
        {
            return (uint8_t)(((r * 77) + (g * 150) + (29 * b)) >> 8);
        }

        //------------------------------------------------------------------------------

        struct JpegHuffman
        {
            uint8_t  fast[1 << JpegFastBits];
            uint16_t code[256];
            uint8_t  values[256];
            uint8_t  size[257];
            unsigned int maxcode[18];
            int delta[17];

            int Build(const int* count)
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
        };

        struct JpegImgComp
        {
            int id;
            int h, v;
            int tq;
            int hd, ha;
            int dc_pred;
            int x, y, w2, h2;
            Array8u bufD, bufC, bufL;
            uint8_t* data;
            short* coeff;
            int coeff_w, coeff_h;
        };

        struct JpegContext
        {
            InputMemoryStream* stream;
            uint32_t img_x, img_y;
            int img_n, img_out_n;
            JpegHuffman huff_dc[4];
            JpegHuffman huff_ac[4];
            uint16_t dequant[4][64];
            int16_t fast_ac[4][1 << JpegFastBits];

            // sizes for components, interleaved MCUs
            int img_h_max, img_v_max;
            int img_mcu_x, img_mcu_y;
            int img_mcu_w, img_mcu_h;

            JpegImgComp img_comp[4];

            uint32_t   code_buffer; // jpeg entropy-coded buffer
            int            code_bits;   // number of valid bits
            unsigned char  marker;      // marker seen while filling entropy buffer
            int            nomore;      // flag if we saw a marker so must stop

            int            progressive;
            int            spec_start;
            int            spec_end;
            int            succ_high;
            int            succ_low;
            int            eob_run;
            int            jfif;
            int            app14_color_transform; // Adobe APP14 tag
            int            rgb;

            int scan_n, order[4];
            int restart_interval, todo;

            Array8u out;

            // kernels
            void (*idct_block_kernel)(uint8_t* out, int out_stride, short data[64]);
            void (*YCbCr_to_RGB_kernel)(uint8_t* out, const uint8_t* y, const uint8_t* pcb, const uint8_t* pcr, int count, int step);
            uint8_t* (*resample_row_hv_2_kernel)(uint8_t* out, uint8_t* in_near, uint8_t* in_far, int w, int hs);
        };

        //static int jpeg__build_huffman(JpegHuffman* h, const int* count)
        //{
        //    int i, j, k = 0;
        //    unsigned int code;
        //    // build size list for each symbol (from JPEG spec)
        //    for (i = 0; i < 16; ++i)
        //        for (j = 0; j < count[i]; ++j)
        //            h->size[k++] = (uint8_t)(i + 1);
        //    h->size[k] = 0;

        //    // compute actual symbols (from jpeg spec)
        //    code = 0;
        //    k = 0;
        //    for (j = 1; j <= 16; ++j) {
        //        // compute delta to add to code to compute symbol id
        //        h->delta[j] = k - code;
        //        if (h->size[k] == j) {
        //            while (h->size[k] == j)
        //                h->code[k++] = (uint16_t)(code++);
        //            if (code - 1 >= (1u << j)) return JpegLoadError("bad code lengths", "Corrupt JPEG");
        //        }
        //        // compute largest code + 1 for this size, preshifted as needed later
        //        h->maxcode[j] = code << (16 - j);
        //        code <<= 1;
        //    }
        //    h->maxcode[j] = 0xffffffff;

        //    // build non-spec acceleration table; 255 is flag for not-accelerated
        //    memset(h->fast, 255, 1 << JpegFastBits);
        //    for (i = 0; i < k; ++i) {
        //        int s = h->size[i];
        //        if (s <= JpegFastBits) {
        //            int c = h->code[i] << (JpegFastBits - s);
        //            int m = 1 << (JpegFastBits - s);
        //            for (j = 0; j < m; ++j) {
        //                h->fast[c + j] = (uint8_t)i;
        //            }
        //        }
        //    }
        //    return 1;
        //}

        // build a table that decodes both magnitude and value of small ACs in
        // one go.
        static void jpeg__build_fast_ac(int16_t* fast_ac, JpegHuffman* h)
        {
            int i;
            for (i = 0; i < (1 << JpegFastBits); ++i) {
                uint8_t fast = h->fast[i];
                fast_ac[i] = 0;
                if (fast < 255) {
                    int rs = h->values[fast];
                    int run = (rs >> 4) & 15;
                    int magbits = rs & 15;
                    int len = h->size[fast];

                    if (magbits && len + magbits <= JpegFastBits) {
                        // magnitude code followed by receive_extend code
                        int k = ((i << len) & ((1 << JpegFastBits) - 1)) >> (JpegFastBits - magbits);
                        int m = 1 << (magbits - 1);
                        if (k < m) k += (~0U << magbits) + 1;
                        // if the result is small enough, we can fit it in fast_ac table
                        if (k >= -128 && k <= 127)
                            fast_ac[i] = (int16_t)((k * 256) + (run * 16) + (len + magbits));
                    }
                }
            }
        }

        static void jpeg__grow_buffer_unsafe(JpegContext* j)
        {
            do {
                unsigned int b = j->nomore ? 0 : j->stream->Get8u();
                if (b == 0xff) {
                    int c = j->stream->Get8u();
                    while (c == 0xff) 
                        c = j->stream->Get8u(); // consume fill bytes
                    if (c != 0) {
                        j->marker = (unsigned char)c;
                        j->nomore = 1;
                        return;
                    }
                }
                j->code_buffer |= b << (24 - j->code_bits);
                j->code_bits += 8;
            } while (j->code_bits <= 24);
        }

        // (1 << n) - 1
        static const uint32_t jpeg__bmask[17] = { 0,1,3,7,15,31,63,127,255,511,1023,2047,4095,8191,16383,32767,65535 };

        // decode a jpeg huffman value from the bitstream
        SIMD_INLINE static int jpeg__jpeg_huff_decode(JpegContext* j, JpegHuffman* h)
        {
            unsigned int temp;
            int c, k;

            if (j->code_bits < 16) jpeg__grow_buffer_unsafe(j);

            // look at the top FAST_BITS and determine what symbol ID it is,
            // if the code is <= FAST_BITS
            c = (j->code_buffer >> (32 - JpegFastBits)) & ((1 << JpegFastBits) - 1);
            k = h->fast[c];
            if (k < 255) {
                int s = h->size[k];
                if (s > j->code_bits)
                    return -1;
                j->code_buffer <<= s;
                j->code_bits -= s;
                return h->values[k];
            }

            // naive test is to shift the code_buffer down so k bits are
            // valid, then test against maxcode. To speed this up, we've
            // preshifted maxcode left so that it has (16-k) 0s at the
            // end; in other words, regardless of the number of bits, it
            // wants to be compared against something shifted to have 16;
            // that way we don't need to shift inside the loop.
            temp = j->code_buffer >> 16;
            for (k = JpegFastBits + 1; ; ++k)
                if (temp < h->maxcode[k])
                    break;
            if (k == 17) {
                // error! code not found
                j->code_bits -= 16;
                return -1;
            }

            if (k > j->code_bits)
                return -1;

            // convert the huffman code to the symbol id
            c = ((j->code_buffer >> (32 - k)) & jpeg__bmask[k]) + h->delta[k];
            assert((((j->code_buffer) >> (32 - h->size[c])) & jpeg__bmask[h->size[c]]) == h->code[c]);

            // convert the id to a symbol
            j->code_bits -= k;
            j->code_buffer <<= k;
            return h->values[c];
        }

        // bias[n] = (-1<<n) + 1
        static const int jpeg__jbias[16] = { 0,-1,-3,-7,-15,-31,-63,-127,-255,-511,-1023,-2047,-4095,-8191,-16383,-32767 };

        // combined JPEG 'receive' and JPEG 'extend', since baseline
        // always extends everything it receives.
        SIMD_INLINE static int jpeg__extend_receive(JpegContext* j, int n)
        {
            unsigned int k;
            int sgn;
            if (j->code_bits < n) jpeg__grow_buffer_unsafe(j);

            sgn = (int32_t)j->code_buffer >> 31; // sign bit is always in MSB
            k = jpeg_lrot(j->code_buffer, n);
            if (n < 0 || n >= (int)(sizeof(jpeg__bmask) / sizeof(*jpeg__bmask))) return 0;
            j->code_buffer = k & ~jpeg__bmask[n];
            k &= jpeg__bmask[n];
            j->code_bits -= n;
            return k + (jpeg__jbias[n] & ~sgn);
        }

        // get some unsigned bits
        SIMD_INLINE static int jpeg__jpeg_get_bits(JpegContext* j, int n)
        {
            unsigned int k;
            if (j->code_bits < n) jpeg__grow_buffer_unsafe(j);
            k = jpeg_lrot(j->code_buffer, n);
            j->code_buffer = k & ~jpeg__bmask[n];
            k &= jpeg__bmask[n];
            j->code_bits -= n;
            return k;
        }

        SIMD_INLINE static int jpeg__jpeg_get_bit(JpegContext* j)
        {
            unsigned int k;
            if (j->code_bits < 1) jpeg__grow_buffer_unsafe(j);
            k = j->code_buffer;
            j->code_buffer <<= 1;
            --j->code_bits;
            return k & 0x80000000;
        }

        // decode one 64-entry block--
        static int jpeg__jpeg_decode_block(JpegContext* j, short data[64], JpegHuffman* hdc, JpegHuffman* hac, int16_t* fac, int b, uint16_t* dequant)
        {
            int diff, dc, k;
            int t;

            if (j->code_bits < 16) jpeg__grow_buffer_unsafe(j);
            t = jpeg__jpeg_huff_decode(j, hdc);
            if (t < 0) return JpegLoadError("bad huffman code", "Corrupt JPEG");

            // 0 all the ac values now so we can do it 32-bits at a time
            memset(data, 0, 64 * sizeof(data[0]));

            diff = t ? jpeg__extend_receive(j, t) : 0;
            dc = j->img_comp[b].dc_pred + diff;
            j->img_comp[b].dc_pred = dc;
            data[0] = (short)(dc * dequant[0]);

            // decode AC components, see JPEG spec
            k = 1;
            do {
                unsigned int zig;
                int c, r, s;
                if (j->code_bits < 16) jpeg__grow_buffer_unsafe(j);
                c = (j->code_buffer >> (32 - JpegFastBits)) & ((1 << JpegFastBits) - 1);
                r = fac[c];
                if (r) { // fast-AC path
                    k += (r >> 4) & 15; // run
                    s = r & 15; // combined length
                    j->code_buffer <<= s;
                    j->code_bits -= s;
                    // decode into unzigzag'd location
                    zig = Base::JpegDeZigZag[k++];
                    data[zig] = (short)((r >> 8) * dequant[zig]);
                }
                else {
                    int rs = jpeg__jpeg_huff_decode(j, hac);
                    if (rs < 0) return JpegLoadError("bad huffman code", "Corrupt JPEG");
                    s = rs & 15;
                    r = rs >> 4;
                    if (s == 0) {
                        if (rs != 0xf0) break; // end block
                        k += 16;
                    }
                    else {
                        k += r;
                        // decode into unzigzag'd location
                        zig = Base::JpegDeZigZag[k++];
                        data[zig] = (short)(jpeg__extend_receive(j, s) * dequant[zig]);
                    }
                }
            } while (k < 64);
            return 1;
        }

        static int jpeg__jpeg_decode_block_prog_dc(JpegContext* j, short data[64], JpegHuffman* hdc, int b)
        {
            int diff, dc;
            int t;
            if (j->spec_end != 0) return JpegLoadError("can't merge dc and ac", "Corrupt JPEG");

            if (j->code_bits < 16) jpeg__grow_buffer_unsafe(j);

            if (j->succ_high == 0) {
                // first scan for DC coefficient, must be first
                memset(data, 0, 64 * sizeof(data[0])); // 0 all the ac values now
                t = jpeg__jpeg_huff_decode(j, hdc);
                if (t == -1) return JpegLoadError("can't merge dc and ac", "Corrupt JPEG");
                diff = t ? jpeg__extend_receive(j, t) : 0;

                dc = j->img_comp[b].dc_pred + diff;
                j->img_comp[b].dc_pred = dc;
                data[0] = (short)(dc << j->succ_low);
            }
            else {
                // refinement scan for DC coefficient
                if (jpeg__jpeg_get_bit(j))
                    data[0] += (short)(1 << j->succ_low);
            }
            return 1;
        }

        // @OPTIMIZE: store non-zigzagged during the decode passes,
        // and only de-zigzag when dequantizing
        static int jpeg__jpeg_decode_block_prog_ac(JpegContext* j, short data[64], JpegHuffman* hac, int16_t* fac)
        {
            int k;
            if (j->spec_start == 0) return JpegLoadError("can't merge dc and ac", "Corrupt JPEG");

            if (j->succ_high == 0) {
                int shift = j->succ_low;

                if (j->eob_run) {
                    --j->eob_run;
                    return 1;
                }

                k = j->spec_start;
                do {
                    unsigned int zig;
                    int c, r, s;
                    if (j->code_bits < 16) jpeg__grow_buffer_unsafe(j);
                    c = (j->code_buffer >> (32 - JpegFastBits)) & ((1 << JpegFastBits) - 1);
                    r = fac[c];
                    if (r) { // fast-AC path
                        k += (r >> 4) & 15; // run
                        s = r & 15; // combined length
                        j->code_buffer <<= s;
                        j->code_bits -= s;
                        zig = Base::JpegDeZigZag[k++];
                        data[zig] = (short)((r >> 8) << shift);
                    }
                    else {
                        int rs = jpeg__jpeg_huff_decode(j, hac);
                        if (rs < 0) return JpegLoadError("bad huffman code", "Corrupt JPEG");
                        s = rs & 15;
                        r = rs >> 4;
                        if (s == 0) {
                            if (r < 15) {
                                j->eob_run = (1 << r);
                                if (r)
                                    j->eob_run += jpeg__jpeg_get_bits(j, r);
                                --j->eob_run;
                                break;
                            }
                            k += 16;
                        }
                        else {
                            k += r;
                            zig = Base::JpegDeZigZag[k++];
                            data[zig] = (short)(jpeg__extend_receive(j, s) << shift);
                        }
                    }
                } while (k <= j->spec_end);
            }
            else {
                // refinement scan for these AC coefficients

                short bit = (short)(1 << j->succ_low);

                if (j->eob_run) {
                    --j->eob_run;
                    for (k = j->spec_start; k <= j->spec_end; ++k) {
                        short* p = &data[Base::JpegDeZigZag[k]];
                        if (*p != 0)
                            if (jpeg__jpeg_get_bit(j))
                                if ((*p & bit) == 0) {
                                    if (*p > 0)
                                        *p += bit;
                                    else
                                        *p -= bit;
                                }
                    }
                }
                else {
                    k = j->spec_start;
                    do {
                        int r, s;
                        int rs = jpeg__jpeg_huff_decode(j, hac); // @OPTIMIZE see if we can use the fast path here, advance-by-r is so slow, eh
                        if (rs < 0) return JpegLoadError("bad huffman code", "Corrupt JPEG");
                        s = rs & 15;
                        r = rs >> 4;
                        if (s == 0) {
                            if (r < 15) {
                                j->eob_run = (1 << r) - 1;
                                if (r)
                                    j->eob_run += jpeg__jpeg_get_bits(j, r);
                                r = 64; // force end of block
                            }
                            else {
                                // r=15 s=0 should write 16 0s, so we just do
                                // a run of 15 0s and then write s (which is 0),
                                // so we don't have to do anything special here
                            }
                        }
                        else {
                            if (s != 1) return JpegLoadError("bad huffman code", "Corrupt JPEG");
                            // sign bit
                            if (jpeg__jpeg_get_bit(j))
                                s = bit;
                            else
                                s = -bit;
                        }

                        // advance by r
                        while (k <= j->spec_end) {
                            short* p = &data[Base::JpegDeZigZag[k++]];
                            if (*p != 0) {
                                if (jpeg__jpeg_get_bit(j))
                                    if ((*p & bit) == 0) {
                                        if (*p > 0)
                                            *p += bit;
                                        else
                                            *p -= bit;
                                    }
                            }
                            else {
                                if (r == 0) {
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

        // take a -128..127 value and jpeg__clamp it and convert to 0..255
        SIMD_INLINE static uint8_t jpeg__clamp(int x)
        {
            // trick to use a single test to catch both cases
            if ((unsigned int)x > 255) {
                if (x < 0) return 0;
                if (x > 255) return 255;
            }
            return (uint8_t)x;
        }

#define jpeg__f2f(x)  ((int) (((x) * 4096 + 0.5)))
#define jpeg__fsh(x)  ((x) * 4096)

        // derived from jidctint -- DCT_ISLOW
#define JPEG__IDCT_1D(s0,s1,s2,s3,s4,s5,s6,s7) \
   int t0,t1,t2,t3,p1,p2,p3,p4,p5,x0,x1,x2,x3; \
   p2 = s2;                                    \
   p3 = s6;                                    \
   p1 = (p2+p3) * jpeg__f2f(0.5411961f);       \
   t2 = p1 + p3*jpeg__f2f(-1.847759065f);      \
   t3 = p1 + p2*jpeg__f2f( 0.765366865f);      \
   p2 = s0;                                    \
   p3 = s4;                                    \
   t0 = jpeg__fsh(p2+p3);                      \
   t1 = jpeg__fsh(p2-p3);                      \
   x0 = t0+t3;                                 \
   x3 = t0-t3;                                 \
   x1 = t1+t2;                                 \
   x2 = t1-t2;                                 \
   t0 = s7;                                    \
   t1 = s5;                                    \
   t2 = s3;                                    \
   t3 = s1;                                    \
   p3 = t0+t2;                                 \
   p4 = t1+t3;                                 \
   p1 = t0+t3;                                 \
   p2 = t1+t2;                                 \
   p5 = (p3+p4)*jpeg__f2f( 1.175875602f);      \
   t0 = t0*jpeg__f2f( 0.298631336f);           \
   t1 = t1*jpeg__f2f( 2.053119869f);           \
   t2 = t2*jpeg__f2f( 3.072711026f);           \
   t3 = t3*jpeg__f2f( 1.501321110f);           \
   p1 = p5 + p1*jpeg__f2f(-0.899976223f);      \
   p2 = p5 + p2*jpeg__f2f(-2.562915447f);      \
   p3 = p3*jpeg__f2f(-1.961570560f);           \
   p4 = p4*jpeg__f2f(-0.390180644f);           \
   t3 += p1+p4;                                \
   t2 += p2+p3;                                \
   t1 += p2+p4;                                \
   t0 += p1+p3;

        static void jpeg__idct_block(uint8_t* out, int out_stride, short data[64])
        {
            int i, val[64], * v = val;
            uint8_t* o;
            short* d = data;

            // columns
            for (i = 0; i < 8; ++i, ++d, ++v) {
                // if all zeroes, shortcut -- this avoids dequantizing 0s and IDCTing
                if (d[8] == 0 && d[16] == 0 && d[24] == 0 && d[32] == 0
                    && d[40] == 0 && d[48] == 0 && d[56] == 0) {
                    //    no shortcut                 0     seconds
                    //    (1|2|3|4|5|6|7)==0          0     seconds
                    //    all separate               -0.047 seconds
                    //    1 && 2|3 && 4|5 && 6|7:    -0.047 seconds
                    int dcterm = d[0] * 4;
                    v[0] = v[8] = v[16] = v[24] = v[32] = v[40] = v[48] = v[56] = dcterm;
                }
                else {
                    JPEG__IDCT_1D(d[0], d[8], d[16], d[24], d[32], d[40], d[48], d[56])
                        // constants scaled things up by 1<<12; let's bring them back
                        // down, but keep 2 extra bits of precision
                        x0 += 512; x1 += 512; x2 += 512; x3 += 512;
                    v[0] = (x0 + t3) >> 10;
                    v[56] = (x0 - t3) >> 10;
                    v[8] = (x1 + t2) >> 10;
                    v[48] = (x1 - t2) >> 10;
                    v[16] = (x2 + t1) >> 10;
                    v[40] = (x2 - t1) >> 10;
                    v[24] = (x3 + t0) >> 10;
                    v[32] = (x3 - t0) >> 10;
                }
            }

            for (i = 0, v = val, o = out; i < 8; ++i, v += 8, o += out_stride) {
                // no fast case since the first 1D IDCT spread components out
                JPEG__IDCT_1D(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])
                    // constants scaled things up by 1<<12, plus we had 1<<2 from first
                    // loop, plus horizontal and vertical each scale by sqrt(8) so together
                    // we've got an extra 1<<3, so 1<<17 total we need to remove.
                    // so we want to round that, which means adding 0.5 * 1<<17,
                    // aka 65536. Also, we'll end up with -128 to 127 that we want
                    // to encode as 0..255 by adding 128, so we'll add that before the shift
                    x0 += 65536 + (128 << 17);
                x1 += 65536 + (128 << 17);
                x2 += 65536 + (128 << 17);
                x3 += 65536 + (128 << 17);
                // tried computing the shifts into temps, or'ing the temps to see
                // if any were out of range, but that was slower
                o[0] = jpeg__clamp((x0 + t3) >> 17);
                o[7] = jpeg__clamp((x0 - t3) >> 17);
                o[1] = jpeg__clamp((x1 + t2) >> 17);
                o[6] = jpeg__clamp((x1 - t2) >> 17);
                o[2] = jpeg__clamp((x2 + t1) >> 17);
                o[5] = jpeg__clamp((x2 - t1) >> 17);
                o[3] = jpeg__clamp((x3 + t0) >> 17);
                o[4] = jpeg__clamp((x3 - t0) >> 17);
            }
        }

#define JPEG__MARKER_none  0xff
        // if there's a pending marker from the entropy stream, return that
        // otherwise, fetch from the stream and get a marker. if there's no
        // marker, return 0xff, which is never a valid marker value
        static uint8_t jpeg__get_marker(JpegContext* j)
        {
            uint8_t x;
            if (j->marker != JPEG__MARKER_none) { x = j->marker; j->marker = JPEG__MARKER_none; return x; }
            x = j->stream->Get8u();
            if (x != 0xff) return JPEG__MARKER_none;
            while (x == 0xff)
                x = j->stream->Get8u(); // consume repeated 0xff fill bytes
            return x;
        }

        // in each scan, we'll have scan_n components, and the order
        // of the components is specified by order[]
#define JPEG__RESTART(x)     ((x) >= 0xd0 && (x) <= 0xd7)

// after a restart interval, jpeg__jpeg_reset the entropy decoder and
// the dc prediction
        static void jpeg__jpeg_reset(JpegContext* j)
        {
            j->code_bits = 0;
            j->code_buffer = 0;
            j->nomore = 0;
            j->img_comp[0].dc_pred = j->img_comp[1].dc_pred = j->img_comp[2].dc_pred = j->img_comp[3].dc_pred = 0;
            j->marker = JPEG__MARKER_none;
            j->todo = j->restart_interval ? j->restart_interval : 0x7fffffff;
            j->eob_run = 0;
            // no more than 1<<31 MCUs if no restart_interal? that's plenty safe,
            // since we don't even allow 1<<30 pixels
        }

        static int jpeg__parse_entropy_coded_data(JpegContext* z)
        {
            jpeg__jpeg_reset(z);
            if (!z->progressive) {
                if (z->scan_n == 1) {
                    int i, j;
                    JPEG_SIMD_ALIGN(short, data[64]);
                    int n = z->order[0];
                    // non-interleaved data, we just need to process one block at a time,
                    // in trivial scanline order
                    // number of blocks to do just depends on how many actual "pixels" this
                    // component has, independent of interleaved MCU blocking and such
                    int w = (z->img_comp[n].x + 7) >> 3;
                    int h = (z->img_comp[n].y + 7) >> 3;
                    for (j = 0; j < h; ++j) {
                        for (i = 0; i < w; ++i) {
                            int ha = z->img_comp[n].ha;
                            if (!jpeg__jpeg_decode_block(z, data, z->huff_dc + z->img_comp[n].hd, z->huff_ac + ha, z->fast_ac[ha], n, z->dequant[z->img_comp[n].tq])) return 0;
                            z->idct_block_kernel(z->img_comp[n].data + z->img_comp[n].w2 * j * 8 + i * 8, z->img_comp[n].w2, data);
                            // every data block is an MCU, so countdown the restart interval
                            if (--z->todo <= 0) {
                                if (z->code_bits < 24) jpeg__grow_buffer_unsafe(z);
                                // if it's NOT a restart, then just bail, so we get corrupt data
                                // rather than no data
                                if (!JPEG__RESTART(z->marker)) return 1;
                                jpeg__jpeg_reset(z);
                            }
                        }
                    }
                    return 1;
                }
                else { // interleaved
                    int i, j, k, x, y;
                    JPEG_SIMD_ALIGN(short, data[64]);
                    for (j = 0; j < z->img_mcu_y; ++j) {
                        for (i = 0; i < z->img_mcu_x; ++i) {
                            // scan an interleaved mcu... process scan_n components in order
                            for (k = 0; k < z->scan_n; ++k) {
                                int n = z->order[k];
                                // scan out an mcu's worth of this component; that's just determined
                                // by the basic H and V specified for the component
                                for (y = 0; y < z->img_comp[n].v; ++y) {
                                    for (x = 0; x < z->img_comp[n].h; ++x) {
                                        int x2 = (i * z->img_comp[n].h + x) * 8;
                                        int y2 = (j * z->img_comp[n].v + y) * 8;
                                        int ha = z->img_comp[n].ha;
                                        if (!jpeg__jpeg_decode_block(z, data, z->huff_dc + z->img_comp[n].hd, z->huff_ac + ha, z->fast_ac[ha], n, z->dequant[z->img_comp[n].tq])) return 0;
                                        z->idct_block_kernel(z->img_comp[n].data + z->img_comp[n].w2 * y2 + x2, z->img_comp[n].w2, data);
                                    }
                                }
                            }
                            // after all interleaved components, that's an interleaved MCU,
                            // so now count down the restart interval
                            if (--z->todo <= 0) {
                                if (z->code_bits < 24) jpeg__grow_buffer_unsafe(z);
                                if (!JPEG__RESTART(z->marker)) return 1;
                                jpeg__jpeg_reset(z);
                            }
                        }
                    }
                    return 1;
                }
            }
            else {
                if (z->scan_n == 1) {
                    int i, j;
                    int n = z->order[0];
                    // non-interleaved data, we just need to process one block at a time,
                    // in trivial scanline order
                    // number of blocks to do just depends on how many actual "pixels" this
                    // component has, independent of interleaved MCU blocking and such
                    int w = (z->img_comp[n].x + 7) >> 3;
                    int h = (z->img_comp[n].y + 7) >> 3;
                    for (j = 0; j < h; ++j) {
                        for (i = 0; i < w; ++i) {
                            short* data = z->img_comp[n].coeff + 64 * (i + j * z->img_comp[n].coeff_w);
                            if (z->spec_start == 0) {
                                if (!jpeg__jpeg_decode_block_prog_dc(z, data, &z->huff_dc[z->img_comp[n].hd], n))
                                    return 0;
                            }
                            else {
                                int ha = z->img_comp[n].ha;
                                if (!jpeg__jpeg_decode_block_prog_ac(z, data, &z->huff_ac[ha], z->fast_ac[ha]))
                                    return 0;
                            }
                            // every data block is an MCU, so countdown the restart interval
                            if (--z->todo <= 0) {
                                if (z->code_bits < 24) jpeg__grow_buffer_unsafe(z);
                                if (!JPEG__RESTART(z->marker)) return 1;
                                jpeg__jpeg_reset(z);
                            }
                        }
                    }
                    return 1;
                }
                else { // interleaved
                    int i, j, k, x, y;
                    for (j = 0; j < z->img_mcu_y; ++j) {
                        for (i = 0; i < z->img_mcu_x; ++i) {
                            // scan an interleaved mcu... process scan_n components in order
                            for (k = 0; k < z->scan_n; ++k) {
                                int n = z->order[k];
                                // scan out an mcu's worth of this component; that's just determined
                                // by the basic H and V specified for the component
                                for (y = 0; y < z->img_comp[n].v; ++y) {
                                    for (x = 0; x < z->img_comp[n].h; ++x) {
                                        int x2 = (i * z->img_comp[n].h + x);
                                        int y2 = (j * z->img_comp[n].v + y);
                                        short* data = z->img_comp[n].coeff + 64 * (x2 + y2 * z->img_comp[n].coeff_w);
                                        if (!jpeg__jpeg_decode_block_prog_dc(z, data, &z->huff_dc[z->img_comp[n].hd], n))
                                            return 0;
                                    }
                                }
                            }
                            // after all interleaved components, that's an interleaved MCU,
                            // so now count down the restart interval
                            if (--z->todo <= 0) {
                                if (z->code_bits < 24) jpeg__grow_buffer_unsafe(z);
                                if (!JPEG__RESTART(z->marker)) return 1;
                                jpeg__jpeg_reset(z);
                            }
                        }
                    }
                    return 1;
                }
            }
        }

        static void jpeg__jpeg_dequantize(short* data, uint16_t* dequant)
        {
            int i;
            for (i = 0; i < 64; ++i)
                data[i] *= dequant[i];
        }

        static void jpeg__jpeg_finish(JpegContext* z)
        {
            if (z->progressive) {
                // dequantize and idct the data
                int i, j, n;
                for (n = 0; n < z->img_n; ++n) {
                    int w = (z->img_comp[n].x + 7) >> 3;
                    int h = (z->img_comp[n].y + 7) >> 3;
                    for (j = 0; j < h; ++j) {
                        for (i = 0; i < w; ++i) {
                            short* data = z->img_comp[n].coeff + 64 * (i + j * z->img_comp[n].coeff_w);
                            jpeg__jpeg_dequantize(data, z->dequant[z->img_comp[n].tq]);
                            z->idct_block_kernel(z->img_comp[n].data + z->img_comp[n].w2 * j * 8 + i * 8, z->img_comp[n].w2, data);
                        }
                    }
                }
            }
        }

        static int jpeg__process_marker(JpegContext* z, int m)
        {
            int L;
            switch (m) {
            case JPEG__MARKER_none: // no marker found
                return JpegLoadError("expected marker", "Corrupt JPEG");

            case 0xDD: // DRI - specify restart interval
                if (z->stream->GetBe16u() != 4) return JpegLoadError("bad DRI len", "Corrupt JPEG");
                z->restart_interval = z->stream->GetBe16u();
                return 1;

            case 0xDB: // DQT - define quantization table
                L = z->stream->GetBe16u() - 2;
                while (L > 0) {
                    int q = z->stream->Get8u();
                    int p = q >> 4, sixteen = (p != 0);
                    int t = q & 15, i;
                    if (p != 0 && p != 1) return JpegLoadError("bad DQT type", "Corrupt JPEG");
                    if (t > 3) return JpegLoadError("bad DQT table", "Corrupt JPEG");

                    for (i = 0; i < 64; ++i)
                        z->dequant[t][Base::JpegDeZigZag[i]] = (uint16_t)(sixteen ? z->stream->GetBe16u() : z->stream->Get8u());
                    L -= (sixteen ? 129 : 65);
                }
                return L == 0;

            case 0xC4: // DHT - define huffman table
                L = z->stream->GetBe16u() - 2;
                while (L > 0) {
                    uint8_t* v;
                    int sizes[16], i, n = 0;
                    int q = z->stream->Get8u();
                    int tc = q >> 4;
                    int th = q & 15;
                    if (tc > 1 || th > 3) return JpegLoadError("bad DHT header", "Corrupt JPEG");
                    for (i = 0; i < 16; ++i) {
                        sizes[i] = z->stream->Get8u();
                        n += sizes[i];
                    }
                    L -= 17;
                    if (tc == 0) {
                        if (!z->huff_dc[th].Build(sizes)) 
                            return 0;
                        v = z->huff_dc[th].values;
                    }
                    else {
                        if (!z->huff_ac[th].Build(sizes)) 
                            return 0;
                        v = z->huff_ac[th].values;
                    }
                    for (i = 0; i < n; ++i)
                        v[i] = z->stream->Get8u();
                    if (tc != 0)
                        jpeg__build_fast_ac(z->fast_ac[th], z->huff_ac + th);
                    L -= n;
                }
                return L == 0;
            }

            // check for comment block or APP blocks
            if ((m >= 0xE0 && m <= 0xEF) || m == 0xFE) {
                L = z->stream->GetBe16u();
                if (L < 2) {
                    if (m == 0xFE)
                        return JpegLoadError("bad COM len", "Corrupt JPEG");
                    else
                        return JpegLoadError("bad APP len", "Corrupt JPEG");
                }
                L -= 2;

                if (m == 0xE0 && L >= 5) { // JFIF APP0 segment
                    static const unsigned char tag[5] = { 'J','F','I','F','\0' };
                    int ok = 1;
                    int i;
                    for (i = 0; i < 5; ++i)
                        if (z->stream->Get8u() != tag[i])
                            ok = 0;
                    L -= 5;
                    if (ok)
                        z->jfif = 1;
                }
                else if (m == 0xEE && L >= 12) { // Adobe APP14 segment
                    static const unsigned char tag[6] = { 'A','d','o','b','e','\0' };
                    int ok = 1;
                    int i;
                    for (i = 0; i < 6; ++i)
                        if (z->stream->Get8u() != tag[i])
                            ok = 0;
                    L -= 6;
                    if (ok) {
                        z->stream->Get8u(); // version
                        z->stream->GetBe16u(); // flags0
                        z->stream->GetBe16u(); // flags1
                        z->app14_color_transform = z->stream->Get8u(); // color transform
                        L -= 6;
                    }
                }

                if (L > 0)
                    z->stream->Skip(L);

                return 1;
            }

            return JpegLoadError("unknown marker", "Corrupt JPEG");
        }

        // after we see SOS
        static int jpeg__process_scan_header(JpegContext* z)
        {
            int i;
            int Ls = z->stream->GetBe16u();
            z->scan_n = z->stream->Get8u();
            if (z->scan_n < 1 || z->scan_n > 4 || z->scan_n > (int)z->img_n) return JpegLoadError("bad SOS component count", "Corrupt JPEG");
            if (Ls != 6 + 2 * z->scan_n) return JpegLoadError("bad SOS len", "Corrupt JPEG");
            for (i = 0; i < z->scan_n; ++i) {
                int id = z->stream->Get8u(), which;
                int q = z->stream->Get8u();
                for (which = 0; which < z->img_n; ++which)
                    if (z->img_comp[which].id == id)
                        break;
                if (which == z->img_n) return 0; // no match
                z->img_comp[which].hd = q >> 4;   if (z->img_comp[which].hd > 3) return JpegLoadError("bad DC huff", "Corrupt JPEG");
                z->img_comp[which].ha = q & 15;   if (z->img_comp[which].ha > 3) return JpegLoadError("bad AC huff", "Corrupt JPEG");
                z->order[i] = which;
            }

            {
                int aa;
                z->spec_start = z->stream->Get8u();
                z->spec_end = z->stream->Get8u(); // should be 63, but might be 0
                aa = z->stream->Get8u();
                z->succ_high = (aa >> 4);
                z->succ_low = (aa & 15);
                if (z->progressive) {
                    if (z->spec_start > 63 || z->spec_end > 63 || z->spec_start > z->spec_end || z->succ_high > 13 || z->succ_low > 13)
                        return JpegLoadError("bad SOS", "Corrupt JPEG");
                }
                else {
                    if (z->spec_start != 0) return JpegLoadError("bad SOS", "Corrupt JPEG");
                    if (z->succ_high != 0 || z->succ_low != 0) return JpegLoadError("bad SOS", "Corrupt JPEG");
                    z->spec_end = 63;
                }
            }

            return 1;
        }

        static int jpeg__process_frame_header(JpegContext* z, int scan)
        {
            int Lf, p, i, q, h_max = 1, v_max = 1, c;
            Lf = z->stream->GetBe16u();         if (Lf < 11) return JpegLoadError("bad SOF len", "Corrupt JPEG"); // JPEG
            p = z->stream->Get8u();            if (p != 8) return JpegLoadError("only 8-bit", "JPEG format not supported: 8-bit only"); // JPEG baseline
            z->img_y = z->stream->GetBe16u();   if (z->img_y == 0) return JpegLoadError("no header height", "JPEG format not supported: delayed height"); // Legal, but we don't handle it--but neither does IJG
            z->img_x = z->stream->GetBe16u();   if (z->img_x == 0) return JpegLoadError("0 width", "Corrupt JPEG"); // JPEG requires
            if (z->img_y > JpegMaxDimensions) return JpegLoadError("too large", "Very large image (corrupt?)");
            if (z->img_x > JpegMaxDimensions) return JpegLoadError("too large", "Very large image (corrupt?)");
            c = z->stream->Get8u();
            if (c != 3 && c != 1 && c != 4) return JpegLoadError("bad component count", "Corrupt JPEG");
            z->img_n = c;
            for (i = 0; i < c; ++i) {
                z->img_comp[i].data = NULL;
                //z->img_comp[i].linebuf = NULL;
            }

            if (Lf != 8 + 3 * z->img_n) return JpegLoadError("bad SOF len", "Corrupt JPEG");

            z->rgb = 0;
            for (i = 0; i < z->img_n; ++i) {
                static const unsigned char rgb[3] = { 'R', 'G', 'B' };
                z->img_comp[i].id = z->stream->Get8u();
                if (z->img_n == 3 && z->img_comp[i].id == rgb[i])
                    ++z->rgb;
                q = z->stream->Get8u();
                z->img_comp[i].h = (q >> 4);  if (!z->img_comp[i].h || z->img_comp[i].h > 4) return JpegLoadError("bad H", "Corrupt JPEG");
                z->img_comp[i].v = q & 15;    if (!z->img_comp[i].v || z->img_comp[i].v > 4) return JpegLoadError("bad V", "Corrupt JPEG");
                z->img_comp[i].tq = z->stream->Get8u();  if (z->img_comp[i].tq > 3) return JpegLoadError("bad TQ", "Corrupt JPEG");
            }

            if (scan) 
                return 1;

            if (z->img_x* z->img_y * z->img_n > INT_MAX) return JpegLoadError("too large", "Image too large to decode");

            for (i = 0; i < z->img_n; ++i) {
                if (z->img_comp[i].h > h_max) h_max = z->img_comp[i].h;
                if (z->img_comp[i].v > v_max) v_max = z->img_comp[i].v;
            }

            // compute interleaved mcu info
            z->img_h_max = h_max;
            z->img_v_max = v_max;
            z->img_mcu_w = h_max * 8;
            z->img_mcu_h = v_max * 8;
            // these sizes can't be more than 17 bits
            z->img_mcu_x = (z->img_x + z->img_mcu_w - 1) / z->img_mcu_w;
            z->img_mcu_y = (z->img_y + z->img_mcu_h - 1) / z->img_mcu_h;

            for (i = 0; i < z->img_n; ++i) {
                // number of effective pixels (e.g. for non-interleaved MCU)
                z->img_comp[i].x = (z->img_x * z->img_comp[i].h + h_max - 1) / h_max;
                z->img_comp[i].y = (z->img_y * z->img_comp[i].v + v_max - 1) / v_max;
                // to simplify generation, we'll allocate enough memory to decode
                // the bogus oversized data from using interleaved MCUs and their
                // big blocks (e.g. a 16x16 iMCU on an image of width 33); we won't
                // discard the extra data until colorspace conversion
                //
                // img_mcu_x, img_mcu_y: <=17 bits; comp[i].h and .v are <=4 (checked earlier)
                // so these muls can't overflow with 32-bit ints (which we require)
                z->img_comp[i].w2 = z->img_mcu_x * z->img_comp[i].h * 8;
                z->img_comp[i].h2 = z->img_mcu_y * z->img_comp[i].v * 8;
                z->img_comp[i].coeff = 0;
                //z->img_comp[i].raw_coeff = 0;
                //z->img_comp[i].linebuf = NULL;
                z->img_comp[i].bufD.Resize(z->img_comp[i].w2 * z->img_comp[i].h2);
                if (z->img_comp[i].bufD.Empty())
                    return JpegLoadError("outofmem", "Out of memory");
                z->img_comp[i].data = z->img_comp[i].bufD.data;
                if (z->progressive) {
                    // w2, h2 are multiples of 8 (see above)
                    z->img_comp[i].coeff_w = z->img_comp[i].w2 / 8;
                    z->img_comp[i].coeff_h = z->img_comp[i].h2 / 8;
                    z->img_comp[i].bufC.Resize(z->img_comp[i].w2 * z->img_comp[i].h2 * sizeof(short));
                    if (z->img_comp[i].bufC.Empty())
                        return JpegLoadError("outofmem", "Out of memory");
                    z->img_comp[i].coeff = (short*)z->img_comp[i].bufC.data;
                }
            }

            return 1;
        }

        // use comparisons since in some cases we handle more than one case (e.g. SOF)
#define jpeg__DNL(x)         ((x) == 0xdc)
#define jpeg__SOI(x)         ((x) == 0xd8)
#define jpeg__EOI(x)         ((x) == 0xd9)
#define jpeg__SOF(x)         ((x) == 0xc0 || (x) == 0xc1 || (x) == 0xc2)
#define jpeg__SOS(x)         ((x) == 0xda)

#define jpeg__SOF_progressive(x)   ((x) == 0xc2)

        static int DecodeJpegHeader(JpegContext* z, int scan)
        {
            int m;
            z->jfif = 0;
            z->app14_color_transform = -1; // valid values are 0,1,2
            z->marker = JPEG__MARKER_none; // initialize cached marker to empty
            m = jpeg__get_marker(z);
            if (!jpeg__SOI(m)) return JpegLoadError("no SOI", "Corrupt JPEG");
            if (scan) 
                return 1;
            m = jpeg__get_marker(z);
            while (!jpeg__SOF(m)) {
                if (!jpeg__process_marker(z, m)) return 0;
                m = jpeg__get_marker(z);
                while (m == JPEG__MARKER_none) {
                    // some files have extra padding after their blocks, so ok, we'll scan
                    if (z->stream->Eof()) 
                        return JpegLoadError("no SOF", "Corrupt JPEG");
                    m = jpeg__get_marker(z);
                }
            }
            z->progressive = jpeg__SOF_progressive(m);
            if (!jpeg__process_frame_header(z, scan)) return 0;
            return 1;
        }

        // decode image to YCbCr format
        static int jpeg__decode_jpeg_image(JpegContext* j)
        {
            int m;
            j->restart_interval = 0;
            if (!DecodeJpegHeader(j, 0)) return 0;
            m = jpeg__get_marker(j);
            while (!jpeg__EOI(m)) {
                if (jpeg__SOS(m)) {
                    if (!jpeg__process_scan_header(j)) return 0;
                    if (!jpeg__parse_entropy_coded_data(j)) return 0;
                    if (j->marker == JPEG__MARKER_none) {
                        // handle 0s at the end of image data from IP Kamera 9060
                        while (!j->stream->Eof()) {
                            int x = j->stream->Get8u();
                            if (x == 255) {
                                j->marker = j->stream->Get8u();
                                break;
                            }
                        }
                        // if we reach eof without hitting a marker, jpeg__get_marker() below will fail and we'll eventually return 0
                    }
                }
                else if (jpeg__DNL(m)) {
                    int Ld = j->stream->GetBe16u();
                    uint32_t NL = j->stream->GetBe16u();
                    if (Ld != 4) return JpegLoadError("bad DNL len", "Corrupt JPEG");
                    if (NL != j->img_y) return JpegLoadError("bad DNL height", "Corrupt JPEG");
                }
                else {
                    if (!jpeg__process_marker(j, m)) return 0;
                }
                m = jpeg__get_marker(j);
            }
            if (j->progressive)
                jpeg__jpeg_finish(j);
            return 1;
        }

        // static jfif-centered resampling (across block boundaries)

        typedef uint8_t* (*ResampleRowPtr)(uint8_t* out, uint8_t* in0, uint8_t* in1, int w, int hs);

#define jpeg__div4(x) ((uint8_t) ((x) >> 2))

        static uint8_t* resample_row_1(uint8_t* out, uint8_t* in_near, uint8_t* in_far, int w, int hs)
        {
            JPEG_NOTUSED(out);
            JPEG_NOTUSED(in_far);
            JPEG_NOTUSED(w);
            JPEG_NOTUSED(hs);
            return in_near;
        }

        static uint8_t* jpeg__resample_row_v_2(uint8_t* out, uint8_t* in_near, uint8_t* in_far, int w, int hs)
        {
            // need to generate two samples vertically for every one in input
            int i;
            JPEG_NOTUSED(hs);
            for (i = 0; i < w; ++i)
                out[i] = jpeg__div4(3 * in_near[i] + in_far[i] + 2);
            return out;
        }

        static uint8_t* jpeg__resample_row_h_2(uint8_t* out, uint8_t* in_near, uint8_t* in_far, int w, int hs)
        {
            // need to generate two samples horizontally for every one in input
            int i;
            uint8_t* input = in_near;

            if (w == 1) {
                // if only one sample, can't do any interpolation
                out[0] = out[1] = input[0];
                return out;
            }

            out[0] = input[0];
            out[1] = jpeg__div4(input[0] * 3 + input[1] + 2);
            for (i = 1; i < w - 1; ++i) {
                int n = 3 * input[i] + 2;
                out[i * 2 + 0] = jpeg__div4(n + input[i - 1]);
                out[i * 2 + 1] = jpeg__div4(n + input[i + 1]);
            }
            out[i * 2 + 0] = jpeg__div4(input[w - 2] * 3 + input[w - 1] + 2);
            out[i * 2 + 1] = input[w - 1];

            JPEG_NOTUSED(in_far);
            JPEG_NOTUSED(hs);

            return out;
        }

#define jpeg__div16(x) ((uint8_t) ((x) >> 4))

        static uint8_t* jpeg__resample_row_hv_2(uint8_t* out, uint8_t* in_near, uint8_t* in_far, int w, int hs)
        {
            // need to generate 2x2 samples for every one in input
            int i, t0, t1;
            if (w == 1) {
                out[0] = out[1] = jpeg__div4(3 * in_near[0] + in_far[0] + 2);
                return out;
            }

            t1 = 3 * in_near[0] + in_far[0];
            out[0] = jpeg__div4(t1 + 2);
            for (i = 1; i < w; ++i) {
                t0 = t1;
                t1 = 3 * in_near[i] + in_far[i];
                out[i * 2 - 1] = jpeg__div16(3 * t0 + t1 + 8);
                out[i * 2] = jpeg__div16(3 * t1 + t0 + 8);
            }
            out[w * 2 - 1] = jpeg__div4(t1 + 2);

            JPEG_NOTUSED(hs);

            return out;
        }

        static uint8_t* jpeg__resample_row_generic(uint8_t* out, uint8_t* in_near, uint8_t* in_far, int w, int hs)
        {
            // resample with nearest-neighbor
            int i, j;
            JPEG_NOTUSED(in_far);
            for (i = 0; i < w; ++i)
                for (j = 0; j < hs; ++j)
                    out[i * hs + j] = in_near[i];
            return out;
        }

        // this is a reduced-precision calculation of YCbCr-to-RGB introduced
        // to make sure the code produces the same results in both SIMD and scalar
#define jpeg__float2fixed(x)  (((int) ((x) * 4096.0f + 0.5f)) << 8)
        static void jpeg__YCbCr_to_RGB_row(uint8_t* out, const uint8_t* y, const uint8_t* pcb, const uint8_t* pcr, int count, int step)
        {
            int i;
            for (i = 0; i < count; ++i) {
                int y_fixed = (y[i] << 20) + (1 << 19); // rounding
                int r, g, b;
                int cr = pcr[i] - 128;
                int cb = pcb[i] - 128;
                r = y_fixed + cr * jpeg__float2fixed(1.40200f);
                g = y_fixed + (cr * -jpeg__float2fixed(0.71414f)) + ((cb * -jpeg__float2fixed(0.34414f)) & 0xffff0000);
                b = y_fixed + cb * jpeg__float2fixed(1.77200f);
                r >>= 20;
                g >>= 20;
                b >>= 20;
                if ((unsigned)r > 255) { if (r < 0) r = 0; else r = 255; }
                if ((unsigned)g > 255) { if (g < 0) g = 0; else g = 255; }
                if ((unsigned)b > 255) { if (b < 0) b = 0; else b = 255; }
                out[0] = (uint8_t)r;
                out[1] = (uint8_t)g;
                out[2] = (uint8_t)b;
                out[3] = 255;
                out += step;
            }
        }

        // set up the kernels
        static void jpeg__setup_jpeg(JpegContext* j)
        {
            j->idct_block_kernel = jpeg__idct_block;
            j->YCbCr_to_RGB_kernel = jpeg__YCbCr_to_RGB_row;
            j->resample_row_hv_2_kernel = jpeg__resample_row_hv_2;
        }

        typedef struct
        {
            ResampleRowPtr resample;
            uint8_t* line0, * line1;
            int hs, vs;   // expansion factor in each axis
            int w_lores; // horizontal pixels pre-expansion
            int ystep;   // how far through vertical expansion we are
            int ypos;    // which pre-expansion row we're on
        } jpeg__resample;

        // fast 0..255 * 0..255 => 0..255 rounded multiplication
        static uint8_t jpeg__blinn_8x8(uint8_t x, uint8_t y)
        {
            unsigned int t = x * y + 128;
            return (uint8_t)((t + (t >> 8)) >> 8);
        }

        static int load_jpeg_image(JpegContext* z, int* out_x, int* out_y, int* comp, int req_comp)
        {
            int n, decode_n, is_rgb;
            z->img_n = 0; // make jpeg__cleanup_jpeg safe

            // validate req_comp
            if (req_comp < 0 || req_comp > 4) return JpegLoadError("bad req_comp", "Internal error");

            // load a jpeg image from whichever source, but leave in YCbCr format
            if (!jpeg__decode_jpeg_image(z))
                return 0;

            // determine actual number of components to generate
            n = req_comp ? req_comp : z->img_n >= 3 ? 3 : 1;

            is_rgb = z->img_n == 3 && (z->rgb == 3 || (z->app14_color_transform == 0 && !z->jfif));

            if (z->img_n == 3 && n < 3 && !is_rgb)
                decode_n = 1;
            else
                decode_n = z->img_n;

            // resample and color-convert
            {
                int k;
                unsigned int i, j;
                uint8_t* coutput[4] = { NULL, NULL, NULL, NULL };

                jpeg__resample res_comp[4];

                for (k = 0; k < decode_n; ++k) 
                {
                    jpeg__resample* r = &res_comp[k];

                    // allocate line buffer big enough for upsampling off the edges
                    // with upsample factor of 4
                    z->img_comp[k].bufL.Resize(z->img_x + 3);
                    if (z->img_comp[k].bufL.Empty()) 
                        return JpegLoadError("outofmem", "Out of memory");

                    r->hs = z->img_h_max / z->img_comp[k].h;
                    r->vs = z->img_v_max / z->img_comp[k].v;
                    r->ystep = r->vs >> 1;
                    r->w_lores = (z->img_x + r->hs - 1) / r->hs;
                    r->ypos = 0;
                    r->line0 = r->line1 = z->img_comp[k].data;

                    if (r->hs == 1 && r->vs == 1) r->resample = resample_row_1;
                    else if (r->hs == 1 && r->vs == 2) r->resample = jpeg__resample_row_v_2;
                    else if (r->hs == 2 && r->vs == 1) r->resample = jpeg__resample_row_h_2;
                    else if (r->hs == 2 && r->vs == 2) r->resample = z->resample_row_hv_2_kernel;
                    else                               r->resample = jpeg__resample_row_generic;
                }

                // can't error after this so, this is safe
                z->out.Resize(n * z->img_x * z->img_y + 1);
                if (z->out.Empty()) return JpegLoadError("outofmem", "Out of memory");

                // now go ahead and resample
                for (j = 0; j < z->img_y; ++j) {
                    uint8_t* out = z->out.data + n * z->img_x * j;
                    for (k = 0; k < decode_n; ++k) {
                        jpeg__resample* r = &res_comp[k];
                        int y_bot = r->ystep >= (r->vs >> 1);
                        coutput[k] = r->resample(z->img_comp[k].bufL.data,
                            y_bot ? r->line1 : r->line0,
                            y_bot ? r->line0 : r->line1,
                            r->w_lores, r->hs);
                        if (++r->ystep >= r->vs) {
                            r->ystep = 0;
                            r->line0 = r->line1;
                            if (++r->ypos < z->img_comp[k].y)
                                r->line1 += z->img_comp[k].w2;
                        }
                    }
                    if (n >= 3) {
                        uint8_t* y = coutput[0];
                        if (z->img_n == 3) {
                            if (is_rgb) {
                                for (i = 0; i < z->img_x; ++i) {
                                    out[0] = y[i];
                                    out[1] = coutput[1][i];
                                    out[2] = coutput[2][i];
                                    out[3] = 255;
                                    out += n;
                                }
                            }
                            else {
                                z->YCbCr_to_RGB_kernel(out, y, coutput[1], coutput[2], z->img_x, n);
                            }
                        }
                        else if (z->img_n == 4) {
                            if (z->app14_color_transform == 0) { // CMYK
                                for (i = 0; i < z->img_x; ++i) {
                                    uint8_t m = coutput[3][i];
                                    out[0] = jpeg__blinn_8x8(coutput[0][i], m);
                                    out[1] = jpeg__blinn_8x8(coutput[1][i], m);
                                    out[2] = jpeg__blinn_8x8(coutput[2][i], m);
                                    out[3] = 255;
                                    out += n;
                                }
                            }
                            else if (z->app14_color_transform == 2) { // YCCK
                                z->YCbCr_to_RGB_kernel(out, y, coutput[1], coutput[2], z->img_x, n);
                                for (i = 0; i < z->img_x; ++i) {
                                    uint8_t m = coutput[3][i];
                                    out[0] = jpeg__blinn_8x8(255 - out[0], m);
                                    out[1] = jpeg__blinn_8x8(255 - out[1], m);
                                    out[2] = jpeg__blinn_8x8(255 - out[2], m);
                                    out += n;
                                }
                            }
                            else { // YCbCr + alpha?  Ignore the fourth channel for now
                                z->YCbCr_to_RGB_kernel(out, y, coutput[1], coutput[2], z->img_x, n);
                            }
                        }
                        else
                            for (i = 0; i < z->img_x; ++i) {
                                out[0] = out[1] = out[2] = y[i];
                                out[3] = 255; // not used if n==3
                                out += n;
                            }
                    }
                    else {
                        if (is_rgb) 
                        {
                            if (n == 1)
                                for (i = 0; i < z->img_x; ++i)
                                    *out++ = jpeg__compute_y(coutput[0][i], coutput[1][i], coutput[2][i]);
                            else {
                                for (i = 0; i < z->img_x; ++i, out += 2) {
                                    out[0] = jpeg__compute_y(coutput[0][i], coutput[1][i], coutput[2][i]);
                                    out[1] = 255;
                                }
                            }
                        }
                        else if (z->img_n == 4 && z->app14_color_transform == 0) {
                            for (i = 0; i < z->img_x; ++i) {
                                uint8_t m = coutput[3][i];
                                uint8_t r = jpeg__blinn_8x8(coutput[0][i], m);
                                uint8_t g = jpeg__blinn_8x8(coutput[1][i], m);
                                uint8_t b = jpeg__blinn_8x8(coutput[2][i], m);
                                out[0] = jpeg__compute_y(r, g, b);
                                out[1] = 255;
                                out += n;
                            }
                        }
                        else if (z->img_n == 4 && z->app14_color_transform == 2) {
                            for (i = 0; i < z->img_x; ++i) {
                                out[0] = jpeg__blinn_8x8(255 - coutput[0][i], coutput[3][i]);
                                out[1] = 255;
                                out += n;
                            }
                        }
                        else {
                            uint8_t* y = coutput[0];
                            if (n == 1)
                                for (i = 0; i < z->img_x; ++i) out[i] = y[i];
                            else
                                for (i = 0; i < z->img_x; ++i) { *out++ = y[i]; *out++ = 255; }
                        }
                    }
                }
                *out_x = z->img_x;
                *out_y = z->img_y;
                if (comp) *comp = z->img_n >= 3 ? 3 : 1; // report original components, not output
                return 1;
            }
        }

        //---------------------------------------------------------------------

        ImageJpegLoader::ImageJpegLoader(const ImageLoaderParam& param)
            : ImageLoader(param)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatRgb24;
        }

        bool ImageJpegLoader::FromStream()
        {
            int x, y, comp;
            JpegContext j;
            j.stream = &_stream;
            jpeg__setup_jpeg(&j);
            if (load_jpeg_image(&j, &x, &y, &comp, 4))
            {
                size_t stride = 4 * x;
                _image.Recreate(x, y, (Image::Format)_param.format);
                switch (_param.format)
                {
                case SimdPixelFormatGray8:
                    Base::RgbaToGray(j.out.data, x, y, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatBgr24:
                    Base::BgraToRgb(j.out.data, x, y, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatBgra32:
                    Base::BgraToRgba(j.out.data, x, y, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatRgb24:
                    Base::BgraToBgr(j.out.data, x, y, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatRgba32:
                    Base::Copy(j.out.data, stride, x, y, 4, _image.data, _image.stride);
                    break;
                default: 
                    break;
                }
                return true;
            }
            return false;
        }
    }
}
