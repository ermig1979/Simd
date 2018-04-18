/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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

namespace Simd
{
    namespace Base
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t) * 3 * width);
                    isc0 = (uint16_t*)_p;
                    isc1 = isc0 + width;
                    iscp = isc1 + width;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * isc0;
                uint16_t * isc1;
                uint16_t * iscp;
            private:
                void *_p;
            };
        }


        /**************************************************************************************************
        *  The Burt & Adelson Reduce operation. This function use 2-D version of algorithm;
        *
        *  Reference:
        *  Frederick M. Waltz and John W.V. Miller. An efficient algorithm for Gaussian blur using
        *  finite-state machines.
        *  SPIE Conf. on Machine Vision Systems for Inspection and Metrology VII. November 1998.
        *
        *
        *  2-D explanation:
        *
        *  src image pixels:   A  B  C  D  E       dst image pixels:   a     b     c
        *                      F  G  H  I  J
        *                      K  L  M  N  O                           d     e     f
        *                      P  Q  R  S  T
        *                      U  V  W  X  Y                           g     h     i
        *
        *  Algorithm visits all src image pixels from left to right and top to bottom.
        *  When visiting src pixel Y, the value of e will be written to the dst image.
        *
        *  State variables before visiting Y:
        *  sr0 = W
        *  sr1 = U + 4V
        *  srp = 4X
        *  sc0[2] = K + 4L + 6M + 4N + O
        *  sc1[2] = (A + 4B + 6C + 4D + E) + 4*(F + 4G + 6H + 4I + J)
        *  scp[2] = 4*(P + 4Q + 6R + 4S + T)
        *
        *  State variables after visiting Y:
        *  sr0 = Y
        *  sr1 = W + 4X
        *  srp = 4X
        *  sc0[2] = U + 4V + 6W + 4X + Y
        *  sc1[2] = (K + 4L + 6M + 4N + O) + 4*(P + 4Q + 6R + 4S + T)
        *  scp[2] = 4*(P + 4Q + 6R + 4S + T)
        *  e =   1 * (A + 4B + 6C + 4D + E)
        *      + 4 * (F + 4G + 6H + 4I + J)
        *      + 6 * (K + 4L + 6M + 4N + O)
        *      + 4 * (P + 4Q + 6R + 4S + T)
        *      + 1 * (U + 4V + 6W + 4X + Y)
        *
        *  Updates when visiting (even x, even y) source pixel:
        *  (all updates occur in parallel)
        *  sr0 <= current
        *  sr1 <= sr0 + srp
        *  sc0[x] <= sr1 + 6*sr0 + srp + current
        *  sc1[x] <= sc0[x] + scp[x]
        *  dst(-1,-1) <= sc1[x] + 6*sc0[x] + scp + (new sc0[x])
        *
        *  Updates when visiting (odd x, even y) source pixel:
        *  srp <= 4*current
        *
        *  Updates when visiting (even x, odd y) source pixel:
        *  sr0 <= current
        *  sr1 <= sr0 + srp
        *  scp[x] <= 4*(sr1 + 6*sr0 + srp + current)
        *
        *  Updates when visting (odd x, odd y) source pixel:
        *  srp <= 4*current
        **************************************************************************************************/
        template <bool compensation> SIMD_INLINE int DivideBy256(int value);

        template <> SIMD_INLINE int DivideBy256<true>(int value)
        {
            return (value + 128) >> 8;
        }

        template <> SIMD_INLINE int DivideBy256<false>(int value)
        {
            return value >> 8;
        }

        template <bool compensation> void ReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);

            Buffer buffer(dstWidth + 1);

            unsigned short isr0, isr1, isrp;

            const short zeroPixel = 0;

            uint8_t * dy = dst;
            uint8_t * dx = dy;
            const uint8_t * sy = src;
            const uint8_t * sx = sy;

            bool evenY = true;
            bool evenX = true;
            size_t srcy = 0;
            size_t srcx = 0;
            size_t dstx = 0;

            // First row
            {
                isr0 = *sy;
                isr1 = zeroPixel;
                isrp = (unsigned short)(*sy) * 4;

                // Main pixels in first row
                for (sx = sy, evenX = true, srcx = 0, dstx = 0; srcx < srcWidth; ++srcx, ++sx)
                {
                    unsigned short icurrent(*sx);

                    if (evenX)
                    {
                        buffer.isc0[dstx] = isr1 + 6 * isr0 + isrp + icurrent;
                        buffer.isc1[dstx] = 5 * buffer.isc0[dstx];
                        isr1 = isr0 + isrp;
                        isr0 = icurrent;
                    }
                    else
                    {
                        isrp = icurrent * 4;
                        ++dstx;
                    }
                    evenX = !evenX;
                }

                // Last entries in first row
                if (!evenX)
                {
                    // previous srcx was even
                    ++dstx;
                    buffer.isc0[dstx] = isr1 + 11 * isr0;
                    buffer.isc1[dstx] = 5 * buffer.isc0[dstx];
                }
                else
                {
                    // previous srcx was odd
                    buffer.isc0[dstx] = isr1 + 6 * isr0 + isrp + (isrp >> 2);
                    buffer.isc1[dstx] = 5 * buffer.isc0[dstx];
                }
            }
            sy += srcStride;

            // Main Rows
            {
                for (evenY = false, srcy = 1; srcy < srcHeight; ++srcy, sy += srcStride)
                {
                    isr0 = (unsigned short)(*sy);
                    isr1 = zeroPixel;
                    isrp = (unsigned short)(*sy) * 4;

                    if (evenY)
                    {
                        // Even-numbered row
                        // First entry in row
                        sx = sy;
                        isr1 = isr0 + isrp;
                        isr0 = (unsigned short)(*sx);
                        ++sx;
                        dx = dy;

                        unsigned short * p_isc0 = buffer.isc0;
                        unsigned short * p_isc1 = buffer.isc1;
                        unsigned short * p_iscp = buffer.iscp;

                        // Main entries in row
                        for (evenX = false, srcx = 1, dstx = 0; srcx < (srcWidth - 1); srcx += 2, ++sx)
                        {
                            p_isc0++;
                            p_isc1++;
                            p_iscp++;

                            unsigned short icurrent = (unsigned short)(*sx);

                            isrp = icurrent * 4;
                            icurrent = (unsigned short)(*(++sx));

                            unsigned short ip;
                            ip = *p_isc1 + 6 * (*p_isc0) + *p_iscp;
                            *p_isc1 = *p_isc0 + *p_iscp;
                            *p_isc0 = isr1 + 6 * isr0 + isrp + icurrent;
                            isr1 = isr0 + isrp;
                            isr0 = icurrent;
                            ip = ip + *p_isc0;
                            *dx = DivideBy256<compensation>(ip);
                            ++dx;
                        }
                        dstx += p_isc0 - buffer.isc0;

                        //doing the last operation due to even number of operations in previous cycle
                        if (!(srcWidth & 1))
                        {
                            unsigned short icurrent = (unsigned short)(*sx);
                            isrp = icurrent * 4;
                            ++dstx;
                            evenX = !evenX;
                            ++sx;
                        }

                        // Last entries in row
                        if (!evenX)
                        {
                            // previous srcx was even
                            ++dstx;

                            unsigned short ip;
                            ip = buffer.isc1[dstx] + 6 * buffer.isc0[dstx] + buffer.iscp[dstx];
                            buffer.isc1[dstx] = buffer.isc0[dstx] + buffer.iscp[dstx];
                            buffer.isc0[dstx] = isr1 + 11 * isr0;
                            ip = ip + buffer.isc0[dstx];
                            *dx = DivideBy256<compensation>(ip);
                        }
                        else
                        {
                            // Previous srcx was odd
                            unsigned short ip;
                            ip = buffer.isc1[dstx] + 6 * buffer.isc0[dstx] + buffer.iscp[dstx];
                            buffer.isc1[dstx] = buffer.isc0[dstx] + buffer.iscp[dstx];
                            buffer.isc0[dstx] = isr1 + 6 * isr0 + isrp + (isrp >> 2);
                            ip = ip + buffer.isc0[dstx];
                            *dx = DivideBy256<compensation>(ip);
                        }

                        dy += dstStride;
                    }
                    else
                    {
                        // First entry in odd-numbered row
                        sx = sy;
                        isr1 = isr0 + isrp;
                        isr0 = (unsigned short)(*sx);
                        ++sx;

                        // Main entries in odd-numbered row
                        unsigned short * p_iscp = buffer.iscp;

                        for (evenX = false, srcx = 1, dstx = 0; srcx < (srcWidth - 1); srcx += 2, ++sx)
                        {
                            unsigned short icurrent = (unsigned short)(*sx);
                            isrp = icurrent * 4;

                            p_iscp++;

                            icurrent = (unsigned short)(*(++sx));

                            *p_iscp = (isr1 + 6 * isr0 + isrp + icurrent) * 4;
                            isr1 = isr0 + isrp;
                            isr0 = icurrent;
                        }
                        dstx += p_iscp - buffer.iscp;

                        //doing the last operation due to even number of operations in previous cycle
                        if (!(srcWidth & 1))
                        {
                            unsigned short icurrent = (unsigned short)(*sx);
                            isrp = icurrent * 4;
                            ++dstx;
                            evenX = !evenX;
                            ++sx;
                        }

                        // Last entries in row
                        if (!evenX)
                        {
                            // previous srcx was even
                            ++dstx;
                            buffer.iscp[dstx] = (isr1 + 11 * isr0) * 4;
                        }
                        else
                        {
                            buffer.iscp[dstx] = (isr1 + 6 * isr0 + isrp + (isrp >> 2)) * 4;
                        }
                    }
                    evenY = !evenY;
                }
            }

            // Last Rows
            {
                if (!evenY)
                {
                    for (dstx = 1, dx = dy; dstx < (dstWidth + 1); ++dstx, ++dx)
                        *dx = DivideBy256<compensation>(buffer.isc1[dstx] + 11 * buffer.isc0[dstx]);
                }
                else
                {
                    for (dstx = 1, dx = dy; dstx < (dstWidth + 1); ++dstx, ++dx)
                        *dx = DivideBy256<compensation>(buffer.isc1[dstx] + 6 * buffer.isc0[dstx] + buffer.iscp[dstx] + (buffer.iscp[dstx] >> 2));
                }
            }
        }

        void ReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray5x5<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray5x5<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
}
