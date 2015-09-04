/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
                const int size;
                float * cos, * sin;
                int * index;
                float * value;

                Buffer(size_t width, size_t quantization)
                    : size((int)quantization/2)
                {
                    _p = Allocate(width*(sizeof(int) + sizeof(float)) + sizeof(float)*2*size);
                    index = (int*)_p;
                    value = (float*)index + width;
                    cos = value + width;
                    sin = cos + size;
                    for(int i = 0; i < size; ++i)
                    {
                        cos[i] = (float)::cos(i*M_PI/size);
                        sin[i] = (float)::sin(i*M_PI/size);
                    }
                }

                ~Buffer()
                {
                    Free(_p);
                }

            private:
                void *_p;
            }; 
        }

        void AddRowToHistograms(int * indexes, float * values, size_t row, size_t width, size_t height, size_t cellX, size_t cellY, size_t quantization, float * histograms)
        {
            int blockX = int(width/cellX);
            int blockY = int(height/cellY);
            int blockStride = int(quantization*blockX);

            float yp = ((float)row + 0.5f)/(float)cellY - 0.5f;
            int iyp = (int)floor(yp);
            float vy0 = yp - iyp;
            float vy1 = 1.0f - vy0;

            size_t noseEnd = cellX/2;
            size_t bodyEnd = width - cellX/2;

            if(iyp < 0)
            {
                float * h = histograms + (iyp + 1)*blockStride;
                for (size_t col = 1; col < width - 1; ++col) 
                {
                    float value = values[col];
                    int index = indexes[col];

                    float xp = ((float)col + 0.5f)/(float)cellX - 0.5f;
                    int ixp = (int)floor(xp);
                    float vx0 = xp - ixp;
                    float vx1 = 1.0f - vx0;

                    if (ixp >= 0) 
                        h[ixp*quantization + index] += vx1*vy0*value;
                    if (ixp + 1 < blockX) 
                        h[(ixp + 1)*quantization + index] += vx0*vy0*value;
                }
            }
            else if(iyp + 1 == blockY)
            {
                float * h = histograms + iyp*blockStride;
                for (size_t col = 1; col < width - 1; ++col) 
                {
                    float value = values[col];
                    int index = indexes[col];

                    float xp = ((float)col + 0.5f)/(float)cellX - 0.5f;
                    int ixp = (int)floor(xp);
                    float vx0 = xp - ixp;
                    float vx1 = 1.0f - vx0;

                    if (ixp >= 0) 
                        h[ixp*quantization + index] += vx1*vy1*value;
                    if (ixp + 1 < blockX) 
                        h[(ixp + 1)*quantization + index] += vx0*vy1*value;
                }
            }
            else
            {
                float * h0 = histograms + iyp*blockStride;
                float * h1 = histograms + (iyp + 1)*blockStride;
                size_t col = 1;
                for (; col < noseEnd; ++col) 
                {
                    float value = values[col];
                    int index = indexes[col];

                    float xp = ((float)col + 0.5f)/(float)cellX - 0.5f;
                    int ixp = (int)floor(xp);
                    float vx0 = xp - ixp;

                    h0[(ixp + 1)*quantization + index] += vx0*vy1*value;
                    h1[(ixp + 1)*quantization + index] += vx0*vy0*value;
                }

                for (; col < bodyEnd; ++col) 
                {
                    float value = values[col];
                    int index = indexes[col];

                    float xp = ((float)col + 0.5f)/(float)cellX - 0.5f;
                    int ixp = (int)floor(xp);
                    float vx0 = xp - ixp;
                    float vx1 = 1.0f - vx0;

                    h0[ixp*quantization + index] += vx1*vy1*value;
                    h1[ixp*quantization + index] += vx1*vy0*value;
                    h0[(ixp + 1)*quantization + index] += vx0*vy1*value;
                    h1[(ixp + 1)*quantization + index] += vx0*vy0*value;
                }

                for (; col < width - 1; ++col) 
                {
                    float value = values[col];
                    int index = indexes[col];

                    float xp = ((float)col + 0.5f)/(float)cellX - 0.5f;
                    int ixp = (int)floor(xp);
                    float vx0 = xp - ixp;
                    float vx1 = 1.0f - vx0;

                    h0[ixp*quantization + index] += vx1*vy1*value;
                    h1[ixp*quantization + index] += vx1*vy0*value;
                }
            }
        }

        void HogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height, 
            size_t cellX, size_t cellY, size_t quantization, float * histograms)
        {
            assert(width%cellX == 0 && height%cellY == 0 && quantization%2 == 0);

            Buffer buffer(width, quantization);

            memset(histograms, 0, quantization*(width/cellX)*(height/cellY)*sizeof(float));

            for (size_t row = 1; row < height - 1; ++row) 
            {
                const uint8_t * src1 = src + stride*row;
                const uint8_t * src0 = src1 - stride;
                const uint8_t * src2 = src1 + stride;

                for (size_t col = 1; col < width - 1; ++col) 
                {
                    float dy = (float)(src2[col] - src0[col]);
                    float dx = (float)(src1[col + 1] - src1[col - 1]);
                    float value = (float)::sqrt(dx*dx + dy*dy);

                    float bestDot = 0;
                    int index = 0;
                    for (int direction = 0; direction < buffer.size; direction++) 
                    {
                        float dot = buffer.cos[direction]*dx + buffer.sin[direction]*dy;
                        if (dot > bestDot) 
                        {
                            bestDot = dot;
                            index = direction;
                        } 
                        else if (-dot > bestDot) 
                        {
                            bestDot = -dot;
                            index = direction + buffer.size;
                        }
                    }

                    buffer.value[col] = value;
                    buffer.index[col] = index;
                }

                AddRowToHistograms(buffer.index, buffer.value, row, width, height, cellX, cellY, quantization, histograms);
            }
        }
    }
}
