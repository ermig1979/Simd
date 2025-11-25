/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#ifndef __SimdShiftDetector_h__
#define __SimdShiftDetector_h__

#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
        struct Implementation;

        struct ShiftDetector : public Deletable
        {
            ShiftDetector(size_t bkgWidth, size_t bkgHeight, size_t levelCount, SimdShiftDetectorTextureType textureType, SimdShiftDetectorDifferenceType differenceType);

            virtual ~ShiftDetector();

            void SetBackground(const uint8_t* bkg, size_t bkgStride, SimdBool makeCopy);

            SimdBool Estimate(const uint8_t* curr, size_t currStride, size_t currWidth, size_t currHeight,
                size_t initShiftX, size_t initShiftY, size_t maxShiftX, size_t maxShiftY, const double* hiddenAreaPenalty, ptrdiff_t regionAreaMin);

            void GetShift(ptrdiff_t* shift, double* refinedShift, double* stability, double* correlation);

        private:
            Implementation* _implementation;
        };
    }
}
#endif
