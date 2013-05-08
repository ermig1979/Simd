/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#ifndef __Simd_h__
#define __Simd_h__

#include "Simd/SimdVersion.h"
#include "Simd/SimdConfig.h"
#include "Simd/SimdDefs.h"
#include "Simd/SimdTypes.h"
#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdInit.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdView.h"

#include "Simd/SimdAbsDifferenceSum.h"
#include "Simd/SimdAbsGradientSaturatedSum.h"
#include "Simd/SimdAbsSecondDerivativeHistogram.h"
#include "Simd/SimdAverage.h"
#include "Simd/SimdBgraToBgr.h"
#include "Simd/SimdBgraToGray.h"
#include "Simd/SimdBgrToBgra.h"
#include "Simd/SimdBgrToGray.h"
#include "Simd/SimdBinarization.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdCrc32.h"
#include "Simd/SimdDeinterleaveUv.h"
#include "Simd/SimdGaussianBlur3x3.h"
#include "Simd/SimdInterleaveBgra.h"
#include "Simd/SimdMedianFilterSquare3x3.h"
#include "Simd/SimdMedianFilterSquare5x5.h"
#include "Simd/SimdReduceGray2x2.h"
#include "Simd/SimdReduceGray3x3.h"
#include "Simd/SimdReduceGray4x4.h"
#include "Simd/SimdReduceGray5x5.h"
#include "Simd/SimdResizeBilinear.h"
#include "Simd/SimdShiftBilinear.h"
#include "Simd/SimdSquaredDifferenceSum.h"
#include "Simd/SimdStatistic.h"
#include "Simd/SimdYuvToBgra.h"
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdYuvToHue.h"

#endif//__Simd_h__
