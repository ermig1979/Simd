/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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

/** \file SimdTypes.h
* This file contains description of basic types of Simd Library.
*/

#ifndef __SimdTypes_h__
#define __SimdTypes_h__

#include <stddef.h>

#if defined(__GNUC__) || (defined(_MSC_VER) && (_MSC_VER >= 1600))
#include <stdint.h>
#else
#  if (_MSC_VER < 1300)
typedef signed char       int8_t;
typedef signed short      int16_t;
typedef signed int        int32_t;
typedef unsigned char     uint8_t;
typedef unsigned short    uint16_t;
typedef unsigned int      uint32_t;
#  else
typedef signed __int8     int8_t;
typedef signed __int16    int16_t;
typedef signed __int32    int32_t;
typedef unsigned __int8   uint8_t;
typedef unsigned __int16  uint16_t;
typedef unsigned __int32  uint32_t;
#  endif
typedef signed __int64    int64_t;
typedef unsigned __int64  uint64_t;
#endif

 /**
 * \enum SimdCompareType
 *
 * Describes types of compare operation.
 * Operation compare(a, b) is 
 */
typedef enum SimdCompareType
{
    /// equal to: a == b
    SimdCompareEqual, 
    /// equal to: a != b          
    SimdCompareNotEqual,   
    /// equal to: a > b     
    SimdCompareGreater,        
    /// equal to: a >= b 
    SimdCompareGreaterOrEqual,  
    /// equal to: a < b
    SimdCompareLesser,       
    /// equal to: a <= b   
    SimdCompareLesserOrEqual,   
} SimdCompareType;

 /**
 * \enum SimdOperationBinary8uType
 *
 * Describes types of binary operation between two images performed by function ::SimdOperationBinary8u.
 * Images must have the same format (unsigned 8-bit integer for every channel).
 */
typedef enum SimdOperationBinary8uType
{
    /// Computes the average value for every channel of every point of two images. Average(a, b) = (a + b + 1)/2.
    SimdOperationBinary8uAverage,
    /// Computes the bitwise AND between two images.
    SimdOperationBinary8uAnd,
    /// Computes maximal value for every channel of every point of two images.
    SimdOperationBinary8uMaximum,
    ///Subtracts unsigned 8-bit integer b from unsigned 8-bit integer a and saturates (for every channel of every point of the images).
    SimdOperationBinary8uSaturatedSubtraction,
} SimdOperationBinary8uType;

 /**
 * \enum SimdOperationBinary16iType
 *
 * Describes types of binary operation between two images performed by function ::SimdOperationBinary16i.
 * Images must have ::SimdPixelFormatInt16 pixel format (signed 16-bit integer for every point).
 */
typedef enum SimdOperationBinary16iType
{
    /// Perform addition of two images for every point. 
    SimdOperationBinary16iAddition,
} SimdOperationBinary16iType;

 /**
 * \enum SimdPixelFormatType
 *
 * Describes pixel format types of an image.
 */
typedef enum SimdPixelFormatType
{
    /// An undefined pixel format.
    SimdPixelFormatNone = 0,
    /// A 8-bit gray pixel format.
    SimdPixelFormatGray8,
    /// A 16-bit (2 8-bit channels) pixel format (UV plane of NV12 pixel format).
    SimdPixelFormatUv16,
    /// A 24-bit (3 8-bit channels) BGR (Blue, Green, Red) pixel format.
    SimdPixelFormatBgr24,
    /// A 32-bit (4 8-bit channels) BGRA (Blue, Green, Red, Alpha) pixel format.
    SimdPixelFormatBgra32,
    /// A single channel 16-bit integer pixel format.
    SimdPixelFormatInt16,
    /// A single channel 32-bit integer pixel format.
    SimdPixelFormatInt32,
    /// A single channel 64-bit integer pixel format.
    SimdPixelFormatInt64,
    /// A single channel 32-bit float point pixel format.
    SimdPixelFormatFloat,
    /// A single channel 64-bit float point pixel format.
    SimdPixelFormatDouble,
    /// A 8-bit Bayer pixel format (GRBG).
    SimdPixelFormatBayerGrbg,
    /// A 8-bit Bayer pixel format (GBRG).
    SimdPixelFormatBayerGbrg,
    /// A 8-bit Bayer pixel format (RGGB).
    SimdPixelFormatBayerRggb,
    /// A 8-bit Bayer pixel format (BGGR).
    SimdPixelFormatBayerBggr,
} SimdPixelFormatType;

#endif//__SimdTypes_h__
