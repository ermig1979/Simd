/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar,
*               2014-2015 Antonenka Mikhail.
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

#ifndef __SimdLib_h__
#define __SimdLib_h__

#include <stddef.h>

#if defined(_MSC_VER)

#define SIMD_INLINE __forceinline

#elif defined(__GNUC__)

#define SIMD_INLINE inline __attribute__ ((always_inline))

#else

#error This platform is unsupported!

#endif

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

/*! @ingroup c_types
    Describes types of compare operation.
    Operation compare(a, b) is 
*/
typedef enum 
{
    /*! equal to: a == b */
    SimdCompareEqual,
    /*! equal to: a != b */          
    SimdCompareNotEqual,   
    /*! equal to: a > b */    
    SimdCompareGreater,        
    /*! equal to: a >= b */
    SimdCompareGreaterOrEqual,  
    /*! equal to: a < b */
    SimdCompareLesser,       
    /*! equal to: a <= b */   
    SimdCompareLesserOrEqual,   
} SimdCompareType;

/*! @ingroup c_types
    Describes types of binary operation between two images performed by function ::SimdOperationBinary8u.
    Images must have the same format (unsigned 8-bit integer for every channel).
*/
typedef enum
{
    /*! Computes the average value for every channel of every point of two images. \n Average(a, b) = (a + b + 1)/2. */
    SimdOperationBinary8uAverage,
    /*! Computes the bitwise AND between two images. */
    SimdOperationBinary8uAnd,
    /*! Computes maximal value for every channel of every point of two images. */
    SimdOperationBinary8uMaximum,
    /*!Subtracts unsigned 8-bit integer b from unsigned 8-bit integer a and saturates (for every channel of every point of the images). */
    SimdOperationBinary8uSaturatedSubtraction,
    /*!Adds unsigned 8-bit integer b from unsigned 8-bit integer a and saturates (for every channel of every point of the images). */
    SimdOperationBinary8uSaturatedAddition,
} SimdOperationBinary8uType;

/*! @ingroup c_types
    Describes types of binary operation between two images performed by function ::SimdOperationBinary16i.
    Images must have ::SimdPixelFormatInt16 pixel format (signed 16-bit integer for every point).
*/
typedef enum
{
    /*! Perform addition of two images for every point.  */
    SimdOperationBinary16iAddition,
} SimdOperationBinary16iType;

/*! @ingroup c_types
    Describes pixel format types of an image.
    In particular this type is used in functions ::SimdBayerToBgr, ::SimdBayerToBgra, ::SimdBgraToBayer and ::SimdBgrToBayer.
*/
typedef enum
{
    /*! An undefined pixel format. */
    SimdPixelFormatNone = 0,
    /*! A 8-bit gray pixel format. */
    SimdPixelFormatGray8,
    /*! A 16-bit (2 8-bit channels) pixel format (UV plane of NV12 pixel format). */
    SimdPixelFormatUv16,
    /*! A 24-bit (3 8-bit channels) BGR (Blue, Green, Red) pixel format. */
    SimdPixelFormatBgr24,
    /*! A 32-bit (4 8-bit channels) BGRA (Blue, Green, Red, Alpha) pixel format. */
    SimdPixelFormatBgra32,
    /*! A single channel 16-bit integer pixel format. */
    SimdPixelFormatInt16,
    /*! A single channel 32-bit integer pixel format. */
    SimdPixelFormatInt32,
    /*! A single channel 64-bit integer pixel format. */
    SimdPixelFormatInt64,
    /*! A single channel 32-bit float point pixel format. */
    SimdPixelFormatFloat,
    /*! A single channel 64-bit float point pixel format. */
    SimdPixelFormatDouble,
    /*! A 8-bit Bayer pixel format (GRBG). */
    SimdPixelFormatBayerGrbg,
    /*! A 8-bit Bayer pixel format (GBRG). */
    SimdPixelFormatBayerGbrg,
    /*! A 8-bit Bayer pixel format (RGGB). */
    SimdPixelFormatBayerRggb,
    /*! A 8-bit Bayer pixel format (BGGR). */
    SimdPixelFormatBayerBggr,
    /*! A 24-bit (3 8-bit channels) HSV (Hue, Saturation, Value) pixel format. */
    SimdPixelFormatHsv24,
    /*! A 24-bit (3 8-bit channels) HSL (Hue, Saturation, Lightness) pixel format. */
    SimdPixelFormatHsl24,
} SimdPixelFormatType;

#if defined(WIN32) && !defined(SIMD_STATIC)
#  ifdef SIMD_EXPORTS
#    define SIMD_API __declspec(dllexport)
#  else//SIMD_EXPORTS
#    define SIMD_API __declspec(dllimport)
#  endif//SIMD_EXPORTS
#else //WIN32
#    define SIMD_API
#endif//WIN32

#ifdef __cplusplus
extern "C"
{
#endif//__cplusplus

    /*! @ingroup info

        \fn const char * SimdVersion();

        \short Gets version of Simd Library.

        \return string with version of Simd Library (major version number, minor version number, release number, number of SVN's commits).
    */
    SIMD_API const char * SimdVersion();

    /*! @ingroup memory

        \fn void * SimdAllocate(size_t size, size_t align);

        \short Allocates aligned memory block.

        \note The memory allocated by this function is must be deleted by function ::SimdFree.

        \param [in] size - a size of memory block.
        \param [in] align - a required alignment of memory block.

        \return a pointer to allocated memory.
    */
    SIMD_API void * SimdAllocate(size_t size, size_t align);

    /*! @ingroup memory

        \fn void SimdFree(void * ptr);

        \short Frees aligned memory block.

        \note This function frees a memory allocated by function ::SimdAllocate.

        \param [in] ptr - a pointer to the memory to be deleted.
    */
    SIMD_API void SimdFree(void * ptr);

    /*! @ingroup memory

        \fn size_t SimdAlign(size_t size, size_t align);

        \short Gets aligned size.

        \param [in] size - an original size.
        \param [in] align - a required alignment.

        \return an aligned size.
    */
    SIMD_API size_t SimdAlign(size_t size, size_t align);

    /*! @ingroup memory

        \fn size_t SimdAlignment();

        \short Gets required alignment for Simd Library.

        \return a required alignment.
    */
    SIMD_API size_t SimdAlignment();

    /*! @ingroup hash

        \fn uint32_t SimdCrc32c(const void * src, size_t size);

        \short Gets 32-bit cyclic redundancy check (CRC32c) for current data.

        Calculation is performed for for polynomial 0x1EDC6F41 (Castagnoli-crc).

        \param [in] src - a pointer to data.
        \param [in] size - a size of the data.
        \return 32-bit cyclic redundancy check (CRC32c).
    */
    SIMD_API uint32_t SimdCrc32c(const void * src, size_t size);

    /*! @ingroup correlation

        \fn void SimdAbsDifferenceSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum);

        \short Gets sum of absolute difference of two gray 8-bit images. 

        Both images must have the same width and height.

        \param [in] a - a pointer to pixels data of first image.
        \param [in] aStride - a row size of first image.
        \param [in] b - a pointer to pixels data of second image.
        \param [in] bStride - a row size of second image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - the result sum of absolute difference of two images.
    */
    SIMD_API void SimdAbsDifferenceSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        size_t width, size_t height, uint64_t * sum);

    /*! @ingroup correlation

        \fn void SimdAbsDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum);

        \short Gets sum of absolute difference of two gray 8-bit images based on gray 8-bit mask. 

        Gets the absolute difference sum for all points when mask[i] == index.
        Both images and mask must have the same width and height.

        \param [in] a - a pointer to pixels data of first image.
        \param [in] aStride - a row size of first image.
        \param [in] b - a pointer to pixels data of second image.
        \param [in] bStride - a row size of second image.
        \param [in] mask - a pointer to pixels data of mask image.
        \param [in] maskStride - a row size of mask image.
        \param [in] index - a mask index.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - the result sum of absolute difference of two images.
    */
    SIMD_API void SimdAbsDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum);

    /*! @ingroup correlation

        \fn void SimdAbsDifferenceSums3x3(const uint8_t * current, size_t currentStride, const uint8_t * background, size_t backgroundStride, size_t width, size_t height, uint64_t * sums);

        \short Gets 9 sums of absolute difference of two gray 8-bit images with various relative shifts in neighborhood 3x3. 

        Both images must have the same width and height. The image height and width must be equal or greater 3.
        The sums are calculated with central part (indent width = 1) of current image and with part of background image with corresponding shift.
        The shifts are lain in the range [-1, 1] for axis x and y.

        \param [in] current - a pointer to pixels data of current image.
        \param [in] currentStride - a row size of the current image.
        \param [in] background - a pointer to pixels data of the background image.
        \param [in] backgroundStride - a row size of the background image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sums - the pointer to buffer with result sums. Buffer size must be equal or greater 9.
    */
    SIMD_API void SimdAbsDifferenceSums3x3(const uint8_t * current, size_t currentStride, const uint8_t * background, size_t backgroundStride,
        size_t width, size_t height, uint64_t * sums);

    /*! @ingroup correlation

        \fn void SimdAbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride, const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums);

        \short Gets 9 sums of absolute difference of two gray 8-bit images with various relative shifts in neighborhood 3x3 based on gray 8-bit mask. 

        Gets the absolute difference sums for all points when mask[i] == index.
        Both images and mask must have the same width and height. The image height and width must be equal or greater 3.
        The sums are calculated with central part (indent width = 1) of current image and with part of background image with corresponding shift.
        The shifts are lain in the range [-1, 1] for axis x and y.

        \param [in] current - a pointer to pixels data of current image.
        \param [in] currentStride - a row size of the current image.
        \param [in] background - a pointer to pixels data of the background image.
        \param [in] backgroundStride - a row size of the background image.
        \param [in] mask - a pointer to pixels data of mask image.
        \param [in] maskStride - a row size of mask image.
        \param [in] index - a mask index.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sums - the pointer to buffer with result sums. Buffer size must be equal or greater 9.
    */
    SIMD_API void SimdAbsDifferenceSums3x3Masked(const uint8_t *current, size_t currentStride, const uint8_t *background, size_t backgroundStride,
        const uint8_t *mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sums);

    /*! @ingroup other_filter

        \fn void SimdAbsGradientSaturatedSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Puts to destination 8-bit gray image saturated sum of absolute gradient for every point of source 8-bit gray image. 

        Both images must have the same width and height.

        For border pixels dst[x, y] = 0, for other pixels: 
        \n dst[x, y] = min(dx[x, y] + dy[x, y], 255), where 
        \n dx[x, y] = abs(src[x + 1, y] - src[x - 1, y]), 
        \n dy[x, y] = abs(src[x, y + 1] - src[x, y - 1]).

        \param [in] src - a pointer to pixels data of source 8-bit gray image.
        \param [in] srcStride - a row size of source image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of destination 8-bit gray image.
        \param [in] dstStride - a row size of destination image.
    */
    SIMD_API void SimdAbsGradientSaturatedSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t * dst, size_t dstStride);

    /*! @ingroup difference_estimation

        \fn void SimdAddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height, const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, uint16_t weight, uint8_t * difference, size_t differenceStride);

        \short Adds feature difference to common difference sum. 

        All images must have the same width, height and format (8-bit gray).

        For every point: 
        \n difference[i] += (weight * excess[i]*excess[i]) >> 16, 
        \n where 
        \n excess[i] = max(lo[i] - value[i], 0) + max(value[i] - hi[i], 0).

        This function is used for difference estimation in algorithm of motion detection.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] lo - a pointer to pixels data of feature lower bound of dynamic background.
        \param [in] loStride - a row size of the lo image.
        \param [in] hi - a pointer to pixels data of feature upper bound of dynamic background.
        \param [in] hiStride - a row size of the hi image.
        \param [in] weight - a current feature weight (unsigned 16-bit value).
        \param [in, out] difference - a pointer to pixels data of image with total difference.
        \param [in] differenceStride - a row size of difference image.
    */
    SIMD_API void SimdAddFeatureDifference(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride,
        uint16_t weight, uint8_t * difference, size_t differenceStride);

    /*! @ingroup drawing

        \fn void SimdAlphaBlending(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, const uint8_t * alpha, size_t alphaStride, uint8_t * dst, size_t dstStride);

        \short Performs alpha blending operation. 

        All images must have the same width and height. Source and destination images must have the same format (8 bit per channel, for example GRAY8, BGR24 or BGRA32). Alpha must be 8-bit gray image.

        For every point: 
        \n dst[i] = (src[i]*alpha[i] + dst[i]*(255 - alpha[i]))/255.

        This function is used for image drawing.

        \param [in] src - a pointer to pixels data of foreground image.
        \param [in] srcStride - a row size of the foreground image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count for foreground and background images (1 <= channelCount <= 4).
        \param [in] alpha - a pointer to pixels data of image with alpha channel.
        \param [in] alphaStride - a row size of the alpha image.
        \param [in, out] dst - a pointer to pixels data of background image.
        \param [in] dstStride - a row size of the background image.
    */
    SIMD_API void SimdAlphaBlending(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
        const uint8_t * alpha, size_t alphaStride, uint8_t * dst, size_t dstStride);

    /*! @ingroup background

        \fn void SimdBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

        \short Performs background update (initial grow, slow mode). 

        All images must have the same width, height and format (8-bit gray). 

        For every point: 
        \n lo[i] -= value[i] < lo[i] ? 1 : 0; 
        \n hi[i] += value[i] > hi[i] ? 1 : 0.

        This function is used for background updating in motion detection algorithm.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] lo - a pointer to pixels data of feature lower bound of dynamic background.
        \param [in] loStride - a row size of the lo image.
        \param [in] hi - a pointer to pixels data of feature upper bound of dynamic background.
        \param [in, out] hiStride - a row size of the hi image.
    */
    SIMD_API void SimdBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

    /*! @ingroup background

        \fn void SimdBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

        \short Performs background update (initial grow, fast mode). 

        All images must have the same width, height and format (8-bit gray). 

        For every point: 
        \n lo[i] = value[i] < lo[i] ? value[i] : lo[i]; 
        \n hi[i] = value[i] > hi[i] ? value[i] : hi[i].

        This function is used for background updating in motion detection algorithm.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] lo - a pointer to pixels data of feature lower bound of dynamic background.
        \param [in] loStride - a row size of the lo image.
        \param [in] hi - a pointer to pixels data of feature upper bound of dynamic background.
        \param [in, out] hiStride - a row size of the hi image.
    */
    SIMD_API void SimdBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

    /*! @ingroup background

        \fn void SimdBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height, const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride, uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride);

        \short Performs collection of background statistic. 

        All images must have the same width, height and format (8-bit gray). 

        Updates background statistic counters for every point: 
        \n loCount[i] += (value[i] < loValue[i] && loCount[i] < 255) ? 1 : 0; 
        \n hiCount[i] += (value[i] > hiValue[i] && hiCount[i] < 255) ? 1 : 0;

        This function is used for background updating in motion detection algorithm.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] loValue - a pointer to pixels data of value of feature lower bound of dynamic background.
        \param [in] loValueStride - a row size of the loValue image.
        \param [in] hiValue - a pointer to pixels data of value of feature upper bound of dynamic background.
        \param [in] hiValueStride - a row size of the hiValue image.
        \param [in, out] loCount - a pointer to pixels data of count of feature lower bound of dynamic background.
        \param [in] loCountStride - a row size of the loCount image.
        \param [in, out] hiCount - a pointer to pixels data of count of feature upper bound of dynamic background.
        \param [in] hiCountStride - a row size of the hiCount image.
    */
    SIMD_API void SimdBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride,
        uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride);

    /*! @ingroup background

        \fn void SimdBackgroundAdjustRange(uint8_t * loCount, size_t loCountStride, size_t width, size_t height, uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride, uint8_t * hiValue, size_t hiValueStride, uint8_t threshold);

        \short Performs adjustment of background range. 

        All images must have the same width, height and format (8-bit gray). 

        Adjusts background range for every point: 
        \n loValue[i] -= (loCount[i] > threshold && loValue[i] > 0) ? 1 : 0;
        \n loValue[i] += (loCount[i] < threshold && loValue[i] < 255) ? 1 : 0; 
        \n loCount[i] = 0;
        \n hiValue[i] += (hiCount[i] > threshold && hiValue[i] < 255) ? 1 : 0;
        \n hiValue[i] -= (hiCount[i] < threshold && hiValue[i] > 0) ? 1 : 0; 
        \n hiCount[i] = 0;

        This function is used for background updating in motion detection algorithm.

        \param [in, out] loCount - a pointer to pixels data of count of feature lower bound of dynamic background.
        \param [in] loCountStride - a row size of the loCount image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] hiCount - a pointer to pixels data of count of feature upper bound of dynamic background.
        \param [in] hiCountStride - a row size of the hiCount image.
        \param [in, out] loValue - a pointer to pixels data of value of feature lower bound of dynamic background.
        \param [in] loValueStride - a row size of the loValue image.
        \param [in, out] hiValue - a pointer to pixels data of value of feature upper bound of dynamic background.
        \param [in] hiValueStride - a row size of the hiValue image.
        \param [in] threshold - a count threshold.
    */
    SIMD_API void SimdBackgroundAdjustRange(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
        uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
        uint8_t * hiValue, size_t hiValueStride, uint8_t threshold);

    /*! @ingroup background

        \fn void SimdBackgroundAdjustRangeMasked(uint8_t * loCount, size_t loCountStride, size_t width, size_t height, uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride, uint8_t * hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride);

        \short Performs adjustment of background range with using adjust range mask. 

        All images must have the same width, height and format (8-bit gray). 

        Adjusts background range for every point when mask[i] != 0 : 
        \n loValue[i] -= (loCount[i] > threshold && loValue[i] > 0) ? 1 : 0;
        \n loValue[i] += (loCount[i] < threshold && loValue[i] < 255) ? 1 : 0; 
        \n loCount[i] = 0;
        \n hiValue[i] += (hiCount[i] > threshold && hiValue[i] < 255) ? 1 : 0;
        \n hiValue[i] -= (hiCount[i] < threshold && hiValue[i] > 0) ? 1 : 0; 
        \n hiCount[i] = 0;

        This function is used for background updating in motion detection algorithm.

        \param [in] loCount - a pointer to pixels data of count of feature lower bound of dynamic background.
        \param [in, out] loCountStride - a row size of the loCount image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] hiCount - a pointer to pixels data of count of feature upper bound of dynamic background.
        \param [in] hiCountStride - a row size of the hiCount image.
        \param [in, out] loValue - a pointer to pixels data of value of feature lower bound of dynamic background.
        \param [in] loValueStride - a row size of the loValue image.
        \param [in, out] hiValue - a pointer to pixels data of value of feature upper bound of dynamic background.
        \param [in] hiValueStride - a row size of the hiValue image.
        \param [in] threshold - a count threshold.
        \param [in] mask - a pointer to pixels data of adjust range mask.
        \param [in] maskStride - a row size of the mask image.
    */
    SIMD_API void SimdBackgroundAdjustRangeMasked(uint8_t * loCount, size_t loCountStride, size_t width, size_t height,
        uint8_t * loValue, size_t loValueStride, uint8_t * hiCount, size_t hiCountStride,
        uint8_t * hiValue, size_t hiValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride);

    /*! @ingroup background

        \fn void SimdBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

        \short Shifts background range. 

        All images must have the same width, height and format (8-bit gray). 

        For every point: 
    \verbatim
    if (value[i] > hi[i])
    {
        lo[i] = min(lo[i] + value[i] - hi[i], 0xFF);
        hi[i] = value[i];
    }
    if (lo[i] > value[i])
    {
        lo[i] = value[i];
        hi[i] = max(hi[i] - lo[i] + value[i], 0);
    }
    \endverbatim

        This function is used for fast background updating in motion detection algorithm.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] lo - a pointer to pixels data of feature lower bound of dynamic background.
        \param [in] loStride - a row size of the lo image.
        \param [in] hi - a pointer to pixels data of feature upper bound of dynamic background.
        \param [in, out] hiStride - a row size of the hi image.
    */
    SIMD_API void SimdBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

    /*! @ingroup background

        \fn void SimdBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride);

        \short Shifts background range with using shift range mask. 

        All images must have the same width, height and format (8-bit gray). 

        For every point when mask[i] != 0 : 
    \verbatim
    if (value[i] > hi[i])
    {
        lo[i] = min(lo[i] + value[i] - hi[i], 0xFF);
        hi[i] = value[i];
    }
    if (lo[i] > value[i])
    {
        lo[i] = value[i];
        hi[i] = max(hi[i] - lo[i] + value[i], 0);
    }
    \endverbatim

        This function is used for fast background updating in motion detection algorithm.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] lo - a pointer to pixels data of feature lower bound of dynamic background.
        \param [in] loStride - a row size of the lo image.
        \param [in] hi - a pointer to pixels data of feature upper bound of dynamic background.
        \param [in, out] hiStride - a row size of the hi image.
        \param [in] mask - a pointer to pixels data of shift range mask.
        \param [in] maskStride - a row size of the mask image.
    */
    SIMD_API void SimdBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride);

    /*! @ingroup background

        \fn void SimdBackgroundInitMask(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride);

        \short Creates background update mask. 

        All images must have the same width, height and format (8-bit gray). 

        For every point when mask[i] == index: 
        \n dst[i] = value; 

        This function is used for background updating in motion detection algorithm.

        \param [in] src - a pointer to pixels data of input mask image.
        \param [in] srcStride - a row size of input mask image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] index - a mask index into input mask.
        \param [in] value - a value to fill the output mask.
        \param [out] dst - a pointer to pixels data of output mask image.
        \param [in] dstStride - a row size of output mask image.
    */
    SIMD_API void SimdBackgroundInitMask(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride);

    /*! @ingroup bayer_conversion

        \fn void SimdBayerToBgr(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgr, size_t bgrStride);

        \short Converts 8-bit Bayer image to 24-bit BGR. 

        All images must have the same width and height. The width and the height must be even.

        \param [in] bayer - a pointer to pixels data of output 8-bit Bayer image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bayerStride - a row size of the bayer image.
        \param [in] bayerFormat - a format of the input bayer image. It can be ::SimdPixelFormatBayerGrbg, ::SimdPixelFormatBayerGbrg, ::SimdPixelFormatBayerRggb or ::SimdPixelFormatBayerBggr.
        \param [out] bgr - a pointer to pixels data of output 24-bit BGR image.
        \param [in] bgrStride - a row size of the bgr image.
    */
    SIMD_API void SimdBayerToBgr(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgr, size_t bgrStride);

    /*! @ingroup bayer_conversion

        \fn void SimdBayerToBgra(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

        \short Converts 8-bit Bayer image to 32-bit BGRA. 

        All images must have the same width and height. The width and the height must be even.

        \param [in] bayer - a pointer to pixels data of output 8-bit Bayer image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bayerStride - a row size of the bayer image.
        \param [in] bayerFormat - a format of the input bayer image. It can be ::SimdPixelFormatBayerGrbg, ::SimdPixelFormatBayerGbrg, ::SimdPixelFormatBayerRggb or ::SimdPixelFormatBayerBggr.
        \param [out] bgra - a pointer to pixels data of output 32-bit BGRA image.
        \param [in] bgraStride - a row size of the bgra image.
        \param [in] alpha - a value of alpha channel.
    */
    SIMD_API void SimdBayerToBgra(const uint8_t * bayer, size_t width, size_t height, size_t bayerStride, SimdPixelFormatType bayerFormat, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

    /*! @ingroup bgra_conversion

        \fn void SimdBgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat);

        \short Converts 32-bit BGRA image to 8-bit Bayer image. 

        All images must have the same width and height. The width and the height must be even.

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgraStride - a row size of the bgra image.
        \param [out] bayer - a pointer to pixels data of output 8-bit Bayer image.
        \param [in] bayerStride - a row size of the bayer image.
        \param [in] bayerFormat - a format of the output bayer image. It can be ::SimdPixelFormatBayerGrbg, ::SimdPixelFormatBayerGbrg, ::SimdPixelFormatBayerRggb or ::SimdPixelFormatBayerBggr.
    */
    SIMD_API void SimdBgraToBayer(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat);

    /*! @ingroup bgra_conversion

        \fn void SimdBgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride);

        \short Converts 32-bit BGRA image to 24-bit BGR image. 

        All images must have the same width and height. 

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgraStride - a row size of the bgra image.
        \param [out] bgr - a pointer to pixels data of output 24-bit BGR image.
        \param [in] bgrStride - a row size of the bgr image.
    */
    SIMD_API void SimdBgraToBgr(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * bgr, size_t bgrStride);

    /*! @ingroup bgra_conversion

        \fn void SimdBgraToGray(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * gray, size_t grayStride);

        \short Converts 32-bit BGRA image to 8-bit gray image. 

        All images must have the same width and height. 

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgraStride - a row size of the bgra image.
        \param [out] gray - a pointer to pixels data of output 8-bit gray image.
        \param [in] grayStride - a row size of the gray image.
    */
    SIMD_API void SimdBgraToGray(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * gray, size_t grayStride);

    /*! @ingroup bgra_conversion

	* \fn void SimdBgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Converts 32-bit BGRA image to YUV420P. 

	* The input BGRA and output Y images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component). 
	*
	* \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
	* \param [in] width - an image width.
	* \param [in] height - an image height.
	* \param [in] bgraStride - a row size of the BGRA image.
	* \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
	* \param [in] yStride - a row size of the y image.
	* \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
	* \param [in] uStride - a row size of the u image.
	* \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
	* \param [in] vStride - a row size of the v image.
	*/
    SIMD_API void SimdBgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup bgra_conversion

	* \fn void SimdBgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Converts 32-bit BGRA image to YUV444P. 

	* The input BGRA and output Y, U and V images must have the same width and height.
	*
	* \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
	* \param [in] width - an image width.
	* \param [in] height - an image height.
	* \param [in] bgraStride - a row size of the BGRA image.
	* \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
	* \param [in] yStride - a row size of the y image.
	* \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
	* \param [in] uStride - a row size of the u image.
	* \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
	* \param [in] vStride - a row size of the v image.
	*/
    SIMD_API void SimdBgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup bgr_conversion

        \fn void SimdBgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat);

        \short Converts 24-bit BGR image to 8-bit Bayer image. 

        All images must have the same width and height. The width and the height must be even.

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgrStride - a row size of the bgr image.
        \param [out] bayer - a pointer to pixels data of output 8-bit Bayer image.
        \param [in] bayerStride - a row size of the bayer image.
        \param [in] bayerFormat - a format of the output bayer image. It can be ::SimdPixelFormatBayerGrbg, ::SimdPixelFormatBayerGbrg, ::SimdPixelFormatBayerRggb or ::SimdPixelFormatBayerBggr.
    */
    SIMD_API void SimdBgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat);

    /*! @ingroup bgr_conversion

        \fn void SimdBgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

        \short Converts 24-bit BGR image to 32-bit BGRA image. 

        All images must have the same width and height. 

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgrStride - a row size of the bgr image.
        \param [out] bgra - a pointer to pixels data of output 32-bit BGRA image.
        \param [in] bgraStride - a row size of the bgra image.
        \param [in] alpha - a value of alpha channel.
    */
    SIMD_API void SimdBgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

    /*! @ingroup other_conversion

        \fn void SimdBgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height, const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

        \short Converts 48-bit planar BGR image to 32-bit BGRA image. 

        All images must have the same width and height. 

        \param [in] blue - a pointer to pixels data of input 16-bit image with blue color plane.
        \param [in] blueStride - a row size of the blue image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] green - a pointer to pixels data of input 16-bit image with green color plane.
        \param [in] greenStride - a row size of the blue image.
        \param [in] red - a pointer to pixels data of input 16-bit image with red color plane.
        \param [in] redStride - a row size of the red image.
        \param [out] bgra - a pointer to pixels data of output 32-bit BGRA image.
        \param [in] bgraStride - a row size of the bgra image.
        \param [in] alpha - a value of alpha channel.
    */
    SIMD_API void SimdBgr48pToBgra32(const uint8_t * blue, size_t blueStride, size_t width, size_t height,
        const uint8_t * green, size_t greenStride, const uint8_t * red, size_t redStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

    /*! @ingroup bgr_conversion

        \fn void SimdBgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride);

        \short Converts 24-bit BGR image to 8-bit gray image. 

        All images must have the same width and height. 

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgrStride - a row size of the bgr image.
        \param [out] gray - a pointer to pixels data of output 8-bit gray image.
        \param [in] grayStride - a row size of the gray image.
    */
    SIMD_API void SimdBgrToGray(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * gray, size_t grayStride);

    /*! @ingroup bgr_conversion

        \fn void SimdBgrToHsl(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsl, size_t hslStride);

        \short Converts 24-bit BGR image to 24-bit HSL(Hue, Saturation, Lightness) image. 

        All images must have the same width and height. 

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgrStride - a row size of the bgr image.
        \param [out] hsl - a pointer to pixels data of output 24-bit HSL image.
        \param [in] hslStride - a row size of the hsl image.
    */
    SIMD_API void SimdBgrToHsl(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsl, size_t hslStride);

    /*! @ingroup bgr_conversion

        \fn void SimdBgrToHsv(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsv, size_t hsvStride);

        \short Converts 24-bit BGR image to 24-bit HSV(Hue, Saturation, Value) image. 

        All images must have the same width and height. 

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgrStride - a row size of the bgr image.
        \param [out] hsv - a pointer to pixels data of output 24-bit HSV image.
        \param [in] hsvStride - a row size of the hsv image.
    */
    SIMD_API void SimdBgrToHsv(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsv, size_t hsvStride);

    /*! @ingroup bgr_conversion

	* \fn void SimdBgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Converts 24-bit BGR image to YUV420P. 

	* The input BGR and output Y images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component). 
	*
	* \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
	* \param [in] width - an image width.
	* \param [in] height - an image height.
	* \param [in] bgrStride - a row size of the BGR image.
	* \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
	* \param [in] yStride - a row size of the y image.
	* \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
	* \param [in] uStride - a row size of the u image.
	* \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
	* \param [in] vStride - a row size of the v image.
	*/
    SIMD_API void SimdBgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup bgr_conversion

	* \fn void SimdBgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Converts 24-bit BGR image to YUV444P. 

	* The input BGR and output Y, U and V images must have the same width and height.
	*
	* \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
	* \param [in] width - an image width.
	* \param [in] height - an image height.
	* \param [in] bgrStride - a row size of the BGR image.
	* \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
	* \param [in] yStride - a row size of the y image.
	* \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
	* \param [in] uStride - a row size of the u image.
	* \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
	* \param [in] vStride - a row size of the v image.
	*/
    SIMD_API void SimdBgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup binarization

        \fn void SimdBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType compareType);

        \short Performs binarization of 8-bit gray image. 

        All images must have 8-bit gray format and must have the same width and height.

        For every point:
        \n dst[i] = compare(src[i], value) ? positive : negative,
        \n compare(a, b) depends from compareType (see ::SimdCompareType).

        \param [in] src - a pointer to pixels data of input 8-bit gray image (first value for compare operation).
        \param [in] srcStride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] value - a second value for compare operation.
        \param [in] positive - a destination value if comparison operation has a positive result.
        \param [in] negative - a destination value if comparison operation has a negative result.
        \param [out] dst - a pointer to pixels data of output 8-bit gray binarized image.
        \param [in] dstStride - a row size of the dst image.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
    */
    SIMD_API void SimdBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType compareType);

    /*! @ingroup binarization

        \fn void SimdAveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType compareType);

        \short Performs averaging binarization of 8-bit gray image. 

        All images must have 8-bit gray format and must have the same width and height.

        For every point:
        \n dst[i] = sum[i]*255 > area[i]*threshold ? positive : negative,
        \n where sum[i] is a sum of positive compare(src[i], value) operation (see ::SimdCompareType) in the point neighborhood (from -neighborhood to neighborhood for x and y),
        \n area[i] - an area of the point neighborhood ( (2*neighborhood + 1)^2 for central part of the image).

        \param [in] src - a pointer to pixels data of input 8-bit gray image (first value for compare operation).
        \param [in] srcStride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] value - a second value for compare operation.
        \param [in] neighborhood - an averaging neighborhood.
        \param [in] threshold - a threshold value for binarization. It can range from 0 to 255.
        \param [in] positive - a destination value if for neighborhood of this point number of positive comparison is greater then threshold.
        \param [in] negative - a destination value if for neighborhood of this point number of positive comparison is lesser or equal then threshold.
        \param [out] dst - a pointer to pixels data of output 8-bit gray binarized image.
        \param [in] dstStride - a row size of the dst image.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
    */
    SIMD_API void SimdAveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative,
        uint8_t * dst, size_t dstStride, SimdCompareType compareType);

    /*! @ingroup conditional

        \fn void SimdConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, SimdCompareType compareType, uint32_t * count);

        \short Calculates number of points satisfying certain condition for 8-bit gray image. 

        For every point:
        \n count += compare(src[i], value) ? 1 : 0,
        \n compare(a, b) depends from compareType (see ::SimdCompareType).

        \param [in] src - a pointer to pixels data of input 8-bit gray image (first value for compare operation).
        \param [in] stride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] count - a pointer to result unsigned 32-bit value.
    */
    SIMD_API void SimdConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, 
        uint8_t value, SimdCompareType compareType, uint32_t * count);
    
    /*! @ingroup conditional

        \fn void SimdConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, SimdCompareType compareType, uint32_t * count);

        \short Calculates number of points satisfying certain condition for 16-bit signed integer image. 

        For every point:
        \n count += compare(src[i], value) ? 1 : 0,
        \n compare(a, b) depends from compareType (see ::SimdCompareType).

        \param [in] src - a pointer to pixels data of input 16-bit signed integer image (first value for compare operation).
        \param [in] stride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] count - a pointer to result unsigned 32-bit value.
    */
    SIMD_API void SimdConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, 
        int16_t value, SimdCompareType compareType, uint32_t * count);

    /*! @ingroup conditional

        \fn void SimdConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

        \short Calculates sum of image points when mask points satisfying certain condition. 

        All images must have 8-bit gray format and must have the same width and height.

        For every point:
        \n sum += compare(mask[i], value) ? src[i] : 0,
        \n compare(a, b) depends from compareType (see ::SimdCompareType).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] mask - a pointer to pixels data of 8-bit gray mask (first value for compare operation).
        \param [in] maskStride - a row size of the mask image.
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] sum - a pointer to result unsigned 64-bit value.
    */
    SIMD_API void SimdConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

    /*! @ingroup conditional

        \fn void SimdConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

        \short Calculates sum of squared image points when mask points satisfying certain condition. 

        All images must have 8-bit gray format and must have the same width and height.

        For every point:
        \n sum += compare(mask[i], value) ? src[i]*src[i] : 0,
        \n compare(a, b) depends from compareType (see ::SimdCompareType).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] mask - a pointer to pixels data of 8-bit gray mask (first value for compare operation).
        \param [in] maskStride - a row size of the mask image.
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] sum - a pointer to result unsigned 64-bit value.
    */
    SIMD_API void SimdConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

    /*! @ingroup conditional

        \fn void SimdConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

        \short Calculates sum of squared gradient of image points when mask points satisfying certain condition. 

        All images must have 8-bit gray format and must have the same width and height. The image height and width must be equal or greater 3.

        For every point:
        \n sum += compare(mask[x, y], value) ? dx[x, y]*dx[x, y] + dy[x, y]*dy[x, y] : 0, 
        \n where for border pixels dx[x, y] = 0 and dy[x, y] = 0, for other pixels: 
        \n dx[x, y] = src[x + 1, y] - src[x - 1, y], 
        \n dy[x, y] = src[x, y + 1] - src[x, y - 1];
        \n compare(a, b) depends from compareType (see ::SimdCompareType).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] mask - a pointer to pixels data of 8-bit gray mask (first value for compare operation).
        \param [in] maskStride - a row size of the mask image.
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] sum - a pointer to result unsigned 64-bit value.
    */
    SIMD_API void SimdConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum);

    /*! @ingroup copying

        \fn void SimdCopy(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, uint8_t * dst, size_t dstStride);

        \short Copies pixels data of image from source to destination. 

        All images must have the same width, height and format.

        \param [in] src - a pointer to pixels data of source image.
        \param [in] srcStride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] pixelSize - a size of the image pixel.
        \param [out] dst - a pointer to pixels data of destination image.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdCopy(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, uint8_t * dst, size_t dstStride);

    /*! @ingroup copying

        \fn void SimdCopyFrame(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t * dst, size_t dstStride);

        \short Copies pixels data of image from source to destination except for the portion bounded frame. 

        All images must have the same width, height and format.

        \param [in] src - a pointer to pixels data of source image.
        \param [in] srcStride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] pixelSize - a size of the image pixel.
        \param [in] frameLeft - a frame left side.
        \param [in] frameTop - a frame top side.
        \param [in] frameRight - a frame right side.
        \param [in] frameBottom - a frame bottom side.
        \param [out] dst - a pointer to pixels data of destination image.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdCopyFrame(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize,
        size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t * dst, size_t dstStride);

    /*! @ingroup other_conversion

        \fn void SimdDeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Deinterleaves 16-bit UV interleaved image into separated 8-bit U and V planar images. 

        All images must have the same width and height.
        This function used for NV12 to YUV420P conversion.

        \param [in] uv - a pointer to pixels data of input 16-bit UV interleaved image.
        \param [in] uvStride - a row size of the uv image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] u - a pointer to pixels data of 8-bit U planar image.
        \param [in] uStride - a row size of the u image.
        \param [out] v - a pointer to pixels data of 8-bit V planar image.
        \param [in] vStride - a row size of the v image.
    */
    SIMD_API void SimdDeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
        uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup edge_background

        \fn void SimdEdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * background, size_t backgroundStride);

        \short Performs edge background update (initial grow, slow mode). 

        All images must have the same width, height and format (8-bit gray). 

        For every point: 
        \n background[i] += value[i] > background[i] ? 1 : 0; 

        This function is used for edge background updating in motion detection algorithm.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] background - a pointer to pixels data of feature value of edge dynamic background.
        \param [in] backgroundStride - a row size of the background image.
    */
    SIMD_API void SimdEdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * background, size_t backgroundStride);

    /*! @ingroup edge_background

        \fn void SimdEdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * background, size_t backgroundStride);

        \short Performs edge background update (initial grow, fast mode). 

        All images must have the same width, height and format (8-bit gray). 

        For every point: 
        \n background[i] = value[i] > background[i] ? value[i] : background[i]; 

        This function is used for edge background updating in motion detection algorithm.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] background - a pointer to pixels data of feature value of edge dynamic background.
        \param [in] backgroundStride - a row size of the background image.
    */
    SIMD_API void SimdEdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * background, size_t backgroundStride);

    /*! @ingroup edge_background

        \fn void SimdEdgeBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height, const uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t * backgroundCount, size_t backgroundCountStride);

        \short Performs collection of edge background statistic. 

        All images must have the same width, height and format (8-bit gray). 

        Updates background statistic counters for every point: 
        \n backgroundCount[i] += (value[i] > backgroundValue[i] && backgroundCount[i] < 255) ? 1 : 0;

        This function is used for edge background updating in motion detection algorithm.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] backgroundValue - a pointer to pixels data of value of feature of edge dynamic background.
        \param [in] backgroundValueStride - a row size of the backgroundValue image.
        \param [in, out] backgroundCount - a pointer to pixels data of count of feature of edge dynamic background.
        \param [in] backgroundCountStride - a row size of the backgroundCount image.
    */
    SIMD_API void SimdEdgeBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        const uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t * backgroundCount, size_t backgroundCountStride);

    /*! @ingroup edge_background

        \fn void SimdEdgeBackgroundAdjustRange(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold);

        \short Performs adjustment of edge background range. 

        All images must have the same width, height and format (8-bit gray). 

        Adjusts edge background range for every point: 
        \n backgroundValue[i] += (backgroundCount[i] > threshold && backgroundValue[i] < 255) ? 1 : 0;
        \n backgroundValue[i] -= (backgroundCount[i] < threshold && backgroundValue[i] > 0) ? 1 : 0; 
        \n backgroundCount[i] = 0;

        This function is used for edge background updating in motion detection algorithm.

        \param [in, out] backgroundCount - a pointer to pixels data of count of feature of edge dynamic background.
        \param [in] backgroundCountStride - a row size of the backgroundCount image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] backgroundValue - a pointer to pixels data of value of feature of edge dynamic background.
        \param [in] backgroundValueStride - a row size of the backgroundValue image.
        \param [in] threshold - a count threshold.
    */
    SIMD_API void SimdEdgeBackgroundAdjustRange(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
        uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold);

    /*! @ingroup edge_background

        \fn void SimdEdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height, uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride);

        \short Performs adjustment of edge background range with using adjust range mask. 

        All images must have the same width, height and format (8-bit gray). 

        Adjusts edge background range for every point when mask[i] != 0: 
        \n backgroundValue[i] += (backgroundCount[i] > threshold && backgroundValue[i] < 255) ? 1 : 0;
        \n backgroundValue[i] -= (backgroundCount[i] < threshold && backgroundValue[i] > 0) ? 1 : 0; 
        \n backgroundCount[i] = 0;

        This function is used for edge background updating in motion detection algorithm.

        \param [in, out] backgroundCount - a pointer to pixels data of count of feature of edge dynamic background.
        \param [in] backgroundCountStride - a row size of the backgroundCount image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] backgroundValue - a pointer to pixels data of value of feature of edge dynamic background.
        \param [in] backgroundValueStride - a row size of the backgroundValue image.
        \param [in] threshold - a count threshold.
        \param [in] mask - a pointer to pixels data of adjust range mask.
        \param [in] maskStride - a row size of the mask image.
    */
    SIMD_API void SimdEdgeBackgroundAdjustRangeMasked(uint8_t * backgroundCount, size_t backgroundCountStride, size_t width, size_t height,
        uint8_t * backgroundValue, size_t backgroundValueStride, uint8_t threshold, const uint8_t * mask, size_t maskStride);

    /*! @ingroup edge_background

        \fn void SimdEdgeBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * background, size_t backgroundStride);

        \short Shifts edge background range. 

        All images must have the same width, height and format (8-bit gray). 

        For every point: 
        \n background[i] = value[i];

        This function is used for fast edge background updating in motion detection algorithm.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] background - a pointer to pixels data of feature of edge dynamic background.
        \param [in] backgroundStride - a row size of the background image.
    */
    SIMD_API void SimdEdgeBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * background, size_t backgroundStride);

    /*! @ingroup edge_background

        \fn void SimdEdgeBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * background, size_t backgroundStride, const uint8_t * mask, size_t maskStride);

        \short Shifts edge background range with using shift range mask. 

        All images must have the same width, height and format (8-bit gray). 

        For every point when mask[i] != 0 : 
        \n background[i] = value[i];

        This function is used for fast edge background updating in motion detection algorithm.

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] background - a pointer to pixels data of feature of edge dynamic background.
        \param [in] backgroundStride - a row size of the background image.
        \param [in] mask - a pointer to pixels data of shift range mask.
        \param [in] maskStride - a row size of the mask image.
    */
    SIMD_API void SimdEdgeBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * background, size_t backgroundStride, const uint8_t * mask, size_t maskStride);

    /*! @ingroup filling

        \fn void SimdFill(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize, uint8_t value);

        \short Fills pixels data of image by given value. 

        \param [out] dst - a pointer to pixels data of destination image.
        \param [in] stride - a row size of the dst image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] pixelSize - a size of the image pixel.
        \param [in] value - a value to fill image.
    */
    SIMD_API void SimdFill(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize, uint8_t value);

    /*! @ingroup filling

        \fn void SimdFillFrame(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize, size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t value);

        \short Fills pixels data of image except for the portion bounded frame by given value. 

        \param [out] dst - a pointer to pixels data of destination image.
        \param [in] stride - a row size of the dst image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] pixelSize - a size of the image pixel.
        \param [in] frameLeft - a frame left side.
        \param [in] frameTop - a frame top side.
        \param [in] frameRight - a frame right side.
        \param [in] frameBottom - a frame bottom side.
        \param [in] value - a value to fill image.
    */
    SIMD_API void SimdFillFrame(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize,
        size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t value);

    /*! @ingroup filling

        \fn void SimdFillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red);

        \short Fills pixels data of 24-bit BGR image by given color(blue, green, red). 

        \param [out] dst - a pointer to pixels data of destination image.
        \param [in] stride - a row size of the dst image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] blue - a blue channel of BGR to fill image.
        \param [in] green - a green channel of BGR to fill image.
        \param [in] red - a red channel of BGR to fill image.
    */
    SIMD_API void SimdFillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red);

    /*! @ingroup filling

        \fn void SimdFillBgra(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha);

        \short Fills pixels data of 32-bit BGRA image by given color(blue, green, red, alpha). 

        \param [out] dst - a pointer to pixels data of destination image.
        \param [in] stride - a row size of the dst image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] blue - a blue channel of BGRA to fill image.
        \param [in] green - a green channel of BGRA to fill image.
        \param [in] red - a red channel of BGRA to fill image.
        \param [in] alpha - a alpha channel of BGRA to fill image.
    */
    SIMD_API void SimdFillBgra(uint8_t * dst, size_t stride, size_t width, size_t height,
        uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha);

    /*! @ingroup other_filter

        \fn void SimdGaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);

        \short Performs Gaussian blur filtration with window 3x3. 

        For every point:
        \n dst[x, y] = (src[x-1, y-1] + 2*src[x, y-1] + src[x+1, y-1] + 
        \n 2*(src[x-1, y] + 2*src[x, y] + src[x+1, y]) +
        \n src[x-1, y+1] + 2*src[x, y+1] + src[x+1, y+1] + 8) / 16; 

        All images must have the same width, height and format (8-bit gray, 24-bit BGR or 32-bit BGRA).

        \param [in] src - a pointer to pixels data of source image.
        \param [in] srcStride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count.
        \param [out] dst - a pointer to pixels data of destination image.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdGaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    /*! @ingroup gray_conversion

        \fn void SimdGrayToBgr(const uint8_t * gray, size_t width, size_t height, size_t grayStride, uint8_t * bgr, size_t bgrStride);

        \short Converts 8-bit gray image to 24-bit BGR image. 

        All images must have the same width and height. 

        \param [in] gray - a pointer to pixels data of input 8-bit gray image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] grayStride - a row size of the gray image.
        \param [out] bgr - a pointer to pixels data of output 24-bit BGR image.
        \param [in] bgrStride - a row size of the bgr image.
    */
    SIMD_API void SimdGrayToBgr(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgr, size_t bgrStride);

    /*! @ingroup gray_conversion

        \fn void SimdGrayToBgra(const uint8_t * gray, size_t width, size_t height, size_t grayStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

        \short Converts 8-bit gray image to 32-bit BGRA image. 

        All images must have the same width and height. 

        \param [in] gray - a pointer to pixels data of input 8-bit gray image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] grayStride - a row size of the gray image.
        \param [out] bgra - a pointer to pixels data of output 32-bit BGRA image.
        \param [in] bgraStride - a row size of the bgra image.
        \param [in] alpha - a value of alpha channel.
    */
    SIMD_API void SimdGrayToBgra(const uint8_t *gray, size_t width, size_t height, size_t grayStride,
        uint8_t *bgra, size_t bgraStride, uint8_t alpha);

    /*! @ingroup histogram

        \fn void SimdAbsSecondDerivativeHistogram(const uint8_t * src, size_t width, size_t height, size_t stride, size_t step, size_t indent, uint32_t * histogram);

        \short Calculates histogram of second derivative for 8-bit gray image. 

        For all points except the boundary (defined by parameter indent): 
        \n histogram[max(dx, dy)]++, where
        \n dx = abs(src[x, y] - average(src[x+step, y], src[x-step, y])),
        \n dy = abs(src[x, y] - average(src[x, y+step], src[x, y-step])).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] stride - a row size of the image.
        \param [in] step - a step for second derivative calculation.
        \param [in] indent - a indent from image boundary.
        \param [out] histogram - a pointer to histogram (array of 256 unsigned 32-bit values).
    */
    SIMD_API void SimdAbsSecondDerivativeHistogram(const uint8_t * src, size_t width, size_t height, size_t stride,
        size_t step, size_t indent, uint32_t * histogram);

    /*! @ingroup histogram

        \fn void SimdHistogram(const uint8_t * src, size_t width, size_t height, size_t stride, uint32_t * histogram);

        \short Calculates histogram for 8-bit gray image. 

        For all points: 
        \n histogram[src(i)]++.

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] stride - a row size of the image.
        \param [out] histogram - a pointer to histogram (array of 256 unsigned 32-bit values).
    */
    SIMD_API void SimdHistogram(const uint8_t * src, size_t width, size_t height, size_t stride, uint32_t * histogram);

    /*! @ingroup histogram

        \fn void SimdHistogramMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t index, uint32_t * histogram);

        \short Calculates histogram for 8-bit gray image with using mask. 

        For every point where mask[i] == index: 
        \n histogram[src(i)]++.

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] mask - a pointer to pixels data of the mask 8-bit image.
        \param [in] maskStride - a row size of the mask image.
        \param [in] index - a mask index.
        \param [out] histogram - a pointer to histogram (array of 256 unsigned 32-bit values).
    */
    SIMD_API void SimdHistogramMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        const uint8_t * mask, size_t maskStride, uint8_t index, uint32_t * histogram);

    /*! @ingroup integral

        \fn void SimdIntegral(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * sum, size_t sumStride, uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride, SimdPixelFormatType sumFormat, SimdPixelFormatType sqsumFormat);

        \short Calculates integral images for input 8-bit gray image. 

        The function can calculates sum integral image, square sum integral image (optionally) and tilted sum integral image (optionally). 
        A integral images must have width and height per unit greater than that of the input image. 

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - a pointer to pixels data of sum image. 
        \param [in] sumStride - a row size of sum image (in bytes).
        \param [out] sqsum - a pointer to pixels data of square sum image. It can be NULL.
        \param [in] sqsumStride - a row size of sqsum image (in bytes).
        \param [out] tilted - a pointer to pixels data of tilted sum image.
        \param [in] tiltedStride - a row size of tilted image (in bytes). It can be NULL.
        \param [in] sumFormat - a format of sum image and tilted image. It can be equal to ::SimdPixelFormatInt32.
        \param [in] sqsumFormat - a format of sqsum image. It can be equal to ::SimdPixelFormatInt32 or ::SimdPixelFormatDouble.
    */
    SIMD_API void SimdIntegral(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        uint8_t * sum, size_t sumStride, uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride, 
        SimdPixelFormatType sumFormat, SimdPixelFormatType sqsumFormat);

    /*! @ingroup interference

        \fn void SimdInterferenceIncrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t increment, int16_t saturation);

        \short Increments statistic of interference detector. 

        For every point: 
        \n statistic[i] = min(statistic[i] + increment, saturation); 

        This function is used for interference detection in motion detection algorithm.

        \param [in, out] statistic - a pointer to pixels data of 16-bit signed integer image with statistic.
        \param [in] stride - a row size of statistic image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] increment - an increment of statistic.
        \param [in] saturation - an upper saturation of statistic.
    */
    SIMD_API void SimdInterferenceIncrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t increment, int16_t saturation);

    /*! @ingroup interference

        \fn void SimdInterferenceIncrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height, uint8_t increment, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index);

        \short Increments statistic of interference detector with using segmentation mask. 

        For every point when mask[i] == index: 
        \n statistic[i] = min(statistic[i] + increment, saturation); 

        All images must have the same width, height. 
        This function is used for interference detection in motion detection algorithm.

        \param [in, out] statistic - a pointer to pixels data of 16-bit signed integer image with statistic.
        \param [in] statisticStride - a row size of statistic image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] increment - an increment of statistic.
        \param [in] saturation - an upper saturation of statistic.
        \param [in] mask - a pointer to pixels data of 8-bit gray image with mask.
        \param [in] maskStride - a row size of mask image.
        \param [in] index - an index of mask.
    */
    SIMD_API void SimdInterferenceIncrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height, 
        uint8_t increment, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index);

    /*! @ingroup interference

        \fn void SimdInterferenceDecrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t decrement, int16_t saturation);

        \short Decrements statistic of interference detector. 

        For every point: 
        \n statistic[i] = max(statistic[i] - decrement, saturation); 

        This function is used for interference detection in motion detection algorithm.

        \param [in, out] statistic - a pointer to pixels data of 16-bit signed integer image with statistic.
        \param [in] stride - a row size of statistic image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] decrement - an decrement of statistic.
        \param [in] saturation - an lower saturation of statistic.
    */
    SIMD_API void SimdInterferenceDecrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t decrement, int16_t saturation);

    /*! @ingroup interference

        \fn void SimdInterferenceDecrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height, uint8_t decrement, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index);

        \short Decrements statistic of interference detector with using segmentation mask. 

        For every point when mask[i] == index: 
        \n statistic[i] = max(statistic[i] - decrement, saturation); 

        All images must have the same width, height. 
        This function is used for interference detection in motion detection algorithm.

        \param [in, out] statistic - a pointer to pixels data of 16-bit signed integer image with statistic.
        \param [in] statisticStride - a row size of statistic image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] decrement - an decrement of statistic.
        \param [in] saturation - an lower saturation of statistic.
        \param [in] mask - a pointer to pixels data of 8-bit gray image with mask.
        \param [in] maskStride - a row size of mask image.
        \param [in] index - an index of mask.
    */
    SIMD_API void SimdInterferenceDecrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height, 
        uint8_t decrement, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index);

    /*! @ingroup other_filter

        \fn void SimdLbpEstimate(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Calculates LBP (Local Binary Patterns) for 8-bit gray image. 

        All images must have the same width and height. 

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of output 8-bit gray image with LBP.
        \param [in] dstStride - a row size of dst image.
    */
    SIMD_API void SimdLbpEstimate(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup median_filter

        \fn void SimdMedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);

        \short Performs median filtration of input image (filter window is a rhomb 3x3). 

        All images must have the same width, height and format (8-bit gray, 24-bit BGR or 32-bit BGRA). 

        \param [in] src - a pointer to pixels data of original input image.
        \param [in] srcStride - a row size of src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count.
        \param [out] dst - a pointer to pixels data of filtered output image.
        \param [in] dstStride - a row size of dst image.
    */
    SIMD_API void SimdMedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    /*! @ingroup median_filter

        \fn void SimdMedianFilterRhomb5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);

        \short Performs median filtration of input image (filter window is a rhomb 5x5). 

        All images must have the same width, height and format (8-bit gray, 24-bit BGR or 32-bit BGRA). 

        \param [in] src - a pointer to pixels data of original input image.
        \param [in] srcStride - a row size of src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count.
        \param [out] dst - a pointer to pixels data of filtered output image.
        \param [in] dstStride - a row size of dst image.
    */
    SIMD_API void SimdMedianFilterRhomb5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    /*! @ingroup median_filter

        \fn void SimdMedianFilterSquare3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);

        \short Performs median filtration of input image (filter window is a square 3x3). 

        All images must have the same width, height and format (8-bit gray, 24-bit BGR or 32-bit BGRA). 

        \param [in] src - a pointer to pixels data of original input image.
        \param [in] srcStride - a row size of src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count.
        \param [out] dst - a pointer to pixels data of filtered output image.
        \param [in] dstStride - a row size of dst image.
    */
    SIMD_API void SimdMedianFilterSquare3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    /*! @ingroup median_filter

        \fn void SimdMedianFilterSquare5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);

        \short Performs median filtration of input image (filter window is a square 5x5). 

        All images must have the same width, height and format (8-bit gray, 24-bit BGR or 32-bit BGRA). 

        \param [in] src - a pointer to pixels data of original input image.
        \param [in] srcStride - a row size of src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count.
        \param [out] dst - a pointer to pixels data of filtered output image.
        \param [in] dstStride - a row size of dst image.
    */
    SIMD_API void SimdMedianFilterSquare5x5(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    /*! @ingroup operation

        \fn void SimdOperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type);

        \short Performs given operation between two images. 

        All images must have the same width, height and format (8-bit gray, 16-bit UV (UV plane of NV12 pixel format), 24-bit BGR or 32-bit BGRA). 

        \param [in] a - a pointer to pixels data of the first input image.
        \param [in] aStride - a row size of the first image.
        \param [in] b - a pointer to pixels data of the second input image.
        \param [in] bStride - a row size of the second image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count.
        \param [out] dst - a pointer to pixels data of filtered output image.
        \param [in] dstStride - a row size of dst image.
        \param [in] type - a type of operation (see ::SimdOperationBinary8uType).
    */
    SIMD_API void SimdOperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type);

    /*! @ingroup operation

        \fn void SimdOperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type);

        \short Performs given operation between two images. 

        All images must have the same width, height and ::SimdPixelFormatInt16 pixel format. 

        \param [in] a - a pointer to pixels data of the first input image.
        \param [in] aStride - a row size of the first image.
        \param [in] b - a pointer to pixels data of the second input image.
        \param [in] bStride - a row size of the second image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of filtered output image.
        \param [in] dstStride - a row size of dst image.
        \param [in] type - a type of operation (see ::SimdOperationBinary16iType).
    */
    SIMD_API void SimdOperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type);

    /*! @ingroup operation

        \fn void SimdVectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height);

        \short Calculates result 8-bit gray image as product of two vectors. 

        For all points: 
        \n dst[x, y] = horizontal[x]*vertical[y]/255.

        \param [in] vertical - a pointer to pixels data of vertical vector. It length is equal to result image height.
        \param [in] horizontal - a pointer to pixels data of horizontal vector. It length is equal to result image width.
        \param [out] dst - a pointer to pixels data of result image.
        \param [in] stride - a row size of dst image.
        \param [in] width - an image width.
        \param [in] height - an image height.
    */
    SIMD_API void SimdVectorProduct(const uint8_t * vertical, const uint8_t * horizontal,
        uint8_t * dst, size_t stride, size_t width, size_t height);

    /*! @ingroup resizing

        \fn void SimdReduceGray2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

        \short Performs reducing and Gaussian blurring (in two time) a 8-bit gray image with using window 2x2. 

        For input and output image must be performed: dstWidth = (srcWidth + 1)/2,  dstHeight = (srcHeight + 1)/2.

        For all points: 
        \n dst[x, y] = (src[2*x, 2*y] + src[2*x, 2*y + 1] + src[2*x + 1, 2*y] + src[2*x + 1, 2*y + 1] + 2)/4.

        \param [in] src - a pointer to pixels data of the original input image.
        \param [in] srcWidth - a width of the input image.
        \param [in] srcHeight - a height of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [out] dst - a pointer to pixels data of the reduced output image.
        \param [in] dstWidth - a width of the output image.
        \param [in] dstHeight - a height of the output image.
        \param [in] dstStride - a row size of the output image.
    */
    SIMD_API void SimdReduceGray2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

    /*! @ingroup resizing

        \fn void SimdReduceGray3x3(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation);

        \short Performs reducing and Gaussian blurring (in two time) a 8-bit gray image with using window 3x3. 

        For input and output image must be performed: dstWidth = (srcWidth + 1)/2,  dstHeight = (srcHeight + 1)/2.

        For every point:
        \n dst[x, y] = (src[2*x-1, 2*y-1] + 2*src[2*x, 2*y-1] + src[2*x+1, 2*y-1] + 
        \n 2*(src[2*x-1, 2*y] + 2*src[2*x, 2*y] + src[2*x+1, 2*y]) +
        \n src[2*x-1, 2*y+1] + 2*src[2*x, 2*y+1] + src[2*x+1, 2*y+1] + compensation ? 8 : 0) / 16; 

        \param [in] src - a pointer to pixels data of the original input image.
        \param [in] srcWidth - a width of the input image.
        \param [in] srcHeight - a height of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [out] dst - a pointer to pixels data of the reduced output image.
        \param [in] dstWidth - a width of the output image.
        \param [in] dstHeight - a height of the output image.
        \param [in] dstStride - a row size of the output image.
        \param [in] compensation - a flag of compensation of rounding.
    */
    SIMD_API void SimdReduceGray3x3(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation);

    /*! @ingroup resizing

        \fn void SimdReduceGray4x4(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

        \short Performs reducing and Gaussian blurring (in two time) a 8-bit gray image with using window 4x4. 

        For input and output image must be performed: dstWidth = (srcWidth + 1)/2,  dstHeight = (srcHeight + 1)/2.

        For every point:
        \n dst[x, y] = (src[2*x-1, 2*y-1] + 3*src[2*x, 2*y-1] + 3*src[2*x+1, 2*y-1] + src[2*x+2, 2*y-1]
        \n 3*(src[2*x-1, 2*y] + 3*src[2*x, 2*y] + 3*src[2*x+1, 2*y] + src[2*x+2, 2*y]) +
        \n 3*(src[2*x-1, 2*y+1] + 3*src[2*x, 2*y+1] + 3*src[2*x+1, 2*y+1] + src[2*x+2, 2*y+1]) +
        \n src[2*x-1, 2*y+2] + 3*src[2*x, 2*y+2] + 3*src[2*x+1, 2*y+2] + src[2*x+2, 2*y+2] + 32) / 64; 

        \param [in] src - a pointer to pixels data of the original input image.
        \param [in] srcWidth - a width of the input image.
        \param [in] srcHeight - a height of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [out] dst - a pointer to pixels data of the reduced output image.
        \param [in] dstWidth - a width of the output image.
        \param [in] dstHeight - a height of the output image.
        \param [in] dstStride - a row size of the output image.
    */
    SIMD_API void SimdReduceGray4x4(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

    /*! @ingroup resizing

        \fn void SimdReduceGray5x5(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation);

        \short Performs reducing and Gaussian blurring (in two time) a 8-bit gray image with using window 5x5. 

        For input and output image must be performed: dstWidth = (srcWidth + 1)/2,  dstHeight = (srcHeight + 1)/2.

        For every point:
        \n dst[x, y] = (src[2*x-2, 2*y-2] + 4*src[2*x-1, 2*y-2] + 6*src[2*x, 2*y-2] + 4*src[2*x+1, 2*y-2] + src[2*x+2, 2*y-2] +
        \n 4*(src[2*x-2, 2*y-1] + 4*src[2*x-1, 2*y-1] + 6*src[2*x, 2*y-1] + 4*src[2*x+1, 2*y-1] + src[2*x+2, 2*y-1]) +
        \n 6*(src[2*x-2, 2*y] + 4*src[2*x-1, 2*y] + 6*src[2*x, 2*y] + 4*src[2*x+1, 2*y] + src[2*x+2, 2*y]) +
        \n 4*(src[2*x-2, 2*y+1] + 4*src[2*x-1, 2*y+1] + 6*src[2*x, 2*y+1] + 4*src[2*x+1, 2*y+1] + src[2*x+2, 2*y+1]) +
        \n src[2*x-2, 2*y+2] + 4*src[2*x-1, 2*y+2] + 6*src[2*x, 2*y+2] + 4*src[2*x+1, 2*y+2] + src[2*x+2, 2*y+2] + compensation ? 128 : 0) / 256; 

        \param [in] src - a pointer to pixels data of the original input image.
        \param [in] srcWidth - a width of the input image.
        \param [in] srcHeight - a height of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [out] dst - a pointer to pixels data of the reduced output image.
        \param [in] dstWidth - a width of the output image.
        \param [in] dstHeight - a height of the output image.
        \param [in] dstStride - a row size of the output image.
        \param [in] compensation - a flag of compensation of rounding.
    */
    SIMD_API void SimdReduceGray5x5(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation);

    /*! @ingroup reordering

        \fn void SimdReorder16bit(const uint8_t * src, size_t size, uint8_t * dst);

        \short Performs bytes reordering for data array. 

        For every 2 bytes:
        \n dst[2*i + 0] = src[2*i + 1];
        \n dst[2*i + 1] = src[2*i + 0];
        
        The data size must be a multiple of 2. 

        \param [in] src - a pointer to the input data.
        \param [in] size - a size of input and output data.
        \param [out] dst - a pointer to the output data.
    */
    SIMD_API void SimdReorder16bit(const uint8_t * src, size_t size, uint8_t * dst);
    
    /*! @ingroup reordering

        \fn void SimdReorder32bit(const uint8_t * src, size_t size, uint8_t * dst);

        \short Performs bytes reordering for data array. 

        For every 4 bytes:
        \n dst[4*i + 0] = src[4*i + 3];
        \n dst[4*i + 1] = src[4*i + 2];
        \n dst[4*i + 2] = src[4*i + 1];
        \n dst[4*i + 3] = src[4*i + 0];
        
        The data size must be a multiple of 4. 

        \param [in] src - a pointer to the input data.
        \param [in] size - a size of input and output data.
        \param [out] dst - a pointer to the output data.
    */
    SIMD_API void SimdReorder32bit(const uint8_t * src, size_t size, uint8_t * dst);

    /*! @ingroup reordering

        \fn void SimdReorder64bit(const uint8_t * src, size_t size, uint8_t * dst);

        \short Performs bytes reordering for data array. 

        For every 8 bytes:
        \n dst[8*i + 0] = src[8*i + 7];
        \n dst[8*i + 1] = src[8*i + 6];
        \n dst[8*i + 2] = src[8*i + 5];
        \n dst[8*i + 3] = src[8*i + 4];
        \n dst[8*i + 4] = src[8*i + 3];
        \n dst[8*i + 5] = src[8*i + 2];
        \n dst[8*i + 6] = src[8*i + 1];
        \n dst[8*i + 7] = src[8*i + 0];
        
        The data size must be a multiple of 8. 

        \param [in] src - a pointer to the input data.
        \param [in] size - a size of input and output data.
        \param [out] dst - a pointer to the output data.
    */
    SIMD_API void SimdReorder64bit(const uint8_t * src, size_t size, uint8_t * dst);

    /*! @ingroup resizing

        \fn void SimdResizeBilinear(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount);

        \short Performs resizing of input image with using bilinear interpolation. 

        All images must have the same format (8-bit gray, 24-bit BGR or 32-bit BGRA). 

        \param [in] src - a pointer to pixels data of the original input image.
        \param [in] srcWidth - a width of the input image.
        \param [in] srcHeight - a height of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [out] dst - a pointer to pixels data of the reduced output image.
        \param [in] dstWidth - a width of the output image.
        \param [in] dstHeight - a height of the output image.
        \param [in] dstStride - a row size of the output image.
        \param [in] channelCount - a channel count.
    */
    SIMD_API void SimdResizeBilinear(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount);

    /*! @ingroup segmentation

        \fn void SimdSegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex);

        \short Changes certain index in mask. 

        Mask must has 8-bit gray pixel format. 

        For every point:
        \n mask[i] = mask[i] == oldIndex ? newIndex : mask[i]. 

        \param [in, out] mask - a pointer to pixels data of 8-bit gray mask image.
        \param [in] stride - a row size of the mask image.
        \param [in] width - a mask width.
        \param [in] height - a mask height.
        \param [in] oldIndex - a mask old index.
        \param [in] newIndex - a mask new index.
    */
    SIMD_API void SimdSegmentationChangeIndex(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t oldIndex, uint8_t newIndex);

    /*! @ingroup segmentation

        \fn void SimdSegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index);

        \short Fill single holes in mask. 

        Mask must has 8-bit gray pixel format. 

        \param [in, out] mask - a pointer to pixels data of 8-bit gray mask image.
        \param [in] stride - a row size of the mask image.
        \param [in] width - an mask width.
        \param [in] height - an mask height.
        \param [in] index - a mask index.
    */
    SIMD_API void SimdSegmentationFillSingleHoles(uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index);

    /*! @ingroup segmentation

        \fn void SimdSegmentationPropagate2x2(const uint8_t * parent, size_t parentStride, size_t width, size_t height, uint8_t * child, size_t childStride, const uint8_t * difference, size_t differenceStride, uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold);

        \short Propagates mask index from parent (upper) to child (lower) level of mask pyramid with using 2x2 scan window. 

        For parent and child image must be performed: parentWidth = (childWidth + 1)/2, parentHeight = (childHeight + 1)/2.
        All images must have 8-bit gray pixel format. Size of different image is equal to child image.

        \param [in] parent - a pointer to pixels data of 8-bit gray parent mask image.
        \param [in] parentStride - a row size of the parent mask image.
        \param [in] width - a parent mask width.
        \param [in] height - a parent mask height.
        \param [in, out] child - a pointer to pixels data of 8-bit gray child mask image.
        \param [in] childStride - a row size of the child mask image.
        \param [in] difference - a pointer to pixels data of 8-bit gray difference image.
        \param [in] differenceStride - a row size of the difference image.
        \param [in] currentIndex - propagated mask index.
        \param [in] invalidIndex - invalid mask index.
        \param [in] emptyIndex - empty mask index.
        \param [in] differenceThreshold - a difference threshold for conditional index propagating.
    */
    SIMD_API void SimdSegmentationPropagate2x2(const uint8_t * parent, size_t parentStride, size_t width, size_t height, 
        uint8_t * child, size_t childStride, const uint8_t * difference, size_t differenceStride, 
        uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t differenceThreshold);

    /*! @ingroup segmentation

        \fn void SimdSegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom);

        \short Finds actual region of mask index location. 

        Mask must has 8-bit gray pixel format. 

        \param [in] mask - a pointer to pixels data of 8-bit gray mask image.
        \param [in] stride - a row size of the mask image.
        \param [in] width - an mask width.
        \param [in] height - an mask height.
        \param [in] index - a mask index.
        \param [in, out] left - a pointer to left side.
        \param [in, out] top - a pointer to top side.
        \param [in, out] right - a pointer to right side.
        \param [in, out] bottom - a pointer to bottom side.
    */
    SIMD_API void SimdSegmentationShrinkRegion(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
        ptrdiff_t * left, ptrdiff_t * top, ptrdiff_t * right, ptrdiff_t * bottom);

    /*! @ingroup shifting

        \fn void SimdShiftBilinear(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, const uint8_t * bkg, size_t bkgStride, double shiftX, double shiftY, size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride);

        \short Performs shifting of input image with using bilinear interpolation. 

        All images must have the same width, height and format (8-bit gray, 24-bit BGR or 32-bit BGRA). 

        \param [in] src - a pointer to pixels data of the foreground input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count.
        \param [in] bkg - a pointer to pixels data of the background input image.
        \param [in] bkgStride - a row size of the background image.
        \param [in] shiftX - an image shift along x axis.
        \param [in] shiftY - an image shift along y axis.
        \param [in] cropLeft - a crop left side.
        \param [in] cropTop - a crop top side.
        \param [in] cropRight - a crop right side.
        \param [in] cropBottom - a crop bottom side.
        \param [out] dst - a pointer to pixels data of the output image.
        \param [in] dstStride - a row size of the output image.
    */
    SIMD_API void SimdShiftBilinear(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
        const uint8_t * bkg, size_t bkgStride, double shiftX, double shiftY,
        size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride);

    /*! @ingroup sobel_filter

        \fn void SimdSobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Calculates Sobel's filter along x axis. 

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format. 

        For every point: 
        \n dst[x, y] = (src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1]).

        \param [in] src - a pointer to pixels data of the foreground input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of the output image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdSobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup sobel_filter

        \fn void SimdSobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Calculates absolute value of Sobel's filter along x axis. 

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format. 

        For every point: 
        \n dst[x, y] = abs((src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1])).

        \param [in] src - a pointer to pixels data of the foreground input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of the output image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdSobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup sobel_filter

        \fn void SimdSobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Calculates Sobel's filter along y axis. 

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format. 

        For every point: 
        \n dst[x, y] = (src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]).

        \param [in] src - a pointer to pixels data of the foreground input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of the output image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdSobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);
    
    /*! @ingroup sobel_filter

        \fn void SimdSobelDyAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Calculates absolute value of Sobel's filter along y axis. 

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format. 

        For every point: 
        \n dst[x, y] = abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1])).

        \param [in] src - a pointer to pixels data of the foreground input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of the output image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdSobelDyAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup contour

        \fn void SimdContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)

        \short Calculates contour metrics based on absolute value and direction of Sobel's filter along y and y axis. 

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format. 
        This function is used for contour extraction. 

        For every point: 
        \n dy = abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1])).
        \n dx = abs((src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1])).
        \n dst[x, y] = (dx + dy)*2 + (dx >= dy ? 0 : 1).

        \param [in] src - a pointer to pixels data of the gray 8-bit input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of the output 16-bit image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup contour

        \fn void SimdContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t indexMin, uint8_t * dst, size_t dstStride)

        \short Calculates contour metrics based on absolute value and direction of Sobel's filter along y and y axis with using mask. 

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format. 
        This function is used for contour extraction. 

        For every point: 
        \n dy = abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1])).
        \n dx = abs((src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1])).
        \n dst[x, y] = mask[x, y] < indexMin ? 0 : (dx + dy)*2 + (dx >= dy ? 0 : 1).

        \param [in] src - a pointer to pixels data of the gray 8-bit input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] mask - a pointer to pixels data of the mask 8-bit image.
        \param [in] maskStride - a row size of the mask image.
        \param [in] indexMin - a mask minimal permissible index.
        \param [out] dst - a pointer to pixels data of the output 16-bit image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
        const uint8_t * mask, size_t maskStride, uint8_t indexMin, uint8_t * dst, size_t dstStride);

    /*! @ingroup contour

        \fn void SimdContourAnchors(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t step, int16_t threshold, uint8_t * dst, size_t dstStride);

        \short Extract contour anchors from contour metrics. 

        All images must have the same width and height. Input image must has 16-bit integer format, output image must has 8-bit gray format. 
        Input image with metrics can be estimated by using ::SimdContourMetrics or ::SimdContourMetricsMasked functions. 
        This function is used for contour extraction. 

        For every point (except border): 
        \n a[x, y] = src[x, y] >> 1.
        \n if(src[x, y] & 1)
        \n dst[x, y] = a[x, y] > 0 && (a[x, y] - a[x + 1, y] >= threshold) && (a[x, y] - a[x - 1, y] >= threshold) ? 255 : 0;
        \n else
        \n dst[x, y] = a[x, y] > 0 && (a[x, y] - a[x, y + 1] >= threshold) && (a[x, y] - a[x, y - 1] >= threshold) ? 255 : 0;

        \param [in] src - a pointer to pixels data of the 16-bit input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] step - a row step (to skip some rows).
        \param [in] threshold - a threshold of anchor creation.
        \param [out] dst - a pointer to pixels data of the output 8-bit gray image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdContourAnchors(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t step, int16_t threshold, uint8_t * dst, size_t dstStride);

    /*! @ingroup correlation

        \fn void SimdSquaredDifferenceSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum);

        \short Calculates sum of squared differences for two 8-bit gray images. 

        All images must have the same width and height. 

        For every point: 
        \n sum += (a[i] - b[i])*(a[i] - b[i]).

        \param [in] a - a pointer to pixels data of the first image.
        \param [in] aStride - a row size of the first image.
        \param [in] b - a pointer to pixels data of the second image.
        \param [in] bStride - a row size of the second image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - a pointer to unsigned 64-bit integer value with result sum.
    */
    SIMD_API void SimdSquaredDifferenceSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        size_t width, size_t height, uint64_t * sum);

    /*! @ingroup correlation

        \fn void SimdSquaredDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum);

        \short Calculates sum of squared differences for two images with using mask. 

        All images must have the same width, height and format (8-bit gray). 

        For every point where mask[i] == index: 
        \n sum += (a[i] - b[i])*(a[i] - b[i]).

        \param [in] a - a pointer to pixels data of the first image.
        \param [in] aStride - a row size of the first image.
        \param [in] b - a pointer to pixels data of the second image.
        \param [in] bStride - a row size of the second image.
        \param [in] mask - a pointer to pixels data of the mask image.
        \param [in] maskStride - a row size of the mask image.
        \param [in] index - a mask index.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - a pointer to unsigned 64-bit integer value with result sum.
    */
    SIMD_API void SimdSquaredDifferenceSumMasked(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        const uint8_t * mask, size_t maskStride, uint8_t index, size_t width, size_t height, uint64_t * sum);

    /*! @ingroup correlation

        \fn float SimdSquaredDifferenceSum32f(const float * a, const float * b, size_t size);

        \short Calculates sum of squared differences for two 32-bit float arrays. 

        All arrays must have the same size. 

        For every point: 
        \n sum += (a[i] - b[i])*(a[i] - b[i]).

        \param [in] a - a pointer to the first array.
        \param [in] b - a pointer to the second array.
        \param [in] size - a size of arrays.
        \return sum of squared differences.
    */
    SIMD_API float SimdSquaredDifferenceSum32f(const float * a, const float * b, size_t size);

    /*! @ingroup other_statistic

        \fn void SimdGetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t * min, uint8_t * max, uint8_t * average);

        \short Finds minimal, maximal and average pixel values for given image. 

        The image must has 8-bit gray format. 

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] stride - a row size of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] min - a pointer to unsigned 8-bit integer value with found minimal pixel value.
        \param [out] max - a pointer to unsigned 8-bit integer value with found maximal pixel value.
        \param [out] average - a pointer to unsigned 8-bit integer value with found average pixel value.
    */
    SIMD_API void SimdGetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height,
        uint8_t * min, uint8_t * max, uint8_t * average);

    /*! @ingroup other_statistic

        \fn void SimdGetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index, uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);

        \short Calculate statistical characteristics (moments) of pixels with given index. 

        The image must has 8-bit gray format.

        For every point where mask[X, Y] == index: 
        \n area += 1.
        \n x += X.
        \n y += Y.
        \n xx += X*X.
        \n xy += X*Y.
        \n yy += Y*Y.

        \param [in] mask - a pointer to pixels data of the mask image.
        \param [in] stride - a row size of the mask image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] index - an mask index.
        \param [out] area - a pointer to unsigned 64-bit integer value with found area (number of pixels with given index).
        \param [out] x - a pointer to unsigned 64-bit integer value with found first-order moment x.
        \param [out] y - a pointer to unsigned 64-bit integer value with found first-order moment y.
        \param [out] xx - a pointer to unsigned 64-bit integer value with found second-order moment xx.
        \param [out] xy - a pointer to unsigned 64-bit integer value with found second-order moment xy.
        \param [out] yy - a pointer to unsigned 64-bit integer value with found second-order moment yy.
    */
    SIMD_API void SimdGetMoments(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
        uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);

    /*! @ingroup row_statistic

        \fn void SimdGetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

        \short Calculate sums of rows for given 8-bit gray image. 

        For all rows: 
        \n sums[y] += src[x, y]; 
        \n where x changes from 0 to width.

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] stride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sums - a pointer to array of unsigned 32-bit integers result sums of rows. It length must be equal to image height.
    */
    SIMD_API void SimdGetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

    /*! @ingroup col_statistic

        \fn void SimdGetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

        \short Calculate sums of columns for given 8-bit gray image. 

        For all columns: 
        \n sums[x] += src[x, y]; 
        \n where y changes from 0 to height.

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] stride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sums - a pointer to array of unsigned 32-bit integers result sums of columns. It length must be equal to image width.
    */
    SIMD_API void SimdGetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

    /*! @ingroup row_statistic

        \fn void SimdGetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

        \short Calculate sums of absolute derivate along y axis for rows for given 8-bit gray image. 

        For all rows except the last: 
        \n sums[y] += abs::(src[x, y+1] - src[x, y]); 
        \n where x changes from 0 to width.
        \n For the last row sums[height-1] = 0; 

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] stride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sums - a pointer to array of unsigned 32-bit integers result sums. It length must be equal to image height.
    */
    SIMD_API void SimdGetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

    /*! @ingroup col_statistic

        \fn void SimdGetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

        \short Calculate sums of absolute derivate along x axis for columns for given 8-bit gray image. 

        For all columns except the last: 
        \n sums[x] += abs::(src[x+1, y] - src[x, y]); 
        \n where y changes from 0 to height.
        \n For the last column sums[width-1] = 0; 

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] stride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sums - a pointer to array of unsigned 32-bit integers result columns. It length must be equal to image width.
    */
    SIMD_API void SimdGetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums);

    /*! @ingroup other_statistic

        \fn void SimdValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

        \short Gets sum of value of pixels for gray 8-bit image. 

        \param [in] src - a pointer to pixels data of the image.
        \param [in] stride - a row size of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - the result sum.
    */
    SIMD_API void SimdValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

    /*! @ingroup other_statistic

        \fn void SimdSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

        \short Gets sum of squared value of pixels for gray 8-bit image . 

        \param [in] src - a pointer to pixels data of the image.
        \param [in] stride - a row size of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - the result sum.
    */
    SIMD_API void SimdSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

    /*! @ingroup resizing

        \fn void SimdStretchGray2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

        \short Stretches input 8-bit gray image in two times. 

        \param [in] src - a pointer to pixels data of the original input image.
        \param [in] srcWidth - a width of the input image.
        \param [in] srcHeight - a height of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [out] dst - a pointer to pixels data of the stretched output image.
        \param [in] dstWidth - a width of the output image.
        \param [in] dstHeight - a height of the output image.
        \param [in] dstStride - a row size of the output image.
    */
    SIMD_API void SimdStretchGray2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

    /*! @ingroup texture_estimation

        \fn void SimdTextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride);

        \short Calculates boosted saturated gradients for given input image. 

        All images must have the same width, height and format (8-bit gray).

        For border pixels dx[x, y] = 0 and dy[x, y] = 0, for other pixels: 
        \n dx[x, y] = (saturation + max(-saturation, min(saturation, (src[x + 1, y] - src[x - 1, y]))))*boost, 
        \n dy[x, y] = (saturation + max(-saturation, min(saturation, (src[x, y + 1] - src[x, y - 1]))))*boost.

        \param [in] src - a pointer to pixels data of source 8-bit gray image.
        \param [in] srcStride - a row size of source image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] saturation - a saturation of gradient.
        \param [in] boost - a boost coefficient.
        \param [out] dx - a pointer to pixels data of image with boosted saturated gradient along x axis.
        \param [in] dxStride - a row size of dx image.
        \param [out] dy - a pointer to pixels data of image with boosted saturated gradient along y axis.
        \param [in] dyStride - a row size of dy image.
    */
    SIMD_API void SimdTextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride);

    /*! @ingroup texture_estimation

        \fn void SimdTextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t boost, uint8_t * dst, size_t dstStride);

        \short Calculates boosted colorized texture feature of input image (actual for U and V components of YUV format). 

        All images must have the same width, height and format (8-bit gray).

        For every pixel: 
        \n dst[x, y] = max(lo, min(hi, src[i]))*boost, 
        \n where lo = 128 - (128/boost), hi = 255 - lo. 

        \param [in] src - a pointer to pixels data of source 8-bit gray image.
        \param [in] srcStride - a row size of source image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] boost - a boost coefficient.
        \param [out] dst - a pointer to pixels data of result image.
        \param [in] dstStride - a row size of destination image.
    */
    SIMD_API void SimdTextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t boost, uint8_t * dst, size_t dstStride);

    /*! @ingroup texture_estimation

        \fn void SimdTextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum);

        \short Calculates difference between current image and background. 

        All images must have the same width, height and format (8-bit gray).

        For every pixel: 
        \n sum += current - average(lo[i], hi[i]);

        \param [in] src - a pointer to pixels data of current image.
        \param [in] srcStride - a row size of current image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] lo - a pointer to pixels data of image with lower bound of background feature.
        \param [in] loStride - a row size of lo image.
        \param [in] hi - a pointer to pixels data of image with upper bound of background feature.
        \param [in] hiStride - a row size of hi image.
        \param [out] sum - a pointer to 64-bit integer with result sum.
    */
    SIMD_API void SimdTextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum);

    /*! @ingroup texture_estimation

        \fn void SimdTexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height, int32_t shift, uint8_t * dst, size_t dstStride);

        \short Performs brightness compensation of input image. 

        All images must have the same width, height and format (8-bit gray).

        For every pixel: 
        \n dst[i] = max(0, min(255, src[i] + shift));

        \param [in] src - a pointer to pixels data of input image.
        \param [in] srcStride - a row size of input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] shift - a compensation shift.
        \param [out] dst - a pointer to pixels data of output image.
        \param [in] dstStride - a row size of output image.
    */
    SIMD_API void SimdTexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        int32_t shift, uint8_t * dst, size_t dstStride);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

        \short Converts YUV420P image to 24-bit BGR image. 

        The input Y and output BGR images must have the same width and height. 
        The input U and V images must have the same width and height (half size relative to Y component). 

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] bgr - a pointer to pixels data of output 24-bit BGR image.
        \param [in] bgrStride - a row size of the bgr image.
    */
    SIMD_API void SimdYuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv444pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

        \short Converts YUV444P image to 24-bit BGR image. 

        The input Y, U, V and output BGR images must have the same width and height. 

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] bgr - a pointer to pixels data of output 24-bit BGR image.
        \param [in] bgrStride - a row size of the bgr image.
    */
    SIMD_API void SimdYuv444pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

        \short Converts YUV420P image to 32-bit BGRA image. 

        The input Y and output BGRA images must have the same width and height. 
        The input U and V images must have the same width and height (half size relative to Y component). 

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] bgra - a pointer to pixels data of output 32-bit BGRA image.
        \param [in] bgraStride - a row size of the bgra image.
        \param [in] alpha - a value of alpha channel.
    */
    SIMD_API void SimdYuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

        \short Converts YUV444P image to 32-bit BGRA image. 

        The input Y, U, V and output BGRA images must have the same width and height. 

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] bgra - a pointer to pixels data of output 32-bit BGRA image.
        \param [in] bgraStride - a row size of the bgra image.
        \param [in] alpha - a value of alpha channel.
    */
    SIMD_API void SimdYuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv444pToHsl(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * hsl, size_t hslStride);

        \short Converts YUV444P image to 24-bit HSL(Hue, Saturation, Lightness) image. 

        The input Y, U, V and output HSL images must have the same width and height. 

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] hsl - a pointer to pixels data of output 24-bit HSL image.
        \param [in] hslStride - a row size of the hsl image.
    */
    SIMD_API void SimdYuv444pToHsl(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * hsl, size_t hslStride);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv444pToHsv(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * hsv, size_t hsvStride);

        \short Converts YUV444P image to 24-bit HSV(Hue, Saturation, Value) image. 

        The input Y, U, V and output HSV images must have the same width and height. 

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] hsv - a pointer to pixels data of output 24-bit HSV image.
        \param [in] hsvStride - a row size of the hsv image.
    */
    SIMD_API void SimdYuv444pToHsv(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * hsv, size_t hsvStride);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * hue, size_t hueStride);

        \short Converts YUV420P image to 8-bit image with Hue component of HSV or HSL color space. 

        The input Y and output Hue images must have the same width and height. 
        The input U and V images must have the same width and height (half size relative to Y component). 

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] hue - a pointer to pixels data of output 8-bit Hue image.
        \param [in] hueStride - a row size of the hue image.
    */
    SIMD_API void SimdYuv420pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * hue, size_t hueStride);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv444pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * hue, size_t hueStride);

        \short Converts YUV444P image to 8-bit image with Hue component of HSV or HSL color space. 

        The input Y, U, V and output Hue images must have the same width and height. 

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] hue - a pointer to pixels data of output 8-bit Hue image.
        \param [in] hueStride - a row size of the hue image.
    */
    SIMD_API void SimdYuv444pToHue(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * hue, size_t hueStride);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif//__SimdLib_h__
