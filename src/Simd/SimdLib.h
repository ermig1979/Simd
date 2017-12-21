/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar,
*               2014-2016 Antonenka Mikhail.
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

#include "Simd/SimdConfig.h"

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
    Describes types of SIMD extensions which supported by current CPU and Simd Library (see function ::SimdCpuInfo).
*/
typedef enum
{
    SimdCpuInfoSse = 0, /*!< SSE (x86). */
    SimdCpuInfoSse2, /*!< SSE2 (x86). */
    SimdCpuInfoSse3, /*!< SSE3 (x86). */
    SimdCpuInfoSsse3, /*!< SSSE3 (x86). */
    SimdCpuInfoSse41, /*!< SSE4.1 (x86). */
    SimdCpuInfoSse42, /*!< SSE4.2 (x86). */
    SimdCpuInfoAvx, /*!< AVX (x86). */
    SimdCpuInfoAvx2, /*!< AVX2 (x86). */
    SimdCpuInfoAvx512f, /*!< AVX-512F (x86). */
    SimdCpuInfoAvx512bw, /*!< AVX-512BW (x86). */
    SimdCpuInfoVmx, /*!< VMX or Altivec (PowerPC). */
    SimdCpuInfoVsx, /*!< VSX (PowerPC). */
    SimdCpuInfoNeon, /*!< NEON (ARM). */
    SimdCpuInfoMsa, /*!< MSA (MIPS). */
} SimdCpuInfoFlags;

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
    /*! Computes the bitwise OR between two images. */
    SimdOperationBinary8uOr,
    /*! Computes maximal value for every channel of every point of two images. */
    SimdOperationBinary8uMaximum,
    /*! Computes minimal value for every channel of every point of two images. */
    SimdOperationBinary8uMinimum,
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
    /*! Performs addition of two images for every point.  */
    SimdOperationBinary16iAddition,
    /*! Performs subtraction of two images for every point.  */
    SimdOperationBinary16iSubtraction,
} SimdOperationBinary16iType;

/*! @ingroup c_types
    Describes pixel format types of an image.
    In particular this type is used in functions ::SimdBayerToBgr, ::SimdBayerToBgra, ::SimdBgraToBayer and ::SimdBgrToBayer.
    \note This type is corresponds to C++ type Simd::View::Format.
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

/*! @ingroup c_types
    Describes types and flags to get information about classifier cascade with using function ::SimdDetectionInfo.
    \note This type is used for implementation of Simd::Detection.
*/
typedef enum
{
    /*! A HAAR cascade classifier type. */
    SimdDetectionInfoFeatureHaar = 0,
    /*! A LBP cascade classifier type. */
    SimdDetectionInfoFeatureLbp,
    /*! A mask to select cascade classifier type. */
    SimdDetectionInfoFeatureMask = 3,
    /*! A flag which defines existence of tilted features in the HAAR cascade. */
    SimdDetectionInfoHasTilted = 4,
    /*! A flag which defines possibility to use 16-bit integers for calculation. */
    SimdDetectionInfoCanInt16 = 8,
} SimdDetectionInfoFlags;

/*! @ingroup c_types
    Describes type of algorithm used for image reducing (downscale in 2 times) (see function Simd::ReduceGray).
*/
enum SimdReduceType
{
    SimdReduce2x2, /*!< Using of function ::SimdReduceGray2x2 for image reducing. */
    SimdReduce3x3, /*!< Using of function ::SimdReduceGray3x3 for image reducing. */
    SimdReduce4x4, /*!< Using of function ::SimdReduceGray4x4 for image reducing. */
    SimdReduce5x5, /*!< Using of function ::SimdReduceGray5x5 for image reducing. */
};

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

        \short Gets version of %Simd Library.

        \return string with version of %Simd Library (major version number, minor version number, release number, number of SVN's commits).
    */
    SIMD_API const char * SimdVersion();

    /*! @ingroup info

        \fn int SimdCpuInfo();

        \short Gets info about SIMD extensions supported by CPU and %Simd Library.

        \note See enumeration ::SimdCpuInfoFlags.

        Using example:
        \verbatim
        #include "Simd/SimdLib.h"
        #include <iostream>

        int main()
        {
            int info = SimdCpuInfo();
            std::cout << "SSE: " << (info&(1 << SimdCpuInfoSse) ? "Yes" : "No") << std::endl;
            std::cout << "SSE2: " << (info&(1 << SimdCpuInfoSse2) ? "Yes" : "No") << std::endl;
            std::cout << "SSE3: " << (info&(1 << SimdCpuInfoSse3) ? "Yes" : "No") << std::endl;
            std::cout << "SSSE3: " << (info&(1 << SimdCpuInfoSsse3) ? "Yes" : "No") << std::endl;
            std::cout << "SSE4.1: " << (info&(1 << SimdCpuInfoSse41) ? "Yes" : "No") << std::endl;
            std::cout << "SSE4.2: " << (info&(1 << SimdCpuInfoSse42) ? "Yes" : "No") << std::endl;
            std::cout << "AVX: " << (info&(1 << SimdCpuInfoAvx) ? "Yes" : "No") << std::endl;
            std::cout << "AVX2: " << (info&(1 << SimdCpuInfoAvx2) ? "Yes" : "No") << std::endl;
            std::cout << "AVX-512F: " << (info&(1 << SimdCpuInfoAvx512f) ? "Yes" : "No") << std::endl;
            std::cout << "AVX-512BW: " << (info&(1 << SimdCpuInfoAvx512bw) ? "Yes" : "No") << std::endl;
            std::cout << "PowerPC-Altivec: " << (info&(1 << SimdCpuInfoVmx) ? "Yes" : "No") << std::endl;
            std::cout << "PowerPC-VSX: " << (info&(1 << SimdCpuInfoVsx) ? "Yes" : "No") << std::endl;
            std::cout << "ARM-NEON: " << (info&(1 << SimdCpuInfoNeon) ? "Yes" : "No") << std::endl;
            std::cout << "MIPS-MSA: " << (info&(1 << SimdCpuInfoMsa) ? "Yes" : "No") << std::endl;
            return 0;
        }
        \endverbatim

        \return an integer value which bits contains information about SIMD extensions supported by CPU and %Simd Library.
    */
    SIMD_API int SimdCpuInfo();

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

        \short Gets alignment required for the most productive work of the Simd Library.

        \return a required alignment.
    */
    SIMD_API size_t SimdAlignment();

    /*! @ingroup hash

        \fn uint32_t SimdCrc32c(const void * src, size_t size);

        \short Gets 32-bit cyclic redundancy check (CRC32c) for current data.

        Calculation is performed for polynomial 0x1EDC6F41 (Castagnoli-crc).

        \param [in] src - a pointer to data.
        \param [in] size - a size of the data.
        \return 32-bit cyclic redundancy check (CRC32c).
    */
    SIMD_API uint32_t SimdCrc32c(const void * src, size_t size);

    /*! @ingroup correlation

        \fn void SimdAbsDifferenceSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum);

        \short Gets sum of absolute difference of two gray 8-bit images.

        Both images must have the same width and height.

        \note This function has a C++ wrapper Simd::AbsDifferenceSum(const View<A> & a, const View<A> & b, uint64_t & sum).

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

        \note This function has a C++ wrapper Simd::AbsDifferenceSum(const View<A>& a, const View<A>& b, const View<A>& mask, uint8_t index, uint64_t & sum).

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

        \note This function has a C++ wrapper Simd::AbsDifferenceSums3x3(const View<A>& current, const View<A>& background, uint64_t * sums).

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

        \note This function has a C++ wrapper Simd::AbsDifferenceSums3x3(const View<A>& current, const View<A>& background, const View<A>& mask, uint8_t index, uint64_t * sums).

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

        For border pixels:
        \verbatim
        dst[x, y] = 0;
        \endverbatim

        For other pixels:
        \verbatim
        dx = abs(src[x + 1, y] - src[x - 1, y]);
        dy = abs(src[x, y + 1] - src[x, y - 1]);
        dst[x, y] = min(dx + dy, 255);
        \endverbatim

        \note This function has a C++ wrapper Simd::AbsGradientSaturatedSum(const View<A>& src, View<A>& dst).

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
        \verbatim
        excess = max(lo[i] - value[i], 0) + max(value[i] - hi[i], 0);
        difference[i] += (weight * excess*excess) >> 16;
        \endverbatim

        This function is used for difference estimation in algorithm of motion detection.

        \note This function has a C++ wrapper Simd::AddFeatureDifference(const View<A>& value, const View<A>& lo, const View<A>& hi, uint16_t weight, View<A>& difference).

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
        \verbatim
        dst[x, y, c] = (src[x, y, c]*alpha[x, y] + dst[x, y, c]*(255 - alpha[x, y]))/255;
        \endverbatim

        This function is used for image drawing.

        \note This function has a C++ wrapper Simd::AlphaBlending(const View<A>& src, const View<A>& alpha, View<A>& dst).

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

    /*! @ingroup drawing

        \fn void SimdAlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, size_t channelCount, const uint8_t * alpha, size_t alphaStride);

        \short Performs alpha filling operation.

        All images must have the same width and height. Destination images must have 8 bit per channel (for example GRAY8, BGR24 or BGRA32). Alpha must be 8-bit gray image.

        For every point:
        \verbatim
        dst[x, y, c] = (channel[c]*alpha[x, y] + dst[x, y, c]*(255 - alpha[x, y]))/255;
        \endverbatim

        This function is used for image drawing.

        \note This function has a C++ wrapper Simd::AlphaFilling(View<A> & dst, const Pixel & pixel, const View<A> & alpha).

        \param [in, out] dst - a pointer to pixels data of background image.
        \param [in] dstStride - a row size of the background image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channel - a pointer to pixel with foreground color.
        \param [in] channelCount - a channel count for foreground color and background images (1 <= channelCount <= 4).
        \param [in] alpha - a pointer to pixels data of image with alpha channel.
        \param [in] alphaStride - a row size of the alpha image.
    */
    SIMD_API void SimdAlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, size_t channelCount, const uint8_t * alpha, size_t alphaStride);

    /*! @ingroup background

        \fn void SimdBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

        \short Performs background update (initial grow, slow mode).

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        lo[i] -= value[i] < lo[i] ? 1 : 0;
        hi[i] += value[i] > hi[i] ? 1 : 0;
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::BackgroundGrowRangeSlow(const View<A>& value, View<A>& lo, View<A>& hi).

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] lo - a pointer to pixels data of feature lower bound of dynamic background.
        \param [in] loStride - a row size of the lo image.
        \param [in, out] hi - a pointer to pixels data of feature upper bound of dynamic background.
        \param [in] hiStride - a row size of the hi image.
    */
    SIMD_API void SimdBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

    /*! @ingroup background

        \fn void SimdBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

        \short Performs background update (initial grow, fast mode).

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        lo[i] = value[i] < lo[i] ? value[i] : lo[i];
        hi[i] = value[i] > hi[i] ? value[i] : hi[i];
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::BackgroundGrowRangeFast(const View<A>& value, View<A>& lo, View<A>& hi).

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] lo - a pointer to pixels data of feature lower bound of dynamic background.
        \param [in] loStride - a row size of the lo image.
        \param [in, out] hi - a pointer to pixels data of feature upper bound of dynamic background.
        \param [in] hiStride - a row size of the hi image.
    */
    SIMD_API void SimdBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

    /*! @ingroup background

        \fn void SimdBackgroundIncrementCount(const uint8_t * value, size_t valueStride, size_t width, size_t height, const uint8_t * loValue, size_t loValueStride, const uint8_t * hiValue, size_t hiValueStride, uint8_t * loCount, size_t loCountStride, uint8_t * hiCount, size_t hiCountStride);

        \short Performs collection of background statistic.

        All images must have the same width, height and format (8-bit gray).

        Updates background statistic counters for every point:
        \verbatim
        loCount[i] += (value[i] < loValue[i] && loCount[i] < 255) ? 1 : 0;
        hiCount[i] += (value[i] > hiValue[i] && hiCount[i] < 255) ? 1 : 0;
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::BackgroundIncrementCount(const View<A>& value, const View<A>& loValue, const View<A>& hiValue, View<A>& loCount, View<A>& hiCount).

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
        \verbatim
        loValue[i] -= (loCount[i] > threshold && loValue[i] > 0) ? 1 : 0;
        loValue[i] += (loCount[i] < threshold && loValue[i] < 255) ? 1 : 0;
        loCount[i] = 0;
        hiValue[i] += (hiCount[i] > threshold && hiValue[i] < 255) ? 1 : 0;
        hiValue[i] -= (hiCount[i] < threshold && hiValue[i] > 0) ? 1 : 0;
        hiCount[i] = 0;
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::BackgroundAdjustRange(View<A>& loCount, View<A>& loValue, View<A>& hiCount, View<A>& hiValue, uint8_t threshold).

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

        Adjusts background range for every point:
        \verbatim
        if(mask[i])
        {
            loValue[i] -= (loCount[i] > threshold && loValue[i] > 0) ? 1 : 0;
            loValue[i] += (loCount[i] < threshold && loValue[i] < 255) ? 1 : 0;
            loCount[i] = 0;
            hiValue[i] += (hiCount[i] > threshold && hiValue[i] < 255) ? 1 : 0;
            hiValue[i] -= (hiCount[i] < threshold && hiValue[i] > 0) ? 1 : 0;
            hiCount[i] = 0;
        }
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::BackgroundAdjustRange(View<A>& loCount, View<A>& loValue, View<A>& hiCount, View<A>& hiValue, uint8_t threshold, const View<A>& mask).

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
            lo[i] = min(lo[i] + value[i] - hi[i], 255);
            hi[i] = value[i];
        }
        if (lo[i] > value[i])
        {
            lo[i] = value[i];
            hi[i] = max(hi[i] - lo[i] + value[i], 0);
        }
        \endverbatim

        This function is used for fast background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::BackgroundShiftRange(const View<A>& value, View<A>& lo, View<A>& hi).

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] lo - a pointer to pixels data of feature lower bound of dynamic background.
        \param [in] loStride - a row size of the lo image.
        \param [in, out] hi - a pointer to pixels data of feature upper bound of dynamic background.
        \param [in] hiStride - a row size of the hi image.
    */
    SIMD_API void SimdBackgroundShiftRange(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride);

    /*! @ingroup background

        \fn void SimdBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride);

        \short Shifts background range with using shift range mask.

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        if(mask[i])
        {
            if (value[i] > hi[i])
            {
                lo[i] = min(lo[i] + value[i] - hi[i], 255);
                hi[i] = value[i];
            }
            if (lo[i] > value[i])
            {
                lo[i] = value[i];
                hi[i] = max(hi[i] - lo[i] + value[i], 0);
            }
        }
        \endverbatim

        This function is used for fast background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::BackgroundShiftRange(const View<A>& value, View<A>& lo, View<A>& hi, const View<A>& mask).

        \param [in] value - a pointer to pixels data of current feature value.
        \param [in] valueStride - a row size of the value image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in, out] lo - a pointer to pixels data of feature lower bound of dynamic background.
        \param [in] loStride - a row size of the lo image.
        \param [in, out] hi - a pointer to pixels data of feature upper bound of dynamic background.
        \param [in] hiStride - a row size of the hi image.
        \param [in] mask - a pointer to pixels data of shift range mask.
        \param [in] maskStride - a row size of the mask image.
    */
    SIMD_API void SimdBackgroundShiftRangeMasked(const uint8_t * value, size_t valueStride, size_t width, size_t height,
        uint8_t * lo, size_t loStride, uint8_t * hi, size_t hiStride, const uint8_t * mask, size_t maskStride);

    /*! @ingroup background

        \fn void SimdBackgroundInitMask(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t index, uint8_t value, uint8_t * dst, size_t dstStride);

        \short Creates background update mask.

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        if(mask[i] == index)
            dst[i] = value;
        \endverbatim

        This function is used for background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::BackgroundInitMask(const View<A>& src, uint8_t index, uint8_t value, View<A>& dst).

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

        \note This function has a C++ wrapper Simd::BayerToBgr(const View<A>& bayer, View<A>& bgr).

        \param [in] bayer - a pointer to pixels data of input 8-bit Bayer image.
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

        \note This function has a C++ wrapper Simd::BayerToBgra(const View<A>& bayer, View<A>& bgra, uint8_t alpha).

        \param [in] bayer - a pointer to pixels data of input 8-bit Bayer image.
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

        \note This function has a C++ wrapper Simd::BgraToBayer(const View<A>& bgra, View<A>& bayer).

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

        \note This function has a C++ wrapper Simd::BgraToBgr(const View<A>& bgra, View<A>& bgr).

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

        \note This function has a C++ wrapper Simd::BgraToGray(const View<A>& bgra, View<A>& gray).

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgraStride - a row size of the bgra image.
        \param [out] gray - a pointer to pixels data of output 8-bit gray image.
        \param [in] grayStride - a row size of the gray image.
    */
    SIMD_API void SimdBgraToGray(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * gray, size_t grayStride);

    /*! @ingroup bgra_conversion

        \fn void SimdBgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Converts 32-bit BGRA image to YUV420P.

        The input BGRA and output Y images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function has a C++ wrapper Simd::BgraToYuv420p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v).

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgraStride - a row size of the BGRA image.
        \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
    */
    SIMD_API void SimdBgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup bgra_conversion

        \fn void SimdBgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Converts 32-bit BGRA image to YUV422P.

        The input BGRA and output Y images must have the same width and height.
        The input U and V images must have the same width and height (their width is equal to half width of Y component).

        \note This function has a C++ wrapper Simd::BgraToYuv422p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v).

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgraStride - a row size of the BGRA image.
        \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
    */
    SIMD_API void SimdBgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup bgra_conversion

        \fn void SimdBgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Converts 32-bit BGRA image to YUV444P.

        The input BGRA and output Y, U and V images must have the same width and height.

        \note This function has a C++ wrapper Simd::BgraToYuv444p(const View<A>& bgra, View<A>& y, View<A>& u, View<A>& v).

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgraStride - a row size of the BGRA image.
        \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
    */
    SIMD_API void SimdBgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup bgr_conversion

        \fn void SimdBgrToBayer(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bayer, size_t bayerStride, SimdPixelFormatType bayerFormat);

        \short Converts 24-bit BGR image to 8-bit Bayer image.

        All images must have the same width and height. The width and the height must be even.

        \note This function has a C++ wrapper Simd::BgrToBayer(const View<A>& bgr, View<A>& bayer).

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

        \note This function has a C++ wrapper Simd::BgrToBgra(const View<A>& bgr, View<A>& bgra, uint8_t alpha).

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

        \note This function has a C++ wrapper Simd::Bgr48pToBgra32(const View<A>& blue, const View<A>& green, const View<A>& red, View<A>& bgra, uint8_t alpha).

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

        \note This function has a C++ wrapper Simd::BgrToGray(const View<A>& bgr, View<A>& gray).

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

        \note This function has a C++ wrapper Simd::BgrToHsl(const View<A>& bgr, View<A>& hsl).

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

        \note This function has a C++ wrapper Simd::BgrToHsv(const View<A>& bgr, View<A>& hsv).

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgrStride - a row size of the bgr image.
        \param [out] hsv - a pointer to pixels data of output 24-bit HSV image.
        \param [in] hsvStride - a row size of the hsv image.
    */
    SIMD_API void SimdBgrToHsv(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * hsv, size_t hsvStride);

    /*! @ingroup bgr_conversion

        \fn void SimdBgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Converts 24-bit BGR image to YUV420P.

        The input BGR and output Y images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function has a C++ wrapper Simd::BgrToYuv420p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v).

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgrStride - a row size of the BGR image.
        \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
    */
    SIMD_API void SimdBgrToYuv420p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup bgr_conversion

        \fn void SimdBgrToYuv422p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Converts 24-bit BGR image to YUV422P.

        The input BGR and output Y images must have the same width and height.
        The input U and V images must have the same width and height (their width is equal to half width of Y component).

        \note This function has a C++ wrapper Simd::BgrToYuv422p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v).

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgrStride - a row size of the BGR image.
        \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
    */
    SIMD_API void SimdBgrToYuv422p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup bgr_conversion

        \fn void SimdBgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

        \short Converts 24-bit BGR image to YUV444P.

        The input BGR and output Y, U and V images must have the same width and height.

        \note This function has a C++ wrapper Simd::BgrToYuv444p(const View<A>& bgr, View<A>& y, View<A>& u, View<A>& v).

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgrStride - a row size of the BGR image.
        \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
    */
    SIMD_API void SimdBgrToYuv444p(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

    /*! @ingroup binarization

        \fn void SimdBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType compareType);

        \short Performs binarization of 8-bit gray image.

        All images must have 8-bit gray format and must have the same width and height.

        For every point:
        \verbatim
        dst[i] = compare(src[i], value) ? positive : negative;
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function has a C++ wrapper Simd::Binarization(const View<A>& src, uint8_t value, uint8_t positive, uint8_t negative, View<A>& dst, SimdCompareType compareType).

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
        \verbatim
        sum = 0; area = 0;
        for(dy = -neighborhood; dy <= neighborhood; ++dy)
        {
            for(dx = -neighborhood; dx <= neighborhood; ++dx)
            {
                if(x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height)
                {
                    area++;
                    if(compare(src[x + dx, x + dy], value))
                        sum++;
                }
            }
        }
        dst[x, y] = sum*255 > area*threshold ? positive : negative;
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function has a C++ wrapper Simd::AveragingBinarization(const View<A>& src, uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, View<A>& dst, SimdCompareType compareType).

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
        \verbatim
        if(compare(src[i], value))
            count++;
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function has a C++ wrapper Simd::ConditionalCount8u(const View<A> & src, uint8_t value, SimdCompareType compareType, uint32_t & count).

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
        \verbatim
        if(compare(src[i], value))
            count++;
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function has a C++ wrapper Simd::ConditionalCount16i(const View<A> & src, int16_t value, SimdCompareType compareType, uint32_t & count).

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
        \verbatim
        if(compare(mask[i], value))
            sum += src[i];
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function has a C++ wrapper Simd::ConditionalSum(const View<A> & src, const View<A> & mask, uint8_t value, SimdCompareType compareType, uint64_t & sum).

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
        \verbatim
        if(compare(mask[i], value))
            sum += src[i]*src[i];
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function has a C++ wrapper Simd::ConditionalSquareSum(const View<A> & src, const View<A> & mask, uint8_t value, SimdCompareType compareType, uint64_t & sum).

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

        For every point except border:
        \verbatim
        if(compare(mask[x, y], value))
        {
            dx = src[x + 1, y] - src[x - 1, y];
            dy = src[x, y + 1] - src[x, y - 1];
            sum += dx*dx + dy*dy;
        }
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function has a C++ wrapper Simd::ConditionalSquareGradientSum(const View<A> & src, const View<A> & mask, uint8_t value, SimdCompareType compareType, uint64_t & sum).

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

    /*! @ingroup conditional

        \fn void SimdConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t threshold, SimdCompareType compareType, uint8_t value, uint8_t * dst, size_t dstStride);

        \short Fills pixels of 8-bit gray image by given value if corresponding pixels of input 8-bit gray image satisfy certain condition.

        All images must have the same width and height.

        For every point:
        \verbatim
        if(compare(src[i], threshold))
            dst[i] = value;
        \endverbatim
        where compare(a, b) depends from compareType (see ::SimdCompareType).

        \note This function has a C++ wrapper Simd::ConditionalFill(const View<A> & src, uint8_t threshold, SimdCompareType compareType, uint8_t value, View<A> & dst).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] threshold - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [in] value - a value for fill operation.
        \param [in, out] dst - a pointer to pixels data of the output 8-bit gray image.
        \param [in] dstStride - a row size of output image.
    */
    SIMD_API void SimdConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        uint8_t threshold, SimdCompareType compareType, uint8_t value, uint8_t * dst, size_t dstStride);

    /*! @ingroup copying

        \fn void SimdCopy(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, uint8_t * dst, size_t dstStride);

        \short Copies pixels data of image from source to destination.

        All images must have the same width, height and format.

        \note This function has a C++ wrapper Simd::Copy(const View<A> & src, View<B> & dst).

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

        \note This function has a C++ wrapper Simd::CopyFrame(const View<A>& src, const Rectangle<ptrdiff_t> & frame, View<A>& dst).

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

        \note This function has a C++ wrapper Simd::DeinterleaveUv(const View<A>& uv, View<A>& u, View<A>& v).

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

    /*! @ingroup other_conversion

        \fn void SimdDeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height, uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride);

        \short Deinterleaves 24-bit BGR interleaved image into separated 8-bit Blue, Green and Red planar images.

        All images must have the same width and height.

        \note This function has a C++ wrapper Simd::DeinterleaveBgr(const View<A>& bgr, View<A>& b, View<A>& g, View<A>& r).

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR interleaved image.
        \param [in] bgrStride - a row size of the bgr image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] b - a pointer to pixels data of 8-bit Blue planar image.
        \param [in] bStride - a row size of the b image.
        \param [out] g - a pointer to pixels data of 8-bit Green planar image.
        \param [in] gStride - a row size of the g image.
        \param [out] r - a pointer to pixels data of 8-bit Red planar image.
        \param [in] rStride - a row size of the r image.
    */
    SIMD_API void SimdDeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
        uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride);

    /*! @ingroup other_conversion

        \fn void SimdDeinterleaveBgra(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride, uint8_t * a, size_t aStride);

        \short Deinterleaves 32-bit BGRA interleaved image into separated 8-bit Blue, Green, Red and Alpha planar images.

        All images must have the same width and height.

        \note This function has a C++ wrapper Simd::DeinterleaveBgra(const View<A>& bgra, View<A>& b, View<A>& g, View<A>& r, View<A>& a).

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA interleaved image.
        \param [in] bgraStride - a row size of the bgra image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] b - a pointer to pixels data of 8-bit Blue planar image.
        \param [in] bStride - a row size of the b image.
        \param [out] g - a pointer to pixels data of 8-bit Green planar image.
        \param [in] gStride - a row size of the g image.
        \param [out] r - a pointer to pixels data of 8-bit Red planar image.
        \param [in] rStride - a row size of the r image.
        \param [out] a - a pointer to pixels data of 8-bit Alpha planar image.
        \param [in] aStride - a row size of the a image.
    */
    SIMD_API void SimdDeinterleaveBgra(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height,
        uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride, uint8_t * a, size_t aStride);

    /*! @ingroup object_detection

        \fn void * SimdDetectionLoadA(const char * path);

        \short Loads a classifier cascade from file.

        This function supports OpenCV HAAR and LBP cascades type.
        Tree based cascades and old cascade formats are not supported.

        \note This function is used for implementation of Simd::Detection.

        \param [in] path - a path to cascade.
        \return a pointer to loaded cascade. On error it returns NULL.
                This pointer is used in functions ::SimdDetectionInfo and ::SimdDetectionInit, and must be released with using function ::SimdDetectionFree.
    */
    SIMD_API void * SimdDetectionLoadA(const char * path);

    /*! @ingroup object_detection

        \fn void SimdDetectionInfo(const void * data, size_t * width, size_t * height, SimdDetectionInfoFlags * flags);

        \short Gets information about the classifier cascade.

        \note This function is used for implementation of Simd::Detection.

        \param [in] data - a pointer to cascade which was received with using of function ::SimdDetectionLoadA.
        \param [out] width - a pointer to returned width of cascade window.
        \param [out] height - a pointer to returned height of cascade window.
        \param [out] flags - a pointer to flags with other information (See ::SimdDetectionInfoFlags).
    */
    SIMD_API void SimdDetectionInfo(const void * data, size_t * width, size_t * height, SimdDetectionInfoFlags * flags);

    /*! @ingroup object_detection

        \fn void * SimdDetectionInit(const void * data, uint8_t * sum, size_t sumStride, size_t width, size_t height, uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride, int throughColumn, int int16);

        \short Initializes hidden classifier cascade structure to work with given size of input 8-bit gray image.

        \note This function is used for implementation of Simd::Detection.

        \param [in] data - a pointer to cascade which was received with using of function ::SimdDetectionLoadA.
        \param [in] sum - a pointer to pixels data of 32-bit integer image with integral sum of given input 8-bit gray image.
                          See function ::SimdIntegral in order to estimate this integral sum.
        \param [in] sumStride - a row size of the sum image.
        \param [in] width - a width of the sum image. It must be per unit greater than width of input 8-bit gray image.
        \param [in] height - a height of the sum image. It must be per unit greater than height of input 8-bit gray image.
        \param [in] sqsum - a pointer to pixels data of 32-bit integer image with squared integral sum of given input 8-bit gray image.
                            Its size must be equal to sum image. See function ::SimdIntegral in order to estimate this squared integral sum. Its
        \param [in] sqsumStride - a row size of the sqsum image.
        \param [in] tilted - a pointer to pixels data of 32-bit integer image with tilted integral sum of given input 8-bit gray image.
                             Its size must be equal to sum image. See function ::SimdIntegral in order to estimate this tilted integral sum.
        \param [in] tiltedStride - a row size of the tilted image.
        \param [in] throughColumn - a flag to detect objects only in even columns and rows (to increase performance).
        \param [in] int16 - a flag use for 16-bit integer version of detection algorithm. (See ::SimdDetectionInfo).
        \return a pointer to hidden cascade. On error it returns NULL.
                This pointer is used in functions ::SimdDetectionPrepare, ::SimdDetectionHaarDetect32fp, ::SimdDetectionHaarDetect32fi,
                ::SimdDetectionLbpDetect32fp, ::SimdDetectionLbpDetect32fi, ::SimdDetectionLbpDetect16ip and ::SimdDetectionLbpDetect16ii.
                It must be released with using function ::SimdDetectionFree.
    */
    SIMD_API void * SimdDetectionInit(const void * data, uint8_t * sum, size_t sumStride, size_t width, size_t height,
        uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride, int throughColumn, int int16);

    /*! @ingroup object_detection

        \fn void SimdDetectionPrepare(void * hid);

        \short Prepares hidden classifier cascade structure to work with given input 8-bit gray image.

        You must call this function before calling of functions ::SimdDetectionHaarDetect32fp, ::SimdDetectionHaarDetect32fi,
         ::SimdDetectionLbpDetect32fp, ::SimdDetectionLbpDetect32fi, ::SimdDetectionLbpDetect16ip and ::SimdDetectionLbpDetect16ii.

        \note This function is used for implementation of Simd::Detection.

        \param [in] hid - a pointer to hidden cascade which was received with using of function ::SimdDetectionInit.
    */
    SIMD_API void SimdDetectionPrepare(void * hid);

    /*! @ingroup object_detection

        \fn void SimdDetectionHaarDetect32fp(const void * hid, const uint8_t * mask, size_t maskStride, ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

        \short Performs object detection with using of HAAR cascade classifier (uses 32-bit float numbers, processes all points).

        You must call function ::SimdDetectionPrepare before calling of this functions.
        All restriction (input mask and bounding box) affects to left-top corner of scanning window.

        \note This function is used for implementation of Simd::Detection.

        \param [in] hid - a pointer to hidden cascade which was received with using of function ::SimdDetectionInit.
        \param [in] mask - a pointer to pixels data of 8-bit image with mask. The mask restricts detection region.
        \param [in] maskStride - a row size of the mask image.
        \param [in] left - a left side of bounding box which restricts detection region.
        \param [in] top - a top side of bounding box which restricts detection region.
        \param [in] right - a right side of bounding box which restricts detection region.
        \param [in] bottom - a bottom side of bounding box which restricts detection region.
        \param [out] dst - a pointer to pixels data of 8-bit image with output result. None zero points refer to left-top corner of detected objects.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdDetectionHaarDetect32fp(const void * hid, const uint8_t * mask, size_t maskStride,
        ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

    /*! @ingroup object_detection

        \fn void SimdDetectionHaarDetect32fi(const void * hid, const uint8_t * mask, size_t maskStride, ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

        \short Performs object detection with using of HAAR cascade classifier (uses 32-bit float numbers, processes only even points).

        You must call function ::SimdDetectionPrepare before calling of this functions.
        All restriction (input mask and bounding box) affects to left-top corner of scanning window.

        \note This function is used for implementation of Simd::Detection.

        \param [in] hid - a pointer to hidden cascade which was received with using of function ::SimdDetectionInit.
        \param [in] mask - a pointer to pixels data of 8-bit image with mask. The mask restricts detection region.
        \param [in] maskStride - a row size of the mask image.
        \param [in] left - a left side of bounding box which restricts detection region.
        \param [in] top - a top side of bounding box which restricts detection region.
        \param [in] right - a right side of bounding box which restricts detection region.
        \param [in] bottom - a bottom side of bounding box which restricts detection region.
        \param [out] dst - a pointer to pixels data of 8-bit image with output result. None zero points refer to left-top corner of detected objects.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdDetectionHaarDetect32fi(const void * hid, const uint8_t * mask, size_t maskStride,
        ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

    /*! @ingroup object_detection

        \fn void SimdDetectionLbpDetect32fp(const void * hid, const uint8_t * mask, size_t maskStride, ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

        \short Performs object detection with using of LBP cascade classifier (uses 32-bit float numbers, processes all points).

        You must call function ::SimdDetectionPrepare before calling of this functions.
        All restriction (input mask and bounding box) affects to left-top corner of scanning window.

        \note This function is used for implementation of Simd::Detection.

        \param [in] hid - a pointer to hidden cascade which was received with using of function ::SimdDetectionInit.
        \param [in] mask - a pointer to pixels data of 8-bit image with mask. The mask restricts detection region.
        \param [in] maskStride - a row size of the mask image.
        \param [in] left - a left side of bounding box which restricts detection region.
        \param [in] top - a top side of bounding box which restricts detection region.
        \param [in] right - a right side of bounding box which restricts detection region.
        \param [in] bottom - a bottom side of bounding box which restricts detection region.
        \param [out] dst - a pointer to pixels data of 8-bit image with output result. None zero points refer to left-top corner of detected objects.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdDetectionLbpDetect32fp(const void * hid, const uint8_t * mask, size_t maskStride,
        ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

    /*! @ingroup object_detection

        \fn void SimdDetectionLbpDetect32fi(const void * hid, const uint8_t * mask, size_t maskStride, ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

        \short Performs object detection with using of LBP cascade classifier (uses 32-bit float numbers, processes only even points).

        You must call function ::SimdDetectionPrepare before calling of this functions.
        All restriction (input mask and bounding box) affects to left-top corner of scanning window.

        \note This function is used for implementation of Simd::Detection.

        \param [in] hid - a pointer to hidden cascade which was received with using of function ::SimdDetectionInit.
        \param [in] mask - a pointer to pixels data of 8-bit image with mask. The mask restricts detection region.
        \param [in] maskStride - a row size of the mask image.
        \param [in] left - a left side of bounding box which restricts detection region.
        \param [in] top - a top side of bounding box which restricts detection region.
        \param [in] right - a right side of bounding box which restricts detection region.
        \param [in] bottom - a bottom side of bounding box which restricts detection region.
        \param [out] dst - a pointer to pixels data of 8-bit image with output result. None zero points refer to left-top corner of detected objects.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdDetectionLbpDetect32fi(const void * hid, const uint8_t * mask, size_t maskStride,
        ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

    /*! @ingroup object_detection

        \fn void SimdDetectionLbpDetect16ip(const void * hid, const uint8_t * mask, size_t maskStride, ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

        \short Performs object detection with using of LBP cascade classifier (uses 16-bit integer numbers, processes all points).

        You must call function ::SimdDetectionPrepare before calling of this functions.
        All restriction (input mask and bounding box) affects to left-top corner of scanning window.

        \note This function is used for implementation of Simd::Detection.

        \param [in] hid - a pointer to hidden cascade which was received with using of function ::SimdDetectionInit.
        \param [in] mask - a pointer to pixels data of 8-bit image with mask. The mask restricts detection region.
        \param [in] maskStride - a row size of the mask image.
        \param [in] left - a left side of bounding box which restricts detection region.
        \param [in] top - a top side of bounding box which restricts detection region.
        \param [in] right - a right side of bounding box which restricts detection region.
        \param [in] bottom - a bottom side of bounding box which restricts detection region.
        \param [out] dst - a pointer to pixels data of 8-bit image with output result. None zero points refer to left-top corner of detected objects.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdDetectionLbpDetect16ip(const void * hid, const uint8_t * mask, size_t maskStride,
        ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

    /*! @ingroup object_detection

        \fn void SimdDetectionLbpDetect16ii(const void * hid, const uint8_t * mask, size_t maskStride, ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

        \short Performs object detection with using of LBP cascade classifier (uses 16-bit integer numbers, processes only even points).

        You must call function ::SimdDetectionPrepare before calling of this functions.
        All restriction (input mask and bounding box) affects to left-top corner of scanning window.

        \note This function is used for implementation of Simd::Detection.

        \param [in] hid - a pointer to hidden cascade which was received with using of function ::SimdDetectionInit.
        \param [in] mask - a pointer to pixels data of 8-bit image with mask. The mask restricts detection region.
        \param [in] maskStride - a row size of the mask image.
        \param [in] left - a left side of bounding box which restricts detection region.
        \param [in] top - a top side of bounding box which restricts detection region.
        \param [in] right - a right side of bounding box which restricts detection region.
        \param [in] bottom - a bottom side of bounding box which restricts detection region.
        \param [out] dst - a pointer to pixels data of 8-bit image with output result. None zero points refer to left-top corner of detected objects.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdDetectionLbpDetect16ii(const void * hid, const uint8_t * mask, size_t maskStride,
        ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

    /*! @ingroup object_detection

        \fn void SimdDetectionFree(void * ptr);

        \short Frees pointers which was received with using of functions ::SimdDetectionLoadA and ::SimdDetectionInit.

        \note This function is used for implementation of Simd::Detection.

        \param [in] ptr - a pointer which was received with using of functions ::SimdDetectionLoadA and ::SimdDetectionInit.
    */    SIMD_API void SimdDetectionFree(void * ptr);

    /*! @ingroup edge_background

        \fn void SimdEdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height, uint8_t * background, size_t backgroundStride);

        \short Performs edge background update (initial grow, slow mode).

        All images must have the same width, height and format (8-bit gray).

        For every point:
        \verbatim
        background[i] += value[i] > background[i] ? 1 : 0;
        \endverbatim

        This function is used for edge background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::EdgeBackgroundGrowRangeSlow(const View<A>& value, View<A>& background).

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
        \verbatim
        background[i] = value[i] > background[i] ? value[i] : background[i];
        \endverbatim

        This function is used for edge background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::EdgeBackgroundGrowRangeFast(const View<A>& value, View<A>& background).

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
        \verbatim
        backgroundCount[i] += (value[i] > backgroundValue[i] && backgroundCount[i] < 255) ? 1 : 0;
        \endverbatim

        This function is used for edge background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::EdgeBackgroundIncrementCount(const View<A>& value, const View<A>& backgroundValue, View<A>& backgroundCount).

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
        \verbatim
        backgroundValue[i] += (backgroundCount[i] > threshold && backgroundValue[i] < 255) ? 1 : 0;
        backgroundValue[i] -= (backgroundCount[i] < threshold && backgroundValue[i] > 0) ? 1 : 0;
        backgroundCount[i] = 0;
        \endverbatim

        This function is used for edge background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::EdgeBackgroundAdjustRange(View<A>& backgroundCount, View<A>& backgroundValue, uint8_t threshold).

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

        Adjusts edge background range for every point:
        \verbatim
        if(mask[i])
        {
            backgroundValue[i] += (backgroundCount[i] > threshold && backgroundValue[i] < 255) ? 1 : 0;
            backgroundValue[i] -= (backgroundCount[i] < threshold && backgroundValue[i] > 0) ? 1 : 0;
            backgroundCount[i] = 0;
        }
        \endverbatim

        This function is used for edge background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::EdgeBackgroundAdjustRange(View<A>& backgroundCount, View<A>& backgroundValue, uint8_t threshold, const View<A>& mask).

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
        \verbatim
        background[i] = value[i];
        \endverbatim

        This function is used for fast edge background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::EdgeBackgroundShiftRange(const View<A>& value, View<A>& background).

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

        For every point:
        \verbatim
        if(mask[i]])
            background[i] = value[i];
        \endverbatim

        This function is used for fast edge background updating in motion detection algorithm.

        \note This function has a C++ wrapper Simd::EdgeBackgroundShiftRange(const View<A>& value, View<A>& background, const View<A>& mask).

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

        \note This function has a C++ wrapper Simd::Fill(View<A>& dst, uint8_t value).

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

        \note This function has a C++ wrapper Simd::FillFrame(View<A>& dst, const Rectangle<ptrdiff_t> & frame, uint8_t value).

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

        \note This function has a C++ wrapper Simd::FillBgr(View<A>& dst, uint8_t blue, uint8_t green, uint8_t red).

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

        \note This function has a C++ wrapper Simd::FillBgra(View<A>& dst, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha).

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

    /*! @ingroup filling

        \fn void SimdFillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const uint8_t * pixel, size_t pixelSize);

        \short Fills image by value of given pixel.

        \note This function has a C++ wrapper Simd::FillPixel(View<A> & dst, const Pixel & pixel).

        \param [out] dst - a pointer to pixels data of destination image.
        \param [in] stride - a row size of the dst image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] pixel - a pointer to pixel to fill.
        \param [in] pixelSize - a size of the image pixel. Parameter is restricted by range [1, 4]. 
    */
    SIMD_API void SimdFillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const uint8_t * pixel, size_t pixelSize);

    /*! @ingroup float16

        \fn void SimdFloat32ToFloat16(const float * src, size_t size, uint16_t * dst);

        \short Converts numbers in the array from 32-bit float to 16-bit float format.

        \param [in] src - a pointer to the input array with 32-bit float point numbers.
        \param [in] size - a size of input and output array.
        \param [out] dst - a pointer to the output array with 16-bit float point numbers.
    */
    SIMD_API void SimdFloat32ToFloat16(const float * src, size_t size, uint16_t * dst);

    /*! @ingroup float16

        \fn void SimdFloat16ToFloat32(const uint16_t* src, size_t size, float  * dst);

        \short Converts numbers in the array from 16-bit float to 32-bit float format.

        \param [in] src - a pointer to the input array with 16-bit float point numbers.
        \param [in] size - a size of input and output array.
        \param [out] dst - a pointer to the output array with 32-bit float point numbers.
    */
    SIMD_API void SimdFloat16ToFloat32(const uint16_t * src, size_t size, float * dst);

    /*! @ingroup float16

        \fn void SimdSquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum);

        \short Calculates sum of squared differences for two 16-bit float arrays.

        All arrays must have the same size.

        For every element:
        \verbatim
        sum += (a[i] - b[i])*(a[i] - b[i]);
        \endverbatim

        \param [in] a - a pointer to the first 16-bit float array.
        \param [in] b - a pointer to the second 16-bit float array.
        \param [in] size - a size of arrays.
        \param [out] sum - a pointer to 32-bit float point sum of squared differences.
    */
    SIMD_API void SimdSquaredDifferenceSum16f(const uint16_t * a, const uint16_t * b, size_t size, float * sum);

    /*! @ingroup other_conversion

        \fn void SimdFloat32ToUint8(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst);

        \short Converts numbers in the array from 32-bit float to 8-bit unsigned integer format.

        For every element:
        \verbatim
        dst[i] = (min(max(src[i], lower), upper) - lower)*255/(upper - lower);
        \endverbatim

        \param [in] src - a pointer to the input array with 32-bit float point numbers.
        \param [in] size - a size of input and output array.
        \param [in] lower - a pointer to lower saturated bound of the input array.
        \param [in] upper - a pointer to upper saturated bound of the input array.
        \param [out] dst - a pointer to the output array with 8-bit unsigned integer numbers.
    */
    SIMD_API void SimdFloat32ToUint8(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst);

    /*! @ingroup other_conversion

        \fn void SimdUint8ToFloat32(const uint8_t* src, size_t size, const float * lower, const float * upper, float * dst);

        \short Converts numbers in the array from 8-bit unsigned integer to 32-bit float format.

        For every element:
        \verbatim
        dst[i] = src[i]*(upper - lower)/255 + lower;
        \endverbatim

        \param [in] src - a pointer to the input array with 8-bit unsigned integer numbers.
        \param [in] size - a size of input and output array.
        \param [in] lower - a pointer to lower bound of the output array.
        \param [in] upper - a pointer to upper bound of the output array.
        \param [out] dst - a pointer to the output array with 32-bit float point numbers.
    */
    SIMD_API void SimdUint8ToFloat32(const uint8_t * src, size_t size, const float * lower, const float * upper, float * dst);

    /*! @ingroup other_filter

        \fn void SimdGaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);

        \short Performs Gaussian blur filtration with window 3x3.

        For every point:
        \verbatim
        dst[x, y] = (src[x-1, y-1] + 2*src[x, y-1] + src[x+1, y-1] +
                    2*(src[x-1, y] + 2*src[x, y] + src[x+1, y]) +
                    src[x-1, y+1] + 2*src[x, y+1] + src[x+1, y+1] + 8) / 16;
        \endverbatim

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function has a C++ wrapper Simd::GaussianBlur3x3(const View<A>& src, View<A>& dst).

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

        \note This function has a C++ wrapper Simd::GrayToBgr(const View<A>& gray, View<A>& bgr).

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

        \note This function has a C++ wrapper Simd::GrayToBgra(const View<A>& gray, View<A>& bgra, uint8_t alpha).

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
        \verbatim
        dx = abs(src[x, y] - average(src[x+step, y], src[x-step, y]));
        dy = abs(src[x, y] - average(src[x, y+step], src[x, y-step]));
        histogram[max(dx, dy)]++;
        \endverbatim

        \note This function has a C++ wrapper Simd::AbsSecondDerivativeHistogram(const View<A>& src, size_t step, size_t indent, uint32_t * histogram).

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
        \verbatim
        histogram[src[i]]++.
        \endverbatim

        \note This function has a C++ wrapper Simd::Histogram(const View<A>& src, uint32_t * histogram).

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

        For every point:
        \verbatim
        if(mask[i] == index)
            histogram[src[i]]++.
        \endverbatim

        \note This function has a C++ wrapper Simd::HistogramMasked(const View<A> & src, const View<A> & mask, uint8_t index, uint32_t * histogram).

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

    /*! @ingroup histogram

        \fn void SimdHistogramConditional(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint32_t * histogram);

        \short Calculates histogram of 8-bit gray image for those points when mask points satisfying certain condition.

        For every point:
        \verbatim
        if(compare(mask[x, y], value))
            histogram[src[x, y]]++.
        \endverbatim

        \note This function has a C++ wrapper Simd::HistogramConditional(const View<A>& src, const View<A>& mask, uint8_t value, SimdCompareType compareType, uint32_t * histogram).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] mask - a pointer to pixels data of the mask 8-bit image.
        \param [in] maskStride - a row size of the mask image.
        \param [in] value - a second value for compare operation.
        \param [in] compareType - a compare operation type (see ::SimdCompareType).
        \param [out] histogram - a pointer to histogram (array of 256 unsigned 32-bit values).
    */
    SIMD_API void SimdHistogramConditional(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint32_t * histogram);

    /*! @ingroup histogram

        \fn void SimdNormalizedColors(const uint32_t * histogram, uint8_t * colors);

        \short Gets normalized color map for given histogram.

        \param [in] histogram - a pointer to histogram (array of 256 unsigned 32-bit values).
        \param [out] colors - a pointer to the color map (array of 256 unsigned 8-bit values).
    */
    SIMD_API void SimdNormalizedColors(const uint32_t * histogram, uint8_t * colors);

    /*! @ingroup histogram

        \fn void SimdChangeColors(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * colors, uint8_t * dst, size_t dstStride);

        \short Changes colors for 8-bit gray image with using of color map.

        The input and output 8-bit gray images must have the same size. 

        \note This function has a C++ wrapper Simd::ChangeColors(const View<A> & src, const uint8_t * colors, View<A> & dst).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] colors - a pointer to the color map (array of 256 unsigned 8-bit values).
        \param [out] dst - a pointer to pixels data of output 8-bit gray image.
        \param [in] dstStride - a row size of the output gray image.
    */
    SIMD_API void SimdChangeColors(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * colors, uint8_t * dst, size_t dstStride);

    /*! @ingroup histogram

        \fn void SimdNormalizeHistogram(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Normalizes histogram for 8-bit gray image.

        The input and output 8-bit gray images must have the same size.

        \note This function has a C++ wrapper Simd::NormalizeHistogram(const View<A> & src, View<A> & dst).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of output 8-bit image with normalized histogram.
        \param [in] dstStride - a row size of the output image.
    */
    SIMD_API void SimdNormalizeHistogram(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup hog

        \fn void SimdHogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height, size_t cellX, size_t cellY, size_t quantization, float * histograms);

        \short Calculates HOG direction histograms for 8-bit gray image.

        Calculates HOG direction histogram for every cell of 8-bit gray image. This function is useful for face recognition.

        \note This function has a C++ wrapper Simd::HogDirectionHistograms(const View<A> & src, const Point<ptrdiff_t> & cell, size_t quantization, float * histograms).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] stride - a row size of the image.
        \param [in] width - an image width. It must be a multiple of cellX.
        \param [in] height - an image height. It must be a multiple of cellY.
        \param [in] cellX - a width of cell.
        \param [in] cellY - a height of cell.
        \param [in] quantization - a direction quantization. Must be even.
        \param [out] histograms - a pointer to buffer with histograms. Array must has size grater or equal to (width/cellX)*(height/cellY)*quantization.
    */
    SIMD_API void SimdHogDirectionHistograms(const uint8_t * src, size_t stride, size_t width, size_t height,
        size_t cellX, size_t cellY, size_t quantization, float * histograms);

    /*! @ingroup hog

        \fn void SimdHogExtractFeatures(const uint8_t * src, size_t stride, size_t width, size_t height, float * features);

        \short Extracts HOG features for 8-bit gray image.

        Extracts HOG features 8-bit gray image. 31 features are extracted for 8x8 cell size and 2x2 block size. This function is useful for face recognition.

        \note This function has a C++ wrapper Simd::HogExtractFeatures(const View<A> & src, float * features).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] stride - a row size of the image.
        \param [in] width - an image width. It must be a multiple of 8. Its minimal value is 16.
        \param [in] height - an image height. It must be a multiple of 8. Its minimal value is 16.
        \param [out] features - a pointer to buffer with features. Array must has size grater or equal to (width/8)*(height/8)*31.
    */
    SIMD_API void SimdHogExtractFeatures(const uint8_t * src, size_t stride, size_t width, size_t height, float * features);

    /*! @ingroup hog

        \fn void SimdHogDeinterleave(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride);

        \short Separates one interleaved 32-bit float point image to separate planes.

        \param [in] src - a pointer to the input interleaved 32-bit float point image.
        \param [in] srcStride - a row size of input image.
        \param [in] width - a width of input and output images.
        \param [in] height - a height of input and output images.
        \param [in] count - the number of output planes.
        \param [out] dst - a pointer to array with pointers to output planes.
        \param [in] dstStride - a row size of output images.
    */
    SIMD_API void SimdHogDeinterleave(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride);

    /*! @ingroup hog

        \fn void SimdHogFilterSeparable(const float * src, size_t srcStride, size_t width, size_t height, const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add);

        \short Applies separable filter to given image of 32-bit float point format.

        For every point (except border):
        \verbatim
        sum = 0;
        for(dy = 0; dy < colSize; dy++)
            for(dx = 0; dx < rowSize; dx++)
                sum += src[x + dx, y + dy]*colFilter[dy]*rowFilter[dx];
        if(add)
            dst[x, y] += sum;
        else
            dst[x, y] = sum;
        \endverbatim

        \note Input image has to have size at least not less then size of filter: (width <= rowSize and height <= colSize).

        \param [in] src - a pointer to input 32-bit float point image.
        \param [in] srcStride - a row size of input image.
        \param [in] width - a width of input image. It must be not less then size of row filter.
        \param [in] height - a height of input image. It must be not less then size of column filter.
        \param [in] rowFilter - a pointer to 32-bit float point array with row filter.
        \param [in] rowSize- a size of row filter.
        \param [in] colFilter - a pointer to 32-bit float point array with column filter.
        \param [in] colSize- a size of column filter.
        \param [in, out] dst - a pointer to output 32-bit float point image.
        \param [in] dstStride - a row size of output image.
        \param [in] add - a flag which signalizes that result has to be added to existing image.
    */
    SIMD_API void SimdHogFilterSeparable(const float * src, size_t srcStride, size_t width, size_t height, const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add);

    /*! @ingroup hog

        \fn void SimdHogLiteExtractFeatures(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t cell, float * features, size_t featuresStride);

        \short Extracts lite HOG features for 8-bit gray image.

        Extracts lite (for 8 directions) HOG features 8-bit gray image. 16 features are extracted for 8x8 or 4x4 cell size and 2x2 block size. 

        \note This function has a C++ wrapper Simd::HogLiteExtractFeatures(const View<A> & src, size_t cell, float * features, size_t featuresStride).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of the image.
        \param [in] width - an image width. Its minimal value is cell*3.
        \param [in] height - an image height. Its minimal value is cell*3.
        \param [in] cell - a size of cell. It must be 4 or 8. 
        \param [out] features - a pointer to buffer with features. Array must has size greater or equal to (height/cell - 2)*featuresStride.
        \param [in] featuresStride - a row size of the buffer with features. It must be greater or equal to (width/cell - 2)*16.
    */
    SIMD_API void SimdHogLiteExtractFeatures(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t cell, float * features, size_t featuresStride);

    /*! @ingroup hog

        \fn void SimdHogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterSize, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride);

        \short Applies filter to lite HOG features.

        Applies filter of square shape to lite HOG features. 

        For every point of output image:
        \verbatim
        if(mask[x, y])
            sum = 0;
            for(dy = 0; dy < filterSize; dy++)
                for(dx = 0; dx < filterSize*featureSize; dx++)
                    sum += src[x*featureSize + dx, y + dy]*filter[dx, dy];
            dst[x, y] = sum;
        else
            dst[x, y] = -FLT_MAX;
        \endverbatim

        \param [in] src - a pointer to the input 32-bit float array with features.
        \param [in] srcStride - a row size of input array with features.
        \param [in] srcWidth - a width of input array with features. Its minimal value is filterSize.
        \param [in] srcHeight - a height of input array with features. Its minimal value is filterSize.
        \param [in] featureSize - a size of cell with features. It must be 8 or 16.
        \param [in] filter - a pointer to the 32-bit float array with filter values. 
                    Array must have size equal to filterSize*filterSize*featureSize.
        \param [in] filterSize - a size (width and height) of used filter. 
        \param [in] mask - a pointer to the 32-bit integer array with mask (0 or -1). 
                    Pointer can be null otherwise the array must have size greater then (srcHeight - filterSize)*(srcWidth - filterSize).
                    A function ::SimdHogLiteCreateMask is usefull in order to create this mask.
        \param [in] maskStride - a row size of mask array. 
        \param [out] dst - a pointer to output buffer with result of filtration. Array must have size greater then (srcHeight - filterSize)*(srcWidth - filterSize).
        \param [in] dstStride - a row size of the output buffer with result of filtration.
    */
    SIMD_API void SimdHogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterSize, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride);

    /*! @ingroup hog

        \fn void SimdHogLiteResizeFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight);

        \short Resizes 2D-array with lite HOG features.

        Resizes 2D-array with lite HOG features. It use method of bilinear interpolation.

        \param [in] src - a pointer to the input 32-bit float array with features.
        \param [in] srcStride - a row size of input array with features.
        \param [in] srcWidth - a width of input array with features. 
        \param [in] srcHeight - a height of input array with features. 
        \param [in] featureSize - a size of cell with features. It must be 8 or 16.
        \param [out] dst - a pointer to the output 32-bit float array with features.
        \param [in] dstStride - a row size of output array with features.
        \param [in] dstWidth - a width of output array with features.
        \param [in] dstHeight - a height of output array with features.
        */
    SIMD_API void SimdHogLiteResizeFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight);

    /*! @ingroup hog

        \fn void SimdHogLiteCompressFeatures(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride);

        \short Compresses 16 features to 8 features for 2D-array.

        Compresses 16 features to 8 features for 2D-array. The method uses PCA.

        \param [in] src - a pointer to the input 32-bit float array with uncompessed features.
        \param [in] srcStride - a row size of input array with uncompessed features.
        \param [in] width - a width of 2D-array with features.
        \param [in] height - a height of 2D-array with features.
        \param [in] pca - a pointer to the PCA matrix with size 16x8.
        \param [out] dst - a pointer to the output 32-bit float array with compessed features.
        \param [in] dstStride - a row size of output array with compessed features.
    */
    SIMD_API void SimdHogLiteCompressFeatures(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride);

    /*! @ingroup hog

        \fn void SimdHogLiteFilterSeparable(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add);

        \short Applies separable filter to lite HOG features.

        For every point (except border):
        \verbatim
        sum = 0;
        for(dy = 0; dy < vSize; dy++)
            for(dx = 0; dx < hSize*featureSize; dx++)
                sum += src[x*featureSize + dx, y + dy]*vFilter[dy]*hFilter[dx];
        if(add)
            dst[x, y] += sum;
        else
            dst[x, y] = sum;
        \endverbatim

        \note Input image has to have size at least not less then size of filter: (srcWidth <= hSize and srcHeight <= vSize).

        \param [in] src - a pointer to the input 32-bit float array with features.
        \param [in] srcStride - a row size of input array with features.
        \param [in] srcWidth - a width of input array with features. Its minimal value is hSize.
        \param [in] srcHeight - a height of input array with features. Its minimal value is vSize.
        \param [in] featureSize - a size of cell with features. It must be 8 or 16.
        \param [in] hFilter - a pointer to 32-bit float point array with horizontal filter.
        \param [in] hSize - a size of horizontal filter (in featureSize). Total size of horizontal filter is hSize*featureSize.
        \param [in] vFilter - a pointer to 32-bit float point array with vertical filter.
        \param [in] vSize- a size of vertical filter.
        \param [in, out] dst - a pointer to output 32-bit float point image.
        \param [in] dstStride - a row size of output image.
        \param [in] add - a flag which signalizes that result has to be added to existing image.
    */
    SIMD_API void SimdHogLiteFilterSeparable(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add);

    /*! @ingroup hog

        \fn void SimdHogLiteFindMax7x7(const float * a, size_t aStride, const float * b, size_t bStride, size_t height, float * value, size_t * col, size_t * row);

        \short Adds two 32-bit float point 2D-array with size 7x7 and finds value and position of maximum in the result array.

        Algorithm description:
        \verbatim
        value = -FLT_MAX;
        for (y = 0; y < height; ++y)
        {
            for (x = 0; x < 7; ++x)
            {
                v = a[x, y] + b[x, y];
                if (v > value)
                {
                    value = v;
                    col = x;
                    row = y;
                    break;
                }
            }
        }
        \endverbatim

        \param [in] a - a pointer to the first input 32-bit float array with size 7x7.
        \param [in] aStride - a row size of the first input array.
        \param [in] b - a pointer to the second input 32-bit float array with size 7x7.
        \param [in] bStride - a row size of the second input array.
        \param [in] height - a height of the input arrays. It must be equal or less then 7.
        \param [out] value - a pointer to the output 32-bit float value with maximum.
        \param [out] col - a pointer to the output integer value with x-position of maximum.
        \param [out] row - a pointer to the output integer value with y-position of maximum.
    */
    SIMD_API void SimdHogLiteFindMax7x7(const float * a, size_t aStride, const float * b, size_t bStride, size_t height, float * value, size_t * col, size_t * row);

    /*! @ingroup hog

        \fn void SimdHogLiteCreateMask(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride);

        \short Creates mask for function ::SimdHogLiteFilterFeatures.

        Zeroes destination mask. Then for every source point:
        \verbatim
        if(src[x, y] > threshold)
            for (dy = 0; dy < size; ++dy)
                for (dx = 0; dx < size; ++dx)
                    dst[x*scale + dx, y*scale + dy] = -1;
        \endverbatim

        \param [in] src - a pointer to the input 32-bit float 2D array.
        \param [in] srcStride - a row size of the input array.
        \param [in] srcWidth - a width of input array.
        \param [in] srcHeight - a height of input array.
        \param [in] threshold - a pointer to 32-bit float threshold.
        \param [in] scale - a scale coefficient between input and output array.
        \param [in] size - a size of neighborhood.
        \param [out] dst - a pointer to the output 32-bit integer array with mask (0 or -1).
        \param [in] dstStride - a row size of the output array.
    */
    SIMD_API void SimdHogLiteCreateMask(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride);

    /*! @ingroup other_conversion

        \fn void SimdInt16ToGray(const uint8_t * src, size_t width, size_t height, size_t srcStride, uint8_t * dst, size_t dstStride);

        \short Converts 16-bit signed integer image to 8-bit gray image with saturation

        All images must have the same width and height.

        For every point:
        \verbatim
        dst[i] = Max(0, Min(255, src[i]));
        \endverbatim

        \note This function has a C++ wrapper Simd::Int16ToGray(const View<A> & src, View<A> & dst).

        \param [in] src - a pointer to pixels data of input 16-bit signed integer image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] srcStride - a row size of the 16-bit signed integer image.
        \param [out] dst - a pointer to pixels data of input 8-bit gray image.
        \param [out] dstStride - a row size of the gray image.
    */
    SIMD_API void SimdInt16ToGray(const uint8_t * src, size_t width, size_t height, size_t srcStride, uint8_t * dst, size_t dstStride);

    /*! @ingroup integral

        \fn void SimdIntegral(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * sum, size_t sumStride, uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride, SimdPixelFormatType sumFormat, SimdPixelFormatType sqsumFormat);

        \short Calculates integral images for input 8-bit gray image.

        The function can calculates sum integral image, square sum integral image (optionally) and tilted sum integral image (optionally).
        A integral images must have width and height per unit greater than that of the input image.

        \note This function has a C++ wrappers:
        \n Simd::Integral(const View<A>& src, View<A>& sum),
        \n Simd::Integral(const View<A>& src, View<A>& sum, View<A>& sqsum),
        \n Simd::Integral(const View<A>& src, View<A>& sum, View<A>& sqsum, View<A>& tilted).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - a pointer to pixels data of 32-bit integer sum image.
        \param [in] sumStride - a row size of sum image (in bytes).
        \param [out] sqsum - a pointer to pixels data of 32-bit integer or 64-bit float point square sum image. It can be NULL.
        \param [in] sqsumStride - a row size of sqsum image (in bytes).
        \param [out] tilted - a pointer to pixels data of 32-bit integer tilted sum image. It can be NULL.
        \param [in] tiltedStride - a row size of tilted image (in bytes).
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
        \verbatim
        statistic[i] = min(statistic[i] + increment, saturation);
        \endverbatim

        This function is used for interference detection in motion detection algorithm.

        \note This function has a C++ wrappers: Simd::InterferenceIncrement(View<A> & dst, uint8_t increment, int16_t saturation).

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

        For every point:
        \verbatim
        if(mask[i] == index)
            statistic[i] = min(statistic[i] + increment, saturation);
        \endverbatim

        All images must have the same width, height.
        This function is used for interference detection in motion detection algorithm.

        \note This function has a C++ wrappers: Simd::InterferenceIncrementMasked(View<A> & dst, uint8_t increment, int16_t saturation, const View<A>& mask, uint8_t index).

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
        \verbatim
        statistic[i] = max(statistic[i] - decrement, saturation);
        \endverbatim

        This function is used for interference detection in motion detection algorithm.

        \note This function has a C++ wrappers: Simd::InterferenceDecrement(View<A> & dst, uint8_t decrement, int16_t saturation).

        \param [in, out] statistic - a pointer to pixels data of 16-bit signed integer image with statistic.
        \param [in] stride - a row size of statistic image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] decrement - a decrement of statistic.
        \param [in] saturation - a lower saturation of statistic.
    */
    SIMD_API void SimdInterferenceDecrement(uint8_t * statistic, size_t stride, size_t width, size_t height, uint8_t decrement, int16_t saturation);

    /*! @ingroup interference

        \fn void SimdInterferenceDecrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height, uint8_t decrement, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index);

        \short Decrements statistic of interference detector with using segmentation mask.

        For every point:
        \verbatim
        if(mask[i] == index)
            statistic[i] = max(statistic[i] - decrement, saturation);
        \endverbatim

        All images must have the same width, height.
        This function is used for interference detection in motion detection algorithm.

        \note This function has a C++ wrappers: Simd::InterferenceDecrementMasked(View<A> & dst, uint8_t decrement, int16_t saturation, const View<A>& mask, uint8_t index).

        \param [in, out] statistic - a pointer to pixels data of 16-bit signed integer image with statistic.
        \param [in] statisticStride - a row size of statistic image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] decrement - a decrement of statistic.
        \param [in] saturation - a lower saturation of statistic.
        \param [in] mask - a pointer to pixels data of 8-bit gray image with mask.
        \param [in] maskStride - a row size of mask image.
        \param [in] index - an index of mask.
    */
    SIMD_API void SimdInterferenceDecrementMasked(uint8_t * statistic, size_t statisticStride, size_t width, size_t height,
        uint8_t decrement, int16_t saturation, const uint8_t * mask, size_t maskStride, uint8_t index);

    /*! @ingroup other_conversion

        \fn void SimdInterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * uv, size_t uvStride);

        \short Interleaves 8-bit U and V planar images into one 16-bit UV interleaved image.

        All images must have the same width and height.
        This function used for YUV420P to NV12 conversion.

        \note This function has a C++ wrapper Simd::InterleaveUv(const View<A>& u, const View<A>& v, View<A>& uv).

        \param [in] u - a pointer to pixels data of input 8-bit U planar image.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit V planar image.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] uv - a pointer to pixels data of output 16-bit UV interleaved image.
        \param [in] uvStride - a row size of the uv image.
    */
    SIMD_API void SimdInterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * uv, size_t uvStride);

    /*! @ingroup other_conversion

        \fn void SimdInterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

        \short Interleaves 8-bit Blue, Green and Red planar images into one 24-bit BGR interleaved image.

        All images must have the same width and height.

        \note This function has a C++ wrapper Simd::InterleaveBgr(const View<A>& b, const View<A>& g, const View<A>& r, View<A>& bgr).

        \param [in] b - a pointer to pixels data of input 8-bit Blue planar image.
        \param [in] bStride - a row size of the b image.
        \param [in] g - a pointer to pixels data of input 8-bit Green planar image.
        \param [in] gStride - a row size of the g image.
        \param [in] r - a pointer to pixels data of input 8-bit Red planar image.
        \param [in] rStride - a row size of the r image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] bgr - a pointer to pixels data of output 24-bit BGR interleaved image.
        \param [in] bgrStride - a row size of the bgr image.
    */
    SIMD_API void SimdInterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride,
        size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

    /*! @ingroup other_conversion

        \fn void SimdInterleaveBgra(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride);

        \short Interleaves 8-bit Blue, Green, Red and Alpha planar images into one 32-bit BGRA interleaved image.

        All images must have the same width and height.

        \note This function has a C++ wrapper Simd::InterleaveBgra(const View<A>& b, const View<A>& g, const View<A>& r, const View<A>& a, View<A>& bgra).

        \param [in] b - a pointer to pixels data of input 8-bit Blue planar image.
        \param [in] bStride - a row size of the b image.
        \param [in] g - a pointer to pixels data of input 8-bit Green planar image.
        \param [in] gStride - a row size of the g image.
        \param [in] r - a pointer to pixels data of input 8-bit Red planar image.
        \param [in] rStride - a row size of the r image.
        \param [in] a - a pointer to pixels data of input 8-bit Alpha planar image.
        \param [in] aStride - a row size of the a image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] bgra - a pointer to pixels data of output 32-bit BGRA interleaved image.
        \param [in] bgraStride - a row size of the bgr image.
    */
    SIMD_API void SimdInterleaveBgra(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, const uint8_t * a, size_t aStride,
        size_t width, size_t height, uint8_t * bgra, size_t bgraStride);

    /*! @ingroup laplace_filter

        \fn void SimdLaplace(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Calculates Laplace's filter.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.

        For every point:
        \verbatim
        dst[x, y] =
            - src[x-1, y-1] -   src[x, y-1] - src[x+1, y-1]
            - src[x-1, y]   + 8*src[x, y]   - src[x+1, y]
            - src[x-1, y+1] -   src[x, y+1] - src[x+1, y+1].
        \endverbatim

        \note This function has a C++ wrappers: Simd::Laplace(const View<A>& src, View<A>& dst).

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of the output image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdLaplace(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup laplace_filter

        \fn void SimdLaplaceAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Calculates absolute value of Laplace's filter.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.

        For every point:
        \verbatim
        dst[x, y] = abs(
            - src[x-1, y-1] -   src[x, y-1] - src[x+1, y-1]
            - src[x-1, y]   + 8*src[x, y]   - src[x+1, y]
            - src[x-1, y+1] -   src[x, y+1] - src[x+1, y+1]).
        \endverbatim

        \note This function has a C++ wrappers: Simd::LaplaceAbs(const View<A>& src, View<A>& dst).

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of the output image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdLaplaceAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup other_statistic

        \fn void SimdLaplaceAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

        \short Calculates sum of absolute value of Laplace's filter.

        Input image must has 8-bit gray format.

        For every point:
        \verbatim
        sum += abs(
            - src[x-1, y-1] -   src[x, y-1] - src[x+1, y-1]
            - src[x-1, y]   + 8*src[x, y]   - src[x+1, y]
            - src[x-1, y+1] -   src[x, y+1] - src[x+1, y+1]).
        \endverbatim

        \note This function has a C++ wrappers: Simd::LaplaceAbsSum(const View<A>& src, uint64_t & sum).

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] stride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - a pointer to result sum.
    */
    SIMD_API void SimdLaplaceAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

    /*! @ingroup other_filter

        \fn void SimdLbpEstimate(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Calculates LBP (Local Binary Patterns) for 8-bit gray image.

        All images must have the same width and height.

        \note This function has a C++ wrappers: Simd::LbpEstimate(const View<A>& src, View<A>& dst).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of output 8-bit gray image with LBP.
        \param [in] dstStride - a row size of dst image.
    */
    SIMD_API void SimdLbpEstimate(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup other_filter

        \fn void SimdMeanFilter3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);

        \short Performs an averaging with window 3x3.

        For every point:
        \verbatim
        dst[x, y] = (src[x-1, y-1] + src[x, y-1] + src[x+1, y-1] +
                     src[x-1, y] + src[x, y] + src[x+1, y] +
                     src[x-1, y+1] + src[x, y+1] + src[x+1, y+1] + 4) / 9;
        \endverbatim

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function has a C++ wrapper Simd::MeanFilter3x3(const View<A>& src, View<A>& dst).

        \param [in] src - a pointer to pixels data of source image.
        \param [in] srcStride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count.
        \param [out] dst - a pointer to pixels data of destination image.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdMeanFilter3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
        size_t channelCount, uint8_t * dst, size_t dstStride);

    /*! @ingroup median_filter

        \fn void SimdMedianFilterRhomb3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);

        \short Performs median filtration of input image (filter window is a rhomb 3x3).

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function has a C++ wrappers: Simd::MedianFilterRhomb3x3(const View<A>& src, View<A>& dst).

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

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function has a C++ wrappers: Simd::MedianFilterRhomb5x5(const View<A>& src, View<A>& dst).

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

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function has a C++ wrappers: Simd::MedianFilterSquare3x3(const View<A>& src, View<A>& dst).

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

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function has a C++ wrappers: Simd::MedianFilterSquare5x5(const View<A>& src, View<A>& dst).

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

    /*! @ingroup neural

        \fn void SimdNeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride, int inversion);

        \short Converts a 8-bit gray image to the 32-bit float array.

        The length of output array must be equal to the area of input image.

        For every point:
        \verbatim
        dst[i] = inversion ? (255 - src[col]) / 255 : src[i]/255;
        \endverbatim

        \note This function has a C++ wrapper Simd::NeuralConvert(const View<A>& src, float * dst, bool inversion).

        \param [in] src - a pointer to pixels data of input image.
        \param [in] srcStride - a row size (in bytes) of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to output array.
        \param [in] dstStride - a row size of the output array.
        \param [in] inversion - a flag of color inversion.
    */
    SIMD_API void SimdNeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride, int inversion);

    /*! @ingroup neural

        \fn void SimdNeuralSigmoid(const float * src, size_t size, const float * slope, float * dst);

        \short Calculates sigmoid for 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        dst[i] = 1/(1 + exp(-slope*src[i]));
        \endverbatim

        \param [in] src - a pointer to the input array.
        \param [in] size - a size of arrays.
        \param [in] slope - a pointer to the slope parameter.
        \param [out] dst - a pointer to output array.
    */
    SIMD_API void SimdNeuralSigmoid(const float * src, size_t size, const float * slope, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst);

        \short Calculates rough sigmoid for 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        x = ::abs(src[i]*slope);
        e = 1 + x + x*x*0.5417 + x*x*x*x*0.1460;
        dst[i] = 1 / (1 + (src[i] > 0 ? 1 / e : e));
        \endverbatim
        It is approximate way (maximal absolute error is 0.002294 (~0.23%) ) of sigmoid function (::SimdNeuralSigmoid) calculation:
        \verbatim
        dst[i] = 1/(1 + exp(-slope*src[i]));
        \endverbatim

        \note This function is used in Simd::Neural::Function.

        \param [in] src - a pointer to the input array.
        \param [in] size - a size of arrays.
        \param [in] slope - a pointer to the slope parameter.
        \param [out] dst - a pointer to output array.
    */
    SIMD_API void SimdNeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst);

        \short Calculates rough sigmoid for 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        x = -src[i]*slope;
        e = max(1 + x/128, 0.5)^128;
        dst[i] = 1 / (1 + e);
        \endverbatim
        It is approximate way (maximal absolute error is 0.001721 (~0.17%) ) of sigmoid function (::SimdNeuralSigmoid) calculation:
        \verbatim
        dst[i] = 1/(1 + exp(-slope*src[i]));
        \endverbatim

        \note This function is used in Simd::Neural::Function.

        \param [in] src - a pointer to the input array.
        \param [in] size - a size of arrays.
        \param [in] slope - a pointer to the slope parameter.
        \param [out] dst - a pointer to output array.
    */
    SIMD_API void SimdNeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralDerivativeSigmoid(const float * src, size_t size, const float * slope, float * dst);

        \short Multiplies output 32-bit float array by derivative of sigmoid from input 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        dst[i] *= slope*(1 - src[i])*src[i];
        \endverbatim

        \note This function is used in Simd::Neural::Function.

        \param [in] src - a pointer to the input array.
        \param [in] size - a size of arrays.
        \param [in] slope - a pointer to the slope parameter.
        \param [in, out] dst - a pointer to output array.
    */
    SIMD_API void SimdNeuralDerivativeSigmoid(const float * src, size_t size, const float * slope, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralTanh(const float * src, size_t size, const float * slope, float * dst);

        \short Calculates hyperbolic tangent for 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        x = slope*src[i];
        dst[i] = (exp(x) - exp(-x))/(exp(x) + exp(-x));
        \endverbatim

        \param [in] src - a pointer to the input array.
        \param [in] size - a size of arrays.
        \param [in] slope - a pointer to the slope parameter.
        \param [out] dst - a pointer to output array.
    */
    SIMD_API void SimdNeuralTanh(const float * src, size_t size, const float * slope, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralRoughTanh(const float * src, size_t size, const float * slope, float * dst);

        \short Calculates rough hyperbolic tangent for 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        x = ::abs(src[i]*slope);
        e = 1 + x + x*x*0.5658 + x*x*x*x*0.1430;
        dst[i] = (src[i] > 0 ? 1 : -1)*(e - 1/e)/(e + 1/e);
        \endverbatim
        It is approximate way (maximal absolute error is 0.001514 (~0.15%) ) of hyperbolic tangent (::SimdNeuralTanh)  function calculation:
        \verbatim
        x = slope*src[i];
        dst[i] = (exp(x) - exp(-x))/(exp(x) + exp(-x));
        \endverbatim

        \note This function is used in Simd::Neural::Function.

        \param [in] src - a pointer to the input array.
        \param [in] size - a size of arrays.
        \param [in] slope - a pointer to the slope parameter.
        \param [out] dst - a pointer to output array.
    */
    SIMD_API void SimdNeuralRoughTanh(const float * src, size_t size, const float * slope, float * dst);

    /*! @ingroup neural

    \fn void SimdNeuralDerivativeTanh(const float * src, size_t size, const float * slope, float * dst);

        \short Multiplies output 32-bit float array by derivative of hyperbolic tangent from input 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        dst[i] *= slope*(1 - src[i]*src[i]);
        \endverbatim

        \note This function is used in Simd::Neural::Function.

        \param [in] src - a pointer to the input array.
        \param [in] size - a size of arrays.
        \param [in] slope - a pointer to the slope parameter.
        \param [in, out] dst - a pointer to output array.
    */
    SIMD_API void SimdNeuralDerivativeTanh(const float * src, size_t size, const float * slope, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralRelu(const float * src, size_t size, const float * slope, float * dst);

        \short Calculates Relu (rectified linear unit) function for 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        dst[i] =  src[i] > 0 ? src[i] : slope*src[i];
        \endverbatim

        \note This function is used in Simd::Neural::Function.

        \param [in] src - a pointer to the input array.
        \param [in] size - a size of arrays.
        \param [in] slope - a pointer to the slope parameter.
        \param [out] dst - a pointer to output array.
    */
    SIMD_API void SimdNeuralRelu(const float * src, size_t size, const float * slope, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralDerivativeRelu(const float * src, size_t size, const float * slope, float * dst);

        \short Multiplies output 32-bit float array by derivative of Relu (rectified linear unit) from input 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        dst[i] *=  src[i] > 0 ? 1 : slope;
        \endverbatim

        \note This function is used in Simd::Neural::Function.

        \param [in] src - a pointer to the input array.
        \param [in] size - a size of arrays.
        \param [in] slope - a pointer to the slope parameter.
        \param [in, out] dst - a pointer to output array.
    */
    SIMD_API void SimdNeuralDerivativeRelu(const float * src, size_t size, const float * slope, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralPow(const float * src, size_t size, const float * exponent, float * dst);

        \short Calculates Pow function for 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        dst[i] =  Pow(src[i], exponent[0]);
        \endverbatim

        \note This function is used in Simd::Neural::Function.

        \param [in] src - a pointer to the input array.
        \param [in] size - a size of arrays.
        \param [in] exponent - a pointer to exponent parameter.
        \param [out] dst - a pointer to output array.
    */
    SIMD_API void SimdNeuralPow(const float * src, size_t size, const float * exponent, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralProductSum(const float * a, const float * b, size_t size, float * sum);

        \short Calculates sum of products for two 32-bit float arrays.

        All arrays must have the same size.

        For every element:
        \verbatim
        sum += a[i]*b[i];
        \endverbatim

        \note This function is used in Simd::Neural.

        \param [in] a - a pointer to the first 32-bit float array.
        \param [in] b - a pointer to the second 32-bit float array.
        \param [in] size - a size of arrays.
        \param [out] sum - a pointer to 32-bit float sum of products.
    */
    SIMD_API void SimdNeuralProductSum(const float * a, const float * b, size_t size, float * sum);

    /*! @ingroup neural

        \fn void SimdNeuralAddVectorMultipliedByValue(const float * src, size_t size, const float * value, float * dst);

        \short Adds the product of a vector and a scalar to given vector.

        All arrays must have the same size.

        For every element:
        \verbatim
        dst[i] += src[i]*value[0];
        \endverbatim

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float array.
        \param [in] size - a size of arrays.
        \param [in] value - a pointer to the scalar 32-bit float value.
        \param [in, out] dst - a pointer to cumulative 32-bit float array.
    */
    SIMD_API void SimdNeuralAddVectorMultipliedByValue(const float * src, size_t size, const float * value, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralAddVector(const float * src, size_t size, float * dst);

        \short Adds a vector to given vector.

        All arrays must have the same size.

        For every element:
        \verbatim
        dst[i] += src[i];
        \endverbatim

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float array.
        \param [in] size - a size of the arrays.
        \param [in, out] dst - a pointer to cumulative 32-bit float array.
    */
    SIMD_API void SimdNeuralAddVector(const float * src, size_t size, float * dst);

    /*! @ingroup neural

        \fn void SimdNeuralAddValue(const float * value, float * dst, size_t size);

        \short Adds a value to each elements of given vector.

        For every element:
        \verbatim
        dst[i] += value;
        \endverbatim

        \note This function is used in Simd::Neural.

        \param [in] value - a pointer to the scalar 32-bit float value.
        \param [in, out] dst - a pointer to cumulative 32-bit float array.
        \param [in] size - a size of the array.
    */
    SIMD_API void SimdNeuralAddValue(const float * value, float * dst, size_t size);

    /*! @ingroup neural

        \fn void SimdNeuralUpdateWeights(const float * x, size_t size, const float * a, const float * b, float * d, float * w);

        \short Updates ANN weights.

        All arrays must have the same size.

        The algorithm performs:
        \verbatim
        for (size_t k = 0; k < size; ++k)
        {
            d[k] = a[0]*d[k] + b[0]*x[k];
            w[k] += d[k];
        }
        \endverbatim

        \param [in] x - a pointer to the X array.
        \param [in] size - a size of arrays.
        \param [in] a - a pointer to the first parameter.
        \param [in] b - a pointer to the second parameter.
        \param [in, out] d - a pointer to the D array.
        \param [in, out] w - a pointer to the W array.
    */
    SIMD_API void SimdNeuralUpdateWeights(const float * x, size_t size, const float * a, const float * b, float * d, float * w);

    /*! @ingroup neural

        \fn void SimdNeuralAdaptiveGradientUpdate(const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight);

        \short Updates neural network weights with using of adaptive gradients method.

        Adaptive gradients method.
        J Duchi, E Hazan and Y Singer,
        "Adaptive subgradient methods for online learning and stochastic optimization"
        The Journal of Machine Learning Research, pages 2121-2159, 2011.

        The algorithm performs:
        \verbatim
        for (i = 0; i < size; ++i)
        {
            d = delta[i]/batch;
            gradient[i] += d*d;
            weight[i] -= alpha * d / sqrt(gradient[i] + epsilon);
        }
        \endverbatim

        \note All arrays must have the same size. This function is used in Simd::Neural.

        \param [in] delta - a pointer to the array with error (delta).
        \param [in] size - a size of arrays.
        \param [in] batch - a batch size.
        \param [in] alpha - a pointer to alpha parameter (update speed).
        \param [in] epsilon - a pointer to epsilon parameter (a small number used to avoid division by zero).
        \param [in, out] gradient - a pointer to the array with gradients.
        \param [in, out] weight - a pointer to the array with weights.
    */
    SIMD_API void SimdNeuralAdaptiveGradientUpdate(const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution2x2Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

        \short Adds 2x2 convolution of 32-bit float image.

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the output image (input image width must be equal to output image width + 1).
        \param [in] height - a height of the output image (input image height must be equal to output image height + 1).
        \param [in] weights - a pointer to the array with weights (its size must be at least 4).
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralAddConvolution2x2Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution3x3Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

        \short Adds 3x3 convolution of 32-bit float image.

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the output image (input image width must be equal to output image width + 2).
        \param [in] height - a height of the output image (input image height must be equal to output image height + 2).
        \param [in] weights - a pointer to the array with weights (its size must be at least 9).
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralAddConvolution3x3Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution4x4Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

        \short Adds 4x4 convolution of 32-bit float image.

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the output image (input image width must be equal to output image width + 3).
        \param [in] height - a height of the output image (input image height must be equal to output image height + 3).
        \param [in] weights - a pointer to the array with weights (its size must be at least 16).
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralAddConvolution4x4Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);


    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution5x5Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

        \short Adds 5x5 convolution of 32-bit float image (forward propagation).

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the output image (input image width must be equal to output image width + 4).
        \param [in] height - a height of the output image (input image height must be equal to output image height + 4).
        \param [in] weights - a pointer to the array with weights (its size must be at least 25).
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralAddConvolution5x5Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution2x2Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

        \short Adds 2x2 convolution of 32-bit float image (backward propagation).

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the input image (output image width must be equal to input image width + 1).
        \param [in] height - a height of the input image (output image height must be equal to input image height + 1).
        \param [in] weights - a pointer to the array with weights (its size must be at least 4).
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralAddConvolution2x2Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution3x3Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

        \short Adds 3x3 convolution of 32-bit float image (backward propagation).

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the input image (output image width must be equal to input image width + 2).
        \param [in] height - a height of the input image (output image height must be equal to input image height + 2).
        \param [in] weights - a pointer to the array with weights (its size must be at least 9).
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralAddConvolution3x3Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution4x4Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

        \short Adds 4x4 convolution of 32-bit float image (backward propagation).

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the input image (output image width must be equal to input image width + 3).
        \param [in] height - a height of the input image (output image height must be equal to input image height + 3).
        \param [in] weights - a pointer to the array with weights (its size must be at least 16).
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralAddConvolution4x4Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution5x5Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

        \short Adds 5x5 convolution of 32-bit float image (backward propagation).

         \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the input image (output image width must be equal to input image width + 4).
        \param [in] height - a height of the input image (output image height must be equal to input image height + 4).
        \param [in] weights - a pointer to the array with weights (its size must be at least 25).
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralAddConvolution5x5Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution2x2Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);

        \short Accumulates changes of weights for 2x2 convolution of 32-bit float image during backward propagation.

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
        \param [in] width - a width of the output image (input image width must be equal to output image width + 1).
        \param [in] height - a height of the output image (input image height must be equal to output image height + 1).
        \param [in, out] sums - a pointer to the array with changes of weights (its size must be at least 4).
    */
    SIMD_API void SimdNeuralAddConvolution2x2Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);

        \short Accumulates changes of weights for 3x3 convolution of 32-bit float image during backward propagation.

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
        \param [in] width - a width of the output image (input image width must be equal to output image width + 2).
        \param [in] height - a height of the output image (input image height must be equal to output image height + 2).
        \param [in, out] sums - a pointer to the array with changes of weights (its size must be at least 9).
    */
    SIMD_API void SimdNeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution4x4Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);

        \short Accumulates changes of weights for 4x4 convolution of 32-bit float image during backward propagation.

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
        \param [in] width - a width of the output image (input image width must be equal to output image width + 3).
        \param [in] height - a height of the output image (input image height must be equal to output image height + 3).
        \param [in, out] sums - a pointer to the array with changes of weights (its size must be at least 16).
    */
    SIMD_API void SimdNeuralAddConvolution4x4Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);

    /*! @ingroup neural

        \fn void SimdNeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);

        \short Accumulates changes of weights for 5x5 convolution of 32-bit float image during backward propagation.

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
        \param [in] width - a width of the output image (input image width must be equal to output image width + 4).
        \param [in] height - a height of the output image (input image height must be equal to output image height + 4).
        \param [in, out] sums - a pointer to the array with changes of weights (its size must be at least 25).
    */
    SIMD_API void SimdNeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);

    /*! @ingroup neural

        \fn void SimdNeuralPooling1x1Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride);

        \short Takes maximum value in 3x3 window of input 32-bit float image and copies to the output image.

        \note This function is used in Simd::Neural. Output image must have the same size.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the input image.
        \param [in] height - a height of the input image.
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralPooling1x1Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride);

    /*! @ingroup neural

        \fn void SimdNeuralPooling2x2Max2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride);

        \short Reduces input 32-bit float image in two times (takes maximum value in 2x2 window and copies to the output image).

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the input image (output image width must have size (width + 1)/2).
        \param [in] height - a height of the input image (output image height must have size (height + 1)/2).
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralPooling2x2Max2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride);

    /*! @ingroup neural

        \fn void SimdNeuralPooling2x2Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride);

        \short Reduces input 32-bit float image in two times (takes maximum value in 3x3 window and copies to the output image).

        \note This function is used in Simd::Neural.

        \param [in] src - a pointer to the input 32-bit float image.
        \param [in] srcStride - a row size of the input image (in 32-float values).
        \param [in] width - a width of the input image (output image width must have size width/2).
        \param [in] height - a height of the input image (output image height must have size height/2).
        \param [in, out] dst - a pointer to the output 32-bit float image.
        \param [in] dstStride - a row size of the output image (in 32-float values).
    */
    SIMD_API void SimdNeuralPooling2x2Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride);

    /*! @ingroup neural

        \fn void SimdNeuralConvolutionForward(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth, const float * weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, void * buffer, size_t * size, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add);

        \short Adds convolution of the input multichannel 32-bit float image to the output multichannel 32-bit float image.

        \note There is a restriction to the size of output image:
        \verbatim
        dstWidth = (srcWidth + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1.
        dstHeight = (srcHeight + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1.
        \endverbatim

        \param [in] src - a pointer to the input multichannel 32-bit float image. Total size of the input image is equal to srcWidth*srcHeight*srcDepth.
        \param [in] srcWidth - a width of the input image.
        \param [in] srcHeight - a height of the input image.
        \param [in] srcDepth - a number of channels in the input image.
        \param [in] weight - a pointer to the convolution weights. Total size of the weights is equal to `kernelX*kernelY*srcDepth*dstDepth`.
        \param [in] kernelX - a width of the convolution kernel.
        \param [in] kernelY - a height of the convolution kernel.
        \param [in] padX - a pad to the x-coordinate of the input image.
        \param [in] padY - a pad to the y-coordinate of the input image.
        \param [in] strideX - a x-stride of the convolution.
        \param [in] strideY - a y-stride of the convolution.
        \param [in] dilationX - a x-stride of the convolution.
        \param [in] dilationY - a y-stride of the convolution.
        \param [in, out] buffer - a pointer to the external temporal buffer used by the algorithm. Can be NULL (the algorithm uses internal buffer).
        \param [in, out] size - a pointer to the size of the external temporal buffer. If the size is too small it will contain required value. Required size is approximately equal to `dstWidth*dstHeight*srcDepth*kernelX*kernelY*sizeof(float)`. Can be NULL.
        \param [in, out] dst - a pointer to the output multichannel 32-bit float image. Total size of the output image is equal to `dstWidth*dstHeight*dstDepth`.
        \param [in] dstWidth - a width of the output image.
        \param [in] dstHeight - a height of the output image.
        \param [in] dstDepth - a number of channels in the output image.
        \param [in] add - a flag which signalizes that we want add or assign value of convolution to the output image.
    */
    SIMD_API void SimdNeuralConvolutionForward(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth, const float * weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, void * buffer, size_t * size, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add);

    /*! @ingroup operation

        \fn void SimdOperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type);

        \short Performs given operation between two images.

        All images must have the same width, height and format (8-bit gray, 16-bit UV (UV plane of NV12 pixel format), 24-bit BGR or 32-bit BGRA).

        \note This function has a C++ wrappers: Simd::OperationBinary8u(const View<A>& a, const View<A>& b, View<A>& dst, SimdOperationBinary8uType type).

        \param [in] a - a pointer to pixels data of the first input image.
        \param [in] aStride - a row size of the first image.
        \param [in] b - a pointer to pixels data of the second input image.
        \param [in] bStride - a row size of the second image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count.
        \param [out] dst - a pointer to pixels data of output image.
        \param [in] dstStride - a row size of dst image.
        \param [in] type - a type of operation (see ::SimdOperationBinary8uType).
    */
    SIMD_API void SimdOperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type);

    /*! @ingroup operation

        \fn void SimdOperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type);

        \short Performs given operation between two images.

        All images must have the same width, height and ::SimdPixelFormatInt16 pixel format.

        \note This function has a C++ wrappers: Simd::OperationBinary16i(const View<A>& a, const View<A>& b, View<A>& dst, SimdOperationBinary16iType type).

        \param [in] a - a pointer to pixels data of the first input image.
        \param [in] aStride - a row size of the first image.
        \param [in] b - a pointer to pixels data of the second input image.
        \param [in] bStride - a row size of the second image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of output image.
        \param [in] dstStride - a row size of dst image.
        \param [in] type - a type of operation (see ::SimdOperationBinary16iType).
    */
    SIMD_API void SimdOperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
        size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type);

    /*! @ingroup operation

        \fn void SimdVectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height);

        \short Calculates result 8-bit gray image as product of two vectors.

        For all points:
        \verbatim
        dst[x, y] = horizontal[x]*vertical[y]/255;
        \endverbatim

        \note This function has a C++ wrappers: Simd::VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, View<A>& dst).

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
        \verbatim
        dst[x, y] = (src[2*x, 2*y] + src[2*x, 2*y + 1] + src[2*x + 1, 2*y] + src[2*x + 1, 2*y + 1] + 2)/4;
        \endverbatim

        \note This function has a C++ wrappers: Simd::ReduceGray2x2(const View<A>& src, View<A>& dst).

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
        \verbatim
        dst[x, y] = (src[2*x-1, 2*y-1] + 2*src[2*x, 2*y-1] + src[2*x+1, 2*y-1] +
                  2*(src[2*x-1, 2*y]   + 2*src[2*x, 2*y]   + src[2*x+1, 2*y]) +
                     src[2*x-1, 2*y+1] + 2*src[2*x, 2*y+1] + src[2*x+1, 2*y+1] + compensation ? 8 : 0) / 16;
        \endverbatim

        \note This function has a C++ wrappers: Simd::ReduceGray3x3(const View<A>& src, View<A>& dst, bool compensation).

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
        \verbatim
        dst[x, y] = (src[2*x-1, 2*y-1] + 3*src[2*x, 2*y-1] + 3*src[2*x+1, 2*y-1] + src[2*x+2, 2*y-1]
                  3*(src[2*x-1, 2*y]   + 3*src[2*x, 2*y]   + 3*src[2*x+1, 2*y]   + src[2*x+2, 2*y]) +
                  3*(src[2*x-1, 2*y+1] + 3*src[2*x, 2*y+1] + 3*src[2*x+1, 2*y+1] + src[2*x+2, 2*y+1]) +
                     src[2*x-1, 2*y+2] + 3*src[2*x, 2*y+2] + 3*src[2*x+1, 2*y+2] + src[2*x+2, 2*y+2] + 32) / 64;
        \endverbatim

        \note This function has a C++ wrappers: Simd::ReduceGray4x4(const View<A>& src, View<A>& dst).

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
        \verbatim
        dst[x, y] = (
               src[2*x-2, 2*y-2] + 4*src[2*x-1, 2*y-2] + 6*src[2*x, 2*y-2] + 4*src[2*x+1, 2*y-2] + src[2*x+2, 2*y-2] +
            4*(src[2*x-2, 2*y-1] + 4*src[2*x-1, 2*y-1] + 6*src[2*x, 2*y-1] + 4*src[2*x+1, 2*y-1] + src[2*x+2, 2*y-1]) +
            6*(src[2*x-2, 2*y]   + 4*src[2*x-1, 2*y]   + 6*src[2*x, 2*y]   + 4*src[2*x+1, 2*y]   + src[2*x+2, 2*y]) +
            4*(src[2*x-2, 2*y+1] + 4*src[2*x-1, 2*y+1] + 6*src[2*x, 2*y+1] + 4*src[2*x+1, 2*y+1] + src[2*x+2, 2*y+1]) +
               src[2*x-2, 2*y+2] + 4*src[2*x-1, 2*y+2] + 6*src[2*x, 2*y+2] + 4*src[2*x+1, 2*y+2] + src[2*x+2, 2*y+2] +
            compensation ? 128 : 0) / 256;
        \endverbatim

        \note This function has a C++ wrappers: Simd::ReduceGray5x5(const Viewc<A>& src, View<A>& dst, bool compensation).

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
        \verbatim
        dst[2*i + 0] = src[2*i + 1];
        dst[2*i + 1] = src[2*i + 0];
        \endverbatim

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
        \verbatim
        dst[4*i + 0] = src[4*i + 3];
        dst[4*i + 1] = src[4*i + 2];
        dst[4*i + 2] = src[4*i + 1];
        dst[4*i + 3] = src[4*i + 0];
        \endverbatim

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
        \verbatim
        dst[8*i + 0] = src[8*i + 7];
        dst[8*i + 1] = src[8*i + 6];
        dst[8*i + 2] = src[8*i + 5];
        dst[8*i + 3] = src[8*i + 4];
        dst[8*i + 4] = src[8*i + 3];
        dst[8*i + 5] = src[8*i + 2];
        dst[8*i + 6] = src[8*i + 1];
        dst[8*i + 7] = src[8*i + 0];
        \endverbatim

        The data size must be a multiple of 8.

        \param [in] src - a pointer to the input data.
        \param [in] size - a size of input and output data.
        \param [out] dst - a pointer to the output data.
    */
    SIMD_API void SimdReorder64bit(const uint8_t * src, size_t size, uint8_t * dst);

    /*! @ingroup resizing

        \fn void SimdResizeBilinear(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount);

        \short Performs resizing of input image with using bilinear interpolation.

        All images must have the same format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function has a C++ wrappers: Simd::ResizeBilinear(const View<A>& src, View<A>& dst).

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
        \verbatim
        if(mask[i] == oldIndex)
            mask[i] = newIndex;
        \endverbatim

        \note This function has a C++ wrappers: Simd::SegmentationChangeIndex(View<A> & mask, uint8_t oldIndex, uint8_t newIndex).

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

        \note This function has a C++ wrappers: Simd::SegmentationFillSingleHoles(View<A> & mask, uint8_t index).

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

        \note This function has a C++ wrappers: Simd::SegmentationPropagate2x2(const View<A> & parent, View<A> & child, const View<A> & difference, uint8_t currentIndex, uint8_t invalidIndex, uint8_t emptyIndex, uint8_t thresholdDifference).

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

        \note This function has a C++ wrappers: Simd::SegmentationShrinkRegion(const View<A> & mask, uint8_t index, Rectangle<ptrdiff_t> & rect).

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

        \fn void SimdShiftBilinear(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, const uint8_t * bkg, size_t bkgStride, const double * shiftX, const double * shiftY, size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride);

        \short Performs shifting of input image with using bilinear interpolation.

        All images must have the same width, height and format (8-bit gray, 16-bit UV, 24-bit BGR or 32-bit BGRA).

        \note This function has a C++ wrappers: Simd::ShiftBilinear(const View<A> & src, const View<A> & bkg, const Point<double> & shift, const Rectangle<ptrdiff_t> & crop, View<A> & dst).

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
        const uint8_t * bkg, size_t bkgStride, const double * shiftX, const double * shiftY,
        size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride);

    /*! @ingroup sobel_filter

        \fn void SimdSobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Calculates Sobel's filter along x axis.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.

        For every point:
        \n dst[x, y] = (src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1]).

        \note This function has a C++ wrappers: Simd::SobelDx(const View<A>& src, View<A>& dst).

        \param [in] src - a pointer to pixels data of the input image.
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
        \verbatim
        dst[x, y] = (src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1]).
        \endverbatim

        \note This function has a C++ wrappers: Simd::SobelDxAbs(const View<A>& src, View<A>& dst).

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of the output image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdSobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup sobel_statistic

        \fn void SimdSobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

        \short Calculates sum of absolute value of Sobel's filter along x axis.

        Input image must has 8-bit gray format.

        For every point:
        \verbatim
        dst[x, y] = abs((src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1])).
        \endverbatim

        \note This function has a C++ wrappers: Simd::SobelDxAbsSum(const View<A>& src, uint64_t & sum).

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] stride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - a pointer to unsigned 64-bit integer value with result sum.
    */
    SIMD_API void SimdSobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

    /*! @ingroup sobel_filter

        \fn void SimdSobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

        \short Calculates Sobel's filter along y axis.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.

        For every point:
        \verbatim
        dst[x, y] = (src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]);
        \endverbatim

        \note This function has a C++ wrappers: Simd::SobelDy(const View<A>& src, View<A>& dst).

        \param [in] src - a pointer to pixels data of the input image.
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
        \verbatim
        dst[x, y] = abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]));
        \endverbatim

        \note This function has a C++ wrappers: Simd::SobelDyAbs(const View<A>& src, View<A>& dst).

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of the output image.
        \param [in] dstStride - a row size of the output image (in bytes).
    */
    SIMD_API void SimdSobelDyAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

    /*! @ingroup sobel_statistic

        \fn void SimdSobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

        \short Calculates sum of absolute value of Sobel's filter along y axis.

        Input image must has 8-bit gray format.

        For every point:
        \verbatim
        sum += abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]));
        \endverbatim

        \note This function has a C++ wrappers: Simd::SobelDyAbsSum(const View<A>& src, uint64_t & sum).

        \param [in] src - a pointer to pixels data of the input image.
        \param [in] stride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - a pointer to unsigned 64-bit integer value with result sum.
    */
    SIMD_API void SimdSobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

    /*! @ingroup contour

        \fn void SimdContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)

        \short Calculates contour metrics based on absolute value and direction of Sobel's filter along y and y axis.

        All images must have the same width and height. Input image must has 8-bit gray format, output image must has 16-bit integer format.
        This function is used for contour extraction.

        For every point:
        \verbatim
        dy = abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]));
        dx = abs((src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1]));
        dst[x, y] = (dx + dy)*2 + (dx >= dy ? 0 : 1);
        \endverbatim

        \note This function has a C++ wrappers: Simd::ContourMetrics(const View<A>& src, View<A>& dst).

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
        \verbatim
        dy = abs((src[x-1,y+1] + 2*src[x, y+1] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x, y-1] + src[x+1, y-1]));
        dx = abs((src[x+1,y-1] + 2*src[x+1, y] + src[x+1, y+1]) - (src[x-1,y-1] + 2*src[x-1, y] + src[x-1, y+1]));
        dst[x, y] = mask[x, y] < indexMin ? 0 : (dx + dy)*2 + (dx >= dy ? 0 : 1);
        \endverbatim

        \note This function has a C++ wrappers: Simd::ContourMetrics(const View<A>& src, const View<A>& mask, uint8_t indexMin, View<A>& dst).

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
        \verbatim
        a[x, y] = src[x, y] >> 1.
        if(src[x, y] & 1)
            dst[x, y] = a[x, y] > 0 && (a[x, y] - a[x + 1, y] >= threshold) && (a[x, y] - a[x - 1, y] >= threshold) ? 255 : 0;
        else
            dst[x, y] = a[x, y] > 0 && (a[x, y] - a[x, y + 1] >= threshold) && (a[x, y] - a[x, y - 1] >= threshold) ? 255 : 0;
        \endverbatim

        \note This function has a C++ wrappers: Simd::ContourAnchors(const View<A>& src, size_t step, int16_t threshold, View<A>& dst).

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
        \verbatim
        sum += (a[i] - b[i])*(a[i] - b[i]);
        \endverbatim

        \note This function has a C++ wrappers: Simd::SquaredDifferenceSum(const View<A>& a, const View<A>& b, uint64_t & sum).

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

        For every point:
        \verbatim
        if(mask[i] == index)
            sum += (a[i] - b[i])*(a[i] - b[i]);
        \endverbatim

        \note This function has a C++ wrappers: Simd::SquaredDifferenceSum(const View<A>& a, const View<A>& b, const View<A>& mask, uint8_t index, uint64_t & sum).

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

        \fn void SimdSquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum);

        \short Calculates sum of squared differences for two 32-bit float arrays.

        All arrays must have the same size.

        For every element:
        \verbatim
        sum += (a[i] - b[i])*(a[i] - b[i]);
        \endverbatim

        \param [in] a - a pointer to the first array.
        \param [in] b - a pointer to the second array.
        \param [in] size - a size of arrays.
        \param [out] sum - a sum of squared differences.
    */
    SIMD_API void SimdSquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum);

    /*! @ingroup correlation

        \fn void SimdSquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum);

        \short Calculates sum of squared differences for two 32-bit float arrays with using Kahan summation algorithm.

        All arrays must have the same size.

        Algorithm pseudo code:
        \verbatim
        sum = 0; corr = 0;
        for(i = 0; i < size; ++i)
        {
            diff = (a[i] - b[i])*(a[i] - b[i]) - corr;
            temp = sum + diff;
            corr = (temp - sum) - diff;
            sum = temp;
        }
        \endverbatim

        \param [in] a - a pointer to the first array.
        \param [in] b - a pointer to the second array.
        \param [in] size - a size of arrays.
        \param [out] sum - a sum of squared differences.
    */
    SIMD_API void SimdSquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum);

    /*! @ingroup other_statistic

        \fn void SimdGetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t * min, uint8_t * max, uint8_t * average);

        \short Finds minimal, maximal and average pixel values for given image.

        The image must has 8-bit gray format.

        \note This function has a C++ wrappers: Simd::GetStatistic(const View<A>& src, uint8_t & min, uint8_t & max, uint8_t & average).

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

        For every point:
        \verbatim
        if(mask[X, Y] == index)
        {
            area += 1.
            x += X.
            y += Y.
            xx += X*X.
            xy += X*Y.
            yy += Y*Y.
        }
        \endverbatim

        \note This function has a C++ wrappers: Simd::GetMoments(const View<A>& mask, uint8_t index, uint64_t & area, uint64_t & x, uint64_t & y, uint64_t & xx, uint64_t & xy, uint64_t & yy).

        \param [in] mask - a pointer to pixels data of the mask image.
        \param [in] stride - a row size of the mask image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] index - a mask index.
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
        \verbatim
        for(x = 0; x < width; ++x)
            sums[y] += src[x, y];
        \endverbatim

        \note This function has a C++ wrappers: Simd::GetRowSums(const View<A>& src, uint32_t * sums).

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
        \verbatim
        for(y = 0; y < height; ++y)
            sums[x] += src[x, y];
        \endverbatim

        \note This function has a C++ wrappers: Simd::GetColSums(const View<A>& src, uint32_t * sums).

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
        \verbatim
        for(x = 0; x < width; ++x)
            sums[y] += abs(src[x, y+1] - src[x, y]);
        \endverbatim
        For the last row:
        \verbatim
        sums[height-1] = 0;
        \endverbatim

        \note This function has a C++ wrappers: Simd::GetAbsDyRowSums(const View<A>& src, uint32_t * sums).

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
        \verbatim
        for(y = 0; y < height; ++y)
            sums[y] += abs(src[x+1, y] - src[x, y]);
        \endverbatim
        For the last column:
        \verbatim
        sums[width-1] = 0;
        \endverbatim

        \note This function has a C++ wrappers: Simd::GetAbsDxColSums(const View<A>& src, uint32_t * sums).

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

        \note This function has a C++ wrappers: Simd::ValueSum(const View<A>& src, uint64_t & sum).

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

        \note This function has a C++ wrappers: Simd::SquareSum(const View<A>& src, uint64_t & sum).

        \param [in] src - a pointer to pixels data of the image.
        \param [in] stride - a row size of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] sum - the result sum.
    */
    SIMD_API void SimdSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum);

    /*! @ingroup other_statistic

        \fn void SimdCorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum);

        \short Gets sum of pixel correlation for two gray 8-bit images.

        For all points:
        \verbatim
        sum += a[i]*b[i];
        \endverbatim

        All images must have the same width and height and 8-bit gray pixel format.

        \note This function has a C++ wrappers: Simd::CorrelationSum(const View<A> & a, const View<A> & b, uint64_t & sum).

        \param [in] a - a pointer to pixels data of the first image.
        \param [in] aStride - a row size of the first image.
        \param [in] b - a pointer to pixels data of the second image.
        \param [in] bStride - a row size of the second image.
        \param [in] width - an images width.
        \param [in] height - an images height.
        \param [out] sum - a pointer to result sum.
    */
    SIMD_API void SimdCorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum);

    /*! @ingroup resizing

        \fn void SimdStretchGray2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

        \short Stretches input 8-bit gray image in two times.

        \note This function has a C++ wrappers: Simd::StretchGray2x2(const View<A>& src, View<A>& dst).

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

    /*! @ingroup svm

        \fn void SimdSvmSumLinear(const float * x, const float * svs, const float * weights, size_t length, size_t count, float * sum);

        \short It is a part of linear SVM (Support Vector Machine) prediction algorithm.

        Algorithm's details:
        \verbatim
        sum = 0;
        for(i = 0; i < count; ++i)
            for(j = 0; j < length; ++j)
                sum += x[j]*svs[j][i]*weight[i];
        \endverbatim

        \note The array with support vectors must has following structure: svs[length][count].

        \param [in] x - a vector of features which need to predict with using SVM.
        \param [in] svs - an array with support vectors.
        \param [in] weights - a weight coefficient of each support vector.
        \param [in] length - a length of these current and support vectors.
        \param [in] count - a count of support vectors.
        \param [out] sum - a pointer to result sum.
    */
    SIMD_API void SimdSvmSumLinear(const float * x, const float * svs, const float * weights, size_t length, size_t count, float * sum);

    /*! @ingroup texture_estimation

        \fn void SimdTextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride);

        \short Calculates boosted saturated gradients for given input image.

        All images must have the same width, height and format (8-bit gray).

        For border pixels:
        \verbatim
        dx[x, y] = 0;
        dy[x, y] = 0;
        \endverbatim
        For other pixels:
        \verbatim
        dx[x, y] = (saturation + max(-saturation, min(saturation, (src[x + 1, y] - src[x - 1, y]))))*boost;
        dy[x, y] = (saturation + max(-saturation, min(saturation, (src[x, y + 1] - src[x, y - 1]))))*boost;
        \endverbatim

        \note This function has a C++ wrappers: Simd::TextureBoostedSaturatedGradient(const View<A>& src, uint8_t saturation, uint8_t boost, View<A>& dx, View<A>& dy).

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
        \verbatim
        lo = 128 - (128/boost);
        hi = 255 - lo;
        dst[x, y] = max(lo, min(hi, src[i]))*boost;
        \endverbatim

        \note This function has a C++ wrappers: Simd::TextureBoostedUv(const View<A>& src, uint8_t boost, View<A>& dst).

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
        \verbatim
        sum += current - average(lo[i], hi[i]);
        \endverbatim

        \note This function has a C++ wrappers: Simd::TextureGetDifferenceSum(const View<A>& src, const View<A>& lo, const View<A>& hi, int64_t & sum).

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
        \verbatim
        dst[i] = max(0, min(255, src[i] + shift));
        \endverbatim

        \note This function has a C++ wrappers: Simd::TexturePerformCompensation(const View<A>& src, int shift, View<A>& dst).

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

        \note This function has a C++ wrappers: Simd::Yuv420pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr);

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

        \fn void SimdYuv422pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

        \short Converts YUV422P image to 24-bit BGR image.

        The input Y and output BGR images must have the same width and height.
        The input U and V images must have the same width and height (their width is equal to half width of Y component).

        \note This function has a C++ wrappers: Simd::Yuv422pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr);

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
    SIMD_API void SimdYuv422pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv444pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

        \short Converts YUV444P image to 24-bit BGR image.

        The input Y, U, V and output BGR images must have the same width and height.

        \note This function has a C++ wrappers: Simd::Yuv444pToBgr(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgr);

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

        \note This function has a C++ wrappers: Simd::Yuv420pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha).

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

        \fn void SimdYuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

        \short Converts YUV422P image to 32-bit BGRA image.

        The input Y and output BGRA images must have the same width and height.
        The input U and V images must have the same width and height (their width is equal to half width of Y component).

        \note This function has a C++ wrappers: Simd::Yuv422pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha).

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
    SIMD_API void SimdYuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
        size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

        \short Converts YUV444P image to 32-bit BGRA image.

        The input Y, U, V and output BGRA images must have the same width and height.

        \note This function has a C++ wrappers: Simd::Yuv444pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& bgra, uint8_t alpha).

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

        \note This function has a C++ wrappers: Simd::Yuv444pToHsl(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hsl).

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

        \note This function has a C++ wrappers: Simd::Yuv444pToHsv(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hsv).

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

        \note This function has a C++ wrappers: Simd::Yuv420pToHue(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hue).

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

        \note This function has a C++ wrappers: Simd::Yuv444pToHue(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& hue).

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
