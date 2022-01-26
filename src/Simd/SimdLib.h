/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar,
*               2014-2019 Antonenka Mikhail,
*               2019-2019 Facundo Galan.
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

#if defined(_MSC_VER) || defined(__CODEGEARC__)

#define SIMD_INLINE __forceinline

#elif defined(__GNUC__)

#define SIMD_INLINE inline __attribute__ ((always_inline))

#else

#error This platform is unsupported!

#endif

#if defined(__GNUC__) || (defined(_MSC_VER) && (_MSC_VER >= 1600)) || (defined(__CODEGEARC__) && (__CODEGEARC__ >= 1840))
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
    Describes Bayer pixel layout.
*/
typedef enum
{
    /*! A Bayer pixel layout (GRBG). */
    SimdBayerLayoutGrbg,
    /*! A Bayer pixel layout (GBRG). */
    SimdBayerLayoutGbrg,
    /*! A Bayer pixel layout (RGGB). */
    SimdBayerLayoutRggb,
    /*! A Bayer pixel layout (BGGR). */
    SimdBayerLayoutBggr,
} SimdBayerLayoutType;

/*! @ingroup c_types
    Describes boolean type.
*/
typedef enum
{
    SimdFalse = 0, /*!< False value. */
    SimdTrue = 1, /*!< True value. */
} SimdBool;

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

/*! @ingroup synet
    Describes type of activation function. 
    It is used in ::SimdSynetConvolution32fInit, ::SimdSynetConvolution8iInit, ::SimdSynetDeconvolution32fInit, 
    ::SimdSynetInnerProduct32fInit, ::SimdSynetMergedConvolution32fInit and ::SimdSynetMergedConvolution8iInit.
*/
typedef enum
{
    /*!
        Identity (activation function is absent).
    */
    SimdConvolutionActivationIdentity = 0,
    /*!
        ReLU activation function.
        \verbatim
        dst[i] = Max(0, src[i]);
        \endverbatim
    */
    SimdConvolutionActivationRelu,
    /*!
        Leaky ReLU activation function.
        It has one parameter: slope (params[0]).
        \verbatim
        dst[i] = src[i] > 0 ? src[i] : slope*src[i];
        \endverbatim
    */
    SimdConvolutionActivationLeakyRelu,
    /*!
        The activation function restricts range.
        It has two parameters: lower (params[0]) and upper (params[1]) bound.
        \verbatim
        dst[i] = Min(Max(lower, src[i]), upper);
        \endverbatim
    */
    SimdConvolutionActivationRestrictRange,
    /*!
        Leaky PReLU activation function.
        It has m parameters: slopes[m] (m = dstC, n = dstH*dstW).
        \verbatim
        dst[i*n + j] = src[i*n + j] > 0 ? src[i*n + j] : slopes[i]*src[i*n + j];
        \endverbatim
    */
    SimdConvolutionActivationPrelu,
    /*!
        Leaky ELU activation function.
        It has one parameter: alpha (params[0]).
        \verbatim
        dst[i] = src[i] >= 0 ? src[i] : alpha*(Exp(src[i]) - 1);
        \endverbatim
    */
    SimdConvolutionActivationElu,
    /*!
        H-Swish (https://arxiv.org/pdf/1905.02244.pdf) activation function.
        It has two parameters: shift (params[0]) and scale (params[1]).
        \verbatim
        dst[i] = Max(Min(src[i], shift) + shift, 0)*scale*src[i];
        \endverbatim
    */
    SimdConvolutionActivationHswish,
    /*!
        Mish (https://arxiv.org/abs/1908.08681) activation function.
        It has parameter: threshold (params[0]).
        \verbatim
        dst[i] = src[i] > threshold ? src[i] : src[i] * tanh(log(exp(src[i]) + 1));
        \endverbatim
    */
    SimdConvolutionActivationMish,
    /*!
        HardSigmoid (https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html) activation function.
        It has two parameters: scale (params[0]) and shift (params[1]).
        \verbatim
        dst[i] = Max(0, Min(src[i] * scale + shift, 1));
        \endverbatim
    */
    SimdConvolutionActivationHardSigmoid,
    /*!
        Swish (https://en.wikipedia.org/wiki/Swish_function) activation function.
        It has one parameter: slope (params[0]).
        \verbatim
        dst[i] = src[i]/(1 + Exp(-slope*src[i]));
        \endverbatim
    */
    SimdConvolutionActivationSwish,
} SimdConvolutionActivationType;

/*! @ingroup c_types
    Describes type of information which can return function ::SimdCpuInfo.
*/
typedef enum
{
    SimdCpuInfoSockets,/*!< A number of sockets. */
    SimdCpuInfoCores, /*!< A number of psysical CPU cores. */
    SimdCpuInfoThreads, /*!< A number of logical CPU cores. */
    SimdCpuInfoCacheL1, /*!< A size of level 1 data cache. */
    SimdCpuInfoCacheL2, /*!< A size of level 2 cache. */
    SimdCpuInfoCacheL3, /*!< A size of level 3 cache. */
    SimdCpuInfoSse2, /*!< Availability of SSE2 (x86). */
    SimdCpuInfoSse41, /*!< Availability of SSE4.1 (x86). */
    SimdCpuInfoAvx, /*!< Availability of AVX (x86). */
    SimdCpuInfoAvx2, /*!< Availability of AVX2 (x86). */
    SimdCpuInfoAvx512f, /*!< Availability of AVX-512F (x86). */
    SimdCpuInfoAvx512bw, /*!< Availability of AVX-512BW (x86). */
    SimdCpuInfoAvx512vnni, /*!< Availability of AVX-512VNNI (x86). */
    SimdCpuInfoVmx, /*!< Availability of VMX or Altivec (PowerPC). */
    SimdCpuInfoVsx, /*!< Availability of VSX (PowerPC). */
    SimdCpuInfoNeon, /*!< Availability of NEON (ARM). */
} SimdCpuInfoType;

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
    Describes formats of image file. It is used in functions ::SimdImageSaveToMemory and ::SimdImageSaveToFile.
*/
typedef enum
{
    /*! An undefined image file format (format auto choice). */
    SimdImageFileUndefined = 0,
    /*! A PGM (Portable Gray Map) text (P2) image file format. */
    SimdImageFilePgmTxt,
    /*! A PGM (Portable Gray Map) binary (P5) image file format. */
    SimdImageFilePgmBin,
    /*! A PGM (Portable Pixel Map) text (P3) image file format. */
    SimdImageFilePpmTxt,
    /*! A PGM (Portable Pixel Map) binary (P6) image file format. */
    SimdImageFilePpmBin,
    /*! A PNG (Portable Network Graphics) image file format. */
    SimdImageFilePng,
    /*! A JPEG (Joint Photographic Experts Group) image file format. */
    SimdImageFileJpeg,
} SimdImageFileType;

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
    /*! A 24-bit (3 8-bit channels) RGB (Red, Green, Blue) pixel format. */
    SimdPixelFormatRgb24,
    /*! A 32-bit (4 8-bit channels) RGBA (Red, Green, Blue, Alpha) pixel format. */
    SimdPixelFormatRgba32,
    /*! A 16-bit (2 8-bit channels) UYVY422 pixel format. */
    SimdPixelFormatUyvy16,
} SimdPixelFormatType;

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

/*! @ingroup resizing
    Describes resized image channel types.
*/
typedef enum
{
    /*! 8-bit integer channel type.  */
    SimdResizeChannelByte,
    /*! 16-bit integer channel type.  */
    SimdResizeChannelShort,
    /*! 32-bit float channel type.  */
    SimdResizeChannelFloat,
} SimdResizeChannelType;

/*! @ingroup resizing
    Describes methods used in order to resize image.
*/
typedef enum
{
    /*! Nearest method. */
    SimdResizeMethodNearest,
    /*! Nearest Pytorch compatible method. */
    SimdResizeMethodNearestPytorch,
    /*! Bilinear method. */
    SimdResizeMethodBilinear,
    /*! Bilinear Caffe compatible method. It is relevant only for ::SimdResizeChannelFloat (32-bit float channel type).*/
    SimdResizeMethodBilinearCaffe,
    /*! Bilinear Pytorch compatible method. It is relevant only for ::SimdResizeChannelFloat (32-bit float channel type).*/
    SimdResizeMethodBilinearPytorch,
    /*! Bicubic method. */
    SimdResizeMethodBicubic,
    /*! Area method. */
    SimdResizeMethodArea,
} SimdResizeMethodType;

/*! @ingroup synet
    Describes Synet compatibility flags. This type used in functions ::SimdSynetScaleLayerForward, ::SimdSynetConvert32fTo8u, 
    ::SimdSynetConvolution8iInit, and ::SimdSynetMergedConvolution8iInit.
*/
typedef enum
{
    SimdSynetCompatibilityFmaUse = 0, /*!< Fast (No compatibility for fast code). */
    SimdSynetCompatibilityFmaNoTail = 1, /*!< Not use FMA instructions at row tail. */
    SimdSynetCompatibilityFmaAvoid = 2, /*!< Not use FMA instructions. */
    SimdSynetCompatibilityFmaMask = 3, /*!< Bit mask of options of FMA instructions using. */
    SimdSynetCompatibility8iPrecise = 0, /*!< Using of precise 8-bit integer multiplication (VNNI, or its 16-bit emulation). */
    SimdSynetCompatibility8iOverflow = 4, /*!< Allow 16-bit integer overflow. */
    SimdSynetCompatibility8iNarrowed = 8, /*!< Using of narrowed range (signed: [-90 .. 90], unsigned: [0 .. 180]) to awoid 16-bit integer overflow. */
    SimdSynetCompatibility8iMask = 12, /*!< Bit mask of options of 8-bit integer multiplication. */
    SimdSynetCompatibilityFloatZero = 16, /*!< Bit flag of asymmetric 8-bit integer quantization. */
} SimdSynetCompatibilityType;

/*! @ingroup synet
    Describes operation type used in function ::SimdSynetEltwiseLayerForward.
*/
typedef enum
{
    SimdSynetEltwiseOperationProduct, /*!< Product. */
    SimdSynetEltwiseOperationSum, /*!< Weighted sum. */
    SimdSynetEltwiseOperationMax, /*!< Maximum. */
    SimdSynetEltwiseOperationMin, /*!< Minimum. */
} SimdSynetEltwiseOperationType;

/*! @ingroup synet
    Describes operation type used in function ::SimdSynetUnaryOperation32fLayerForward.
*/
typedef enum
{
    /*! Gets absolute value for every point of input tensor. */
    SimdSynetUnaryOperation32fAbs,
    /*! Gets exponent for every point of input tensor. */
    SimdSynetUnaryOperation32fExp,
    /*! Gets logarithm for every point of input tensor. */
    SimdSynetUnaryOperation32fLog,
    /*! Gets negative for every point of input tensor. */
    SimdSynetUnaryOperation32fNeg,
    /*! Gets reverse square root for every point of input tensor. */
    SimdSynetUnaryOperation32fRsqrt,
    /*! Gets square root for every point of input tensor. */
    SimdSynetUnaryOperation32fSqrt,
    /*! Gets hyperbolic tangent for every point of input tensor. */
    SimdSynetUnaryOperation32fTanh,
    /*! Gets zero value for every point of input tensor. */
    SimdSynetUnaryOperation32fZero,
} SimdSynetUnaryOperation32fType;

/*! @ingroup synet
    Describes <a href="http://github.com/ermig1979/Synet">Synet Framework</a> 4D-tensor format type.
*/
typedef enum
{
    SimdTensorFormatUnknown = -1, /*!< Unknown tensor format. */
    SimdTensorFormatNchw, /*!< NCHW (N - batch, C - channels, H - height, W - width) 4D-tensor format of (input/output) image. */
    SimdTensorFormatNhwc, /*!< NHWC (N - batch, H - height, W - width, C - channels) 4D-tensor format of (input/output) image. */
    SimdTensorFormatNchw4c, /*!< NCHW4c (N - batch, C - (channels + 3) / 4, H - height, W - width, 4c - channels gropped by 4) special 5D-tensor format of (input/output) image optimized for SSE and NEON. */
    SimdTensorFormatNchw8c, /*!< NCHW8c (N - batch, C - (channels + 7) / 8, H - height, W - width, 8c - channels gropped by 8) special 5D-tensor format of (input/output) image optimized for AVX and AVX2. */
    SimdTensorFormatNchw16c, /*!< NCHW16c (N - batch, C - (channels + 15) / 16, H - height, W - width, 16c - channels gropped by 16) special 5D-tensor format of (input/output) image optimized for AVX-512. */
    SimdTensorFormatNchwXc, /*!< Unspecified hardware optimized 5D-tensor format of (input/output) image. Specific format (::SimdTensorFormatNchw4c, ::SimdTensorFormatNchw8c or ::SimdTensorFormatNchw16c) is determinated by function ::SimdSynetSpecifyTensorFormat. */
    SimdTensorFormatOiyx, /*!< OIYX (O - output channels, I - input channels, Y - kernel height, X - kernel width) 4D-tensor format of 2D-convolution filter. */
    SimdTensorFormatYxio, /*!< YXIO (Y - kernel height, X - kernel width, I - input channels, O - output channels) 4D-tensor format of 2D-convolution filter. */
    SimdTensorFormatOyxi4o, /*!< OYXI4o (O - (output channels + 3)/4, Y - kernel height, X - kernel width, I - input channels, 4o - output channels gropped by 4) special 5D-tensor format of 2D-convolution filter optimized for SSE and NEON. */
    SimdTensorFormatOyxi8o, /*!< OYXI8o (O - (output channels + 7)/8, Y - kernel height, X - kernel width, I - input channels, 8o - output channels gropped by 8) special 5D-tensor format of 2D-convolution filter optimized for AVX and AVX2. */
    SimdTensorFormatOyxi16o, /*!< OYXI16o (O - (output channels + 15)/16, Y - kernel height, X - kernel width, I - input channels, 16o - output channels gropped by 16) special 5D-tensor format of 2D-convolution filter optimized for AVX-512. */
    SimdTensorFormatOyxiXo, /*!< Unspecified hardware optimized 5D-tensor format of 2D-convolution filter. Specific format (::SimdTensorFormatOyxi4o, ::SimdTensorFormatOyxi8o or ::SimdTensorFormatOyxi16o) is determinated by function ::SimdSynetSpecifyTensorFormat. */
} SimdTensorFormatType;

/*! @ingroup synet
    Describes <a href="http://github.com/ermig1979/Synet">Synet Framework</a> tensor data type.
*/
typedef enum
{
    SimdTensorDataUnknown = -1, /*!< Unknown tensor data type. */
    SimdTensorData32f, /*!< 32-bit float point. */
    SimdTensorData32i, /*!< 32-bit signed integer. */
    SimdTensorData8i, /*!< 8-bit signed integer. */
    SimdTensorData8u, /*!< 8-bit unsigned integer. */
} SimdTensorDataType;

/*! @ingroup transform
    Describes transform type used in function ::SimdTransformImage in order to describe result of transformation.
*/
typedef enum
{
    SimdTransformRotate0 = 0, /*!< An original image. The output image has the same size as input image.*/
    SimdTransformRotate90, /*!< Image rotated 90 degrees counterclockwise. The output width and height are equal to the input height and widht. */
    SimdTransformRotate180, /*!< Image rotated 180 degrees counterclockwise. The output image has the same size as input image. */
    SimdTransformRotate270, /*!< Image rotated 270 degrees counterclockwise. The output width and height are equal to the input height and widht. */
    SimdTransformTransposeRotate0, /*!< Transposed image. The output width and height are equal to the input height and widht. */
    SimdTransformTransposeRotate90, /*!< Image transposed and rotated 90 degrees counterclockwise. It is equal to horizontal mirroring of image. The output image has the same size as input image.*/
    SimdTransformTransposeRotate180, /*!< Image transposed and rotated 180 degrees counterclockwise. The output width and height are equal to the input height and widht. */
    SimdTransformTransposeRotate270, /*!< Image transposed and rotated 270 degrees counterclockwise. It is equal to vertical mirroring of image. The output image has the same size as input image.*/
} SimdTransformType;

/*! @ingroup yuv_conversion
    Describes YUV format type. It is uses in YUV to BGR forward and backward conversions.
*/
typedef enum
{
    SimdYuvUnknown = -1, /*!< Unknown YUV standard. */
    SimdYuvBt601, /*!< Corresponds to BT.601 standard. Uses Kr=0.299, Kb=0.114. Restricts Y to range [16..235], U and V to [16..240]. */
    SimdYuvBt709, /*!< Corresponds to BT.709 standard. Uses Kr=0.2126, Kb=0.0722. Restricts Y to range [16..235], U and V to [16..240]. */
    SimdYuvBt2020, /*!< Corresponds to BT.2020 standard. Uses Kr=0.2627, Kb=0.0593. Restricts Y to range [16..235], U and V to [16..240]. */
    SimdYuvTrect871, /*!< Corresponds to T-REC-T.871 standard. Uses Kr=0.299, Kb=0.114. Y, U and V use full range [0..255]. */
} SimdYuvType;

/*! @ingroup synet
    \brief Callback function type "SimdGemm32fNNPtr";

    The function has to perform general matrix multiplication (for 32-bit float numbers).

    \verbatim
    C(M, N) = alpha*A(M, K)*B(K, N) + beta*C(M, N);
    \endverbatim

    \param [in] M - a height of A and height of C matrices.
    \param [in] N - a width of B and width of C matrices.
    \param [in] K - a width of A and height of B matrices.
    \param [in] alpha - a pointer to multiplier of the first term.
    \param [in] A - a pointer to input A matrix.
    \param [in] lda - a leading dimension of A matrix.
    \param [in] B - a pointer to input B matrix.
    \param [in] ldb - a leading dimension of B matrix.
    \param [in] beta - a pointer to multiplier of the second term.
    \param [out] C - a pointer to output C matrix.
    \param [in] ldc - a leading dimension of C matrix.
*/
typedef void(*SimdGemm32fNNPtr)(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc);

/*! @ingroup synet
    Describes convolution (deconvolution) parameters. It is used in ::SimdSynetConvolution32fInit, ::SimdSynetConvolution8iInit, 
    ::SimdSynetDeconvolution32fInit, ::SimdSynetMergedConvolution32fInit and ::SimdSynetMergedConvolution8iInit.
*/
typedef struct SimdConvolutionParameters
{
    /*!
        A number of input tensor channels.
    */
    size_t srcC;
    /*!
        An input tensor height.
    */
    size_t srcH;
    /*!
        An input tensor width.
    */
    size_t srcW;
    /*!
        An input tensor data type.
    */
    SimdTensorDataType srcT;
    /*!
        An input tensor data format.
    */
    SimdTensorFormatType srcF;
    /*!
        A number of output tensor channels.
    */
    size_t dstC;
    /*!
        An output tensor height.
    */
    size_t dstH;
    /*!
        An output tensor width.
    */
    size_t dstW;
    /*!
        An output tensor data type.
    */
    SimdTensorDataType dstT;
    /*!
        An output tensor data format.
    */
    SimdTensorFormatType dstF;
    /*!
        A convolution (deconvolution) kernel window height.
    */
    size_t kernelY;
    /*!
        A convolution (deconvolution) kernel window width.
    */
    size_t kernelX;
    /*!
        A convolution (deconvolution) dilation along Y-axis.
    */
    size_t dilationY;
    /*!
        A convolution (deconvolution) dilation along X-axis.
    */
    size_t dilationX;
    /*!
        A convolution (deconvolution) stride along Y-axis.
    */
    size_t strideY;
    /*!
        A convolution (deconvolution) stride along X-axis.
    */
    size_t strideX;
    /*!
        An additional zero padding of input image at the beginning of Y-axis.
    */
    size_t padY;
    /*!
        An additional zero padding of input image at the beginning of X-axis.
    */
    size_t padX;
    /*!
        An additional zero padding of input image at the end of Y-axis.
    */
    size_t padH;
    /*!
        An additional zero padding of input image at the end of X-axis.
    */
    size_t padW;
    /*!
        A number of convolution (deconvolution) groups.
    */
    size_t group;
    /*!
        An activation function type used after convolution (deconvolution).
    */
    SimdConvolutionActivationType activation;
} SimdConvolutionParameters;

#if defined(WIN32) && !defined(SIMD_STATIC)
#  ifdef SIMD_EXPORTS
#    define SIMD_API __declspec(dllexport)
#  else
#    define SIMD_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && defined(SIMD_HIDE_INTERNAL)
#    define SIMD_API __attribute__ ((visibility ("default")))
#else
#    define SIMD_API
#endif

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

        \fn size_t SimdCpuInfo(SimdCpuInfoType type);

        \short Gets info about CPU and %Simd Library.

        \note See enumeration ::SimdCpuInfoType.

        Using example:
        \verbatim
        #include "Simd/SimdLib.h"
        #include <iostream>

        int main()
        {
            std::cout << "Sockets : " << SimdCpuInfo(SimdCpuInfoSockets) << std::endl;
            std::cout << "Cores : " << SimdCpuInfo(SimdCpuInfoCores) << std::endl;
            std::cout << "Threads : " << SimdCpuInfo(SimdCpuInfoThreads) << std::endl;
            std::cout << "L1D Cache : " << SimdCpuInfo(SimdCpuInfoCacheL1) / 1024  << " KB" << std::endl;
            std::cout << "L2 Cache : " << SimdCpuInfo(SimdCpuInfoCacheL2) / 1024  << " KB" << std::endl;
            std::cout << "L3 Cache : " << SimdCpuInfo(SimdCpuInfoCacheL3) / 1024  << " KB" << std::endl;
            std::cout << "SSE2: " << (SimdCpuInfo(SimdCpuInfoSse2) ? "Yes" : "No") << std::endl;
            std::cout << "SSE4.1: " << (SimdCpuInfo(SimdCpuInfoSse41) ? "Yes" : "No") << std::endl;
            std::cout << "AVX: " << (SimdCpuInfo(SimdCpuInfoAvx) ? "Yes" : "No") << std::endl;
            std::cout << "AVX2: " << (SimdCpuInfo(SimdCpuInfoAvx2) ? "Yes" : "No") << std::endl;
            std::cout << "AVX-512F: " << (SimdCpuInfo(SimdCpuInfoAvx512f) ? "Yes" : "No") << std::endl;
            std::cout << "AVX-512BW: " << (SimdCpuInfo(SimdCpuInfoAvx512bw) ? "Yes" : "No") << std::endl;
            std::cout << "AVX-512VNNI: " << (SimdCpuInfo(SimdCpuInfoAvx512vnni) ? "Yes" : "No") << std::endl;
            std::cout << "PowerPC-Altivec: " << (SimdCpuInfo(SimdCpuInfoVmx) ? "Yes" : "No") << std::endl;
            std::cout << "PowerPC-VSX: " << (SimdCpuInfo(SimdCpuInfoVsx) ? "Yes" : "No") << std::endl;
            std::cout << "ARM-NEON: " << (SimdCpuInfo(SimdCpuInfoNeon) ? "Yes" : "No") << std::endl;
            return 0;
        }
        \endverbatim

        \param [in] type - a type of required information.
        \return a value which contains information about CPU and %Simd Library.
    */
    SIMD_API size_t SimdCpuInfo(SimdCpuInfoType type);

    /*! @ingroup info

        \fn const char *SimdPerformanceStatistic();

        \short Gets internal performance statistics of %Simd Library.

        \note %Simd Library have to be build with defined SIMD_PERFORMANCE_STATISTIC macro.

        \return string with internal performance statistics of %Simd Library.
    */
    SIMD_API const char * SimdPerformanceStatistic();

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

        \short Gets alignment required for the most productive work of Simd Library.

        \return a required alignment.
    */
    SIMD_API size_t SimdAlignment();

    /*! @ingroup memory

        \fn void SimdRelease(void * context);

        \short Releases context created with using of Simd Library API.

        \note This function releases a context created by functions ::SimdDetectionLoadA and ::SimdDetectionInit.

        \param [in] context - a context to be released.
    */    
    SIMD_API void SimdRelease(void * context);

    /*! @ingroup thread

        \fn size_t SimdGetThreadNumber();

        \short Gets number of threads used by Simd Library to parallelize some algorithms.

        \return current thread number.
    */
    SIMD_API size_t SimdGetThreadNumber();

    /*! @ingroup thread

        \fn void SimdSetThreadNumber(size_t threadNumber);

        \short Sets number of threads used by Simd Library to parallelize some algorithms.

        \param [in] threadNumber - a number of threads.
    */
    SIMD_API void SimdSetThreadNumber(size_t threadNumber);

    /*! @ingroup cpu_flags

        \fn SimdBool SimdGetFastMode();

        \short Gets current CPU Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) flags. It is used in order to process subnormal numbers.

        \return current 'fast' mode.
    */
    SIMD_API SimdBool SimdGetFastMode();

    /*! @ingroup cpu_flags

        \fn void SimdSetFastMode(SimdBool value);

        \short Sets current CPU Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) flags. It is used in order to process subnormal numbers.

        \param [in] value - a value of 'fast' mode.
    */
    SIMD_API void SimdSetFastMode(SimdBool value);

    /*! @ingroup hash

        \fn uint32_t SimdCrc32(const void * src, size_t size);

        \short Gets 32-bit cyclic redundancy check (CRC32) for current data.

        Calculation is performed for polynomial 0xEDB88320.

        \param [in] src - a pointer to data.
        \param [in] size - a size of the data.
        \return 32-bit cyclic redundancy check (CRC32).
    */
    SIMD_API uint32_t SimdCrc32(const void* src, size_t size);

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

        \fn void SimdAbsDifference(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, uint8_t * c, size_t cStride, size_t width, size_t height);

        \short Gets absolute difference of two gray 8-bit images, pyxel by pixel.

        The three images must have the same width and height.

        \note This function has a C++ wrapper Simd::AbsDifference(const View<A> & a, const View<A> & b, View<A> & c).

        \param [in] a - a pointer to pixels data of first image.
        \param [in] aStride - a row size of first image.
        \param [in] b - a pointer to pixels data of second image.
        \param [in] bStride - a row size of second image.
        \param [out] c - a pointer to pixels data of destination image.
        \param [in] cStride - a row size of destination image.
        \param [in] width - an image width.
        \param [in] height - an image height.
    */
    SIMD_API void SimdAbsDifference(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, uint8_t * c, size_t cStride, size_t width, size_t height);

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
    SIMD_API void SimdAbsDifferenceSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum);

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

        All images must have the same width and height. Source and destination images must have the same format (8 bit per channel, for example GRAY8, UV16, BGR24 or BGRA32). Alpha must be 8-bit gray image.

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

        \fn void SimdAlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t alpha, uint8_t* dst, size_t dstStride);

        \short Performs uniform alpha blending operation.

        All images must have the same width and height. Source and destination images must have the same format (8 bit per channel, for example GRAY8, UV16, BGR24 or BGRA32).

        For every point:
        \verbatim
        dst[x, y, c] = (src[x, y, c]*alpha[x, y] + dst[x, y, c]*(255 - alpha))/255;
        \endverbatim

        This function is used for image drawing.

        \note This function has a C++ wrapper Simd::AlphaBlending(const View<A>& src, uint8_t alpha, View<A>& dst).

        \param [in] src - a pointer to pixels data of foreground image.
        \param [in] srcStride - a row size of the foreground image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channelCount - a channel count for foreground and background images (1 <= channelCount <= 4).
        \param [in] alpha - a pvalue of alpha.
        \param [in, out] dst - a pointer to pixels data of background image.
        \param [in] dstStride - a row size of the background image.
    */
    SIMD_API void SimdAlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t alpha, uint8_t* dst, size_t dstStride);

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

    /*! @ingroup drawing

        \fn void SimdAlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

        \short Performs premultiply operation.

        All images must have the same width, height and format (BGRA32).

        For every point:
        \verbatim
         dst[x, y, 0] = src[x, y, 0] * src[x, y, 3] / 255;
         dst[x, y, 1] = src[x, y, 1] * src[x, y, 3] / 255;
         dst[x, y, 2] = src[x, y, 2] * src[x, y, 3] / 255;
         dst[x, y, 3] = src[x, y, 3];
        \endverbatim

        This function is used for image drawing as a part of alpha blending operation.

        \note This function has a C++ wrapper Simd::AlphaPremultiply(const View<A>& src, View<A>& dst).

        \param [in] src - a pointer to pixels data of input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of output premultiplyed image.
        \param [in] dstStride - a row size of the output premultiplyed image.
    */
    SIMD_API void SimdAlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

    /*! @ingroup drawing

        \fn void SimdAlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

        \short Performs unpremultiply operation.

        All images must have the same width, height and format (BGRA32).

        For every point:
        \verbatim
         dst[x, y, 0] = src[x, y, 0] / src[x, y, 3] * 255;
         dst[x, y, 1] = src[x, y, 1] / src[x, y, 3] * 255;
         dst[x, y, 2] = src[x, y, 2] / src[x, y, 3] * 255;
         dst[x, y, 3] = src[x, y, 3];
        \endverbatim

        This function is used for image drawing as a part of alpha blending operation.

        \note This function has a C++ wrapper Simd::AlphaUnpremultiply(const View<A>& src, View<A>& dst).

        \param [in] src - a pointer to pixels data of input image.
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] dst - a pointer to pixels data of output unpremultiplyed image.
        \param [in] dstStride - a row size of the output unpremultiplyed image.
    */
    SIMD_API void SimdAlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

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

    /*! @ingroup base64

        \fn void SimdBase64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t* dstSize);

        \short Decode string from Base64.

        \note This function has a C++ wrapper std::string Simd::Base64Decode(const std::string & src).

        \param [in] src - a pointer to Base64 encoded input string.
        \param [in] srcSize - a length of input string.
        \param [out] dst - a pointer to the output buffer with decoded string. The size of the buffer is must be at least srcSize / 4 * 3.
        \param [out] dstSize - a pointer to the value with lenght of decoded string. 
    */
    SIMD_API void SimdBase64Decode(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t* dstSize);

    /*! @ingroup base64

        \fn void SimdBase64Encode(const uint8_t* src, size_t size, uint8_t* dst);

        \short Encode string to Base64.

        \note This function has a C++ wrapper std::string Simd::Base64Encode(const std::string & src).

        \param [in] src - a pointer to original string.
        \param [in] size - a length of input string.
        \param [out] dst - a pointer to the output buffer with Base64 encoded string. The size of the buffer is must be at least (size + 2) / 3 * 4.
    */
    SIMD_API void SimdBase64Encode(const uint8_t* src, size_t size, uint8_t* dst);

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

        \short Converts 32-bit BGRA image to 24-bit BGR image. Also it can be used for 32-bit RGBA to 24-bit RGB conversion.

        All images must have the same width and height.

        \note This function has C++ wrappers: Simd::BgraToBgr(const View<A>& bgra, View<A>& bgr)
            and Simd::RgbaToRgb(const View<A>& rgba, View<A>& rgb).

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA (or 32-bit RGBA) image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgraStride - a row size of the bgra image.
        \param [out] bgr - a pointer to pixels data of output 24-bit BGR (or 24-bit RGB) image.
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

        \fn void SimdBgraToRgb(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * rgb, size_t rgbStride);

        \short Converts 32-bit BGRA image to 24-bit RGB image. Also it can be used for 32-bit RGBA to 24-bit BGR conversion.

        All images must have the same width and height.

        \note This function has C++ wrappers: Simd::BgraToRgb(const View<A>& bgra, View<A>& rgb)
            and Simd::RgbaToBgr(const View<A>& rgba, View<A>& bgr).

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA (or 32-bit RGBA) image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgraStride - a row size of the bgra image.
        \param [out] rgb - a pointer to pixels data of output 24-bit RGB (or 24-bit BGR) image.
        \param [in] rgbStride - a row size of the rgb image.
    */
    SIMD_API void SimdBgraToRgb(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgb, size_t rgbStride);

    /*! @ingroup bgra_conversion

        \fn void SimdBgraToRgba(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * rgba, size_t rgbaStride);

        \short Converts 32-bit BGRA image to 32-bit RGBA image. Also it can be used for 32-bit RGBA to 32-bit BGRA conversion.

        All images must have the same width and height.

        \note This function has C++ wrappers: Simd::BgraToRgba(const View<A>& bgra, View<A>& rgba)
            and Simd::RgbaToBgra(const View<A>& rgba, View<A>& bgra).

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA (or 32-bit RGBA) image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgraStride - a row size of the bgra image.
        \param [out] rgba - a pointer to pixels data of output 32-bit RGBA (or 32-bit BGRA) image.
        \param [in] rgbaStride - a row size of the rgb image.
    */
    SIMD_API void SimdBgraToRgba(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* rgba, size_t rgbaStride);

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

        \fn void SimdBgraToYuv420pV2(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride, SimdYuvType yuvType);

        \short Converts 32-bit BGRA image to YUV420P.

        The input BGRA and output Y images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
        \param [in] bgraStride - a row size of the BGRA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] yuvType - a type of input YUV image (see descriprion of ::SimdYuvType).
    */
    SIMD_API void SimdBgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType);

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

    /*! @ingroup bgra_conversion

        \fn void SimdBgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType);

        \short Converts 32-bit BGRA image to YUV444P.

        The input BGRA and output Y, U and V images must have the same width and height.

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
        \param [in] bgraStride - a row size of the BGRA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] yuvType - a type of input YUV image (see descriprion of ::SimdYuvType).
    */
    SIMD_API void SimdBgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, 
        uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType);

    /*! @ingroup bgra_conversion

        \fn void SimdBgraToYuva420p(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride, uint8_t * a, size_t aStride);

        \short Converts 32-bit BGRA image to YUVA420P.

        The input BGRA and output Y and A images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function has a C++ wrapper Simd::BgraToYuva420p(const View<A> & bgra, View<A> & y, View<A> & u, View<A> & v, View<A> & a).

        \param [in] bgra - a pointer to pixels data of input 32-bit BGRA image.
        \param [in] bgraStride - a row size of the BGRA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] y - a pointer to pixels data of output 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [out] u - a pointer to pixels data of output 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [out] v - a pointer to pixels data of output 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [out] a - a pointer to pixels data of output 8-bit image with alpha plane.
        \param [in] aStride - a row size of the a image.
    */
    SIMD_API void SimdBgraToYuva420p(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, 
        uint8_t * y, size_t yStride, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride, uint8_t * a, size_t aStride);

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

        \fn void SimdBgrToRgb(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * rgb, size_t rgbStride);

        \short Converts 24-bit BGR image to 24-bit RGB image. Also it can be used for 24-bit RGB to 24-bit BGR conversion.

        All images must have the same width and height.

        \note This function has C++ wrappers: Simd::BgrToRgb(const View<A> & bgr, View<A> & rgb) 
            and Simd::RgbToBgr(const View<A>& rgb, View<A>& bgr).

        \param [in] bgr - a pointer to pixels data of input 24-bit BGR image (or 24-bit RGB image).
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] bgrStride - a row size of the bgr image.
        \param [out] rgb - a pointer to pixels data of output 24-bit RGB image (or 24-bit BGR image).
        \param [in] rgbStride - a row size of the rgb image.
    */
    SIMD_API void SimdBgrToRgb(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * rgb, size_t rgbStride);

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

    /*! @ingroup binarization

        \fn void SimdAveragingBinarizationV2(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride);

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
                    sum += src[x + dx, x + dy];
                }
            }
        }
        dst[x, y] = (src[x, y] + shift)*area > sum ? positive : negative;
        \endverbatim

        \note This function has a C++ wrapper Simd::AveragingBinarizationV2(const View<A>& src, size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, View<A>& dst).

        \param [in] src - a pointer to pixels data of input 8-bit gray image.
        \param [in] srcStride - a row size of the src image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] neighborhood - an averaging neighborhood.
        \param [in] shift - a shift value for binarization. It can range from -255 to 255.
        \param [in] positive - a destination value for positive value of condition (seen before).
        \param [in] negative - a destination value for negative value of condition (seen before).
        \param [out] dst - a pointer to pixels data of output 8-bit gray binarized image.
        \param [in] dstStride - a row size of the dst image.
    */
    SIMD_API void SimdAveragingBinarizationV2(const uint8_t* src, size_t srcStride, size_t width, size_t height,
        size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride);

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

    /*! @ingroup deinterleave_conversion

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

    /*! @ingroup deinterleave_conversion

        \fn void SimdDeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height, uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride);

        \short Deinterleaves 24-bit BGR interleaved image into separated 8-bit Blue, Green and Red planar images.

        All images must have the same width and height.

        \note This function has C++ wrappers:
            Simd::DeinterleaveBgr(const View<A>& bgr, View<A>& b, View<A>& g, View<A>& r),
            Simd::DeinterleaveRgb(const View<A>& rgb, View<A>& r, View<A>& g, View<A>& b).

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

    /*! @ingroup deinterleave_conversion

        \fn void SimdDeinterleaveBgra(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride, uint8_t * a, size_t aStride);

        \short Deinterleaves 32-bit BGRA interleaved image into separated 8-bit Blue, Green, Red and Alpha planar images.

        All images must have the same width and height.

        \note This function has C++ wrappers:
            Simd::DeinterleaveBgra(const View<A>& bgra, View<A>& b, View<A>& g, View<A>& r, View<A>& a),
            Simd::DeinterleaveBgra(const View<A>& bgra, View<A>& b, View<A>& g, View<A>& r),
            Simd::DeinterleaveRgba(const View<A>& rgba, View<A>& r, View<A>& g, View<A>& b, View<A>& a),
            Simd::DeinterleaveRgba(const View<A>& rgba, View<A>& r, View<A>& g, View<A>& b).

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
        \param [out] a - a pointer to pixels data of 8-bit Alpha planar image. It can be NULL.
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
                This pointer is used in functions ::SimdDetectionInfo and ::SimdDetectionInit, and must be released with using of function ::SimdRelease.
    */
    SIMD_API void * SimdDetectionLoadA(const char * path);

    /*! @ingroup object_detection

        \fn void * SimdDetectionLoadStringXml(char * xml);

        \short Loads a classifier cascade from a string.

        This function supports OpenCV HAAR and LBP cascades type.
        Tree based cascades and old cascade formats are not supported.

        \note This function is used for implementation of Simd::Detection.

        \param [in,out] xml - A string with the xml of a classifier cascade.
        \return a pointer to loaded cascade. On error it returns NULL.
                This pointer is used in functions ::SimdDetectionInfo and ::SimdDetectionInit, and must be released with using of function ::SimdRelease.
    */
    SIMD_API void * SimdDetectionLoadStringXml(char * xml);

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
                It must be released with using of function ::SimdRelease.
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

    /*! @ingroup filling

        \fn void SimdFill32f(float * dst, size_t size, const float * value);

        \short Fills 32-bit float array by given value.

        \param [out] dst - a pointer to 32-bit float array.
        \param [in] size - a size of the array.
        \param [in] value - a pointer to value to fill. Can be NULL (filling value is assumed to be equal to zero).
    */
    SIMD_API void SimdFill32f(float * dst, size_t size, const float * value);

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

    /*! @ingroup float16

        \fn void SimdCosineDistance16f(const uint16_t * a, const uint16_t * b, size_t size, float * distance);

        \short Calculates cosine distance of two 16-bit float arrays.

        All arrays must have the same size.

        Algorithm description:
        \verbatim
        distance = 1 - Sum(a[i]*b[i])/Sqrt(Sum(a[i]*a[i])*Sum(b[i]*b[i]));
        \endverbatim

        \param [in] a - a pointer to the first 16-bit float array.
        \param [in] b - a pointer to the second 16-bit float array.
        \param [in] size - a size of arrays.
        \param [out] distance - a pointer to 32-bit float with cosine distance.
    */
    SIMD_API void SimdCosineDistance16f(const uint16_t * a, const uint16_t * b, size_t size, float * distance);

    /*! @ingroup float16

        \fn void SimdCosineDistancesMxNa16f(size_t M, size_t N, size_t K, const uint16_t * const * A, const uint16_t * const * B, float * distances);

        \short Calculates mutual cosine distance of two arrays of 16-bit float arrays.

        Algorithm description:
        \verbatim
        distances[i, j] = 1 - Sum(A[i][k]*B[j][k])/Sqrt(Sum(A[i][k]*A[i][k])*Sum(B[j][k]*B[j][k]));
        \endverbatim

        \param [in] M - a number of A arrays.
        \param [in] N - a number of B arrays.
        \param [in] K - a size of A and B arrays.
        \param [in] A - a pointer to the first array with pointers to 16-bit float arrays.
        \param [in] B - a pointer to the second array with pointers to 16-bit float arrays.
        \param [out] distances - a pointer to result 32-bit float array with cosine distances. It size must be M*N.
    */
    SIMD_API void SimdCosineDistancesMxNa16f(size_t M, size_t N, size_t K, const uint16_t * const * A, const uint16_t * const * B, float * distances);

    /*! @ingroup float16

        \fn void SimdCosineDistancesMxNp16f(size_t M, size_t N, size_t K, const uint16_t* A, const uint16_t* B, float* distances);

        \short Calculates mutual cosine distance of two arrays of 16-bit float arrays.

        Algorithm description:
        \verbatim
        distances[i, j] = 1 - Sum(A[i*K + k]*B[j*K + k])/Sqrt(Sum(A[i*K + k]*A[i*K + k])*Sum(B[j*K + k]*B[j*K + k]));
        \endverbatim

        \param [in] M - a number of A arrays.
        \param [in] N - a number of B arrays.
        \param [in] K - a size of A and B arrays.
        \param [in] A - a pointer to 16-bit float arrays.
        \param [in] B - a pointer to 16-bit float arrays.
        \param [out] distances - a pointer to result 32-bit float array with cosine distances. It size must be M*N.
    */
    SIMD_API void SimdCosineDistancesMxNp16f(size_t M, size_t N, size_t K, const uint16_t* A, const uint16_t* B, float* distances);

    /*! @ingroup float16

        \fn void SimdVectorNormNa16f(size_t N, size_t K, const uint16_t* const* A, float* norms);

        \short Calculates vector norms for array of 16-bit float arrays.

        Algorithm description:
        \verbatim
        norms[j] = Sqrt(Sum(A[j][k]*A[j][k]));
        \endverbatim

        \param [in] N - a number of A arrays.
        \param [in] K - a size of A arrays.
        \param [in] A - a pointer to the array with pointers to 16-bit float arrays.
        \param [out] norms - a pointer to result 32-bit float array with vector norms. It size must be N.
    */
    SIMD_API void SimdVectorNormNa16f(size_t N, size_t K, const uint16_t* const* A, float* norms);

    /*! @ingroup float16

        \fn void SimdVectorNormNp16f(size_t N, size_t K, const uint16_t* A, float* norms);

        \short Calculates vector norms for array of 16-bit float arrays.

        Algorithm description:
        \verbatim
        norms[j] = Sqrt(Sum(A[j*K + k]*A[j*K + k]));
        \endverbatim

        \param [in] N - a number of A arrays.
        \param [in] K - a size of A arrays.
        \param [in] A - a pointer to 16-bit float arrays.
        \param [out] norms - a pointer to result 32-bit float array with vector norms. It size must be N.
    */
    SIMD_API void SimdVectorNormNp16f(size_t N, size_t K, const uint16_t* A, float* norms);

    /*! @ingroup float16

        \fn void SimdCosineDistancesMxNp16f(size_t M, size_t N, size_t K, const uint16_t* A, const uint16_t* B, float* distances);

        \short Calculates mutual cosine distance of two arrays of 16-bit float arrays.

        Algorithm description:
        \verbatim
        distances[i, j] = 1 - Sum(A[i*K + k]*B[j*K + k])/Sqrt(Sum(A[i*K + k]*A[i*K + k])*Sum(B[j*K + k]*B[j*K + k]));
        \endverbatim

        \param [in] M - a number of A arrays.
        \param [in] N - a number of B arrays.
        \param [in] K - a size of A and B arrays.
        \param [in] A - a pointer to 16-bit float arrays.
        \param [in] B - a pointer to 16-bit float arrays.
        \param [out] distances - a pointer to result 32-bit float array with cosine distances. It size must be M*N.
    */
    SIMD_API void SimdCosineDistancesMxNp16f(size_t M, size_t N, size_t K, const uint16_t* A, const uint16_t* B, float* distances);

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

    /*! @ingroup correlation

        \fn void SimdCosineDistance32f(const float * a, const float * b, size_t size, float * distance);

        \short Calculates cosine distance of two 32-bit float arrays.

        All arrays must have the same size.

        Algorithm description:
        \verbatim
        distance = 1 - Sum(a[i]*b[i])/Sqrt(Sum(a[i]*a[i])*Sum(b[i]*b[i]));
        \endverbatim

        \param [in] a - a pointer to the first 32-bit float array.
        \param [in] b - a pointer to the second 32-bit float array.
        \param [in] size - a size of arrays.
        \param [out] distance - a pointer to 32-bit float with cosine distance.
    */
    SIMD_API void SimdCosineDistance32f(const float * a, const float * b, size_t size, float * distance);

    /*! @ingroup gaussian_filter

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

    /*! @ingroup gaussian_filter

        \fn void * SimdGaussianBlurInit(size_t width, size_t height, size_t channels, const float * sigma, const float* epsilon);

        \short Creates Gaussian blur filter context.

        In particular calculates Gaussian blur coefficients:
        \verbatim
        half = floor(sqrt(log(1/epsilon)) * sigma);
        weight[2*half + 1];

        for(x = -half; x <= half; ++x)
            weight[x + half] = exp(-sqr(x / sigma) / 2);

        sum = 0;
        for (x = -half; x <= half; ++x)
            sum += weight[x + half];

        for (x = -half; x <= half; ++x)
            weight[x + half] /= sum;
        \endverbatim

        \param [in] width - a width of input and output image.
        \param [in] height - a height of input and output image.    
        \param [in] channels - a channel number of input and output image. Its value must be in range [1..4].
        \param [in] sigma - a pointer to sigma parameter (blur radius). MIts value must be greater than 0.000001.
        \param [in] epsilon - a pointer to epsilon parameter (permissible relative error). 
                              Its value must be greater than 0.000001. Pointer can be NULL and by default value 0.001 is used.
        \return a pointer to filter context. On error it returns NULL.
                This pointer is used in functions ::SimdGaussianBlurRun.
                It must be released with using of function ::SimdRelease.
    */
    SIMD_API void* SimdGaussianBlurInit(size_t width, size_t height, size_t channels, const float * sigma, const float* epsilon);

    /*! @ingroup gaussian_filter

        \fn void SimdGaussianBlurRun(const void* filter, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);

        \short Performs image Gaussian bluring.

        Bluring algorithm for every point:
        \verbatim
        sum = 0;
        for(x = -half; x <= half; ++x)
        {
            sx = min(max(0, dx + x), width - 1);
            for(y = -half; y <= half; ++y)
            {
                sy = min(max(0, dy + y), height - 1);
                sum += src[sx, sy]*weight[x + half]*weight[y + half];
            }
        }
        dst[dx, dy] = sum;
        \endverbatim

        \param [in] filter - a filter context. It must be created by function ::SimdGaussianBlurInit and released by function ::SimdRelease.
        \param [in] src - a pointer to pixels data of the original input image.
        \param [in] srcStride - a row size (in bytes) of the input image.
        \param [out] dst - a pointer to pixels data of the filtered output image.
        \param [in] dstStride - a row size (in bytes) of the output image.
    */
    SIMD_API void SimdGaussianBlurRun(const void* filter, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);

    /*! @ingroup matrix

        \fn void SimdGemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc);

        \short Performs general matrix multiplication (for 32-bit float numbers).

        \verbatim
        C(M, N) = alpha*A(M, K)*B(K, N) + beta*C(M, N);
        \endverbatim

        \note This function supports multithreading (See functions ::SimdGetThreadNumber and ::SimdSetThreadNumber).

        \param [in] M - a height of A and height of C matrices.
        \param [in] N - a width of B and width of C matrices.
        \param [in] K - a width of A and height of B matrices.
        \param [in] alpha - a pointer to multiplier of the first term.
        \param [in] A - a pointer to input A matrix.
        \param [in] lda - a leading dimension of A matrix.
        \param [in] B - a pointer to input B matrix.
        \param [in] ldb - a leading dimension of B matrix.
        \param [in] beta - a pointer to multiplier of the second term.
        \param [out] C - a pointer to output C matrix.
        \param [in] ldc - a leading dimension of C matrix.
    */
    SIMD_API void SimdGemm32fNN(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc);

    /*! @ingroup matrix

        \fn void SimdGemm32fNT(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc);

        \short Performs general matrix multiplication (for 32-bit float numbers).

        \verbatim
        C(M, N) = alpha*A(M, K)*Trans(B(N, K)) + beta*C(M, N);
        \endverbatim

        \note This function supports multithreading (See functions ::SimdGetThreadNumber and ::SimdSetThreadNumber).

        \param [in] M - a height of A and height of C matrices.
        \param [in] N - a height of B and width of C matrices.
        \param [in] K - a width of A and width of B matrices.
        \param [in] alpha - a pointer to multiplier of the first term.
        \param [in] A - a pointer to input A matrix.
        \param [in] lda - a leading dimension of A matrix.
        \param [in] B - a pointer to input B matrix.
        \param [in] ldb - a leading dimension of B matrix.
        \param [in] beta - a pointer to multiplier of the second term.
        \param [out] C - a pointer to output C matrix.
        \param [in] ldc - a leading dimension of C matrix.
    */
    SIMD_API void SimdGemm32fNT(size_t M, size_t N, size_t K, const float * alpha, const float * A, size_t lda, const float * B, size_t ldb, const float * beta, float * C, size_t ldc);

    /*! @ingroup gray_conversion

        \fn void SimdGrayToBgr(const uint8_t * gray, size_t width, size_t height, size_t grayStride, uint8_t * bgr, size_t bgrStride);

        \short Converts 8-bit gray image to 24-bit BGR image. Also it can be used for 8-bit gray to 24-bit RGB conversion.

        All images must have the same width and height.

        \note This function has C++ wrappers: Simd::GrayToBgr(const View<A>& gray, View<A>& bgr) 
            and Simd::GrayToRgb(const View<A>& gray, View<A>& rgb).

        \param [in] gray - a pointer to pixels data of input 8-bit gray image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] grayStride - a row size of the gray image.
        \param [out] bgr - a pointer to pixels data of output 24-bit BGR (or 24-bit RGB) image.
        \param [in] bgrStride - a row size of the bgr image.
    */
    SIMD_API void SimdGrayToBgr(const uint8_t *gray, size_t width, size_t height, size_t grayStride, uint8_t *bgr, size_t bgrStride);

    /*! @ingroup gray_conversion

        \fn void SimdGrayToBgra(const uint8_t * gray, size_t width, size_t height, size_t grayStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

        \short Converts 8-bit gray image to 32-bit BGRA image. Also it can be used for 8-bit gray to 32-bit RGBA conversion.

        All images must have the same width and height.

        \note This function has C++ wrappers: Simd::GrayToBgra(const View<A>& gray, View<A>& bgra, uint8_t alpha) 
            and Simd::GrayToRgba(const View<A>& gray, View<A>& rgba, uint8_t alpha).

        \param [in] gray - a pointer to pixels data of input 8-bit gray image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] grayStride - a row size of the gray image.
        \param [out] bgra - a pointer to pixels data of output 32-bit BGRA (or 32-bit RGBA) image.
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
        Algorithm description:
        \verbatim
        for(y = 0; y < height; ++y)
            for(x = 0; x < width; ++x)
                dst[x, y] = colors[src[x, y]];
        \endverbatim

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

        \fn void SimdHogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride);

        \short Applies filter to lite HOG features.

        Applies filter of square shape to lite HOG features. 

        For every point of output image:
        \verbatim
        if(mask[x, y])
            sum = 0;
            for(dy = 0; dy < filterHeight; dy++)
                for(dx = 0; dx < filterWidth*featureSize; dx++)
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
        \param [in] filterWidth - a width of used filter. 
        \param [in] filterHeight - a height of used filter.
        \param [in] mask - a pointer to the 32-bit integer array with mask (0 or -1).
                    Pointer can be null otherwise the array must have size greater then (srcHeight - filterSize)*(srcWidth - filterSize).
                    A function ::SimdHogLiteCreateMask is usefull in order to create this mask.
        \param [in] maskStride - a row size of mask array. 
        \param [out] dst - a pointer to output buffer with result of filtration. Array must have size greater then (srcHeight - filterSize)*(srcWidth - filterSize).
        \param [in] dstStride - a row size of the output buffer with result of filtration.
    */
    SIMD_API void SimdHogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride);

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

    /*! @ingroup image_io

        \fn uint8_t* SimdImageSaveToMemory(const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, size_t * size);

        \short Saves an image to memory in given image file format.

        \param [in] src - a pointer to pixels data of input image. 
        \param [in] stride - a row size of input image in bytes.
        \param [in] width - a width of input image.
        \param [in] height - a height of input image.
        \param [in] format - a pixel format of input image. 
            Supported pixel formats: ::SimdPixelFormatGray8, ::SimdPixelFormatBgr24, ::SimdPixelFormatBgra32, ::SimdPixelFormatRgb24, ::SimdPixelFormatRgba32.
        \param [in] file - a format of output image file. To auto choise format of output file set this parameter to ::SimdImageFileUndefined.
        \param [in] quality - a parameter of compression quality (if file format supports it).
        \param [out] size - a pointer to the size of output image file in bytes.
        \return a pointer to memory buffer with output image file. 
            It has to be deleted after use by function ::SimdFree. On error it returns NULL.
    */
    SIMD_API uint8_t* SimdImageSaveToMemory(const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, size_t * size);

    /*! @ingroup image_io

        \fn SimdBool SimdImageSaveToFile(const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, const char * path);

        \short Saves an image to memory in given image file format.

        \param [in] src - a pointer to pixels data of input image.
        \param [in] stride - a row size of input image in bytes.
        \param [in] width - a width of input image.
        \param [in] height - a height of input image.
        \param [in] format - a pixel format of input image. 
            Supported pixel formats: ::SimdPixelFormatGray8, ::SimdPixelFormatBgr24, ::SimdPixelFormatBgra32, ::SimdPixelFormatRgb24, ::SimdPixelFormatRgba32.
        \param [in] file - a format of output image file. To auto choise format of output file set this parameter to ::SimdImageFileUndefined.
        \param [in] quality - a parameter of compression quality (if file format supports it).
        \param [in] path - a path to output image file.
        \return result of the operation.
    */
    SIMD_API SimdBool SimdImageSaveToFile(const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, const char * path);

    /*! @ingroup image_io

        \fn uint8_t* SimdNv12SaveAsJpegToMemory(const uint8_t* y, size_t yStride, const uint8_t* uv, size_t uvStride, size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size);

        \short Saves image in NV12 format to memory as JPEG.

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] uv - a pointer to pixels data of input 8-bit image with UV color plane.
        \param [in] uvStride - a row size of the uv image.
        \param [in] width - a width of input image. It must be even number.
        \param [in] height - a height of input image. It must be even number.
        \param [in] yuvType - a type of input YUV image(see descriprion of::SimdYuvType). Now only ::SimdYuvTrect871 (T-REC-T.871 format) is supported.
        \param [in] quality - a parameter of compression quality.
        \param [out] size - a pointer to the size of output image file in bytes.
        \return a pointer to memory buffer with output image file.
            It has to be deleted after use by function ::SimdFree. On error it returns NULL.
    */
    SIMD_API uint8_t* SimdNv12SaveAsJpegToMemory(const uint8_t* y, size_t yStride, const uint8_t* uv, size_t uvStride, size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size);

    /*! @ingroup image_io

        \fn uint8_t* SimdYuv420pSaveAsJpegToMemory(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size);

        \short Saves image in YUV420P format to memory as JPEG.

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - a width of input image. It must be even number.
        \param [in] height - a height of input image. It must be even number.
        \param [in] yuvType - a type of input YUV image(see descriprion of::SimdYuvType). Now only ::SimdYuvTrect871 (T-REC-T.871 format) is supported.
        \param [in] quality - a parameter of compression quality.
        \param [out] size - a pointer to the size of output image file in bytes.
        \return a pointer to memory buffer with output image file.
            It has to be deleted after use by function ::SimdFree. On error it returns NULL.
    */
    SIMD_API uint8_t* SimdYuv420pSaveAsJpegToMemory(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, 
        size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size);

    /*! @ingroup image_io

        \fn uint8_t* SimdImageLoadFromMemory(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType * format);

        \short Loads an image from memory buffer.

        \param [in] data - a pointer to memory buffer with input image file.
        \param [in] size - a size of input image file in bytes.
        \param [out] stride - a pointer to row size of output image in bytes.
        \param [out] width - a pointer to width of output image.
        \param [out] height - a pointer to height of output image.
        \param [in, out] format - a pointer to pixel format of output image. 
            Here you can set desired pixel format (it can be ::SimdPixelFormatGray8, ::SimdPixelFormatBgr24, ::SimdPixelFormatBgra32, ::SimdPixelFormatRgb24, ::SimdPixelFormatRgba32).
            Or set ::SimdPixelFormatNone and use pixel format of input image file.
        \return a pointer to pixels data of output image. 
            It has to be deleted after use by function ::SimdFree. On error it returns NULL.
    */
    SIMD_API uint8_t* SimdImageLoadFromMemory(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType * format);

    /*! @ingroup image_io

        \fn uint8_t* SimdImageLoadFromFile(const char* path, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType * format);

        \short Loads an image from file.

        \param [in] path - a path to input image file.
        \param [out] stride - a pointer to row size of output image in bytes.
        \param [out] width - a pointer to width of output image.
        \param [out] height - a pointer to height of output image.
        \param [in, out] format - a pointer to pixel format of output image.
            Here you can set desired pixel format (it can be ::SimdPixelFormatGray8, ::SimdPixelFormatBgr24, ::SimdPixelFormatBgra32, ::SimdPixelFormatRgb24, ::SimdPixelFormatRgba32).
            Or set ::SimdPixelFormatNone and use pixel format of input image file.
        \return a pointer to pixels data of output image.
            It has to be deleted after use by function ::SimdFree. On error it returns NULL.
    */
    SIMD_API uint8_t* SimdImageLoadFromFile(const char* path, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType * format);

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

    /*! @ingroup interleave_conversion

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

    /*! @ingroup interleave_conversion

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

    /*! @ingroup interleave_conversion

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

        \fn void SimdNeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst);

        \short Calculates rough sigmoid for 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        x = abs(src[i]*slope);
        e = 1 + x + x*x*0.5417 + x*x*x*x*0.1460;
        dst[i] = 1 / (1 + (src[i] > 0 ? 1 / e : e));
        \endverbatim
        It is approximate way (maximal absolute error is 0.002294 (~0.23%) ) of sigmoid function (::SimdSynetSigmoid32f) calculation:
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
        It is approximate way (maximal absolute error is 0.001721 (~0.17%) ) of sigmoid function (::SimdSynetSigmoid32f) calculation:
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

        \fn void SimdNeuralRoughTanh(const float * src, size_t size, const float * slope, float * dst);

        \short Calculates rough hyperbolic tangent for 32-bit float array.

        All arrays must have the same size.

        For every element:
        \verbatim
        x = ::abs(src[i]*slope);
        e = 1 + x + x*x*0.5658 + x*x*x*x*0.1430;
        dst[i] = (src[i] > 0 ? 1 : -1)*(e - 1/e)/(e + 1/e);
        \endverbatim
        It is approximate way (maximal absolute error is 0.001514 (~0.15%) ) of hyperbolic tangent (::SimdSynetTanh32f)  function calculation:
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

        \fn void SimdReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount);

        \short Performs reducing and Gaussian blurring (in two time) a 8-bit channel color image with using window 2x2.

        For input and output image must be performed: dstWidth = (srcWidth + 1)/2,  dstHeight = (srcHeight + 1)/2.

        For all points:
        \verbatim
        dst[x, y, c] = (src[2*x, 2*y, c] + src[2*x, 2*y + 1, c] + src[2*x + 1, 2*y, c] + src[2*x + 1, 2*y + 1, c] + 2)/4;
        \endverbatim

        \note This function has a C++ wrappers: Simd::Reduce2x2(const View<A> & src, View<A> & dst).

        \param [in] src - a pointer to pixels data of the original input image.
        \param [in] srcWidth - a width of the input image.
        \param [in] srcHeight - a height of the input image.
        \param [in] srcStride - a row size of the input image.
        \param [out] dst - a pointer to pixels data of the reduced output image.
        \param [in] dstWidth - a width of the output image.
        \param [in] dstHeight - a height of the output image.
        \param [in] dstStride - a row size of the output image.
        \param [in] channelCount - a nmber of channels for input and output images.
    */
    SIMD_API void SimdReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
        uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount);

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

    /*! @ingroup resizing

        \fn void * SimdResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);

        \short Creates resize context.

        An using example (resize of RGBA64 image):
        \verbatim
        void * resizer = SimdResizerInit(srcX, srcY, dstX, dstY, 4, SimdResizeChannelShort, SimdResizeMethodBilinear);
        if (resizer)
        {
             SimdResizerRun(resizer, (uint8_t*)src, srcStride, (uint8_t*)dst, dstStride);
             SimdRelease(resizer);
        }
        \endverbatim

        \param [in] srcX - a width of the input image.
        \param [in] srcY - a height of the input image.
        \param [in] dstX - a width of the output image.
        \param [in] dstY - a height of the output image.
        \param [in] channels - a channel number of input and output image.
        \param [in] type - a type of input and output image channel.
        \param [in] method - a method used in order to resize image.
        \return a pointer to resize context. On error it returns NULL. 
                This pointer is used in functions ::SimdResizerRun. 
                It must be released with using of function ::SimdRelease.
    */
    SIMD_API void * SimdResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);

    /*! @ingroup resizing

        \fn void SimdResizerRun(const void * resizer, const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);

        \short Performs image resizing.

        \param [in] resizer - a resize context. It must be created by function ::SimdResizerInit and released by function ::SimdRelease.
        \param [in] src - a pointer to pixels data of the original input image.
        \param [in] srcStride - a row size (in bytes) of the input image.
        \param [out] dst - a pointer to pixels data of the resized output image.
        \param [in] dstStride - a row size (in bytes) of the output image.
    */
    SIMD_API void SimdResizerRun(const void * resizer, const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);

    /*! @ingroup rgb_conversion

        \fn void SimdRgbToBgra(const uint8_t * rgb, size_t width, size_t height, size_t rgbStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha);

        \short Converts 24-bit RGB image to 32-bit BGRA image. Also it can be used for 24-bit BGR to 32-bit RGBA conversion.

        All images must have the same width and height.

        \note This function has C++ wrappers: Simd::RgbToBgra(const View<A>& rgb, View<A>& bgra, uint8_t alpha)
            and Simd::BgrToRgba(const View<A>& bgr, View<A>& rgba, uint8_t alpha).

        \param [in] rgb - a pointer to pixels data of input 24-bit RGB (or 24-bit BGR) image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] rgbStride - a row size of the rgb image.
        \param [out] bgra - a pointer to pixels data of output 32-bit BGRA (or 32-bit RGBA) image.
        \param [in] bgraStride - a row size of the bgra image.
        \param [in] alpha - a value of alpha channel.
    */
    SIMD_API void SimdRgbToBgra(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* bgra, size_t bgraStride, uint8_t alpha);

    /*! @ingroup rgb_conversion

        \fn void SimdRgbToGray(const uint8_t * rgb, size_t width, size_t height, size_t rgbStride, uint8_t * gray, size_t grayStride);

        \short Converts 24-bit RGB image to 8-bit gray image.

        All images must have the same width and height.

        \note This function has a C++ wrapper Simd::RgbToGray(const View<A>& rgb, View<A>& gray).

        \param [in] rgb - a pointer to pixels data of input 24-bit RGB image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] rgbStride - a row size of the rgb image.
        \param [out] gray - a pointer to pixels data of output 8-bit gray image.
        \param [in] grayStride - a row size of the gray image.
    */
    SIMD_API void SimdRgbToGray(const uint8_t* rgb, size_t width, size_t height, size_t rgbStride, uint8_t* gray, size_t grayStride);

    /*! @ingroup rgba_conversion

        \fn void SimdRgbaToGray(const uint8_t * rgba, size_t width, size_t height, size_t rgbaStride, uint8_t * gray, size_t grayStride);

        \short Converts 32-bit RGBA image to 8-bit gray image.

        All images must have the same width and height.

        \note This function has a C++ wrapper Simd::RgbaToGray(const View<A>& rgba, View<A>& gray).

        \param [in] rgba - a pointer to pixels data of input 32-bit RGBA image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] rgbaStride - a row size of the rgba image.
        \param [out] gray - a pointer to pixels data of output 8-bit gray image.
        \param [in] grayStride - a row size of the gray image.
    */
    SIMD_API void SimdRgbaToGray(const uint8_t* rgba, size_t width, size_t height, size_t rgbaStride, uint8_t* gray, size_t grayStride);

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

    /*! @ingroup other_statistic

        \fn void SimdGetObjectMoments(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t index, uint64_t * n, uint64_t * s,  uint64_t * sx, uint64_t * sy, uint64_t * sxx, uint64_t * sxy, uint64_t * syy);

        \short Calculate statistical characteristics (moments) of given object.

        The images must has 8-bit gray format and equal size. One of them can be empty.

        For every point:
        \verbatim
        if(mask[X, Y] == index || mask == 0)
        {
            S = src ? src[X, Y] : 1;
            n += 1.
            s += S;
            sx += S*X.
            sy += S*Y.
            sxx += S*X*X.
            sxy += S*X*Y.
            syy += S*Y*Y.
        }
        \endverbatim

        \note This function has a C++ wrappers: Simd::GetObjectMoments(const View<A> & src, const View<A> & mask, uint8_t index, uint64_t & n, uint64_t & s,  uint64_t & sx, uint64_t & sy, uint64_t & sxx, uint64_t & sxy, uint64_t & syy).

        \param [in] src - a pointer to pixels data of the input image. Can be NULL (its behaviour is equal to function SimdGetMoments).
        \param [in] srcStride - a row size of the input image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] mask - a pointer to pixels data of the mask image. Can be NULL (the moments will be collected over whole image).
        \param [in] maskStride - a row size of the mask image.
        \param [in] index - a mask index.
        \param [out] n - a pointer to unsigned 64-bit integer value with found area of given object.
        \param [out] s - a pointer to unsigned 64-bit integer value with sum of image values of given object.
        \param [out] sx - a pointer to unsigned 64-bit integer value with found first-order moment x of given object.
        \param [out] sy - a pointer to unsigned 64-bit integer value with found first-order moment y of given object.
        \param [out] sxx - a pointer to unsigned 64-bit integer value with found second-order moment xx of given object.
        \param [out] sxy - a pointer to unsigned 64-bit integer value with found second-order moment xy of given object.
        \param [out] syy - a pointer to unsigned 64-bit integer value with found second-order moment yy of given object.
    */
    SIMD_API void SimdGetObjectMoments(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t index,
        uint64_t * n, uint64_t * s,  uint64_t * sx, uint64_t * sy, uint64_t * sxx, uint64_t * sxy, uint64_t * syy);

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

        \fn void SimdValueSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * valueSum, uint64_t * squareSum);

        \short Gets sum and squared sum of value of pixels for gray 8-bit image.

        \note This function has a C++ wrappers: Simd::ValueSquareSum(const View<A>& src, uint64_t & valueSum, uint64_t & squareSum).

        \param [in] src - a pointer to pixels data of the image.
        \param [in] stride - a row size of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] valueSum - the result value sum.
        \param [out] squareSum - the result square sum.
    */
    SIMD_API void SimdValueSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * valueSum, uint64_t * squareSum);

    /*! @ingroup other_statistic

        \fn void SimdValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums);

        \short Gets image channels value sums and squared value sums for image. The image must have 8-bit depth per channel.

        \verbatim
        for(c = 0; c < channels; c++)
        {
            valueSums[c] = 0;
            squareSums[c] = 0;
        }
        for(y = 0; y < height; y++) 
            for(x = 0; x < width; x++)
                for(c = 0; c < channels; c++)
                {
                    value = src[y * stride + x * channels + c];
                    valueSums[c] += value;
                    squareSums[c] += value * value;
                }
        \endverbatim

        \note This function has a C++ wrappers: Simd::ValueSquareSums(const View<A>& src, uint64_t * valueSums, uint64_t * squareSums).

        \param [in] src - a pointer to pixels data of the image.
        \param [in] stride - a row size of the image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [in] channels - an image channels count. It my be equal to 1, 2, 3 or 4.
        \param [out] valueSums - the pointer to output buffer with value sums. Size of the buffer must be at least channels count.
        \param [out] squareSums - the pointer to output buffer with square sums. Size of the buffer must be at least channels count.
    */
    SIMD_API void SimdValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums);
    
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

    /*! @ingroup synet

        \fn void SimdSynetAddBias(const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

        \short Adds a bias to given vector.

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(j = 0; j < spatial; ++j)
                 dst[c*spatial + s] += bias[c];
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] bias - a pointer to the 32-bit float array with bias coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] channels - a number of channels in the image tensor.
        \param [in] spatial - a spatial size of image tensor.
        \param [in, out] dst - a pointer to cumulative 32-bit image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] format - a format of image tensor.
    */
    SIMD_API void SimdSynetAddBias(const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

    /*! @ingroup synet

        \fn void SimdSynetAdd8i(const uint8_t * aData, const float * aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift, uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);

        \short Adds two INT8 tensors.

         Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(b = 0; b < batch; ++b)
            for(c = 0; c < channels; ++c)
                for(s = 0; s < spatial; ++s)
                {
                     offs = (b*channels + c)*spatial + s;
                     A = aData[offs]*aScale[c] + aShift[c]; 
                     B = bData[offs]*bScale[c] + bShift[c];
                     cData[offs] = round((A + B)*cScale[c] + cShift[c]);
                }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] aData - a pointer to the first input 8-bit integer tensor.
        \param [in] aScale - a pointer to the 32-bit float array with scale coefficients of the first input tensor.
        \param [in] aShift - a pointer to the 32-bit float array with shift coefficients of the first input tensor.
        \param [in] bData - a pointer to the second input 8-bit integer tensor.
        \param [in] bScale - a pointer to the 32-bit float array with scale coefficients of the second input tensor.
        \param [in] bShift - a pointer to the 32-bit float array with shift coefficients of the second input tensor.
        \param [out] cData - a pointer to the output 8-bit integer tensor.
        \param [in] cScale - a pointer to the 32-bit float array with scale coefficients of the output tensor.
        \param [in] cShift - a pointer to the 32-bit float array with shift coefficients of the output tensor.
        \param [in] batch - a batch size of input and output image tensors.
        \param [in] channels - a number of channels in input and output image tensors.
        \param [in] spatial - a spatial size of input and output image tensors.
        \param [in] format - a format of input and output image tensors.
        \param [in] compatibility - a flags of bitwise compatibility.
    */
    SIMD_API void SimdSynetAdd8i(const uint8_t * aData, const float * aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
        uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);

    /*! @ingroup synet_conversion

        \fn void SimdSynetConvert32fTo8u(const float * src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float * shift, uint8_t * dst, SimdSynetCompatibilityType compatibility);

        \short Converts 32-bit float point image to 8-bit unsigned integer image.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>. 

        \param [in] src - a pointer to the 32-bit float array with input image tensor. 
        \param [in] batch - a number of images in the batch of (input/output) image tensor.
        \param [in] channels - a number of channels in the (input/output) image tensor.
        \param [in] height - a height of (input/output) image tensor.
        \param [in] width - a width of (input/output) image tensor.
        \param [in] format - a format of (input/output) image tensor.
        \param [in] scale - a pointer to the 32-bit float array with scale coefficients. 
        \param [in] shift - a pointer to the 32-bit float array with shift coefficients. 
        \param [out] dst - a pointer to the 8-bit unsigned integer array with output image tensor. 
        \param [in] compatibility - a flags of bitwise compatibility.
    */
    SIMD_API void SimdSynetConvert32fTo8u(const float * src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float * shift, uint8_t* dst, SimdSynetCompatibilityType compatibility);

    /*! @ingroup synet_conversion

        \fn void SimdSynetConvert8uTo32f(const uint8_t* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, float* dst, SimdSynetCompatibilityType compatibility);

        \short Converts 8-bit unsigned integer image to 32-bit float point image.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 8-bit unsigned integer array with input image tensor.
        \param [in] batch - a number of images in the batch of (input/output) image tensor.
        \param [in] channels - a number of channels in the (input/output) image tensor.
        \param [in] height - a height of (input/output) image tensor.
        \param [in] width - a width of (input/output) image tensor.
        \param [in] format - a format of (input/output) image tensor.
        \param [in] scale - a pointer to the 32-bit float array with scale coefficients.
        \param [in] shift - a pointer to the 32-bit float array with shift coefficients.
        \param [out] dst - a pointer to the array with 32-bit float output image tensor.
        \param [in] compatibility - a flags of bitwise compatibility.
    */
    SIMD_API void SimdSynetConvert8uTo32f(const uint8_t* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, float* dst, SimdSynetCompatibilityType compatibility);

    /*! @ingroup synet_convolution_fp32

        \fn void * SimdSynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);

        \short Initilizes FP32 convolution algorithm.

        \param [in] batch - a batch size.
        \param [in] conv - a pointer to convolution parameters.
        \param [in] gemm - a pointer to external function of matrix multiplication. Can be NULL.
        \return a pointer to FP32 convolution context. On error it returns NULL. It must be released with using of function ::SimdRelease.
            This pointer is used in functions ::SimdSynetConvolution32fExternalBufferSize, ::SimdSynetConvolution32fInternalBufferSize, 
            ::SimdSynetConvolution32fInfo, ::SimdSynetConvolution32fSetParams and ::SimdSynetConvolution32fForward.
    */
    SIMD_API void * SimdSynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);

    /*! @ingroup synet_convolution_fp32

        \fn size_t SimdSynetConvolution32fExternalBufferSize(const void * context);

        \short Gets size of external temporary buffer required for FP32 convolution algorithm.

        \param [in] context - a pointer to FP32 convolution context. It must be created by function ::SimdSynetConvolution32fInit and released by function ::SimdRelease.
        \return size of external temporary buffer required for FP32 convolution algorithm.
    */
    SIMD_API size_t SimdSynetConvolution32fExternalBufferSize(const void * context);

    /*! @ingroup synet_convolution_fp32

        \fn size_t SimdSynetConvolution32fInternalBufferSize(const void * context);

        \short Gets size of internal buffer used inside FP32 convolution algorithm.

        \param [in] context - a pointer to FP32 convolution context. It must be created by function ::SimdSynetConvolution32fInit and released by function ::SimdRelease.
        \return size of internal buffer used inside FP32 convolution algorithm.
    */
    SIMD_API size_t SimdSynetConvolution32fInternalBufferSize(const void * context);

    /*! @ingroup synet_convolution_fp32

        \fn const char* SimdSynetConvolution32fInfo(const void* context);

        \short Gets description of internal implementation of FP32 convolution algorithm.

        \param [in] context - a pointer to FP32 convolution context. It must be created by function ::SimdSynetConvolution32fInit and released by function ::SimdRelease.
        \return string with description of internal implementation of FP32 convolution algorithm.
    */
    SIMD_API const char* SimdSynetConvolution32fInfo(const void* context);

    /*! @ingroup synet_convolution_fp32

        \fn void SimdSynetConvolution32fSetParams(void * context, const float * weight, SimdBool * internal, const float * bias, const float * params);

        \short Sets weights, biases and parameters of activation function required for FP32 convolution algorithm.

        \param [in, out] context - a pointer to FP32 convolution context. It must be created by function ::SimdSynetConvolution32fInit and released by function ::SimdRelease.
        \param [in] weight - a pointer to convolution weights.
        \param [out] internal - a flag signalized that weight is stored in the internal buffer. Can be NULL.
        \param [in] bias - a pointer to bias. Can be NULL.
        \param [in] params - a pointer to parameters of activation functions (see ::SimdConvolutionActivationType). Can be NULL.
    */
    SIMD_API void SimdSynetConvolution32fSetParams(void * context, const float * weight, SimdBool * internal, const float * bias, const float * params);

    /*! @ingroup synet_convolution_fp32

        \fn void SimdSynetConvolution32fForward(void * context, const float * src, float * buf, float * dst);

        \short Performs forward propagation of FP32 convolution algorithm.

        \param [in] context - a pointer to FP32 convolution context. It must be created by function ::SimdSynetConvolution32fInit and released by function ::SimdRelease.
        \param [in] src - a pointer to input tensor.
        \param [out] buf - a pointer to external temporary buffer. The size of the external temporary buffer is determined by function ::SimdSynetConvolution32fExternalBufferSize. Can be NULL (it causes usage of internal buffer).
        \param [out] dst - a pointer to output tensor.
    */
    SIMD_API void SimdSynetConvolution32fForward(void * context, const float * src, float * buf, float * dst);

    /*! @ingroup synet_convolution_int8

        \fn void * SimdSynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility);

        \short Initilizes INT8 convolution algorithm.

        \param [in] batch - a batch size.
        \param [in] conv - a pointer to convolution parameters.
        \param [in] compatibility - a flags of bitwise compatibility.
        \return a pointer to INT8 convolution context. On error it returns NULL. It must be released with using of function ::SimdRelease.
            This pointer is used in functions ::SimdSynetConvolution8iExternalBufferSize, ::SimdSynetConvolution8iInternalBufferSize, 
            ::SimdSynetConvolution8iInfo, ::SimdSynetConvolution8iSetParams and ::SimdSynetConvolution8iForward.
    */
    SIMD_API void * SimdSynetConvolution8iInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility);

    /*! @ingroup synet_convolution_int8

        \fn size_t SimdSynetConvolution8iExternalBufferSize(const void * context);

        \short Gets size in bytes of external temporary buffer required for INT8 convolution algorithm.

        \param [in] context - a pointer to INT8 convolution context. It must be created by function ::SimdSynetConvolution8iInit and released by function ::SimdRelease.
        \return size of external temporary buffer required for INT8 convolution algorithm.
    */
    SIMD_API size_t SimdSynetConvolution8iExternalBufferSize(const void * context);

    /*! @ingroup synet_convolution_int8

        \fn size_t SimdSynetConvolution8iInternalBufferSize(const void * context);

        \short Gets size of internal buffer used inside INT8 convolution algorithm.

        \param [in] context - a pointer to INT8 convolution context. It must be created by function ::SimdSynetConvolution8iInit and released by function ::SimdRelease.
        \return size of internal buffer used inside INT8 convolution algorithm.
    */
    SIMD_API size_t SimdSynetConvolution8iInternalBufferSize(const void * context);

    /*! @ingroup synet_convolution_int8

        \fn const char* SimdSynetConvolution8iInfo(const void* context);

        \short Gets description of internal implementation of INT8 convolution algorithm.

        \param [in] context - a pointer to INT8 convolution context. It must be created by function ::SimdSynetConvolution8iInit and released by function ::SimdRelease.
        \return string with description of internal implementation of INT8 convolution algorithm.
    */
    SIMD_API const char* SimdSynetConvolution8iInfo(const void* context);

    /*! @ingroup synet_convolution_int8

        \fn void SimdSynetConvolution8iSetParams(void * context, const float * weight, const float * bias, const float * params, const float * const * stats);

        \short Sets weights, biases, parameters of activation function, input/output tensor statistics required for INT8 convolution algorithm.

        \param [in, out] context - a pointer to INT8 convolution context. It must be created by function ::SimdSynetConvolution8iInit and released by function ::SimdRelease.
        \param [in] weight - a pointer to original (32-bit float point) convolution weights.
        \param [in] bias - a pointer to original (32-bit float point) bias. Can be NULL.
        \param [in] params - a pointer to original (32-bit float point) parameters of activation functions (see ::SimdConvolutionActivationType). Can be NULL.
        \param [in] stats - a pointer to pointers with statistics of input(min - stats[0], max - stats[1]) and output(min - stats[2], max - stats[3]) tensors.
    */
    SIMD_API void SimdSynetConvolution8iSetParams(void * context, const float * weight, const float * bias, const float * params, const float * const* stats);

    /*! @ingroup synet_convolution_int8

        \fn void SimdSynetConvolution8iForward(void * context, const uint8_t * src, uint8_t * buf, uint8_t * dst);

        \short Performs forward propagation of INT8 convolution algorithm.

        \param [in] context - a pointer to INT8 convolution context. It must be created by function ::SimdSynetConvolution8iInit and released by function ::SimdRelease.
        \param [in] src - a pointer to input tensor.
        \param [out] buf - a pointer to external temporary buffer. The size of the external temporary buffer is determined by function ::SimdSynetConvolution8iExternalBufferSize. Can be NULL (it causes usage of internal buffer).
        \param [out] dst - a pointer to output tensor.
    */
    SIMD_API void SimdSynetConvolution8iForward(void * context, const uint8_t * src, uint8_t * buf, uint8_t * dst);

    /*! @ingroup synet_deconvolution_fp32

        \fn void * SimdSynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);

        \short Initilizes FP32 deconvolution algorithm.

        \param [in] batch - a batch size.
        \param [in] conv - a pointer to deconvolution parameters.
        \param [in] gemm - a pointer to external function of matrix multiplication. Can be NULL.
        \return a pointer to FP32 deconvolution context. On error it returns NULL. It must be released with using of function ::SimdRelease.
            This pointer is used in functions ::SimdSynetDeconvolution32fExternalBufferSize, ::SimdSynetDeconvolution32fInternalBufferSize, 
            ::SimdSynetDeconvolution32fInfo, ::SimdSynetDeconvolution32fSetParams and ::SimdSynetDeconvolution32fForward.
    */
    SIMD_API void * SimdSynetDeconvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);

    /*! @ingroup synet_deconvolution_fp32

        \fn size_t SimdSynetDeconvolution32fExternalBufferSize(const void * context);

        \short Gets size of external temporary buffer required for FP32 deconvolution algorithm.

        \param [in] context - a pointer to FP32 deconvolution context. It must be created by function ::SimdSynetDeconvolution32fInit and released by function ::SimdRelease.
        \return size of external temporary buffer required for FP32 deconvolution algorithm.
    */
    SIMD_API size_t SimdSynetDeconvolution32fExternalBufferSize(const void * context);

    /*! @ingroup synet_deconvolution_fp32

        \fn size_t SimdSynetDeconvolution32fInternalBufferSize(const void * context);

        \short Gets size of internal buffer used inside FP32 deconvolution algorithm.

        \param [in] context - a pointer to FP32 deconvolution context. It must be created by function ::SimdSynetDeconvolution32fInit and released by function ::SimdRelease.
        \return size of internal buffer used inside FP32 deconvolution algorithm.
    */
    SIMD_API size_t SimdSynetDeconvolution32fInternalBufferSize(const void * context);

    /*! @ingroup synet_deconvolution_fp32

        \fn const char* SimdSynetDeconvolution32fInfo(const void* context);

        \short Gets description of internal implementation of FP32 deconvolution algorithm.

        \param [in] context - a pointer to FP32 deconvolution context. It must be created by function ::SimdSynetDeconvolution32fInit and released by function ::SimdRelease.
        \return string with description of internal implementation of FP32 deconvolution algorithm.
    */
    SIMD_API const char* SimdSynetDeconvolution32fInfo(const void* context);

    /*! @ingroup synet_deconvolution_fp32

        \fn void SimdSynetDeconvolution32fSetParams(void * context, const float * weight, SimdBool * internal, const float * bias, const float * params);

        \short Sets weights, beases and parameters of activation function required for FP32 deconvolution algorithm.

        \param [in, out] context - a pointer to FP32 deconvolution context. It must be created by function ::SimdSynetDeconvolution32fInit and released by function ::SimdRelease.
        \param [in] weight - a pointer to deconvolution weights.
        \param [out] internal - a flag signalized that weight is stored in the internal buffer. Can be NULL.
        \param [in] bias - a pointer to bias. Can be NULL.
        \param [in] params - a pointer to parameters of activation functions (see ::SimdConvolutionActivationType). Can be NULL.
    */
    SIMD_API void SimdSynetDeconvolution32fSetParams(void * context, const float * weight, SimdBool * internal, const float * bias, const float * params);

    /*! @ingroup synet_deconvolution_fp32

        \fn void SimdSynetDeconvolution32fForward(void * context, const float * src, float * buf, float * dst);

        \short Performs forward propagation of FP32 deconvolution algorithm.

        \param [in] context - a pointer to FP32 deconvolution context. It must be created by function ::SimdSynetDeconvolution32fInit and released by function ::SimdRelease.
        \param [in] src - a pointer to input tensor.
        \param [out] buf - a pointer to external temporary buffer. The size of the external temporary buffer is determined by function ::SimdSynetDeconvolution32fExternalBufferSize. Can be NULL (it causes usage of internal buffer).
        \param [out] dst - a pointer to output tensor.
    */
    SIMD_API void SimdSynetDeconvolution32fForward(void * context, const float * src, float * buf, float * dst);

    /*! @ingroup synet

        \fn void SimdSynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst);

        \short This function is used for forward propagation of EltwiseLayer.

        Algorithm's details for ::SimdSynetEltwiseOperationProduct:
        \verbatim
        for(j = 0; j < size; ++j)
            dst[j] = 1;
        for(i = 0; i < count; ++i)
            for(j = 0; j < size; ++j)
                dst[j] *= src[i][j];
        \endverbatim

        Algorithm's details for ::SimdSynetEltwiseOperationSum:
        \verbatim
        for(j = 0; j < size; ++j)
            dst[j] = 0;
        for(i = 0; i < count; ++i)
            for(j = 0; j < size; ++j)
                dst[j] += src[i][j]*weight[i];
        \endverbatim

        Algorithm's details for ::SimdSynetEltwiseOperationMax:
        \verbatim
        for(j = 0; j < size; ++j)
            dst[j] = -FLT_MAX;
        for(i = 0; i < count; ++i)
            for(j = 0; j < size; ++j)
                dst[j] = Max(dst[j], src[i][j]);
        \endverbatim

        Algorithm's details for ::SimdSynetEltwiseOperationMin:
        \verbatim
        for(j = 0; j < size; ++j)
            dst[j] = FLT_MAX;
        for(i = 0; i < count; ++i)
            for(j = 0; j < size; ++j)
                dst[j] = Min(dst[j], src[i][j]);
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to poitres to the input 32-bit float arrays. 
        \param [in] weight - a pointer to the 32-bit float array with sum coefficients. It is need only for ::SimdSynetEltwiseOperationSum operation type otherwise it can be NULL.
        \param [in] count - a count of input arrays. Must be at least 2.
        \param [in] size - a size of the input and output arrays.
        \param [in] type - a type of operation (see ::SimdSynetEltwiseOperationType).
        \param [out] dst - a pointer to the output 32-bit float array.
    */
    SIMD_API void SimdSynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst);

    /*! @ingroup synet_activation

        \fn void SimdSynetElu32f(const float * src, size_t size, const float * alpha, float * dst);

        \short Calculates ELU activation function for 32-bit float array.

        The input and output arrays must have the same size.

        Algorithm's details:
        \verbatim
        for(i = 0; i < size; ++i)
            dst[i] = src[i] >= 0 ? src[i] : alpha*(Exp(src[i]) - 1);
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array.
        \param [in] size - a size of input and output arrays.
        \param [in] alpha - a pointer to alpha parameter.
        \param [out] dst - a pointer to the output 32-bit float array.
    */
    SIMD_API void SimdSynetElu32f(const float * src, size_t size, const float * alpha, float * dst);

    /*! @ingroup synet_fused

        \fn void SimdSynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

        \short This function is used for forward propagation of FusedLayer (type 0).

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(s = 0; s < spatial; ++s)
            {
                o = c*spatial + s;
                x = src[o] + bias[c];
                dst[o] = (x - abs(x))*scale[c] + max(0, x);
            }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 32-bit float array with input image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] bias - a pointer to the 32-bit float array with bias coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] scale - a pointer to the 32-bit float array with scale coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] channels - a number of channels in the (input/output) image tensor.
        \param [in] spatial - a spatial size of (input/output) image tensor.
        \param [out] dst - a pointer to the 32-bit float array with output image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] format - a format of (input/output) image tensor.
    */
    SIMD_API void SimdSynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

    /*! @ingroup synet_fused

        \fn void SimdSynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

        \short This function is used for forward propagation of FusedLayer (type 1).

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(s = 0; s < spatial; ++s)
            {
                o = c*spatial + s;
                x = src[o] + bias0[c];
                dst[o] = max(0, -x)*scale1[c] + bias1[c] + max(0, x);
            }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 32-bit float array with input image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] bias0 - a pointer to the 32-bit float array with bias0 coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] scale1 - a pointer to the 32-bit float array with scale1 coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] bias1 - a pointer to the 32-bit float array with bias1 coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] channels - a number of channels in the (input/output) image tensor.
        \param [in] spatial - a spatial size of (input/output) image tensor.
        \param [out] dst - a pointer to the 32-bit float array with output image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] format - a format of (input/output) image tensor.
        */
    SIMD_API void SimdSynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

    /*! @ingroup synet_fused

        \fn void SimdSynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst, SimdTensorFormatType format);

        \short This function is used for forward propagation of FusedLayer (type 2).

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(s = 0; s < spatial; ++s)
            {
                o = c*spatial + s;
                x = src[o]*scale[c]  + bias[c];
                dst[o] = max(0, x) + min(0, x)*slope[0];
            }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 32-bit float array with input image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] scale - a pointer to the 32-bit float array with scale coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] bias - a pointer to the 32-bit float array with bias coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] channels - a number of channels in the (input/output) image tensor.
        \param [in] spatial - a spatial size of (input/output) image tensor.
        \param [in] slope - a pointer to the 32-bit float slope coefficient.
        \param [out] dst - a pointer to the 32-bit float array with output image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] format - a format of (input/output) image tensor.
        */
    SIMD_API void SimdSynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst, SimdTensorFormatType format);

    /*! @ingroup synet_fused

        \fn void SimdSynetFusedLayerForward3(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

        \short This function is used for forward propagation of FusedLayer (type 3).

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(s = 0; s < spatial; ++s)
            {
                o = c*spatial + s;
                x = src[o] + bias[c];
                dst[o] = max(0, x) + min(0, x)*scale[c];
            }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 32-bit float array with input image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] bias - a pointer to the 32-bit float array with bias coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] scale - a pointer to the 32-bit float array with scale coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] channels - a number of channels in the (input/output) image tensor.
        \param [in] spatial - a spatial size of (input/output) image tensor.
        \param [out] dst - a pointer to the 32-bit float array with output image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] format - a format of (input/output) image tensor.
        */
    SIMD_API void SimdSynetFusedLayerForward3(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

    /*! @ingroup synet_fused

        \fn void SimdSynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

        \short This function is used for forward propagation of FusedLayer (type 4).

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(s = 0; s < spatial; ++s)
            {
                x = src[c*spatial + s] + bias0[c];
                dst[c*spatial + s] = std::max((T)0, x);
                dst[(c + channels)*spatial + s] = std::max((T)0, x*scale1[0] + bias1[0]);
            }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 32-bit float array with input image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] bias0 - a pointer to the 32-bit float array with bias0 coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] scale1 - a pointer to the 32-bit float array with scale1 coefficients. The size of the array is 1.
        \param [in] bias1 - a pointer to the 32-bit float array with bias1 coefficients. The size of the array is 1.
        \param [in] channels - a number of channels in the input image tensor. Output image tensor has 2 * channels.
        \param [in] spatial - a spatial size of (input/output) image tensor.
        \param [out] dst - a pointer to the 32-bit float array with output image tensor. The size of the array is ::SimdAlign (2 * channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] format - a format of (input/output) image tensor.
        */
    SIMD_API void SimdSynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

    /*! @ingroup synet_fused

        \fn void SimdSynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

        \short This function is used for forward propagation of FusedLayer (type 8).

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(s = 0; s < spatial; ++s)
            {
                o = c*spatial + s;
                dst[o] = src0[o] + src1[o]*src2[c];
            }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src0 - a pointer to the first input 32-bit float array. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] src1 - a pointer to the second input 32-bit float array. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] src2 - a pointer to the third input 32-bit float array. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] channels - a number of channels in the (input/output) image tensor. 
        \param [in] spatial - a spatial size of (input/output) image tensor.
        \param [out] dst - a pointer to the output 32-bit float array. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] format - a format of (input/output) image tensor.
        */
    SIMD_API void SimdSynetFusedLayerForward8(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

    /*! @ingroup synet_fused

        \fn void SimdSynetFusedLayerForward9(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, SimdTensorFormatType format);

        \short This function is used for forward propagation of FusedLayer (type 9).

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels0; ++c)
            for(s = 0; s < spatial; ++s)
            {
                dst0[c*spatial + s] = max(0, src0[c*spatial + s]*scale[c] + bias[c]);
                if(dst1)
                    dst1[c*spatial + s] = src0[c*spatial + s];
            }
        for(c = 0; c < channels1; ++c)
            for(s = 0; s < spatial; ++s)
            {
                dst0[(c + channels0)*spatial + s] = max(0, src1[c*spatial + s]*scale[channels0 + c] + bias[channels0 + c]);
                if(dst1)
                    dst1[(c + channels0)*spatial + s] = src1[c*spatial + s];
            }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src0 - a pointer to the first input 32-bit float array. The size of the array is ::SimdAlign (channels0, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] src1 - a pointer to the second input 32-bit float array. The size of the array is ::SimdAlign (channels1, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] scale - a pointer to the 32-bit float array with scale coefficients. The size of the array is ::SimdAlign (channels0 + channels1, ::SimdSynetTensorAlignment (format)).
        \param [in] bias - a pointer to the 32-bit float array with bias coefficients. The size of the array is ::SimdAlign (channels0 + channels1, ::SimdSynetTensorAlignment (format)).
        \param [in] channels0 - a number of channels in the first input image tensor.
        \param [in] channels1 - a number of channels in the second input image tensor.
        \param [in] spatial - a spatial size of (input/output) image tensor.
        \param [out] dst0 - a pointer to the first output 32-bit float array. The size of the array is ::SimdAlign (channels0 + channels1, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [out] dst1 - a pointer to the second output 32-bit float array. The size of the array is ::SimdAlign (channels0 + channels1, ::SimdSynetTensorAlignment (format)) * spatial. The pointer can be NULL.
        \param [in] format - a format of (input/output) image tensor.
    */
    SIMD_API void SimdSynetFusedLayerForward9(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, SimdTensorFormatType format);

    /*! @ingroup synet_activation

        \fn void SimdSynetHardSigmoid32f(const float * src, size_t size, const float * scale, const float * shift, float * dst);

        \short Calculates HardSigmoid activation function (https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html) for 32-bit float array.

        Input and output arrays must have the same size.

        Algorithm's details:
        \verbatim
        for(i = 0; i < size; ++i)
            dst[i] = Max(0, Min(src[i] * scale + shift, 1));
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array.
        \param [in] size - a size of input and output arrays.
        \param [in] scale - a pointer to scale parameter. This parameter is equal to 1/6 in Pytorch documentation.
        \param [in] shift - a pointer to shift parameter. This parameter is equal to 1/2 in Pytorch documentation.
        \param [out] dst - a pointer to the output 32-bit float array.
    */
    SIMD_API void SimdSynetHardSigmoid32f(const float * src, size_t size, const float * scale, const float * shift, float * dst);

    /*! @ingroup synet_activation

        \fn void SimdSynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst);

        \short Calculates H-Swish activation function (https://arxiv.org/pdf/1905.02244.pdf) for 32-bit float array.

        Input and output arrays must have the same size.

        Algorithm's details:
        \verbatim
        for(i = 0; i < size; ++i)
            dst[i] = Max(Min(src[i], shift) + shift, 0)*scale*src[i];
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array.
        \param [in] size - a size of input and output arrays.
        \param [in] shift - a pointer to shift parameter. It is equal to 3 in original paper.
        \param [in] scale - a pointer to scale parameter. It is equal to 1/6 in original paper.
        \param [out] dst - a pointer to the output 32-bit float array.
    */
    SIMD_API void SimdSynetHswish32f(const float* src, size_t size, const float* shift, const float* scale, float* dst);

    /*! @ingroup synet_inner_product

        \fn void * SimdSynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation);

        \short Initilizes FP32 inner product algorithm.

        \param [in] batch - a batch size.
        \param [in] input - a input vector size.
        \param [in] output - a output vector size.
        \param [in] transpose - a flag of transposing of weight matrix.
        \param [in] activation - an activation function type used after inner product.
        \return a pointer to FP32 inner product context. On error it returns NULL. It must be released with using of function ::SimdRelease.
            This pointer is used in functions ::SimdSynetInnerProduct32fInternalBufferSize, :SimdSynetInnerProduct32fSetParams and ::SimdSynetInnerProduct32fForward.
    */
    SIMD_API void* SimdSynetInnerProduct32fInit(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation);

    /*! @ingroup synet_inner_product

        \fn size_t SimdSynetInnerProduct32fInternalBufferSize(const void * context);

        \short Gets size of internal buffer used inside FP32 inner product algorithm.

        \param [in] context - a pointer to FP32 inner product context. It must be created by function ::SimdSynetInnerProduct32fInit and released by function ::SimdRelease.
        \return size of internal buffer used inside FP32 deconvolution algorithm.
    */
    SIMD_API size_t SimdSynetInnerProduct32fInternalBufferSize(const void* context);

    /*! @ingroup synet_inner_product

        \fn void SimdSynetInnerProduct32fSetParams(void* context, const float* weight, SimdBool* internal, const float* bias, const float* params);

        \short Sets weights, beases and parameters of activation function required for FP32 inner product algorithm.

        \param [in, out] context - a pointer to FP32 inner product context. It must be created by function ::SimdSynetInnerProduct32fInit and released by function ::SimdRelease.
        \param [in] weight - a pointer to inner product weights.
        \param [out] internal - a flag signalized that weight is stored in the internal buffer. Can be NULL.
        \param [in] bias - a pointer to bias. Can be NULL.
        \param [in] params - a pointer to parameters of activation functions (see ::SimdConvolutionActivationType). Can be NULL.
    */
    SIMD_API void SimdSynetInnerProduct32fSetParams(void* context, const float* weight, SimdBool* internal, const float* bias, const float* params);

    /*! @ingroup synet_inner_product

        \fn void SimdSynetInnerProduct32fForward(void* context, const float* src, float* dst);

        \short Performs forward propagation of FP32 inner product algorithm.

        \param [in] context - a pointer to FP32 inner product context. It must be created by function ::SimdSynetInnerProduct32fInit and released by function ::SimdRelease.
        \param [in] src - a pointer to input tensor.
        \param [out] dst - a pointer to output tensor.
    */
    SIMD_API void SimdSynetInnerProduct32fForward(void* context, const float* src, float* dst);

    /*! @ingroup synet_inner_product

        \fn void SimdSynetInnerProductLayerForward(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst);

        \short This function is used for forward propagation of InnerProductLayer.

        Algorithm's details:
        \verbatim
        for(i = 0; i < count; ++i)
        {
            dst[i] = (bias ? bias[i] : 0);
            for(j = 0; j < size; ++j)
               dst[i] += src[j]*weight[i*size + j];
        }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array. The size of the array must be equal to size.
        \param [in] weight - a pointer to the 32-bit float array with weight coefficients. The size of the array must be equal to count*size.
        \param [in] bias - a pointer to the 32-bit float array with bias coefficients. The size of the array must be equal to count. Can be NULL. 
        \param [in] count - a size of output array.
        \param [in] size - a size of input array.
        \param [out] dst - a pointer to the output 32-bit float array. The size of the array must be equal to count.
    */
    SIMD_API void SimdSynetInnerProductLayerForward(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst);

    /*! @ingroup synet_inner_product

        \fn void SimdSynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t * src, const int8_t * weight, int32_t * dst, SimdSynetCompatibilityType compatibility);

        \short This function is used for INT8 forward propagation of InnerProductLayer.

        Algorithm's details:
        \verbatim
        for (i = 0; i < M; ++i)
        {
            for (j = 0; j < N; ++j)
            {
                sum = 0;
                for (k = 0; k < K; ++k)
                    sum += src[i * K + k] * weight[j * K + k];
                dst[i*N + j] = sum;
            }
        }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] M - a batch size.
        \param [in] N - an output size.
        \param [in] K - an input size.
        \param [in] src - a pointer to the input 8-bit unsigned integer array. The size of the array must be equal to M*K.
        \param [in] weight - a pointer to the 8-bit signed integer array with weight. The size of the array must be equal to N*K.
        \param [out] dst - a pointer to the output 32-bit integer array. The size of the array must be equal to M*N.
        \param [in] compatibility - a flags of bitwise compatibility.
    */
    SIMD_API void SimdSynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t * src, const int8_t * weight, int32_t * dst, SimdSynetCompatibilityType compatibility);

    /*! @ingroup synet

        \fn void SimdSynetLrnLayerCrossChannels(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst, SimdTensorFormatType format);

        \short This function is used for forward propagation of LrnLayer (cross channels normalization).

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(s = 0; s < spatial; ++s)
            {
                lo = Max(0, c - half);
                hi = Min(channels, c + half + 1);
                sum = 0;
                for(i = lo; i < ln; ++i)
                    sum += Square(src[i*spatial + s]);
                dst[c*spatial + s] = src[c*spatial + s]*Pow(k[0] + sum*k[1], k[2]);
            }
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 32-bit float array with input image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] half - a local normalization half size.
        \param [in] channels - a number of channels in the (input/output) image tensor
        \param [in] spatial - a spatial size of (input/output) image tensor.
        \param [in] k - a pointer to the 32-bit float array with 3 coefficients (see algorithm details). 
        \param [out] dst - a pointer to the 32-bit float array with output image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] format - a format of (input/output) image tensor.
    */
    SIMD_API void SimdSynetLrnLayerCrossChannels(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst, SimdTensorFormatType format);

    /*! @ingroup synet_merged_convolution_fp32

        \fn void * SimdSynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);

        \short Initilizes FP32 merged convolution algorithm.

        \param [in] batch - a batch size.
        \param [in] convs - an array with convolutions parameters.
        \param [in] count - a number of merged convolutions.
        \param [in] add - a flag that signilizes if we need to add output to source value.
        \return a pointer to FP32 merged convolution context. On error it returns NULL. It must be released with using of function ::SimdRelease.
            This pointer is used in functions ::SimdSynetMergedConvolution32fExternalBufferSize, ::SimdSynetMergedConvolution32fInternalBufferSize, 
            ::SimdSynetMergedConvolution32fInfo, ::SimdSynetMergedConvolution32fSetParams and ::SimdSynetMergedConvolution32fForward.
    */
    SIMD_API void * SimdSynetMergedConvolution32fInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdBool add);

    /*! @ingroup synet_merged_convolution_fp32

        \fn size_t SimdSynetMergedConvolution32fExternalBufferSize(const void * context);

        \short Gets size of external temporary buffer required for FP32 merged convolution algorithm.

        \param [in] context - a pointer to FP32 merged convolution context. It must be created by function ::SimdSynetMergedConvolution32fInit and released by function ::SimdRelease.
        \return size of external temporary buffer required for FP32 merged convolution algorithm.
    */
    SIMD_API size_t SimdSynetMergedConvolution32fExternalBufferSize(const void * context);

    /*! @ingroup synet_merged_convolution_fp32

        \fn size_t SimdSynetMergedConvolution32fInternalBufferSize(const void * context);

        \short Gets size of internal buffer used inside FP32 merged convolution algorithm.

        \param [in] context - a pointer to FP32 merged convolution context. It must be created by function ::SimdSynetMergedConvolution32fInit and released by function ::SimdRelease.
        \return size of internal buffer used inside FP32 merged convolution algorithm.
    */
    SIMD_API size_t SimdSynetMergedConvolution32fInternalBufferSize(const void * context);

    /*! @ingroup synet_merged_convolution_fp32

        \fn const char* SimdSynetMergedConvolution32fInfo(const void* context);

        \short Gets description of internal implementation of FP32 merged convolution algorithm.

        \param [in] context - a pointer to FP32 merged convolution context. It must be created by function ::SimdSynetMergedConvolution32fInit and released by function ::SimdRelease.
        \return string with description of internal implementation of FP32 merged convolution algorithm.
    */
    SIMD_API const char* SimdSynetMergedConvolution32fInfo(const void* context);

    /*! @ingroup synet_merged_convolution_fp32

        \fn void SimdSynetMergedConvolution32fSetParams(void * context, const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params);

        \short Sets weights, beases and parameters of activation function required for FP32 merged convolution algorithm.

        \param [in, out] context - a pointer to FP32 merged convolution context. It must be created by function ::SimdSynetMergedConvolution32fInit and released by function ::SimdRelease.
        \param [in] weight - a pointer to the array with pointers to convolution weights. The array size is determined by number of merged convolutions.
        \param [out] internal - a ponter to the array of flags signalized that weights are stored in the internal buffer. The array size is determined by number of merged convolutions. Can be NULL.
        \param [in] bias - a pointer to the array with pointers to bias. The array size is determined by number of merged convolutions. Can be NULL.
        \param [in] params - a pointer to the array with pointers to parameters of the activation functions (see ::SimdConvolutionActivationType). The array size is determined by number of merged convolutions. Can be NULL.
    */
    SIMD_API void SimdSynetMergedConvolution32fSetParams(void * context, const float * const * weight, SimdBool * internal, const float * const * bias, const float * const * params);

    /*! @ingroup synet_merged_convolution_fp32

        \fn void SimdSynetMergedConvolution32fForward(void * context, const float * src, float * buf, float * dst);

        \short Performs forward propagation of FP32 merged convolution algorithm.

        \param [in] context - a pointer to FP32 merged convolution context. It must be created by function ::SimdSynetMergedConvolution32fInit and released by function ::SimdRelease.
        \param [in] src - a pointer to input image.
        \param [out] buf - a pointer to external temporary buffer. The size of the external temporary buffer is determined by function ::SimdSynetMergedConvolution32fExternalBufferSize. Can be NULL (it causes usage of internal buffer).
        \param [out] dst - a pointer to output image.
    */
    SIMD_API void SimdSynetMergedConvolution32fForward(void * context, const float * src, float * buf, float * dst);

    /*! @ingroup synet_merged_convolution_int8

        \fn void * SimdSynetMergedConvolution8iInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdSynetCompatibilityType compatibility);

        \short Initilizes INT8 merged convolution algorithm.

        \param [in] batch - a batch size.
        \param [in] convs - an array with convolutions parameters.
        \param [in] count - a number of merged convolutions.
        \param [in] compatibility - a flags of bitwise compatibility.
        \return a pointer to INT8 merged convolution context. On error it returns NULL. It must be released with using of function ::SimdRelease.
            This pointer is used in functions ::SimdSynetMergedConvolution8iExternalBufferSize, ::SimdSynetMergedConvolution8iInternalBufferSize, 
            ::SimdSynetMergedConvolution8iInfo, ::SimdSynetMergedConvolution8iSetParams and ::SimdSynetMergedConvolution8iForward.
    */
    SIMD_API void* SimdSynetMergedConvolution8iInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdSynetCompatibilityType compatibility);

    /*! @ingroup synet_merged_convolution_int8

        \fn size_t SimdSynetMergedConvolution8iExternalBufferSize(const void * context);

        \short Gets size in bytes of external temporary buffer required for INT8 merged convolution algorithm.

        \param [in] context - a pointer to INT8 merged convolution context. It must be created by function ::SimdSynetMergedConvolution8iInit and released by function ::SimdRelease.
        \return size in bytes of external temporary buffer required for INT8 merged convolution algorithm.
    */
    SIMD_API size_t SimdSynetMergedConvolution8iExternalBufferSize(const void* context);

    /*! @ingroup synet_merged_convolution_int8

        \fn size_t SimdSynetMergedConvolution8iInternalBufferSize(const void * context);

        \short Gets size in bytes of internal buffer used inside INT8 merged convolution algorithm.

        \param [in] context - a pointer to INT8 merged convolution context. It must be created by function ::SimdSynetMergedConvolution8iInit and released by function ::SimdRelease.
        \return size in bytes of internal buffer used inside INT8 merged convolution algorithm.
    */
    SIMD_API size_t SimdSynetMergedConvolution8iInternalBufferSize(const void* context);

    /*! @ingroup synet_merged_convolution_int8

        \fn const char* SimdSynetMergedConvolution8iInfo(const void* context);

        \short Gets description of internal implementation of INT8 merged convolution algorithm.

        \param [in] context - a pointer to INT8 merged convolution context. It must be created by function ::SimdSynetMergedConvolution8iInit and released by function ::SimdRelease.
        \return string with description of internal implementation of INT8 merged convolution algorithm.
    */
    SIMD_API const char* SimdSynetMergedConvolution8iInfo(const void* context);

    /*! @ingroup synet_merged_convolution_int8

        \fn void SimdSynetMergedConvolution8iSetParams(void* context, const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params, const float* const* stats);

        \short Sets weights, beases and parameters of activation function required for INT8 merged convolution algorithm.

        \param [in, out] context - a pointer to INT8 merged convolution context. It must be created by function ::SimdSynetMergedConvolution8iInit and released by function ::SimdRelease.
        \param [in] weight - a pointer to the array with pointers to convolution weights. The array size is determined by number of merged convolutions.
        \param [out] internal - a ponter to the array of flags signalized that weights are stored in the internal buffer. The array size is determined by number of merged convolutions. Can be NULL.
        \param [in] bias - a pointer to the array with pointers to bias. The array size is determined by number of merged convolutions. Can be NULL.
        \param [in] params - a pointer to the array with pointers to parameters of the activation functions (see ::SimdConvolutionActivationType). The array size is determined by number of merged convolutions. Can be NULL.
        \param [in] stats - a pointer to pointers with statistics of input(min - stats[0], max - stats[1]), interim(min - stats[2], max - stats[3]) and output(min - stats[4], max - stats[5]) tensors.
    */
    SIMD_API void SimdSynetMergedConvolution8iSetParams(void* context, const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params, const float* const* stats);

    /*! @ingroup synet_merged_convolution_int8

        \fn void SimdSynetMergedConvolution8iForward(void * context, const uint8_t* src, uint8_t* buf, uint8_t* dst);

        \short Performs forward propagation of INT8 merged convolution algorithm.

        \param [in] context - a pointer to INT8 merged convolution context. It must be created by function ::SimdSynetMergedConvolution8iInit and released by function ::SimdRelease.
        \param [in] src - a pointer to input image.
        \param [out] buf - a pointer to external temporary buffer. The sizein bytes of the external temporary buffer is determined by function ::SimdSynetMergedConvolution8iExternalBufferSize. Can be NULL (it causes usage of internal buffer).
        \param [out] dst - a pointer to output image.
    */
    SIMD_API void SimdSynetMergedConvolution8iForward(void* context, const uint8_t* src, uint8_t* buf, uint8_t* dst);

    /*! @ingroup synet_activation

        \fn void SimdSynetMish32f(const float* src, size_t size, const float* threshold, float* dst);

        Calculates Mish activation function (https://arxiv.org/abs/1908.08681) for 32-bit float array

        Algorithm's details:
        \verbatim
        for(i = 0; i < size; ++i)
            dst[i] = src[i] > threshold ? src[i] : src[i] * tanh(log(exp(src[i]) + 1));
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array.
        \param [in] size - a size of input and output arrays.
        \param [in] threshold - a pointer to 'threshold' parameter.
        \param [out] dst - a pointer to the output 32-bit float array.
    */
    SIMD_API void SimdSynetMish32f(const float* src, size_t size, const float* threshold, float* dst);

    /*! @ingroup synet

        \fn void SimdSynetPoolingForwardAverage(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, SimdBool excludePad, SimdTensorFormatType format);

        \short This function is used for forward propagation of PoolingLayer (AveragePooling).

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array. The size of the array must be equal to srcC*srcH*srcW.
        \param [in] srcC - a number of input and output channels.
        \param [in] srcH - an input height.
        \param [in] srcW - an input width.
        \param [in] kernelY - a height of the pooling kernel.
        \param [in] kernelX - a width of the pooling kernel.
        \param [in] strideY - a y-stride of the pooling.
        \param [in] strideX - a x-stride of the pooling.
        \param [in] padY - a pad to the top of the input image.
        \param [in] padX - a pad to the left of the input image.
        \param [out] dst - a pointer to the output 32-bit float array. The size of the array must be equal to srcC*dstH*dstW.
        \param [in] dstH - an output height.
        \param [in] dstW - an output width.
        \param [in] excludePad - a flag of exclude pad from average value calculation.
        \param [in] format - a format of (input/output) image tensor.
    */
    SIMD_API void SimdSynetPoolingForwardAverage(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
        size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, SimdBool excludePad, SimdTensorFormatType format);

    /*! @ingroup synet

        \fn void SimdSynetPoolingForwardMax32f(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, SimdTensorFormatType format);

        \short This function is used for forward propagation of PoolingLayer (MaxPooling, 32-bit float).

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array. The size of the array must be equal to srcC*srcH*srcW.
        \param [in] srcC - a number of input and output channels.
        \param [in] srcH - an input height.
        \param [in] srcW - an input width.
        \param [in] kernelY - a height of the pooling kernel.
        \param [in] kernelX - a width of the pooling kernel.
        \param [in] strideY - a y-stride of the pooling.
        \param [in] strideX - a x-stride of the pooling.
        \param [in] padY - a pad to the top of the input image.
        \param [in] padX - a pad to the left of the input image.
        \param [out] dst - a pointer to the output 32-bit float array. The size of the array must be equal to srcC*dstH*dstW.
        \param [in] dstH - an output height.
        \param [in] dstW - an output width.
        \param [in] format - a format of (input/output) image tensor.
    */
    SIMD_API void SimdSynetPoolingForwardMax32f(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX, 
        size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, SimdTensorFormatType format);

    /*! @ingroup synet

        \fn void SimdSynetPoolingForwardMax8u(const uint8_t * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t * dst, size_t dstH, size_t dstW, SimdTensorFormatType format);

        \short This function is used for forward propagation of PoolingLayer (MaxPooling, 8-bit unsigned integer).

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 8-bit unsigned integer array. The size of the array must be equal to srcC*srcH*srcW.
        \param [in] srcC - a number of input and output channels.
        \param [in] srcH - an input height.
        \param [in] srcW - an input width.
        \param [in] kernelY - a height of the pooling kernel.
        \param [in] kernelX - a width of the pooling kernel.
        \param [in] strideY - a y-stride of the pooling.
        \param [in] strideX - a x-stride of the pooling.
        \param [in] padY - a pad to the top of the input image.
        \param [in] padX - a pad to the left of the input image.
        \param [out] dst - a pointer to the output 8-bit unsigned integer array. The size of the array must be equal to srcC*dstH*dstW.
        \param [in] dstH - an output height.
        \param [in] dstW - an output width.
        \param [in] format - a format of (input/output) image tensor.
    */
    SIMD_API void SimdSynetPoolingForwardMax8u(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
        size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format);


    /*! @ingroup synet_activation

        \fn void SimdSynetPreluLayerForward(const float * src, const float * slope, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

        \short This function is used for forward propagation of PreluLayer (PReLU).

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(s = 0; s < spatial; ++s)
                dst[c*spatial + s] = src[c*spatial + s] > 0 ? src[c*spatial + s] : slope[c]*src[c*spatial + s];
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 32-bit float array with input image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] slope - a pointer to the 32-bit float array with slope coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format).
        \param [in] channels - a number of channels in the (input/output) image tensor
        \param [in] spatial - a spatial size of (input/output) image tensor.
        \param [out] dst - a pointer to the 32-bit float array with output image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] format - a format of (input/output) image tensor.
    */
    SIMD_API void SimdSynetPreluLayerForward(const float * src, const float * slope, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

    /*! @ingroup synet_activation

        \fn void SimdSynetRelu32f(const float* src, size_t size, const float* slope, float* dst);

        \short Calculates ReLU (rectified linear unit) function for 32-bit float array.

        Algorithm's details:
        \verbatim
        for(i = 0; i < size; ++i)
            dst[i] =  src[i] > 0 ? src[i] : slope*src[i];
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array.
        \param [in] size - a size of input and output arrays.
        \param [in] slope - a pointer to the 'slope' parameter.
        \param [out] dst - a pointer to output 32-bit float array.
    */
    SIMD_API void SimdSynetRelu32f(const float* src, size_t size, const float* slope, float* dst);

    /*! @ingroup synet_conversion

        \fn void SimdSynetReorderImage(size_t batch, size_t channels, size_t spatial, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat);

        \short Converts (input/output) image between different formats of 4D-tensor.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>. Conversion between ::SimdTensorFormatNchw4c, ::SimdTensorFormatNchw8c, ::SimdTensorFormatNchw16c is not supported.

        \param [in] batch - a batch (number of images in the batch).
        \param [in] channels - a number of image channels.
        \param [in] spatial - a spatial size (height*width) of image.
        \param [in] src - a pointer to input image data.
        \param [in] srcFormat - a format of input image. It can be ::SimdTensorFormatNchw, ::SimdTensorFormatNhwc, ::SimdTensorFormatNchw4c, ::SimdTensorFormatNchw8c, ::SimdTensorFormatNchw16c.
        \param [out] dst - a pointer to output image data.
        \param [in] dstFormat - a format of output image. It can be ::SimdTensorFormatNchw, ::SimdTensorFormatNhwc, ::SimdTensorFormatNchw4c, ::SimdTensorFormatNchw8c, ::SimdTensorFormatNchw16c.
    */
    SIMD_API void SimdSynetReorderImage(size_t batch, size_t channels, size_t spatial, const float* src, SimdTensorFormatType srcFormat, float* dst, SimdTensorFormatType dstFormat);

    /*! @ingroup synet_conversion

        \fn void SimdSynetReorderFilter(size_t output, size_t input, size_t kernel, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat);

        \short Converts 2d-convolution filter weight between different formats of 4D-tensor.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>. Conversion between ::SimdTensorFormatOyxi4o, ::SimdTensorFormatOyxi8o, ::SimdTensorFormatOyxi16o is not supported.

        \param [in] output - a number of output channels in filter.
        \param [in] input - a number of intput channels in filter.
        \param [in] kernel - a size (width*height) of filter kernel.
        \param [in] src - a pointer to input filter data.
        \param [in] srcFormat - a format of input filter. It can be ::SimdTensorFormatOiyx, ::SimdTensorFormatYxio, ::SimdTensorFormatOyxi4o, ::SimdTensorFormatOyxi8o, ::SimdTensorFormatOyxi16o.
        \param [out] dst - a pointer to output filter data.
        \param [in] dstFormat - a format of output filter. It can be SimdTensorFormatOiyx, ::SimdTensorFormatYxio, ::SimdTensorFormatOyxi4o, ::SimdTensorFormatOyxi8o, ::SimdTensorFormatOyxi16o.
    */
    SIMD_API void SimdSynetReorderFilter(size_t output, size_t input, size_t kernel, const float* src, SimdTensorFormatType srcFormat, float* dst, SimdTensorFormatType dstFormat);

    /*! @ingroup synet_activation

        \fn void SimdSynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst);

        \short This function is used in order to restrict range for given 320bit float array.

        Algorithm's details:
        \verbatim
        for(i = 0; i < size; ++i)
            dst[i] = Min(Max(lower, src[i]), upper);
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array.
        \param [in] size - a size of input and output arrays.
        \param [in] lower - a pointer to lower restrict bound.
        \param [in] upper - a pointer to upper restrict bound.
        \param [out] dst - a pointer to the output 32-bit float array.
    */
    SIMD_API void SimdSynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst);

    /*! @ingroup synet_scale

        \fn void SimdSynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);

        \short This function is used for forward propagation of ScaleLayer.

        Algorithm's details (example for NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(h = 0; h < height; ++h)
                for(w = 0; w < width; ++w)
                    dst[(c*height + h)*width + w] = src[(c*height + h)*width + w]*scale[c] + (bias ? bias[c] : 0);
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 32-bit float array with input image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] scale - a pointer to the 32-bit float array with scale coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)).
        \param [in] bias - a pointer to the 32-bit float array with bias coefficients. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)). Can be NULL.
        \param [in] channels - a number of channels in the (input/output) image tensor.
        \param [in] height - a height of (input/output) image tensor.
        \param [in] width - a width of (input/output) image tensor.
        \param [out] dst - a pointer to the 32-bit float array with output image tensor. The size of the array is ::SimdAlign (channels, ::SimdSynetTensorAlignment (format)) * spatial.
        \param [in] format - a format of (input/output) image tensor.
        \param [in] compatibility - a flags of bitwise compatibility.
    */
    SIMD_API void SimdSynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);

    /*! @ingroup synet_scale

        \fn void * SimdSynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);

        \short Initilizes INT8 scale algorithm.

        \param [in] batch - a batch size.
        \param [in] channels - a numbeo of channels.
        \param [in] spatial - a spatial image size.
        \param [in] srcType - an input data type (SimdTensorData32f or SimdTensorData8u).
        \param [in] dstType - an output data type (SimdTensorData32f or SimdTensorData8u).
        \param [in] format - a format of (input/output) image tensor.
        \param [in] compatibility - a flags of bitwise compatibility.
        \return a pointer to INT8 scale context. On error it returns NULL. It must be released with using of function ::SimdRelease.
            This pointer is used in functions ::SimdSynetScale8iInternalBufferSize, ::SimdSynetScale8iSetParams and ::SimdSynetScale8iForward.
    */
    SIMD_API void* SimdSynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);

    /*! @ingroup synet_scale

        \fn size_t SimdSynetScale8iInternalBufferSize(const void * context);

        \short Gets size of internal buffer used inside INT8 scale algorithm.

        \param [in] context - a pointer to INT8 scale context. It must be created by function ::SimdSynetScale8iInit and released by function ::SimdRelease.
        \return size of internal buffer used inside INT8 scale algorithm.
    */
    SIMD_API size_t SimdSynetScale8iInternalBufferSize(const void* context);

    /*! @ingroup synet_scale

        \fn void SimdSynetScale8iSetParams(void * context, const float * scale, const float * bias, const float * const * stats);

        \short Sets scale, bias, parameters of activation function, input/output tensor statistics required for INT8 scale algorithm.

        \param [in, out] context - a pointer to INT8 convolution context. It must be created by function ::SimdSynetScale8iInit and released by function ::SimdRelease.
        \param [in] scale - a pointer to original (32-bit float point) scale.
        \param [in] bias - a pointer to original (32-bit float point) bias. Can be NULL.
        \param [in] stats - a pointer to pointers with statistics of input(min - stats[0], max - stats[1]) and output(min - stats[2], max - stats[3]) tensors. Can be NULL for subsequent calls of this function.
    */
    SIMD_API void SimdSynetScale8iSetParams(void* context, const float* scale, const float* bias, const float* const* stats);

    /*! @ingroup synet_scale

        \fn void SimdSynetScale8iForward(void * context, const uint8_t * src, uint8_t * dst);

        \short Performs forward propagation of INT8 scale algorithm.

        \param [in] context - a pointer to INT8 scale context. It must be created by function ::SimdSynetScale8iInit and released by function ::SimdRelease.
        \param [in] src - a pointer to input tensor.
        \param [out] dst - a pointer to output tensor.
    */
    SIMD_API void SimdSynetScale8iForward(void* context, const uint8_t* src, uint8_t* dst);

    /*! @ingroup synet_conversion

        \fn void void SimdSynetSetInput(const uint8_t * src, size_t width, size_t height, size_t stride, SimdPixelFormatType srcFormat, const float * lower, const float * upper, float * dst, size_t channels, SimdTensorFormatType dstFormat);

        \short Sets image to the input of neural network of <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        Algorithm's details (example for BGRA pixel format and NCHW tensor format):
        \verbatim
        for(c = 0; c < channels; ++c)
            for(y = 0; y < height; ++y)
                for(x = 0; x < width; ++x)
                    dst[(c*height + y)*width + x] = src[stride*y + width*4 + c]*(upper[c] - lower[c])/255 + lower[c];
        \endverbatim

        \note This function has a C++ wrappers: Simd::SynetSetInput(const View<A> & src, const float * lower, const float * upper, float * dst, size_t channels, SimdTensorFormatType format).

        \param [in] src - a pointer to pixels data of input image.
        \param [in] width - a width of input image and output image tensor.
        \param [in] height - a height of input image and output image tensor.
        \param [in] stride - a row size of input image.
        \param [in] srcFormat - a pixel format of input image. There are supported following pixel formats: ::SimdPixelFormatGray8, ::SimdPixelFormatBgr24, ::SimdPixelFormatBgra32, ::SimdPixelFormatRgb24.
        \param [in] lower - a pointer to the array with lower bound of values of the output tensor. The size of the array have to correspond number of channels in the output image tensor.
        \param [in] upper - a pointer to the array with upper bound of values of the output tensor. The size of the array have to correspond number of channels in the output image tensor.
        \param [out] dst - a pointer to the output 32-bit float image tensor.
        \param [in] channels - a number of channels in the output image tensor. It can be 1 or 3.
        \param [in] dstFormat - a format of output image tensor. There are supported following tensor formats: ::SimdTensorFormatNchw, ::SimdTensorFormatNhwc.
    */
    SIMD_API void SimdSynetSetInput(const uint8_t * src, size_t width, size_t height, size_t stride, SimdPixelFormatType srcFormat, 
        const float * lower, const float * upper, float * dst, size_t channels, SimdTensorFormatType dstFormat);

    /*! @ingroup synet

        \fn void SimdSynetShuffleLayerForward(const float * src0, const float * src1, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, SimdTensorFormatType format, int type);

        \short This function is used for forward propagation of ShuffleLayer.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src0 - a pointer to the 32-bit float array with the first input image tensor.
        \param [in] src1 - a pointer to the 32-bit float array with the second input image tensor.
        \param [in] channels0 - a number of channels in the first input (type == 0) or output (type == 1) image tensor. It must be even number.
        \param [in] channels1 - a number of channels in the second input (type == 0) or output (type == 1) image tensor. It must be even number.
        \param [in] spatial - a spatial size of (input/output) image tensors.
        \param [out] dst0 - a pointer to the 32-bit float array with the first output image tensor.
        \param [out] dst1 - a pointer to the 32-bit float array with the second output image tensor.
        \param [in] format - a format of (input/output) image tensors.
        \param [in] type - a shuffle type (it can be 0 or 1).
    */
    SIMD_API void SimdSynetShuffleLayerForward(const float * src0, const float * src1, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, SimdTensorFormatType format, int type);

    /*! @ingroup synet_activation

        \fn void SimdSynetSigmoid32f(const float * src, size_t size, const float * slope, float * dst);

        \short This function is used for forward propagation of SigmoidLayer.

        Algorithm's details:
        \verbatim
        for(i = 0; i < size; ++i)
            dst[i] = 1/(1 + exp(-slope*src[i]));
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 32-bit float array.
        \param [in] size - a size of input and output arrays.
        \param [in] slope - a pointer to the 'slope' parameter.
        \param [out] dst - a pointer to output 32-bit float array.
    */
    SIMD_API void SimdSynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst);

    /*! @ingroup synet

        \fn void SimdSynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst);

        \short This function is used for forward propagation of SoftmaxLayer.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array. The size of the array must be equal to outer*count*inner.
        \param [in] outer - an outer size of input and output arrays.
        \param [in] count - a size of softmax dimmension.
        \param [in] inner - an inner size of input and output arrays.
        \param [out] dst - a pointer to the output 32-bit float array. The size of the array must be equal to outer*count*inner.
    */
    SIMD_API void SimdSynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst);

    /*! @ingroup synet_activation

        \fn void SimdSynetSoftplus32f(const float* src, size_t size, const float * beta, const float * threshold, float * dst);

        \short This function is used for forward propagation of SoftplusLayer.

        Algorithm's details:
        \verbatim
        for(i = 0; i < size; ++i)
            dst[i] = src[i] > threshold ? src[i] : log(1 + exp(src[i]*beta))/beta;
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array.
        \param [in] size - a size of input and output arrays.
        \param [in] beta - a pointer to 'beta' parameter.
        \param [in] threshold - a pointer to 'threshold' parameter.
        \param [out] dst - a pointer to the output 32-bit float array.
    */
    SIMD_API void SimdSynetSoftplus32f(const float* src, size_t size, const float * beta, const float * threshold, float * dst);

    /*! @ingroup synet

        \fn SimdTensorFormatType SimdSynetSpecifyTensorFormat(SimdTensorFormatType format);

        \short Specifies hardware optimized tensor format of 5D-tensor for (input/output) image or 2D-convolution filter.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>. 

        \param [in] format - an unspecified hardware optimized 5D-tensor format of (input/output) image or 2D-convolution filter. It can be ::SimdTensorFormatNchwXc or ::SimdTensorFormatOyxiXo.
        \return specified hardware optimized 5D-tensor format. 
    */
    SIMD_API SimdTensorFormatType SimdSynetSpecifyTensorFormat(SimdTensorFormatType format);

    /*! @ingroup synet_activation

        \fn void SimdSynetSwish32f(const float * src, size_t size, const float * slope, float * dst);

        \short This function is used for forward propagation of SwishLayer.

        Algorithm's details:
        \verbatim
        for(i = 0; i < size; ++i)
            dst[i] = src[i]/(1 + exp(-slope*src[i]));
        \endverbatim

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the 32-bit float array.
        \param [in] size - a size of input and output arrays.
        \param [in] slope - a pointer to the 'slope' parameter.
        \param [out] dst - a pointer to output 32-bit float array.
    */
    SIMD_API void SimdSynetSwish32f(const float* src, size_t size, const float* slope, float* dst);

    /*! @ingroup synet_activation

        \fn void SimdSynetTanh32f(const float * src, size_t size, const float * slope, float * dst);

        \short Calculates hyperbolic tangent for 32-bit float array.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        Algorithm's details:
        \verbatim
        for(i = 0; i < size; ++i)
        {
            x = slope*src[i];
            dst[i] = (exp(x) - exp(-x))/(exp(x) + exp(-x));
        }
        \endverbatim

        \param [in] src - a pointer to the input 32-bit float array.
        \param [in] size - a size of input and output arrays.
        \param [in] slope - a pointer to the 'slope' parameter.
        \param [out] dst - a pointer to output 32-bit float array.
    */
    SIMD_API void SimdSynetTanh32f(const float* src, size_t size, const float* slope, float* dst);

    /*! @ingroup synet

        \fn size_t SimdSynetTensorAlignment(SimdTensorFormatType format);

        \short Gets alignment requred for current tensor format.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] format - a tensor format.
        \return alignment requred for current tensor format.
    */
    SIMD_API size_t SimdSynetTensorAlignment(SimdTensorFormatType format);

    /*! @ingroup synet

        \fn void SimdSynetUnaryOperation32fLayerForward(const float * src, size_t size, SimdSynetUnaryOperation32fType type, float* dst);

        \short This function is used for forward propagation of UnaryOperationLayer.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to poitres to the input 32-bit float arrays.
        \param [in] size - a size of the input and output arrays.
        \param [in] type - an unary operation type (see ::SimdSynetUnaryOperation32fType).
        \param [out] dst - a pointer to the output 32-bit float array.
    */
    SIMD_API void SimdSynetUnaryOperation32fLayerForward(const float * src, size_t size, SimdSynetUnaryOperation32fType type, float * dst);

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

    /*! @ingroup transform

        \fn void SimdTransformImage(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, SimdTransformType transform, uint8_t * dst, size_t dstStride);

        \short Performs transformation of input image. The type of transformation is defined by ::SimdTransformType enumeration.

        \note This function has a C++ wrappers: Simd::TransformImage(const View<A> & src, ::SimdTransformType transform, View<A> & dst).

        \param [in] src - a pointer to pixels data of input image.
        \param [in] srcStride - a row size of input image.
        \param [in] width - an input image width. 
        \param [in] height - an input image height.
        \param [in] pixelSize - a pixel size in input and output images. It can be 1, 2, 3, 4.
        \param [in] transform - a type of image transformation.
        \param [out] dst - a pointer to pixels data of output image.
        \param [in] dstStride - a row size of output image.
    */
    SIMD_API void SimdTransformImage(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, SimdTransformType transform, uint8_t * dst, size_t dstStride);

    /*! @ingroup uyvy_conversion

        \fn void SimdUyvy422ToBgr(const uint8_t * uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride, SimdYuvType yuvType);

        \short Converts 16-bit UYVY422 image to 24-bit BGR image.

        The input and output images must have the same width and height. Width must be even number.

        \note This function has a C++ wrappers: Simd::Uyvy422ToBgr(const View<A>& uyvy, View<A>& bgr, SimdYuvType yuvType = SimdYuvBt601);

        \param [in] uyvy - a pointer to pixels data of input 16-bit UYVY422 image.
        \param [in] uyvyStride - a row size of the UYVY422 image.
        \param [in] width - an image width. Width must be even number.
        \param [in] height - an image height.
        \param [out] bgr - a pointer to pixels data of output 24-bit BGR image.
        \param [in] bgrStride - a row size of the bgr image.
        \param [in] yuvType - a type of input YUV image (see descriprion of ::SimdYuvType).
    */
    SIMD_API void SimdUyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType);

    /*! @ingroup uyvy_conversion

        \fn void SimdUyvy422ToYuv420p(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride);

        \short Converts 16-bit UYVY422 image to YUV420P.

        The input UYVY422 and output Y images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function has a C++ wrapper Simd::Uyvy422ToYuv420p(const View<A>& uyvy, View<A>& y, View<A>& u, View<A>& v).

        \param [in] uyvy - a pointer to pixels data of input 16-bit UYVY422 image.
        \param [in] uyvyStride - a row size of the UYVY422 image.
        \param [in] width - an image width. Width must be even number.
        \param [in] height - an image height.
        \param[out] y - a pointer to pixels data of output 8 - bit image with Y color plane.
        \param[in] yStride - a row size of the y image.
        \param[out] u - a pointer to pixels data of output 8 - bit image with U color plane.
        \param[in] uStride - a row size of the u image.
        \param[out] v - a pointer to pixels data of output 8 - bit image with V color plane.
        \param[in] vStride - a row size of the v image.
    */
    SIMD_API void SimdUyvy422ToYuv420p(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, 
        uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel1x3Block1x4SetFilter(const float * src, size_t size, float * dst, SimdBool trans);

        \short This function is used for filter conversion in Winograd F(1x4,1x3) or F(4x1,3x1) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array with filter weights.
        \param [in] size - (number of input channels)*(number of output channels).
        \param [out] dst - a pointer to the output 32-bit float array with filter weights.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel1x3Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel1x3Block1x4SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t padY, size_t padX, size_t padH, size_t padW, float * dst, size_t dstStride, SimdBool trans);

        \short This function is used for input image conversion in Winograd F(1x4,1x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcChannels - a number of input channels.
        \param [in] srcHeight - a height of input image.
        \param [in] srcWidth - a width of input image.
        \param [in] padY - an additional zero padding of input image at the beginning of Y-axis.
        \param [in] padX - an additional zero padding of input image at the beginning of X-axis.
        \param [in] padH - an additional zero padding of input image at the end of Y-axis.
        \param [in] padW - an additional zero padding of input image at the end of X-axis.
        \param [out] dst - a pointer to the output array with converted image.
        \param [in] dstStride - a stride of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel1x3Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
        size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel1x3Block1x4SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        \short This function is used for output image conversion in Winograd F(1x4,1x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcStride - a stride of input image.
        \param [out] dst - a pointer to the output image.
        \param [in] dstChannels - a number of output channels.
        \param [in] dstHeight - a height of output image.
        \param [in] dstWidth - a width of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel1x3Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel1x5Block1x4SetFilter(const float * src, size_t size, float * dst, SimdBool trans);

        \short This function is used for filter conversion in Winograd F(1x4,1x5) or F(4x1,5x1) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array with filter weights.
        \param [in] size - (number of input channels)*(number of output channels).
        \param [out] dst - a pointer to the output 32-bit float array with filter weights.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel1x5Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel1x5Block1x4SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t padY, size_t padX, size_t padH, size_t padW, float * dst, size_t dstStride, SimdBool trans);

        \short This function is used for input image conversion in Winograd F(1x4,1x5) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcChannels - a number of input channels.
        \param [in] srcHeight - a height of input image.
        \param [in] srcWidth - a width of input image.
        \param [in] padY - an additional zero padding of input image at the beginning of Y-axis.
        \param [in] padX - an additional zero padding of input image at the beginning of X-axis.
        \param [in] padH - an additional zero padding of input image at the end of Y-axis.
        \param [in] padW - an additional zero padding of input image at the end of X-axis.
        \param [out] dst - a pointer to the output array with converted image.
        \param [in] dstStride - a stride of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel1x5Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
        size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel1x5Block1x4SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        \short This function is used for output image conversion in Winograd F(1x4,1x5) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcStride - a stride of input image.
        \param [out] dst - a pointer to the output image.
        \param [in] dstChannels - a number of output channels.
        \param [in] dstHeight - a height of output image.
        \param [in] dstWidth - a width of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel1x5Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel2x2Block2x2SetFilter(const float * src, size_t size, float * dst, SimdBool trans);

        \short This function is used for filter conversion in Winograd F(2x2,2x2) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array with filter weights.
        \param [in] size - (number of input channels)*(number of output channels).
        \param [out] dst - a pointer to the output 32-bit float array with filter weights.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel2x2Block2x2SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel2x2Block2x2SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t padY, size_t padX, size_t padH, size_t padW, float * dst, size_t dstStride, SimdBool trans);

        \short This function is used for input image conversion in Winograd F(2x2,2x2) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcChannels - a number of input channels.
        \param [in] srcHeight - a height of input image.
        \param [in] srcWidth - a width of input image.
        \param [in] padY - an additional zero padding of input image at the beginning of Y-axis.
        \param [in] padX - an additional zero padding of input image at the beginning of X-axis.
        \param [in] padH - an additional zero padding of input image at the end of Y-axis.
        \param [in] padW - an additional zero padding of input image at the end of X-axis.
        \param [out] dst - a pointer to the output array with converted image.
        \param [in] dstStride - a stride of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel2x2Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
        size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel2x2Block2x2SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        \short This function is used for output image conversion in Winograd F(2x2,2x2) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcStride - a stride of input image.
        \param [out] dst - a pointer to the output image.
        \param [in] dstChannels - a number of output channels.
        \param [in] dstHeight - a height of output image.
        \param [in] dstWidth - a width of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel2x2Block2x2SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel2x2Block4x4SetFilter(const float * src, size_t size, float * dst, SimdBool trans);

        \short This function is used for filter conversion in Winograd F(4x4,2x2) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array with filter weights.
        \param [in] size - (number of input channels)*(number of output channels).
        \param [out] dst - a pointer to the output 32-bit float array with filter weights.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel2x2Block4x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel2x2Block4x4SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t padY, size_t padX, size_t padH, size_t padW, float * dst, size_t dstStride, SimdBool trans);

        \short This function is used for input image conversion in Winograd F(4x4,2x2) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcChannels - a number of input channels.
        \param [in] srcHeight - a height of input image.
        \param [in] srcWidth - a width of input image.
        \param [in] padY - an additional zero padding of input image at the beginning of Y-axis.
        \param [in] padX - an additional zero padding of input image at the beginning of X-axis.
        \param [in] padH - an additional zero padding of input image at the end of Y-axis.
        \param [in] padW - an additional zero padding of input image at the end of X-axis.
        \param [out] dst - a pointer to the output array with converted image.
        \param [in] dstStride - a stride of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel2x2Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
        size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel2x2Block4x4SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        \short This function is used for output image conversion in Winograd F(4x4,2x2) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcStride - a stride of input image.
        \param [out] dst - a pointer to the output image.
        \param [in] dstChannels - a number of output channels.
        \param [in] dstHeight - a height of output image.
        \param [in] dstWidth - a width of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel2x2Block4x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel3x3Block2x2SetFilter(const float * src, size_t size, float * dst, SimdBool trans);

        \short This function is used for filter conversion in Winograd F(2x2,3x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array with filter weights.
        \param [in] size - (number of input channels)*(number of output channels).
        \param [out] dst - a pointer to the output 32-bit float array with filter weights.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel3x3Block2x2SetFilter(const float * src, size_t size, float * dst, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel3x3Block2x2SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t padY, size_t padX, size_t padH, size_t padW, float * dst, size_t dstStride, SimdBool trans);

        \short This function is used for input image conversion in Winograd F(2x2,3x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcChannels - a number of input channels.
        \param [in] srcHeight - a height of input image.
        \param [in] srcWidth - a width of input image.
        \param [in] padY - an additional zero padding of input image at the beginning of Y-axis.
        \param [in] padX - an additional zero padding of input image at the beginning of X-axis.
        \param [in] padH - an additional zero padding of input image at the end of Y-axis.
        \param [in] padW - an additional zero padding of input image at the end of X-axis.
        \param [out] dst - a pointer to the output array with converted image.
        \param [in] dstStride - a stride of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel3x3Block2x2SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
        size_t padY, size_t padX, size_t padH, size_t padW, float * dst, size_t dstStride, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel3x3Block2x2SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        \short This function is used for output image conversion in Winograd F(2x2,3x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcStride - a stride of input image.
        \param [out] dst - a pointer to the output image.
        \param [in] dstChannels - a number of output channels.
        \param [in] dstHeight - a height of output image.
        \param [in] dstWidth - a width of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel3x3Block2x2SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel3x3Block3x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans);

        \short This function is used for filter conversion in Winograd F(3x3,3x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array with filter weights.
        \param [in] size - (number of input channels)*(number of output channels).
        \param [out] dst - a pointer to the output 32-bit float array with filter weights.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel3x3Block3x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel3x3Block3x3SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t padY, size_t padX, size_t padH, size_t padW, float * dst, size_t dstStride, SimdBool trans);

        \short This function is used for input image conversion in Winograd F(3x3,3x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcChannels - a number of input channels.
        \param [in] srcHeight - a height of input image.
        \param [in] srcWidth - a width of input image.
        \param [in] padY - an additional zero padding of input image at the beginning of Y-axis.
        \param [in] padX - an additional zero padding of input image at the beginning of X-axis.
        \param [in] padH - an additional zero padding of input image at the end of Y-axis.
        \param [in] padW - an additional zero padding of input image at the end of X-axis.
        \param [out] dst - a pointer to the output array with converted image.
        \param [in] dstStride - a stride of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel3x3Block3x3SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
        size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel3x3Block3x3SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        \short This function is used for output image conversion in Winograd F(3x3,3x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcStride - a stride of input image.
        \param [out] dst - a pointer to the output image.
        \param [in] dstChannels - a number of output channels.
        \param [in] dstHeight - a height of output image.
        \param [in] dstWidth - a width of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel3x3Block3x3SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel3x3Block4x4SetFilter(const float * src, size_t size, float * dst, SimdBool trans);

        \short This function is used for filter conversion in Winograd F(4x4,3x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input 32-bit float array with filter weights.
        \param [in] size - (number of input channels)*(number of output channels).
        \param [out] dst - a pointer to the output 32-bit float array with filter weights.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel3x3Block4x4SetFilter(const float * src, size_t size, float * dst, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel3x3Block4x4SetInput(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t padY, size_t padX, size_t padH, size_t padW, float * dst, size_t dstStride, SimdBool trans);

        \short This function is used for input image conversion in Winograd F(4x4,3x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcChannels - a number of input channels.
        \param [in] srcHeight - a height of input image.
        \param [in] srcWidth - a width of input image.
        \param [in] padY - an additional zero padding of input image at the beginning of Y-axis.
        \param [in] padX - an additional zero padding of input image at the beginning of X-axis.
        \param [in] padH - an additional zero padding of input image at the end of Y-axis.
        \param [in] padW - an additional zero padding of input image at the end of X-axis.
        \param [out] dst - a pointer to the output array with converted image.
        \param [in] dstStride - a stride of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel3x3Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
        size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

    /*! @ingroup synet_winograd

        \fn void SimdWinogradKernel3x3Block4x4SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        \short This function is used for output image conversion in Winograd F(4x4,3x3) convolution algorithm.

        \note This function is used in <a href="http://github.com/ermig1979/Synet">Synet Framework</a>.

        \param [in] src - a pointer to the input image.
        \param [in] srcStride - a stride of input image.
        \param [out] dst - a pointer to the output image.
        \param [in] dstChannels - a number of output channels.
        \param [in] dstHeight - a height of output image.
        \param [in] dstWidth - a width of output image.
        \param [in] trans - a flag of transposed data.
    */
    SIMD_API void SimdWinogradKernel3x3Block4x4SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

    /*! @ingroup yuv_conversion

        \fn void SimdYuva420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride);

        \short Converts YUVA420P image to 32-bit BGRA image.

        The input Y, A and output BGRA images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function has a C++ wrappers: Simd::Yuva420pToBgra(const View<A>& y, const View<A>& u, const View<A>& v, const View<A> & a, View<A>& bgra).

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] a - a pointer to pixels data of input 8-bit image with alpha channel.
        \param [in] aStride - a row size of the a image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] bgra - a pointer to pixels data of output 32-bit BGRA image.
        \param [in] bgraStride - a row size of the bgra image.
    */
    SIMD_API void SimdYuva420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, 
        const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride);

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

        \fn void SimdYuv420pToBgraV2(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType);

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
        \param [in] yuvType - a type of input YUV image (see descriprion of ::SimdYuvType).
    */
    SIMD_API void SimdYuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
        size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType);

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

        \fn void SimdYuv444pToBgraV2(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType);

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
        \param [in] yuvType - a type of input YUV image (see descriprion of ::SimdYuvType).
    */
    SIMD_API void SimdYuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
        size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType);

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

    /*! @ingroup yuv_conversion

        \fn void SimdYuv420pToRgb(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * rgb, size_t rgbStride);

        \short Converts YUV420P image to 24-bit RGB image.

        The input Y and output RGB images must have the same width and height.
        The input U and V images must have the same width and height (half size relative to Y component).

        \note This function has a C++ wrappers: Simd::Yuv420pToRgb(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& rgb);

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] rgb - a pointer to pixels data of output 24-bit RGB image.
        \param [in] rgbStride - a row size of the rgb image.
    */
    SIMD_API void SimdYuv420pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
        size_t width, size_t height, uint8_t* rgb, size_t rgbStride);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv422pToRgb(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * rgb, size_t rgbStride);

        \short Converts YUV422P image to 24-bit RGB image.

        The input Y and output RGB images must have the same width and height.
        The input U and V images must have the same width and height (their width is equal to half width of Y component).

        \note This function has a C++ wrappers: Simd::Yuv422pToRgb(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& rgb);

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] rgb - a pointer to pixels data of output 24-bit RGB image.
        \param [in] rgbStride - a row size of the rgb image.
    */
    SIMD_API void SimdYuv422pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
        size_t width, size_t height, uint8_t* rgb, size_t rgbStride);

    /*! @ingroup yuv_conversion

        \fn void SimdYuv444pToRgb(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * rgb, size_t rgbStride);

        \short Converts YUV444P image to 24-bit RGB image.

        The input Y, U, V and output RGB images must have the same width and height.

        \note This function has a C++ wrappers: Simd::Yuv444pToRgb(const View<A>& y, const View<A>& u, const View<A>& v, View<A>& rgb);

        \param [in] y - a pointer to pixels data of input 8-bit image with Y color plane.
        \param [in] yStride - a row size of the y image.
        \param [in] u - a pointer to pixels data of input 8-bit image with U color plane.
        \param [in] uStride - a row size of the u image.
        \param [in] v - a pointer to pixels data of input 8-bit image with V color plane.
        \param [in] vStride - a row size of the v image.
        \param [in] width - an image width.
        \param [in] height - an image height.
        \param [out] rgb - a pointer to pixels data of output 24-bit RGB image.
        \param [in] rgbStride - a row size of the rgb image.
    */
    SIMD_API void SimdYuv444pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
        size_t width, size_t height, uint8_t* rgb, size_t rgbStride);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif//__SimdLib_h__
