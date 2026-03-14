/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */
#ifndef __SimdHvx_h__
#define __SimdHvx_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_HVX_ENABLE
    namespace Hvx
    {
        void AbsDifference(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, uint8_t* c, size_t cStride,
            size_t width, size_t height);

        void AbsDifferenceSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, uint64_t* sum);

        void AbsGradientSaturatedSum(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t* dst, size_t dstStride);

        void AddFeatureDifference(const uint8_t* value, size_t valueStride, size_t width, size_t height,
            const uint8_t* lo, size_t loStride, const uint8_t* hi, size_t hiStride,
            uint16_t weight, uint8_t* difference, size_t differenceStride);

        void BgrToGray(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride,
            uint8_t* gray, size_t grayStride);

        void BgrToRgb(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride,
            uint8_t* rgb, size_t rgbStride);

        void FillBgra(uint8_t* dst, size_t stride, size_t width, size_t height,
            uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha);

        void FillPixel(uint8_t* dst, size_t stride, size_t width, size_t height,
            const uint8_t* pixel, size_t pixelSize);

        void OperationBinary8u(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride,
            SimdOperationBinary8uType type);

        void GetStatistic(const uint8_t* src, size_t stride, size_t width, size_t height,
            uint8_t* min, uint8_t* max, uint8_t* average);
    }
#endif
}
#endif
