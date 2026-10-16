#include "Simd/SimdArray.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdLib.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace
{
    unsigned Decode(uint8_t value)
    {
        return static_cast<uint8_t>(value - 32);
    }

    unsigned Dimension(const uint8_t* bytes)
    {
        return Decode(bytes[0]) % 16 + 16 * (Decode(bytes[1]) % 16);
    }

    bool Reserve(size_t rows, size_t stride, size_t& remaining, size_t& count)
    {
        if (stride == 0 || rows > remaining / stride)
            return false;
        count = rows * stride;
        remaining -= count;
        return true;
    }

    void Initialize(Simd::Array32f& array, const uint8_t* payload, size_t payloadSize, size_t salt)
    {
        for (size_t index = 0; index < array.size; ++index)
        {
            const int value = static_cast<int>((payload[(index + salt) % payloadSize] + index % 129 + salt) % 129);
            array[index] = (value - 64) / 64.0f;
        }
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (size < 13)
        return 0;

    const bool transpose = Decode(data[0]) % 2 != 0;
    const size_t rows = 1 + Dimension(data + 1) % 128;
    const size_t columns = 1 + Dimension(data + 3) % 128;
    const size_t depth = transpose ? 64 + Dimension(data + 5) % 193 : 1 + Dimension(data + 5);
    const size_t strideA = depth + Decode(data[7]) % 8;
    const size_t strideB = (transpose ? depth : columns) + Decode(data[8]) % 8;
    const size_t strideC = columns + Decode(data[9]) % 8;
    const float scales[] = { 0.0f, 1.0f, -1.0f, 0.5f, 1.5f };
    const float alpha = scales[Decode(data[10]) % 5];
    const float beta = scales[Decode(data[11]) % 5];

    if (rows > size_t(1048576) / columns / depth)
        return 0;
    size_t remaining = size_t(1048576) / sizeof(float);
    size_t countA = 0, countB = 0, countC = 0;
    if (!Reserve(rows, strideA, remaining, countA) ||
        !Reserve(transpose ? columns : depth, strideB, remaining, countB) ||
        !Reserve(rows, strideC, remaining, countC) ||
        !Reserve(rows, strideC, remaining, countC) ||
        !Reserve(rows, strideC, remaining, countC))
        return 0;

    Simd::Array32f matrixA(countA);
    Simd::Array32f matrixB(countB);
    Simd::Array32f initialC(countC);
    Simd::Array32f reference(countC);
    Simd::Array32f actual(countC);
    if (!matrixA.data || !matrixB.data || !initialC.data || !reference.data || !actual.data)
        return 0;
    const size_t payloadSize = std::min(size - 12, size_t(8192));
    Initialize(matrixA, data + 12, payloadSize, 0);
    Initialize(matrixB, data + 12, payloadSize, 43);
    Initialize(initialC, data + 12, payloadSize, 86);
    std::copy(initialC.data, initialC.data + countC, reference.data);
    std::copy(initialC.data, initialC.data + countC, actual.data);

    if (transpose)
    {
        Simd::Base::Gemm32fNT(rows, columns, depth, &alpha, matrixA.data, strideA,
            matrixB.data, strideB, &beta, reference.data, strideC);
        SimdGemm32fNT(rows, columns, depth, &alpha, matrixA.data, strideA,
            matrixB.data, strideB, &beta, actual.data, strideC);
    }
    else
    {
        Simd::Base::Gemm32fNN(rows, columns, depth, &alpha, matrixA.data, strideA,
            matrixB.data, strideB, &beta, reference.data, strideC);
        SimdGemm32fNN(rows, columns, depth, &alpha, matrixA.data, strideA,
            matrixB.data, strideB, &beta, actual.data, strideC);
    }
    return 0;
}
