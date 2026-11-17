/*
 * Fuzzer for the Synet float convolution runtime (SimdSynetConvolution32f*).
 *
 * Threat model: downstream consumers (Synet inference) run convolution stages
 * whose shapes, strides, dilations, paddings, groups and activation types come
 * from untrusted model files, with tensors holding untrusted data. A
 * memory-safety violation (OOB read/write) or a wrong-result dispatch bug on
 * such attacker-influenced parameters is a real bug.
 *
 * Design (mirrors the project's existing fuzzers, e.g. simd_pooling_fuzzer):
 *  - budgeted dimensions decoded from fuzz bytes (buffers capped well below
 *    8 MiB so slow units and OOMs stay out of the picture);
 *  - deterministic payload initialization (no uninitialized memory);
 *  - differential oracle: the same valid convolution runs through the
 *    dispatching public API (SimdSynetConvolution32fInit) and through the
 *    reference implementation (Simd::Base::SynetConvolution32fInit); results
 *    must agree within float reassociation tolerance and stay finite;
 *  - method selection (GemmNN/GemmNT/DirectNchw/NhwcDirect/NhwcDepthwise/
 *    DepthwiseDotProduct/NhwcGroupedBlock1x2/Winograd2/Winograd3) is decided
 *    by the library from the fuzzed parameters (Preferable() gates in
 *    src/Simd/SimdBaseSynetConvolution32f*.cpp), so one harness sweeps all
 *    convolution method classes including the dark Winograd paths.
 */
#include "Simd/SimdArray.h"
#include "Simd/SimdLib.h"
#include "Simd/SimdSynetConvolution32f.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace
{
    unsigned Decode(uint8_t value)
    {
        return static_cast<unsigned>(uint8_t(value - 32));
    }

    size_t Div(size_t value, size_t divisor)
    {
        return divisor ? value % divisor : 0;
    }
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    const size_t header = 19;
    if (size <= header)
        return 0;

    const size_t kernels[] = { 1, 2, 3, 5, 7 };
    const size_t strides[] = { 1, 1, 2, 3 };

    const size_t batch = 1 + Div(Decode(data[0]), 2);
    const SimdTensorFormatType format = Div(Decode(data[1]), 2) ? SimdTensorFormatNhwc : SimdTensorFormatNchw;
    const size_t kernelY = kernels[Div(Decode(data[2]), 5)];
    const size_t kernelX = kernels[Div(Decode(data[3]), 5)];
    const size_t strideY = strides[Div(Decode(data[4]), 4)];
    const size_t strideX = strides[Div(Decode(data[5]), 4)];
    const size_t dilationY = 1 + Div(Decode(data[6]), 2);
    const size_t dilationX = 1 + Div(Decode(data[7]), 2);
    const size_t srcC = 1 + Div(Decode(data[8]), 64);
    const size_t dstC = 1 + Div(Decode(data[9]), 64);
    const size_t srcH = 1 + Div(Decode(data[10]), 56);
    const size_t srcW = 1 + Div(Decode(data[11]), 56);
    const size_t padY = Div(Decode(data[12]), 3);
    const size_t padX = Div(Decode(data[13]), 3);
    const size_t padH = Div(Decode(data[14]), 3);
    const size_t padW = Div(Decode(data[15]), 3);
    const size_t group = 1 + Div(Decode(data[16]), 4);
    const SimdConvolutionActivationType activation =
        (SimdConvolutionActivationType)Div(Decode(data[17]), 11);
    SimdConvolutionParameters conv;
    memset(&conv, 0, sizeof(conv));
    conv.srcC = srcC;
    conv.srcH = srcH;
    conv.srcW = srcW;
    conv.srcT = SimdTensorData32f;
    conv.srcF = format;
    conv.dstT = SimdTensorData32f;
    conv.dstF = format;
    conv.kernelY = kernelY;
    conv.kernelX = kernelX;
    conv.dilationY = dilationY;
    conv.dilationX = dilationX;
    conv.strideY = strideY;
    conv.strideX = strideX;
    conv.padY = padY;
    conv.padX = padX;
    conv.padH = padH;
    conv.padW = padW;
    conv.group = group;
    conv.activation = activation;

    if ((srcC % group != 0) || (dstC % group != 0))
        return 0;

    const size_t extentY = srcH + padY + padH;
    const size_t extentX = srcW + padX + padW;
    const size_t windowY = dilationY * (kernelY - 1) + 1;
    const size_t windowX = dilationX * (kernelX - 1) + 1;
    if (windowY > extentY || windowX > extentX)
        return 0;
    const size_t dstH = (extentY - windowY) / strideY + 1;
    const size_t dstW = (extentX - windowX) / strideX + 1;
    if (dstH == 0 || dstW == 0)
        return 0;
    conv.dstH = dstH;
    conv.dstW = dstW;

    const size_t sizeS = batch * srcC * srcH * srcW;
    const size_t sizeD = batch * dstC * dstH * dstW;
    const size_t sizeW = kernelY * kernelX * (srcC / group) * dstC;
    const size_t budget = size_t(1024) * size_t(1024) / sizeof(float) / 2;
    if (sizeS == 0 || sizeD == 0 || sizeW == 0 ||
        sizeS > budget || sizeD > budget || sizeW > budget)
        return 0;

    void* referenceContext = Simd::Base::SynetConvolution32fInit(batch, &conv);
    void* actualContext = SimdSynetConvolution32fInit(batch, &conv);
    if ((referenceContext == NULL) != (actualContext == NULL))
    {
        SimdRelease(referenceContext);
        SimdRelease(actualContext);
        std::abort();
    }
    if (referenceContext == NULL)
        return 0;

    const size_t bufSize = std::max(
        SimdSynetConvolution32fExternalBufferSize(referenceContext),
        SimdSynetConvolution32fExternalBufferSize(actualContext));
    if (bufSize == 0 || bufSize > 4 * budget)
    {
        SimdRelease(referenceContext);
        SimdRelease(actualContext);
        return 0;
    }

    Simd::Array32f src(sizeS);
    Simd::Array32f weight(sizeW);
    Simd::Array32f bias(dstC);
    Simd::Array32f params(dstC);
    Simd::Array32f bufRef(bufSize);
    Simd::Array32f bufAct(bufSize);
    Simd::Array32f dstRef(sizeD);
    Simd::Array32f dstAct(sizeD);
    if (!src.data || !weight.data || !bias.data || !params.data ||
        !bufRef.data || !bufAct.data || !dstRef.data || !dstAct.data)
    {
        SimdRelease(referenceContext);
        SimdRelease(actualContext);
        return 0;
    }

    const uint8_t* payload = data + header;
    const size_t payloadSize = std::min(size - header, size_t(4096));
    for (size_t index = 0; index < sizeS; ++index)
        src[index] = (static_cast<int>(payload[index % payloadSize] % 129) - 64) / 64.0f;
    for (size_t index = 0; index < sizeW; ++index)
        weight[index] = (static_cast<int>(payload[(index + 71) % payloadSize] % 129) - 64) / 256.0f;
    for (size_t index = 0; index < dstC; ++index)
    {
        bias[index] = (static_cast<int>(payload[(index + 131) % payloadSize] % 129) - 64) / 64.0f;
        params[index] = (static_cast<int>(payload[(index + 193) % payloadSize] % 129)) / 128.0f + 0.1f;
    }
    if (dstC > 1)
    {
        if (activation == SimdConvolutionActivationHswish)
        {
            params[0] = 3.0f;
            params[1] = 1.0f / 6.0f;
        }
        else if (activation == SimdConvolutionActivationMish)
            params[0] = 20.0f;
        else if (activation == SimdConvolutionActivationHardSigmoid)
        {
            params[0] = 1.0f / 6.0f;
            params[1] = 0.5f;
        }
        else if (activation == SimdConvolutionActivationPrelu || activation == SimdConvolutionActivationLeakyRelu)
            params[0] = std::max(0.01f, params[0]);
    }
    for (size_t index = 0; index < bufSize; ++index)
    {
        bufRef[index] = 1.0f;
        bufAct[index] = 1.0f;
    }
    for (size_t index = 0; index < sizeD; ++index)
    {
        dstRef[index] = 0.1f;
        dstAct[index] = 0.1f;
    }

    SimdSynetConvolution32fSetParams(referenceContext, weight.data, NULL, bias.data, params.data);
    SimdSynetConvolution32fSetParams(actualContext, weight.data, NULL, bias.data, params.data);

    SimdSynetConvolution32fForward(referenceContext, src.data, bufRef.data, dstRef.data);
    SimdSynetConvolution32fForward(actualContext, src.data, bufAct.data, dstAct.data);

    for (size_t index = 0; index < sizeD; ++index)
    {
        const float reference = dstRef[index];
        const float actual = dstAct[index];
        if (!std::isfinite(reference) || !std::isfinite(actual))
            std::abort();
        const float difference = std::fabs(reference - actual);
        if (difference > 0.05f + 0.002f * std::fabs(reference))
            std::abort();
    }

    SimdRelease(referenceContext);
    SimdRelease(actualContext);
    return 0;
}
