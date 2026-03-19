/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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

#include "Simd/SimdSynetGatherElements.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        GatherElementsParam::GatherElementsParam(SimdTensorDataType dt, SimdTensorDataType it, SimdBool iC, size_t iu, const size_t* o, size_t os, size_t sc, size_t i, size_t ic)
            : dataType(dt)
            , indexType(it)
            , indexConst(iC)
            , indexUsers(iu)
            , outer(o, o + os)
            , srcCount(sc)
            , inner(i)
            , idxCount(ic)
        {
        }

        bool GatherElementsParam::Valid() const
        {
            if (dataType != SimdTensorData32f && dataType != SimdTensorData16b && dataType != SimdTensorData8u)
                return false;
            if (indexType != SimdTensorData64i && indexType != SimdTensorData32i)
                return false;
            return true;
        }

        //-------------------------------------------------------------------------------------------------


        template <class D, class I, int check> void GatherElements(const uint8_t* src8, size_t batch, size_t outer, size_t srcCount, size_t inner, const uint8_t* idx8, size_t idxCount, uint8_t* dst8)
        {
            const D* src = (const D*)src8;
            const I* idx = (I*)idx8;
            D* dst = (D*)dst8;
            if (inner == 1)
            {
                for (size_t b = 0; b < batch; ++b)
                {
                    const I* pi = idx;
                    for (size_t o = 0; o < outer; ++o)
                    {
                        for (size_t c = 0; c < idxCount; ++c)
                        {
                            I ic = pi[c];
                            if (check)
                            {
                                if (ic < 0)
                                    ic += I(srcCount);
                            }
                            dst[c] = src[ic];
                        }
                        src += srcCount;
                        pi += idxCount;
                        dst += idxCount;
                    }
                }
            }
            else
            {
                for (size_t b = 0; b < batch; ++b)
                {
                    const I* pi = idx;
                    for (size_t o = 0; o < outer; ++o)
                    {
                        for (size_t c = 0; c < idxCount; ++c)
                        {
                            for (size_t i = 0; i < inner; ++i)
                            {
                                I ii = pi[i];
                                if (check)
                                {
                                    if (ii < 0)
                                        ii += I(srcCount);
                                }
                                dst[i] = src[ii * inner + i];
                            }
                            pi += inner;
                            dst += inner;
                        }
                        src += srcCount * inner;
                    }
                }
            }
        }

        template <class D, class I> SynetGatherElements::GatherElementsPtr GetGatherElements(int c)
        {
            return c ? GatherElements<D, I, 1> : GatherElements<D, I, 0>;
        }

        template <class D> SynetGatherElements::GatherElementsPtr GetGatherElements(SimdTensorDataType i, int c)
        {
            switch (i)
            {
            case SimdTensorData32i: return GetGatherElements<D, int32_t>(c);
            case SimdTensorData64i: return GetGatherElements<D, int64_t>(c);
            default:
                return NULL;
            }
        }

        SynetGatherElements::GatherElementsPtr GetGatherElements(SimdTensorDataType d, SimdTensorDataType i, int c)
        {
            switch (d)
            {
            case SimdTensorData8u: return GetGatherElements<uint8_t>(i, c);
            case SimdTensorData16b: return GetGatherElements<uint16_t>(i, c);
            case SimdTensorData32f: return GetGatherElements<uint32_t>(i, c);
            default:
                return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetGatherElements::SynetGatherElements(const GatherElementsParam& p)
            : _param(p)
            , _gatherElements(NULL)
        {
            _check = 1;
            _gatherElements = GetGatherElements(p.dataType, p.indexType, SimdTrue);
            _batch = 1, _outer = 1; 
            for (size_t i = 0; i < p.outer.size(); ++i)
                _outer *= p.outer[i];
        }

        size_t SynetGatherElements::InternalBufferSize() const
        {
            return _index.RawSize();
        }

        void SynetGatherElements::SetIndex(const uint8_t* idx8)
        {
            const GatherElementsParam& p = _param;
            if (p.indexConst == SimdFalse)
                return;
            size_t elem = 1;
            if (p.indexType == SimdTensorData32i)
            {
                const int32_t* idx = (const int32_t*)idx8;
                _check = 0;
                for (size_t i = 0, n = _batch * _outer * p.idxCount * p.inner; i < n; ++i)
                {
                    if (idx[i] < 0)
                    {
                        _check = 1;
                        break;
                    }
                }
                _gatherElements = GetGatherElements(p.dataType, p.indexType, _check);
                elem = sizeof(int32_t);
            }
            else if (p.indexType == SimdTensorData64i)
            {
                const int64_t* idx = (const int64_t*)idx8;
                _check = 0;
                for (size_t i = 0, n = _batch * _outer * p.idxCount * p.inner; i < n; ++i)
                {
                    if (idx[i] < 0)
                    {
                        _check = 1;
                        break;
                    }
                }
                _gatherElements = GetGatherElements(p.dataType, p.indexType, _check);
                elem = sizeof(int64_t);
            }
            _batch = 1, _outer = 1;
            for (size_t i = 0; i < p.outer.size(); ++i)
                _outer *= p.outer[i];
            for (size_t i = 0; i < p.outer.size(); ++i)
            {
                if (p.outer[i] == 1)
                    continue;
                size_t batch = 1;
                for (size_t j = 0; j <= i; ++j)
                    batch *= p.outer[j];
                size_t size = p.idxCount * p.inner * elem;
                for (size_t j = i + 1; j < p.outer.size(); ++j)
                    size *= p.outer[j];
                bool equal = true;
                for (size_t b = 1; b < batch && equal; b++)
                    equal = memcmp(idx8, idx8 + b * size, size) == 0;
                if (equal)
                {
                    _batch *= p.outer[i];
                    _outer /= p.outer[i];
                }
                else
                    break;
            }
        }
       
        void SynetGatherElements::Forward(const uint8_t* src, const uint8_t* idx, uint8_t* dst)
        {
            const GatherElementsParam& p = _param;
            if (_index.size)
                _gatherElements(src, _batch, _outer, p.srcCount, p.inner, _index.data, p.idxCount, dst);
            else
                _gatherElements(src, _batch, _outer, p.srcCount, p.inner, idx, p.idxCount, dst);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetGatherElementsInit(SimdTensorDataType dataType, SimdTensorDataType indexType, SimdBool indexConst, size_t indexUsers, const size_t* outer, size_t outerSize, size_t srcCount, size_t inner, size_t idxCount)
        {
            GatherElementsParam param(dataType, indexType, indexConst, indexUsers, outer, outerSize, srcCount, inner, idxCount);
            if (!param.Valid())
                return NULL;
            return new SynetGatherElements(param);
        }
    }
#endif
}
