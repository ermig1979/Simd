/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
        template <class D, class I, int check> void GatherElements(const uint8_t* src8, size_t srcOuter, size_t srcCount, size_t srcInner, const uint8_t* idx8, size_t idxCount, uint8_t* dst8)
        {
            const D* src = (const D*)src8;
            const I* idx = (I*)idx8;
            D* dst = (D*)dst8;
            if (srcInner == 1)
            {
                for (size_t o = 0; o < srcOuter; ++o)
                {
                    for (size_t c = 0; c < idxCount; ++c)
                    {
                        I ic = idx[c];
                        if (check)
                        {
                            if (ic < 0)
                                ic += I(srcCount);
                        }
                        dst[c] = src[ic];
                    }
                    src += srcCount;
                    idx += idxCount;
                    dst += idxCount;
                }
            }
            else
            {
                for (size_t o = 0; o < srcOuter; ++o)
                {
                    for (size_t c = 0; c < idxCount; ++c)
                    {
                        for (size_t i = 0; i < srcInner; ++i)
                        {
                            I ii = idx[i];
                            if (check)
                            {
                                if (ii < 0)
                                    ii += I(srcCount);
                            }
                            dst[i] = src[ii * srcInner + i];
                        }
                        idx += srcInner;
                        dst += srcInner;
                    }
                    src += srcCount * srcInner;
                }
            }
        }

        template <class D, class I> SynetGatherElements::GatherElementsPtr GetGatherElements(SimdBool c)
        {
            return c ? GatherElements<D, I, 0> : GatherElements<D, I, 1>;
        }

        template <class D> SynetGatherElements::GatherElementsPtr GetGatherElements(SimdTensorDataType i, SimdBool c)
        {
            switch (i)
            {
            case SimdTensorData32i: return GetGatherElements<D, int32_t>(c);
            case SimdTensorData64i: return GetGatherElements<D, int64_t>(c);
            default:
                return NULL;
            }
        }

        SynetGatherElements::GatherElementsPtr GetGatherElements(SimdTensorDataType d, SimdTensorDataType i, SimdBool c)
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
            _gatherElements = GetGatherElements(p.dataType, p.indexType, p.indexConst);
        }

        size_t SynetGatherElements::InternalBufferSize() const
        {
            return _idx32i.RawSize();
        }

        void SynetGatherElements::SetIndex(const uint8_t* idx)
        {
            const GatherElementsParam& p = _param;
            if (p.indexType == SimdTensorData32i)
            {
                _idx32i.Resize(p.srcOuter * p.idxCount * p.srcInner);
                for (size_t i = 0; i < _idx32i.size; ++i)
                {
                    int32_t val = ((int32_t*)idx)[i];
                    if (val < 0)
                        val += int32_t(p.srcCount);
                    _idx32i[i] = val;
                }
            }
            else if (p.indexType == SimdTensorData64i)
            {
                _idx32i.Resize(p.srcOuter * p.idxCount * p.srcInner);
                for (size_t i = 0; i < _idx32i.size; ++i)
                {
                    int32_t val = int32_t(((int64_t*)idx)[i]);
                    if (val < 0)
                        val += int32_t(p.srcCount);
                    _idx32i[i] = val;
                }
            }
        }
       
        void SynetGatherElements::Forward(const uint8_t* src, const uint8_t* idx, uint8_t* dst)
        {
            const GatherElementsParam& p = _param;
            if (p.indexConst)
            {
                if(_idx32i.data) 
                    _gatherElements(src, p.srcOuter, p.srcCount, p.srcInner, (const uint8_t * )_idx32i.data, p.idxCount, dst);
            }
            else
                _gatherElements(src, p.srcOuter, p.srcCount, p.srcInner, idx, p.idxCount, dst);
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetGatherElementsInit(SimdTensorDataType dataType, SimdTensorDataType indexType, SimdBool indexConst, size_t srcOuter, size_t srcCount, size_t srcInner, size_t idxCount)
        {
            GatherElementsParam param(dataType, indexType, indexConst, srcOuter, srcCount, srcInner, idxCount);
            if (!param.Valid())
                return NULL;
            return new SynetGatherElements(param);
        }
    }
#endif
}
