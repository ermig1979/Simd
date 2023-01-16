/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdTransform.h"
#include "Simd/SimdSynetPermute.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<class T> void Permute2(const uint8_t* src, const Base::Shape& shape, const Base::Shape& stride, uint8_t* dst)
        {
            static ImageTransforms transforms = ImageTransforms();

            transforms.TransformImage(src, shape[0] * sizeof(T), shape[0], shape[1], sizeof(T), SimdTransformTransposeRotate0, dst, shape[1] * sizeof(T));
        }

        //-------------------------------------------------------------------------------------------------

        SynetPermute::SynetPermute(const Base::PermuteParam& param)
            : Avx2::SynetPermute(param)
        {
            if (_count == 2)
            {            
                switch (_param.type)
                {
                case SimdTensorData32f:
                case SimdTensorData32i:
                    _permute = Permute2<uint32_t>;
                    break;
                case SimdTensorData8i:
                case SimdTensorData8u:
                    _permute = Permute2<uint8_t>;
                    break;
                case SimdTensorData16b:
                case SimdTensorData16f:
                    _permute = Permute2<uint16_t>;
                    break;
                default:
                    assert(0);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetPermuteInit(const size_t* shape, const size_t* order, size_t count, SimdTensorDataType type)
        {
            Base::PermuteParam param(shape, order, count, type, A);
            if (!param.Valid())
                return NULL;
            return new SynetPermute(param);
        }
    }
#endif
}
