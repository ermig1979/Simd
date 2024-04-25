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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_AVX512VNNI_ENABLE    
    namespace Avx512vnni
    {
        DescrInt::DescrInt(size_t size, size_t depth)
            : Avx512bw::DescrInt(size, depth)
        {
            _cosineDistance = GetCosineDistance(_depth);
            _macroCosineDistancesDirect = GetMacroCosineDistancesDirect(_depth);
            _microMd = 4;
            _microNd = 4;

            if (_depth != 8)
            {
                _macroCosineDistancesUnpack = GetMacroCosineDistancesUnpack(_depth);
                _microMu = 12;
                _microNu = 32;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth)
        {
            if (!Base::DescrInt::Valid(size, depth))
                return NULL;
            return new Avx512vnni::DescrInt(size, depth);
        }
    }
#endif
}
