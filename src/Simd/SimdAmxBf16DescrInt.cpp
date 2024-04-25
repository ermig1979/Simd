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
#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
        DescrInt::DescrInt(size_t size, size_t depth)
            : Avx512vnni::DescrInt(size, depth)
        {
            if (_depth >= 5 && _depth <= 7)
            {
                _cosineDistance = GetCosineDistance(_depth);
                _macroCosineDistancesDirect = GetMacroCosineDistancesDirect(_depth);
            }

            //if (_depth != 8)
            //{
                _unpackDataA = GetUnpackData(_depth, false);
                _unpackDataB = GetUnpackData(_depth, true);
                _macroCosineDistancesUnpack = GetMacroCosineDistancesUnpack(_depth);
                _unpSize = AlignHi(_size, 64);
                _microMu = 32;
                _microNu = 32;
            //}
        }

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth)
        {
            if (!Base::DescrInt::Valid(size, depth))
                return NULL;
            return new AmxBf16::DescrInt(size, depth);
        }
    }
#endif
}
