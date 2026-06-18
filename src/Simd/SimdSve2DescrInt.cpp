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

#include "Simd/SimdDescrInt.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        DescrInt::DescrInt(size_t size, size_t depth)
#ifdef SIMD_NEON_ENABLE
            : Neon::DescrInt(size, depth)
#else
            : Base::DescrInt(size, depth)
#endif
        {
            _encode32f = GetEncode32f(_depth);
            _encode16f = GetEncode16f(_depth);
            _decode32f = GetDecode32f(_depth);

            _cosineDistance = GetCosineDistance(_depth);
            _macroCosineDistancesDirect = GetMacroCosineDistancesDirect(_depth);
            _microMd = 4;
            _microNd = 1;

            _unpackNormA = GetUnpackNorm(false);
            _unpackNormB = GetUnpackNorm(true);
            _unpackDataA = GetUnpackData(_depth);
            _unpackDataB = GetUnpackData(_depth);
            _macroCosineDistancesUnpack = GetMacroCosineDistancesUnpack(_depth);
            _unpSize = _size;
            _microMu = 4;
            _microNu = 1;
        }

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth)
        {
            if (!Base::DescrInt::Valid(size, depth))
                return NULL;
            return new Sve2::DescrInt(size, depth);
        }
    }
#endif// SIMD_SVE2_ENABLE
}
