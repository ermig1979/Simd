/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::UnpackDataPtr GetUnpackData(size_t depth, bool transpose)
        {
            switch (depth)
            {
            //case 4: return transpose ? UnpackDataB<4> : UnpackDataA<4>;
            //case 5: return transpose ? UnpackDataB<5> : UnpackDataA<5>;
            //case 6: return transpose ? UnpackDataB<6> : UnpackDataA<6>;
            //case 7: return transpose ? UnpackDataB<7> : UnpackDataA<7>;
            //case 8: return transpose ? UnpackDataB8 : UnpackDataA8;
            default: return NULL;
            }
        }

        Base::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth)
        {
            return NULL;// depth == 8 ? MacroCorrelation16 : MacroCorrelation8;
        }
    }
#endif// SIMD_NEON_ENABLE
}
