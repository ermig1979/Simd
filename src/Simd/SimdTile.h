/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#ifndef __SimdTile_h__
#define __SimdTile_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
    struct TileConf
    {
        uint8_t paletteId;
        uint8_t startRow;
        uint8_t reserved[14];
        uint16_t colsb[16];
        uint8_t rows[16];

        SIMD_INLINE TileConf(uint8_t paletteId = 1, uint8_t startRow = 0)
        {
            memset(this, 0, sizeof(TileConf));
            this->paletteId = paletteId;
            this->startRow = startRow;
        }
    };

    union SIMD_ALIGNED(64) TileReg
    {
        int8_t i8[16][64];
        uint8_t u8[16][64];
        int16_t i16[16][32];
        uint16_t u16[16][32];
        int32_t i32[16][16];
        uint32_t u32[16][16];
        float f32[16][16];
    };

    const size_t TileRegCount = 8;

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        void TileLoadConfig(const TileConf* tileConf);

        void TileStoreConfig(TileConf* tileConf);

        void TileZero(int dst);

        void TileLoad(int dst, const void* base, int stride);

        void TileStore(int src, void* base, int stride);

        void TileMatMulBf16(int dst, int a, int b);
    }
#endif

#ifdef SIMD_AMX_ENABLE    
    namespace Amx
    {
    }
#endif
}
#endif
