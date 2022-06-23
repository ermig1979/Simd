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

        void TileRelease();

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

#if defined(SIMD_AMX_EMULATE)

#ifdef _tile_loadconfig
#undef _tile_loadconfig
#endif
#define _tile_loadconfig Simd::Avx512bw::TileLoadConfig

#ifdef _tile_storeconfig
#undef _tile_storeconfig
#endif
#define _tile_storeconfig Simd::Avx512bw::TileStoreConfig

#ifdef _tile_release
#undef _tile_release
#endif
#define _tile_release Simd::Avx512bw::TileRelease

#ifdef _tile_loadd
#undef _tile_loadd
#endif
#define _tile_loadd Simd::Avx512bw::TileLoad

#ifdef _tile_stream_loadd
#undef _tile_stream_loadd
#endif
#define _tile_stream_loadd Simd::Avx512bw::TileLoad

#ifdef _tile_stored
#undef _tile_stored
#endif
#define _tile_stored Simd::Avx512bw::TileStore

#ifdef _tile_zero
#undef _tile_zero
#endif
#define _tile_zero Simd::Avx512bw::TileZero

#ifdef _tile_dpbf16ps
#undef _tile_dpbf16ps
#endif
#define _tile_dpbf16ps Simd::Avx512bw::TileMatMulBf16

#endif

#endif
