/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#ifndef __SimdMemoryStream_h__
#define __SimdMemoryStream_h__

#include "Simd/SimdMemory.h"

namespace Simd
{
    class OutputMemoryStream
    {
        const size_t CAPACITY_MIN = 4096;

        uint8_t * _data;
        size_t _pos, _size, _capacity;

        SIMD_INLINE void Reset(bool owner)
        {
            if (_data && owner)
                Free(_data);
            _data = NULL;
            _pos = 0;
            _size = 0;
            _capacity = 0;
        }

    public:
        SIMD_INLINE OutputMemoryStream()
            : _data(NULL)
            , _pos(0)
            , _size(0)
            , _capacity(0)
        {
        }

        SIMD_INLINE ~OutputMemoryStream()
        {
            Reset(true);
        }

        SIMD_INLINE void Seek(size_t pos)
        {
            _pos = pos;
            _size = Max(_size, _pos);
            Reserve(_pos);
        }

        SIMD_INLINE size_t Pos() const
        {
            return _pos;
        }

        SIMD_INLINE size_t Size() const
        {
            return _size;
        }

        SIMD_INLINE uint8_t* Data()
        {
            return _data;
        }

        SIMD_INLINE const uint8_t * Data() const
        {
            return _data;
        }

        SIMD_INLINE void Write(const void * data, size_t size)
        {
            Reserve(_pos + size);
            memcpy(_data + _pos, data, size);
            _pos += size;
            _size = Max(_size, _pos);
        }

        SIMD_INLINE uint8_t* Release(size_t* size = NULL)
        {
            uint8_t* data = _data;
            if(size)
                *size = _size;
            Reset(false);
            return data;
        }

        SIMD_INLINE void Reserve(size_t size)
        {
            if (size > _capacity)
            {
                size_t capacity = Max(CAPACITY_MIN, Max(_capacity * 2, AlignHi(size, CAPACITY_MIN)));
                uint8_t* data = (uint8_t*)Allocate(capacity, SIMD_ALIGN);
                if (_data)
                {
                    memcpy(data, _data, _size);
                    Free(_data);
                }
                _data = data;
                _capacity = capacity;
            }
        }
    };
}

#endif//__SimdMemoryStream_h__
