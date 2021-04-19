/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
    class InputMemoryStream
    {
        const uint8_t* _data;
        size_t _pos, _size;
    public:
        SIMD_INLINE InputMemoryStream(const uint8_t* data = NULL, size_t size = 0)
        {
            Init(data, size);
        }

        SIMD_INLINE void Init(const uint8_t* data, size_t size)
        {
            _pos = 0;
            _data = data;
            _size = size;
        }

        SIMD_INLINE bool Seek(size_t pos)
        {
            if (pos <= _size)
            {
                _pos = pos;
                return true;
            }
            return false;
        }

        SIMD_INLINE size_t Size() const
        {
            return _size;
        }

        SIMD_INLINE const uint8_t* Data() const
        {
            return _data;
        }

        SIMD_INLINE size_t Pos() const
        {
            return _pos;
        }
        
        SIMD_INLINE size_t Read(size_t size, void* data)
        {
            size = Min(_size - _pos, size);
            memcpy(data, _data + _pos, size);
            _pos += size;
            return size;
        }

        template <class Value> SIMD_INLINE bool Read(Value & value)
        {
            return Read(sizeof(Value), &value) == sizeof(Value);
        }

        template<class Unsigned> SIMD_INLINE bool ReadUnsigned(Unsigned& value)
        {
            if (!SkipGap())
                return false;
            value = 0;
            while (!IsGap(_data[_pos]) && _pos < _size)
            {
                if (_data[_pos] >= '0' && _data[_pos] <= '9')
                    value = value * 10 + Unsigned(_data[_pos] - '0');
                else
                    return false;
                _pos++;
            }
            return true;
        }

        SIMD_INLINE bool Skip(size_t size)
        {
            if (_pos + size < _size)
            {
                _pos += size;
                return true;
            }
            return false;
        }

        SIMD_INLINE bool SkipValue(uint8_t value)
        {
            while (_data[_pos] == value && _pos < _size)
                _pos++;
            return _pos < _size;
        }

        SIMD_INLINE bool SkipNotGap()
        {
            while (!IsGap(_data[_pos]) && _pos < _size)
                _pos++;
            return _pos < _size;
        }        
        
        SIMD_INLINE bool SkipGap()
        {
            while (IsGap(_data[_pos]) && _pos < _size)
                _pos++;
            return _pos < _size;
        }

        static SIMD_INLINE bool IsGap(uint8_t value)
        {
            return value == ' ' || value == '\t' || value == '\n' || value == '\r';
        }
    };

    //-------------------------------------------------------------------------

    class OutputMemoryStream
    {
        const size_t CAPACITY_MIN = 4096;

        uint8_t * _data;
        size_t _pos, _size, _capacity, _bitCount, _bitBuffer;

        SIMD_INLINE void Reset(bool owner)
        {
            if (_data && owner)
                Free(_data);
            _data = NULL;
            _pos = 0;
            _size = 0;
            _capacity = 0;
            _bitBuffer = 0;
            _bitCount = 0;
        }

    public:
        SIMD_INLINE OutputMemoryStream()
        {
            Reset(false);
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

        SIMD_INLINE uint8_t* Current()
        {
            return _data + _pos;
        }

        SIMD_INLINE const uint8_t* Current() const
        {
            return _data + _pos;
        }

        SIMD_INLINE void Write(const void * data, size_t size)
        {
            Reserve(_pos + size);
            memcpy(_data + _pos, data, size);
            _pos += size;
            _size = Max(_size, _pos);
        }

        template <class Value> SIMD_INLINE void Write(const Value& value)
        {
            Write(&value, sizeof(Value));
        }

        SIMD_INLINE void Write8u(uint8_t value)
        {
#if 1
            Reserve(_pos + 1);
            _data[_pos++] = value;
            _size = Max(_size, _pos);
#else
            Write(&value, 1);
#endif
        }

        SIMD_INLINE void WriteBe32(const uint32_t & value)
        {
#if defined(SIMD_BIG_ENDIAN)
            Write<uint32_t>(value);
#else
            Write<uint32_t>(
                (value & 0x000000FF) << 24 | 
                (value & 0x0000FF00) << 8 |
                (value & 0x00FF0000) >> 8 | 
                (value & 0xFF000000) >> 24);
#endif
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

        SIMD_INLINE void WriteBits(const size_t bits, size_t count)
        {
            _bitBuffer |= (bits) << _bitCount;
            _bitCount += count;
            FlushBits(false);
        }

        SIMD_INLINE void FlushBits(bool tail)
        {
            while (_bitCount >= 8)
            {
                Write(&_bitBuffer, 1);
                _bitBuffer >>= 8;
                _bitCount -= 8;
            }
            if (tail && _bitCount)
            {
                Write(&_bitBuffer, 1);
                _bitBuffer = 0;
                _bitCount = 0;
            }
        }

        SIMD_INLINE size_t& BitBuffer()
        {
            return _bitBuffer;
        }

        SIMD_INLINE size_t& BitCount()
        {
            return _bitCount;
        }
    };
}

#endif//__SimdMemoryStream_h__
