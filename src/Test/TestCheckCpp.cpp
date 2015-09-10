/*
* Tests for Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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

/*                                                                                       
* File name   : SimdCheckCpp.cpp                                                       
* Description : This file is needed to verify the C++ API of Simd Library.               
*/

#include "Simd/SimdLib.hpp"
#include "Simd/SimdFrame.hpp"

namespace Test
{
    static void CheckCpp()
    {
        typedef Simd::View<Simd::Allocator> View;

        View v(1, 1, View::Gray8);

		typedef Simd::Frame<Simd::Allocator> Frame;

		Frame fs(2, 2, Frame::Yuv420p);
		Frame fd(2, 2, Frame::Bgr24);
		Simd::Convert(fs, fd);
	}
}


