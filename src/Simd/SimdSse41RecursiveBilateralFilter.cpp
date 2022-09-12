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
#include "Simd/SimdMemory.h"
#include "Simd/SimdRecursiveBilateralFilter.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
//#define SIMD_RBF_SSE2_TEST
#ifdef SIMD_RBF_SSE2_TEST
        namespace RbfSse2
        {
#define RBF_MAX_THREADS 8
#define STAGE_BUFFER_COUNT 3

            class CRBFilterSSE2
            {
                int				m_reserved_width = 0;
                int				m_reserved_height = 0;
                int				m_thread_count = 0;
                bool			m_pipelined = false;

                float			m_sigma_spatial = 0.f;
                float			m_sigma_range = 0.f;
                float			m_inv_alpha_f = 0.f;
                float* m_range_table = nullptr;

                int				m_filter_counter = 0; // used in pipelined mode
                unsigned char* m_stage_buffer[STAGE_BUFFER_COUNT] = { nullptr }; // size width * height * 4, 2nd one null if not pipelined
                float** m_h_line_cache = nullptr; // single line cache for horizontal filter pass, one per thread
                float** m_v_line_cache = nullptr; // if not pipelined mode, this is equal to 'm_h_line_cache'
                unsigned char* m_out_buffer[STAGE_BUFFER_COUNT] = { nullptr }; // used for keeping track of current output buffer in pipelined mode 
                int				m_image_width = 0; // cache of sizes for pipelined mode
                int				m_image_height = 0;
                int				m_image_pitch = 0;

                // core filter functions
                void horizontalFilter(int thread_index, const unsigned char* img_src, unsigned char* img_dst, int width, int height, int pitch);
                void verticalFilter(int thread_index, const unsigned char* img_src, unsigned char* img_dst, int width, int height, int pitch);

            public:

                CRBFilterSSE2();
                ~CRBFilterSSE2();

                // 'sigma_spatial' - unlike the original implementation of Recursive Bilateral Filter, 
                // the value if sigma_spatial is not influence by image width/height.
                // In this implementation, sigma_spatial is assumed over image width 255, height 255
                void setSigma(float sigma_spatial, float sigma_range);

                // Source and destination images are assumed to be 4 component
                // 'width' - maximum image width
                // 'height' - maximum image height
                // 'thread_count' - total thread count to use for each filter stage (horizontal and vertical), recommended thread count = 4
                // 'pipelined' - if true, then horizontal and vertical filter passes are split into separate stages,
                // where each stage uses 'thread_count' of threads (so basically double)
                // Return true if successful, had very basic error checking
                bool initialize(int width, int height, int thread_count = 1, bool pipelined = false);

                // de-initialize, free memory
                void release();

                // synchronous filter function, returns only when everything finished, goes faster if there's multiple threads
                // initialize() and setSigma() should be called before this
                // 'out_data' - output image buffer, assumes 4 byte per pixel
                // 'in_data' - input image buffer, assumes 4 byte per pixel
                // 'width' - width of both input and output buffers, must be same for both
                // 'height' - height of both input and output buffers, must be same for both
                // 'pitch' - row size in bytes, must be same for both buffers (ideally, this should be divisible by 16)
                // return false if failed for some reason
                bool filter(unsigned char* out_data, const unsigned char* in_data, int width, int height, int pitch);
            };

#define MAX_RANGE_TABLE_SIZE 255
#define ALIGN_SIZE 16

			// only 1 of following 2 should be defined
#define EDGE_COLOR_USE_MAXIMUM
//#define EDGE_COLOR_USE_ADDITION

// if EDGE_COLOR_USE_MAXIMUM is defined, then edge color detection works by calculating
// maximum difference among 3 components (RGB) of 2 colors, which tends to result in lower differences (since only largest among 3 is selected)
// if EDGE_COLOR_USE_ADDITION is defined, then edge color detection works by calculating
// sum of all 3 components, while enforcing 255 maximum. This method is much more sensitive to small differences 

#if defined(EDGE_COLOR_USE_MAXIMUM) && defined(EDGE_COLOR_USE_ADDITION)
#error Only 1 of those can be defined
#endif

#if !defined(EDGE_COLOR_USE_MAXIMUM) && !defined(EDGE_COLOR_USE_ADDITION)
#error 1 of those must be defined
#endif

			CRBFilterSSE2::CRBFilterSSE2()
			{
				m_range_table = new float[MAX_RANGE_TABLE_SIZE + 1];
				memset(m_range_table, 0, (MAX_RANGE_TABLE_SIZE + 1) * sizeof(float));
			}

			CRBFilterSSE2::~CRBFilterSSE2()
			{
				release();

				delete[] m_range_table;
			}

			bool CRBFilterSSE2::initialize(int width, int height, int thread_count, bool pipelined)
			{
				// basic sanity check, not strict
				if (width < 16 || width > 10000)
					return false;

				if (height < 2 || height > 10000)
					return false;

				if (thread_count < 1 || thread_count > RBF_MAX_THREADS)
					return false;

				release();

				// round width up to nearest ALIGN_SIZE * thread_count
				int round_up = (ALIGN_SIZE / 4) * thread_count;
				if (width % round_up)
				{
					width += round_up - width % round_up;
				}
				m_reserved_width = width;
				m_reserved_height = height;
				m_thread_count = thread_count;

				m_stage_buffer[0] = (unsigned char*)_mm_malloc(m_reserved_width * m_reserved_height * 4, ALIGN_SIZE);
				if (!m_stage_buffer[0])
					return false;

				if (pipelined)
				{
					for (int i = 1; i < STAGE_BUFFER_COUNT; i++)
					{
						m_stage_buffer[i] = (unsigned char*)_mm_malloc(m_reserved_width * m_reserved_height * 4, ALIGN_SIZE);
						if (!m_stage_buffer[i])
							return false;
					}
				}

				m_h_line_cache = new (std::nothrow) float* [m_thread_count];
				if (!m_h_line_cache)
					return false;

				// zero just in case
				for (int i = 0; i < m_thread_count; i++)
					m_h_line_cache[i] = nullptr;

				for (int i = 0; i < m_thread_count; i++)
				{
					m_h_line_cache[i] = (float*)_mm_malloc(m_reserved_width * 12 * sizeof(float), ALIGN_SIZE);
					if (!m_h_line_cache[i])
						return false;
				}

				//	if (m_pipelined)
				{
					m_v_line_cache = new (std::nothrow) float* [m_thread_count];
					if (!m_v_line_cache)
						return false;

					for (int i = 0; i < m_thread_count; i++)
						m_v_line_cache[i] = nullptr;

					for (int i = 0; i < m_thread_count; i++)
					{
						m_v_line_cache[i] = (float*)_mm_malloc((m_reserved_width * 8 * sizeof(float)) / m_thread_count, ALIGN_SIZE);
						if (!m_v_line_cache[i])
							return false;
					}
				}


				return true;
			}

			void CRBFilterSSE2::release()
			{
				for (int i = 0; i < STAGE_BUFFER_COUNT; i++)
				{
					if (m_stage_buffer[i])
					{
						_mm_free(m_stage_buffer[i]);
						m_stage_buffer[i] = nullptr;
					}
				}

				if (m_h_line_cache)
				{
					for (int i = 0; i < m_thread_count; i++)
					{
						if (m_h_line_cache[i])
							_mm_free(m_h_line_cache[i]);
					}
					delete[] m_h_line_cache;
					m_h_line_cache = nullptr;
				}

				//	if (m_pipelined)
				{
					for (int i = 0; i < m_thread_count; i++)
					{
						if (m_v_line_cache[i])
							_mm_free(m_v_line_cache[i]);
					}
					delete[] m_v_line_cache;
				}
				m_v_line_cache = nullptr;

				m_reserved_width = 0;
				m_reserved_height = 0;
				m_thread_count = 0;
				m_pipelined = false;
				m_filter_counter = 0;
			}

			void CRBFilterSSE2::setSigma(float sigma_spatial, float sigma_range)
			{
				if (m_sigma_spatial != sigma_spatial || m_sigma_range != sigma_range)
				{
					m_sigma_spatial = sigma_spatial;
					m_sigma_range = sigma_range;

					double alpha_f = (exp(-sqrt(2.0) / (sigma_spatial * 255.0)));
					m_inv_alpha_f = (float)(1.0 - alpha_f);
					double inv_sigma_range = 1.0 / (sigma_range * MAX_RANGE_TABLE_SIZE);
					{
						double ii = 0.f;
						for (int i = 0; i <= MAX_RANGE_TABLE_SIZE; i++, ii -= 1.0)
						{
							m_range_table[i] = (float)(alpha_f * exp(ii * inv_sigma_range));
						}
					}
				}
			}

			// example of edge color difference calculation from original implementation
			// idea is to fit maximum edge color difference as single number in 0-255 range
			// colors are added then 2 components are scaled 4x while 1 complement is scaled 2x
			// this means 1 of the components is more dominant 

			//int getDiffFactor(const unsigned char* color1, const unsigned char* color2)
			//{
			//	int c1 = abs(color1[0] - color2[0]);
			//	int c2 = abs(color1[1] - color2[1]);
			//	int c3 = abs(color1[2] - color2[2]);
			//
			//	return ((c1 + c3) >> 2) + (c2 >> 1);
			//}


			inline void getDiffFactor3x(__m128i pix4, __m128i pix4p, __m128i* diff4x)
			{
				static __m128i byte_mask = _mm_set1_epi32(255);

				// get absolute difference for each component per pixel
				__m128i diff = _mm_sub_epi8(_mm_max_epu8(pix4, pix4p), _mm_min_epu8(pix4, pix4p));

#ifdef EDGE_COLOR_USE_MAXIMUM
				// get maximum of 3 components
				__m128i diff_shift1 = _mm_srli_epi32(diff, 8); // 2nd component
				diff = _mm_max_epu8(diff, diff_shift1);
				diff_shift1 = _mm_srli_epi32(diff_shift1, 8); // 3rd component
				diff = _mm_max_epu8(diff, diff_shift1);
				// skip alpha component
				diff = _mm_and_si128(diff, byte_mask); // zero out all but 1st byte
#endif

#ifdef EDGE_COLOR_USE_ADDITION
				// add all component differences and saturate 
				__m128i diff_shift1 = _mm_srli_epi32(diff, 8); // 2nd component
				diff = _mm_adds_epu8(diff, diff_shift1);
				diff_shift1 = _mm_srli_epi32(diff_shift1, 8); // 3rd component
				diff = _mm_adds_epu8(diff, diff_shift1);
				diff = _mm_and_si128(diff, byte_mask); // zero out all but 1st byte
#endif

				_mm_store_si128(diff4x, diff);
			}


			void CRBFilterSSE2::horizontalFilter(int thread_index, const unsigned char* img_src, unsigned char* img_dst, int width, int height, int pitch)
			{
				int height_segment = height / m_thread_count;
				int buffer_offset = thread_index * height_segment * pitch;
				img_src += buffer_offset;
				img_dst += buffer_offset;

				if (thread_index + 1 == m_thread_count) // last segment should account for uneven height
					height_segment += height % m_thread_count;

				float* line_cache = m_h_line_cache[thread_index];
				const float* range_table = m_range_table;

				__m128 inv_alpha = _mm_set_ps1(m_inv_alpha_f);
				__m128 half_value = _mm_set_ps1(0.5f);
				__m128i mask_pack = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
				__m128i mask_unpack = _mm_setr_epi8(12, -1, -1, -1, 13, -1, -1, -1, 14, -1, -1, -1, 15, -1, -1, -1);

				// used to store maximum difference between 2 pixels
				SIMD_ALIGNED(16) long color_diff[4];

				for (int y = 0; y < height_segment; y++)
				{
					//////////////////////
					// right to left pass, results of this pass get stored in 'line_cache'
					{
						int pixels_left = width - 1;

						// get end of line buffer
						float* line_buffer = line_cache + pixels_left * 12;

						///////
						// handle last pixel in row separately as special case
						{
							const unsigned char* last_src = img_src + (y + 1) * pitch - 4;

							// result color
							line_buffer[8] = (float)last_src[0];
							line_buffer[9] = (float)last_src[1];
							line_buffer[10] = (float)last_src[2];
							line_buffer[11] = (float)last_src[3];

							// premultiplied source
							// caching pre-multiplied allows saving 1 multiply operation in 2nd pass loop, not a big difference
							line_buffer[4] = m_inv_alpha_f * line_buffer[8];
							line_buffer[5] = m_inv_alpha_f * line_buffer[9];
							line_buffer[6] = m_inv_alpha_f * line_buffer[10];
							line_buffer[7] = m_inv_alpha_f * line_buffer[11];
						}

						// "previous" pixel color
						__m128 pixel_prev = _mm_load_ps(line_buffer + 8);
						// "previous" pixel factor
						__m128 alpha_f_prev4 = _mm_set_ps1(1.f);

						///////
						// handle most middle pixels in 16 byte intervals using xmm registers
						// process 4x pixels at a time
						int buffer_inc = y * pitch + (pixels_left - 1) * 4 - 16;
						const __m128i* src_4xCur = (const __m128i*)(img_src + buffer_inc);
						const __m128i* src_4xPrev = (const __m128i*)(img_src + buffer_inc + 4);

						while (pixels_left > 0) // outer loop 4x pixel
						{
							// load 4x pixel, may read backward past start of buffer, but it's OK since that extra data won't be used
							__m128i pix4 = _mm_loadu_si128(src_4xCur--);
							__m128i pix4p = _mm_loadu_si128(src_4xPrev--);

							// get color differences
							getDiffFactor3x(pix4, pix4p, (__m128i*)color_diff);

							for (int i = 3; i >= 0 && pixels_left-- > 0; i--) // inner loop
							{
								float alpha_f = range_table[color_diff[i]];
								__m128 alpha_f_4x = _mm_set_ps1(alpha_f);

								// cache weights for next filter pass
								line_buffer -= 12;
								_mm_store_ps(line_buffer, alpha_f_4x);

								// color factor
								alpha_f_prev4 = _mm_mul_ps(alpha_f_prev4, alpha_f_4x);
								alpha_f_prev4 = _mm_add_ps(alpha_f_prev4, inv_alpha);

								// unpack current source pixel
								__m128i pix1 = _mm_shuffle_epi8(pix4, mask_unpack); // extracts 1 pixel components from BYTE to DWORD
								pix4 = _mm_slli_si128(pix4, 4); // shift left so next loop unpacks next pixel data 
								__m128 pixel_F = _mm_cvtepi32_ps(pix1); // convert to floats


								// apply color filter
								pixel_F = _mm_mul_ps(pixel_F, inv_alpha);
								_mm_store_ps(line_buffer + 4, pixel_F); // cache pre-multiplied source color for next filter pass
								alpha_f_4x = _mm_mul_ps(pixel_prev, alpha_f_4x);
								pixel_F = _mm_add_ps(pixel_F, alpha_f_4x);

								// store current color as previous for next cycle
								pixel_prev = pixel_F;

								// calculate final color
								pixel_F = _mm_div_ps(pixel_F, alpha_f_prev4);

								// cache filtered color for next filter pass
								_mm_store_ps(line_buffer + 8, pixel_F);
							}
						}
					}

					//////////////////////
					// left to right pass
					{
						int pixels_left = width - 1;

						// process 4x pixels at a time
						int buffer_inc = y * pitch;
						const __m128i* src_4xCur = (const __m128i*)(img_src + buffer_inc + 4); // shifted by 1 pixel
						const __m128i* src_4xPrev = (const __m128i*)(img_src + buffer_inc);

						// use float type only to enable 4 byte write using MOVSS
						float* out_result = (float*)(img_dst + buffer_inc + 4); // start at 2nd pixel from left

						const float* line_buffer = line_cache;

						///////
						// handle first pixel in row separately as special case
						{
							unsigned char* first_dst = img_dst + buffer_inc;
							// average new pixel with one already in output
							// source color was pre-multipled, so get original
							float inv_factor = 1.f / m_inv_alpha_f;
							first_dst[0] = (unsigned char)((line_buffer[4] * inv_factor + line_buffer[8]) * 0.5f);
							first_dst[1] = (unsigned char)((line_buffer[5] * inv_factor + line_buffer[9]) * 0.5f);
							first_dst[2] = (unsigned char)((line_buffer[6] * inv_factor + line_buffer[10]) * 0.5f);
							first_dst[3] = (unsigned char)((line_buffer[7] * inv_factor + line_buffer[11]) * 0.5f);
						}

						// initialize "previous pixel" with 4 components of last row pixel
						__m128 pixel_prev = _mm_load_ps(line_buffer + 8);
						line_buffer += 12;
						__m128 alpha_f_prev4 = _mm_set_ps1(1.f);


						///////
						// handle most pixels in 16 byte intervals using xmm registers
						while (pixels_left > 0) // outer loop 4x pixel
						{
							for (int i = 0; i <= 3 && pixels_left-- > 0; i++) // inner loop
							{
								// load cached factor
								__m128 alpha_f_4x = _mm_load_ps(line_buffer);
								line_buffer += 12;

								// color factor
								alpha_f_prev4 = _mm_mul_ps(alpha_f_prev4, alpha_f_4x);
								alpha_f_prev4 = _mm_add_ps(alpha_f_prev4, inv_alpha);

								// load current source pixel, pre-multiplied
								__m128 pixel_F = _mm_load_ps(line_buffer + 4);


								// apply color filter
								alpha_f_4x = _mm_mul_ps(pixel_prev, alpha_f_4x);
								pixel_F = _mm_add_ps(pixel_F, alpha_f_4x);

								// store current color as previous for next cycle
								pixel_prev = pixel_F;

								// calculate final color
								pixel_F = _mm_div_ps(pixel_F, alpha_f_prev4);

								// average this result with result from previous pass
								__m128 prev_pix4 = _mm_load_ps(line_buffer + 8);

								pixel_F = _mm_add_ps(pixel_F, prev_pix4);
								pixel_F = _mm_mul_ps(pixel_F, half_value);

								// pack float pixel into byte pixel
								__m128i pixB = _mm_cvtps_epi32(pixel_F); // convert to integer
								pixB = _mm_shuffle_epi8(pixB, mask_pack);
								_mm_store_ss(out_result++, _mm_castsi128_ps(pixB));

							}
						}
					}
				}
			}


			void CRBFilterSSE2::verticalFilter(int thread_index, const unsigned char* img_src, unsigned char* img_dst, int width, int height, int pitch)
			{
				int width_segment = width / m_thread_count;
				// make sure width segments round to 16 byte boundary except for last one
				width_segment -= width_segment % 4;
				int start_offset = width_segment * thread_index;
				if (thread_index == m_thread_count - 1) // last one
					width_segment = width - start_offset;

				int width4 = width_segment / 4;

				// adjust img buffer starting positions
				img_src += start_offset * 4;
				img_dst += start_offset * 4;

				float* line_cache = m_v_line_cache[thread_index];
				const float* range_table = m_range_table;

				__m128 inv_alpha = _mm_set_ps1(m_inv_alpha_f);
				__m128 half_value = _mm_set_ps1(0.5f);
				__m128i mask_pack = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
				__m128i mask_unpack = _mm_setr_epi8(0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1);

				// used to store maximum difference between 2 pixels
				SIMD_ALIGNED(16) long color_diff[4];

				/////////////////
				// Bottom to top pass first
				{
					// last line processed separately since no previous
					{
						unsigned char* dst_line = img_dst + (height - 1) * pitch;
						const unsigned char* src_line = img_src + (height - 1) * pitch;
						float* line_buffer = line_cache;

						memcpy(dst_line, src_line, width_segment * 4); // copy last line

						// initialize line cache
						for (int x = 0; x < width_segment; x++)
						{
							// set factor to 1
							line_buffer[0] = 1.f;
							line_buffer[1] = 1.f;
							line_buffer[2] = 1.f;
							line_buffer[3] = 1.f;

							// set result color
							line_buffer[4] = (float)src_line[0];
							line_buffer[5] = (float)src_line[1];
							line_buffer[6] = (float)src_line[2];
							line_buffer[7] = (float)src_line[3];

							src_line += 4;
							line_buffer += 8;
						}
					}

					// process other lines
					for (int y = height - 2; y >= 0; y--)
					{
						float* dst_line = (float*)(img_dst + y * pitch);
						float* line_buffer = line_cache;

						__m128i* src_4xCur = (__m128i*)(img_src + y * pitch);
						__m128i* src_4xPrev = (__m128i*)(img_src + (y + 1) * pitch);

						int pixels_left = width_segment;
						while (pixels_left > 0)
						{
							// may read past end of buffer, but that data won't be used
							__m128i pix4 = _mm_loadu_si128(src_4xCur++); // load 4x pixel
							__m128i pix4p = _mm_loadu_si128(src_4xPrev++);

							// get color differences
							getDiffFactor3x(pix4, pix4p, (__m128i*)color_diff);

							for (int i = 0; i < 4 && pixels_left-- > 0; i++) // inner loop
							{
								float alpha_f = range_table[color_diff[i]];
								__m128 alpha_f_4x = _mm_set_ps1(alpha_f);

								// load previous line color factor
								__m128 alpha_f_prev4 = _mm_load_ps(line_buffer);
								// load previous line color
								__m128 pixel_prev = _mm_load_ps(line_buffer + 4);

								// color factor
								alpha_f_prev4 = _mm_mul_ps(alpha_f_prev4, alpha_f_4x);
								alpha_f_prev4 = _mm_add_ps(alpha_f_prev4, inv_alpha);

								// unpack current source pixel
								__m128i pix1 = _mm_shuffle_epi8(pix4, mask_unpack);
								pix4 = _mm_srli_si128(pix4, 4); // shift right
								__m128 pixel_F = _mm_cvtepi32_ps(pix1); // convert to floats


								// apply color filter
								pixel_F = _mm_mul_ps(pixel_F, inv_alpha);
								alpha_f_4x = _mm_mul_ps(pixel_prev, alpha_f_4x);
								pixel_F = _mm_add_ps(pixel_F, alpha_f_4x);

								// store current factor and color as previous for next cycle
								_mm_store_ps(line_buffer, alpha_f_prev4);
								_mm_store_ps(line_buffer + 4, pixel_F);
								line_buffer += 8;

								// calculate final color
								pixel_F = _mm_div_ps(pixel_F, alpha_f_prev4);

								// pack float pixel into byte pixel
								__m128i pixB = _mm_cvtps_epi32(pixel_F); // convert to integer
								pixB = _mm_shuffle_epi8(pixB, mask_pack);
								_mm_store_ss(dst_line++, _mm_castsi128_ps(pixB));
							}
						}
					}
				}

				/////////////////
				// Top to bottom pass last
				{
					mask_pack = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12);

					// first line handled separately since no previous
					{
						unsigned char* dst_line = img_dst;
						const unsigned char* src_line = img_src;
						float* line_buffer = line_cache;

						for (int x = 0; x < width_segment; x++)
						{
							// average ccurrent destanation color with current source
							dst_line[0] = (dst_line[0] + src_line[0]) / 2;
							dst_line[1] = (dst_line[1] + src_line[1]) / 2;
							dst_line[2] = (dst_line[2] + src_line[2]) / 2;
							dst_line[3] = (dst_line[3] + src_line[3]) / 2;

							// set factor to 1
							line_buffer[0] = 1.f;
							line_buffer[1] = 1.f;
							line_buffer[2] = 1.f;
							line_buffer[3] = 1.f;

							// set result color
							line_buffer[4] = (float)src_line[0];
							line_buffer[5] = (float)src_line[1];
							line_buffer[6] = (float)src_line[2];
							line_buffer[7] = (float)src_line[3];

							dst_line += 4;
							src_line += 4;
							line_buffer += 8;
						}
					}

					// process other lines
					for (int y = 1; y < height; y++)
					{
						//	const unsigned char* src_line = img_src + y * pitch;
						float* line_buffer = line_cache;

						__m128i* src_4xCur = (__m128i*)(img_src + y * pitch);
						__m128i* src_4xPrev = (__m128i*)(img_src + (y - 1) * pitch);
						__m128i* dst_4x = (__m128i*)(img_dst + y * pitch);

						for (int x = 0; x < width4; x++)
						{
							// get color difference
							__m128i pix4 = _mm_loadu_si128(src_4xCur++); // load 4x pixel
							__m128i pix4p = _mm_loadu_si128(src_4xPrev++);

							// get color differences
							getDiffFactor3x(pix4, pix4p, (__m128i*)color_diff);

							__m128i out_pix4 = _mm_setzero_si128();
							for (int i = 0; i < 4; i++) // inner loop
							{
								float alpha_f = range_table[color_diff[i]];
								__m128 alpha_f_4x = _mm_set_ps1(alpha_f);

								// load previous line color factor
								__m128 alpha_f_prev4 = _mm_load_ps(line_buffer);
								// load previous line color
								__m128 pixel_prev = _mm_load_ps(line_buffer + 4);

								// color factor
								//	alpha_f_prev = m_inv_alpha_f + alpha_f * alpha_f_prev;
								alpha_f_prev4 = _mm_mul_ps(alpha_f_prev4, alpha_f_4x);
								alpha_f_prev4 = _mm_add_ps(alpha_f_prev4, inv_alpha);

								// unpack current source pixel
								__m128i pix1 = _mm_shuffle_epi8(pix4, mask_unpack);
								pix4 = _mm_srli_si128(pix4, 4); // shift right
								__m128 pixel_F = _mm_cvtepi32_ps(pix1); // convert to floats

								// apply color filter
								pixel_F = _mm_mul_ps(pixel_F, inv_alpha);
								alpha_f_4x = _mm_mul_ps(pixel_prev, alpha_f_4x);
								pixel_F = _mm_add_ps(pixel_F, alpha_f_4x);

								// store current factor and color as previous for next cycle
								_mm_store_ps(line_buffer, alpha_f_prev4);
								_mm_store_ps(line_buffer + 4, pixel_F);
								line_buffer += 8;

								// calculate final color
								pixel_F = _mm_div_ps(pixel_F, alpha_f_prev4);

								// pack float pixel into byte pixel
								__m128i pixB = _mm_cvtps_epi32(pixel_F); // convert to integer
								pixB = _mm_shuffle_epi8(pixB, mask_pack);

								out_pix4 = _mm_srli_si128(out_pix4, 4); // shift 
								out_pix4 = _mm_or_si128(out_pix4, pixB);

							}

							// average result 4x pixel with what is already in destination
							__m128i dst4 = _mm_loadu_si128(dst_4x);
							out_pix4 = _mm_avg_epu8(out_pix4, dst4);
							_mm_storeu_si128(dst_4x++, out_pix4); // store 4x pixel
						}

						// have to handle leftover 1-3 pixels if last width segment isn't divisble by 4
						if (width_segment % 4)
						{
							// this should be avoided by having image buffers with pitch divisible by 16
						}
					}
				}

			}

			bool CRBFilterSSE2::filter(unsigned char* out_data, const unsigned char* in_data, int width, int height, int pitch)
			{
				// basic error checking
				if (!m_stage_buffer[0])
					return false;

				if (width < 16 || width > m_reserved_width)
					return false;

				if (height < 16 || height > m_reserved_height)
					return false;

				if (pitch < width * 4)
					return false;

				if (!out_data || !in_data)
					return false;

				if (m_inv_alpha_f == 0.f)
					return false;

				horizontalFilter(0, in_data, m_stage_buffer[0], width, height, pitch);

				verticalFilter(0, m_stage_buffer[0], out_data, width, height, pitch);

				return true;
			}
        }
#endif //SIMD_RBF_SSE2_TEST

        //-----------------------------------------------------------------------------------------

        namespace Rbf
        {
            template<size_t channels> int DiffFactor(const uint8_t* color1, const uint8_t* color2)
            {
                int final_diff, component_diff[4];
                for (int i = 0; i < channels; i++)
                    component_diff[i] = abs(color1[i] - color2[i]);
                switch (channels)
                {
                case 1:
                    final_diff = component_diff[0];
                    break;
                case 2:
                    final_diff = ((component_diff[0] + component_diff[1]) >> 1);
                    break;
                case 3:
                    final_diff = ((component_diff[0] + component_diff[2]) >> 2) + (component_diff[1] >> 1);
                    break;
                case 4:
                    final_diff = ((component_diff[0] + component_diff[1] + component_diff[2] + component_diff[3]) >> 2);
                    break;
                default:
                    final_diff = 0;
                }
                assert(final_diff >= 0 && final_diff <= 255);
                return final_diff;
            }

            template<size_t channels> void SetOut(const float* bc, const float* bf, const float* ec, const float* ef,
                size_t width, size_t height, uint8_t* dst, size_t dstStride)
            {
                size_t tail = dstStride - width * channels;
                for (size_t y = 0; y < height; ++y)
                {
                    for (size_t x = 0; x < width; x++)
                    {
                        float factor = 1.f / (bf[x] + ef[x]);
                        for (size_t c = 0; c < channels; c++)
                        {
                            dst[c] = uint8_t(factor * (bc[c] + ec[c]));
                        }
                        bc += channels;
                        ec += channels;
                        dst += channels;
                    }
                    bf += width;
                    ef += width;
                    dst += tail;
                }
            }

            template<size_t channels>
            void HorizontalFilter(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride, size_t width, size_t height,
                float* ranges, float alpha, float* left_Color_Buffer, float* left_Factor_Buffer, float* right_Color_Buffer, float* right_Factor_Buffer)
            {
                size_t size = width * channels, cLast = size - 1, fLast = width - 1;
                for (size_t y = 0; y < height; y++)
                {
                    const uint8_t* src_left_color = src + y * srcStride;
                    float* left_Color = left_Color_Buffer + y * size;
                    float* left_Factor = left_Factor_Buffer + y * width;

                    const uint8_t* src_right_color = src + y * srcStride + cLast;
                    float* right_Color = right_Color_Buffer + y * size + cLast;
                    float* right_Factor = right_Factor_Buffer + y * width + fLast;

                    const uint8_t* src_left_prev = src_left_color;
                    const float* left_prev_factor = left_Factor;
                    const float* left_prev_color = left_Color;

                    const uint8_t* src_right_prev = src_right_color;
                    const float* right_prev_factor = right_Factor;
                    const float* right_prev_color = right_Color;

                    *left_Factor++ = 1.f;
                    *right_Factor-- = 1.f;
                    for (int c = 0; c < channels; c++)
                    {
                        *left_Color++ = *src_left_color++;
                        *right_Color-- = *src_right_color--;
                    }
                    for (size_t x = 1; x < width; x++)
                    {
                        int left_diff = DiffFactor<channels>(src_left_color, src_left_prev);
                        src_left_prev = src_left_color;

                        int right_diff = DiffFactor<channels>(src_right_color + 1 - channels, src_right_prev + 1 - channels);
                        src_right_prev = src_right_color;

                        float left_alpha_f = ranges[left_diff];
                        float right_alpha_f = ranges[right_diff];
                        *left_Factor++ = alpha + left_alpha_f * (*left_prev_factor++);
                        *right_Factor-- = alpha + right_alpha_f * (*right_prev_factor--);

                        for (int c = 0; c < channels; c++)
                        {
                            *left_Color++ = (alpha * (*src_left_color++) + left_alpha_f * (*left_prev_color++));
                            *right_Color-- = (alpha * (*src_right_color--) + right_alpha_f * (*right_prev_color--));
                        }
                    }
                }
                SetOut<channels>(left_Color_Buffer, left_Factor_Buffer, right_Color_Buffer, right_Factor_Buffer, width, height, dst, dstStride);
            }

            template<size_t channels>
            void VerticalFilter(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride, int width, int height,
                float* range_table_f, float inv_alpha_f, float* down_Color_Buffer, float* down_Factor_Buffer, float* up_Color_Buffer, float* up_Factor_Buffer)
            {
                size_t size = width * channels, srcTail = srcStride - size, dstTail = dstStride - size;

                const uint8_t* src_color_first_hor = dst;
                const uint8_t* src_down_color = src;
                float* down_color = down_Color_Buffer;
                float* down_factor = down_Factor_Buffer;

                const uint8_t* src_down_prev = src_down_color;
                const float* down_prev_color = down_color;
                const float* down_prev_factor = down_factor;

                int last_index = size * height - 1;
                const uint8_t* src_up_color = src + srcStride * (height - 1) + size - 1;
                const uint8_t* src_color_last_hor = dst + dstStride * (height - 1) + size - 1;
                float* up_color = up_Color_Buffer + last_index;
                float* up_factor = up_Factor_Buffer + (width * height - 1);

                const float* up_prev_color = up_color;
                const float* up_prev_factor = up_factor;

                for (int x = 0; x < width; x++)
                {
                    *down_factor++ = 1.f;
                    *up_factor-- = 1.f;
                    for (int c = 0; c < channels; c++)
                    {
                        *down_color++ = *src_color_first_hor++;
                        *up_color-- = *src_color_last_hor--;
                    }
                    src_down_color += channels;
                    src_up_color -= channels;
                }
                src_color_first_hor += dstTail;
                src_color_last_hor -= dstTail;
                src_down_color += srcTail;
                src_up_color -= srcTail;
                for (int y = 1; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int down_diff = DiffFactor<channels>(src_down_color, src_down_color - srcStride);
                        src_down_prev += channels;
                        src_down_color += channels;
                        src_up_color -= channels;
                        int up_diff = DiffFactor<channels>(src_up_color + 1, src_up_color + srcStride + 1);
                        float down_alpha_f = range_table_f[down_diff];
                        float up_alpha_f = range_table_f[up_diff];

                        *down_factor++ = inv_alpha_f + down_alpha_f * (*down_prev_factor++);
                        *up_factor-- = inv_alpha_f + up_alpha_f * (*up_prev_factor--);

                        for (int c = 0; c < channels; c++)
                        {
                            *down_color++ = inv_alpha_f * (*src_color_first_hor++) + down_alpha_f * (*down_prev_color++);
                            *up_color-- = inv_alpha_f * (*src_color_last_hor--) + up_alpha_f * (*up_prev_color--);
                        }
                    }
                    src_color_first_hor += dstTail;
                    src_color_last_hor -= dstTail;
                    src_down_color += srcTail;
                    src_up_color -= srcTail;
                }

                SetOut<channels>(down_Color_Buffer, down_Factor_Buffer, up_Color_Buffer, up_Factor_Buffer, width, height, dst, dstStride);
            }

            template<size_t channels>
            void Filter(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride, int Width, int Height, float sigmaSpatial, float sigmaRange)
            {
                int reserveWidth = Width;
                int reserveHeight = Height;

                assert(reserveWidth >= 10 && reserveWidth < 10000);
                assert(reserveHeight >= 10 && reserveHeight < 10000);
                assert(channels >= 1 && channels <= 4);

                int reservePixels = reserveWidth * reserveHeight;
                int numberOfPixels = reservePixels * channels;

                float* leftColorBuffer = (float*)calloc(sizeof(float) * numberOfPixels, 1);
                float* leftFactorBuffer = (float*)calloc(sizeof(float) * reservePixels, 1);
                float* rightColorBuffer = (float*)calloc(sizeof(float) * numberOfPixels, 1);
                float* rightFactorBuffer = (float*)calloc(sizeof(float) * reservePixels, 1);

                if (leftColorBuffer == NULL || leftFactorBuffer == NULL || rightColorBuffer == NULL || rightFactorBuffer == NULL)
                {
                    if (leftColorBuffer)  free(leftColorBuffer);
                    if (leftFactorBuffer) free(leftFactorBuffer);
                    if (rightColorBuffer) free(rightColorBuffer);
                    if (rightFactorBuffer) free(rightFactorBuffer);

                    return;
                }
                float* downColorBuffer = leftColorBuffer;
                float* downFactorBuffer = leftFactorBuffer;
                float* upColorBuffer = rightColorBuffer;
                float* upFactorBuffer = rightFactorBuffer;

                float alpha_f = static_cast<float>(exp(-sqrt(2.0) / (sigmaSpatial * 255)));
                float inv_alpha_f = 1.f - alpha_f;


                float range_table_f[255 + 1];
                float inv_sigma_range = 1.0f / (sigmaRange * 255);

                float ii = 0.f;
                for (int i = 0; i <= 255; i++, ii -= 1.f)
                {
                    range_table_f[i] = alpha_f * exp(ii * inv_sigma_range);
                }

                HorizontalFilter<channels>(src, srcStride, dst, dstStride, Width, Height,
                    range_table_f, inv_alpha_f, leftColorBuffer, leftFactorBuffer, rightColorBuffer, rightFactorBuffer);

                VerticalFilter<channels>(src, srcStride, dst, dstStride, Width, Height,
                    range_table_f, inv_alpha_f, downColorBuffer, downFactorBuffer, upColorBuffer, upFactorBuffer);

                if (leftColorBuffer)
                {
                    free(leftColorBuffer);
                    leftColorBuffer = NULL;
                }

                if (leftFactorBuffer)
                {
                    free(leftFactorBuffer);
                    leftFactorBuffer = NULL;
                }

                if (rightColorBuffer)
                {
                    free(rightColorBuffer);
                    rightColorBuffer = NULL;
                }

                if (rightFactorBuffer)
                {
                    free(rightFactorBuffer);
                    rightFactorBuffer = NULL;
                }
            }
        }

		//-----------------------------------------------------------------------------------------

        RecursiveBilateralFilterDefault::RecursiveBilateralFilterDefault(const RbfParam& param)
            :Base::RecursiveBilateralFilterDefault(param)
        {
        }

        void RecursiveBilateralFilterDefault::Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            switch (_param.channels)
            {
            case 1: Rbf::Filter<1>(src, srcStride, dst, dstStride, (int)_param.width, (int)_param.height, _param.spatial, _param.range); break;
            case 2: Rbf::Filter<2>(src, srcStride, dst, dstStride, (int)_param.width, (int)_param.height, _param.spatial, _param.range); break;
            case 3: Rbf::Filter<3>(src, srcStride, dst, dstStride, (int)_param.width, (int)_param.height, _param.spatial, _param.range); break;
#ifndef SIMD_RBF_SSE2_TEST
			case 4:	Rbf::Filter<4>(src, srcStride, dst, dstStride, (int)_param.width, (int)_param.height, _param.spatial, _param.range); break;
#else
            case 4: 
			{
				RbfSse2::CRBFilterSSE2 filter;
				filter.initialize((int)_param.width, (int)_param.height);
				filter.setSigma(_param.spatial, _param.range);
				filter.filter(dst, src, (int)_param.width, (int)_param.height, srcStride);
				break;
			}
#endif
            default:
                assert(0);
            }
        }

        //-----------------------------------------------------------------------------------------

        void* RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, const float* sigmaSpatial, const float* sigmaRange)
        {
            RbfParam param(width, height, channels, sigmaSpatial, sigmaRange, sizeof(void*));
            if (!param.Valid())
                return NULL;
            return new RecursiveBilateralFilterDefault(param);
        }
    }
#endif//SIMD_SSE41_ENABLE
}

