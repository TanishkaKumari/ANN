#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include <functional>
#include <iomanip>
#include <iostream>

namespace sp
{
	template<typename T>

	class Matrix
	{
	public:
		uint32_t _cols;
		uint32_t _rows;
		std::vector<T> _vals;
	public:
		Matrix()
			: _cols(0),
			_rows(0),
			_vals({})
		{}

		Matrix(uint32_t cols, uint32_t rows)
			: _cols(cols),
			_rows(rows),
			_vals({})
		{
			_vals.resize(rows * cols, T());
		}

		Matrix applyFunction(std::function<T(const T&)> func)
		{
			Matrix output(_cols, _rows);

			for (uint32_t y = 0; y < output._rows; y++)
			{
				for (uint32_t x = 0; x < output._cols; x++)
				{
					output.at(x, y) = func(at(x, y));
				}
			}
			return output;
		}

		T& at(uint32_t col, uint32_t row) {
			return _vals[row * _cols + col];
		}

		Matrix multiply(Matrix& target)
		{
			assert(_cols == target._rows);
			Matrix output(target._cols, _rows);

			for (uint32_t y = 0; y < output._rows; y++)
				for (uint32_t x = 0; x < output._cols; x++)
				{
					T result = T();
					for (uint32_t k = 0; k < _cols; k++)
					{
						result += at(k, y) * target.at(x, k);
					}
					output.at(x, y) = result;
				}
			return output;
		}

		Matrix multiplyScaler(float s)
		{
			Matrix output(_cols, _rows);

			for (uint32_t y = 0; y < output._rows; y++)
			{
				for (uint32_t x = 0; x < output._cols; x++)
				{
					output.at(x, y) = at(x, y) * s;
				}
			}
			return output;
		}

		Matrix multiplyElement(Matrix& target)
		{
			assert(_rows == target._rows && _cols == target._cols);
			Matrix output(_cols, _rows);

			for (uint32_t y = 0; y < output._rows; y++)
			{
				for (uint32_t x = 0; x < output._cols; x++)
				{
					output.at(x, y) = at(x, y) * target.at(x, y);
				}
			}
			return output;
		}

		Matrix add(Matrix& target)
		{
			assert(_rows == target._rows && _cols == target._cols);
			Matrix output(_cols, _rows);
			for (uint32_t y = 0; y < output._rows; y++)
			{
				for (uint32_t x = 0; x < output._cols; x++)
				{
					output.at(x, y) = at(x, y) + target.at(x, y);
				}
			}
			return output;
		}


		Matrix addScaler(float s)
		{
			Matrix output(_cols, _rows);

			for (uint32_t y = 0; y < output._rows; y++)
			{
				for (uint32_t x = 0; x < output._cols; x++)
				{
					output.at(x, y) = at(x, y) + s;
				}
			}
			return output;
		}

		Matrix negetive()
		{
			Matrix output(_cols, _rows);

			for (uint32_t y = 0; y < output._rows; y++)
			{
				for (uint32_t x = 0; x < output._cols; x++)
				{
					output.at(x, y) = -at(x, y);
				}
			}
			return output;
		}

		Matrix transpose()
		{
			Matrix output(_rows, _cols);
			for (uint32_t y = 0; y < _rows; y++)
				for (uint32_t x = 0; x < _cols; x++)
				{
					output.at(y, x) = at(x, y);
				}
			return output;
		}
	};

	template<typename T>
	void LogMatrix(Matrix<T>& mat)
	{
		for (uint32_t y = 0; y < mat._rows; y++)
		{
			for (uint32_t x = 0; x < mat._cols; x++)
				std::cout << std::setw(10) << mat.at(x, y) << " ";
			std::cout << std::endl;
		}
	}
}






