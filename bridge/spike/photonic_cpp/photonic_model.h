#ifndef PHOTONIC_CPP_PHOTONIC_MODEL_H
#define PHOTONIC_CPP_PHOTONIC_MODEL_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <iostream>

namespace photonic {

/**
 * Abstract base class for photonic models.
 *
 * All photonic models must implement MVM (Matrix-Vector Multiplication)
 * and DOTP (Dot Product) operations.
 */
class PhotonicModel {
public:
    virtual ~PhotonicModel() = default;

    /**
     * Matrix-Vector Multiplication.
     * result = matrix * vector
     *
     * @param result Output vector (size: rows)
     * @param matrix Input matrix (size: rows x cols, row-major)
     * @param vector Input vector (size: cols)
     * @param rows Number of rows in matrix
     * @param cols Number of columns in matrix
     */
    virtual void mvm(int16_t* result, const int16_t* matrix,
                     const int16_t* vector, size_t rows, size_t cols) = 0;

    /**
     * Dot Product.
     * result = vec1 . vec2
     *
     * @param vec1 First vector
     * @param vec2 Second vector
     * @param len Length of vectors
     * @return Dot product result
     */
    virtual int16_t dotp(const int16_t* vec1, const int16_t* vec2, size_t len) = 0;

    /**
     * Get model name.
     */
    virtual std::string name() const = 0;

    /**
     * Check if model applies noise/non-idealities.
     */
    virtual bool is_ideal() const { return false; }
};

/**
 * Float version of PhotonicModel for internal computations.
 * Noise models work with float internally and quantize at the end.
 */
class PhotonicModelFloat {
public:
    virtual ~PhotonicModelFloat() = default;

    virtual void mvm(float* result, const float* matrix,
                     const float* vector, size_t rows, size_t cols) = 0;

    virtual float dotp(const float* vec1, const float* vec2, size_t len) = 0;

    virtual std::string name() const = 0;
    virtual bool is_ideal() const { return false; }
};

} // namespace photonic

#endif // PHOTONIC_CPP_PHOTONIC_MODEL_H
