#include <stddef.h>
#include <stdint.h>

/*
 * Independent sequential decoders for the DFloat11 v0.5.0 and ECF8 v0.2.0
 * tensor formats.
 *
 * The official implementations decode independent 8-byte regions in CUDA
 * kernels.  This verifier deliberately follows each logical Huffman bitstream
 * from the beginning instead.  It shares no decoder code or execution path
 * with either official CUDA implementation.
 */

enum {
    DFLOAT11_OK = 0,
    DFLOAT11_INVALID_ARGUMENT = 1,
    DFLOAT11_TRUNCATED_CODE = 2,
    DFLOAT11_INVALID_LUT_POINTER = 3,
    DFLOAT11_INVALID_CODE_LENGTH = 4
};

static uint32_t read_prefix(
    const uint8_t *codes,
    size_t n_code_bytes,
    uint64_t bit_position
) {
    const size_t byte_position = (size_t)(bit_position >> 3);
    const unsigned bit_offset = (unsigned)(bit_position & 7U);
    uint64_t window = 0;

    for (unsigned index = 0; index < 5; ++index) {
        const size_t position = byte_position + index;
        window = (window << 8) |
            (uint64_t)(position < n_code_bytes ? codes[position] : 0);
    }
    return (uint32_t)(window >> (8U - bit_offset));
}

int brevis_dfloat11_decode(
    const uint8_t *luts,
    size_t n_lut_rows,
    const uint8_t *codes,
    size_t n_code_bytes,
    const uint8_t *sign_mantissa,
    size_t n_elements,
    uint64_t *bit_position,
    uint16_t *output,
    size_t *failed_element
) {
    if (
        luts == NULL ||
        codes == NULL ||
        sign_mantissa == NULL ||
        bit_position == NULL ||
        output == NULL ||
        n_lut_rows < 2
    ) {
        return DFLOAT11_INVALID_ARGUMENT;
    }

    const size_t length_row = n_lut_rows - 1;
    const uint64_t n_code_bits = (uint64_t)n_code_bytes * 8U;
    uint64_t position = *bit_position;

    for (size_t element = 0; element < n_elements; ++element) {
        if (position >= n_code_bits) {
            if (failed_element != NULL) {
                *failed_element = element;
            }
            return DFLOAT11_TRUNCATED_CODE;
        }

        const uint32_t prefix = read_prefix(codes, n_code_bytes, position);
        uint8_t decoded = luts[(prefix >> 24) & 0xffU];
        unsigned depth = 0;

        while (decoded >= 240U) {
            const size_t row = (size_t)(256U - decoded);
            ++depth;
            if (row >= length_row || depth > 3U) {
                if (failed_element != NULL) {
                    *failed_element = element;
                }
                return DFLOAT11_INVALID_LUT_POINTER;
            }
            decoded = luts[
                row * 256U +
                ((prefix >> (24U - depth * 8U)) & 0xffU)
            ];
        }

        const uint8_t code_length = luts[length_row * 256U + decoded];
        if (
            code_length == 0U ||
            code_length > 32U ||
            code_length > n_code_bits - position
        ) {
            if (failed_element != NULL) {
                *failed_element = element;
            }
            return DFLOAT11_INVALID_CODE_LENGTH;
        }

        const uint8_t other_bits = sign_mantissa[element];
        output[element] =
            ((uint16_t)(other_bits & 0x80U) << 8) |
            ((uint16_t)decoded << 7) |
            (uint16_t)(other_bits & 0x7fU);
        position += code_length;
    }

    *bit_position = position;
    return DFLOAT11_OK;
}

int brevis_ecf8_decode(
    const uint8_t *luts,
    size_t n_lut_rows,
    const uint8_t *codes,
    size_t n_code_bytes,
    const uint8_t *packed_other_4bits,
    size_t n_packed_bytes,
    size_t start_element,
    size_t n_elements,
    uint64_t *bit_position,
    uint8_t *output,
    size_t *failed_element
) {
    if (
        luts == NULL ||
        codes == NULL ||
        packed_other_4bits == NULL ||
        bit_position == NULL ||
        output == NULL ||
        n_lut_rows < 2 ||
        start_element > SIZE_MAX - n_elements
    ) {
        return DFLOAT11_INVALID_ARGUMENT;
    }
    const size_t end_element = start_element + n_elements;
    if (end_element / 2U + end_element % 2U > n_packed_bytes) {
        return DFLOAT11_INVALID_ARGUMENT;
    }

    const size_t length_row = n_lut_rows - 1;
    const uint64_t n_code_bits = (uint64_t)n_code_bytes * 8U;
    uint64_t position = *bit_position;

    for (size_t element = 0; element < n_elements; ++element) {
        if (position >= n_code_bits) {
            if (failed_element != NULL) {
                *failed_element = element;
            }
            return DFLOAT11_TRUNCATED_CODE;
        }

        const uint32_t prefix = read_prefix(codes, n_code_bytes, position);
        uint8_t decoded = luts[(prefix >> 24) & 0xffU];
        unsigned depth = 0;

        while (decoded >= 240U) {
            const size_t row = (size_t)(256U - decoded);
            ++depth;
            if (row >= length_row || depth > 1U) {
                if (failed_element != NULL) {
                    *failed_element = element;
                }
                return DFLOAT11_INVALID_LUT_POINTER;
            }
            decoded = luts[
                row * 256U +
                ((prefix >> (24U - depth * 8U)) & 0xffU)
            ];
        }

        const uint8_t code_length = luts[length_row * 256U + decoded];
        if (
            code_length == 0U ||
            code_length > 16U ||
            code_length > n_code_bits - position ||
            decoded > 15U
        ) {
            if (failed_element != NULL) {
                *failed_element = element;
            }
            return DFLOAT11_INVALID_CODE_LENGTH;
        }

        const size_t global_element = start_element + element;
        const uint8_t packed = packed_other_4bits[global_element / 2U];
        const uint8_t other_bits = (
            global_element % 2U == 0U ? packed >> 4 : packed & 0x0fU
        );
        output[element] =
            (uint8_t)(decoded << 3) |
            (uint8_t)((other_bits & 0x08U) << 4) |
            (uint8_t)(other_bits & 0x07U);
        position += code_length;
    }

    *bit_position = position;
    return DFLOAT11_OK;
}
