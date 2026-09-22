/* SignalMint portable integer inference runtime.
 *
 * Streaming-friendly, dependency-free C11 that mirrors the Python integer
 * reference (signalmint/quantize/int_infer.py) bit-for-bit: INT8 conv weights,
 * Q(FRAC_BITS) fixed-point activations, round-half-away-from-zero requantization,
 * and shared tanh/sigmoid LUTs. Given identical inputs it produces identical
 * int32 logits to the reference -- the property verified by the parity test.
 */
#ifndef SIGNALMINT_RT_H
#define SIGNALMINT_RT_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    const int8_t*  weight;      /* [out_ch * in_ch * kernel], row-major */
    const int32_t* bias;        /* [out_ch], Q.F */
    int32_t out_ch;
    int32_t in_ch;
    int32_t kernel;
    int32_t dilation;
    int32_t multiplier;         /* requant multiplier M */
    int32_t shift;              /* requant total shift */
} QConv;

typedef struct {
    int32_t num_bins;
    int32_t res_ch;
    int32_t skip_ch;
    int32_t kernel;
    int32_t num_layers;
    int32_t dil_cycle;
    const int32_t* embed_qf;    /* [num_bins * res_ch], Q.F */
    const QConv*   conv;        /* [num_layers] gated convs (2*res_ch out) */
    const QConv*   residual;    /* [num_layers] 1x1 */
    const QConv*   skip;        /* [num_layers] 1x1 */
    QConv head1;
    QConv head2;
    const int32_t* tanh_lut;    /* [SM_LUT_SIZE], Q.F */
    const int32_t* sigmoid_lut; /* [SM_LUT_SIZE], Q.F */
} QModel;

/* Fixed-point / LUT constants must match signalmint.quantize.fixedpoint. */
#define SM_RT_FRAC_BITS 12
#define SM_RT_LUT_SIZE  4096
#define SM_RT_LUT_RANGE 8

/* round(acc * multiplier / 2^shift), round half away from zero. */
int64_t sm_requantize(int64_t acc, int32_t multiplier, int32_t shift);

/* round(prod / 2^shift), round half away from zero. */
int64_t sm_round_shift(int64_t prod, int32_t shift);

/* Compute int32 Q.F logits for a length-T symbol frame.
 * logits_out must have room for T * num_bins int32 values (row-major (T, bins)).
 * Returns 0 on success, non-zero if T exceeds SM_RT_MAX_T. */
int sm_forward(const QModel* model, const int32_t* symbols, int32_t T,
               int32_t* logits_out);

/* Maximum frame length supported by the static scratch buffers. */
#ifndef SM_RT_MAX_T
#define SM_RT_MAX_T 2048
#endif
#ifndef SM_RT_MAX_CH
#define SM_RT_MAX_CH 256
#endif

#endif /* SIGNALMINT_RT_H */
