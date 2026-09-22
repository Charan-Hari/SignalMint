/* SignalMint integer inference engine. Mirrors signalmint/quantize/int_infer.py. */
#include "signalmint_rt.h"
#include <stdlib.h>
#include <string.h>

int64_t sm_round_shift(int64_t prod, int32_t shift) {
    if (shift <= 0) {
        return prod << (-shift);
    }
    int64_t half = (int64_t)1 << (shift - 1);
    if (prod >= 0) {
        return (prod + half) >> shift;
    }
    return -(((-prod) + half) >> shift);
}

int64_t sm_requantize(int64_t acc, int32_t multiplier, int32_t shift) {
    return sm_round_shift(acc * (int64_t)multiplier, shift);
}

static int32_t sm_lut_lookup(const int32_t* lut, int64_t x) {
    int64_t range_qf = (int64_t)SM_RT_LUT_RANGE << SM_RT_FRAC_BITS;
    if (x < -range_qf) x = -range_qf;
    if (x > range_qf) x = range_qf;
    int64_t idx = ((x + range_qf) * (SM_RT_LUT_SIZE - 1)) / (2 * range_qf);
    return lut[idx];
}

/* Dilated causal conv on Q.F input x[in_ch*T] -> Q.F out[out_ch*T]. */
static void sm_causal_conv(const int64_t* x, int32_t T, const QConv* c, int64_t* out) {
    int32_t pad = (c->kernel - 1) * c->dilation;
    for (int32_t o = 0; o < c->out_ch; ++o) {
        const int8_t* wrow = c->weight + (size_t)o * c->in_ch * c->kernel;
        int64_t bias = c->bias[o];
        for (int32_t t = 0; t < T; ++t) {
            int64_t acc = 0;
            for (int32_t k = 0; k < c->kernel; ++k) {
                int32_t ti = t - (c->kernel - 1 - k) * c->dilation;
                if (ti < 0) continue;
                for (int32_t i = 0; i < c->in_ch; ++i) {
                    acc += (int64_t)wrow[i * c->kernel + k] * x[(size_t)i * T + ti];
                }
            }
            out[(size_t)o * T + t] = sm_requantize(acc, c->multiplier, c->shift) + bias;
        }
    }
    (void)pad;
}

int sm_forward(const QModel* m, const int32_t* symbols, int32_t T, int32_t* logits_out) {
    if (T <= 0) return 1;
    int32_t C = m->res_ch, S = m->skip_ch, B = m->num_bins;

    int64_t* x        = (int64_t*)malloc((size_t)C * T * sizeof(int64_t));
    int64_t* skip_tot = (int64_t*)calloc((size_t)S * T, sizeof(int64_t));
    int64_t* gate     = (int64_t*)malloc((size_t)2 * C * T * sizeof(int64_t));
    int64_t* z        = (int64_t*)malloc((size_t)C * T * sizeof(int64_t));
    int64_t* res_out  = (int64_t*)malloc((size_t)C * T * sizeof(int64_t));
    int64_t* skip_out = (int64_t*)malloc((size_t)S * T * sizeof(int64_t));
    int64_t* hbuf     = (int64_t*)malloc((size_t)S * T * sizeof(int64_t));
    int64_t* logits   = (int64_t*)malloc((size_t)B * T * sizeof(int64_t));
    if (!x || !skip_tot || !gate || !z || !res_out || !skip_out || !hbuf || !logits) {
        free(x); free(skip_tot); free(gate); free(z);
        free(res_out); free(skip_out); free(hbuf); free(logits);
        return 2;
    }

    /* Embedding lookup -> x[C][T] (Q.F). */
    for (int32_t t = 0; t < T; ++t) {
        const int32_t* row = m->embed_qf + (size_t)symbols[t] * C;
        for (int32_t c = 0; c < C; ++c) {
            x[(size_t)c * T + t] = row[c];
        }
    }

    for (int32_t l = 0; l < m->num_layers; ++l) {
        sm_causal_conv(x, T, &m->conv[l], gate);   /* (2C, T) */
        /* Gated activation: z = tanh(f) * sigmoid(g). */
        for (int32_t c = 0; c < C; ++c) {
            const int64_t* fr = gate + (size_t)c * T;
            const int64_t* gr = gate + (size_t)(c + C) * T;
            int64_t* zr = z + (size_t)c * T;
            for (int32_t t = 0; t < T; ++t) {
                int64_t tf = sm_lut_lookup(m->tanh_lut, fr[t]);
                int64_t sg = sm_lut_lookup(m->sigmoid_lut, gr[t]);
                zr[t] = sm_round_shift(tf * sg, SM_RT_FRAC_BITS);
            }
        }
        sm_causal_conv(z, T, &m->residual[l], res_out);
        for (size_t idx = 0; idx < (size_t)C * T; ++idx) res_out[idx] += x[idx];
        sm_causal_conv(z, T, &m->skip[l], skip_out);
        for (size_t idx = 0; idx < (size_t)S * T; ++idx) skip_tot[idx] += skip_out[idx];
        memcpy(x, res_out, (size_t)C * T * sizeof(int64_t));
    }

    /* ReLU -> head1 -> ReLU -> head2. */
    for (size_t idx = 0; idx < (size_t)S * T; ++idx)
        skip_tot[idx] = skip_tot[idx] > 0 ? skip_tot[idx] : 0;
    sm_causal_conv(skip_tot, T, &m->head1, hbuf);
    for (size_t idx = 0; idx < (size_t)S * T; ++idx)
        hbuf[idx] = hbuf[idx] > 0 ? hbuf[idx] : 0;
    sm_causal_conv(hbuf, T, &m->head2, logits);

    /* Transpose (B, T) -> (T, B) int32. */
    for (int32_t t = 0; t < T; ++t)
        for (int32_t o = 0; o < B; ++o)
            logits_out[(size_t)t * B + o] = (int32_t)logits[(size_t)o * T + t];

    free(x); free(skip_tot); free(gate); free(z);
    free(res_out); free(skip_out); free(hbuf); free(logits);
    return 0;
}
