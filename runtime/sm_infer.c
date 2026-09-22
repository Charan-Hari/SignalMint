/* SignalMint parity/inference harness.
 *
 * Reads a whitespace-separated list of integer symbols from a file (argv[1]),
 * runs the integer inference engine, and writes the resulting int32 logits
 * (one per line, row-major over (T, num_bins)) to stdout. The Python parity test
 * compares this output to signalmint.quantize.int_infer bit-for-bit.
 *
 * Usage: sm_infer <symbols.txt>
 */
#include "model_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <symbols.txt> [--bench <reps>]\n", argv[0]);
        return 1;
    }
    FILE* f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }

    int32_t capacity = 4096, T = 0;
    int32_t* symbols = (int32_t*)malloc(capacity * sizeof(int32_t));
    long v;
    while (fscanf(f, "%ld", &v) == 1) {
        if (T >= capacity) {
            capacity *= 2;
            symbols = (int32_t*)realloc(symbols, capacity * sizeof(int32_t));
        }
        symbols[T++] = (int32_t)v;
    }
    fclose(f);

    if (T == 0) {
        fprintf(stderr, "no symbols read\n");
        free(symbols);
        return 1;
    }

    int32_t B = signalmint_model.num_bins;
    int32_t* logits = (int32_t*)malloc((size_t)T * B * sizeof(int32_t));

    int bench = (argc >= 4 && strcmp(argv[2], "--bench") == 0);
    if (bench) {
        int reps = atoi(argv[3]);
        if (reps < 1) reps = 1;
        clock_t t0 = clock();
        for (int r = 0; r < reps; ++r) {
            if (sm_forward(&signalmint_model, symbols, T, logits) != 0) {
                fprintf(stderr, "sm_forward failed\n");
                free(symbols); free(logits);
                return 1;
            }
        }
        clock_t t1 = clock();
        double total_us = (double)(t1 - t0) * 1e6 / CLOCKS_PER_SEC;
        /* stdout: total_us reps T  -> Python derives us/sample */
        printf("%.3f %d %d\n", total_us, reps, T);
    } else {
        if (sm_forward(&signalmint_model, symbols, T, logits) != 0) {
            fprintf(stderr, "sm_forward failed\n");
            free(symbols); free(logits);
            return 1;
        }
        for (int32_t i = 0; i < T * B; ++i) {
            printf("%d\n", logits[i]);
        }
    }

    free(symbols);
    free(logits);
    return 0;
}
