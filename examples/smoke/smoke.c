/*
 * smoke.c — minimal end-to-end proof the shim can load a TTS model and
 * synthesise. Links only audiocpp_c.h + the shared lib (no C++, no ggml).
 *
 *   cc smoke.c -I../../shim -L<build> -laudiocpp -o smoke
 *   ./smoke <model.gguf> <family> <backend> <voice> "text to speak"
 */
#include "audiocpp_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <model.gguf> <family> <backend> <voice> <text>\n", argv[0]);
        return 2;
    }
    const char * model = argv[1];
    const char * family = argv[2];
    const char * backend = argv[3];
    const char * voice = argv[4];
    const char * text = argv[5];

    char err[512] = {0};
    printf("shim: %s\n", audiocpp_version());
    printf("loading %s (family=%s backend=%s)...\n", model, family, backend);

    audiocpp_ctx * ctx = audiocpp_load(model, family, backend, 0, 1,
                                       "{\"max_new_tokens\":\"512\"}", err, sizeof(err));
    if (ctx == NULL) {
        fprintf(stderr, "load failed: %s\n", err);
        return 1;
    }

    float * samples = NULL;
    int n = 0, rate = 0, ch = 0;
    printf("synthesising voice=%s...\n", voice);
    int rc = audiocpp_synthesize(ctx, "tts", text, voice,
                                 NULL, 0, 0,
                                 "{\"seed\":\"1234\",\"do_sample\":\"false\"}",
                                 &samples, &n, &rate, &ch, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "synthesize failed (code %d): %s\n", rc, err);
        audiocpp_free(ctx);
        return 1;
    }

    double seconds = (rate > 0) ? (double)n / (double)(rate * (ch > 0 ? ch : 1)) : 0.0;
    printf("OK: %d samples, %d Hz, %d ch  (%.2f s of audio)\n", n, rate, ch, seconds);

    /* sanity: non-empty, plausible rate */
    int ok = (n > 0 && rate == 24000);

    audiocpp_free_samples(samples);
    audiocpp_free(ctx);
    if (!ok) {
        fprintf(stderr, "FAIL: expected non-empty PCM at 24000 Hz\n");
        return 1;
    }
    printf("smoke: PASS\n");
    return 0;
}
