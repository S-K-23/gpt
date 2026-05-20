#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

void linear_fwd(const float *restrict x, const float *restrict w, int out_size, int in_size, float *restrict out);
float rmsnorm_fwd(const float *x, int n, float *out);
void softmax_fwd(const float *logits, int n, float *probs);

// --- EPSILON FOR FLOATING POINT COMPARISONS ---
#define EPSILON 1e-5f

// --- HELPER MACRO FOR TESTING ---
#define RUN_TEST(test_func) \
    do { \
        printf("Running %s...", #test_func); \
        test_func(); \
        printf(" PASSED\n"); \
    } while (0)

// --- 1. TESTS FOR LINEAR_FWD ---
void test_linear_fwd_basic() {
    // 2x3 weight matrix (Flattened row-major)
    // [ 1.0, 2.0, 3.0 ]
    // [ 4.0, 5.0, 6.0 ]
    const float w[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float x[3] = {0.5f, 2.0f, -1.0f};
    float out[2] = {0};

    linear_fwd(x, w, 2, 3, out);

    // Row 0: (1.0*0.5) + (2.0*2.0) + (3.0*-1.0) = 0.5 + 4.0 - 3.0 = 1.5
    // Row 1: (4.0*0.5) + (5.0*2.0) + (6.0*-1.0) = 2.0 + 10.0 - 6.0 = 6.0
    assert(fabsf(out[0] - 1.5f) < EPSILON);
    assert(fabsf(out[1] - 6.0f) < EPSILON);
}

void test_linear_fwd_zeros() {
    const float w[4] = {0, 0, 0, 0};
    const float x[2] = {5.5f, 10.2f};
    float out[2] = {99.0f, 99.0f}; // Seed with garbage to ensure overwrite

    linear_fwd(x, w, 2, 2, out);

    assert(fabsf(out[0] - 0.0f) < EPSILON);
    assert(fabsf(out[1] - 0.0f) < EPSILON);
}


// --- 2. TESTS FOR RMSNORM_FWD ---
void test_rmsnorm_fwd_basic() {
    const float x[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float out[4] = {0};

    float scale = rmsnorm_fwd(x, 4, out);

    // ms = 2^2 + 2^2 + 2^2 + 2^2 = 16
    // scale = 1 / sqrt(16 + 1e-5) approx 1/4 = 0.25
    // out = 2 * 0.25 = 0.5
    assert(fabsf(scale - (1.0f / sqrtf(16.0f + 1e-5f))) < EPSILON);
    for (int i = 0; i < 4; i++) {
        assert(fabsf(out[i] - (2.0f * scale)) < EPSILON);
    }
}

void test_rmsnorm_fwd_zeros() {
    const float x[3] = {0.0f, 0.0f, 0.0f};
    float out[3] = {0};

    float scale = rmsnorm_fwd(x, 3, out);

    // ms = 0. scale should be 1 / sqrt(1e-5) approx 316.22
    // out should be 0 * scale = 0 (prevents division by zero crash)
    assert(fabsf(out[0] - 0.0f) < EPSILON);
    assert(fabsf(out[1] - 0.0f) < EPSILON);
    assert(fabsf(out[2] - 0.0f) < EPSILON);
    assert(!isnan(scale) && !isinf(scale));
}


// --- 3. TESTS FOR SOFTMAX_FWD ---
void test_softmax_fwd_basic() {
    const float logits[3] = {1.0f, 2.0f, 3.0f};
    float probs[3] = {0};

    softmax_fwd(logits, 3, probs);

    // Check that probabilities sum up to 1.0
    float sum = probs[0] + probs[1] + probs[2];
    assert(fabsf(sum - 1.0f) < EPSILON);

    // Hardcoded expected values:
    // max = 3. exp(1-3)=0.1353, exp(2-3)=0.3678, exp(3-3)=1.0. Sum = 1.5031
    // probs: 0.1353/1.5031=0.0900, 0.3678/1.5031=0.2447, 1.0/1.5031=0.6652
    assert(fabsf(probs[0] - 0.09003057f) < EPSILON);
    assert(fabsf(probs[1] - 0.24472847f) < EPSILON);
    assert(fabsf(probs[2] - 0.66524096f) < EPSILON);
}

void test_softmax_fwd_numerical_stability() {
    // If your code didn't subtract 'max', expf(1000.0f) would overflow to Infinity
    const float logits[2] = {1000.0f, 1000.0f};
    float probs[2] = {0};

    softmax_fwd(logits, 2, probs);

    // Both values are identical, so they should split probabilities 50/50 safely
    assert(fabsf(probs[0] - 0.5f) < EPSILON);
    assert(fabsf(probs[1] - 0.5f) < EPSILON);
}

void test_softmax_fwd_extreme_negative() {
    const float logits[2] = {-500.0f, -500.0f};
    float probs[2] = {0};

    softmax_fwd(logits, 2, probs);

    // Even with deep negative values, max subtraction protects it
    assert(fabsf(probs[0] - 0.5f) < EPSILON);
    assert(fabsf(probs[1] - 0.5f) < EPSILON);
}


// --- MAIN TEST RUNNER ---
int main() {
    printf("=== STARTING TRANSFORMER LAYER TESTS ===\n");

    // Linear Forward Tests
    RUN_TEST(test_linear_fwd_basic);
    RUN_TEST(test_linear_fwd_zeros);

    // RMSNorm Tests
    RUN_TEST(test_rmsnorm_fwd_basic);
    RUN_TEST(test_rmsnorm_fwd_zeros);

    // Softmax Tests
    RUN_TEST(test_softmax_fwd_basic);
    RUN_TEST(test_softmax_fwd_numerical_stability);
    RUN_TEST(test_softmax_fwd_extreme_negative);

    printf("=== ALL TESTS PASSED SUCCESSFULLY ===\n");
    return 0;
}
