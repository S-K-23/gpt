#include <math.h>

inline void linear_fwd(const float *restrict x, const float *restrict w, int out_size,
                       int in_size, float *restrict out)
{
    for (int r = 0; r < out_size; r++)
    {
        float s = 0;
        const float *wr = w + r * in_size;

        for (int c = 0; c < in_size; c++)
        {
            s += wr[c] * x[c];
        }

        out[r] = s;
    }
}

inline float rmsnorm_fwd(const float *x, int n, float *out)
{
    float ms = 0;

    for (int i = 0; i < n; i++)
    {
        ms += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf(ms + 1e-5f);

    for (int i = 0; i < n; i++)
    {
        out[i] = x[i] * scale;
    }
    return scale;
}

inline void softmax_fwd(const float *logits, int n, float *probs)
{
    float max = logits[0];
    for (int i = 0; i < n; i++)
    {
        if (logits[i] > max)
        {
            max = logits[i];
        }
    }
    // EXP(x - max)
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        probs[i] = expf(logits[i] - max);
        sum += probs[i];
    }
    // EXP(x-max) / sum(EXP(x-max))
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++)
        probs[i] *= inv;
}

inline void linear_bwd_x (){
    
}