#include <math.h>

void linear_fwd(const float *restrict x, const float *restrict w, int out_size,
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

float rmsnorm_fwd(const float *x, int n, float *out)
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

void softmax_fwd(const float *logits, int n, float *probs)
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

/**
 * The gradients of the current layer (dx) are increased by the sum of the gradients
 * in subsequent layer (d_out) * the wieghts of the current layer (w[r * n_in + c])
 */
void linear_bwd_x(const float *restrict w, const float *restrict d_out, int n_out,
                  int n_in, float *restrict dx)
{
    for (int c = 0; c < n_in; c++)
    {
        float s = 0;
        for (int r = 0; r < n_out; r++)
        {
            s += d_out[r] * w[r * n_in + c];
        }

        dx[c] += s;
    }
}

/**
 * Accumulates the gradients for the weights matrix (dw) by summing the
 * gradient error of each neuron (d_out[r]) multiplied by its corresponding
 * original input element (x[c]).
 */

void linear_bwd_w(const float *restrict x, const float *restrict d_out, int n_out,
                  int n_in, float *restrict dw)
{
    for (int r = 0; r < n_out; r++)
    {
        float dr = d_out[r];
        float *dwr = dw + r * n_in;

        for (int c = 0; c < n_in; c++)
        {
            dwr[c] += dr * x[c];
        }
    }
}

void rmsnorm_bwd(const float *x, float scale, const float *d_out,
                  int n, float *dx)
{
    float dot = 0;
    for (int i = 0; i < n; i++)
    {
        dot += d_out[i] * x[i];
    }

    float coef = scale * scale * scale / n;

    for (int i = 0; i < n; i++)
    {
        dx[i] += scale * d_out[i] - coef * x[i] * dot;
    }
    
}