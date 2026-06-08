#include <stdlib.h>
#include <assert.h>

#include "model.h"
#include "random.c"
#include "character_tokenizer.c"
#include "model_functions.c"

// Token + Position Embeddings
static float *wte, *d_wte;
static float *wpe, *d_wpe;
static float *lm_head, *d_lm_head;

// Attention Mechanism: QKV + Output
static float *attn_qry[N_LAYER], *d_attn_qry[N_LAYER]; //(128,32)
static float *attn_key[N_LAYER], *d_attn_key[N_LAYER]; //(128,32)
static float *attn_val[N_LAYER], *d_attn_val[N_LAYER]; //(128,32)
static float *attn_out[N_LAYER], *d_attn_out[N_LAYER]; //(128,128)

// MLP Layers
static float *mlp_exp[N_LAYER], *d_mlp_exp[N_LAYER]; //(128, 512)
static float *mlp_con[N_LAYER], *d_mlp_con[N_LAYER]; //(512, 128)

// Optimizer State
static float *adam_m_wte, *adam_v_wte;
static float *adam_m_wpe, *adam_v_wpe;
static float *adam_m_lm, *adam_v_lm;

static float *adam_m_qry[N_LAYER], *adam_v_qry[N_LAYER]; //(128,32)
static float *adam_m_key[N_LAYER], *adam_v_key[N_LAYER]; //(128,32)
static float *adam_m_val[N_LAYER], *adam_v_val[N_LAYER]; //(128,32)
static float *adam_m_out[N_LAYER], *adam_v_out[N_LAYER]; //(128,128)

static float *adam_m_exp[N_LAYER], *adam_v_exp[N_LAYER]; //(128, 512)
static float *adam_m_con[N_LAYER], *adam_v_con[N_LAYER]; //(512, 128)

static int num_params = 0;

static float *make_param(int size, float std)
{
    float *param = (float *)calloc(size, sizeof(float));
    assert(param);

    for (int i = 0; i < size; i++)
    {
        param[i] = random_gauss(0, std);
    }

    num_params += size;
    return param;
}

static float *make_zero_param(int size)
{
    float *param = calloc(size, sizeof(float));
    assert(param);

    return (float *)param;
}

static void init_params()
{
    int emb_s = vocab_size * N_EMBED;
    int pemb_s = CON_WINDOW * N_EMBED;
    int attn_s = N_EMBED * N_EMBED;
    int mlp_s = MLP_DIM * N_EMBED;

    wte = make_param(emb_s, 0.02f);
    d_wte = make_zero_param(emb_s);
    adam_m_wte = make_zero_param(emb_s);
    adam_v_wte = make_zero_param(emb_s);

    wpe = make_param(pemb_s, 0.02f);
    d_wpe = make_zero_param(pemb_s);
    adam_m_wpe = make_zero_param(pemb_s);
    adam_v_wpe = make_zero_param(pemb_s);

    lm_head = make_param(emb_s, 0.02f);
    d_lm_head = make_zero_param(emb_s);
    adam_m_lm = make_zero_param(emb_s);
    adam_v_lm = make_zero_param(emb_s);

    printf("Loaded Token, Position, and LM Head\n");

    for (int i = 0; i < N_LAYER; i++)
    {
        attn_qry[i] = make_param(attn_s, 0.02f);
        d_attn_qry[i] = make_zero_param(attn_s);
        adam_m_qry[i] = make_zero_param(attn_s);
        adam_v_qry[i] = make_zero_param(attn_s);

        attn_key[i] = make_param(attn_s, 0.02f);
        d_attn_key[i] = make_zero_param(attn_s);
        adam_m_key[i] = make_zero_param(attn_s);
        adam_v_key[i] = make_zero_param(attn_s);

        attn_val[i] = make_param(attn_s, 0.02f);
        d_attn_val[i] = make_zero_param(attn_s);
        adam_m_val[i] = make_zero_param(attn_s);
        adam_v_val[i] = make_zero_param(attn_s);

        attn_out[i] = make_param(attn_s, 0.02f);
        d_attn_out[i] = make_zero_param(attn_s);
        adam_m_out[i] = make_zero_param(attn_s);
        adam_v_out[i] = make_zero_param(attn_s);

        mlp_exp[i] = make_param(mlp_s, 0.02f);
        d_mlp_exp[i] = make_zero_param(mlp_s);
        adam_m_exp[i] = make_zero_param(mlp_s);
        adam_v_exp[i] = make_zero_param(mlp_s);

        mlp_con[i] = make_param(mlp_s, 0.02f);
        d_mlp_con[i] = make_zero_param(mlp_s);
        adam_m_con[i] = make_zero_param(mlp_s);
        adam_v_con[i] = make_zero_param(mlp_s);
    }

    printf("Initialized Model with | %d | params\n", num_params);
}

PosVals saved[CON_WINDOW];                    // Saved Positon Values for Each Token
float saved_probs[CON_WINDOW][MAX_CHARS + 1]; // Saved Probability Distribution for Each Token

// Key Value(KV) Cache
float kv_keys[N_LAYER][CON_WINDOW][N_EMBED];
float kv_vals[N_LAYER][CON_WINDOW][N_EMBED];

float dk_accum[N_LAYER][CON_WINDOW][N_EMBED];
float dv_accum[N_LAYER][CON_WINDOW][N_EMBED];

void gpt_forward(int token_id, int pos_id, float *logits_out, PosVals *vals)
{
    assert(token_id >= 0 && token_id < vocab_size);
    float x[N_EMBED], tmp[MLP_DIM > N_EMBED ? MLP_DIM : N_EMBED];

    // Embed Tokens, Add Positional Encoding
    for (int i = 0; i < N_EMBED; i++)
    {
        x[i] = wte[token_id * N_EMBED + i] + wpe[pos_id * N_EMBED + i];
    }
    memcpy(vals->x_emb, x, sizeof(x));

    // Normalize X
    vals->rms_scale_init = rmsnorm_fwd(x, N_EMBED, x);

    // Transformer Layer
    for (int layer_in = 0; layer_in < N_LAYER; layer_in++)
    {
        memcpy(vals->x_inp[layer_in], x, sizeof(x));

        // Normalize X
        float x_norm[N_EMBED];
        vals->rms_scale_attn[layer_in] = rmsnorm_fwd(x, N_EMBED, x_norm);
        memcpy(vals->x_norm_attn[layer_in], x_norm, sizeof(x_norm));

        // Project to Query, Key, Value
        float qry[N_EMBED], key[N_EMBED], val[N_EMBED];
        linear_fwd(x_norm, attn_qry[layer_in], N_EMBED, N_EMBED, qry);
        linear_fwd(x_norm, attn_key[layer_in], N_EMBED, N_EMBED, key);
        linear_fwd(x_norm, attn_val[layer_in], N_EMBED, N_EMBED, val);
        memcpy(vals->query[layer_in], qry, sizeof(qry));

        // Save K,V in KV Cache, Save # of Tokens Seen
        memcpy(kv_keys[layer_in][pos_id], key, sizeof(key));
        memcpy(kv_vals[layer_in][pos_id], val, sizeof(val));
        int seq_len = pos_id + 1;

        float scale = 1.0f / sqrt((float)N_EMBED / (float)N_HEAD);
        float attn_o[N_EMBED];

        for (int h = 0; h < N_HEAD; h++)
        {
            int head_index = h * HEAD_DIM;

            // Compute attention logits (Current query dot product with cached historical keys)
            float attn_logits[CON_WINDOW];
            for (int tt = 0; tt < seq_len; tt++)
            {
                float dp = 0;
                for (int j = 0; j < HEAD_DIM; j++)
                {
                    dp += qry[head_index + j] * kv_keys[layer_in][tt][head_index + j];
                }
                attn_logits[tt] = dp * scale;
            }
            // Softmax to Get Attention Weights
            float max = attn_logits[0];
            for (int tt = 1; tt < seq_len; tt++)
            {
                if (attn_logits[tt] > max)
                    max = attn_logits[tt];
            }

            float sum = 0;
            for (int tt = 0; tt < seq_len; tt++)
            {
                attn_logits[tt] = expf(attn_logits[tt] - max);
                sum += attn_logits[tt];
            }

            float inv = 1.0f / sum;
            for (int tt = 0; tt < seq_len; tt++)
            {
                attn_logits[tt] *= inv;
            }

            for (int tt = 0; tt < seq_len; tt++)
            {
                vals->attnw[layer_in][h][tt] = attn_logits[tt];
            }

            // Compute attention output (Weighted sum of cached values using attention probabilities (logits))
            for (int i = 0; i < HEAD_DIM; i++)
            {
                float s = 0;

                for (int tt = 0; tt < seq_len; tt++)
                {
                    s += attn_logits[tt] * kv_vals[layer_in][tt][head_index + i];
                }
                attn_o[head_index + i] = s;
            }
        }

        memcpy(vals->attn_out[layer_in], attn_o, sizeof(attn_o));

        // Expand Attention Output and Add Residuals
        linear_fwd(attn_o, attn_out[layer_in], N_EMBED, N_EMBED, tmp);
        for (int i = 0; i < N_EMBED; i++)
        {
            x[i] = tmp[i] + vals->x_inp[layer_in][i];
        }
        memcpy(vals->x_mid[layer_in], x, sizeof(x));

        // Normalize
        float xn_m[N_EMBED];
        vals->rms_scale_mlp[layer_in] = rmsnorm_fwd(x, N_EMBED, xn_m);
        memcpy(vals->x_norm_mlp[layer_in], xn_m, sizeof(xn_m));

        // First MLP Layer
        float h1[MLP_DIM];
        linear_fwd(xn_m, mlp_exp[layer_in], MLP_DIM, N_EMBED, h1);
        memcpy(vals->mlp_pre[layer_in], h1, MLP_DIM * sizeof(float));

        // Squared ReLU Activation After First Layer
        float h2[MLP_DIM];
        for (int i = 0; i < MLP_DIM; i++)
        {
            h2[i] = h1[i] > 0 ? h1[i] * h1[i] : 0;
        }
        memcpy(vals->mlp_post[layer_in], h2, MLP_DIM * sizeof(float));

        // Second MLP Layer and Residual
        linear_fwd(h2, mlp_con[layer_in], N_EMBED, MLP_DIM, tmp);

        for (int i = 0; i < N_EMBED; i++)
        {
            x[i] = tmp[i] + vals->x_mid[layer_in][i];
        }
    }

    // Final Output Projection to Vocabulary
    memcpy(vals->x_out, x, sizeof(x));
    linear_fwd(x, lm_head, vocab_size, N_EMBED, logits_out);
}

void gpt_backward(int n, const int *tokens, const int *targets)
{
    memset(dk_accum, 0, sizeof(dk_accum));
    memset(dv_accum, 0, sizeof(dv_accum));
    float inv_n = 1.0f / n;

    for (int pos = n - 1; pos >= 0; pos--)
    {
        PosVals *vals = &saved[pos];
        int seq_len = pos + 1;

        float dl[vocab_size];
        for (int i = 0; i < vocab_size; i++)
        {
            dl[i] = (saved_probs[pos][i] - (i == targets[pos] ? 1.0f : 0.0f)) * inv_n;
        }

        float dx[N_EMBED];
        memset(dx, 0, sizeof(dx));
        linear_bwd_x(lm_head, dl, vocab_size, N_EMBED, dx);
        linear_bwd_w(vals->x_out, dl, vocab_size, N_EMBED, d_lm_head);

        for (int li = N_LAYER - 1; li >= 0; li--)
        {

            float d_h2[MLP_DIM];
            memset(d_h2, 0, sizeof(d_h2));
            linear_bwd_x(mlp_con[li], dx, N_EMBED, MLP_DIM, d_h2);
            linear_bwd_w(vals->mlp_post[li], dx, N_EMBED, MLP_DIM, d_mlp_con[li]);

            float d_h1[MLP_DIM];
            for (int i = 0; i < MLP_DIM; i++)
            {
                d_h1[i] = vals->mlp_pre[li][i] > 0 ? 2.0f * vals->mlp_pre[li][i] * d_h2[i] : 0;
            }

            float d_xn_mlp[N_EMBED];
            memset(d_xn_mlp, 0, sizeof(d_xn_mlp));
            linear_bwd_x(mlp_exp[li], d_h1, MLP_DIM, N_EMBED, d_xn_mlp);
            linear_bwd_w(vals->x_norm_mlp[li], d_h1, MLP_DIM, N_EMBED, d_mlp_exp[li]);

            float d_x_mid[N_EMBED];
            memset(d_x_mid, 0, sizeof(d_x_mid));
            rmsnorm_bwd(vals->x_mid[li], vals->rms_scale_mlp[li], d_xn_mlp, N_EMBED, d_x_mid);

            for (int i = 0; i < N_EMBED; i++)
            {
                dx[i] += d_x_mid[i];
            }

            float d_ao[N_EMBED];
            memset(d_ao, 0, sizeof(d_ao));
            linear_bwd_x(attn_out[li], dx, N_EMBED, N_EMBED, d_ao);
            linear_bwd_w(vals->attn_out[li], dx, N_EMBED, N_EMBED, d_attn_out[li]);

            float d_q[N_EMBED];
            memset(d_q, 0, sizeof(d_q));
            float scale = 1.0f / sqrt((float)N_EMBED / (float)N_HEAD);

            for (int h = 0; h < N_HEAD; h++)
            {
                int hs = h * HEAD_DIM;

                float d_aw[CON_WINDOW];
                memset(d_aw, 0, sizeof(d_aw));

                for (int j = 0; j < HEAD_DIM; j++)
                {
                    for (int tt = 0; tt < seq_len; tt++)
                    {
                        d_aw[tt] += d_ao[hs + j] * kv_vals[li][tt][hs + j];
                        dv_accum[li][tt][hs + j] += vals->attnw[li][h][tt] * d_ao[hs + j];
                    }
                }

                float dot = 0;
                for (int tt = 0; tt < seq_len; tt++)
                {
                    dot += d_aw[tt] * vals->attnw[li][h][tt];
                }

                float d_al[CON_WINDOW];
                for (int tt = 0; tt < seq_len; tt++)
                {
                    d_al[tt] = vals->attnw[li][h][tt] * (d_aw[tt] - dot);
                }

                for (int tt = 0; tt < seq_len; tt++)
                {
                    for (int j = 0; j < HEAD_DIM; j++)
                    {
                        d_q[hs + j] += d_al[tt] * kv_keys[li][tt][hs + j] * scale;
                        dk_accum[li][tt][hs + j] += d_al[tt] * vals->query[li][hs + j] * scale;
                    }
                }
            }

            float d_xn[N_EMBED];
            memset(d_xn, 0, sizeof(d_xn));

            linear_bwd_x(attn_qry[li], d_q, N_EMBED, N_EMBED, d_xn);
            linear_bwd_w(vals->x_norm_attn[li], d_q, N_EMBED, N_EMBED, d_attn_qry[li]);

            linear_bwd_x(attn_key[li], dk_accum[li][pos], N_EMBED, N_EMBED, d_xn);
            linear_bwd_w(vals->x_norm_attn[li], dk_accum[li][pos], N_EMBED, N_EMBED, d_attn_key[li]);

            linear_bwd_x(attn_val[li], dv_accum[li][pos], N_EMBED, N_EMBED, d_xn);
            linear_bwd_w(vals->x_norm_attn[li], dv_accum[li][pos], N_EMBED, N_EMBED, d_attn_val[li]);

            float d_x_in[N_EMBED];
            memset(d_x_in, 0, sizeof(d_x_in));
            rmsnorm_bwd(vals->x_inp[li], vals->rms_scale_attn[li], d_xn, N_EMBED, d_x_in);

            for (int i = 0; i < N_EMBED; i++)
            {
                dx[i] = dx[i] + d_x_in[i];
            }
        }

        float d_embed[N_EMBED];
        memset(d_embed, 0, sizeof(d_embed));
        rmsnorm_bwd(vals->x_emb, vals->rms_scale_init, dx, N_EMBED, d_embed);

        int tk = tokens[pos];
        assert(tk >= 0 && tk < vocab_size);
        for (int i = 0; i < N_EMBED; i++)
        {
            d_wte[tk * N_EMBED + i] += d_embed[i];
            d_wpe[pos * N_EMBED + i] += d_embed[i];
        }
    }
}

void adam_update(float *p, float *g, float *m, float *v, int sz,
                 float lr, float b1, float b2, float eps, int step)
{
    float b1c = 1.0f - powf(b1, step + 1);
    float b2c = 1.0f - powf(b2, step + 1);

    for (int i = 0; i < sz; i++)
    {
        m[i] = b1 * m[i] + (1 - b1) * g[i];

        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];

        p[i] -= lr * (m[i] / b1c) / (sqrtf(v[i] / b2c) + eps);

        g[i] = 0;
    }
}

int weighted_choice(const float *w, int n)
{
    float total = 0;
    for (int i = 0; i < n; i++)
    {
        total += w[i];
    }

    float r = (float)stand_distro() * total;
    float c = 0;

    for (int i = 0; i < n; i++)
    {
        c += w[i];
        if (r < c)
        {
            return i;
        }
    }

    return n - 1;
}
