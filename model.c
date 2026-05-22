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

static float *attn_qry[N_LAYER], *d_attn_qry[N_LAYER]; //(128,32)
static float *attn_key[N_LAYER], *d_attn_key[N_LAYER]; //(128,32)
static float *attn_val[N_LAYER], *d_attn_val[N_LAYER]; //(128,32)
static float *attn_out[N_LAYER], *d_attn_out[N_LAYER]; //(128,128)

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
    float x[N_EMBED], tmp[MLP_DIM > N_EMBED ? MLP_DIM : N_EMBED];

    //Embed Tokens, Add Positional Encoding
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
                if (attn_logits[tt] > max) max = attn_logits[tt];
            }

            float sum = 0;
            for (int tt = 0; tt < seq_len; tt++)
            {
                attn_logits[tt] = expf(attn_logits[tt] - max);
                sum += attn_logits[tt];
            }

            float inv = 1.0f / sum;
            for(int tt = 0; tt < seq_len; tt++)
            {
                attn_logits[tt] *= inv;
            }

            for (int tt = 0; tt < seq_len; tt++){
                vals->attnw[layer_in][h][tt] = attn_logits[tt];
            }

            // Weighted Sum of values
            for (int i = 0; i < HEAD_DIM; i++)
            {
                float s = 0;

                for(int tt = 0; tt < seq_len; tt++)
                {
                    s += attn_logits[tt] * kv_vals[layer_in][tt][head_index + i];
                }
                attn_o[head_index + i] = s; 
            }
        }

        memcpy(vals->attn_out[layer_in], attn_o, sizeof(attn_o));

        // Expand Attention Output and Add Residuals
        linear_fwd(attn_o, attn_out[layer_in], N_EMBED, N_EMBED, tmp);
        for(int i = 0; i < N_EMBED; i++)
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
        for (int i = 0; i <MLP_DIM; i++)
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

int main()
{
    init_params();
    return 0;
}