#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "model.c"

void train(int steps)
{
    float l_rate = 1e-3f;   
    float beta1 = 0.9f;
    float beta2 = 0.95f;    
    float epsilon = 1e-8f;
    int warmup_steps = 500; 

    for (int s = 0; s < steps; s++)
    {
        // Tokenize
        char *doc = docs[s % curr_doc_count];
        int doc_len = (int)strlen(doc);

        int tokens[MAX_DOC_LEN + 2], targets[CON_WINDOW];
        tokens[0] = BOS;

        for (int i = 0; i < doc_len; i++)
        {
            tokens[i + 1] = char_to_id(doc[i]);
        }
        tokens[doc_len + 1] = BOS;

        int n = CON_WINDOW < (doc_len + 1) ? CON_WINDOW : (doc_len + 1);

        // Forward Pass
        float total_loss = 0;
        float logits[MAX_CHARS + 1];

        for (int pos = 0; pos < n; pos++)
        {
            targets[pos] = tokens[pos + 1];

            gpt_forward(tokens[pos], pos, logits, &saved[pos]);

            softmax_fwd(logits, vocab_size, saved_probs[pos]);

            total_loss += -logf(saved_probs[pos][targets[pos]] + 1e-30f);
        }

        float loss = total_loss / n;

        // Backprop
        gpt_backward(n, tokens, targets);

        // Learning rate with warmup then cosine decay
        float lr_t;
        if (s < warmup_steps) {
            lr_t = l_rate * (s + 1) / warmup_steps;
        } else {
            float progress = (float)(s - warmup_steps) / (float)(steps - warmup_steps);
            lr_t = l_rate * (0.1f + 0.9f * 0.5f * (1.0f + cosf((float)M_PI * progress)));
        }

        int es = vocab_size * N_EMBED;
        int ps = CON_WINDOW * N_EMBED;
        int as = N_EMBED * N_EMBED;
        int ms = MLP_DIM * N_EMBED;

        adam_update(wte, d_wte, adam_m_wte, adam_v_wte, es, lr_t, beta1, beta2, epsilon, s);
        adam_update(wpe, d_wpe, adam_m_wpe, adam_v_wpe, ps, lr_t, beta1, beta2, epsilon, s);
        adam_update(lm_head, d_lm_head, adam_m_lm, adam_v_lm, es, lr_t, beta1, beta2, epsilon, s);

        for (int i = 0; i < N_LAYER; i++)
        {
            adam_update(attn_qry[i], d_attn_qry[i], adam_m_qry[i], adam_v_qry[i], as, lr_t, beta1, beta2, epsilon, s);
            adam_update(attn_key[i], d_attn_key[i], adam_m_key[i], adam_v_key[i], as, lr_t, beta1, beta2, epsilon, s);
            adam_update(attn_val[i], d_attn_val[i], adam_m_val[i], adam_v_val[i], as, lr_t, beta1, beta2, epsilon, s);
            adam_update(attn_out[i], d_attn_out[i], adam_m_out[i], adam_v_out[i], as, lr_t, beta1, beta2, epsilon, s);

            adam_update(mlp_exp[i], d_mlp_exp[i], adam_m_exp[i], adam_v_exp[i], ms, lr_t, beta1, beta2, epsilon, s);
            adam_update(mlp_con[i], d_mlp_con[i], adam_m_con[i], adam_v_con[i], ms, lr_t, beta1, beta2, epsilon, s);
        }

        // Print every 500 steps
        if ((s + 1) % 500 == 0 || s == 0) {
            printf("Step %5d / %5d | loss %.4f | lr %.2e\n", s + 1, steps, loss, lr_t);
        }
    }
}

void inference(float temp)
{
    for (int si = 0; si < 20; si++)
    {
        char sample[CON_WINDOW + 1];
        int slen = 0, token_id = BOS;
        PosVals tmp_act;

        for (int pos = 0; pos < CON_WINDOW; pos++)
        {
            float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];

            gpt_forward(token_id, pos, logits, &tmp_act);

            float inv_t = 1.0f / temp;
            for (int i = 0; i < vocab_size; i++)
            {
                logits[i] *= inv_t;
            }

            softmax_fwd(logits, vocab_size, probs);
            token_id = weighted_choice(probs, vocab_size);

            if (token_id == BOS)
                break;
            if (token_id < num_uchars)
            {
                sample[slen++] = uniq_char_arr[token_id];
            }
        }

        sample[slen] = '\0';
        printf("sample %2d: %s\n", si + 1, sample);

        memset(kv_keys, 0, sizeof(kv_keys));
        memset(kv_vals, 0, sizeof(kv_vals));
    }
}

void free_params(void)
{
    free(wte);
    free(d_wte);
    free(adam_m_wte);
    free(adam_v_wte);
    free(wpe);
    free(d_wpe);
    free(adam_m_wpe);
    free(adam_v_wpe);
    free(lm_head);
    free(d_lm_head);
    free(adam_m_lm);
    free(adam_v_lm);

    for (int i = 0; i < N_LAYER; i++)
    {
        free(attn_qry[i]);
        free(d_attn_qry[i]);
        free(adam_m_qry[i]);
        free(adam_v_qry[i]);
        free(attn_key[i]);
        free(d_attn_key[i]);
        free(adam_m_key[i]);
        free(adam_v_key[i]);
        free(attn_val[i]);
        free(d_attn_val[i]);
        free(adam_m_val[i]);
        free(adam_v_val[i]);
        free(attn_out[i]);
        free(d_attn_out[i]);
        free(adam_m_out[i]);
        free(adam_v_out[i]);
        free(mlp_exp[i]);
        free(d_mlp_exp[i]);
        free(adam_m_exp[i]);
        free(adam_v_exp[i]);
        free(mlp_con[i]);
        free(d_mlp_con[i]);
        free(adam_m_con[i]);
        free(adam_v_con[i]);
    }
}

int main()
{
    setvbuf(stdout, NULL, _IONBF, 0);

    load_data("input.txt");

    int *doc_order = (int *)malloc(curr_doc_count * sizeof(int));

    for (int i = 0; i < curr_doc_count; i++)
    {
        doc_order[i] = i;
    }
    shuffle_ints(doc_order, curr_doc_count);

    char (*docs_tmp)[MAX_DOC_LEN] = malloc((size_t)curr_doc_count * MAX_DOC_LEN);
    for (int i = 0; i < curr_doc_count; i++)
    {
        memcpy(docs_tmp[i], docs[doc_order[i]], MAX_DOC_LEN);
    }
    memcpy(docs, docs_tmp, (size_t)curr_doc_count * MAX_DOC_LEN);
    free(docs_tmp);
    free(doc_order);

    printf("Number of Docs: %d\n", curr_doc_count);

    build_tokenizer();
    printf("=================================\n");
    printf("Vocab Size: %d\n", vocab_size);

    init_params();

    int num_steps = 50000;
    train(num_steps);

    printf("=================================\n");
    printf("Inference\n");
    float temperature = 0.6f;
    inference(temperature);

    free_params();
    return 0;
}