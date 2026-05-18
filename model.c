#include <stdlib.h>
#include <assert.h>

#include "model.h"
#include "random.c"

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

static float *make_param(int size, float std) {
    float *param = (float *) calloc(size, sizeof(float));
    assert(param);
    
    for (int i =0; i < size; i++) {
        param[i] = random_gauss(0, std);
    }

    num_params += size;
    return param;
}

static float *make_zero_param(int size) {
    float *param = calloc(size, sizeof(float));
    assert(param);

    return (float *) param;
}
