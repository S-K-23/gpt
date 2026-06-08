#define N_EMBED 64                     // Embed Dim
#define N_HEAD 4                        // Attn Heads
#define N_LAYER 2                       // Transformer Layers
#define CON_WINDOW 32                   // Sequence Length
#define HEAD_DIM ((N_EMBED) / (N_HEAD)) // Head Internal Dim
#define MLP_DIM (4 * (N_EMBED))         // MLP Expanded Dim

/**
 * Postional struct describing attention state and buffers for a single token
*/
typedef struct
{
    float x_emb[N_EMBED];                     // Embedding for Current Token
    float rms_scale_init;                     // Initial RMS Scale Value
    float x_inp[N_LAYER][N_EMBED];            // Raw Layer Input
    float x_norm_attn[N_LAYER][N_EMBED];      // Normalized Input for Attention
    float rms_scale_attn[N_LAYER];            // RMS for attention [N_EMBED]
    float query[N_LAYER][N_EMBED];            // Query vector for token
    float attnw[N_LAYER][N_HEAD][CON_WINDOW]; // Attention Score for Current Token
    float attn_out[N_LAYER][N_EMBED];         // Raw Attenton Output
    float x_mid[N_LAYER][N_EMBED];            // Residual Data
    float x_norm_mlp[N_LAYER][N_EMBED];       // Normalized Residual Data
    float rms_scale_mlp[N_LAYER];             // RMS for MLP
    float mlp_pre[N_LAYER][MLP_DIM];          // Preactivation Hidden Layer
    float mlp_post[N_LAYER][MLP_DIM];         // Post activation Hidden Layer
    float x_out[N_EMBED];                     // Final Representation Vector for Current Token
} PosVals;
