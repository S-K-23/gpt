#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "model.c"

void train (int steps) {
    float l_rate = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float epsilon = 1e-8f;

    for (int s = 0; s < steps; s++)
    {
        char *doc = docs[s % curr_doc_count];
        int doc_len = (int)strlen(doc);


    }
}

int main() {
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

    int num_steps = 5000;
    train(num_steps);

}