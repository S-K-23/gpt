#include <string.h>
#include <stdlib.h>

#include "data.h"
#include "load_data.c"

/**
 * Tokenizes dataset by characters
 */

static char uniq_char_arr[MAX_CHARS];
static int vocab_size, BOS, num_uchars = 0;

/**
 * Finds ID of a character in uniq_char_arr
 * 
 * @param c character to lookup
 */
int char_to_id(char c) {
    for (int i = 0; i < MAX_CHARS; i++) {
        if (uniq_char_arr[i] == c) return i;
    }
    return -1;
}

/**
 * Compares the value of two characters
 * 
 * @param a first char
 * @param b second char
 */

int cmp_char(const void *a, const void *b) {
    return (*(const char *) a) - (*(const char *)b);
}

/**
 * Provides Unique ID to each unqiue character in dataset
 */
void build_tokenizer () {
    int seen[256] = {0};

    for (int d = 0; d < curr_doc_count; d++) {
        for (int i = 0; docs[d][i]; i++) {
            seen[(unsigned char)docs[d][i]] = 1;
        }
    }

    for (int i = 0; i < 256; i++) {
        if (seen[i]) {
            uniq_char_arr[num_uchars++] = (char)i;
        }
    }
    qsort(uniq_char_arr, num_uchars, sizeof(char), cmp_char);

    BOS = num_uchars;
    vocab_size = num_uchars + 1;
}
