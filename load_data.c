#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "data.h"

/**
 * File to load the dataset
 */

char docs[MAX_DOCS][MAX_DOC_LEN];
int curr_doc_count = 0;

/**
 * Loads each line of a file into docs array and null terminates
 *
 * @param file filename
 */
void load_data(char *file)
{
    FILE *fp = fopen(file, "r");
    assert(fp);

    char line[256];

    while ((fgets(line, sizeof(line), fp)) && (curr_doc_count < MAX_DOCS))
    {

        int len = (int)strlen(line);

        while ((len > 0) && ((line[len - 1] == "\n") || (line[len - 1] == "\r")))
        {
            line[len - 1] = 0;
        }

        if (len > 0)
        {
            strncpy(docs[curr_doc_count], line, MAX_DOC_LEN - 1);
            docs[curr_doc_count][MAX_DOC_LEN] = 0;
        }
        curr_doc_count++;
    }

    fclose(fp);
}