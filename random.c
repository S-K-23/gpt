#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * Handle Random Number Generation and Random Distribution Creation
 */

#ifndef PI
#define PI 3.14159265358979323846
#endif

// Random Seed = 34
static unsigned long long random_state = 34;

/**
 * Shuffles random_state sequence
 */
static unsigned long long random_next () {
    random_state ^= random_state << 21;
    random_state ^= random_state >> 25;
    random_state ^= random_state << 4;
    return random_state;
}

// 1/ 2^53
#define RANDOM_SCALE 1.0 / 9007199254740992.0 

/**
 * Normalizes random number between [0, 1)
 */

static double stand_distro () {
    return (random_next() >> 11) * (RANDOM_SCALE);
}

/**
 * Converts random numbers into Gaussian number
 * 
 * @param mean shift number
 * @param std scale factor
 */

static float random_gauss(float mean, float std) {
    double u1 = stand_distro();
    double u2 = stand_distro();

    if (u1 < 1e-30) u1 = 1e-30;

    double r = sqrt(-2 * log(u1));
    double theta = 2 * PI * u2;

    double z = r * cos(theta);
    
    return mean + std * (float)z;
}

/** 
 * Performes a Fisher-Yates shuffle on dataset
 * 
 * @param arr dataset
 * @param n size of dataset
 */
static void shuffle_ints(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(stand_distro() * (i + 1)); //0 - i

        int t = arr[j];
        arr[j] = arr[i];
        arr[i] = t;

    }
}