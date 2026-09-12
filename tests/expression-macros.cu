// MIT probe: parenthesized expression macros, including NVIDIA's IMAD pattern.
#define IMAD(a, b, c) (__mul24((a), (b)) + (c))
#define SQUARE(x) ((x) * (x))
#define RADIUS 8
#define LENGTH (2 * RADIUS + 1)
__global__ void expressionMacros(const int* input, int* output) {
    output[0] = IMAD(input[0], input[2], input[3]);
    output[1] = IMAD(input[1], 2, 1);
    output[2] = IMAD(input[0] + 1, input[2] + 1, 7 - 2);
    output[3] = SQUARE(input[2] + 3);
    output[4] = LENGTH;
}
