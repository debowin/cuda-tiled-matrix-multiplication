#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;


#endif // _MATRIXMUL_H_

