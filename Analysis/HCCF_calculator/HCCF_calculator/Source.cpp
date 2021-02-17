#include "pch.h"
#include <iostream>

extern "C"
{

	__declspec(dllexport) double square(double x)
    {
        return x * x;
    }

    double line_sum;
    double image_sum;

    const int rows = 512;
    const int cols = 512;

    __declspec(dllexport) double HHCF(double(*image)[rows][cols], int r) {

    double line_sum = 0.0;
    double image_sum = 0.0;


    for (int i = 0; i < rows; i++) {            // iterates over fast scanning lines
        line_sum = 0.0;

        for (int j = 0; j < cols - r; j++) {    // iterates through fast scanning lines
            line_sum = line_sum + square((*image)[i][j] - (*image)[i][j + r]);
        };
        image_sum = image_sum + (line_sum / (cols - r));
    };
    image_sum = image_sum / rows;
    return image_sum;
    };
};	