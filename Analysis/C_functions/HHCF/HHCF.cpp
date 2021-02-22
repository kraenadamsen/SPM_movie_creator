#include <iostream>
#include <vector>


#ifdef __cplusplus
extern "C" {
#endif

int main() {
    std::cout << "Opened" << std::endl;
}



double square(double x)
    {
        return x * x;
    }

double line_sum;
double image_sum;


/* 

std::vector<double> HHCF (double(*image)[], int im_size, int R) {

    double line_sum = 0.0;
    double image_sum_val = 0.0;
    std::vector<double> image_sum(R);


    for (int r = 1; r < R; r++){
        image_sum_val = 0.0;

        for (int i = 0; i < im_size; i++) {            // iterates over fast scanning lines
            line_sum = 0.0;

            for (int j = 0; j < im_size - r; j++) {    // iterates through fast scanning lines
                line_sum = line_sum + square((*image)[i*im_size + j] - (*image)[i*im_size + j + r]);
            };
            image_sum_val= image_sum_val + (line_sum / (im_size - r));
        };

        image_sum[r] =  (image_sum_val / im_size);
    }
    return image_sum;
}

 */


double HHCF (double(*image)[], int im_size, int r) {

    double line_sum = 0.0;
    double image_sum = 0.0;


    
    image_sum = 0.0;

    for (int i = 0; i < im_size; i++) {            // iterates over fast scanning lines
        line_sum = 0.0;

        for (int j = 0; j < im_size - r; j++) {    // iterates through fast scanning lines
            line_sum = line_sum + square((*image)[i*im_size + j] - (*image)[i*im_size + j + r]);
        };
        image_sum= image_sum + (line_sum / (im_size - r));
    };

    image_sum =  (image_sum / im_size);
    
    return image_sum;
}


#ifdef __cplusplus
}
#endif



