//
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//

#include <torch/extension.h>


/**
 * Compute a*b^T and save it into out.
 *
 * a \in R^A
 * b \in R^B
 */
inline void vvt_dot(float *a, float *b, float *out, int A, int B) {
    for (int i=0; i<A; i++) {
        float * bi = b;
        for (int j=0; j<B; j++) {
            *out += (*a) * (*bi);
            out++;
            bi++;
        }
        a++;
    }
}


/**
 * Implement a vector matrix product v*m and save it into out.
 *
 * v \in R^A
 * m \in R^{AxB}
 */
inline void vm_dot(float *v, float *m, float *out, int A, int B) {
    // TODO: Consider removing the zeroing part and assuming out already
    //       contains 0s
    for (int i=0; i<B; i++) {
        out[i] = 0;
    }

    for (int i=0; i<A; i++) {
        float *oi = out;
        for (int j=0; j<B; j++) {
            *oi += (*v) * (*m);
            oi++;
            m++;
        }
        v++;
    }
}


/**
 * Implement a vector transposed-matrix product and save it into out.
 *
 * v \in R^B
 * m \in R^{AxB}
 */
inline void vmt_dot(float *v, float *m, float *out, int A, int B) {
    for (int i=0; i<A; i++) {
        float *vi = v;
        float s = 0;
        for (int j=0; j<B; j++) {
            s += (*vi) * (*m);
            vi++;
            m++;
        }
        // TODO: Should we be aggregating? See the comment on vm_dot.
        *out = s;
        out++;
    }
}


/**
 * Compute the causally masked dot products of queries, keys and values.
 *
 * Basically compute V_j' = (Q_{0:j} * K_{0:j}^T) * V_{0:j} for all j. The
 * computation is done efficiently by changing the order of the dot products.
 */
void causal_dot_product(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor values,
    torch::Tensor product
) {
    // Extract some shapes
    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);

    // Create accessors for all the arguments
    auto qa = queries.accessor<float, 4>();
    auto ka = keys.accessor<float, 4>();
    auto va = values.accessor<float, 4>();
    auto pa = product.accessor<float, 4>();

    #pragma omp parallel for collapse(2)
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            auto kv = torch::zeros({E, M}, queries.options());
            float *kvp = kv.data_ptr<float>();
            for (int l=0; l<L; l++) {
                vvt_dot(
                    &ka[n][h][l][0],
                    &va[n][h][l][0],
                    kvp,
                    E,
                    M
                );
                vm_dot(
                    &qa[n][h][l][0],
                    kvp,
                    &pa[n][h][l][0],
                    E,
                    M
                );
            }
        }
    }
}


/**
 * Compute the gradients of queries, keys and values given the gradient of the
 * causal_dot_product output.
 *
 * Make sure that everything is computed in O(N D^2) complexity.
 */
void causal_dot_backward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor values,
    const torch::Tensor grad_out,
    torch::Tensor grad_queries,
    torch::Tensor grad_keys,
    torch::Tensor grad_values
) {
    // Extract some shapes
    int N = queries.size(0);
    int H = queries.size(1);
    int L = queries.size(2);
    int E = queries.size(3);
    int M = values.size(3);

    // Create accessors for all the arguments
    auto qa = queries.accessor<float, 4>();
    auto ka = keys.accessor<float, 4>();
    auto va = values.accessor<float, 4>();
    auto ga = grad_out.accessor<float, 4>();
    auto gqa = grad_queries.accessor<float, 4>();
    auto gka = grad_keys.accessor<float, 4>();
    auto gva = grad_values.accessor<float, 4>();

    #pragma omp parallel for collapse(2)
    for (int n=0; n<N; n++) {
        for (int h=0; h<H; h++) {
            auto kv = torch::zeros({E, M}, queries.options());
            float *kvp = kv.data_ptr<float>();

            // Compute the gradient wrt the queries
            for (int l=0; l<L; l++) {
                vvt_dot(
                    &ka[n][h][l][0],
                    &va[n][h][l][0],
                    kvp,
                    E,
                    M
                );
                vmt_dot(
                    &ga[n][h][l][0],
                    kvp,
                    &gqa[n][h][l][0],
                    E,
                    M
                );
            }

            // Compute the gradient wrt the keys and values
            kv.zero_();
            for (int l=L-1; l>=0; l--) {
                vvt_dot(
                    &qa[n][h][l][0],
                    &ga[n][h][l][0],
                    kvp,
                    E,
                    M
                );
                vmt_dot(
                    &va[n][h][l][0],
                    kvp,
                    &gka[n][h][l][0],
                    E,
                    M
                );
                vm_dot(
                    &ka[n][h][l][0],
                    kvp,
                    &gva[n][h][l][0],
                    E,
                    M
                );
            }
        }
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "causal_dot_product",
        &causal_dot_product,
        "Compute the weighted sum of values but attending only to previous "
        "values."
    );
    m.def(
        "causal_dot_backward",
        &causal_dot_backward,
        "Compute the gradient of queries, keys and values given the gradient "
        "of causal_dot_product."
    );
}
