#include "tsne_core.cpp"

extern "C" {
  int run_tSNE_float64(double *inputData, double *outputData, int Nsamples, int in_dims, int out_dims, double theta, double perplexity, int rand_seed, bool verbose, int max_iter) {
    return run_tSNE<double>(inputData, outputData, Nsamples, in_dims, out_dims, theta, perplexity, rand_seed, verbose, max_iter);
    }

  int run_tSNE_float32(float *inputData, float *outputData, int Nsamples, int in_dims, int out_dims, float theta, float perplexity, int rand_seed, bool verbose, int max_iter) {
    return run_tSNE<float>(inputData, outputData, Nsamples, in_dims, out_dims, theta, perplexity, rand_seed, verbose, max_iter);
    }
}
