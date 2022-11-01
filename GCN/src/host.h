#ifndef __HOST_H__
#define __HOST_H__

#include "dcl.h"
#include "xcl2.hpp"
#include "dataset.hpp"

constexpr int NUM_TRIALS = 25;

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

extern aligned_vector<WT_TYPE> node_embedding_weight_fixed;
extern aligned_vector<WT_TYPE> edge_embedding_weight_fixed;
extern aligned_vector<WT_TYPE> convs_weight_fixed;
extern aligned_vector<WT_TYPE> convs_bias_fixed;
extern aligned_vector<WT_TYPE> convs_root_emb_weight_fixed;
extern aligned_vector<WT_TYPE> bn_weight_fixed;
extern aligned_vector<WT_TYPE> bn_bias_fixed;
extern aligned_vector<WT_TYPE> bn_mean_fixed;
extern aligned_vector<WT_TYPE> bn_var_fixed;
extern aligned_vector<WT_TYPE> graph_pred_linear_weight_fixed;
extern aligned_vector<WT_TYPE> graph_pred_linear_bias_fixed;

void GCN_compute_one_graph();
void load_weights();
void fetch_one_graph(
    int g,
    char* graph_name,
    aligned_vector<node_feature_t>& node_feature,
    aligned_vector<edge_t>& edge_list,
    aligned_vector<edge_attr_t>& edge_attr,
    int num_of_nodes,
    int num_of_edges
);

#endif
