#ifndef __HOST_H__
#define __HOST_H__

#include "dcl.h"
#include "xcl2.hpp"
#include "dataset.hpp"

constexpr int NUM_TRIALS = 25;

template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

extern aligned_vector<WT_TYPE> node_mlp_1_weights_fixed;
extern aligned_vector<WT_TYPE> node_mlp_1_bias_fixed;
extern aligned_vector<WT_TYPE> node_mlp_2_weights_fixed;
extern aligned_vector<WT_TYPE> node_mlp_2_bias_fixed;
extern aligned_vector<WT_TYPE> node_embedding_table_fixed;
extern aligned_vector<WT_TYPE> edge_embedding_table_fixed;
extern aligned_vector<WT_TYPE> graph_pred_linear_weight_fixed;
extern aligned_vector<WT_TYPE> graph_pred_linear_bias_fixed;
extern aligned_vector<WT_TYPE> eps_fixed;

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
