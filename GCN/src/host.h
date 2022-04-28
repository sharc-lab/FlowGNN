#ifndef __HOST_H__
#define __HOST_H__

#include <ap_fixed.h>
#include <array>
#include "xcl2.hpp"

// #region Model Parameters
constexpr int MAX_EDGE = 500;
constexpr int MAX_NODE = 500;
constexpr int ND_FEATURE = 9;
constexpr int ND_FEATURE_TOTAL = 173;
constexpr int EDGE_ATTR = 3;
constexpr int ED_FEATURE_PER_LAYER = 13;
constexpr int EMB_DIM = 100;
constexpr int NUM_LAYERS = 5;
constexpr int NUM_TASK = 1;
// #endregion

// #region Data Types
typedef ap_fixed<16, 6> FM_TYPE;
typedef ap_fixed<16, 6> WT_TYPE;

typedef std::array<int, ND_FEATURE> node_feature_t;
typedef std::array<int, EDGE_ATTR> edge_attr_t;

typedef struct {
    int u;
    int v;
} edge_t;
// #endregion

constexpr int NUM_GRAPHS = 4113;
constexpr int NUM_TRIALS = 100;

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
