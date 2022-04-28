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
constexpr int EMB_DIM = 80;
constexpr int NUM_LAYERS = 4;
constexpr int NUM_TASK = 1;
constexpr int GRAPH_MLP_1_OUT = 40;
constexpr int GRAPH_MLP_2_OUT = 20;

typedef enum {
    AGGR_MEAN,
    AGGR_MIN,
    AGGR_MAX,
    AGGR_STD,
    NUM_AGGRS
} aggregator_t;

typedef enum {
    SCALER_NONE,
    SCALER_T,
    SCALER_SCALE,
    NUM_SCALERS
} scaler_t;
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
extern aligned_vector<WT_TYPE> node_conv_weights_fixed;
extern aligned_vector<WT_TYPE> node_conv_bias_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_1_weights_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_1_bias_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_2_weights_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_2_bias_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_3_weights_fixed;
extern aligned_vector<WT_TYPE> graph_mlp_3_bias_fixed;
extern aligned_vector<WT_TYPE> avg_deg_fixed;

void load_weights();
void fetch_one_graph(
    int g,
    char* graph_name,
    aligned_vector<node_feature_t>& node_feature,
    aligned_vector<edge_t>& edge_list,
    int num_of_nodes,
    int num_of_edges
);

#endif
