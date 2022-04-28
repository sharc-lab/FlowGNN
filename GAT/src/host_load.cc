#include <stdlib.h>
#include <stdio.h>
#include "host.h"

float scoring_fn_target_float[NUM_LAYERS][NUM_HEADS][EMB_DIM];
float scoring_fn_source_float[NUM_LAYERS][NUM_HEADS][EMB_DIM];
float linear_proj_weights_float_0[NUM_HEADS][EMB_DIM][1][ND_FEATURE];
float skip_proj_weights_float_0[NUM_HEADS][EMB_DIM][1][ND_FEATURE];
float linear_proj_weights_float_1[NUM_LAYERS - 1][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM];
float skip_proj_weights_float_1[NUM_LAYERS - 1][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM];
float graph_pred_weights_float[NUM_TASK][EMB_DIM];
float graph_pred_bias_float[NUM_TASK];

void load_weights()
{
    printf("Loading weights for GAT ...\n");

    FILE* f;

    f = fopen("gat_ep1_pred_weights_layer5.bin", "r");
    fread(graph_pred_weights_float, sizeof(float), NUM_TASK * EMB_DIM, f);
    fclose(f);

    f = fopen("gat_ep1_pred_bias_layer5.bin", "r");
    fread(graph_pred_bias_float, sizeof(float), NUM_TASK, f);
    fclose(f);

    f = fopen("gat_ep1_scoring_fn_target_layer5.bin", "r");
    fread(scoring_fn_target_float, sizeof(float), NUM_LAYERS * NUM_HEADS * EMB_DIM, f);
    fclose(f);

    f = fopen("gat_ep1_scoring_fn_source_layer5.bin", "r");
    fread(scoring_fn_source_float, sizeof(float), NUM_LAYERS * NUM_HEADS * EMB_DIM, f);
    fclose(f);

    f = fopen("gat_ep1_linear_proj_weight_0_layer5.bin", "r");
    fread(linear_proj_weights_float_0, sizeof(float), NUM_HEADS * EMB_DIM * ND_FEATURE, f);
    fclose(f);

    f = fopen("gat_ep1_linear_proj_weight_1_layer5.bin", "r");
    fread(linear_proj_weights_float_1, sizeof(float), (NUM_LAYERS - 1) * NUM_HEADS * EMB_DIM * NUM_HEADS * EMB_DIM, f);
    fclose(f);

    f = fopen("gat_ep1_skip_proj_weight_0_layer5.bin", "r");
    fread(skip_proj_weights_float_0, sizeof(float), NUM_HEADS * EMB_DIM * ND_FEATURE, f);
    fclose(f);

    f = fopen("gat_ep1_skip_proj_weight_1_layer5.bin", "r");
    fread(skip_proj_weights_float_1, sizeof(float), (NUM_LAYERS - 1) * NUM_HEADS * EMB_DIM * NUM_HEADS * EMB_DIM, f);
    fclose(f);

    /// convert to fixed point
    for(int i = 0; i < NUM_TASK; i++) {
        graph_pred_bias_fixed[i] = (WT_TYPE)graph_pred_bias_float[i];
        for(int j = 0; j < EMB_DIM; j++) {
            graph_pred_weights_fixed[i * EMB_DIM + j] = (WT_TYPE)graph_pred_weights_float[i][j];
        }
    }

    for(int i = 0; i < NUM_LAYERS; i++) {
        for(int j = 0; j < NUM_HEADS; j++) {
            for (int k = 0; k < EMB_DIM; k++) {
                scoring_fn_target_fixed[(i * NUM_HEADS + j) * EMB_DIM + k] = (WT_TYPE)scoring_fn_target_float[i][j][k];
                scoring_fn_source_fixed[(i * NUM_HEADS + j) * EMB_DIM + k] = (WT_TYPE)scoring_fn_source_float[i][j][k];
            }
        }
    }

    for(int head_out = 0; head_out < NUM_HEADS; head_out++) {
        for (int dim_out = 0; dim_out < EMB_DIM; dim_out++) {
            for (int head_in = 0; head_in < 1; head_in++) {
                for (int dim_in = 0; dim_in < ND_FEATURE; dim_in++) {
                    linear_proj_weights_fixed[((head_out * EMB_DIM + dim_out) * NUM_HEADS + head_in) * EMB_DIM + dim_in] = (WT_TYPE)linear_proj_weights_float_0[head_out][dim_out][head_in][dim_in];
                    skip_proj_weights_fixed[((head_out * EMB_DIM + dim_out) * NUM_HEADS + head_in) * EMB_DIM + dim_in] = (WT_TYPE)skip_proj_weights_float_0[head_out][dim_out][head_in][dim_in];
                }
            }
        }
    }

    for(int layer = 1; layer < NUM_LAYERS; layer++) {
        for(int head_out = 0; head_out < NUM_HEADS; head_out++) {
            for (int dim_out = 0; dim_out < EMB_DIM; dim_out++) {
                for (int head_in = 0; head_in < NUM_HEADS; head_in++) {
                    for (int dim_in = 0; dim_in < EMB_DIM; dim_in++) {
                        linear_proj_weights_fixed[(((layer * NUM_HEADS + head_out) * EMB_DIM + dim_out) * NUM_HEADS + head_in) * EMB_DIM + dim_in] = (WT_TYPE)linear_proj_weights_float_1[layer - 1][head_out][dim_out][head_in][dim_in];
                        skip_proj_weights_fixed[(((layer * NUM_HEADS + head_out) * EMB_DIM + dim_out) * NUM_HEADS + head_in) * EMB_DIM + dim_in] = (WT_TYPE)skip_proj_weights_float_1[layer - 1][head_out][dim_out][head_in][dim_in];
                    }
                }
            }
        }
    }
}

void fetch_one_graph(
    int g,
    char* graph_name,
    aligned_vector<node_feature_t>& node_feature,
    aligned_vector<edge_t>& edge_list,
    int num_of_nodes,
    int num_of_edges
)
{
    printf("(%d/%d) Loading graph %s ...\n", g, NUM_GRAPHS, graph_name);

    FILE* f;

    char f_node_feature[128];
    char f_edge_list[128];

    sprintf(f_node_feature, "%s_node_feature.bin", graph_name);
    sprintf(f_edge_list, "%s_edge_list.bin", graph_name);

    f = fopen(f_node_feature, "rb");
    size_t node_feature_start = node_feature.size();
    node_feature.resize(node_feature_start + num_of_nodes);
    node_feature_t* node_feature_ptr = &node_feature.data()[node_feature_start];
    fread(node_feature_ptr, sizeof(node_feature_t), num_of_nodes, f);
    fclose(f);

    f = fopen(f_edge_list, "rb");
    size_t edge_list_start = edge_list.size();
    edge_list.resize(edge_list_start + num_of_edges);
    edge_t* edge_list_ptr = &edge_list.data()[edge_list_start];
    fread(edge_list_ptr, sizeof(edge_t), num_of_edges, f);
    fclose(f);
}
