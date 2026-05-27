# Large-Scale Graph Generation with a Fine-Tuned MDLM

## Overview

The original graph generation setup used adjacency-dictionary serialisation. This representation became unstable for larger graphs because generated dictionaries were often incomplete, repetitive, or syntactically malformed.

To improve generation, we added an edge-list representation:

```text
N=<number_of_nodes>; M=<number_of_edges>; E=(u,v),(u,v),...

## training:

CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch --num_processes 1 \
  examples/a2d/mdlm/graph_pt.py \
  --model_name_or_path ".models/a2d/Qwen3-0.6B" \
  --pyg_dataset "MalNetTiny" \
  --data_root "./data/pyg" \
  --token_strategy "edge_list" \
  --malnet_num_hops 2 \
  --max_graphs 1024 \
  --test_size 0.25 \
  --max_nodes_per_graph 12000 \
  --max_length 16384 \
  --num_train_epochs 1 \
  --max_steps 5000 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --bf16 true \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 2 \
  --logging_steps 100 \
  --report_to "none" \
  --graph_eval_num_generated_graphs 128 \
  --graph_eval_generation_batch_size 1 \
  --graph_eval_max_new_tokens 512 \
  --graph_eval_temperature 0.1 \
  --output_dir ".models/debug/graph-mdlm-malnettiny-edgelist-h100-v1"

## to assemble the graphs

python scripts/assemble.py \
  --eval_json ".models/debug/graph-mdlm-malnettiny-edgelist-h100-v1/graph_generation_eval.json" \
  --target_nodes 5000 \
  --out_dir ".models/debug/final-assembled-newprompt-5000-min100" \
  --seed 42 \
  --min_chunk_nodes 100

### to assemble part of the graphs:
for SEED in $(seq 0 999); do
  OUTDIR=".models/debug/assembled_1000_newprompt/graph_seed_${SEED}"

  python scripts/assemble.py \
    --eval_json ".models/debug/graph-mdlm-malnettiny-edgelist-h100-v1/graph_generation_eval.json" \
    --target_nodes 5000 \
    --out_dir "${OUTDIR}" \
    --seed "${SEED}" \
    --min_chunk_nodes 100
done

## evaluating graphs 

cd ~/dllm

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/src:${PYTHONPATH:-}"

python scripts/evaluate_graphs.py \
  --generated_graph_dir ".models/debug/assembled_1000_newprompt" \
  --out_json ".models/debug/assembled_1000_newprompt/evaluation_metrics_1000_global.json" \
  --reference_dataset "MalNetTiny" \
  --data_root "./data/pyg" \
  --reference_max_graphs 512 \
  --reference_max_nodes_per_graph 6000 \
  --malnet_num_hops 2

## 2d visualization

python scripts/visualize.py \
  --graph_json ".models/debug/assembled_1000_newprompt/graph_seed_8/assembled_graph.json" \
  --out_png ".models/debug/assembled_1000_newprompt/graph_seed_8/full_5000_nodes.png" \
  --mode sample \
  --max_nodes 5000

## 3d visualization

python scripts/visu3d.py \
  --graph_json ".models/debug/assembled_1000_newprompt/graph_seed_8/assembled_graph.json" \
  --out_html ".models/debug/assembled_1000_newprompt/graph_seed_8/largest_component_3d.html" \
  --mode largest \
  --max_nodes 400