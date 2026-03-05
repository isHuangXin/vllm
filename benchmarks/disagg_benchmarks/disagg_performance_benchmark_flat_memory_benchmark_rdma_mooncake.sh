#!/bin/bash

# Requirement: 2x GPUs.


# Model: meta-llama/Meta-Llama-3.1-8B-Instruct
# Query: 1024 input tokens, 6 output tokens, QPS 2/4/6/8, 100 requests
# Resource: 2x GPU
# Approaches:
# 2. Chunked prefill: 2 vllm instance with tp=4, equivalent to 1 tp=4 instance with QPS 4
# 3. Disaggregated prefill: 1 prefilling instance and 1 decoding instance
# Prefilling instance: max_output_token=1
# Decoding instance: force the input tokens be the same across requests to bypass prefilling

set -euo pipefail
set -x

BASE=/data/qinjin/.cache
mkdir -p $BASE/tmp $BASE/torchinductor $BASE/triton $BASE/torch_extensions
# 1) 系统临时目录
export TMPDIR=$BASE/tmp
export TEMP=$BASE/tmp
export TMP=$BASE/tmp
# 2) torch.compile / torch inductor 缓存
export TORCHINDUCTOR_CACHE_DIR=$BASE/torchinductor
# 3) triton 缓存
export TRITON_CACHE_DIR=$BASE/triton
# 4) C++/CUDA extension 编译
export TORCH_EXTENSIONS_DIR=$BASE/torch_extensions

PIDS=()

register_pid() {
  local pid="$1"
  if [[ -n "${pid:-}" ]]; then
    PIDS+=("$pid")
  fi
}

kill_gpu_processes() {
  set +x
  # 1) 杀本脚本启动的进程
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    for pid in "${PIDS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${PIDS[@]}"; do
      kill -9 "$pid" 2>/dev/null || true
    done
  fi

  # 2) 杀占用端口的进程
  for port in 8000 8100 8200 14579 14580 30001; do
    pids=$(lsof -t -iTCP:$port -sTCP:LISTEN -u "$(id -u)" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      # 只杀匹配 vllm / proxy 的
      for pid in $pids; do
        cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
        if echo "$cmd" | egrep -q 'vllm serve|disagg_prefill_proxy_server|uvicorn|quart'; then
          kill "$pid" 2>/dev/null || true
          sleep 0.2
          kill -9 "$pid" 2>/dev/null || true
        fi
      done
    fi
  done

  PIDS=()
  sleep 1
  set -x
}

cleanup() {
  # 只杀 PIDS，不扫端口
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    kill "${PIDS[@]}" 2>/dev/null || true
    sleep 1
    kill -9 "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  # wait for vllm server to start
  # return 1 if vllm server crashes
  local port=$1
  echo "Waiting for server at port $port..."
  timeout 600 bash -c "
    until curl -fsS --connect-timeout 1 --max-time 1 http://127.0.0.1:${port}/v1/models >/dev/null 2>&1; do
      sleep 1
    done"
}

launch_disagg_prefill() {
  model="/data/qinjin/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

  # Force 127.0.0.1 to avoid multi-NIC confusion
  export VLLM_HOST_IP=127.0.0.1

  # --- FORCE RDMA CONFIGURATION ---
  export NCCL_P2P_DISABLE=1
  export NCCL_SHM_DISABLE=1
  export NCCL_IB_DISABLE=0
  export NCCL_NET_GDR_READ=1
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  
  # Ensure NCCL control plane uses loopback to avoid timeouts
  export NCCL_SOCKET_IFNAME=lo
  export GLOO_SOCKET_IFNAME=lo

  # export NCCL_DEBUG=INFO
  # export NCCL_DEBUG_SUBSYS=NET,GRAPH
  # export NCCL_DEBUG_FILE=/data/qinjin/workspace/vllm/vllm/benchmarks/disagg_benchmarks/nccl_%h_%p.log
  # export NCCL_NET_GDR_LEVEL=SYS

  # 
  # disagg prefill: Producer (Prefill) on GPU 2 -> mapped to IB mlx5_4 (Active/200Gb)
  NCCL_IB_HCA=mlx5_4:1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 vllm serve $model \
    --host 127.0.0.1 \
    --port 8100 \
    --gpu-memory-utilization 0.8 \
    --max_model_len 10000 \
    --max-num-seqs 64 \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":8e9,"kv_port":"14579","kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_ip":"127.0.0.1","http_port":"8100","send_type":"PUT_ASYNC"}}' &
  register_pid $!

  # disagg prefill: Consumer (Decode) on GPU 0/1 -> mapped to IB mlx5_3 (Active/200Gb)
  NCCL_IB_HCA=mlx5_3:1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 vllm serve $model \
    --host 127.0.0.1 \
    --port 8200 \
    --max_model_len 10000 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 64 \
    --kv-transfer-config \
    '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":20e9,"kv_port":"14580","kv_connector_extra_config":{"proxy_ip":"127.0.0.1","proxy_port":"30001","http_ip":"127.0.0.1","http_port":"8200","send_type":"PUT_ASYNC"}}' &
  register_pid $!

  wait_for_server 8100
  wait_for_server 8200
  
  # Explicitly bind proxy to localhost environment
  python3 disagg_prefill_proxy_server.py \
    --prefill-url http://127.0.0.1:8100 \
    --decode-url http://127.0.0.1:8200 \
    --kv-host 127.0.0.1 &
  register_pid $!
  sleep 1
}


benchmark() {
  results_folder="./results"
  model="/data/qinjin/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
  # dataset_name="sharegpt"
  dataset_path="/data/qinjin/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json"
  num_prompts=640
  qps=$1
  prefix_len=50
  input_len=1024
  output_len=$2
  tag=$3

  # --dataset-name $dataset_name \
  # --dataset-path $dataset_path \
  # Use openai backend for proxy, set timeout to 0 for readiness check
  vllm bench serve \
    --backend vllm \
    --random-input-len $input_len \
    --random-output-len $output_len \
    --model $model \
    --num-prompts $num_prompts \
    --port 8000 \
    --num-warmups 16 \
    --max-concurrency 64 \
    --save-result \
    --result-dir $results_folder \
    --result-filename "$tag"-qps-"$qps".json \
    --request-rate "$qps" \
    --ready-check-timeout-sec 0

  sleep 2
}


main() {

  (which wget && which curl) || (apt-get update && apt-get install -y wget curl)
  (which jq) || (apt-get -y install jq)
  (which socat) || (apt-get -y install socat)
  (which lsof) || (apt-get -y install lsof)

  pip install --break-system-packages quart httpx matplotlib aiohttp datasets

  cd "$(dirname "$0")"

  rm -rf results
  mkdir results

  default_output_len=512

  export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')

  # Ensure clean slate from the previous manual run
  # kill_gpu_processes

  launch_disagg_prefill
  for qps in 32 64 128; do
    benchmark $qps $default_output_len disagg_prefill
  done
  kill_gpu_processes

  python3 visualize_benchmark_results.py

}


main "$@"
