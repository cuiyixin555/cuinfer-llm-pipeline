#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this script: source startup_env.sh" >&2
  exit 1
fi

_cuinfer_prepend_path() {
  local entry="$1" remaining="$2" item
  local result="$entry"

  if [[ -n "$remaining" ]]; then
    while :; do
      item="${remaining%%:*}"
      if [[ "$item" != "$entry" ]]; then
        result+=":$item"
      fi
      if [[ "$remaining" != *:* ]]; then
        break
      fi
      remaining="${remaining#*:}"
    done
  fi
  printf '%s' "$result"
}

_cuinfer_setup_cuda() {
  local cuda_home="${CUDA_HOME:-/usr/local/cuda}"
  local new_path="${PATH:-}"
  local new_library_path="${LD_LIBRARY_PATH:-}"
  local wsl_driver_dir="/usr/lib/wsl/lib"

  if [[ "$cuda_home" != /* || "$cuda_home" == *:* ]]; then
    echo "CUDA_HOME must be an absolute path without ':' (got: $cuda_home)" >&2
    return 1
  fi
  if [[ ! -x "$cuda_home/bin/nvcc" ]]; then
    echo "CUDA compiler not found: $cuda_home/bin/nvcc" >&2
    echo "Install the CUDA Toolkit for WSL or set CUDA_HOME to its location." >&2
    return 1
  fi
  if [[ ! -f "$cuda_home/include/cuda_runtime.h" || ! -f "$cuda_home/lib64/libcudart.so" ]]; then
    echo "CUDA headers or runtime library missing under $cuda_home" >&2
    return 1
  fi

  if [[ -f "$wsl_driver_dir/libcuda.so.1" ]]; then
    new_library_path="$(_cuinfer_prepend_path "$wsl_driver_dir" "$new_library_path")"
    if [[ -x "$wsl_driver_dir/nvidia-smi" ]]; then
      new_path="$(_cuinfer_prepend_path "$wsl_driver_dir" "$new_path")"
    fi
  else
    echo "Warning: WSL CUDA driver library not found at $wsl_driver_dir/libcuda.so.1." >&2
    echo "Ensure the Windows NVIDIA driver supports CUDA in WSL." >&2
  fi

  new_path="$(_cuinfer_prepend_path "$cuda_home/bin" "$new_path")"
  new_library_path="$(_cuinfer_prepend_path "$cuda_home/lib64" "$new_library_path")"
  export CUDA_HOME="$cuda_home"
  export CUDA_PATH="$cuda_home"
  export PATH="$new_path"
  export LD_LIBRARY_PATH="$new_library_path"
  printf 'CUDA environment configured: %s\n' "$CUDA_HOME"
}

if _cuinfer_setup_cuda; then
  unset -f _cuinfer_setup_cuda _cuinfer_prepend_path
else
  unset -f _cuinfer_setup_cuda _cuinfer_prepend_path
  return 1
fi