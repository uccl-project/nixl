## TCPX Backend Plugin (NCCL GPUDirectTCPX Wrapper)

This plugin keeps the same shape as the PR‑895 UCCL backend but now expects an
external `libuccl_engine.so` built with `USE_TCPX=1`, which internally uses
NCCL `ncclSend/ncclRecv` over GPUDirectTCPX. The legacy `tcpx::Endpoint`
implementation has been removed; TCPX naming is retained only for metadata
compatibility with existing NIXL configs.

## Prerequisites

- CUDA Toolkit (nvcc, headers, cudart) on every build node.
- Google’s TCPX NCCL runtime (`libnccl-net-tcpx.so`) installed wherever NIXL
  agents run; expose it via `LD_LIBRARY_PATH` or `NCCL_PLUGIN_P2P`.
- `make`-based Phase A sources under `p2p/` (already vendored in this repo).

## Building the engine library

```bash
git clone https://github.com/uccl-project/uccl.git   # or reuse the vendored copy
cd uccl/p2p
make USE_TCPX=1 -j
sudo install -m 0755 libuccl_engine.so /usr/local/lib/
sudo ldconfig
# Or equivalently:
sudo make USE_TCPX=1 install
```

The `USE_TCPX=1` flag builds `libuccl_engine.so` with NCCL GPUDirectTCPX
support (via `p2p/nccl_tcpx_endpoint.{h,cc}`) while disabling the RDMA
components. After `make install`, the system should expose
`libuccl_engine.so` on a path discoverable by Meson and the dynamic loader.

## Building the plugin

Once `libuccl_engine.so` is available, build the TCPX plugin from the NIXL repo
root:

```bash
# from the NIXL repo root
meson setup build -Ddisable_tcpx_backend=false         # first time
# or: meson setup --reconfigure build -Ddisable_tcpx_backend=false

# build the plugin directly by output path
ninja -C build src/plugins/tcpx/libplugin_TCPX.so
```

For static builds, pass `-Dstatic_plugins=TCPX` (and rebuild the core library),
so the plugin is embedded in the agent binary.

## Plugin discovery

The plugin manager scans a single directory for `libplugin_*.so` files (no
recursion). Point `NIXL_PLUGIN_DIR` to the directory that directly contains the
plugin file or install it to the default location:

- Development (no install):
  - `export NIXL_PLUGIN_DIR=$(pwd)/build/src/plugins/tcpx`
  - Optional: `export NIXL_LOG_LEVEL=DEBUG` to see loader debug messages
- System install:
  - `sudo install -d /usr/local/lib/plugins`
  - `sudo install -m 0755 build/src/plugins/tcpx/libplugin_TCPX.so /usr/local/lib/plugins/`
  - Plugin path: `/usr/local/lib/plugins/libplugin_TCPX.so`
  - Either rely on default discovery or set `NIXL_PLUGIN_DIR=/usr/local/lib/plugins`

Quick Python check:

```bash
export NIXL_PLUGIN_DIR=$(pwd)/build/src/plugins/tcpx
python3 - <<'PY'
from nixl._api import nixl_agent
a = nixl_agent("check")
print("Loaded plugins:", a.plugin_list)
PY
```

Force TCPX backend only (omit extra args if your package has older API):

```bash
python3 - <<'PY'
from nixl._api import nixl_agent, nixl_agent_config
cfg = nixl_agent_config(backends=["TCPX"])  # If older API, just keep backends
a = nixl_agent("check", cfg)
print("Loaded plugins:", a.plugin_list)
PY
```

## Runtime knobs

- Backend options: `device_idx`, `num_cpus`, and `in_python` (parity with UCCL).
- Required environment (minimum):
	- `NCCL_MIN_ZCOPY_SIZE=4096`
	- `NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE=4096`
	- `NCCL_GPUDIRECTTCPX_RECV_SYNC=1`
	- `UCCL_TCPX_CHUNK_BYTES` ≤ 4 MiB
	- `UCCL_TCPX_PORT_RETRIES` (optional) to probe adjacent ports if the base is busy
	- TCPX NIC binding variables (see `p2p/tcpx_plugin_usage.md` for a ready-made
	    script)
- Set `NIXL_BACKEND=tcp-x` (or the corresponding configuration entry) so the
  agent picks this plugin.

## Validation

1. Confirm `libuccl_engine.so` resolves to the TCPX-enabled build:
   `ldd build/src/plugins/tcpx/libplugin_TCPX.so | grep uccl_engine`.
2. Export the TCPX environment variables on all participating nodes.
3. Run a simple NIXL workload issuing remote GPU reads/writes. Throughput and
   error modes should match the Phase A smoke tests.

For additional guidance, refer to `p2p/tcpx_plugin_usage.md` and
`p2p/tcpx_phase_b_plan.md`.

### Troubleshooting

- Unknown Ninja target: list targets and pick `TCPX`.
  - `ninja -C build -t targets all | grep -i tcpx`
- `libuccl_engine.so` not found at link/run time:
  - `sudo ldconfig -p | grep libuccl_engine`
  - or `export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`
- Debug build plugin discovery file:
  - `grep TCPX build/pluginlist`

- The loader does not recurse into subdirectories:
  - Set `NIXL_PLUGIN_DIR` to the directory that directly contains
    `libplugin_TCPX.so` (e.g., `build/src/plugins/tcpx`), not its parent.

### Troubleshooting: Build/Install and dlopen

- Build only this plugin target (no global install):
  ```bash
  ninja -C build src/plugins/tcpx/libplugin_TCPX.so
  export NIXL_PLUGIN_DIR=$(pwd)/build/src/plugins/tcpx
  ```

- If `ninja -C build install` fails on unrelated plugins (e.g. POSIX needs libaio):
  - Either install the missing dependency (`sudo apt-get install -y libaio-dev`), or
  - Manually install just this plugin:
    ```bash
    sudo install -d /usr/local/lib/plugins
    sudo install -m 0755 build/src/plugins/tcpx/libplugin_TCPX.so /usr/local/lib/plugins/
    export NIXL_PLUGIN_DIR=/usr/local/lib/plugins
    ```

- If Python logs `Failed to load plugin ... libuccl_engine.so: cannot open shared object file`:
  - Verify the engine library is present and visible:
    ```bash
    ls -l /usr/local/lib/libuccl_engine.so || true
    sudo ldconfig -p | grep libuccl_engine || true
    ```
  - Fix by installing/pointing the library:
    ```bash
    # If you have the built .so locally
    sudo install -m 0755 /path/to/libuccl_engine.so /usr/local/lib/
    sudo ldconfig
    # Or set a search path
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
    ```
  - Re-check linkage:
    ```bash
    ldd $(pwd)/build/src/plugins/tcpx/libplugin_TCPX.so | grep uccl_engine
    ```

## Quick Start: blocking_send_recv_example.py

Run a functional read/verify test over TCPX using the bundled example.

Environment (both sides):

```bash
# Plugin discovery (installed)
export NIXL_PLUGIN_DIR=/usr/local/lib/plugins

# TCPX NCCL runtime (pick one)
export NCCL_PLUGIN_P2P=/usr/local/tcpx/lib64/libnccl-net-tcpx.so
# or: export LD_LIBRARY_PATH=/usr/local/tcpx/lib64:$LD_LIBRARY_PATH

# Minimal TCPX/NCCL settings
export NCCL_MIN_ZCOPY_SIZE=4096
export NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE=4096
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1
export UCCL_TCPX_CHUNK_BYTES=4194304

# Control plane NIC (lo for same-host tests; replace with eth0 on multi-node)
export NCCL_SOCKET_IFNAME=lo
export NCCL_GPUDIRECTTCPX_CTRL_DEV=lo

# Optional: debug logging
export NIXL_LOG_LEVEL=DEBUG
```

Target (listener):

```bash
python3 examples/python/blocking_send_recv_example.py \
  --mode=target --port=28900 --backend=TCPX --use_cuda=False
```

Initiator:

```bash
python3 examples/python/blocking_send_recv_example.py \
  --mode=initiator --ip=127.0.0.1 --port=28900 --backend=TCPX --use_cuda=False
```

For multi-node runs, replace `lo` with your control NIC (e.g., `eth0`) and
adjust the `--ip` accordingly. For higher performance, configure the NIC
binding variables from `p2p/tcpx_plugin_usage.md`.

## Full Environment Profile (example)

The following profile mirrors the perf harness defaults and is suitable for
gVNIC A3 hosts. Adjust NIC names and CPU bindings to your cluster.

```bash
# Discovery (installed path)
export NIXL_PLUGIN_DIR=/usr/local/lib/plugins

# TCPX runtime (choose one)
export NCCL_PLUGIN_P2P=/usr/local/tcpx/lib64/libnccl-net-tcpx.so
# or: export LD_LIBRARY_PATH=/usr/local/tcpx/lib64:$LD_LIBRARY_PATH

# CUDA/toolchain convenience
export PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Minimal TCPX/NCCL essentials
export NCCL_MIN_ZCOPY_SIZE=4096
export NCCL_GPUDIRECTTCPX_MIN_ZCOPY_SIZE=4096
export NCCL_GPUDIRECTTCPX_RECV_SYNC=1
export UCCL_TCPX_CHUNK_BYTES=4194304            # ≤ 4 MiB

# Control-plane + data-plane NICs
export NCCL_GPUDIRECTTCPX_CTRL_DEV=eth0         # control plane (metadata)
export NCCL_SOCKET_IFNAME=eth0                  # NCCL control socket NIC

# Socket threading
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=1

# Chunk and buffer sizing
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_BUFFSIZE=8388608

# Static NIC CPU bindings (example; tune per host)
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"

# Flow steering and completion tuning
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=100
export NCCL_GPUDIRECTTCPX_RX_COMPLETION_NANOSLEEP=100

# NCCL algorithmic hints
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=8
export NCCL_MIN_NCHANNELS=8

# TCPX multi-channel knobs (if supported by your engine)
# export UCCL_TCPX_NUM_CHANNELS=2

# Optional: logging
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-ENV}
export NIXL_LOG_LEVEL=DEBUG
```

To exercise GPU unpack on the target, run the example with `--use_cuda=True`
on both ends (or set `CUDA_VISIBLE_DEVICES=...` to pin GPUs).


# CUDA & 路径设置
export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/var/lib/tcpx/lib64:$LD_LIBRARY_PATH"

# TCPX 性能参数
export UCCL_TCPX_NUM_CHANNELS=2                 # 每个GPU 2个通道
export UCCL_TCPX_BOOTSTRAP_PORT_BASE=20000      # 启动端口基址
export UCCL_TCPX_PERF_SIZE=67108864             # 每次传输字节数 (64MB)
export UCCL_TCPX_PERF_ITERS=20                  # 迭代次数
export UCCL_TCPX_CHUNK_BYTES=524288             # 块大小 (512KB)

# NCCL + TCPX 绑定参数
export NCCL_GPUDIRECTTCPX_CTRL_DEV="eth0"
export NCCL_NSOCKS_PERTHREAD=2
export NCCL_SOCKET_NTHREADS=1
export NCCL_DYNAMIC_CHUNK_SIZE=524288
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_BUFFSIZE=8388608
export NCCL_GPUDIRECTTCPX_TX_BINDINGS="eth1:8-21,112-125;eth2:8-21,112-125;eth3:60-73,164-177;eth4:60-73,164-177"
export NCCL_GPUDIRECTTCPX_RX_BINDINGS="eth1:22-35,126-139;eth2:22-35,126-139;eth3:74-87,178-191;eth4:74-87,178-191"
export NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS=50000
export NCCL_GPUDIRECTTCPX_FORCE_ACK=0
export NCCL_GPUDIRECTTCPX_TX_COMPLETION_NANOSLEEP=100
export NCCL_GPUDIRECTTCPX_RX_COMPLETION_NANOSLEEP=100

# 其他NCCL通用配置
export NCCL_SOCKET_IFNAME=eth0
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_P2P_PXN_LEVEL=0
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=8
export NCCL_MIN_NCHANNELS=8
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ENV
