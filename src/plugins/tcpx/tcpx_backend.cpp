/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tcpx_backend.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <limits>
#include <algorithm>

// Parse connection string in format: ip_addr:port?gpu_index
bool
parseConnectionString(const std::string &conn_str, char *&ip_addr, int &port, int &gpu_index) {
    // Exit with error if neither : or ? is found in conn_str
    size_t colon_pos = conn_str.find(':');
    if (colon_pos == std::string::npos) {
        NIXL_ERROR << "Invalid connection string format: missing colon separator";
        return false;
    }
    size_t question_pos = conn_str.find('?', colon_pos);
    if (question_pos == std::string::npos) {
        NIXL_ERROR << "Invalid connection string format: missing question mark separator";
        return false;
    }

    std::string ip_str = conn_str.substr(0, colon_pos);
    ip_addr = new char[ip_str.length() + 1];
    strcpy(ip_addr, ip_str.c_str());

    std::string port_str = conn_str.substr(colon_pos + 1, question_pos - colon_pos - 1);
    try {
        port = std::stoi(port_str);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Invalid port number: " << port_str;
        delete[] ip_addr;
        return false;
    }

    std::string gpu_str = conn_str.substr(question_pos + 1);
    try {
        gpu_index = std::stoi(gpu_str);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Invalid GPU index: " << gpu_str;
        delete[] ip_addr;
        return false;
    }

    return true;
}

int
getNixlParam(const nixl_b_params_t *custom_params, const std::string &key, int default_value) {
    if (!custom_params) {
        return default_value;
    }

    auto it = custom_params->find(key);
    if (it == custom_params->end()) {
        return default_value;
    }

    try {
        return std::stoi(it->second);
    }
    catch (const std::exception &) {
        return default_value;
    }
}

namespace {

int
getEnvIntOrDefault(const char *key, int default_value) {
    const char *value = std::getenv(key);
    if (!value || !*value) {
        return default_value;
    }
    char *endptr = nullptr;
    long parsed = std::strtol(value, &endptr, 10);
    if (endptr == value) {
        return default_value;
    }
    if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

int
getBasePort() {
    int base = getEnvIntOrDefault("UCCL_TCPX_OOB_PORT", -1);
    if (base < 0) base = getEnvIntOrDefault("UCCL_TCPX_CTRL_PORT", -1);
    if (base < 0) base = 28900;
    return base;
}

int
getPortRetryCount() {
    int retries = getEnvIntOrDefault("UCCL_TCPX_PORT_RETRIES", 16);
    if (retries < 1) return 1;
    return retries;
}

int
distanceToAllowedRange(int port, int base_port, int retry_count) {
    if (port <= 0 || port > 0xFFFF) {
        return std::numeric_limits<int>::max();
    }
    int min_port = base_port;
    int max_port = base_port + retry_count - 1;
    if (port < min_port) return min_port - port;
    if (port > max_port) return port - max_port;
    return 0;
}

int
chooseHostOrderPort(int raw_port) {
    if (raw_port <= 0 || raw_port > 0xFFFF) {
        return raw_port;
    }

    uint16_t swapped16 = ntohs(static_cast<uint16_t>(raw_port));
    int swapped = static_cast<int>(swapped16);
    if (swapped == raw_port) {
        return raw_port;
    }

    int base_port = getBasePort();
    int retry_count = getPortRetryCount();
    int raw_distance = distanceToAllowedRange(raw_port, base_port, retry_count);
    int swapped_distance = distanceToAllowedRange(swapped, base_port, retry_count);

    if (swapped_distance < raw_distance) {
        return swapped;
    }
    if (swapped_distance > raw_distance) {
        return raw_port;
    }

    // If both distances are equal, fall back to whichever is closer to the base port.
    if (std::abs(swapped - base_port) < std::abs(raw_port - base_port)) {
        return swapped;
    }

    return raw_port;
}

} // namespace

namespace {

size_t
getChunkBytes() {
    static size_t cached_chunk_bytes = []() -> size_t {
        int chunk = getEnvIntOrDefault("UCCL_TCPX_CHUNK_BYTES", 0);
        if (chunk <= 0) {
            return 0;
        }
        return static_cast<size_t>(chunk);
    }();
    return cached_chunk_bytes;
}

} // namespace

nixlTcpxEngine::nixlTcpxEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    local_agent_name_ = init_params->localAgent;
    nixl_b_params_t *custom_params = init_params->customParams;

    bool has_device_idx = false;
    if (custom_params) {
        has_device_idx = (custom_params->find("device_idx") != custom_params->end());
    }
    size_t dev_idx = getNixlParam(custom_params, "device_idx", 0);
    size_t num_cpus = getNixlParam(custom_params, "num_cpus", 4);
    int in_python = getNixlParam(custom_params, "in_python", 1);
    NIXL_DEBUG << "Creating TCPX Engine for dev: " << dev_idx << ", num_cpus: " << num_cpus;
    if (has_device_idx) {
        std::string dev_env = std::to_string(dev_idx);
        setenv("UCCL_TCPX_LOCAL_DEVICE", dev_env.c_str(), /*overwrite=*/1);
        NIXL_DEBUG << "TCPX local device override via env: " << dev_env;
    }
    engine_ = uccl_engine_create(num_cpus, (in_python == 1));
    NIXL_DEBUG << "TCPX engine created";

    listener_thread_ = std::thread(&nixlTcpxEngine::startListener, this);
}

nixlTcpxEngine::~nixlTcpxEngine() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &[addr, priv] : mem_reg_info_) {
            if (priv && priv->mr_id != 0) {
                uccl_mr_t *mr = reinterpret_cast<uccl_mr_t *>(priv->mr_id);
                if (mr) {
                    uccl_engine_mr_destroy(mr);
                }
            }
            delete priv;
        }
        mem_reg_info_.clear();
    }
    std::set<std::string> destroyed_agents;
    for (auto &[agent_name, conn_id] : connected_agents_) {
        if (destroyed_agents.find(agent_name) == destroyed_agents.end()) {
            uccl_conn_t *conn = reinterpret_cast<uccl_conn_t *>(conn_id);
            if (conn) {
                uccl_engine_conn_destroy(conn);
                destroyed_agents.insert(agent_name);
            }
        }
    }

    connected_agents_.clear();
    if (engine_) {
        // Add a small delay to allow TCPX internal cleanup to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        uccl_engine_destroy(engine_);
        engine_ = nullptr;
    }

    if (listener_thread_.joinable()) {
        listener_thread_.detach();
    }
}

void
nixlTcpxEngine::startListener() {
    // The listener waits for connections from remote agents
    NIXL_DEBUG << "TCPX accepting connections";
    while (true) {
        char ip_buf[256];
        int remote_gpu_idx;
        uccl_conn_t *conn = uccl_engine_accept(engine_, ip_buf, sizeof(ip_buf), &remote_gpu_idx);
        if (!conn) {
            NIXL_ERROR << "Failed to accept connection from remote agent";
            continue;
        }
        // Start the listener thread to send/get notifications from the remote agent
        uccl_engine_start_listener(conn);
        NIXL_DEBUG << "Connected to remote agent: " << ip_buf;
        connected_agents_[ip_buf] = reinterpret_cast<uint64_t>(conn);
    }
}

nixl_mem_list_t
nixlTcpxEngine::getSupportedMems() const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);

    return mems;
}

nixl_status_t
nixlTcpxEngine::getPublicData(const nixlBackendMD *meta, std::string &str) const {
    nixlTcpxBackendMD *priv = (nixlTcpxBackendMD *)meta;
    str = std::to_string(priv->mr_id);

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::getConnInfo(std::string &str) const {
    std::cerr << "[TCPX-plugin] getConnInfo() invoked" << std::endl;
    if (!engine_) {
        return NIXL_ERR_BACKEND;
    }

    char *metadata = nullptr;
    int result = uccl_engine_get_metadata(engine_, &metadata);
    if (result == 0 && metadata) {
        str = std::string(metadata);
        delete[] metadata;
        bool port_corrected = false;
        size_t colon_pos = str.find(':');
        size_t question_pos = (colon_pos == std::string::npos) ? std::string::npos
                                                               : str.find('?', colon_pos);
        if (colon_pos != std::string::npos && question_pos != std::string::npos &&
            question_pos > colon_pos + 1) {
            std::string ip_part = str.substr(0, colon_pos);
            std::string port_part = str.substr(colon_pos + 1, question_pos - colon_pos - 1);
            std::string suffix = str.substr(question_pos);
            try {
                int parsed_port = std::stoi(port_part);
                int host_port = chooseHostOrderPort(parsed_port);
                if (host_port != parsed_port) {
                    std::ostringstream os;
                    os << ip_part << ":" << host_port << suffix;
                    str = os.str();
                    port_corrected = true;
                }
            }
            catch (const std::exception &) {
                // Leave metadata unchanged if parsing fails.
            }
        }
        NIXL_DEBUG << "TCPX engine metadata: " << str;
        if (port_corrected) {
            NIXL_WARN << "TCPX getConnInfo: adjusted metadata port to host byte order: " << str;
        }
        std::cerr << "[TCPX-plugin] getConnInfo() using engine metadata: " << str << std::endl;
        return NIXL_SUCCESS;
    }

    NIXL_WARN << "TCPX getConnInfo: uccl_engine_get_metadata failed result=" << result
              << " metadata=" << (metadata ? metadata : "(null)");
    std::cerr << "[TCPX-plugin] getConnInfo() metadata failed result=" << result
              << " metadata=" << (metadata ? metadata : "(null)") << std::endl;
    if (metadata) {
        delete[] metadata;
        metadata = nullptr;
    }

    // Fallback path: construct metadata from environment if TCPX didn't produce it.
    // Format: ip:port?gpu_index (matches loadRemoteConnInfo/parseConnectionString)
    auto getenv_str = [](const char *k) -> std::string {
        const char *v = std::getenv(k);
        return v ? std::string(v) : std::string();
    };

    std::string ctrl_dev = getenv_str("NCCL_GPUDIRECTTCPX_CTRL_DEV");
    if (ctrl_dev.empty()) ctrl_dev = getenv_str("NCCL_SOCKET_IFNAME");
    std::string port_s = getenv_str("UCCL_TCPX_OOB_PORT");
    if (port_s.empty()) port_s = "28900"; // align with tcpx::Endpoint default

    NIXL_WARN << "TCPX getConnInfo fallback env ctrl_dev='" << ctrl_dev << "' port='" << port_s
              << "'";
    std::cerr << "[TCPX-plugin] getConnInfo() fallback env ctrl_dev='" << ctrl_dev
              << "' port='" << port_s << "'" << std::endl;

    // Resolve IP for the control device
    std::string ip;
    if (!ctrl_dev.empty()) {
        struct ifaddrs *ifaddr = nullptr;
        if (getifaddrs(&ifaddr) == 0) {
            for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
                if (!ifa->ifa_addr || (ifa->ifa_addr->sa_family != AF_INET)) continue;
                if (ctrl_dev == ifa->ifa_name) {
                    char host[INET_ADDRSTRLEN];
                    auto *sa_in = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
                    const char *res = inet_ntop(AF_INET, &(sa_in->sin_addr), host, sizeof(host));
                    if (res) ip = host;
                    break;
                }
            }
            freeifaddrs(ifaddr);
        }
    }

    if (ip.empty()) {
        ip = getenv_str("UCCL_TCPX_CTRL_IP");
        if (!ip.empty()) {
        NIXL_WARN << "TCPX getConnInfo fallback: using UCCL_TCPX_CTRL_IP=" << ip;
        std::cerr << "[TCPX-plugin] getConnInfo() using UCCL_TCPX_CTRL_IP=" << ip << std::endl;
        }
    }

    if (ip.empty()) {
        // As a last resort, do not fail backend creation; advertise loopback.
        // This still lets higher layers proceed and unit tests to run on a single host.
        ip = "127.0.0.1";
        NIXL_WARN << "TCPX getConnInfo fallback: defaulting to 127.0.0.1 (ctrl_dev='"
                  << ctrl_dev << "', metadata error=" << result << ")";
        std::cerr << "[TCPX-plugin] getConnInfo() defaulting IP to 127.0.0.1" << std::endl;
    }

    // Default GPU index to 0; Python can pass device_idx via custom params if needed later.
    std::string gpu_idx = "0";
    std::ostringstream os;
    os << ip << ":" << port_s << "?" << gpu_idx;
    str = os.str();
    NIXL_WARN << "TCPX getConnInfo: using constructed metadata '" << str
              << "' (original TCPX metadata generation failed with code " << result << ")";
    std::cerr << "[TCPX-plugin] getConnInfo() constructed metadata '" << str << "'" << std::endl;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::loadRemoteConnInfo(const std::string &remote_agent,
                                   const std::string &remote_conn_info) {
    // Parse remote_conn_info and establish connection using TCPX engine
    NIXL_DEBUG << "TCPX engine remote_agent: " << remote_agent
               << " loadRemoteConnInfo: " << remote_conn_info;
    std::lock_guard<std::mutex> lock(mutex_);

    char *ip_addr = nullptr;
    int port = 0;
    int gpu_index = 0;

    if (!parseConnectionString(remote_conn_info, ip_addr, port, gpu_index)) {
        return NIXL_ERR_BACKEND;
    }

    uccl_conn_t *conn = nullptr;

    int chosen_port = chooseHostOrderPort(port);
    int fallback_port = -1;
    if (chosen_port != port) {
        fallback_port = port;
        port = chosen_port;
    }
    else if (port > 0 && port <= 0xFFFF) {
        uint16_t swapped16 = ntohs(static_cast<uint16_t>(port));
        int swapped_port = static_cast<int>(swapped16);
        if (swapped_port != port) {
            fallback_port = swapped_port;
        }
    }

    NIXL_DEBUG << "Connecting to " << ip_addr << ":" << port << "?gpu=" << gpu_index
               << std::endl;
    conn = uccl_engine_connect(engine_, ip_addr, gpu_index, port);
    if (!conn) {
        if (fallback_port > 0 && fallback_port <= 0xFFFF) {
            NIXL_WARN << "Primary connect to " << ip_addr << ":" << port
                      << " failed; retrying with port " << fallback_port;
            conn = uccl_engine_connect(engine_, ip_addr, gpu_index, fallback_port);
            if (conn) {
                port = fallback_port;
            }
        }
        if (!conn) {
            NIXL_ERROR << "Failed to connect to remote agent " << remote_agent;
            delete[] ip_addr;
            return NIXL_ERR_BACKEND;
        }
    }

    NIXL_DEBUG << "Successfully connected to remote agent " << remote_agent;
    // Start the listener thread for notifications
    uccl_engine_start_listener(conn);

    connected_agents_[remote_agent] = reinterpret_cast<uint64_t>(conn);

    delete[] ip_addr;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::connect(const std::string &remote_agent) {
    // Unused
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::disconnect(const std::string &remote_agent) {
    auto conn_iter = connected_agents_.find(remote_agent);
    if (conn_iter == connected_agents_.end()) {
        NIXL_ERROR << "No connection found for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }
    uccl_conn_t *conn = reinterpret_cast<uccl_conn_t *>(conn_iter->second);
    if (!conn) {
        NIXL_ERROR << "Invalid connection for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }

    if (conn) {
        NIXL_DEBUG << "Disconnecting from agent: " << remote_agent;
        uccl_engine_conn_destroy(conn);
        connected_agents_.erase(remote_agent);
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::registerMem(const nixlBlobDesc &mem,
                            const nixl_mem_t &nixl_mem,
                            nixlBackendMD *&out) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (mem_reg_info_.count(mem.addr)) {
        auto priv = mem_reg_info_[mem.addr];
        NIXL_DEBUG << "Registering memory: " << mem.addr << ", len:  " << mem.len;
        priv->ref_cnt++;
        out = priv;
        return NIXL_SUCCESS;
    }

    // Register memory with TCPX engine
    uccl_mr_t *mr = uccl_engine_reg(engine_, mem.addr, mem.len);
    if (!mr) {
        NIXL_ERROR << "Failed to register memory with TCPX engine";
        return NIXL_ERR_BACKEND;
    }

    auto priv = new nixlTcpxBackendMD(true);
    priv->addr = (void *)mem.addr;
    priv->length = mem.len;
    priv->ref_cnt = 1;
    priv->mr_id = reinterpret_cast<uint64_t>(mr); // Store the memory region handle
    out = priv;
    mem_reg_info_[mem.addr] = priv;
    NIXL_DEBUG << "Registering memory: " << mem.addr << "Device: " << mem.devId
               << " ref_cnt: " << priv->ref_cnt << " mr_id: " << priv->mr_id;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::deregisterMem(nixlBackendMD *meta) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto priv = static_cast<nixlTcpxBackendMD *>(meta);
    priv->ref_cnt--;
    if (priv->ref_cnt > 0) return NIXL_SUCCESS;

    // Deregister memory from TCPX engine
    if (priv->mr_id != 0) {
        uccl_mr_t *mr = reinterpret_cast<uccl_mr_t *>(priv->mr_id);
        if (mr) {
            uccl_engine_mr_destroy(mr);
            NIXL_DEBUG << "Deregistered memory: " << priv->addr << " mr_id: " << priv->mr_id;
        }
        priv->mr_id = 0;
    }

    mem_reg_info_.erase((uint64_t)priv->addr);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) {
    nixlTcpxBackendMD *input_md = (nixlTcpxBackendMD *)input;
    NIXL_DEBUG << "TCPX Load Local MD: " << input_md->addr << "Meta Info:" << input_md->mr_id;

    nixlTcpxBackendMD *output_md = (nixlTcpxBackendMD *)output;
    output_md->addr = (void *)input_md->addr;
    output_md->length = input_md->length;
    output_md->ref_cnt = 1;
    output_md->mr_id = reinterpret_cast<uint64_t>(input_md->mr_id);

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::loadRemoteMD(const nixlBlobDesc &input,
                             const nixl_mem_t &nixl_mem,
                             const std::string &remote_agent,
                             nixlBackendMD *&output) {
    NIXL_DEBUG << "TCPX Load Remote MD: " << input.addr << "Meta Info:" << input.metaInfo
               << " remote_agent: " << remote_agent;

    output = new nixlTcpxBackendMD(true);
    nixlTcpxBackendMD *output_md = static_cast<nixlTcpxBackendMD *>(output);
    output_md->addr = (void *)input.addr;
    output_md->length = input.len;
    output_md->ref_cnt = 1;
    output_md->mr_id = strtoul(input.metaInfo.c_str(), NULL, 10);

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::unloadMD(nixlBackendMD *input) {
    nixlTcpxBackendMD *md = (nixlTcpxBackendMD *)input;
    delete md;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::prepXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    int result = 0;
    nixlTcpxBackendMD *lmd;
    nixlTcpxBackendMD *rmd;
    handle = nullptr;
    NIXL_DEBUG << "TCPX PrepXfer: " << operation << " remote_agent: " << remote_agent;
    
    // For WRITE operations, all logic is in postXfer
    if (operation == NIXL_WRITE) {
        NIXL_DEBUG << "WRITE operation: skipping prepareXfer (all logic in postXfer)";
        return NIXL_SUCCESS;
    }
    // Get the connection for this remote agent
    auto conn_iter = connected_agents_.find(remote_agent);
    if (conn_iter == connected_agents_.end()) {
        NIXL_ERROR << "No connection found for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }
    uccl_conn_t *conn = reinterpret_cast<uccl_conn_t *>(conn_iter->second);
    if (!conn) {
        NIXL_ERROR << "Invalid connection for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }

    size_t lcnt = local.descCount();
    size_t rcnt = remote.descCount();

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local and remote descriptor counts don't match: " << lcnt << " != " << rcnt;
        return NIXL_ERR_INVALID_PARAM;
    }

    // Collect all tx_data into vectors for batch sending
    std::vector<md_t> md_vector;
    std::vector<nixlTcpxBackendMD *> local_priv_vector;
    md_vector.reserve(lcnt);
    local_priv_vector.reserve(lcnt);

    size_t chunk_bytes = getChunkBytes();
    bool enable_chunking = (chunk_bytes > 0) && (operation == NIXL_WRITE);

    for (size_t i = 0; i < lcnt; i++) {
        lmd = (nixlTcpxBackendMD *)local[i].metadataP;
        rmd = (nixlTcpxBackendMD *)remote[i].metadataP;
        size_t rsize = remote[i].len;

        NIXL_DEBUG << "lmd: " << lmd->addr << ", " << lmd->mr_id << " rmd: " << rmd->addr << ", "
                   << rmd->mr_id;

        // Validate the local address is registered
        auto local_mem_iter = mem_reg_info_.find((uint64_t)lmd->addr);
        if (local_mem_iter == mem_reg_info_.end()) {
            NIXL_ERROR << "Local memory not registered for address: " << lmd->addr;
            return NIXL_ERR_BACKEND;
        }

        auto local_priv = local_mem_iter->second;
        if (local_priv->mr_id == 0) {
            NIXL_ERROR << "Local memory region not properly registered";
            return NIXL_ERR_BACKEND;
        }

        if (enable_chunking && rsize > chunk_bytes) {
            size_t remaining = rsize;
            size_t offset = 0;

            while (remaining > 0) {
                size_t chunk = std::min(remaining, chunk_bytes);

                tx_msg_t tx_data;
                tx_data.data_ptr =
                    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rmd->addr) + offset);
                tx_data.data_size = chunk;

                md_t md;
                md.op = UCCL_WRITE;
                md.data.tx_data = tx_data;

                md_vector.push_back(md);
                local_priv_vector.push_back(local_priv);

                offset += chunk;
                remaining -= chunk;
            }
        }
        else {
            tx_msg_t tx_data;
            tx_data.data_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rmd->addr));
            tx_data.data_size = rsize;

            md_t md;
            md.op = (operation == NIXL_READ) ? UCCL_READ : UCCL_WRITE;
            md.data.tx_data = tx_data;

            md_vector.push_back(md);
            local_priv_vector.push_back(local_priv);
        }
    }

    // Send all tx_data as a vector
    result = uccl_engine_send_tx_md_vector(conn, md_vector.data(), md_vector.size());
    if (result < 0) {
        NIXL_ERROR << "Failed to send transfer metadata vector";
        return NIXL_ERR_BACKEND;
    }

    // Get FIFO items one by one for READ operations
    if (operation == NIXL_READ) {
        for (size_t i = 0; i < local_priv_vector.size(); i++) {
            char fifo_item[FIFO_ITEM_SIZE];
            int retry_count = 0;
            const int max_retries = 5;
            do {
                result = uccl_engine_get_fifo_item(conn, i, &fifo_item);
                if (result == 0) {
                    // Successfully got fifo_item
                    NIXL_DEBUG << "Got the FIFO item to perform read operation for item " << i;
                    memcpy(local_priv_vector[i]->fifo_item_data, fifo_item, FIFO_ITEM_SIZE);
                    break;
                }
                retry_count++;
                if (retry_count < max_retries) {
                    NIXL_DEBUG << "Failed to get FIFO item, retry " << retry_count << "/"
                               << max_retries << " for item " << i;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } while (retry_count < max_retries);

            if (result != 0) {
                NIXL_ERROR << "Failed to get FIFO item after " << max_retries
                           << " retries for item " << i;
                return NIXL_ERR_BACKEND;
            }
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::postXfer(const nixl_xfer_op_t &operation,
                         const nixl_meta_dlist_t &local,
                         const nixl_meta_dlist_t &remote,
                         const std::string &remote_agent,
                         nixlBackendReqH *&handle,
                         const nixl_opt_b_args_t *opt_args) const {
    nixlTcpxReqH *tcpx_handle;
    nixlTcpxBackendMD *lmd;
    nixlTcpxBackendMD *rmd;

    NIXL_DEBUG << "TCPX PostXfer: " << operation << " remote_agent: " << remote_agent;

    // Get the connection for this remote agent
    auto conn_iter = connected_agents_.find(remote_agent);
    if (conn_iter == connected_agents_.end()) {
        NIXL_ERROR << "No connection found for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }

    uccl_conn_t *conn = reinterpret_cast<uccl_conn_t *>(conn_iter->second);
    if (!conn) {
        NIXL_ERROR << "Invalid connection for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }

    size_t lcnt = local.descCount();
    size_t rcnt = remote.descCount();

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local and remote descriptor counts don't match: " << lcnt << " != " << rcnt;
        return NIXL_ERR_INVALID_PARAM;
    }

    // Process each descriptor pair
    // TODO: Use a vector send async API to send all the transfers at once
    for (size_t i = 0; i < lcnt; i++) {
        lmd = (nixlTcpxBackendMD *)local[i].metadataP;
        rmd = (nixlTcpxBackendMD *)remote[i].metadataP;
        size_t lsize = local[i].len;
        size_t rsize = remote[i].len;

        NIXL_DEBUG << "lmd: " << lmd->addr << ", " << lmd->mr_id << " rmd: " << rmd->addr << ", "
                   << rmd->mr_id;

        if (lsize != rsize) {
            NIXL_ERROR << "Local and remote sizes don't match: " << lsize << " != " << rsize;
            return NIXL_ERR_INVALID_PARAM;
        }

        // Validate the local address is registered
        auto local_mem_iter = mem_reg_info_.find((uint64_t)lmd->addr);
        if (local_mem_iter == mem_reg_info_.end()) {
            NIXL_ERROR << "Local memory not registered for address: " << lmd->addr;
            return NIXL_ERR_BACKEND;
        }

        auto local_priv = local_mem_iter->second;
        if (local_priv->mr_id == 0) {
            NIXL_ERROR << "Local memory region not properly registered";
            return NIXL_ERR_BACKEND;
        }

        uccl_mr_t *local_mr = reinterpret_cast<uccl_mr_t *>(local_priv->mr_id);

        // BUG修复：对于TCPX backend，禁用NIXL层面的chunking
        //
        // 问题：NIXL plugin之前有自己的chunking逻辑，对每个chunk调用一次uccl_engine_write
        //       每次调用都会创建一个独立的transfer（每个transfer只有1个chunk）
        //       结果：窗口很快被填满，后续chunks无法post
        //
        // 正确做法：
        // - 对于WRITE操作，不要在NIXL层面分chunk
        // - 发送一次metadata（包含整个size）
        // - 调用一次uccl_engine_write（包含整个size）
        // - UCCL engine内部会自动分成多个chunks（通过UCCL_TCPX_CHUNK_BYTES控制）
        //
        // 注意：READ操作保持原有的chunking逻辑（因为READ是pull模式，不需要metadata）

        int result = 0;
        uint64_t transfer_id = 0;

        switch (operation) {
        case NIXL_READ: {
            // READ操作：保持原有的chunking逻辑
            size_t chunk_bytes = getChunkBytes();
            bool enable_chunking = (chunk_bytes > 0);
            size_t remaining = lsize;
            size_t offset = 0;

            while (remaining > 0) {
                size_t chunk = remaining;
                if (enable_chunking && remaining > chunk_bytes) {
                    chunk = std::min(remaining, chunk_bytes);
                }

                void *local_addr = reinterpret_cast<void *>(
                    reinterpret_cast<uintptr_t>(lmd->addr) + offset);

                result = uccl_engine_read(conn, local_mr, local_addr, chunk,
                                          local_priv->fifo_item_data, &transfer_id);
                if (result != 0) {
                    NIXL_ERROR << "TCPX READ failed at offset=" << offset
                               << " size=" << chunk;
                    return NIXL_ERR_BACKEND;
                }

                if (!handle) {
                    handle = new nixlTcpxReqH(conn);
                }
                tcpx_handle = static_cast<nixlTcpxReqH *>(handle);
                tcpx_handle->transfer_ids.push_back(transfer_id);

                NIXL_DEBUG << "Successfully posted READ chunk: offset=" << offset
                           << " size=" << chunk << " transfer_id=" << transfer_id;

                offset += chunk;
                remaining -= chunk;
            }
            break;
        }

        case NIXL_WRITE: {
            // WRITE操作：不要在NIXL层面分chunk，让UCCL engine来处理

            // 1. 发送metadata（只发送一次，包含整个传输的大小）
            md_t md;
            md.op = UCCL_WRITE;
            md.data.tx_data.data_ptr = reinterpret_cast<uint64_t>(rmd->addr);
            md.data.tx_data.data_size = lsize;  // 整个传输的大小

            int md_result = uccl_engine_send_tx_md(conn, &md);
            if (md_result < 0) {
                NIXL_ERROR << "Failed to send WRITE metadata: remote_addr="
                           << rmd->addr << " size=" << lsize;
                return NIXL_ERR_BACKEND;
            }

            NIXL_DEBUG << "Sent WRITE metadata: remote_addr=" << rmd->addr
                       << " total_size=" << lsize;

            // 2. 调用一次uccl_engine_write（包含整个传输）
            //    UCCL engine内部会自动分成多个chunks
            result = uccl_engine_write(conn, local_mr, lmd->addr, lsize, &transfer_id);
            if (result != 0) {
                NIXL_ERROR << "TCPX WRITE failed: size=" << lsize;
                return NIXL_ERR_BACKEND;
            }

            if (!handle) {
                handle = new nixlTcpxReqH(conn);
            }
            tcpx_handle = static_cast<nixlTcpxReqH *>(handle);
            tcpx_handle->transfer_ids.push_back(transfer_id);

            NIXL_DEBUG << "Successfully posted WRITE: size=" << lsize
                       << " transfer_id=" << transfer_id;
            break;
        }

        default:
            NIXL_ERROR << "Unsupported operation type: " << operation;
            return NIXL_ERR_INVALID_PARAM;
        }
    }
    if (opt_args && opt_args->hasNotif) {
        tcpx_handle->notif_msg = opt_args->notifMsg;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlTcpxEngine::checkXfer(nixlBackendReqH *handle) const {
    if (!handle) {
        NIXL_ERROR << "Invalid handle provided to checkXfer";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlTcpxReqH *tcpx_handle = dynamic_cast<nixlTcpxReqH *>(handle);
    if (!tcpx_handle) {
        NIXL_ERROR << "Invalid handle type for TCPX backend";
        return NIXL_ERR_INVALID_PARAM;
    }

    uccl_conn_t *conn = tcpx_handle->conn;
    if (!conn) {
        NIXL_ERROR << "No connection found in handle";
        return NIXL_ERR_BACKEND;
    }

    bool all_done = true;
    uccl_engine_progress_conn(conn);
    for (uint64_t transfer_id : tcpx_handle->transfer_ids) {
        if (std::find(tcpx_handle->completed_transfer_ids.begin(),
                      tcpx_handle->completed_transfer_ids.end(),
                      transfer_id) != tcpx_handle->completed_transfer_ids.end()) {
            continue;
        }

        int is_done = uccl_engine_xfer_status(conn, transfer_id);
        if (is_done) {
            tcpx_handle->completed_transfer_ids.push_back(transfer_id);
        } else {
            all_done = false;
            continue;
        }
    }
    if (all_done && !tcpx_handle->notif_msg.empty()) {
        notify_msg_t notify_msg = {};
        strncpy(notify_msg.name, local_agent_name_.c_str(), sizeof(notify_msg.name) - 1);
        strncpy(notify_msg.msg, tcpx_handle->notif_msg.c_str(), sizeof(notify_msg.msg) - 1);
        uccl_engine_send_notif(conn, &notify_msg);
        NIXL_DEBUG << "All transfers in handle completed, sent notification: "
                   << tcpx_handle->notif_msg;
    }

    return (all_done) ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t
nixlTcpxEngine::releaseReqH(nixlBackendReqH *handle) const {
    if (!handle) {
        return NIXL_SUCCESS;
    }

    nixlTcpxReqH *tcpx_handle = dynamic_cast<nixlTcpxReqH *>(handle);
    if (tcpx_handle) {
        delete tcpx_handle;
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::getNotifs(notif_list_t &notif_list) {
    if (notif_list.size() != 0) return NIXL_ERR_INVALID_PARAM;

    std::vector<notify_msg_t> notify_msgs = uccl_engine_get_notifs();
    for (size_t i = 0; i < notify_msgs.size(); i++) {
        notif_list.push_back(std::make_pair(notify_msgs[i].name, notify_msgs[i].msg));
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlTcpxEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    auto conn_iter = connected_agents_.find(remote_agent);
    if (conn_iter == connected_agents_.end()) {
        NIXL_ERROR << "No connection found for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }

    uccl_conn_t *conn = reinterpret_cast<uccl_conn_t *>(conn_iter->second);
    if (!conn) {
        NIXL_ERROR << "Invalid connection for remote agent: " << remote_agent;
        return NIXL_ERR_BACKEND;
    }

    notify_msg_t notify_msg;
    memset(&notify_msg, 0, sizeof(notify_msg));
    strncpy(notify_msg.name, local_agent_name_.c_str(), sizeof(notify_msg.name) - 1);
    strncpy(notify_msg.msg, msg.c_str(), sizeof(notify_msg.msg) - 1);
    int result = uccl_engine_send_notif(conn, &notify_msg);
    if (result < 0) {
        NIXL_ERROR << "Failed to send notify message";
        return NIXL_ERR_BACKEND;
    }

    return NIXL_SUCCESS;
}
