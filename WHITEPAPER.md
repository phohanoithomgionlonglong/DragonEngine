# Dragon Engine: A Kernel-Mode Spatial-Logic Cache

**White Paper — Version 1.1**  
**Author:** Nguyễn Thành Long, 14 years old  
**Location:** Vietnam  
**Date:** June 2026

---

## Abstract

Dragon Engine is an open-source, kernel-mode driver and companion service suite designed to accelerate repetitive spatial computations in real-time graphics, physics simulation, and AI inference. It operates by caching the results of expensive calculations in a shared, lock-free knowledge map, and serving subsequent requests with O(1) lookups. The system uses Cuckoo hashing with Force Eviction, an atomic Passport-Key locking mechanism, zero-copy memory mapping, SIMD dispatch on the CPU, and fused GPU kernels via OpenCL. This paper describes the complete architecture, the technical decisions behind each component, and the current limitations of the project. Dragon Engine is a proof-of-concept built by a 14-year-old developer working under severe hardware constraints; it has passed Microsoft Driver Verifier in a virtualized environment.

---

## 1. Problem Statement

In modern real-time applications — video games, physics engines, ray tracing, and AI inference — many computational workloads exhibit strong spatial locality. Identical or near-identical inputs are processed repeatedly across frames:

- Collision queries against the same geometry
- Ray intersections in static or slowly changing scenes
- Matrix transformations of recurring coordinate sets
- Neural network activations over cached feature maps

Conventional architectures recompute these results from scratch on every invocation. This wastes CPU/GPU cycles, increases power consumption, and limits the frame budget available for non-redundant work. While application-level caches exist, they are siloed within individual programs and cannot be shared across processes or between CPU and GPU without significant data copying.

Dragon Engine addresses this by moving the cache into the operating system kernel, where it can be shared across all user-mode processes and GPU contexts with zero-copy access.

---

## 2. System Architecture

Dragon Engine consists of three independent components that communicate through a Named Pipe and shared kernel memory.

### 2.1 Component Overview

| Component | Type | Role |
|-----------|------|------|
| `Dragon.sys` | Kernel Driver | Maintains the knowledge map, handles IOCTL requests, manages zero-copy mappings |
| `dragon_compute.exe` | User-mode Service (CPU) | Receives external work via Named Pipe, dispatches to the best available SIMD path, writes results back to the kernel cache |
| `dragon_controller.exe` | User-mode Service (GPU) | Manages GPU devices, builds and dispatches fused OpenCL kernels, shards work across multiple GPUs |

### 2.2 Inter-Component Communication
+-------------------+  Zero-copy mapping   +-------------------+
| dragon_compute    | <===================> | dragon.sys        |
| (CPU SIMD)        |                      | (Kernel Cache)    |
+-------------------+                      +-------------------+
        ^                                          ^
        | Named Pipe                              | Zero-copy mapping
        | (cổng nhận dữ liệu                      |
        |  từ ứng dụng ngoài)                      |
+-------------------+                      +-------------------+
| Ứng dụng ngoài    |                      | dragon_controller |
| (game, AI, ...)   |                      | (GPU OpenCL)      |
+-------------------+                      +-------------------+

- **CPU path:** External applications send spatial computation requests to `\\.\pipe\DragonCompute`. The `dragon_compute` service reads these requests, processes them using SIMD, and writes results asynchronously to the kernel cache via IOCTL and zero-copy mapping.
- **GPU path:** `dragon_controller` accesses the kernel cache directly through the same zero-copy mapped memory. It dispatches fused OpenCL kernels that check the cache, compute missing results, and store them back — all in a single kernel invocation.
- **Driver communication:** Both `dragon_compute` and `dragon_controller` talk to `dragon.sys` exclusively through IOCTLs and a shared zero-copy memory region. The Named Pipe is reserved for external application input into `dragon_compute`.

---

## 3. Kernel Driver (`Dragon.sys`)

### 3.1 Knowledge Map Structure

The Knowledge Map is a hash table stored in non-paged kernel memory. It uses a two-table Cuckoo hashing scheme:

- Each entry has two possible positions: one in Table A, one in Table B
- Lookup is O(1): at most two memory accesses
- Insertion displaces existing occupants when both positions are filled
- If a displacement chain grows beyond a threshold, Force Eviction removes the oldest node in the chain

The table is sized to 2% of system RAM, rounded down to the nearest power of 2.

### 3.2 Spatial Indexing: Morton Codes

3D coordinates (x, y, z) are converted into a 1D key using Morton encoding (Z-order curve). This interleaves the bits of the three coordinates, ensuring that points that are spatially close receive numerically close keys. This property:

- Improves cache locality during batch lookups
- Enables efficient SIMD processing of spatially grouped data
- Simplifies sharding across multiple GPUs by key range

**Limitation:** Morton codes are not rotation-invariant. Scenes with heavy rotational variation will see reduced cache hit rates (see Section 7).

### 3.3 Synchronization: Passport-Key Locking

Synchronization between CPU and GPU access to the cache is achieved through a custom lock-free mechanism called Passport-Key Locking:

- Each node in the hash table contains a 32-bit `owner_ticket` field
- A reader or writer attempts to acquire the node using an atomic Compare-And-Swap (CAS) on this field
- If the CAS succeeds, the caller holds the ticket and may safely read or write
- If the CAS fails, the caller retries after a brief spin
- The ticket also serves as a dirty flag, indicating whether the node contains valid cached data

This approach avoids traditional spinlocks and mutexes, which are either unavailable at certain IRQL levels or too heavy for the intended workload. Because only 32 bits are used for synchronization, the mechanism respects the 128-byte atomic operation boundary imposed by some hardware platforms for CPU-GPU shared memory.

### 3.4 Zero-Copy Memory Mapping

The driver allocates non-paged memory using `ExAllocatePoolWithTag` with the `NonPagedPool` pool type, then maps it directly into the address space of user-mode processes via `IoAllocateMdl`, `MmBuildMdlForNonPagedPool`, and `MmMapLockedPagesSpecifyCache`. This allows:

- `dragon_controller.exe` to access kernel cache data without a single copy
- GPU OpenCL kernels to operate on the same physical memory through sub-buffers

The tradeoff is safety: a buggy user-mode process could corrupt kernel memory. The driver includes pointer validation, but this remains an inherent risk of zero-copy designs.

### 3.5 Node Aging and Cache Management

A dedicated worker thread periodically scans the hash table and reduces a `frequency` counter for each node. Nodes with low frequency are candidates for eviction. This ensures that:

- Frequently accessed ("hot") data remains in the cache
- Cold data is gradually removed, freeing space for new entries
- The cache adapts to changing workloads without manual tuning

### 3.6 Sharding for Parallelism

The hash table is divided into shards, where the number of shards is a power of two not exceeding the number of logical CPU cores. Each shard can be locked independently, reducing contention during concurrent access from multiple threads or GPU compute units.

In the GPU path (`dragon_controller`), the same sharding principle is applied: the knowledge map is partitioned into contiguous segments (also a power of two in size) and each segment is mapped to a separate OpenCL sub-buffer. This allows multiple GPUs to work on disjoint ranges of the cache simultaneously, enabling linear scaling with additional GPUs.

---

## 4. CPU Compute Service (`dragon_compute.exe`)

### 4.1 Named Pipe Interface

The CPU service listens on `\\.\pipe\DragonCompute`. External applications connect and send batches of spatial data as binary messages. The service processes each batch and returns results synchronously while also writing to the kernel cache asynchronously.

### 4.2 SIMD Runtime Dispatch

At startup, the service detects the best SIMD instruction set available on the host CPU:

- SSE2 (baseline for x86-64)
- SSE4.1
- AVX
- AVX2
- AVX-512

It then selects the corresponding kernel implementation. This ensures maximum throughput on any x86-64 processor without requiring separate binaries.

### 4.3 Zero-Value Skip

Before processing, the service scans input batches for blocks that are entirely zero. These blocks are skipped, saving computation when the workload contains large sparse regions.

---

## 5. GPU Compute Controller (`dragon_controller.exe`)

### 5.1 Dynamic OpenCL Loading

To avoid build-time dependencies on any specific OpenCL SDK, the controller dynamically loads all OpenCL functions from `OpenCL.dll` at runtime using `LoadLibrary` and `GetProcAddress`. This makes the executable portable across different GPU vendors and driver versions.

### 5.2 Fused Kernel Architecture

Traditional GPU compute pipelines split work into sequential kernels (Lookup → Compute → Store). Dragon Engine fuses all three stages into a single OpenCL kernel:

1. **Lookup:** Check whether the requested computation result is already in the cache
2. **Compute:** If not found, perform the spatial computation directly on the GPU
3. **Store:** Write the result back to the cache for future reuse

This fusion eliminates PCIe round-trips and kernel dispatch overhead between stages.

### 5.3 Multi-GPU Sharding

The knowledge map is divided into contiguous segments based on Morton code ranges. Each GPU is assigned a subset of segments and operates independently on its own workload. This enables linear scaling with additional GPUs, limited primarily by the bandwidth of the shared knowledge map.

---

## 6. Safety and Verification

- The driver has passed **Microsoft Driver Verifier** with standard settings on a Windows virtual machine
- All synchronization primitives use atomic operations compatible with kernel-mode execution at IRQL ≤ DISPATCH_LEVEL
- Pointer and buffer bounds are validated on every IOCTL path

**What has not been done:**
- Formal verification of the lock-free algorithms (e.g., with TLA+ or Spin)
- Real-hardware stress testing under sustained GPU load
- Security audit of the zero-copy memory sharing

---

## 7. Current Limitations

1.  **No automatic interception.** The engine cannot yet hook into existing game or graphics API calls. Applications must explicitly send work through the Named Pipe. A kernel-level hooking or user-mode API wrapper layer is required for transparent integration.

2.  **No real-world benchmarks.** The author's computer failed shortly after the code was published. All performance characteristics are theoretical and based on the asymptotic behavior of the chosen algorithms.

3.  **Rotation-variant hashing.** Morton codes do not handle rotated coordinate systems well. A rotation-invariant encoding would improve cache hit rates for dynamic scenes.

4.  **No formal proof of lock-freedom.** The Passport-Key mechanism has been tested empirically but not verified with formal methods.

5.  **Windows-only.** The driver is tightly coupled to the Windows kernel API and cannot run on Linux or other platforms without a complete rewrite.

---

## 8. Comparison to Related Work

| Approach | Cache Location | Cross-Process | GPU Access | Locking |
|----------|---------------|---------------|------------|---------|
| Application-level cache (game engine) | User-mode heap | No | Via API | Mutex |
| GPU texture cache | GPU VRAM | No | Native | Hardware |
| Dragon Engine | Kernel non-paged memory | Yes | Zero-copy | Lock-free CAS |

Dragon Engine is the only approach that provides a shared, hardware-agnostic cache accessible to both CPU and GPU across process boundaries without data copies.

---

## 9. Roadmap

The following items are planned for future development, contingent on hardware availability:

1.  **Interception layer:** Investigate kernel-level hooking and user-mode API wrappers for transparent integration with existing games
2.  **Rotation-invariant encoding:** Prototype alternative spatial hashing methods (e.g., radial basis functions)
3.  **Linux port:** Explore porting the cache logic to a Linux kernel module
4.  **Formal verification:** Model the Passport-Key algorithm in a formal verification tool

---

## 10. Conclusion

Dragon Engine demonstrates that a kernel-resident, lock-free spatial cache shared between CPU and GPU is feasible. The project combines well-understood data structures (Cuckoo hashing, Morton codes) with a custom synchronization primitive (Passport-Key locking) to achieve a design that is simple in concept but careful in execution.

It is not a finished product. It is a foundation — and an open invitation for collaboration.

---

## References

- Pagh, R., & Rodler, F. F. (2004). Cuckoo Hashing. *Journal of Algorithms*, 51(2), 122–144.
- Morton, G. M. (1966). A computer oriented geodetic data base and a new technique in file sequencing. *IBM Technical Report*.
- Microsoft. (n.d.). *Driver Verifier*. Windows Driver Kit Documentation.
- Khronos Group. (n.d.). *OpenCL 2.0 Specification*.

---

*"I am 14. This is my first project. My PC was 12 years old. I did what I could with what I had. Thank you for reading."*

— Nguyễn Thành Long