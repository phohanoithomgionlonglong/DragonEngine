/*
 * dragon_compute.cpp – Dragon Compute Service (Universal CPU Dispatch)
 * - Tự nhận diện tập lệnh mạnh nhất: SSE2, SSE4.1, AVX, AVX2, AVX‑512.
 * - Các kernel được biên dịch với target attribute riêng, không cần cờ toàn cục.
 * - Hàm _xgetbv được cô lập trong target("xsave") để tránh lỗi.
 * - Giao tiếp Named Pipe \\.\pipe\DragonCompute.
 * - Ghi Knowledge Map bất đồng bộ.
 * Chỉ dùng thư viện nội bộ Dev‑C++ (không cần linker/compiler ngoài).
 * Yêu cầu tối thiểu: -msse2 (đã có sẵn trong Dev‑C++).
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <thread>
#include <atomic>
#include <intrin.h>
#include <emmintrin.h>    // SSE2
#include <smmintrin.h>    // SSE4.1
#include <immintrin.h>    // AVX, AVX2, AVX-512

// ==================== CPU Feature Detection ====================
static bool s_SSE2   = true;   // baseline
static bool s_SSE41  = false;
static bool s_AVX    = false;
static bool s_AVX2   = false;
static bool s_AVX512 = false;

// Hàm helper gọi _xgetbv – bắt buộc phải có target("xsave") để biên dịch được
__attribute__((target("xsave")))
static uint64_t GetXCR0() {
    return _xgetbv(0);
}

void DetectCPU() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    s_SSE41 = (cpuInfo[2] & (1 << 19)) != 0;   // SSE4.1
    bool osXSave = (cpuInfo[2] & (1 << 27)) != 0; // OSXSAVE

    if (osXSave) {
        uint64_t xcr0 = GetXCR0();
        bool avx = (cpuInfo[2] & (1 << 28)) != 0;
        bool avx2 = false, avx512f = false, avx512dq = false;
        __cpuid(cpuInfo, 7);
        avx2    = (cpuInfo[1] & (1 << 5)) != 0;
        avx512f = (cpuInfo[1] & (1 << 16)) != 0;
        avx512dq= (cpuInfo[1] & (1 << 17)) != 0;

        if (avx && (xcr0 & 6) == 6) s_AVX = true;
        if (avx2 && (xcr0 & 6) == 6) s_AVX2 = true;
        if (avx512f && avx512dq && (xcr0 & 0xE0) == 0xE0) s_AVX512 = true;
    }
    printf("CPU features: SSE4.1:%d AVX:%d AVX2:%d AVX512:%d\n",
           s_SSE41, s_AVX, s_AVX2, s_AVX512);
}

// ==================== Driver interface ====================
struct alignas(64) SpatialNode {
    uint64_t spatial_key;
    uint32_t owner_ticket;
    uint32_t frequency;
    uint64_t lookup_pointer;
    float    physical_attributes[10];
};

#define FILE_DEVICE_DRAGONENGINE 0x8000
#define IOCTL_DRAGON_MAP_MEMORY   CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x802, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_DRAGON_QUERY_STATS  CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x804, METHOD_BUFFERED, FILE_ANY_ACCESS)

static HANDLE        hDevice   = INVALID_HANDLE_VALUE;
static SpatialNode*  mappedMap = nullptr;
static size_t        mapSize   = 0;
static std::atomic<bool> running(true);

bool ConnectDriver() {
    hDevice = CreateFileW(L"\\\\.\\DragonEngine", GENERIC_READ | GENERIC_WRITE,
                          0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hDevice == INVALID_HANDLE_VALUE) {
        printf("Driver not available, compute only.\n");
        return false;
    }
    DWORD ret;
    SIZE_T stats[2];
    DeviceIoControl(hDevice, IOCTL_DRAGON_QUERY_STATS, NULL, 0, stats, sizeof(stats), &ret, NULL);
    mapSize = stats[0];
    PVOID uAddr = nullptr;
    DeviceIoControl(hDevice, IOCTL_DRAGON_MAP_MEMORY, NULL, 0, &uAddr, sizeof(uAddr), &ret, NULL);
    mappedMap = (SpatialNode*)uAddr;
    return mappedMap != nullptr;
}

static uint64_t MortonKeyFromValue(float value) {
    if (value == 0.0f) return 0;
    float v = fabsf(value);
    uint32_t x = (uint32_t)(v * 1000.0f) & 0x1FFFFF;
    uint32_t y = (uint32_t)(v * 2000.0f) & 0x1FFFFF;
    uint32_t z = (uint32_t)(v * 3000.0f) & 0x1FFFFF;
    uint64_t morton = 0;
    for (int i = 0; i < 21; i++) {
        uint64_t bx = (x >> i) & 1, by = (y >> i) & 1, bz = (z >> i) & 1;
        morton |= (bx << (3*i)) | (by << (3*i+1)) | (bz << (3*i+2));
    }
    return morton;
}

static bool WriteNode(uint64_t key, const float* attrs, int n) {
    if (!mappedMap || mapSize == 0) return false;
    SIZE_T idx = key & (mapSize - 1);
    SpatialNode* node = &mappedMap[idx];
    LONG desired = (LONG)(GetCurrentThreadId() | 0x80000000);
    LONG old = InterlockedCompareExchange(&node->owner_ticket, desired, 0);
    if (old == 0 || old == desired) {
        node->spatial_key = key;
        for (int i = 0; i < n && i < 10; i++) node->physical_attributes[i] = attrs[i];
        InterlockedExchange(&node->frequency, 1);
        InterlockedExchange(&node->owner_ticket, 0);
        return true;
    }
    for (int p = 1; p < 64; p++) {
        SIZE_T idx2 = (idx + p) & (mapSize - 1);
        SpatialNode* n2 = &mappedMap[idx2];
        old = InterlockedCompareExchange(&n2->owner_ticket, desired, 0);
        if (old == 0 || old == desired) {
            n2->spatial_key = key;
            for (int i = 0; i < n && i < 10; i++) n2->physical_attributes[i] = attrs[i];
            InterlockedExchange(&n2->frequency, 1);
            InterlockedExchange(&n2->owner_ticket, 0);
            return true;
        }
    }
    return false;
}

// ==================== SIMD Kernels (mỗi kernel có target riêng) ====================
using ComputeFunc = void (*)(const float*, float*, int, float, float, float);

__attribute__((target("sse2")))
void Kernel_SSE2(const float* in, float* out, int N, float a, float b, float c) {
    __m128 va = _mm_set1_ps(a), vb = _mm_set1_ps(b), vc = _mm_set1_ps(c), vz = _mm_setzero_ps();
    int i = 0;
    for (; i <= N - 16; i += 16) {
        __m128 v0 = _mm_loadu_ps(&in[i]), v1 = _mm_loadu_ps(&in[i+4]);
        __m128 v2 = _mm_loadu_ps(&in[i+8]), v3 = _mm_loadu_ps(&in[i+12]);
        if (_mm_movemask_ps(_mm_cmpeq_ps(_mm_or_ps(_mm_or_ps(v0,v1),_mm_or_ps(v2,v3)), vz)) == 0x0F) {
            _mm_storeu_ps(&out[i], vz); _mm_storeu_ps(&out[i+4], vz);
            _mm_storeu_ps(&out[i+8], vz); _mm_storeu_ps(&out[i+12], vz);
            continue;
        }
        __m128 t0 = _mm_add_ps(_mm_mul_ps(va, v0), vb);
        _mm_storeu_ps(&out[i],   _mm_add_ps(_mm_mul_ps(t0, v0), vc));
        __m128 t1 = _mm_add_ps(_mm_mul_ps(va, v1), vb);
        _mm_storeu_ps(&out[i+4], _mm_add_ps(_mm_mul_ps(t1, v1), vc));
        __m128 t2 = _mm_add_ps(_mm_mul_ps(va, v2), vb);
        _mm_storeu_ps(&out[i+8], _mm_add_ps(_mm_mul_ps(t2, v2), vc));
        __m128 t3 = _mm_add_ps(_mm_mul_ps(va, v3), vb);
        _mm_storeu_ps(&out[i+12],_mm_add_ps(_mm_mul_ps(t3, v3), vc));
    }
    for (; i < N; ++i) {
        float x = in[i];
        out[i] = (x == 0.0f) ? 0.0f : (a * x + b) * x + c;
    }
}

__attribute__((target("sse4.1")))
void Kernel_SSE41(const float* in, float* out, int N, float a, float b, float c) {
    __m128 va = _mm_set1_ps(a), vb = _mm_set1_ps(b), vc = _mm_set1_ps(c), vz = _mm_setzero_ps();
    int i = 0;
    for (; i <= N - 16; i += 16) {
        __m128 v0 = _mm_loadu_ps(&in[i]), v1 = _mm_loadu_ps(&in[i+4]);
        __m128 v2 = _mm_loadu_ps(&in[i+8]), v3 = _mm_loadu_ps(&in[i+12]);
        if (_mm_movemask_ps(_mm_cmpeq_ps(_mm_or_ps(_mm_or_ps(v0,v1),_mm_or_ps(v2,v3)), vz)) == 0x0F) {
            _mm_storeu_ps(&out[i], vz); _mm_storeu_ps(&out[i+4], vz);
            _mm_storeu_ps(&out[i+8], vz); _mm_storeu_ps(&out[i+12], vz);
            continue;
        }
        __m128 t0 = _mm_add_ps(_mm_mul_ps(va, v0), vb);
        _mm_storeu_ps(&out[i],   _mm_add_ps(_mm_mul_ps(t0, v0), vc));
        __m128 t1 = _mm_add_ps(_mm_mul_ps(va, v1), vb);
        _mm_storeu_ps(&out[i+4], _mm_add_ps(_mm_mul_ps(t1, v1), vc));
        __m128 t2 = _mm_add_ps(_mm_mul_ps(va, v2), vb);
        _mm_storeu_ps(&out[i+8], _mm_add_ps(_mm_mul_ps(t2, v2), vc));
        __m128 t3 = _mm_add_ps(_mm_mul_ps(va, v3), vb);
        _mm_storeu_ps(&out[i+12],_mm_add_ps(_mm_mul_ps(t3, v3), vc));
    }
    // Phần dư: blend thay vì rẽ nhánh
    for (; i < N; i += 4) {
        int rem = N - i;
        if (rem >= 4) {
            __m128 vin = _mm_loadu_ps(&in[i]);
            __m128 mask = _mm_cmpeq_ps(vin, vz);
            __m128 t = _mm_add_ps(_mm_mul_ps(va, vin), vb);
            __m128 res = _mm_add_ps(_mm_mul_ps(t, vin), vc);
            res = _mm_blendv_ps(res, vz, mask);
            _mm_storeu_ps(&out[i], res);
        } else {
            for (int j = 0; j < rem; ++j) {
                float x = in[i+j];
                out[i+j] = (x == 0.0f) ? 0.0f : (a * x + b) * x + c;
            }
        }
    }
}

__attribute__((target("avx")))
void Kernel_AVX(const float* in, float* out, int N, float a, float b, float c) {
    __m256 va = _mm256_set1_ps(a), vb = _mm256_set1_ps(b), vc = _mm256_set1_ps(c), vz = _mm256_setzero_ps();
    int i = 0;
    for (; i <= N - 32; i += 32) {
        __m256 v0 = _mm256_loadu_ps(&in[i]), v1 = _mm256_loadu_ps(&in[i+8]);
        __m256 v2 = _mm256_loadu_ps(&in[i+16]), v3 = _mm256_loadu_ps(&in[i+24]);
        if (_mm256_movemask_ps(_mm256_cmp_ps(_mm256_or_ps(_mm256_or_ps(v0,v1),_mm256_or_ps(v2,v3)), vz, _CMP_EQ_OQ)) == 0xFF) {
            _mm256_storeu_ps(&out[i], vz); _mm256_storeu_ps(&out[i+8], vz);
            _mm256_storeu_ps(&out[i+16], vz); _mm256_storeu_ps(&out[i+24], vz);
            continue;
        }
        __m256 t0 = _mm256_add_ps(_mm256_mul_ps(va, v0), vb);
        _mm256_storeu_ps(&out[i],   _mm256_add_ps(_mm256_mul_ps(t0, v0), vc));
        __m256 t1 = _mm256_add_ps(_mm256_mul_ps(va, v1), vb);
        _mm256_storeu_ps(&out[i+8], _mm256_add_ps(_mm256_mul_ps(t1, v1), vc));
        __m256 t2 = _mm256_add_ps(_mm256_mul_ps(va, v2), vb);
        _mm256_storeu_ps(&out[i+16],_mm256_add_ps(_mm256_mul_ps(t2, v2), vc));
        __m256 t3 = _mm256_add_ps(_mm256_mul_ps(va, v3), vb);
        _mm256_storeu_ps(&out[i+24],_mm256_add_ps(_mm256_mul_ps(t3, v3), vc));
    }
    for (; i < N; ++i) {
        float x = in[i];
        out[i] = (x == 0.0f) ? 0.0f : (a * x + b) * x + c;
    }
}

__attribute__((target("avx2,fma")))
void Kernel_AVX2(const float* in, float* out, int N, float a, float b, float c) {
    __m256 va = _mm256_set1_ps(a), vb = _mm256_set1_ps(b), vc = _mm256_set1_ps(c), vz = _mm256_setzero_ps();
    int i = 0;
    for (; i <= N - 32; i += 32) {
        __m256 v0 = _mm256_loadu_ps(&in[i]), v1 = _mm256_loadu_ps(&in[i+8]);
        __m256 v2 = _mm256_loadu_ps(&in[i+16]), v3 = _mm256_loadu_ps(&in[i+24]);
        __m256 orall = _mm256_or_ps(_mm256_or_ps(v0,v1), _mm256_or_ps(v2,v3));
        if (_mm256_testz_ps(orall, orall)) {
            _mm256_storeu_ps(&out[i], vz); _mm256_storeu_ps(&out[i+8], vz);
            _mm256_storeu_ps(&out[i+16], vz); _mm256_storeu_ps(&out[i+24], vz);
            continue;
        }
        __m256 t0 = _mm256_fmadd_ps(va, v0, vb);
        _mm256_storeu_ps(&out[i],   _mm256_fmadd_ps(t0, v0, vc));
        __m256 t1 = _mm256_fmadd_ps(va, v1, vb);
        _mm256_storeu_ps(&out[i+8], _mm256_fmadd_ps(t1, v1, vc));
        __m256 t2 = _mm256_fmadd_ps(va, v2, vb);
        _mm256_storeu_ps(&out[i+16],_mm256_fmadd_ps(t2, v2, vc));
        __m256 t3 = _mm256_fmadd_ps(va, v3, vb);
        _mm256_storeu_ps(&out[i+24],_mm256_fmadd_ps(t3, v3, vc));
    }
    for (; i < N; ++i) {
        float x = in[i];
        out[i] = (x == 0.0f) ? 0.0f : (a * x + b) * x + c;
    }
}

__attribute__((target("avx512f,avx512dq,fma")))
void Kernel_AVX512(const float* in, float* out, int N, float a, float b, float c) {
    __m512 va = _mm512_set1_ps(a), vb = _mm512_set1_ps(b), vc = _mm512_set1_ps(c), vz = _mm512_setzero_ps();
    int i = 0;
    for (; i <= N - 64; i += 64) {
        __m512 v0 = _mm512_loadu_ps(&in[i]), v1 = _mm512_loadu_ps(&in[i+16]);
        __m512 v2 = _mm512_loadu_ps(&in[i+32]), v3 = _mm512_loadu_ps(&in[i+48]);
        __m512 orall = _mm512_or_ps(_mm512_or_ps(v0,v1), _mm512_or_ps(v2,v3));
        if (_mm512_cmp_ps_mask(orall, vz, _CMP_EQ_OQ) == 0xFFFF) {
            _mm512_storeu_ps(&out[i], vz); _mm512_storeu_ps(&out[i+16], vz);
            _mm512_storeu_ps(&out[i+32], vz); _mm512_storeu_ps(&out[i+48], vz);
            continue;
        }
        __m512 t0 = _mm512_fmadd_ps(va, v0, vb);
        _mm512_storeu_ps(&out[i],   _mm512_fmadd_ps(t0, v0, vc));
        __m512 t1 = _mm512_fmadd_ps(va, v1, vb);
        _mm512_storeu_ps(&out[i+16],_mm512_fmadd_ps(t1, v1, vc));
        __m512 t2 = _mm512_fmadd_ps(va, v2, vb);
        _mm512_storeu_ps(&out[i+32],_mm512_fmadd_ps(t2, v2, vc));
        __m512 t3 = _mm512_fmadd_ps(va, v3, vb);
        _mm512_storeu_ps(&out[i+48],_mm512_fmadd_ps(t3, v3, vc));
    }
    for (; i < N; ++i) {
        float x = in[i];
        out[i] = (x == 0.0f) ? 0.0f : (a * x + b) * x + c;
    }
}

ComputeFunc SelectBestKernel() {
    if (s_AVX512) return Kernel_AVX512;
    if (s_AVX2)   return Kernel_AVX2;
    if (s_AVX)    return Kernel_AVX;
    if (s_SSE41)  return Kernel_SSE41;
    return Kernel_SSE2;
}

ComputeFunc g_Compute = nullptr;

// ==================== Named Pipe Server ====================
#define PIPE_NAME L"\\\\.\\pipe\\DragonCompute"

void AsyncWriteMap(std::vector<float> out, int N) {
    for (int i = 0; i < N; ++i) {
        if (out[i] == 0.0f) continue;
        uint64_t key = MortonKeyFromValue(out[i]);
        WriteNode(key, &out[i], 1);
    }
}

void HandlePipe(HANDLE hPipe) {
    DWORD bytesRead, bytesWritten;
    int N = 0;
    if (!ReadFile(hPipe, &N, sizeof(N), &bytesRead, NULL) || bytesRead != sizeof(N) ||
        N <= 0 || N > 100000) {
        DisconnectNamedPipe(hPipe); CloseHandle(hPipe); return;
    }
    int dataSize = N * sizeof(float);
    if (dataSize <= 0) { DisconnectNamedPipe(hPipe); CloseHandle(hPipe); return; }
    std::vector<float> in(N), out(N);
    char* buf = (char*)in.data();
    DWORD total = 0;
    while (total < (DWORD)dataSize) {
        DWORD chunk;
        if (!ReadFile(hPipe, buf + total, dataSize - total, &chunk, NULL) || chunk == 0) {
            DisconnectNamedPipe(hPipe); CloseHandle(hPipe); return;
        }
        total += chunk;
    }
    float params[3];
    if (!ReadFile(hPipe, params, sizeof(params), &bytesRead, NULL) || bytesRead != sizeof(params)) {
        DisconnectNamedPipe(hPipe); CloseHandle(hPipe); return;
    }

    g_Compute(in.data(), out.data(), N, params[0], params[1], params[2]);

    if (!WriteFile(hPipe, out.data(), dataSize, &bytesWritten, NULL) || bytesWritten != (DWORD)dataSize) {
        DisconnectNamedPipe(hPipe); CloseHandle(hPipe); return;
    }
    FlushFileBuffers(hPipe);
    DisconnectNamedPipe(hPipe);
    CloseHandle(hPipe);

    if (mappedMap && mapSize > 0) {
        std::thread(AsyncWriteMap, std::move(out), N).detach();
    }
}

void ServiceLoop() {
    printf("ComputeService listening on pipe %ls\n", PIPE_NAME);
    while (running) {
        HANDLE hPipe = CreateNamedPipeW(PIPE_NAME, PIPE_ACCESS_DUPLEX,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
            PIPE_UNLIMITED_INSTANCES, 1024*1024, 1024*1024, 0, NULL);
        if (hPipe == INVALID_HANDLE_VALUE) { printf("Pipe error\n"); break; }
        if (ConnectNamedPipe(hPipe, NULL) || GetLastError() == ERROR_PIPE_CONNECTED) {
            std::thread(HandlePipe, hPipe).detach();
        } else {
            CloseHandle(hPipe);
        }
    }
}

BOOL WINAPI CtrlHandler(DWORD fdwCtrlType) {
    if (fdwCtrlType == CTRL_C_EVENT || fdwCtrlType == CTRL_CLOSE_EVENT) {
        running = false; Sleep(200); return TRUE;
    }
    return FALSE;
}

int main() {
    DetectCPU();
    g_Compute = SelectBestKernel();
    if (!g_Compute) { printf("No SIMD kernel!\n"); return 1; }
    SetConsoleCtrlHandler(CtrlHandler, TRUE);
    ConnectDriver();
    ServiceLoop();
    if (hDevice != INVALID_HANDLE_VALUE) CloseHandle(hDevice);
    return 0;
}