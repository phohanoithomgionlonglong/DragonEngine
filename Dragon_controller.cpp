/*
 * dragon_controller.cpp - Dragon Spatial-Logic Engine Controller (FIXED)
 * Hỗ trợ đa GPU thực sự: mỗi segment một kernel riêng, set args một lần.
 * Vòng lặp 2ms, workload cố định, mọi ứng dụng đều được hưởng.
 * Chỉ dùng thư viện nội bộ Dev‑C++ và dynamic loading OpenCL.
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>

// ==================== OpenCL Dynamic Loading ====================
#ifndef CL_API_CALL
#define CL_API_CALL __stdcall
#endif

typedef int32_t          cl_int;
typedef uint32_t         cl_uint;
typedef uint64_t         cl_ulong;
typedef cl_uint          cl_bool;
typedef cl_ulong         cl_mem_flags;
typedef cl_ulong         cl_device_type;
typedef intptr_t         cl_context_properties;
typedef intptr_t         cl_command_queue_properties;
typedef intptr_t         cl_map_flags;
typedef void*            cl_platform_id;
typedef void*            cl_device_id;
typedef void*            cl_context;
typedef void*            cl_command_queue;
typedef void*            cl_mem;
typedef void*            cl_program;
typedef void*            cl_kernel;
typedef void*            cl_event;

#define CL_SUCCESS                    0
#define CL_TRUE                       1
#define CL_FALSE                      0
#define CL_DEVICE_TYPE_GPU            (1 << 2)
#define CL_MEM_READ_WRITE             (1 << 0)
#define CL_MEM_USE_HOST_PTR           (1 << 3)
#define CL_MEM_COPY_HOST_PTR          (1 << 5)
#define CL_MEM_READ_ONLY              (1 << 2)
#define CL_MEM_WRITE_ONLY             (1 << 3)
#define CL_PROGRAM_BUILD_LOG          0x1183
#define CL_BUFFER_CREATE_TYPE_REGION  0x1220

// Định nghĩa cl_buffer_region (không có trong header, tự thêm)
typedef struct _cl_buffer_region {
    size_t origin;
    size_t size;
} cl_buffer_region;

// Kiểu cl_buffer_create_type (thực ra là cl_uint, nhưng dùng intptr_t cho an toàn)
typedef cl_uint cl_buffer_create_type;

typedef cl_int              (CL_API_CALL *PCL_GET_PLATFORM_IDS)         (cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int              (CL_API_CALL *PCL_GET_DEVICE_IDS)           (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context          (CL_API_CALL *PCL_CREATE_CONTEXT)           (const cl_context_properties*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue    (CL_API_CALL *PCL_CREATE_COMMAND_QUEUE)     (cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_program          (CL_API_CALL *PCL_CREATE_PROGRAM_WITH_SOURCE)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int              (CL_API_CALL *PCL_BUILD_PROGRAM)            (cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_kernel           (CL_API_CALL *PCL_CREATE_KERNEL)            (cl_program, const char*, cl_int*);
typedef cl_int              (CL_API_CALL *PCL_SET_KERNEL_ARG)           (cl_kernel, cl_uint, size_t, const void*);
typedef cl_int              (CL_API_CALL *PCL_ENQUEUE_ND_RANGE_KERNEL)  (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int              (CL_API_CALL *PCL_FINISH)                   (cl_command_queue);
typedef cl_int              (CL_API_CALL *PCL_RELEASE_MEM_OBJECT)       (cl_mem);
typedef cl_int              (CL_API_CALL *PCL_RELEASE_KERNEL)           (cl_kernel);
typedef cl_int              (CL_API_CALL *PCL_RELEASE_PROGRAM)          (cl_program);
typedef cl_int              (CL_API_CALL *PCL_RELEASE_COMMAND_QUEUE)    (cl_command_queue);
typedef cl_int              (CL_API_CALL *PCL_RELEASE_CONTEXT)          (cl_context);
typedef cl_mem              (CL_API_CALL *PCL_CREATE_BUFFER)            (cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int              (CL_API_CALL *PCL_GET_PROGRAM_BUILD_INFO)   (cl_program, cl_device_id, cl_int, size_t, void*, size_t*);
typedef cl_int              (CL_API_CALL *PCL_ENQUEUE_READ_BUFFER)      (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int              (CL_API_CALL *PCL_ENQUEUE_WRITE_BUFFER)     (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_mem              (CL_API_CALL *PCL_CREATE_SUB_BUFFER)        (cl_mem, cl_mem_flags, cl_buffer_create_type, const void*, cl_int*);
typedef cl_int              (CL_API_CALL *PCL_WAIT_FOR_EVENTS)         (cl_uint, const cl_event*);
typedef cl_int              (CL_API_CALL *PCL_RELEASE_EVENT)           (cl_event);

static PCL_GET_PLATFORM_IDS          pclGetPlatformIDs          = nullptr;
static PCL_GET_DEVICE_IDS            pclGetDeviceIDs            = nullptr;
static PCL_CREATE_CONTEXT            pclCreateContext           = nullptr;
static PCL_CREATE_COMMAND_QUEUE      pclCreateCommandQueue      = nullptr;
static PCL_CREATE_PROGRAM_WITH_SOURCE pclCreateProgramWithSource = nullptr;
static PCL_BUILD_PROGRAM             pclBuildProgram            = nullptr;
static PCL_CREATE_KERNEL             pclCreateKernel            = nullptr;
static PCL_SET_KERNEL_ARG            pclSetKernelArg            = nullptr;
static PCL_ENQUEUE_ND_RANGE_KERNEL   pclEnqueueNDRangeKernel    = nullptr;
static PCL_FINISH                    pclFinish                  = nullptr;
static PCL_RELEASE_MEM_OBJECT        pclReleaseMemObject        = nullptr;
static PCL_RELEASE_KERNEL            pclReleaseKernel           = nullptr;
static PCL_RELEASE_PROGRAM           pclReleaseProgram          = nullptr;
static PCL_RELEASE_COMMAND_QUEUE     pclReleaseCommandQueue     = nullptr;
static PCL_RELEASE_CONTEXT           pclReleaseContext          = nullptr;
static PCL_CREATE_BUFFER             pclCreateBuffer            = nullptr;
static PCL_GET_PROGRAM_BUILD_INFO    pclGetProgramBuildInfo     = nullptr;
static PCL_ENQUEUE_READ_BUFFER       pclEnqueueReadBuffer       = nullptr;
static PCL_ENQUEUE_WRITE_BUFFER      pclEnqueueWriteBuffer      = nullptr;
static PCL_CREATE_SUB_BUFFER         pclCreateSubBuffer         = nullptr;
static PCL_WAIT_FOR_EVENTS           pclWaitForEvents           = nullptr;
static PCL_RELEASE_EVENT            pclReleaseEvent            = nullptr;

bool LoadOpenCL() {
    HMODULE h = LoadLibraryW(L"OpenCL.dll");
    if (!h) return false;

    pclGetPlatformIDs          = (PCL_GET_PLATFORM_IDS)         GetProcAddress(h, "clGetPlatformIDs");
    pclGetDeviceIDs            = (PCL_GET_DEVICE_IDS)           GetProcAddress(h, "clGetDeviceIDs");
    pclCreateContext           = (PCL_CREATE_CONTEXT)           GetProcAddress(h, "clCreateContext");
    pclCreateCommandQueue      = (PCL_CREATE_COMMAND_QUEUE)     GetProcAddress(h, "clCreateCommandQueue");
    pclCreateProgramWithSource = (PCL_CREATE_PROGRAM_WITH_SOURCE)GetProcAddress(h, "clCreateProgramWithSource");
    pclBuildProgram            = (PCL_BUILD_PROGRAM)            GetProcAddress(h, "clBuildProgram");
    pclCreateKernel            = (PCL_CREATE_KERNEL)            GetProcAddress(h, "clCreateKernel");
    pclSetKernelArg            = (PCL_SET_KERNEL_ARG)           GetProcAddress(h, "clSetKernelArg");
    pclEnqueueNDRangeKernel    = (PCL_ENQUEUE_ND_RANGE_KERNEL)  GetProcAddress(h, "clEnqueueNDRangeKernel");
    pclFinish                  = (PCL_FINISH)                   GetProcAddress(h, "clFinish");
    pclReleaseMemObject        = (PCL_RELEASE_MEM_OBJECT)       GetProcAddress(h, "clReleaseMemObject");
    pclReleaseKernel           = (PCL_RELEASE_KERNEL)           GetProcAddress(h, "clReleaseKernel");
    pclReleaseProgram          = (PCL_RELEASE_PROGRAM)          GetProcAddress(h, "clReleaseProgram");
    pclReleaseCommandQueue     = (PCL_RELEASE_COMMAND_QUEUE)    GetProcAddress(h, "clReleaseCommandQueue");
    pclReleaseContext          = (PCL_RELEASE_CONTEXT)          GetProcAddress(h, "clReleaseContext");
    pclCreateBuffer            = (PCL_CREATE_BUFFER)            GetProcAddress(h, "clCreateBuffer");
    pclGetProgramBuildInfo     = (PCL_GET_PROGRAM_BUILD_INFO)   GetProcAddress(h, "clGetProgramBuildInfo");
    pclEnqueueReadBuffer       = (PCL_ENQUEUE_READ_BUFFER)      GetProcAddress(h, "clEnqueueReadBuffer");
    pclEnqueueWriteBuffer      = (PCL_ENQUEUE_WRITE_BUFFER)     GetProcAddress(h, "clEnqueueWriteBuffer");
    pclCreateSubBuffer         = (PCL_CREATE_SUB_BUFFER)        GetProcAddress(h, "clCreateSubBuffer");
    pclWaitForEvents           = (PCL_WAIT_FOR_EVENTS)          GetProcAddress(h, "clWaitForEvents");
    pclReleaseEvent            = (PCL_RELEASE_EVENT)            GetProcAddress(h, "clReleaseEvent");

    return (pclGetPlatformIDs && pclGetDeviceIDs && pclCreateContext && pclCreateCommandQueue &&
            pclCreateProgramWithSource && pclBuildProgram && pclCreateKernel && pclSetKernelArg &&
            pclEnqueueNDRangeKernel && pclFinish && pclReleaseMemObject && pclReleaseKernel &&
            pclReleaseProgram && pclReleaseCommandQueue && pclReleaseContext && pclCreateBuffer &&
            pclGetProgramBuildInfo && pclEnqueueReadBuffer && pclEnqueueWriteBuffer &&
            pclCreateSubBuffer && pclWaitForEvents && pclReleaseEvent);
}

// ==================== Structures matching driver ====================
struct alignas(64) SpatialNode {
    uint64_t spatial_key;
    uint32_t owner_ticket;
    uint32_t frequency;
    uint64_t lookup_pointer;
    float    physical_attributes[10];
};

#define FILE_DEVICE_DRAGONENGINE 0x8000
#define IOCTL_DRAGON_START        CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x800, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_DRAGON_STOP         CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x801, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_DRAGON_MAP_MEMORY   CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x802, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_DRAGON_PAUSE_RESUME CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x803, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_DRAGON_QUERY_STATS  CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x804, METHOD_BUFFERED, FILE_ANY_ACCESS)

// ==================== Fused Kernel Source ====================
static const char* kernelSource = R"(
typedef struct __attribute__((aligned(64))) {
    ulong  spatial_key;
    uint   owner_ticket;
    uint   frequency;
    ulong  lookup_pointer;
    float  attributes[10];
} SpatialNode;

ulong morton_key(uint x, uint y, uint z) {
    ulong m = 0;
    for (int i=0; i<21; i++) {
        ulong bx = (x>>i)&1, by = (y>>i)&1, bz = (z>>i)&1;
        m |= (bx << (3*i)) | (by << (3*i+1)) | (bz << (3*i+2));
    }
    return m;
}

kernel void fused_kernel(global volatile SpatialNode* map, ulong mapSize,
                        global const float* input, global float* output,
                        const uint width, const float param, const int MAX_PROBE) {
    uint gid = get_global_id(0);
    float v = input[gid];
    uint x=(uint)(v*1000), y=(uint)(v*2000), z=(uint)(v*3000);
    ulong key = morton_key(x,y,z);
    ulong idx = key & (mapSize - 1);   // mapSize là lũy thừa 2

    // Lookup
    global volatile SpatialNode* node = &map[idx];
    if (node->spatial_key == key) {
        output[gid] = node->attributes[gid%10];
        atomic_inc(&node->frequency);   // tăng tần suất sử dụng
        return;
    }

    // Compute
    float result = v;
    result = fma(result, param, result*0.5f);
    result = fma(result, param, result*0.3f);
    result = native_sqrt(result);
    output[gid] = result;

    // Evolution
    uint ticket = (gid | 0x80000000);
    uint old = atomic_cmpxchg(&node->owner_ticket, 0, ticket);
    if (old == 0 || old == ticket) {
        node->spatial_key = key;
        node->attributes[gid%10] = result;
        atomic_inc(&node->frequency);
        atomic_xchg(&node->owner_ticket, 0);
        return;
    }

    for (int p=1; p<MAX_PROBE; p++) {
        ulong idx2 = (idx + p) & (mapSize - 1);
        global SpatialNode* n2 = &map[idx2];
        old = atomic_cmpxchg(&n2->owner_ticket, 0, ticket);
        if (old == 0 || old == ticket) {
            n2->spatial_key = key;
            n2->attributes[gid%10] = result;
            atomic_inc(&n2->frequency);
            atomic_xchg(&n2->owner_ticket, 0);
            return;
        }
    }

    // Forced eviction
    atomic_xchg(&node->owner_ticket, ticket);
    node->spatial_key = key;
    node->attributes[gid%10] = result;
    atomic_inc(&node->frequency);
    atomic_xchg(&node->owner_ticket, 0);
}
)";

// ==================== Driver Installation ====================
static bool IsAdmin() {
    BOOL isAdmin = FALSE;
    PSID adminGroup;
    SID_IDENTIFIER_AUTHORITY ntAuthority = SECURITY_NT_AUTHORITY;
    if (AllocateAndInitializeSid(&ntAuthority, 2, SECURITY_BUILTIN_DOMAIN_RID,
                                DOMAIN_ALIAS_RID_ADMINS, 0, 0, 0, 0, 0, 0, &adminGroup)) {
        CheckTokenMembership(NULL, adminGroup, &isAdmin);
        FreeSid(adminGroup);
    }
    return isAdmin != FALSE;
}

static void ElevateToAdmin() {
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    ShellExecuteA(NULL, "runas", exePath, NULL, NULL, SW_SHOWNORMAL);
    ExitProcess(0);
}

static bool InstallAndStartDriver(const char* sysPath) {
    SC_HANDLE scm = OpenSCManager(NULL, NULL, SC_MANAGER_ALL_ACCESS);
    if (!scm) return false;
    SC_HANDLE svc = OpenServiceA(scm, "DragonEngine", SERVICE_ALL_ACCESS);
    if (!svc) {
        svc = CreateServiceA(scm, "DragonEngine", "Dragon Spatial Engine",
            SERVICE_ALL_ACCESS, SERVICE_KERNEL_DRIVER, SERVICE_DEMAND_START,
            SERVICE_ERROR_NORMAL, sysPath, NULL, NULL, NULL, NULL, NULL);
        if (!svc) { CloseServiceHandle(scm); return false; }
    }
    SERVICE_STATUS status;
    if (!QueryServiceStatus(svc, &status) || status.dwCurrentState != SERVICE_RUNNING)
        StartServiceA(svc, 0, NULL);
    CloseServiceHandle(svc);
    CloseServiceHandle(scm);
    return true;
}

// ==================== DragonEngine Class ====================
class DragonEngine {
    HANDLE hDevice = INVALID_HANDLE_VALUE;
    SpatialNode* mappedMap = nullptr;
    size_t mapSize = 0;               // tổng số node

    cl_context context = nullptr;
    std::vector<cl_device_id> devices;
    std::vector<cl_command_queue> queues;
    cl_program program = nullptr;
    std::vector<cl_kernel> kernels;   // mỗi segment một kernel riêng
    std::vector<cl_mem> subMapBuffers; // sub-buffer cho từng segment
    cl_mem wholeMapBuffer = nullptr;

    size_t segmentSize = 0;           // kích thước mỗi segment (lũy thừa 2)

    std::atomic<bool> running{true};

public:
    bool ConnectDriver() {
        hDevice = CreateFileW(L"\\\\.\\DragonEngine", GENERIC_READ | GENERIC_WRITE,
                              0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hDevice == INVALID_HANDLE_VALUE) { printf("Driver not found.\n"); return false; }
        DWORD ret;
        SIZE_T stats[2];
        DeviceIoControl(hDevice, IOCTL_DRAGON_QUERY_STATS, NULL, 0, stats, sizeof(stats), &ret, NULL);
        mapSize = stats[0];
        PVOID uAddr = nullptr;
        DeviceIoControl(hDevice, IOCTL_DRAGON_MAP_MEMORY, NULL, 0, &uAddr, sizeof(uAddr), &ret, NULL);
        mappedMap = (SpatialNode*)uAddr;
        return mappedMap != nullptr;
    }

    bool InitOpenCL() {
        if (!LoadOpenCL()) return false;
        cl_uint nplat;
        pclGetPlatformIDs(0, NULL, &nplat);
        std::vector<cl_platform_id> plats(nplat);
        pclGetPlatformIDs(nplat, plats.data(), NULL);
        for (auto& p : plats) {
            cl_uint nd;
            pclGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, NULL, &nd);
            if (nd) {
                std::vector<cl_device_id> devs(nd);
                pclGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), NULL);
                devices.insert(devices.end(), devs.begin(), devs.end());
            }
        }
        if (devices.empty()) { printf("No GPU.\n"); return false; }

        cl_int err;
        context = pclCreateContext(NULL, (cl_uint)devices.size(), devices.data(), NULL, NULL, &err);
        if (err != CL_SUCCESS) return false;
        size_t srcLen = strlen(kernelSource);
        program = pclCreateProgramWithSource(context, 1, &kernelSource, &srcLen, &err);
        if (err != CL_SUCCESS) return false;
        err = pclBuildProgram(program, (cl_uint)devices.size(), devices.data(), "-cl-std=CL2.0", NULL, NULL);
        if (err != CL_SUCCESS) {
            char log[4096];
            pclGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
            printf("Build error: %s\n", log);
            return false;
        }

        // Tạo command queues (mỗi GPU một queue)
        for (size_t i = 0; i < devices.size(); i++) {
            cl_command_queue q = pclCreateCommandQueue(context, devices[i], 0, &err);
            if (err != CL_SUCCESS) return false;
            queues.push_back(q);
        }

        wholeMapBuffer = pclCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                         mapSize * sizeof(SpatialNode), mappedMap, &err);
        if (err != CL_SUCCESS) return false;

        // Chia map thành các segment lũy thừa 2
        size_t numGPUs = devices.size();
        // Tính segSize là lũy thừa 2 lớn nhất ≤ mapSize / numGPUs
        size_t segSize = 1;
        while (segSize * 2 <= mapSize / numGPUs) segSize <<= 1;
        segmentSize = segSize;
        size_t numSegments = mapSize / segSize;   // số segment thực sự

        // Tạo sub-buffers và kernels cho từng segment
        subMapBuffers.clear();
        kernels.clear();
        for (size_t i = 0; i < numSegments; i++) {
            size_t offset = i * segSize;
            cl_buffer_region region = { offset * sizeof(SpatialNode), segSize * sizeof(SpatialNode) };
            cl_int err2;
            cl_mem sub = pclCreateSubBuffer(wholeMapBuffer, CL_MEM_READ_WRITE,
                                            CL_BUFFER_CREATE_TYPE_REGION, &region, &err2);
            if (err2 != CL_SUCCESS) return false;
            subMapBuffers.push_back(sub);

            // Kernel riêng cho segment này
            cl_kernel k = pclCreateKernel(program, "fused_kernel", &err);
            if (err != CL_SUCCESS) return false;
            kernels.push_back(k);
        }

        return true;
    }

    void ProcessLoop() {
        size_t numSegments = subMapBuffers.size();
        if (numSegments == 0) return;

        const size_t totalThreads = 65536;
        size_t threadsPerSegment = totalThreads / numSegments;
        size_t remainder = totalThreads % numSegments;

        std::vector<std::vector<float>> inputs(numSegments);
        std::vector<std::vector<float>> outputs(numSegments);
        std::vector<cl_mem> inBuffers(numSegments);
        std::vector<cl_mem> outBuffers(numSegments);
        cl_int err;

        for (size_t i = 0; i < numSegments; i++) {
            size_t thr = threadsPerSegment + (i == numSegments - 1 ? remainder : 0);
            inputs[i].resize(thr, 0.5f);
            outputs[i].resize(thr);
            inBuffers[i] = pclCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           thr * sizeof(float), inputs[i].data(), &err);
            if (err != CL_SUCCESS) return;
            outBuffers[i] = pclCreateBuffer(context, CL_MEM_READ_WRITE,
                                            thr * sizeof(float), NULL, &err);
            if (err != CL_SUCCESS) return;
        }

        float param = 3.14159f;
        int maxProbe = 64;

        // Set kernel arguments một lần trước vòng lặp (vì chúng không thay đổi)
        for (size_t i = 0; i < numSegments; i++) {
            cl_kernel k = kernels[i];
            cl_uint dim = (cl_uint)inputs[i].size();

            pclSetKernelArg(k, 0, sizeof(cl_mem), &subMapBuffers[i]);
            pclSetKernelArg(k, 1, sizeof(segmentSize), &segmentSize);
            pclSetKernelArg(k, 2, sizeof(cl_mem), &inBuffers[i]);
            pclSetKernelArg(k, 3, sizeof(cl_mem), &outBuffers[i]);
            pclSetKernelArg(k, 4, sizeof(cl_uint), &dim);
            pclSetKernelArg(k, 5, sizeof(float), &param);
            pclSetKernelArg(k, 6, sizeof(int), &maxProbe);
        }

        while (running) {
            std::vector<cl_event> events(numSegments, nullptr);

            for (size_t i = 0; i < numSegments; i++) {
                size_t global = inputs[i].size();
                err = pclEnqueueNDRangeKernel(queues[i % queues.size()], kernels[i], 1, NULL, &global, NULL, 0, NULL, &events[i]);
                if (err != CL_SUCCESS) { printf("Kernel enqueue error on segment %zu\n", i); break; }
            }

            // Chờ tất cả GPU hoàn thành
            if (pclWaitForEvents((cl_uint)numSegments, events.data()) != CL_SUCCESS) {
                // fallback nếu wait thất bại
                for (size_t i = 0; i < numSegments; i++) pclFinish(queues[i % queues.size()]);
            }

            // Giải phóng events
            for (auto& ev : events) if (ev) pclReleaseEvent(ev);

            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        for (size_t i = 0; i < numSegments; i++) {
            pclReleaseMemObject(inBuffers[i]);
            pclReleaseMemObject(outBuffers[i]);
        }
    }

    void Run() {
        printf("Dragon Engine starting...\n");
        if (!ConnectDriver() || !InitOpenCL()) {
            printf("Initialization failed.\n");
            return;
        }
        DeviceIoControl(hDevice, IOCTL_DRAGON_START, NULL, 0, NULL, 0, NULL, NULL);
        printf("Dragon Engine running (multi-GPU, 2ms loop). All apps benefit. Press Ctrl+C to stop.\n");
        ProcessLoop();
        DeviceIoControl(hDevice, IOCTL_DRAGON_STOP, NULL, 0, NULL, 0, NULL, NULL);
        for (auto& sub : subMapBuffers) pclReleaseMemObject(sub);
        if (wholeMapBuffer) pclReleaseMemObject(wholeMapBuffer);
        for (auto k : kernels) pclReleaseKernel(k);
        if (program) pclReleaseProgram(program);
        for (auto q : queues) pclReleaseCommandQueue(q);
        if (context) pclReleaseContext(context);
        CloseHandle(hDevice);
    }
};

int main() {
    if (!IsAdmin()) { ElevateToAdmin(); return 0; }
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);
    char* slash = strrchr(exePath, '\\');
    if (slash) *slash = '\0';
    char sysPath[MAX_PATH];
    snprintf(sysPath, MAX_PATH, "%s\\Dragon.sys", exePath);
    if (!InstallAndStartDriver(sysPath)) {
        printf("Failed to install/start driver.\n");
        system("pause");
        return 1;
    }
    printf("Driver ready.\n");
    DragonEngine engine;
    engine.Run();
    printf("Engine stopped.\n");
    system("pause");
    return 0;
}