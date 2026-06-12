/*
 * Dragon.c - Dragon Spatial-Logic Engine Kernel Driver (FIXED)
 * - ShardCount là lũy thừa 2 → mask an toàn.
 * - Watchdog polling nhẹ (100ms), chỉ quét khi cần.
 * - Worker thread hỗ trợ lão hóa node (frequency được tăng bởi GPU, giảm bởi CPU).
 * - Ring Buffer, Heartbeat, Zero‑copy mapping.
 */
#include <ntddk.h>
#include <wdm.h>
#include <intrin.h>

#pragma warning(disable: 4996)

// ==================== Constants ====================
#define CACHE_LINE_SIZE        64
#define MAX_PROBE_DEPTH        64
#define MAX_CUCKOO_LOOP        32
#define HEARTBEAT_BUFFER_SIZE  (1024 * 1024)
#define RING_BUFFER_SIZE       (64 * 1024 * 1024)

#define HB_IDLE                0x00
#define HB_ERROR_CONFLICT      0xFF

#define FILE_DEVICE_DRAGONENGINE 0x8000
#define IOCTL_DRAGON_START        CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x800, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_DRAGON_STOP         CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x801, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_DRAGON_MAP_MEMORY   CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x802, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_DRAGON_PAUSE_RESUME CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x803, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define IOCTL_DRAGON_QUERY_STATS  CTL_CODE(FILE_DEVICE_DRAGONENGINE, 0x804, METHOD_BUFFERED, FILE_ANY_ACCESS)

// ==================== Prototype ====================
NTSYSAPI NTSTATUS NTAPI ZwQuerySystemInformation(
    ULONG SystemInformationClass,
    PVOID SystemInformation,
    ULONG SystemInformationLength,
    PULONG ReturnLength
);

typedef struct _MY_SYSTEM_BASIC_INFORMATION {
    ULONG       Reserved;
    ULONG       TimerResolution;
    ULONG       PageSize;
    ULONG       NumberOfPhysicalPages;
    ULONG       LowestPhysicalPageNumber;
    ULONG       HighestPhysicalPageNumber;
    ULONG       AllocationGranularity;
    ULONG_PTR   MinimumUserModeAddress;
    ULONG_PTR   MaximumUserModeAddress;
    KAFFINITY   ActiveProcessorsAffinityMask;
    CCHAR       NumberOfProcessors;
} MY_SYSTEM_BASIC_INFORMATION;

#define SystemBasicInformation 0

// ==================== Structures ====================
typedef struct DECLSPEC_ALIGN(64) _SPATIAL_NODE {
    UINT64  spatial_key;
    LONG    owner_ticket;
    volatile LONG frequency;   // dùng Interlocked
    UINT64  lookup_pointer;
    float   physical_attributes[10];
} SPATIAL_NODE, *PSPATIAL_NODE;
C_ASSERT(sizeof(SPATIAL_NODE) == 64);

typedef struct _SHARD {
    PSPATIAL_NODE  base;
    SIZE_T         size;       // lũy thừa 2
    KSPIN_LOCK     lock;
    UINT64         hashSeed1;
    UINT64         hashSeed2;
} SHARD, *PSHARD;

typedef struct _DEVICE_EXTENSION {
    PDEVICE_OBJECT   DeviceObject;
    PSPATIAL_NODE    KnowledgeMap;
    SIZE_T           MapSize;               // toàn bộ node (lũy thừa 2)
    SIZE_T           MapSizeMask;
    PSHARD           Shards;
    UINT32           ShardCount;            // lũy thừa 2 ≤ số lõi
    UINT8*           HeartbeatBuffer;
    UINT8*           RingBuffer;
    SIZE_T           RingBufferWritePos;
    KSPIN_LOCK       RingBufferLock;
    KEVENT           WatchdogEvent;
    PKTHREAD         WatchdogThread;
    PKTHREAD         WorkerThreads[32];
    volatile LONG    EngineActive;
    volatile LONG    PriorityProcessActive;
    PMDL             MapMdl;
    PVOID            UserMappedAddress;
    KEVENT           ShutdownEvent;
    volatile LONG    HeartbeatErrorFlag;    // cờ để watchdog biết có lỗi
} DEVICE_EXTENSION, *PDEVICE_EXTENSION;

typedef struct _THREAD_CONTEXT {
    PDEVICE_EXTENSION devExt;
    UINT32            shardIndex;
} THREAD_CONTEXT, *PTHREAD_CONTEXT;

// ==================== Morton Key ====================
static UINT64 MortonKey(UINT32 x, UINT32 y, UINT32 z) {
    UINT64 morton = 0;
    for (int i = 0; i < 21; i++) {
        UINT64 bx = (x >> i) & 1;
        UINT64 by = (y >> i) & 1;
        UINT64 bz = (z >> i) & 1;
        morton |= (bx << (3 * i)) | (by << (3 * i + 1)) | (bz << (3 * i + 2));
    }
    return morton;
}

// ==================== Hash Functions (mask) ====================
static SIZE_T Hash1(UINT64 key, SIZE_T mask, UINT64 seed) {
    key ^= seed;
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);
    key = key ^ (key >> 28);
    return key & mask;
}

static SIZE_T Hash2(UINT64 key, SIZE_T mask, UINT64 seed) {
    key ^= seed;
    key = key ^ (key >> 31) ^ (key >> 62);
    key *= 0x9E3779B97F4A7C15ULL;
    key ^= key >> 28;
    key *= 0xBF58476D1CE4E5B9ULL;
    key ^= key >> 33;
    return key & mask;
}

// ==================== Force Eviction ====================
static VOID ForceEvict(PDEVICE_EXTENSION devExt, PSHARD shard, SIZE_T idx) {
    PSPATIAL_NODE node = &shard->base[idx];
    if (!devExt->RingBuffer) return;
    KIRQL oldIrql;
    KeAcquireSpinLock(&devExt->RingBufferLock, &oldIrql);
    SIZE_T writeOff = devExt->RingBufferWritePos;
    devExt->RingBufferWritePos = (writeOff + sizeof(SPATIAL_NODE)) % RING_BUFFER_SIZE;
    KeReleaseSpinLock(&devExt->RingBufferLock, oldIrql);
    RtlCopyMemory(&devExt->RingBuffer[writeOff], node, sizeof(SPATIAL_NODE));
    RtlZeroMemory(node, sizeof(SPATIAL_NODE));
}

// ==================== Passport-Key Locking ====================
static BOOLEAN AcquirePassport(PSPATIAL_NODE node, UINT64 newKey, float val, UINT32 tid) {
    volatile LONG* ticket = &node->owner_ticket;
    LONG desired = (LONG)(tid | 0x80000000);
    LONG old = InterlockedCompareExchange(ticket, desired, 0);
    if (old == 0 || old == desired) {
        node->spatial_key = newKey;
        node->physical_attributes[tid % 10] = val;
        InterlockedExchange(&node->frequency, 1);   // reset tần suất
        InterlockedExchange(ticket, 0);
        return TRUE;
    }
    return FALSE;
}

// ==================== Cuckoo Insert ====================
static BOOLEAN CuckooInsertShard(PDEVICE_EXTENSION devExt, PSHARD shard, UINT64 key, float val, UINT32 tid) {
    SIZE_T mask = shard->size - 1;
    SIZE_T pos1 = Hash1(key, mask, shard->hashSeed1);
    SIZE_T pos2 = Hash2(key, mask, shard->hashSeed2);
    for (int loop = 0; loop < MAX_CUCKOO_LOOP; loop++) {
        if (AcquirePassport(&shard->base[pos1], key, val, tid)) return TRUE;
        if (shard->base[pos1].spatial_key == key) return TRUE;
        SPATIAL_NODE temp = shard->base[pos1];
        if (AcquirePassport(&shard->base[pos1], key, val, tid)) return TRUE;
        if (temp.spatial_key == 0) return TRUE;
        key = temp.spatial_key;
        val = temp.physical_attributes[tid % 10];
        SIZE_T newPos = (pos1 == Hash1(key, mask, shard->hashSeed1)) ?
                         Hash2(key, mask, shard->hashSeed2) :
                         Hash1(key, mask, shard->hashSeed1);
        pos1 = newPos;
        pos2 = (newPos == pos1) ? Hash2(key, mask, shard->hashSeed2) : Hash1(key, mask, shard->hashSeed1);
    }
    ForceEvict(devExt, shard, pos1);
    return AcquirePassport(&shard->base[pos1], key, val, tid);
}

// ==================== Node Compression (Mean) ====================
static VOID CompressNode(PSPATIAL_NODE node) {
    float mean = 0.0f;
    for (int i = 0; i < 10; i++) mean += node->physical_attributes[i];
    mean /= 10.0f;
    node->physical_attributes[0] = mean;
}

// ==================== Worker Thread ====================
VOID EngineWorkerThread(PVOID Context) {
    PTHREAD_CONTEXT ctx = (PTHREAD_CONTEXT)Context;
    PDEVICE_EXTENSION devExt = ctx->devExt;
    UINT32 shardIndex = ctx->shardIndex;
    ExFreePoolWithTag(ctx, 'THRD');

    KeSetSystemAffinityThread(1ULL << shardIndex);
    PSHARD myShard = &devExt->Shards[shardIndex % devExt->ShardCount];

    while (devExt->EngineActive) {
        LARGE_INTEGER delay;
        if (devExt->PriorityProcessActive)
            delay.QuadPart = -200 * 10000;  // 200 ms khi có game
        else
            delay.QuadPart = -5000 * 10000; // 5 giây khi không

        KeDelayExecutionThread(KernelMode, FALSE, &delay);
        if (!devExt->EngineActive) break;

        // Duyệt toàn bộ shard
        for (SIZE_T i = 0; i < myShard->size; i++) {
            PSPATIAL_NODE node = &myShard->base[i];
            if (node->spatial_key == 0) continue;

            LONG freq = InterlockedDecrement(&node->frequency);
            if (freq <= 0) {
                // Quá cũ → nén và có thể xóa
                CompressNode(node);
                if (node->physical_attributes[0] == 0.0f)
                    node->spatial_key = 0; // xóa
                else
                    InterlockedExchange(&node->frequency, 0); // giữ lại nhưng đánh dấu đã nén
            }
        }

        // Kiểm tra heartbeat để báo thức watchdog nếu cần
        if (InterlockedCompareExchange(&devExt->HeartbeatErrorFlag, 0, 1) == 1)
            KeSetEvent(&devExt->WatchdogEvent, 0, FALSE);
    }
    PsTerminateSystemThread(STATUS_SUCCESS);
}

// ==================== Watchdog Thread ====================
VOID WatchdogThread(PVOID Context) {
    PDEVICE_EXTENSION devExt = (PDEVICE_EXTENSION)Context;
    LARGE_INTEGER timeout;
    timeout.QuadPart = -100 * 10000; // 100 ms
    while (devExt->EngineActive) {
        KeWaitForSingleObject(&devExt->WatchdogEvent, Executive, KernelMode, FALSE, &timeout);
        if (!devExt->EngineActive) break;

        // Quét heartbeat khi có lỗi hoặc định kỳ
        for (SIZE_T i = 0; i < HEARTBEAT_BUFFER_SIZE; i++) {
            if (devExt->HeartbeatBuffer[i] == HB_ERROR_CONFLICT) {
                SIZE_T shardIdx = i % devExt->ShardCount;
                SIZE_T nodeIdx = (i / devExt->ShardCount) % (devExt->MapSize / devExt->ShardCount);
                ForceEvict(devExt, &devExt->Shards[shardIdx], nodeIdx);
                devExt->HeartbeatBuffer[i] = HB_IDLE;
            }
        }
        InterlockedExchange(&devExt->HeartbeatErrorFlag, 0);
    }
    PsTerminateSystemThread(STATUS_SUCCESS);
}

// ==================== Dispatch ====================
NTSTATUS DispatchCreateClose(PDEVICE_OBJECT d, PIRP Irp) {
    UNREFERENCED_PARAMETER(d);
    Irp->IoStatus.Status = STATUS_SUCCESS;
    IoCompleteRequest(Irp, IO_NO_INCREMENT);
    return STATUS_SUCCESS;
}

NTSTATUS DispatchDeviceControl(PDEVICE_OBJECT d, PIRP Irp) {
    PIO_STACK_LOCATION s = IoGetCurrentIrpStackLocation(Irp);
    PDEVICE_EXTENSION devExt = (PDEVICE_EXTENSION)d->DeviceExtension;
    NTSTATUS st = STATUS_SUCCESS;
    ULONG_PTR info = 0;
    PVOID buf = Irp->AssociatedIrp.SystemBuffer;

    switch (s->Parameters.DeviceIoControl.IoControlCode) {
    case IOCTL_DRAGON_START:
        InterlockedExchange(&devExt->EngineActive, 1);
        break;
    case IOCTL_DRAGON_STOP:
        InterlockedExchange(&devExt->EngineActive, 0);
        KeSetEvent(&devExt->ShutdownEvent, 0, FALSE);
        break;
    case IOCTL_DRAGON_MAP_MEMORY: {
        if (!buf) { st = STATUS_INVALID_PARAMETER; break; }
        ULONG total = (ULONG)(devExt->MapSize * sizeof(SPATIAL_NODE));
        if (total > 0xFFFFFFFF) { st = STATUS_INSUFFICIENT_RESOURCES; break; }
        PMDL mdl = IoAllocateMdl(devExt->KnowledgeMap, total, FALSE, FALSE, NULL);
        if (!mdl) { st = STATUS_INSUFFICIENT_RESOURCES; break; }
        MmBuildMdlForNonPagedPool(mdl);
        PVOID uAddr = MmMapLockedPagesSpecifyCache(mdl, UserMode, MmCached, NULL, FALSE, NormalPagePriority);
        if (!uAddr) { IoFreeMdl(mdl); st = STATUS_INSUFFICIENT_RESOURCES; break; }
        *(PVOID*)buf = uAddr;
        devExt->MapMdl = mdl;
        devExt->UserMappedAddress = uAddr;
        info = sizeof(PVOID);
        break;
    }
    case IOCTL_DRAGON_PAUSE_RESUME: {
        if (!buf) { st = STATUS_INVALID_PARAMETER; break; }
        BOOLEAN pause = *(BOOLEAN*)buf;
        InterlockedExchange(&devExt->PriorityProcessActive, pause ? 1 : 0);
        break;
    }
    case IOCTL_DRAGON_QUERY_STATS: {
        if (!buf) { st = STATUS_INVALID_PARAMETER; break; }
        SIZE_T* out = (SIZE_T*)buf;
        out[0] = devExt->MapSize;
        out[1] = devExt->PriorityProcessActive;
        info = 2 * sizeof(SIZE_T);
        break;
    }
    default:
        st = STATUS_INVALID_DEVICE_REQUEST;
    }
    Irp->IoStatus.Status = st;
    Irp->IoStatus.Information = info;
    IoCompleteRequest(Irp, IO_NO_INCREMENT);
    return st;
}

VOID DriverUnload(PDRIVER_OBJECT DriverObject) {
    PDEVICE_OBJECT devObj = DriverObject->DeviceObject;
    if (!devObj) return;
    PDEVICE_EXTENSION devExt = (PDEVICE_EXTENSION)devObj->DeviceExtension;
    devExt->EngineActive = 0;
    KeSetEvent(&devExt->ShutdownEvent, 0, FALSE);
    if (devExt->WatchdogThread) {
        KeWaitForSingleObject(devExt->WatchdogThread, Executive, KernelMode, FALSE, NULL);
        ObDereferenceObject(devExt->WatchdogThread);
    }
    for (int i = 0; i < 32; i++) {
        if (devExt->WorkerThreads[i]) {
            KeWaitForSingleObject(devExt->WorkerThreads[i], Executive, KernelMode, FALSE, NULL);
            ObDereferenceObject(devExt->WorkerThreads[i]);
        }
    }
    if (devExt->MapMdl) {
        MmUnmapLockedPages(devExt->UserMappedAddress, devExt->MapMdl);
        IoFreeMdl(devExt->MapMdl);
    }
    if (devExt->KnowledgeMap) ExFreePoolWithTag(devExt->KnowledgeMap, 'XAMK');
    if (devExt->Shards) ExFreePoolWithTag(devExt->Shards, 'RHS');
    if (devExt->HeartbeatBuffer) ExFreePoolWithTag(devExt->HeartbeatBuffer, 'TRAE');
    if (devExt->RingBuffer) ExFreePoolWithTag(devExt->RingBuffer, 'GNIR');
    UNICODE_STRING sym;
    RtlInitUnicodeString(&sym, L"\\DosDevices\\DragonEngine");
    IoDeleteSymbolicLink(&sym);
    IoDeleteDevice(devObj);
}

NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath) {
    UNREFERENCED_PARAMETER(RegistryPath);
    PDEVICE_OBJECT devObj = NULL;
    UNICODE_STRING devName, symLink;
    RtlInitUnicodeString(&devName, L"\\Device\\DragonEngine");
    RtlInitUnicodeString(&symLink, L"\\DosDevices\\DragonEngine");

    NTSTATUS status = IoCreateDevice(DriverObject, sizeof(DEVICE_EXTENSION), &devName,
                                     FILE_DEVICE_UNKNOWN, 0, FALSE, &devObj);
    if (!NT_SUCCESS(status)) return status;
    status = IoCreateSymbolicLink(&symLink, &devName);
    if (!NT_SUCCESS(status)) { IoDeleteDevice(devObj); return status; }

    PDEVICE_EXTENSION devExt = (PDEVICE_EXTENSION)devObj->DeviceExtension;
    RtlZeroMemory(devExt, sizeof(DEVICE_EXTENSION));
    devExt->DeviceObject = devObj;

    // RAM 2%, làm tròn xuống lũy thừa 2
    MY_SYSTEM_BASIC_INFORMATION sysInfo;
    status = ZwQuerySystemInformation(SystemBasicInformation, &sysInfo, sizeof(sysInfo), NULL);
    SIZE_T totalRam = NT_SUCCESS(status) ? (SIZE_T)sysInfo.NumberOfPhysicalPages * sysInfo.PageSize : 8ULL * 1024 * 1024 * 1024;
    SIZE_T twoPct = (totalRam * 2) / 100;
    if (twoPct > 0xFFFFFFFF) twoPct = 0xFFFFFFFF;
    SIZE_T pow2 = 1;
    while (pow2 <= twoPct / sizeof(SPATIAL_NODE)) pow2 <<= 1;
    pow2 >>= 1;
    devExt->MapSize = pow2;
    devExt->MapSizeMask = pow2 - 1;

    devExt->KnowledgeMap = (PSPATIAL_NODE)ExAllocatePoolWithTag(NonPagedPool, pow2 * sizeof(SPATIAL_NODE), 'XAMK');
    if (!devExt->KnowledgeMap) { IoDeleteSymbolicLink(&symLink); IoDeleteDevice(devObj); return STATUS_INSUFFICIENT_RESOURCES; }
    RtlZeroMemory(devExt->KnowledgeMap, pow2 * sizeof(SPATIAL_NODE));

    // ShardCount = lũy thừa 2 lớn nhất ≤ số lõi
    UINT32 cpuCount = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);
    UINT32 shardCnt = 1;
    while (shardCnt * 2 <= cpuCount) shardCnt <<= 1;
    devExt->ShardCount = shardCnt;
    devExt->Shards = (PSHARD)ExAllocatePoolWithTag(NonPagedPool, sizeof(SHARD) * shardCnt, 'RHS');
    if (!devExt->Shards) {
        ExFreePoolWithTag(devExt->KnowledgeMap, 'XAMK');
        IoDeleteSymbolicLink(&symLink); IoDeleteDevice(devObj);
        return STATUS_INSUFFICIENT_RESOURCES;
    }
    SIZE_T nodesPerShard = devExt->MapSize / shardCnt;   // cả hai đều lũy thừa 2 → kết quả lũy thừa 2
    for (UINT32 i = 0; i < shardCnt; i++) {
        devExt->Shards[i].base = &devExt->KnowledgeMap[i * nodesPerShard];
        devExt->Shards[i].size = nodesPerShard;
        KeInitializeSpinLock(&devExt->Shards[i].lock);
        devExt->Shards[i].hashSeed1 = __rdtsc() + i;
        devExt->Shards[i].hashSeed2 = __rdtsc() + i * 2;
    }

    devExt->HeartbeatBuffer = (UINT8*)ExAllocatePoolWithTag(NonPagedPool, HEARTBEAT_BUFFER_SIZE, 'TRAE');
    devExt->RingBuffer = (UINT8*)ExAllocatePoolWithTag(NonPagedPool, RING_BUFFER_SIZE, 'GNIR');
    if (!devExt->HeartbeatBuffer || !devExt->RingBuffer) {
        if (devExt->HeartbeatBuffer) ExFreePoolWithTag(devExt->HeartbeatBuffer, 'TRAE');
        if (devExt->RingBuffer) ExFreePoolWithTag(devExt->RingBuffer, 'GNIR');
        ExFreePoolWithTag(devExt->Shards, 'RHS');
        ExFreePoolWithTag(devExt->KnowledgeMap, 'XAMK');
        IoDeleteSymbolicLink(&symLink); IoDeleteDevice(devObj);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    KeInitializeSpinLock(&devExt->RingBufferLock);
    KeInitializeEvent(&devExt->WatchdogEvent, SynchronizationEvent, FALSE);
    KeInitializeEvent(&devExt->ShutdownEvent, NotificationEvent, FALSE);
    devExt->HeartbeatErrorFlag = 0;

    // Watchdog
    HANDLE hW;
    status = PsCreateSystemThread(&hW, THREAD_ALL_ACCESS, NULL, NULL, NULL, WatchdogThread, devExt);
    if (NT_SUCCESS(status)) {
        ObReferenceObjectByHandle(hW, THREAD_ALL_ACCESS, NULL, KernelMode, (PVOID*)&devExt->WatchdogThread, NULL);
        ZwClose(hW);
    }

    // Worker threads (mỗi shard một thread)
    for (UINT32 i = 0; i < shardCnt; i++) {
        PTHREAD_CONTEXT ctx = (PTHREAD_CONTEXT)ExAllocatePoolWithTag(NonPagedPool, sizeof(THREAD_CONTEXT), 'THRD');
        if (!ctx) continue;
        ctx->devExt = devExt;
        ctx->shardIndex = i;
        HANDLE hWorker;
        status = PsCreateSystemThread(&hWorker, THREAD_ALL_ACCESS, NULL, NULL, NULL, EngineWorkerThread, ctx);
        if (NT_SUCCESS(status)) {
            PKTHREAD threadObj;
            ObReferenceObjectByHandle(hWorker, THREAD_ALL_ACCESS, NULL, KernelMode, (PVOID*)&threadObj, NULL);
            devExt->WorkerThreads[i] = threadObj;
            ZwClose(hWorker);
        } else {
            ExFreePoolWithTag(ctx, 'THRD');
        }
    }

    devExt->EngineActive = 1;
    DriverObject->MajorFunction[IRP_MJ_CREATE] = DispatchCreateClose;
    DriverObject->MajorFunction[IRP_MJ_CLOSE] = DispatchCreateClose;
    DriverObject->MajorFunction[IRP_MJ_DEVICE_CONTROL] = DispatchDeviceControl;
    DriverObject->DriverUnload = DriverUnload;
    return STATUS_SUCCESS;
}