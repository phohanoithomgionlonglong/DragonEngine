# Dragon Engine – Spatial-Logic Accelerator

**Author:** Nguyễn Thành Long, 14 years old  
**Country:** Vietnam

This is my first project: a kernel‑mode driver and companion software that speed up
repetitive spatial calculations in games, graphics, and AI.

The core idea is simple: when the system performs a heavy computation (collision
detection, ray tracing, matrix multiplication…), it stores the result. Next time the
same or a very similar input appears, the result is reused instead of being recomputed
from scratch. This works across applications because the engine lives in the kernel
and shares a common cache between the CPU and GPU.

> ⚠️ **Important:** My computer is 12 years old. I cannot run demanding tests.
> The code may contain bugs. I would be very grateful if you try it and let me know
> your results – good or bad.

## What's Included

This repository contains only the source code (3 files):

| File | Description |
|------|-------------|
| `Dragon.c` | Kernel driver (`Dragon.sys`) – memory‑mapped spatial cache, Cuckoo hashing, lock‑free synchronization, zero‑copy sharing with user mode. |
| `dragon_controller.cpp` | OpenCL controller – distributes fused (lookup → compute → store) kernels across multiple GPUs. Uses dynamic OpenCL loading. |
| `dragon_compute.cpp` | CPU compute service – automatically picks the strongest SIMD instruction set (SSE2/SSE4.1/AVX/AVX2/AVX‑512), communicates via Named Pipe. |

There is no pre‑compiled binary. You have to build from source.

## ⚙️ System Requirements

- Windows 10 / 11 (64‑bit)
- **Test Mode** enabled (because the driver is self‑signed)
- Administrator privileges
- For the GPU controller: OpenCL 2.0 capable GPU and driver
- For the CPU service: any x86‑64 CPU (the software auto‑detects the best SIMD path)

## 🚀 How to Build & Run

### 1. Enable Test Mode (one time)
Open **Command Prompt as Administrator** and run:
bcdedit /set testsigning on
Restart your computer. "Test Mode" will appear in the bottom‑right corner of the desktop.

### 2. Build the driver (`Dragon.sys`)
- Install the **Windows Driver Kit (WDK)** or use Visual Studio with the `Kernel‑Mode Driver` template.
- Create a new empty driver project, add `Dragon.c`.
- Build for your platform (x64). You will get `Dragon.sys`.

### 3. Build the user‑mode executables
- Use **MinGW‑w64** (e.g., from Dev‑C++) or MSVC.
- Compile `dragon_controller.cpp`:
sc stop DragonEngine
sc delete DragonEngine

### 5. Verifying it works
Since I cannot provide a benchmark, you can write a small program that sends repeated batches of floats to the pipe. The first run (cold cache) will be slower; subsequent runs with the same data (hot cache) should be noticeably faster.

If you run a game, you may see higher FPS only if the game's computations are repetitive and the game (or a wrapper) communicates with the engine. **Right now, the system does not automatically intercept game calls** – that is a major piece that I could not finish with my old hardware.

## 🧪 Feedback & Contributions

I am 14 and this is my first project. I know it is far from perfect. If you:
- find a bug
- see a way to improve it
- or just want to tell me your benchmark results

please open an issue on GitHub or contact me through the email in my profile.

## 🛡️ License

All source code is licensed under the **Creative Commons Attribution‑NonCommercial‑ShareAlike 4.0 International** license (CC BY‑NC‑SA 4.0).

You are free to:
- Share – copy and redistribute the material in any medium or format
- Adapt – remix, transform, and build upon the material

Under the following terms:
- **Attribution** – You must give appropriate credit to Nguyễn Thành Long provide a link to the license, and indicate if changes were made.
- **NonCommercial** – You may not use the material for commercial purposes.
- **ShareAlike** – If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

See the full legal code in the `LICENSE` file.

**Moral request:** If the ideas or algorithms in this project inspire you, I would deeply appreciate a mention of my name and age somewhere in your project. This is not a legal requirement, just a way to support a young developer.

---

*“I am 14. This is my first work. My PC is 12 years old. I did what I could with what I had.”*
