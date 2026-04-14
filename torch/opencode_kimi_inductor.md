# PyTorch Inductor编译器深度分析

## 目录
1. [架构概览](#1-架构概览)
2. [编译流程](#2-编译流程)
3. [GraphLowering详解](#3-graphlowering详解)
4. [Scheduler调度器](#4-scheduler调度器)
5. [代码生成](#5-代码生成)
6. [内存规划](#6-内存规划)
7. [Triton Kernel生成](#7-triton-kernel生成)
8. [优化策略](#8-优化策略)
9. [AOT编译](#9-aot编译)
10. [Inductor配置](#10-inductor配置)
11. [CPU后端](#11-cpu后端)

---

## 1. 架构概览

### 1.1 核心文件位置

| 组件 | 文件路径 | 行数 |
|------|----------|------|
| compile_fx | torch/_inductor/compile_fx.py | ~1286行 |
| graph | torch/_inductor/graph.py | ~2766行 |
| scheduler | torch/_inductor/scheduler.py | ~2800行 |
| wrapper | torch/_inductor/codegen/wrapper.py | ~2400行 |
| triton | torch/_inductor/codegen/triton.py | ~3800行 |
| ir | torch/_inductor/ir.py | ~5200行 |
| codecache | torch/_inductor/codecache.py | ~1841行 |

### 1.2 整体架构

```mermaid
flowchart TD
    A["torch.compile model"] --> B["Dynamo Frame Evaluation"]
    B --> C["Python字节码捕获 → FX Graph"]
    C --> D["AOT Autograd"]
    D --> E["前向/反向图分离"]
    E --> F["Inductor: compile_fx"]
    
    subgraph "Inductor Compilation"
        F --> G["Pre-grad Passes: 模式匹配"]
        G --> H["Joint Graph Passes: 前向/反向优化"]
        H --> I["Post-grad Passes: 布局优化"]
        I --> J["GraphLowering: FX → Inductor IR"]
        J --> K["Scheduler: 融合与调度"]
        K --> L["Code Generation"]
    end
    
    L --> M["Python Wrapper + Kernels"]
    M --> N["Triton Kernels GPU"]
    M --> O["C++ Kernels CPU"]
    N --> P["Cache Compiled Code"]
    O --> P
    P --> Q["Execute Compiled Graph"]
```

---

## 2. 编译流程

### 2.1 torch.compile完整流程

```mermaid
flowchart TD
    A["torch.compile model"] --> B["Dynamo: Python Frame Evaluation"]
    B --> C{"Graph Break?"}
    C -->|"Yes"| D["Partial Graph"]
    C -->|"No"| E["Complete FX Graph"]
    D --> B
    E --> F["AOT Autograd: Forward/Backward Split"]
    F --> G["Inductor: compile_fx"]
    
    subgraph "Inductor Compilation"
        G --> H["Pre-grad Passes<br/>Pattern Matching"]
        H --> I["Joint Graph Passes"]
        I --> J["Post-grad Passes<br/>Layout Optimization"]
        J --> K["GraphLowering<br/>FX → Inductor IR"]
        K --> L["Scheduler<br/>Fusion & Ordering"]
        L --> M["Code Generation"]
    end
    
    M --> N["Python Wrapper<br/>Triton Kernels<br/>C++ Kernels"]
    N --> O["Cache Compiled Code"]
    O --> P["Execute Compiled Graph"]
```

### 2.2 compile_fx_inner核心代码

```python
# 来自torch/_inductor/compile_fx.py
def compile_fx_inner(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    cudagraphs: Optional[BoxedBool] = None,
    num_static_inputs: Optional[int] = None,
    is_backward: bool = False,
    graph_id: Optional[str] = None,
    **kwargs,
) -> OutputCode:
    # 1. 图预处理 - 检查和转换
    # 2. 缓存查找 - 避免重复编译
    # 3. 代码生成和编译
    # 4. 编译后设置
    
    # 核心调用
    return _compile_fx_inner(
        gm, example_inputs, cudagraphs, num_static_inputs,
        is_backward, graph_id, **kwargs
    )

def _compile_fx_inner(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    ...
) -> OutputCode:
    # 应用post-grad passes
    # 创建GraphLowering
    # 运行scheduler
    # 生成wrapper代码
    # 编译和缓存
```

### 2.3 编译阶段

| 阶段 | 文件 | 描述 |
|------|------|------|
| Pre-grad | torch/_inductor/fx_passes/pre_grad.py | 在grad之前应用的模式匹配 |
| Joint Graph | torch/_inductor/fx_passes/joint_graph.py | 前向/反向联合优化 |
| Post-grad | torch/_inductor/fx_passes/post_grad.py | 布局优化，内存传递 |
| Pattern Matching | torch/_inductor/pattern_matcher.py | 通用模式匹配框架 |

---

## 3. GraphLowering详解

### 3.1 GraphLowering类

```python
# 来自torch/_inductor/graph.py
class GraphLowering(torch.fx.Interpreter):
    """
    将FX图转换为Inductor IR的关键职责：
    - 通过ShapeEnv进行符号形状跟踪
    - 缓冲区管理和分配
    - 操作降级注册表
    - 布局优化决策
    """
    
    def __init__(self, gm: GraphModule, ...):
        self.sizevars = SizeVarAllocator(shape_env)
        self.buffers: list[ir.Buffer] = []
        self.operations: list[ir.Operation] = []
        self.name_to_buffer: dict[str, ir.Buffer] = {}
        self.graph_outputs: list[ir.Buffer] = []
        
        # 设备信息
        self.device_types: set[str] = set()
        self.device_idxs: set[int] = set()
        
    def call_function(self, target: Callable, args: Any, kwargs: dict[str, Any]) -> Any:
        """将FX操作降级为Inductor IR"""
        if target in lowerings:
            return lowerings[target](*args, **kwargs)
        # 对不支持的操作进行回退处理
        return fallback_handler(target)(*args, **kwargs)
```

### 3.2 IR数据结构

```python
# 来自torch/_inductor/ir.py
class Buffer(IRNode):
    """内存缓冲区的基类"""
    def __init__(self, name, layout):
        self.name = name
        self.layout = layout  # 包含device, dtype, sizes, strides
        self.users: list[IRNode] = []
        
class ComputedBuffer(Buffer):
    """存储计算结果的缓冲区"""
    def __init__(self, name, layout, data):
        super().__init__(name, layout)
        self.data = data  # Loops/Body containing computation
        
class ExternKernel(Buffer):
    """外部内核调用（cuBLAS等）"""
    def __init__(self, name, layout, inputs):
        super().__init__(name, layout)
        self.inputs = inputs
        
class Pointwise(Loops):
    """逐元素操作"""
    pass
    
class Reduction(Loops):
    """归约操作（sum, max等）"""
    pass
    
class Layout:
    """内存布局描述"""
    def __init__(self, device, dtype, size, stride):
        self.device = device
        self.dtype = dtype
        self.size = size
        self.stride = stride
```

### 3.3 GraphLowering流程

```mermaid
flowchart TD
    A["FX Graph from Dynamo"] --> B["GraphLowering.run"]
    
    subgraph "Graph Lowering"
        B --> C["Process Placeholders<br/>Create Input Buffers"]
        C --> D["Call Functions<br/>Lower to IR"]
        D --> E["Track Buffer Dependencies"]
        E --> F["Layout Decisions<br/>Channels-last Heuristics"]
        F --> G["Output Processing<br/>Realize Outputs"]
    end
    
    G --> H["Scheduler Initialization"]
```

---

## 4. Scheduler调度器

### 4.1 Scheduler类

```python
# 来自torch/_inductor/scheduler.py
class Scheduler:
    """
    管理调度节点、融合和内核生成。
    """
    
    def __init__(self, nodes: list[ir.Operation]) -> None:
        self.nodes = [self.create_scheduler_node(n) for n in nodes]
        self.compute_dependencies()
        self.nodes = self.fuse_nodes(self.nodes)  # 关键融合步骤
        self.merge_loops()
        self.compute_stream_order()
```

### 4.2 Scheduler Node类型

```python
class SchedulerNode:
    """标准计算操作（逐元素、归约）"""
    def __init__(self, node: ir.Operation):
        self.node = node
        self.group = node.group  # 融合组
        self.users: list[SchedulerNode] = []
        self.ancestors: set[str] = set()

class FusedSchedulerNode:
    """融合的操作"""
    def __init__(self, nodes: list[SchedulerNode]):
        self.nodes = nodes
        self.group = nodes[0].group
        
class ExternKernelSchedulerNode:
    """外部内核调用（cuBLAS等）"""
    def __init__(self, node: ir.ExternKernel):
        self.node = node
        
class NopKernelSchedulerNode:
    """无操作"""
    pass
    
class ForeachKernelSchedulerNode:
    """批处理操作（组合内核）"""
    def __init__(self, nodes: list[SchedulerNode]):
        self.nodes = nodes
        
class GroupedSchedulerNode:
    """临时分组的节点"""
    def __init__(self, nodes: list[SchedulerNode]):
        self.nodes = nodes
```

### 4.3 融合策略

```mermaid
flowchart TD
    A["Scheduler Nodes"] --> B["Fusion Passes"]
    
    subgraph "Fusion Decision"
        B --> C{"Can Fuse?"}
        C -->|"Same Loop Structure"| D["Vertical Fusion<br/>Producer-Consumer"]
        C -->|"Same Device"| E["Horizontal Fusion<br/>Independent Ops"]
        C -->|"Template Match"| F["Epilogue Fusion<br/>GEMM + Pointwise"]
        D --> G["FusedSchedulerNode"]
        E --> G
        F --> G
    end
    
    G --> H["Final Schedule"]
```

### 4.4 can_fuse逻辑

```python
def can_fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode) -> bool:
    """确定两个调度器节点是否可以融合。"""
    # 检查设备兼容性
    if node1.get_device() != node2.get_device():
        return False
        
    # 检查是否在同一融合组
    if node1.group != node2.group:
        return False
        
    # 检查循环结构兼容性
    if not self.have_compatible_loop_structures(node1, node2):
        return False
        
    # 检查数据依赖
    if node2.get_name() in node1.ancestors:
        # 生产者-消费者：检查融合是否有益
        return self.should_fuse_producer_consumer(node1, node2)
    
    # 检查独立节点（水平融合）
    if self.are_independent(node1, node2):
        return self.should_fuse_independent(node1, node2)
        
    return False

def should_fuse_producer_consumer(self, producer, consumer) -> bool:
    """检查生产者-消费者融合是否有益。"""
    # 检查内存访问模式
    # 检查计算密度
    # 检查是否会导致寄存器压力过高
    pass
```

---

## 5. 代码生成

### 5.1 Wrapper Code Generation

```python
# 来自torch/_inductor/codegen/wrapper.py
class PythonWrapperCodegen(CodeGen):
    """
    生成编排内核执行的外部包装代码。
    """
    
    def __init__(self):
        self.imports = IndentedBuffer()
        self.header = IndentedBuffer()
        self.prefix = IndentedBuffer()
        self.wrapper_call = IndentedBuffer()
        self.lines: list[Line] = []  # 内存规划行
        self.allocated_buffers: set[str] = set()
        self.allocated_comm_buffers: set[str] = set()
        
    def generate(self, is_inference):
        # 1. 运行wrapper IR passes
        self.run_wrapper_ir_passes(is_inference)
        # 2. 生成内存规划
        # 3. 组装最终代码
        
    def generate_extern_kernel_out(
        self, 
        output: str,
        kernel: str,
        args: list[str]
    ):
        # 生成外部内核调用代码
        pass
```

### 5.2 代码生成流程

```mermaid
flowchart TD
    A["Scheduler Output"] --> B{"Device Type"}
    
    B -->|"CUDA/Hopper"| C["Triton Code Generation"]
    B -->|"CPU"| D["C++ Code Generation"]
    
    subgraph "Triton Kernel Generation"
        C --> E["TritonKernel.codegen<br/>Generate Kernel"]
        E --> F["Generate Block Pointers<br/>or Tensor Descriptors"]
        F --> G["Create Loop Structure<br/>XBLOCK, YBLOCK, RBLOCK"]
        G --> H["Emit Pointwise Operations"]
        H --> I["Emit Reduction Operations<br/>tl.reduce"]
        I --> J["Handle Masking<br/>Boundary Conditions"]
        J --> K["Generate Kernel Signature"]
    end
    
    subgraph "C++ Kernel Generation"
        D --> L["CppKernel.codegen"]
        L --> M["Vectorized Operations"]
        M --> N["OpenMP Parallelization"]
    end
    
    K --> O["PythonWrapperCodegen"]
    N --> O
    
    subgraph "Wrapper Generation"
        O --> P["Memory Planning<br/>Buffer Reuse"]
        P --> Q["Generate Allocations"]
        Q --> R["Generate Kernel Calls<br/>with Grid/Block Config"]
        R --> S["Generate Free Operations"]
        S --> T["Assemble Final Module"]
    end
```

---

## 6. 内存规划

### 6.1 MemoryPlanningState

```python
class MemoryPlanningState:
    def __init__(self):
        self.reuse_pool: dict[ReuseKey, list[FreeIfNotReusedLine]] = \
            collections.defaultdict(list)
    
    def pop(self, key: ReuseKey) -> FreeIfNotReusedLine:
        item = self.reuse_pool[key].pop()
        return item
        
    def push(self, key: ReuseKey, item: FreeIfNotReusedLine) -> None:
        self.reuse_pool[key].append(item)

def buffer_reuse_key(node: BufferLike) -> ReuseKey:
    """为缓冲区重用匹配创建键。"""
    return (
        node.get_device_or_error(),
        node.get_dtype(),
        sympy_str(V.graph.sizevars.simplify(storage_size)),
        alignment,
    )
```

### 6.2 内存规划流程

```mermaid
flowchart TD
    A["Scheduler Nodes with Buffer Info"] --> B["MemoryPlanningState"]
    
    subgraph "Buffer Reuse Strategy"
        B --> C["Calculate Reuse Key<br/>Device, Dtype, Size, Alignment"]
        C --> D["Maintain Reuse Pool"]
        D --> E{"Allocation Request"}
        E -->|"Key in Pool"| F["Pop from Pool<br/>ReuseLine"]
        E -->|"Key not in Pool"| G["AllocateLine"]
        F --> H["Mark FreeIfNotReusedLine<br/>as Reused"]
        G --> I["Track New Buffer"]
    end
    
    subgraph "Peak Memory Estimation"
        A --> J["EfficientPeakEstimate"]
        J --> K["Build Segmented Tree<br/>of Allocations"]
        K --> L["Calculate Peak Memory<br/>Between Allocations"]
        L --> M{"Reuse Decision"}
        M -->|"Would Exceed Peak"| N["Skip Reuse"]
        M -->|"Within Budget"| F
    end
    
    subgraph "Special Cases"
        O["Comm Buffer"] -->|"Separate Pool"| P["Comm-Comm Reuse Only"]
        Q["Multi-Stream"] -->|"Same Stream Check"| R["Stream-Aware Reuse"]
        S["Inplace Update"] --> T["Direct Buffer Overwrite"]
    end
```

---

## 7. Triton Kernel生成

### 7.1 TritonKernel类

```python
# 来自torch/_inductor/codegen/triton.py
class TritonKernel(SIMDKernel[TritonCSEVariable]):
    """
    为逐元素/归约操作生成Triton内核代码。
    """
    
    def __init__(self, tiling: dict[str, sympy.Expr], ...):
        self.cse = TritonCSE(...)  # 公共子表达式消除
        self.range_trees: list[IterationRangesRoot] = []  # 循环结构
        self.block_ptr_id = itertools.count()
        self.use_block_ptr = config.triton.use_block_ptr
        
    def codegen_kernel(self, name: str = None) -> str:
        """生成完整的Triton内核源代码"""
        code = IndentedBuffer()
        
        # 写入内核签名
        code.writeline(f"@triton.jit")
        code.writeline(f"def {name or self.name}(")
        
        # 写入参数
        for arg in self.args:
            code.writeline(f"    {arg},")
        code.writeline("):")
        
        with code.indent():
            # 初始化块指针
            if self.use_block_ptr:
                for block_ptr in self.block_ptrs.values():
                    code.writeline(f"{block_ptr.name} = {block_ptr.codegen_init()}")
            
            # 生成循环结构
            for tree in self.range_trees:
                code.writeline(f"{tree.codegen_start()}")
                
            # 生成计算主体
            code.splice(self.compute)
            
            # 处理归约
            if self.is_reduction():
                code.splice(self.post_loop_combine)
                code.splice(self.post_loop_store)
        
        return code.getvalue()
```

### 7.2 Triton代码生成示例

```python
# 生成的Triton内核示例
"""
@triton.jit
def triton_fused_add_mul(
    x_ptr, y_ptr, z_ptr, out_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    # 程序ID映射到数据范围
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    
    # 计算：out = (x + y) * z
    tmp = x + y
    out = tmp * z
    
    # 存储结果
    tl.store(out_ptr + offsets, out, mask=mask)
"""
```

### 7.3 Tensor Descriptor (Hopper)

```python
# Hopper架构支持Tensor Descriptor用于更高效的内存访问
class TritonTensorDescriptor:
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims
    
    def codegen_init(self):
        # 生成tl.make_tensor_descriptor调用
        return f"tl.make_tensor_descriptor({self.name}, {self.dims})"
    
    def codegen_load(self, offsets):
        # 生成tl.load_via_tensor_descriptor
        return f"tl.load_via_tensor_descriptor({self.name}, {offsets})"
```

---

## 8. 优化策略

### 8.1 布局优化

```python
def decide_layout_opt(gm: GraphModule, *, is_inference: bool) -> bool:
    """
    决定是否应启用channels-last布局优化。
    """
    conv_nodes = [n for n in gm.graph.nodes 
                  if n.target is torch.ops.aten.convolution.default]
    
    # 启发式：卷积太少则跳过
    if len(conv_nodes) == 0:
        return False
        
    # 检查组卷积（channels-last通常较慢）
    if any(is_grouped(n) for n in conv_nodes):
        return False
        
    # 检查小通道尺寸
    if all(is_small_channel(n) for n in conv_nodes):
        return False
        
    # 加权FLOP分析
    return weighted_flops_analysis(conv_nodes) <= total_flops
```

### 8.2 融合优化

1. **垂直融合**：生产者-消费者融合（如conv + relu）
2. **水平融合**：具有相同循环结构的独立操作
3. **模板融合**：将尾缀操作融合到GEMM模板
4. **组合内核**：批处理小独立内核

### 8.3 分块和自动调优

```python
class TritonKernel:
    def select_tiling(self):
        """选择最佳分块配置"""
        # 考虑因素：
        # - 数据形状
        # - 内存访问模式
        # - GPU架构（SM数量、共享内存）
        # - 自动调优缓存
        
    def autotune(self):
        """自动调优内核配置"""
        # 尝试不同的BLOCK_SIZE
        # 选择性能最佳的配置
        # 缓存结果供重用
```

---

## 9. AOT编译

### 9.1 AOT Inductor Runtime

```cpp
// 来自torch/csrc/inductor/aoti_runtime/interface.h
// C API for compiled models

// 加载和执行AOT编译模型
AOTInductorModelHandle model;
aoti_load_model("model.so", &model);

// 准备输入
AOTInductorTensorHandle inputs[] = {...};

// 执行
AOTInductorTensorHandle outputs[];
aoti_run_model(model, inputs, 1, outputs, 1);

// 清理
aoti_free_model(model);
```

### 9.2 AOT编译流程

```mermaid
flowchart TD
    A["PyTorch Model"] --> B["torch.export.export"]
    B --> C["ExportedProgram"]
    C --> D["AOT Inductor Compile"]
    D --> E["Generate C++ Wrapper"]
    E --> F["Compile to .so"]
    F --> G["Deploy to Production"]
    G --> H["Load in C++/Python"]
    H --> I["Execute"]
```

### 9.3 AOTInductor编译Python API

```python
import torch._inductor.aot_compilation as aot

# 编译模型
so_path = aot.aot_compile(
    model,
    example_inputs,
    options={
        "aot_inductor.output_path": "/path/to/output.so",
        "aot_inductor.package": True,  # 打包为完整包
    }
)

# 加载和运行
runner = aot.load(so_path)
output = runner(*inputs)
```

---

## 10. Inductor配置

### 10.1 关键配置选项

```python
# torch/_inductor/config.py
class config:
    # 调试
    debug = False
    trace = False
    
    # 优化
    pattern_matcher = True
    epilogue_fusion = True
    split_reductions = True
    
    # Triton特定
    triton = TritonConfig(
        use_block_ptr = True,  # 使用块指针（Hopper）
        cooperative_reductions = False,
        dense_indexing = False,
    )
    
    # 内存
    memory_planning = True
    max_autotune = False
    max_autotune_gemm = False
    
    # AOT
    aot_inductor = AOTInductorConfig(
        output_path = "",
        package = False,
        use_runtime_constant_folding = True,
    )
    
    # 回退
    fallback_random = False
    allow_buffer_reuse = True
```

### 10.2 环境变量配置

```bash
# 启用最大自动调优
TORCHINDUCTOR_MAX_AUTOTUNE=1

# 启用Triton块指针
TORCHINDUCTOR_TRITON_USE_BLOCK_PTR=1

# 禁用某些优化
TORCHINDUCTOR_PATTERN_MATCHER=0
TORCHINDUCTOR_EPILOGUE_FUSION=0

# 调试输出
TORCHINDUCTOR_DEBUG=1
TORCHINDUCTOR_TRACE=1
```

---

## 11. CPU后端

### 11.1 CppKernel生成

```python
# 来自torch/_inductor/codegen/cpp.py
class CppKernel(SIMDKernel):
    """生成C++内核代码"""
    
    def __init__(self, ...):
        self.vec_isa = pick_vec_isa()  # 选择向量指令集（AVX2, AVX512）
        
    def codegen_kernel(self, name: str) -> str:
        code = IndentedBuffer()
        
        # 包含头文件
        code.writeline("#include <ATen/ATen.h>")
        code.writeline("#include <omp.h>")
        
        # 函数签名
        code.writeline(f"void {name}(...)")
        code.writeline("{")
        
        with code.indent():
            # OpenMP并行
            if self.num_threads > 1:
                code.writeline("#pragma omp parallel for")
            
            # 循环结构
            for tree in self.range_trees:
                code.writeline(f"for (int {tree.name} = ...)")
            
            # 向量化操作
            if self.vec_isa:
                code.writeline(f"using vec = at::vec::Vectorized<float>;")
                # 生成向量化代码
        
        code.writeline("}")
        return code.getvalue()
```

### 11.2 CPU特定优化

1. **向量指令集**：AVX2, AVX512
2. **OpenMP并行**：多线程并行化
3. **线程池**：重用线程避免创建开销
4. **内存对齐**：确保向量化内存访问对齐

---

## 12. 关键文件汇总

| 文件 | 用途 | 关键类/函数 |
|------|------|-------------|
| torch/_inductor/compile_fx.py | 主编译入口 | compile_fx_inner(), _compile_fx_inner() |
| torch/_inductor/graph.py | 图降级 | GraphLowering, lowerings registry |
| torch/_inductor/scheduler.py | 操作调度 | Scheduler, FusedSchedulerNode, can_fuse() |
| torch/_inductor/codegen/wrapper.py | 包装代码生成 | PythonWrapperCodegen, MemoryPlanningState |
| torch/_inductor/codegen/triton.py | Triton内核生成 | TritonKernel, TritonOverrides |
| torch/_inductor/codegen/cpp.py | C++内核生成 | CppKernel, CppOverrides |
| torch/_inductor/ir.py | 中间表示 | Buffer, ComputedBuffer, Pointwise, Reduction, ExternKernel |
| torch/_inductor/codecache.py | 代码缓存 | CodeCache, FxGraphCache |
| torch/_inductor/config.py | 配置 | config, TritonConfig, AOTInductorConfig |
| torch/_inductor/fx_passes/ | 优化passes | joint_graph.py, post_grad.py, pre_grad.py |
| torch/csrc/inductor/aoti_runtime/interface.h | AOT Inductor运行时 | C API for compiled models |
| torch/csrc/inductor/cpp_wrapper/ | C++包装模板 | 设备特定包装代码 |

---

## 13. 总结

PyTorch Inductor是一个先进的深度学习编译器：

1. **降低FX图**：到支持符号形状的IR，通过GraphLowering转换

2. **调度操作**：使用激进的融合策略（垂直、水平、模板融合）

3. **生成优化内核**：
   - GPU：使用Triton生成高性能内核，支持自动调优
   - CPU：生成C++向量化代码，支持OpenMP并行

4. **内存管理**：通过智能缓冲区重用和规划，降低峰值内存

5. **支持AOT编译**：用于部署场景，生成独立的.so文件

6. **灵活配置**：通过config和environment变量控制优化行为

7. **多后端支持**：CUDA, CPU, XPU等多种设备后端

8. **模式匹配**：基于pattern matcher的图优化

9. **自动调优**：自动搜索最佳内核配置

编译流程经过多个优化阶段，调度器在确定融合机会和操作排序方面发挥核心作用。代码生成是设备特定的，Triton通过自动调优提供高性能GPU内核，CPU后端则通过向量化和OpenMP提供优化。
