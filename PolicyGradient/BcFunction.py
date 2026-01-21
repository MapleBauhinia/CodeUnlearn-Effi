import json
import os
import ast
from typing import List, Dict, Any
import sys
# 采样utf-8编码的文件
sys.stdout.reconfigure(encoding='utf-8')


# 由于 radon 库在某些环境下安装不便，我们采用动态导入的方式加载 radon 的 complexity 模块。
import importlib.util
module_path = rf"C:\Program Files\Python311\Lib\site-packages\radon\complexity.py"
spec = importlib.util.spec_from_file_location(
    "radon_complexity", module_path
)
radon_complexity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(radon_complexity)
cc_visit = radon_complexity.cc_visit


class BaseEfficiencyVisitor(ast.NodeVisitor):
    """基础访问器，用于维护上下文（如是否在循环内）"""
    def __init__(self):
        self.in_loop = False
        self.loop_depth = 0
        self.score = 0.0

    def visit_For(self, node):
        self.in_loop = True
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1
        if self.loop_depth == 0: self.in_loop = False

    def visit_While(self, node):
        self.in_loop = True
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1
        if self.loop_depth == 0: self.in_loop = False


# Category 1: Inefficient function or API usage
# Sub-category 1.1: Suboptimal method or API selection
class BadAPIVisitor(BaseEfficiencyVisitor):
    """
    检测逻辑： 建立一个“黑名单”映射。检测 O(N) 的操作被用于本该 O(1) 或更优的场景。
    典型例子： 使用 list.pop(0) (O(N)) 而非 deque.popleft()；在循环中使用 list.count()。
    评分公式： B(y) = \sum (\text{Frequency of bad API calls})
    """
    def visit_Call(self, node):
        # 检测 list.pop(0)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'pop':
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0:
                self.score += 1.0 # 惩罚
        
        # 检测循环内的 list.count 或 index (通常暗示可用 dict/set)
        if self.in_loop and isinstance(node.func, ast.Attribute):
            if node.func.attr in ['count', 'index']:
                self.score += 0.5
        self.generic_visit(node)
def get_bad_api_score(code_str):
    visitor = BadAPIVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Sub-category 1.2: Unnecessary or excessive recursion
class RecursionVisitor(ast.NodeVisitor):
    """
    检测逻辑： 简单的递归不仅效率低（栈溢出风险），而且在大模型生成的简单代码中往往不如迭代。检测函数体内是否调用自身。
    评分公式： $B(y) = \mathbb{I}(\text{Is Recursive}) \times (\text{Depth Factor})$
    """
    def __init__(self):
        self.funcs = []
        self.score = 0.0
    
    def visit_FunctionDef(self, node):
        self.funcs.append(node.name)
        # 检查函数体中是否有调用自己
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id == node.name:
                    self.score += 1.0 # 发现递归
        self.funcs.pop()

def get_recursion_score(code_str):
    visitor = RecursionVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Category 2: Algorithmic inefficiencies
# Sub-category 2.1: Brute-force or suboptimal algorithmic strategy
# Sub-category 2.2: Absence of established optimization techniques
# Sub-category 2.3: Insufficient mathematical abstraction and optimization
def get_complexity_score(code):
    """
    检测逻辑： 这是一个语义难点。我们用圈复杂度 (Cyclomatic Complexity, CC) 作为代理指标。如果代码没有引用高效库（如 bisect, heapq）但 CC 极高，判定为“低效的复杂逻辑”。
    评分公式： $B(y) = \text{Norm}(CC) \times (1 - \mathbb{I}(\text{Has Optimization Imports}))$
    """
    has_opt_lib = any(lib in code for lib in ['bisect', 'heapq', 'numpy', 'itertools'])
    try:
        blocks = cc_visit(code)
        total_cc = sum(b.complexity for b in blocks)
        # 如果复杂度高且没用优化库，得分高
        score = max(0, total_cc - 10) * (0.5 if has_opt_lib else 1.0)
        return score
    except: return 0.0

# Sub-category 2.4: Inefficient conditional logic
class ConditionalVisitor(BaseEfficiencyVisitor):
    """
    检测逻辑： 检测过深的 if-elif-else 嵌套，或者在循环内部进行恒定条件的判断。
    评分公式： $B(y) = \text{Max If Depth} + \text{Count}(If \text{ inside } Loop)$
    """
    def visit_If(self, node):
        if self.in_loop:
            self.score += 0.5 # 循环内的判断成本更高
        self.generic_visit(node)

def get_conditional_score(code_str):
    visitor = ConditionalVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Sub-category 2.5: Avoidable nested-loop complexity
class NestedLoopVisitor(ast.NodeVisitor):
    """
    检测逻辑： 核心指标。检测嵌套层数。
    评分公式： $B(y) = \sum_{loop} (2^{\text{depth}} - 1)$
    """
    def __init__(self):
        self.depth = 0
        self.max_depth = 0
    
    def visit_For(self, node):
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        self.generic_visit(node)
        self.depth -= 1
        
    def visit_While(self, node): # 同上
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        self.generic_visit(node)
        self.depth -= 1

    def get_score(self):
        return max(0, self.max_depth - 2) * 2.0 # 容忍2层，超过重罚

def get_nested_loop_score(code_str):
    visitor = NestedLoopVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.get_score()
    except: return 0.0

# Sub-category 2.6: Unnecessary multi-pass processing
# Sub-category 2.7: Redundant recomputation
class MultiPassVisitor(ast.NodeVisitor):
    """
    检测逻辑： 在循环中反复调用同一个函数且参数看起来没变（纯函数假设），或者多次遍历同一个 List。
    代码实现： 需要构建简单的 Control Flow Graph (CFG) 较难，简化版：检测多个同级循环遍历同一变量。
    """
    def __init__(self):
        self.iterated_vars = []
        self.score = 0.0
        
    def visit_For(self, node):
        if isinstance(node.iter, ast.Name):
            var_name = node.iter.id
            if var_name in self.iterated_vars:
                self.score += 1.0 # 再次遍历同一变量
            self.iterated_vars.append(var_name)
        self.generic_visit(node)

def get_multipass_score(code_str):
    visitor = MultiPassVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Category 3: Inefficient data structure usage
# Sub-category 3.1: Inappropriate data structure selection
class BadDSSelectionVisitor(BaseEfficiencyVisitor):
    """
    检测逻辑： 最典型的是在列表（List）中进行查找（Membership Test x in y），而该列表从未被修改，本应用 set。
    评分公式： $B(y) = \mathbb{I}(x \text{ in List inside Loop})$
    """
    def visit_Compare(self, node):
        # 检测 x in y 操作
        if self.in_loop and isinstance(node.ops[0], (ast.In, ast.NotIn)):
             # 这里无法静态确定 y 是 list 还是 set，但可以根据变量名或初始化猜测
             # 简化策略：惩罚循环内所有的 'in' 操作，除非它是显然的小范围
             self.score += 0.5 
        self.generic_visit(node)

def get_ds_selection_score(code_str):
    visitor = BadDSSelectionVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Sub-category 3.2: Inefficient operations on selected data structure
class InefficientListOpVisitor(BaseEfficiencyVisitor):
    """
    检测逻辑：
        1、检测是否在 循环 (Loop) 内部。
        2、检测特定的列表方法调用：insert(0, ...) 或 pop(0)。
        3、检测列表的 remove(x) 或 count(x)（也是 $O(N)$），如果在循环内使用则非常危险。
    评分公式： $$B(y) = \sum_{op \in \{insert, pop, remove, count\}} \mathbb{I}(op \text{ inside loop}) \times w_{op}$$
    """
    def visit_Call(self, node):
        # 仅检测循环内的操作
        if self.in_loop and isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            
            # 检测 list.insert(0, x) -> 建议使用 collections.deque
            if method_name == 'insert':
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0:
                    self.score += 1.0  # 重罚头部插入
            
            # 检测 list.pop(0) -> 建议使用 collections.deque
            elif method_name == 'pop':
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0:
                    self.score += 1.0  # 重罚头部弹出
            
            # 检测 list.remove(x) 或 list.count(x) -> O(N) 操作在循环内
            elif method_name in ['remove', 'count']:
                self.score += 0.5

        self.generic_visit(node)

def get_list_op_score(code_str):
    visitor = InefficientListOpVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Sub-category 3.3: Inefficient string concatenation
class StringConcatVisitor(ast.NodeVisitor):
    """
    检测逻辑： 寻找 For 或 While 循环内部的 AugAssign (例如 +=) 节点，且操作数往往涉及字符串变量（由于 AST 不含类型信息，我们采用启发式规则：变量名包含 str, text, s 或上下文推断）。
    评分公式： $$B(y) = \sum \mathbb{I}(\text{String Concat in Loop})$$
    """
    def __init__(self):
        self.score = 0
        self.in_loop = False
        
    def visit_For(self, node):
        self.in_loop = True
        self.generic_visit(node)
        self.in_loop = False

    def visit_While(self, node):
        self.in_loop = True
        self.generic_visit(node)
        self.in_loop = False
        
    def visit_AugAssign(self, node):
        if self.in_loop and isinstance(node.op, ast.Add):
            # 启发式检测：检查变量名是否像字符串，或者是否在上下文初始化为 ""
            # 这里简化为检测所有循环内的 +=，实际应用可结合变量名过滤
            if isinstance(node.target, ast.Name):
                # 强检测：如果这是一个 += 操作
                self.score += 1.0 
        self.generic_visit(node)

def get_ds_score(code_str):
    visitor = StringConcatVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except:
        return 0.0

# Sub-category 3.4: Repeated sequence slicing in loops
class SlicingVisitor(BaseEfficiencyVisitor):
    """
    检测逻辑： 循环内出现 x[i:j]。这会产生新的拷贝，开销巨大。
    评分公式： $B(y) = \text{Count}(\text{Slicing inside Loop})$
    """
    def visit_Subscript(self, node):
        if self.in_loop and isinstance(node.slice, ast.Slice):
            self.score += 1.0
        self.generic_visit(node)

def get_slicing_score(code_str):
    visitor = SlicingVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Sub-category 3.5: Unnecessary data creation, duplication, or conversion
class DataCreationVisitor(BaseEfficiencyVisitor):
    """
    检测逻辑： 循环内调用 list(), dict(), set(), sorted(), copy() 等构造函数。
    """
    def visit_Call(self, node):
        if self.in_loop and isinstance(node.func, ast.Name):
            if node.func.id in ['list', 'dict', 'set', 'sorted', 'copy']:
                self.score += 1.0
        self.generic_visit(node)

def get_data_creation_score(code_str):
    visitor = DataCreationVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Category 4: Underutilization of language-specific features
# Sub-category 4.1: Failure to utilize built-in functions or libraries
class BuiltinFailureVisitor(ast.NodeVisitor):
    """
    检测逻辑 (Pattern Matching)： 捕捉“手动累积”模式。即：
        1、检测到一个 For 循环。
        2、循环体非常短（通常只有 1 行）。
        3、循环体内是 AugAssign（+=, *=）或者简单的 If-Compare-Assign（用于找最大值）。
        4、如果匹配此模式，意味着应该使用 sum() 或 max()。
    评分公式： $$B_{4.1}(y) = \mathbb{I}(\text{Manual Summation Loop}) + \mathbb{I}(\text{Manual Min/Max Loop})$$
    """
    def __init__(self):
        self.score = 0.0

    def visit_For(self, node):
        # 模式1: 手动求和 (Manual Summation)
        # 结构: for x in y: s += x
        if len(node.body) == 1 and isinstance(node.body[0], ast.AugAssign):
            aug = node.body[0]
            if isinstance(aug.op, ast.Add): # s += x
                # 这是一个强烈的信号，虽然可能有假阳性，但在生成代码中极高概率是手动求和
                self.score += 1.0
        
        # 模式2: 手动求最大/最小值 (Manual Min/Max)
        # 结构: for x in y: if x > m: m = x
        if len(node.body) == 1 and isinstance(node.body[0], ast.If):
            if_node = node.body[0]
            # 检查是否有 if x > m: ...
            if isinstance(if_node.test, ast.Compare):
                 # 这里简化检测：如果是单行 if，且做比较，极大可能是手动 min/max
                 self.score += 0.5

        self.generic_visit(node)

def get_builtin_failure_score(code_str):
    visitor = BuiltinFailureVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Sub-category 4.2: Lack of idiomatic constructs
class UnidiomaticVisitor(ast.NodeVisitor):
    def __init__(self):
        self.score = 0.0
    def visit_Call(self, node):
        # range(len(...))
        if isinstance(node.func, ast.Name) and node.func.id == 'range':
            if node.args and isinstance(node.args[0], ast.Call):
                if isinstance(node.args[0].func, ast.Name) and node.args[0].func.id == 'len':
                    self.score += 1.0
        self.generic_visit(node)
        
    def visit_Compare(self, node):
        # == True / == False
        if isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
             if isinstance(node.comparators[0], ast.Constant) and isinstance(node.comparators[0].value, bool):
                 self.score += 0.5
        self.generic_visit(node)

def get_unidiomatic_score(code_str):
    visitor = UnidiomaticVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Category 5: Memory inefficiencies
# Sub-category 5.1: Unnecessary buffering or intermediate storage
# Sub-category 5.2: Creation of large or avoidable temporary data
class MemoryInefficiencyVisitor(ast.NodeVisitor):
    """
    检测逻辑： 使用列表推导式（List Comprehension [...]）赋值给一个变量，随后只在循环中遍历它。应使用生成器表达式（Generator Expression (...)）。
    评分公式： $B(y) = \text{Count}(\text{Large ListComp assigned to var})$
    """
    def __init__(self):
        self.score = 0.0
    def visit_Assign(self, node):
        # 检测 v = [x for x in y]
        if isinstance(node.value, ast.ListComp):
            self.score += 0.8 # 建议用生成器
        self.generic_visit(node)

def get_memory_score(code_str):
    visitor = MemoryInefficiencyVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Category 6: Other inefficiencies
# Sub-category 6.1: Lack of input-scale awareness
class ScaleAwarenessVisitor(ast.NodeVisitor):
    """
    检测逻辑：
        1、检测文件操作方法 readlines()（读取所有行到列表）和 read()（读取整个文件）。高效做法是直接迭代文件句柄 for line in f:。
        2、检测 list(generator) 的转换，如果这个 generator 可能很大（例如 range(10**9)），虽然静态难以判断大小，但转换本身由于破坏了流式特性，在大数据场景下是低效风险点。
    评分公式： $$B_{6.1}(y) = \sum \mathbb{I}(\text{Call to readlines/read}) + 0.5 \times \mathbb{I}(\text{Materializing huge generator})$$
    """
    def __init__(self):
        self.score = 0.0

    def visit_Call(self, node):
        # 1. 检测文件全量读取
        if isinstance(node.func, ast.Attribute):
            # f.readlines() -> 内存爆炸风险
            if node.func.attr == 'readlines':
                self.score += 1.0
            # f.read() 且没有参数 -> 读取整个文件
            elif node.func.attr == 'read':
                if not node.args: # 无参数
                    self.score += 0.8
        
        # 2. 检测范围实体化: list(range(huge))
        # 这在 Python 3 中是低效的，如果只是为了迭代
        if isinstance(node.func, ast.Name) and node.func.id == 'list':
            if node.args and isinstance(node.args[0], ast.Call):
                inner = node.args[0]
                if isinstance(inner.func, ast.Name) and inner.func.id == 'range':
                    # 这是一个 heuristic，惩罚 list(range(...)) 写法
                    self.score += 0.5
                    
        self.generic_visit(node)

def get_scale_score(code_str):
    visitor = ScaleAwarenessVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Sub-category 6.2: Inefficient I/O processing
class IOVisitor(BaseEfficiencyVisitor):
    """
    检测逻辑： 在循环内部频繁调用 print() 或文件 input()。这是 IO 瓶颈的主要来源。
    评分公式： $B(y) = \text{Count}(\text{I/O in Loop})$
    """
    def visit_Call(self, node):
        if self.in_loop and isinstance(node.func, ast.Name) and (node.func.id == 'print' or node.func.id == 'input'):
            self.score += 1.0
        self.generic_visit(node)

def get_io_score(code_str):
    visitor = IOVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Sub-category 6.3: Inefficient exception handling patterns
class ExceptionVisitor(BaseEfficiencyVisitor):
    """
    检测逻辑： 捕获所有异常 except: 或 except Exception: 且处理块为空（吞掉错误），或者在循环内部频繁 try-except。
    """
    def visit_Try(self, node):
        if self.in_loop:
            self.score += 0.5 # 循环内 try-except 开销大
        for handler in node.handlers:
            if handler.type is None: # bare except
                self.score += 1.0
        self.generic_visit(node)

def get_exception_score(code_str):
    visitor = ExceptionVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

# Sub-category 6.4: Unnecessary or redundant code constructs
class RedundancyVisitor(ast.NodeVisitor):
    """
    检测逻辑：
        1、冗余的布尔返回 (Redundant Boolean Return)
        2、死代码 (Dead Code / Pass)
        3、无用赋值 (Unused Assignment)
    评分公式： $$B_{6.4}(y) = \text{Count}(\text{Redundant Bool Logic}) + \text{Count}(\text{Statement after Return})$$
    """
    def __init__(self):
        self.score = 0.0
    
    def visit_If(self, node):
        # 检测 if x: return True else: return False
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Return) and
            len(node.orelse) == 1 and isinstance(node.orelse[0], ast.Return)):
            
            ret_body = node.body[0].value
            ret_else = node.orelse[0].value
            
            if (isinstance(ret_body, ast.Constant) and ret_body.value is True and
                isinstance(ret_else, ast.Constant) and ret_else.value is False):
                self.score += 1.0 # 发现冗余逻辑
        
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # 检测 Return 后的死代码 (简易版)
        has_returned = False
        for child in node.body:
            if has_returned:
                self.score += 1.0 # Return 后还有代码，明显的冗余/错误
                break
            if isinstance(child, ast.Return):
                has_returned = True
        self.generic_visit(node)

def get_redundancy_score(code_str):
    visitor = RedundancyVisitor()
    try:
        visitor.visit(ast.parse(code_str))
        return visitor.score
    except: return 0.0

def get_all_scores(code: str) -> dict:
    return {
        "bad_api": get_bad_api_score(code),
        "recursion": get_recursion_score(code),
        "complexity": get_complexity_score(code),
        "conditional": get_conditional_score(code),
        "nested_loop": get_nested_loop_score(code),
        "multipass": get_multipass_score(code),
        "ds_selection": get_ds_selection_score(code),
        "list_op": get_list_op_score(code),
        "slicing": get_slicing_score(code),
        "data_creation": get_data_creation_score(code),
        "builtin_failure": get_builtin_failure_score(code),
        "unidiomatic": get_unidiomatic_score(code),
        "memory": get_memory_score(code),
        "scale": get_scale_score(code),
        "io": get_io_score(code),
        "exception": get_exception_score(code),
        "redundancy": get_redundancy_score(code),
    }

if __name__ == "__main__":
    # quick local test harness
    sample1 = """
def twoSum(nums: list[int], target: int) -> list[int]:
    res = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                res.append(i)
                res.append(j)
    return res
"""
    sample2 = """
def twoSum(nums: list[int], target: int) -> list[int]:
    num_map = \{\}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []
"""
    print(get_all_scores(sample1))
    print(get_all_scores(sample2))