import ast
import json
from typing import List, Dict, Any, Optional, Set, Tuple
import sys
import math
import statistics
sys.stdout.reconfigure(encoding='utf-8')

# Import radon for cyclomatic complexity
try:
    import importlib.util
    module_path = rf"C:\Program Files\Python311\Lib\site-packages\radon\complexity.py"
    spec = importlib.util.spec_from_file_location("radon_complexity", module_path)
    radon_complexity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(radon_complexity)
    cc_visit = radon_complexity.cc_visit
    RADON_AVAILABLE = True
except Exception:
    RADON_AVAILABLE = False
    cc_visit = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _safe_parse(code: str) -> Optional[ast.AST]:
    """
    Safely parse Python code into an AST.
    Returns None if syntax errors occur.
    """
    try:
        return ast.parse(code, mode="exec")
    except (SyntaxError, ValueError, TypeError):
        return None


def _is_constant_value(node: ast.AST, value: Any) -> bool:
    """Check if a node is a Constant with specific value."""
    return isinstance(node, ast.Constant) and node.value == value


def _is_name(node: ast.AST, name: str) -> bool:
    """Check if a node is a Name with specific id."""
    return isinstance(node, ast.Name) and node.id == name


def _extract_name(node: ast.AST) -> Optional[str]:
    """Extract variable name from various node types."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            return node.value.id
    return None


# ============================================================================
# BASE VISITOR CLASS
# ============================================================================

class BaseVisitor(ast.NodeVisitor):
    def __init__(self):
        self.loop_depth = 0
        self.comprehension_depth = 0
        self.function_depth = 0
        self.score = 0.0

    def _in_loop(self) -> bool:
        """Check if currently inside a loop."""
        return self.loop_depth > 0

    def _in_comprehension(self) -> bool:
        """Check if currently inside a comprehension."""
        return self.comprehension_depth > 0

    def _in_function(self) -> bool:
        """Check if currently inside a function."""
        return self.function_depth > 0

    def visit_For(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_AsyncFor(self, node):
        self.loop_depth += 1
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_ListComp(self, node):
        self.comprehension_depth += 1
        self.generic_visit(node)
        self.comprehension_depth -= 1

    def visit_SetComp(self, node):
        self.comprehension_depth += 1
        self.generic_visit(node)
        self.comprehension_depth -= 1

    def visit_DictComp(self, node):
        self.comprehension_depth += 1
        self.generic_visit(node)
        self.comprehension_depth -= 1

    def visit_GeneratorExp(self, node):
        self.comprehension_depth += 1
        self.generic_visit(node)
        self.comprehension_depth -= 1

    def visit_FunctionDef(self, node):
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_AsyncFunctionDef(self, node):
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1


# ============================================================================
# CATEGORY 0: SYNTAX CORRECTNESS
# ============================================================================

def get_syntax_correctness(code: str) -> float:
    """
    Return inf if syntax error, else 0.0.
    IMPROVEMENT: Added better error categorization.
    """
    if not code or not code.strip():
        return float("inf")
    
    tree = _safe_parse(code)
    return float("inf") if tree is None else 0.0


# ============================================================================
# CATEGORY 1: INEFFICIENT FUNCTION OR API USAGE
# ============================================================================
# Sub-category 1.1: Suboptimal method or API selection
class _BadAPIVisitor(BaseVisitor):
    """
    Detect suboptimal API usage.
    IMPROVEMENTS:
    1. Better detection of O(n) operations in loops
    2. Track variable types to reduce false positives
    3. Detect append in loop when extend would work
    4. Detect sorted() when sort() would suffice
    """
    def __init__(self):
        super().__init__()
        self.list_vars = set()  # Variables known to be lists
        self.dict_vars = set()  # Variables known to be dicts
        self.set_vars = set()   # Variables known to be sets

    def visit_Assign(self, node):
        """Track variable types from assignments."""
        if isinstance(node.value, ast.List):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.list_vars.add(target.id)
        elif isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.dict_vars.add(target.id)
        elif isinstance(node.value, ast.Set):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.set_vars.add(target.id)
        elif isinstance(node.value, ast.Call):
            func_name = _extract_name(node.value.func) if hasattr(node.value, 'func') else None
            if func_name == 'list':
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.list_vars.add(target.id)
            elif func_name == 'dict':
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.dict_vars.add(target.id)
            elif func_name == 'set':
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.set_vars.add(target.id)
        
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            obj_name = _extract_name(node.func.value)
            
            # IMPROVEMENT: Check list.pop(0) and list.insert(0, x) - O(n) operations
            if method == "pop" and node.args:
                if _is_constant_value(node.args[0], 0):
                    self.score += 3.0 if self._in_loop() else 1.5
            
            if method == "insert" and len(node.args) >= 2:
                if _is_constant_value(node.args[0], 0):
                    self.score += 3.0 if self._in_loop() else 1.5
            
            # IMPROVEMENT: Detect list operations in loops more accurately
            if self._in_loop():
                # count(), index(), remove() are O(n) - in loop becomes O(n²)
                if method in ("count", "index", "remove"):
                    # Only penalize if we know it's a list or likely a list
                    if obj_name in self.list_vars or method == "remove":
                        self.score += 1.5
                    else:
                        self.score += 0.5  # Conservative penalty
                
                # IMPROVEMENT: Detect repeated append when extend would work
                if method == "append":
                    self.score += 0.3  # Mild penalty, might be necessary
                
                # IMPROVEMENT: Detect sorted() creating new list in loop
                if method == "sort" and obj_name:
                    # sort() modifies in place - this is OK
                    pass
        
        # IMPROVEMENT: Check for sorted() when .sort() would work
        if isinstance(node.func, ast.Name):
            if node.func.id == "sorted" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Name) and arg.id in self.list_vars:
                    # sorted() creates new list, .sort() modifies in place
                    self.score += 0.5
                
                # In loop, sorted() is expensive
                if self._in_loop():
                    self.score += 1.0
        
        self.generic_visit(node)

    def visit_Compare(self, node):
        """
        IMPROVEMENT: Better detection of linear search patterns.
        Distinguish between set/dict lookup (O(1)) and list lookup (O(n)).
        """
        if self._in_loop():
            for i, op in enumerate(node.ops):
                if isinstance(op, (ast.In, ast.NotIn)) and i < len(node.comparators):
                    comp = node.comparators[i]
                    
                    # Direct list/tuple literals
                    if isinstance(comp, (ast.List, ast.Tuple)):
                        self.score += 1.0
                    
                    # List comprehensions
                    elif isinstance(comp, ast.ListComp):
                        self.score += 1.2
                    
                    # Calls that create lists
                    elif isinstance(comp, ast.Call):
                        func_name = None
                        if isinstance(comp.func, ast.Name):
                            func_name = comp.func.id
                        
                        if func_name in ("list", "sorted", "reversed"):
                            self.score += 1.0
                        elif func_name == "range":
                            # range is efficient for membership testing
                            self.score += 0.2
                    
                    # Variable - check if we know its type
                    elif isinstance(comp, ast.Name):
                        if comp.id in self.list_vars:
                            self.score += 0.9
                        elif comp.id in self.set_vars or comp.id in self.dict_vars:
                            # Set/dict membership is O(1) - no penalty
                            pass
                        else:
                            # Unknown type - mild penalty
                            self.score += 0.3
                    
                    # Attribute access (e.g., self.items)
                    elif isinstance(comp, ast.Attribute):
                        self.score += 0.4
        
        self.generic_visit(node)

def get_bad_api_score(code: str) -> float:
    """
    Score for suboptimal API usage.
    Returns float indicating severity.
    """
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _BadAPIVisitor()
    v.visit(tree)
    return float(v.score)

# Sub-category 1.2: Unnecessary or excessive recursion
class _RecursionVisitor(ast.NodeVisitor):
    """
    Detect recursion patterns.
    IMPROVEMENTS:
    1. Detect tail recursion vs general recursion
    2. Identify memoization patterns to reduce score
    3. Detect mutual recursion
    4. Better handling of class methods
    """
    def __init__(self):
        self.scores = {}
        self.function_calls = {}  # Track which functions call which
        self.has_memoization = set()  # Functions with @lru_cache or similar

    def visit_FunctionDef(self, node):
        fname = node.name
        
        # Check for memoization decorators
        has_cache = any(
            isinstance(d, ast.Name) and d.id in ('lru_cache', 'cache', 'memoize')
            or isinstance(d, ast.Attribute) and d.attr in ('lru_cache', 'cache')
            for d in node.decorator_list
        )
        
        if has_cache:
            self.has_memoization.add(fname)
        
        # Count self-calls
        self_calls = []
        last_stmt_is_return_call = False
        
        for i, n in enumerate(ast.walk(node)):
            # Direct function call: foo(...)
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name) and n.func.id == fname:
                    self_calls.append(n)
                
                # Method call: self.foo(...) or cls.foo(...)
                elif isinstance(n.func, ast.Attribute) and n.func.attr == fname:
                    if isinstance(n.func.value, ast.Name) and n.func.value.id in ("self", "cls"):
                        self_calls.append(n)
        
        # IMPROVEMENT: Check if last statement is return with recursive call (tail recursion)
        if node.body:
            last = node.body[-1]
            if isinstance(last, ast.Return) and isinstance(last.value, ast.Call):
                call = last.value
                is_tail_recursive = False
                if isinstance(call.func, ast.Name) and call.func.id == fname:
                    is_tail_recursive = True
                elif isinstance(call.func, ast.Attribute) and call.func.attr == fname:
                    if isinstance(call.func.value, ast.Name) and call.func.value.id in ("self", "cls"):
                        is_tail_recursive = True
                
                if is_tail_recursive:
                    last_stmt_is_return_call = True
        
        if self_calls:
            # Base score for having recursion
            score = 1.0
            
            # IMPROVEMENT: Multiple recursive calls (tree recursion) is more complex
            if len(self_calls) > 1:
                score += 2.0 * (len(self_calls) - 1)
            
            # IMPROVEMENT: Reduce penalty if memoized
            if fname in self.has_memoization:
                score *= 0.3  # Memoization makes recursion acceptable
            
            # IMPROVEMENT: Tail recursion is slightly better (can be optimized)
            elif last_stmt_is_return_call and len(self_calls) == 1:
                score *= 0.7  # Tail recursion is more optimizable
            
            self.scores[fname] = score
        
        self.generic_visit(node)


def get_recursion_score(code: str) -> float:
    """
    Score for recursion usage.
    Lower score if recursion is memoized or tail-recursive.
    """
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _RecursionVisitor()
    v.visit(tree)
    return float(sum(v.scores.values()))


# ============================================================================
# CATEGORY 2: ALGORITHMIC INEFFICIENCIES
# ============================================================================
# Sub-category 2.1: Brute-force or suboptimal algorithmic strategy
# Sub-category 2.2: Absence of established optimization techniques
# Sub-category 2.3: Insufficient mathematical abstraction and optimization

def _get_node_length(node: ast.AST) -> Optional[int]:
    """
    Estimate node length in lines using available attributes (lineno, end_lineno).
    Returns None if not available.
    """
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None) or getattr(node, "endline", None) or None
    if start is None or end is None:
        return None
    return max(0, end - start + 1)


def _get_radon_block_length(block) -> Optional[int]:
    """
    Radon's CCResult may expose lineno and endline-like attributes; try several names.
    """
    for attr in ("endline", "end_lineno", "lastline", "lineno"):
        if hasattr(block, attr):
            # If using lineno and endline combination:
            start = getattr(block, "lineno", None)
            end = getattr(block, "endline", None) or getattr(block, "end_lineno", None) or getattr(block, "lastline", None)
            if start is not None and end is not None:
                return max(0, end - start + 1)
    # fallback: some radon versions may store 'lineno' only or none
    return None


def get_algorithmic_score(code: str) -> float:
    """
    Calculate a robustness-focused complexity score for a code snippet (module).
    """
    # 1) Try Radon first (best-effort)
    if RADON_AVAILABLE and cc_visit is not None:
        try:
            blocks = cc_visit(code)
            if blocks:
                complexities: List[float] = [getattr(b, "complexity", 0) or 0 for b in blocks]
                total_cc = sum(complexities)
                num_blocks = len(blocks)
                avg_cc = total_cc / num_blocks if num_blocks else 0.0
                max_cc = max(complexities) if complexities else 0.0
                stdev_cc = statistics.pstdev(complexities) if len(complexities) > 1 else 0.0

                # Try to estimate block lengths when radon provides that info.
                lengths: List[int] = []
                for b in blocks:
                    ln = _get_radon_block_length(b)
                    if ln:
                        lengths.append(ln)

                avg_len = statistics.mean(lengths) if lengths else 0.0
                max_len = max(lengths) if lengths else 0.0

                # Compose penalties (non-linear to penalize extremes more)
                # - average complexity penalty (threshold ~4)
                avg_pen = max(0.0, (avg_cc - 4.0)) ** 1.2 / 2.5

                # - maximum complexity penalty (threshold ~10)
                max_pen = max(0.0, (max_cc - 10.0)) ** 1.25 / 3.0

                # - variability penalty (high variance indicates inconsistent hotspots)
                stdev_pen = (stdev_cc / 4.0) ** 1.15 if stdev_cc > 0 else 0.0

                # - length penalty (long functions are suspicious)
                length_pen = max(0.0, (avg_len - 60.0)) ** 1.1 / 120.0  # scaled down

                # - absolute long-block penalty (very long blocks)
                long_block_pen = max(0.0, (max_len - 200.0)) ** 1.05 / 400.0

                raw_score = avg_pen + max_pen + stdev_pen + length_pen + long_block_pen

                # Smooth and scale final score to avoid huge numbers; keep monotonicity.
                score = math.log1p(1.0 + raw_score) * 1.6

                # Small floor to keep type consistent
                return float(max(0.0, score))
        except Exception:
            # If radon fails for any reason, fall back to AST-based analysis.
            pass

    # 2) AST fallback
    tree = _safe_parse(code)
    if not tree:
        return 0.0

    class ComplexityCounter(ast.NodeVisitor):
        def __init__(self):
            self.count = 0  # branching-like constructs count
            self.depth = 0
            self.max_depth = 0
            self.bool_ops = 0  # number of boolean ops (and/or) contributing extra branches
            self.comprehensions = 0
            self.ifexp = 0  # ternary expressions
            self.match_cases = 0
            self.function_ranges: List[Tuple[int, int]] = []  # (start, end) lineno
            self._current_function_start: Optional[int] = None

        def _enter_block(self):
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            self.count += 1

        def _exit_block(self):
            self.depth = max(0, self.depth - 1)

        def visit_If(self, node: ast.If):
            self._enter_block()
            self.generic_visit(node)
            self._exit_block()

        def visit_For(self, node: ast.For):
            self._enter_block()
            self.generic_visit(node)
            self._exit_block()

        def visit_While(self, node: ast.While):
            self._enter_block()
            self.generic_visit(node)
            self._exit_block()

        def visit_Try(self, node: ast.Try):
            # try itself doesn't add many branches, but except handlers do.
            # Count try as a block and each except separately.
            self._enter_block()
            # except handlers
            self.count += len([h for h in node.handlers if isinstance(h, ast.ExceptHandler)])
            self.generic_visit(node)
            self._exit_block()

        def visit_ExceptHandler(self, node: ast.ExceptHandler):
            # in case handlers are visited directly
            self._enter_block()
            self.generic_visit(node)
            self._exit_block()

        def visit_With(self, node: ast.With):
            # context managers can hide complexity (resources, multi-withs)
            self._enter_block()
            self.generic_visit(node)
            self._exit_block()

        def visit_Match(self, node: ast.Match):
            # pattern matching: number of cases increases branching
            self._enter_block()
            self.match_cases += len(node.cases) if getattr(node, "cases", None) else 0
            self.generic_visit(node)
            self._exit_block()

        def visit_BoolOp(self, node: ast.BoolOp):
            # Boolean operators like "and/or" add implicit branching depending on values
            # If n values in BoolOp, it contributes (n-1) additional short-circuit decisions.
            self.bool_ops += max(0, len(node.values) - 1)
            self.generic_visit(node)

        def visit_IfExp(self, node: ast.IfExp):
            # Ternary conditional
            self.ifexp += 1
            self.generic_visit(node)

        def visit_ListComp(self, node: ast.ListComp):
            self.comprehensions += 1
            self.generic_visit(node)

        def visit_SetComp(self, node: ast.SetComp):
            self.comprehensions += 1
            self.generic_visit(node)

        def visit_DictComp(self, node: ast.DictComp):
            self.comprehensions += 1
            self.generic_visit(node)

        def visit_GeneratorExp(self, node: ast.GeneratorExp):
            self.comprehensions += 1
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            # Record function start/end lines if available for length estimation
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or getattr(node, "endline", None) or None
            if start is not None and end is not None:
                self.function_ranges.append((start, end))
            # treat each function body as its own "module" for nested analysis
            self._enter_block()
            self.generic_visit(node)
            self._exit_block()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.visit_FunctionDef(node)

    counter = ComplexityCounter()
    counter.visit(tree)

    # Compose fallback metrics into a score
    # Base counts: treat 'count' as primary branching events
    count_score = 0.0
    if counter.count > 5:
        count_score = ((counter.count - 5) / 3.0) ** 1.05

    # Nesting depth penalty (penalize deep nesting more aggressively)
    depth_score = 0.0
    if counter.max_depth > 3:
        depth_score = ((counter.max_depth - 3) / 2.0) ** 1.25

    # Boolean/op/comprehension penalties (smaller weight)
    bool_score = counter.bool_ops / 4.0  # each ~4 boolean ops yields +1
    comp_score = counter.comprehensions / 2.0
    ifexp_score = counter.ifexp * 0.6
    match_score = max(0.0, (counter.match_cases - 3) / 3.0)

    # Function length penalty (long functions are suspicious)
    func_len_pen = 0.0
    if counter.function_ranges:
        lengths = [max(0, end - start + 1) for (start, end) in counter.function_ranges]
        avg_len = statistics.mean(lengths)
        max_len = max(lengths)
        if avg_len > 50:
            func_len_pen += ((avg_len - 50) / 30.0) ** 1.1
        if max_len > 150:
            func_len_pen += ((max_len - 150) / 100.0) ** 1.05

    raw_score = count_score + depth_score + bool_score + comp_score + ifexp_score + match_score + func_len_pen
    return float(max(0.0, raw_score))


# Sub-category 2.4: Inefficient conditional logic
class _ConditionalVisitor(BaseVisitor):
    """
    IMPROVEMENTS:
    1. Detect redundant conditions (if x == True)
    2. Detect conditions that could be simplified with 'in'
    3. Better elif chain detection
    4. Detect boolean expression complexity
    """
    def __init__(self):
        super().__init__()
        self.if_depth = 0
        self.max_if_depth = 0
        self.elif_chains = []

    def visit_If(self, node):
        self.if_depth += 1
        self.max_if_depth = max(self.max_if_depth, self.if_depth)

        # Penalty for if inside loop
        if self._in_loop():
            self.score += 0.5

        # IMPROVEMENT: Count elif chain length
        elif_chain_len = 0
        current = node
        while current.orelse and len(current.orelse) == 1:
            orelse = current.orelse[0]
            if isinstance(orelse, ast.If):
                elif_chain_len += 1
                current = orelse
            else:
                break

        # Long elif chains suggest dictionary dispatch might be better
        if elif_chain_len > 4:
            self.score += 0.5 * (elif_chain_len - 4)
        
        # IMPROVEMENT: Detect redundant boolean comparisons
        if isinstance(node.test, ast.Compare):
            ops = node.test.ops
            comparators = node.test.comparators
            
            for op, comp in zip(ops, comparators):
                # if x == True or if x == False
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    if _is_constant_value(comp, True) or _is_constant_value(comp, False):
                        self.score += 0.5
        
        # IMPROVEMENT: Check for complex boolean expressions that could be simplified
        bool_op_count = sum(1 for n in ast.walk(node.test) if isinstance(n, ast.BoolOp))
        if bool_op_count > 3:
            self.score += 0.3 * (bool_op_count - 3)

        self.generic_visit(node)
        self.if_depth -= 1


def get_conditional_score(code: str) -> float:
    """Score for inefficient conditional logic."""
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _ConditionalVisitor()
    v.visit(tree)
    
    # Extra penalty for extreme nesting
    extra = max(0, v.max_if_depth - 4) * 0.5
    return float(v.score + extra)


# Sub-category 2.5: Avoidable nested-loop complexity
class _NestedLoopVisitor(BaseVisitor):
    """
    IMPROVEMENTS:
    1. Detect actual nested iterations (not just structural nesting)
    2. Track loop variables to detect potential optimizations
    3. Identify break/continue patterns that might reduce actual complexity
    """
    def __init__(self):
        super().__init__()
        self.max_depth = 0
        self.current_depth = 0
        self.loop_has_break = []  # Track if loop has break statement

    def visit_For(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        
        # Check if loop has break (might reduce actual iterations)
        has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
        self.loop_has_break.append(has_break)
        
        self.generic_visit(node)
        
        self.loop_has_break.pop()
        self.current_depth -= 1

    def visit_While(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        
        has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
        self.loop_has_break.append(has_break)
        
        self.generic_visit(node)
        
        self.loop_has_break.pop()
        self.current_depth -= 1


def get_nested_loop_score(code: str) -> float:
    """
    Score for nested loops.
    IMPROVEMENT: Consider early exits and actual iteration patterns.
    """
    tree = _safe_parse(code)
    if not tree:
        return 0.0
    
    v = _NestedLoopVisitor()
    v.visit(tree)
    
    # Allow single nesting (common and often necessary)
    depth_over_threshold = max(0, v.max_depth - 1)
    
    if depth_over_threshold == 0:
        return 0.0
    
    # IMPROVEMENT: Exponential penalty but capped
    # 2 levels: 1 point, 3 levels: 3 points, 4 levels: 7 points, etc.
    penalty = (2 ** v.max_depth) - 1
    
    return float(min(penalty, 31.0))  # Cap at 30 to avoid extreme values


# Sub-category 2.6/2.7: Unnecessary multi-pass processing / redundant recomputation
class _MultiPassVisitor(BaseVisitor):
    """
    IMPROVEMENTS:
    1. Better tracking of what's being iterated
    2. Distinguish between necessary and unnecessary multi-pass
    3. Track comprehension usage separately
    4. Detect filter-then-process patterns
    """
    def __init__(self):
        super().__init__()
        self.iterated_vars = {}  # var -> count
        self.comprehension_targets = set()
        self.iteration_contexts = []  # Track what we're iterating over

    def _extract_iter_name(self, node):
        """Extract the base iterable name from various expressions."""
        if isinstance(node, ast.Name):
            return node.id
        
        if isinstance(node, ast.Call):
            # range(x), enumerate(x), etc.
            if node.args and isinstance(node.args[0], ast.Name):
                return node.args[0].id
            # Nested calls
            if node.args and isinstance(node.args[0], ast.Call):
                return self._extract_iter_name(node.args[0])
        
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return node.value.id
        
        if isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Name):
                return base.id
        
        return None

    def visit_For(self, node):
        iter_name = self._extract_iter_name(node.iter)
        
        if iter_name:
            # Track iterations
            self.iterated_vars.setdefault(iter_name, 0)
            self.iterated_vars[iter_name] += 1
            
            # IMPROVEMENT: Multiple iterations of same collection
            if self.iterated_vars[iter_name] > 1:
                # Check if it's in comprehension (might be intentional)
                if iter_name not in self.comprehension_targets:
                    self.score += 1.5
                else:
                    # Lower penalty if used in comprehension
                    self.score += 0.5
        
        self.iteration_contexts.append(iter_name)
        self.generic_visit(node)
        self.iteration_contexts.pop()

    def visit_ListComp(self, node):
        for gen in node.generators:
            name = self._extract_iter_name(gen.iter)
            if name:
                self.comprehension_targets.add(name)
        self.generic_visit(node)

    def visit_SetComp(self, node):
        for gen in node.generators:
            name = self._extract_iter_name(gen.iter)
            if name:
                self.comprehension_targets.add(name)
        self.generic_visit(node)

    def visit_DictComp(self, node):
        for gen in node.generators:
            name = self._extract_iter_name(gen.iter)
            if name:
                self.comprehension_targets.add(name)
        self.generic_visit(node)


def get_multipass_score(code: str) -> float:
    """Score for unnecessary multi-pass processing."""
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _MultiPassVisitor()
    v.visit(tree)
    return float(v.score)


# ============================================================================
# CATEGORY 3: INEFFICIENT DATA STRUCTURE USAGE
# ============================================================================
# Sub-category 3.1: Inappropriate data structure selection
class _BadDSSelectionVisitor(BaseVisitor):
    """
    IMPROVEMENTS:
    1. Better type tracking across assignments
    2. Detect when set/dict would be better than list
    3. Track collection sizes (if possible)
    """
    def __init__(self):
        super().__init__()
        self.list_vars = set()
        self.set_vars = set()
        self.dict_vars = set()

    def visit_Assign(self, node):
        # Track variable types
        for target in node.targets:
            if isinstance(target, ast.Name):
                if isinstance(node.value, ast.List):
                    self.list_vars.add(target.id)
                elif isinstance(node.value, ast.Set):
                    self.set_vars.add(target.id)
                elif isinstance(node.value, ast.Dict):
                    self.dict_vars.add(target.id)
                elif isinstance(node.value, ast.Call):
                    func_name = _extract_name(node.value.func)
                    if func_name == 'list':
                        self.list_vars.add(target.id)
                    elif func_name == 'set':
                        self.set_vars.add(target.id)
                    elif func_name == 'dict':
                        self.dict_vars.add(target.id)
        
        self.generic_visit(node)

    def visit_Compare(self, node):
        if self._in_loop():
            for i, op in enumerate(node.ops):
                if isinstance(op, (ast.In, ast.NotIn)) and i < len(node.comparators):
                    comp = node.comparators[i]
                    
                    # Literal collections
                    if isinstance(comp, (ast.List, ast.Tuple)):
                        self.score += 1.0
                    elif isinstance(comp, ast.ListComp):
                        self.score += 1.2
                    
                    # Known list variables
                    elif isinstance(comp, ast.Name):
                        if comp.id in self.list_vars:
                            self.score += 1.0
                        elif comp.id in self.set_vars or comp.id in self.dict_vars:
                            # Good - O(1) lookup
                            pass
                        else:
                            # Unknown - mild penalty
                            self.score += 0.4
                    
                    # Method calls that return lists
                    elif isinstance(comp, ast.Call):
                        func_name = _extract_name(comp.func) if hasattr(comp, 'func') else None
                        if func_name in ('list', 'sorted', 'reversed'):
                            self.score += 1.0
        
        self.generic_visit(node)


def get_ds_selection_score(code: str) -> float:
    """Score for poor data structure selection."""
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _BadDSSelectionVisitor()
    v.visit(tree)
    return float(v.score)


# Sub-category 3.2: Inefficient operations on selected data structure
class _InefficientDSOpVisitor(BaseVisitor):
    """
    Detects inefficient operations on data structures.
    
    IMPROVEMENTS:
    1. Better type inference through multiple assignment patterns
    2. Detect chained operations that could be optimized
    3. Consider nested loop severity multipliers
    4. Track set operations that should use set methods
    5. Detect unnecessary conversions between types
    """
    def __init__(self):
        super().__init__()
        self.score = 0.0
        self.var_types: Dict[str, str] = {}
        self.loop_invariant_calls: Set[str] = set()

    def _get_loop_multiplier(self) -> float:
        """Return penalty multiplier based on loop nesting depth."""
        if self.loop_depth == 0:
            return 1.0
        elif self.loop_depth == 1:
            return 2.0
        else:
            # Nested loops are exponentially worse
            return 2.0 ** self.loop_depth

    def _call_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract the name of a function call."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        if isinstance(call_node.func, ast.Attribute):
            if isinstance(call_node.func.value, ast.Name):
                return f"{call_node.func.value.id}.{call_node.func.attr}"
            return call_node.func.attr
        return None

    def _obj_name(self, node) -> Optional[str]:
        """Extract object name from expression."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return node.value.id
        return None

    def _infer_type(self, node) -> Optional[str]:
        """Infer type from various node patterns."""
        if isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Set):
            return "set"
        elif isinstance(node, ast.Tuple):
            return "tuple"
        elif isinstance(node, ast.Call):
            cname = self._call_name(node)
            if cname in ("list", "dict", "set", "tuple", "deque", "collections.deque"):
                return cname.split('.')[-1]
        return None

    def visit_Assign(self, node):
        """Track variable types through assignment."""
        inferred = self._infer_type(node.value)
        if inferred:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.var_types[t.id] = inferred
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        """Track types from annotated assignments."""
        if isinstance(node.annotation, ast.Name):
            type_name = node.annotation.id.lower()
            if type_name in ("list", "dict", "set", "tuple", "deque"):
                if isinstance(node.target, ast.Name):
                    self.var_types[node.target.id] = type_name
        self.generic_visit(node)

    def visit_Call(self, node):
        """Detect inefficient method calls and operations."""
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            obj = node.func.value
            obj_name = self._obj_name(obj)
            obj_type = self.var_types.get(obj_name) if obj_name else None

            # IMPROVEMENT: Check for pop(0) and insert(0, x) on lists
            if method == "pop" and node.args:
                if isinstance(node.args[0], ast.Constant) and node.args[0].value == 0:
                    if obj_type == "deque":
                        # Using deque correctly, minor penalty
                        self.score += 0.1 * self._get_loop_multiplier()
                    else:
                        # O(n) operation on list
                        self.score += 1.0 * self._get_loop_multiplier()

            if method == "insert" and len(node.args) >= 2:
                if isinstance(node.args[0], ast.Constant) and node.args[0].value == 0:
                    if obj_type != "deque":
                        # O(n) operation on list
                        self.score += 1.0 * self._get_loop_multiplier()

            # IMPROVEMENT: Linear search operations
            if method in ("remove", "index"):
                if obj_type in ("list", None):
                    self.score += 0.8 * self._get_loop_multiplier()

            # count() is always O(n), even on lists
            if method == "count":
                self.score += 0.5 * self._get_loop_multiplier()

            # IMPROVEMENT: Detect dict.keys()/values()/items() in membership tests
            if method in ("keys", "values", "items"):
                if self._in_loop():
                    # These create views/iterators, but repeated calls are wasteful
                    self.score += 0.5

            # IMPROVEMENT: Detect append in comprehension (should use list comp)
            if method == "append" and self._in_comprehension():
                self.score += 0.8

            # File I/O in loops
            if method in ("write", "writelines", "flush"):
                if self._in_loop():
                    self.score += 0.8 * self._get_loop_multiplier()

        else:
            # Direct function calls
            if isinstance(node.func, ast.Name):
                fname = node.func.id

                # IMPROVEMENT: Unnecessary conversions in loops
                if fname in ("list", "tuple", "set") and self._in_loop():
                    self.score += 0.8 * self._get_loop_multiplier()

                # sorted() in loop - consider if it's on loop-invariant data
                if fname == "sorted" and self._in_loop():
                    self.score += 1.0 * self._get_loop_multiplier()

                # IMPROVEMENT: copy/deepcopy in loops
                if fname in ("copy", "deepcopy") and self._in_loop():
                    self.score += 1.2 * self._get_loop_multiplier()

        self.generic_visit(node)

    def visit_Compare(self, node):
        """Detect inefficient membership tests."""
        if self._in_loop():
            for i, (op, comp) in enumerate(zip(node.ops, node.comparators)):
                # Check for: x in d.values() or x in d.items()
                if isinstance(op, (ast.In, ast.NotIn)):
                    if isinstance(comp, ast.Call):
                        if isinstance(comp.func, ast.Attribute):
                            method = comp.func.attr
                            if method in ("values", "items"):
                                # O(n) membership test, should use keys or refactor
                                self.score += 1.5
                            elif method == "keys":
                                # Less severe but still unnecessary
                                self.score += 0.3

                    # IMPROVEMENT: Check for list membership in nested loops
                    if self.loop_depth > 1 and isinstance(comp, ast.Name):
                        obj_type = self.var_types.get(comp.id)
                        if obj_type == "list":
                            # Should consider using set for membership tests
                            self.score += 1.0

        self.generic_visit(node)


def get_ds_operations_score(code: str) -> float:
    """
    Score inefficient data structure operations.
    Higher scores indicate more inefficiencies.
    
    Returns:
        float: Penalty score (0.0 = efficient, higher = less efficient)
    """
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _InefficientDSOpVisitor()
    v.visit(tree)
    # Cap to prevent extreme scores
    return min(50.0, float(v.score))


# Sub-category 3.3: Inefficient string concatenation
class _StringConcatVisitor(BaseVisitor):
    """
    Detects inefficient string concatenation patterns.
    
    IMPROVEMENTS:
    1. Better detection of += patterns in loops
    2. Recognize good patterns (join with list)
    3. Track string building across multiple statements
    4. Consider f-strings vs concatenation
    5. Detect repeated string operations
    """
    def __init__(self):
        super().__init__()
        self.string_vars: Set[str] = set()
        self.concat_counts: Dict[str, int] = {}
        self.parts_lists: Set[str] = set()
        self.joined_lists: Set[str] = set()

    def visit_Assign(self, node):
        """Track string variables and list-building patterns."""
        # Track string literals
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.string_vars.add(t.id)

        # Track str() calls
        if isinstance(node.value, ast.Call):
            cname = self._call_name(node.value) if hasattr(self, '_call_name') else None
            if isinstance(node.value.func, ast.Name):
                if node.value.func.id == "str":
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            self.string_vars.add(t.id)
                # Track list initialization (good pattern for join)
                elif node.value.func.id in ("list", "[]") or not node.value.args:
                    if isinstance(node.value, ast.List):
                        for t in node.targets:
                            if isinstance(t, ast.Name):
                                self.parts_lists.add(t.id)

        # IMPROVEMENT: Detect s = s + "..." pattern (should use +=)
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    # Check if target appears in RHS
                    for n in ast.walk(node.value):
                        if isinstance(n, ast.Name) and n.id == t.id:
                            multiplier = 2.0 if self._in_loop() else 0.8
                            self.concat_counts[t.id] = self.concat_counts.get(t.id, 0) + 1
                            self.score += multiplier
                            break

        self.generic_visit(node)

    def visit_AugAssign(self, node):
        """Detect += string concatenation patterns."""
        if isinstance(node.op, ast.Add) and isinstance(node.target, ast.Name):
            name = node.target.id
            
            # Track concatenation count
            self.concat_counts[name] = self.concat_counts.get(name, 0) + 1
            
            if self._in_loop():
                # IMPROVEMENT: Exponential penalty for nested loops
                if self.loop_depth == 1:
                    penalty = 1.0
                elif self.loop_depth == 2:
                    penalty = 2.0
                else:
                    penalty = 3.0
                self.score += penalty
            else:
                # Single concatenation outside loop is fine
                self.score += 0.3

        self.generic_visit(node)

    def visit_Call(self, node):
        """Detect good (join) and bad patterns."""
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            obj = node.func.value

            # IMPROVEMENT: Recognize good pattern: "".join(parts)
            if method == "join":
                if isinstance(obj, ast.Constant) and isinstance(obj.value, str):
                    # This is the good pattern! Reward by reducing score
                    if node.args and isinstance(node.args[0], ast.Name):
                        list_name = node.args[0].id
                        if list_name in self.parts_lists:
                            # Confirmed good pattern, reduce penalty
                            self.joined_lists.add(list_name)
                            self.score = max(0, self.score - 1.0)

            # IMPROVEMENT: Detect % formatting in loops (old style)
            # This is less efficient than f-strings or join
            if method in ("format", "__mod__") and self._in_loop():
                self.score += 0.5

            # append() to parts list is good if followed by join
            if method == "append" and isinstance(obj, ast.Name):
                if obj.id in self.parts_lists and self._in_loop():
                    # This is potentially good, but we'll verify in finalize
                    pass

        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        """Track f-strings - these are generally efficient."""
        # f-strings in loops are better than +=, but still have some cost
        if self._in_loop():
            self.score += 0.2
        self.generic_visit(node)

    def finalize(self):
        """
        IMPROVEMENT: Adjust scores based on patterns observed.
        Penalize parts lists that were never joined.
        """
        # Check for parts lists that were never joined
        orphan_lists = self.parts_lists - self.joined_lists
        for orphan in orphan_lists:
            # List created but never joined - wasteful
            self.score += 1.0

        # IMPROVEMENT: Severe penalty for many concatenations on same var
        for var, count in self.concat_counts.items():
            if count > 10:
                # Excessive concatenations on one variable
                self.score += (count - 10) * 0.5


def get_string_concat_score(code: str) -> float:
    """
    Score inefficient string concatenation patterns.
    
    Returns:
        float: Penalty score (0.0 = efficient, higher = less efficient)
    """
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _StringConcatVisitor()
    v.visit(tree)
    v.finalize()
    # Normalize and cap
    return min(20.0, float(v.score))


# Sub-category 3.4: Repeated sequence slicing in loops
class _SlicingVisitor(BaseVisitor):
    """
    Detects inefficient slicing patterns.
    
    IMPROVEMENTS:
    1. Distinguish between constant and variable slices
    2. Track slice results that are immediately discarded
    3. Detect overlapping slices that could be optimized
    4. Consider string vs list slicing differences
    """
    def __init__(self):
        super().__init__()
        self.slice_vars: Dict[str, int] = {}

    def _is_constant_slice(self, slice_node: ast.Slice) -> bool:
        """Check if slice uses only constants."""
        parts = [slice_node.lower, slice_node.upper, slice_node.step]
        for part in parts:
            if part is not None:
                if not isinstance(part, ast.Constant):
                    # Has a variable component
                    return False
        return True

    def visit_Subscript(self, node):
        """Detect slicing operations."""
        slice_node = node.slice
        
        if isinstance(slice_node, ast.Slice):
            if self._in_loop():
                # IMPROVEMENT: Distinguish constant vs variable slices
                is_constant = self._is_constant_slice(slice_node)
                
                if is_constant:
                    # Constant slice - relatively efficient if small
                    # But repeated in loop is still wasteful
                    self.score += 0.5 * self.loop_depth
                else:
                    # Variable slice - O(n) each time, very inefficient
                    self.score += 1.5 * self.loop_depth
                
                # IMPROVEMENT: Track slicing on same variable
                if isinstance(node.value, ast.Name):
                    var_name = node.value.id
                    self.slice_vars[var_name] = self.slice_vars.get(var_name, 0) + 1
            
            elif self._in_comprehension():
                # Slicing in comprehension - moderate concern
                self.score += 0.3

        # IMPROVEMENT: Detect chained slicing (x[:10][:5])
        if isinstance(node.value, ast.Subscript):
            if isinstance(node.value.slice, ast.Slice):
                # Chained slicing - creates intermediate copy
                self.score += 1.0

        self.generic_visit(node)

    def finalize(self):
        """Adjust score based on repeated slicing patterns."""
        for var, count in self.slice_vars.items():
            if count > 5:
                # Same variable sliced many times in loop
                self.score += (count - 5) * 0.5


def get_slicing_score(code: str) -> float:
    """
    Score inefficient slicing patterns.
    
    Returns:
        float: Penalty score (0.0 = efficient, higher = less efficient)
    """
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _SlicingVisitor()
    v.visit(tree)
    v.finalize()
    return min(15.0, float(v.score))


# Sub-category 3.5: Unnecessary data creation, duplication, or conversion
class _DataCreationVisitor(BaseVisitor):
    """
    Detects unnecessary data structure creation and conversion.
    
    IMPROVEMENTS:
    1. Track if created data is actually used
    2. Detect redundant conversions (list -> set -> list)
    3. Identify large data structures in hot paths
    4. Check for defensive copies that aren't needed
    """
    def __init__(self):
        super().__init__()
        self.created_vars: Dict[str, str] = {}
        self.used_vars: Set[str] = set()
        self.conversion_chains: Dict[str, List[str]] = {}

    def visit_Assign(self, node):
        """Track data structure creation."""
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name):
                func_name = node.value.func.id
                
                # Track what was created
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        var_name = t.id
                        
                        # IMPROVEMENT: Track conversion chains
                        if node.value.args and isinstance(node.value.args[0], ast.Name):
                            source_var = node.value.args[0].id
                            if source_var in self.created_vars:
                                chain = self.conversion_chains.get(source_var, [self.created_vars[source_var]])
                                chain.append(func_name)
                                self.conversion_chains[var_name] = chain
                        
                        self.created_vars[var_name] = func_name

        self.generic_visit(node)

    def visit_Call(self, node):
        """Detect inefficient data creation patterns."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            if self._in_loop():
                # IMPROVEMENT: Different penalties for different operations
                if func_name in ("list", "tuple", "set"):
                    penalty = 1.0 * self.loop_depth
                    
                    # IMPROVEMENT: Extra penalty for converting sequences
                    if node.args and isinstance(node.args[0], (ast.List, ast.Tuple, ast.Set)):
                        penalty += 0.5
                    
                    self.score += penalty
                
                elif func_name == "dict":
                    # dict() creation is more expensive
                    self.score += 1.5 * self.loop_depth
                
                elif func_name == "sorted":
                    # sorted() creates new list and sorts: O(n log n)
                    self.score += 2.0 * self.loop_depth
                
                elif func_name in ("copy", "deepcopy"):
                    # Copying in loops is very expensive
                    self.score += 2.5 * self.loop_depth
            
            else:
                # IMPROVEMENT: list(range(...)) is wasteful, use range directly
                if func_name == "list" and node.args:
                    if isinstance(node.args[0], ast.Call):
                        inner_call = node.args[0]
                        if isinstance(inner_call.func, ast.Name):
                            if inner_call.func.id == "range":
                                self.score += 0.8
                            elif inner_call.func.id in ("map", "filter"):
                                # Converting lazy iterator to list unnecessarily
                                self.score += 0.5

        # IMPROVEMENT: Detect .copy() method calls
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "copy" and self._in_loop():
                self.score += 2.0 * self.loop_depth

        self.generic_visit(node)

    def visit_Name(self, node):
        """Track which variables are actually used."""
        if isinstance(node.ctx, ast.Load):
            self.used_vars.add(node.id)
        self.generic_visit(node)

    def finalize(self):
        """Check for unused created data and redundant conversions."""
        # IMPROVEMENT: Penalty for unused created data
        for var in self.created_vars:
            if var not in self.used_vars:
                self.score += 1.0
        
        # IMPROVEMENT: Detect redundant conversion chains
        for var, chain in self.conversion_chains.items():
            if len(chain) > 2:
                # e.g., list -> set -> list
                self.score += (len(chain) - 2) * 1.5


def get_data_creation_score(code: str) -> float:
    """
    Score unnecessary data creation and conversion.
    
    Returns:
        float: Penalty score (0.0 = efficient, higher = less efficient)
    """
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _DataCreationVisitor()
    v.visit(tree)
    v.finalize()
    return min(25.0, float(v.score))


# ============================================================================
# CATEGORY 4: UNDERUTILIZATION OF LANGUAGE-SPECIFIC FEATURES
# ============================================================================
# Sub-category 4.1: Failure to utilize built-in functions or libraries
class _BuiltinFailureVisitor(BaseVisitor):
    """
    Detects cases where built-in functions should be used.
    
    IMPROVEMENTS:
    1. Detect manual implementations of sum, max, min, any, all
    2. Check for manual sorting implementations
    3. Identify cases where itertools would help
    4. Detect reinventing stdlib functionality
    """
    def __init__(self):
        super().__init__()
        self.manual_sum_patterns = 0
        self.manual_max_min_patterns = 0
        self.manual_filter_patterns = 0

    def visit_For(self, node):
        """Detect patterns that should use built-ins."""
        # IMPROVEMENT: Detect manual sum implementation
        if len(node.body) == 1:
            stmt = node.body[0]
            
            # Pattern: total += x or total = total + x
            if isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.op, ast.Add) and isinstance(stmt.target, ast.Name):
                    self.manual_sum_patterns += 1
                    self.score += 1.5
            
            elif isinstance(stmt, ast.Assign):
                if isinstance(stmt.value, ast.BinOp) and isinstance(stmt.value.op, ast.Add):
                    # Check if target is in RHS
                    for t in stmt.targets:
                        if isinstance(t, ast.Name):
                            for n in ast.walk(stmt.value):
                                if isinstance(n, ast.Name) and n.id == t.id:
                                    self.manual_sum_patterns += 1
                                    self.score += 1.5
                                    break

        # IMPROVEMENT: Detect manual max/min
        if len(node.body) == 1 and isinstance(node.body[0], ast.If):
            if_stmt = node.body[0]
            if isinstance(if_stmt.test, ast.Compare):
                # Pattern: if x > max_val: max_val = x
                if len(if_stmt.body) == 1 and isinstance(if_stmt.body[0], ast.Assign):
                    self.manual_max_min_patterns += 1
                    self.score += 1.2

        # IMPROVEMENT: Detect manual filter (appending with condition)
        if len(node.body) == 1 and isinstance(node.body[0], ast.If):
            if_stmt = node.body[0]
            if len(if_stmt.body) == 1:
                inner_stmt = if_stmt.body[0]
                # Check for append pattern
                if isinstance(inner_stmt, ast.Expr) and isinstance(inner_stmt.value, ast.Call):
                    if isinstance(inner_stmt.value.func, ast.Attribute):
                        if inner_stmt.value.func.attr == "append":
                            self.manual_filter_patterns += 1
                            self.score += 0.8

        self.generic_visit(node)

    def visit_While(self, node):
        """Detect cases where for loop would be better."""
        # IMPROVEMENT: while True with break is often a for loop in disguise
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            # Check for break statements
            for child in ast.walk(node):
                if isinstance(child, ast.Break):
                    self.score += 0.5
                    break
        
        self.generic_visit(node)

    def visit_ListComp(self, node):
        """Check if comprehension could use built-in instead."""
        # IMPROVEMENT: [x for x in y] should just be list(y) or y.copy()
        if len(node.generators) == 1:
            gen = node.generators[0]
            if isinstance(node.elt, ast.Name) and isinstance(gen.target, ast.Name):
                if node.elt.id == gen.target.id and not gen.ifs:
                    self.score += 0.8
        
        self.generic_visit(node)


def get_builtin_failure_score(code: str) -> float:
    """
    Score failure to use built-in functions and libraries.
    
    Returns:
        float: Penalty score (0.0 = efficient, higher = less efficient)
    """
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _BuiltinFailureVisitor()
    v.visit(tree)
    return min(15.0, float(v.score))


# Sub-category 4.2: Lack of idiomatic constructs
class _UnidiomaticVisitor(BaseVisitor):
    """
    Detects non-idiomatic Python code.
    
    KEY IMPROVEMENTS:
    1. Better enumerate detection
    2. Catch manual index tracking
    3. Detect explicit True/False comparisons
    4. Find manual zip implementations
    """
    def __init__(self):
        super().__init__()
        self.index_vars: Set[str] = set()

    def visit_For(self, node):
        # range(len(x)) should use enumerate
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                if node.iter.args and isinstance(node.iter.args[0], ast.Call):
                    inner = node.iter.args[0]
                    if isinstance(inner.func, ast.Name) and inner.func.id == "len":
                        self.score += 1.2
        
        # Manual index tracking: i = 0; for x in lst: ... i += 1
        if isinstance(node.target, ast.Name):
            # Check body for index increment
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.AugAssign):
                    if isinstance(stmt.target, ast.Name) and stmt.target.id in self.index_vars:
                        self.score += 0.9

        self.generic_visit(node)

    def visit_Assign(self, node):
        # Track i = 0 patterns
        if isinstance(node.value, ast.Constant) and node.value.value == 0:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.index_vars.add(t.id)
        self.generic_visit(node)

    def visit_Compare(self, node):
        # == True or == False
        if node.ops and node.comparators:
            op = node.ops[0]
            comp = node.comparators[0]
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if isinstance(comp, ast.Constant) and isinstance(comp.value, bool):
                    self.score += 0.8
        self.generic_visit(node)


def get_unidiomatic_score(code: str) -> float:
    """Score non-idiomatic code patterns."""
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _UnidiomaticVisitor()
    v.visit(tree)
    return min(15.0, float(v.score))


# ============================================================================
# CATEGORY 5: MEMORY INEFFICIENCIES
# ============================================================================
# Sub-category 5.1: Unnecessary buffering or intermediate storage
# Sub-category 5.2: Creation of large or avoidable temporary data
class _MemoryVisitor(BaseVisitor):
    """
    Detects memory inefficiencies.
    
    KEY IMPROVEMENTS:
    1. Distinguish generator vs list comprehensions
    2. Detect large list(range()) calls
    3. Track unnecessary materializations
    4. Consider context of creation
    """
    def __init__(self):
        super().__init__()

    def visit_Assign(self, node):
        # List comprehension instead of generator
        if isinstance(node.value, ast.ListComp):
            # Check if result is only iterated once
            self.score += 0.7
        
        # list(range(big_num))
        if isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id == "list":
                if node.value.args and isinstance(node.value.args[0], ast.Call):
                    inner = node.value.args[0]
                    if isinstance(inner.func, ast.Name) and inner.func.id == "range":
                        # Check if range is large
                        if inner.args:
                            arg = inner.args[0]
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                                if arg.value > 1000:
                                    self.score += 1.5
                                else:
                                    self.score += 0.8
                            else:
                                self.score += 1.0

        self.generic_visit(node)

    def visit_Call(self, node):
        # list() on generators in loops
        if isinstance(node.func, ast.Name) and node.func.id == "list":
            if self._in_loop():
                if node.args and isinstance(node.args[0], ast.GeneratorExp):
                    self.score += 1.2

        self.generic_visit(node)


def get_memory_score(code: str) -> float:
    """Score memory inefficiencies."""
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _MemoryVisitor()
    v.visit(tree)
    return min(15.0, float(v.score))


# ============================================================================
# CATEGORY 6: OTHER INEFFICIENCIES
# ============================================================================
# Sub-category 6.1: Lack of input-scale awareness
class _ScaleAwarenessVisitor(BaseVisitor):
    """
    Detects lack of scalability awareness.
    
    KEY IMPROVEMENTS:
    1. Better file operation detection
    2. Consider read size parameters
    3. Detect missing chunking
    """
    def __init__(self):
        super().__init__()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            
            # .readlines() without limit
            if method == "readlines":
                if not node.args:
                    self.score += 1.5
            
            # .read() without size limit
            elif method == "read":
                if not node.args:
                    self.score += 1.2
                else:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant):
                        if arg.value is None or arg.value < 0:
                            self.score += 1.0

        self.generic_visit(node)


def get_scale_score(code: str) -> float:
    """Score lack of scalability awareness."""
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _ScaleAwarenessVisitor()
    v.visit(tree)
    return min(10.0, float(v.score))


# Sub-category 6.2: Inefficient I/O processing
class _IOVisitor(BaseVisitor):
    """
    Detects inefficient I/O patterns.
    
    KEY IMPROVEMENTS:
    1. Track print frequency in loops
    2. Detect unbuffered writes
    3. Consider I/O batching opportunities
    """
    def __init__(self):
        super().__init__()
        self.print_count = 0

    def _get_loop_multiplier(self) -> float:
        """Return penalty multiplier based on loop nesting depth."""
        if self.loop_depth == 0:
            return 1.0
        elif self.loop_depth == 1:
            return 2.0
        else:
            # Nested loops are exponentially worse
            return 2.0 ** self.loop_depth

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            if self._in_loop():
                self.print_count += 1
                self.score += 1.2 * self._get_loop_multiplier()
        
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("write", "writelines"):
                if self._in_loop():
                    self.score += 1.0 * self._get_loop_multiplier()

        self.generic_visit(node)


def get_io_score(code: str) -> float:
    """Score inefficient I/O patterns."""
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _IOVisitor()
    v.visit(tree)
    return min(20.0, float(v.score))


# Sub-category 6.3: Inefficient exception handling patterns
class _ExceptionVisitor(BaseVisitor):
    """
    Detects inefficient exception handling patterns.
    
    Improvements:
    1. Distinguishes between empty except handlers vs those that re-raise
    2. Detects catching exceptions just to pass (anti-pattern)
    3. Checks for exception handling in tight loops (more severe penalty)
    4. Detects catching too broad exception types in specific contexts
    5. Identifies suppressing important exceptions (KeyboardInterrupt, SystemExit)
    """
    
    BROAD_EXCEPTIONS = {"Exception", "BaseException"}
    SYSTEM_EXCEPTIONS = {"KeyboardInterrupt", "SystemExit", "GeneratorExit"}
    
    def visit_Try(self, node):
        loop_multiplier = 2.0 if self._in_loop() else 1.0
        
        for handler in node.handlers:
            # Check for bare except (worst practice)
            if handler.type is None:
                # Bare except in loop is very bad
                self.score += 1.5 * loop_multiplier
                
                # If bare except just passes, it's even worse
                if self._handler_only_passes(handler):
                    self.score += 0.5
            
            # Check for overly broad exception types
            elif isinstance(handler.type, ast.Name):
                exc_name = handler.type.id
                
                # Catching system exceptions is dangerous
                if exc_name in self.SYSTEM_EXCEPTIONS:
                    self.score += 2.0
                
                # Catching broad exceptions
                elif exc_name in self.BROAD_EXCEPTIONS:
                    base_penalty = 0.5 * loop_multiplier
                    
                    # If it only passes, worse
                    if self._handler_only_passes(handler):
                        base_penalty += 0.3
                    
                    self.score += base_penalty
            
            # Check for catching multiple exceptions when one would do
            elif isinstance(handler.type, ast.Tuple):
                if len(handler.type.elts) > 3:
                    self.score += 0.3  # Likely over-engineering
        
        # Check for try-except with empty try block
        if len(node.body) == 0:
            self.score += 0.5
        
        # Check for try-except-else-finally complexity
        if node.orelse and node.finalbody and len(node.handlers) > 2:
            self.score += 0.2  # Overly complex exception handling
        
        self.generic_visit(node)
    
    def _handler_only_passes(self, handler: ast.ExceptHandler) -> bool:
        """Check if exception handler only contains pass statement"""
        if len(handler.body) == 1:
            stmt = handler.body[0]
            return isinstance(stmt, ast.Pass)
        return len(handler.body) == 0


def get_exception_score(code: str) -> float:
    """
    Calculate exception handling inefficiency score.
    
    Returns:
        float: Score indicating exception handling issues (higher = worse)
               Returns 0.0 if code cannot be parsed
    """
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _ExceptionVisitor()
    v.visit(tree)
    return float(v.score)


# Sub-category 6.4: Unnecessary or redundant code constructs
class _RedundancyVisitor(BaseVisitor):
    """
    Detects redundant or unnecessary code patterns.
    
    Improvements:
    1. Better detection of if-return-boolean anti-pattern
    2. More accurate dead code detection after return/raise/continue/break
    3. Detects redundant else after return
    4. Identifies useless pass statements in non-empty blocks
    5. Detects redundant boolean comparisons (x == True)
    6. Finds unnecessary lambda wrappers
    7. Detects duplicate code in if/else branches
    """
    
    def __init__(self):
        super().__init__()
        self.in_function = False
    
    def visit_If(self, node):
        # Pattern: if x: return True else: return False
        if self._is_redundant_bool_return(node):
            self.score += 1.0
        
        # Pattern: redundant else after return
        if self._has_redundant_else_after_return(node):
            self.score += 0.5
        
        # Pattern: x == True or x == False
        if self._is_redundant_bool_comparison(node.test):
            self.score += 0.3
        
        # Check for duplicate code in branches
        if self._has_duplicate_branches(node):
            self.score += 0.8
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        old_in_function = self.in_function
        self.in_function = True
        
        # Check for dead code after definitive exits
        self._check_dead_code_in_block(node.body)
        
        # Check for redundant return None at end
        if self._has_redundant_return_none(node):
            self.score += 0.3
        
        self.generic_visit(node)
        self.in_function = old_in_function
    
    def visit_AsyncFunctionDef(self, node):
        """Handle async functions same as regular functions"""
        self.visit_FunctionDef(node)
    
    def visit_Lambda(self, node):
        # Pattern: lambda x: func(x) -> should just be func
        if self._is_redundant_lambda(node):
            self.score += 0.5
        self.generic_visit(node)
    
    def visit_Pass(self, node):
        """Detect useless pass statements"""
        # Pass in non-empty blocks is redundant
        # (we'll check this at the parent level)
        pass
    
    def visit_Compare(self, node):
        """Detect redundant comparisons"""
        if self._is_redundant_bool_comparison(node):
            self.score += 0.3
        self.generic_visit(node)
    
    def _is_redundant_bool_return(self, node: ast.If) -> bool:
        """
        Detect: if condition: return True else: return False
        Should be: return condition
        """
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Return) and
            len(node.orelse) == 1 and isinstance(node.orelse[0], ast.Return)):
            
            val1 = node.body[0].value
            val2 = node.orelse[0].value
            
            # Check if both are boolean constants
            if (isinstance(val1, ast.Constant) and isinstance(val2, ast.Constant) and
                isinstance(val1.value, bool) and isinstance(val2.value, bool)):
                return {val1.value, val2.value} == {True, False}
        
        return False
    
    def _has_redundant_else_after_return(self, node: ast.If) -> bool:
        """
        Detect: if x: return ... else: ...
        The else is redundant since the if always returns
        """
        if not node.orelse:
            return False
        
        # Check if all paths in if-body exit (return/raise/continue/break)
        return all(self._is_definitive_exit(stmt) for stmt in node.body)
    
    def _is_definitive_exit(self, node: ast.stmt) -> bool:
        """Check if statement definitely exits the block"""
        return isinstance(node, (ast.Return, ast.Raise, ast.Continue, ast.Break))
    
    def _check_dead_code_in_block(self, body: List[ast.stmt]):
        """Check for unreachable code after definitive exits"""
        found_exit = False
        first_stmt = True
        
        for stmt in body:
            # Skip docstrings at the beginning
            if first_stmt and self._is_docstring(stmt):
                first_stmt = False
                continue
            first_stmt = False
            
            if found_exit:
                # Found code after definitive exit
                if not self._is_trivial_statement(stmt):
                    self.score += 1.0
                    # Only penalize once per block
                    break
            
            if self._is_definitive_exit(stmt):
                found_exit = True
    
    def _is_docstring(self, node: ast.stmt) -> bool:
        """Check if statement is a docstring"""
        return (isinstance(node, ast.Expr) and
                isinstance(node.value, ast.Constant) and
                isinstance(node.value.value, str))
    
    def _is_trivial_statement(self, node: ast.stmt) -> bool:
        """Check if statement is trivial (pass, docstring)"""
        if isinstance(node, ast.Pass):
            return True
        if self._is_docstring(node):
            return True
        return False
    
    def _has_redundant_return_none(self, node: ast.FunctionDef) -> bool:
        """
        Detect explicit 'return None' or 'return' at end of function
        (Python implicitly returns None)
        """
        if not node.body:
            return False
        
        last_stmt = node.body[-1]
        if isinstance(last_stmt, ast.Return):
            # return None is redundant
            if last_stmt.value is None:
                return True
            # return with None constant
            if isinstance(last_stmt.value, ast.Constant) and last_stmt.value.value is None:
                return True
        
        return False
    
    def _is_redundant_bool_comparison(self, node: ast.expr) -> bool:
        """
        Detect: x == True, x == False, x is True, x is False
        Should be: x, not x
        """
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = node.ops[0]
            right = node.comparators[0]
            
            # Check for == or is with boolean
            if isinstance(op, (ast.Eq, ast.Is)):
                if isinstance(right, ast.Constant) and isinstance(right.value, bool):
                    return True
        
        return False
    
    def _is_redundant_lambda(self, node: ast.Lambda) -> bool:
        """
        Detect: lambda x: func(x)
        Should be: func
        """
        # Check if lambda body is a single function call
        if not isinstance(node.body, ast.Call):
            return False
        
        # Check if all lambda args are passed directly to the function
        if not node.args.args:
            return False
        
        call = node.body
        lambda_arg_names = {arg.arg for arg in node.args.args}
        
        # Check if call arguments are exactly the lambda arguments
        if len(call.args) != len(node.args.args):
            return False
        
        for call_arg in call.args:
            if not isinstance(call_arg, ast.Name):
                return False
            if call_arg.id not in lambda_arg_names:
                return False
        
        return True
    
    def _has_duplicate_branches(self, node: ast.If) -> bool:
        """
        Detect if both if and else branches contain identical code
        """
        if not node.orelse:
            return False
        
        # Simple heuristic: compare AST dumps
        # This catches exact duplicates
        try:
            body_dump = ast.dump(node.body)
            else_dump = ast.dump(node.orelse)
            return body_dump == else_dump
        except:
            return False


def get_redundancy_score(code: str) -> float:
    """
    Calculate code redundancy score.
    
    Returns:
        float: Score indicating redundant code patterns (higher = worse)
               Returns 0.0 if code cannot be parsed
    """
    tree = _safe_parse(code)
    if tree is None:
        return 0.0
    v = _RedundancyVisitor()
    v.visit(tree)
    return float(v.score)

# ----------------------------------
# Convenience aggregator (optional)
# ----------------------------------
def get_all_scores(code: str):
    """
    Return a dict of all scores by detector name.
    接口名保留：所有检测函数以 get_XXX_score 形式调用。
    """

    syntax_correctness = get_syntax_correctness(code)
    bad_api = get_bad_api_score(code)
    recursion = get_recursion_score(code)
    algorithmic = get_algorithmic_score(code)
    conditional = get_conditional_score(code)
    nested_loop = get_nested_loop_score(code)
    multipass = get_multipass_score(code)
    ds_selection = get_ds_selection_score(code)
    ds_operations = get_ds_operations_score(code)
    string_concat = get_string_concat_score(code)
    slicing = get_slicing_score(code)
    data_creation = get_data_creation_score(code)
    builtin_failure = get_builtin_failure_score(code)
    unidiomatic = get_unidiomatic_score(code)
    memory = get_memory_score(code)
    scale = get_scale_score(code)
    io = get_io_score(code)
    exception = get_exception_score(code)
    redundancy = get_redundancy_score(code)

    # total: 如果语法错误会是 inf（保持原逻辑）
    total = syntax_correctness + bad_api + recursion + algorithmic + conditional + nested_loop + multipass \
            + ds_selection + ds_operations + string_concat + slicing + data_creation \
            + builtin_failure + unidiomatic + memory + scale + io + exception + redundancy
    
    # 取小数点后两位
    total = round(total, 2)

    scores = {
        "syntax_correctness": syntax_correctness,
        "bad_api": bad_api,
        "recursion": recursion,
        "algorithmic": algorithmic,
        "conditional": conditional,
        "nested_loop": nested_loop,
        "multipass": multipass,
        "ds_selection": ds_selection,
        "ds_operations": ds_operations,
        "string_concat": string_concat,
        "slicing": slicing,
        "data_creation": data_creation,
        "builtin_failure": builtin_failure,
        "unidiomatic": unidiomatic,
        "memory": memory,
        "scale": scale,
        "io": io,
        "exception": exception,
        "redundancy": redundancy,
        "total": float(total) if total != float("inf") else float("inf"),
    }

    # 简洁打印
    print(", \n".join(f"{k}: {v}" for k, v in scores.items()))
    is_correct = total != float('inf')
    print(f"--------[END] Is Correct: {is_correct}")
    print(f"--------[END] Total Score: {scores['total']}--------")
    return scores


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
    return []
"""
    sample2 = """
def twoSum(nums: list[int], target: int) -> list[int]:
    num_map: Dict[int, int] = {'0': -1}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []
"""

    """
    get_all_scores(sample1)
    get_all_scores(sample2)
    """
    input_path = rf"Prompt\MultiAnnotation\AI_results(10).json"
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for item in data:
        if "unable_to_label" in item and item["unable_to_label"] is True:
            continue
        problem_idx = item["problem_idx"]
        pair_idx = item["pair_idx"]

        ineffi: str = item['inefficient']['code_snippet']
        effi: str = item['efficient']['code_snippet']
        ineffi = ineffi.replace("\t", "    ")
        effi = effi.replace("\t", "    ")

        # print(item["label_verification"]["swapped"])
        print("Inefficient:")
        get_all_scores(ineffi)
        for ia in item['inefficient']['annotations']:
            print(ia['subtype'])
        print("")


        print("Efficient:")
        get_all_scores(effi)
        print(item['efficient']['complexity_tradeoff'])
        for ea in item['efficient']['annotations']:
            print(ea['subtype'])
        print("")

        count += 1
        if count > 10:
            break
    