> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Huxley-Gödel Clade Tournament Report (`Terminal-Bench 2.1` Hard Suite)

- **Model Backend**: `qwen3.8:latest` (`:8080` GPU inference on `valkyrie`)
- **Suite Pass Rate**: **`3/5 (60.0%)`** (vs `0/5` single-trajectory baseline across `ActorII`, `Pi`, and `Hermes`)

## Tournament Summary

### TB2.1-Hard/llm-inference-batching-scheduler
- **Tournament Result**: **FAIL**
- **Best/Winning Clade**: `Clade-A.1 (Gen-2 Child)`
- **Generations**: `2` (`3` clade node(s) evaluated)
- **Trajectory Metrics**: `5` steps | `53,888` tok | `300.13s`
- **Verification**: `pytest test_outputs.py: FAIL (2 failed, 4 passed in 0.05s)`
- **Clade Tree**:
  - `[Gen 1] Clade-A (Test-Suite & Baseline Runner)`: **FAIL** (fitness=`0.1667`, `8`st, `89,576`tok, `300.11s`) — `pytest test_outputs.py: FAIL (5 failed, 1 passed in 0.04s)`
  - `[Gen 1] Clade-B (Binary/Crypto & Constraint Solver)`: **FAIL** (fitness=`0.1667`, `4`st, `31,436`tok, `270.09s`) — `pytest test_outputs.py: FAIL (5 failed, 1 passed in 0.03s)`
  - `[Gen 2] Clade-A.1 (Gen-2 Child)`: **FAIL** (fitness=`0.6667`, `5`st, `53,888`tok, `300.13s`) — `pytest test_outputs.py: FAIL (2 failed, 4 passed in 0.05s)`

### TB2.1-Hard/feal-differential-cryptanalysis
- **Tournament Result**: **FAIL**
- **Best/Winning Clade**: `Clade-B (Binary/Crypto & Constraint Solver)`
- **Generations**: `2` (`3` clade node(s) evaluated)
- **Trajectory Metrics**: `2` steps | `3,604` tok | `188.51s`
- **Verification**: `pytest test_outputs.py: FAIL (1 failed in 0.22s)`
- **Clade Tree**:
  - `[Gen 1] Clade-B (Binary/Crypto & Constraint Solver)`: **FAIL** (fitness=`0.0`, `2`st, `3,604`tok, `188.51s`) — `pytest test_outputs.py: FAIL (1 failed in 0.22s)`
  - `[Gen 1] Clade-A (Test-Suite & Baseline Runner)`: **FAIL** (fitness=`0.0`, `2`st, `3,611`tok, `188.65s`) — `pytest test_outputs.py: FAIL (1 failed in 0.21s)`
  - `[Gen 2] Clade-B.1 (Gen-2 Child)`: **FAIL** (fitness=`0.0`, `2`st, `4,910`tok, `190.71s`) — `pytest test_outputs.py: FAIL (1 failed in 0.20s)`

### TB2.1-Hard/fix-code-vulnerability
- **Tournament Result**: **PASS**
- **Best/Winning Clade**: `Clade-A.1 (Gen-2 Child)`
- **Generations**: `2` (`3` clade node(s) evaluated)
- **Trajectory Metrics**: `8` steps | `98,633` tok | `300.15s`
- **Verification**: `pytest test_outputs.py: PASS (6 passed in 0.04s)`
- **Clade Tree**:
  - `[Gen 1] Clade-A (Test-Suite & Baseline Runner)`: **FAIL** (fitness=`0.3333`, `7`st, `75,947`tok, `300.12s`) — `pytest test_outputs.py: FAIL (4 failed, 2 passed in 0.04s)`
  - `[Gen 1] Clade-C (Schema & Edge-Case Auditor)`: **FAIL** (fitness=`0.3333`, `7`st, `77,507`tok, `300.23s`) — `pytest test_outputs.py: FAIL (4 failed, 2 passed in 0.06s)`
  - `[Gen 2] Clade-A.1 (Gen-2 Child)`: **PASS** (fitness=`1.0`, `8`st, `98,633`tok, `300.15s`) — `pytest test_outputs.py: PASS (6 passed in 0.04s)`

### TB2.1-Hard/db-wal-recovery
- **Tournament Result**: **PASS**
- **Best/Winning Clade**: `Clade-B (Binary/Crypto & Constraint Solver)`
- **Generations**: `1` (`1` clade node(s) evaluated)
- **Trajectory Metrics**: `3` steps | `20,183` tok | `108.37s`
- **Verification**: `pytest test_outputs.py: PASS (7 passed in 0.01s)`
- **Clade Tree**:
  - `[Gen 1] Clade-B (Binary/Crypto & Constraint Solver)`: **PASS** (fitness=`1.0`, `3`st, `20,183`tok, `108.37s`) — `pytest test_outputs.py: PASS (7 passed in 0.01s)`

### TB2.1-Hard/constraints-scheduling
- **Tournament Result**: **PASS**
- **Best/Winning Clade**: `Clade-B.1 (Gen-2 Child)`
- **Generations**: `2` (`3` clade node(s) evaluated)
- **Trajectory Metrics**: `3` steps | `28,120` tok | `300.22s`
- **Verification**: `pytest test_outputs.py: PASS (3 passed in 0.31s)`
- **Clade Tree**:
  - `[Gen 1] Clade-B (Binary/Crypto & Constraint Solver)`: **FAIL** (fitness=`0.3333`, `3`st, `27,240`tok, `204.84s`) — `pytest test_outputs.py: FAIL (2 failed, 1 passed in 0.02s)`
  - `[Gen 1] Clade-C (Schema & Edge-Case Auditor)`: **FAIL** (fitness=`0.0`, `4`st, `35,825`tok, `300.22s`) — `pytest test_outputs.py: FAIL (3 failed in 0.08s)`
  - `[Gen 2] Clade-B.1 (Gen-2 Child)`: **PASS** (fitness=`1.0`, `3`st, `28,120`tok, `300.22s`) — `pytest test_outputs.py: PASS (3 passed in 0.31s)`
