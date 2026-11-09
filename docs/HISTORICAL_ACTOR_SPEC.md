# Historical 1993 Actor Professional 4.1 Specification & Ground Truth

> [!NOTE]
> This document details the authentic syntax, object model, and runtime semantics of The Whitewater Group's Actor Professional 4.1 (1993), extracted directly from the original installation archives.

---

## 1. Core Language Grammar & Syntax

Actor combines Smalltalk object-oriented semantics with a C/Pascal-like statement structure.

### 1.1 Class Definitions (`inherit`)
Classes are defined dynamically via the `inherit` function:
```actor
inherit(SuperClass, #SubClassName, #(instVar1 instVar2 ...), classType, nil)!!
```
- `SuperClass`: Name of the parent class (single inheritance root is `Object`).
- `#SubClassName`: Symbol representing the new class name.
- `#(instVar1 instVar2 ...)`: Array of symbols defining instance variables.
- `classType`: Integer flag (typically `2` for standard pointer-based classes).
- `!!`: Top-level statement delimiter (evaluates immediately in the workspace/compiler).

### 1.2 Method Declarations (`Def`)
Methods are defined using `Def`:
```actor
Def methodName(self, arg1, arg2 | local1 local2)
{
    local1 := arg1 + arg2;
    doSomething(self, local1);
    ^local1;
}!!
```
- First parameter is always `self` (the receiver).
- Parameters followed by `|` declare local variables for the method frame.
- `^`: Return operator (returns value to the caller, similar to Smalltalk).
- `:=`: Assignment operator.
- `!!`: Method termination delimiter.

### 1.3 Event / Message Handlers (`Action`)
Actor binds UI events and Windows messages to methods using `Action` with selector maps:
```actor
Action onClick(self, msg) #[#okButton->#selected #menuItem->#clicked]
{
    if condition
    then handleSuccess(self);
    else handleFallback(self);
    endif;
}!!
```

### 1.4 Literal Syntax
- **Symbols**: `#symbolName` (e.g. `#ok`, `#selected`, `#dialog`).
- **Literal Arrays**: `#(#item1 #item2 "string" 123)`.
- **Comments**: C-style `/* comment */`.
- **Conditionals**:
  ```actor
  if cond then ... else ... endif;
  ```
- **Loops**:
  ```actor
  while cond do ... endwhile;
  do ... until cond;
  ```

---

## 2. Directory Layout of Extracted Source (`historical_actor_41_source/`)

| Directory | Count | Contents & Significance |
| :--- | :--- | :--- |
| `actcls/` | 195 | Core class hierarchy (`OBJECT.CLS`, `WINDOW.CLS`, `COLLECT.CLS`, `LIST.CLS`, `DIALOG.CLS`, `BUTTON.CLS`). |
| `actact/` | 23 | Action/algorithm suites (`FRACTAL.ACT`, `SIEVE.ACT`, `LIST.ACT`, `QUEEN1.ACT`, `DEEPCOPY.ACT`). Ideal for unit tests! |
| `actres/` | 68 | Window resource layouts, dialog templates, and menu definitions. |
| `drawcls/` | 18 | *ObjectDraw* vector drawing classes (primitives, canvas, selection handles). |
| `oglcls/` | 32 | *ObjectGraphics* library (2D/3D transformations, pen/brush contexts). |
| `sqlcls/` | 8 | Actor SQL database connectivity layer. |
| `wrt/` | 12 | Whitewater Resource Toolkit integration. |

---

## 3. How to Bridge Historical Actor to Modern ActorII

1. **Parser Target**:
   `actorii.parser` can support either:
   - Modern simplified Actor (Python-like `class Foo: def bar():`)
   - Authentic 1993 Actor syntax (`inherit()`, `Def ...(self | locals)`, `^return`, `:=`, `#symbols`, `!!`).
2. **Benchmark Test Suite**:
   Use `historical_actor_41_source/actact/SIEVE.ACT` and `FRACTAL.ACT` as authentic integration test benchmarks.
