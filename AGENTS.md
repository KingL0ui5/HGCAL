# Agent instructions

## Environment

- Use `uv` for dependency management (`uv add <package>`, `uv sync`). Don't edit `pyproject.toml`/`uv.lock` by hand.
- Run Python through the project venv: `.venv/bin/python`, not a bare `python`/`python3`.

## Verification

Before considering a change to `.py` files complete, run:

```
.venv/bin/ruff check <files>
.venv/bin/ty check <files>
```

## Typing

- Use type hints on function/method signatures (parameters and return types).
- Use strong/explicit typecasts wherever a value's type is ambiguous or inferred loosely
  (e.g. `int(x)`, `np.ndarray.astype(...)`/`to_numpy(dtype=...)`, explicit `torch.dtype` on
  tensor construction) rather than relying on implicit coercion. This keeps `ruff` and `ty`
  checks meaningful and catches type mismatches early.

## Code style

- Always include a blank line before control statements (`if`, `for`, `while`, `try`,
  `return`, etc.) unless it is the first line in its block.
- Use good, clear naming conventions (variables, functions, classes) so the code reads as
  its own documentation. Do not add comments unless a line is inherently complex to read
  (e.g. a long/non-obvious calculation) and cannot be made clear through better naming or
  OOP structure instead.
