# Persistent user preferences

- When creating any new training or result directory under `output/` or
  `outputs/`, include the Beijing-time date and hour in its name (at least
  `YYYY-MM-DD_HH`); never use a date-only directory name.
- GPU workloads may use only physical GPU indices 0 and 2 (`cuda:0` or
  `cuda:2`). Never launch work on GPU indices 1, 3, 4, 5, or any other
  device unless the user explicitly changes this preference.
