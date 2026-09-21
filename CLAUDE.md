# CLAUDE.md

Guidance for working in this repo.

## Run it

```bash
uv sync
uv run main.py          # launches the GUI
uv run pytest           # engine + lifecycle test suite
```

`bootstrap-gui` is also registered as a `uv run` script entry point
(`uv run bootstrap-gui`), equivalent to `uv run main.py`.

## Where things live

- `engine/` — pure Python/numpy. No Tkinter, no `print`, no interactive I/O.
  Importable from notebooks, CLI scripts, or the GUI unchanged. This is
  where all numerical logic lives, including `engine/lifecycle.py` (the
  Life Strategy Simulator's math — see its module docstring for the exact
  formulas: monthly block bootstrap, average-cost capital-gains tax,
  inflation indexing) and `engine/plotprep.py` (result dicts → numeric
  matrices, plus the density-preserving thinning the Space Explorer's
  scatter needs to stay interactive at 100k points).
- `bootstrap_gui/` — shared UI infrastructure: `theme.py` (design tokens +
  `ttk.Style`), `widgets.py` (themed widgets), `runner_mixin.py`
  (`BackgroundJobMixin` — the thread/queue/poll pattern, used by every
  long-running Run button), `library.py` (JSON-backed portfolio/plan
  persistence, atomic writes), `assets.py` (the only place the GUI touches
  `engine.data` directly), `logsetup.py`.
  `bootstrap_gui/sections/lifecycle.py` is the Life Strategy panel — the
  only section built entirely in the new package. The three original
  sections (Portfolio Builder, Space Explorer, Single Bootstrap) still live
  in the top-level `gui.py`; they import their shared building blocks from
  `bootstrap_gui` rather than defining local copies (which is what used to
  cause tk/ttk visual inconsistency).
- `gui.py` — the three legacy sections + `BootstrapApp` (the window shell,
  nav, status bar, lazy per-tab construction).
- `tests/` — one file per `engine/` module. `test_lifecycle.py` is the
  most important: every test there is a closed-form check with a hand-computed
  expected value, not just a "does it run" smoke test.

## Conventions

- **The engine never imports Tkinter or matplotlib pyplot state.** UI code
  contains no numerical logic — every computation belongs in `engine/`.
- **Vectorise over `n_sim`, always.** A Python loop over months is fine
  (bounded, small); a Python loop over simulations is not.
- **Background jobs use `BackgroundJobMixin`** (`bootstrap_gui/runner_mixin.py`),
  not hand-rolled `threading.Thread` + `queue.Queue` + `self.after()` —
  that pattern used to be duplicated three times with the same bugs (Run
  button not reliably re-enabled on error).
- **No blocking dialogs from a background-triggered callback.**
  `simpledialog.askstring()` from an `after()` poll used to freeze the
  mainloop; the click-to-library flow in Space Explorer is now a
  non-modal "pending selections" panel instead.
- **`Figure()` explicit, never `plt.subplots()`**, in any code that runs
  more than once (Run buttons) — `plt.subplots()` registers with pyplot's
  global state and never gets garbage collected, which used to leak memory
  on every re-run.
- **Widgets**: prefer `ttk` over `tk` for anything interactive (buttons,
  entries, labels that need themed foreground/background, checkbuttons,
  radiobuttons). Plain `tk.Frame` is fine as a background container — it
  takes `bg=` directly and there's no meaningful visual difference from
  `ttk.Frame` for an unstyled rectangle. `tk.Listbox` / `tk.Text` /
  `tk.Canvas` have no `ttk` equivalent; style them explicitly with the
  tokens from `bootstrap_gui.theme`.
- **Nothing per-point in a Plotly figure is a Python string.** The Space
  Explorer routinely plots 100k portfolios; a pre-rendered hover string
  and a `{ticker: weight}` dict per point is what used to write a 137 MB
  `results/scatter.html`. Use one `hovertemplate` plus a numeric
  `customdata` matrix (plotly base64-encodes numpy arrays), and thin the
  cloud with `engine.plotprep.thin_scatter` before it leaves Python.
- **`sim_seed` cancels relative noise, not absolute.** Every candidate in
  a multi-bootstrap run shares one `sim_seed` (common random numbers) so
  score DIFFERENCES between portfolios reflect weights, not sampling —
  see `run_multi_streaming`'s docstring in `engine/runner.py`. But the
  whole cloud still rides on that one shared draw, and moves by up to
  ~0.7pp (p50) / ~2pp (p1) at `n_sim=1000` depending on which `sim_seed`
  was used. Never compare a cloud/overlay computed with one `sim_seed`
  against one computed with another — an overlay point especially must be
  computed against the exact same data + `sim_seed` as the cloud it's
  drawn on, see `bootstrap_gui.assets.overlay_key`.
- **Evolutionary search keeps one Pool alive across all generations.**
  `run_evolutionary_streaming` (`engine/runner.py`) opens the
  multiprocessing `Pool` once, before the generation loop, not once per
  generation — spawning workers is slow (measured: several seconds for a
  ~20-worker Pool), which would dominate total run time otherwise. This
  also means `sim_seed` is fixed for the entire run by construction (the
  same `_init_worker` params dict is reused every generation), which is
  required, not incidental: comparing a later generation's candidate
  against an earlier one only means anything if they share a draw — same
  rule as `sim_seed` above, just across generations instead of across a
  single batch.
- **The Space Explorer chart is a shell page, not a written-out figure.**
  `results/scatter.html` fetches `figure.json` and `Plotly.react`s it,
  polling the click server's `/version`; a regenerated chart therefore
  lands in the already-open tab (zoom preserved via `uirevision`) instead
  of needing a manual reload. `fig.write_html` is only the fallback for
  when the local click server couldn't bind a port.
- Numbers formatted for the UI go through `bootstrap_gui/fmt.py`
  (`money`, `pct`, `ratio`) — don't hand-roll `f"{x:.1%}"` in a new call site.

## Known scope boundaries (deliberate, not oversights)

- `PortfolioBuilderSection`, `SpaceExplorerSection`, and `SingleBootstrapSection`
  still live inside `gui.py` rather than in `bootstrap_gui/sections/`.
  Splitting ~2000 lines of working, load-bearing UI code across files for
  organisation's sake, with no behavioural change, was judged higher risk
  than payoff given everything else in scope; they already consume the
  shared `bootstrap_gui` infrastructure, which was the part that actually
  fixes visual inconsistency and duplicated bugs.
- Loss carry-forward against future capital gains is not modelled in the
  Life Strategy Simulator (`engine.lifecycle.LOSS_CARRYFORWARD_SUPPORTED = False`).
