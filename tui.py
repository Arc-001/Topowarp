"""Interactive TUI wizard for Topowarp dataset generation.

Run with: python tui.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    Select,
    Static,
)

from topowarp.export import export_dataset
from topowarp.generators import (
    archimedean_spirals,
    concentric_annuli,
    disjoint_clusters,
    nd_checkerboard,
)
from topowarp.noise import apply_feature_noise, apply_label_noise
from topowarp.visualizer import render_all


# -- Data structures ----------------------------------------------------------

TOPOLOGIES = {
    "concentric_annuli": {
        "label": "Concentric Annuli",
        "desc": "Ring-shaped clusters with configurable thickness and margin",
        "params": {
            "n_rings": {"default": "3", "placeholder": "2-10", "help": "Number of concentric rings (classes)"},
            "thickness": {"default": "0.3", "placeholder": "0.1-1.0", "help": "Radial width of each ring"},
            "margin": {"default": "0.5", "placeholder": "0.1-2.0", "help": "Gap between adjacent rings"},
        },
    },
    "archimedean_spirals": {
        "label": "Archimedean Spirals",
        "desc": "Interleaved spiral arms winding outward from the origin",
        "params": {
            "n_arms": {"default": "2", "placeholder": "2-5", "help": "Number of spiral arms (classes)"},
            "turns": {"default": "2.0", "placeholder": "0.5-5.0", "help": "Number of full rotations per arm"},
            "margin": {"default": "0.3", "placeholder": "0.1-1.0", "help": "Radial jitter controlling arm width"},
        },
    },
    "nd_checkerboard": {
        "label": "N-D Checkerboard",
        "desc": "Binary checkerboard pattern in arbitrary dimensions",
        "params": {
            "freq": {"default": "2", "placeholder": "1-5", "help": "Number of cells per dimension"},
        },
    },
    "disjoint_clusters": {
        "label": "Disjoint Clusters",
        "desc": "Isotropic Gaussian blobs with controlled separation",
        "params": {
            "k": {"default": "4", "placeholder": "2-10", "help": "Number of clusters (classes)"},
            "separation": {"default": "5.0", "placeholder": "1.0-20.0", "help": "Distance from origin to centroids"},
        },
    },
}

NOISE_DISTRIBUTIONS = [("None", "none"), ("Gaussian", "gaussian"), ("Uniform", "uniform"), ("Laplacian", "laplacian"), ("Exponential (mean-centered)", "exponential"), ("Erlang k=2 (mean-centered)", "erlang")]
NOISE_SCALE_PRESETS = ["0.0", "0.1", "0.5", "1.5"]
SPARSITY_OPTIONS = [("100%", "1.0"), ("50%", "0.5"), ("10%", "0.1")]
FLIP_PROB_OPTIONS = [("0%", "0.0"), ("5%", "0.05"), ("15%", "0.15"), ("30%", "0.3")]
TARGETING_OPTIONS = [("Uniform", "uniform"), ("Boundary", "boundary")]

SWEEP_PARAMS = {
    "noise_scale": {"label": "Noise Scale", "values": [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5]},
    "flip_prob": {"label": "Flip Probability", "values": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]},
    "sparsity": {"label": "Feature Sparsity", "values": [0.1, 0.25, 0.5, 0.75, 1.0]},
}


# -- Help modal ---------------------------------------------------------------


class HelpModal(ModalScreen[None]):
    """Shows contextual help text."""

    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self) -> ComposeResult:
        with Vertical(id="help-modal"):
            yield Label(self._title, id="help-title")
            yield Static(self._body, id="help-body")
            yield Button("Close", id="help-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss()


# -- Screen 1: Topology selection ---------------------------------------------


class TopologyScreen(Screen):
    """Select manifold topology and optionally enable sweep mode."""

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("question_mark", "show_help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="topology-container"):
            yield Label("Select Topology", id="screen-title")
            yield RadioSet(
                *[RadioButton(f"{v['label']} -- {v['desc']}", id=k) for k, v in TOPOLOGIES.items()],
                id="topology-radio",
            )
            yield Checkbox("Enable OFAT sweep mode", id="sweep-toggle")
            with Horizontal(id="nav-buttons"):
                yield Button("Next", variant="primary", id="next-btn")
                yield Button("Quit", variant="error", id="quit-btn")
        yield Footer()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_show_help(self) -> None:
        self.app.push_screen(HelpModal(
            "Topology Selection",
            "Choose the geometric structure of your dataset.\n\n"
            "Each topology produces distinct decision boundaries that "
            "challenge classifiers differently. Annuli test radial separation, "
            "spirals test angular separation, checkerboard tests axis-aligned "
            "partitions, and clusters test centroid-based separation.",
        ))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit-btn":
            self.app.exit()
        elif event.button.id == "next-btn":
            radio_set = self.query_one("#topology-radio", RadioSet)
            if radio_set.pressed_button is None:
                self.notify("Select a topology first", severity="warning")
                return
            topo_key = radio_set.pressed_button.id
            sweep = self.query_one("#sweep-toggle", Checkbox).value
            self.app.tui_state["topology"] = topo_key
            self.app.tui_state["sweep"] = sweep
            self.app.push_screen(ParamsScreen())


# -- Screen 2: Topological parameters ----------------------------------------


class ParamsScreen(Screen):
    """Configure sample size, dimensions, and topology-specific parameters."""

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("escape", "go_back", "Back"),
        Binding("question_mark", "show_help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        topo_key = self.app.tui_state["topology"]
        topo = TOPOLOGIES[topo_key]
        yield Header()
        with VerticalScroll(id="params-container"):
            yield Label(f"Parameters: {topo['label']}", id="screen-title")
            yield Label("Sample count")
            yield Input(value="1000", placeholder="100-50000", id="param-n")
            yield Label("Dimensions")
            yield Input(value="2", placeholder="2-30", id="param-d")
            yield Label("Random seed")
            yield Input(value="42", placeholder="integer", id="param-seed")
            for pname, pinfo in topo["params"].items():
                yield Label(f"{pname}")
                yield Input(value=pinfo["default"], placeholder=pinfo["placeholder"], id=f"param-{pname}")
            with Horizontal(id="nav-buttons"):
                yield Button("Back", id="back-btn")
                if self.app.tui_state.get("sweep"):
                    yield Button("Next: Sweep Config", variant="primary", id="next-btn")
                else:
                    yield Button("Next: Feature Noise", variant="primary", id="next-btn")
        yield Footer()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_show_help(self) -> None:
        topo_key = self.app.tui_state["topology"]
        lines = []
        for pname, pinfo in TOPOLOGIES[topo_key]["params"].items():
            lines.append(f"{pname}: {pinfo['help']} (range: {pinfo['placeholder']})")
        self.app.push_screen(HelpModal("Parameter Help", "\n\n".join(lines)))

    def _read_params(self) -> dict | None:
        """Read and validate all parameter inputs. Returns dict or None on error."""
        topo_key = self.app.tui_state["topology"]
        topo = TOPOLOGIES[topo_key]

        try:
            n = int(self.query_one("#param-n", Input).value)
            if not 100 <= n <= 50000:
                raise ValueError("n must be between 100 and 50000")
        except ValueError as e:
            self.notify(str(e), severity="error")
            return None

        try:
            d = int(self.query_one("#param-d", Input).value)
            if not 2 <= d <= 30:
                raise ValueError("d must be between 2 and 30")
        except ValueError as e:
            self.notify(str(e), severity="error")
            return None

        try:
            seed = int(self.query_one("#param-seed", Input).value)
        except ValueError:
            self.notify("seed must be an integer", severity="error")
            return None

        extra = {}
        for pname in topo["params"]:
            raw = self.query_one(f"#param-{pname}", Input).value
            try:
                val = float(raw)
                if val == int(val) and "." not in raw:
                    val = int(val)
                extra[pname] = val
            except ValueError:
                self.notify(f"Invalid value for {pname}", severity="error")
                return None

        return {"n": n, "d": d, "seed": seed, **extra}

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "next-btn":
            params = self._read_params()
            if params is None:
                return
            self.app.tui_state["params"] = params
            if self.app.tui_state.get("sweep"):
                self.app.push_screen(SweepScreen())
            else:
                self.app.push_screen(FeatureNoiseScreen())


# -- Screen 2.5: Sweep configuration -----------------------------------------


class SweepScreen(Screen):
    """Configure which parameter to sweep in OFAT mode."""

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="sweep-container"):
            yield Label("OFAT Sweep Configuration", id="screen-title")
            yield Label("Parameter to sweep")
            yield Select(
                [(v["label"], k) for k, v in SWEEP_PARAMS.items()],
                id="sweep-param",
                value="noise_scale",
            )
            yield Label("Values will be swept from the predefined range for the selected parameter.")
            with Horizontal(id="nav-buttons"):
                yield Button("Back", id="back-btn")
                yield Button("Next: Feature Noise", variant="primary", id="next-btn")
        yield Footer()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "next-btn":
            sel = self.query_one("#sweep-param", Select)
            self.app.tui_state["sweep_param"] = sel.value
            self.app.tui_state["sweep_values"] = SWEEP_PARAMS[sel.value]["values"]
            self.app.push_screen(FeatureNoiseScreen())


# -- Screen 3: Feature noise -------------------------------------------------


class FeatureNoiseScreen(Screen):
    """Configure feature noise distribution, scale, and sparsity."""

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("escape", "go_back", "Back"),
        Binding("question_mark", "show_help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="fnoise-container"):
            yield Label("Feature Noise", id="screen-title")
            yield Label("Distribution")
            yield Select(
                [(label, val) for label, val in NOISE_DISTRIBUTIONS],
                id="noise-dist",
                value="gaussian",
            )
            yield Label("Scale")
            yield Input(value="0.1", placeholder="0.0-5.0", id="noise-scale")
            yield Label("Presets:")
            yield RadioSet(
                *[RadioButton(v, id=f"preset-{i}") for i, v in enumerate(NOISE_SCALE_PRESETS)],
                id="scale-presets",
            )
            yield Label("Sparsity (fraction of features affected)")
            yield Select(
                [(label, val) for label, val in SPARSITY_OPTIONS],
                id="noise-sparsity",
                value="1.0",
            )
            with Horizontal(id="nav-buttons"):
                yield Button("Back", id="back-btn")
                yield Button("Next: Label Noise", variant="primary", id="next-btn")
        yield Footer()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_show_help(self) -> None:
        self.app.push_screen(HelpModal(
            "Feature Noise",
            "Additive noise applied to feature columns.\n\n"
            "Scale: standard deviation (gaussian/laplacian) or half-width (uniform).\n"
            "Sparsity: fraction of feature columns that receive noise. "
            "1.0 corrupts all columns, 0.1 corrupts only 10%.",
        ))

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "scale-presets" and event.pressed is not None:
            self.query_one("#noise-scale", Input).value = str(event.pressed.label)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "next-btn":
            dist = self.query_one("#noise-dist", Select).value
            try:
                scale = float(self.query_one("#noise-scale", Input).value)
                if scale < 0:
                    raise ValueError
            except ValueError:
                self.notify("Scale must be a non-negative number", severity="error")
                return
            sparsity = float(self.query_one("#noise-sparsity", Select).value)

            self.app.tui_state["feature_noise"] = {
                "distribution": dist,
                "scale": scale,
                "sparsity": sparsity,
            }
            self.app.push_screen(LabelNoiseScreen())


# -- Screen 4: Label noise ---------------------------------------------------


class LabelNoiseScreen(Screen):
    """Configure label flip probability and targeting mode."""

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("escape", "go_back", "Back"),
        Binding("question_mark", "show_help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="lnoise-container"):
            yield Label("Label Noise", id="screen-title")
            yield Label("Flip probability")
            yield Select(
                [(label, val) for label, val in FLIP_PROB_OPTIONS],
                id="flip-prob",
                value="0.05",
            )
            yield Label("Targeting mode")
            yield RadioSet(
                *[RadioButton(label, id=f"target-{val}") for label, val in TARGETING_OPTIONS],
                id="targeting-radio",
            )
            with Horizontal(id="nav-buttons"):
                yield Button("Back", id="back-btn")
                yield Button("Next: Export", variant="primary", id="next-btn")
        yield Footer()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_show_help(self) -> None:
        self.app.push_screen(HelpModal(
            "Label Noise",
            "Flips a fraction of class labels to simulate annotation errors.\n\n"
            "Uniform: selects points to flip at random.\n"
            "Boundary: prioritizes points near class boundaries, simulating "
            "the kind of errors that occur where classes overlap.",
        ))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "next-btn":
            flip_prob = float(self.query_one("#flip-prob", Select).value)
            radio = self.query_one("#targeting-radio", RadioSet)
            if radio.pressed_button is None:
                targeting = "uniform"
            else:
                targeting = radio.pressed_button.id.replace("target-", "")

            self.app.tui_state["label_noise"] = {
                "flip_prob": flip_prob,
                "targeting": targeting,
            }
            self.app.push_screen(ExportScreen())


# -- Screen 5: Export options -------------------------------------------------


class ExportScreen(Screen):
    """Configure output directory, formats, and auto-visualize toggle."""

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="export-container"):
            yield Label("Export Options", id="screen-title")
            yield Label("Output directory")
            yield Input(value="output", placeholder="path/to/directory", id="output-dir")
            yield Label("Dataset name (optional)")
            yield Input(placeholder="auto-generated if empty", id="dataset-name")
            yield Label("Formats")
            yield Checkbox("NPZ (compressed NumPy)", value=True, id="fmt-npz")
            yield Checkbox("CSV", id="fmt-csv")
            yield Checkbox("PT (PyTorch)", id="fmt-pt")
            yield Checkbox("Auto-visualize after generation", value=True, id="auto-viz")
            with Horizontal(id="nav-buttons"):
                yield Button("Back", id="back-btn")
                yield Button("Review", variant="primary", id="next-btn")
        yield Footer()

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "next-btn":
            formats = []
            if self.query_one("#fmt-npz", Checkbox).value:
                formats.append("npz")
            if self.query_one("#fmt-csv", Checkbox).value:
                formats.append("csv")
            if self.query_one("#fmt-pt", Checkbox).value:
                formats.append("pt")
            if not formats:
                self.notify("Select at least one export format", severity="warning")
                return

            name = self.query_one("#dataset-name", Input).value.strip()

            self.app.tui_state["export"] = {
                "output_dir": self.query_one("#output-dir", Input).value.strip() or "output",
                "formats": formats,
                "name": name if name else None,
                "auto_viz": self.query_one("#auto-viz", Checkbox).value,
            }
            self.app.push_screen(ReviewScreen())


# -- Screen 6: Review and confirm --------------------------------------------


class ReviewScreen(Screen):
    """Summary of all parameters with generate/back/quit controls."""

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="review-container"):
            yield Label("Review Configuration", id="screen-title")
            yield DataTable(id="review-table")
            with Horizontal(id="nav-buttons"):
                yield Button("Back", id="back-btn")
                yield Button("Generate", variant="success", id="generate-btn")
                yield Button("Quit", variant="error", id="quit-btn")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#review-table", DataTable)
        table.add_columns("Parameter", "Value")
        state = self.app.tui_state

        table.add_row("Topology", TOPOLOGIES[state["topology"]]["label"])
        for k, v in state["params"].items():
            table.add_row(k, str(v))

        fn = state["feature_noise"]
        table.add_row("Feature noise", fn["distribution"])
        table.add_row("Noise scale", str(fn["scale"]))
        table.add_row("Sparsity", str(fn["sparsity"]))

        ln = state["label_noise"]
        table.add_row("Flip prob", str(ln["flip_prob"]))
        table.add_row("Targeting", ln["targeting"])

        ex = state["export"]
        table.add_row("Output dir", ex["output_dir"])
        table.add_row("Formats", ", ".join(ex["formats"]))
        table.add_row("Auto-visualize", str(ex["auto_viz"]))

        if state.get("sweep"):
            table.add_row("Sweep param", state["sweep_param"])
            table.add_row("Sweep values", str(state["sweep_values"]))

    def action_quit_app(self) -> None:
        self.app.exit()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "quit-btn":
            self.app.exit()
        elif event.button.id == "generate-btn":
            self.app.push_screen(GeneratingScreen())


# -- Screen 7: Generation progress -------------------------------------------

GENERATORS = {
    "concentric_annuli": concentric_annuli,
    "archimedean_spirals": archimedean_spirals,
    "nd_checkerboard": nd_checkerboard,
    "disjoint_clusters": disjoint_clusters,
}


class GeneratingScreen(Screen):
    """Runs generation, noise injection, export, and optional visualization."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="gen-container"):
            yield Label("Generating...", id="screen-title")
            yield ProgressBar(total=100, id="progress")
            yield Static("", id="status-text")
            yield Button("Done", variant="primary", id="done-btn", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.run_generation()

    @work(thread=True)
    def run_generation(self) -> None:
        state = self.app.tui_state
        topo_key = state["topology"]
        params = state["params"]
        fn = state["feature_noise"]
        ln = state["label_noise"]
        ex = state["export"]

        gen_func = GENERATORS[topo_key]
        n, d, seed = params["n"], params["d"], params["seed"]
        extra = {k: v for k, v in params.items() if k not in ("n", "d", "seed")}

        if state.get("sweep"):
            configs = self._build_sweep_configs(state, fn, ln)
        else:
            configs = [("single", fn, ln)]

        total = len(configs)
        bar = self.query_one("#progress", ProgressBar)
        bar.update(total=total)
        status = self.query_one("#status-text", Static)

        # Accumulate viz jobs to run on the main thread after all generation is done.
        # Matplotlib is not thread-safe; calling it from a worker thread causes SIGSEGV.
        viz_jobs: list[tuple] = []

        for i, (suffix, fn_cfg, ln_cfg) in enumerate(configs):
            self.app.call_from_thread(status.update, f"[{i + 1}/{total}] Generating {suffix}...")

            X_clean, y_clean = gen_func(n=n, d=d, seed=seed, **extra)

            if fn_cfg["distribution"] != "none" and fn_cfg["scale"] > 0:
                X_noisy = apply_feature_noise(
                    X_clean,
                    distribution=fn_cfg["distribution"],
                    scale=fn_cfg["scale"],
                    sparsity=fn_cfg["sparsity"],
                    seed=seed,
                )
            else:
                X_noisy = X_clean.copy()

            if ln_cfg["flip_prob"] > 0:
                y_noisy, flip_mask = apply_label_noise(
                    y_clean,
                    flip_prob=ln_cfg["flip_prob"],
                    targeting=ln_cfg["targeting"],
                    X=X_clean if ln_cfg["targeting"] == "boundary" else None,
                    seed=seed,
                )
            else:
                y_noisy = y_clean.copy()
                flip_mask = np.zeros(len(y_clean), dtype=bool)

            metadata = {
                "topology": topo_key,
                "params": params,
                "feature_noise": fn_cfg,
                "label_noise": ln_cfg,
            }

            name = ex["name"]
            if total > 1:
                name = f"{name or 'dataset'}_{suffix}"

            export_dataset(
                X_clean, y_clean, X_noisy, y_noisy, flip_mask,
                metadata, ex["output_dir"], ex["formats"], name,
            )

            if ex["auto_viz"]:
                plots_dir = str(Path(ex["output_dir"]) / "plots")
                viz_jobs.append((X_clean, y_clean, X_noisy, y_noisy, flip_mask, plots_dir, name or "dataset"))

            self.app.call_from_thread(bar.advance, 1)

        # Hand visualization back to the main thread
        if viz_jobs:
            self.app.call_from_thread(self._run_visualization, viz_jobs, ex["output_dir"], total)
        else:
            done_msg = f"Done. {total} dataset(s) exported to {ex['output_dir']}/"
            self.app.call_from_thread(status.update, done_msg)
            done_btn = self.query_one("#done-btn", Button)
            self.app.call_from_thread(setattr, done_btn, "disabled", False)

    def _run_visualization(
        self,
        viz_jobs: list[tuple],
        output_dir: str,
        total: int,
    ) -> None:
        """Run matplotlib visualization on the main thread."""
        status = self.query_one("#status-text", Static)
        status.update(f"Rendering {len(viz_jobs)} plot suite(s)...")
        for X_clean, y_clean, X_noisy, y_noisy, flip_mask, plots_dir, name in viz_jobs:
            render_all(X_clean, y_clean, X_noisy, y_noisy, flip_mask, plots_dir, name)
        status.update(f"Done. {total} dataset(s) exported to {output_dir}/")
        self.query_one("#done-btn", Button).disabled = False

    def _build_sweep_configs(self, state: dict, fn: dict, ln: dict) -> list[tuple]:
        """Build list of (suffix, feature_noise_cfg, label_noise_cfg) for sweep."""
        param = state["sweep_param"]
        values = state["sweep_values"]
        configs = []
        for val in values:
            fn_cfg = dict(fn)
            ln_cfg = dict(ln)
            if param == "noise_scale":
                fn_cfg["scale"] = val
            elif param == "flip_prob":
                ln_cfg["flip_prob"] = val
            elif param == "sparsity":
                fn_cfg["sparsity"] = val
            suffix = f"{param}_{val}"
            configs.append((suffix, fn_cfg, ln_cfg))
        return configs

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "done-btn":
            self.app.exit()


# -- App ----------------------------------------------------------------------

APP_CSS = """
Screen {
    align: center middle;
}

#screen-title {
    text-style: bold;
    margin-bottom: 1;
}

#topology-container, #params-container, #fnoise-container,
#lnoise-container, #export-container, #review-container,
#gen-container, #sweep-container {
    width: 80;
    height: 1fr;
    padding: 1 2;
}

#nav-buttons {
    margin-top: 1;
    height: 3;
}

#nav-buttons Button {
    margin-right: 1;
}

#help-modal {
    width: 60;
    max-height: 20;
    padding: 1 2;
    background: $surface;
    border: thick $primary;
}

#help-title {
    text-style: bold;
    margin-bottom: 1;
}

#review-table {
    height: auto;
    max-height: 20;
}

#progress {
    margin: 1 0;
}

#status-text {
    margin: 1 0;
}
"""


class TopowarpApp(App):
    """Topowarp dataset generation wizard."""

    TITLE = "Topowarp"
    CSS = APP_CSS
    BINDINGS = [Binding("ctrl+q", "quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self.tui_state: dict = {}

    def on_mount(self) -> None:
        self.push_screen(TopologyScreen())


def main() -> None:
    app = TopowarpApp()
    app.run()


if __name__ == "__main__":
    main()
