"""Tests for the TUI wizard.

Uses textual's pilot API for headless testing of screen flow
and generation dispatch.
"""

import shutil
from pathlib import Path

import pytest
from textual.widgets import Button, Checkbox, Input, RadioButton, RadioSet, Select

from tui import (
    ExportScreen,
    FeatureNoiseScreen,
    GeneratingScreen,
    HelpModal,
    LabelNoiseScreen,
    ParamsScreen,
    ReviewScreen,
    SweepScreen,
    TopologyScreen,
    TopowarpApp,
)


@pytest.fixture
def app():
    return TopowarpApp()


TERMINAL_SIZE = (120, 60)


class TestTopologyScreen:
    @pytest.mark.asyncio
    async def test_initial_screen_renders(self, app):
        async with app.run_test(size=TERMINAL_SIZE) as pilot:
            assert app.screen.query_one("#topology-radio", RadioSet) is not None
            assert app.screen.query_one("#sweep-toggle", Checkbox) is not None

    @pytest.mark.asyncio
    async def test_next_without_selection_warns(self, app):
        async with app.run_test(size=TERMINAL_SIZE) as pilot:
            await pilot.click("#next-btn")
            # Should still be on the same screen (no push happened)
            assert isinstance(app.screen, TopologyScreen)

    @pytest.mark.asyncio
    async def test_next_with_selection_advances(self, app):
        async with app.run_test(size=TERMINAL_SIZE) as pilot:
            radio = app.screen.query_one("#topology-radio", RadioSet)
            buttons = radio.query(RadioButton)
            await pilot.click(f"#{buttons.first().id}")
            await pilot.click("#next-btn")
            assert isinstance(app.screen, ParamsScreen)

    @pytest.mark.asyncio
    async def test_quit_exits(self, app):
        async with app.run_test(size=TERMINAL_SIZE) as pilot:
            await pilot.click("#quit-btn")
            assert app.return_code is not None or app._exit


class TestParamsScreen:
    @pytest.mark.asyncio
    async def test_renders_topology_specific_params(self, app):
        app.tui_state["topology"] = "concentric_annuli"
        app.tui_state["sweep"] = False
        async with app.run_test(size=TERMINAL_SIZE) as pilot:
            app.push_screen(ParamsScreen())
            await pilot.pause()
            assert app.screen.query_one("#param-n_rings", Input) is not None
            assert app.screen.query_one("#param-thickness", Input) is not None
            assert app.screen.query_one("#param-margin", Input) is not None

    @pytest.mark.asyncio
    async def test_back_pops_screen(self, app):
        app.tui_state["topology"] = "nd_checkerboard"
        app.tui_state["sweep"] = False
        async with app.run_test(size=TERMINAL_SIZE) as pilot:
            app.push_screen(ParamsScreen())
            await pilot.pause()
            await pilot.click("#back-btn")
            await pilot.pause()
            assert not isinstance(app.screen, ParamsScreen)


class TestHelpModal:
    @pytest.mark.asyncio
    async def test_help_modal_opens_and_closes(self, app):
        async with app.run_test(size=TERMINAL_SIZE) as pilot:
            await pilot.press("question_mark")
            await pilot.pause()
            assert isinstance(app.screen, HelpModal)
            await pilot.click("#help-close")
            await pilot.pause()
            assert not isinstance(app.screen, HelpModal)


class TestFullWizardFlow:
    @pytest.mark.asyncio
    async def test_end_to_end_generation(self, app, tmp_path):
        """Walk through the entire wizard and generate a dataset."""
        async with app.run_test(size=TERMINAL_SIZE) as pilot:
            # Screen 1: select topology
            radio = app.screen.query_one("#topology-radio", RadioSet)
            buttons = radio.query(RadioButton)
            await pilot.click(f"#{buttons.first().id}")
            await pilot.click("#next-btn")
            await pilot.pause()
            assert isinstance(app.screen, ParamsScreen)

            # Screen 2: set params (use small n for speed)
            app.screen.query_one("#param-n", Input).value = "100"
            app.screen.query_one("#param-d", Input).value = "2"
            await pilot.click("#next-btn")
            await pilot.pause()
            assert isinstance(app.screen, FeatureNoiseScreen)

            # Screen 3: feature noise (defaults are fine)
            await pilot.click("#next-btn")
            await pilot.pause()
            assert isinstance(app.screen, LabelNoiseScreen)

            # Screen 4: label noise (defaults are fine)
            await pilot.click("#next-btn")
            await pilot.pause()
            assert isinstance(app.screen, ExportScreen)

            # Screen 5: export options
            app.screen.query_one("#output-dir", Input).value = str(tmp_path)
            app.screen.query_one("#dataset-name", Input).value = "test_run"
            app.screen.query_one("#auto-viz", Checkbox).value = False
            await pilot.click("#next-btn")
            await pilot.pause()
            assert isinstance(app.screen, ReviewScreen)

            # Screen 6: review and generate
            await pilot.click("#generate-btn")
            await pilot.pause(delay=3.0)
            assert isinstance(app.screen, GeneratingScreen)

            # Wait for generation to complete
            for _ in range(20):
                done_btn = app.screen.query_one("#done-btn", Button)
                if not done_btn.disabled:
                    break
                await pilot.pause(delay=0.5)

            # Verify output files exist
            npz_files = list(tmp_path.glob("*.npz"))
            json_files = list(tmp_path.glob("*.json"))
            assert len(npz_files) == 1, f"Expected 1 npz file, got {npz_files}"
            assert len(json_files) == 1, f"Expected 1 json file, got {json_files}"


class TestSweepMode:
    @pytest.mark.asyncio
    async def test_sweep_toggle_routes_to_sweep_screen(self, app):
        async with app.run_test(size=TERMINAL_SIZE) as pilot:
            # Select topology
            radio = app.screen.query_one("#topology-radio", RadioSet)
            buttons = radio.query(RadioButton)
            await pilot.click(f"#{buttons.first().id}")
            # Enable sweep
            await pilot.click("#sweep-toggle")
            await pilot.click("#next-btn")
            await pilot.pause()
            assert isinstance(app.screen, ParamsScreen)

            # Advance past params
            app.screen.query_one("#param-n", Input).value = "100"
            await pilot.click("#next-btn")
            await pilot.pause()
            assert isinstance(app.screen, SweepScreen)
