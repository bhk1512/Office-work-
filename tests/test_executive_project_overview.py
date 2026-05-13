from __future__ import annotations

import unittest
from typing import Any, Iterable
from unittest.mock import patch

import pandas as pd
from dash import Dash

from dashboard.callbacks import register_callbacks
from dashboard.config import AppConfig
from dashboard.layout import build_layout


def _iter_components(root: Any) -> Iterable[Any]:
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        yield node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(reversed([child for child in children if child is not None]))
        elif children is not None:
            stack.append(children)


def _callback_entry(app: Dash, output_token: str) -> dict[str, Any]:
    for entry in app.callback_map.values():
        if output_token in str(entry.get("output")):
            return entry
    raise AssertionError(f"Callback not found for token: {output_token}")


def _output_index(entry: dict[str, Any], component_id: str, component_property: str) -> int:
    outputs = entry.get("output") or []
    if not isinstance(outputs, (list, tuple)):
        raise AssertionError("Expected callback output list")
    for idx, output in enumerate(outputs):
        if (
            getattr(output, "component_id", None) == component_id
            and getattr(output, "component_property", None) == component_property
        ):
            return idx
    raise AssertionError(f"Output not found: {component_id}.{component_property}")


class ExecutiveProjectOverviewTests(unittest.TestCase):
    @staticmethod
    def _portfolio_rows(rendered: Any) -> list[Any]:
        table = rendered
        children = getattr(rendered, "children", None)
        if isinstance(children, list) and children:
            maybe_table = children[0]
            if getattr(maybe_table, "className", "") == "portfolio-table":
                table = maybe_table
        return getattr(getattr(table, "children", [None, None])[1], "children", [])

    def _build_app(self, stretch_section: pd.DataFrame | None = None) -> Dash:
        app = Dash(__name__)
        app.layout = build_layout("")

        status_activity = pd.DataFrame(
            [
                {
                    "project_code": "TA 419",
                    "project_display": "TA 419",
                    "line_name": "L1",
                    "report_date": "2020-03-29",
                    "activity_norm": "foundation",
                    "quantity_primary": 100,
                    "cumulative_progress": 40,
                    "plan_for_month": 10,
                    "progress_for_month": 7,
                },
                {
                    "project_code": "TA 419",
                    "project_display": "TA 419",
                    "line_name": "L1",
                    "report_date": "2020-04-29",
                    "activity_norm": "foundation",
                    "quantity_primary": 100,
                    "cumulative_progress": 55,
                    "plan_for_month": 12,
                    "progress_for_month": 9,
                },
                {
                    "project_code": "TA 505",
                    "project_display": "TA 505",
                    "line_name": "L2",
                    "report_date": "2020-01-29",
                    "activity_norm": "foundation",
                    "quantity_primary": 200,
                    "cumulative_progress": 120,
                    "plan_for_month": 20,
                    "progress_for_month": 11,
                },
            ]
        )
        status_snapshot_project = pd.DataFrame(
            [
                {
                    "project_code": "TA 419",
                    "project_display": "TA 419",
                    "line_name": "L1",
                    "month": "2020-03-01",
                    "completion_pct": 40.0,
                    "cumulative_progress_sum": 40,
                    "plan_for_month_sum": 10,
                    "progress_for_month_sum": 7,
                    "overall_rag": "AMBER",
                    "foundation_completion_pct": 40.0,
                },
                {
                    "project_code": "TA 419",
                    "project_display": "TA 419",
                    "line_name": "L1",
                    "month": "2020-04-01",
                    "completion_pct": 55.0,
                    "cumulative_progress_sum": 55,
                    "plan_for_month_sum": 12,
                    "progress_for_month_sum": 9,
                    "overall_rag": "GREEN",
                    "foundation_completion_pct": 55.0,
                },
                {
                    "project_code": "TA 505",
                    "project_display": "TA 505",
                    "line_name": "L2",
                    "month": "2020-01-01",
                    "completion_pct": 60.0,
                    "cumulative_progress_sum": 120,
                    "plan_for_month_sum": 20,
                    "progress_for_month_sum": 11,
                    "overall_rag": "AMBER",
                    "foundation_completion_pct": 60.0,
                },
            ]
        )
        status_snapshot_overall = pd.DataFrame(
            [
                {"month": "2020-03-01", "completion_pct": 40.0, "plan_for_month_sum": 10, "progress_for_month_sum": 7},
                {"month": "2020-04-01", "completion_pct": 56.5, "plan_for_month_sum": 32, "progress_for_month_sum": 20},
            ]
        )
        project_info = pd.DataFrame(
            [
                {"project_code": "TA 419", "PCH": "Mr Arun Felbin"},
                {"project_code": "TA 505", "PCH": "Mr Nabajit Baruah"},
            ]
        )
        empty = pd.DataFrame()
        stretch_section_frame = stretch_section if isinstance(stretch_section, pd.DataFrame) else empty

        register_callbacks(
            app,
            data_provider=lambda: pd.DataFrame({"Date": []}),
            config=AppConfig(enable_stringing=False),
            status_activity_provider=lambda: status_activity,
            status_snapshot_project_provider=lambda: status_snapshot_project,
            status_snapshot_overall_provider=lambda: status_snapshot_overall,
            stretch_section_provider=lambda: stretch_section_frame,
            stretch_readiness_summary_provider=lambda: empty,
            manpower_productivity_provider=lambda: empty,
            project_info_provider=lambda: project_info,
            stringing_compiled_provider=lambda: empty,
        )
        return app

    def test_exec_payload_trend_and_portfolio_activity_cells(self) -> None:
        app = self._build_app()
        compute_cb = _callback_entry(app, "executive-overview-payload.data")["callback"].__wrapped__
        exec_payload, _proj_payload = compute_cb([], [], [], None, [], "monthly", 0, 0)

        status_trend = exec_payload.get("overall_trends", {}).get("status", [])
        self.assertGreaterEqual(len(status_trend), 2, "Expected monthly status trend to include Mar + Apr points")

        render_portfolio_cb = app.callback_map["exec-portfolio-table-container.children"]["callback"].__wrapped__
        rendered = render_portfolio_cb(exec_payload, "cum", [])
        rows = self._portfolio_rows(rendered)
        project_rows = [
            row
            for row in rows
            if isinstance(getattr(row, "id", None), dict) and row.id.get("type") == "project-tile-trigger"
        ]
        row_by_project = {row.id.get("project"): row for row in project_rows}
        ta505_row = row_by_project.get("TA 505")
        self.assertIsNotNone(ta505_row, "TA 505 row should remain present even with NaT-only period values")
        foundation_cell = ta505_row.children[1].children
        self.assertEqual(getattr(foundation_cell, "className", ""), "prog-cell")

    def test_month_view_marks_stale_projects_without_target_month_data(self) -> None:
        app = self._build_app()
        compute_cb = _callback_entry(app, "executive-overview-payload.data")["callback"].__wrapped__
        exec_payload, _proj_payload = compute_cb([], [], [], None, [], "monthly", 0, 0)

        render_portfolio_cb = app.callback_map["exec-portfolio-table-container.children"]["callback"].__wrapped__
        rendered = render_portfolio_cb(exec_payload, "month", [])
        rows = self._portfolio_rows(rendered)
        project_rows = [
            row
            for row in rows
            if isinstance(getattr(row, "id", None), dict) and row.id.get("type") == "project-tile-trigger"
        ]

        labels_by_project: dict[str, str] = {}
        for row in project_rows:
            base_name = str(row.id.get("project") or "")
            label_text = str(getattr(row.children[0], "children", "") or "")
            labels_by_project[base_name] = label_text

        self.assertIn("TA 419", labels_by_project)
        self.assertIn("TA 505", labels_by_project)
        self.assertIn("[STALE:", labels_by_project["TA 419"])
        self.assertIn("[STALE:", labels_by_project["TA 505"])

    def test_month_view_uses_selected_month_for_stale_logic(self) -> None:
        app = self._build_app()
        compute_cb = _callback_entry(app, "executive-overview-payload.data")["callback"].__wrapped__
        exec_payload, _proj_payload = compute_cb([], [], ["2020-04"], None, [], "monthly", 0, 0)

        self.assertEqual(exec_payload.get("target_month"), "2020-04-01")

        render_portfolio_cb = app.callback_map["exec-portfolio-table-container.children"]["callback"].__wrapped__
        rendered = render_portfolio_cb(exec_payload, "month", [])
        rows = self._portfolio_rows(rendered)
        project_rows = [
            row
            for row in rows
            if isinstance(getattr(row, "id", None), dict) and row.id.get("type") == "project-tile-trigger"
        ]

        labels_by_project: dict[str, str] = {}
        for row in project_rows:
            base_name = str(row.id.get("project") or "")
            label_text = str(getattr(row.children[0], "children", "") or "")
            labels_by_project[base_name] = label_text

        self.assertIn("TA 419", labels_by_project)
        self.assertIn("TA 505", labels_by_project)
        self.assertNotIn("[STALE:", labels_by_project["TA 419"])
        self.assertIn("[STALE:", labels_by_project["TA 505"])

    def test_project_overview_donut_and_lag_fallback_contract(self) -> None:
        app = self._build_app()
        entry = _callback_entry(app, "proj-stretch-state-graph.figure")
        render_proj_cb = entry["callback"].__wrapped__
        payload = {
            "scope": {"project": "TA 419", "line": "L1", "period_label": "30 Apr 2026"},
            "kpis": {
                "completion_pct": 55.0,
                "plan_attainment_pct": 75.0,
                "readiness_pct": 60.0,
                "gap_days_avg": 45.0,
                "manpower_availability_pct": 80.0,
                "rag": "AMBER",
            },
            "stretch": {
                "ready_km": 73.8,
                "total_km": 122.2,
                "readiness_km_pct": 60.4,
                "state_split": [
                    {"state": "READY", "count": 476},
                    {"state": "UNKNOWN", "count": 344},
                    {"state": "NOT_READY", "count": 180},
                    {"state": "PARTIAL", "count": 0},
                ],
                "blocked_sections": [{"section_id": "S-1"}],
            },
            "stringing_erection": {
                "gap_trend": [],
                "gap_days_avg": 45.0,
                "reason": "No section-level READY→P/O join available for selected scope.",
            },
            "manpower_productivity": {"scatter": []},
            "ranking": [{"project_display": "TA 419", "overall_rag": "AMBER"}],
        }
        outputs = render_proj_cb(payload, "TA 419")
        fig_state = outputs[_output_index(entry, "proj-stretch-state-graph", "figure")]
        fig_lag = outputs[_output_index(entry, "proj-es-lag-chart", "figure")]

        self.assertEqual(fig_state.layout.legend.orientation, "v")
        self.assertAlmostEqual(float(fig_state.layout.legend.x), 1.0, places=2)
        self.assertGreaterEqual(float(fig_state.data[0]["domain"]["x"][1]), 0.62)
        self.assertEqual(len(fig_lag.data), 0)
        annotation_text = str(fig_lag.layout.annotations[0]["text"]).lower()
        self.assertIn("gap", annotation_text)
        self.assertIn("data", annotation_text)

    def test_project_overview_selector_preserves_current_value(self) -> None:
        app = self._build_app()
        sync_selector_cb = _callback_entry(app, "proj-overview-project-select.value")["callback"].__wrapped__
        payload = {
            "project_ranking": [
                {"project_display": "TA 419", "overall_rag": "RED", "plan_slippage_pct": 12},
                {"project_display": "TA 505", "overall_rag": "AMBER", "plan_slippage_pct": 4},
            ]
        }
        options, value = sync_selector_cb(payload, "TA 505")
        self.assertTrue(any(opt.get("value") == "TA 505" for opt in options))
        self.assertEqual(value, "TA 505")

    def test_project_modal_donut_matches_overview_layout(self) -> None:
        stretch_section = pd.DataFrame(
            [
                {"project_code": "TA 419", "project_display": "TA 419", "readiness_state": "READY"},
                {"project_code": "TA 419", "project_display": "TA 419", "readiness_state": "PARTIAL"},
                {"project_code": "TA 419", "project_display": "TA 419", "readiness_state": "NOT_READY"},
                {"project_code": "TA 419", "project_display": "TA 419", "readiness_state": "UNKNOWN"},
            ]
        )
        app = self._build_app(stretch_section=stretch_section)
        render_modal_cb = _callback_entry(app, "project-modal-stretch-pie.figure")["callback"].__wrapped__
        outputs = render_modal_cb(
            {"project": "TA 419", "display": "TA 419", "code": "TA 419"},
            [],
            None,
            [],
            "all",
        )
        fig_state = outputs[4]
        self.assertEqual(fig_state.layout.legend.orientation, "v")
        self.assertAlmostEqual(float(fig_state.layout.legend.x), 1.0, places=2)
        self.assertGreaterEqual(float(fig_state.data[0]["domain"]["x"][1]), 0.62)
        self.assertEqual(int(fig_state.layout.height), 280)

    def test_exec_portfolio_lag_star_marker_and_note(self) -> None:
        app = self._build_app()
        render_portfolio_cb = app.callback_map["exec-portfolio-table-container.children"]["callback"].__wrapped__
        payload = {
            "kpis": {
                "gap_per_project": {"TA 419::L1": 34.0},
                "gap_fallback_per_project": {"TA 419::L1": True, "TA 505::L2": True},
            },
            "project_ranking": [
                {"project_display": "TA 419", "project_code": "TA 419", "line_name": "L1", "overall_rag": "AMBER"},
                {"project_display": "TA 505", "project_code": "TA 505", "line_name": "L2", "overall_rag": "RED"},
            ],
        }

        container = render_portfolio_cb(payload, "cum", [])
        self.assertEqual(getattr(container, "id", None), None)
        children = getattr(container, "children", [])
        self.assertTrue(isinstance(children, list) and len(children) == 2)
        note = children[1]
        self.assertEqual(getattr(note, "id", ""), "exec-portfolio-lag-note")
        self.assertIn("fallback", str(getattr(note, "children", "")).lower())

        rows = self._portfolio_rows(container)
        project_rows = [
            row
            for row in rows
            if isinstance(getattr(row, "id", None), dict) and row.id.get("type") == "project-tile-trigger"
        ]
        by_project = {str(row.id.get("project")): row for row in project_rows}
        self.assertIn("TA 419", by_project)
        self.assertIn("TA 505", by_project)
        lag_419 = str(getattr(by_project["TA 419"].children[6].children, "children", by_project["TA 419"].children[6].children))
        lag_505 = str(getattr(by_project["TA 505"].children[6].children, "children", by_project["TA 505"].children[6].children))
        self.assertIn("days*", lag_419)
        self.assertEqual(lag_505, "-*")

    def test_exec_portfolio_respects_expanded_store_for_row_visibility(self) -> None:
        app = self._build_app()
        render_portfolio_cb = app.callback_map["exec-portfolio-table-container.children"]["callback"].__wrapped__
        payload = {
            "project_ranking": [
                {"project_display": "TA 419", "project_code": "TA 419", "overall_rag": "AMBER"},
                {"project_display": "TA 505", "project_code": "TA 505", "overall_rag": "RED"},
            ],
            "kpis": {},
        }

        rendered = render_portfolio_cb(payload, "cum", ["mr-arun-felbin"])
        rows = self._portfolio_rows(rendered)
        project_rows = [
            row
            for row in rows
            if isinstance(getattr(row, "id", None), dict) and row.id.get("type") == "project-tile-trigger"
        ]
        by_project = {str(row.id.get("project")): row for row in project_rows}

        self.assertEqual((getattr(by_project["TA 419"], "style", {}) or {}).get("display"), "table-row")
        self.assertEqual((getattr(by_project["TA 505"], "style", {}) or {}).get("display"), "none")

        pch_rows = [row for row in rows if str(getattr(row, "id", "")).startswith("pch-row-")]
        by_pch = {str(row.id): row for row in pch_rows}
        arun_toggle = by_pch["pch-row-mr-arun-felbin"].children[0].children
        nabajit_toggle = by_pch["pch-row-mr-nabajit-baruah"].children[0].children
        self.assertIn("open", str(getattr(arun_toggle.children[0], "className", "")))
        self.assertNotIn("open", str(getattr(nabajit_toggle.children[0], "className", "")))
        self.assertEqual(getattr(arun_toggle, "type", None), "button")

    def test_exec_pch_toggle_store_multi_expand_and_single_click_toggle(self) -> None:
        app = self._build_app()
        sync_cb = _callback_entry(app, "exec-pch-expanded.data")["callback"].__wrapped__
        payload = {
            "project_ranking": [
                {"project_display": "TA 419", "project_code": "TA 419", "overall_rag": "AMBER"},
                {"project_display": "TA 505", "project_code": "TA 505", "overall_rag": "RED"},
            ],
        }

        with patch("dashboard.callbacks._resolve_triggered_id", return_value={"type": "exec-pch-toggle", "pch": "mr-arun-felbin"}):
            state = sync_cb([1, 0], payload, "cum", [])
        self.assertEqual(state, ["mr-arun-felbin"])

        with patch("dashboard.callbacks._resolve_triggered_id", return_value={"type": "exec-pch-toggle", "pch": "mr-nabajit-baruah"}):
            state = sync_cb([1, 1], payload, "cum", state)
        self.assertEqual(state, ["mr-arun-felbin", "mr-nabajit-baruah"])

        with patch("dashboard.callbacks._resolve_triggered_id", return_value={"type": "exec-pch-toggle", "pch": "mr-arun-felbin"}):
            state = sync_cb([2, 1], payload, "cum", state)
        self.assertEqual(state, ["mr-nabajit-baruah"])

    def test_exec_pch_store_prunes_removed_groups_on_payload_refresh(self) -> None:
        app = self._build_app()
        sync_cb = _callback_entry(app, "exec-pch-expanded.data")["callback"].__wrapped__
        payload = {
            "project_ranking": [
                {"project_display": "TA 505", "project_code": "TA 505", "overall_rag": "RED"},
            ],
        }
        previous = ["mr-arun-felbin", "mr-nabajit-baruah"]

        with patch("dashboard.callbacks._resolve_triggered_id", return_value="executive-overview-payload"):
            pruned = sync_cb([0], payload, "cum", previous)
        self.assertEqual(pruned, ["mr-nabajit-baruah"])

    def test_overview_and_modal_dpr_strip_share_union_fields(self) -> None:
        app = self._build_app()
        render_overview_dpr_cb = _callback_entry(app, "proj-dpr-strip.children")["callback"].__wrapped__
        render_modal_summary_cb = _callback_entry(app, "project-modal-dpr-strip.children")["callback"].__wrapped__
        payload = {
            "scope": {"project": "TA 419"},
            "kpis": {"rag": "AMBER"},
            "ranking": [{"project_display": "TA 419", "overall_rag": "AMBER"}],
        }
        focus = {"project": "TA 419", "display": "TA 419", "code": "TA 419"}
        overview_children = render_overview_dpr_cb(payload, "TA 419")
        modal_outputs = render_modal_summary_cb(focus, [], None, [], "all")
        modal_children = modal_outputs[2]

        def _labels(children: list[Any]) -> list[str]:
            labels: list[str] = []
            for child in children:
                nodes = getattr(child, "children", None) or []
                if not isinstance(nodes, list) or not nodes:
                    continue
                label_node = nodes[0]
                labels.append(str(getattr(label_node, "children", "")).strip())
            return labels

        expected = ["Project", "Voltage", "Latest DPR Date", "Scope (km)", "DPR Staleness", "RAG"]
        self.assertEqual(_labels(overview_children), expected)
        self.assertEqual(_labels(modal_children), expected)

    def test_layout_has_light_dropdown_and_this_month_toggle(self) -> None:
        layout = build_layout("")
        project_select = None
        modal_project_select = None
        month_filter = None
        portfolio_toggle = None
        overview_perf_collapse = None
        modal_perf_collapse = None
        overview_raw = None
        modal_raw = None
        for node in _iter_components(layout):
            node_id = getattr(node, "id", None)
            if node_id == "proj-overview-project-select":
                project_select = node
            if node_id == "project-modal-project-select":
                modal_project_select = node
            if node_id == "f-month":
                month_filter = node
            if node_id == "exec-portfolio-view":
                portfolio_toggle = node
            if node_id == "proj-overview-perf-collapse":
                overview_perf_collapse = node
            if node_id == "project-modal-perf-collapse":
                modal_perf_collapse = node
            if node_id == "proj-overview-erections-table":
                overview_raw = node
            if node_id == "project-modal-erections-table":
                modal_raw = node
        self.assertIsNotNone(project_select)
        self.assertIn("filter-select--light", str(getattr(project_select, "className", "")))
        self.assertIsNotNone(modal_project_select)
        self.assertIn("filter-select--light", str(getattr(modal_project_select, "className", "")))
        self.assertIsNotNone(month_filter)
        self.assertEqual(getattr(month_filter, "value", None), [])
        self.assertFalse(bool(getattr(month_filter, "persistence", False)))
        self.assertIsNotNone(portfolio_toggle)
        labels = [str(option.get("label")) for option in getattr(portfolio_toggle, "options", [])]
        self.assertIn("This Month", labels)
        self.assertIsNotNone(overview_perf_collapse)
        self.assertFalse(bool(getattr(overview_perf_collapse, "is_open", True)))
        self.assertIsNotNone(modal_perf_collapse)
        self.assertFalse(bool(getattr(modal_perf_collapse, "is_open", True)))
        self.assertIsNotNone(overview_raw)
        self.assertIsNotNone(modal_raw)


if __name__ == "__main__":
    unittest.main()
