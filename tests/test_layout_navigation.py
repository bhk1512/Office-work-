from __future__ import annotations

import unittest
from typing import Any, Iterable

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


def _find_component_by_id(root: Any, target_id: str) -> Any:
    for node in _iter_components(root):
        if getattr(node, "id", None) == target_id:
            return node
    raise AssertionError(f"Component with id '{target_id}' not found")


def _count_components_by_id(root: Any, target_id: str) -> int:
    count = 0
    for node in _iter_components(root):
        if getattr(node, "id", None) == target_id:
            count += 1
    return count


class LayoutNavigationTests(unittest.TestCase):
    def test_main_tabs_excludes_dashboard(self) -> None:
        layout = build_layout("")
        tabs = _find_component_by_id(layout, "main-tabs")
        labels = [getattr(tab, "label", None) for tab in getattr(tabs, "children", [])]
        tab_ids = [getattr(tab, "tab_id", None) for tab in getattr(tabs, "children", [])]

        self.assertEqual(
            labels,
            [
                "Executive Overview",
                "Project Overview",
                "Tower Erection Analytics",
                "Stringing Analytics",
            ],
        )
        self.assertEqual(
            tab_ids,
            [
                "executive-overview",
                "project-overview",
                "analytics",
                "stringing-analytics",
            ],
        )
        self.assertNotIn("Dashboard", labels)

    def test_global_performance_triggers_and_stringing_scope_exist_once(self) -> None:
        layout = build_layout("")

        self.assertEqual(_count_components_by_id(layout, "btn-open-global-performance-erection"), 1)
        self.assertEqual(_count_components_by_id(layout, "btn-open-global-performance-stringing"), 1)
        self.assertEqual(_count_components_by_id(layout, "f-stringing-scope"), 1)

        # Legacy components remain mounted for callback compatibility after removing the tab.
        _find_component_by_id(layout, "legacy-dashboard-mount")


if __name__ == "__main__":
    unittest.main()
