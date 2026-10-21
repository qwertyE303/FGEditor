"""PySide6 GUI for the FlexGravity NMO editor."""

from __future__ import annotations

import html
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import __version__
from .expr import (
    CARTESIAN_VARS,
    SPHERICAL_VARS,
    check_domain,
    check_function,
    help_html,
)
from .model import (
    CARTESIAN_PARAMS,
    CUSTOM_PARAMS,
    FIELD_LABELS,
    FIELD_PARAMS,
    MODE_CARTESIAN,
    MODE_SPHERICAL,
    SPHERICAL_PARAMS,
    CustomRow,
    Project,
    ProjectError,
)

APP_TITLE = f"FGEditor - v{__version__}"
EMPTY_OBJECT = ""
SUMMARY_GROUPS = ("FG_U", "FG_M", "FG_L", "FG_S", "FG_R")
CUSTOM_LABELS = {"Domain": "作用区域"}
CUSTOM_PREVIEW_LIMIT = 24


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def asset_path(name: str) -> Path:
    """Locate a bundled data file (icon) in frozen and source runs."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        return base / name
    path = Path(__file__).resolve().parents[2] / "assets" / name
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parents[3] / "build" / "gen" / name
    return fallback if fallback.exists() else path


def brief_reason(message: str, limit: int = 26) -> str:
    """Condense one warning/error to a few words for the status bar."""
    text = message.strip()
    for separator in ("；", "，"):
        text = text.split(separator, 1)[0]
    if "：" in text and len(text) > limit:
        text = text.split("：", 1)[1]
    return text if len(text) <= limit else text[: limit - 1] + "…"


def brief_reasons(messages: list[str], limit: int = 2) -> str:
    text = "；".join(brief_reason(message) for message in messages[:limit])
    if len(messages) > limit:
        text += f"…（共 {len(messages)} 条）"
    return text


class Settings:
    def __init__(self) -> None:
        self.path = app_dir() / "FGEditor.ini"
        self.data: dict[str, str] = {}
        try:
            if self.path.exists():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}

    def get(self, key: str, default: str = "") -> str:
        return self.data.get(key, default)

    def set(self, key: str, value: str) -> None:
        self.data[key] = value
        try:
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        except Exception:
            pass


def log_line(message: str) -> None:
    try:
        path = app_dir() / "FGEditor.log"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as fp:
            fp.write(f"[{stamp}] {message}\n")
    except Exception:
        pass


def fmt(value: float) -> str:
    return f"{float(value):.9g}"


def parse_float(text: str) -> float:
    text = text.strip()
    if not text:
        return 0.0
    return float(text)


class SourcePickerDialog(QDialog):
    def __init__(self, project: Project, selected: list[int], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("选择起作用对象")
        self.resize(520, 520)
        self.project = project
        self.selected = set(selected)

        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索对象名…")
        layout.addWidget(self.search)
        self.list = QListWidget()
        layout.addWidget(self.list, 1)

        self.type_map = project.mls_object_types()
        all_indices = project.mls_objects()
        for index in selected:
            if index not in all_indices:
                all_indices.append(index)
        all_indices = sorted(set(all_indices), key=project.object_key)
        for index in all_indices:
            name = project.object_display(index)
            types = "/".join(sorted(self.type_map.get(index, [])))
            label = f"{name}    [{types}]" if types else f"{name}    [未归组]"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if index in self.selected else Qt.Unchecked)
            self.list.addItem(item)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.search.textChanged.connect(self._filter)

    def _filter(self, text: str) -> None:
        needle = text.strip().casefold()
        for row in range(self.list.count()):
            item = self.list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().casefold())

    def selected_indices(self) -> list[int]:
        result = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.checkState() == Qt.Checked:
                result.append(int(item.data(Qt.UserRole)))
        return result


class HalfWidthTable(QTableWidget):
    """Table whose first column takes half the width, all other columns the rest."""

    def __init__(self, rows: int, columns: int, parent: QWidget | None = None):
        super().__init__(rows, columns, parent)
        self._fitted: tuple[int, int] | None = None
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(False)
        self.viewport().installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.viewport() and event.type() == QEvent.Resize:
            self.fit_columns()
        return super().eventFilter(watched, event)

    def fit_columns(self) -> None:
        columns = self.columnCount()
        width = self.viewport().width()
        if columns < 1 or width <= 0:
            return
        key = (width, columns)
        if key == self._fitted:
            return
        self._fitted = key
        head = width // 2 if columns > 1 else width
        self.setColumnWidth(0, head)
        if columns > 1:
            rest = width - head
            share = rest // (columns - 1)
            for index in range(1, columns):
                width_i = rest - share * (columns - 2) if index == columns - 1 else share
                self.setColumnWidth(index, width_i)


class FieldTab(QWidget):
    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self.params = FIELD_PARAMS[name]
        self.project: Project | None = None
        self._loading = False

        layout = QVBoxLayout(self)
        batch = QGroupBox("批量操作")
        batch_layout = QHBoxLayout(batch)
        self.spins: list[QDoubleSpinBox] = []
        self.checks: list[QCheckBox] = []
        for param in self.params:
            if name == "FG_U":
                check = QCheckBox(param)
                check.setChecked(True)
                batch_layout.addWidget(check)
                self.checks.append(check)
            else:
                batch_layout.addWidget(QLabel(param))
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1.0e12, 1.0e12)
            spin.setSingleStep(1.0)
            spin.setMinimumWidth(130)
            batch_layout.addWidget(spin)
            self.spins.append(spin)
        batch_layout.addStretch(1)
        self.apply_button = QPushButton("应用到选中行")
        self.apply_button.clicked.connect(self.apply_batch)
        batch_layout.addWidget(self.apply_button)
        layout.addWidget(batch)

        self.table = HalfWidthTable(0, 1 + len(self.params))
        self.table.setHorizontalHeaderLabels(["对象"] + self.params)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked |
            QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.reload()

    def reload(self) -> None:
        self._loading = True
        self.table.setRowCount(0)
        if self.project is not None:
            rows = self.project.fields[self.name].rows
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                object_item = QTableWidgetItem(
                    self.project.object_display(row.object_index) if row.object_index is not None else EMPTY_OBJECT)
                object_item.setFlags(object_item.flags() & ~Qt.ItemIsEditable)
                object_item.setData(Qt.UserRole, row.object_index)
                object_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, 0, object_item)
                for c, value in enumerate(row.values):
                    item = QTableWidgetItem(fmt(value))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(r, 1 + c, item)
        self._loading = False
        self.table.fit_columns()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or self.project is None:
            return
        if item.column() == 0:
            return
        try:
            value = parse_float(item.text())
        except ValueError:
            value = 0.0
        item.setText(fmt(value))

    def apply_batch(self) -> None:
        if self.project is None:
            return
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(self, "批量操作", "请先在表格里选中要应用的行。")
            return
        active = list(range(len(self.spins)))
        if self.checks:
            active = [index for index, check in enumerate(self.checks) if check.isChecked()]
            if not active:
                QMessageBox.information(self, "批量操作", "请至少勾选一个要应用的分量。")
                return
        self._loading = True
        for row in rows:
            for c in active:
                value = self.spins[c].value()
                cell = self.table.item(row, 1 + c)
                if cell is None:
                    cell = QTableWidgetItem()
                    self.table.setItem(row, 1 + c, cell)
                cell.setText(fmt(value))
        self._loading = False
        self.table.viewport().update()

    def sync_to_project(self) -> None:
        if self.project is None:
            return
        target = self.project.fields[self.name].rows
        if len(target) != self.table.rowCount():
            return
        for r, row in enumerate(target):
            values = list(row.values)
            for c in range(len(self.params)):
                cell = self.table.item(r, 1 + c)
                try:
                    values[c] = parse_float(cell.text()) if cell is not None else 0.0
                except ValueError:
                    values[c] = 0.0
            row.values = values


class RTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: Project | None = None
        self.batch_sources: list[int] = []

        layout = QVBoxLayout(self)
        batch = QGroupBox("批量操作")
        batch_layout = QHBoxLayout(batch)
        self.pick_batch = QPushButton("选择起作用对象…")
        self.pick_batch.clicked.connect(self._pick_batch)
        self.batch_label = QLabel("未选择")
        self.apply_batch_button = QPushButton("应用到选中作用区域")
        self.apply_batch_button.clicked.connect(self._apply_batch)
        batch_layout.addWidget(self.pick_batch)
        batch_layout.addWidget(self.batch_label, 1)
        batch_layout.addWidget(self.apply_batch_button)
        layout.addWidget(batch)

        self.table = HalfWidthTable(0, 2)
        self.table.setHorizontalHeaderLabels(["作用区域", "起作用对象"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(self._edit_row)
        layout.addWidget(self.table, 1)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.batch_sources = []
        self.batch_label.setText("未选择")
        self.reload()

    def _source_text(self, sources: list[int]) -> str:
        if self.project is None:
            return ""
        return ", ".join(self.project.object_display(i) for i in sources)

    def reload(self) -> None:
        self.table.setRowCount(0)
        if self.project is None:
            return
        rows = self.project.r_rows
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            region = QTableWidgetItem(
                self.project.object_display(row.region_index) if row.region_index is not None else "")
            region.setFlags(region.flags() & ~Qt.ItemIsEditable)
            region.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 0, region)
            sources = QTableWidgetItem(self._source_text(row.sources))
            sources.setFlags(sources.flags() & ~Qt.ItemIsEditable)
            sources.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 1, sources)
        self.table.fit_columns()

    def _edit_row(self, row: int, column: int) -> None:
        if self.project is None or row >= len(self.project.r_rows):
            return
        current = self.project.r_rows[row]
        dialog = SourcePickerDialog(self.project, current.sources, self)
        if dialog.exec() == QDialog.Accepted:
            sources = sorted(set(dialog.selected_indices()), key=self.project.object_key)
            current.sources = sources
            self.table.item(row, 1).setText(self._source_text(sources))

    def _pick_batch(self) -> None:
        if self.project is None:
            return
        selected_rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        initial = self.batch_sources
        if selected_rows and not initial:
            first = self.project.r_rows[selected_rows[0]]
            initial = list(first.sources)
        dialog = SourcePickerDialog(self.project, initial, self)
        if dialog.exec() == QDialog.Accepted:
            self.batch_sources = sorted(set(dialog.selected_indices()), key=self.project.object_key)
            self.batch_label.setText(
                ", ".join(self.project.object_display(i) for i in self.batch_sources) or "空")

    def _apply_batch(self) -> None:
        if self.project is None:
            return
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(self, "批量操作", "请先选中要应用的作用区域。")
            return
        for row in rows:
            if 0 <= row < len(self.project.r_rows):
                self.project.r_rows[row].sources = list(self.batch_sources)
                self.table.item(row, 1).setText(self._source_text(self.batch_sources))


class RGTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: Project | None = None
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        self.pick = QPushButton("选择起作用对象…")
        self.pick.clicked.connect(self._pick)
        row.addWidget(self.pick)
        row.addStretch(1)
        layout.addLayout(row)

        self.table = HalfWidthTable(0, 1)
        self.table.setHorizontalHeaderLabels(["全局作用对象"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.reload()

    def reload(self) -> None:
        self.table.setRowCount(0)
        if self.project is None:
            return
        sources = self.project.rg_sources
        self.table.setRowCount(len(sources))
        for row, index in enumerate(sources):
            item = QTableWidgetItem(self.project.object_display(index))
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, item)
        self.table.fit_columns()

    def _pick(self) -> None:
        if self.project is None:
            return
        dialog = SourcePickerDialog(self.project, self.project.rg_sources, self)
        if dialog.exec() == QDialog.Accepted:
            self.project.rg_sources = sorted(
                set(dialog.selected_indices()), key=self.project.object_key)
            self.reload()


def preview_function(text: str, limit: int = CUSTOM_PREVIEW_LIMIT) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


class FunctionEditorDialog(QDialog):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("编辑函数")
        self.resize(720, 460)
        layout = QVBoxLayout(self)
        self.editor = QPlainTextEdit()
        self.editor.setPlainText(text)
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.editor.setFont(font)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.editor.setTabStopDistance(self.editor.fontMetrics().horizontalAdvance(" ") * 4)
        layout.addWidget(self.editor, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        save = QPushButton("保存")
        save.setDefault(True)
        save.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)
        self.editor.setFocus()

    def text(self) -> str:
        return self.editor.toPlainText()


class FunctionHelpDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("函数填写说明")
        self.resize(860, 660)
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(help_html())
        layout.addWidget(browser, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)


class CustomTable(QTableWidget):
    """Nine-column table with a two-level header and per-row coordinate radios."""

    def __init__(self, rows: int, columns: int, parent: QWidget | None = None):
        super().__init__(rows, columns, parent)
        self._fitted: tuple[int, int] | None = None
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(28)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked |
            QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed)
        self.setWordWrap(False)
        self.viewport().installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.viewport() and event.type() == QEvent.Resize:
            self.fit_columns()
        return super().eventFilter(watched, event)

    def fit_columns(self) -> None:
        width = self.viewport().width()
        if width <= 0 or self.columnCount() != 9:
            return
        key = (width, self.columnCount())
        if key == self._fitted:
            return
        self._fitted = key
        ratios = (0.17, 0.11, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.27)
        used = 0
        for index, ratio in enumerate(ratios[:-1]):
            value = max(50, int(width * ratio))
            self.setColumnWidth(index, value)
            used += value
        self.setColumnWidth(8, max(80, width - used))


class CustomTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: Project | None = None
        self._loading = False
        self.param_columns = {
            param: 2 + index for index, param in enumerate(CUSTOM_PARAMS)
        }

        layout = QVBoxLayout(self)

        batch = QGroupBox("批量操作")
        grid = QGridLayout(batch)
        self.batch_checks: dict[str, QCheckBox] = {}
        self.batch_edits: dict[str, QLineEdit] = {}
        for index, param in enumerate(CUSTOM_PARAMS):
            label = CUSTOM_LABELS.get(param, param)
            check = QCheckBox(label)
            check.setChecked(True)
            edit = QLineEdit()
            edit.setPlaceholderText("函数…")
            edit.setMinimumWidth(120)
            pair = QWidget()
            pair_layout = QHBoxLayout(pair)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            pair_layout.setSpacing(4)
            pair_layout.addWidget(check)
            pair_layout.addWidget(edit, 1)
            grid.addWidget(pair, index // 4, index % 4)
            grid.setColumnStretch(index % 4, 1)
            self.batch_checks[param] = check
            self.batch_edits[param] = edit
        self.apply_button = QPushButton("应用到选中行")
        self.apply_button.clicked.connect(self.apply_batch)
        grid.addWidget(self.apply_button, 0, 4, 2, 1,
                       Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(batch)

        buttons = QHBoxLayout()
        self.help_button = QPushButton("函数填写说明")
        self.check_button = QPushButton("检查")
        self.new_button = QPushButton("新建")
        self.delete_button = QPushButton("删除")
        self.help_button.clicked.connect(self.show_help)
        self.check_button.clicked.connect(self.check)
        self.new_button.clicked.connect(self.new_row)
        self.delete_button.clicked.connect(self.delete_rows)
        buttons.addWidget(self.help_button)
        buttons.addWidget(self.check_button)
        buttons.addStretch(1)
        buttons.addWidget(self.new_button)
        buttons.addWidget(self.delete_button)
        layout.addLayout(buttons)

        self.table = CustomTable(0, 9)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellDoubleClicked.connect(self._edit_function)
        layout.addWidget(self.table, 1)

        self.reload()
        self._update_buttons()

    # ------------------------------------------------------------------
    def _update_buttons(self) -> None:
        has_project = self.project is not None
        self.check_button.setEnabled(has_project)
        self.new_button.setEnabled(has_project)
        self.delete_button.setEnabled(has_project)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.reload()
        self._update_buttons()

    def _rows(self) -> list[CustomRow]:
        return self.project.custom_rows if self.project is not None else []

    def _header_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        item.setFlags(Qt.ItemIsEnabled)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        palette = self.table.palette()
        item.setBackground(QBrush(palette.color(QPalette.ColorRole.Button)))
        item.setForeground(QBrush(palette.color(QPalette.ColorRole.ButtonText)))
        return item

    def _setup_header(self) -> None:
        self.table.clearSpans()
        self.table.setSpan(0, 0, 2, 1)   # 名称
        self.table.setSpan(0, 1, 2, 1)   # 坐标系
        self.table.setSpan(0, 2, 1, 3)   # 直角坐标
        self.table.setSpan(0, 5, 1, 3)   # 球坐标
        self.table.setSpan(0, 8, 2, 1)   # 作用区域
        self.table.setItem(0, 0, self._header_item("名称"))
        self.table.setItem(0, 1, self._header_item("坐标系"))
        self.table.setItem(0, 2, self._header_item("直角坐标"))
        self.table.setItem(0, 5, self._header_item("球坐标"))
        self.table.setItem(0, 8, self._header_item("作用区域"))
        for column, text in ((2, "X"), (3, "Y"), (4, "Z"),
                             (5, "R"), (6, "Theta"), (7, "Phi")):
            self.table.setItem(1, column, self._header_item(text))
        self.table.setRowHeight(0, 24)
        self.table.setRowHeight(1, 24)

    def _function_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(preview_function(text))
        item.setData(Qt.UserRole, text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        item.setTextAlignment(Qt.AlignCenter)
        item.setToolTip(text if text else "（空）双击编辑函数")
        return item

    def _set_function_item(self, row: int, column: int, text: str) -> None:
        item = self.table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, column, item)
        item.setText(preview_function(text))
        item.setData(Qt.UserRole, text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        item.setTextAlignment(Qt.AlignCenter)
        item.setToolTip(text if text else "（空）双击编辑函数")

    def _mode_widget(self, model_index: int, mode: int) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        cartesian = QRadioButton("直角")
        spherical = QRadioButton("球")
        cartesian.setChecked(mode == MODE_CARTESIAN)
        spherical.setChecked(mode == MODE_SPHERICAL)
        group = QButtonGroup(widget)
        group.setExclusive(True)
        group.addButton(cartesian, MODE_CARTESIAN)
        group.addButton(spherical, MODE_SPHERICAL)

        def on_toggled(button_id: int, checked: bool) -> None:
            if checked and not self._loading:
                self._set_row_mode(model_index, button_id)

        group.idToggled.connect(on_toggled)
        widget._button_group = group  # keep the Python wrapper alive
        layout.addWidget(cartesian)
        layout.addWidget(spherical)
        layout.addStretch(1)
        return widget

    def _theme_colors(self):
        palette = self.table.palette()
        base = palette.color(QPalette.ColorRole.Base)
        text = palette.color(QPalette.ColorRole.Text)
        inactive_bg = QColor(base)
        if inactive_bg.lightness() < 128:
            inactive_bg = inactive_bg.lighter(115)
        else:
            inactive_bg = inactive_bg.darker(105)
        inactive_fg = palette.color(
            QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        return base, text, inactive_bg, inactive_fg

    def _apply_row_style(self, row: int, mode: int) -> None:
        active = set(CARTESIAN_PARAMS if mode == MODE_CARTESIAN else SPHERICAL_PARAMS)
        base, text, inactive_bg, inactive_fg = self._theme_colors()
        for param in CUSTOM_PARAMS:
            column = self.param_columns[param]
            item = self.table.item(row, column)
            if item is None:
                continue
            text_value = item.data(Qt.UserRole)
            text_value = "" if text_value is None else str(text_value)
            is_active = param == "Domain" or param in active
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setTextAlignment(Qt.AlignCenter)
            if is_active:
                item.setBackground(QBrush(base))
                item.setForeground(QBrush(text))
                item.setToolTip(text_value if text_value else "（空）双击编辑函数")
            else:
                item.setBackground(QBrush(inactive_bg))
                item.setForeground(QBrush(inactive_fg))
                text_mode = "直角" if mode == MODE_CARTESIAN else "球"
                item.setToolTip(f"当前行使用{text_mode}坐标；此列只读")

    def _set_row_mode(self, model_index: int, mode: int) -> None:
        if self.project is None or not (0 <= model_index < len(self.project.custom_rows)):
            return
        row = self.project.custom_rows[model_index]
        if row.mode == mode:
            return
        row.mode = mode
        self._apply_row_style(model_index + 2, mode)

    def reload(self) -> None:
        rows = self._rows()
        self._loading = True
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.table.setRowCount(2 + len(rows))
        self._setup_header()
        for model_index, row in enumerate(rows):
            table_row = model_index + 2
            name_item = QTableWidgetItem(row.name)
            name_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(table_row, 0, name_item)
            self.table.setCellWidget(table_row, 1, self._mode_widget(model_index, row.mode))
            for param in CUSTOM_PARAMS:
                column = self.param_columns[param]
                self.table.setItem(table_row, column, self._function_item(row.get(param)))
            self._apply_row_style(table_row, row.mode)
        self.table.blockSignals(False)
        self._loading = False
        self.table.fit_columns()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or self.project is None:
            return
        if item.column() != 0 or item.row() < 2:
            return
        index = item.row() - 2
        if 0 <= index < len(self.project.custom_rows):
            self.project.custom_rows[index].name = item.text()

    def _edit_function(self, row: int, column: int) -> None:
        if self.project is None or row < 2 or not (2 <= column <= 8):
            return
        index = row - 2
        if not (0 <= index < len(self.project.custom_rows)):
            return
        model_row = self.project.custom_rows[index]
        param = CUSTOM_PARAMS[column - 2]
        if param != "Domain" and param not in model_row.active_params:
            return
        item = self.table.item(row, column)
        full = item.data(Qt.UserRole) if item is not None else ""
        dialog = FunctionEditorDialog("" if full is None else str(full), self)
        if dialog.exec() == QDialog.Accepted:
            value = dialog.text()
            model_row.set(param, value)
            self._set_function_item(row, column, value)
            self._apply_row_style(row, model_row.mode)

    def sync_to_project(self) -> None:
        if self.project is None:
            return
        if self.table.rowCount() != len(self.project.custom_rows) + 2:
            return
        for index, row in enumerate(self.project.custom_rows):
            table_row = index + 2
            name_item = self.table.item(table_row, 0)
            if name_item is not None:
                row.name = name_item.text()
            values = []
            for param in CUSTOM_PARAMS:
                item = self.table.item(table_row, self.param_columns[param])
                value = item.data(Qt.UserRole) if item is not None else ""
                if value is None:
                    value = item.text() if item is not None else ""
                values.append(str(value))
            row.values = values

    def _selected_model_rows(self) -> list[int]:
        rows = sorted({
            index.row() - 2
            for index in self.table.selectionModel().selectedRows()
            if index.row() >= 2
        })
        return [row for row in rows if 0 <= row < len(self._rows())]

    def _next_name(self) -> str:
        used = set()
        for row in self._rows():
            match = re.fullmatch(r"CustomGravity_(\d+)", row.name.strip())
            if match:
                used.add(int(match.group(1)))
        number = 1
        while number in used:
            number += 1
        return f"CustomGravity_{number:02d}"

    def new_row(self) -> None:
        if self.project is None:
            return
        self.project.custom_rows.append(CustomRow(self._next_name()))
        self.reload()
        self.table.selectRow(2 + len(self.project.custom_rows) - 1)
        self.table.scrollToBottom()

    def delete_rows(self) -> None:
        if self.project is None:
            return
        selected = self._selected_model_rows()
        if not selected:
            QMessageBox.information(self, "删除", "请先选中要删除的行。")
            return
        answer = QMessageBox.question(
            self, "删除",
            f"确定删除选中的 {len(selected)} 个自定义场吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        for index in sorted(selected, reverse=True):
            del self.project.custom_rows[index]
        self.reload()

    def apply_batch(self) -> None:
        if self.project is None:
            return
        selected = self._selected_model_rows()
        if not selected:
            QMessageBox.information(self, "批量操作", "请先在表格里选中要应用的行。")
            return
        params = [param for param in CUSTOM_PARAMS
                  if self.batch_checks[param].isChecked()]
        if not params:
            QMessageBox.information(self, "批量操作", "请至少勾选一个参数。")
            return
        applied = False
        for index in selected:
            row = self.project.custom_rows[index]
            for param in params:
                if param != "Domain" and param not in row.active_params:
                    continue
                value = self.batch_edits[param].text()
                row.set(param, value)
                self._set_function_item(
                    index + 2, self.param_columns[param], value)
                applied = True
            self._apply_row_style(index + 2, row.mode)
        if not applied:
            QMessageBox.information(
                self, "批量操作", "选中行的坐标系与勾选参数不匹配。")

    def show_help(self) -> None:
        FunctionHelpDialog(self).exec()

    def _mark_error(self, row: int, param: str, message: str) -> QTableWidgetItem | None:
        item = self.table.item(row, self.param_columns[param])
        if item is None:
            return None
        dark = self.table.palette().color(QPalette.ColorRole.Base).lightness() < 128
        item.setBackground(QBrush(QColor("#5a1f1f" if dark else "#ffd6d6")))
        item.setForeground(QBrush(QColor("#ffb4b4" if dark else "#b00020")))
        item.setToolTip(message)
        return item

    def check(self) -> None:
        if self.project is None:
            return
        self.sync_to_project()
        for index, row in enumerate(self.project.custom_rows):
            self._apply_row_style(index + 2, row.mode)
        errors = 0
        first: QTableWidgetItem | None = None
        for index, row in enumerate(self.project.custom_rows):
            table_row = index + 2
            active = CARTESIAN_PARAMS if row.mode == MODE_CARTESIAN else SPHERICAL_PARAMS
            allowed = CARTESIAN_VARS if row.mode == MODE_CARTESIAN else SPHERICAL_VARS
            for param in active:
                value = row.get(param)
                if not value.strip():
                    error = "函数不能为空"
                else:
                    error = check_function(value, allowed)
                if error:
                    item = self._mark_error(table_row, param, error)
                    if first is None:
                        first = item
                    errors += 1
            domain = row.get("Domain")
            if domain.strip():
                error = check_domain(domain)
                if error:
                    item = self._mark_error(table_row, "Domain", error)
                    if first is None:
                        first = item
                    errors += 1
        if errors:
            if first is not None:
                self.table.setCurrentCell(first.row(), first.column())
            QMessageBox.warning(self, "检查", "发现函数语法错误！")
        else:
            QMessageBox.information(self, "检查", "检查通过！")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1180, 780)
        self.settings = Settings()
        self.project: Project | None = None
        self.current_path: Path | None = None
        self._build()

    def _build(self) -> None:
        toolbar = self.addToolBar("主工具栏")
        toolbar.setMovable(False)
        import_action = QAction("导入 NMO", self)
        import_action.triggered.connect(self.import_nmo)
        export_action = QAction("导出 NMO", self)
        export_action.triggered.connect(self.export_nmo)
        toolbar.addAction(import_action)
        toolbar.addAction(export_action)

        self.tabs = QTabWidget()
        self.field_tabs = {name: FieldTab(name) for name in ("FG_U", "FG_M", "FG_L", "FG_S")}
        self.r_tab = RTab()
        self.rg_tab = RGTab()
        self.custom_tab = CustomTab()

        for name, tab in self.field_tabs.items():
            self.tabs.addTab(tab, FIELD_LABELS[name])
        self.tabs.addTab(self.r_tab, "R（作用区域）")
        self.tabs.addTab(self.rg_tab, "RG（全局作用）")
        self.tabs.addTab(self.custom_tab, "Custom（自定义场）")
        self.setCentralWidget(self.tabs)

        self.status_label = QLabel("就绪。导入一张分好 FG Group 的 NMO 开始。")
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.statusBar().addWidget(self.status_label, 1)

    # ------------------------------------------------------------------
    def _notify(self, text: str, timeout: int = 5000) -> None:
        """Short lived message in the status bar; the summary comes back after."""
        log_line(text)
        self.statusBar().showMessage(text, timeout)

    def _summary_html(self) -> str:
        project = self.project
        if project is None:
            return "就绪。导入一张分好 FG Group 的 NMO 开始。"
        name = self.current_path.name if self.current_path else ""
        counts = ", ".join(
            f"{group.removeprefix('FG_')}:{len(project.group_indices(group))}"
            for group in SUMMARY_GROUPS)
        counts += f", RG:{len(project.rg_sources)}"
        counts += f", Custom:{len(project.custom_rows)}"
        pieces = [
            f"文件：{html.escape(name)}",
            f"对象数：{len(project.nmo.objects)}",
            counts,
        ]
        if project.warnings:
            text = f"警告 {len(project.warnings)}：{brief_reasons(project.warnings)}"
            pieces.append(f'<span style="color:#b26a00;">{html.escape(text)}</span>')
        else:
            pieces.append("警告 0")
        if project.errors:
            text = f"错误 {len(project.errors)}：{brief_reasons(project.errors)}"
            pieces.append(f'<span style="color:#b00020;">{html.escape(text)}</span>')
        else:
            pieces.append("错误 0")
        return "&nbsp;&nbsp;&nbsp;&nbsp;".join(pieces)

    def _summary_tooltip(self) -> str:
        project = self.project
        if project is None:
            return ""
        lines = [f"文件：{self.current_path}"] if self.current_path else []
        lines.extend(f"警告：{message}" for message in project.warnings)
        lines.extend(f"错误：{message}" for message in project.errors)
        return "\n".join(lines)

    def _refresh(self) -> None:
        for tab in self.field_tabs.values():
            tab.set_project(self.project)
        self.r_tab.set_project(self.project)
        self.rg_tab.set_project(self.project)
        self.custom_tab.set_project(self.project)
        self.status_label.setText(self._summary_html())
        self.status_label.setToolTip(self._summary_tooltip())

    def import_nmo(self) -> None:
        default_dir = self.settings.get("last_import", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self, "导入 NMO", default_dir, "NMO 文件 (*.nmo);;所有文件 (*.*)")
        if not path:
            return
        try:
            project = Project.load(path)
        except Exception as exc:
            self._notify(f"导入失败：{exc}", timeout=8000)
            QMessageBox.critical(self, "导入失败", str(exc))
            log_line(traceback.format_exc())
            return
        self.project = project
        self.current_path = Path(path)
        self.settings.set("last_import", str(Path(path).parent))
        self._refresh()
        self.tabs.setCurrentIndex(0)
        for message in project.warnings:
            log_line(f"警告：{message}")
        for message in project.errors:
            log_line(f"错误：{message}")
        log_line(f"导入完成：{Path(path).name}；"
                 f"警告 {len(project.warnings)}，错误 {len(project.errors)}")

    def _sync(self) -> None:
        if self.project is None:
            return
        for tab in self.field_tabs.values():
            tab.sync_to_project()
        self.custom_tab.sync_to_project()

    def export_nmo(self) -> None:
        if self.project is None:
            QMessageBox.information(self, "导出", "请先导入一张 NMO。")
            return
        self._sync()
        if not self.project.can_export():
            self._notify("导出失败：导入时存在 array 格式错误，请先处理。", timeout=8000)
            return
        default_dir = self.settings.get("last_export", str(self.current_path.parent if self.current_path else Path.home()))
        default_name = self.current_path.name if self.current_path else "output.nmo"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 NMO", str(Path(default_dir) / default_name), "NMO 文件 (*.nmo);;所有文件 (*.*)")
        if not path:
            return
        try:
            self.project.export(path, update_level=False)
        except Exception as exc:
            self._notify(f"导出失败：{exc}", timeout=8000)
            QMessageBox.critical(self, "导出失败", str(exc))
            log_line(traceback.format_exc())
            return
        self.settings.set("last_export", str(Path(path).parent))
        self._notify("导出成功。")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    smoke = "--smoke-test" in args
    args = [arg for arg in args if arg != "--smoke-test"]
    app = QApplication(args)
    app.setApplicationName("FGEditor")
    app.setApplicationVersion(__version__)
    icon = asset_path("FGEditor.ico")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    app.setFont(QFont("Microsoft YaHei UI", 9))
    window = MainWindow()
    window.show()
    log_line("FGEditor started" + (" (smoke test)" if smoke else ""))
    if smoke:
        QTimer.singleShot(500, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
