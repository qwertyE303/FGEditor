"""PySide6 GUI for the FlexGravity NMO editor (v2.1 schema)."""

from __future__ import annotations

import html
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, QRect, QSize, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPalette,
)
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
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
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
    domain_help_html,
    help_html,
)
from .model import (
    CARTESIAN_PARAMS,
    CFIELD_PARAMS,
    FIELD_LABELS,
    FIELD_PARAMS,
    MODE_CARTESIAN,
    MODE_SPHERICAL,
    SPHERICAL_PARAMS,
    CFieldRow,
    CRegionRow,
    Project,
    Ref,
)

APP_TITLE = f"FGEditor - v{__version__}"
EMPTY_OBJECT = ""
ELLIPSIS = "…"
CUSTOM_PREVIEW_LIMIT = 24
MAX_MEASURED_CHARS = 4096


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def asset_path(name: str) -> Path:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        return base / name
    path = Path(__file__).resolve().parents[2] / "assets" / name
    if path.exists():
        return path
    fallback = Path(__file__).resolve().parents[3] / "build" / "gen" / name
    return fallback if fallback.exists() else path


def brief_reason(message: str, limit: int = 26) -> str:
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


def single_line_text(text: str) -> str:
    """Collapse whitespace so expressions stay on one table line."""
    return " ".join(str(text).split())


def preview_text(text: str, limit: int = CUSTOM_PREVIEW_LIMIT) -> str:
    """Short item text so Qt size hints stay cheap for huge expressions."""
    collapsed = single_line_text(text)
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + ELLIPSIS


def measured_width(fm: QFontMetrics, text: str) -> int:
    """Width of ``text``; huge strings are treated as definitely too wide."""
    if len(text) > MAX_MEASURED_CHARS:
        return 10 ** 9
    return fm.horizontalAdvance(text)


def fit_prefix(text: str, fm: QFontMetrics, width: int) -> str:
    """Longest prefix of ``text`` that fits ``width``, without an ellipsis."""
    if width <= 0:
        return ""
    sample = text[:MAX_MEASURED_CHARS]
    low, high = 0, len(sample)
    while low < high:
        middle = (low + high + 1) // 2
        if fm.horizontalAdvance(sample[:middle]) <= width:
            low = middle
        else:
            high = middle - 1
    return sample[:low].rstrip()


def elide_text(text: str, fm: QFontMetrics, width: int) -> str:
    if width <= 0:
        return ""
    if measured_width(fm, text) <= width:
        return text
    ellipsis_width = fm.horizontalAdvance(ELLIPSIS)
    if width <= ellipsis_width:
        return ELLIPSIS
    return fit_prefix(text, fm, width - ellipsis_width) + ELLIPSIS


# A segment is (text, is_custom, atomic).  ``atomic`` segments (the four-space
# type marker and the ", " separator) are never cut in half; a source name is
# only cut when it does not fit on its own.
Segment = tuple[str, bool, bool]


def ref_segments(project: Project, ref: Ref | None, *,
                 include_builtin: bool = True) -> list[Segment]:
    if ref is None:
        return []
    name = single_line_text(project.ref_display(ref))
    segments: list[Segment] = [(name, False, False)] if name else []
    kind = project.ref_type_label(ref)
    if kind and (include_builtin or kind == "C"):
        segments.append((f"    [{kind}]", kind == "C", True))
    return segments


def region_segments(project: Project, ref: Ref | None) -> list[Segment]:
    return ref_segments(project, ref, include_builtin=False)


def source_segments(project: Project, sources: list[Ref]) -> list[Segment]:
    segments: list[Segment] = []
    for index, ref in enumerate(sources):
        if index:
            segments.append((", ", False, True))
        segments.extend(ref_segments(project, ref))
    return segments


def source_tooltip(project: Project, sources: list[Ref]) -> str:
    if not sources:
        return "（无）"
    labels = [project.ref_label(ref) for ref in sources]
    return f"共 {len(labels)} 个：\n" + "\n".join(labels)


def paint_segments(painter: QPainter, rect: QRect, opt: QStyleOptionViewItem,
                   segments: list[Segment]) -> None:
    """Paint one line of rich segments, eliding at the available width."""
    if not segments:
        return
    fm = QFontMetrics(opt.font)
    selected = bool(opt.state & QStyle.State_Selected)
    role = (
        QPalette.ColorRole.HighlightedText
        if selected
        else QPalette.ColorRole.Text
    )
    normal = opt.palette.color(role)
    custom = QColor("#d4a017")

    total = sum(measured_width(fm, text) for text, _custom, _atomic in segments)
    ops: list[tuple[str, bool]] = []
    used = 0
    if total <= rect.width():
        for text, is_custom, _atomic in segments:
            ops.append((text, is_custom))
            used += fm.horizontalAdvance(text)
    else:
        ellipsis_width = fm.horizontalAdvance(ELLIPSIS)
        budget = rect.width() - ellipsis_width
        if budget <= 0:
            ops.append((ELLIPSIS, False))
            used = ellipsis_width
        else:
            for text, is_custom, atomic in segments:
                width = measured_width(fm, text)
                if width <= budget:
                    ops.append((text, is_custom))
                    used += fm.horizontalAdvance(text)
                    budget -= width
                    continue
                if not atomic:
                    prefix = fit_prefix(text, fm, budget)
                    if prefix:
                        ops.append((prefix, is_custom))
                        used += fm.horizontalAdvance(prefix)
                ops.append((ELLIPSIS, False))
                used += ellipsis_width
                break

    align = opt.displayAlignment
    if align & Qt.AlignmentFlag.AlignRight:
        x = rect.left() + max(0, rect.width() - used)
    elif align & Qt.AlignmentFlag.AlignHCenter:
        x = rect.left() + max(0, (rect.width() - used) // 2)
    else:
        x = rect.left()
    baseline = rect.top() + (rect.height() + fm.ascent() - fm.descent()) // 2
    painter.save()
    painter.setClipRect(rect)
    painter.setFont(opt.font)
    for text, is_custom in ops:
        painter.setPen(custom if is_custom else normal)
        painter.drawText(x, baseline, text)
        x += fm.horizontalAdvance(text)
    painter.restore()


def text_segments(text: str) -> list[Segment]:
    text = single_line_text(text)
    return [(text, False, False)] if text else []


class ElidedSegmentsDelegate(QStyledItemDelegate):
    """Single-line rich-text cell; ellipsizes instead of wrapping/overflowing."""

    def sizeHint(self, option, index) -> QSize:
        hint = super().sizeHint(option, index)
        widget = option.widget
        if widget is not None and hasattr(widget, "viewport"):
            viewport = widget.viewport()
            if viewport is not None and viewport.width() > 0:
                return QSize(viewport.width(), hint.height())
        return hint

    def paint(self, painter, option, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        segments = index.data(Qt.UserRole + 2)
        if not segments:
            segments = text_segments(opt.text)
        opt.text = ""
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        if not segments:
            return
        rect = QRect(opt.rect)
        if index.data(Qt.CheckStateRole) is not None:
            width = style.pixelMetric(QStyle.PM_IndicatorWidth, opt, opt.widget) + 8
            rect.setLeft(rect.left() + width)
        rect.adjust(4, 0, -4, 0)
        if rect.width() <= 0:
            return
        painter.save()
        painter.setClipRect(opt.rect)
        paint_segments(painter, rect, opt, segments)
        painter.restore()


class ElidedLabel(QLabel):
    """QLabel that never widens the layout; long text gets an ellipsis."""

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def sizeHint(self) -> QSize:
        return QSize(0, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        fm = self.fontMetrics()
        rect = self.contentsRect()
        text = elide_text(self.text(), fm, rect.width())
        group = (
            QPalette.ColorGroup.Disabled
            if not self.isEnabled()
            else QPalette.ColorGroup.Active
        )
        painter.setPen(self.palette().color(group, self.foregroundRole()))
        painter.drawText(rect, int(self.alignment()), text)


class DomainLineEdit(QLineEdit):
    """Batch domain input; double-click opens the large editor."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.on_double_click = None

    def mouseDoubleClickEvent(self, event) -> None:
        if self.on_double_click is not None:
            self.on_double_click()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)



class CheckListWidget(QListWidget):
    """Checkbox list with Shift range toggling."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._anchor: int | None = None

    def mousePressEvent(self, event) -> None:
        if event.modifiers() & Qt.ShiftModifier and self._anchor is not None:
            row = self.row(self.itemAt(event.position().toPoint()))
            if row >= 0:
                state = (
                    Qt.Unchecked
                    if self.item(row).checkState() == Qt.Checked
                    else Qt.Checked
                )
                start, end = sorted((self._anchor, row))
                for index in range(start, end + 1):
                    self.item(index).setCheckState(state)
                self._anchor = row
                event.accept()
                return
        super().mousePressEvent(event)
        row = self.row(self.itemAt(event.position().toPoint()))
        if row >= 0:
            self._anchor = row


class SourcePickerDialog(QDialog):
    def __init__(self, project: Project, selected: list[Ref], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("选择起作用对象")
        self.resize(560, 560)
        self.project = project
        self.options = project.source_options()
        self.selected = set(selected)

        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索对象名…")
        layout.addWidget(self.search)
        self.list = CheckListWidget()
        self.list.setItemDelegate(ElidedSegmentsDelegate(self.list))
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setWordWrap(False)
        layout.addWidget(self.list, 1)

        for option in self.options:
            label = project.ref_label(option)
            item = QListWidgetItem(single_line_text(label))
            item.setData(Qt.UserRole, self.options.index(option))
            item.setData(Qt.UserRole + 2, ref_segments(project, option))
            item.setToolTip(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if option in self.selected else Qt.Unchecked)
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

    def selected_refs(self) -> list[Ref]:
        result = []
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.checkState() == Qt.Checked:
                result.append(self.options[int(item.data(Qt.UserRole))])
        return result


class HalfWidthTable(QTableWidget):
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
        self.spins = []
        self.checks = []
        for param in self.params:
            if name == "FG_UField":
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
                object_item = QTableWidgetItem(self.project.object_display(row.object_index))
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
        if self._loading or self.project is None or item.column() == 0:
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


class CFieldTable(QTableWidget):
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
        if width <= 0 or self.columnCount() != 8:
            return
        key = (width, self.columnCount())
        if key == self._fitted:
            return
        self._fitted = key
        name_width = max(100, int(width * 0.18))
        mode_width = max(70, int(width * 0.11))
        rest = max(6 * 40, width - name_width - mode_width)
        share = rest // 6
        self.setColumnWidth(0, name_width)
        self.setColumnWidth(1, mode_width)
        for index in range(2, 8):
            self.setColumnWidth(index, share)


class CFieldTab(QWidget):
    def __init__(self, on_name_changed=None, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: Project | None = None
        self._loading = False
        self.on_name_changed = on_name_changed
        self.param_columns = {param: 2 + index for index, param in enumerate(CFIELD_PARAMS)}

        layout = QVBoxLayout(self)
        batch = QGroupBox("批量操作")
        grid = QGridLayout(batch)
        self.batch_checks = {}
        self.batch_edits = {}
        for index, param in enumerate(CFIELD_PARAMS):
            check = QCheckBox(param)
            check.setChecked(True)
            edit = DomainLineEdit()
            edit.setToolTip("双击打开大编辑框")
            edit.on_double_click = lambda p=param, e=edit: self._edit_batch_function(p, e)
            edit.setMinimumWidth(110)
            pair = QWidget()
            pair_layout = QHBoxLayout(pair)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            pair_layout.setSpacing(4)
            pair_layout.addWidget(check)
            pair_layout.addWidget(edit, 1)
            grid.addWidget(pair, index // 3, index % 3)
            grid.setColumnStretch(index % 3, 1)
            self.batch_checks[param] = check
            self.batch_edits[param] = edit
        self.apply_button = QPushButton("应用到选中行")
        self.apply_button.clicked.connect(self.apply_batch)
        grid.addWidget(self.apply_button, 0, 3, 2, 1, Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(batch)

        buttons = QHBoxLayout()
        self.help_button = QPushButton("场函数填写说明")
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

        self.table = CFieldTable(0, 8)
        self.elided_delegate = ElidedSegmentsDelegate(self.table)
        for column in range(2, 8):
            self.table.setItemDelegateForColumn(column, self.elided_delegate)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellDoubleClicked.connect(self._edit_function)
        layout.addWidget(self.table, 1)
        self.reload()

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.reload()

    def _rows(self) -> list[CFieldRow]:
        return self.project.cfield_rows if self.project is not None else []

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
        self.table.setSpan(0, 0, 2, 1)
        self.table.setSpan(0, 1, 2, 1)
        self.table.setSpan(0, 2, 1, 3)
        self.table.setSpan(0, 5, 1, 3)
        self.table.setItem(0, 0, self._header_item("名称"))
        self.table.setItem(0, 1, self._header_item("坐标系"))
        self.table.setItem(0, 2, self._header_item("直角坐标"))
        self.table.setItem(0, 5, self._header_item("球坐标"))
        for column, text in ((2, "X"), (3, "Y"), (4, "Z"), (5, "R"), (6, "Theta"), (7, "Phi")):
            self.table.setItem(1, column, self._header_item(text))
        self.table.setRowHeight(0, 24)
        self.table.setRowHeight(1, 24)

    def _function_item(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(preview_text(text))
        item.setData(Qt.UserRole, text)
        item.setData(Qt.UserRole + 2, text_segments(text))
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        item.setTextAlignment(Qt.AlignCenter)
        item.setToolTip(text if text else "（空）双击编辑函数")
        return item

    def _set_function_item(self, row: int, column: int, text: str) -> None:
        item = self.table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, column, item)
        item.setText(preview_text(text))
        item.setData(Qt.UserRole, text)
        item.setData(Qt.UserRole + 2, text_segments(text))
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
        widget._button_group = group
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
        inactive_fg = palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        return base, text, inactive_bg, inactive_fg

    def _apply_row_style(self, row: int, mode: int) -> None:
        active = set(CARTESIAN_PARAMS if mode == MODE_CARTESIAN else SPHERICAL_PARAMS)
        base, text, inactive_bg, inactive_fg = self._theme_colors()
        for param in CFIELD_PARAMS:
            column = self.param_columns[param]
            item = self.table.item(row, column)
            if item is None:
                continue
            text_value = item.data(Qt.UserRole)
            text_value = "" if text_value is None else str(text_value)
            is_active = param in active
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setTextAlignment(Qt.AlignCenter)
            if is_active:
                item.setBackground(QBrush(base))
                item.setForeground(QBrush(text))
                item.setToolTip(text_value if text_value else "（空）双击编辑函数")
            else:
                item.setBackground(QBrush(inactive_bg))
                item.setForeground(QBrush(inactive_fg))
                item.setToolTip("当前行未使用这组坐标；此列只读")

    def _set_row_mode(self, model_index: int, mode: int) -> None:
        if self.project is None or not (0 <= model_index < len(self.project.cfield_rows)):
            return
        row = self.project.cfield_rows[model_index]
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
            for param in CFIELD_PARAMS:
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
        if 0 <= index < len(self.project.cfield_rows):
            self.project.cfield_rows[index].name = item.text()
            if self.on_name_changed is not None:
                QTimer.singleShot(0, self.on_name_changed)

    def _clear_cell_error(self, row: int, column: int) -> None:
        item = self.table.item(row, column)
        if item is None:
            return
        previous = self.table.blockSignals(True)
        try:
            item.setData(Qt.BackgroundRole, None)
            item.setData(Qt.ForegroundRole, None)
            item.setBackground(QBrush())
            item.setForeground(QBrush(self.table.palette().color(QPalette.ColorRole.Text)))
        finally:
            self.table.blockSignals(previous)

    def _edit_batch_function(self, param: str, edit: QLineEdit) -> None:
        dialog = FunctionEditorDialog(edit.text(), self, "编辑函数")
        if dialog.exec() == QDialog.Accepted:
            edit.setText(dialog.text())

    def _edit_function(self, row: int, column: int) -> None:
        if self.project is None or row < 2 or not (2 <= column <= 7):
            return
        index = row - 2
        if not (0 <= index < len(self.project.cfield_rows)):
            return
        model_row = self.project.cfield_rows[index]
        param = CFIELD_PARAMS[column - 2]
        if param not in model_row.active_params:
            return
        item = self.table.item(row, column)
        full = item.data(Qt.UserRole) if item is not None else ""
        dialog = FunctionEditorDialog("" if full is None else str(full), self, "编辑函数")
        if dialog.exec() == QDialog.Accepted:
            value = dialog.text()
            model_row.set(param, value)
            self._set_function_item(row, column, value)

    def sync_to_project(self) -> None:
        if self.project is None:
            return
        if self.table.rowCount() != len(self.project.cfield_rows) + 2:
            return
        for index, row in enumerate(self.project.cfield_rows):
            table_row = index + 2
            name_item = self.table.item(table_row, 0)
            if name_item is not None:
                row.name = name_item.text()
            values = []
            for param in CFIELD_PARAMS:
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

    def new_row(self) -> None:
        if self.project is None:
            return
        self.project.new_cfield()
        self.reload()
        self.table.selectRow(2 + len(self.project.cfield_rows) - 1)
        self.table.scrollToBottom()
        if self.on_name_changed is not None:
            self.on_name_changed()

    def delete_rows(self) -> None:
        if self.project is None:
            return
        selected = self._selected_model_rows()
        if not selected:
            QMessageBox.information(self, "删除", "请先选中要删除的行。")
            return
        answer = QMessageBox.question(
            self, "删除", f"确定删除选中的 {len(selected)} 个自定义场吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        for index in sorted(selected, reverse=True):
            self.project.remove_cfield(self.project.cfield_rows[index])
        self.reload()
        if self.on_name_changed is not None:
            self.on_name_changed()

    def apply_batch(self) -> None:
        if self.project is None:
            return
        selected = self._selected_model_rows()
        if not selected:
            QMessageBox.information(self, "批量操作", "请先在表格里选中要应用的行。")
            return
        params = [param for param in CFIELD_PARAMS if self.batch_checks[param].isChecked()]
        if not params:
            QMessageBox.information(self, "批量操作", "请至少勾选一个参数。")
            return
        applied = False
        for index in selected:
            row = self.project.cfield_rows[index]
            for param in params:
                if param not in row.active_params:
                    continue
                value = self.batch_edits[param].text()
                row.set(param, value)
                self._set_function_item(index + 2, self.param_columns[param], value)
                applied = True
        if not applied:
            QMessageBox.information(self, "批量操作", "选中行的坐标系与勾选参数不匹配。")

    def show_help(self) -> None:
        HelpDialog("场函数填写说明", help_html(), self).exec()

    def _mark_error(self, row: int, column: int, message: str) -> QTableWidgetItem | None:
        item = self.table.item(row, column)
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
        for table_row in range(2, self.table.rowCount()):
            self._clear_cell_error(table_row, 0)
            for column in range(2, 8):
                self._clear_cell_error(table_row, column)
        for index, row in enumerate(self.project.cfield_rows):
            self._apply_row_style(index + 2, row.mode)
        errors = 0
        first = None
        name_counts: dict[str, int] = {}
        for row in self.project.cfield_rows:
            name_counts[row.name.casefold()] = name_counts.get(row.name.casefold(), 0) + 1
        for index, row in enumerate(self.project.cfield_rows):
            table_row = index + 2
            name = row.name.strip()
            if not name:
                item = self._mark_error(table_row, 0, "名称不能为空")
                first = first or item
                errors += 1
            elif name_counts.get(name.casefold(), 0) > 1:
                item = self._mark_error(table_row, 0, "名称重复")
                first = first or item
                errors += 1
            active = CARTESIAN_PARAMS if row.mode == MODE_CARTESIAN else SPHERICAL_PARAMS
            allowed = CARTESIAN_VARS if row.mode == MODE_CARTESIAN else SPHERICAL_VARS
            for param in active:
                value = row.get(param)
                error = "函数不能为空" if not value.strip() else check_function(value, allowed)
                if error:
                    tooltip = error
                    if value.strip():
                        tooltip += f"\n\n完整函数：\n{value}"
                    item = self._mark_error(table_row, self.param_columns[param], tooltip)
                    first = first or item
                    errors += 1
        if errors:
            if first is not None:
                self.table.setCurrentCell(first.row(), first.column())
                self.table.clearSelection()
            QMessageBox.warning(self, "检查", "发现函数语法错误！")
        else:
            QMessageBox.information(self, "检查", "检查通过！")


class RegionTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: Project | None = None
        layout = QVBoxLayout(self)
        self.table = HalfWidthTable(0, 1)
        self.table.setHorizontalHeaderLabels(["作用区域"])
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
        rows = self.project.region_indices
        self.table.setRowCount(len(rows))
        for r, index in enumerate(rows):
            item = QTableWidgetItem(self.project.object_display(index))
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(r, 0, item)
        self.table.fit_columns()


class CRegionTab(QWidget):
    def __init__(self, on_name_changed=None, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: Project | None = None
        self._loading = False
        self.on_name_changed = on_name_changed
        layout = QVBoxLayout(self)

        batch = QGroupBox("批量操作")
        batch_layout = QHBoxLayout(batch)
        batch_layout.addWidget(QLabel("定义域"))
        self.batch_edit = DomainLineEdit()
        self.batch_edit.setToolTip("双击打开大编辑框")
        self.batch_edit.on_double_click = self._edit_batch_domain
        batch_layout.addWidget(self.batch_edit, 1)
        self.apply_button = QPushButton("应用到选中行")
        self.apply_button.clicked.connect(self.apply_batch)
        batch_layout.addWidget(self.apply_button)
        layout.addWidget(batch)

        buttons = QHBoxLayout()
        self.help_button = QPushButton("作用域填写说明")
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

        self.table = HalfWidthTable(0, 2)
        self.table.setHorizontalHeaderLabels(["名称", "定义域"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked |
            QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed)
        self.table.verticalHeader().setVisible(False)
        self.domain_delegate = ElidedSegmentsDelegate(self.table)
        self.table.setItemDelegateForColumn(1, self.domain_delegate)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellDoubleClicked.connect(self._edit_domain)
        layout.addWidget(self.table, 1)
        self.reload()

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.reload()

    def reload(self) -> None:
        self._loading = True
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if self.project is not None:
            rows = self.project.cregion_rows
            self.table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                name_item = QTableWidgetItem(row.name)
                name_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, 0, name_item)
                domain_item = QTableWidgetItem(preview_text(row.domain))
                domain_item.setData(Qt.UserRole, row.domain)
                domain_item.setData(Qt.UserRole + 2, text_segments(row.domain))
                domain_item.setFlags(domain_item.flags() & ~Qt.ItemIsEditable)
                domain_item.setTextAlignment(Qt.AlignCenter)
                domain_item.setToolTip(row.domain if row.domain else "（空）双击编辑定义域")
                self.table.setItem(r, 1, domain_item)
        self.table.blockSignals(False)
        self._loading = False
        self.table.fit_columns()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or self.project is None or item.column() != 0:
            return
        index = item.row()
        if 0 <= index < len(self.project.cregion_rows):
            self.project.cregion_rows[index].name = item.text()
            if self.on_name_changed is not None:
                QTimer.singleShot(0, self.on_name_changed)

    def _edit_batch_domain(self) -> None:
        dialog = FunctionEditorDialog(self.batch_edit.text(), self, "编辑作用域")
        if dialog.exec() == QDialog.Accepted:
            self.batch_edit.setText(dialog.text())

    def _clear_cell_error(self, row: int, column: int) -> None:
        item = self.table.item(row, column)
        if item is None:
            return
        previous = self.table.blockSignals(True)
        try:
            item.setData(Qt.BackgroundRole, None)
            item.setData(Qt.ForegroundRole, None)
            item.setBackground(QBrush())
            item.setForeground(QBrush(self.table.palette().color(QPalette.ColorRole.Text)))
        finally:
            self.table.blockSignals(previous)

    def _edit_domain(self, row: int, column: int) -> None:
        if self.project is None or column != 1 or not (0 <= row < len(self.project.cregion_rows)):
            return
        model_row = self.project.cregion_rows[row]
        dialog = FunctionEditorDialog(model_row.domain, self, "编辑作用域")
        if dialog.exec() == QDialog.Accepted:
            model_row.domain = dialog.text()
            item = self.table.item(row, 1)
            if item is not None:
                item.setText(preview_text(model_row.domain))
                item.setData(Qt.UserRole, model_row.domain)
                item.setData(Qt.UserRole + 2, text_segments(model_row.domain))
                item.setToolTip(model_row.domain if model_row.domain else "（空）双击编辑定义域")


    def _selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.table.selectionModel().selectedRows()})

    def new_row(self) -> None:
        if self.project is None:
            return
        self.project.new_cregion()
        self.reload()
        self.table.selectRow(len(self.project.cregion_rows) - 1)
        self.table.scrollToBottom()
        if self.on_name_changed is not None:
            self.on_name_changed()

    def delete_rows(self) -> None:
        if self.project is None:
            return
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "删除", "请先选中要删除的行。")
            return
        answer = QMessageBox.question(
            self, "删除", f"确定删除选中的 {len(rows)} 个自定义作用域吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        for index in sorted(rows, reverse=True):
            self.project.remove_cregion(self.project.cregion_rows[index])
        self.reload()
        if self.on_name_changed is not None:
            self.on_name_changed()

    def apply_batch(self) -> None:
        if self.project is None:
            return
        rows = self._selected_rows()
        if not rows:
            QMessageBox.information(self, "批量操作", "请先选中要应用的行。")
            return
        value = self.batch_edit.text()
        for row in rows:
            if 0 <= row < len(self.project.cregion_rows):
                self.project.cregion_rows[row].domain = value
                item = self.table.item(row, 1)
                if item is not None:
                    item.setText(preview_text(value))
                    item.setData(Qt.UserRole, value)
                    item.setData(Qt.UserRole + 2, text_segments(value))
                    item.setToolTip(value if value else "（空）双击编辑定义域")


    def show_help(self) -> None:
        HelpDialog("作用域填写说明", domain_help_html(), self).exec()

    def _mark_error(self, row: int, column: int, message: str) -> QTableWidgetItem | None:
        item = self.table.item(row, column)
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
        for row in range(self.table.rowCount()):
            self._clear_cell_error(row, 0)
            self._clear_cell_error(row, 1)
        for index, row in enumerate(self.project.cregion_rows):
            item = self.table.item(index, 1)
            if item is not None:
                item.setToolTip(row.domain if row.domain else "（空）双击编辑定义域")
        counts: dict[str, int] = {}
        for row in self.project.cregion_rows:
            counts[row.name.casefold()] = counts.get(row.name.casefold(), 0) + 1
        errors = 0
        first = None
        for index, row in enumerate(self.project.cregion_rows):
            name = row.name.strip()
            if not name:
                item = self._mark_error(index, 0, "名称不能为空")
                first = first or item
                errors += 1
            elif counts.get(name.casefold(), 0) > 1:
                item = self._mark_error(index, 0, "名称重复")
                first = first or item
                errors += 1
            error = check_domain(row.domain)
            if error:
                tooltip = error
                if row.domain.strip():
                    tooltip += f"\n\n完整定义域：\n{row.domain}"
                item = self._mark_error(index, 1, tooltip)
                first = first or item
                errors += 1
        if errors:
            if first is not None:
                self.table.setCurrentCell(first.row(), first.column())
                self.table.clearSelection()
            QMessageBox.warning(self, "检查", "发现语法错误！")
        else:
            QMessageBox.information(self, "检查", "检查通过！")


class LocalTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.project: Project | None = None
        self.batch_sources: list[Ref] = []
        layout = QVBoxLayout(self)
        batch = QGroupBox("批量操作")
        batch_layout = QHBoxLayout(batch)
        self.pick_batch = QPushButton("选择起作用对象…")
        self.pick_batch.clicked.connect(self._pick_batch)
        self.batch_label = ElidedLabel("未选择")
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
        self.table.setItemDelegate(ElidedSegmentsDelegate(self.table))
        self.table.cellDoubleClicked.connect(self._edit_row)
        layout.addWidget(self.table, 1)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.batch_sources = []
        self._update_batch_label()
        self.reload()

    def _rows(self):
        if self.project is None:
            return []
        return self.project.local_rows + self.project.local_orphans

    def _source_text(self, sources: list[Ref]) -> str:
        if self.project is None:
            return ""
        return ", ".join(single_line_text(self.project.ref_display(ref)) for ref in sources)

    def _update_batch_label(self) -> None:
        if self.project is None or not self.batch_sources:
            self.batch_label.setText("未选择")
            self.batch_label.setToolTip("")
            return
        names = ", ".join(self.project.ref_display(ref) for ref in self.batch_sources)
        self.batch_label.setText(f"已选择 {len(self.batch_sources)} 个：{names}")
        self.batch_label.setToolTip(source_tooltip(self.project, self.batch_sources))

    def reload(self) -> None:
        self.table.setRowCount(0)
        if self.project is None:
            return
        rows = self._rows()
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            region_text = single_line_text(self.project.ref_display(row.region))
            region_parts = region_segments(self.project, row.region)
            region_tooltip = self.project.ref_label(row.region)
            if row.orphan:
                region_text += "（额外行）"
                region_parts = list(region_parts) + [("（额外行）", False, False)]
                region_tooltip += "（额外行）"
            region = QTableWidgetItem(region_text)
            region.setData(Qt.UserRole + 2, region_parts)
            region.setToolTip(region_tooltip)
            region.setFlags(region.flags() & ~Qt.ItemIsEditable)
            region.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 0, region)
            sources_text = self._source_text(row.sources)
            sources = QTableWidgetItem(sources_text)
            sources.setData(Qt.UserRole + 2, source_segments(self.project, row.sources))
            sources.setToolTip(source_tooltip(self.project, row.sources))
            sources.setFlags(sources.flags() & ~Qt.ItemIsEditable)
            sources.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 1, sources)
        self.table.fit_columns()

    def _edit_row(self, row: int, column: int) -> None:
        if self.project is None:
            return
        rows = self._rows()
        if not (0 <= row < len(rows)):
            return
        current = rows[row]
        dialog = SourcePickerDialog(self.project, current.sources, self)
        if dialog.exec() == QDialog.Accepted:
            current.sources = sorted(set(dialog.selected_refs()), key=self.project.ref_sort_key)
            self.reload()

    def _pick_batch(self) -> None:
        if self.project is None:
            return
        dialog = SourcePickerDialog(self.project, self.batch_sources, self)
        if dialog.exec() == QDialog.Accepted:
            self.batch_sources = sorted(set(dialog.selected_refs()), key=self.project.ref_sort_key)
            self._update_batch_label()

    def _apply_batch(self) -> None:
        if self.project is None:
            return
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        if not rows:
            QMessageBox.information(self, "批量操作", "请先选中要应用的作用区域。")
            return
        all_rows = self._rows()
        for row in rows:
            if 0 <= row < len(all_rows):
                all_rows[row].sources = list(self.batch_sources)
        self.reload()


class GlobalTab(QWidget):
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
        self.table.setItemDelegate(ElidedSegmentsDelegate(self.table))
        layout.addWidget(self.table, 1)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.reload()

    def reload(self) -> None:
        self.table.setRowCount(0)
        if self.project is None:
            return
        sources = self.project.global_sources
        self.table.setRowCount(len(sources))
        for row, ref in enumerate(sources):
            label = self.project.ref_label(ref)
            item = QTableWidgetItem(single_line_text(label))
            item.setData(Qt.UserRole + 2, ref_segments(self.project, ref))
            item.setToolTip(label)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)
        self.table.fit_columns()

    def _pick(self) -> None:
        if self.project is None:
            return
        dialog = SourcePickerDialog(self.project, self.project.global_sources, self)
        if dialog.exec() == QDialog.Accepted:
            self.project.global_sources = sorted(
                set(dialog.selected_refs()), key=self.project.ref_sort_key)
            self.reload()


class FunctionEditorDialog(QDialog):
    def __init__(self, text: str = "", parent: QWidget | None = None, title: str = "编辑函数"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 480)
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


class HelpDialog(QDialog):
    def __init__(self, title: str, page: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 700)
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(page)
        layout.addWidget(browser, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1220, 820)
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

        self.field_tabs = {name: FieldTab(name) for name in FIELD_PARAMS}
        self.cfield_tab = CFieldTab(self._pairing_changed)
        self.region_tab = RegionTab()
        self.cregion_tab = CRegionTab(self._pairing_changed)
        self.local_tab = LocalTab()
        self.global_tab = GlobalTab()

        setup_tabs = QTabWidget()
        for name, tab in self.field_tabs.items():
            setup_tabs.addTab(tab, FIELD_LABELS[name])
        setup_tabs.addTab(self.cfield_tab, "CF（自定义场）")
        setup_tabs.addTab(self.region_tab, "R（作用域）")
        setup_tabs.addTab(self.cregion_tab, "CR（自定义作用域）")

        pairing_tabs = QTabWidget()
        pairing_tabs.addTab(self.local_tab, "Local（局部作用）")
        pairing_tabs.addTab(self.global_tab, "Global（全局作用）")

        self.tabs = QTabWidget()
        setup_page = QWidget()
        setup_layout = QVBoxLayout(setup_page)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.addWidget(setup_tabs)
        pairing_page = QWidget()
        pairing_layout = QVBoxLayout(pairing_page)
        pairing_layout.setContentsMargins(0, 0, 0, 0)
        pairing_layout.addWidget(pairing_tabs)
        self.tabs.addTab(setup_page, "场、作用域设置")
        self.tabs.addTab(pairing_page, "场-作用域配对")
        self.setCentralWidget(self.tabs)

        self.status_label = QLabel("就绪。导入一张分好新 FG Group 的 NMO 开始。")
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.statusBar().addWidget(self.status_label, 1)

    def _notify(self, text: str, timeout: int = 5000) -> None:
        log_line(text)
        self.statusBar().showMessage(text, timeout)

    def _pairing_changed(self) -> None:
        self.local_tab.reload()
        self.global_tab.reload()

    def _summary_html(self) -> str:
        project = self.project
        if project is None:
            return "就绪。导入一张分好新 FG Group 的 NMO 开始。"
        name = self.current_path.name if self.current_path else ""
        counts = ", ".join(
            f"{key.replace('FG_', '').replace('Field', '')}:{len(project.fields[key].rows)}"
            for key in project.fields
        )
        counts += f", CF:{len(project.cfield_rows)}, R:{len(project.region_indices)}, CR:{len(project.cregion_rows)}"
        local_pairs = sum(
            len(row.sources) for row in project.local_rows + project.local_orphans
        )
        counts += f", Local:{local_pairs}, Global:{len(project.global_sources)}"
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
        self.cfield_tab.set_project(self.project)
        self.region_tab.set_project(self.project)
        self.cregion_tab.set_project(self.project)
        self.local_tab.set_project(self.project)
        self.global_tab.set_project(self.project)
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
        for message in project.warnings:
            log_line(f"警告：{message}")
        for message in project.errors:
            log_line(f"错误：{message}")
        log_line(f"导入完成：{Path(path).name}；警告 {len(project.warnings)}，错误 {len(project.errors)}")

    def _sync(self) -> None:
        if self.project is None:
            return
        for tab in self.field_tabs.values():
            tab.sync_to_project()
        self.cfield_tab.sync_to_project()
        # CRegionTab keeps the model in sync through itemChanged / edit dialog.

    def export_nmo(self) -> None:
        if self.project is None:
            QMessageBox.information(self, "导出", "请先导入一张 NMO。")
            return
        self._sync()
        if not self.project.can_export():
            self._notify("导出失败：导入时存在 array 格式错误，请先处理。", timeout=8000)
            return
        default_dir = self.settings.get(
            "last_export", str(self.current_path.parent if self.current_path else Path.home()))
        default_name = self.current_path.name if self.current_path else "output.nmo"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 NMO", str(Path(default_dir) / default_name),
            "NMO 文件 (*.nmo);;所有文件 (*.*)")
        if not path:
            return
        try:
            self.project.export(path)
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
