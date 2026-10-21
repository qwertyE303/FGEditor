"""FlexGravity editor data model: import NMO groups/arrays, edit, export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .dataarray import (
    NULL_OBJECT,
    ArraySpec,
    Column,
    decode_array,
    encode_array,
)
from .nmo import (
    CK_ARRAYTYPE_FLOAT,
    CK_ARRAYTYPE_INT,
    CK_ARRAYTYPE_OBJECT,
    CK_ARRAYTYPE_STRING,
    CK_CLASS_DATAARRAY,
    CK_CLASS_GROUP,
    CK_CLASS_3DOBJECT,
    CK_CLASS_CURVE,
    NmoFile,
    ObjectRecord,
)

GROUP_NAMES = ("FG_U", "FG_M", "FG_L", "FG_S", "FG_R")
ARRAY_NAMES = ("FG_U", "FG_M", "FG_L", "FG_S", "FG_R", "FG_RG", "FG_Custom")
FIELD_NAMES = {"FG_U": "U", "FG_M": "M", "FG_L": "L", "FG_S": "S"}
FIELD_PARAMS = {
    "FG_U": ["X", "Y", "Z"],
    "FG_M": ["G"],
    "FG_L": ["Lambda"],
    "FG_S": ["Sigma"],
}
PARAM_TYPES = {
    "FG_U": [CK_ARRAYTYPE_FLOAT] * 3,
    "FG_M": [CK_ARRAYTYPE_FLOAT],
    "FG_L": [CK_ARRAYTYPE_FLOAT],
    "FG_S": [CK_ARRAYTYPE_FLOAT],
}
FIELD_LABELS = {
    "FG_U": "U（匀强场）",
    "FG_M": "M（单极子场）",
    "FG_L": "L（线场）",
    "FG_S": "S（面场）",
}

CUSTOM_ARRAY = "FG_Custom"
CUSTOM_COLUMNS = ("Name", "Mode", "X", "Y", "Z", "R", "Theta", "Phi", "Domain")
CUSTOM_PARAMS = ("X", "Y", "Z", "R", "Theta", "Phi", "Domain")
CARTESIAN_PARAMS = ("X", "Y", "Z")
SPHERICAL_PARAMS = ("R", "Theta", "Phi")
CUSTOM_PARAM_INDEX = {name: index for index, name in enumerate(CUSTOM_PARAMS)}
MODE_CARTESIAN = 0
MODE_SPHERICAL = 1

# Groups may contain curves (L) and 3D objects (U/M/S/R).  We keep every group
# member in the table and only warn about suspicious classes.
VALID_FIELD_CLASSES = {33, 41, 43}


class ProjectError(Exception):
    pass


@dataclass
class FieldRow:
    object_index: int | None
    values: list[float]

    def clone(self) -> "FieldRow":
        return FieldRow(self.object_index, list(self.values))


@dataclass
class FieldTable:
    name: str
    params: list[str]
    rows: list[FieldRow] = field(default_factory=list)

    def clone(self) -> "FieldTable":
        return FieldTable(self.name, list(self.params), [r.clone() for r in self.rows])


@dataclass
class RRow:
    region_index: int | None
    sources: list[int]

    def clone(self) -> "RRow":
        return RRow(self.region_index, list(self.sources))


@dataclass
class CustomRow:
    name: str = ""
    mode: int = MODE_CARTESIAN
    values: list[str] = field(default_factory=lambda: [""] * len(CUSTOM_PARAMS))

    def clone(self) -> "CustomRow":
        return CustomRow(self.name, self.mode, list(self.values))

    def get(self, param: str) -> str:
        return self.values[CUSTOM_PARAM_INDEX[param]]

    def set(self, param: str, value: str) -> None:
        self.values[CUSTOM_PARAM_INDEX[param]] = value

    @property
    def active_params(self) -> tuple[str, ...]:
        return CARTESIAN_PARAMS if self.mode == MODE_CARTESIAN else SPHERICAL_PARAMS


@dataclass
class Project:
    nmo: NmoFile
    fields: dict[str, FieldTable] = field(default_factory=dict)
    r_rows: list[RRow] = field(default_factory=list)
    r_extra_cols: int = 0
    rg_sources: list[int] = field(default_factory=list)
    rg_extra_cols: int = 0
    custom_rows: list[CustomRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Project":
        project = cls(NmoFile.load(path))
        project._load()
        return project

    def _warn(self, message: str) -> None:
        self.warnings.append(message)

    def _error(self, message: str) -> None:
        self.errors.append(message)

    def _load(self) -> None:
        self.fields = {
            name: FieldTable(name, list(FIELD_PARAMS[name]))
            for name in ("FG_U", "FG_M", "FG_L", "FG_S")
        }
        self._load_fields()
        self._load_r()
        self._load_rg()
        self._load_custom()
        self._validate()

    def group_indices(self, name: str) -> list[int]:
        return [obj.index for obj in self.nmo.group_members(name)]

    def object_name(self, index: int | None) -> str:
        obj = self.nmo.object_by_index(index)
        return obj.name if obj else ""

    def object_key(self, index: int | None) -> tuple:
        obj = self.nmo.object_by_index(index)
        if obj is None:
            return (1, "", 0)
        return (0, obj.name.casefold(), obj.name, index)

    def _load_fields(self) -> None:
        for name, table in self.fields.items():
            group = self.group_indices(name)
            existing = self.nmo.dataarray_by_name(name)
            if existing is not None:
                try:
                    spec = decode_array(existing.chunk)
                    table.rows = self._decode_field_array(name, spec)
                except Exception as exc:
                    self._error(f"{name} array 格式错误：{exc}")
            seen = {row.object_index for row in table.rows if row.object_index is not None}
            for index in group:
                if index not in seen:
                    table.rows.append(FieldRow(index, [0.0] * len(table.params)))
            table.rows.sort(key=lambda row: self.object_key(row.object_index))

    def _decode_field_array(self, name: str, spec: ArraySpec) -> list[FieldRow]:
        expected = ["Object"] + FIELD_PARAMS[name]
        columns = [c.name.casefold() for c in spec.columns]
        if len(spec.columns) != len(expected):
            raise ProjectError(f"需要 {len(expected)} 列，实际 {len(spec.columns)} 列")
        indices = {}
        for expected_name in expected:
            try:
                indices[expected_name] = columns.index(expected_name.casefold())
            except ValueError:
                raise ProjectError(f"缺少列 {expected_name!r}")
        rows: list[FieldRow] = []
        for row in spec.rows:
            object_index = row[indices["Object"]]
            values = []
            for param in FIELD_PARAMS[name]:
                value = row[indices[param]]
                if value is None:
                    value = 0.0
                values.append(float(value))
            rows.append(FieldRow(object_index, values))
        return rows

    def _load_r(self) -> None:
        existing = self.nmo.dataarray_by_name("FG_R")
        if existing is not None:
            try:
                spec = decode_array(existing.chunk)
                if not spec.columns or any(not c.is_object for c in spec.columns):
                    raise ProjectError("所有列都必须是 Object")
                if spec.column_count < 2:
                    raise ProjectError("至少需要 2 列（作用区域 + 至少一个源）")
                for row in spec.rows:
                    region = row[0]
                    sources = []
                    for value in row[1:]:
                        if value is not None and value not in sources:
                            sources.append(value)
                    self.r_rows.append(RRow(region, sources))
                max_used = max((len(row.sources) for row in self.r_rows), default=0)
                self.r_extra_cols = max(0, spec.column_count - 1 - max_used)
            except Exception as exc:
                self.r_rows = []
                self.r_extra_cols = 0
                self._error(f"FG_R array 格式错误：{exc}")
        seen = {row.region_index for row in self.r_rows if row.region_index is not None}
        for index in self.group_indices("FG_R"):
            if index not in seen:
                self.r_rows.append(RRow(index, []))
        for row in self.r_rows:
            row.sources.sort(key=self.object_key)
        self.r_rows.sort(key=lambda row: self.object_key(row.region_index))

    def _load_rg(self) -> None:
        existing = self.nmo.dataarray_by_name("FG_RG")
        if existing is not None:
            try:
                spec = decode_array(existing.chunk)
                if not spec.columns or any(not c.is_object for c in spec.columns):
                    raise ProjectError("所有列都必须是 Object")
                sources = []
                for row in spec.rows:
                    for value in row:
                        if value is not None and value not in sources:
                            sources.append(value)
                self.rg_sources = sources
                self.rg_extra_cols = max(0, spec.column_count - len(sources))
            except Exception as exc:
                self.rg_sources = []
                self.rg_extra_cols = 0
                self._error(f"FG_RG array 格式错误：{exc}")
        self.rg_sources.sort(key=self.object_key)

    def _load_custom(self) -> None:
        existing = self.nmo.dataarray_by_name(CUSTOM_ARRAY)
        if existing is None:
            return
        try:
            spec = decode_array(existing.chunk)
            self.custom_rows = self._decode_custom_array(spec)
        except Exception as exc:
            self.custom_rows = []
            self._error(f"FG_Custom array 格式错误：{exc}")

    def _decode_custom_array(self, spec: ArraySpec) -> list[CustomRow]:
        names = [column.name.casefold() for column in spec.columns]
        if len(set(names)) != len(names):
            raise ProjectError("列名重复")
        has_mode = "mode" in names
        expected = CUSTOM_COLUMNS if has_mode else tuple(
            name for name in CUSTOM_COLUMNS if name != "Mode")
        if len(spec.columns) != len(expected):
            raise ProjectError(f"需要 {len(expected)} 列，实际 {len(spec.columns)} 列")
        indices = {}
        for expected_name in expected:
            key = expected_name.casefold()
            if key not in names:
                raise ProjectError(f"缺少列 {expected_name!r}")
            indices[expected_name] = names.index(key)
        for column in spec.columns:
            if column.name.casefold() == "mode":
                if not column.is_int:
                    raise ProjectError("Mode 列必须是 int")
            elif not column.is_string:
                raise ProjectError(f"列 {column.name} 必须是 string")

        rows: list[CustomRow] = []
        for raw in spec.rows:
            if has_mode:
                mode_value = raw[indices["Mode"]]
                if mode_value is None:
                    mode = MODE_CARTESIAN
                else:
                    mode = int(mode_value)
                    if mode not in (MODE_CARTESIAN, MODE_SPHERICAL):
                        self._warn(
                            f"FG_Custom: Mode={mode} 不是 0/1，按 0 处理")
                        mode = MODE_CARTESIAN
            else:
                mode = MODE_CARTESIAN
            values = []
            for param in CUSTOM_PARAMS:
                value = raw[indices[param]]
                values.append("" if value is None else str(value))
            rows.append(CustomRow(
                name="" if raw[indices["Name"]] is None else str(raw[indices["Name"]]),
                mode=mode,
                values=values,
            ))
        return rows

    def _validate(self) -> None:
        for group in GROUP_NAMES:
            if self.nmo.object_by_name(group, CK_CLASS_GROUP) is None:
                self._warn(f"缺少 Group {group}；对应表格为空")
        for name, table in self.fields.items():
            for row in table.rows:
                if row.object_index is None:
                    continue
                obj = self.nmo.object_by_index(row.object_index)
                if obj is None:
                    self._warn(f"{name}: 数组行引用了不存在的对象索引 {row.object_index}")
                    continue
                if obj.class_id not in VALID_FIELD_CLASSES:
                    self._warn(f"{name}: {obj.name} 的 class={obj.class_id}，可能不是 3D 对象/曲线")
                if name == "FG_L" and obj.class_id != CK_CLASS_CURVE:
                    self._warn(f"FG_L: {obj.name} 不是 curve，mod 会退回包围盒最长轴")
                if name in ("FG_U", "FG_M", "FG_S") and obj.class_id == CK_CLASS_CURVE:
                    self._warn(f"{name}: {obj.name} 是 curve，mod 可能会跳过或使用包围盒")
                group_indices = set(self.group_indices(name))
                if row.object_index not in group_indices:
                    self._warn(f"{name}: {obj.name} 不在 {name} Group 里（孤行，导出保留）")
        mls = set()
        for name in ("FG_M", "FG_L", "FG_S"):
            mls.update(self.group_indices(name))
            for row in self.fields[name].rows:
                if row.object_index is not None:
                    mls.add(row.object_index)
        for row in self.r_rows:
            if row.region_index is not None:
                if row.region_index not in set(self.group_indices("FG_R")):
                    self._warn(f"FG_R: {self.object_name(row.region_index)} 不在 FG_R Group 里（孤行）")
            for source in row.sources:
                if source not in mls:
                    self._warn(f"FG_R: 源 {self.object_name(source)} 不在 M/L/S 表里")
        for source in self.rg_sources:
            if source not in mls:
                self._warn(f"FG_RG: 源 {self.object_name(source)} 不在 M/L/S 表里")

    # ------------------------------------------------------------------
    # Editing helpers
    # ------------------------------------------------------------------
    def field_rows(self, name: str) -> list[FieldRow]:
        return self.fields[name].rows

    def mls_objects(self) -> list[int]:
        indices = set()
        for name in ("FG_M", "FG_L", "FG_S"):
            for row in self.fields[name].rows:
                if row.object_index is not None:
                    indices.add(row.object_index)
            indices.update(self.group_indices(name))
        return sorted(indices, key=self.object_key)

    def mls_object_types(self) -> dict[int, set[str]]:
        result: dict[int, set[str]] = {}
        for name, label in (("FG_M", "M"), ("FG_L", "L"), ("FG_S", "S")):
            for index in self.group_indices(name):
                result.setdefault(index, set()).add(label)
            for row in self.fields[name].rows:
                if row.object_index is not None:
                    result.setdefault(row.object_index, set()).add(label)
        return result

    def object_display(self, index: int | None) -> str:
        if index is None:
            return ""
        obj = self.nmo.object_by_index(index)
        return obj.name if obj else f"<{index}>"

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def can_export(self) -> bool:
        return not self.errors

    def _build_field_spec(self, name: str) -> ArraySpec:
        table = self.fields[name]
        columns = [Column("Object", CK_ARRAYTYPE_OBJECT)]
        columns += [Column(param, CK_ARRAYTYPE_FLOAT) for param in table.params]
        rows = []
        for row in table.rows:
            values = [float(v) for v in row.values]
            rows.append([row.object_index] + values)
        return ArraySpec(name, columns, rows)

    def _build_r_spec(self) -> ArraySpec:
        max_sources = max((len(row.sources) for row in self.r_rows), default=0)
        source_cols = max(1, max_sources) + self.r_extra_cols
        columns = [Column("Object", CK_ARRAYTYPE_OBJECT)] * (1 + source_cols)
        rows = []
        for row in self.r_rows:
            values = list(row.sources)
            values += [None] * (source_cols - len(values))
            rows.append([row.region_index] + values)
        return ArraySpec("FG_R", columns, rows)

    def _build_rg_spec(self) -> ArraySpec:
        source_cols = max(1, len(self.rg_sources)) + self.rg_extra_cols
        columns = [Column("Object", CK_ARRAYTYPE_OBJECT)] * source_cols
        if not self.rg_sources:
            return ArraySpec("FG_RG", columns, [])
        values = list(self.rg_sources)
        values += [None] * (source_cols - len(values))
        return ArraySpec("FG_RG", columns, [values])

    def _build_custom_spec(self) -> ArraySpec:
        columns = [
            Column("Name", CK_ARRAYTYPE_STRING),
            Column("Mode", CK_ARRAYTYPE_INT),
            Column("X", CK_ARRAYTYPE_STRING),
            Column("Y", CK_ARRAYTYPE_STRING),
            Column("Z", CK_ARRAYTYPE_STRING),
            Column("R", CK_ARRAYTYPE_STRING),
            Column("Theta", CK_ARRAYTYPE_STRING),
            Column("Phi", CK_ARRAYTYPE_STRING),
            Column("Domain", CK_ARRAYTYPE_STRING),
        ]
        rows = []
        for row in self.custom_rows:
            values = list(row.values)
            values += [""] * (len(CUSTOM_PARAMS) - len(values))
            rows.append([row.name, int(row.mode)] + values)
        return ArraySpec(CUSTOM_ARRAY, columns, rows)

    def build_specs(self) -> dict[str, ArraySpec]:
        specs = {name: self._build_field_spec(name) for name in ("FG_U", "FG_M", "FG_L", "FG_S")}
        specs["FG_R"] = self._build_r_spec()
        specs["FG_RG"] = self._build_rg_spec()
        specs[CUSTOM_ARRAY] = self._build_custom_spec()
        return specs

    def export(self, path: str | Path, update_level: bool = False) -> list[str]:
        if self.errors:
            raise ProjectError("导入时存在 array 格式错误，不能导出")
        specs = self.build_specs()
        records: list[ObjectRecord] = []
        existing: dict[str, ObjectRecord] = {}
        for obj in self.nmo.objects:
            record = ObjectRecord.from_object(obj)
            records.append(record)
            if obj.class_id == CK_CLASS_DATAARRAY and obj.name in specs:
                existing[obj.name] = record

        max_id = max((record.object_id for record in records), default=0)
        for name in ARRAY_NAMES:
            spec = specs[name]
            chunk = encode_array(spec)
            if name in existing:
                existing[name].chunk = chunk
            else:
                max_id += 1
                records.append(ObjectRecord.new(max_id, CK_CLASS_DATAARRAY, name, chunk))

        # Phase 0/1: the Level object list is kept unchanged.  The game creates
        # objects from the object table; whether the Level list is required is
        # being verified in Virtools.  A generic updater can be added later.
        data = self.nmo.build(records)
        Path(path).write_bytes(data)
        return list(self.warnings)
