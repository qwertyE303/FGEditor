"""FlexGravity editor data model for the v2.1 schema.

The model keeps *logical references* to CField/CRegion rows (their ``uid``)
because a new row has no placeholder 3D object until export.  On export the
placeholder objects are created, the final object table is assembled, and
every index-based reference (DataArray Object cells, Group members, Level
scene object IDs) is rebuilt.
"""

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
from .encoding import encode_ansi
from .nmo import (
    CK_ARRAYTYPE_FLOAT,
    CK_ARRAYTYPE_INT,
    CK_ARRAYTYPE_OBJECT,
    CK_ARRAYTYPE_STRING,
    CK_CLASS_3DENTITY,
    CK_CLASS_3DOBJECT,
    CK_CLASS_CURVE,
    CK_CLASS_DATAARRAY,
    CK_CLASS_GROUP,
    CK_CLASS_LEVEL,
    NmoFile,
    ObjectRecord,
    StateChunk,
)
from .scene import (
    LevelScene,
    decode_group_members,
    encode_group,
    encode_placeholder_object,
)

PREFIX = "\x01__"

GROUP_NAMES = (
    "FG_UField", "FG_MField", "FG_LField", "FG_SField",
    "FG_Region", "FG_CField", "FG_CRegion",
)
FIELD_NAMES = ("FG_UField", "FG_MField", "FG_LField", "FG_SField")
ARRAY_NAMES = (
    "FG_UField", "FG_MField", "FG_LField", "FG_SField",
    "FG_CField", "FG_CRegion", "FG_Local", "FG_Global",
)
FIELD_PARAMS = {
    "FG_UField": ["X", "Y", "Z"],
    "FG_MField": ["G"],
    "FG_LField": ["Lambda"],
    "FG_SField": ["Sigma"],
}
PARAM_TYPES = {
    "FG_UField": [CK_ARRAYTYPE_FLOAT] * 3,
    "FG_MField": [CK_ARRAYTYPE_FLOAT],
    "FG_LField": [CK_ARRAYTYPE_FLOAT],
    "FG_SField": [CK_ARRAYTYPE_FLOAT],
}
FIELD_LABELS = {
    "FG_UField": "U（匀强场）",
    "FG_MField": "M（单极子场）",
    "FG_LField": "L（线场）",
    "FG_SField": "S（面场）",
}

CFIELD_PARAMS = ("X", "Y", "Z", "R", "Theta", "Phi")
CARTESIAN_PARAMS = ("X", "Y", "Z")
SPHERICAL_PARAMS = ("R", "Theta", "Phi")
CFIELD_PARAM_INDEX = {name: index for index, name in enumerate(CFIELD_PARAMS)}
MODE_CARTESIAN = 0
MODE_SPHERICAL = 1

VALID_FIELD_CLASSES = {33, 41, 43}


class ProjectError(Exception):
    pass


def is_placeholder_name(name: str) -> bool:
    return bool(name) and name.startswith(PREFIX)


def strip_prefix(name: str) -> str:
    return name[len(PREFIX):] if is_placeholder_name(name) else name


def make_placeholder_name(name: str) -> str:
    return PREFIX + name


@dataclass(frozen=True)
class Ref:
    """A logical reference: a map object index, CField uid, or CRegion uid."""
    kind: str
    key: int


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
class CFieldRow:
    uid: int
    name: str = ""
    object_index: int | None = None
    mode: int = MODE_CARTESIAN
    values: list[str] = field(default_factory=lambda: [""] * len(CFIELD_PARAMS))

    def get(self, param: str) -> str:
        return self.values[CFIELD_PARAM_INDEX[param]]

    def set(self, param: str, value: str) -> None:
        self.values[CFIELD_PARAM_INDEX[param]] = value

    @property
    def active_params(self) -> tuple[str, ...]:
        return CARTESIAN_PARAMS if self.mode == MODE_CARTESIAN else SPHERICAL_PARAMS


@dataclass
class CRegionRow:
    uid: int
    name: str = ""
    object_index: int | None = None
    domain: str = ""


@dataclass
class LocalRow:
    region: Ref | None
    sources: list[Ref] = field(default_factory=list)
    orphan: bool = False


class Project:
    def __init__(self, nmo: NmoFile) -> None:
        self.nmo = nmo
        self.fields: dict[str, FieldTable] = {
            name: FieldTable(name, list(FIELD_PARAMS[name])) for name in FIELD_NAMES
        }
        self.region_indices: list[int] = []
        self.cfield_rows: list[CFieldRow] = []
        self.cregion_rows: list[CRegionRow] = []
        self.local_rows: list[LocalRow] = []
        self.local_orphans: list[LocalRow] = []
        self.global_sources: list[Ref] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self._next_uid = 1

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

    def _uid(self) -> int:
        value = self._next_uid
        self._next_uid += 1
        return value

    def _load(self) -> None:
        self._load_fields()
        self._load_region()
        self._load_cfield()
        self._load_cregion()
        self._load_local()
        self._load_global()
        self._validate()

    def group_indices(self, name: str) -> list[int]:
        return [obj.index for obj in self.nmo.group_members(name)]

    def object_name(self, index: int | None) -> str:
        obj = self.nmo.object_by_index(index)
        return obj.name if obj else ""

    def object_display(self, index: int | None) -> str:
        name = self.object_name(index)
        if not name:
            return "" if index is None else f"<{index}>"
        return strip_prefix(name)

    def object_key(self, index: int | None) -> tuple:
        obj = self.nmo.object_by_index(index)
        if obj is None:
            return (1, "", "", -1)
        return (0, strip_prefix(obj.name).casefold(), strip_prefix(obj.name), obj.index)

    def _object_in_group(self, index: int | None, group: str) -> bool:
        return index is not None and index in set(self.group_indices(group))

    # ----- built-in fields --------------------------------------------
    def _load_fields(self) -> None:
        for name, table in self.fields.items():
            group = self.group_indices(name)
            if not group and self.nmo.object_by_name(name, CK_CLASS_GROUP) is None:
                self._warn(f"缺少 Group {name}；对应表格为空")
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
                values.append(0.0 if value is None else float(value))
            rows.append(FieldRow(object_index, values))
        return rows

    # ----- Region -----------------------------------------------------
    def _load_region(self) -> None:
        self.region_indices = sorted(set(self.group_indices("FG_Region")), key=self.object_key)

    # ----- CField -----------------------------------------------------
    def _load_cfield(self) -> None:
        existing = self.nmo.dataarray_by_name("FG_CField")
        seen_objects: set[int] = set()
        if existing is not None:
            try:
                spec = decode_array(existing.chunk)
                self.cfield_rows = self._decode_cfield_array(spec, seen_objects)
            except Exception as exc:
                self.cfield_rows = []
                self._error(f"FG_CField array 格式错误：{exc}")
        group = set(self.group_indices("FG_CField"))
        for index in sorted(group, key=self.object_key):
            if index in seen_objects:
                continue
            name = strip_prefix(self.object_name(index))
            if not is_placeholder_name(self.object_name(index)):
                self._warn(f"FG_CField Group 里的 {self.object_name(index)} 缺少 {PREFIX!r} 前缀")
            if existing is not None:
                self._warn(f"FG_CField Group 有对象 {name} 但 Array 无对应行，已生成空行")
            self.cfield_rows.append(CFieldRow(
                uid=self._uid(), name=name, object_index=index,
                mode=MODE_CARTESIAN, values=[""] * len(CFIELD_PARAMS),
            ))
            seen_objects.add(index)
        self.cfield_rows.sort(key=lambda row: (row.name.casefold(), row.name, row.object_index or -1))
        self._warn_duplicate_names(self.cfield_rows, "FG_CField")

    def _decode_cfield_array(self, spec: ArraySpec, seen_objects: set[int]) -> list[CFieldRow]:
        names = [column.name.casefold() for column in spec.columns]
        if len(set(names)) != len(names):
            raise ProjectError("列名重复")
        expected = ["Name", "Mode"] + list(CFIELD_PARAMS)
        if len(spec.columns) != len(expected):
            raise ProjectError(f"需要 {len(expected)} 列，实际 {len(spec.columns)} 列")
        indices = {}
        for expected_name in expected:
            key = expected_name.casefold()
            if key not in names:
                raise ProjectError(f"缺少列 {expected_name!r}")
            indices[expected_name] = names.index(key)
        if not spec.columns[indices["Name"]].is_object:
            raise ProjectError("Name 列必须是 Object")
        if not spec.columns[indices["Mode"]].is_int:
            raise ProjectError("Mode 列必须是 int")
        for param in CFIELD_PARAMS:
            if not spec.columns[indices[param]].is_string:
                raise ProjectError(f"列 {param} 必须是 string")

        rows: list[CFieldRow] = []
        for raw in spec.rows:
            object_index = raw[indices["Name"]]
            mode_value = raw[indices["Mode"]]
            mode = MODE_CARTESIAN if mode_value is None else int(mode_value)
            if mode not in (MODE_CARTESIAN, MODE_SPHERICAL):
                self._warn(f"FG_CField: Mode={mode} 不是 0/1，按 0 处理")
                mode = MODE_CARTESIAN
            values = []
            for param in CFIELD_PARAMS:
                value = raw[indices[param]]
                values.append("" if value is None else str(value))
            if object_index is None:
                name = "<无效对象>"
                self._error("FG_CField 有一行缺少 Name 对象")
            else:
                name = strip_prefix(self.object_name(object_index))
                if not self.object_name(object_index):
                    name = "<无效对象>"
                    self._error(f"FG_CField 行引用了不存在的对象索引 {object_index}")
                elif not is_placeholder_name(self.object_name(object_index)):
                    self._warn(f"FG_CField 的 {self.object_name(object_index)} 缺少 {PREFIX!r} 前缀")
                if not self._object_in_group(object_index, "FG_CField"):
                    self._warn(f"FG_CField: {name} 不在 FG_CField Group 里")
                if object_index in seen_objects:
                    self._warn(f"FG_CField: {name} 指向重复对象，保留多行")
                seen_objects.add(object_index)
            rows.append(CFieldRow(
                uid=self._uid(), name=name, object_index=object_index,
                mode=mode, values=values,
            ))
        return rows

    # ----- CRegion ----------------------------------------------------
    def _load_cregion(self) -> None:
        existing = self.nmo.dataarray_by_name("FG_CRegion")
        seen_objects: set[int] = set()
        if existing is not None:
            try:
                spec = decode_array(existing.chunk)
                self.cregion_rows = self._decode_cregion_array(spec, seen_objects)
            except Exception as exc:
                self.cregion_rows = []
                self._error(f"FG_CRegion array 格式错误：{exc}")
        group = set(self.group_indices("FG_CRegion"))
        for index in sorted(group, key=self.object_key):
            if index in seen_objects:
                continue
            name = strip_prefix(self.object_name(index))
            if not is_placeholder_name(self.object_name(index)):
                self._warn(f"FG_CRegion Group 里的 {self.object_name(index)} 缺少 {PREFIX!r} 前缀")
            if existing is not None:
                self._warn(f"FG_CRegion Group 有对象 {name} 但 Array 无对应行，已生成空行")
            self.cregion_rows.append(CRegionRow(uid=self._uid(), name=name, object_index=index))
            seen_objects.add(index)
        self.cregion_rows.sort(key=lambda row: (row.name.casefold(), row.name, row.object_index or -1))
        self._warn_duplicate_names(self.cregion_rows, "FG_CRegion")

    def _decode_cregion_array(self, spec: ArraySpec, seen_objects: set[int]) -> list[CRegionRow]:
        names = [column.name.casefold() for column in spec.columns]
        if len(set(names)) != len(names):
            raise ProjectError("列名重复")
        expected = ["Name", "Domain"]
        if len(spec.columns) != len(expected):
            raise ProjectError(f"需要 {len(expected)} 列，实际 {len(spec.columns)} 列")
        indices = {}
        for expected_name in expected:
            key = expected_name.casefold()
            if key not in names:
                raise ProjectError(f"缺少列 {expected_name!r}")
            indices[expected_name] = names.index(key)
        if not spec.columns[indices["Name"]].is_object:
            raise ProjectError("Name 列必须是 Object")
        if not spec.columns[indices["Domain"]].is_string:
            raise ProjectError("Domain 列必须是 string")

        rows: list[CRegionRow] = []
        for raw in spec.rows:
            object_index = raw[indices["Name"]]
            domain = raw[indices["Domain"]]
            domain = "" if domain is None else str(domain)
            if object_index is None:
                name = "<无效对象>"
                self._error("FG_CRegion 有一行缺少 Name 对象")
            else:
                name = strip_prefix(self.object_name(object_index))
                if not self.object_name(object_index):
                    name = "<无效对象>"
                    self._error(f"FG_CRegion 行引用了不存在的对象索引 {object_index}")
                elif not is_placeholder_name(self.object_name(object_index)):
                    self._warn(f"FG_CRegion 的 {self.object_name(object_index)} 缺少 {PREFIX!r} 前缀")
                if not self._object_in_group(object_index, "FG_CRegion"):
                    self._warn(f"FG_CRegion: {name} 不在 FG_CRegion Group 里")
                if object_index in seen_objects:
                    self._warn(f"FG_CRegion: {name} 指向重复对象，保留多行")
                seen_objects.add(object_index)
            rows.append(CRegionRow(
                uid=self._uid(), name=name, object_index=object_index, domain=domain,
            ))
        return rows

    def _warn_duplicate_names(self, rows: list, label: str) -> None:
        seen: dict[str, int] = {}
        for row in rows:
            key = row.name.casefold()
            seen[key] = seen.get(key, 0) + 1
        for key, count in seen.items():
            if count > 1 and key:
                self._warn(f"{label}: 名称 {key!r} 重复 {count} 次（导出前需处理）")

    # ----- Local / Global --------------------------------------------
    def _mls_indices(self) -> set[int]:
        result: set[int] = set()
        for name in ("FG_MField", "FG_LField", "FG_SField"):
            result.update(self.group_indices(name))
            for row in self.fields[name].rows:
                if row.object_index is not None:
                    result.add(row.object_index)
        return result

    def _u_indices(self) -> set[int]:
        result = set(self.group_indices("FG_UField"))
        for row in self.fields["FG_UField"].rows:
            if row.object_index is not None:
                result.add(row.object_index)
        return result

    def _region_refs(self) -> dict[int, Ref]:
        result: dict[int, Ref] = {}
        for index in self.region_indices:
            result[index] = Ref("object", index)
        for row in self.cregion_rows:
            if row.object_index is not None:
                result[row.object_index] = Ref("cregion", row.uid)
        return result

    def _source_refs(self) -> dict[int, Ref]:
        result: dict[int, Ref] = {}
        for index in self._mls_indices():
            result[index] = Ref("object", index)
        for row in self.cfield_rows:
            if row.object_index is not None:
                result[row.object_index] = Ref("cfield", row.uid)
        return result

    def _load_local(self) -> None:
        region_refs = self._region_refs()
        source_refs = self._source_refs()
        u_indices = self._u_indices()
        cregion_objects = {
            row.object_index for row in self.cregion_rows if row.object_index is not None
        }
        existing = self.nmo.dataarray_by_name("FG_Local")
        rows_by_ref: dict[Ref, LocalRow] = {}
        if existing is not None:
            try:
                spec = decode_array(existing.chunk)
                if not spec.columns or any(not c.is_object for c in spec.columns):
                    raise ProjectError("所有列都必须是 Object")
                if spec.column_count < 2:
                    raise ProjectError("至少需要 2 列（作用区域 + 至少一个源）")
                for raw in spec.rows:
                    region_index = raw[0]
                    if region_index is None:
                        self._warn("FG_Local: 有一行 Region 为空，已忽略")
                        continue
                    region_ref = region_refs.get(region_index)
                    orphan = region_ref is None
                    if orphan:
                        region_ref = Ref("object", region_index)
                        self._warn(f"FG_Local: 未知 Region {self.object_display(region_index)}，保留为额外行")
                    sources: list[Ref] = []
                    for value in raw[1:]:
                        if value is None:
                            continue
                        if value in source_refs:
                            ref = source_refs[value]
                        elif value in u_indices:
                            self._warn(f"FG_Local: U {self.object_display(value)} 不能作为源，已忽略")
                            continue
                        elif value in cregion_objects:
                            self._warn(f"FG_Local: CRegion {self.object_display(value)} 不能作为源，已忽略")
                            continue
                        else:
                            ref = Ref("object", value)
                            self._warn(f"FG_Local: 未知源 {self.object_display(value)}，保留")
                        if ref not in sources:
                            sources.append(ref)
                    sources.sort(key=self.ref_sort_key)
                    if region_ref in rows_by_ref:
                        target = rows_by_ref[region_ref]
                        for ref in sources:
                            if ref not in target.sources:
                                target.sources.append(ref)
                        target.sources.sort(key=self.ref_sort_key)
                        self._warn("FG_Local: 同一 Region 有多行，已合并源")
                    else:
                        rows_by_ref[region_ref] = LocalRow(region_ref, sources, orphan)
            except Exception as exc:
                self._error(f"FG_Local array 格式错误：{exc}")
        for region_ref in self._all_region_refs():
            rows_by_ref.setdefault(region_ref, LocalRow(region_ref, []))
        self.local_rows = sorted(
            [row for row in rows_by_ref.values() if not row.orphan],
            key=lambda row: self.region_sort_key(row.region),
        )
        self.local_orphans = [row for row in rows_by_ref.values() if row.orphan]

    def _load_global(self) -> None:
        source_refs = self._source_refs()
        u_indices = self._u_indices()
        cregion_objects = {
            row.object_index for row in self.cregion_rows if row.object_index is not None
        }
        existing = self.nmo.dataarray_by_name("FG_Global")
        sources: list[Ref] = []
        if existing is not None:
            try:
                spec = decode_array(existing.chunk)
                if not spec.columns or any(not c.is_object for c in spec.columns):
                    raise ProjectError("所有列都必须是 Object")
                for raw in spec.rows:
                    for value in raw:
                        if value is None:
                            continue
                        if value in source_refs:
                            ref = source_refs[value]
                        elif value in u_indices:
                            self._warn(f"FG_Global: U {self.object_display(value)} 不能作为源，已忽略")
                            continue
                        elif value in cregion_objects:
                            self._warn(f"FG_Global: CRegion {self.object_display(value)} 不能作为源，已忽略")
                            continue
                        else:
                            ref = Ref("object", value)
                            self._warn(f"FG_Global: 未知源 {self.object_display(value)}，保留")
                        if ref not in sources:
                            sources.append(ref)
            except Exception as exc:
                self._error(f"FG_Global array 格式错误：{exc}")
        sources.sort(key=self.ref_sort_key)
        self.global_sources = sources

    # ----- validation -------------------------------------------------
    def _validate(self) -> None:
        for group in GROUP_NAMES:
            if group in ("FG_CField", "FG_CRegion"):
                # 这两个 Group 由导出器创建；首次导入的图通常没有。
                continue
            if self.nmo.object_by_name(group, CK_CLASS_GROUP) is None:
                self._warn(f"缺少 Group {group}；对应表格为空")
        for name, table in self.fields.items():
            group_indices = set(self.group_indices(name))
            for row in table.rows:
                if row.object_index is None:
                    continue
                obj = self.nmo.object_by_index(row.object_index)
                if obj is None:
                    self._warn(f"{name}: 数组行引用了不存在的对象索引 {row.object_index}")
                    continue
                if obj.class_id not in VALID_FIELD_CLASSES:
                    self._warn(f"{name}: {obj.name} 的 class={obj.class_id}，可能不是 3D 对象/曲线")
                if name == "FG_LField" and obj.class_id != CK_CLASS_CURVE:
                    self._warn(f"FG_LField: {obj.name} 不是 curve，mod 会退回包围盒最长轴")
                if name in ("FG_UField", "FG_MField", "FG_SField") and obj.class_id == CK_CLASS_CURVE:
                    self._warn(f"{name}: {obj.name} 是 curve，mod 可能会跳过或使用包围盒")
                if row.object_index not in group_indices:
                    self._warn(f"{name}: {obj.name} 不在 {name} Group 里（孤行，导出保留）")
        cfield_names = {}
        for row in self.cfield_rows:
            cfield_names[row.name.casefold()] = cfield_names.get(row.name.casefold(), 0) + 1
        cregion_names = {}
        for row in self.cregion_rows:
            cregion_names[row.name.casefold()] = cregion_names.get(row.name.casefold(), 0) + 1
        for row in self.cfield_rows:
            if not row.name or row.name == "<无效对象>":
                self._warn("FG_CField: 存在空名称/无效对象的行")
        for row in self.cregion_rows:
            if not row.name or row.name == "<无效对象>":
                self._warn("FG_CRegion: 存在空名称/无效对象的行")

    # ------------------------------------------------------------------
    # Display / selection helpers
    # ------------------------------------------------------------------
    def ref_display(self, ref: Ref | None) -> str:
        if ref is None:
            return ""
        if ref.kind == "cfield":
            row = self.cfield_by_uid(ref.key)
            return row.name if row else "<缺失CField>"
        if ref.kind == "cregion":
            row = self.cregion_by_uid(ref.key)
            return row.name if row else "<缺失CRegion>"
        return self.object_display(ref.key)

    def ref_sort_key(self, ref: Ref | None) -> tuple:
        if ref is None:
            return (2, "", -1)
        if ref.kind == "cfield":
            row = self.cfield_by_uid(ref.key)
            return (1, row.name.casefold() if row else "", ref.key)
        if ref.kind == "cregion":
            row = self.cregion_by_uid(ref.key)
            return (1, row.name.casefold() if row else "", ref.key)
        return (0, self.object_display(ref.key).casefold(), ref.key)

    def region_sort_key(self, ref: Ref | None) -> tuple:
        if ref is None:
            return (2, "", -1)
        if ref.kind == "cregion":
            row = self.cregion_by_uid(ref.key)
            return (1, row.name.casefold() if row else "", ref.key)
        return (0, self.object_display(ref.key).casefold(), ref.key)

    def ref_type_label(self, ref: Ref | None) -> str:
        if ref is None:
            return ""
        if ref.kind == "cfield":
            return "C"
        if ref.kind == "cregion":
            return "C"
        labels = []
        for name, label in (("FG_MField", "M"), ("FG_LField", "L"), ("FG_SField", "S")):
            if ref.key in set(self.group_indices(name)):
                labels.append(label)
            for row in self.fields[name].rows:
                if row.object_index == ref.key:
                    labels.append(label)
                    break
        if ref.key in self._u_indices():
            labels.append("U")
        return "/".join(dict.fromkeys(labels))

    def ref_label(self, ref: Ref | None) -> str:
        name = self.ref_display(ref)
        kind = self.ref_type_label(ref)
        return f"{name}    [{kind}]" if kind else name

    def source_options(self) -> list[Ref]:
        refs: list[Ref] = [Ref("object", index) for index in self._mls_indices()]
        refs += [Ref("cfield", row.uid) for row in self.cfield_rows]
        return sorted(set(refs), key=self.ref_sort_key)

    def region_options(self) -> list[Ref]:
        refs = [Ref("object", index) for index in self.region_indices]
        refs += [Ref("cregion", row.uid) for row in self.cregion_rows]
        return sorted(set(refs), key=self.region_sort_key)

    def cfield_by_uid(self, uid: int) -> CFieldRow | None:
        for row in self.cfield_rows:
            if row.uid == uid:
                return row
        return None

    def cregion_by_uid(self, uid: int) -> CRegionRow | None:
        for row in self.cregion_rows:
            if row.uid == uid:
                return row
        return None

    def _all_region_refs(self) -> list[Ref]:
        return self.region_options()

    # ------------------------------------------------------------------
    # Editing helpers
    # ------------------------------------------------------------------
    def next_name(self, prefix: str, existing: list[str]) -> str:
        used = set()
        for name in existing:
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                if suffix.isdigit():
                    used.add(int(suffix))
        number = 1
        while number in used:
            number += 1
        return f"{prefix}{number:02d}"

    def new_cfield(self) -> CFieldRow:
        row = CFieldRow(
            uid=self._uid(),
            name=self.next_name("CustomField_", [r.name for r in self.cfield_rows]),
            values=[""] * len(CFIELD_PARAMS),
        )
        self.cfield_rows.append(row)
        return row

    def remove_cfield(self, row: CFieldRow) -> None:
        self.cfield_rows = [item for item in self.cfield_rows if item.uid != row.uid]
        ref = Ref("cfield", row.uid)
        for local in self.local_rows + self.local_orphans:
            local.sources = [item for item in local.sources if item != ref]
        self.global_sources = [item for item in self.global_sources if item != ref]

    def new_cregion(self) -> CRegionRow:
        row = CRegionRow(
            uid=self._uid(),
            name=self.next_name("CustomRegion_", [r.name for r in self.cregion_rows]),
        )
        self.cregion_rows.append(row)
        self.local_rows.append(LocalRow(Ref("cregion", row.uid), []))
        return row

    def remove_cregion(self, row: CRegionRow) -> None:
        self.cregion_rows = [item for item in self.cregion_rows if item.uid != row.uid]
        ref = Ref("cregion", row.uid)
        self.local_rows = [item for item in self.local_rows if item.region != ref]
        self.local_orphans = [item for item in self.local_orphans if item.region != ref]
        self.global_sources = [item for item in self.global_sources if item != ref]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def can_export(self) -> bool:
        return not self.errors

    def _validate_export_names(self) -> list[str]:
        problems: list[str] = []
        names: list[str] = []
        for label, rows in (("FG_CField", self.cfield_rows), ("FG_CRegion", self.cregion_rows)):
            for row in rows:
                name = row.name.strip()
                if not name or name == "<无效对象>":
                    problems.append(f"{label}: 存在空名称")
                    continue
                if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
                    problems.append(f"{label}: 名称 {name!r} 含控制字符")
                names.append(name)
                full = make_placeholder_name(name)
                try:
                    encode_ansi(full, strict=True)
                except (UnicodeEncodeError, LookupError):
                    problems.append(f"{label}: 名称 {name!r} 无法用 ANSI/GBK 编码")
                for obj in self.nmo.objects:
                    if obj.name == full and not is_placeholder_name(obj.name):
                        problems.append(f"{label}: 名称 {name!r} 与地图对象重名")
        seen: set[str] = set()
        for name in names:
            key = name.casefold()
            if key in seen:
                problems.append(f"CField/CRegion 名称 {name!r} 重复")
            seen.add(key)
        return problems

    def _map_old_index(self, index: int | None, mapping: dict[int, int]) -> int | None:
        if index is None:
            return None
        return mapping.get(index)

    def _map_ref(self, ref: Ref | None, mapping: dict[int, int],
                 cfield_index: dict[int, int], cregion_index: dict[int, int]) -> int | None:
        if ref is None:
            return None
        if ref.kind == "cfield":
            return cfield_index.get(ref.key)
        if ref.kind == "cregion":
            return cregion_index.get(ref.key)
        return mapping.get(ref.key)

    def _build_field_spec(self, name: str, mapping: dict[int, int]) -> ArraySpec:
        table = self.fields[name]
        columns = [Column("Object", CK_ARRAYTYPE_OBJECT)]
        columns += [Column(param, CK_ARRAYTYPE_FLOAT) for param in table.params]
        rows = []
        for row in table.rows:
            rows.append([self._map_old_index(row.object_index, mapping)] + [float(v) for v in row.values])
        return ArraySpec(name, columns, rows)

    def _build_cfield_spec(self, cfield_index: dict[int, int]) -> ArraySpec:
        columns = [
            Column("Name", CK_ARRAYTYPE_OBJECT),
            Column("Mode", CK_ARRAYTYPE_INT),
        ]
        columns += [Column(param, CK_ARRAYTYPE_STRING) for param in CFIELD_PARAMS]
        rows = []
        for row in self.cfield_rows:
            values = list(row.values) + [""] * (len(CFIELD_PARAMS) - len(row.values))
            rows.append([cfield_index.get(row.uid)] + [int(row.mode)] + values[:len(CFIELD_PARAMS)])
        return ArraySpec("FG_CField", columns, rows)

    def _build_cregion_spec(self, cregion_index: dict[int, int]) -> ArraySpec:
        columns = [Column("Name", CK_ARRAYTYPE_OBJECT), Column("Domain", CK_ARRAYTYPE_STRING)]
        rows = [[cregion_index.get(row.uid), row.domain] for row in self.cregion_rows]
        return ArraySpec("FG_CRegion", columns, rows)

    def _build_local_spec(self, mapping: dict[int, int],
                          cfield_index: dict[int, int],
                          cregion_index: dict[int, int]) -> ArraySpec:
        rows_all = self.local_rows + self.local_orphans
        max_sources = max((len(row.sources) for row in rows_all), default=0)
        source_cols = max(1, max_sources)
        columns = [Column("Region", CK_ARRAYTYPE_OBJECT)]
        columns += [Column(f"Source{i + 1}", CK_ARRAYTYPE_OBJECT) for i in range(source_cols)]
        rows = []
        for row in rows_all:
            region = self._map_ref(row.region, mapping, cfield_index, cregion_index)
            sources = [
                self._map_ref(ref, mapping, cfield_index, cregion_index)
                for ref in row.sources
            ]
            sources = [value for value in sources if value is not None]
            values = [region] + sources + [None] * (source_cols - len(sources))
            rows.append(values)
        return ArraySpec("FG_Local", columns, rows)

    def _build_global_spec(self, mapping: dict[int, int],
                           cfield_index: dict[int, int],
                           cregion_index: dict[int, int]) -> ArraySpec:
        values = [
            self._map_ref(ref, mapping, cfield_index, cregion_index)
            for ref in self.global_sources
        ]
        values = [value for value in values if value is not None]
        source_cols = max(1, len(values))
        columns = [Column(f"Source{i + 1}", CK_ARRAYTYPE_OBJECT) for i in range(source_cols)]
        rows = [values + [None] * (source_cols - len(values))] if values else []
        return ArraySpec("FG_Global", columns, rows)

    @staticmethod
    def _remap_array_chunk(chunk: bytes, mapping: dict[int, int]) -> bytes:
        spec = decode_array(chunk)
        rows = []
        for raw in spec.rows:
            row = []
            for column, value in zip(spec.columns, raw):
                if column.is_object:
                    row.append(None if value is None else mapping.get(value))
                else:
                    row.append(value)
            rows.append(row)
        return encode_array(ArraySpec("", spec.columns, rows))

    def export(self, path: str | Path) -> list[str]:
        if self.errors:
            raise ProjectError("导入时存在 array 格式错误，不能导出")
        name_problems = self._validate_export_names()
        if name_problems:
            raise ProjectError("；".join(name_problems))

        records: list[ObjectRecord] = []
        for obj in self.nmo.objects:
            if is_placeholder_name(obj.name):
                continue
            records.append(ObjectRecord.from_object(obj))

        old_to_new: dict[int, int] = {}
        for index, record in enumerate(records):
            if record.source_index is not None:
                old_to_new[record.source_index] = index

        max_id = max((record.object_id for record in records), default=0)

        def allocate_id() -> int:
            nonlocal max_id
            max_id += 1
            return max_id

        def find_record(name: str, class_id: int) -> ObjectRecord | None:
            for record in records:
                if record.name == name and record.class_id == class_id:
                    return record
            return None

        cfield_index: dict[int, int] = {}
        for row in self.cfield_rows:
            record = ObjectRecord.new(
                allocate_id(), CK_CLASS_3DOBJECT,
                make_placeholder_name(row.name), encode_placeholder_object(),
            )
            records.append(record)
            cfield_index[row.uid] = len(records) - 1

        cregion_index: dict[int, int] = {}
        for row in self.cregion_rows:
            record = ObjectRecord.new(
                allocate_id(), CK_CLASS_3DOBJECT,
                make_placeholder_name(row.name), encode_placeholder_object(),
            )
            records.append(record)
            cregion_index[row.uid] = len(records) - 1

        def ensure(name: str, class_id: int) -> ObjectRecord:
            existing = find_record(name, class_id)
            if existing is not None:
                return existing
            chunk = encode_group([]) if class_id == CK_CLASS_GROUP else b""
            record = ObjectRecord.new(allocate_id(), class_id, name, chunk)
            records.append(record)
            return record

        cfield_group = ensure("FG_CField", CK_CLASS_GROUP)
        cregion_group = ensure("FG_CRegion", CK_CLASS_GROUP)
        array_records = {name: ensure(name, CK_CLASS_DATAARRAY) for name in ARRAY_NAMES}

        # Remap every non-editor group and DataArray.
        for record in records:
            if record.class_id == CK_CLASS_GROUP and record.name not in ("FG_CField", "FG_CRegion"):
                members = decode_group_members(record.chunk)
                record.chunk = encode_group([old_to_new[m] for m in members if m in old_to_new])
            elif record.class_id == CK_CLASS_DATAARRAY and record.name not in ARRAY_NAMES:
                try:
                    record.chunk = self._remap_array_chunk(record.chunk, old_to_new)
                except Exception as exc:
                    raise ProjectError(f"无法重映射 DataArray {record.name}：{exc}")

        cfield_group.chunk = encode_group([cfield_index[row.uid] for row in self.cfield_rows])
        cregion_group.chunk = encode_group([cregion_index[row.uid] for row in self.cregion_rows])

        for name in FIELD_NAMES:
            array_records[name].chunk = encode_array(self._build_field_spec(name, old_to_new))
        array_records["FG_CField"].chunk = encode_array(self._build_cfield_spec(cfield_index))
        array_records["FG_CRegion"].chunk = encode_array(self._build_cregion_spec(cregion_index))
        array_records["FG_Local"].chunk = encode_array(
            self._build_local_spec(old_to_new, cfield_index, cregion_index))
        array_records["FG_Global"].chunk = encode_array(
            self._build_global_spec(old_to_new, cfield_index, cregion_index))

        level_record = next((record for record in records if record.class_id == CK_CLASS_LEVEL), None)
        if level_record is None:
            raise ProjectError("NMO 中没有 Level 对象")
        level_state = StateChunk.parse(level_record.chunk)
        scene_payload = level_state.find(0x80000000)
        if scene_payload is None:
            raise ProjectError("Level 缺少场景数据")
        scene = LevelScene.from_payload(scene_payload)
        record_ids = {record.object_id for record in records}
        desired_ids = [
            entry.object_id for entry in scene.entries if entry.object_id in record_ids
        ]
        wanted = set(desired_ids)
        for record in records:
            is_new = record.source_index is None
            is_editor_group = (
                record.class_id == CK_CLASS_GROUP
                and record.name in ("FG_CField", "FG_CRegion")
            )
            is_editor_array = (
                record.class_id == CK_CLASS_DATAARRAY
                and record.name in ARRAY_NAMES
            )
            if (is_new or is_editor_group or is_editor_array) and record.object_id not in wanted:
                desired_ids.append(record.object_id)
                wanted.add(record.object_id)
        scene.update_entries(desired_ids)
        level_record.chunk = scene.to_chunk(level_state)

        data = self.nmo.build(records)
        Path(path).write_bytes(data)
        return list(self.warnings)
