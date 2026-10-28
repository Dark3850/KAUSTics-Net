"""Auditable source-image partitions; no torch dependency or data mutation."""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

PATCH_RE = re.compile(r"^(\d+)_n_patch_(\d+)_(\d+)(?=_|\.|$)")
PARTITIONS = ("train", "val", "test")
KINDS = ("base", "blur", "color_1", "color_2", "color_3")


def patch_identity(path):
    match = PATCH_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"Cannot recover source image and patch coordinates: {path}")
    source, row, col = map(int, match.groups())
    return f"{source}_n_patch_{row}_{col}", str(source)


def assign_sources(sources, seed=42, val_fraction=0.1, test_fraction=0.1,
                   spatial_groups=None):
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("Require positive train, validation and test fractions")
    sources = sorted(set(sources), key=int)
    mapping = spatial_groups if spatial_groups is not None else {s: s for s in sources}
    if set(mapping) != set(sources) or any(not str(v).strip() for v in mapping.values()):
        raise ValueError("Spatial-group map must cover every source ID exactly once")
    groups = sorted(set(str(mapping[s]) for s in sources))
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required")
    random.Random(seed).shuffle(groups)
    nv = max(1, int(len(groups) * val_fraction + 0.5))
    nt = max(1, int(len(groups) * test_fraction + 0.5))
    if nv + nt >= len(groups):
        raise ValueError("Requested fractions leave no training groups")
    assignment = {g: "train" for g in groups}
    assignment.update({g: "val" for g in groups[:nv]})
    assignment.update({g: "test" for g in groups[nv:nv + nt]})
    return {s: (str(mapping[s]), assignment[str(mapping[s])]) for s in sources}


def digest(manifest):
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summarize(records):
    result = {}
    for split in PARTITIONS:
        rows = [r for r in records if r["split"] == split]
        result[split] = {
            "source_ids": sorted({r["source_id"] for r in rows}, key=int),
            "group_ids": sorted({r["group_id"] for r in rows}),
            "source_images": len({r["source_id"] for r in rows}),
            "base_patches": len(rows),
            "rendered_base_pairs": sum(v["kind"] == "base" for r in rows for v in r["variants"]),
            "degraded_views": sum(len(r["variants"]) for r in rows),
            "by_variant": dict(sorted(Counter(f"shape{v['shape']}/{v['kind']}" for r in rows for v in r["variants"]).items())),
        }
    return result


def validate(manifest, directory, check_files=True):
    if manifest.get("schema_version") != 1 or manifest.get("sha256") != digest(manifest):
        raise ValueError("Manifest version/digest invalid; regenerate through the preparation tool")
    seen, path_owner, source_splits, group_splits = set(), {}, {}, {}
    for r in manifest["records"]:
        patch, source = patch_identity(r["patch_id"])
        if patch != r["patch_id"] or source != r["source_id"] or patch in seen:
            raise ValueError(f"Duplicate or inconsistent patch identity: {patch}")
        seen.add(patch)
        split = r["split"]
        if split not in PARTITIONS:
            raise ValueError(f"Invalid partition {split}")
        for key, table in [(source, source_splits), (r["group_id"], group_splits)]:
            if table.setdefault(key, split) != split:
                raise ValueError(f"Leakage: group/source {key} occurs in multiple partitions")
        variants = [(v["shape"], v["kind"]) for v in r["variants"]]
        expected = {(s, k) for s in (1, 2) for k in KINDS}
        if len(variants) != 10 or set(variants) != expected:
            raise ValueError(f"Incomplete/duplicated variant family: {patch}")
        for value in [r["gt"]] + [v["path"] for v in r["variants"]]:
            p = (Path(directory) / value).resolve()
            if p in path_owner:
                raise ValueError(f"File reused by multiple records/variants: {p}")
            path_owner[p] = patch
            if patch_identity(p)[0] != patch:
                raise ValueError(f"Path does not belong to patch {patch}: {p}")
            if check_files and not p.is_file():
                raise FileNotFoundError(p)
    if summarize(manifest["records"]) != manifest["counts"]:
        raise ValueError("Stored partition counts do not match records")
    if any(not manifest["counts"][s]["base_patches"] for s in PARTITIONS):
        raise ValueError("Empty partition")
    return manifest


def load_manifest(path, check_files=True):
    path = Path(path).resolve()
    m = validate(json.loads(path.read_text()), path.parent, check_files)
    return m


def resolve_records(manifest, directory, split):
    rows = []
    for r in manifest["records"]:
        if r["split"] == split:
            rows.append({**r, "gt": str((Path(directory) / r["gt"]).resolve()),
                         "variants": [{**v, "path": str((Path(directory) / v["path"]).resolve())}
                                      for v in r["variants"]]})
    return rows
