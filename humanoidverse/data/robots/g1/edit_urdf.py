#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inject or update <dynamics damping="..." friction="..."/> for URDF joints.

- Targets revolute joints by default; can include continuous/prismatic via flags.
- Inserts <dynamics> before <limit> if present, else after <axis>, else append.
- If a joint already has <dynamics>, values are updated.

Usage:
  python inject_urdf_dynamics.py \
      --in robot.urdf \
      --out robot.with_dynamics.urdf \
      --damping 0.05 \
      --friction 0.2 \
      --include-continuous \
      --include-prismatic \
      --format raw|pretty|compact \
      --dry-run
"""

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def _local(tag: str) -> str:
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def find_child(elem, name):
    for c in list(elem):
        if _local(c.tag) == name:
            return c
    return None

def insert_before(parent, new_elem, before_tag_names):
    children = list(parent)
    for i, c in enumerate(children):
        if _local(c.tag) in before_tag_names:
            parent.insert(i, new_elem)
            return True
    return False

def insert_after(parent, new_elem, after_tag_names):
    children = list(parent)
    for i, c in enumerate(children):
        if _local(c.tag) in after_tag_names:
            parent.insert(i + 1, new_elem)
            return True
    return False

def indent(elem, level=0):
    """In-place pretty-printer without adding extra blank lines (no minidom)."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for idx, child in enumerate(list(elem)):
            indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                # last child: tail to current level; others: same
                child.tail = i + "  " if idx < len(elem) - 1 else i
    else:
        if not elem.text or not elem.text.strip():
            elem.text = ""
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = "\n" + (level - 1) * "  "

def write_xml(root, outpath: Path, fmt: str = "raw"):
    # 1) raw：不改缩进，最大程度保持原样
    if fmt == "raw":
        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        outpath.write_bytes(xml_bytes)
        return

    # 2) pretty：自定义缩进
    root_copy = _deepcopy_element(root)
    indent(root_copy, 0)
    xml_bytes = ET.tostring(root_copy, encoding="utf-8", xml_declaration=True)
    text = xml_bytes.decode("utf-8")

    if fmt == "compact":
        # 去掉纯空白行
        lines = [ln for ln in text.splitlines() if ln.strip() != ""]
        text = "\n".join(lines) + ("\n" if not text.endswith("\n") else "")

    outpath.write_text(text, encoding="utf-8")

def _deepcopy_element(elem):
    """Deep copy an ElementTree element (without depending on copy module)."""
    new = ET.Element(elem.tag, elem.attrib)
    new.text = elem.text
    new.tail = elem.tail
    for child in list(elem):
        new.append(_deepcopy_element(child))
    return new

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inpath", required=False, default="/home/embodied/data/zxlei/embodied/humanoid/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.urdf", help="Input URDF path")
    ap.add_argument("--out", dest="outpath", required=False, default="/home/embodied/data/zxlei/embodied/humanoid/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof_with_dynamics.urdf", help="Output URDF path")
    ap.add_argument("--damping", type=float, default=0.05, help="damping value")
    ap.add_argument("--friction", type=float, default=0.2, help="friction value (URDF field)")
    ap.add_argument("--include-continuous", action="store_true", help="also patch type='continuous'")
    ap.add_argument("--include-prismatic", action="store_true", help="also patch type='prismatic'")
    ap.add_argument("--dry-run", action="store_true", help="only print summary, do not write file")
    ap.add_argument("--format", choices=["raw", "pretty", "compact"], default="compact",
                    help="Output formatting style")
    args = ap.parse_args()
    
    inpath = Path(args.inpath)
    outpath = Path(args.outpath)
    if not inpath.exists():
        print(f"[ERROR] Input file not found: {inpath}", file=sys.stderr)
        sys.exit(1)

    # 解析
    parser = ET.XMLParser()
    tree = ET.parse(str(inpath), parser=parser)
    root = tree.getroot()

    target_types = {"revolute"}
    if args.include_continuous:
        target_types.add("continuous")
    if args.include_prismatic:
        target_types.add("prismatic")

    changed = 0
    total_targets = 0

    for joint in root.iter():
        if _local(joint.tag) != "joint":
            continue
        jtype = joint.attrib.get("type", "").strip()
        jname = joint.attrib.get("name", "").strip()
        if jtype not in target_types:
            continue
        total_targets += 1

        # 查找/创建 <dynamics>
        dyn = find_child(joint, "dynamics")
        if dyn is None:
            dyn = ET.Element("dynamics")
            inserted = insert_before(joint, dyn, {"limit"})
            if not inserted:
                inserted = insert_after(joint, dyn, {"axis"})
            if not inserted:
                joint.append(dyn)

        prev_damping = dyn.attrib.get("damping")
        prev_friction = dyn.attrib.get("friction")
        new_damping = f"{args.damping:g}"
        new_friction = f"{args.friction:g}"

        if prev_damping != new_damping or prev_friction != new_friction:
            dyn.set("damping", new_damping)
            dyn.set("friction", new_friction)
            changed += 1
            print(f"[UPDATE] joint='{jname}' type={jtype}  dynamics: "
                  f"damping {prev_damping} -> {new_damping}, "
                  f"friction {prev_friction} -> {new_friction}")

    print(f"\n[SUMMARY] target joints: {total_targets}, updated/inserted: {changed}")

    if args.dry_run:
        print("[DRY-RUN] no file written.")
        return

    # 覆盖写同名文件时先备份
    if outpath.resolve() == inpath.resolve():
        backup = inpath.with_suffix(inpath.suffix + ".bak")
        shutil.copyfile(inpath, backup)
        print(f"[BACKUP] original saved to: {backup}")

    write_xml(root, outpath, fmt=args.format)
    print(f"[WRITE] wrote updated URDF to: {outpath} (format={args.format})")

if __name__ == "__main__":
    main()