#!/usr/bin/env python3
# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
r"""Report which IPs are identical and identically configured across tops.

Each top instantiates its own copy of shared block-level IP, so regressions and
CI re-run block-level DV once per top. Some of those runs exercise genuinely
different RTL (the IP is parameterised differently per top) and some do not.
This tool answers which is which, and how much simulation sits behind each
answer, so the duplication can be quantified before any flow is changed.

It never modifies the source tree: it parses checked-in hjson, and only writes
where it is asked to -- ``--json`` for the machine-readable dump and ``--html``
for a report under the scratch root, which ``--serve`` then hosts on localhost.

Two parameter sources are needed, because they cover different things:

  * ``hw/top_*/data/autogen/top_*.gen.hjson`` -- the fully elaborated top. Every
    module carries a ``param_list`` of resolved SystemVerilog instantiation
    parameters. Covers all IPs, but only SV parameters.
  * ``hw/top_*/ip_autogen/*/data/*.ipconfig.hjson`` -- ipgen template parameters,
    which drive code generation rather than SV elaboration and so never appear in
    ``param_list``. Covers only the ipgen IPs.

Neither alone is sufficient. gpio is the worked example: its ``param_list``
differs across tops on ``GpioAsHwStrapsEn`` while its ipconfig differs on
``num_inp_period_counters``, and the two disagree about which tops match.
"""

import argparse
import functools
import hashlib
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import hjson

# The checkout being analysed. Resolved at startup by resolve_repo_root(), not
# from __file__: this tool is routinely run out of one checkout against another
# (a public tree against an embargoed one, say), and deriving the root from the
# script's own location silently analysed the wrong repo. Every use is inside a
# function, so rebinding it in main() before any work starts is safe.
REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_repo_root(arg_repo_root):
    """Pick the checkout to analyse, mirroring how dvsim resolves proj_root.

    dvsim (cli/run.py::get_proj_root) runs `git rev-parse --show-toplevel` in
    the current directory, so invoking it from inside a checkout targets that
    checkout no matter where the tool itself lives. Matching that means
    `../opentitan/util/ip_equivalence.py` run from another tree analyses the
    tree you are standing in, which is what people expect.
    """
    if arg_repo_root:
        return Path(arg_repo_root).resolve()
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=5,
                             check=True)
        top = out.stdout.strip()
        if top:
            return Path(top).resolve()
    except (OSError, subprocess.SubprocessError):
        pass
    # Not inside a git repo: fall back to the checkout holding this script.
    return Path(__file__).resolve().parent.parent

# Keys that encode *where* an instance lives rather than *what* it is. Two
# otherwise identical configs always differ on these, so comparing without
# dropping them finds nothing. topname in particular is injected into every
# IpConfig unconditionally by util/topgen.py.
IDENTITY_KEYS = {"topname", "instance_name", "module_instance_name",
                 "uniquified_modules"}

# Marker for "this instance does not declare that parameter at all".
ABSENT = "<absent>"

# A name-mangling artifact of the top (e.g. Gpio + GpioAsyncOn), not a value.
PARAM_META_IGNORE = {"name_top"}

# Whitespace is legal between the base and the digits, and is used in-tree:
# lc_ctrl writes ProductId as "16'h 4000". Without \s* here such values fall
# through to the string comparison, so 16'h4000 and 16'h 4000 would read as
# different parameters.
SV_LITERAL_RE = re.compile(
    r"^\s*(\d+)?'([sS])?([bBoOdDhH])\s*([0-9a-fA-F_xXzZ?]+)\s*$")
SV_BASE = {"b": 2, "o": 8, "d": 10, "h": 16}


def normalise_value(val):
    """Canonicalise a parameter value so equal values compare equal.

    Parameter defaults reach us as SystemVerilog literals, quoted decimals, or
    native hjson types, and the same value is spelled differently in different
    tops -- englishbreakfast writes ``1'b1`` where darjeeling writes ``"1"``.
    Without this, every comparison is dominated by spelling.

    Values that are not recognisably numeric are returned as a stripped string,
    which compares fine as long as it is spelled consistently.
    """
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return val
    if isinstance(val, (list, tuple)):
        return [normalise_value(v) for v in val]
    if isinstance(val, dict):
        return {k: normalise_value(v) for k, v in sorted(val.items())}
    if val is None:
        return None

    text = str(val).strip()
    if not text:
        return ""

    low = text.lower()
    if low in ("true", "false"):
        return 1 if low == "true" else 0

    m = SV_LITERAL_RE.match(text)
    if m:
        digits = m.group(4).replace("_", "")
        # x/z/? are not a single value; keep them symbolic but canonically cased.
        if re.search(r"[xXzZ?]", digits):
            return f"sv:{m.group(3).lower()}:{digits.lower()}"
        try:
            return int(digits, SV_BASE[m.group(3).lower()])
        except ValueError:
            return text

    # Bare decimal or hex, quoted or not.
    try:
        return int(text, 0)
    except ValueError:
        pass
    try:
        return int(text, 10)
    except ValueError:
        pass
    return text


def canonical_params(params):
    """Drop identity keys and normalise every value, ready for fingerprinting."""
    out = {}
    for key, val in params.items():
        if key in IDENTITY_KEYS:
            continue
        out[key] = normalise_value(val)
    return out


def fingerprint(params):
    """A stable digest of a canonical parameter dict."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


@functools.cache
def parse_hjson(path):
    """Parse one hjson file, memoised.

    hjson is slow, and the shared includes under hw/dv/tools/dvsim are imported
    by nearly every block cfg, so without this the same handful of files gets
    parsed close to a hundred times. Returns None if the file is missing or
    unparseable, so callers can distinguish that from an empty file.
    """
    try:
        return hjson.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, hjson.HjsonDecodeError):
        return None


def discover_tops(only=None):
    """Find each top's fully elaborated description."""
    tops = {}
    for path in sorted(REPO_ROOT.glob("hw/top_*/data/autogen/top_*.gen.hjson")):
        # Skip the secrets sidecar, which is not a top description.
        if ".secrets" in path.name:
            continue
        name = path.parent.parent.parent.name  # hw/top_<x>/...
        if only and name not in only and name.removeprefix("top_") not in only:
            continue
        tops[name] = path
    return tops


def load_sv_params(gen_hjson):
    """Per-instance SystemVerilog parameters from an elaborated top.

    Returns {instance_name: {"type": ip_type, "params": {...}, "local": [...]}}.
    """
    data = parse_hjson(str(gen_hjson)) or {}
    out = {}
    for mod in data.get("module", []):
        params = {}
        locals_ = []
        for p in mod.get("param_list", []):
            pname = p.get("name")
            if pname is None or pname in PARAM_META_IGNORE:
                continue
            params[pname] = p.get("default")
            if str(p.get("local", "false")).lower() == "true":
                locals_.append(pname)
        out[mod["name"]] = {
            "type": mod.get("type", mod["name"]),
            "params": params,
            "local": locals_,
            "attr": mod.get("attr", "normal"),
        }
    return out


def load_ipgen_params(top):
    """ipgen template parameters, keyed by the ip_autogen directory name.

    These drive code generation, so they never show up in the elaborated top's
    param_list, yet they change the RTL and therefore the DV.
    """
    out = {}
    for path in sorted((REPO_ROOT / "hw" / top / "ip_autogen").glob(
            "*/data/*.ipconfig.hjson")):
        ip_dir = path.parent.parent.name
        cfg = parse_hjson(str(path)) or {}
        out[ip_dir] = cfg.get("param_values", {})
    return out


def build_instances(tops):
    """Join both parameter sources into one record per (top, instance)."""
    instances = []
    for top, gen_path in tops.items():
        sv = load_sv_params(gen_path)
        ipgen = load_ipgen_params(top)
        for inst, info in sv.items():
            ip_type = info["type"]
            merged = dict(info["params"])
            ipgen_params = ipgen.get(ip_type, {})
            # Namespace ipgen keys so an SV param and a template param that
            # happen to share a name cannot silently collide. Drop the identity
            # keys *before* namespacing -- otherwise "topname" survives as
            # "ipgen:topname" and makes every ipgen IP differ trivially.
            for k, v in ipgen_params.items():
                if k in IDENTITY_KEYS:
                    continue
                merged[f"ipgen:{k}"] = v
            canon = canonical_params(merged)
            instances.append({
                "top": top,
                "instance": inst,
                "ip": ip_type,
                "attr": info["attr"],
                "params": canon,
                "local": info["local"],
                "has_ipgen": ip_type in ipgen,
                "fingerprint": fingerprint(canon),
            })
    return instances


WILDCARD_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z_0-9]*)\}")


def subst_wildcards(text, subs):
    """Resolve the wildcards we can; report the rest as unresolved.

    dvsim substitutes a large set of wildcards, most of which depend on the tool
    and scratch layout. For cost accounting we only need the ones that appear in
    include and testplan paths, so anything else marks the path unresolvable and
    we skip it rather than guess.
    """
    missing = []

    def repl(m):
        key = m.group(1)
        if key in subs:
            return str(subs[key])
        missing.append(key)
        return m.group(0)

    return WILDCARD_RE.sub(repl, text), missing


def _merge_cfg(target, new):
    """Approximate dvsim's import merge: lists concatenate, scalars first-wins.

    See dvsim's flow/hjson.py. We only need `tests`, `regressions`, `testplan`
    and `import_cfgs`, all of which follow this rule.
    """
    for key, val in new.items():
        if key not in target:
            target[key] = val
        elif isinstance(target[key], list) and isinstance(val, list):
            target[key] = target[key] + val
        # Scalars: the first definition wins, matching "default-looking values
        # lose" closely enough for counting purposes.


def load_cfg_tree(path, seen=None):
    """Load a sim cfg and everything it imports, flattened.

    dvsim errors on diamonds in the include graph, so a plain recursive walk is
    faithful; `seen` only guards against pathological self-inclusion.
    """
    path = Path(path)
    seen = seen if seen is not None else set()
    if path in seen or not path.is_file():
        return {}
    seen.add(path)

    data = parse_hjson(str(path))
    if data is None:
        return {}

    merged = {k: v for k, v in data.items() if k != "import_cfgs"}
    subs = {"proj_root": str(REPO_ROOT), "self_dir": str(path.parent)}
    for inc in data.get("import_cfgs", []):
        resolved, missing = subst_wildcards(str(inc), subs)
        if missing:
            # Tool-dependent include (e.g. {tool}.hjson); irrelevant to reseed
            # counts, so skipping it does not bias the result.
            continue
        _merge_cfg(merged, load_cfg_tree(Path(resolved), seen))
    return merged


DEFAULT_RESEED = 1


def cfg_cost(path):
    """Reseed-weighted simulation runs and test count for one sim cfg.

    Reseed weighting matters: csrng declares 8 tests but a reseed sum of 1235,
    so counting test entries understates its cost by two orders of magnitude.
    """
    cfg = load_cfg_tree(path)
    tests = cfg.get("tests", [])
    if not isinstance(tests, list):
        return {"tests": 0, "runs": 0}

    # A cfg-level `reseed` is the default for every test that does not set its
    # own, so it must be picked up or blocks like prim_esc (cfg reseed 20) are
    # counted at one run per test.
    try:
        default_reseed = int(normalise_value(cfg.get("reseed", DEFAULT_RESEED)))
    except (TypeError, ValueError):
        default_reseed = DEFAULT_RESEED

    runs = 0
    for t in tests:
        if not isinstance(t, dict):
            continue
        try:
            runs += int(normalise_value(t.get("reseed", default_reseed)))
        except (TypeError, ValueError):
            runs += default_reseed
    return {"tests": len(tests), "runs": runs}


IP_FROM_PATH_RE = [
    re.compile(r"/hw/ip/([^/]+)/"),
    re.compile(r"/hw/top_[^/]+/ip_autogen/([^/]+)/"),
    re.compile(r"/hw/top_[^/]+/ip/([^/]+)/"),
    re.compile(r"/hw/ip_templates/([^/]+)/"),
]


def ip_from_cfg_path(path):
    text = str(path)
    for rx in IP_FROM_PATH_RE:
        m = rx.search(text)
        if m:
            return m.group(1)
    return None


def load_batch(top):
    """The sim cfgs a top's batch regression actually runs."""
    batch = REPO_ROOT / "hw" / top / "dv" / f"{top}_sim_cfgs.hjson"
    if not batch.is_file():
        return None, []
    data = parse_hjson(str(batch)) or {}
    subs = {"proj_root": str(REPO_ROOT), "self_dir": str(batch.parent)}
    cfgs = []
    for entry in data.get("use_cfgs", []):
        if not isinstance(entry, str):
            continue  # in-line dict cfg; no path to compare across tops
        resolved, missing = subst_wildcards(entry, subs)
        if missing:
            continue
        cfgs.append(Path(resolved))
    return batch, cfgs


def classify(instances):
    """Group instances of the same IP into parameter equivalence classes.

    Three relations are kept separate because they license different
    conclusions:

      1. no exposed parameters at all -- top-agnostic by construction, so the
         DV is redundant across tops no matter what the tests do;
      2. parameters present and all equal -- same RTL under a different name,
         so likewise redundant;
      3. parameters differ -- the RTL genuinely differs, and whether a given
         test is redundant depends on whether it touches the differing
         parameter. This is a worklist, not a saving.
    """
    by_ip = defaultdict(list)
    for inst in instances:
        by_ip[inst["ip"]].append(inst)

    report = {}
    for ip, insts in sorted(by_ip.items()):
        classes = defaultdict(list)
        for i in insts:
            classes[i["fingerprint"]].append(i)
        tops = sorted({i["top"] for i in insts})
        no_params = all(not i["params"] for i in insts)

        # What matters for cross-top dedup is whether a class *spans tops*, not
        # how many classes there are. edn splits into two classes because edn0
        # and edn1 differ from each other (NumEndPoints 8 vs 1), but each class
        # contains the matching instance from every top, so both are redundant
        # across tops. Counting classes alone would wrongly call that differing.
        class_tops = {fp: sorted({m["top"] for m in ms})
                      for fp, ms in classes.items()}
        spanning = [fp for fp, ts in class_tops.items() if len(ts) > 1]

        if len(tops) < 2:
            relation = "single-top"
        elif no_params:
            relation = "top-agnostic"
        elif len(spanning) == len(classes):
            relation = "identical-config"
        elif spanning:
            relation = "partially-shared"
        else:
            relation = "differing-config"

        # Which parameters are responsible for the split.
        differing = []
        if relation in ("differing-config", "partially-shared"):
            all_keys = set()
            for i in insts:
                all_keys |= set(i["params"])
            for key in sorted(all_keys):
                # Key by top/instance, not top: several instances of one IP can
                # live in the same top (uart0..uart3, sram_ctrl main vs ret) and
                # keying by top alone would silently keep only the last of them.
                # ABSENT, not None: a parameter that this instance does not
                # declare at all is a different thing from one explicitly set
                # to null, and the two must not render identically.
                vals = {f"{i['top']}/{i['instance']}": i["params"].get(key, ABSENT)
                        for i in insts}
                if len({json.dumps(v, sort_keys=True, default=str)
                        for v in vals.values()}) > 1:
                    differing.append({"param": key, "values": vals})

        report[ip] = {
            "relation": relation,
            "tops": tops,
            "instances": len(insts),
            "classes": [
                {"fingerprint": fp,
                 "spans_tops": len(class_tops[fp]) > 1,
                 "tops": class_tops[fp],
                 "members": sorted(f"{m['top']}/{m['instance']}" for m in ms)}
                for fp, ms in sorted(classes.items())
            ],
            "spanning_classes": len(spanning),
            "differing_params": differing,
        }
    return report


def batch_analysis(tops):
    """Per-top batch regressions, and the cfgs shared verbatim between them."""
    per_top = {}
    for top in tops:
        batch, cfgs = load_batch(top)
        if batch is None:
            per_top[top] = None
            continue
        costs = {}
        for c in cfgs:
            costs[str(c.relative_to(REPO_ROOT))] = cfg_cost(c)
        per_top[top] = {
            "batch": str(batch.relative_to(REPO_ROOT)),
            "cfgs": costs,
            "total_tests": sum(v["tests"] for v in costs.values()),
            "total_runs": sum(v["runs"] for v in costs.values()),
        }

    # A cfg path referenced by more than one top's batch is not a per-top copy;
    # it is the same file, run once per top, with no parameterisation at all.
    counts = defaultdict(list)
    for top, info in per_top.items():
        if info:
            for cfg in info["cfgs"]:
                counts[cfg].append(top)
    shared = {c: ts for c, ts in counts.items() if len(ts) > 1}

    dup_runs = 0
    dup_tests = 0
    for cfg, ts in shared.items():
        cost = next(per_top[t]["cfgs"][cfg] for t in ts)
        # First run is real work; every additional top repeats it.
        dup_runs += cost["runs"] * (len(ts) - 1)
        dup_tests += cost["tests"] * (len(ts) - 1)

    return per_top, shared, {"runs": dup_runs, "tests": dup_tests}


def render_value(val, width=46):
    """Render a parameter value compactly for the terminal report.

    Some parameters are large structures -- alert_handler's async_on is a
    77-element vector, otp_ctrl's otp_mmap a nested dict -- and printing them
    verbatim buries the report. The full values are always in --json.
    """
    if isinstance(val, list):
        return f"<{len(val)} items>" if len(val) > 4 else repr(val)
    if isinstance(val, dict):
        return f"<{len(val)} keys>"
    text = repr(val)
    return text if len(text) <= width else text[:width - 3] + "..."


RELATION_ORDER = ["top-agnostic", "identical-config", "partially-shared",
                  "differing-config", "single-top"]
RELATION_BLURB = {
    "top-agnostic": "no exposed parameters -- DV is redundant across tops",
    "identical-config": "every config class spans tops -- DV is redundant "
                        "across tops",
    "partially-shared": "some config classes span tops, some are top-specific",
    "differing-config": "no config class spans tops -- needs per-test "
                        "relevance review",
    "single-top": "instantiated in one top only -- nothing to dedup",
}


def runs_by_ip(per_top):
    """Reseed-weighted runs attributed to each IP across every top's batch."""
    ip_runs = defaultdict(int)
    for info in per_top.values():
        if not info:
            continue
        for cfg, cost in info["cfgs"].items():
            ip = ip_from_cfg_path("/" + cfg)
            if ip:
                ip_runs[ip] += cost["runs"]
    return ip_runs


def worklist_rows(classes, ip_runs, relations=("differing-config",)):
    """Differing IPs ranked by the DV cost sitting behind their parameters."""
    rows = [
        (ip_runs.get(ip, 0), ip, [d["param"] for d in c["differing_params"]])
        for ip, c in classes.items() if c["relation"] in relations
    ]
    return sorted(rows, reverse=True)


def total_runs(per_top):
    return sum(i["total_runs"] for i in per_top.values() if i)


def print_report(classes, per_top, shared, dup, verbose):
    w = sys.stdout.write

    w("\n=== IP equivalence across tops ===\n\n")
    for relation in RELATION_ORDER:
        ips = {ip: c for ip, c in classes.items() if c["relation"] == relation}
        if not ips:
            continue
        w(f"-- {relation}: {RELATION_BLURB[relation]}\n")
        if relation == "single-top" and not verbose:
            w(f"   {len(ips)} IPs (use --verbose to list)\n\n")
            continue
        for ip, c in sorted(ips.items()):
            short_tops = ",".join(t.removeprefix("top_") for t in c["tops"])
            w(f"   {ip:<20} {c['instances']:>2} inst  "
              f"{len(c['classes'])} class(es)  tops={short_tops}\n")
            if relation in ("differing-config", "partially-shared"):
                for d in c["differing_params"]:
                    vals = ", ".join(
                        f"{t.removeprefix('top_')}={render_value(d['values'][t])}"
                        for t in sorted(d["values"]))
                    w(f"        {d['param']}: {vals}\n")
        w("\n")

    w("=== Batch regression cost (reseed-weighted simulation runs) ===\n\n")
    for top, info in sorted(per_top.items()):
        if info is None:
            w(f"   {top:<24} no batch sim cfg\n")
            continue
        w(f"   {top:<24} {len(info['cfgs']):>3} cfgs  "
          f"{info['total_tests']:>5} tests  {info['total_runs']:>6} runs\n")
    w("\n")

    w(f"=== Cfgs shared verbatim by more than one top: {len(shared)} ===\n\n")
    for cfg, tops in sorted(shared.items()):
        cost = next(per_top[t]["cfgs"][cfg] for t in tops)
        w(f"   {cost['runs']:>6} runs x{len(tops)-1} repeat  {cfg}\n")
    w(f"\n   Duplicated: {dup['runs']} reseed-weighted runs "
      f"({dup['tests']} test entries)\n")

    total = total_runs(per_top)
    if total:
        w(f"   That is {100 * dup['runs'] / total:.0f}% of the "
          f"{total} runs across all batch regressions.\n")
    w("\n")

    w("=== Parameter worklist: differing params ranked by DV cost behind them ===\n\n")
    rows = worklist_rows(classes, runs_by_ip(per_top))
    if not rows:
        w("   (none)\n")
    for runs, ip, params in rows:
        w(f"   {runs:>6} runs  {ip:<20} {', '.join(params) or '(name only)'}\n")
    w("\n")


def resolve_scratch_root(arg_scratch_root):
    """Mirror dvsim's scratch-root precedence so reports land where expected.

    dvsim (cli/run.py::resolve_scratch_root) prefers an explicit argument, then
    $SCRATCH_ROOT if it is usable, then <proj_root>/scratch. Matching that means
    an HTML report shows up beside the flow output people already look at.
    """
    if arg_scratch_root:
        return Path(arg_scratch_root).resolve()
    env = os.environ.get("SCRATCH_ROOT")
    if env and Path(env).is_dir():
        return Path(env).resolve()
    return REPO_ROOT / "scratch"


def current_branch():
    """Branch name, for the {scratch_root}/{branch} layout the project uses."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return "no-branch"
    name = out.stdout.strip() or "no-branch"
    # Branch names may contain slashes, which would nest the report directory.
    return name.replace("/", "_")


def report_dir(arg_scratch_root):
    """Where the HTML report is written: {scratch_root}/{branch}/ip_equivalence."""
    return resolve_scratch_root(arg_scratch_root) / current_branch() / "ip_equivalence"


def serve(directory, port):
    """Serve the report directory over HTTP until interrupted.

    Bound to localhost: this is a developer report that may describe unreleased
    configuration, so it should not be reachable from the network by default.
    """
    directory = str(directory)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def log_message(self, fmt, *args):
            # The default handler logs every asset request to stderr, which
            # buries the "serving at" line people actually need.
            pass

    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        actual = httpd.server_address[1]
        print(f"serving {directory}\n  http://127.0.0.1:{actual}/\n"
              f"Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


# Bootstrap supplies the page chrome (typography, cards, tables, badges,
# progress bars) and, importantly, a theme whose text/background contrast is
# already correct in both modes. Only the data-mark colours are ours.
BOOTSTRAP_CSS = ("https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/"
                 "bootstrap.min.css")

# Validated categorical slots 1-3 from the project data-viz palette, checked
# against Bootstrap's own surfaces (#212529 dark, #ffffff light) rather than a
# generic one. Colour is bound to meaning, not rank: blue = work shared between
# tops, orange = top-specific, aqua = partially shared. Dark passes every gate
# including contrast; light leaves aqua at 2.82:1, so the relief rule applies --
# every bar carries a visible value and every number is repeated in a table.
BASE_CSS = """
/* Ground colours are set with literal values, never through a custom property.
   The first version of this report defined its tokens on an inner wrapper and
   consumed them on `body`; custom properties only inherit downward, so `body`
   resolved neither one, `color` fell back to initial black and the background
   stayed transparent -- black text on the browser's dark canvas. Literals here
   mean the page is legible even if the Bootstrap stylesheet fails to load. */
html { background: #212529; color: #dee2e6; }
html[data-bs-theme="light"] { background: #ffffff; color: #212529; }

:root {
  --mark-shared: #3987e5;
  --mark-specific: #d95926;
  --mark-partial: #199e70;
}
[data-bs-theme="light"] {
  --mark-shared: #2a78d6;
  --mark-specific: #eb6834;
  --mark-partial: #1baf7a;
}

.hero-fig { font-size: 3.25rem; font-weight: 600; line-height: 1; }
.metric { font-size: 1.6rem; font-weight: 600; }

/* One source of truth for the bar rhythm. These are px, not rem, and the row
   height is derived from the bar height rather than set independently: the two
   used to be a hard-coded 18px bar inside a 1.625rem row, which only lines up
   at a 16px root font. On a browser whose root is 12px that row computed to
   19.5px and collapsed onto the bar, leaving 1.5px of slack and a rhythm that
   drifted with the reader's font settings. */
:root { --bar-h: 18px; --bar-row-gap: 6px; }

/* Bars: thin, square at the baseline, 4px rounded at the data end. */
.progress, .progress-stacked { height: var(--bar-h); border-radius: 2px; }
.progress-bar { border-radius: 0 4px 4px 0; }
.bar-shared { background-color: var(--mark-shared); }
.bar-specific { background-color: var(--mark-specific); }
.bar-partial { background-color: var(--mark-partial); }
/* A 2px gap in the surface colour separates touching segments -- never a
   stroke, which would add ink that is not data. */
.progress-stacked > .progress { margin-right: 2px; }
.progress-stacked > .progress:last-child { margin-right: 0; }

/* Every row is pinned to the same height so the bars line up as an even
   column. Without grid-auto-rows the row grew to fit a wrapped label, and rows
   whose label fitted on one line sat tighter than those that did not -- the
   ragged spacing that made the worklist hard to scan. Labels are authored
   short enough to fit, so the ellipsis is a safety net rather than the plan;
   either way the full text is on the hover title and in the tables below, so
   nothing is only available by truncation. */
.bar-grid {
  display: grid; grid-template-columns: minmax(11em, 1.1fr) 3fr auto;
  grid-auto-rows: var(--bar-h); row-gap: var(--bar-row-gap);
  column-gap: 12px; align-items: center;
}
/* Labels and values inherit the body size rather than shrinking to .8125rem,
   which rendered at 9.75px on a 12px root. At the body size their line box
   matches --bar-h, so text and bar occupy the same height on every row. */
.bar-label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-val {
  font-variant-numeric: tabular-nums;
  white-space: nowrap; text-align: right;
}
.key-dot {
  width: .625rem; height: .625rem; border-radius: 50%;
  display: inline-block; flex: none;
}
td code, .param-cell code { overflow-wrap: anywhere; }
.param-cell { line-height: 1.55; }
"""


def esc(text):
    """Escape for HTML text and quoted attributes."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def fmt(n):
    return f"{n:,}"


def bar_rows(items, mark, total=None):
    """Ranked bars, widest first, value labelled at every tip.

    Single series, so no legend -- the section heading names what is plotted.
    The per-row value doubles as this chart's table view, which is also the
    relief the light-mode palette requires.

    Items are ``(label, value)`` or ``(label, value, detail)``. Labels must stay
    short enough to sit on one line: the grid pins every row to the same height
    so the bars form an even column, and a label that wrapped would either be
    cut off or reintroduce the ragged spacing. Anything longer belongs in
    ``detail``, which rides the hover title.
    """
    if not items:
        return "<p class='text-body-secondary mb-0'>(none)</p>"
    scale = total or max((it[1] for it in items), default=0) or 1
    out = ["<div class='bar-grid'>"]
    for item in items:
        label, value = item[0], item[1]
        detail = item[2] if len(item) > 2 else label
        pct = 100.0 * value / scale
        out.append(
            f"<div class='bar-label text-body-secondary'>{esc(label)}</div>"
            f"<div class='progress' role='progressbar'"
            f" aria-label='{esc(detail)}' aria-valuenow='{value}'"
            f" aria-valuemin='0' aria-valuemax='{scale}'"
            f" title='{esc(detail)}: {fmt(value)} runs'>"
            f"<div class='progress-bar {mark}' style='width:{pct:.2f}%'></div>"
            f"</div>"
            f"<div class='bar-val'>{fmt(value)}</div>")
    out.append("</div>")
    return "".join(out)


def legend(*entries):
    keys = "".join(
        f"<span class='d-inline-flex align-items-center gap-2 me-3'>"
        f"<span class='key-dot' style='background:var(--mark-{mark})'></span>"
        f"<span class='text-body-secondary small'>{esc(label)}</span></span>"
        for mark, label in entries)
    return f"<div class='mb-3'>{keys}</div>"


def shared_split(info, shared):
    """Split one top's runs into work shared with another top vs top-specific."""
    sh = sum(cost["runs"] for cfg, cost in info["cfgs"].items() if cfg in shared)
    return sh, info["total_runs"] - sh


def stacked_top_rows(per_top, shared):
    """Per-top batch cost, split shared vs top-specific.

    Two series, so a legend is mandatory. Widths use one common scale across
    tops so the bars are comparable rather than each normalised to itself.
    """
    scale = max((i["total_runs"] for i in per_top.values() if i), default=0) or 1
    out = [legend(("shared", "Shared with another top"),
                  ("specific", "Top-specific")),
           "<div class='bar-grid'>"]
    for top, info in sorted(per_top.items()):
        label = esc(top.removeprefix("top_"))
        if info is None:
            out.append(
                # Not .small: at the body size this note's line box matches
                # --bar-h, so the row sits on the same rhythm as the bar rows.
                f"<div class='bar-label text-body-secondary'>{label}</div>"
                f"<div class='text-body-secondary fst-italic' "
                f"style='grid-column:2/4'>no batch sim cfg</div>")
            continue
        sh, sp = shared_split(info, shared)
        out.append(
            f"<div class='bar-label text-body-secondary'>{label}</div>"
            f"<div class='progress-stacked'>"
            f"<div class='progress' role='progressbar' aria-label='shared'"
            f" aria-valuenow='{sh}' aria-valuemin='0' aria-valuemax='{scale}'"
            f" style='width:{100.0 * sh / scale:.2f}%'"
            f" title='shared: {fmt(sh)} runs'>"
            f"<div class='progress-bar bar-shared'></div></div>"
            f"<div class='progress' role='progressbar' aria-label='top-specific'"
            f" aria-valuenow='{sp}' aria-valuemin='0' aria-valuemax='{scale}'"
            f" style='width:{100.0 * sp / scale:.2f}%'"
            f" title='top-specific: {fmt(sp)} runs'>"
            f"<div class='progress-bar bar-specific'></div></div>"
            f"</div>"
            f"<div class='bar-val'>{fmt(info['total_runs'])}</div>")
    out.append("</div>")

    # Interior stacked segments carry no inline label, so the numbers need a
    # non-hover home.
    out.append(
        "<details class='mt-3'><summary class='text-body-secondary small'>"
        "Table view</summary><div class='table-responsive mt-2'>"
        "<table class='table table-sm align-middle mb-0'><thead><tr>"
        "<th>Top</th><th class='text-end'>Shared</th>"
        "<th class='text-end'>Top-specific</th><th class='text-end'>Total runs</th>"
        "<th class='text-end'>Cfgs</th><th class='text-end'>Tests</th>"
        "</tr></thead><tbody>")
    for top, info in sorted(per_top.items()):
        label = esc(top.removeprefix("top_"))
        if info is None:
            out.append(f"<tr><td>{label}</td>"
                       f"<td colspan='5' class='text-body-secondary'>"
                       f"no batch sim cfg</td></tr>")
            continue
        sh, sp = shared_split(info, shared)
        out.append(
            f"<tr><td>{label}</td>"
            f"<td class='text-end font-monospace'>{fmt(sh)}</td>"
            f"<td class='text-end font-monospace'>{fmt(sp)}</td>"
            f"<td class='text-end font-monospace'>{fmt(info['total_runs'])}</td>"
            f"<td class='text-end font-monospace'>{len(info['cfgs'])}</td>"
            f"<td class='text-end font-monospace'>{fmt(info['total_tests'])}</td>"
            f"</tr>")
    out.append("</tbody></table></div></details>")
    return "".join(out)


# Relation -> the mark colour carrying its meaning, and the Bootstrap text
# utility for the badge. The badge always shows its text label, so hue never
# has to carry the distinction alone.
RELATION_ROLE = {
    "top-agnostic": "shared",
    "identical-config": "shared",
    "partially-shared": "partial",
    "differing-config": "specific",
    "single-top": None,
}


def relation_badge(relation):
    role = RELATION_ROLE.get(relation)
    colour = f"var(--mark-{role})" if role else "var(--bs-secondary-color)"
    return (f"<span class='d-inline-flex align-items-center gap-2'>"
            f"<span class='key-dot' style='background:{colour}'></span>"
            f"{esc(relation)}</span>")


def param_cell(params):
    if not params:
        return "<span class='text-body-tertiary'>&mdash;</span>"
    return "<br>".join(
        f"<code>{esc(d['param'])}</code> "
        + ", ".join(
            f"{esc(k.removeprefix('top_'))}=<code>"
            f"{esc(render_value(d['values'][k]))}</code>"
            for k in sorted(d["values"]))
        for d in params)


def equivalence_tables(classes, ip_runs):
    """One table per relation, with the differing parameters spelled out."""
    out = []
    for relation in RELATION_ORDER:
        ips = {ip: c for ip, c in classes.items() if c["relation"] == relation}
        if not ips:
            continue
        out.append(
            f"<h2 class='h6 mt-5 mb-1 pb-2 border-bottom'>"
            f"{relation_badge(relation)}"
            f"<span class='text-body-secondary fw-normal ms-2'>"
            f"{len(ips)} IPs</span></h2>"
            f"<p class='text-body-secondary small'>"
            f"{esc(RELATION_BLURB[relation])}</p>"
            f"<div class='card'><div class='table-responsive'>"
            f"<table class='table table-sm table-hover align-middle mb-0'>"
            f"<thead><tr><th>IP</th><th class='text-end'>Inst</th>"
            f"<th class='text-end'>Classes</th><th>Tops</th>"
            f"<th class='text-end'>Runs</th>"
            f"<th>Differing parameters</th></tr></thead><tbody>")
        for ip, c in sorted(ips.items()):
            tops = ", ".join(t.removeprefix("top_") for t in c["tops"])
            out.append(
                f"<tr><td><code>{esc(ip)}</code></td>"
                f"<td class='text-end'>{c['instances']}</td>"
                f"<td class='text-end'>{len(c['classes'])}</td>"
                f"<td class='small'>{esc(tops)}</td>"
                f"<td class='text-end font-monospace'>"
                f"{fmt(ip_runs.get(ip, 0))}</td>"
                f"<td class='param-cell'>{param_cell(c['differing_params'])}</td>"
                f"</tr>")
        out.append("</tbody></table></div></div>")
    return "".join(out)


# Dark is the default; the toggle only needs to persist a deliberate override.
THEME_JS = """
(function () {
  var KEY = 'ip-equivalence-theme';
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem(KEY);
    if (saved) root.setAttribute('data-bs-theme', saved);
  } catch (e) { /* private mode: keep the default theme */ }
  var btn = document.getElementById('theme-toggle');
  if (!btn) return;
  var sync = function () {
    var dark = root.getAttribute('data-bs-theme') !== 'light';
    btn.textContent = dark ? 'Light mode' : 'Dark mode';
  };
  sync();
  btn.addEventListener('click', function () {
    var next = root.getAttribute('data-bs-theme') === 'light' ? 'dark' : 'light';
    root.setAttribute('data-bs-theme', next);
    try { localStorage.setItem(KEY, next); } catch (e) { /* ignore */ }
    sync();
  });
})();
"""


def render_html(classes, per_top, shared, dup, tops):
    """A Bootstrap-based HTML report, dark by default.

    Bootstrap comes from a CDN, so unlike the first version this page is no
    longer strictly offline: if the stylesheet does not load the layout falls
    back to unstyled flow. The literal ground colours in BASE_CSS keep it
    readable in that case rather than reproducing the black-on-dark failure.
    """
    ip_runs = runs_by_ip(per_top)
    total = total_runs(per_top)
    pct = (100.0 * dup["runs"] / total) if total else 0.0

    shared_items = sorted(
        ((cfg.split("/")[-1].removesuffix("_sim_cfg.hjson"),
          next(per_top[t]["cfgs"][cfg] for t in ts)["runs"] * (len(ts) - 1))
         for cfg, ts in shared.items()),
        key=lambda kv: -kv[1])

    # The bar label is the IP name plus a parameter count -- always one line, so
    # every row is the same height. Spelling the parameter names out here is
    # what made the rows ragged: rv_core_ibex has 24 of them. The names ride the
    # hover title and are listed in full in the per-relation tables below.
    work_items = []
    for runs, ip, params in worklist_rows(
            classes, ip_runs, ("differing-config", "partially-shared")):
        n = len(params)
        if n == 0:
            label, detail = f"{ip}  (name only)", f"{ip}: differs by name only"
        else:
            label = f"{ip}  ({n} param{'s' if n > 1 else ''})"
            detail = f"{ip}: {', '.join(params)}"
        work_items.append((label, runs, detail))

    redundant_ips = sum(
        1 for c in classes.values()
        if c["relation"] in ("top-agnostic", "identical-config"))

    rev = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=False).stdout.strip() or "unknown"

    tiles = [
        ("Total runs, all batches", fmt(total)),
        ("Duplicated", f"{pct:.0f}%"),
        ("Cfgs shared verbatim", str(len(shared))),
        ("IPs redundant across tops", str(redundant_ips)),
    ]
    tile_html = "".join(
        f"<div class='col-6 col-lg-3'><div class='card h-100'>"
        f"<div class='card-body py-3'>"
        f"<div class='text-body-secondary small'>{esc(label)}</div>"
        f"<div class='metric'>{esc(value)}</div></div></div></div>"
        for label, value in tiles)

    return f"""<!doctype html>
<html lang="en" data-bs-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IP equivalence across tops</title>
<link rel="stylesheet" href="{BOOTSTRAP_CSS}">
<style>{BASE_CSS}</style>
</head>
<body>
<div class="container py-4" style="max-width: 1140px">

  <div class="d-flex justify-content-between align-items-start gap-3 mb-1">
    <h1 class="h3 mb-0">IP equivalence across tops</h1>
    <button id="theme-toggle" type="button"
            class="btn btn-sm btn-outline-secondary flex-none">Light mode</button>
  </div>
  <p class="text-body-secondary mb-1">Which block-level DV is genuinely
  duplicated between tops, and how much simulation sits behind each answer.</p>
  <p class="text-body-secondary small"><code>{esc(REPO_ROOT.name)}</code>
  &middot; {esc(len(tops))} tops
  ({esc(', '.join(sorted(t.removeprefix('top_') for t in tops)))})
  &middot; branch <code>{esc(current_branch())}</code>
  &middot; revision <code>{esc(rev)}</code>
  &middot; reseed-weighted run counts</p>

  <div class="card mb-3">
    <div class="card-body">
      <div class="row g-4 align-items-center">
        <div class="col-md-4">
          <div class="hero-fig">{fmt(dup['runs'])}</div>
          <div class="text-body-secondary small mt-1">duplicated
          reseed-weighted runs</div>
        </div>
        <div class="col-md-8">
          <p class="mb-0 text-body-secondary">{len(shared)} sim cfgs are
          referenced by more than one top's batch regression &mdash; the
          <em class="text-body">same file</em>, with no per-top
          parameterisation. Re-running them per top costs
          <span class="text-body">{fmt(dup['runs'])} runs</span>
          ({pct:.0f}% of all {fmt(total)}) and {fmt(dup['tests'])} test
          entries for no additional coverage.</p>
        </div>
      </div>
    </div>
  </div>

  <div class="row g-3 mb-2">{tile_html}</div>

  <h2 class="h6 mt-5 mb-1 pb-2 border-bottom">Batch regression cost per top</h2>
  <p class="text-body-secondary small">Bars share one scale, so tops are
  directly comparable.</p>
  <div class="card"><div class="card-body">
    {stacked_top_rows(per_top, shared)}
  </div></div>

  <h2 class="h6 mt-5 mb-1 pb-2 border-bottom">Cfgs shared verbatim between
  tops</h2>
  <p class="text-body-secondary small">Duplicated runs per cfg &mdash; the cost
  of every repeat beyond the first top. Zero coverage risk to remove.</p>
  <div class="card"><div class="card-body">
    {bar_rows(shared_items, 'bar-shared')}
  </div></div>

  <h2 class="h6 mt-5 mb-1 pb-2 border-bottom">Parameter worklist</h2>
  <p class="text-body-secondary small">IPs whose parameters differ across tops,
  ranked by the DV cost behind them. These are <em class="text-body">not</em>
  savings: each needs a per-test check of whether the test actually depends on
  the differing parameter.</p>
  <div class="card"><div class="card-body">
    {bar_rows(work_items, 'bar-specific')}
  </div></div>

  {equivalence_tables(classes, ip_runs)}

  <h2 class="h6 mt-5 mb-1 pb-2 border-bottom">How to read this</h2>
  <div class="card"><div class="card-body">
    <p class="mb-0 text-body-secondary">A run is redundant only if the test
    <em class="text-body">and</em> the RTL under it match. Testplans are
    near-identical across tops, but almost every generated IP is parameterised
    differently, so identical testplans alone prove nothing. The blue sections
    are provable today; the orange one is the worklist that needs per-test
    parameter relevance before anything can be cut.</p>
  </div></div>

</div>
<script>{THEME_JS}</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="Report which IPs are identical and identically "
                    "configured across tops.")
    ap.add_argument("--tops", nargs="*",
                    help="restrict to these tops (default: all)")
    ap.add_argument("--json", metavar="PATH",
                    help="also write the full report as JSON")
    ap.add_argument("--verbose", action="store_true",
                    help="list single-top IPs too")
    ap.add_argument("--html", nargs="?", const=True, metavar="PATH",
                    help="write an HTML report; with no PATH it goes to "
                         "{scratch_root}/{branch}/ip_equivalence/index.html")
    ap.add_argument("--scratch-root", metavar="PATH",
                    help="override the scratch root (default: $SCRATCH_ROOT, "
                         "else <proj_root>/scratch)")
    ap.add_argument("--serve", nargs="?", const=8000, type=int, metavar="PORT",
                    help="write the HTML report and serve it on 127.0.0.1:PORT "
                         "(default 8000); implies --html")
    ap.add_argument("--repo-root", metavar="PATH",
                    help="checkout to analyse (default: the git toplevel of "
                         "the current directory, like dvsim's proj_root)")
    ap.add_argument("--quiet", action="store_true",
                    help="skip the text report (useful with --html/--serve)")
    args = ap.parse_args()

    global REPO_ROOT
    REPO_ROOT = resolve_repo_root(args.repo_root)

    tops = discover_tops(args.tops)
    if not tops:
        sys.exit(f"no tops found under {REPO_ROOT}/hw/top_*/data/autogen/\n"
                 f"Use --repo-root to point at a different checkout.")
    # Always say which checkout this is: the tool used to derive the root from
    # its own location, so running it from another tree reported that tree's
    # tops without a word about it.
    print(f"repo: {REPO_ROOT}")
    print(f"tops: {', '.join(sorted(t.removeprefix('top_') for t in tops))}")

    instances = build_instances(tops)
    classes = classify(instances)
    per_top, shared, dup = batch_analysis(tops)

    if not args.quiet:
        print_report(classes, per_top, shared, dup, args.verbose)

    if args.json:
        out = {
            "tops": sorted(tops),
            "instances": instances,
            "equivalence": classes,
            "batches": per_top,
            "shared_cfgs": {c: sorted(t) for c, t in shared.items()},
            "duplicated": dup,
        }
        Path(args.json).write_text(json.dumps(out, indent=2, sort_keys=True,
                                              default=str))
        print(f"wrote {args.json}")

    # --serve implies --html: there is nothing to serve otherwise.
    want_html = args.html is not None or args.serve is not None
    if not want_html:
        return

    if isinstance(args.html, str):
        html_path = Path(args.html).resolve()
    else:
        html_path = report_dir(args.scratch_root) / "index.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(classes, per_top, shared, dup, tops))
    print(f"wrote {html_path}")

    if args.serve is not None:
        serve(html_path.parent, args.serve)


if __name__ == "__main__":
    main()
