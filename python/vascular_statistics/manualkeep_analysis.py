"""Reproducible analysis for the manually retained skeleton experiment."""
from __future__ import annotations
import csv, hashlib, json, math, re, shutil, tempfile
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

POPS = {"full": "", "d0-10um": "_d0-10um", "d10+um": "_d10+um"}
METRICS = ("avg_diameter_um", "avg_length_um", "segment_density_per_mm3", "avg_tortuosity_au")
LABELS = {"\u5e73\u5747\u76f4\u5f84": METRICS[0], "\u5e73\u5747\u957f\u5ea6": METRICS[1], "\u6bb5\u5bc6\u5ea6": METRICS[2], "\u5e73\u5747\u5f2f\u66f2\u5ea6": METRICS[3]}

def parse_summary(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\u5e73\u5747\u76f4\u5f84|\u5e73\u5747\u957f\u5ea6|\u6bb5\u5bc6\u5ea6|\u5e73\u5747\u5f2f\u66f2\u5ea6)(?:\s*[\(\uff08].*?[\)\uff09])?\s*:\s*(-?[\d.eE+]+)", line)
        if m:
            out[LABELS[m.group(1)]] = float(m.group(2))
    if set(out) != set(METRICS) or not all(math.isfinite(v) for v in out.values()):
        raise ValueError(f"invalid statistics file: {path}")
    return out


def aggregate_sample(sample: Path, population: str):
    values, paths = [], []
    for run in sorted(sample.glob("run_*")):
        path = run / f"statistics_summary{POPS[population]}.txt"
        if run.is_dir() and path.is_file():
            values.append(parse_summary(path)); paths.append(path)
    if not values:
        raise ValueError(f"no {population} statistics: {sample}")
    return {m: float(np.mean([v[m] for v in values])) for m in METRICS}, len(values), paths


def bh(p_values):
    p = np.asarray(p_values, dtype=float); out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
        out[order] = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 1)
    return out.tolist()


def rank_biserial(delta):
    nz = delta[delta != 0]
    if not len(nz): return 0.0
    ranks = rankdata(np.abs(nz)); pos = ranks[nz > 0].sum(); neg = ranks[nz < 0].sum()
    return float((pos - neg) / (pos + neg))


def bootstrap_ci(delta, rng, n_boot):
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    return [float(v) for v in np.percentile(np.median(delta[idx], axis=1), [2.5, 97.5])]


def input_hash(paths):
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=str):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def collect(final_root: Path, sources: dict[str, Path]):
    manifest_path = final_root / "manual_cut_manifest.tsv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    rows, paths, errors, mismatches = [], [manifest_path, final_root / "manual_cut_experiment.json"], [], []
    status_counts, experimenter_counts = {}, {}
    for item in manifest:
        try:
            exp = item["experimenter"]; key = item["unified_sample_key"]
            final_dir = final_root / key; source_dir = sources[exp] / item["source_sample_key"]
            meta_path = final_dir / "sample_metadata.json"; meta = json.loads(meta_path.read_text(encoding="utf-8"))
            paths.append(meta_path); parsed = meta.get("path_parsed", {}); parts = item["source_sample_key"].split("/")
            context = {
                "experimenter": exp, "sample_key": key, "source_sample_key": item["source_sample_key"],
                "status": item["status"], "series": parts[0], "group": str(parsed.get("group", parts[0])),
                "subgroup": parts[1] if exp == "xiaoqian" and len(parts) > 1 else parts[0],
                "batch_id": str(parsed.get("batch_id", parsed.get("animal_id", "?"))),
                "study_day": parsed.get("study_day"), "study_day_label": str(parsed.get("study_day_label", parsed.get("daypoint", "?"))),
                "acquisition_date": str(parsed.get("acquisition_date", "?")),
                "original_volume_mm3": float(item["original_volume_mm3"]),
                "retained_volume_mm3": float(item["retained_volume_mm3"]),
                "retained_fraction": float(item["retained_fraction"]),
                "deleted_nodes": int(item["deleted_nodes"]), "deleted_edges": int(item["deleted_edges"]),
            }
            for pop in POPS:
                final, fn, fp = aggregate_sample(final_dir, pop)
                source, sn, sp = aggregate_sample(source_dir, pop); paths.extend(fp + sp)
                if fn != sn or fn != int(item["run_count"]):
                    mismatches.append({"sample_key": key, "population": pop, "manifest": int(item["run_count"]), "source": sn, "final": fn})
                for metric in METRICS:
                    s, f = source[metric], final[metric]
                    rows.append({**context, "population": pop, "metric": metric, "source_value": s, "final_value": f,
                                 "delta_abs": f-s, "delta_pct": (f/s-1)*100 if s else math.nan,
                                 "source_run_count": sn, "final_run_count": fn})
            status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
            experimenter_counts[exp] = experimenter_counts.get(exp, 0) + 1
        except Exception as exc:
            errors.append({"sample_key": item.get("unified_sample_key", "?"), "error": str(exc)})
    audit = {"manifest_samples": len(manifest), "paired_samples": sum(experimenter_counts.values()), "metric_rows": len(rows),
             "status_counts": status_counts, "experimenter_counts": experimenter_counts,
             "run_pair_mismatches": mismatches, "errors": errors, "input_sha256": input_hash(paths)}
    return rows, audit


def grouped(rows, cut_only=True):
    out = {}
    for row in rows:
        if cut_only and row["status"] != "quality_region_retained": continue
        for stratum in ("all", row["experimenter"]):
            out.setdefault((stratum, row["population"], row["metric"]), []).append(row)
    return out


def paired_table(rows, seed, n_boot):
    rng = np.random.default_rng(seed); out = []
    for (stratum, pop, metric), group in sorted(grouped(rows).items()):
        source = np.array([r["source_value"] for r in group]); final = np.array([r["final_value"] for r in group]); delta = final-source
        try: p = float(wilcoxon(delta, zero_method="wilcox").pvalue)
        except ValueError: p = 1.0
        low, high = bootstrap_ci(delta, rng, n_boot)
        out.append({"stratum": stratum, "population": pop, "metric": metric, "n_samples": len(group),
                    "source_mean": source.mean(), "source_median": np.median(source), "final_mean": final.mean(),
                    "final_median": np.median(final), "median_delta_abs": np.median(delta),
                    "median_delta_pct": np.median([r["delta_pct"] for r in group]),
                    "bootstrap_median_delta_ci_low": low, "bootstrap_median_delta_ci_high": high,
                    "wilcoxon_p": p, "rank_biserial": rank_biserial(delta)})
    for row, q in zip(out, bh([r["wilcoxon_p"] for r in out])): row["bh_fdr_q"] = q
    return out


def correlation_table(rows):
    out = []
    for (stratum, pop, metric), group in sorted(grouped(rows).items()):
        for predictor in ("retained_fraction", "deleted_nodes", "deleted_edges"):
            result = spearmanr([r[predictor] for r in group], [r["delta_pct"] for r in group])
            out.append({"stratum": stratum, "population": pop, "metric": metric, "predictor": predictor,
                        "n_samples": len(group), "spearman_rho": result.statistic, "spearman_p": result.pvalue})
    for row, q in zip(out, bh([r["spearman_p"] for r in out])): row["bh_fdr_q"] = q
    return out


def descriptive_table(rows):
    groups = {}
    for row in rows:
        levels = (("experimenter", row["experimenter"], "all", "all", "all"),
                  ("group", row["experimenter"], row["group"], "all", "all"),
                  ("group_day", row["experimenter"], row["group"], "all", row["study_day_label"]),
                  ("subgroup", row["experimenter"], row["group"], row["subgroup"], "all"),
                  ("subgroup_day", row["experimenter"], row["group"], row["subgroup"], row["study_day_label"]))
        for level, exp, group, subgroup, day in levels:
            key = (level, exp, group, subgroup, day, row["population"], row["metric"])
            groups.setdefault(key, []).append(row["final_value"])
    out = []
    for key, values in sorted(groups.items(), key=lambda x: tuple(map(str, x[0]))):
        a = np.array(values)
        out.append({"level": key[0], "experimenter": key[1], "group": key[2],
                    "subgroup": key[3], "study_day_label": key[4],
                    "population": key[5], "metric": key[6], "n_samples": len(a), "mean": a.mean(),
                    "stdev": a.std(ddof=1) if len(a)>1 else math.nan, "median": np.median(a),
                    "q1": np.percentile(a,25), "q3": np.percentile(a,75), "min": a.min(), "max": a.max()})
    return out

def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def make_plots(rows, output):
    cut = [r for r in rows if r["status"] == "quality_region_retained"]
    fig, axes = plt.subplots(3,4,figsize=(15,9),constrained_layout=True)
    for i,pop in enumerate(POPS):
        for j,metric in enumerate(METRICS):
            ax=axes[i,j]; values=[r["delta_pct"] for r in cut if r["population"]==pop and r["metric"]==metric]
            ax.axhline(0,color="#777",lw=.8); ax.boxplot(values,showfliers=False)
            ax.scatter(np.ones(len(values))+np.linspace(-.08,.08,len(values)),values,s=8,alpha=.4,color="#0072B2")
            ax.set_xticks([]); ax.set_title(f"{pop} | {metric}",fontsize=8); ax.set_ylabel("Change (%)"); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Manual retention effect in 105 adjusted samples"); fig.savefig(output/"paired_changes.png",dpi=180); plt.close(fig)
    full=[r for r in cut if r["population"]=="full"]; colors={"xiaoqian":"#0072B2","huaien":"#D55E00"}
    fig,axes=plt.subplots(1,4,figsize=(15,3.8),constrained_layout=True)
    for ax,metric in zip(axes,METRICS):
        for exp,color in colors.items():
            part=[r for r in full if r["metric"]==metric and r["experimenter"]==exp]
            ax.scatter([r["retained_fraction"] for r in part],[r["delta_pct"] for r in part],s=16,alpha=.7,label=exp,color=color)
        ax.axhline(0,color="#777",lw=.8); ax.set_title(metric,fontsize=8); ax.set_xlabel("Retained fraction"); ax.set_ylabel("Change (%)"); ax.grid(alpha=.2)
    axes[0].legend(frameon=False); fig.suptitle("Retained volume and full-population metric change")
    fig.savefig(output/"retention_association.png",dpi=180); plt.close(fig)


def run_analysis(final_root: Path, output_root: Path, sources: dict[str, Path], seed=20260720, n_boot=10000, overwrite=False):
    final_root=final_root.resolve(); output_root=output_root.resolve()
    if output_root.exists() and not overwrite: raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix=f".{output_root.name}.",dir=output_root.parent))
    try:
        rows,audit=collect(final_root,sources)
        if audit["errors"] or audit["run_pair_mismatches"] or audit["paired_samples"]!=145 or len(rows)!=1740:
            raise ValueError(f"pairing audit failed: {audit}")
        no_cut=[r for r in rows if r["status"]=="no_cut_keep_full"]
        audit["no_cut_max_abs_metric_delta"]=max(abs(r["delta_abs"]) for r in no_cut)
        if audit["no_cut_max_abs_metric_delta"]>1e-12: raise ValueError("no-cut identity control failed")
        paired=paired_table(rows,seed,n_boot); corr=correlation_table(rows); desc=descriptive_table(rows)
        unique=list({r["sample_key"]:r for r in rows}.values()); cut=[r for r in unique if r["status"]=="quality_region_retained"]
        fractions=np.array([r["retained_fraction"] for r in cut])
        retention={"n_all":len(unique),"n_cut":len(cut),"n_full_retained":len(unique)-len(cut),
                   "retained_fraction_cut":{"mean":fractions.mean(),"median":np.median(fractions),"q1":np.percentile(fractions,25),
                                            "q3":np.percentile(fractions,75),"min":fractions.min(),"max":fractions.max()},
                   "deleted_nodes_total":sum(r["deleted_nodes"] for r in cut),"deleted_edges_total":sum(r["deleted_edges"] for r in cut)}
        write_csv(work/"sample_metrics.csv",rows); write_csv(work/"paired_summary.csv",paired)
        write_csv(work/"correlation_summary.csv",corr); write_csv(work/"final_descriptive.csv",desc)
        outliers=sorted([r for r in rows if r["status"]=="quality_region_retained"],key=lambda r:abs(r["delta_pct"]),reverse=True)[:100]
        write_csv(work/"largest_relative_changes.csv",outliers); make_plots(rows,work)
        payload={"schema_version":"1.0","final_root":str(final_root),"source_roots":{k:str(v.resolve()) for k,v in sources.items()},
                 "parameters":{"seed":seed,"bootstrap_iterations":n_boot,"bh_fdr_scope":"all tests within each output table"},
                 "audit":audit,"retention":retention}
        (work/"analysis_manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=float)+"\n",encoding="utf-8")
        (work/"README.md").write_text("# Manual-keep statistical analysis artifacts\n\nInferential tests use the 105 adjusted samples; 40 full-retained samples are identity controls. Biological groups are descriptive only.\n",encoding="utf-8")
        if output_root.exists(): shutil.rmtree(output_root)
        work.rename(output_root); return payload
    except Exception:
        shutil.rmtree(work,ignore_errors=True); raise
