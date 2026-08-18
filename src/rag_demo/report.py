import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from string import Template

REPORTS_DIR = Path(__file__).resolve().parents[2] / "docs" / "reports"
RUNS_DIR = Path(__file__).resolve().parents[2] / "docs" / "eval-runs"

MODEL_LABELS = {"e5": "e5", "bge_m3": "bge-m3"}
MODEL_COLORS = {
    "e5": {"light": "#2a78d6", "dark": "#3987e5"},
    "bge_m3": {"light": "#eb6834", "dark": "#d95926"},
}
FALLBACK_COLOR = {"light": "#6b6b68", "dark": "#9a9a95"}

HIGHER_IS_BETTER = {"not_found_rate": False, "avg_query_ms": False}


def _label(model_key: str) -> str:
    return MODEL_LABELS.get(model_key, model_key)


def _color(model_key: str, mode: str) -> str:
    return MODEL_COLORS.get(model_key, FALLBACK_COLOR)[mode]


def _metric_keys(results: list[dict]) -> tuple[list[str], list[str], list[str]]:
    sample = results[0]
    recall_keys = sorted(
        (k for k in sample if k.startswith("recall@")),
        key=lambda k: int(k.split("@")[1]),
    )
    mrr_keys = [k for k in sample if k.startswith("mrr@")]
    other_keys = [k for k in sample if k not in {"model", "n", *recall_keys, *mrr_keys}]
    return recall_keys, mrr_keys, other_keys


METRIC_LABELS = {"not_found_rate": "Not found rate", "avg_query_ms": "Avg query latency"}


def _metric_label(key: str) -> str:
    if key in METRIC_LABELS:
        return METRIC_LABELS[key]
    if key.startswith("recall@"):
        return f"Recall@{key.split('@')[1]}"
    if key.startswith("mrr@"):
        return f"MRR@{key.split('@')[1]}"
    return key


def _fmt(key: str, value: float) -> str:
    if key.startswith("recall@") or key == "not_found_rate":
        return f"{value * 100:.2f}%"
    if key.startswith("mrr@"):
        return f"{value:.4f}"
    if key == "avg_query_ms":
        return f"{value:.1f}ms"
    return f"{value}"


def _winner(results: list[dict], key: str) -> str:
    higher_is_better = HIGHER_IS_BETTER.get(key, True)
    pick = max if higher_is_better else min
    best = pick(results, key=lambda r: r[key])
    return best["model"]


def _metric_row(results: list[dict], key: str) -> str:
    cells = "".join(f'<td class="num tnum">{_fmt(key, r[key])}</td>' for r in results)
    winner = _winner(results, key)
    delta_cell = "<td class=\"num tnum\">-</td>"
    if len(results) == 2:
        delta = abs(results[0][key] - results[1][key])
        delta_cell = f'<td class="num tnum">{_fmt(key, delta)}</td>'
    return (
        f"<tr><td>{_metric_label(key)}</td>{cells}{delta_cell}"
        f'<td><span class="winner"><span class="dot" style="background:{_color(winner, "light")}"></span>'
        f"{_label(winner)}</span></td></tr>"
    )


def _stat_tile(results: list[dict], key: str, label: str) -> str:
    higher_is_better = HIGHER_IS_BETTER.get(key, True)
    ordered = sorted(results, key=lambda r: r[key], reverse=higher_is_better)
    best, rest = ordered[0], ordered[1:]
    rest_str = " · ".join(f"{_label(r['model'])} {_fmt(key, r[key])}" for r in rest)
    return f"""
        <div class="stat">
          <span class="label">{label}</span>
          <span class="value tnum"><span class="win" style="color:{_color(best['model'], 'light')}">{_fmt(key, best[key])}</span><span class="lose">{_label(best['model'])}</span></span>
          <span class="foot">{rest_str}</span>
        </div>"""


def render_html(run: dict) -> str:
    results = run["results"]
    recall_keys, mrr_keys, other_keys = _metric_keys(results)

    n = run.get("n_samples")
    seed = run.get("seed")
    run_at = run.get("run_at", "")
    run_date = run_at.split("T")[0] if "T" in run_at else run_at
    model_names = ", ".join(_label(r["model"]) for r in results)

    stat_tiles = "".join(
        [
            _stat_tile(results, recall_keys[0], _metric_label(recall_keys[0])) if recall_keys else "",
            _stat_tile(results, recall_keys[-1], _metric_label(recall_keys[-1])) if recall_keys else "",
            _stat_tile(results, "avg_query_ms", "평균 쿼리 지연") if "avg_query_ms" in other_keys else "",
        ]
    )

    metric_rows = "".join(
        _metric_row(results, k) for k in [*recall_keys, *mrr_keys, *other_keys]
    )

    legend = "".join(
        f'<span class="item"><span class="swatch" style="background:{_color(r["model"], "light")}"></span>{_label(r["model"])}</span>'
        for r in results
    )

    metric_header = "".join(f'<th class="num">{_label(r["model"])}</th>' for r in results)

    chart_data = {
        "categories": recall_keys,
        "series": [
            {
                "key": r["model"],
                "label": _label(r["model"]),
                "colorLight": _color(r["model"], "light"),
                "colorDark": _color(r["model"], "dark"),
                "values": [r[k] for k in recall_keys],
            }
            for r in results
        ],
    }

    template = Template(HTML_TEMPLATE)
    return template.substitute(
        run_at=run_at,
        run_date=run_date,
        n=n,
        seed=seed,
        model_names=model_names,
        stat_tiles=stat_tiles,
        legend=legend,
        metric_header=metric_header,
        metric_rows=metric_rows,
        chart_data_json=json.dumps(chart_data, ensure_ascii=False),
    )


HTML_TEMPLATE = """<title>Retrieval Eval $run_date</title>
<style>
  .re {
    color-scheme: light;
    --page:        #f9f9f7;
    --surface:     #fcfcfb;
    --surface-2:   #f3f2ee;
    --text-1:      #0b0b0b;
    --text-2:      #52514e;
    --muted:       #898781;
    --grid:        #e1e0d9;
    --border:      rgba(11,11,11,0.10);
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .re {
      color-scheme: dark;
      --page:        #0d0d0d;
      --surface:     #1a1a19;
      --surface-2:   #202020;
      --text-1:      #ffffff;
      --text-2:      #c3c2b7;
      --muted:       #898781;
      --grid:        #2c2c2a;
      --border:      rgba(255,255,255,0.10);
    }
  }
  :root[data-theme="dark"] .re {
    color-scheme: dark;
    --page:        #0d0d0d;
    --surface:     #1a1a19;
    --surface-2:   #202020;
    --text-1:      #ffffff;
    --text-2:      #c3c2b7;
    --muted:       #898781;
    --grid:        #2c2c2a;
    --border:      rgba(255,255,255,0.10);
  }
  .re { background: var(--page); color: var(--text-1); font-family: system-ui, -apple-system, "Segoe UI", "Malgun Gothic", sans-serif; padding: 56px 24px 80px; box-sizing: border-box; }
  .re * { box-sizing: border-box; }
  .re .wrap { max-width: 860px; margin: 0 auto; display: flex; flex-direction: column; gap: 40px; }
  .re .mono { font-family: ui-monospace, "SFMono-Regular", "Consolas", "Cascadia Code", monospace; }
  .re .tnum { font-variant-numeric: tabular-nums; }
  .re header { display: flex; flex-direction: column; gap: 10px; border-bottom: 1px solid var(--border); padding-bottom: 28px; }
  .re .eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-2); font-family: ui-monospace, monospace; }
  .re h1 { font-size: 26px; line-height: 1.25; font-weight: 800; margin: 0; letter-spacing: -0.01em; }
  .re .sub { font-size: 14px; color: var(--text-2); margin: 0; max-width: 66ch; line-height: 1.55; }
  .re .meta { display: flex; gap: 18px; flex-wrap: wrap; font-size: 12.5px; color: var(--muted); font-family: ui-monospace, monospace; }
  .re section { display: flex; flex-direction: column; gap: 16px; }
  .re .section-title { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  .re h2 { font-size: 13px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-1); margin: 0; }
  .re .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .re .stat { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; display: flex; flex-direction: column; gap: 6px; }
  .re .stat .label { font-size: 11.5px; color: var(--muted); font-weight: 600; letter-spacing: 0.02em; }
  .re .stat .value { font-size: 24px; font-weight: 800; line-height: 1; display: flex; align-items: baseline; gap: 8px; }
  .re .stat .value .lose { font-size: 13px; font-weight: 600; color: var(--muted); }
  .re .stat .foot { font-size: 12px; color: var(--text-2); }
  .re .legend { display: flex; gap: 16px; font-size: 12.5px; color: var(--text-2); }
  .re .legend .item { display: flex; align-items: center; gap: 6px; }
  .re .legend .swatch { width: 10px; height: 10px; border-radius: 2px; }
  .re .chart-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px 20px 8px; overflow-x: auto; }
  .re .chart-inner { min-width: 560px; }
  .re svg text { font-family: ui-monospace, monospace; fill: var(--muted); }
  .re .bar { cursor: pointer; transition: opacity 0.12s ease; }
  .re .bar:hover { opacity: 0.82; }
  .re .cat-label { font-size: 11.5px; fill: var(--text-2); font-family: system-ui, -apple-system, "Malgun Gothic", sans-serif; }
  .re .axis-label { font-size: 10.5px; }
  .re .val-label { font-size: 10px; font-weight: 600; }
  .re .gridline { stroke: var(--grid); stroke-width: 1; }
  .re .baseline { stroke: var(--muted); stroke-width: 1; }
  .re #tooltip { position: absolute; pointer-events: none; opacity: 0; transition: opacity 0.1s ease; background: var(--text-1); color: var(--page); font-size: 12px; font-family: ui-monospace, monospace; padding: 6px 9px; border-radius: 6px; white-space: nowrap; z-index: 5; transform: translate(-50%, -100%); }
  .re .table-scroll { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
  .re table { width: 100%; border-collapse: collapse; font-size: 13px; min-width: 560px; }
  .re th { text-align: left; font-size: 11px; letter-spacing: 0.03em; text-transform: uppercase; color: var(--muted); font-weight: 600; padding: 10px 14px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  .re td { padding: 11px 14px; border-bottom: 1px solid var(--border); color: var(--text-1); white-space: nowrap; }
  .re tr:last-child td { border-bottom: none; }
  .re td.num { text-align: right; }
  .re .winner { display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px; font-weight: 700; }
  .re .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
  .re footer { font-size: 11.5px; color: var(--muted); font-family: ui-monospace, monospace; padding-top: 8px; border-top: 1px solid var(--border); }
  @media (max-width: 640px) { .re .stats { grid-template-columns: 1fr; } }
</style>

<div class="re">
  <div class="wrap">
    <header>
      <div class="eyebrow">RAG retrieval eval · auto-generated</div>
      <h1>Retrieval 평가 리포트</h1>
      <p class="sub">$model_names 임베딩 테이블에 동일 질의 셋을 던져 Recall@k · MRR · 검색 지연시간을 측정한 자동 생성 리포트입니다.</p>
      <div class="meta">
        <span>run_at · $run_at</span>
        <span>samples · $n</span>
        <span>seed · $seed</span>
      </div>
    </header>

    <section>
      <div class="stats">$stat_tiles</div>
    </section>

    <section>
      <div class="section-title">
        <h2>Recall@k 비교</h2>
        <div class="legend">$legend</div>
      </div>
      <div class="chart-card">
        <div class="chart-inner">
          <svg id="chart" width="100%" viewBox="0 0 680 300" role="img" aria-label="Recall@k 비교 막대그래프"></svg>
        </div>
      </div>
    </section>

    <section>
      <h2>상세 지표</h2>
      <div class="table-scroll">
        <table>
          <thead><tr><th>지표</th>$metric_header<th class="num">차이(Δ)</th><th>우세</th></tr></thead>
          <tbody>$metric_rows</tbody>
        </table>
      </div>
    </section>

    <footer>generated via rag_demo.report · rag-demo</footer>
  </div>
</div>

<div id="tooltip"></div>

<script>
(function () {
  var chartData = $chart_data_json;
  var svg = document.getElementById("chart");
  var ns = "http://www.w3.org/2000/svg";
  var W = 680, H = 300;
  var plotTop = 14, plotBottom = 240, plotLeft = 40, plotRight = 20;
  var plotH = plotBottom - plotTop;
  var plotW = W - plotLeft - plotRight;
  var yMax = 1.0;
  var isDark = matchMedia("(prefers-color-scheme: dark)").matches;
  var themeAttr = document.documentElement.getAttribute("data-theme");
  if (themeAttr === "dark") isDark = true;
  if (themeAttr === "light") isDark = false;

  function y(v) { return plotBottom - (v / yMax) * plotH; }
  function el(tag, attrs) {
    var e = document.createElementNS(ns, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  var ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  ticks.forEach(function (t) {
    svg.appendChild(el("line", { class: "gridline", x1: plotLeft, x2: W - plotRight, y1: y(t), y2: y(t) }));
    var lbl = el("text", { class: "axis-label", x: plotLeft - 8, y: y(t) + 3, "text-anchor": "end" });
    lbl.textContent = Math.round(t * 100) + "%";
    svg.appendChild(lbl);
  });
  svg.appendChild(el("line", { class: "baseline", x1: plotLeft, x2: W - plotRight, y1: plotBottom, y2: plotBottom }));

  var nSeries = chartData.series.length;
  var groupW = plotW / chartData.categories.length;
  var gap = 6;
  var barW = Math.min(36, (groupW * 0.7 - gap * (nSeries - 1)) / nSeries);

  var tooltip = document.getElementById("tooltip");
  var chartCard = document.querySelector(".chart-card");

  function roundedTopPath(x, yTop, w, h, r) {
    var yBot = yTop + h;
    if (h < r) r = h;
    return "M" + x + "," + (yTop + r) + " Q" + x + "," + yTop + " " + (x + r) + "," + yTop +
      " L" + (x + w - r) + "," + yTop + " Q" + (x + w) + "," + yTop + " " + (x + w) + "," + (yTop + r) +
      " L" + (x + w) + "," + yBot + " L" + x + "," + yBot + " Z";
  }

  function addBar(x, w, value, color, catLabel, seriesLabel) {
    var h = (value / yMax) * plotH;
    var top = plotBottom - h;
    var path = el("path", { class: "bar", d: roundedTopPath(x, top, w, h, 4), fill: color });
    svg.appendChild(path);
    var vlbl = el("text", { class: "val-label", x: x + w / 2, y: top - 6, "text-anchor": "middle", fill: color });
    vlbl.textContent = (value * 100).toFixed(1) + "%";
    svg.appendChild(vlbl);
    path.addEventListener("mousemove", function (ev) {
      var rect = chartCard.getBoundingClientRect();
      tooltip.style.left = (ev.clientX - rect.left + chartCard.scrollLeft) + "px";
      tooltip.style.top = (ev.clientY - rect.top - 10) + "px";
      tooltip.style.opacity = 1;
      tooltip.textContent = catLabel + " · " + seriesLabel + " · " + (value * 100).toFixed(2) + "%";
    });
    path.addEventListener("mouseleave", function () { tooltip.style.opacity = 0; });
  }

  chartData.categories.forEach(function (cat, ci) {
    var gx = plotLeft + ci * groupW;
    var pairW = barW * nSeries + gap * (nSeries - 1);
    var startX = gx + (groupW - pairW) / 2;
    chartData.series.forEach(function (s, si) {
      var color = isDark ? s.colorDark : s.colorLight;
      addBar(startX + si * (barW + gap), barW, s.values[ci], color, cat, s.label);
    });
    var lbl = el("text", { class: "cat-label", x: gx + groupW / 2, y: plotBottom + 22, "text-anchor": "middle" });
    lbl.textContent = cat;
    svg.appendChild(lbl);
  });
})();
</script>
"""


def _latest_run_file() -> Path:
    files = sorted(RUNS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"평가 결과 JSON이 없습니다: {RUNS_DIR}")
    return files[-1]


def generate_report(run: dict, run_id: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html = render_html(run)
    out_path = REPORTS_DIR / f"retrieval-eval-{run_id}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None, help="평가 결과 JSON 경로 (기본: 최신 실행)")
    args = parser.parse_args()

    run_file = args.run or _latest_run_file()
    run = json.loads(run_file.read_text(encoding="utf-8"))
    out_path = generate_report(run, run_file.stem)
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
