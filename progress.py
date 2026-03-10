"""Plot autoresearch experiment progress from results.tsv.

Outputs:
- logs/progress.png: matplotlib chart for headless/CI
- reports/index.html: interactive Plotly.js dashboard
- reports/results.json: data file for live auto-refresh

Usage:
  python progress.py                    # default: results.tsv -> both outputs
  python progress.py -i results.tsv     # explicit input
  python progress.py --html-only        # skip matplotlib, just generate HTML
"""

import argparse
import csv
import json
import os
import time

HIGHER_IS_BETTER = {"dna"}

METRIC_LABELS = {
    "lm": ("val_bpb", "Validation BPB", True),
    "dna": ("accuracy", "Accuracy", False),
    "ts": ("val_mse", "Validation MSE", True),
}


def read_results(path):
    """Read results.tsv into a list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                row["val_metric"] = float(row["val_metric"])
            except (ValueError, TypeError):
                row["val_metric"] = None
            try:
                row["params"] = int(row["params"])
            except (ValueError, TypeError):
                row["params"] = 0
            rows.append(row)
    return rows


def compute_stats(rows):
    """Compute summary statistics from experiment rows."""
    task = rows[0]["task"] if rows else "lm"
    higher_better = task in HIGHER_IS_BETTER

    kept = [r for r in rows if r.get("status", "").strip() == "keep" and r["val_metric"] is not None]
    discarded = [r for r in rows if r.get("status", "").strip() == "discard"]
    crashes = [r for r in rows if r.get("status", "").strip() == "crash"]

    if kept:
        best_fn = max if higher_better else min
        best_row = best_fn(kept, key=lambda r: r["val_metric"])
        best_metric = best_row["val_metric"]
    else:
        best_metric = None

    return {
        "task": task,
        "total": len(rows),
        "kept": len(kept),
        "discarded": len(discarded),
        "crashes": len(crashes),
        "keep_rate": len(kept) / len(rows) * 100 if rows else 0,
        "best_metric": best_metric,
        "higher_is_better": higher_better,
    }


def format_params(n):
    """Format parameter count: 431000 -> '431K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


# ---------------------------------------------------------------------------
# Matplotlib output (for headless/CI)
# ---------------------------------------------------------------------------


def plot_matplotlib(rows, out_path="logs/progress.png"):
    """Plot experiment progress chart with matplotlib."""
    import matplotlib.pyplot as plt

    if not rows:
        print("No results to plot")
        return

    task = rows[0]["task"]
    higher_better = task in HIGHER_IS_BETTER

    kept_x, kept_y, kept_desc = [], [], []
    discard_x, discard_y = [], []
    crash_x, crash_y = [], []

    for i, row in enumerate(rows):
        if row["val_metric"] is None:
            continue
        status = row.get("status", "").strip()
        if status == "keep":
            kept_x.append(i)
            kept_y.append(row["val_metric"])
            kept_desc.append(row.get("description", ""))
        elif status == "crash":
            crash_x.append(i)
            crash_y.append(row["val_metric"])
        else:
            discard_x.append(i)
            discard_y.append(row["val_metric"])

    best_x, best_y = [], []
    running_best = None
    for x, y in zip(kept_x, kept_y):
        if running_best is None:
            running_best = y
        elif higher_better:
            running_best = max(running_best, y)
        else:
            running_best = min(running_best, y)
        best_x.append(x)
        best_y.append(running_best)

    n_total = len(rows)
    n_kept = len(kept_x)
    _, ylabel, _ = METRIC_LABELS.get(task, ("val_metric", "val_metric", True))

    fig, ax = plt.subplots(figsize=(14, 6))
    if discard_x:
        ax.scatter(discard_x, discard_y, c="#cccccc", s=20, alpha=0.6, label="Discarded", zorder=2)
    if crash_x:
        ax.scatter(crash_x, crash_y, c="#ff4444", s=30, marker="x", alpha=0.8, label="Crash", zorder=3)
    if kept_x:
        ax.scatter(kept_x, kept_y, c="#22cc66", s=50, alpha=0.9, label="Kept", zorder=4)
    if best_x:
        ax.step(best_x, best_y, where="post", c="#22cc66", linewidth=1.5, alpha=0.7, label="Running best", zorder=3)

    for x, y, desc in zip(kept_x, kept_y, kept_desc):
        if desc and desc.lower() != "baseline":
            label = desc[:40] + "..." if len(desc) > 40 else desc
            ax.annotate(label, (x, y), fontsize=5, alpha=0.6, rotation=15, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Experiment #")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Autotune Progress: {n_total} Experiments, {n_kept} Kept Improvements")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# HTML dashboard output
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nanostate autoresearch</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    --bg: #2c3039;
    --bg-card: #353a45;
    --border: #434956;
    --border-active: #545c6b;
    --text: #eaecf0;
    --text-secondary: #a0a8b8;
    --text-muted: #6e7787;
    --green: #34d399;
    --green-muted: rgba(52, 211, 153, 0.15);
    --red: #f87171;
    --red-muted: rgba(248, 113, 113, 0.15);
    --gray: #6e7787;
    --gray-muted: rgba(110, 119, 135, 0.25);
    --chart-bg: #393f4a;
    --chart-grid: #434956;
    --font-mono: "JetBrains Mono", monospace;
    --font-body: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
  }
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  body {
    font-family: var(--font-body);
    background: var(--bg);
    color: var(--text);
    line-height: 1.45;
    -webkit-font-smoothing: antialiased;
    padding: 32px;
    max-width: 1200px;
    margin: 0 auto;
  }
  @media (max-width: 600px) {
    body { padding: 16px; }
  }

  .header { margin-bottom: 28px; }
  .header .label {
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 1.5px;
    text-transform: uppercase;
  }
  .header h1 {
    font-size: 24px;
    font-weight: 600;
    margin-top: 4px;
    letter-spacing: -0.5px;
  }
  .header .meta {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
  }

  .stats {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }
  .stat {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    flex: 1;
    min-width: 130px;
  }
  .stat .value {
    font-family: var(--font-mono);
    font-size: 22px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  .stat .label {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .stat .sub {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
  }
  .stat.highlight { border-color: var(--green); }
  .stat .value.green { color: var(--green); }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
  }
  @media (max-width: 600px) {
    .card { padding: 16px; border-radius: 10px; }
  }
  .card-header { margin-bottom: 16px; }
  .card-title { font-size: 14px; font-weight: 600; }
  .card-subtitle {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  #chart { width: 100%; height: 400px; }

  .chart-legend {
    display: flex;
    gap: 16px;
    margin-top: 12px;
    justify-content: center;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-secondary);
  }
  .legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .legend-line {
    width: 14px;
    height: 2px;
    flex-shrink: 0;
    border-radius: 1px;
  }

  .table-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .filters { display: flex; gap: 6px; }
  .filter-btn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 11px;
    color: var(--text-secondary);
    cursor: pointer;
    font-family: var(--font-body);
    transition: all 0.15s;
  }
  .filter-btn.active {
    background: var(--green-muted);
    border-color: var(--green);
    color: var(--green);
    font-weight: 500;
  }
  .filter-btn:hover { border-color: var(--border-active); }

  table {
    width: 100%;
    font-size: 12px;
    border-collapse: collapse;
  }
  th {
    padding: 8px;
    text-align: left;
    font-weight: 500;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  th:hover { color: var(--text-secondary); }
  td { padding: 8px; border-bottom: 1px solid var(--bg); }
  tr:hover td { background: var(--bg); }
  td.mono {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
  }
  td.metric {
    font-family: var(--font-mono);
    font-weight: 500;
    font-variant-numeric: tabular-nums;
  }

  .badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.3px;
  }
  .badge.keep { background: var(--green-muted); color: var(--green); }
  .badge.discard { background: var(--gray-muted); color: var(--text-muted); }
  .badge.crash { background: var(--red-muted); color: var(--red); }

  .live-indicator {
    text-align: center;
    margin-top: 20px;
    font-size: 11px;
    color: var(--text-muted);
  }
  .dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
  }
  .dot.live { background: var(--green); animation: pulse 2s infinite; }
  .dot.static { background: var(--gray); }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* custom plotly tooltip override */
  .plotly .hoverlayer .hovertext rect {
    rx: 8 !important;
    ry: 8 !important;
  }
</style>
</head>
<body>

<div class="header">
  <div class="label">nanostate autoresearch</div>
  <h1>Experiment Progress</h1>
  <div class="meta" id="meta"></div>
</div>

<div class="stats" id="stats"></div>

<div class="card">
  <div class="card-header">
    <div class="card-title">Autotune Progress</div>
    <div class="card-subtitle" id="chart-subtitle">Experiment outcomes over time</div>
  </div>
  <div id="chart"></div>
  <div class="chart-legend" id="chart-legend"></div>
</div>

<div class="card">
  <div class="table-controls">
    <div class="card-header" style="margin-bottom:0">
      <div class="card-title">Experiment Log</div>
      <div class="card-subtitle">Click column headers to sort</div>
    </div>
    <div class="filters" id="filters"></div>
  </div>
  <table>
    <thead>
      <tr>
        <th data-col="idx">#</th>
        <th data-col="commit">Commit</th>
        <th data-col="val_metric">Metric</th>
        <th data-col="params">Params</th>
        <th data-col="status">Status</th>
        <th data-col="description">Description</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<div class="live-indicator" id="live-indicator"></div>

<script>
var DATA = __DATA_PLACEHOLDER__;
var isLive = false;
var currentFilter = 'all';
var sortCol = 'idx';
var sortAsc = true;

function render(data) {
  var s = data.stats;
  var mk = data.metric_key;

  // meta
  document.getElementById('meta').textContent =
    'Task: ' + s.task + '  \\u00b7  Updated: ' + data.generated;

  // chart subtitle
  document.getElementById('chart-subtitle').textContent =
    s.total + ' experiments, ' + s.kept + ' kept improvements';

  // stats (DOM-safe)
  var best = s.best_metric !== null ? s.best_metric.toFixed(4) : '\\u2014';
  var improvement = '';
  if (s.best_metric !== null && data.rows.length > 0) {
    var baseline = null;
    for (var i = 0; i < data.rows.length; i++) {
      if (data.rows[i].status === 'keep' && data.rows[i].val_metric !== null) {
        baseline = data.rows[i].val_metric; break;
      }
    }
    if (baseline !== null && baseline !== s.best_metric) {
      var pct = ((baseline - s.best_metric) / baseline * 100).toFixed(1);
      improvement = (data.lower_is_better ? '-' : '+') + pct + '% from baseline';
    }
  }

  var statsEl = document.getElementById('stats');
  while (statsEl.firstChild) statsEl.removeChild(statsEl.firstChild);
  var statsData = [
    { value: best, label: 'Best ' + mk, sub: improvement, cls: 'green', highlight: true },
    { value: String(s.total), label: 'Experiments', sub: '', cls: '', highlight: false },
    { value: String(s.kept), label: 'Kept', sub: s.keep_rate.toFixed(1) + '% keep rate', cls: 'green', highlight: false },
    { value: String(s.discarded), label: 'Discarded', sub: '', cls: '', highlight: false },
    { value: String(s.crashes), label: 'Crashes', sub: '', cls: '', highlight: false },
  ];
  statsData.forEach(function(item) {
    var div = document.createElement('div');
    div.className = 'stat' + (item.highlight ? ' highlight' : '');
    var valDiv = document.createElement('div');
    valDiv.className = 'value' + (item.cls ? ' ' + item.cls : '');
    valDiv.textContent = item.value;
    div.appendChild(valDiv);
    var labDiv = document.createElement('div');
    labDiv.className = 'label';
    labDiv.textContent = item.label;
    div.appendChild(labDiv);
    if (item.sub) {
      var subDiv = document.createElement('div');
      subDiv.className = 'sub';
      subDiv.textContent = item.sub;
      div.appendChild(subDiv);
    }
    statsEl.appendChild(div);
  });

  renderChart(data);
  renderLegend();
  renderFilters();
  renderTable(data);

  // live indicator
  var indEl = document.getElementById('live-indicator');
  while (indEl.firstChild) indEl.removeChild(indEl.firstChild);
  var dot = document.createElement('span');
  dot.className = isLive ? 'dot live' : 'dot static';
  indEl.appendChild(dot);
  indEl.appendChild(document.createTextNode(
    isLive ? 'Live \\u2014 refreshing every 30s' : 'Static snapshot \\u2014 ' + data.generated
  ));
}

function renderChart(data) {
  var rows = data.rows;
  var kept = rows.filter(function(r) { return r.status === 'keep' && r.val_metric !== null; });
  var discarded = rows.filter(function(r) { return r.status === 'discard' && r.val_metric !== null; });
  var crashes = rows.filter(function(r) { return r.status === 'crash' && r.val_metric !== null; });

  var bestX = [], bestY = [], running = null;
  var hb = data.stats.higher_is_better;
  kept.forEach(function(r) {
    if (running === null) running = r.val_metric;
    else running = hb ? Math.max(running, r.val_metric) : Math.min(running, r.val_metric);
    bestX.push(r.idx);
    bestY.push(running);
  });

  var traces = [
    {
      x: discarded.map(function(r){return r.idx;}),
      y: discarded.map(function(r){return r.val_metric;}),
      mode: 'markers', type: 'scatter', name: 'Discarded', showlegend: false,
      marker: { color: '#6e7787', size: 6, opacity: 0.7 },
      text: discarded.map(function(r){return r.description;}),
      customdata: discarded.map(function(r){return [r.commit, r.params_fmt];}),
      hovertemplate: '<b>%{text}</b><br>%{y:.4f}<br>%{customdata[0]} \\u00b7 %{customdata[1]}<extra>#%{x}</extra>',
    },
    {
      x: crashes.map(function(r){return r.idx;}),
      y: crashes.map(function(r){return r.val_metric;}),
      mode: 'markers', type: 'scatter', name: 'Crash', showlegend: false,
      marker: { color: '#f87171', size: 8, symbol: 'x', line: { width: 2 } },
      text: crashes.map(function(r){return r.description;}),
      customdata: crashes.map(function(r){return [r.commit, r.params_fmt];}),
      hovertemplate: '<b>%{text}</b><br>CRASH<br>%{customdata[0]}<extra>#%{x}</extra>',
    },
    {
      x: kept.map(function(r){return r.idx;}),
      y: kept.map(function(r){return r.val_metric;}),
      mode: 'markers', type: 'scatter', name: 'Kept', showlegend: false,
      marker: { color: '#34d399', size: 9 },
      text: kept.map(function(r){return r.description;}),
      customdata: kept.map(function(r){return [r.commit, r.params_fmt];}),
      hovertemplate: '<b>%{text}</b><br>%{y:.4f}<br>%{customdata[0]} \\u00b7 %{customdata[1]}<extra>#%{x} \\u2714</extra>',
    },
    {
      x: bestX, y: bestY,
      mode: 'lines', type: 'scatter', name: 'Running best', showlegend: false,
      line: { color: 'rgba(52, 211, 153, 0.35)', width: 2, shape: 'hv' },
      hoverinfo: 'skip',
    },
  ];

  var layout = {
    xaxis: {
      title: { text: 'Experiment #', font: { size: 12, color: '#5c6478' } },
      gridcolor: '#434956', zeroline: false,
      tickfont: { family: 'JetBrains Mono', size: 11, color: '#6e7787' },
    },
    yaxis: {
      title: { text: data.metric_label, font: { size: 12, color: '#6e7787' } },
      gridcolor: '#434956', zeroline: false,
      autorange: data.lower_is_better ? 'reversed' : true,
      tickfont: { family: 'JetBrains Mono', size: 11, color: '#6e7787' },
    },
    plot_bgcolor: '#393f4a',
    paper_bgcolor: '#353a45',
    margin: { l: 60, r: 20, t: 10, b: 50 },
    showlegend: false,
    font: { family: 'Inter, -apple-system, sans-serif', size: 12 },
    hovermode: 'closest',
    hoverlabel: {
      bgcolor: '#353a45',
      bordercolor: '#434956',
      font: { family: 'JetBrains Mono, monospace', size: 11, color: '#eaecf0' },
    },
  };

  Plotly.newPlot('chart', traces, layout, { responsive: true, displayModeBar: false });
}

function renderLegend() {
  var container = document.getElementById('chart-legend');
  while (container.firstChild) container.removeChild(container.firstChild);
  var items = [
    { type: 'dot', color: '#6e7787', label: 'Discarded' },
    { type: 'dot', color: '#34d399', label: 'Kept' },
    { type: 'dot', color: '#f87171', label: 'Crash' },
    { type: 'line', color: 'rgba(52, 211, 153, 0.35)', label: 'Running best' },
  ];
  items.forEach(function(item) {
    var div = document.createElement('div');
    div.className = 'legend-item';
    var indicator = document.createElement('span');
    indicator.className = item.type === 'dot' ? 'legend-dot' : 'legend-line';
    indicator.style.backgroundColor = item.color;
    div.appendChild(indicator);
    var text = document.createElement('span');
    text.textContent = item.label;
    div.appendChild(text);
    container.appendChild(div);
  });
}

function renderFilters() {
  var container = document.getElementById('filters');
  while (container.firstChild) container.removeChild(container.firstChild);
  ['all', 'keep', 'discard', 'crash'].forEach(function(f) {
    var btn = document.createElement('button');
    btn.className = 'filter-btn' + (currentFilter === f ? ' active' : '');
    btn.textContent = f === 'all' ? 'All' : f === 'keep' ? 'Kept' : f === 'discard' ? 'Discarded' : 'Crashes';
    btn.addEventListener('click', function() { filterTable(f); });
    container.appendChild(btn);
  });
}

function renderTable(data) {
  var rows = data.rows.slice();

  if (currentFilter !== 'all') {
    rows = rows.filter(function(r) { return r.status === currentFilter; });
  }

  rows.sort(function(a, b) {
    var va = a[sortCol], vb = b[sortCol];
    if (va === null) return 1;
    if (vb === null) return -1;
    if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  var tbody = document.getElementById('tbody');
  while (tbody.firstChild) tbody.removeChild(tbody.firstChild);

  rows.forEach(function(r) {
    var tr = document.createElement('tr');

    var tdIdx = document.createElement('td');
    tdIdx.className = 'mono';
    tdIdx.textContent = r.idx;
    tr.appendChild(tdIdx);

    var tdCommit = document.createElement('td');
    tdCommit.className = 'mono';
    tdCommit.textContent = r.commit;
    tr.appendChild(tdCommit);

    var tdMetric = document.createElement('td');
    tdMetric.className = 'metric';
    tdMetric.textContent = r.val_metric !== null ? r.val_metric.toFixed(4) : '\\u2014';
    tr.appendChild(tdMetric);

    var tdParams = document.createElement('td');
    tdParams.className = 'mono';
    tdParams.textContent = r.params ? r.params_fmt : '\\u2014';
    tr.appendChild(tdParams);

    var tdStatus = document.createElement('td');
    if (r.status) {
      var badge = document.createElement('span');
      badge.className = 'badge ' + r.status;
      badge.textContent = r.status;
      tdStatus.appendChild(badge);
    }
    tr.appendChild(tdStatus);

    var tdDesc = document.createElement('td');
    tdDesc.textContent = r.description || '';
    tr.appendChild(tdDesc);

    tbody.appendChild(tr);
  });
}

function filterTable(status) {
  currentFilter = status;
  renderFilters();
  renderTable(DATA);
}

document.querySelectorAll('th[data-col]').forEach(function(th) {
  th.addEventListener('click', function() {
    var col = th.getAttribute('data-col');
    if (sortCol === col) { sortAsc = !sortAsc; }
    else { sortCol = col; sortAsc = true; }
    renderTable(DATA);
  });
});

render(DATA);

setInterval(function() {
  fetch('results.json?t=' + Date.now())
    .then(function(resp) { if (resp.ok) return resp.json(); throw new Error(); })
    .then(function(data) { DATA = data; isLive = true; render(DATA); })
    .catch(function() {});
}, 30000);

fetch('results.json?t=' + Date.now())
  .then(function(resp) { if (resp.ok) return resp.json(); throw new Error(); })
  .then(function(data) { DATA = data; isLive = true; render(DATA); })
  .catch(function() {});
</script>
</body>
</html>"""


def generate_html(rows, out_dir="reports"):
    """Generate interactive HTML dashboard and JSON data file."""
    os.makedirs(out_dir, exist_ok=True)
    stats = compute_stats(rows)
    task = stats["task"]
    metric_key, metric_label, lower_is_better = METRIC_LABELS.get(task, ("val_metric", "val_metric", True))

    data = []
    for i, row in enumerate(rows):
        data.append(
            {
                "idx": i,
                "commit": row.get("commit", ""),
                "task": row.get("task", ""),
                "val_metric": row["val_metric"],
                "params": row["params"],
                "params_fmt": format_params(row["params"]),
                "status": row.get("status", "").strip(),
                "description": row.get("description", ""),
            }
        )

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "stats": stats,
        "metric_key": metric_key,
        "metric_label": metric_label,
        "lower_is_better": lower_is_better,
        "rows": data,
    }

    # write JSON for live refresh
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f)
    print(f"Saved {json_path}")

    # write HTML with baked-in data
    html_path = os.path.join(out_dir, "index.html")
    data_json = json.dumps(payload)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Saved {html_path}")


# ---------------------------------------------------------------------------
# Run summary (markdown)
# ---------------------------------------------------------------------------


def generate_summary(rows, tag, out_dir="reports/runs"):
    """Generate a markdown run summary for the given tag."""
    os.makedirs(out_dir, exist_ok=True)
    stats = compute_stats(rows)
    task = stats["task"]
    metric_key, _, _ = METRIC_LABELS.get(task, ("val_metric", "val_metric", True))

    kept = [r for r in rows if r.get("status", "").strip() == "keep" and r["val_metric"] is not None]
    crashes = [r for r in rows if r.get("status", "").strip() == "crash"]

    # baseline and best
    baseline = kept[0]["val_metric"] if kept else None
    best = stats["best_metric"]
    higher_better = stats["higher_is_better"]

    if baseline is not None and best is not None and baseline != best:
        delta = ((best - baseline) / baseline) * 100
        if not higher_better:
            delta = -delta
        delta_str = f"{delta:+.1f}%"
    else:
        delta_str = "N/A"

    # top improvements (by delta from previous kept)
    improvements = []
    prev = baseline
    for r in kept[1:]:
        if prev is not None:
            d = prev - r["val_metric"] if not higher_better else r["val_metric"] - prev
            improvements.append((r["description"], r["val_metric"], d))
            prev = r["val_metric"]

    improvements.sort(key=lambda x: x[2], reverse=True)

    lines = [
        f"# Run: {tag}",
        "",
        f"**Task**: {task}  ",
        f"**Experiments**: {stats['total']} ({stats['kept']} kept, {stats['discarded']} discarded, {stats['crashes']} crashes)  ",
        f"**Baseline {metric_key}**: {baseline:.4f}  " if baseline else "",
        f"**Best {metric_key}**: {best:.4f} ({delta_str} from baseline)  " if best else "",
        f"**Keep rate**: {stats['keep_rate']:.1f}%  ",
        "",
        "## Top improvements",
        "",
    ]

    for desc, val, delta in improvements[:10]:
        lines.append(f"- **{desc}** → {val:.4f}")

    if crashes:
        lines.append("")
        lines.append(f"## Crashes ({len(crashes)})")
        lines.append("")
        for r in crashes:
            lines.append(f"- {r.get('description', 'unknown')}")

    lines.append("")
    lines.append(f"*Generated {time.strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    out_path = os.path.join(out_dir, f"{tag}.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plot autoresearch experiment progress")
    parser.add_argument("-i", "--input", default="results.tsv", help="Path to results.tsv")
    parser.add_argument("-o", "--output", default="logs/progress.png", help="Output path for matplotlib PNG")
    parser.add_argument("--html-only", action="store_true", help="Skip matplotlib, just generate HTML report")
    parser.add_argument("--summary", metavar="TAG", help="Generate run summary markdown for the given tag")
    parser.add_argument("--report-dir", default="reports", help="Output directory for HTML report")
    args = parser.parse_args()

    rows = read_results(args.input)
    if not rows:
        print(f"No results found in {args.input}")
        return

    if not args.html_only:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        plot_matplotlib(rows, args.output)

    generate_html(rows, args.report_dir)

    if args.summary:
        generate_summary(rows, args.summary)


if __name__ == "__main__":
    main()
