"""Visualize SSM hidden state during inference.

Runs a trained model in recurrent mode and records the internal state at every
position. Generates an interactive HTML heatmap showing how the model's memory
evolves as it reads text.

Usage:
  python visualize.py checkpoints/lm "To be, or not to be"
  python visualize.py checkpoints/lm --file hamlet.txt --max-chars 500
  python visualize.py checkpoints/lm "Hello world" -o state.html
"""

import argparse
import json
import sys

import mlx.core as mx
import numpy as np

from engine import RecurrentState, load_model


def record_states(model, text):
    """Run inference and record state norms at every position.

    Returns dict with:
        chars: list of display strings per position
        tokens: list of byte values
        norms: (n_positions, n_layers, d_model) float32 array
        deltas: (n_positions, n_layers, d_model) float32 array
        top_preds: list of [(byte, prob), ...] top-5 predictions per position
        n_layers: int
        d_model: int
        state_dim: int
    """
    state = RecurrentState(model)
    n_layers = len(model.blocks)
    d_model = model.blocks[0].ssm.log_A.shape[0]
    state_dim = model.blocks[0].ssm.log_A.shape[1]

    text_bytes = text.encode("utf-8")
    n_pos = len(text_bytes)

    norms = np.zeros((n_pos, n_layers, d_model), dtype=np.float32)
    deltas = np.zeros((n_pos, n_layers, d_model), dtype=np.float32)
    prev_norms = np.zeros((n_layers, d_model), dtype=np.float32)
    top_preds = []
    chars = []
    tokens = []

    for pos, byte_val in enumerate(text_bytes):
        tokens.append(int(byte_val))
        if byte_val == 10:
            chars.append("\\n")
        elif byte_val == 9:
            chars.append("\\t")
        elif 32 <= byte_val < 127:
            chars.append(chr(byte_val))
        else:
            chars.append(f"\\x{byte_val:02x}")

        logits = state.step(byte_val)

        for li in range(n_layers):
            s = state.states[li]
            norm = np.array(mx.sqrt(mx.sum(s * s, axis=1)))
            norms[pos, li, :] = norm
            deltas[pos, li, :] = norm - prev_norms[li]
            prev_norms[li] = norm.copy()

        probs = np.array(mx.softmax(logits))
        top_idx = np.argsort(probs)[-5:][::-1]
        top_preds.append([(int(i), round(float(probs[i]), 4)) for i in top_idx])

    return {
        "chars": chars,
        "tokens": tokens,
        "norms": norms,
        "deltas": deltas,
        "top_preds": top_preds,
        "n_layers": n_layers,
        "d_model": d_model,
        "state_dim": state_dim,
    }


def generate_html(data, config, output_path):
    """Generate self-contained interactive HTML visualization."""
    # Prepare JSON-serializable data
    payload = {
        "chars": data["chars"],
        "tokens": data["tokens"],
        "norms": data["norms"].tolist(),
        "deltas": data["deltas"].tolist(),
        "top_preds": data["top_preds"],
        "n_layers": data["n_layers"],
        "d_model": data["d_model"],
        "state_dim": data["state_dim"],
        "config": config,
    }

    html = (
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nanostate · state viewer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
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
    --orange: #f97316;
    --blue: #60a5fa;
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

  .header { margin-bottom: 24px; }
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
    font-family: var(--font-mono);
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }
  .btn {
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
  .btn.active {
    background: var(--green-muted);
    border-color: var(--green);
    color: var(--green);
    font-weight: 500;
  }
  .btn:hover { border-color: var(--border-active); }
  .btn-play {
    width: 32px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
  }
  .control-group {
    display: flex;
    gap: 4px;
  }
  .control-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    align-self: center;
    margin-right: 4px;
  }

  .slider-container {
    flex: 1;
    min-width: 200px;
  }
  .slider {
    width: 100%;
    -webkit-appearance: none;
    appearance: none;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    outline: none;
  }
  .slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--green);
    cursor: pointer;
  }

  .text-display {
    font-family: var(--font-mono);
    font-size: 13px;
    line-height: 1.8;
    padding: 12px 16px;
    background: var(--bg);
    border-radius: 8px;
    margin-bottom: 16px;
    overflow-x: auto;
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
  }
  .text-display .char {
    display: inline-block;
    padding: 1px 0;
    border-bottom: 2px solid transparent;
    transition: border-color 0.1s;
  }
  .text-display .char.active {
    border-bottom-color: var(--green);
    color: var(--green);
  }
  .text-display .char.past {
    color: var(--text-secondary);
  }
  .text-display .char.future {
    color: var(--text-muted);
  }

  .heatmap-container {
    position: relative;
    overflow-x: auto;
  }
  #heatmap {
    image-rendering: pixelated;
    display: block;
    border-radius: 4px;
  }
  .heatmap-cursor {
    position: absolute;
    top: 0;
    width: 2px;
    background: var(--green);
    pointer-events: none;
    opacity: 0.8;
  }

  .legend {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    font-size: 10px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }
  .legend-gradient {
    width: 120px;
    height: 8px;
    border-radius: 2px;
  }

  .detail-panel {
    display: flex;
    gap: 24px;
    margin-top: 16px;
    font-size: 12px;
  }
  .detail-section {
    flex-shrink: 0;
  }
  .detail-section.channels {
    width: 420px;
  }
  .detail-section.predictions {
    width: 220px;
    margin-left: auto;
  }
  .detail-section h3 {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }
  .channel-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .channel-tag {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 6px;
    font-family: var(--font-mono);
    font-size: 10px;
  }
  .pred-row {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 22px;
  }
  .pred-char {
    font-family: var(--font-mono);
    background: var(--bg);
    padding: 1px 6px;
    border-radius: 3px;
    width: 28px;
    height: 18px;
    line-height: 18px;
    text-align: center;
    font-size: 11px;
  }
  .pred-bar {
    height: 10px;
    background: var(--green);
    border-radius: 2px;
    opacity: 0.6;
  }
  .pred-pct {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-muted);
    width: 36px;
  }
</style>
</head>
<body>

<div class="header">
  <div class="label">nanostate</div>
  <h1>State Viewer</h1>
  <div class="meta" id="meta"></div>
</div>

<div class="card">
  <div class="controls">
    <button class="btn btn-play" id="playBtn" title="Play/Pause"></button>
    <div class="slider-container">
      <input type="range" class="slider" id="posSlider" min="0" max="0" value="0">
    </div>
    <span id="posLabel" style="font-family:var(--font-mono);font-size:12px;color:var(--text-secondary);width:160px;text-align:right"></span>
  </div>

  <div class="controls">
    <span class="control-label">Layer</span>
    <div class="control-group" id="layerBtns"></div>
    <span class="control-label" style="margin-left:12px">View</span>
    <div class="control-group" id="viewBtns"></div>
    <span class="control-label" style="margin-left:12px">Speed</span>
    <div class="control-group" id="speedBtns"></div>
  </div>

  <div class="text-display" id="textDisplay"></div>

  <div class="heatmap-container" id="heatmapContainer">
    <canvas id="heatmap"></canvas>
    <div class="heatmap-cursor" id="cursor"></div>
  </div>
  <div class="legend" id="legend"></div>

  <div class="detail-panel" id="details"></div>
</div>

<script>
var DATA = """
        + json.dumps(payload)
        + """;

var CELL_H = Math.max(1, Math.min(4, Math.floor(400 / DATA.d_model)));
var PLAY_SYMBOL = '\\u25B6';
var PAUSE_SYMBOL = '\\u23F8';

var currentPos = 0;
var currentLayer = 0;
var currentView = 'norm';
var playing = false;
var playSpeed = 50;
var playTimer = null;

function init() {
  var meta = document.getElementById('meta');
  var c = DATA.config;
  meta.textContent = 'd=' + c.d_model + ' L=' + c.n_layers + ' N=' + c.state_dim +
    ' \\u00b7 ' + DATA.chars.length + ' tokens \\u00b7 state: ' +
    (c.n_layers * c.d_model * c.state_dim * 4).toLocaleString() + ' bytes';

  document.getElementById('playBtn').textContent = PLAY_SYMBOL;
  buildTextDisplay();
  buildLayerButtons();
  buildViewButtons();
  buildSpeedButtons();

  var slider = document.getElementById('posSlider');
  slider.max = DATA.chars.length - 1;
  slider.addEventListener('input', function() {
    currentPos = parseInt(this.value);
    update();
  });

  document.getElementById('playBtn').addEventListener('click', togglePlay);

  renderHeatmap();
  update();
}

function buildTextDisplay() {
  var el = document.getElementById('textDisplay');
  for (var i = 0; i < DATA.chars.length; i++) {
    var span = document.createElement('span');
    span.className = 'char';
    span.dataset.pos = i;
    span.textContent = DATA.chars[i];
    span.addEventListener('click', (function(pos) {
      return function() {
        currentPos = pos;
        document.getElementById('posSlider').value = currentPos;
        update();
      };
    })(i));
    el.appendChild(span);
  }
}

function buildLayerButtons() {
  var container = document.getElementById('layerBtns');
  for (var i = 0; i < DATA.n_layers; i++) {
    var btn = document.createElement('button');
    btn.className = 'btn' + (i === 0 ? ' active' : '');
    btn.textContent = (i + 1).toString();
    btn.addEventListener('click', (function(layer) {
      return function() {
        currentLayer = layer;
        container.querySelectorAll('.btn').forEach(function(b) { b.classList.remove('active'); });
        this.classList.add('active');
        renderHeatmap();
        update();
      };
    })(i));
    container.appendChild(btn);
  }
}

function buildViewButtons() {
  var container = document.getElementById('viewBtns');
  ['norm', 'delta'].forEach(function(view) {
    var btn = document.createElement('button');
    btn.className = 'btn' + (view === 'norm' ? ' active' : '');
    btn.textContent = view;
    btn.addEventListener('click', (function(v) {
      return function() {
        currentView = v;
        container.querySelectorAll('.btn').forEach(function(b) { b.classList.remove('active'); });
        this.classList.add('active');
        renderHeatmap();
        update();
      };
    })(view));
    container.appendChild(btn);
  });
}

function buildSpeedButtons() {
  var container = document.getElementById('speedBtns');
  [{label: '1x', ms: 100}, {label: '2x', ms: 50}, {label: '5x', ms: 20}].forEach(function(s, i) {
    var btn = document.createElement('button');
    btn.className = 'btn' + (i === 1 ? ' active' : '');
    btn.textContent = s.label;
    btn.addEventListener('click', (function(ms) {
      return function() {
        playSpeed = ms;
        container.querySelectorAll('.btn').forEach(function(b) { b.classList.remove('active'); });
        this.classList.add('active');
        if (playing) { clearInterval(playTimer); playTimer = setInterval(stepForward, playSpeed); }
      };
    })(s.ms));
    container.appendChild(btn);
  });
}

function normColor(t) {
  t = Math.min(Math.max(t, 0), 1);
  return [Math.round(44 + t * 8), Math.round(58 + t * 153), Math.round(69 + t * 84)];
}

function deltaColor(t) {
  t = Math.min(Math.max(t, -1), 1);
  if (t >= 0) {
    return [Math.round(53 + t * 196), Math.round(58 + t * 57), Math.round(69 - t * 47)];
  }
  var s = -t;
  return [Math.round(53 - s * 17), Math.round(58 + s * 107), Math.round(69 + s * 181)];
}

function renderHeatmap() {
  var canvas = document.getElementById('heatmap');
  var nPos = DATA.chars.length;
  var nChan = DATA.d_model;
  // Render at 1px per position, CELL_H px per channel. CSS stretches to fill container.
  var w = nPos;
  var h = nChan * CELL_H;
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = '100%';
  canvas.style.height = Math.max(h, 300) + 'px';

  var ctx = canvas.getContext('2d');
  var imageData = ctx.createImageData(w, h);
  var pixels = imageData.data;

  var source = currentView === 'norm' ? DATA.norms : DATA.deltas;
  var maxVal = 0;
  for (var p = 0; p < nPos; p++) {
    for (var c = 0; c < nChan; c++) {
      var v = Math.abs(source[p][currentLayer][c]);
      if (v > maxVal) maxVal = v;
    }
  }
  if (maxVal === 0) maxVal = 1;

  for (var p = 0; p < nPos; p++) {
    for (var c = 0; c < nChan; c++) {
      var val = source[p][currentLayer][c];
      var rgb = currentView === 'norm' ? normColor(val / maxVal) : deltaColor(val / maxVal);
      for (var dy = 0; dy < CELL_H; dy++) {
        var idx = ((c * CELL_H + dy) * w + p) * 4;
        pixels[idx] = rgb[0];
        pixels[idx + 1] = rgb[1];
        pixels[idx + 2] = rgb[2];
        pixels[idx + 3] = 255;
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);

  document.getElementById('cursor').style.height = canvas.style.height;

  // Legend
  var legendEl = document.getElementById('legend');
  while (legendEl.firstChild) legendEl.removeChild(legendEl.firstChild);

  var labelLow = document.createElement('span');
  labelLow.textContent = currentView === 'norm' ? '0' : '-max';
  legendEl.appendChild(labelLow);

  var grad = document.createElement('span');
  grad.className = 'legend-gradient';
  if (currentView === 'norm') {
    grad.style.background = 'linear-gradient(to right, rgb(44,58,69), rgb(52,211,153))';
  } else {
    grad.style.background = 'linear-gradient(to right, rgb(36,165,250), rgb(53,58,69), rgb(249,115,22))';
  }
  legendEl.appendChild(grad);

  var labelHigh = document.createElement('span');
  labelHigh.textContent = currentView === 'norm' ? maxVal.toFixed(2) : '\\u00b1' + maxVal.toFixed(2);
  legendEl.appendChild(labelHigh);

  var dimLabel = document.createElement('span');
  dimLabel.style.marginLeft = '12px';
  dimLabel.textContent = 'x: position (' + nPos + ')  y: channel (' + nChan + ')';
  legendEl.appendChild(dimLabel);
}

function update() {
  var label = document.getElementById('posLabel');
  var ch = DATA.chars[currentPos];
  var tok = DATA.tokens[currentPos];
  label.textContent = currentPos + '/' + (DATA.chars.length - 1) + "  '" + ch + "' (" + tok + ')';

  var spans = document.getElementById('textDisplay').querySelectorAll('.char');
  for (var i = 0; i < spans.length; i++) {
    spans[i].className = 'char ' + (i < currentPos ? 'past' : i === currentPos ? 'active' : 'future');
  }
  if (spans[currentPos]) {
    spans[currentPos].scrollIntoView({block: 'nearest', inline: 'center'});
  }

  var heatmap = document.getElementById('heatmap');
  var displayW = heatmap.getBoundingClientRect().width;
  var pxPerPos = displayW / DATA.chars.length;
  document.getElementById('cursor').style.left = (currentPos * pxPerPos) + 'px';
  document.getElementById('posSlider').value = currentPos;
  updateDetails();
}

function updateDetails() {
  var el = document.getElementById('details');
  while (el.firstChild) el.removeChild(el.firstChild);

  // Top channels
  var norms = DATA.norms[currentPos][currentLayer];
  var indexed = [];
  for (var i = 0; i < norms.length; i++) indexed.push({idx: i, val: norms[i]});
  indexed.sort(function(a, b) { return b.val - a.val; });

  var chanSection = document.createElement('div');
  chanSection.className = 'detail-section channels';
  var chanTitle = document.createElement('h3');
  chanTitle.textContent = 'Top channels (layer ' + (currentLayer + 1) + ')';
  chanSection.appendChild(chanTitle);
  var chanList = document.createElement('div');
  chanList.className = 'channel-list';
  for (var i = 0; i < 8 && i < indexed.length; i++) {
    var tag = document.createElement('span');
    tag.className = 'channel-tag';
    tag.textContent = '[' + indexed[i].idx + '] ' + indexed[i].val.toFixed(3);
    chanList.appendChild(tag);
  }
  chanSection.appendChild(chanList);
  el.appendChild(chanSection);

  // Predictions
  var predSection = document.createElement('div');
  predSection.className = 'detail-section predictions';
  var predTitle = document.createElement('h3');
  predTitle.textContent = 'Next token predictions';
  predSection.appendChild(predTitle);

  var preds = DATA.top_preds[currentPos];
  var maxProb = preds.length > 0 ? preds[0][1] : 1;
  for (var i = 0; i < preds.length; i++) {
    var row = document.createElement('div');
    row.className = 'pred-row';

    var ch = document.createElement('span');
    ch.className = 'pred-char';
    var code = preds[i][0];
    if (code >= 32 && code < 127) {
      ch.textContent = String.fromCharCode(code);
    } else if (code === 10) {
      ch.textContent = '\\n';
    } else {
      ch.textContent = '0x' + code.toString(16).padStart(2, '0');
    }
    row.appendChild(ch);

    var bar = document.createElement('span');
    bar.className = 'pred-bar';
    bar.style.width = Math.round(preds[i][1] / maxProb * 80) + 'px';
    row.appendChild(bar);

    var pct = document.createElement('span');
    pct.className = 'pred-pct';
    pct.textContent = (preds[i][1] * 100).toFixed(1) + '%';
    row.appendChild(pct);

    predSection.appendChild(row);
  }
  el.appendChild(predSection);
}

function togglePlay() {
  playing = !playing;
  var btn = document.getElementById('playBtn');
  btn.textContent = playing ? PAUSE_SYMBOL : PLAY_SYMBOL;
  if (playing) {
    playTimer = setInterval(stepForward, playSpeed);
  } else {
    if (playTimer) { clearInterval(playTimer); playTimer = null; }
  }
}

function stepForward() {
  currentPos++;
  if (currentPos >= DATA.chars.length) currentPos = 0;
  update();
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowRight') { currentPos = Math.min(currentPos + 1, DATA.chars.length - 1); update(); }
  if (e.key === 'ArrowLeft') { currentPos = Math.max(currentPos - 1, 0); update(); }
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
});

document.getElementById('heatmap').addEventListener('mousemove', function(e) {
  var rect = this.getBoundingClientRect();
  var x = e.clientX - rect.left;
  var pxPerPos = rect.width / DATA.chars.length;
  var pos = Math.floor(x / pxPerPos);
  if (pos >= 0 && pos < DATA.chars.length) {
    currentPos = pos;
    document.getElementById('posSlider').value = pos;
    update();
  }
});

init();
</script>
</body>
</html>"""
    )

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SSM state during inference")
    parser.add_argument("checkpoint", help="Path to checkpoint directory")
    parser.add_argument("text", nargs="?", help="Text to process")
    parser.add_argument("--file", help="Read text from file instead")
    parser.add_argument("--max-chars", type=int, default=500, help="Max characters to process")
    parser.add_argument("-o", "--output", default="state.html", help="Output HTML file")
    parser.add_argument("--gif", metavar="PATH", help="Also generate an animated GIF")
    parser.add_argument("--gif-layer", type=int, default=0, help="Layer to render in GIF (default: 0)")
    parser.add_argument("--gif-fps", type=int, default=15, help="GIF frame rate (default: 15)")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read()[: args.max_chars]
    elif args.text:
        text = args.text[: args.max_chars]
    else:
        print("Provide text as argument or via --file", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.checkpoint}...", file=sys.stderr)
    model, config = load_model(args.checkpoint)
    if config["task"] != "lm":
        print(f"Visualization only supported for LM task, got '{config['task']}'", file=sys.stderr)
        sys.exit(1)

    print(f"Recording state for {len(text)} characters...", file=sys.stderr)
    data = record_states(model, text)

    generate_html(data, config, args.output)
    print(f"Open {args.output} in a browser", file=sys.stderr)

    if args.gif:
        generate_gif(data, config, args.gif, layer=args.gif_layer, fps=args.gif_fps)


def generate_gif(data, config, output_path, layer=0, fps=15):
    """Render animated GIF of state heatmap evolving over positions."""
    from PIL import Image, ImageDraw, ImageFont

    norms = data["norms"]  # (n_pos, n_layers, d_model)
    chars = data["chars"]
    n_pos, n_layers, d_model = norms.shape

    # Dimensions
    text_h = 32
    heatmap_h = min(d_model * 3, 384)
    cell_h = heatmap_h / d_model
    width = 800
    height = text_h + heatmap_h + 4
    px_per_pos = width / n_pos

    # Colormap: dark → green
    max_val = float(norms[:, layer, :].max()) or 1.0

    def norm_color(v):
        t = min(max(v / max_val, 0), 1)
        return (int(44 + t * 8), int(58 + t * 153), int(69 + t * 84))

    bg = (44, 48, 57)
    green = (52, 211, 153)
    muted = (110, 119, 135)
    text_color = (234, 236, 240)

    # Try to load a monospace font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFMono-Regular.otf", 13)
    except (OSError, AttributeError):
        font = ImageFont.load_default()

    frames = []
    # Sample frames: every position is too many for long texts, step through
    step = max(1, n_pos // 200)
    frame_positions = list(range(0, n_pos, step))
    if frame_positions[-1] != n_pos - 1:
        frame_positions.append(n_pos - 1)

    for pos in frame_positions:
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        # Draw text row
        x_offset = 8
        for i, ch in enumerate(chars):
            if i == pos:
                color = green
            elif i < pos:
                color = text_color
            else:
                color = muted
            draw.text((x_offset, 8), ch, fill=color, font=font)
            x_offset += max(8, int(px_per_pos))
            if x_offset > width - 10:
                break

        # Draw heatmap
        y_start = text_h + 2
        for p in range(n_pos):
            x0 = int(p * px_per_pos)
            x1 = int((p + 1) * px_per_pos)
            if x1 <= x0:
                x1 = x0 + 1
            for c in range(d_model):
                y0 = y_start + int(c * cell_h)
                y1 = y_start + int((c + 1) * cell_h)
                if y1 <= y0:
                    y1 = y0 + 1
                color = norm_color(norms[p, layer, c]) if p <= pos else bg
                draw.rectangle([x0, y0, x1, y1], fill=color)

        # Cursor line
        cx = int(pos * px_per_pos)
        draw.rectangle([cx, y_start, cx + 1, height - 2], fill=green)

        frames.append(img)

    # Hold last frame longer
    durations = [int(1000 / fps)] * len(frames)
    durations[-1] = 1500

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Saved {output_path} ({len(frames)} frames)", file=sys.stderr)


if __name__ == "__main__":
    main()
