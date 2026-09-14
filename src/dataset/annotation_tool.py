from pathlib import Path
import csv
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = Path("data/raw/host_samples/Sample Images").resolve()
OUTPUT = Path("data/manifests/host_annotations.csv").resolve()

images = sorted(ROOT.rglob("*.jpg"))

existing = {}
if OUTPUT.exists():
    with OUTPUT.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing[row["image_path"]] = row

data = []

for p in images:
    rel = p.relative_to(ROOT)
    label = rel.parts[0]
    case_id = rel.parts[1]

    old = existing.get(str(p), {})

    data.append({
        "image_path": str(p),
        "relative_path": str(rel),
        "case_id": case_id,
        "label": label,
        "view_type": old.get("view_type", ""),
        "tie_down_visible": old.get("tie_down_visible", ""),
        "suspension_visible": old.get("suspension_visible", ""),
        "engine_visible": old.get("engine_visible", ""),
        "whole_truck_visible": old.get("whole_truck_visible", ""),
        "close_up": old.get("close_up", ""),
        "notes": old.get("notes", "")
    })

DATA_JSON = json.dumps(data).replace("</", "<\\/")

PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Aircraft Tie-Down Annotation</title>

<style>
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #202124;
    color: white;
}

header {
    background: #111;
    padding: 15px 20px;
}

h2 {
    margin: 0 0 6px 0;
}

#progress {
    color: #8ab4f8;
}

#layout {
    display: flex;
    height: calc(100vh - 80px);
}

#imagePanel {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #303134;
    overflow: auto;
}

#image {
    max-width: 96%;
    max-height: 96%;
    object-fit: contain;
}

#controls {
    width: 370px;
    padding: 20px;
    background: #181818;
    overflow-y: auto;
}

.case {
    color: #aaa;
    margin-bottom: 15px;
}

label {
    display: block;
    margin-top: 14px;
    font-weight: bold;
}

select,
textarea,
button {
    width: 100%;
    box-sizing: border-box;
    margin-top: 6px;
    padding: 9px;
    font-size: 14px;
}

button {
    cursor: pointer;
    border: 0;
    border-radius: 4px;
}

#save {
    background: #2e7d32;
    color: white;
}

.nav {
    display: flex;
    gap: 10px;
}

.nav button {
    width: 50%;
}

#prev {
    background: #555;
    color: white;
}

#next {
    background: #1565c0;
    color: white;
}

.hint {
    margin-top: 20px;
    padding: 10px;
    background: #252525;
    color: #bbb;
    font-size: 13px;
    line-height: 1.4;
}
</style>
</head>

<body>

<header>
<h2>Aircraft Engine Tie-Down - Visual Annotation</h2>
<div id="info"></div>
<div id="progress"></div>
</header>

<div id="layout">

<div id="imagePanel">
<img id="image">
</div>

<div id="controls">

<div class="case" id="case"></div>

<label>View Type</label>
<select id="view_type">
<option value="">-- Select --</option>
<option value="left">Left side</option>
<option value="right">Right side</option>
<option value="front">Front</option>
<option value="rear">Rear</option>
<option value="whole_truck">Whole truck</option>
<option value="suspension">Suspension</option>
<option value="close_up">Close-up</option>
<option value="unknown">Unknown</option>
</select>

<label>Tie-down visible?</label>
<select id="tie_down_visible">
<option value="">-- Select --</option>
<option value="yes">Yes</option>
<option value="no">No</option>
</select>

<label>Suspension visible?</label>
<select id="suspension_visible">
<option value="">-- Select --</option>
<option value="yes">Yes</option>
<option value="no">No</option>
</select>

<label>Engine visible?</label>
<select id="engine_visible">
<option value="">-- Select --</option>
<option value="yes">Yes</option>
<option value="no">No</option>
</select>

<label>Whole truck visible?</label>
<select id="whole_truck_visible">
<option value="">-- Select --</option>
<option value="yes">Yes</option>
<option value="no">No</option>
</select>

<label>Close-up?</label>
<select id="close_up">
<option value="">-- Select --</option>
<option value="yes">Yes</option>
<option value="no">No</option>
</select>

<label>Notes</label>
<textarea id="notes" rows="5"
placeholder="Describe only what is visually visible."></textarea>

<button id="save">Save Annotation</button>

<div class="nav">
<button id="prev">Previous</button>
<button id="next">Next</button>
</div>

<div class="hint">
Keyboard shortcuts:<br>
Left Arrow = Previous<br>
Right Arrow = Next
</div>

</div>
</div>

<script>

const DATA = __DATA__;

let index = 0;

const fields = [
    "view_type",
    "tie_down_visible",
    "suspension_visible",
    "engine_visible",
    "whole_truck_visible",
    "close_up",
    "notes"
];

function saveCurrent() {

    for (const field of fields) {
        DATA[index][field] =
            document.getElementById(field).value;
    }

    fetch("/save", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(DATA)
    });
}

function loadCurrent() {

    const item = DATA[index];

    document.getElementById("image").src =
        "/image?path=" +
        encodeURIComponent(item.image_path);

    document.getElementById("info").textContent =
        item.relative_path;

    document.getElementById("case").textContent =
        item.case_id + " - " + item.label;

    document.getElementById("progress").textContent =
        "Image " + (index + 1) +
        " / " + DATA.length;

    for (const field of fields) {
        document.getElementById(field).value =
            item[field] || "";
    }
}

document.getElementById("save").onclick = function() {
    saveCurrent();
    alert("Annotation saved.");
};

document.getElementById("next").onclick = function() {

    saveCurrent();

    if (index < DATA.length - 1) {
        index++;
    }

    loadCurrent();
};

document.getElementById("prev").onclick = function() {

    saveCurrent();

    if (index > 0) {
        index--;
    }

    loadCurrent();
};

document.addEventListener("keydown", function(event) {

    if (event.target.tagName === "TEXTAREA") {
        return;
    }

    if (event.key === "ArrowRight") {
        document.getElementById("next").click();
    }

    if (event.key === "ArrowLeft") {
        document.getElementById("prev").click();
    }
});

loadCurrent();

</script>

</body>
</html>
"""

PAGE = PAGE.replace("__DATA__", DATA_JSON)


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):

        parsed = urlparse(self.path)

        if parsed.path == "/":

            content = PAGE.encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )
            self.send_header(
                "Content-Length",
                str(len(content))
            )
            self.end_headers()

            self.wfile.write(content)
            return

        if parsed.path == "/image":

            params = parse_qs(parsed.query)

            if "path" not in params:
                self.send_response(400)
                self.end_headers()
                return

            requested = Path(params["path"][0]).resolve()

            try:
                requested.relative_to(ROOT)
            except ValueError:
                self.send_response(403)
                self.end_headers()
                return

            if not requested.exists():
                self.send_response(404)
                self.end_headers()
                return

            content = requested.read_bytes()

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "image/jpeg"
            )
            self.send_header(
                "Content-Length",
                str(len(content))
            )
            self.end_headers()

            self.wfile.write(content)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):

        if self.path != "/save":
            self.send_response(404)
            self.end_headers()
            return

        length = int(
            self.headers.get("Content-Length", 0)
        )

        body = self.rfile.read(length)

        rows = json.loads(body)

        OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        fields = [
            "image_path",
            "relative_path",
            "case_id",
            "label",
            "view_type",
            "tie_down_visible",
            "suspension_visible",
            "engine_visible",
            "whole_truck_visible",
            "close_up",
            "notes"
        ]

        with OUTPUT.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields
            )

            writer.writeheader()
            writer.writerows(rows)

        self.send_response(200)
        self.end_headers()

        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


server = HTTPServer(
    ("127.0.0.1", 8765),
    Handler
)

print("Annotation tool running.")
print("Open: http://127.0.0.1:8765")
print("Press Ctrl+C to stop.")

webbrowser.open(
    "http://127.0.0.1:8765"
)

server.serve_forever()
