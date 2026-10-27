"""
Shared runtime for hosted deployments (Colab, Kaggle, Docker, ephemeral VMs).

The notebooks and the Docker entrypoint are deliberately thin: everything they
need to do that is not "show the user a form" lives here, so the Colab and
Kaggle variants cannot drift apart and the behaviour is testable without a
notebook or a GPU.

Three rules shape this module:

* **No heavy imports at module import time.** `import torch` is not here; it is
  imported lazily inside `hardware_report`. A caller can import this module in a
  bare interpreter and still get the pure helpers (platform detection, runtime
  root choice, URL building).

* **`configure_runtime` must run before `app` is imported.** `engine/app.py`
  reads `UPLOAD_DIR` (and the sweep settings) from the environment at import
  time and creates the folder. Redirecting the working tree afterwards would be
  too late, so the caller runs this first, before anything imports the app.

* **One process, one worker.** `start_server` runs uvicorn in a thread of the
  *current* process: the pipeline holds multi-gigabyte models and an in-process
  job registry, so a second worker would double the memory and split the
  registry. A notebook cannot afford a subprocess tree either, since the kernel
  must stay responsive.

Nothing here is Multiva-specific except the health path and the environment
variable names; the layout mirrors the shape a hosted deployment always needs.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

__all__ = [
    "ServerHandle",
    "configure_runtime",
    "default_runtime_root",
    "detect_platform",
    "ensure_ffmpeg",
    "hardware_report",
    "is_notebook",
    "notebook_public_url",
    "public_url",
    "runtime_dirs",
    "start_server",
    "wait_until_ready",
]

# Where each platform is happy to write. Colab and Kaggle both mount a
# persistent, writable volume; writing into the checkout is not durable there
# and on Kaggle the checkout can be read-only.
_PLATFORM_ROOTS = {
    "colab": "/content/multiva-runtime",
    "kaggle": "/kaggle/working/multiva-runtime",
    "docker": "/data",
}

# The directory names under the runtime root. Declared once so `runtime_dirs`
# and `configure_runtime` cannot disagree about what exists.
_SUBDIRS = ("uploads", "output")


def detect_platform() -> str:
    """
    Which hosted environment this is: `colab`, `kaggle`, `docker`, or `local`.

    Detection is by the marker each platform actually sets, not by guessing from
    the hostname: Colab sets `COLAB_*` environment variables and mounts
    `/content`, Kaggle ships a `/kaggle` directory, and Docker leaves
    `/.dockerenv`.
    """
    if os.environ.get("COLAB_GPU") is not None or os.path.isdir("/content"):
        return "colab"
    if os.path.isdir("/kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.path.exists("/.dockerenv"):
        return "docker"
    return "local"


def is_notebook() -> bool:
    """
    Whether we are running inside an IPython kernel.

    Checked via the shell class rather than `__file__` sniffing, which is wrong
    for a notebook launched from a path without a name.
    """
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
    except Exception:  # noqa: BLE001 - IPython is optional
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


def default_runtime_root(platform: str | None = None) -> Path:
    """
    The directory the runtime tree should live under on this platform.

    An explicit `RUNTIME_DIR` environment variable always wins, so a user can pin
    the location (for example onto a mounted Drive) without editing a notebook.
    """
    override = os.environ.get("RUNTIME_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    platform = platform or detect_platform()
    return Path(_PLATFORM_ROOTS.get(platform, "/tmp/multiva-runtime"))


def runtime_dirs(root: str | Path | None = None) -> dict[str, Path]:
    """
    The runtime layout as absolute paths, without creating anything.

    Pure and side-effect-free so a caller can show the user where things will go
    before committing to it.
    """
    base = Path(root) if root is not None else default_runtime_root()
    return {name: base / name for name in _SUBDIRS}


def configure_runtime(root: str | Path | None = None) -> dict[str, Path]:
    """
    Point the application's runtime directories at a writable tree and create it.

    Sets `MULTIVA_UPLOAD_DIR` and `MULTIVA_OUTPUT_DIR` in the environment, then
    creates each directory. **Call this before importing `app`**: `engine/app.py`
    snapshots the upload directory at import time and creates it, and
    `engines.output_dir()` reads the output variable.

    Returns the layout it created. Idempotent: calling twice with the same root
    is a no-op beyond re-setting the same variables.
    """
    dirs = runtime_dirs(root)
    base = next(iter(dirs.values())).parent
    base.mkdir(parents=True, exist_ok=True)

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    os.environ["MULTIVA_UPLOAD_DIR"] = str(dirs["uploads"])
    os.environ["MULTIVA_OUTPUT_DIR"] = str(dirs["output"])
    return dirs


def ensure_ffmpeg() -> bool:
    """
    Make sure FFmpeg is on PATH, installing it on Debian-based hosted images.

    Returns whether FFmpeg is available afterwards. Never raises: a failed
    install leaves the answer `False` and the caller decides what to do. On
    Colab and Kaggle `apt-get` works without a password prompt; elsewhere the
    install is skipped and the caller is expected to have provided FFmpeg.
    """
    if shutil.which("ffmpeg"):
        return True

    platform = detect_platform()
    if platform not in ("colab", "kaggle", "docker"):
        return False

    commands = [
        ["apt-get", "update", "-qq"],
        ["apt-get", "install", "-y", "-qq", "ffmpeg"],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=600)
        except (OSError, subprocess.SubprocessError):
            return False
        if result.returncode != 0 and cmd[1] != "update":
            # `apt-get update` can fail on a flaky mirror while the package is
            # already cached; only the install result decides.
            return False
    return shutil.which("ffmpeg") is not None


def hardware_report() -> dict:
    """
    What the machine offers, and which preset it can realistically run.

    Imports torch lazily so this can be called (and tested) before the pipeline
    is set up. Returns a plain dict with a `recommended_preset` key, which the
    notebook shows and uses to pick the flow-matching step count.

    The mapping is a judgement, stated so it can be argued with: lip-sync is the
    VRAM-hungry stage and IndicF5 is the slow one, so a large GPU gets `HIGH`
    (32 steps), a mid-range one `BALANCED` (16 steps), and a small one `FAST`
    (8 steps).
    """
    report: dict = {
        "platform": detect_platform(),
        "cuda_available": False,
        "gpu_name": None,
        "vram_gb": None,
        "device": "cpu",
        "recommended_preset": "FAST",
        "notes": [],
    }

    try:
        import torch  # local import: keep module import cheap
    except Exception as exc:  # noqa: BLE001 - torch may be absent entirely
        report["notes"].append(f"torch is not importable: {exc}")
        return report

    report["torch_version"] = getattr(torch, "__version__", None)

    if torch.cuda.is_available():
        report["cuda_available"] = True
        report["device"] = "cuda"
        try:
            props = torch.cuda.get_device_properties(0)
            report["gpu_name"] = props.name
            report["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
        except Exception as exc:  # noqa: BLE001
            report["notes"].append(f"Could not read GPU properties: {exc}")
            return report

        vram = report["vram_gb"] or 0
        if vram >= 16:
            report["recommended_preset"] = "HIGH"
        elif vram >= 8:
            report["recommended_preset"] = "BALANCED"
        else:
            report["recommended_preset"] = "FAST"
            report["notes"].append(
                f"{vram} GB VRAM is tight; FAST keeps the render inside it.")
        return report

    # No CUDA. Apple silicon is the other accelerator worth naming, because the
    # engine uses it and a user on a Mac should not be told they are on "CPU".
    try:
        if torch.backends.mps.is_available():
            report["device"] = "mps"
            report["notes"].append(
                "Apple silicon (MPS). Renders work, roughly 1.5x faster than "
                "CPU; the CUDA paths below do not apply.")
            return report
    except Exception:  # noqa: BLE001 - older torch builds have no mps module
        pass

    report["notes"].append(
        "No CUDA GPU. Renders will run on CPU: correct, but slow. Select a GPU "
        "runtime (Runtime/Accelerator menu) to speed this up.")
    return report


def public_url(port: int, host: str = "127.0.0.1") -> str:
    """
    The URL to show the user.

    A wildcard bind is not navigable, so it is shown as loopback. When
    `PUBLIC_HOST` is set (a tunnel, a proxy, a reserved hostname) that wins,
    because on Colab and Kaggle the loopback address is reachable only from
    inside the VM and the user is on a different machine.
    """
    override = os.environ.get("PUBLIC_HOST", "").strip()
    if override:
        scheme = os.environ.get("PUBLIC_SCHEME", "https").strip() or "https"
        return f"{scheme}://{override}"
    display = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"http://{display}:{port}"


def notebook_public_url(port: int, host: str = "0.0.0.0") -> str:
    """
    The URL a notebook user should click, using the platform's own proxy.

    Colab can proxy a port with no account or token
    (`google.colab.kernel.proxyPort`), so that is preferred when available: the
    result is reachable from the user's browser, which `127.0.0.1:port` is not.
    Kaggle has no equivalent built-in proxy, so a tunnel must be supplied
    through `PUBLIC_HOST`; without one the loopback URL is returned and the
    notebook tells the user it is only reachable inside the VM.

    Never raises: any failure falls back to `public_url`.
    """
    if detect_platform() == "colab":
        try:
            from google.colab import output  # type: ignore

            proxy = output.eval_js(f"google.colab.kernel.proxyPort({port})")
            if isinstance(proxy, str) and proxy.startswith("http"):
                return proxy.rstrip("/")
        except Exception:  # noqa: BLE001 - not Colab, or the API moved
            pass
    return public_url(port, host)


def wait_until_ready(url: str, timeout: float = 120.0,
                     interval: float = 0.5) -> bool:
    """
    Poll the health endpoint until it answers, or `timeout` elapses.

    Polls over HTTP rather than a bare TCP connect, so a port that is open but
    serving 500s does not read as ready.
    """
    health = url.rstrip("/") + "/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(interval)
    return False


@dataclasses.dataclass
class ServerHandle:
    """
    A running server in this process, and a way to stop it.

    `stop()` sets uvicorn's `should_exit` flag and joins the thread with a
    timeout, so re-running a notebook cell does not leave a server holding the
    port. Safe to call more than once.
    """

    url: str
    host: str
    port: int
    server: object
    thread: threading.Thread

    def is_running(self) -> bool:
        return self.thread.is_alive()

    def stop(self, timeout: float = 15.0) -> None:
        try:
            self.server.should_exit = True  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - a non-uvicorn server has no flag
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)

    def __enter__(self) -> "ServerHandle":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def start_server(host: str = "0.0.0.0", port: int = 8000,
                 *, ready_timeout: float = 300.0,
                 log_level: str = "info",
                 warmup: bool = False) -> ServerHandle:
    """
    Start the API (and the built UI, if present) in a background thread.

    Imports `app` and serves it with uvicorn in a daemon thread of this process.
    Returns a `ServerHandle` whose `url` is the address to show the user and
    whose `stop()` shuts it down. `PUBLIC_HOST` overrides the displayed URL for
    tunnelled platforms.

    The bind defaults to `0.0.0.0` because a notebook is reached through a proxy
    or tunnel that connects from outside the loopback interface.

    `warmup` preloads the models before the first render. It is off by default
    because the engine starts a warmup thread of its own at boot; pass True only
    when the caller wants the server to answer only once the weights are ready.
    """
    # Make `engine/` importable regardless of the caller's cwd, so `import app`
    # resolves to engine/app.py and not to some other module named "app".
    engine_dir = str(Path(__file__).resolve().parent)
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)

    if warmup:
        os.environ["WARMUP_ON_START"] = "1"

    import uvicorn  # noqa: E402  (after sys.path is set)

    import app as app_module  # noqa: E402  (after sys.path is set)

    config = uvicorn.Config(
        app_module.app, host=host, port=port, log_level=log_level, workers=1)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="multiva-server",
                              daemon=True)
    thread.start()

    url = public_url(port, host)
    if not wait_until_ready(url, timeout=ready_timeout):
        server.should_exit = True
        thread.join(timeout=15)
        raise RuntimeError(
            f"The server did not become ready at {url} within "
            f"{ready_timeout:.0f}s. Check the log output above.")
    return ServerHandle(url=url, host=host, port=port, server=server,
                        thread=thread)
