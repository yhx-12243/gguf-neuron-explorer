"""gne modal: 手元に載らないモデルのために、gne のコマンドを Modal の GPU で回す(docs/modal.md)。

ここは modal コマンドを呼ぶだけで、modal のライブラリは読まない。実際の定義は remote/modal_app.py にある。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import typer

APP_FILE = Path(__file__).resolve().parent / "remote" / "modal_app.py"
FETCH_FILE = APP_FILE.with_name("modal_fetch.py")
UPLOAD_FILE = APP_FILE.with_name("modal_upload.py")
MODELS_VOL, DATA_VOL = "gne-models", "gne-data"
modal_app = typer.Typer(no_args_is_help=True, help="在 Modal 的 GPU 上运行 gne（docs/modal.md）。价格见 https://modal.com/pricing")


def _modal() -> str:
    exe = shutil.which("modal")
    if not exe:
        sys.stderr.write("找不到 modal 命令。请先 uv tool install modal，再用 modal setup 登录\n")
        raise typer.Exit(2)
    return exe


def _call(args: list[str], dry_run: bool = False) -> None:
    cmd = [_modal(), *args]
    if dry_run:
        print(json.dumps(cmd[1:], ensure_ascii=False))
        return
    code = subprocess.call(cmd)
    if code:
        raise typer.Exit(code)


def parse_seconds(s: str) -> int:
    """6h・90m・3600 を秒にする。Modal の上限は 24 時間。"""
    m = re.fullmatch(r"(\d+)([smh]?)", s.strip())
    if not m:
        raise typer.BadParameter(f"{s!r}。请写成 6h、90m、3600 的形式")
    sec = int(m[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[m[2]]
    if not 1 <= sec <= 24 * 3600:
        raise typer.BadParameter("范围为 1 秒到 24 小时")
    return sec


@modal_app.command("fetch")
def fetch(repo: str = typer.Argument(..., help="Hugging Face 仓库（例如 unsloth/Qwen3.8-Flash-Next-GGUF）"),
          include: Optional[list[str]] = typer.Option(None, "--include", help="要下载的文件模式（例如 '*UD-IQ1_S*'）。省略则下载全部"),
          dry_run: bool = typer.Option(False, "--dry-run", help="仅打印要执行的 modal 命令")):
    """モデルを Modal の Volume に直接落とす。手元のディスクは使わず、GPU も使わない。終わると -m に渡すパスを出す。"""
    if not include:
        sys.stderr.write("未指定 --include，将下载仓库的全部文件（每种量化会有数十至数百 GB）\n")
    _call(["run", f"{FETCH_FILE}::main", "--repo", repo, "--include-json", json.dumps(include or [])], dry_run)


@modal_app.command("put")
def put(local: Path = typer.Argument(..., exists=True, help="要上传的文件或文件夹"),
        remote: Optional[str] = typer.Argument(None, help="Volume 上的位置（默认 inputs/<名称>）"),
        force: bool = typer.Option(False, "--force", help="同名时覆盖"),
        dry_run: bool = typer.Option(False, "--dry-run", help="仅打印要执行的 modal 命令")):
    """質問集やレシピを上げる。Modal に上がるのは、ここで名指ししたものと、git で追跡している gne/・configs/ だけ。"""
    dst = (remote or f"inputs/{local.name}").lstrip("/")
    _call(["volume", "put", *(["--force"] if force else []), DATA_VOL, str(local), dst], dry_run)
    if not dry_run:
        print(f"/data/{dst}")


@modal_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context,
        gpu: Optional[str] = typer.Option(None, "--gpu", help="GPU 类型（默认 A100-80GB，无空闲则 H100）。例如 H200。不读取权重的检查用 T4 即可"),
        cpu: float = typer.Option(8, "--cpu", help="CPU 核心数（gne 的线程数也使用此值）"),
        memory: int = typer.Option(32768, "--memory", help="内存（MiB）"),
        timeout: str = typer.Option("6h", "--timeout", help="时间上限。按每次执行计算。最大 24h"),
        script: Optional[str] = typer.Option(None, "--script", help="代替 gne 执行的 Python 脚本（容器内路径。用 gne modal put 上传）"),
        detach: bool = typer.Option(True, "--detach/--attach", help="本地连接断开后继续运行。中途停止用 modal app stop。--attach 未必会停止，请用 modal app list 确认"),
        dry_run: bool = typer.Option(False, "--dry-run", help="仅打印要执行的 modal 命令")):
    """gne のコマンドを Modal で実行する。例: gne modal run -- heads pipeline --spec /data/inputs/s.json -m /models/x.gguf --yes

    動いている時間だけ課金される。コマンドが終わるとコンテナも止まる。結果は /data/gne に残り、gne modal get で持ち帰る。"""
    argv = list(ctx.args)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv and not script:
        raise typer.BadParameter("请在 -- 之后写 gne 命令")
    args = ["run", *(["--detach"] if detach else []), f"{APP_FILE}::main", "--argv-json", json.dumps(argv, ensure_ascii=False),
            "--cpu", str(cpu), "--memory", str(memory), "--timeout", str(parse_seconds(timeout))]
    if gpu:
        args += ["--gpu", gpu]
    if script:
        args += ["--script", script]
    _call(args, dry_run)


TUNNEL_RE = re.compile(r"GNE_TUNNEL (\S+) (\d+)")
APP_RE = re.compile(r"\bap-[A-Za-z0-9]+")


def stop_app(exe: str, app_id: Optional[str]) -> bool:
    """modal run を止めても Modal 側のコンテナは動き続けるので、課金を止めるには明示的に止める。"""
    if not app_id:
        sys.stderr.write("注意：无法获知 App ID，不能停止 Modal 侧。请用 modal app list 确认后执行 modal app stop\n")
        return False
    if subprocess.run([exe, "app", "stop", "-y", app_id], capture_output=True).returncode:
        sys.stderr.write(f"注意：{app_id} 无法停止。会继续计费。请执行 modal app stop {app_id} 将其停止\n")
        return False
    sys.stderr.write(f"{app_id} 已停止\n")
    return True


def ssh_args(host: str, port: int, key: Path, known_hosts: Path, local_port: int, remote_port: int = 8791) -> list[str]:
    """手元の 127.0.0.1 だけに UI を出す。サーバの鍵は作った本人のものだけを信用する。"""
    return ["ssh", "-N", "-i", str(key), "-p", str(port), "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
            "-o", "IdentitiesOnly=yes", "-o", f"UserKnownHostsFile={known_hosts}", "-o", "StrictHostKeyChecking=yes",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", f"root@{host}"]


def make_keys(work: Path) -> tuple[Path, str, str, str]:
    """この接続だけの使い捨ての鍵を作る。返り値は (手元の秘密鍵, 手元の公開鍵, サーバの秘密鍵, サーバの公開鍵)。"""
    for name in ("client", "host"):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "gne-modal", "-f", str(work / name)], check=True)
    return (work / "client", (work / "client.pub").read_text().strip(), (work / "host").read_text(),
            (work / "host.pub").read_text().strip())


@modal_app.command("serve")
def serve(model: str = typer.Option(..., "--model", "-m", help="容器内路径（/models/...）"),
          gpu: Optional[str] = typer.Option(None, "--gpu", help="GPU 类型（默认 A100-80GB，无空闲则 H100）。例如 H200"),
          cpu: float = typer.Option(8, "--cpu", help="CPU 核心数"),
          memory: int = typer.Option(32768, "--memory", help="内存（MiB）"),
          timeout: str = typer.Option("2h", "--timeout", help="时间上限。最大 24h"),
          idle: str = typer.Option("15m", "--idle", help="SSH 连接数低于此值则停止"),
          port: int = typer.Option(8791, "--port", help="本地端口"),
          dry_run: bool = typer.Option(False, "--dry-run", help="仅打印要执行的 modal 命令")):
    """Web UI を Modal の GPU で動かし、手元の 127.0.0.1 で開く。動いている間は GPU の料金がかかる。Ctrl+C で止まる。

    Modal の公開 URL は使わない。gne serve はコンテナの中でも 127.0.0.1 だけで待ち受け、この接続だけの使い捨ての鍵を
    持つ SSH のトンネルでつなぐ。チャットの内容は Modal のコンテナを通るが、ログには出ない。"""
    args = ["run", f"{APP_FILE}::serve_main", "--model", model, "--cpu", str(cpu), "--memory", str(memory),
            "--timeout", str(parse_seconds(timeout)), "--idle", str(parse_seconds(idle))]
    if gpu:
        args += ["--gpu", gpu]
    if dry_run:
        print(json.dumps(args, ensure_ascii=False))
        return
    exe = _modal()
    for tool in ("ssh", "ssh-keygen"):
        if not shutil.which(tool):
            sys.stderr.write(f"{tool} 不存在\n")
            raise typer.Exit(2)
    with tempfile.TemporaryDirectory(prefix="gne-modal-") as d:
        work = Path(d)
        key, client_pub, host_key, host_pub = make_keys(work)
        known = work / "known_hosts"
        proc = subprocess.Popen([exe, *args], env={**os.environ, "GNE_SSH_PUB": client_pub, "GNE_SSH_HOST_KEY": host_key},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        target: dict = {}

        def read_output():
            for line in proc.stdout:
                a = APP_RE.search(line)
                if a:
                    target.setdefault("app", a[0])
                m = TUNNEL_RE.search(line)
                if m:
                    target["host"] = (m[1], int(m[2]))     # 横取りで作り直されると、新しいトンネルの行が来る
                else:
                    sys.stdout.write(line)
                    sys.stdout.flush()
        threading.Thread(target=read_output, daemon=True).start()
        ssh, current = None, None
        try:
            sys.stderr.write("正在启动容器。首次构建镜像会花较长时间\n")
            while proc.poll() is None:
                time.sleep(1)
                want = target.get("host")
                if want and (want != current or ssh is None or ssh.poll() is not None):
                    if ssh is not None and ssh.poll() is None:
                        ssh.terminate()
                    if want != current:
                        known.write_text(f"[{want[0]}]:{want[1]} {host_pub}\n")
                        sys.stderr.write(f"可打开：http://127.0.0.1:{port}/（按 Ctrl+C 停止后计费也会停止）\n")
                    else:
                        time.sleep(5)                       # 切れたらつなぎ直す
                    current = want
                    ssh = subprocess.Popen(ssh_args(want[0], want[1], key, known, port))
        except KeyboardInterrupt:
            pass
        finally:
            if ssh is not None and ssh.poll() is None:
                ssh.terminate()
            stop_app(exe, target.get("app"))
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
    raise typer.Exit(0 if proc.returncode in (0, -15, None) else proc.returncode)


def running_apps(exe: str) -> list[dict]:
    """止まっていない gne のアプリ(gne と gne-fetch)。"""
    out = subprocess.run([exe, "app", "list", "--json"], capture_output=True, text=True)
    if out.returncode:
        sys.stderr.write(out.stderr[-500:])
        raise typer.Exit(out.returncode)
    apps = json.loads(out.stdout or "[]")
    return [a for a in apps if a.get("description") in ("gne", "gne-fetch", "gne-upload") and not str(a.get("state", "")).startswith("stopped")]


@modal_app.command("ps")
def ps():
    """動いている gne のアプリ(止まりかけを含む)。課金が続いているかを確かめるのに使う。"""
    apps = running_apps(_modal())
    for a in apps:
        print(f"{a['app_id']}  {a['description']:9} {a['state']:12} 任务 {a.get('tasks')}  {a.get('created_at')}")
    print(f"运行中的应用：{len(apps)} 个")


@modal_app.command("stop")
def stop(app_ids: Optional[list[str]] = typer.Argument(None, help="要停止的应用 ID（省略时需要 --all）"),
         all_: bool = typer.Option(False, "--all", help="停止所有正在运行的 gne 应用")):
    """アプリを止める(コンテナも止まり、課金も止まる)。止まったかは gne modal ps で確かめる。"""
    exe = _modal()
    ids = list(app_ids or []) or ([a["app_id"] for a in running_apps(exe)] if all_ else [])
    if not ids:
        raise typer.BadParameter("请指定要停止的应用 ID 或 --all")
    bad = [i for i in ids if not stop_app(exe, i)]
    raise typer.Exit(1 if bad else 0)


@modal_app.command("cost")
def cost(period: str = typer.Option("this month", "--for", help="时间段（today / this month / last month 等）")):
    """Modal の課金の合計(modal billing report)。月の上限は Modal の Usage & Billing で決める。"""
    out = subprocess.run([_modal(), "billing", "report", "--for", period, "--json"], capture_output=True, text=True)
    if out.returncode:
        sys.stderr.write(out.stderr[-500:])
        raise typer.Exit(out.returncode)
    rows = json.loads(out.stdout or "[]")
    total = sum(float(r.get("Cost", r.get("cost", 0)) or 0) for r in rows)
    print(f"{period}: ${total:.2f}")


@modal_app.command("hf-upload")
def hf_upload(paths: list[str] = typer.Argument(..., help="Volume 上的文件（例如 export/x-00001-of-00004.gguf）"),
              repo: str = typer.Option(..., "--repo", help="Hugging Face 仓库（例如 user/name）"),
              public: bool = typer.Option(False, "--public", help="设为公开仓库（默认私有）"),
              card: Optional[Path] = typer.Option(None, "--card", exists=True, help="模型卡（作为 README.md 上传）"),
              remove: Optional[list[str]] = typer.Option(None, "--remove", help="要从仓库删除的文件（例如改名后的旧文件）"),
              dry_run: bool = typer.Option(False, "--dry-run", help="仅打印要执行的 modal 命令")):
    """Volume の上のファイルを、手元を通さず Hugging Face に上げる。トークンは Modal のシークレット huggingface(HF_TOKEN)から読む
    (modal secret create huggingface HF_TOKEN=<トークン>)。上げたものは外から見えるので、名前とモデルカードを確かめてから使う。"""
    args = ["run", f"{UPLOAD_FILE}::main", "--repo", repo, "--paths-json", json.dumps([p.lstrip("/") for p in paths])]
    if public:
        args += ["--no-private"]
    if card:
        args += ["--card-path", str(card.resolve())]
    if remove:
        args += ["--remove-json", json.dumps(list(remove))]
    _call(args, dry_run)


@modal_app.command("ls")
def ls(path: str = typer.Argument("", help="Volume 上的位置"),
       models: bool = typer.Option(False, "--models", help="查看模型存放位置（默认是结果存放位置）"),
       dry_run: bool = typer.Option(False, "--dry-run", help="仅打印要执行的 modal 命令")):
    """Volume の中を見る。-m に渡すパスは /models/<ここの表示>、コンテナの中の結果は /data/<ここの表示>。"""
    _call(["volume", "ls", MODELS_VOL if models else DATA_VOL, *([path] if path else [])], dry_run)


@modal_app.command("get")
def get(remote: str = typer.Argument(..., help="Volume 上的位置（例如 gne/<model_id>）"),
        local: Path = typer.Argument(Path("."), help="下载到本地的目标位置"),
        dry_run: bool = typer.Option(False, "--dry-run", help="仅打印要执行的 modal 命令")):
    """結果を手元に持ち帰る。"""
    _call(["volume", "get", DATA_VOL, remote.lstrip("/"), str(local)], dry_run)
