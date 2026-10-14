import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "training", "needle")
DISTRO = "Ubuntu"
VENV = "/home/archit/needle-venv"


def wsl_path(path):
    drive, rest = os.path.splitdrive(os.path.abspath(path))
    return "/mnt/" + drive[0].lower() + rest.replace("\\", "/")


def wsl(command, stream=False, log_path=None):
    full = "cd {} && {}".format(wsl_path(ROOT), command)
    proc = subprocess.Popen(
        ["wsl", "-d", DISTRO, "--", "bash", "-lc", full],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    started = time.time()
    try:
        for line in proc.stdout:
            if stream:
                stamped = "[{:>8.1f}s] {}".format(time.time() - started, line.rstrip())
                print(stamped, flush=True)
                if log:
                    log.write(stamped + "\n")
                    log.flush()
            else:
                print(line.rstrip(), flush=True)
        return proc.wait()
    finally:
        if log:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--data", default=os.path.join(TRAIN_DIR, "train.jsonl"))
    parser.add_argument("--out", default=os.path.join(TRAIN_DIR, "adapter.safetensors"))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Warm-start base from scripts/make_warmstart.py. Defaults to the stock base.",
    )
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    data = wsl_path(args.data)
    out = wsl_path(args.out)

    if not args.skip_check:
        print("preflight: checking every example fits in {} tokens".format(args.max_len), flush=True)
        code = wsl(
            "{}/bin/python scripts/check_needle_budget.py --data {} --max-len {}".format(
                VENV, data, args.max_len
            )
        )
        if code != 0:
            print("preflight failed, not training. raise --max-len and retry.", flush=True)
            return 1

    command = (
        "PYTHONUNBUFFERED=1 {venv}/bin/needle finetune {data} --epochs {epochs} --max-len {max_len} "
        "--batch-size {batch} --lora-rank {rank} --lora-alpha {alpha} --val-split 0.1 {warm}--out {out}"
    ).format(
        venv=VENV,
        data=data,
        epochs=args.epochs,
        max_len=args.max_len,
        batch=args.batch_size,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        warm="--checkpoint {} ".format(wsl_path(args.checkpoint)) if args.checkpoint else "",
        out=out,
    )
    log_path = os.path.splitext(os.path.abspath(args.out))[0] + ".log"
    print("logging to {}".format(os.path.relpath(log_path, ROOT)), flush=True)
    return wsl(command, stream=True, log_path=log_path)


if __name__ == "__main__":
    sys.exit(main())
