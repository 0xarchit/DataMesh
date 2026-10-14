import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.join(ROOT, "checkpoints", "needle3.safetensors"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.base):
        print("missing base checkpoint {}".format(args.base), flush=True)
        return 1

    from needle.model.checkpoints import read_adapter, read_checkpoint, write_checkpoint
    from needle.model.finetune import merge_lora

    checkpoint = read_checkpoint(args.base)
    adapter = read_adapter(args.adapter)
    lora = {tuple(key.split("/")): value for key, value in adapter["lora"].items()}
    scale = adapter["scale"]

    checkpoint["params"] = merge_lora(checkpoint["params"], lora, scale)
    checkpoint["run"] = dict(checkpoint.get("run") or {}, warmstart=os.path.basename(args.adapter))
    write_checkpoint(args.out, checkpoint)

    print(
        "merged {} weight groups at scale {:g} into base, wrote {}".format(
            len(lora), scale, args.out
        ),
        flush=True,
    )
    print("next: needle finetune <data.jsonl> --checkpoint {}".format(args.out), flush=True)
    return 0


def self_check():
    from needle.model.checkpoints import flatten, read_adapter, read_checkpoint

    adapter = read_adapter(os.path.join(ROOT, "training", "needle", "adapter_v10.safetensors"))
    assert adapter["scale"] > 0, "adapter carries no merge scale"
    assert adapter["base"], "adapter does not record its base checkpoint"
    for key, value in adapter["lora"].items():
        a, b = value["A"], value["B"]
        assert a.ndim == 3 and b.ndim == 3, "{} is not stacked over layers".format(key)
        assert a.shape[0] == b.shape[0], key
        assert a.shape[2] == b.shape[1] == adapter["rank"], "{} is not rank {}".format(
            key, adapter["rank"]
        )
        assert key.count("/") >= 1, key
    print(
        "adapter ok: {} groups, rank {}, scale {}".format(
            len(adapter["lora"]), adapter["rank"], adapter["scale"]
        )
    )
    base = read_checkpoint(os.path.join(ROOT, "checkpoints", "needle3.safetensors"))
    flat_keys = set(flatten(base["params"]))
    missing = [k for k in adapter["lora"] if k not in flat_keys]
    assert not missing, "adapter paths absent from base: {}".format(missing[:3])
    print("all {} adapter paths exist in base".format(len(adapter["lora"])))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-check":
        self_check()
        sys.exit(0)
    sys.exit(main())
