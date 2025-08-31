import argparse
import json


def main():
    p = argparse.ArgumentParser(description="Filter a JSONL by task_type")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", required=True)
    p.add_argument("--task-type", dest="task_type", required=True)
    args = p.parse_args()

    kept = 0
    with open(args.inp, "r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("task_type") == args.task_type:
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    print({"kept": kept, "out": args.out})


if __name__ == "__main__":
    main()
