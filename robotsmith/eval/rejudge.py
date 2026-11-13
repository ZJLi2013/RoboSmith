"""python -m robotsmith.eval.rejudge <judge.jsonl>...: print each log's verdict as one JSON line."""

import json
import sys

from robotsmith.eval.judge import judge

if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(json.dumps(judge(path).to_dict()))
