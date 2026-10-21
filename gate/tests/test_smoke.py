"""End-to-end smoke: gate + bench + exactness against the fake server, < 60 s.  python3 tests/test_smoke.py"""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]

import bench  # noqa: E402
import common  # noqa: E402
import exactness  # noqa: E402
import fake_server  # noqa: E402
import gate  # noqa: E402


def main():
    t0 = time.monotonic()
    assert gate.parse_hardmode("  ● TC-45  tool_choice=required Compliance  ✅ PASS  2/2\n│    Quality:        90/100") \
        == {"quality": 90, "tc45": "PASS"}
    _, ref = fake_server.serve()
    _, cand = fake_server.serve(perturb=True)
    d = tempfile.mkdtemp()
    p = lambda n: os.path.join(d, n)  # noqa: E731
    m = ["--model", "fake"]

    rc = gate.main(["--base", ref, *m, "--skip", "hardmode", "--depths", "8000", "--fid-trials", "1",
                    "--session-tokens", "9000", "--long-decode-tokens", "2000", "--out", p("g.json")])
    g = json.load(open(p("g.json")))
    assert rc == 0, {k: v for k, v in g["probes"].items() if v.get("pass") is False}
    assert g["probes"]["hardmode"]["skipped"] and g["probes"]["long_session"]["decoded"] >= 9000
    assert g["probes"]["tool_only_turn"]["prompt_token_delta_kept_reasoning"] >= 50
    rc = gate.main(["--base", ref, *m, "--only", "reasoning,tc45", "--engram", "off", "--compare-to", p("g.json"),
                    "--out", p("g2.json")])
    assert rc == 0 and json.load(open(p("g2.json")))["compare"]["deltas"]["tc45"]["other"] == 1.0

    bench.main(["--base", ref, *m, "--repeats", "2", "--sizes", "4096", "--prefill-conc", "1,4", "--out", p("b.json")])
    b = json.load(open(p("b.json")))
    assert len(b["cells"]) == 27 and all(c["aggregate_median"] for c in b["cells"]) and b["prefill"][0]["tps_median"]
    seeds = {r.get("seed") for c in b["cells"] for r in c["requests"]}
    assert all(c["per_stream_median"] for c in b["cells"]), "decode tps missing"
    c8 = next(c for c in b["cells"] if c["c"] == 8)
    assert len({r["seed"] for r in c8["requests"]}) == 16, "repeats must use distinct seeds"

    for base, tag in ((ref, "r"), (cand, "c")):
        exactness.main(["record-tokens", "--base", base, *m, "--out", p(f"t{tag}.json")])
        exactness.main(["record-logits", "--base", base, *m, "--out", p(f"l{tag}.json")])
    assert exactness.main(["compare", p("tr.json"), p("tr.json")]) == 0
    exactness.main(["record-tokens", "--base", ref, *m, "--conc", "8", "--out", p("t8.json")])
    assert exactness.main(["compare", p("tr.json"), p("t8.json")]) == 0           # batched == serial
    assert exactness.main(["compare", p("tr.json"), p("tc.json")]) == 1           # seed 22 diverges at 5
    assert exactness.main(["compare", p("lr.json"), p("lr.json"), "--exact"]) == 0
    assert exactness.main(["compare", p("lr.json"), p("lc.json"), "--exact"]) == 1
    json.dump({"top1_agreement": 0.997, "top1_first_copy": 0.99, "pass": True}, open(p("tf.json"), "w"))
    assert exactness.main(["compare", p("lr.json"), p("tf.json")]) == 0
    assert exactness.main(["accept", "--base", ref, *m, "--out", p("a.json")]) == 0
    a = json.load(open(p("a.json")))
    assert a["vllm_per_position"]["0"] > a["vllm_per_position"]["2"] and a["dspark_keep_per_draft"] > 0.6
    import pickle
    os.makedirs(p("dl"))
    ev = [("d", {"slot": 0, "k": 3}), ("k", {"slot": 0, "accepted": 2}), ("d", {"slot": 0, "k": 3}),
          ("k", {"slot": 0, "accepted": 0})]
    pickle.dump({"t0": 0, "events": ev}, open(p("dl/draftlog-1-00000.pkl"), "wb"))
    open(p("req.jsonl"), "w").write(json.dumps({"tokens_per_round": 2.0, "decode_tokens": 100}) + "\n")
    assert exactness.main(["accept", "--base", "", "--draft-log", p("dl"), "--request-log", p("req.jsonl"),
                           "--out", p("a2.json")]) == 0
    a2 = json.load(open(p("a2.json")))
    assert a2["draft_log_per_position"] == {"0": 0.5, "1": 0.5, "2": 0.0} and a2["request_log"]["tokens_per_round"] == 2
    assert exactness.kl_topk([[1, -0.1], [2, -2.5]], [[1, -0.1], [2, -2.5]]) < 1e-12
    print(f"smoke OK in {time.monotonic() - t0:.1f}s (seeds seen: {len(seeds)})")


if __name__ == "__main__":
    main()
