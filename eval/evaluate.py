import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.query import qa_chain

with open("eval/eval_set.json") as f:
    eval_set = json.load(f)

RESULTS = "eval/results.json"
done = {}
if os.path.exists(RESULTS):
    for r in json.load(open(RESULTS)):
        if r["answer"] != "ERROR":
            done[r["question"]] = r   # keep good results, redo errors


def norm(s):
    return s.lower().replace("’", "'").strip()


results = []
quota_hit = False
for i, item in enumerate(eval_set, 1):
    if item["question"] in done:
        results.append(done[item["question"]])
        continue
    if quota_hit:
        continue
    try:
        start = time.time()
        resp = qa_chain.invoke({"query": item["question"]})
        latency = time.time() - start
    except Exception as e:
        print(f"[{i}] STOPPING: {str(e)[:120]}")
        quota_hit = True
        continue

    answer = resp["result"]
    retrieved = [(os.path.basename(d.metadata["source"]), d.metadata["page"])
                 for d in resp["source_documents"]]
    retrieval_hit = None
    if item["type"] == "answerable":
        retrieval_hit = any(s == item["source"] and abs(p - item["page"]) <= 1
                            for s, p in retrieved)
    results.append({
        "question": item["question"], "expected": item["expected"],
        "answer": answer, "type": item["type"],
        "correct": norm(item["expected"]) in norm(answer),
        "retrieval_hit": retrieval_hit, "latency": latency,
        "retrieved": retrieved,
        "chunks": [d.page_content[:300] for d in resp["source_documents"]],
    })
    print(f"[{i}/{len(eval_set)}] correct={results[-1]['correct']} ({latency:.1f}s)")
    json.dump(results, open(RESULTS, "w"), indent=2)
    time.sleep(13)

json.dump(results, open(RESULTS, "w"), indent=2)


ans = [r for r in results if r["type"] == "answerable"]
unans = [r for r in results if r["type"] == "unanswerable"]

print("\n" + "=" * 50)
if quota_hit:
    print(f"INCOMPLETE: only {len(results)}/{len(eval_set)} questions answered.")
    print("Rerun the same command later to finish. Numbers below are partial.")
print(f"Questions completed:   {len(results)}/{len(eval_set)}")
if results:
    print(f"Overall accuracy:      {sum(r['correct'] for r in results)}/{len(results)}")
if ans:
    print(f"Answerable accuracy:   {sum(r['correct'] for r in ans)}/{len(ans)}")
    print(f"Retrieval hit rate:    {sum(bool(r['retrieval_hit']) for r in ans)}/{len(ans)}")
if unans:
    print(f"Correct refusals:      {sum(r['correct'] for r in unans)}/{len(unans)}")
if results:
    print(f"Average latency:       {sum(r['latency'] for r in results) / len(results):.2f}s")

print("\nFAILURES:")
for r in results:
    if not r["correct"]:
        print(f"\nQ: {r['question']}")
        print(f"  expected: {r['expected']}")
        print(f"  got:      {r['answer'][:200]}")
        print(f"  retrieval_hit: {r['retrieval_hit']}")

print("\nFull results saved to eval/results.json")