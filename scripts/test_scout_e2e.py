"""End-to-end test: run eval -> create traces -> import into Scout -> verify.

Split into two phases because Inspect's eval() and Scout's async DB
can't share the same event loop.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TRACKING_URI = "http://127.0.0.1:5558"
EXPERIMENT = "scout-e2e-test"
DB_PATH = "/tmp/scout-e2e-db"


def phase1_run_eval():
    """Run eval to create MLflow traces."""
    print("=== Phase 1: Run eval to create MLflow traces ===")
    os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
    os.environ["MLFLOW_INSPECT_TRACING"] = "true"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT

    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import match
    from inspect_ai.solver import generate

    from inspect_mlflow.tracing import MlflowTracingHooks  # noqa: F401
    from inspect_mlflow.tracking import MlflowTrackingHooks  # noqa: F401

    task = Task(
        dataset=[
            Sample(input="What is 2 + 2?", target="4"),
            Sample(input="What is 3 * 5?", target="15"),
            Sample(input="What is 10 - 7?", target="3"),
        ],
        solver=generate(),
        scorer=match(),
    )
    logs = eval(task, model="openai/gpt-4o-mini", log_dir="/tmp/scout-e2e-logs")
    log = logs[0]
    print(f"  Eval status: {log.status}")
    if log.results and log.results.scores:
        for s in log.results.scores:
            for m, v in s.metrics.items():
                print(f"  {s.name}/{m}: {v.value}")

    # Verify traces exist
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    traces = mlflow.search_traces(experiment_ids=[exp.experiment_id])
    print(f"  Traces created: {len(traces)}")
    trace_id = traces.iloc[0]["trace_id"]
    trace = mlflow.get_trace(trace_id)
    print(f"  Spans in first trace: {len(trace.data.spans)}")
    for span in trace.data.spans:
        print(f"    {span.name} ({span.span_type})")
    print("  Phase 1 DONE.\n")


def phase2_import_and_verify():
    """Import traces into Scout and verify. Runs as subprocess to get clean event loop."""
    print("=== Phase 2: Import into Scout and verify ===")
    code = f'''
import asyncio
import sys
sys.path.insert(0, "{os.path.join(os.path.dirname(__file__), "..")}")

async def run():
    from inspect_mlflow.scout import import_mlflow_traces
    from inspect_scout import transcripts_db

    db_path = "{DB_PATH}"
    print(f"  Scout DB: {{db_path}}")

    async with transcripts_db(db_path) as db:
        count = 0
        async for t in import_mlflow_traces(
            experiment_name="{EXPERIMENT}",
            tracking_uri="{TRACKING_URI}",
        ):
            await db.insert([t])
            count += 1
        print(f"  Imported {{count}} transcript(s)")

    print()
    print("  === Verifying Scout transcripts ===")
    async with transcripts_db(db_path) as db:
        transcripts = await db.transcripts().list()
        print(f"  Transcripts in DB: {{len(transcripts)}}")
        for t in transcripts:
            print(f"    ID: {{t.transcript_id}}")
            print(f"    Source: {{t.source_type}}")
            print(f"    Model: {{t.model}}")
            print(f"    Messages: {{t.message_count}}")
            print(f"    Events: {{len(t.events)}}")
            print(f"    Tokens: {{t.total_tokens}}")
            print(f"    Time: {{t.total_time:.2f}}s" if t.total_time else "    Time: N/A")
            print(f"    Score: {{t.score}}")
            print(f"    Metadata: {{t.metadata}}")
            if t.messages:
                print("    Messages:")
                for msg in t.messages[:4]:
                    content = str(msg.content)[:80]
                    print(f"      [{{msg.role}}] {{content}}")
            if t.events:
                print("    Events:")
                for ev in t.events[:5]:
                    etype = type(ev).__name__
                    detail = getattr(ev, "model", getattr(ev, "function", ""))
                    print(f"      {{etype}}: {{detail}}")
            print()

    print("  Phase 2 DONE.")

asyncio.run(run())
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": TRACKING_URI},
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    if result.returncode != 0:
        print(f"  Phase 2 FAILED (exit code {result.returncode})")
    return result.returncode


if __name__ == "__main__":
    phase1_run_eval()
    exit_code = phase2_import_and_verify()
    print("\n=== E2E TEST", "PASSED" if exit_code == 0 else "FAILED", "===")
    pkill_result = os.system("pkill -f 'mlflow server --port 5558' 2>/dev/null")
