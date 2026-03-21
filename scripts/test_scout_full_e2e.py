"""Full E2E: eval -> traces -> Scout import -> Scout read back with messages."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

URI = "http://127.0.0.1:5559"
EXP = "scout-full-e2e"
DB = "/tmp/scout-e2e-db"


def phase1():
    print("=== Phase 1: Run eval ===")
    os.environ["MLFLOW_TRACKING_URI"] = URI
    os.environ["MLFLOW_INSPECT_TRACING"] = "true"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXP

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
    print(f"  Status: {logs[0].status}")
    print("  Phase 1 DONE.\n")


def phase2():
    print("=== Phase 2: Import + Read back ===")
    code = f'''
import asyncio, sys
sys.path.insert(0, "{os.path.join(os.path.dirname(__file__), "..")}")

async def run():
    from inspect_mlflow.scout import import_mlflow_traces
    from inspect_scout import transcripts_db
    from inspect_scout._transcript.types import TranscriptContent

    # Import
    async with transcripts_db("{DB}") as db:
        count = 0
        async for t in import_mlflow_traces(experiment_name="{EXP}", tracking_uri="{URI}"):
            await db.insert([t])
            count += 1
        print(f"  Imported {{count}} transcript(s)")

    # Read back with messages and events
    async with transcripts_db("{DB}") as db:
        c = await db.count()
        print(f"  Transcripts in DB: {{c}}")

        async for info in db.select():
            print(f"  ID: {{info.transcript_id}}")
            print(f"  Source: {{info.source_type}}")
            print(f"  Model: {{info.model}}")
            print(f"  Messages: {{info.message_count}}")
            print(f"  Tokens: {{info.total_tokens}}")
            print(f"  Score: {{info.score}}")

            tc = TranscriptContent(messages="all", events="all", timeline=None)
            t = await db.read(info, tc)
            print(f"  Read back messages: {{len(t.messages)}}")
            for msg in t.messages:
                print(f"    [{{msg.role}}] {{str(msg.content)[:80]}}")
            print(f"  Read back events: {{len(t.events)}}")
            for ev in t.events:
                etype = type(ev).__name__
                detail = getattr(ev, "model", getattr(ev, "function", ""))
                print(f"    {{etype}}: {{detail}}")

    print("  Phase 2 DONE.")

asyncio.run(run())
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": URI},
    )
    print(result.stdout)
    if result.stderr:
        lines = result.stderr.strip().split("\n")
        for line in lines[-5:]:
            print(f"  STDERR: {line}")
    return result.returncode


if __name__ == "__main__":
    phase1()
    rc = phase2()
    print(f"\n=== E2E TEST {'PASSED' if rc == 0 else 'FAILED'} ===")
    os.system("pkill -f 'mlflow server --port 5559' 2>/dev/null")
