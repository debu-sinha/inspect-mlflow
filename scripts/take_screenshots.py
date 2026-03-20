"""Take comprehensive screenshots using direct URL navigation."""

import re

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5557"
OUT = "/Users/debu.sinha/repos/inspect-mlflow/docs/images"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1920, "height": 1080}, color_scheme="dark")
        page = ctx.new_page()

        # 1. Experiment overview
        page.goto(f"{BASE}/#/experiments/1", wait_until="domcontentloaded")
        page.wait_for_timeout(8000)
        page.screenshot(path=f"{OUT}/screenshot-01-overview.png")
        print("1: Overview")

        # Get all run IDs
        content = page.content()
        run_ids = list(set(re.findall(r"runs/([a-f0-9]{32})", content)))
        print(f"  Run IDs found: {len(run_ids)}")

        # 2. Navigate to each run to find the task run with metrics
        for rid in run_ids:
            page.goto(f"{BASE}/#/experiments/1/runs/{rid}", wait_until="domcontentloaded")
            page.wait_for_timeout(5000)
            run_content = page.inner_text("body")
            if "match/accuracy" in run_content or "accuracy" in run_content:
                page.screenshot(path=f"{OUT}/screenshot-02-task-run.png")
                print(f"2: Task run with metrics ({rid[:8]})")

                # Scroll to metrics section
                page.evaluate("window.scrollTo(0, 500)")
                page.wait_for_timeout(2000)
                page.screenshot(path=f"{OUT}/screenshot-03-run-scrolled.png")
                print("3: Run scrolled (metrics)")
                break

        # 3. Traces list
        page.goto(f"{BASE}/#/experiments/1/traces", wait_until="domcontentloaded")
        page.wait_for_timeout(8000)
        page.screenshot(path=f"{OUT}/screenshot-04-traces-list.png")
        print("4: Traces list")

        # 4. Find trace IDs in the page
        traces_content = page.content()
        trace_ids = list(set(re.findall(r"tr-[a-f0-9]{32}", traces_content)))
        print(f"  Trace IDs found: {trace_ids}")

        if trace_ids:
            # Click the first trace link
            page.click(f"text={trace_ids[0][:20]}")
            page.wait_for_timeout(5000)

            # Switch to Details & Timeline
            tabs = page.query_selector_all("[role='tab']")
            for tab in tabs:
                if "Details" in tab.inner_text() or "Timeline" in tab.inner_text():
                    tab.click()
                    break
            page.wait_for_timeout(4000)
            page.screenshot(path=f"{OUT}/screenshot-05-span-tree.png")
            print("5: Span tree")

            # Click tool span
            tool_el = page.query_selector("text=/tool:calculator/")
            if tool_el:
                tool_el.click()
                page.wait_for_timeout(3000)
                page.screenshot(path=f"{OUT}/screenshot-06-tool-detail.png")
                print("6: Tool span detail")

            # Click model span
            model_el = page.query_selector("text=/model:openai/")
            if model_el:
                model_el.click()
                page.wait_for_timeout(3000)
                page.screenshot(path=f"{OUT}/screenshot-07-llm-detail.png")
                print("7: LLM span detail")

        browser.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
