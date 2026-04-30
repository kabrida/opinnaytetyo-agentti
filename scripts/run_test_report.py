import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class TestRunResult:
    name: str
    command: list[str]
    returncode: int
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    deselected: int = 0
    duration_s: float = 0.0
    output: str = ""


def parse_pytest_summary(output: str) -> tuple[int, int, int, int, float]:
    passed = failed = skipped = deselected = 0
    duration_s = 0.0

    # Esim: "1 passed, 19 deselected in 12.21s"
    summary_match = re.search(r"([\w\s,]+) in ([0-9]+\.[0-9]+)s", output)
    if summary_match:
        duration_s = float(summary_match.group(2))
        summary_text = summary_match.group(1)
        for count, label in re.findall(r"(\d+)\s+(passed|failed|skipped|deselected)", summary_text):
            n = int(count)
            if label == "passed":
                passed = n
            elif label == "failed":
                failed = n
            elif label == "skipped":
                skipped = n
            elif label == "deselected":
                deselected = n

    return passed, failed, skipped, deselected, duration_s


def run_pytest_group(name: str, command: list[str], env: dict[str, str]) -> TestRunResult:
    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )

    output = (process.stdout or "") + ("\n" + process.stderr if process.stderr else "")
    passed, failed, skipped, deselected, duration_s = parse_pytest_summary(output)

    return TestRunResult(
        name=name,
        command=command,
        returncode=process.returncode,
        passed=passed,
        failed=failed,
        skipped=skipped,
        deselected=deselected,
        duration_s=duration_s,
        output=output,
    )


def print_results_table(results: list[TestRunResult]) -> None:
    headers = ["Ryhmä", "Status", "Passed", "Failed", "Skipped", "Deselected", "Aika (s)"]
    rows = []

    for r in results:
        status = "OK" if r.returncode == 0 else "FAIL"
        rows.append([
            r.name,
            status,
            str(r.passed),
            str(r.failed),
            str(r.skipped),
            str(r.deselected),
            f"{r.duration_s:.2f}",
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, col in enumerate(row):
            widths[i] = max(widths[i], len(col))

    def fmt_row(cols: list[str]) -> str:
        return " | ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

    separator = "-+-".join("-" * w for w in widths)
    print(fmt_row(headers))
    print(separator)
    for row in rows:
        print(fmt_row(row))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aja testiryhmat ja tulosta raporttitaulukko (sopii kuvakaappaukseen)."
    )
    parser.add_argument(
        "--with-live",
        action="store_true",
        help="Aja myos live_retrieval ja live_e2e testit.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python-suorittimen polku (oletus: nykyinen Python).",
    )
    args = parser.parse_args()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Testiraportti: {now}")
    print(f"Python: {args.python}")
    print()

    base_env = os.environ.copy()
    base_cmd = [args.python, "-m", "pytest", "-v"]

    groups: list[tuple[str, list[str], dict[str, str]]] = []

    unit_env = base_env.copy()
    groups.append(
        (
            "Yksikko/offline",
            base_cmd + ["-m", "not live_retrieval and not live_e2e and not live_retrieval_calibration"],
            unit_env,
        )
    )

    if args.with_live:
        retrieval_env = base_env.copy()
        retrieval_env["RUN_LIVE_RETRIEVAL_TESTS"] = "true"
        groups.append(("Live retrieval", base_cmd + ["-m", "live_retrieval"], retrieval_env))

        e2e_env = base_env.copy()
        e2e_env["RUN_LIVE_E2E_TESTS"] = "true"
        groups.append(("Live smoke", base_cmd + ["-m", "live_e2e"], e2e_env))

    results: list[TestRunResult] = []
    for name, cmd, env in groups:
        print(f"> Ajetaan: {name}")
        print("  " + " ".join(cmd))
        result = run_pytest_group(name=name, command=cmd, env=env)
        results.append(result)

    print("\nYhteenveto")
    print_results_table(results)

    any_fail = any(r.returncode != 0 for r in results)
    if any_fail:
        print("\nEpaonnistuneiden ajojen viimeiset rivit:")
        for r in results:
            if r.returncode == 0:
                continue
            print(f"\n[{r.name}] returncode={r.returncode}")
            tail = "\n".join(r.output.strip().splitlines()[-25:])
            print(tail)

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
