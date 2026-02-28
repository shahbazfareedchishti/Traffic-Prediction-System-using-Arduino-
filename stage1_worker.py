#!/usr/bin/env python3
"""
Background worker for fully automated Stage-1 processing.

It watches `stage1_inbox/*/_READY` and automatically runs:
1) stage1_process_session.py
2) optional pipeline.py trigger via --auto-pipeline passthrough
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_MODEL = BASE_DIR / "UVH-26-MV-YOLOv11-X.pt"


def build_process_command(
    session_id: str,
    inbox_dir: Path,
    output_dir: Path,
    model_path: Path,
    confidence: float,
    vehicle_class_ids: str,
    frame_step: int,
    auto_pipeline: bool,
    pipeline_script: Path,
    dataset: Path,
    pipeline_extra_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        str(BASE_DIR / "stage1_process_session.py"),
        "--session-id",
        session_id,
        "--inbox-dir",
        str(inbox_dir),
        "--output-dir",
        str(output_dir),
        "--model-path",
        str(model_path),
        "--confidence",
        str(confidence),
        "--vehicle-class-ids",
        vehicle_class_ids,
        "--frame-step",
        str(frame_step),
        "--pipeline-script",
        str(pipeline_script),
        "--dataset",
        str(dataset),
    ]
    if auto_pipeline:
        cmd.append("--auto-pipeline")
    if pipeline_extra_args:
        cmd.extend(["--pipeline-extra-args", *pipeline_extra_args])
    return cmd


def process_ready_session(
    session_dir: Path,
    inbox_dir: Path,
    output_dir: Path,
    model_path: Path,
    confidence: float,
    vehicle_class_ids: str,
    frame_step: int,
    auto_pipeline: bool,
    pipeline_script: Path,
    dataset: Path,
    pipeline_extra_args: list[str],
) -> None:
    session_id = session_dir.name
    ready_marker = session_dir / "_READY"
    done_marker = session_dir / "_DONE"
    failed_marker = session_dir / "_FAILED"

    if done_marker.exists():
        return

    cmd = build_process_command(
        session_id=session_id,
        inbox_dir=inbox_dir,
        output_dir=output_dir,
        model_path=model_path,
        confidence=confidence,
        vehicle_class_ids=vehicle_class_ids,
        frame_step=frame_step,
        auto_pipeline=auto_pipeline,
        pipeline_script=pipeline_script,
        dataset=dataset,
        pipeline_extra_args=pipeline_extra_args,
    )
    print(f"[worker] Processing session {session_id}")
    try:
        subprocess.run(cmd, check=True, cwd=str(BASE_DIR))
    except subprocess.CalledProcessError as exc:
        failed_marker.write_text(str(exc), encoding="utf-8")
        print(f"[worker] Failed session {session_id}: {exc}")
        return

    done_marker.write_text("processed", encoding="utf-8")
    if ready_marker.exists():
        ready_marker.unlink()
    if failed_marker.exists():
        failed_marker.unlink()
    print(f"[worker] Completed session {session_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-process ready Stage-1 sessions.")
    parser.add_argument("--inbox-dir", default="stage1_inbox")
    parser.add_argument("--output-dir", default="stage1_output")
    parser.add_argument("--model-path", default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--pipeline-script", default=str(BASE_DIR / "pipeline.py"))
    parser.add_argument("--dataset", default="historical_traffic_hourly.csv")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--vehicle-class-ids", default="2,3,5,7")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--auto-pipeline", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--once", action="store_true", help="Process current ready sessions then exit.")
    parser.add_argument(
        "--pipeline-extra-args",
        nargs="*",
        default=[],
        help="Extra args passed to pipeline.py via stage1_process_session.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inbox_dir = Path(args.inbox_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model_path).resolve()
    pipeline_script = Path(args.pipeline_script).resolve()
    dataset = Path(args.dataset).resolve()

    inbox_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ready_markers = sorted(inbox_dir.glob("*/_READY"))
        if ready_markers:
            for marker in ready_markers:
                process_ready_session(
                    session_dir=marker.parent,
                    inbox_dir=inbox_dir,
                    output_dir=output_dir,
                    model_path=model_path,
                    confidence=args.confidence,
                    vehicle_class_ids=args.vehicle_class_ids,
                    frame_step=max(1, args.frame_step),
                    auto_pipeline=args.auto_pipeline,
                    pipeline_script=pipeline_script,
                    dataset=dataset,
                    pipeline_extra_args=args.pipeline_extra_args,
                )
        elif args.once:
            break

        if args.once:
            break
        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    main()
