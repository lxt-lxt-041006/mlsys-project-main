from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm.openai_client import client


ROOT = Path(__file__).resolve().parents[1]
TARGET_SPEC_PATH = Path("/target/target_spec.json")
OUTPUT_PATH = ROOT / "output.json"
WORK_DIR = ROOT / ".agent_workspace"
DIAG_PATH = WORK_DIR / "diagnostics.json"
GENERATED_CUDA_PATH = WORK_DIR / "generated_probe.cu"
PROMPT_DIR = ROOT / "agent" / "prompts"
MAX_RETRY = 3


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


class ShellTool:
    @staticmethod
    def run(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> CmdResult:
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return CmdResult(result.returncode, result.stdout, result.stderr)
        except Exception as exc:
            return CmdResult(1, "", f"{type(exc).__name__}: {exc}")


class TargetSpecLoader:
    @staticmethod
    def load() -> list[str]:
        if TARGET_SPEC_PATH.exists():
            with TARGET_SPEC_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            sample = ROOT / "target_spec_sample.json"
            with sample.open("r", encoding="utf-8") as f:
                data = json.load(f)

        targets = data.get("targets", [])
        if not isinstance(targets, list) or not all(isinstance(t, str) and t.strip() for t in targets):
            raise ValueError("Invalid target spec format. Expecting {'targets': ['name1', 'name2', ...]}.")
        return [t.strip() for t in targets]


class ProbeAgent:
    def __init__(self) -> None:
        self.targets = TargetSpecLoader.load()
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        self.context = self._collect_context()
        self.system_prompt = self._load_system_prompt()
        self._local_probe_cache: dict[str, float] | None = None
        self._python_probe_cache: dict[str, float] | None = None
        self._driver_probe_cache: dict[str, float] | None = None
        self.diag: dict[str, Any] = {}

    def _load_system_prompt(self) -> str:
        prompt_file = PROMPT_DIR / "system_probe.txt"
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8").strip()
        return (
            "You are an expert GPU hardware probing engineer. "
            "Use measurement-first methodology, avoid static spec-table lookup, "
            "and prefer robust scripts with retries and numeric JSON outputs."
        )

    @staticmethod
    def _extract_fenced_code(text: str) -> str:
        if not text:
            return ""
        # 支持 ```cpp / ```cuda / ```c++ / ```python 等任意围栏语言
        fence = re.search(r"```[a-zA-Z0-9_+\-]*\s*(.*?)```", text, flags=re.DOTALL)
        code = fence.group(1).strip() if fence else text.strip()

        # 某些模型会把语言标记误放进正文首行（如: cpp / cuda）
        lines = code.splitlines()
        if lines and lines[0].strip().lower() in {"cpp", "c++", "cuda", "cc", "c"}:
            code = "\n".join(lines[1:]).strip()
        return code

    @staticmethod
    def _extract_json_blob(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        stripped = text.strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _collect_context(self) -> dict[str, str]:
        commands = {
            "python_version": [sys.executable, "--version"],
            "nvidia_smi_basic": [
                "nvidia-smi",
                "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw",
                "--format=csv,noheader,nounits",
            ],
            "nvidia_smi_q": ["nvidia-smi", "-q"],
            "nvcc_version": ["nvcc", "--version"],
            "ncu_version": ["ncu", "--version"],
        }
        info: dict[str, str] = {}
        for key, cmd in commands.items():
            result = ShellTool.run(cmd, timeout=30)
            text = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
            if key == "nvidia_smi_q":
                info[key] = text.strip()[:50000]
            else:
                info[key] = text.strip()[:8000]
        return info

    def _query_nvidia_smi_fields(self, fields: list[str]) -> dict[str, float]:
        if not fields:
            return {}
        cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
        result = ShellTool.run(cmd, timeout=20)
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        first = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) != len(fields):
            return {}

        out: dict[str, float] = {}
        for k, v in zip(fields, parts):
            try:
                out[k] = float(v)
            except ValueError:
                continue
        return out

    def _query_nvidia_smi_name(self) -> str:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        ]
        result = ShellTool.run(cmd, timeout=15)
        if result.returncode != 0 or not result.stdout.strip():
            return ""
        return result.stdout.strip().splitlines()[0].strip()

    def _build_cuda_probe_prompt(self, previous_error: str | None = None) -> str:
        return (
            "Generate one standalone CUDA C++ source file that probes GPU metrics and prints exactly one JSON object.\n"
            "Requirements:\n"
            "1) Must include kernel + main and compile with nvcc.\n"
            "2) Must measure peak_fp32_tflops by timing a compute-heavy FP32 kernel (not hardcoded).\n"
            "3) Also output these keys when available: physical_sm_count, max_shmem_per_block_kb, l2_cache_size_kb, l2_cache_size_bytes, warp_size, max_threads_per_block, max_threads_per_sm, global_memory_size_mb, actual_core_clock_mhz, memory_clock_mhz.\n"
            "4) Never use static online spec tables.\n"
            "5) Output JSON only to stdout in main().\n"
            f"Requested targets: {json.dumps(self.targets, ensure_ascii=False)}\n"
            + (f"Previous compile/runtime error:\n{previous_error}\n" if previous_error else "")
            + "Return raw CUDA code only."
        )

    def _request_cuda_probe_code(self, previous_error: str | None = None) -> str:
        prompt = self._build_cuda_probe_prompt(previous_error=previous_error)
        response = client.chat.completions.create(
            model=os.getenv("BASE_MODEL", ""),
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return self._extract_fenced_code(content)

    def _run_llm_cuda_probe(self, previous_error: str | None = None) -> tuple[dict[str, Any] | None, str]:
        code = self._request_cuda_probe_code(previous_error=previous_error)
        GENERATED_CUDA_PATH.write_text(code, encoding="utf-8")

        nvcc = shutil.which("nvcc")
        if not nvcc:
            return None, "nvcc not found in PATH"

        exe = WORK_DIR / ("generated_probe.exe" if os.name == "nt" else "generated_probe")
        compile_result = ShellTool.run([nvcc, str(GENERATED_CUDA_PATH), "-O3", "--use_fast_math", "-o", str(exe)], timeout=240)
        if compile_result.returncode != 0:
            err = (compile_result.stdout or "") + "\n" + (compile_result.stderr or "")
            return None, f"CUDA compile failed:\n{err.strip()}"

        run_result = ShellTool.run([str(exe)], timeout=300)
        if run_result.returncode != 0:
            err = (run_result.stdout or "") + "\n" + (run_result.stderr or "")
            return None, f"CUDA run failed:\n{err.strip()}"

        data = self._extract_json_blob(run_result.stdout)
        if data is None:
            return None, f"CUDA probe output is not valid JSON. stdout:\n{run_result.stdout}\nstderr:\n{run_result.stderr}"
        return data, "ok"

    def _validate_output(self, data: dict[str, Any]) -> tuple[bool, str]:
        missing = [t for t in self.targets if t not in data]
        if missing:
            return False, f"Missing targets: {missing}"

        bad = [t for t in self.targets if not self._is_number(data.get(t))]
        if bad:
            return False, f"Non-numeric values for: {bad}"

        return True, "ok"

    def _run_local_nvcc_probe(self) -> dict[str, float]:
        self._local_probe_cache = {}
        return self._local_probe_cache

    def _run_python_runtime_probe(self) -> dict[str, float]:
        self._python_probe_cache = {}
        return self._python_probe_cache

    def _run_cuda_driver_probe(self) -> dict[str, float]:
        self._driver_probe_cache = {}
        return self._driver_probe_cache

    def run(self) -> None:
        last_error: str | None = None
        final_values: dict[str, Any] | None = None
        cuda_path_error: str | None = None

        for attempt in range(1, MAX_RETRY + 1):
            print(f"[Agent] attempt {attempt}/{MAX_RETRY}")
            try:
                data, cuda_info = self._run_llm_cuda_probe(previous_error=last_error)
                if data is not None:
                    ok, reason = self._validate_output(data)
                    if ok:
                        final_values = data
                        print("[Agent] llm->cuda probe succeeded.")
                        self.diag["probe_path"] = "llm_cuda"
                        break
                    cuda_path_error = f"CUDA probe output validation failed: {reason}"
                else:
                    cuda_path_error = cuda_info
            except Exception as exc:
                cuda_path_error = f"LLM->CUDA path exception: {type(exc).__name__}: {exc}"
            last_error = cuda_path_error
            print(f"[Agent] llm->cuda failed in attempt {attempt}.")

        if final_values is None:
            print("[Agent] all llm->cuda attempts failed.")
            final_values = {t: -1.0 for t in self.targets}

        if cuda_path_error:
            self.diag["cuda_path_error"] = cuda_path_error[:3000]

        normalized = {
            t: float(final_values[t]) if self._is_number(final_values.get(t)) else -1.0
            for t in self.targets
        }

        if "peak_fp32_tflops" in normalized and normalized["peak_fp32_tflops"] <= 0:
            normalized["peak_fp32_tflops"] = -1.0
            self.diag["peak_fp32_reason"] = (
                "No executable FP32 CUDA benchmark succeeded. On Windows this is commonly caused by nvcc requiring cl.exe in PATH."
            )

        expected_gpu_regex = os.getenv("EXPECTED_GPU_REGEX", "").strip()
        detected_gpu_name = self._query_nvidia_smi_name()
        warnings: list[str] = []
        if expected_gpu_regex and detected_gpu_name:
            try:
                if not re.search(expected_gpu_regex, detected_gpu_name, flags=re.IGNORECASE):
                    warnings.append(
                        f"GPU name mismatch: expected /{expected_gpu_regex}/ but got '{detected_gpu_name}'."
                    )
            except re.error:
                warnings.append(f"Invalid EXPECTED_GPU_REGEX: {expected_gpu_regex}")

        diag = {
            "detected_gpu_name": detected_gpu_name,
            "expected_gpu_regex": expected_gpu_regex,
            "warnings": warnings,
        }
        diag.update(self.diag)
        DIAG_PATH.write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
        if warnings:
            print(f"[Agent][WARN] {warnings[0]}")

        OUTPUT_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Agent] wrote {OUTPUT_PATH}")


def main() -> None:
    agent = ProbeAgent()
    agent.run()


if __name__ == "__main__":
    main()
