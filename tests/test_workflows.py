"""cloud-runner.yml and the copy embedded in run_cloud_handoff.ps1 must not drift apart."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_handoff_script_embeds_the_current_workflow():
    script = _text(ROOT / "run_cloud_handoff.ps1")
    embedded = re.search(r"^\$CloudRunnerYaml = @'\n(.*?)\n'@$", script, re.S | re.M)
    assert embedded, "embedded cloud-runner.yml here-string not found"
    assert embedded.group(1).rstrip("\n") == _text(ROOT / ".github/workflows/cloud-runner.yml").rstrip("\n")


def test_cloud_model_drives_pipeline_and_cache():
    wf = _text(ROOT / ".github/workflows/cloud-runner.yml")
    assert "AGENT_REACH_OLLAMA_MODEL: ${{ github.event.inputs.model" in wf  # the pipeline labels with the chosen model
    assert wf.count("key: ollama-models-${{ env.MODEL }}-${{ env.EMBED_MODEL }}") == 2  # restore + save
    assert "ollama-models-llama3.1-8b" not in wf
