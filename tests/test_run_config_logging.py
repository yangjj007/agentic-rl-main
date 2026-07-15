import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import main


def test_run_config_summary_logs_teacher_probe_harness_metadata(monkeypatch, capsys):
    monkeypatch.setattr(main, "_is_main_process", lambda: True)

    main._log_run_config_summary(
        config_path="opd_7b_dyme_probe",
        dataset_config={"train_dataset": "data/train.json"},
        training_config={"dyme_args": {"output_dir": "out", "num_train_epochs": 10, "max_steps": -1}},
        opsd_config={
            "enabled": True,
            "mode": "dyme_teacher_probe_opd",
            "teacher_probe": {
                "enabled": True,
                "context_providers": ["format_only", "visual_facts_deplot"],
                "harness": "chartqa_closed_loop_recovery",
                "harness_version": "v12_executable_deplot",
                "prompt_profile": "chartqa_short_answer",
                "prompt_log": {"enabled": True, "max_records_per_rank": 32},
                "candidate_log": {"enabled": True},
                "max_new_tokens": 96,
            },
        },
        model_config={"teacher_model_path": "teacher", "teacher_device_map": "auto"},
        launch_config={},
    )

    line = capsys.readouterr().out.strip()
    assert line.startswith("[DyME-RUN-CONFIG] ")
    payload = json.loads(line.split(" ", 1)[1])
    probe = payload["teacher_probe"]
    assert probe["harness"] == "chartqa_closed_loop_recovery"
    assert probe["harness_version"] == "v12_executable_deplot"
    assert probe["prompt_log"] == {"enabled": True, "max_records_per_rank": 32}
