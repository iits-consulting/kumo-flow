"""Unit tests for the chat module — the deterministic parts only (catalog,
validate_plan, turn gating). No LLM calls; live-model behavior is covered by
the manual acceptance cases in wiki/chat-assistant.md.

    uv run pytest
"""

from chat import EXAMPLE_TURNS, ChatTurn, Plan, PlanStep, catalog, system_prompt, turn_errors, validate_plan


def plan(steps, wiring, requirements=None):
    """Steps as (kind, config?) and wires as (from, port, to, target_port).
    Default requirements: one clause covering every step, so tests about other
    checks stay quiet on the coverage ones."""
    return Plan(
        steps=[PlanStep(kind=s[0], config=s[1] if len(s) > 1 else {}) for s in steps],
        wiring=[{"from": f, "port": p, "to": t, "target_port": tp} for f, p, t, tp in wiring],
        requirements=[{"text": "the goal", "steps": list(range(len(steps)))}] if requirements is None else requirements,
    )


def blur_plan(**overrides):
    """The spec's hand-built blur pipeline: load -> text_prompt -> sam3 -> blur -> view."""
    kw = dict(
        steps=[("load",), ("text_prompt", {"text": "person"}), ("segment",), ("blur",), ("view",)],
        wiring=[
            (0, "image", 2, "image"),
            (1, "prompts", 2, "prompts"),
            (0, "image", 3, "image"),
            (2, "masks", 3, "masks"),
            (3, "image", 4, "image"),
        ],
    )
    kw.update(overrides)
    return plan(**kw)


# --- catalog ------------------------------------------------------------------


def test_catalog_excludes_custom_and_note():
    text = catalog()
    assert "- custom " not in text and "- note " not in text
    assert "- load " in text  # sanity: real nodes are present


def test_catalog_modality_filter_drops_video_only_nodes():
    image_only = catalog("image")
    for kind in ("load_video", "track", "view_video", "sample_frames"):
        assert f"- {kind} " in catalog() and f"- {kind} " not in image_only


def test_catalog_entries_carry_docstrings():
    # MaskOps' docstring is the model's only hint about invert semantics
    entry = next(line for line in catalog().split("- mask_ops")[1].splitlines())
    assert "invert" in entry


# --- validate_plan --------------------------------------------------------------


def test_blur_pipeline_valid():
    assert validate_plan(blur_plan(), "image", "view") == []


def test_unknown_kind():
    p = blur_plan(steps=[("load",), ("frobnicate",), ("view",)])
    assert any("frobnicate" in e for e in validate_plan(p, "image", "view"))


def test_custom_rejected():
    p = plan([("load",), ("custom",), ("view",)], [(0, "image", 2, "image")])
    assert any("custom" in e for e in validate_plan(p, "image", "view"))


def test_missing_input_node():
    no_input = plan([("blur",), ("view",)], [(0, "image", 1, "image")])
    assert any("input node of kind 'load'" in e for e in validate_plan(no_input, "image", "view"))


def test_second_load_for_overlay_media_is_valid():
    # the license-plate case: blur the background, logo (its own load) onto the
    # plates — used to be inexpressible under the exactly-one-input rule
    p = plan(
        [("load",), ("text_prompt", {"text": "license plate"}), ("segment",), ("mask_ops", {"invert": True}),
         ("blur",), ("load",), ("warp_overlay",), ("view",)],
        [
            (0, "image", 2, "image"),
            (1, "prompts", 2, "prompts"),
            (2, "masks", 3, "masks"),
            (0, "image", 4, "image"),
            (3, "masks", 4, "masks"),
            (4, "image", 6, "image"),
            (5, "image", 6, "overlay"),
            (2, "masks", 6, "masks"),
            (6, "image", 7, "image"),
        ],
    )
    assert validate_plan(p, "image", "view") == []


def test_wrong_input_kind_for_brief():
    errs = validate_plan(blur_plan(), "video", "view")
    assert any("'load_video'" in e for e in errs)


def test_export_requires_export_step():
    errs = validate_plan(blur_plan(), "image", "export")
    assert any("export step" in e for e in errs)
    exported = blur_plan(
        steps=[("load",), ("text_prompt", {"text": "person"}), ("segment",), ("blur",), ("export",)]
    )
    assert validate_plan(exported, "image", "export") == []


def test_wire_index_out_of_range():
    p = blur_plan(wiring=[(0, "image", 9, "image")])
    assert any("out of range" in e for e in validate_plan(p, "image", "view"))


def test_non_mating_ports():
    p = plan([("load",), ("view",)], [(0, "labels", 1, "image")])
    assert any("does not mate" in e for e in validate_plan(p, "image", "view"))


def test_unknown_ports_on_wire():
    p = plan([("load",), ("view",)], [(0, "pixels", 1, "pixels")])
    errs = validate_plan(p, "image", "view")
    assert any("no output port" in e for e in errs) and any("no input port" in e for e in errs)


def test_unwired_required_port():
    # view's image is required; segment's prompts is not (its run() validates
    # prompting per model instead), so drop view's wire to hit the check
    p = blur_plan(wiring=[(0, "image", 2, "image"), (1, "prompts", 2, "prompts"), (0, "image", 3, "image"), (2, "masks", 3, "masks")])
    assert any("required input 'image' not wired" in e for e in validate_plan(p, "image", "view"))


def test_bad_config_key():
    p = blur_plan(steps=[("load",), ("text_prompt", {"text": "x"}), ("segment",), ("blur", {"radius": 3}), ("view",)])
    assert any("unknown config keys" in e for e in validate_plan(p, "image", "view"))


def test_bad_config_value():
    p = blur_plan(steps=[("load",), ("text_prompt", {"text": "x"}), ("segment",), ("blur", {"blur": "soft"}), ("view",)])
    assert any("bad config" in e for e in validate_plan(p, "image", "view"))


def test_cycle():
    p = plan(
        [("load",), ("blur",), ("flip",), ("view",)],
        [(0, "image", 1, "image"), (1, "image", 2, "image"), (2, "image", 1, "image"), (2, "image", 3, "image")],
    )
    assert any("cycle" in e for e in validate_plan(p, "image", "view"))


def test_disconnected_step():
    p = blur_plan(steps=[("load",), ("text_prompt", {"text": "x"}), ("segment",), ("blur",), ("view",), ("flip",)])
    assert any("not wired to anything" in e for e in validate_plan(p, "image", "view"))


def test_empty_requirements_rejected():
    p = blur_plan(requirements=[])
    assert any("requirements is empty" in e for e in validate_plan(p, "image", "view"))


def test_requirement_step_index_out_of_range():
    p = blur_plan(requirements=[{"text": "blur people", "steps": [0, 99]}])
    assert any("out of range" in e for e in validate_plan(p, "image", "view"))


# --- ChatTurn gating -------------------------------------------------------------


def test_json_plan_pasted_into_reply_fails_gating():
    # seen live: the model printed the whole plan as a ```json block in `reply`
    # and left `plan` null, so the apply button never appeared
    reply = 'Here is the graph:\n```json\n{"steps": [{"kind": "load"}], "wiring": []}\n```'
    turn = ChatTurn(reply=reply, input_kind="image", output_handling="view", plan=None)
    assert any("prose" in e for e in turn_errors(turn))


def test_plan_with_incomplete_brief_fails_gating():
    turn = ChatTurn(reply="here", input_kind="image", output_handling=None, plan=blur_plan())
    assert any("brief incomplete" in e for e in turn_errors(turn))


def test_complete_turn_passes_gating():
    turn = ChatTurn(reply="here", input_kind="image", output_handling="view", plan=blur_plan())
    assert turn_errors(turn) == []
    no_plan = ChatTurn(reply="which output?", input_kind="image", output_handling=None)
    assert turn_errors(no_plan) == []


def test_unsupported_clause_needs_a_reply():
    reqs = [{"text": "blur people", "steps": [1, 2, 3]}, {"text": "in 3D", "steps": []}]
    silent = ChatTurn(reply="  ", input_kind="image", output_handling="view", plan=blur_plan(requirements=reqs))
    assert any("steps: []" in e for e in turn_errors(silent))
    told = ChatTurn(
        reply="I can't render 3D.", input_kind="image", output_handling="view", plan=blur_plan(requirements=reqs)
    )
    assert turn_errors(told) == []


def test_rules_example_turns_pass_the_gate():
    # the prompt must never teach a shape the validator rejects
    for ex in EXAMPLE_TURNS:
        assert turn_errors(ChatTurn(**ex)) == [], ex["reply"]
    # the second example is the motivating failure: the modifier clause is covered
    kinds = [s["kind"] for s in EXAMPLE_TURNS[1]["plan"]["steps"]]
    assert "mask_ops" in kinds and "warp_overlay" in kinds


def test_system_prompt_embeds_recipes_from_agents_md():
    text = system_prompt()
    assert "Recipes" in text and "warp_overlay" in text
