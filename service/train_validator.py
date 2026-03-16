"""
Validates training JSON batches before sending to /train endpoint.
Based on rules_generator.md section 7.
"""

from typing import Any, Dict, List, Tuple

VALID_TYPES = {
    "single_choice",
    "writing_text",
    "fill_blank",
    "ordering",
    "speaking_record",
    "image",
    # legacy — accepted but not recommended for new data
    "mcq",
    "open",
    "short_answer",
    "essay",
}

REQUIRED_FIELDS_BY_TYPE: Dict[str, List[str]] = {
    "single_choice":   ["question_id", "type", "text", "language", "difficulty", "options", "answer"],
    "mcq":             ["question_id", "type", "text", "language", "difficulty", "options", "answer"],
    "writing_text":    ["question_id", "type", "text", "language", "difficulty", "expected_keywords", "rubric", "examples_answers"],
    "open":            ["question_id", "type", "text", "language", "difficulty", "rubric", "examples_answers"],
    "essay":           ["question_id", "type", "text", "language", "difficulty", "rubric", "examples_answers"],
    "short_answer":    ["question_id", "type", "text", "language", "difficulty"],
    "fill_blank":      ["question_id", "type", "text", "language", "difficulty", "accepted_answers"],
    "ordering":        ["question_id", "type", "text", "language", "difficulty", "elements", "correct_order"],
    "speaking_record": ["question_id", "type", "text", "language", "difficulty", "rubric", "examples_answers"],
    "image":           ["question_id", "type", "text", "language", "difficulty", "image_description"],
}


def validate_training_batch(data: Any) -> Tuple[bool, List[str]]:
    """
    Validate a training batch JSON dict.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return False, ["Root value must be a JSON object"]

    if "train_id" not in data:
        errors.append("Missing required root field: train_id")

    examples = data.get("examples")
    if not isinstance(examples, list) or len(examples) == 0:
        errors.append("Field 'examples' must be a non-empty list")
        return False, errors

    seen_ids: set = set()
    for i, example in enumerate(examples):
        item_errors = _validate_example(example, i)
        errors.extend(item_errors)

        qid = example.get("question_id") if isinstance(example, dict) else None
        if qid:
            if qid in seen_ids:
                errors.append(f"examples[{i}]: duplicate question_id '{qid}'")
            seen_ids.add(qid)

    return len(errors) == 0, errors


def _validate_example(example: Any, idx: int) -> List[str]:
    errors: List[str] = []
    prefix = f"examples[{idx}]"

    if not isinstance(example, dict):
        return [f"{prefix}: item must be a JSON object"]

    q_type = example.get("type")
    if not q_type:
        errors.append(f"{prefix}: missing 'type' field")
        return errors

    if q_type not in VALID_TYPES:
        errors.append(f"{prefix}: invalid type '{q_type}'")
        return errors

    for field in REQUIRED_FIELDS_BY_TYPE.get(q_type, []):
        if field not in example:
            errors.append(f"{prefix}: missing required field '{field}' for type '{q_type}'")

    difficulty = example.get("difficulty")
    if difficulty is not None and not (1 <= int(difficulty) <= 5):
        errors.append(f"{prefix}: difficulty must be 1–5, got {difficulty}")

    if q_type in ("single_choice", "mcq"):
        answer = example.get("answer")
        options = example.get("options", [])
        if answer and options and answer not in options:
            errors.append(f"{prefix}: answer '{answer}' not found in options")
        if isinstance(options, list) and len(options) < 3:
            errors.append(f"{prefix}: single_choice requires at least 3 options")

    if q_type == "fill_blank":
        text = example.get("text", "")
        if "___" not in text:
            errors.append(f"{prefix}: fill_blank text must contain '___'")
        accepted = example.get("accepted_answers", [])
        if not accepted:
            errors.append(f"{prefix}: accepted_answers must not be empty")

    if q_type == "ordering":
        elements = example.get("elements", [])
        correct_order = example.get("correct_order", [])
        if len(elements) != len(correct_order):
            errors.append(
                f"{prefix}: elements ({len(elements)}) and correct_order ({len(correct_order)}) must have same length"
            )
        elif set(elements) != set(correct_order):
            errors.append(f"{prefix}: correct_order must be a permutation of elements")
        if len(elements) < 4:
            errors.append(f"{prefix}: ordering requires at least 4 elements")

    if q_type == "image":
        image_desc = example.get("image_description")
        if not image_desc or not str(image_desc).strip():
            errors.append(f"{prefix}: image_description must not be empty for type 'image'")

        options = example.get("options")
        answer = example.get("answer")
        has_choice_mode = options is not None or answer is not None
        has_descriptive_mode = example.get("rubric") is not None

        if not has_choice_mode and not has_descriptive_mode:
            errors.append(
                f"{prefix}: image type requires either (options + answer) for choice-based "
                "or (rubric) for descriptive evaluation"
            )

        if has_choice_mode:
            if not isinstance(options, list) or len(options) < 3:
                errors.append(f"{prefix}: image choice-based requires at least 3 options")
            if answer and options and answer not in options:
                errors.append(f"{prefix}: answer '{answer}' not found in options")

    for j, ea in enumerate(example.get("examples_answers", [])):
        if not isinstance(ea, dict):
            continue
        if "text" not in ea:
            errors.append(
                f"{prefix}.examples_answers[{j}]: missing required field 'text' "
                "(backend expects 'text', not 'answer')"
            )
        score = ea.get("score")
        if score is not None and not (0.0 <= float(score) <= 1.0):
            errors.append(f"{prefix}.examples_answers[{j}]: score {score} not in [0, 1]")

    rubric = example.get("rubric")
    if isinstance(rubric, dict):
        weights = rubric.get("criteria_weights")
        if isinstance(weights, dict) and weights:
            total = sum(float(v) for v in weights.values())
            if abs(total - 1.0) > 0.02:
                errors.append(
                    f"{prefix}: criteria_weights sum {total:.3f} is not ~1.0 (tolerance ±0.02)"
                )

    return errors
