"""Verifiable reward functions for math and code."""
import re
import subprocess
from collections.abc import Callable


def extract_answer(response: str) -> str | None:
    patterns = [
        r"####\s*(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)",
        r"(?:answer|result|equals?|is)[:\s]+(-?\d+(?:\.\d+)?)",
        r"=\s*(-?\d+(?:\.\d+)?)\s*$",
        r"\\boxed\{(-?\d+(?:\.\d+)?)}",
        r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "")
    return None

def math_reward(response: str, ground_truth: str) -> float:
    answer = extract_answer(response)
    if answer is None:
        return -1.0
    try:
        if abs(float(answer) - float(ground_truth.replace(",", ""))) < 1e-6:
            return 1.0
    except ValueError:
        pass
    if answer.strip() == ground_truth.replace(",", "").strip():
        return 1.0
    return -0.5

def code_reward(code: str, expected_output: str, timeout: float = 5.0) -> float:
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return -1.0
        if result.stdout.strip() == expected_output.strip():
            return 1.0
        return -0.5
    except subprocess.TimeoutExpired:
        return -1.0
    except Exception:
        return -1.0

def reward_with_thinking(
    response: str,
    ground_truth: str,
    base_reward_fn: Callable[[str, str], float] = math_reward,
) -> float:
    has_think = "<think>" in response and "</think>" in response
    base_reward = base_reward_fn(response, ground_truth)

    if base_reward > 0 and has_think:
        return 1.0
    elif base_reward > 0:
        return 0.5
    elif has_think:
        return -0.2
    return -0.5

if __name__ == "__main__":
    test_cases = [
        ("The answer is 42", "42", 1.0),
        ("Let me think... 6 * 7 = 42", "42", 1.0),
        ("I think it's 43", "42", -0.5),
        ("I don't know", "42", -1.0),
    ]

    print("Testing math_reward:")
    for response, truth, expected in test_cases:
        result = math_reward(response, truth)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{response[:30]}...' -> {result} (expected {expected})")
