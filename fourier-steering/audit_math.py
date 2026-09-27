"""Dependency-free mathematical checks for RESEARCH_REVIEW.md.

These are synthetic checks and static prompt checks, not new Llama experiments.
Run: python3 fourier-steering/audit_math.py
"""
import ast
import json
import math
from pathlib import Path


def reconstruct(values, correct_nyquist):
    n = len(values)
    mean = sum(values) / n
    coefficients = []
    for k in range(1, n // 2 + 1):
        nyquist = n % 2 == 0 and k == n // 2
        scale = (1 if correct_nyquist and nyquist else 2) / n
        a = scale * sum(v * math.cos(math.tau * k * d / n)
                        for d, v in enumerate(values))
        b = 0 if correct_nyquist and nyquist else (2 / n) * sum(
            v * math.sin(math.tau * k * d / n) for d, v in enumerate(values))
        coefficients.append((k, a, b))
    return [mean + sum(a * math.cos(math.tau * k * d / n)
                       + b * math.sin(math.tau * k * d / n)
                       for k, a, b in coefficients) for d in range(n)]


def max_error(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def prompt_overlap():
    # Execute only the literal labels and pure prompt-generation functions.
    path = Path(__file__).with_name("fourier_steer_real.py")
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
                "shift", "day_prompts", "month_prompts"}:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {"DAYS", "MONTHS"}
                for t in node.targets):
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return {
        "weekday_test_in_centroid_training": any(
            "Today is Wednesday. Tomorrow is" in namespace["day_prompts"](d)
            for d in namespace["DAYS"]),
        "month_test_in_centroid_training": any(
            "The month after February is" in namespace["month_prompts"](m)
            for m in namespace["MONTHS"]),
    }


def main():
    alternating = [float((-1) ** d) for d in range(12)]
    old = reconstruct(alternating, False)
    fixed = reconstruct(alternating, True)
    odd = [math.sin(d * 1.37) + d / 10 for d in range(7)]
    odd_error = max_error(odd, reconstruct(odd, False))
    assert max_error(old, [2 * v for v in alternating]) < 1e-12
    assert max_error(fixed, alternating) < 1e-12
    assert odd_error < 1e-12

    # Homogeneity of ideal RMSNorm: rescaling the entire state cannot undo
    # competition between its directional components in the normalization.
    def rmsnorm(v):
        scale = math.sqrt(sum(x * x for x in v) / len(v))
        return [x / scale for x in v]

    rms_error = max_error(rmsnorm([3.0, 4.0]), rmsnorm([6.0, 8.0]))
    assert rms_error < 1e-12

    # Nonorthogonal encoders need not imply coupled latent variables:
    # h = a*(1,0) + b*(cos(theta),sin(theta)); dual decoding recovers a,b.
    theta = math.radians(51.3)
    a, b, change = 1.0, 2.0, 3.0
    h = [a + b * math.cos(theta), b * math.sin(theta)]
    edited = [h[0] + change * math.cos(theta), h[1] + change * math.sin(theta)]
    decoded_b = edited[1] / math.sin(theta)
    decoded_a = edited[0] - decoded_b * math.cos(theta)
    assert abs(decoded_a - a) < 1e-12
    assert abs(decoded_b - (b + change)) < 1e-12

    results = {
        "scope": "Synthetic mathematics and static source checks; no model inference",
        "nyquist_original_max_reconstruction_error": max_error(old, alternating),
        "nyquist_corrected_max_reconstruction_error": max_error(fixed, alternating),
        "odd_n_original_max_reconstruction_error": odd_error,
        "ideal_rmsnorm_global_rescale_difference": rms_error,
        "nonorthogonal_encoding_preserved_first_coordinate": decoded_a,
        "nonorthogonal_encoding_changed_second_coordinate": decoded_b,
        **prompt_overlap(),
    }
    assert results["weekday_test_in_centroid_training"]
    assert results["month_test_in_centroid_training"]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
