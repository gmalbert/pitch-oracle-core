import ast
from pathlib import Path


def test_train_and_save_models_does_not_shadow_numpy_alias():
    source = Path(__file__).parents[1].joinpath("train_models.py").read_text()
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_and_save_models"
    )

    local_numpy_imports = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Import)
        and any(alias.name == "numpy" and alias.asname == "np" for alias in node.names)
    ]
    assert not local_numpy_imports
