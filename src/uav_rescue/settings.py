"""项目路径与 TOML 配置读取。"""

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_toml(relative_path: str) -> dict:
    """读取相对于项目根目录的 TOML 配置。"""

    with (PROJECT_ROOT / relative_path).open("rb") as handle:
        return tomllib.load(handle)


def project_path(relative_path: str) -> Path:
    """把配置中的相对路径解析为绝对路径。"""

    return (PROJECT_ROOT / relative_path).resolve()

