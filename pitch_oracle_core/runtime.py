"""Runtime configuration used by migrated scripts and thin league applications."""

from dataclasses import dataclass
from pathlib import Path
import os

from .config import LeagueConfig


@dataclass(frozen=True)
class Runtime:
    league: LeagueConfig
    data_dir: Path
    models_dir: Path

    @classmethod
    def for_league(cls, league: LeagueConfig, root: str | Path = ".") -> "Runtime":
        root = Path(root)
        return cls(league, root / "data_files", root / "models")

    def environment(self) -> dict[str, str]:
        return {
            "PITCH_ORACLE_LEAGUE": self.league.key,
            "PITCH_ORACLE_DIV": self.league.football_data_div,
            "PITCH_ORACLE_ESPN_SLUG": self.league.espn_slug or "",
            "PITCH_ORACLE_CLUBELO_CODE": self.league.clubelo_code or "",
            "PITCH_ORACLE_DATA_DIR": str(self.data_dir),
            "PITCH_ORACLE_MODELS_DIR": str(self.models_dir),
        }

    def apply(self) -> "Runtime":
        os.environ.update(self.environment())
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        return self


def current_runtime(default: LeagueConfig) -> Runtime:
    return Runtime(
        default,
        Path(os.getenv("PITCH_ORACLE_DATA_DIR", "data_files")),
        Path(os.getenv("PITCH_ORACLE_MODELS_DIR", "models")),
    )

