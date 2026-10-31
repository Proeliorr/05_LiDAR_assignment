"""Run context: where the mounted data and the installed tools live inside the container."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

DEFAULT_DATA_DIR = Path("/data")
DEFAULT_INSTALL_DIR = Path("/opt/lidar")
_RUN_NAME = re.compile(r"^\w[\w.+=\-]*$")


class PathOutsideDataError(ValueError):
    """A user-supplied path does not lie inside the mounted data directory."""


@dataclass(frozen=True)
class RunContext:
    """Everything a stage needs to know about the current run.

    ``data_dir`` is the mounted data directory (``/data``); user paths are relative to it. ``install_dir`` holds the pipeline
    code and the upstream tools (``/opt/lidar``).

    Layout of a run (``run_dir``)::

        <data_dir>/out/<run_name>/dfin/       3DFin outputs (+ params.ini)
        <data_dir>/out/<run_name>/split/      per-tree clouds + manifest
        <data_dir>/out/<run_name>/smartqsm/   QSM outputs
        <data_dir>/out/<run_name>/arbor/      arbor outputs
        <data_dir>/out/<run_name>/compare/    comparison tables / plots
        <data_dir>/out/<run_name>/pipeline.log
    """

    run_name: str
    data_dir: Path = DEFAULT_DATA_DIR
    install_dir: Path = DEFAULT_INSTALL_DIR
    input: Path | None = None
    reference: Path | None = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not _RUN_NAME.match(self.run_name) or self.run_name.endswith(".."):
            raise ValueError(
                f"invalid run name {self.run_name!r}: use letters, digits and . _ - + = only (no spaces, slashes or "
                "leading dot); it becomes the folder out/<run name>"
            )

    # -- construction ------------------------------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        run_name: str,
        *,
        input: str | os.PathLike | None = None,
        reference: str | os.PathLike | None = None,
        dry_run: bool = False,
        env: dict | None = None,
    ) -> RunContext:
        """Build a context; ``LIDAR_DATA_DIR`` and ``LIDAR_INSTALL_DIR`` override the default directories."""
        env = os.environ if env is None else env
        data_dir = Path(env.get("LIDAR_DATA_DIR", str(DEFAULT_DATA_DIR)))
        if not data_dir.is_dir():
            raise RuntimeError(f"{data_dir} does not exist: mount your data folder, docker run -v D:\\path\\data:{data_dir} ...")
        ctx = cls(
            run_name=run_name,
            data_dir=data_dir,
            install_dir=Path(env.get("LIDAR_INSTALL_DIR", str(DEFAULT_INSTALL_DIR))),
            dry_run=dry_run,
        )
        return replace(
            ctx,
            input=ctx.resolve(input) if input else None,
            reference=ctx.resolve(reference) if reference else None,
        )

    # -- directories -------------------------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        """``<data_dir>/out/<run_name>``."""
        return self.data_dir / "out" / self.run_name

    def stage_dir(self, stage: str) -> Path:
        """Output directory of a stage (not created)."""
        return self.run_dir / stage

    def reset_stage_dir(self, stage: str) -> Path:
        """Delete everything in the stage directory and recreate it empty. Returns the directory.

        Refuses to touch anything that is not a direct child of ``run_dir``.
        """
        target = self.stage_dir(stage)
        if target.parent != self.run_dir or target.name in ("", ".", ".."):
            raise ValueError(f"refusing to reset {target}: not a stage directory of {self.run_dir}")
        if self.dry_run:
            return target
        if target.exists():
            try:
                shutil.rmtree(target)
            except OSError as exc:
                raise RuntimeError(
                    f"cannot clear stale outputs in {target} ({exc}). Is a file from it open in another program "
                    "(e.g. the .xlsx in Excel)? Close it and retry."
                ) from exc
        target.mkdir(parents=True)
        return target

    # -- paths -------------------------------------------------------------------------------------------
    def resolve(self, path: str | os.PathLike) -> Path:
        """Absolute path for ``path`` (relative paths are relative to ``data_dir``).

        Raises :class:`PathOutsideDataError` if the result is not inside ``data_dir`` (``..`` is collapsed lexically; a
        Windows drive path like ``D:/x`` is rejected with a hint).
        """
        text = str(path)
        if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
            raise PathOutsideDataError(
                f"{text!r} is a Windows path, but the pipeline runs in a Linux container. Give a path relative to the "
                f"mounted data folder, e.g. in/{PurePosixPath(text.replace(chr(92), '/')).name}"
            )
        p = Path(os.path.normpath(self.data_dir / path))
        try:
            p.relative_to(self.data_dir)
        except ValueError:
            raise PathOutsideDataError(
                f"{text!r} is outside the mounted data folder {self.data_dir}. Put the file into that folder."
            ) from None
        return p

    def rel(self, path: str | os.PathLike) -> PurePosixPath:
        """``path`` relative to ``data_dir`` (POSIX spelling)."""
        return PurePosixPath(self.resolve(path).relative_to(self.data_dir).as_posix())
