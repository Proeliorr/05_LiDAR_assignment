"""Run context: where things live inside the bridge container and how to spell them for the docker daemon."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from lidar_bridge.docker_runner import Mount, host_path

DEFAULT_PROJECT_DIR = Path("/workspace")
_RUN_NAME = re.compile(r"^\w[\w.+=\-]*$")


class PathOutsideProjectError(ValueError):
    """A user-supplied path does not lie inside the project directory (and so cannot be mounted)."""


@dataclass(frozen=True)
class RunContext:
    """Everything a stage needs to know about the current run.

    Paths held in the context are *in-bridge* paths (under ``project_dir``, normally ``/workspace``). Use
    :meth:`host` / :meth:`mount` to translate them for ``docker run``: the daemon resolves bind-mount sources on
    the Windows host, where the project lives at ``host_project_dir`` (env ``HOST_PROJECT_DIR``).

    Layout of a run (``run_dir``)::

        data/out/<run_name>/dfin/       3DFin outputs (+ params.ini)
        data/out/<run_name>/split/      per-tree clouds + manifest
        data/out/<run_name>/smartqsm/   QSM outputs
        data/out/<run_name>/arbor/      arbor outputs
        data/out/<run_name>/compare/    comparison tables / plots
        data/out/<run_name>/.hydra/     hydra config snapshot + log of this invocation
    """

    run_name: str
    host_project_dir: str
    project_dir: Path = DEFAULT_PROJECT_DIR
    input: Path | None = None
    reference: Path | None = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        if not _RUN_NAME.match(self.run_name) or self.run_name.endswith(".."):
            raise ValueError(
                f"invalid run_name {self.run_name!r}: use letters, digits and . _ - + = only (no spaces, slashes or "
                "leading dot); it becomes the folder data/out/<run_name>"
            )
        host_path(self.host_project_dir)  # fails early on an empty / comma-containing HOST_PROJECT_DIR

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
        """Build a context from ``HOST_PROJECT_DIR`` (required) and ``LIDAR_PROJECT_DIR`` (default /workspace)."""
        env = os.environ if env is None else env
        host_dir = env.get("HOST_PROJECT_DIR", "").strip()
        if not host_dir:
            raise RuntimeError(
                "HOST_PROJECT_DIR is not set. It must be the Windows project path with forward slashes, e.g. "
                "D:/Podyplomowe/05_Proximity_Lidar (run_pipeline.bat sets it; in PowerShell: "
                "$env:HOST_PROJECT_DIR='D:/Podyplomowe/05_Proximity_Lidar')."
            )
        project_dir = Path(env.get("LIDAR_PROJECT_DIR", str(DEFAULT_PROJECT_DIR)))
        host_path(host_dir)  # validate early
        ctx = cls(run_name=run_name, host_project_dir=host_dir, project_dir=project_dir, dry_run=dry_run)
        return replace(
            ctx,
            input=ctx.resolve(input) if input else None,
            reference=ctx.resolve(reference) if reference else None,
        )

    # -- directories -------------------------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def run_dir(self) -> Path:
        """``<project>/data/out/<run_name>``."""
        return self.project_dir / "data" / "out" / self.run_name

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

    # -- path translation --------------------------------------------------------------------------------
    def resolve(self, path: str | os.PathLike) -> Path:
        """Absolute in-bridge path for ``path`` (relative paths are relative to ``project_dir``).

        Raises :class:`PathOutsideProjectError` if the result is not inside ``project_dir`` (``..`` is
        collapsed lexically; a Windows drive path like ``D:/x`` is rejected with a hint).
        """
        text = str(path)
        if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
            raise PathOutsideProjectError(
                f"{text!r} is a Windows path, but the bridge runs in a Linux container. Give a path relative to the "
                f"project, e.g. data/in/{PurePosixPath(text.replace(chr(92), '/')).name}"
            )
        p = Path(os.path.normpath(self.project_dir / path))
        try:
            p.relative_to(self.project_dir)
        except ValueError:
            raise PathOutsideProjectError(
                f"{text!r} is outside the project directory {self.project_dir}; only files inside the project "
                "can be mounted into the stage containers. Put the file under data/in/."
            ) from None
        return p

    def rel(self, path: str | os.PathLike) -> PurePosixPath:
        """``path`` relative to ``project_dir`` (POSIX spelling)."""
        return PurePosixPath(self.resolve(path).relative_to(self.project_dir).as_posix())

    def host(self, path: str | os.PathLike) -> str:
        """HOST spelling of an in-bridge path (the thing to pass to ``docker run`` as a bind-mount source).

        ``ctx.host("/workspace/data/in/a.laz")`` -> ``"D:/Podyplomowe/05_Proximity_Lidar/data/in/a.laz"``.
        See :func:`lidar_bridge.docker_runner.host_path` for why this spelling was chosen.
        """
        relative = self.rel(path).as_posix()
        return host_path(self.host_project_dir, "" if relative == "." else relative)

    def mount(self, path: str | os.PathLike, target: str, *, readonly: bool = False, must_exist: bool = True) -> Mount:
        """Bind-mount description for an in-bridge path.

        Docker Desktop silently creates a missing source directory on the Windows side (a ghost folder), so
        existence is verified here, on the bridge side. Skipped in ``dry_run``.
        """
        resolved = self.resolve(path)
        if must_exist and not self.dry_run and not resolved.exists():
            raise FileNotFoundError(f"cannot mount {resolved}: it does not exist")
        return Mount(self.host(resolved), target, readonly=readonly)

    def container_path(self, path: str | os.PathLike, root: str = "/data") -> str:
        """Path of an in-bridge ``path`` inside a container that mounts ``data/`` at ``root``.

        Raises :class:`PathOutsideProjectError` if the path is not under ``data/``.
        """
        resolved = self.resolve(path)
        try:
            rel = resolved.relative_to(self.data_dir)
        except ValueError:
            raise PathOutsideProjectError(f"{resolved} is not under {self.data_dir}") from None
        return root if rel.as_posix() == "." else f"{root.rstrip('/')}/{rel.as_posix()}"
