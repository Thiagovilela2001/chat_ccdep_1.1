"""Empacota e distribui o ChromaDB principal como artefato de GitHub Release."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import chromadb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_DIR = ROOT / "rag_principal" / "chroma_db"
DEFAULT_COLLECTION = "estatisticas"
DEFAULT_RELEASE_REPO = "Thiagovilela2001/chat_ccdep_1.1"
DEFAULT_RELEASE_TAG = "vector-index-v4"
DEFAULT_RELEASE_ASSET = "rag-principal-index-v4.tar.gz"
SCHEMA_VERSION = 1
EMBEDDING_MODEL = "BAAI/bge-m3"
REQUIRED_DB_FILES = {"chroma.sqlite3", "bm25_nodes.pkl", "indexed_manifest.json"}
INSTALLED_MANIFEST = ".index_artifact.json"
VERSION_PACKAGES = (
    "chromadb",
    "llama-index",
    "llama-index-vector-stores-chroma",
)


class ArtifactError(RuntimeError):
    """Artefato ausente, incompatível, corrompido ou inseguro."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in VERSION_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _collection_count(db_dir: Path, collection_name: str) -> int:
    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        return client.get_collection(collection_name).count()
    except Exception as exc:
        raise ArtifactError(
            f"Coleção '{collection_name}' ausente ou ilegível em '{db_dir}'."
        ) from exc
    finally:
        system = getattr(client, "_system", None)
        if system is not None:
            try:
                system.stop()
            except Exception:
                pass
        try:
            from chromadb.api.client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        except (ImportError, AttributeError):
            pass


def _db_files(db_dir: Path) -> list[Path]:
    if not db_dir.is_dir():
        raise ArtifactError(f"Diretório ChromaDB não encontrado: {db_dir}")
    missing = sorted(name for name in REQUIRED_DB_FILES if not (db_dir / name).is_file())
    if missing:
        raise ArtifactError(
            "Banco incompleto; arquivo(s) ausente(s): " + ", ".join(missing)
        )
    files = sorted(path for path in db_dir.rglob("*") if path.is_file())
    if not files:
        raise ArtifactError(f"Diretório ChromaDB vazio: {db_dir}")
    return files


def _file_state(files: list[Path], root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in files
    }


def _manifest(db_dir: Path, collection_name: str) -> tuple[dict, list[Path]]:
    vector_count = _collection_count(db_dir, collection_name)
    files = _db_files(db_dir)
    requirements = ROOT / "requirements.txt"
    records: list[dict[str, str | int]] = []
    for path in files:
        relative = Path("chroma_db") / path.relative_to(db_dir)
        records.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": "principal",
        "collection": collection_name,
        "vector_count": vector_count,
        "embedding_model": EMBEDDING_MODEL,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "packages": _package_versions(),
        "requirements_sha256": _sha256(requirements) if requirements.is_file() else None,
        "git_commit": _git_commit(),
        "files": records,
        "total_bytes": sum(int(record["size"]) for record in records),
    }
    return manifest, files


def build_artifact(
    output: Path,
    *,
    db_dir: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> tuple[Path, Path, dict]:
    """Cria tar.gz verificável. Serviços devem estar parados pelo chamador."""
    output = output.expanduser().resolve()
    db_dir = db_dir.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output == db_dir or db_dir in output.parents:
        raise ArtifactError("Arquivo de saída não pode ficar dentro do ChromaDB.")

    manifest, files = _manifest(db_dir, collection_name)
    before = _file_state(files, db_dir)
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")

    temporary = output.with_name(output.name + ".tmp")
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            info = tarfile.TarInfo("index_artifact.json")
            info.size = len(manifest_bytes)
            info.mtime = int(datetime.now(timezone.utc).timestamp())
            archive.addfile(info, io.BytesIO(manifest_bytes))
            for path in files:
                archive.add(
                    path,
                    arcname=(Path("chroma_db") / path.relative_to(db_dir)).as_posix(),
                    recursive=False,
                )
        after = _file_state(files, db_dir)
        if before != after:
            raise ArtifactError(
                "ChromaDB mudou durante exportação. Pare serviços e tente novamente."
            )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    checksum = _sha256(output)
    sidecar = Path(str(output) + ".sha256")
    sidecar.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    return output, sidecar, manifest


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise ArtifactError(f"Caminho inseguro no artefato: {member.name}")
        if member.issym() or member.islnk():
            raise ArtifactError(f"Link não permitido no artefato: {member.name}")
    archive.extractall(destination, filter="data")


def _load_manifest(extracted: Path) -> dict:
    path = extracted / "index_artifact.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("Manifesto do artefato ausente ou inválido.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError("Versão do formato de artefato não suportada.")
    if manifest.get("engine") != "principal":
        raise ArtifactError("Artefato não pertence à engine Principal.")
    return manifest


def _verify_files(extracted: Path, manifest: dict) -> None:
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ArtifactError("Manifesto não contém arquivos do banco.")
    for record in records:
        if not isinstance(record, dict):
            raise ArtifactError("Registro de arquivo inválido no manifesto.")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise ArtifactError("Caminho de arquivo inválido no manifesto.")
        path = (extracted / relative).resolve()
        if extracted.resolve() not in path.parents or not path.is_file():
            raise ArtifactError(f"Arquivo ausente ou inseguro: {relative}")
        if path.stat().st_size != record.get("size") or _sha256(path) != record.get("sha256"):
            raise ArtifactError(f"Checksum inválido: {relative}")


def _verify_versions(manifest: dict, allow_mismatch: bool) -> None:
    expected_python = manifest.get("python")
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    expected_packages = manifest.get("packages") or {}
    current_packages = _package_versions()
    mismatches = []
    if expected_python != current_python:
        mismatches.append(f"python: {expected_python} != {current_python}")
    for package in VERSION_PACKAGES:
        if expected_packages.get(package) != current_packages.get(package):
            mismatches.append(
                f"{package}: {expected_packages.get(package)} != {current_packages.get(package)}"
            )
    if mismatches and not allow_mismatch:
        raise ArtifactError(
            "Versões incompatíveis: "
            + "; ".join(mismatches)
            + ". Use --allow-version-mismatch somente após teste controlado."
        )


def _verify_archive_checksum(archive_path: Path) -> None:
    sidecar = Path(str(archive_path) + ".sha256")
    if not sidecar.is_file():
        return
    expected = sidecar.read_text(encoding="utf-8").split()[0].lower()
    if expected != _sha256(archive_path):
        raise ArtifactError("Checksum SHA-256 do arquivo compactado não confere.")


def verify_artifact(
    archive_path: Path,
    *,
    allow_version_mismatch: bool = False,
) -> dict:
    """Extrai temporariamente e valida checksum, versões e contagem de vetores."""
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise ArtifactError(f"Artefato não encontrado: {archive_path}")

    _verify_archive_checksum(archive_path)

    with tempfile.TemporaryDirectory(prefix="rag-index-verify-") as temporary:
        extracted = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, extracted)
        manifest = _load_manifest(extracted)
        _verify_files(extracted, manifest)
        _verify_versions(manifest, allow_version_mismatch)
        count = _collection_count(
            extracted / "chroma_db",
            str(manifest.get("collection") or DEFAULT_COLLECTION),
        )
        if count != manifest.get("vector_count"):
            raise ArtifactError(
                f"Contagem vetorial divergente: {count} != {manifest.get('vector_count')}"
            )
        return manifest


def _validate_install_target(target: Path) -> Path:
    target = target.expanduser().resolve()
    forbidden = {Path(target.anchor), Path.home().resolve(), ROOT.resolve()}
    if target in forbidden:
        raise ArtifactError(f"Destino inseguro para instalação: {target}")
    return target


def install_artifact(
    archive_path: Path,
    *,
    target: Path = DEFAULT_DB_DIR,
    backup_root: Path | None = None,
    allow_version_mismatch: bool = False,
) -> tuple[dict, Path | None]:
    """Valida e instala banco com troca recuperável do diretório anterior."""
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise ArtifactError(f"Artefato não encontrado: {archive_path}")
    _verify_archive_checksum(archive_path)
    target = _validate_install_target(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = (
        backup_root.expanduser().resolve()
        if backup_root
        else ROOT / f"chroma_backups_{timestamp}"
    )
    backup_path: Path | None = None

    with tempfile.TemporaryDirectory(
        prefix=".rag-index-install-",
        dir=target.parent,
    ) as temporary:
        extracted = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, extracted)
        manifest = _load_manifest(extracted)
        _verify_files(extracted, manifest)
        _verify_versions(manifest, allow_version_mismatch)
        staged = extracted / "chroma_db"
        count = _collection_count(
            staged,
            str(manifest.get("collection") or DEFAULT_COLLECTION),
        )
        if count != manifest.get("vector_count"):
            raise ArtifactError("Contagem vetorial divergente no banco extraído.")

        if target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        if target.exists():
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_path = backup_root / target.name
            if backup_path.exists():
                raise ArtifactError(f"Destino do backup já existe: {backup_path}")
            shutil.move(str(target), str(backup_path))
        try:
            shutil.move(str(staged), str(target))
        except Exception:
            if backup_path and backup_path.exists() and not target.exists():
                shutil.move(str(backup_path), str(target))
            raise
        (target / INSTALLED_MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return manifest, backup_path


def _release_asset_url(repo: str, tag: str, asset: str) -> str:
    parts = repo.split("/")
    if (
        len(parts) != 2
        or not all(parts)
        or any(part in {".", ".."} for part in parts)
        or any(not all(char.isalnum() or char in "._-" for char in part) for part in parts)
    ):
        raise ArtifactError(f"Repositório GitHub inválido: {repo}")
    return (
        f"https://github.com/{parts[0]}/{parts[1]}/releases/download/"
        f"{quote(tag, safe='')}/{quote(asset, safe='')}"
    )


def _download_file(
    url: str,
    destination: Path,
    *,
    token: str | None = None,
    timeout: float = 600.0,
) -> None:
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "rag-ccdep-index-bootstrap/1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ArtifactError(f"Falha ao baixar índice de '{url}': {exc}") from exc


def _installed_vector_count(
    target: Path,
    collection_name: str = DEFAULT_COLLECTION,
) -> int | None:
    """Retorna contagem do banco pronto; vazio significa que download é necessário."""
    target = target.expanduser().resolve()
    if not target.exists():
        return None
    if not target.is_dir():
        raise ArtifactError(f"Destino do índice não é diretório: {target}")
    if not any(target.iterdir()):
        return None

    try:
        _db_files(target)
    except ArtifactError as exc:
        # Um diretório parcialmente criado (por exemplo, só chroma.sqlite3)
        # deve acionar o bootstrap da Release em vez de bloquear o startup.
        if "arquivo(s) ausente(s)" in str(exc):
            return None
        raise
    count = _collection_count(target, collection_name)
    if count <= 0:
        raise ArtifactError(f"Índice existente está vazio ou inválido: {target}")

    installed_manifest = target / INSTALLED_MANIFEST
    if installed_manifest.is_file():
        try:
            manifest = json.loads(installed_manifest.read_text(encoding="utf-8"))
            expected = int(manifest["vector_count"])
            expected_collection = str(manifest["collection"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ArtifactError("Manifesto do índice instalado está inválido.") from exc
        if expected_collection != collection_name or expected != count:
            raise ArtifactError(
                "Índice instalado diverge do manifesto: "
                f"coleção={collection_name}, vetores={count}, esperado={expected}."
            )
    return count


def ensure_release_index(
    *,
    target: Path,
    repo: str = DEFAULT_RELEASE_REPO,
    tag: str = DEFAULT_RELEASE_TAG,
    asset: str = DEFAULT_RELEASE_ASSET,
    collection_name: str = DEFAULT_COLLECTION,
    token: str | None = None,
    timeout: float = 600.0,
) -> tuple[int, bool]:
    """Usa banco local válido ou baixa, valida e instala uma Release uma única vez."""
    target = target.expanduser().resolve()
    count = _installed_vector_count(target, collection_name)
    if count is not None:
        return count, False

    with tempfile.TemporaryDirectory(prefix="rag-index-bootstrap-") as temporary:
        temporary_path = Path(temporary)
        archive_path = temporary_path / asset
        base_url = _release_asset_url(repo, tag, asset)
        _download_file(base_url, archive_path, token=token, timeout=timeout)
        _download_file(
            base_url + ".sha256",
            Path(str(archive_path) + ".sha256"),
            token=token,
            timeout=timeout,
        )
        manifest, _backup = install_artifact(archive_path, target=target)

    installed_count = _installed_vector_count(target, collection_name)
    if installed_count is None or installed_count != manifest["vector_count"]:
        raise ArtifactError("Índice instalado não passou pela validação final.")
    return installed_count, True


def _require_gh() -> str:
    executable = shutil.which("gh")
    if not executable:
        raise ArtifactError("GitHub CLI (`gh`) não encontrado no PATH.")
    return executable


def publish_release(
    archive_path: Path,
    *,
    repo: str,
    tag: str,
    title: str | None = None,
) -> None:
    gh = _require_gh()
    archive_path = archive_path.expanduser().resolve()
    sidecar = Path(str(archive_path) + ".sha256")
    if not archive_path.is_file() or not sidecar.is_file():
        raise ArtifactError("Artefato ou arquivo .sha256 ausente.")

    view = subprocess.run(
        [gh, "release", "view", tag, "--repo", repo],
        capture_output=True,
        text=True,
    )
    if view.returncode == 0:
        command = [
            gh,
            "release",
            "upload",
            tag,
            str(archive_path),
            str(sidecar),
            "--repo",
            repo,
            "--clobber",
        ]
    else:
        command = [
            gh,
            "release",
            "create",
            tag,
            str(archive_path),
            str(sidecar),
            "--repo",
            repo,
            "--title",
            title or f"Índice vetorial {tag}",
            "--notes",
            "ChromaDB Principal pré-indexado. Instale com scripts/index_artifact.py.",
        ]
    subprocess.run(command, check=True)


def download_release(
    *,
    repo: str,
    asset: str,
    tag: str | None,
    target: Path,
    allow_version_mismatch: bool = False,
) -> tuple[dict, Path | None]:
    gh = _require_gh()
    with tempfile.TemporaryDirectory(prefix="rag-index-download-") as temporary:
        command = [gh, "release", "download"]
        if tag:
            command.append(tag)
        command.extend(
            [
                "--repo",
                repo,
                "--pattern",
                asset,
                "--pattern",
                asset + ".sha256",
                "--dir",
                temporary,
            ]
        )
        subprocess.run(command, check=True)
        archive_path = Path(temporary) / asset
        return install_artifact(
            archive_path,
            target=target,
            allow_version_mismatch=allow_version_mismatch,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exporta e instala o índice vetorial Principal via GitHub Releases."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="Cria artefato tar.gz verificável.")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    export.add_argument("--collection", default=DEFAULT_COLLECTION)
    export.add_argument(
        "--confirm-stopped",
        action="store_true",
        help="Confirma que serviços com acesso ao ChromaDB estão parados.",
    )

    verify = subparsers.add_parser("verify", help="Valida artefato sem instalar.")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--allow-version-mismatch", action="store_true")

    install = subparsers.add_parser("install", help="Instala artefato local.")
    install.add_argument("--archive", type=Path, required=True)
    install.add_argument("--target", type=Path, default=DEFAULT_DB_DIR)
    install.add_argument("--allow-version-mismatch", action="store_true")

    publish = subparsers.add_parser("publish", help="Publica artefato em GitHub Release.")
    publish.add_argument("--archive", type=Path, required=True)
    publish.add_argument("--repo", required=True, help="OWNER/REPO")
    publish.add_argument("--tag", required=True)
    publish.add_argument("--title")

    download = subparsers.add_parser(
        "download",
        help="Baixa GitHub Release e instala banco.",
    )
    download.add_argument("--repo", required=True, help="OWNER/REPO")
    download.add_argument("--asset", required=True)
    download.add_argument("--tag", help="Omitir para usar release mais recente.")
    download.add_argument("--target", type=Path, default=DEFAULT_DB_DIR)
    download.add_argument("--allow-version-mismatch", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "export":
            if not args.confirm_stopped:
                raise ArtifactError(
                    "Pare serviços (`docker compose down`) e repita com --confirm-stopped."
                )
            archive, sidecar, manifest = build_artifact(
                args.output,
                db_dir=args.db_dir,
                collection_name=args.collection,
            )
            print(f"Artefato: {archive}")
            print(f"Checksum: {sidecar}")
            print(f"Vetores: {manifest['vector_count']}")
        elif args.command == "verify":
            manifest = verify_artifact(
                args.archive,
                allow_version_mismatch=args.allow_version_mismatch,
            )
            print(f"Artefato válido: {manifest['vector_count']} vetores")
        elif args.command == "install":
            manifest, backup = install_artifact(
                args.archive,
                target=args.target,
                allow_version_mismatch=args.allow_version_mismatch,
            )
            print(f"Índice instalado: {manifest['vector_count']} vetores")
            if backup:
                print(f"Banco anterior preservado em: {backup}")
            print("Defina RAG_INDEX_READ_ONLY=1 antes de iniciar a aplicação.")
        elif args.command == "publish":
            publish_release(
                args.archive,
                repo=args.repo,
                tag=args.tag,
                title=args.title,
            )
            print(f"Release publicada: {args.repo}@{args.tag}")
        elif args.command == "download":
            manifest, backup = download_release(
                repo=args.repo,
                asset=args.asset,
                tag=args.tag,
                target=args.target,
                allow_version_mismatch=args.allow_version_mismatch,
            )
            print(f"Índice instalado: {manifest['vector_count']} vetores")
            if backup:
                print(f"Banco anterior preservado em: {backup}")
            print("Defina RAG_INDEX_READ_ONLY=1 antes de iniciar a aplicação.")
        return 0
    except (ArtifactError, OSError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
