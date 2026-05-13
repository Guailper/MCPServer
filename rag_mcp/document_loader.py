"""Load text documents from files and directories."""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag_mcp.config import Settings
from rag_mcp.schemas import Document


class DocumentLoader:
    """Read supported local files into normalized text documents."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_path(self, path: str, recursive: bool = True) -> list[Document]:
        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"路径不存在：{root}")

        if root.is_file():
            return [self._load_file(root)]

        files = self._iter_files(root, recursive)
        documents: list[Document] = []
        for file_path in files:
            documents.append(self._load_file(file_path))

        return documents

    def _iter_files(self, root: Path, recursive: bool) -> list[Path]:
        pattern = "**/*" if recursive else "*"
        files = [
            path
            for path in root.glob(pattern)
            if path.is_file() and path.suffix.lower() in self.settings.supported_extensions
        ]
        return sorted(files)

    def _load_file(self, file_path: Path) -> Document:
        extension = file_path.suffix.lower()
        if extension not in self.settings.supported_extensions:
            raise ValueError(f"暂不支持的文件类型：{extension}")

        size = file_path.stat().st_size
        if size > self.settings.max_file_size_bytes:
            raise ValueError(
                f"文件过大：{file_path}，大小 {size} bytes，"
                f"上限 {self.settings.max_file_size_bytes} bytes。"
            )

        content = self._read_text(file_path)
        document_id = self._build_document_id(file_path, content)
        return Document(
            document_id=document_id,
            title=file_path.stem,
            source_path=str(file_path),
            content=content,
            metadata={
                "extension": extension,
                "size_bytes": size,
            },
        )

    def _read_text(self, file_path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return file_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        return file_path.read_text(encoding="utf-8", errors="replace")

    def _build_document_id(self, file_path: Path, content: str) -> str:
        digest = hashlib.sha256()
        digest.update(str(file_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        return digest.hexdigest()[:24]
