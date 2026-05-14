"""本地文档加载工具。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag_mcp.config import Settings
from rag_mcp.schemas import Document


class DocumentLoader:
    """把本地文件或目录读取为 RAG 可索引的文本文档。"""

    def __init__(self, settings: Settings) -> None:
        """初始化加载器。

        Args:
            settings: RAG MCP 的运行配置。
        """

        self.settings = settings

    def load_path(self, path: str, recursive: bool = True) -> list[Document]:
        """读取一个文件或目录。

        Args:
            path: 文件或目录路径。
            recursive: 当 path 是目录时，是否递归读取子目录。

        Returns:
            已读取的文档列表。

        Raises:
            FileNotFoundError: path 不存在。
            ValueError: 文件类型不支持或文件过大。
        """

        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"路径不存在：{root}")

        if root.is_file():
            return [self._load_file(root)]

        return [self._load_file(file_path) for file_path in self._iter_files(root, recursive)]

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
            raise ValueError(f"文件过大：{file_path}，大小 {size} bytes，上限 {self.settings.max_file_size_bytes} bytes。")

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
                "source_path": str(file_path),
            },
        )

    def _read_text(self, file_path: Path) -> str:
        # 兼容中文 Windows 文档：优先 UTF-8，再退回 GB18030。
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
