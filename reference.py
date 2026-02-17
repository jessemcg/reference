#!/usr/bin/python3

from __future__ import annotations

import json
import importlib
import re
import shutil
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # type: ignore

APP_ID = "com.mcglaw.Reference"
APP_NAME = "Reference"
GLib.set_application_name(APP_NAME)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"
ODT_DIR = BASE_DIR / "prior_briefing_odt_files"
DATA_DIR = BASE_DIR / "reference_data"
DB_FILE = DATA_DIR / "briefs.sqlite"
CHROMA_DIR = DATA_DIR / "chroma"
TEXT_VERSION = 2

CONFIG_KEY_RAG_API_URL = "rag_api_url"
CONFIG_KEY_RAG_MODEL_ID = "rag_model_id"
CONFIG_KEY_RAG_API_KEY = "rag_api_key"
CONFIG_KEY_RAG_PROMPT = "rag_prompt"
CONFIG_KEY_RAG_PROMPT_NO_CITATIONS = "rag_prompt_no_citations"
CONFIG_KEY_RAG_PROMPT_FULL_CITATIONS = "rag_prompt_full_citations"
CONFIG_KEY_RAG_PROMPT_STATUTES_ONLY = "rag_prompt_statutes_only"
CONFIG_KEY_RAG_TOP_K = "rag_top_k"
CONFIG_KEY_RAG_PROVIDER = "rag_provider"
CONFIG_KEY_RAG_VOYAGE_API_KEY = "rag_voyage_api_key"
CONFIG_KEY_RAG_VOYAGE_MODEL = "rag_voyage_model"
CONFIG_KEY_RAG_ISAACUS_API_KEY = "rag_isaacus_api_key"
CONFIG_KEY_RAG_ISAACUS_MODEL = "rag_isaacus_model"
CONFIG_KEY_DEEP_ASK_TIMEOUT_SECONDS = "deep_ask_timeout_seconds"
CONFIG_KEY_DEEP_ASK_SHOW_REASONING = "deep_ask_show_reasoning"
CONFIG_KEY_VOYAGE_API_KEY = "voyage_api_key"
CONFIG_KEY_VOYAGE_MODEL = "voyage_model"
CONFIG_KEY_RAG_OUTPUT_FONT_SIZE = "rag_output_font_size"
CONFIG_KEY_SEARCH_OUTPUT_FONT_SIZE = "search_output_font_size"

DEFAULT_RAG_PROMPT = (
    "You are a legal research assistant. Answer the user's question using only the provided context. "
    "Include short direct quotes of two to three words in double quotes. "
    "Cite cases, statutes, and rules exactly as they appear in the context. "
    "If the context is insufficient, say so plainly."
)
DEFAULT_RAG_PROMPT_FULL_CITATIONS = (
    "You are a legal research assistant. Answer the user's question using only the provided context. "
    "Include short direct quotes of two to three words in double quotes. "
    "Cite statutes, rules of court, and published decisions exactly as they appear in the context. "
    "If the context is insufficient, say so plainly."
)
DEFAULT_RAG_PROMPT_STATUTES_ONLY = (
    "You are a legal research assistant. Answer the user's question using only the provided context. "
    "Include short direct quotes of two to three words in double quotes. "
    "Cite only statutes and rules of court exactly as they appear in the context. "
    "Do not cite cases or other authorities. If the context is insufficient, say so plainly."
)
DEFAULT_OUTPUT_FONT_SIZE = 12
DEFAULT_TEXT_COLOR = "alpha(@window_fg_color, 0.68)"
DEFAULT_QUOTED_PHRASE_ALPHA = 1.0
DEFAULT_SEARCH_HIGHLIGHT_COLOR = "#ffff00"
DEFAULT_RAG_LINE_HEIGHT = 1.2
DEFAULT_RAG_TOP_K = 6
RAG_OUTPUT_MIN_HEIGHT = 200
RAG_OUTPUT_MAX_HEIGHT = 480
RAG_OUTPUT_BG_COLOR = "alpha(@window_fg_color, 0.06)"
SEARCH_OUTPUT_BG_COLOR = "alpha(@window_fg_color, 0.06)"
BRIEF_TEXT_BG_COLOR = "#ffffff"
BRIEF_TEXT_FG_COLOR = "#000000"
BRIEF_TEXT_FONT_FAMILY = (
    '"Century Schoolbook", "TeX Gyre Schola", "New Century Schoolbook", '
    '"Century Schoolbook L", "URW Schoolbook L", serif'
)
RAG_PROMPT_NO_CITATIONS = "no_citations"
RAG_PROMPT_FULL_CITATIONS = "full_citations"
RAG_PROMPT_STATUTES_ONLY = "statutes_only"
RAG_PROVIDER_VOYAGE = "voyage"
RAG_PROVIDER_ISAACUS = "isaacus"
DEFAULT_RAG_PROVIDER = RAG_PROVIDER_VOYAGE
DEFAULT_RAG_VOYAGE_MODEL = "voyage-law-2"
DEFAULT_RAG_ISAACUS_MODEL = "kanon-2-embedder"
DEFAULT_STREAM_TIMEOUT_SECONDS = 300
DEFAULT_SHOW_REASONING_TRACE = False

AI_LINK_SPAN_RE = re.compile(r'(?:\"|“)(.+?)(?:\"|”)|\*\*(.+?)\*\*', re.DOTALL)
LINK_TRAILING_PUNCTUATION = ",.;:!?)]"


def _read_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_config(data: dict[str, Any]) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def split_link_phrase(phrase: str) -> tuple[str, str]:
    end = len(phrase)
    while end > 0 and phrase[end - 1] in LINK_TRAILING_PUNCTUATION:
        end -= 1
    core = phrase[:end].rstrip()
    trailing = phrase[end:]
    return core, trailing


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.rstrip()


def _normalize_rag_provider(value: str) -> str:
    provider = (value or "").strip().lower()
    if provider not in {RAG_PROVIDER_VOYAGE, RAG_PROVIDER_ISAACUS}:
        return DEFAULT_RAG_PROVIDER
    return provider


def _extract_embedding_vectors(response: Any) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if not isinstance(embeddings, list):
        raise ValueError("Invalid embeddings response format.")
    vectors: list[list[float]] = []
    for item in embeddings:
        vector = getattr(item, "embedding", None)
        if vector is None and isinstance(item, dict):
            vector = item.get("embedding")
        if not isinstance(vector, list):
            raise ValueError("Missing embedding vector in response.")
        vectors.append(vector)
    return vectors


class IsaacusEmbeddings:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._model,
            texts=texts,
            task="retrieval/document",
        )
        return _extract_embedding_vectors(response)

    def embed_query(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self._model,
            texts=[text],
            task="retrieval/query",
        )
        vectors = _extract_embedding_vectors(response)
        if not vectors:
            raise ValueError("Isaacus returned no embedding vectors.")
        return vectors[0]


def _extract_odt_text(path: Path) -> str:
    if pypandoc is None:
        raise RuntimeError("pypandoc is not installed; run `uv add pypandoc`.")
    return pypandoc.convert_file(
        str(path),
        "plain",
        format="odt",
        extra_args=["--wrap=none"],
    )


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS briefs (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            text TEXT NOT NULL,
            text_version INTEGER NOT NULL DEFAULT 1
        );
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(briefs)").fetchall()}
    if "text_version" not in columns:
        conn.execute("ALTER TABLE briefs ADD COLUMN text_version INTEGER NOT NULL DEFAULT 1;")
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS briefs_fts USING fts5(
            text,
            path UNINDEXED,
            title UNINDEXED,
            content='briefs',
            content_rowid='id'
        );
        """
    )
    conn.commit()


@dataclass
class AiSettings:
    rag_api_url: str
    rag_model_id: str
    rag_api_key: str
    rag_prompt_no_citations: str
    rag_prompt_full_citations: str
    rag_prompt_statutes_only: str
    rag_top_k: int
    rag_provider: str
    voyage_api_key: str
    voyage_model: str
    isaacus_api_key: str
    isaacus_model: str
    deep_ask_timeout_seconds: int
    deep_ask_show_reasoning: bool

    def is_rag_ready(self) -> bool:
        return all(
            value.strip()
            for value in (
                self.rag_api_url,
                self.rag_model_id,
                self.rag_api_key,
                self.rag_prompt_no_citations,
            )
        ) and self.embeddings_ready()

    def voyage_ready(self) -> bool:
        return all(value.strip() for value in (self.voyage_api_key, self.voyage_model))

    def isaacus_ready(self) -> bool:
        return all(value.strip() for value in (self.isaacus_api_key, self.isaacus_model))

    def embeddings_ready(self) -> bool:
        provider = _normalize_rag_provider(self.rag_provider)
        if provider == RAG_PROVIDER_ISAACUS:
            return self.isaacus_ready()
        return self.voyage_ready()

    def embeddings_provider_name(self) -> str:
        provider = _normalize_rag_provider(self.rag_provider)
        if provider == RAG_PROVIDER_ISAACUS:
            return "Isaacus"
        return "Voyage"


def load_ai_settings() -> AiSettings:
    config = _read_config()
    raw_top_k = config.get(CONFIG_KEY_RAG_TOP_K, DEFAULT_RAG_TOP_K)
    try:
        rag_top_k = int(raw_top_k)
    except (TypeError, ValueError):
        rag_top_k = DEFAULT_RAG_TOP_K
    rag_top_k = _clamp_rag_top_k(rag_top_k)
    legacy_prompt = str(config.get(CONFIG_KEY_RAG_PROMPT, DEFAULT_RAG_PROMPT) or DEFAULT_RAG_PROMPT).strip()
    rag_provider = _normalize_rag_provider(str(config.get(CONFIG_KEY_RAG_PROVIDER, DEFAULT_RAG_PROVIDER) or ""))
    voyage_model = str(
        config.get(
            CONFIG_KEY_RAG_VOYAGE_MODEL,
            config.get(CONFIG_KEY_VOYAGE_MODEL, DEFAULT_RAG_VOYAGE_MODEL),
        )
        or DEFAULT_RAG_VOYAGE_MODEL
    ).strip()
    deep_ask_timeout_seconds = _coerce_timeout_seconds(
        config.get(CONFIG_KEY_DEEP_ASK_TIMEOUT_SECONDS),
        DEFAULT_STREAM_TIMEOUT_SECONDS,
    )
    deep_ask_show_reasoning = _coerce_bool_config(
        config.get(CONFIG_KEY_DEEP_ASK_SHOW_REASONING),
        DEFAULT_SHOW_REASONING_TRACE,
    )
    return AiSettings(
        rag_api_url=str(config.get(CONFIG_KEY_RAG_API_URL, "") or "").strip(),
        rag_model_id=str(config.get(CONFIG_KEY_RAG_MODEL_ID, "") or "").strip(),
        rag_api_key=str(config.get(CONFIG_KEY_RAG_API_KEY, "") or "").strip(),
        rag_prompt_no_citations=str(
            config.get(CONFIG_KEY_RAG_PROMPT_NO_CITATIONS, legacy_prompt) or legacy_prompt
        ).strip(),
        rag_prompt_full_citations=str(
            config.get(CONFIG_KEY_RAG_PROMPT_FULL_CITATIONS, DEFAULT_RAG_PROMPT_FULL_CITATIONS)
            or DEFAULT_RAG_PROMPT_FULL_CITATIONS
        ).strip(),
        rag_prompt_statutes_only=str(
            config.get(CONFIG_KEY_RAG_PROMPT_STATUTES_ONLY, DEFAULT_RAG_PROMPT_STATUTES_ONLY)
            or DEFAULT_RAG_PROMPT_STATUTES_ONLY
        ).strip(),
        rag_top_k=rag_top_k,
        rag_provider=rag_provider,
        voyage_api_key=str(
            config.get(
                CONFIG_KEY_RAG_VOYAGE_API_KEY,
                config.get(CONFIG_KEY_VOYAGE_API_KEY, ""),
            )
            or ""
        ).strip(),
        voyage_model=voyage_model or DEFAULT_RAG_VOYAGE_MODEL,
        isaacus_api_key=str(config.get(CONFIG_KEY_RAG_ISAACUS_API_KEY, "") or "").strip(),
        isaacus_model=str(
            config.get(CONFIG_KEY_RAG_ISAACUS_MODEL, DEFAULT_RAG_ISAACUS_MODEL) or DEFAULT_RAG_ISAACUS_MODEL
        ).strip(),
        deep_ask_timeout_seconds=deep_ask_timeout_seconds,
        deep_ask_show_reasoning=deep_ask_show_reasoning,
    )


def save_ai_settings(settings: AiSettings) -> None:
    config = _read_config()
    config[CONFIG_KEY_RAG_API_URL] = settings.rag_api_url
    config[CONFIG_KEY_RAG_MODEL_ID] = settings.rag_model_id
    config[CONFIG_KEY_RAG_API_KEY] = settings.rag_api_key
    config[CONFIG_KEY_RAG_PROMPT_NO_CITATIONS] = settings.rag_prompt_no_citations or DEFAULT_RAG_PROMPT
    config[CONFIG_KEY_RAG_PROMPT_FULL_CITATIONS] = (
        settings.rag_prompt_full_citations or DEFAULT_RAG_PROMPT_FULL_CITATIONS
    )
    config[CONFIG_KEY_RAG_PROMPT_STATUTES_ONLY] = (
        settings.rag_prompt_statutes_only or DEFAULT_RAG_PROMPT_STATUTES_ONLY
    )
    config[CONFIG_KEY_RAG_PROMPT] = settings.rag_prompt_no_citations or DEFAULT_RAG_PROMPT
    config[CONFIG_KEY_RAG_TOP_K] = _clamp_rag_top_k(int(settings.rag_top_k))
    config[CONFIG_KEY_RAG_PROVIDER] = _normalize_rag_provider(settings.rag_provider)
    config[CONFIG_KEY_VOYAGE_API_KEY] = settings.voyage_api_key
    config[CONFIG_KEY_VOYAGE_MODEL] = settings.voyage_model or DEFAULT_RAG_VOYAGE_MODEL
    config[CONFIG_KEY_RAG_VOYAGE_API_KEY] = settings.voyage_api_key
    config[CONFIG_KEY_RAG_VOYAGE_MODEL] = settings.voyage_model or DEFAULT_RAG_VOYAGE_MODEL
    config[CONFIG_KEY_RAG_ISAACUS_API_KEY] = settings.isaacus_api_key
    config[CONFIG_KEY_RAG_ISAACUS_MODEL] = settings.isaacus_model or DEFAULT_RAG_ISAACUS_MODEL
    config[CONFIG_KEY_DEEP_ASK_TIMEOUT_SECONDS] = _coerce_timeout_seconds(
        settings.deep_ask_timeout_seconds,
        DEFAULT_STREAM_TIMEOUT_SECONDS,
    )
    config[CONFIG_KEY_DEEP_ASK_SHOW_REASONING] = bool(settings.deep_ask_show_reasoning)
    _write_config(config)


def _clamp_font_size(value: int) -> int:
    if value < 8:
        return 8
    if value > 32:
        return 32
    return value


def _clamp_rag_top_k(value: int) -> int:
    if value < 1:
        return 1
    if value > 20:
        return 20
    return value


def _coerce_timeout_seconds(value: Any, default: int) -> int:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return default
    return min(3600, max(60, seconds))


def _coerce_bool_config(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def load_ui_settings() -> tuple[int, int]:
    config = _read_config()
    raw_rag_size = config.get(CONFIG_KEY_RAG_OUTPUT_FONT_SIZE, DEFAULT_OUTPUT_FONT_SIZE)
    try:
        rag_size = int(raw_rag_size)
    except (TypeError, ValueError):
        rag_size = DEFAULT_OUTPUT_FONT_SIZE
    rag_size = _clamp_font_size(rag_size)
    raw_search_size = config.get(CONFIG_KEY_SEARCH_OUTPUT_FONT_SIZE, DEFAULT_OUTPUT_FONT_SIZE)
    try:
        search_size = int(raw_search_size)
    except (TypeError, ValueError):
        search_size = DEFAULT_OUTPUT_FONT_SIZE
    search_size = _clamp_font_size(search_size)
    return rag_size, search_size


def save_ui_settings(
    rag_font_size: int,
    search_font_size: int,
) -> None:
    config = _read_config()
    config[CONFIG_KEY_RAG_OUTPUT_FONT_SIZE] = int(rag_font_size)
    config[CONFIG_KEY_SEARCH_OUTPUT_FONT_SIZE] = int(search_font_size)
    _write_config(config)


@dataclass
class AiOutputView:
    view: Gtk.TextView | None = None
    buffer: Gtk.TextBuffer | None = None
    scroller: Gtk.ScrolledWindow | None = None
    link_tags: list[Gtk.TextTag] = field(default_factory=list)
    link_lookup: dict[Gtk.TextTag, str] = field(default_factory=dict)
    motion_controller: Gtk.EventControllerMotion | None = None
    click_gesture: Gtk.GestureClick | None = None
    focus_controller: Gtk.EventControllerFocus | None = None


@dataclass
class SearchResult:
    path: str
    title: str
    snippet: str


class ReferenceApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.FLAGS_NONE,
        )
        try:
            style_manager = Adw.StyleManager.get_default()
            style_manager.set_color_scheme(Adw.ColorScheme.DEFAULT)
            style_manager.connect("notify::color-scheme", self._on_color_scheme_changed)
        except Exception:
            pass
        self._window: ReferenceWindow | None = None
        self.connect("activate", self._on_activate)
        self._settings_window: ReferenceSettingsWindow | None = None

        action = Gio.SimpleAction.new("open-settings", None)
        action.connect("activate", self._on_open_settings)
        self.add_action(action)

        action = Gio.SimpleAction.new("update-index", None)
        action.connect("activate", self._on_update_index)
        self.add_action(action)

        action = Gio.SimpleAction.new("focus-rag", None)
        action.connect("activate", self._on_focus_rag)
        self.add_action(action)

        action = Gio.SimpleAction.new("focus-search", None)
        action.connect("activate", self._on_focus_search)
        self.add_action(action)

        self.set_accels_for_action("app.focus-rag", ["<Primary><Shift>a"])
        self.set_accels_for_action("app.focus-search", ["<Primary><Shift>f"])

    def _on_activate(self, _app: Adw.Application) -> None:
        if self._window is None:
            self._window = ReferenceWindow(self)
        self._window.present()

    def _on_open_settings(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        if self._window is None:
            return
        self._window.open_settings()

    def _on_update_index(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        if self._window is None:
            return
        self._window._on_update_index_clicked(None)

    def _on_focus_rag(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        if self._window is None:
            return
        self._window._focus_rag_entry()

    def _on_focus_search(self, _action: Gio.SimpleAction, _param: GLib.Variant | None) -> None:
        if self._window is None:
            return
        self._window._focus_search_entry()

    def _on_color_scheme_changed(self, *_args: object) -> None:
        if self._window is None:
            return
        self._window._refresh_rag_quote_colors()


class ReferenceWindow(Adw.ApplicationWindow):
    def __init__(self, app: ReferenceApp) -> None:
        super().__init__(application=app)
        self.set_title(APP_NAME)
        self.set_default_size(1100, 900)

        self._ai_settings = load_ai_settings()
        (
            self._rag_output_font_size,
            self._search_output_font_size,
        ) = load_ui_settings()
        self._rag_output_state = AiOutputView()
        self._search_output_state = AiOutputView()
        self._indexing = False
        self._css_provider: Gtk.CssProvider | None = None
        self._last_rag_answer = ""
        self._rag_stream_thread: threading.Thread | None = None
        self._rag_cancel_event: threading.Event | None = None
        self._rag_request_generation = 0
        self._rag_vectorstore: Any | None = None
        self._rag_load_thread: threading.Thread | None = None
        self._rag_load_error: str | None = None
        self._rag_loading = False
        self._rag_load_generation = 0
        self._rag_lock = threading.Lock()

        self._toast_overlay: Adw.ToastOverlay | None = None
        self._status_spinner: Gtk.Spinner | None = None
        self._status_label: Gtk.Label | None = None
        self._rag_entry: Gtk.Entry | None = None
        self._search_entry: Gtk.SearchEntry | None = None
        self._rag_no_citations_button: Gtk.Button | None = None
        self._rag_full_citations_button: Gtk.Button | None = None
        self._rag_statutes_only_button: Gtk.Button | None = None
        self._search_button: Gtk.Button | None = None
        self._search_prev_button: Gtk.Button | None = None
        self._search_next_button: Gtk.Button | None = None
        self._search_nav_label: Gtk.Label | None = None
        self._search_title_label: Gtk.Label | None = None
        self._search_download_button: Gtk.Button | None = None
        self._search_results: list[SearchResult] = []
        self._search_result_index = 0
        self._search_terms: list[str] = []

        self._build_ui()
        self._install_shortcuts()
        self._apply_ui_settings()
        self._kickoff_rag_background_load()

    def _build_ui(self) -> None:
        view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.add_css_class("flat")
        title_label = Gtk.Label(label=APP_NAME, xalign=0)
        title_label.add_css_class("app-title")
        header.set_title_widget(title_label)

        menu_model = Gio.Menu()
        menu_model.append("Settings", "app.open-settings")
        menu_model.append("Update Index", "app.update-index")
        menu_button = Gtk.MenuButton(icon_name="open-menu-symbolic")
        menu_button.set_menu_model(menu_model)
        header.pack_end(menu_button)

        view.add_top_bar(header)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        outer.set_margin_top(18)
        outer.set_margin_bottom(12)
        outer.set_margin_start(18)
        outer.set_margin_end(18)

        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        controls.set_hexpand(True)
        controls.set_valign(Gtk.Align.CENTER)

        rag_entry = Gtk.Entry()
        rag_entry.set_hexpand(True)
        rag_entry.set_icon_from_icon_name(Gtk.EntryIconPosition.PRIMARY, "dialog-question-symbolic")
        rag_entry.set_icon_tooltip_text(Gtk.EntryIconPosition.PRIMARY, "Ask a RAG question")
        rag_entry.connect("activate", self._on_rag_question_activate)
        controls.append(rag_entry)
        self._rag_entry = rag_entry

        rag_buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        no_citations_button = Gtk.Button(label="No Citations")
        no_citations_button.add_css_class("suggested-action")
        no_citations_button.add_css_class("flat")
        no_citations_button.add_css_class("no-bold")
        no_citations_button.connect("clicked", self._on_rag_question_clicked, RAG_PROMPT_NO_CITATIONS)
        rag_buttons.append(no_citations_button)
        self._rag_no_citations_button = no_citations_button

        full_citations_button = Gtk.Button(label="Full Citations")
        full_citations_button.add_css_class("flat")
        full_citations_button.add_css_class("no-bold")
        full_citations_button.connect("clicked", self._on_rag_question_clicked, RAG_PROMPT_FULL_CITATIONS)
        rag_buttons.append(full_citations_button)
        self._rag_full_citations_button = full_citations_button

        statutes_only_button = Gtk.Button(label="Statutes/Rules Only")
        statutes_only_button.add_css_class("flat")
        statutes_only_button.add_css_class("no-bold")
        statutes_only_button.connect("clicked", self._on_rag_question_clicked, RAG_PROMPT_STATUTES_ONLY)
        rag_buttons.append(statutes_only_button)
        self._rag_statutes_only_button = statutes_only_button

        controls.append(rag_buttons)

        outer.append(controls)

        status_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        status_row.set_hexpand(True)
        status_row.set_valign(Gtk.Align.CENTER)
        status_spinner = Gtk.Spinner(spinning=False)
        status_spinner.set_visible(False)
        status_row.append(status_spinner)
        self._status_spinner = status_spinner

        status_label = Gtk.Label(label="", xalign=0)
        status_label.add_css_class("dim-label")
        status_label.set_wrap(True)
        status_label.set_hexpand(True)
        status_label.set_visible(False)
        status_row.append(status_label)
        outer.append(status_row)
        self._status_label = status_label

        rag_scroller, rag_state = self._build_rag_output_view()
        outer.append(rag_scroller)
        self._rag_output_state = rag_state

        search_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        search_controls.set_hexpand(True)
        search_controls.set_valign(Gtk.Align.CENTER)

        search_entry = Gtk.SearchEntry()
        search_entry.set_hexpand(True)
        search_entry.connect("activate", self._on_search_activate)
        search_controls.append(search_entry)
        self._search_entry = search_entry

        search_button = Gtk.Button(label="Search")
        search_button.add_css_class("flat")
        search_button.add_css_class("no-bold")
        search_button.connect("clicked", self._on_search_clicked)
        search_controls.append(search_button)
        self._search_button = search_button

        search_highlighted_button = Gtk.Button(label="Search Highlighted")
        search_highlighted_button.add_css_class("flat")
        search_highlighted_button.add_css_class("no-bold")
        search_highlighted_button.connect("clicked", self._on_search_highlighted_clicked)
        search_controls.append(search_highlighted_button)

        outer.append(search_controls)

        search_nav = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        search_nav.set_hexpand(True)
        search_nav.set_halign(Gtk.Align.FILL)

        prev_button = Gtk.Button(icon_name="go-previous-symbolic")
        prev_button.add_css_class("flat")
        prev_button.set_tooltip_text("Previous brief")
        prev_button.connect("clicked", self._on_search_prev_clicked)
        search_nav.append(prev_button)
        self._search_prev_button = prev_button

        next_button = Gtk.Button(icon_name="go-next-symbolic")
        next_button.add_css_class("flat")
        next_button.set_tooltip_text("Next brief")
        next_button.connect("clicked", self._on_search_next_clicked)
        search_nav.append(next_button)
        self._search_next_button = next_button

        nav_label = Gtk.Label(label="0 hits", xalign=0)
        nav_label.add_css_class("dim-label")
        search_nav.append(nav_label)
        self._search_nav_label = nav_label

        title_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        title_box.set_hexpand(True)
        title_box.set_halign(Gtk.Align.END)
        title_box.set_valign(Gtk.Align.CENTER)

        title_label = Gtk.Label(label="", xalign=0)
        title_label.add_css_class("dim-label")
        title_label.set_wrap(True)
        title_label.set_halign(Gtk.Align.END)
        title_label.set_xalign(1.0)
        title_box.append(title_label)
        self._search_title_label = title_label

        download_button = Gtk.Button(label="Download")
        download_button.add_css_class("flat")
        download_button.add_css_class("no-bold")
        download_button.set_visible(False)
        download_button.connect("clicked", self._on_search_download_clicked)
        title_box.append(download_button)
        self._search_download_button = download_button

        search_nav.append(title_box)

        outer.append(search_nav)

        results_scroller, search_state = self._build_search_output_view()
        outer.append(results_scroller)
        self._search_output_state = search_state

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content.append(outer)

        toast_overlay = Adw.ToastOverlay()
        toast_overlay.set_child(content)
        self._toast_overlay = toast_overlay

        view.set_content(toast_overlay)
        self.set_content(view)

    def _install_shortcuts(self) -> None:
        controller = Gtk.ShortcutController()
        controller.set_scope(Gtk.ShortcutScope.GLOBAL)
        rag_trigger = Gtk.KeyvalTrigger.new(
            Gdk.KEY_a,
            Gdk.ModifierType.CONTROL_MASK | Gdk.ModifierType.SHIFT_MASK,
        )
        action = Gtk.CallbackAction.new(self._focus_rag_entry)
        controller.add_shortcut(Gtk.Shortcut.new(rag_trigger, action))

        search_trigger = Gtk.KeyvalTrigger.new(
            Gdk.KEY_f,
            Gdk.ModifierType.CONTROL_MASK | Gdk.ModifierType.SHIFT_MASK,
        )
        search_action = Gtk.CallbackAction.new(self._focus_search_entry)
        controller.add_shortcut(Gtk.Shortcut.new(search_trigger, search_action))
        self.add_controller(controller)

    def _focus_rag_entry(self, *_args: Any) -> None:
        if self._rag_entry:
            self._rag_entry.grab_focus()

    def _focus_search_entry(self, *_args: Any) -> None:
        if self._search_entry:
            self._search_entry.grab_focus()

    def _build_rag_output_view(self) -> tuple[Gtk.Widget, AiOutputView]:
        output_state = AiOutputView()
        view = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        view.add_css_class("rag-output")
        view.set_hexpand(True)
        view.set_vexpand(True)
        view.set_left_margin(12)
        view.set_right_margin(12)
        view.set_top_margin(12)
        view.set_bottom_margin(12)
        view.set_cursor_visible(False)
        output_state.view = view
        output_state.buffer = view.get_buffer()
        self._install_ai_output_link_controllers(output_state)

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_hexpand(True)
        scroller.set_vexpand(False)
        scroller.set_valign(Gtk.Align.START)
        scroller.set_propagate_natural_height(True)
        scroller.set_min_content_height(RAG_OUTPUT_MIN_HEIGHT)
        scroller.set_max_content_height(RAG_OUTPUT_MAX_HEIGHT)
        scroller.set_child(view)
        output_state.scroller = scroller
        frame = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        frame.set_hexpand(True)
        frame.set_vexpand(False)
        frame.set_valign(Gtk.Align.START)
        frame.add_css_class("rag-output-frame")
        frame.append(scroller)
        return frame, output_state

    def _build_search_output_view(self) -> tuple[Gtk.Widget, AiOutputView]:
        output_state = AiOutputView()
        view = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        view.add_css_class("search-output")
        view.set_hexpand(True)
        view.set_vexpand(True)
        view.set_left_margin(12)
        view.set_right_margin(12)
        view.set_top_margin(12)
        view.set_bottom_margin(12)
        view.set_monospace(False)
        view.set_cursor_visible(False)
        output_state.view = view
        output_state.buffer = view.get_buffer()

        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_min_content_height(240)
        scroller.set_child(view)
        output_state.scroller = scroller
        frame = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        frame.set_hexpand(True)
        frame.set_vexpand(True)
        frame.add_css_class("search-output-frame")
        frame.append(scroller)
        return frame, output_state

    def _show_toast(self, text: str) -> None:
        if not self._toast_overlay:
            return
        toast = Adw.Toast.new(text)
        toast.set_timeout(4)
        self._toast_overlay.add_toast(toast)

    def _set_status(self, text: str, *, spinning: bool = False) -> None:
        if self._status_spinner:
            self._status_spinner.set_spinning(spinning)
            self._status_spinner.set_visible(spinning)
        if self._status_label:
            self._status_label.set_text(text)
            self._status_label.set_visible(bool(text))

    def open_settings(self) -> None:
        app = self.get_application()
        if not isinstance(app, ReferenceApp):
            return
        if app._settings_window is None:
            app._settings_window = ReferenceSettingsWindow(app, self)
        app._settings_window.present()

    def _on_rag_question_activate(self, entry: Gtk.Entry) -> None:
        question = entry.get_text().strip()
        if question:
            entry.set_text("")
        self._ask_rag_question(question, RAG_PROMPT_NO_CITATIONS)

    def _on_rag_question_clicked(self, _button: Gtk.Button, prompt_kind: str) -> None:
        if not self._rag_entry:
            return
        question = self._rag_entry.get_text().strip()
        if question:
            self._rag_entry.set_text("")
        self._ask_rag_question(question, prompt_kind)

    def _resolve_rag_prompt(self, prompt_kind: str) -> str:
        settings = self._ai_settings
        if prompt_kind == RAG_PROMPT_FULL_CITATIONS:
            prompt = settings.rag_prompt_full_citations
            return prompt or DEFAULT_RAG_PROMPT_FULL_CITATIONS
        if prompt_kind == RAG_PROMPT_STATUTES_ONLY:
            prompt = settings.rag_prompt_statutes_only
            return prompt or DEFAULT_RAG_PROMPT_STATUTES_ONLY
        prompt = settings.rag_prompt_no_citations
        return prompt or DEFAULT_RAG_PROMPT

    def _ask_rag_question(self, question: str, prompt_kind: str) -> None:
        if not question:
            return
        if not self._ai_settings.is_rag_ready():
            self._show_toast("Configure RAG and embeddings settings first.")
            return
        if not CHROMA_DIR.exists():
            self._show_toast("No embeddings found. Click Update Index first.")
            return
        self._stop_rag_stream_if_running()
        self._rag_request_generation += 1
        generation = self._rag_request_generation
        self._set_status("", spinning=True)
        self._last_rag_answer = ""
        self._apply_ai_output_links("", self._rag_output_state)
        cancel_event = threading.Event()
        self._rag_cancel_event = cancel_event
        prompt_text = self._resolve_rag_prompt(prompt_kind)
        thread = threading.Thread(
            target=self._rag_worker,
            args=(question, prompt_text, self._ai_settings, cancel_event, generation),
            daemon=True,
        )
        self._rag_stream_thread = thread
        thread.start()

    def _rag_worker(
        self,
        question: str,
        prompt_text: str,
        settings: AiSettings,
        cancel_event: threading.Event | None,
        generation: int,
    ) -> None:
        try:
            messages = self._build_rag_messages(question, settings, prompt_text)
            if not messages:
                GLib.idle_add(self._set_rag_answer, "No relevant context found in the briefing files.")
                GLib.idle_add(self._on_rag_stream_finished, generation)
                return
            self._stream_chat_completion(
                api_url=settings.rag_api_url,
                api_key=settings.rag_api_key,
                model_id=settings.rag_model_id,
                messages=messages,
                cancel_event=cancel_event,
                generation=generation,
                include_reasoning=settings.deep_ask_show_reasoning,
                request_timeout_seconds=settings.deep_ask_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            GLib.idle_add(self._on_rag_stream_error, str(exc), generation)

    def _set_rag_answer(self, text: str) -> None:
        self._set_status("")
        self._last_rag_answer = text
        self._apply_ai_output_links(text, self._rag_output_state)
        self._scroll_rag_output_to_bottom()

    def _stop_rag_stream_if_running(self) -> None:
        if self._rag_cancel_event:
            self._rag_cancel_event.set()
        if self._rag_stream_thread and self._rag_stream_thread.is_alive():
            try:
                self._rag_stream_thread.join(timeout=0.2)
            except Exception:
                pass
        self._rag_stream_thread = None
        self._rag_cancel_event = None

    def _append_rag_output(self, text: str, generation: int) -> bool:
        if generation != self._rag_request_generation:
            return False
        if not text:
            return False
        new_text = self._last_rag_answer + text
        self._last_rag_answer = new_text
        self._apply_ai_output_links(new_text, self._rag_output_state)
        self._scroll_rag_output_to_bottom()
        self._set_status("", spinning=True)
        return False

    def _on_rag_stream_finished(self, generation: int) -> bool:
        if generation != self._rag_request_generation:
            return False
        self._rag_stream_thread = None
        self._rag_cancel_event = None
        self._set_status("", spinning=False)
        return False

    def _on_rag_stream_error(self, message: str, generation: int) -> bool:
        if generation != self._rag_request_generation:
            return False
        self._rag_stream_thread = None
        self._rag_cancel_event = None
        self._set_status("RAG failed.", spinning=False)
        self._show_toast(message or "RAG request failed.")
        return False

    def _on_rag_stream_cancelled(self, generation: int) -> bool:
        if generation != self._rag_request_generation:
            return False
        self._rag_stream_thread = None
        self._rag_cancel_event = None
        self._set_status("", spinning=False)
        return False

    def _build_rag_messages(
        self,
        question: str,
        settings: AiSettings,
        prompt_text: str,
    ) -> list[dict[str, str]] | None:
        vectorstore, error = self._ensure_rag_vectorstore_ready(settings)
        if error or vectorstore is None:
            raise RuntimeError(error or "RAG data unavailable.")
        docs = vectorstore.similarity_search(question, k=_clamp_rag_top_k(settings.rag_top_k))
        if not docs:
            return None
        context_blocks = []
        for doc in docs:
            title = doc.metadata.get("title") or doc.metadata.get("source") or "Brief"
            context_blocks.append(f"{title}\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_blocks)

        prompt = prompt_text or DEFAULT_RAG_PROMPT
        if "{context}" in prompt or "{question}" in prompt:
            system_prompt = prompt.format(context=context, question=question)
            messages = [{"role": "system", "content": system_prompt}]
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
            ]
        return messages

    def _kickoff_rag_background_load(self) -> None:
        settings = self._ai_settings
        if not settings.embeddings_ready():
            with self._rag_lock:
                self._rag_vectorstore = None
                self._rag_load_error = f"{settings.embeddings_provider_name()} settings missing."
                self._rag_loading = False
                self._rag_load_thread = None
            return
        if not CHROMA_DIR.exists():
            with self._rag_lock:
                self._rag_vectorstore = None
                self._rag_load_error = "No embeddings found. Click Update Index first."
                self._rag_loading = False
                self._rag_load_thread = None
            return
        self._rag_load_generation += 1
        generation = self._rag_load_generation
        with self._rag_lock:
            self._rag_vectorstore = None
            self._rag_load_error = None
            self._rag_loading = True

        def worker() -> None:
            vectorstore, error = self._load_rag_vectorstore(settings)
            GLib.idle_add(self._on_rag_resources_loaded, generation, vectorstore, error)

        self._rag_load_thread = threading.Thread(target=worker, daemon=True)
        self._rag_load_thread.start()

    def _on_rag_resources_loaded(
        self,
        generation: int,
        vectorstore: Any | None,
        error: str | None,
    ) -> bool:
        if generation != self._rag_load_generation:
            return False
        with self._rag_lock:
            if error:
                self._rag_vectorstore = None
                self._rag_load_error = error
            else:
                self._rag_vectorstore = vectorstore
                self._rag_load_error = None
            self._rag_loading = False
            self._rag_load_thread = None
        return False

    def _ensure_rag_vectorstore_ready(self, settings: AiSettings) -> tuple[Any | None, str | None]:
        thread = self._rag_load_thread
        if thread and thread.is_alive():
            thread.join()
        with self._rag_lock:
            if self._rag_vectorstore is not None:
                return self._rag_vectorstore, None
            if self._rag_load_error:
                return None, self._rag_load_error
        vectorstore, error = self._load_rag_vectorstore(settings)
        with self._rag_lock:
            self._rag_vectorstore = vectorstore
            self._rag_load_error = error
            self._rag_loading = False
            self._rag_load_thread = None
        return vectorstore, error

    def _load_rag_vectorstore(self, settings: AiSettings) -> tuple[Any | None, str | None]:
        provider = _normalize_rag_provider(settings.rag_provider)
        if not settings.embeddings_ready():
            return None, f"{settings.embeddings_provider_name()} settings missing."
        if not CHROMA_DIR.exists():
            return None, "No embeddings found. Click Update Index first."
        try:
            from langchain_chroma import Chroma  # type: ignore
        except ImportError:
            return None, "Install langchain and langchain-chroma to enable RAG questions."
        try:
            if provider == RAG_PROVIDER_ISAACUS:
                try:
                    isaacus_module = importlib.import_module("isaacus")
                    isaacus_client_class = getattr(isaacus_module, "Isaacus")
                except Exception:
                    return None, "Install Isaacus SDK to enable Isaacus RAG embeddings."
                isaacus_client = isaacus_client_class(api_key=settings.isaacus_api_key)
                embeddings = IsaacusEmbeddings(
                    client=isaacus_client,
                    model=settings.isaacus_model,
                )
            else:
                try:
                    from langchain_voyageai import VoyageAIEmbeddings  # type: ignore
                except ImportError:
                    return None, "Install langchain-voyageai and voyageai to enable Voyage RAG embeddings."
                embeddings = VoyageAIEmbeddings(
                    voyage_api_key=settings.voyage_api_key,
                    model=settings.voyage_model,
                )
            vectorstore = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"Failed to load RAG embeddings: {exc}"
        return vectorstore, None

    def _stream_chat_completion(
        self,
        *,
        api_url: str,
        api_key: str,
        model_id: str,
        messages: list[dict[str, str]],
        cancel_event: threading.Event | None,
        generation: int,
        include_reasoning: bool = False,
        request_timeout_seconds: int = DEFAULT_STREAM_TIMEOUT_SECONDS,
    ) -> None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Reference/1.0",
        }
        body = {
            "model": model_id,
            "messages": messages,
            "stream": True,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
        try:
            timeout_seconds = _coerce_timeout_seconds(
                request_timeout_seconds,
                DEFAULT_STREAM_TIMEOUT_SECONDS,
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                for chunk in self._iter_sse_chunks(resp, cancel_event, include_reasoning=include_reasoning):
                    if cancel_event and cancel_event.is_set():
                        GLib.idle_add(self._on_rag_stream_cancelled, generation)
                        return
                    GLib.idle_add(self._append_rag_output, chunk, generation)
            if cancel_event and cancel_event.is_set():
                GLib.idle_add(self._on_rag_stream_cancelled, generation)
            else:
                GLib.idle_add(self._on_rag_stream_finished, generation)
        except urllib.error.HTTPError as exc:
            GLib.idle_add(
                self._on_rag_stream_error,
                f"HTTP error {exc.code}: {exc.reason or 'request failed'}",
                generation,
            )
        except Exception as exc:  # noqa: BLE001
            GLib.idle_add(self._on_rag_stream_error, str(exc), generation)

    def _iter_sse_chunks(
        self,
        resp: Any,
        cancel_event: threading.Event | None,
        *,
        include_reasoning: bool = False,
    ) -> Iterable[str]:
        in_reasoning_trace = False
        while True:
            if cancel_event and cancel_event.is_set():
                break
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].lstrip()
            if data == "[DONE]":
                break
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            answer_text, reasoning_text = self._extract_stream_text_parts(payload)
            if include_reasoning and reasoning_text:
                if not in_reasoning_trace:
                    in_reasoning_trace = True
                    yield "\n[Reasoning Trace]\n"
                yield reasoning_text
            if answer_text:
                if include_reasoning and in_reasoning_trace:
                    in_reasoning_trace = False
                    yield "\n[Answer]\n"
                yield answer_text

    def _extract_stream_text_parts(self, payload: Any) -> tuple[str, str]:
        answer_text = ""
        reasoning_text = ""
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if isinstance(choices, list) and choices:
            first = choices[0] or {}
            delta = first.get("delta") or first.get("message") or first
            if isinstance(delta, dict):
                answer_text = self._coerce_stream_text(
                    delta.get("content") if "content" in delta else delta.get("text")
                )
                reasoning_text = self._coerce_stream_text(
                    delta.get("reasoning_content")
                    if "reasoning_content" in delta
                    else delta.get("reasoning")
                    if "reasoning" in delta
                    else delta.get("thinking")
                )
        if isinstance(payload, dict):
            fallback = payload.get("data") or payload.get("text")
            if isinstance(fallback, str):
                answer_text = answer_text or fallback
        return answer_text, reasoning_text

    def _coerce_stream_text(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if not isinstance(value, list):
            return ""
        merged: list[str] = []
        for item in value:
            if isinstance(item, dict):
                candidate = item.get("text")
                if isinstance(candidate, str):
                    merged.append(candidate)
            elif isinstance(item, str):
                merged.append(item)
        return "".join(merged)

    def _scroll_rag_output_to_bottom(self) -> None:
        scroller = self._rag_output_state.scroller
        if scroller is None:
            return
        vadj = scroller.get_vadjustment()
        if vadj is None:
            return
        lower = vadj.get_lower()
        upper = vadj.get_upper()
        page_size = vadj.get_page_size()
        vadj.set_value(max(lower, upper - page_size))

    def _on_search_activate(self, entry: Gtk.SearchEntry) -> None:
        query = entry.get_text().strip()
        if query:
            entry.set_text("")
        self._run_search(query, from_link=False)

    def _on_search_clicked(self, _button: Gtk.Button) -> None:
        if not self._search_entry:
            return
        query = self._search_entry.get_text().strip()
        if query:
            self._search_entry.set_text("")
        self._run_search(query, from_link=False)

    def _on_search_highlighted_clicked(self, _button: Gtk.Button) -> None:
        if not self._search_entry:
            return
        buffer = self._rag_output_state.buffer
        if not buffer or not buffer.get_has_selection():
            self._show_toast("Highlight text in the RAG answer first.")
            return
        start, end = buffer.get_selection_bounds()
        selection = buffer.get_text(start, end, True).strip()
        if not selection:
            self._show_toast("Highlight text in the RAG answer first.")
            return
        self._search_entry.set_text(selection)
        self._run_search(selection, from_link=True)

    def _activate_ai_link(self, phrase: str) -> None:
        if not self._search_entry:
            return
        self._search_entry.set_text(phrase)
        self._run_search(phrase, from_link=True)

    def _run_search(self, query: str, *, from_link: bool) -> None:
        if not query:
            return
        if not DB_FILE.exists():
            self._show_toast("No search index found. Click Update Index first.")
            return
        phrase = self._normalize_search_phrase(query)
        fts_query = f"\"{phrase}\"" if phrase else ""
        try:
            results = self._search_database(fts_query)
        except sqlite3.Error as exc:
            self._show_toast(f"Search failed: {exc}")
            return
        self._search_results = results
        self._search_result_index = 0
        self._search_terms = self._extract_search_terms(phrase)
        self._update_search_nav()
        self._show_search_result_at_index(self._search_result_index)

    def _search_database(self, query: str) -> list[SearchResult]:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        with conn:
            _ensure_db(conn)
            rows = conn.execute(
                """
                SELECT briefs.path, briefs.title,
                       snippet(briefs_fts, 0, '[', ']', ' ... ', 40) AS snippet
                FROM briefs_fts
                JOIN briefs ON briefs_fts.rowid = briefs.id
                WHERE briefs_fts MATCH ?
                ORDER BY briefs.mtime DESC
                LIMIT 50;
                """,
                (query,),
            ).fetchall()
        conn.close()
        return [
            SearchResult(
                path=row["path"],
                title=row["title"],
                snippet=row["snippet"] or "",
            )
            for row in rows
        ]

    def _normalize_search_phrase(self, query: str) -> str:
        phrase = query.strip()
        if phrase.startswith('"') and phrase.endswith('"') and len(phrase) > 1:
            phrase = phrase[1:-1].strip()
        return phrase

    def _extract_search_terms(self, phrase: str) -> list[str]:
        cleaned = phrase.strip()
        return [cleaned] if cleaned else []

    def _update_search_nav(self) -> None:
        total = len(self._search_results)
        if self._search_nav_label:
            if total == 0:
                self._search_nav_label.set_text("0 hits")
            else:
                current = self._search_result_index + 1
                self._search_nav_label.set_text(f"{current} of {total} hits")
        if self._search_prev_button:
            self._search_prev_button.set_sensitive(total > 1)
        if self._search_next_button:
            self._search_next_button.set_sensitive(total > 1)

    def _show_search_result_at_index(self, index: int) -> None:
        state = self._search_output_state
        if not state.buffer or not state.view:
            return
        if not self._search_results:
            state.buffer.set_text("No matches found.")
            if self._search_title_label:
                self._search_title_label.set_text("")
            self._update_search_download_button_state()
            return
        if index < 0 or index >= len(self._search_results):
            return
        result = self._search_results[index]
        text = self._fetch_brief_text(result.path)
        if text is None:
            state.buffer.set_text("Brief not found in the index.")
            if self._search_title_label:
                self._search_title_label.set_text(result.title)
            self._update_search_download_button_state()
            return
        state.buffer.set_text(text)
        if self._search_title_label:
            self._search_title_label.set_text(result.title)
        self._update_search_download_button_state()
        first_match = self._highlight_search_terms(state.buffer, text, self._search_terms)
        if first_match is not None:
            GLib.idle_add(self._scroll_search_view_to_offset, state.view, state.buffer, first_match)

    def _highlight_search_terms(
        self,
        buffer: Gtk.TextBuffer,
        text: str,
        terms: list[str],
    ) -> int | None:
        table = buffer.get_tag_table()
        tag = table.lookup("search-highlight") if table else None
        if tag is None:
            tag = buffer.create_tag(
                "search-highlight",
                background=DEFAULT_SEARCH_HIGHLIGHT_COLOR,
            )
        else:
            tag.set_property("background", DEFAULT_SEARCH_HIGHLIGHT_COLOR)
            tag.set_property("foreground-set", False)
        start, end = buffer.get_bounds()
        buffer.remove_tag(tag, start, end)
        first_match: int | None = None
        for term in terms:
            if not term:
                continue
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            for match in pattern.finditer(text):
                start_iter = buffer.get_iter_at_offset(match.start())
                end_iter = buffer.get_iter_at_offset(match.end())
                buffer.apply_tag(tag, start_iter, end_iter)
                if first_match is None or match.start() < first_match:
                    first_match = match.start()
        return first_match

    def _scroll_search_view_to_offset(
        self,
        view: Gtk.TextView,
        buffer: Gtk.TextBuffer,
        offset: int,
    ) -> None:
        target_iter = buffer.get_iter_at_offset(offset)
        view.scroll_to_iter(target_iter, 0.1, True, 0.5, 0.5)

    def _on_search_prev_clicked(self, _button: Gtk.Button) -> None:
        if not self._search_results:
            return
        self._search_result_index = (self._search_result_index - 1) % len(self._search_results)
        self._update_search_nav()
        self._show_search_result_at_index(self._search_result_index)

    def _on_search_next_clicked(self, _button: Gtk.Button) -> None:
        if not self._search_results:
            return
        self._search_result_index = (self._search_result_index + 1) % len(self._search_results)
        self._update_search_nav()
        self._show_search_result_at_index(self._search_result_index)

    def _on_search_download_clicked(self, _button: Gtk.Button) -> None:
        if not self._search_results:
            self._show_toast("No search results to download.")
            return
        result = self._search_results[self._search_result_index]
        odt_path = Path(result.path)
        if not odt_path.exists():
            self._show_toast("Brief file not found on disk.")
            return
        dialog = Gtk.FileDialog(title="Save ODT")
        dialog.set_initial_name(odt_path.name)
        dialog.save(self, None, self._on_search_download_save_ready, odt_path)

    def _on_search_download_save_ready(
        self,
        dialog: Gtk.FileDialog,
        result: Gio.AsyncResult,
        odt_path: Path,
    ) -> None:
        try:
            dest_file = dialog.save_finish(result)
        except GLib.Error:
            return
        if dest_file is None:
            return
        dest_path_raw = dest_file.get_path()
        if not dest_path_raw:
            self._show_toast("Please choose a local file destination.")
            return
        dest_path = Path(dest_path_raw)
        self._download_odt_in_background(odt_path, dest_path)

    def _download_odt_in_background(self, odt_path: Path, dest_path: Path) -> None:
        self._set_status("Saving ODT…")
        if self._search_download_button:
            self._search_download_button.set_sensitive(False)
        thread = threading.Thread(
            target=self._download_odt_worker,
            args=(odt_path, dest_path),
            daemon=True,
        )
        thread.start()

    def _download_odt_worker(self, odt_path: Path, dest_path: Path) -> None:
        try:
            self._copy_odt(odt_path, dest_path)
        except Exception as exc:  # noqa: BLE001
            GLib.idle_add(self._show_toast, f"ODT download failed: {exc}")
        else:
            GLib.idle_add(self._show_toast, f"Saved ODT to {dest_path.name}.")
        finally:
            GLib.idle_add(self._set_status, "")
            GLib.idle_add(self._update_search_download_button_state)

    def _copy_odt(self, odt_path: Path, dest_path: Path) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(odt_path, dest_path)

    def _update_search_download_button_state(self) -> None:
        if not self._search_download_button:
            return
        has_result = bool(self._search_results)
        self._search_download_button.set_visible(has_result)
        self._search_download_button.set_sensitive(has_result)

    def _on_open_brief_clicked(self, _button: Gtk.Button, path: str) -> None:
        text = self._fetch_brief_text(path)
        if text is None:
            self._show_toast("Brief not found in the index.")
            return
        window = Adw.Window(transient_for=self, application=self.get_application())
        window.set_title(Path(path).name)
        window.set_default_size(900, 700)

        view = Gtk.TextView(editable=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        view.add_css_class("brief-output")
        view.set_monospace(False)
        view.set_left_margin(12)
        view.set_right_margin(12)
        view.set_top_margin(12)
        view.set_bottom_margin(12)
        buffer = view.get_buffer()
        buffer.set_text(text)
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.add_css_class("brief-output-frame")
        scroller.set_child(view)

        window.set_content(scroller)
        window.present()

    def _fetch_brief_text(self, path: str) -> str | None:
        conn = sqlite3.connect(DB_FILE)
        row = conn.execute("SELECT text FROM briefs WHERE path = ?", (path,)).fetchone()
        conn.close()
        if row:
            return row[0]
        return None

    def _on_update_index_clicked(self, _button: Gtk.Button | None) -> None:
        if self._indexing:
            self._show_toast("Index update already running.")
            return
        self._indexing = True
        self._set_status("Updating index…")
        thread = threading.Thread(target=self._update_index_worker, daemon=True)
        thread.start()

    def _update_index_worker(self) -> None:
        started = time.time()
        try:
            summary = self._update_index(self._ai_settings)
            elapsed = time.time() - started
            GLib.idle_add(self._on_index_complete, summary, elapsed)
        except Exception as exc:  # noqa: BLE001
            GLib.idle_add(self._on_index_error, str(exc))

    def _on_index_complete(self, summary: str, elapsed: float) -> None:
        self._indexing = False
        self._set_status(summary)
        self._show_toast(f"Index updated in {elapsed:.1f}s.")
        self._kickoff_rag_background_load()

    def _on_index_error(self, message: str) -> None:
        self._indexing = False
        self._set_status("Index update failed.")
        self._show_toast(message or "Index update failed.")

    def _update_index(self, settings: AiSettings) -> str:
        _ensure_data_dir()
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        _ensure_db(conn)

        files = sorted(ODT_DIR.glob("*.odt"))
        current_paths = {str(path) for path in files}
        existing = {
            row["path"]: row
            for row in conn.execute(
                "SELECT id, path, mtime, size, text_version FROM briefs"
            ).fetchall()
        }

        changed_paths: list[Path] = []
        for path in files:
            stat = path.stat()
            row = existing.get(str(path))
            if (
                row
                and row["mtime"] == stat.st_mtime
                and row["size"] == stat.st_size
                and row["text_version"] == TEXT_VERSION
            ):
                continue
            text = _normalize_text(_extract_odt_text(path))
            title = path.stem
            if row:
                conn.execute(
                    "UPDATE briefs SET title = ?, mtime = ?, size = ?, text = ?, text_version = ? WHERE path = ?",
                    (title, stat.st_mtime, stat.st_size, text, TEXT_VERSION, str(path)),
                )
                rowid = conn.execute("SELECT id FROM briefs WHERE path = ?", (str(path),)).fetchone()[0]
                conn.execute("DELETE FROM briefs_fts WHERE rowid = ?", (rowid,))
                conn.execute(
                    "INSERT INTO briefs_fts(rowid, text, path, title) VALUES (?, ?, ?, ?)",
                    (rowid, text, str(path), title),
                )
            else:
                cursor = conn.execute(
                    "INSERT INTO briefs(path, title, mtime, size, text, text_version) VALUES (?, ?, ?, ?, ?, ?)",
                    (str(path), title, stat.st_mtime, stat.st_size, text, TEXT_VERSION),
                )
                rowid = cursor.lastrowid
                conn.execute(
                    "INSERT INTO briefs_fts(rowid, text, path, title) VALUES (?, ?, ?, ?)",
                    (rowid, text, str(path), title),
                )
            changed_paths.append(path)

        removed_paths = [path for path in existing.keys() if path not in current_paths]
        for path in removed_paths:
            rowid = conn.execute("SELECT id FROM briefs WHERE path = ?", (path,)).fetchone()
            if rowid:
                conn.execute("DELETE FROM briefs_fts WHERE rowid = ?", (rowid[0],))
            conn.execute("DELETE FROM briefs WHERE path = ?", (path,))

        conn.commit()
        conn.close()

        if settings.embeddings_ready():
            self._update_embeddings(settings, changed_paths, removed_paths)
        elif changed_paths or removed_paths:
            GLib.idle_add(
                self._show_toast,
                f"{settings.embeddings_provider_name()} settings missing; embeddings not updated.",
            )

        return (
            f"Indexed {len(files)} briefs. Updated {len(changed_paths)}, removed {len(removed_paths)}."
        )

    def _update_embeddings(
        self,
        settings: AiSettings,
        changed_paths: list[Path],
        removed_paths: list[str],
    ) -> None:
        from langchain_chroma import Chroma  # type: ignore
        from langchain_core.documents import Document  # type: ignore
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
        provider = _normalize_rag_provider(settings.rag_provider)

        if provider == RAG_PROVIDER_ISAACUS:
            isaacus_module = importlib.import_module("isaacus")
            isaacus_client_class = getattr(isaacus_module, "Isaacus")
            isaacus_client = isaacus_client_class(api_key=settings.isaacus_api_key)
            embeddings: Any = IsaacusEmbeddings(
                client=isaacus_client,
                model=settings.isaacus_model,
            )
        else:
            from langchain_voyageai import VoyageAIEmbeddings  # type: ignore

            embeddings = VoyageAIEmbeddings(
                voyage_api_key=settings.voyage_api_key,
                model=settings.voyage_model,
            )
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )

        for path in removed_paths:
            vectorstore.delete(where={"source": path})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "],
        )

        for path in changed_paths:
            vectorstore.delete(where={"source": str(path)})
            text = _normalize_text(_extract_odt_text(path))
            chunks = splitter.split_text(text)
            docs = []
            for idx, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": str(path),
                            "title": path.stem,
                            "chunk": idx,
                        },
                    )
                )
            if docs:
                vectorstore.add_documents(docs)

        if hasattr(vectorstore, "persist"):
            vectorstore.persist()

    def _resolve_rag_quote_color(self, view: Gtk.TextView | None) -> Gdk.RGBA:
        fallback = Gdk.RGBA()
        fallback.parse("#ffffff")
        if not view:
            return fallback
        if hasattr(view, "get_color"):
            base = view.get_color()
        else:
            context = view.get_style_context()
            try:
                base = context.get_color()
            except TypeError:
                base = context.get_color(Gtk.StateFlags.NORMAL)
        quote = Gdk.RGBA()
        quote.red = base.red
        quote.green = base.green
        quote.blue = base.blue
        quote.alpha = DEFAULT_QUOTED_PHRASE_ALPHA
        return quote

    def _apply_link_spans(
        self,
        text: str,
        buffer: Gtk.TextBuffer | None,
        link_tags: list[Gtk.TextTag],
        link_lookup: dict[Gtk.TextTag, str],
        scroller: Gtk.ScrolledWindow | None,
        view: Gtk.TextView | None,
    ) -> None:
        if not buffer:
            return
        table = buffer.get_tag_table()
        if table is None:
            return
        for tag in link_tags:
            try:
                table.remove(tag)
            except TypeError:
                pass
        link_tags.clear()
        link_lookup.clear()

        rendered_text, spans = self._extract_ai_link_spans(text)
        buffer.set_text(rendered_text)

        quote_color = self._resolve_rag_quote_color(view)
        for start, end, phrase in spans:
            if end <= start:
                continue
            start_iter = buffer.get_iter_at_offset(start)
            end_iter = buffer.get_iter_at_offset(end)
            tag = buffer.create_tag(
                None,
                foreground_rgba=quote_color,
                underline=Pango.Underline.NONE,
            )
            link_lookup[tag] = phrase
            buffer.apply_tag(tag, start_iter, end_iter)
            link_tags.append(tag)
        if scroller:
            scroller.queue_resize()

    def _contrast_text_color(self, color: str) -> str:
        match = re.fullmatch(r"#([0-9a-fA-F]{6})", color.strip())
        if not match:
            return "#1a1a1a"
        value = match.group(1)
        r = int(value[0:2], 16) / 255.0
        g = int(value[2:4], 16) / 255.0
        b = int(value[4:6], 16) / 255.0
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#ffffff" if luminance < 0.5 else "#1a1a1a"

    def _apply_ui_settings(self, *, refresh_content: bool = False) -> None:
        if self._css_provider is None:
            self._css_provider = Gtk.CssProvider()
            display = Gdk.Display.get_default()
            if display:
                Gtk.StyleContext.add_provider_for_display(
                    display,
                    self._css_provider,
                    Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
                )
        css = (
            "textview.rag-output, textview.search-output {"
            " }"
            ".app-title { font-weight: 700; }"
            "button.no-bold { font-weight: normal; }"
            "textview.rag-output { "
            f"color: {DEFAULT_TEXT_COLOR};"
            "background: transparent;"
            "}"
            "textview.search-output { "
            f"color: {BRIEF_TEXT_FG_COLOR};"
            f"background-color: {BRIEF_TEXT_BG_COLOR};"
            f"font-family: {BRIEF_TEXT_FONT_FAMILY};"
            "}"
            "textview.search-output text { "
            f"color: {BRIEF_TEXT_FG_COLOR};"
            f"background-color: {BRIEF_TEXT_BG_COLOR};"
            "}"
            "textview.brief-output { "
            f"color: {BRIEF_TEXT_FG_COLOR};"
            f"background-color: {BRIEF_TEXT_BG_COLOR};"
            f"font-family: {BRIEF_TEXT_FONT_FAMILY};"
            "}"
            "textview.brief-output text { "
            f"color: {BRIEF_TEXT_FG_COLOR};"
            f"background-color: {BRIEF_TEXT_BG_COLOR};"
            "}"
            ".rag-output-frame { "
            f"background-color: {RAG_OUTPUT_BG_COLOR};"
            "border-radius: 16px;"
            "padding: 5px;"
            "}"
            ".rag-output-frame > scrolledwindow, .rag-output-frame > scrolledwindow > viewport { "
            "background: transparent;"
            "}"
            ".search-output-frame { "
            f"background-color: {BRIEF_TEXT_BG_COLOR};"
            "border-radius: 16px;"
            "padding: 5px;"
            "}"
            ".search-output-frame > scrolledwindow, .search-output-frame > scrolledwindow > viewport { "
            f"background-color: {BRIEF_TEXT_BG_COLOR};"
            "}"
            ".search-output-frame scrolledwindow undershoot { "
            "background: transparent;"
            "box-shadow: none;"
            "}"
            ".search-output-frame scrolledwindow overshoot { "
            "background: transparent;"
            "box-shadow: none;"
            "}"
            ".brief-output-frame, .brief-output-frame > viewport { "
            f"background-color: {BRIEF_TEXT_BG_COLOR};"
            "}"
            f"textview.rag-output {{ font-size: {self._rag_output_font_size}pt; line-height: {DEFAULT_RAG_LINE_HEIGHT}; }}"
            f"textview.search-output {{ font-size: {self._search_output_font_size}pt; }}"
        )
        self._css_provider.load_from_data(css.encode("utf-8"))
        if refresh_content:
            if self._last_rag_answer:
                self._apply_ai_output_links(self._last_rag_answer, self._rag_output_state)
            if self._search_results:
                self._show_search_result_at_index(self._search_result_index)
        else:
            self._refresh_output_colors()

    def apply_saved_ui_settings(
        self,
        rag_size: int,
        search_size: int,
    ) -> bool:
        size_changed = (
            rag_size != self._rag_output_font_size
            or search_size != self._search_output_font_size
        )
        self._rag_output_font_size = rag_size
        self._search_output_font_size = search_size
        if size_changed:
            self._refresh_output_colors()
            self._show_toast("Font size saved. Restart Reference to apply.")
        else:
            self._apply_ui_settings()
        return size_changed

    def _prepare_for_font_resize(self) -> None:
        self._stop_rag_stream_if_running()
        if self._rag_output_state.buffer:
            self._rag_output_state.buffer.set_text("")
        self._last_rag_answer = ""
        self._rag_output_state.link_tags.clear()
        self._rag_output_state.link_lookup.clear()
        if self._search_output_state.buffer:
            self._search_output_state.buffer.set_text("")
        self._search_results = []
        self._search_result_index = 0
        self._search_terms = []
        self._update_search_nav()
        if self._search_title_label:
            self._search_title_label.set_text("")
        self._set_status("")
        self._show_toast("Font size updated; rerun search or ask to refresh output.")

    def _refresh_output_colors(self) -> None:
        self._update_link_tag_colors(self._rag_output_state)
        self._update_search_highlight_color(self._search_output_state)

    def _refresh_rag_quote_colors(self) -> None:
        if self._last_rag_answer:
            self._apply_ai_output_links(self._last_rag_answer, self._rag_output_state)
        else:
            self._update_link_tag_colors(self._rag_output_state)

    def _update_link_tag_colors(self, state: AiOutputView) -> None:
        if not state.link_tags:
            return
        for tag in state.link_tags:
            tag.set_property("foreground-rgba", self._resolve_rag_quote_color(state.view))

    def _update_search_highlight_color(self, state: AiOutputView) -> None:
        buffer = state.buffer
        if not buffer:
            return
        table = buffer.get_tag_table()
        tag = table.lookup("search-highlight") if table else None
        if tag is not None:
            tag.set_property("background", DEFAULT_SEARCH_HIGHLIGHT_COLOR)
            tag.set_property("foreground-set", False)

    def _apply_ai_output_links(self, text: str, state: AiOutputView) -> None:
        self._apply_link_spans(
            text,
            state.buffer,
            state.link_tags,
            state.link_lookup,
            state.scroller,
            state.view,
        )

    def _extract_ai_link_spans(self, text: str) -> tuple[str, list[tuple[int, int, str]]]:
        spans: list[tuple[int, int, str]] = []
        parts: list[str] = []
        cursor = 0
        offset = 0
        for match in AI_LINK_SPAN_RE.finditer(text):
            start, end = match.span()
            before = text[cursor:start]
            parts.append(before)
            offset += len(before)
            phrase = (match.group(1) or match.group(2) or "").strip()
            if phrase:
                link_phrase, trailing = split_link_phrase(phrase)
                if link_phrase:
                    parts.append(link_phrase)
                    spans.append((offset, offset + len(link_phrase), link_phrase))
                    offset += len(link_phrase)
                if trailing:
                    parts.append(trailing)
                    offset += len(trailing)
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts), spans

    def _install_ai_output_link_controllers(self, state: AiOutputView) -> None:
        view = state.view
        if not view:
            return
        if not state.motion_controller:
            motion = Gtk.EventControllerMotion()
            motion.connect("motion", self._on_ai_output_motion, view, state.link_lookup)
            motion.connect("enter", self._on_ai_output_motion, view, state.link_lookup)
            motion.connect("leave", self._on_ai_output_leave, view)
            view.add_controller(motion)
            state.motion_controller = motion
        if not state.click_gesture:
            click = Gtk.GestureClick.new()
            click.set_button(Gdk.BUTTON_PRIMARY)
            click.connect("released", self._on_ai_output_click, view, state.link_lookup)
            view.add_controller(click)
            state.click_gesture = click
        if not state.focus_controller:
            focus_controller = Gtk.EventControllerFocus()
            focus_controller.connect("enter", self._ai_output_focus_enter, view)
            focus_controller.connect("leave", self._ai_output_focus_leave, view)
            view.add_controller(focus_controller)
            state.focus_controller = focus_controller

    def _ai_output_focus_enter(self, _controller: Gtk.EventControllerFocus, view: Gtk.TextView) -> None:
        view.set_cursor_visible(False)

    def _ai_output_focus_leave(self, _controller: Gtk.EventControllerFocus, view: Gtk.TextView) -> None:
        view.set_cursor_visible(False)

    def _ai_link_at_coords(
        self,
        textview: Gtk.TextView,
        x: float,
        y: float,
        lookup: dict[Gtk.TextTag, str],
    ) -> str | None:
        bx, by = textview.window_to_buffer_coords(Gtk.TextWindowType.WIDGET, int(x), int(y))
        iter_result = textview.get_iter_at_location(int(bx), int(by))
        if isinstance(iter_result, tuple):
            success, iter_ = iter_result
            if not success:
                return None
        else:
            iter_ = iter_result
        if iter_ is None:
            return None
        for tag in iter_.get_tags():
            link = lookup.get(tag)
            if link is not None:
                return link
        return None

    def _on_ai_output_motion(
        self,
        _controller: Gtk.EventControllerMotion,
        x: float,
        y: float,
        view: Gtk.TextView,
        lookup: dict[Gtk.TextTag, str],
    ) -> None:
        link = self._ai_link_at_coords(view, x, y, lookup)
        if link:
            view.set_cursor_from_name("pointer")
        else:
            view.set_cursor_from_name(None)

    def _on_ai_output_leave(self, _controller: Gtk.EventControllerMotion, view: Gtk.TextView) -> None:
        view.set_cursor_from_name(None)

    def _on_ai_output_click(
        self,
        gesture: Gtk.GestureClick,
        _n_press: int,
        x: float,
        y: float,
        view: Gtk.TextView,
        lookup: dict[Gtk.TextTag, str],
    ) -> None:
        button = gesture.get_current_button()
        if button and button != Gdk.BUTTON_PRIMARY:
            return
        view.grab_focus()
        view.set_cursor_visible(False)
        phrase = self._ai_link_at_coords(view, x, y, lookup)
        if not phrase:
            return
        self._activate_ai_link(phrase)


class ReferenceSettingsWindow(Adw.ApplicationWindow):
    def __init__(self, app: ReferenceApp, parent: ReferenceWindow) -> None:
        super().__init__(application=app, transient_for=parent)
        self.set_title("Settings")
        self.set_default_size(840, 620)
        self.set_resizable(True)
        self._app = app
        self._parent = parent
        self._rag_api_url_row: Adw.EntryRow | None = None
        self._rag_model_row: Adw.EntryRow | None = None
        self._rag_api_key_row: Adw.EntryRow | None = None
        self._rag_top_k_row: Adw.EntryRow | None = None
        self._rag_timeout_row: Adw.EntryRow | None = None
        self._rag_reasoning_row: Adw.SwitchRow | None = None
        self._embeddings_provider_row: Adw.ComboRow | None = None
        self._embeddings_provider_values: list[str] = [RAG_PROVIDER_VOYAGE, RAG_PROVIDER_ISAACUS]
        self._voyage_model_row: Adw.EntryRow | None = None
        self._voyage_key_row: Adw.EntryRow | None = None
        self._isaacus_model_row: Adw.EntryRow | None = None
        self._isaacus_key_row: Adw.EntryRow | None = None
        self._font_size_row: Adw.EntryRow | None = None
        self._search_font_size_row: Adw.EntryRow | None = None
        self._prompt_buffers: dict[str, Gtk.TextBuffer] = {}
        self._status_label: Gtk.Label | None = None
        self._build_ui()
        self._load_settings()
        self.connect("close-request", self._on_close_request)

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        self._app._settings_window = None
        return False

    def _build_ui(self) -> None:
        view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.add_css_class("flat")
        header.set_title_widget(Gtk.Label(label="Settings", xalign=0))
        view.add_top_bar(header)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(18)
        box.set_margin_bottom(12)
        box.set_margin_start(18)
        box.set_margin_end(18)

        display_group = Adw.PreferencesGroup(title="Display")
        display_group.add_css_class("list-stack")
        display_group.set_hexpand(True)
        box.append(display_group)

        font_size_row = Adw.EntryRow(title="RAG Output Font Size (pt)")
        font_size_row.set_hexpand(True)
        display_group.add(font_size_row)
        self._font_size_row = font_size_row

        search_font_size_row = Adw.EntryRow(title="Search Output Font Size (pt)")
        search_font_size_row.set_hexpand(True)
        display_group.add(search_font_size_row)
        self._search_font_size_row = search_font_size_row

        rag_group = Adw.PreferencesGroup(title="RAG LLM (OpenAI Compatible)")
        rag_group.add_css_class("list-stack")
        rag_group.set_hexpand(True)
        box.append(rag_group)

        rag_api_url = Adw.EntryRow(title="API URL")
        rag_api_url.set_hexpand(True)
        rag_group.add(rag_api_url)
        self._rag_api_url_row = rag_api_url

        rag_model = Adw.EntryRow(title="Model ID")
        rag_model.set_hexpand(True)
        rag_group.add(rag_model)
        self._rag_model_row = rag_model

        rag_top_k = Adw.EntryRow(title="RAG Context Chunks (k)")
        rag_top_k.set_hexpand(True)
        rag_group.add(rag_top_k)
        self._rag_top_k_row = rag_top_k

        rag_timeout = Adw.EntryRow(title="RAG Timeout (seconds)")
        rag_timeout.set_hexpand(True)
        rag_group.add(rag_timeout)
        self._rag_timeout_row = rag_timeout

        rag_reasoning = Adw.SwitchRow(
            title="Show Reasoning Trace",
            subtitle="Display streamed reasoning tokens when emitted by the model.",
        )
        rag_group.add(rag_reasoning)
        self._rag_reasoning_row = rag_reasoning

        rag_api_key = self._build_password_row("API Key")
        rag_group.add(rag_api_key)
        self._rag_api_key_row = rag_api_key

        embeddings_group = Adw.PreferencesGroup(title="Embeddings")
        embeddings_group.add_css_class("list-stack")
        embeddings_group.set_hexpand(True)
        box.append(embeddings_group)

        embeddings_provider = Adw.ComboRow(title="Provider")
        embeddings_provider.set_model(Gtk.StringList.new(["VoyageAI", "Isaacus"]))
        embeddings_group.add(embeddings_provider)
        self._embeddings_provider_row = embeddings_provider

        voyage_model = Adw.EntryRow(title="Voyage Model")
        voyage_model.set_hexpand(True)
        embeddings_group.add(voyage_model)
        self._voyage_model_row = voyage_model

        voyage_key = self._build_password_row("Voyage API Key")
        embeddings_group.add(voyage_key)
        self._voyage_key_row = voyage_key

        isaacus_model = Adw.EntryRow(title="Isaacus Model")
        isaacus_model.set_hexpand(True)
        embeddings_group.add(isaacus_model)
        self._isaacus_model_row = isaacus_model

        isaacus_key = self._build_password_row("Isaacus API Key")
        embeddings_group.add(isaacus_key)
        self._isaacus_key_row = isaacus_key

        self._add_prompt_section(
            box,
            "No Citations Prompt",
            DEFAULT_RAG_PROMPT,
            RAG_PROMPT_NO_CITATIONS,
        )
        self._add_prompt_section(
            box,
            "Full Citations Prompt",
            DEFAULT_RAG_PROMPT_FULL_CITATIONS,
            RAG_PROMPT_FULL_CITATIONS,
        )
        self._add_prompt_section(
            box,
            "Statutes/Rules Only Prompt",
            DEFAULT_RAG_PROMPT_STATUTES_ONLY,
            RAG_PROMPT_STATUTES_ONLY,
        )

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        scrolled.set_child(box)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content.append(scrolled)

        buttons = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        buttons.set_margin_top(6)
        buttons.set_margin_bottom(12)
        buttons.set_margin_start(12)
        buttons.set_margin_end(12)
        buttons.set_halign(Gtk.Align.END)
        save_btn = Gtk.Button(label="Save Settings")
        save_btn.add_css_class("suggested-action")
        save_btn.add_css_class("flat")
        save_btn.connect("clicked", self._on_save_clicked)
        buttons.append(save_btn)
        content.append(buttons)

        status_label = Gtk.Label(label="", xalign=0)
        status_label.add_css_class("dim-label")
        status_label.set_wrap(True)
        content.append(status_label)
        self._status_label = status_label

        view.set_content(content)
        self.set_content(view)

    def _build_password_row(self, title: str) -> Adw.EntryRow:
        password_row_cls = getattr(Adw, "PasswordEntryRow", None)
        if password_row_cls:
            row = password_row_cls(title=title)
            if hasattr(row, "set_show_peek_icon"):
                row.set_show_peek_icon(True)
        else:
            row = Adw.EntryRow(title=title)
            if hasattr(row, "set_input_purpose"):
                row.set_input_purpose(Gtk.InputPurpose.PASSWORD)
            if hasattr(row, "set_visibility"):
                try:
                    row.set_visibility(False)
                except Exception:
                    pass
        if hasattr(row, "set_hexpand"):
            row.set_hexpand(True)
        return row

    def _add_prompt_section(self, box: Gtk.Box, title: str, text: str, key: str) -> None:
        prompt_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        prompt_section.set_hexpand(True)
        prompt_section.set_vexpand(True)
        prompt_label = Gtk.Label(label=title, xalign=0)
        prompt_label.add_css_class("dim-label")
        prompt_section.append(prompt_label)
        prompt_scroller, buffer = self._build_prompt_editor(text)
        prompt_section.append(prompt_scroller)
        box.append(prompt_section)
        self._prompt_buffers[key] = buffer

    def _build_prompt_editor(self, text: str) -> tuple[Gtk.ScrolledWindow, Gtk.TextBuffer]:
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scroller.set_hexpand(True)
        scroller.set_vexpand(True)
        scroller.set_has_frame(False)

        buffer = Gtk.TextBuffer()
        buffer.set_text(text)
        prompt_view = Gtk.TextView.new_with_buffer(buffer)
        prompt_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        prompt_view.set_monospace(True)
        prompt_view.set_vexpand(True)
        prompt_view.set_hexpand(True)
        prompt_view.set_top_margin(12)
        prompt_view.set_bottom_margin(12)
        prompt_view.set_left_margin(12)
        prompt_view.set_right_margin(12)
        scroller.set_child(prompt_view)
        return scroller, buffer

    def _prompt_text(self, key: str, fallback: str) -> str:
        buffer = self._prompt_buffers.get(key)
        if not buffer:
            return fallback
        start, end = buffer.get_bounds()
        return buffer.get_text(start, end, True)

    def _load_settings(self) -> None:
        settings = load_ai_settings()
        if self._rag_api_url_row:
            self._rag_api_url_row.set_text(settings.rag_api_url)
        if self._rag_model_row:
            self._rag_model_row.set_text(settings.rag_model_id)
        if self._rag_api_key_row:
            self._rag_api_key_row.set_text(settings.rag_api_key)
        if self._rag_top_k_row:
            self._rag_top_k_row.set_text(str(settings.rag_top_k))
        if self._rag_timeout_row:
            self._rag_timeout_row.set_text(str(settings.deep_ask_timeout_seconds))
        if self._rag_reasoning_row:
            self._rag_reasoning_row.set_active(bool(settings.deep_ask_show_reasoning))
        if self._embeddings_provider_row:
            provider = _normalize_rag_provider(settings.rag_provider)
            if provider in self._embeddings_provider_values:
                self._embeddings_provider_row.set_selected(self._embeddings_provider_values.index(provider))
            else:
                self._embeddings_provider_row.set_selected(0)
        if self._voyage_model_row:
            self._voyage_model_row.set_text(settings.voyage_model)
        if self._voyage_key_row:
            self._voyage_key_row.set_text(settings.voyage_api_key)
        if self._isaacus_model_row:
            self._isaacus_model_row.set_text(settings.isaacus_model)
        if self._isaacus_key_row:
            self._isaacus_key_row.set_text(settings.isaacus_api_key)
        if RAG_PROMPT_NO_CITATIONS in self._prompt_buffers:
            self._prompt_buffers[RAG_PROMPT_NO_CITATIONS].set_text(
                settings.rag_prompt_no_citations or DEFAULT_RAG_PROMPT
            )
        if RAG_PROMPT_FULL_CITATIONS in self._prompt_buffers:
            self._prompt_buffers[RAG_PROMPT_FULL_CITATIONS].set_text(
                settings.rag_prompt_full_citations or DEFAULT_RAG_PROMPT_FULL_CITATIONS
            )
        if RAG_PROMPT_STATUTES_ONLY in self._prompt_buffers:
            self._prompt_buffers[RAG_PROMPT_STATUTES_ONLY].set_text(
                settings.rag_prompt_statutes_only or DEFAULT_RAG_PROMPT_STATUTES_ONLY
            )
        rag_size, search_size = load_ui_settings()
        if self._font_size_row:
            self._font_size_row.set_text(str(rag_size))
        if self._search_font_size_row:
            self._search_font_size_row.set_text(str(search_size))

    def _on_save_clicked(self, _button: Gtk.Button) -> None:
        if not all(
            [
                self._rag_api_url_row,
                self._rag_model_row,
                self._rag_api_key_row,
                self._embeddings_provider_row,
                self._voyage_model_row,
                self._voyage_key_row,
                self._isaacus_model_row,
                self._isaacus_key_row,
                self._rag_timeout_row,
                self._rag_reasoning_row,
            ]
        ):
            return
        settings = AiSettings(
            rag_api_url=self._rag_api_url_row.get_text().strip(),
            rag_model_id=self._rag_model_row.get_text().strip(),
            rag_api_key=self._rag_api_key_row.get_text().strip(),
            rag_prompt_no_citations=self._prompt_text(
                RAG_PROMPT_NO_CITATIONS,
                DEFAULT_RAG_PROMPT,
            ).strip()
            or DEFAULT_RAG_PROMPT,
            rag_prompt_full_citations=self._prompt_text(
                RAG_PROMPT_FULL_CITATIONS,
                DEFAULT_RAG_PROMPT_FULL_CITATIONS,
            ).strip()
            or DEFAULT_RAG_PROMPT_FULL_CITATIONS,
            rag_prompt_statutes_only=self._prompt_text(
                RAG_PROMPT_STATUTES_ONLY,
                DEFAULT_RAG_PROMPT_STATUTES_ONLY,
            ).strip()
            or DEFAULT_RAG_PROMPT_STATUTES_ONLY,
            rag_top_k=DEFAULT_RAG_TOP_K,
            rag_provider=DEFAULT_RAG_PROVIDER,
            voyage_api_key=self._voyage_key_row.get_text().strip(),
            voyage_model=self._voyage_model_row.get_text().strip() or DEFAULT_RAG_VOYAGE_MODEL,
            isaacus_api_key=self._isaacus_key_row.get_text().strip(),
            isaacus_model=self._isaacus_model_row.get_text().strip() or DEFAULT_RAG_ISAACUS_MODEL,
            deep_ask_timeout_seconds=DEFAULT_STREAM_TIMEOUT_SECONDS,
            deep_ask_show_reasoning=bool(self._rag_reasoning_row.get_active()),
        )
        provider_index = int(self._embeddings_provider_row.get_selected())
        if 0 <= provider_index < len(self._embeddings_provider_values):
            settings.rag_provider = self._embeddings_provider_values[provider_index]
        else:
            settings.rag_provider = DEFAULT_RAG_PROVIDER
        if self._rag_top_k_row:
            raw_top_k = self._rag_top_k_row.get_text().strip()
            try:
                settings.rag_top_k = _clamp_rag_top_k(int(raw_top_k))
            except (TypeError, ValueError):
                settings.rag_top_k = DEFAULT_RAG_TOP_K
        if self._rag_timeout_row:
            settings.deep_ask_timeout_seconds = _coerce_timeout_seconds(
                self._rag_timeout_row.get_text().strip(),
                DEFAULT_STREAM_TIMEOUT_SECONDS,
            )
        save_ai_settings(settings)
        self._parent._ai_settings = settings
        self._parent._kickoff_rag_background_load()
        rag_size = DEFAULT_OUTPUT_FONT_SIZE
        search_size = DEFAULT_OUTPUT_FONT_SIZE
        if self._font_size_row:
            raw_size = self._font_size_row.get_text().strip()
            try:
                rag_size = _clamp_font_size(int(raw_size))
            except (TypeError, ValueError):
                rag_size = DEFAULT_OUTPUT_FONT_SIZE
        if self._search_font_size_row:
            raw_size = self._search_font_size_row.get_text().strip()
            try:
                search_size = _clamp_font_size(int(raw_size))
            except (TypeError, ValueError):
                search_size = DEFAULT_OUTPUT_FONT_SIZE
        save_ui_settings(rag_size, search_size)
        size_changed = self._parent.apply_saved_ui_settings(
            rag_size,
            search_size,
        )
        if self._status_label:
            if size_changed:
                self._status_label.set_text("Settings saved. Restart Reference to apply font size changes.")
            else:
                self._status_label.set_text("Settings saved.")


def main() -> int:
    app = ReferenceApp()
    return app.run([])


if __name__ == "__main__":
    raise SystemExit(main())
