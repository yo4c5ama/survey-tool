import ast
from pathlib import Path
from string import Formatter

from vnn_survey.app.i18n import LANGUAGE_NAMES, TRANSLATIONS, language_name, translate


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def test_translation_catalogs_have_matching_keys_and_placeholders() -> None:
    reference_keys = set(TRANSLATIONS["zh"])

    assert set(LANGUAGE_NAMES) == {"en", "zh", "ja", "ko"}
    for language in ["ja", "ko"]:
        assert set(TRANSLATIONS[language]) == reference_keys
    for language, catalog in TRANSLATIONS.items():
        for source, localized in catalog.items():
            assert _fields(localized) == _fields(source), (language, source)


def test_translation_uses_english_fallback_and_formats_values() -> None:
    assert translate("Results", "en") == "Results"
    assert translate("Results", "zh") == "结果"
    assert translate("Run {run_id}", "ja", run_id="2026") == "実行 2026"
    assert translate("Uncatalogued research text", "ko") == "Uncatalogued research text"
    assert language_name("ko") == "한국어"


def test_static_interface_phrases_are_translated() -> None:
    main_path = Path(__file__).parents[1] / "src" / "vnn_survey" / "app" / "main.py"
    tree = ast.parse(main_path.read_text(encoding="utf-8"))
    messages = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_t"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }

    for language, catalog in TRANSLATIONS.items():
        assert not messages.difference(catalog), language
