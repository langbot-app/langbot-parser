from __future__ import annotations

import sys
import types
import unittest
import logging
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
logging.disable(logging.CRITICAL)


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    if '.' in name:
        parent_name, child_name = name.rsplit('.', 1)
        parent = _ensure_module(parent_name)
        setattr(parent, child_name, module)
    return module


def _install_langbot_stubs() -> None:
    parser_module = _ensure_module('langbot_plugin.api.definition.components.parser.parser')
    models_module = _ensure_module('langbot_plugin.api.entities.builtin.rag.models')

    class Parser:
        pass

    @dataclass
    class ParseContext:
        file_content: bytes
        filename: str
        mime_type: str | None = None
        metadata: dict | None = None

    @dataclass
    class ParseResult:
        text: str
        sections: list
        metadata: dict

    @dataclass
    class TextSection:
        content: str
        heading: str
        level: int
        page: int | None = None

    parser_module.Parser = Parser
    models_module.ParseContext = ParseContext
    models_module.ParseResult = ParseResult
    models_module.TextSection = TextSection


def _install_optional_dependency_stubs() -> None:
    try:
        import markdown  # noqa: F401
    except ModuleNotFoundError:
        markdown_module = types.ModuleType('markdown')
        markdown_module.markdown = lambda text, extensions=None: text
        sys.modules['markdown'] = markdown_module

    try:
        import fitz  # noqa: F401
    except ModuleNotFoundError:
        sys.modules['fitz'] = types.ModuleType('fitz')


_install_langbot_stubs()
_install_optional_dependency_stubs()


class GeneralParsersConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_enable_vision_false_suppresses_configured_model(self) -> None:
        from components.general_parsers.general_parsers import GeneralParsers
        from langbot_plugin.api.entities.builtin.rag.models import ParseContext

        class Plugin:
            called = False

            def get_config(self) -> dict:
                return {
                    'enable_vision': False,
                    'vision_llm_model_uuid': 'vision-model',
                }

            async def invoke_llm(self, *args, **kwargs):
                self.called = True
                raise AssertionError('vision model should not be invoked')

        plugin = Plugin()
        parser = GeneralParsers()
        parser.plugin = plugin

        result = await parser.parse(
            ParseContext(
                file_content=b'not a real image',
                filename='diagram.png',
                mime_type='image/png',
            )
        )

        self.assertFalse(plugin.called)
        self.assertEqual(result.text, '[图片文件: diagram.png]')
        self.assertFalse(result.metadata['vision_used'])


class HtmlParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_nested_html_preserves_headings_and_markdown_tables(self) -> None:
        from components.general_parsers.parsers.html_text import parse_html

        html = b'''
        <html><body>
          <main><section>
            <h1>Title</h1>
            <p>Intro paragraph</p>
            <table><tr><th>A</th></tr><tr><td>B</td></tr></table>
          </section></main>
        </body></html>
        '''

        text, metadata = await parse_html(html, 'nested.html')

        self.assertIn('# Title', text)
        self.assertIn('Intro paragraph', text)
        self.assertIn('| A |', text)
        self.assertIn('| --- |', text)
        self.assertIn('| B |', text)
        self.assertLess(text.index('# Title'), text.index('Intro paragraph'))
        self.assertLess(text.index('Intro paragraph'), text.index('| A |'))
        self.assertFalse(metadata['has_images'])

    async def test_inline_image_vision_failure_keeps_text_and_placeholder(self) -> None:
        from components.general_parsers.parsers.html_text import parse_html

        async def failing_vision(image_b64: str, prompt: str) -> str:
            raise RuntimeError('vision service unavailable')

        html = b'''
        <html><body>
          <p>before <img src="data:image/png;base64,AAAA" alt="diagram"> after</p>
        </body></html>
        '''

        text, metadata = await parse_html(html, 'image.html', invoke_vision=failing_vision)

        self.assertIn('before', text)
        self.assertIn('[图片: HTML图片1]', text)
        self.assertIn('diagram', text)
        self.assertIn('after', text)
        self.assertFalse(metadata['vision_used'])
        self.assertEqual(metadata['vision_tasks_count'], 1)
        self.assertEqual(metadata['vision_images_described_count'], 0)
        self.assertEqual(metadata['vision_failed_count'], 1)


class ImageParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_image_vision_failure_returns_placeholder(self) -> None:
        from components.general_parsers.parsers.image import parse_image

        async def failing_vision(image_b64: str, prompt: str) -> str:
            raise RuntimeError('vision service unavailable')

        text, metadata = await parse_image(b'image bytes', 'photo.png', invoke_vision=failing_vision)

        self.assertEqual(text, '[图片文件: photo.png]')
        self.assertFalse(metadata['vision_used'])
        self.assertEqual(metadata['vision_tasks_count'], 1)
        self.assertEqual(metadata['vision_images_described_count'], 0)
        self.assertEqual(metadata['vision_failed_count'], 1)


class PdfVisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_pdf_vision_task_failures_are_counted_without_losing_successes(self) -> None:
        from components.general_parsers.parsers.pdf import _process_vision_tasks

        async def mixed_vision(image_b64: str, prompt: str) -> str:
            if image_b64 == 'fail':
                raise RuntimeError('vision service unavailable')
            return 'recognized diagram'

        text, stats = await _process_vision_tasks(
            '<!-- PAGE:1 -->\n[图片: 第1页-图片1]\n\n<!-- PAGE:2 -->\n',
            [
                {
                    'type': 'embedded_image',
                    'page': 1,
                    'img_idx': 0,
                    'image_b64': 'ok',
                    'placeholder': '[图片: 第1页-图片1]',
                },
                {
                    'type': 'scanned_page',
                    'page': 2,
                    'image_b64': 'fail',
                },
            ],
            mixed_vision,
        )

        self.assertIn('[图片描述: recognized diagram]', text)
        self.assertEqual(stats['vision_images_described_count'], 1)
        self.assertEqual(stats['vision_scanned_pages_count'], 0)
        self.assertEqual(stats['vision_failed_count'], 1)


class UtilsTests(unittest.TestCase):
    def test_count_words_counts_latin_words_and_cjk_characters(self) -> None:
        from components.general_parsers.utils import count_words

        self.assertEqual(count_words("hello world 中国"), 4)
        self.assertEqual(count_words("O'Reilly co-op"), 2)


if __name__ == '__main__':
    unittest.main()
