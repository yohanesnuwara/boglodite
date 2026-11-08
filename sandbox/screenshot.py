"""Screenshot an HTML page with headless Chromium (WebGL via SwiftShader) and report JS errors.

    uv run --no-sync --with playwright python sandbox/screenshot.py PAGE.html OUT.png [WIDTH]
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def main(page: str, out: str, width: int = 1500) -> None:
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = await b.new_page(viewport={"width": width, "height": 1000})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
        await pg.goto(Path(page).resolve().as_uri())
        await pg.wait_for_timeout(7000)
        await pg.screenshot(path=out, full_page=True)
        print("errors:", errs)
        await b.close()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1500))
