import { expect, type Locator, type Page } from "@playwright/test";

export async function setTheme(page: Page, theme: "light" | "dark"): Promise<void> {
  await page.addInitScript((selectedTheme: "light" | "dark") => {
    localStorage.setItem("/.__palette", JSON.stringify({ color: selectedTheme }));
  }, theme);
}

export async function gotoAndStabilize(
  page: Page,
  path: string,
  theme?: "light" | "dark"
): Promise<void> {
  if (theme) {
    await setTheme(page, theme);
  }

  await page.goto(path, { waitUntil: "networkidle" });

  await page.evaluate((selectedTheme?: "light" | "dark") => {
    if (!selectedTheme) return;
    const body = document.body;
    if (!body) return;

    if (selectedTheme === "dark") {
      body.setAttribute("data-md-color-scheme", "slate");
    } else {
      body.setAttribute("data-md-color-scheme", "default");
    }
  }, theme);

  await waitForPageStable(page);
}

export async function waitForPageStable(page: Page): Promise<void> {
  await page.waitForLoadState("networkidle");
  await page.evaluate(async () => {
    if ("fonts" in document) {
      await document.fonts.ready;
    }
  });
}

export async function assertImagesLoaded(page: Page): Promise<void> {
  const failures = await page.evaluate(() => {
    const media = Array.from(document.querySelectorAll("img"));
    return media
      .filter(
        (img) =>
          !(img as HTMLImageElement).complete ||
          (img as HTMLImageElement).naturalWidth === 0
      )
      .map((img) => ({
        src: (img as HTMLImageElement).src,
        alt: (img as HTMLImageElement).alt
      }));
  });

  expect(
    failures,
    `Broken images:\n${JSON.stringify(failures, null, 2)}`
  ).toEqual([]);
}

export async function assertNoDomOverflow(page: Page): Promise<void> {
  const offenders = await page.evaluate(() => {
    const bad: Array<{ tag: string; className: string; text: string }> = [];
    const nodes = Array.from(document.querySelectorAll("body *"));

    for (const el of nodes) {
      const node = el as HTMLElement;
      const style = window.getComputedStyle(node);

      if (style.display === "inline") continue;
      if (!node.innerText?.trim()) continue;
      if (node.children.length > 0) continue;

      const horizontalOverflow = node.scrollWidth - node.clientWidth > 2;
      const verticalOverflow = node.scrollHeight - node.clientHeight > 2;

      if (horizontalOverflow || verticalOverflow) {
        bad.push({
          tag: node.tagName.toLowerCase(),
          className: node.className,
          text: node.innerText.trim().slice(0, 120)
        });
      }
    }

    return bad.slice(0, 20);
  });

  expect(
    offenders,
    `DOM overflow offenders:\n${JSON.stringify(offenders, null, 2)}`
  ).toEqual([]);
}

export async function assertNoTinyVisibleContainers(page: Page): Promise<void> {
  const offenders = await page.evaluate(() => {
    const selectors = [
      ".snapshot-card",
      ".diagram",
      ".diagram-img",
      ".md-content figure",
      ".md-typeset .grid",
      ".md-typeset .card",
      ".md-content__inner article"
    ];

    const bad: Array<{
      selector: string;
      width: number;
      height: number;
      text: string;
    }> = [];

    for (const selector of selectors) {
      const nodes = Array.from(document.querySelectorAll(selector));
      for (const node of nodes) {
        const el = node as HTMLElement;
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        const visible =
          rect.width > 0 &&
          rect.height > 0 &&
          style.visibility !== "hidden" &&
          style.display !== "none";

        if (!visible) continue;

        const looksLikeContent =
          el.innerText.trim().length > 0 ||
          el.querySelector("img, svg, canvas, figure, picture");

        if (!looksLikeContent) continue;

        if (rect.height < 24 || rect.width < 40) {
          bad.push({
            selector,
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            text: el.innerText.trim().slice(0, 120)
          });
        }
      }
    }

    return bad.slice(0, 20);
  });

  expect(
    offenders,
    `Tiny visible containers:\n${JSON.stringify(offenders, null, 2)}`
  ).toEqual([]);
}

export async function assertNoMeaningfulOverlaps(page: Page): Promise<void> {
  const offenders = await page.evaluate(() => {
    type Box = {
      text: string;
      className: string;
      left: number;
      top: number;
      right: number;
      bottom: number;
      width: number;
      height: number;
    };

    function intersects(a: Box, b: Box): boolean {
      return !(
        a.right <= b.left ||
        a.left >= b.right ||
        a.bottom <= b.top ||
        a.top >= b.bottom
      );
    }

    const ignored = new Set([
      "md-header",
      "md-sidebar",
      "md-search",
      "md-overlay",
      "md-dialog"
    ]);

    const nodes = Array.from(
      document.querySelectorAll(
        "main h1, main h2, main h3, main p, main li, main .metric-number, main .metric-value, main .snapshot-label, main .snapshot-copy"
      )
    );

    const boxes: Box[] = [];

    for (const node of nodes) {
      const el = node as HTMLElement;
      const rect = el.getBoundingClientRect();
      const style = window.getComputedStyle(el);

      if (
        rect.width < 8 ||
        rect.height < 8 ||
        style.display === "none" ||
        style.visibility === "hidden"
      ) {
        continue;
      }

      const className = el.className || "";
      if ([...ignored].some((name) => className.includes(name))) {
        continue;
      }

      const text = el.innerText.trim();
      if (!text) continue;

      boxes.push({
        text: text.slice(0, 80),
        className,
        left: rect.left,
        top: rect.top,
        right: rect.right,
        bottom: rect.bottom,
        width: rect.width,
        height: rect.height
      });
    }

    const bad: Array<{ a: string; b: string }> = [];

    for (let i = 0; i < boxes.length; i += 1) {
      for (let j = i + 1; j < boxes.length; j += 1) {
        const a = boxes[i];
        const b = boxes[j];

        if (!intersects(a, b)) continue;

        const sameBlock =
          Math.abs(a.left - b.left) < 4 &&
          Math.abs(a.right - b.right) < 4 &&
          Math.abs(a.top - b.top) < 24;

        if (sameBlock) continue;

        bad.push({
          a: `${a.text} [${a.className}]`,
          b: `${b.text} [${b.className}]`
        });

        if (bad.length >= 20) return bad;
      }
    }

    return bad;
  });

  expect(
    offenders,
    `Meaningful overlaps:\n${JSON.stringify(offenders, null, 2)}`
  ).toEqual([]);
}

export async function expectMainScreenshot(
  page: Page,
  name: string
): Promise<void> {
  const main: Locator = page.locator("main");
  await expect(main).toBeVisible();
  await expect(main).toHaveScreenshot(name, { animations: "disabled" });
}