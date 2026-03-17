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

  const navigationPath = path === "/" ? "./" : path.replace(/^\//, "");
  await page.goto(navigationPath, { waitUntil: "networkidle" });

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

export async function assertNoMissingPage(page: Page): Promise<void> {
  const mainHeading = page.locator("main article > h1").first();

  await expect(mainHeading).toBeVisible();
  await expect(mainHeading).not.toHaveText("404 - Not found");
}

export async function assertThemeDiagramVariants(
  page: Page,
  theme: "light" | "dark"
): Promise<void> {
  const offenders = await page.evaluate((selectedTheme) => {
    const diagrams = Array.from(document.querySelectorAll("figure.diagram"));
    const bad: Array<{
      figure: string;
      visibleLight: number;
      visibleDark: number;
    }> = [];

    const isVisible = (element: Element): boolean => {
      const node = element as HTMLElement;
      const style = window.getComputedStyle(node);
      const rect = node.getBoundingClientRect();
      return (
        style.display !== "none" &&
        style.visibility !== "hidden" &&
        rect.width > 0 &&
        rect.height > 0
      );
    };

    for (const figure of diagrams) {
      const light = Array.from(figure.querySelectorAll("img.diagram-light"));
      const dark = Array.from(figure.querySelectorAll("img.diagram-dark"));
      if (light.length === 0 && dark.length === 0) {
        continue;
      }

      const visibleLight = light.filter(isVisible).length;
      const visibleDark = dark.filter(isVisible).length;
      const expectedLight = selectedTheme === "light" ? 1 : 0;
      const expectedDark = selectedTheme === "dark" ? 1 : 0;

      if (visibleLight !== expectedLight || visibleDark !== expectedDark) {
        bad.push({
          figure: (figure.textContent || "").trim().replace(/\s+/g, " ").slice(0, 120),
          visibleLight,
          visibleDark,
        });
      }
    }

    return bad.slice(0, 20);
  }, theme);

  expect(
    offenders,
    `Theme diagram visibility offenders:\n${JSON.stringify(offenders, null, 2)}`
  ).toEqual([]);
}

export async function assertNoDomOverflow(page: Page): Promise<void> {
  const offenders = await page.evaluate(() => {
    const bad: Array<{ tag: string; className: string; text: string }> = [];
    const root =
      document.querySelector(".md-content__inner") ||
      document.querySelector("main") ||
      document.body;
    const nodes = Array.from(root.querySelectorAll("*"));

    for (const el of nodes) {
      const node = el as HTMLElement;
      const style = window.getComputedStyle(node);

      if (style.display === "inline") continue;
      if (style.textOverflow === "ellipsis") continue;
      if (node.classList.contains("md-ellipsis")) continue;
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
    type Candidate = {
      el: HTMLElement;
      text: string;
      className: string;
      rect: {
        left: number;
        top: number;
        right: number;
        bottom: number;
        width: number;
        height: number;
      };
    };

    const root =
      (document.querySelector(".md-content__inner") as HTMLElement | null) ||
      (document.querySelector("main") as HTMLElement | null) ||
      document.body;

    const selectors = [
      "h1",
      "h2",
      "h3",
      "h4",
      "p",
      "li",
      "figcaption",
      "td",
      "th",
      "blockquote",
      ".metric-number",
      ".metric-value",
      ".snapshot-label",
      ".snapshot-copy"
    ].join(",");

    const hiddenOrIgnoredAncestorSelector = [
      "[hidden]",
      "[aria-hidden='true']",
      ".md-header",
      ".md-sidebar",
      ".md-search",
      ".md-search__inner",
      ".md-overlay",
      ".md-dialog",
      ".md-tabs",
      ".md-footer",
      ".md-top",
      "nav"
    ].join(",");

    function isVisible(el: HTMLElement): boolean {
      const style = window.getComputedStyle(el);
      const rect = el.getBoundingClientRect();

      if (el.closest(hiddenOrIgnoredAncestorSelector)) return false;
      if (style.display === "none") return false;
      if (style.visibility === "hidden") return false;
      if (style.opacity === "0") return false;
      if (rect.width < 8 || rect.height < 8) return false;

      const text = (el.innerText || "").trim();
      return text.length > 0;
    }

    function hasVisibleTextChild(el: HTMLElement): boolean {
      return Array.from(el.children).some((child) => {
        const c = child as HTMLElement;
        return isVisible(c) && (c.innerText || "").trim().length > 0;
      });
    }

    function overlapArea(a: Candidate["rect"], b: Candidate["rect"]): number {
      const width = Math.max(
        0,
        Math.min(a.right, b.right) - Math.max(a.left, b.left)
      );
      const height = Math.max(
        0,
        Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top)
      );
      return width * height;
    }

    function area(rect: Candidate["rect"]): number {
      return rect.width * rect.height;
    }

    const candidates: Candidate[] = Array.from(
      root.querySelectorAll<HTMLElement>(selectors)
    )
      .filter((el) => isVisible(el))
      .filter((el) => !hasVisibleTextChild(el))
      .map((el) => {
        const rect = el.getBoundingClientRect();
        return {
          el,
          text: (el.innerText || "").trim().replace(/\s+/g, " ").slice(0, 120),
          className: (el.className || "").toString(),
          rect: {
            left: rect.left,
            top: rect.top,
            right: rect.right,
            bottom: rect.bottom,
            width: rect.width,
            height: rect.height
          }
        };
      });

    const bad: Array<{ a: string; b: string }> = [];

    for (let i = 0; i < candidates.length; i += 1) {
      for (let j = i + 1; j < candidates.length; j += 1) {
        const a = candidates[i];
        const b = candidates[j];

        if (a.el.contains(b.el) || b.el.contains(a.el)) continue;

        const overlap = overlapArea(a.rect, b.rect);
        if (overlap <= 0) continue;

        const smallerArea = Math.min(area(a.rect), area(b.rect));
        if (smallerArea <= 0) continue;

        // Ignore tiny edge-touching or near-miss cases.
        if (overlap < 12) continue;

        // Only flag genuinely substantial overlap.
        if (overlap / smallerArea < 0.25) continue;

        bad.push({
          a: `${a.text} [${a.className}]`,
          b: `${b.text} [${b.className}]`
        });

        if (bad.length >= 20) {
          return bad;
        }
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