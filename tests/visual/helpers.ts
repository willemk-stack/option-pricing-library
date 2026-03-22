import { expect, type Locator, type Page } from "@playwright/test";

async function stubRepositoryFacts(page: Page): Promise<void> {
  // Full-page docs baselines intentionally exclude async shell chrome like the
  // repo facts widget so page snapshots stay focused on docs content.
  // MkDocs Material augments the repo badge with async GitHub/GitLab facts.
  // Those counters are outside the docs content we care about and can appear
  // nondeterministically across snapshot runs, so force an empty response.
  await page.route(/https:\/\/api\.github\.com\/repos\/.+/i, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: "{}",
    });
  });

  await page.route(
    /https:\/\/[^/]*gitlab[^/]+\/api\/v4\/projects\/.+/i,
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: "{}",
      });
    }
  );
}

export async function mockRepositoryFacts(
  page: Page,
  facts: { version: string; stars: number; forks: number } = {
    version: "v0.4.0",
    stars: 42,
    forks: 7,
  }
): Promise<void> {
  await page.route(
    /https:\/\/api\.github\.com\/repos\/[^/]+\/[^/]+\/releases\/latest$/i,
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ tag_name: facts.version }),
      });
    }
  );

  await page.route(
    /https:\/\/api\.github\.com\/repos\/[^/]+\/[^/]+$/i,
    async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          stargazers_count: facts.stars,
          forks_count: facts.forks,
        }),
      });
    }
  );
}

async function suppressRepositoryFacts(page: Page): Promise<void> {
  await page.addInitScript(() => {
    const suppressionKey = "__playwrightRepoFactsSuppressed";
    const globalWindow = window as Window & {
      __playwrightRepoFactsSuppressed?: boolean;
    };

    if (globalWindow[suppressionKey]) {
      return;
    }

    globalWindow[suppressionKey] = true;

    const styleId = "__playwright-hide-repository-facts";
    const styleContent = `
      .md-source__facts,
      .md-source__repository > ul {
        display: none !important;
      }
    `;

    const ensureStyle = () => {
      if (document.getElementById(styleId)) {
        return;
      }

      if (!document.head) {
        return;
      }

      const style = document.createElement("style");
      style.id = styleId;
      style.textContent = styleContent;
      document.head.appendChild(style);
    };

    const stripFacts = () => {
      document
        .querySelectorAll(".md-source__facts, .md-source__repository > ul")
        .forEach((node) => node.remove());

      document
        .querySelectorAll(".md-source__repository--active")
        .forEach((node) =>
          node.classList.remove("md-source__repository--active")
        );
    };

    const install = () => {
      if (!document.documentElement || !document.head) {
        requestAnimationFrame(install);
        return;
      }

      ensureStyle();
      stripFacts();

      const observer = new MutationObserver(() => {
        ensureStyle();
        stripFacts();
      });

      observer.observe(document.documentElement, {
        subtree: true,
        childList: true,
        attributes: true,
        attributeFilter: ["class"],
      });
    };

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", install, { once: true });
      return;
    }

    install();
  });
}

async function removeRepositoryFacts(page: Page): Promise<void> {
  await page.evaluate(async () => {
    const stripFacts = () => {
      document
        .querySelectorAll(".md-source__facts, .md-source__repository > ul")
        .forEach((node) => node.remove());

      document
        .querySelectorAll(".md-source__repository--active")
        .forEach((node) =>
          node.classList.remove("md-source__repository--active")
        );
    };

    stripFacts();

    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => {
        stripFacts();
        requestAnimationFrame(() => {
          stripFacts();
          resolve();
        });
      });
    });
  });
}

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
  await stubRepositoryFacts(page);
  await suppressRepositoryFacts(page);

  if (theme) {
    await setTheme(page, theme);
  }

  const navigationPath = path === "/" ? "./" : path.replace(/^\//, "");
  await page.goto(navigationPath, { waitUntil: "domcontentloaded" });

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
  await removeRepositoryFacts(page);
}

export async function waitForPageStable(page: Page): Promise<void> {
  await page.waitForLoadState("load");
  await page.waitForLoadState("networkidle").catch(() => {});
  await page.evaluate(async () => {
    const nextFrame = () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      });

    const settleFrames = async (count: number) => {
      for (let index = 0; index < count; index += 1) {
        await nextFrame();
      }
    };

    const decodeImage = async (img: HTMLImageElement) => {
      if (!img.complete) {
        await new Promise<void>((resolve) => {
          const finish = () => resolve();
          img.addEventListener("load", finish, { once: true });
          img.addEventListener("error", finish, { once: true });
        });
      }

      if ("decode" in img && img.naturalWidth > 0) {
        try {
          await img.decode();
        } catch {
          // SVG-backed or lazy-decoded images can reject decode even after load.
        }
      }
    };

    await Promise.all(Array.from(document.images).map((img) => decodeImage(img)));

    if ("fonts" in document) {
      await document.fonts.ready;
    }

    const maybeMathJax = (
      window as Window & {
        MathJax?: {
          startup?: { promise?: Promise<unknown> };
        };
      }
    ).MathJax;

    if (document.querySelector(".arithmatex")) {
      try {
        await maybeMathJax?.startup?.promise;
      } catch {
        // Math-specific audits will report the actual failure details.
      }

      for (let attempt = 0; attempt < 60; attempt += 1) {
        const allTypeset = Array.from(document.querySelectorAll(".arithmatex")).every(
          (node) => !!node.querySelector("mjx-container")
        );
        if (allTypeset) {
          break;
        }
        await settleFrames(1);
      }
    }

    const trackedElements = Array.from(
      document.querySelectorAll<HTMLElement>(
        "main, figure.diagram, figure.diagram img, .snapshot-grid, .md-content img"
      )
    );

    let previousSignature = "";
    let stableFrames = 0;

    for (let attempt = 0; attempt < 10 && stableFrames < 3; attempt += 1) {
      await settleFrames(1);

      const signature = JSON.stringify(
        trackedElements.map((element) => {
          const rect = element.getBoundingClientRect();
          const image = element as HTMLImageElement;
          return [
            Math.round(rect.x),
            Math.round(rect.y),
            Math.round(rect.width),
            Math.round(rect.height),
            image.currentSrc || image.src || "",
            element.scrollWidth,
            element.scrollHeight,
          ];
        }).concat([
          [
            document.documentElement.scrollHeight,
            document.body?.scrollHeight ?? 0,
            document.images.length,
          ],
        ])
      );

      if (signature === previousSignature) {
        stableFrames += 1;
      } else {
        previousSignature = signature;
        stableFrames = 1;
      }
    }

    await settleFrames(trackedElements.length > 0 ? 3 : 2);
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
