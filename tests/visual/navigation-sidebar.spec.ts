import { expect, test, type Page } from "@playwright/test";

import { gotoAndStabilize } from "./helpers";
import { themes } from "./targets";

const singletonPaths = ["/", "/performance/", "/roadmap/"];
const multiOptionPath = "/user_guides/";
const desktopBreakpointWidth = 1280;

async function sidebarDisplay(page: Page, selector: string) {
  return page.locator(selector).evaluate((element) => window.getComputedStyle(element).display);
}

for (const theme of themes) {
  for (const path of singletonPaths) {
    test(`${path} collapses singleton desktop nav in ${theme}`, async ({ page }) => {
      await gotoAndStabilize(page, path, theme);

      const sidebar = page.locator('.md-sidebar--primary[data-nav-singleton="true"]');
      await expect(sidebar).toHaveCount(1);

      const width = page.viewportSize()?.width ?? 0;
      const display = await sidebarDisplay(page, '.md-sidebar--primary[data-nav-singleton="true"]');
      if (width >= desktopBreakpointWidth) {
        expect(display).toBe("none");
      } else {
        expect(display).not.toBe("none");
      }
    });
  }

  test(`${multiOptionPath} keeps multi-option nav in ${theme}`, async ({ page }) => {
    await gotoAndStabilize(page, multiOptionPath, theme);

    await expect(page.locator('.md-sidebar--primary[data-nav-singleton="true"]')).toHaveCount(0);

    const sidebar = page.locator('.md-sidebar--primary[data-md-type="navigation"]');
    await expect(sidebar).toHaveCount(1);

    const width = page.viewportSize()?.width ?? 0;
    if (width >= desktopBreakpointWidth) {
      const display = await sidebarDisplay(page, '.md-sidebar--primary[data-md-type="navigation"]');
      expect(display).not.toBe("none");
    }
  });
}
