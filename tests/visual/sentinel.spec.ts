import { expect, test } from "@playwright/test";

import { gotoAndStabilize } from "./helpers";
import { componentScreenshotName } from "./targets";

const sentinelProjects = new Set(["chromium-375", "chromium-1280"]);
const sentinelTheme = "light" as const;

const sentinelComponents = [
  { path: "/", name: "home-snapshot-grid", selector: ".snapshot-grid" },
  {
    path: "/performance/",
    name: "performance-snapshot-table",
    selector: "table",
  },
] as const;

for (const target of sentinelComponents) {
  test(`${target.path} sentinel component ${target.name} in ${sentinelTheme}`, async ({ page }, testInfo) => {
    test.skip(
      !sentinelProjects.has(testInfo.project.name),
      "Sentinel only covers representative mobile and desktop widths."
    );

    await gotoAndStabilize(page, target.path, sentinelTheme);
    const component = page.locator(target.selector).first();
    await expect(component).toBeVisible();
    await expect(component).toHaveScreenshot(
      componentScreenshotName(target.path, target.name, sentinelTheme, testInfo.project.name),
      { animations: "disabled" }
    );
  });
}
