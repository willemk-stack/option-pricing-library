import { test } from "@playwright/test";
import {
  assertImagesLoaded,
  assertNoMissingPage,
  assertNoDomOverflow,
  assertNoMeaningfulOverlaps,
  assertThemeDiagramVariants,
  assertNoTinyVisibleContainers,
  expectMainScreenshot,
  gotoAndStabilize
} from "./helpers";

import { pages, screenshotName, themes } from "./targets";

for (const theme of themes) {
  for (const path of pages) {
    test(`${path} renders in ${theme}`, async ({ page }, testInfo) => {
      await gotoAndStabilize(page, path, theme);

      await assertNoMissingPage(page);
      await assertImagesLoaded(page);
      await assertThemeDiagramVariants(page, theme);
      await assertNoDomOverflow(page);
      await assertNoTinyVisibleContainers(page);
      await assertNoMeaningfulOverlaps(page);

      await expectMainScreenshot(
        page,
        screenshotName(path, theme, testInfo.project.name)
      );
    });
  }
}