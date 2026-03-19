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

import {
  pageSnapshotProjectNames,
  pageSnapshotReviewConfigs,
  screenshotName,
  themes
} from "./targets";

for (const theme of themes) {
  for (const reviewConfig of pageSnapshotReviewConfigs) {
    test(`${reviewConfig.path} renders in ${theme}`, async ({ page }, testInfo) => {
      test.skip(
        !pageSnapshotProjectNames.has(testInfo.project.name),
        "Representative page baselines only cover the blocking mobile and desktop widths by default."
      );

      await gotoAndStabilize(page, reviewConfig.path, theme);

      await assertNoMissingPage(page);
      await assertImagesLoaded(page);
      await assertThemeDiagramVariants(page, theme);
      await assertNoDomOverflow(page);
      await assertNoTinyVisibleContainers(page);
      await assertNoMeaningfulOverlaps(page);

      await expectMainScreenshot(
        page,
        screenshotName(reviewConfig.path, theme, testInfo.project.name)
      );
    });
  }
}
