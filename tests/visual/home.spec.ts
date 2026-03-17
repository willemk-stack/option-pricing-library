import { test } from "@playwright/test";
import {
  assertImagesLoaded,
  assertNoDomOverflow,
  assertNoMeaningfulOverlaps,
  assertNoTinyVisibleContainers,
  expectMainScreenshot,
  gotoAndStabilize
} from "./helpers";

for (const theme of ["light", "dark"] as const) {
  test(`home renders cleanly in ${theme}`, async ({ page }, testInfo) => {
    await gotoAndStabilize(page, "/", theme);

    await assertImagesLoaded(page);
    await assertNoDomOverflow(page);
    await assertNoTinyVisibleContainers(page);
    await assertNoMeaningfulOverlaps(page);

    await expectMainScreenshot(
      page,
      `home-${theme}-${testInfo.project.name}.png`
    );
  });
}