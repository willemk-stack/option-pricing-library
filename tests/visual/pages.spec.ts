import { test } from "@playwright/test";
import {
  assertImagesLoaded,
  assertNoDomOverflow,
  assertNoMeaningfulOverlaps,
  assertNoTinyVisibleContainers,
  expectMainScreenshot,
  gotoAndStabilize
} from "./helpers";

const paths = [
  "/",
  "/architecture/",
  "/performance/",
  "/user_guides/decision_guide/",
  "/user_guides/localvol_pde_validation/"
];

for (const theme of ["light", "dark"] as const) {
  for (const path of paths) {
    test(`${path} renders in ${theme}`, async ({ page }, testInfo) => {
      await gotoAndStabilize(page, path, theme);

      await assertImagesLoaded(page);
      await assertNoDomOverflow(page);
      await assertNoTinyVisibleContainers(page);
      await assertNoMeaningfulOverlaps(page);

      const safeName =
        path === "/" ? "home" : path.replaceAll("/", "_").replace(/^_+|_+$/g, "");

      await expectMainScreenshot(
        page,
        `${safeName}-${theme}-${testInfo.project.name}.png`
      );
    });
  }
}