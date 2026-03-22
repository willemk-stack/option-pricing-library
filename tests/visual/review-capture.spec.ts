import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

import { expect, test } from "@playwright/test";

import { gotoAndStabilize } from "./helpers";
import {
    componentReviewTargets,
    componentScreenshotName,
    pageReviewConfigs,
    screenshotName,
    themes,
} from "./targets";

const captureDir = process.env.IMPROVEMENT_CAPTURE_DIR
    ? resolve(process.env.IMPROVEMENT_CAPTURE_DIR)
    : "";

function ensureParentDir(path: string): void {
    mkdirSync(dirname(path), { recursive: true });
}

for (const theme of themes) {
    for (const reviewConfig of pageReviewConfigs) {
        test(`${reviewConfig.path} review capture in ${theme}`, async ({ page }, testInfo) => {
            test.skip(!captureDir, "IMPROVEMENT_CAPTURE_DIR is required for review capture runs");

            await gotoAndStabilize(page, reviewConfig.path, theme);

            const main = page.locator("main");
            await expect(main).toBeVisible();

            const screenshotPath = join(
                captureDir,
                screenshotName(reviewConfig.path, theme, testInfo.project.name)
            );
            ensureParentDir(screenshotPath);
            await main.screenshot({ path: screenshotPath, animations: "disabled" });
        });
    }
}

for (const theme of themes) {
    for (const target of componentReviewTargets) {
        test(`${target.path} review component capture ${target.name} in ${theme}`, async ({ page }, testInfo) => {
            test.skip(!captureDir, "IMPROVEMENT_CAPTURE_DIR is required for review capture runs");

            await gotoAndStabilize(page, target.path, theme);

            const component = page.locator(target.selector).first();
            await expect(component).toBeVisible();

            const screenshotPath = join(
                captureDir,
                componentScreenshotName(target.path, target.name, theme, testInfo.project.name)
            );
            ensureParentDir(screenshotPath);
            await component.screenshot({ path: screenshotPath, animations: "disabled" });
        });
    }
}