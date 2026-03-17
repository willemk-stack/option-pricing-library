import { expect, test } from "@playwright/test";

import { gotoAndStabilize } from "./helpers";
import {
    componentReviewTargets,
    componentScreenshotName,
    themes,
} from "./targets";

for (const theme of themes) {
    for (const target of componentReviewTargets) {
        test(`${target.path} component ${target.name} in ${theme}`, async ({ page }, testInfo) => {
            await gotoAndStabilize(page, target.path, theme);

            const component = page.locator(target.selector).first();
            await expect(component).toBeVisible();
            await expect(component).toHaveScreenshot(
                componentScreenshotName(target.path, target.name, theme, testInfo.project.name),
                { animations: "disabled" }
            );
        });
    }
}