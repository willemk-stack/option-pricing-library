import { test } from "@playwright/test";

import {
    assertImagesLoaded,
    assertNoMissingPage,
    gotoAndStabilize,
} from "./helpers";
import {
    assertNoBlockingFindings,
    attachAuditFindings,
    startRuntimeIssueTracking,
} from "./audits";
import { pageReviewConfigs, themes } from "./targets";

for (const theme of themes) {
    for (const reviewConfig of pageReviewConfigs) {
        test(`${reviewConfig.path} smoke in ${theme}`, async ({ page }, testInfo) => {
            const tracker = startRuntimeIssueTracking(page);

            await gotoAndStabilize(page, reviewConfig.path, theme);
            await assertNoMissingPage(page);
            await assertImagesLoaded(page);

            const runtimeFindings = tracker.stop();
            await attachAuditFindings(testInfo, "runtime-findings", runtimeFindings);
            assertNoBlockingFindings(
                runtimeFindings,
                `Runtime smoke findings for ${reviewConfig.path} in ${theme}`
            );
        });
    }
}