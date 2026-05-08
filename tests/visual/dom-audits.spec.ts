import { test } from "@playwright/test";

import { assertNoMissingPage, gotoAndStabilize } from "./helpers";
import {
    assertNoBlockingFindings,
    collectDomAuditFindings,
    recordAuditFindings,
} from "./audits";
import { pageReviewConfigs, themes } from "./targets";

const SUITE_NAME = "dom-audits.spec.ts";

for (const theme of themes) {
    for (const reviewConfig of pageReviewConfigs) {
        test(`${reviewConfig.path} DOM audits in ${theme}`, async ({ page }, testInfo) => {
            await gotoAndStabilize(page, reviewConfig.path, theme);
            await assertNoMissingPage(page);

            const findings = await collectDomAuditFindings(page, reviewConfig);
            await recordAuditFindings(testInfo, {
                name: "dom-audit-findings",
                suite: SUITE_NAME,
                route: reviewConfig.path,
                theme,
                findings,
            });
            assertNoBlockingFindings(
                findings,
                `DOM audit findings for ${reviewConfig.path} in ${theme}`
            );
        });
    }
}