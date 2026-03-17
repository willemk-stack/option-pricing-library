import { test } from "@playwright/test";

import { assertNoMissingPage, gotoAndStabilize } from "./helpers";
import {
    assertNoBlockingFindings,
    attachAuditFindings,
    collectDomAuditFindings,
} from "./audits";
import { pageReviewConfigs, themes } from "./targets";

for (const theme of themes) {
    for (const reviewConfig of pageReviewConfigs) {
        test(`${reviewConfig.path} DOM audits in ${theme}`, async ({ page }, testInfo) => {
            await gotoAndStabilize(page, reviewConfig.path, theme);
            await assertNoMissingPage(page);

            const findings = await collectDomAuditFindings(page, reviewConfig);
            await attachAuditFindings(testInfo, "dom-audit-findings", findings);
            assertNoBlockingFindings(
                findings,
                `DOM audit findings for ${reviewConfig.path} in ${theme}`
            );
        });
    }
}