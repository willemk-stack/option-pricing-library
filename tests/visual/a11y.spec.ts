import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { assertNoMissingPage, gotoAndStabilize } from "./helpers";
import {
    a11yViolationsToFindings,
    assertNoBlockingFindings,
    recordAuditFindings,
} from "./audits";
import { pages, themes } from "./targets";

const SUITE_NAME = "a11y.spec.ts";

for (const theme of themes) {
    for (const path of pages) {
        test(`a11y ${path} in ${theme}`, async ({ page }, testInfo) => {
            await gotoAndStabilize(page, path, theme);
            await assertNoMissingPage(page);

            const results = await new AxeBuilder({ page })
                .include("main")
                .analyze();

            const findings = a11yViolationsToFindings(path, results.violations);
            await recordAuditFindings(testInfo, {
                name: "a11y-findings",
                suite: SUITE_NAME,
                route: path,
                theme,
                findings,
            });

            assertNoBlockingFindings(findings, `Accessibility findings for ${path} in ${theme}`);
            expect(results.violations).toEqual([]);
        });
    }
}