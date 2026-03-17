import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { assertNoMissingPage, gotoAndStabilize } from "./helpers";
import { pages, themes } from "./targets";

for (const theme of themes) {
    for (const path of pages) {
        test(`a11y ${path} in ${theme}`, async ({ page }) => {
            await gotoAndStabilize(page, path, theme);
            await assertNoMissingPage(page);

            const results = await new AxeBuilder({ page })
                .include("main")
                .analyze();

            expect(results.violations).toEqual([]);
        });
    }
}