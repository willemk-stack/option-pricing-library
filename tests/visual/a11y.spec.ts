import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { gotoAndStabilize } from "./helpers";

const paths = [
    "/",
    "/architecture/",
    "/performance/",
    "/user_guides/decision_guide/",
    "/user_guides/localvol_pde_validation/"
];

for (const theme of ["light", "dark"] as const) {
    for (const path of paths) {
        test(`a11y ${path} in ${theme}`, async ({ page }) => {
            await gotoAndStabilize(page, path, theme);

            const results = await new AxeBuilder({ page })
                .include("main")
                .analyze();

            expect(results.violations).toEqual([]);
        });
    }
}