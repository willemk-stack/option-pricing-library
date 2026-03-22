import { expect, test } from "@playwright/test";

import { mockRepositoryFacts, setTheme, waitForPageStable } from "./helpers";

const repoFactsProjects = new Set(["chromium-1280"]);
const repoFactsTheme = "light" as const;
const mockedFacts = {
  version: "v0.4.0",
  stars: 42,
  forks: 7,
} as const;

test("repository facts render deterministic mocked data", async ({ page }, testInfo) => {
  test.skip(
    !repoFactsProjects.has(testInfo.project.name),
    "Repository facts only need a representative desktop assertion."
  );

  await mockRepositoryFacts(page, mockedFacts);
  await setTheme(page, repoFactsTheme);

  await page.goto("./", { waitUntil: "domcontentloaded" });
  await page.evaluate(() => {
    document.body?.setAttribute("data-md-color-scheme", "default");
  });
  await waitForPageStable(page);

  const source = page.locator(".md-header__source .md-source").first();
  const facts = source.locator(".md-source__facts");

  await expect(source).toBeVisible();
  await expect(source.locator(".md-source__repository")).toContainText("GitHub");
  await expect(facts).toBeVisible();
  await expect(facts.locator(".md-source__fact--version")).toHaveText(mockedFacts.version);
  await expect(facts.locator(".md-source__fact--stars")).toHaveText(
    mockedFacts.stars.toString()
  );
  await expect(facts.locator(".md-source__fact--forks")).toHaveText(
    mockedFacts.forks.toString()
  );
});
