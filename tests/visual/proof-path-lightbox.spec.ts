import { expect, test, type Locator, type Page } from "@playwright/test";

import { assertImagesLoaded, assertNoMissingPage, gotoAndStabilize } from "./helpers";
import { themes } from "./targets";

type ProofPathLightboxPage = {
  path: string;
  expectedTriggerCount: number;
};

const proofPathLightboxPages: ProofPathLightboxPage[] = [
  {
    path: "/user_guides/surface_workflow/",
    expectedTriggerCount: 2,
  },
  {
    path: "/user_guides/essvi_smooth_handoff/",
    expectedTriggerCount: 1,
  },
  {
    path: "/user_guides/localvol_pde_validation/",
    expectedTriggerCount: 2,
  },
];

async function openFromTrigger(trigger: Locator, page: Page) {
  await trigger.focus();
  await expect(trigger).toBeFocused();
  await page.keyboard.press("Space");
}

for (const theme of themes) {
  for (const config of proofPathLightboxPages) {
    test(`proof-path support plots open in ${config.path} for ${theme}`, async ({ page }) => {
      await gotoAndStabilize(page, config.path, theme);
      await assertNoMissingPage(page);
      await assertImagesLoaded(page);

      const triggers = page.locator(".proof-path-lightbox-trigger[data-proof-path-lightbox]");
      const figures = page.locator("figure.proof-path-support-figure");
      const overlay = page.locator(".proof-path-lightbox");
      const dialog = overlay.locator(".proof-path-lightbox__dialog");
      const modalImage = overlay.locator(".proof-path-lightbox__image");
      const modalCaption = overlay.locator(".proof-path-lightbox__caption");

      await expect(triggers).toHaveCount(config.expectedTriggerCount);
      await expect(figures).toHaveCount(config.expectedTriggerCount);

      const firstTrigger = triggers.first();
      await expect(firstTrigger).toBeVisible();
      await expect(firstTrigger).toHaveAttribute("aria-haspopup", "dialog");
      await expect(firstTrigger).toHaveAttribute("aria-label", /Open a larger view/i);

      const sourceCaption = await firstTrigger.evaluate((node) =>
        node.closest("figure")?.querySelector("figcaption")?.textContent?.trim() ?? ""
      );
      const expectedSrc = await firstTrigger.getAttribute(theme === "dark" ? "data-dark-src" : "data-light-src");
      expect(expectedSrc).toBeTruthy();

      await openFromTrigger(firstTrigger, page);

      await expect(dialog).toBeVisible();
      await expect(overlay.getByRole("button", { name: "Close enlarged plot" })).toBeFocused();
      await expect(modalImage).toBeVisible();
      await expect(modalCaption).toContainText(sourceCaption);

      const actualSrc = await modalImage.getAttribute("src");
      expect(actualSrc).toBeTruthy();

      const resolvedActualSrc = new URL(actualSrc as string, page.url());
      const resolvedExpectedSrc = new URL(expectedSrc as string, page.url());
      expect(resolvedActualSrc.pathname).toBe(resolvedExpectedSrc.pathname);
      expect(resolvedActualSrc.search).toBe(resolvedExpectedSrc.search);

      await page.keyboard.press("Escape");
      await expect(dialog).toBeHidden();
      await expect(firstTrigger).toBeFocused();
    });
  }

  test(`proof-path lightbox closes via button and backdrop for ${theme}`, async ({ page }) => {
    await gotoAndStabilize(page, "/user_guides/localvol_pde_validation/", theme);
    await assertNoMissingPage(page);
    await assertImagesLoaded(page);

    const trigger = page.locator(".proof-path-lightbox-trigger[data-proof-path-lightbox]").first();
    const overlay = page.locator(".proof-path-lightbox");
    const dialog = overlay.locator(".proof-path-lightbox__dialog");
    const closeButton = overlay.getByRole("button", { name: "Close enlarged plot" });

    await trigger.click();
    await expect(dialog).toBeVisible();
    await closeButton.click();
    await expect(dialog).toBeHidden();
    await expect(trigger).toBeFocused();

    await trigger.click();
    await expect(dialog).toBeVisible();
    await overlay.click({ position: { x: 12, y: 12 } });
    await expect(dialog).toBeHidden();
    await expect(trigger).toBeFocused();
  });
}
