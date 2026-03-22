import { test } from "@playwright/test";

import type { AuditFinding } from "./audits";
import {
  assertNoBlockingFindings,
  attachAuditFindings,
  startRuntimeIssueTracking,
} from "./audits";
import { assertNoMissingPage, gotoAndStabilize } from "./helpers";
import { loadMathRoutes } from "./math-targets";
import { themes } from "./targets";

const TEX_SOURCE_PATTERN = /\\[A-Za-z]+|\\[()[\]]|\$\$/;

async function collectMathAuditFindings(
  page: Parameters<typeof gotoAndStabilize>[0],
  route: string
): Promise<AuditFinding[]> {
  return page.evaluate(
    ({ currentRoute, texSourcePattern }) => {
      type Finding = AuditFinding;

      const findings: Finding[] = [];
      const sourcePattern = new RegExp(texSourcePattern);

      const pushFinding = (
        severity: Finding["severity"],
        category: Finding["category"],
        rule: string,
        message: string,
        details?: unknown
      ): void => {
        if (findings.length >= 60) {
          return;
        }
        findings.push({ severity, category, rule, message, details });
      };

      const isVisible = (element: Element | null): element is HTMLElement => {
        if (!(element instanceof HTMLElement)) {
          return false;
        }

        const style = window.getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          style.opacity !== "0" &&
          rect.width > 0 &&
          rect.height > 0
        );
      };

      const mathNodes = Array.from(document.querySelectorAll(".arithmatex")).filter(
        (node): node is HTMLElement => isVisible(node)
      );

      if (mathNodes.length === 0) {
        pushFinding(
          "critical",
          "content",
          "missing-visible-math",
          `Expected visible math on ${currentRoute}, but no visible .arithmatex nodes remained after load`
        );
        return findings;
      }

      mathNodes.forEach((mathNode, index) => {
        const label = `${currentRoute} math block ${index + 1}`;
        const container = mathNode.querySelector("mjx-container");

        if (!(container instanceof HTMLElement)) {
          pushFinding(
            "critical",
            "content",
            "math-not-typeset",
            `${label} does not contain a rendered MathJax container`
          );
          return;
        }

        const jax = container.getAttribute("jax");
        if (jax !== "SVG") {
          pushFinding(
            "critical",
            "content",
            "unexpected-mathjax-renderer",
            `${label} rendered with ${jax || "unknown"} instead of SVG`,
            { route: currentRoute, jax }
          );
        }

        const svg = container.querySelector("svg");
        if (!(svg instanceof SVGSVGElement)) {
          pushFinding(
            "critical",
            "content",
            "missing-svg-output",
            `${label} is missing SVG output inside the MathJax container`
          );
        }

        const renderedText = (container.textContent || "").replace(/\s+/g, " ").trim();
        if (sourcePattern.test(renderedText)) {
          pushFinding(
            "critical",
            "content",
            "rendered-math-contains-tex-source",
            `${label} still contains TeX-like source after rendering`,
            { route: currentRoute, renderedText: renderedText.slice(0, 160) }
          );
        }

        const clone = mathNode.cloneNode(true) as HTMLElement;
        clone.querySelectorAll("mjx-container").forEach((node) => node.remove());
        const leftoverText = (clone.textContent || "").replace(/\s+/g, " ").trim();
        if (leftoverText && sourcePattern.test(leftoverText)) {
          pushFinding(
            "critical",
            "content",
            "raw-tex-leftover",
            `${label} leaves raw TeX source visible outside the MathJax output`,
            { route: currentRoute, leftoverText: leftoverText.slice(0, 160) }
          );
        }

        if (mathNode.tagName.toLowerCase() !== "div") {
          return;
        }

        const style = window.getComputedStyle(mathNode);
        const horizontalOverflow = mathNode.scrollWidth - mathNode.clientWidth > 2;
        if (horizontalOverflow && !["auto", "scroll"].includes(style.overflowX)) {
          pushFinding(
            "major",
            "layout",
            "display-math-overflow",
            `${label} overflows horizontally without scroll containment`,
            {
              route: currentRoute,
              clientWidth: mathNode.clientWidth,
              scrollWidth: mathNode.scrollWidth,
              overflowX: style.overflowX,
            }
          );
        }

        const verticalOverflow = mathNode.scrollHeight - mathNode.clientHeight > 2;
        if (verticalOverflow && style.overflowY === "hidden") {
          pushFinding(
            "major",
            "layout",
            "display-math-clipped",
            `${label} appears vertically clipped`,
            {
              route: currentRoute,
              clientHeight: mathNode.clientHeight,
              scrollHeight: mathNode.scrollHeight,
              overflowY: style.overflowY,
            }
          );
        }
      });

      return findings;
    },
    { currentRoute: route, texSourcePattern: TEX_SOURCE_PATTERN.source }
  );
}

for (const theme of themes) {
  test(`math audits in ${theme}`, async ({ page }, testInfo) => {
    const allFindings: AuditFinding[] = [];

    for (const route of loadMathRoutes()) {
      await test.step(`${route} in ${theme}`, async () => {
        const runtimeTracker = startRuntimeIssueTracking(page);
        await gotoAndStabilize(page, route, theme);
        await assertNoMissingPage(page);

        const findings = [
          ...runtimeTracker.stop(),
          ...(await collectMathAuditFindings(page, route)),
        ];
        allFindings.push(...findings);
      });
    }

    await attachAuditFindings(testInfo, "math-audit-findings", allFindings);
    assertNoBlockingFindings(allFindings, `Math audit findings in ${theme}`);
  });
}
