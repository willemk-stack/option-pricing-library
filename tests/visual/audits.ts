import { expect, type Page, type TestInfo } from "@playwright/test";

import type { PageReviewConfig } from "./targets";

const DEFAULT_DOCS_BASE_URL = "http://127.0.0.1:8000/option-pricing-library/";

export type AuditSeverity = "critical" | "major" | "minor";
export type AuditCategory = "console" | "content" | "layout" | "media" | "theme";

export type AuditFinding = {
    severity: AuditSeverity;
    category: AuditCategory;
    rule: string;
    message: string;
    details?: unknown;
};

type RuntimeIssueTracker = {
    stop: () => AuditFinding[];
};

function shouldIgnoreConsoleMessage(message: string): boolean {
    const normalized = message.toLowerCase();
    return (
        normalized.includes("favicon") ||
        normalized.includes("content security policy") ||
        normalized.startsWith("failed to load resource")
    );
}

function isTrackedLocalUrl(url: string): boolean {
    const docsBaseUrl = process.env.DOCS_BASE_URL || DEFAULT_DOCS_BASE_URL;
    return (
        url.startsWith(docsBaseUrl) ||
        url.startsWith("http://127.0.0.1:8000/") ||
        url.startsWith("http://localhost:8000/")
    );
}

function shouldIgnoreResponseUrl(url: string): boolean {
    const normalized = url.toLowerCase();
    return normalized.endsWith("/favicon.ico") || normalized.endsWith("favicon.ico");
}

export function startRuntimeIssueTracking(page: Page): RuntimeIssueTracker {
    const findings: AuditFinding[] = [];

    const onConsole = (message: { type: () => string; text: () => string }): void => {
        if (message.type() !== "error") {
            return;
        }

        const text = message.text();
        if (shouldIgnoreConsoleMessage(text)) {
            return;
        }

        findings.push({
            severity: "major",
            category: "console",
            rule: "console-error",
            message: `Console error: ${text}`,
        });
    };

    const onPageError = (error: Error): void => {
        findings.push({
            severity: "major",
            category: "console",
            rule: "page-error",
            message: `Uncaught page error: ${error.message}`,
        });
    };

    const onResponse = (response: { status: () => number; url: () => string }): void => {
        const status = response.status();
        const url = response.url();

        if (status < 400) {
            return;
        }
        if (!isTrackedLocalUrl(url) || shouldIgnoreResponseUrl(url)) {
            return;
        }

        findings.push({
            severity: "major",
            category: "console",
            rule: "http-response-error",
            message: `HTTP ${status} while loading ${url}`,
        });
    };

    page.on("console", onConsole);
    page.on("pageerror", onPageError);
    page.on("response", onResponse);

    return {
        stop: () => {
            page.off("console", onConsole);
            page.off("pageerror", onPageError);
            page.off("response", onResponse);
            return findings.slice(0, 25);
        },
    };
}

export async function collectDomAuditFindings(
    page: Page,
    reviewConfig: PageReviewConfig
): Promise<AuditFinding[]> {
    return page.evaluate((config: PageReviewConfig) => {
        type Finding = {
            severity: AuditSeverity;
            category: AuditCategory;
            rule: string;
            message: string;
            details?: unknown;
        };

        type Candidate = {
            text: string;
            className: string;
            interactive: boolean;
            rect: {
                left: number;
                top: number;
                right: number;
                bottom: number;
                width: number;
                height: number;
            };
        };

        const findings: Finding[] = [];

        const pushFinding = (
            severity: AuditSeverity,
            category: AuditCategory,
            rule: string,
            message: string,
            details?: unknown
        ): void => {
            if (findings.length >= 40) {
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

        const documentRoot = document.documentElement;
        if (documentRoot.scrollWidth - documentRoot.clientWidth > 2) {
            pushFinding(
                "major",
                "layout",
                "unexpected-horizontal-scroll",
                `Page scroll width ${documentRoot.scrollWidth}px exceeds viewport width ${documentRoot.clientWidth}px`
            );
        }

        const mainHeading = document.querySelector("main article > h1");
        if (!isVisible(mainHeading)) {
            pushFinding(
                "critical",
                "content",
                "missing-or-hidden-primary-heading",
                "Primary page heading is missing or hidden"
            );
        } else {
            const headingText = mainHeading.innerText.trim();
            if (!headingText) {
                pushFinding(
                    "critical",
                    "content",
                    "empty-primary-heading",
                    "Primary page heading is empty"
                );
            }

            const headingOverflow =
                mainHeading.scrollWidth - mainHeading.clientWidth > 2 ||
                mainHeading.scrollHeight - mainHeading.clientHeight > 2;
            if (headingOverflow) {
                pushFinding(
                    "critical",
                    "content",
                    "clipped-primary-heading",
                    `Primary heading overflows its container: ${headingText.slice(0, 120)}`
                );
            }
        }

        const brokenImages = Array.from(document.querySelectorAll("img"))
            .filter(
                (img) =>
                    !(img as HTMLImageElement).complete ||
                    (img as HTMLImageElement).naturalWidth === 0
            )
            .map((img) => ({
                src: (img as HTMLImageElement).src,
                alt: (img as HTMLImageElement).alt,
            }));

        for (const brokenImage of brokenImages.slice(0, 10)) {
            pushFinding(
                "critical",
                "media",
                "broken-image",
                `Broken image detected: ${brokenImage.src || "<missing src>"}`,
                brokenImage
            );
        }

        if (config.requiresVisualEvidence) {
            const hasVisibleMedia = config.essentialMediaSelectors.some((selector) =>
                Array.from(document.querySelectorAll(selector)).some((node) => isVisible(node))
            );

            if (!hasVisibleMedia) {
                pushFinding(
                    "critical",
                    "media",
                    "missing-essential-media",
                    `Expected visible visual evidence on ${config.path}, but none of the required selectors matched`,
                    { selectors: config.essentialMediaSelectors }
                );
            }
        }

        const root =
            document.querySelector(".md-content__inner") ||
            document.querySelector("main") ||
            document.body;

        const overflowNodes = Array.from(root.querySelectorAll("*"))
            .filter((node): node is HTMLElement => node instanceof HTMLElement)
            .filter((node) => {
                const style = window.getComputedStyle(node);
                if (style.display === "inline") return false;
                if (style.textOverflow === "ellipsis") return false;
                if (node.classList.contains("md-ellipsis")) return false;
                if (!node.innerText?.trim()) return false;
                if (node.children.length > 0) return false;
                return (
                    node.scrollWidth - node.clientWidth > 2 ||
                    node.scrollHeight - node.clientHeight > 2
                );
            })
            .slice(0, 10);

        for (const node of overflowNodes) {
            pushFinding(
                "major",
                "layout",
                "text-overflow",
                `Leaf content overflows its container: ${node.innerText.trim().slice(0, 120)}`,
                {
                    tag: node.tagName.toLowerCase(),
                    className: node.className,
                }
            );
        }

        for (const selector of config.emptyContainerSelectors) {
            const nodes = Array.from(document.querySelectorAll(selector));
            for (const node of nodes) {
                if (!isVisible(node)) {
                    continue;
                }

                const rect = node.getBoundingClientRect();
                const text = node.textContent?.trim() || "";
                const hasVisualChild = !!node.querySelector("img, svg, canvas, picture, figure");

                if ((rect.height < 24 || rect.width < 40) && (text || hasVisualChild)) {
                    pushFinding(
                        "major",
                        "layout",
                        "tiny-visible-container",
                        `Visible content container is suspiciously small for selector ${selector}`,
                        {
                            selector,
                            width: Math.round(rect.width),
                            height: Math.round(rect.height),
                        }
                    );
                    continue;
                }

                if (rect.width >= 120 && rect.height >= 60 && !text && !hasVisualChild) {
                    pushFinding(
                        "major",
                        "content",
                        "visually-empty-container",
                        `Visible container for selector ${selector} appears empty`,
                        { selector }
                    );
                }
            }
        }

        const diagrams = Array.from(document.querySelectorAll("figure.diagram"));
        for (const figure of diagrams) {
            const light = Array.from(figure.querySelectorAll("img.diagram-light"));
            const dark = Array.from(figure.querySelectorAll("img.diagram-dark"));
            if (light.length === 0 && dark.length === 0) {
                continue;
            }

            const visibleLight = light.filter((node) => isVisible(node)).length;
            const visibleDark = dark.filter((node) => isVisible(node)).length;

            if (visibleLight + visibleDark !== 1) {
                pushFinding(
                    "major",
                    "theme",
                    "theme-diagram-visibility",
                    "Theme-aware diagram does not resolve to exactly one visible variant",
                    {
                        visibleLight,
                        visibleDark,
                        figure: (figure.textContent || "").trim().slice(0, 120),
                    }
                );
            }
        }

        const selectors = [
            "h1",
            "h2",
            "h3",
            "h4",
            "p",
            "li",
            "figcaption",
            "td",
            "th",
            "blockquote",
            ".metric-number",
            ".metric-value",
            ".snapshot-label",
            ".snapshot-copy",
            "a.md-button",
            "button",
        ].join(",");

        const hiddenOrIgnoredAncestorSelector = [
            "[hidden]",
            "[aria-hidden='true']",
            ".md-header",
            ".md-sidebar",
            ".md-search",
            ".md-search__inner",
            ".md-overlay",
            ".md-dialog",
            ".md-tabs",
            ".md-footer",
            ".md-top",
            "nav",
        ].join(",");

        const hasVisibleTextChild = (element: HTMLElement): boolean =>
            Array.from(element.children).some((child) => {
                if (!(child instanceof HTMLElement)) {
                    return false;
                }

                return isVisible(child) && (child.innerText || "").trim().length > 0;
            });

        const overlapArea = (a: Candidate["rect"], b: Candidate["rect"]): number => {
            const width = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
            const height = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
            return width * height;
        };

        const area = (rect: Candidate["rect"]): number => rect.width * rect.height;

        const candidates: Candidate[] = Array.from(root.querySelectorAll<HTMLElement>(selectors))
            .filter((element) => !element.closest(hiddenOrIgnoredAncestorSelector))
            .filter((element) => isVisible(element))
            .filter((element) => !hasVisibleTextChild(element))
            .map((element) => {
                const rect = element.getBoundingClientRect();
                return {
                    text: (element.innerText || "").trim().replace(/\s+/g, " ").slice(0, 120),
                    className: (element.className || "").toString(),
                    interactive:
                        element.matches("a, button, input, select, textarea") ||
                        element.getAttribute("role") === "button",
                    rect: {
                        left: rect.left,
                        top: rect.top,
                        right: rect.right,
                        bottom: rect.bottom,
                        width: rect.width,
                        height: rect.height,
                    },
                };
            });

        for (let index = 0; index < candidates.length; index += 1) {
            for (let innerIndex = index + 1; innerIndex < candidates.length; innerIndex += 1) {
                const a = candidates[index];
                const b = candidates[innerIndex];
                const overlap = overlapArea(a.rect, b.rect);
                const smallerArea = Math.min(area(a.rect), area(b.rect));

                if (overlap <= 0 || smallerArea <= 0) {
                    continue;
                }
                if (overlap < 12 || overlap / smallerArea < 0.25) {
                    continue;
                }

                pushFinding(
                    a.interactive || b.interactive ? "critical" : "major",
                    "layout",
                    "overlapping-elements",
                    `Elements overlap substantially: ${a.text || a.className} <-> ${b.text || b.className}`,
                    {
                        first: a.className,
                        second: b.className,
                    }
                );
            }
        }

        return findings;
    }, reviewConfig);
}

export async function attachAuditFindings(
    testInfo: TestInfo,
    name: string,
    findings: AuditFinding[]
): Promise<void> {
    await testInfo.attach(name, {
        body: Buffer.from(JSON.stringify(findings, null, 2), "utf8"),
        contentType: "application/json",
    });
}

export function assertNoBlockingFindings(
    findings: AuditFinding[],
    label: string
): void {
    const blocking = findings.filter((finding) => finding.severity !== "minor");
    expect(blocking, `${label}\n${JSON.stringify(blocking, null, 2)}`).toEqual([]);
}