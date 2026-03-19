import { readFileSync } from "node:fs";
import { resolve } from "node:path";

export type ThemeName = "light" | "dark";

type ReviewTargets = {
    pages: string[];
    page_snapshot_pages: string[];
    themes: ThemeName[];
    widths: number[];
    page_snapshot_widths: number[];
    priority_asset_globs: string[];
};

export type PageArchetype =
    | "homepage"
    | "section-landing"
    | "long-form-guide"
    | "api-landing"
    | "visual-report";

export type ComponentReviewTarget = {
    path: string;
    name: string;
    selector: string;
};

export type PageReviewConfig = {
    path: string;
    pageKey: string;
    archetype: PageArchetype;
    requiresVisualEvidence: boolean;
    essentialMediaSelectors: string[];
    emptyContainerSelectors: string[];
    componentShots: ComponentReviewTarget[];
};

function deriveFallbackPageKey(path: string): string {
    if (path === "/") {
        return "homepage";
    }

    return path
        .replaceAll("/", "_")
        .replaceAll("-", "_")
        .replace(/^_+|_+$/g, "") || "page";
}

function deriveFallbackArchetype(path: string): PageArchetype {
    if (path === "/") {
        return "homepage";
    }
    if (path.startsWith("/api/")) {
        return "api-landing";
    }
    if (path.startsWith("/user_guides/")) {
        return "long-form-guide";
    }
    return "section-landing";
}

function buildFallbackPageReviewConfig(path: string): PageReviewConfig {
    return {
        path,
        pageKey: deriveFallbackPageKey(path),
        archetype: deriveFallbackArchetype(path),
        requiresVisualEvidence: false,
        essentialMediaSelectors: ["figure.diagram img", ".snapshot-grid", ".cta-row", "table"],
        emptyContainerSelectors: ["figure.diagram", ".snapshot-grid", ".cta-row", "table", "main article"],
        componentShots: [],
    };
}

function loadReviewTargets(): ReviewTargets {
    const filePath = resolve(__dirname, "../../scripts/visual_audit/review_targets.json");
    const raw = JSON.parse(readFileSync(filePath, "utf8")) as Partial<ReviewTargets>;

    const pages = Array.isArray(raw.pages) ? raw.pages.filter((value): value is string => typeof value === "string") : [];
    const pageSnapshotPages = Array.isArray(raw.page_snapshot_pages)
        ? raw.page_snapshot_pages.filter((value): value is string => typeof value === "string")
        : [];
    const themes = Array.isArray(raw.themes)
        ? raw.themes.filter((value): value is ThemeName => value === "light" || value === "dark")
        : [];
    const widths = Array.isArray(raw.widths)
        ? raw.widths.filter((value): value is number => Number.isInteger(value) && value > 0)
        : [];
    const pageSnapshotWidths = Array.isArray(raw.page_snapshot_widths)
        ? raw.page_snapshot_widths.filter((value): value is number => Number.isInteger(value) && value > 0)
        : [];
    const priorityAssetGlobs = Array.isArray(raw.priority_asset_globs)
        ? raw.priority_asset_globs.filter((value): value is string => typeof value === "string")
        : [];

    if (pages.length === 0) {
        throw new Error("review_targets.json must define at least one docs page");
    }
    if (themes.length === 0) {
        throw new Error("review_targets.json must define at least one theme");
    }
    if (widths.length === 0) {
        throw new Error("review_targets.json must define at least one viewport width");
    }

    return {
        pages,
        page_snapshot_pages: pageSnapshotPages.length > 0 ? pageSnapshotPages : pages,
        themes,
        widths,
        page_snapshot_widths: pageSnapshotWidths.length > 0 ? pageSnapshotWidths : widths,
        priority_asset_globs: priorityAssetGlobs,
    };
}

const reviewTargets = loadReviewTargets();

export const themes = reviewTargets.themes;
export const widths = reviewTargets.widths;
export const priorityAssetGlobs = reviewTargets.priority_asset_globs;
const defaultPageSnapshotPageSet = new Set(reviewTargets.page_snapshot_pages);
const defaultPageSnapshotWidths = reviewTargets.page_snapshot_widths;

const pageReviewConfigOverrides: Record<string, Omit<PageReviewConfig, "path">> = {
    "/": {
        pageKey: "homepage",
        archetype: "homepage",
        requiresVisualEvidence: true,
        essentialMediaSelectors: ["figure.diagram img", ".snapshot-card"],
        emptyContainerSelectors: ["figure.diagram", ".snapshot-card", ".snapshot-grid"],
        componentShots: [
            { path: "/", name: "home-snapshot-grid", selector: ".snapshot-grid" },
            { path: "/", name: "home-proof-panel", selector: "figure.diagram" },
        ],
    },
    "/architecture/": {
        pageKey: "architecture",
        archetype: "section-landing",
        requiresVisualEvidence: true,
        essentialMediaSelectors: ["figure.diagram img"],
        emptyContainerSelectors: ["figure.diagram", "main article"],
        componentShots: [],
    },
    "/performance/": {
        pageKey: "visual_report",
        archetype: "visual-report",
        requiresVisualEvidence: true,
        essentialMediaSelectors: ["figure.diagram img", "table"],
        emptyContainerSelectors: ["figure.diagram", "table"],
        componentShots: [
            { path: "/performance/", name: "performance-overview-panel", selector: "figure.diagram" },
            { path: "/performance/", name: "performance-snapshot-table", selector: "table" },
        ],
    },
    "/user_guides/decision_guide/": {
        pageKey: "decision_guide",
        archetype: "section-landing",
        requiresVisualEvidence: false,
        essentialMediaSelectors: [".cta-row", "table"],
        emptyContainerSelectors: [".cta-row", "table"],
        componentShots: [],
    },
    "/user_guides/surface_workflow/": {
        pageKey: "long_form_guide",
        archetype: "long-form-guide",
        requiresVisualEvidence: true,
        essentialMediaSelectors: ["figure.diagram img", ".snapshot-grid"],
        emptyContainerSelectors: ["figure.diagram", ".snapshot-grid"],
        componentShots: [
            { path: "/user_guides/surface_workflow/", name: "surface-guide-primary-figure", selector: "figure.diagram" },
            { path: "/user_guides/surface_workflow/", name: "surface-guide-figure-grid", selector: ".snapshot-grid" },
        ],
    },
    "/user_guides/essvi_smooth_handoff/": {
        pageKey: "essvi_smooth_handoff",
        archetype: "long-form-guide",
        requiresVisualEvidence: true,
        essentialMediaSelectors: ["figure.diagram img", ".snapshot-grid"],
        emptyContainerSelectors: ["figure.diagram", ".snapshot-grid"],
        componentShots: [],
    },
    "/user_guides/localvol_pde_validation/": {
        pageKey: "localvol_pde_validation",
        archetype: "long-form-guide",
        requiresVisualEvidence: true,
        essentialMediaSelectors: ["figure.diagram img", ".snapshot-grid"],
        emptyContainerSelectors: ["figure.diagram", ".snapshot-grid"],
        componentShots: [],
    },
};

function buildDefaultPageReviewConfig(path: string): PageReviewConfig {
    const override = pageReviewConfigOverrides[path];
    if (!override) {
        return buildFallbackPageReviewConfig(path);
    }

    return {
        path,
        ...override,
    };
}

const allPageReviewConfigs: PageReviewConfig[] = reviewTargets.pages.map((path) =>
    buildDefaultPageReviewConfig(path)
);

function readFilterSet(value: string | undefined): Set<string> {
    if (!value) {
        return new Set();
    }

    return new Set(
        value
            .split(",")
            .map((entry) => entry.trim())
            .filter((entry) => entry.length > 0)
    );
}

const selectedPaths = readFilterSet(process.env.REVIEW_PATHS);
const selectedPageKeys = readFilterSet(process.env.REVIEW_PAGE_KEYS);
const hasReviewFilters = selectedPaths.size > 0 || selectedPageKeys.size > 0;

function filterPageReviewConfigs(
    configs: PageReviewConfig[],
    filteredPaths: Set<string>,
    filteredPageKeys: Set<string>
): PageReviewConfig[] {
    if (filteredPaths.size === 0 && filteredPageKeys.size === 0) {
        return configs;
    }

    const filtered = configs.filter(
        (config) =>
            filteredPaths.has(config.path) ||
            filteredPageKeys.has(config.pageKey)
    );

    const knownPaths = new Set(configs.map((config) => config.path));
    for (const path of filteredPaths) {
        if (knownPaths.has(path)) {
            continue;
        }
        filtered.push(buildFallbackPageReviewConfig(path));
    }

    if (filtered.length === 0) {
        throw new Error(
            `No review targets matched REVIEW_PATHS=${process.env.REVIEW_PATHS || ""} REVIEW_PAGE_KEYS=${process.env.REVIEW_PAGE_KEYS || ""}`
        );
    }

    return filtered;
}

export const pageReviewConfigs = filterPageReviewConfigs(
    allPageReviewConfigs,
    selectedPaths,
    selectedPageKeys
);
export const pages = Array.from(new Set(pageReviewConfigs.map((config) => config.path)));
export const pageSnapshotWidths = hasReviewFilters ? widths : defaultPageSnapshotWidths;
export const pageSnapshotProjectNames = new Set(
    pageSnapshotWidths.map((width) => `chromium-${width}`)
);
export const pageSnapshotReviewConfigs = hasReviewFilters
    ? pageReviewConfigs
    : pageReviewConfigs.filter((config) => defaultPageSnapshotPageSet.has(config.path));

export const componentReviewTargets = pageReviewConfigs.flatMap((config) => config.componentShots);

export function screenshotName(path: string, theme: ThemeName, projectName: string): string {
    const safeName =
        path === "/"
            ? "home"
            : path.replaceAll("/", "-").replaceAll("_", "-").replace(/^-+|-+$/g, "");
    return `${safeName}-${theme}-${projectName}.png`;
}

export function componentScreenshotName(
    path: string,
    componentName: string,
    theme: ThemeName,
    projectName: string
): string {
    const safePageName =
        path === "/"
            ? "home"
            : path.replaceAll("/", "-").replaceAll("_", "-").replace(/^-+|-+$/g, "");
    const safeComponentName = componentName.replaceAll("_", "-").replace(/^-+|-+$/g, "");
    return `${safePageName}-${safeComponentName}-${theme}-${projectName}.png`;
}
