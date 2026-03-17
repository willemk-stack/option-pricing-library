import { readFileSync } from "node:fs";
import { resolve } from "node:path";

type ThemeName = "light" | "dark";

type ReviewTargets = {
    pages: string[];
    themes: ThemeName[];
    widths: number[];
    priority_asset_globs: string[];
};

function loadReviewTargets(): ReviewTargets {
    const filePath = resolve(__dirname, "../../scripts/visual_audit/review_targets.json");
    const raw = JSON.parse(readFileSync(filePath, "utf8")) as Partial<ReviewTargets>;

    const pages = Array.isArray(raw.pages) ? raw.pages.filter((value): value is string => typeof value === "string") : [];
    const themes = Array.isArray(raw.themes)
        ? raw.themes.filter((value): value is ThemeName => value === "light" || value === "dark")
        : [];
    const widths = Array.isArray(raw.widths)
        ? raw.widths.filter((value): value is number => Number.isInteger(value) && value > 0)
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
        themes,
        widths,
        priority_asset_globs: priorityAssetGlobs,
    };
}

const reviewTargets = loadReviewTargets();

export const pages = reviewTargets.pages;
export const themes = reviewTargets.themes;
export const widths = reviewTargets.widths;
export const priorityAssetGlobs = reviewTargets.priority_asset_globs;

export function screenshotName(path: string, theme: ThemeName, projectName: string): string {
    const safeName =
        path === "/"
            ? "home"
            : path.replaceAll("/", "-").replaceAll("_", "-").replace(/^-+|-+$/g, "");
    return `${safeName}-${theme}-${projectName}.png`;
}