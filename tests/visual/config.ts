import { readFileSync } from "node:fs";
import { resolve } from "node:path";

type DocsVisualConfig = {
    docs_base_url: string;
    build_profile: string;
};

const configPath = resolve(__dirname, "../../scripts/visual_audit/docs_visual_config.json");
const config = JSON.parse(readFileSync(configPath, "utf8")) as DocsVisualConfig;

export function normalizeBaseURL(rawUrl: string): string {
    return rawUrl.endsWith("/") ? rawUrl : `${rawUrl}/`;
}

export const defaultDocsBaseURL = normalizeBaseURL(config.docs_base_url);
export const defaultDocsBuildProfile = config.build_profile;