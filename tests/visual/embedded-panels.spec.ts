import { expect, test } from "@playwright/test";
import { assertImagesLoaded, assertNoMissingPage, gotoAndStabilize } from "./helpers";
import { embeddedPanelProjectNames, themes } from "./targets";

type Region = {
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

type RiskAsset = {
  pagePath: string;
  srcIncludes: string;
  regions: Region[];
};

const HIGH_RISK_ASSETS: RiskAsset[] = [
  {
    pagePath: "/",
    srcIncludes: "reviewer_proof_panel",
    regions: [
      { label: "thumb-a", x: 760 / 1600, y: 450 / 900, width: 340 / 1600, height: 150 / 900 },
      { label: "thumb-b", x: 1154 / 1600, y: 450 / 900, width: 340 / 1600, height: 150 / 900 },
      { label: "thumb-c", x: 760 / 1600, y: 628 / 900, width: 734 / 1600, height: 178 / 900 },
    ],
  },
  {
    pagePath: "/performance/",
    srcIncludes: "benchmark_overview",
    regions: [
      { label: "iv-panel", x: 74 / 1600, y: 264 / 1080, width: 452 / 1600, height: 290 / 1080 },
      { label: "pde-panel", x: 574 / 1600, y: 264 / 1080, width: 452 / 1600, height: 290 / 1080 },
      { label: "macro-panel", x: 1074 / 1600, y: 264 / 1080, width: 452 / 1600, height: 290 / 1080 },
    ],
  },
];

type RegionStats = {
  label: string;
  stdev: number;
  uniqueBuckets: number;
  mean: number;
};

async function regionStatsForImage(page: Parameters<typeof test>[0]["page"], srcIncludes: string, regions: Region[]) {
  return await page.evaluate(
    async ({ srcIncludes, regions }) => {
      const img = Array.from(document.querySelectorAll("img"))
        .find((node) => (node as HTMLImageElement).src.includes(srcIncludes)) as HTMLImageElement | undefined;

      if (!img) {
        return { found: false, src: null, stats: [] as RegionStats[] };
      }

      if (!img.complete || img.naturalWidth === 0 || img.naturalHeight === 0) {
        return { found: true, src: img.src, broken: true, stats: [] as RegionStats[] };
      }

      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) {
        return { found: true, src: img.src, broken: false, error: "No 2D canvas context", stats: [] as RegionStats[] };
      }

      ctx.drawImage(img, 0, 0);

      function analyzeRegion(region: Region): RegionStats {
        const sx = Math.max(0, Math.floor(region.x * img.naturalWidth));
        const sy = Math.max(0, Math.floor(region.y * img.naturalHeight));
        const sw = Math.max(1, Math.floor(region.width * img.naturalWidth));
        const sh = Math.max(1, Math.floor(region.height * img.naturalHeight));
        const imageData = ctx.getImageData(sx, sy, sw, sh).data;

        let count = 0;
        let sum = 0;
        let sumSq = 0;
        const buckets = new Set<number>();

        // sample every 4th pixel to keep this fast
        for (let i = 0; i < imageData.length; i += 16) {
          const r = imageData[i];
          const g = imageData[i + 1];
          const b = imageData[i + 2];
          const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          count += 1;
          sum += luma;
          sumSq += luma * luma;
          buckets.add(Math.round(luma / 8));
        }

        const mean = count > 0 ? sum / count : 0;
        const variance = count > 0 ? Math.max(0, sumSq / count - mean * mean) : 0;
        const stdev = Math.sqrt(variance);
        return {
          label: region.label,
          stdev: Number(stdev.toFixed(2)),
          uniqueBuckets: buckets.size,
          mean: Number(mean.toFixed(2)),
        };
      }

      return {
        found: true,
        src: img.src,
        broken: false,
        stats: regions.map(analyzeRegion),
      };
    },
    { srcIncludes, regions }
  );
}

function suspiciousRegion(stats: RegionStats): boolean {
  // Designed to catch placeholder-like or nearly flat blank boxes in the known proof/benchmark ROIs.
  return stats.stdev < 10 || stats.uniqueBuckets < 12;
}

for (const theme of themes) {
  for (const asset of HIGH_RISK_ASSETS) {
    test(`embedded panels look non-blank on ${asset.pagePath} in ${theme} (${asset.srcIncludes})`, async ({ page }, testInfo) => {
      test.skip(
        !embeddedPanelProjectNames.has(testInfo.project.name),
        "Embedded-panel checks only cover the representative desktop width by default."
      );

      await gotoAndStabilize(page, asset.pagePath, theme);
      await assertNoMissingPage(page);
      await assertImagesLoaded(page);

      const result = await regionStatsForImage(page, asset.srcIncludes, asset.regions);
      expect(result.found, `Could not find rendered asset matching ${asset.srcIncludes} on ${asset.pagePath}`).toBeTruthy();
      expect((result as any).broken, `Rendered image for ${asset.srcIncludes} is broken`).not.toBeTruthy();

      const bad = (result.stats || []).filter(suspiciousRegion);
      expect(
        bad,
        `Suspiciously blank-looking embedded regions for ${asset.srcIncludes} on ${asset.pagePath} in ${theme}:\n${JSON.stringify(result, null, 2)}`
      ).toEqual([]);
    });
  }
}
