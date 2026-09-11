#!/usr/bin/env node

const { chromium } = require("playwright");

const url = process.argv[2];
if (!url) {
  console.error("usage: node scripts/playwright_consumer_smoke.cjs <url>");
  process.exit(2);
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`console: ${message.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("requestfailed", (request) => {
    errors.push(`requestfailed: ${request.url()} (${request.failure()?.errorText || "unknown"})`);
  });

  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 120000 });
    await page.locator('[data-testid="stApp"]').waitFor({ state: "visible", timeout: 120000 });
    await page.waitForTimeout(2500);
    if (errors.length) {
      throw new Error(errors.join("\n"));
    }
    console.log(`Playwright smoke passed: ${url}`);
  } finally {
    await browser.close();
  }
})().catch((error) => {
  console.error(`Playwright smoke failed: ${error.message}`);
  process.exit(1);
});
