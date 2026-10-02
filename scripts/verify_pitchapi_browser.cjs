/** Repeatable Playwright verification for a running analytics Streamlit app. */
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

async function main() {
  const url = process.argv[2];
  const output = process.argv[3] || 'precomputed/pitchapi-browser';
  const consumer = process.argv[4] === 'consumer';
  if (!url) throw new Error('Usage: node verify_pitchapi_browser.cjs URL [OUTPUT]');
  fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({headless: true});
  const context = await browser.newContext({viewport: {width: 1440, height: 1000}});
  const page = await context.newPage();
  const failures = [];
  page.on('pageerror', error => failures.push(error.message));
  const reports = [];
  try {
    await page.goto(url, {waitUntil: 'domcontentloaded'});
    const assertHealthy = async label => {
      await page.locator('[data-testid="stException"]').count().then(count => {
        if (count) throw new Error(`${label}: Streamlit exception`);
      });
      if (failures.length) throw new Error(`${label}: ${failures.join('; ')}`);
      reports.push({section: label, status: 'passed'});
    };
    if (consumer) {
      await page.locator('[data-testid="stSidebar"]').waitFor({timeout: 60000});
      await page.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), {timeout: 60000});
      await assertHealthy('production-page');
      await page.screenshot({path: path.join(output, 'production-page.png'), fullPage: true});
      await page.locator('[data-testid="stSidebar"]').getByText('Match analytics', {exact: true}).click();
    }
    await page.getByRole('heading', {name: 'Match analytics', exact: true}).waitFor({timeout: 60000});
    for (const section of ['Lineups', 'Forecast history', 'Shots', 'Momentum', 'Team comparison', 'Players', 'Passing network', 'Heatmaps']) {
      await page.getByText(section, {exact: true}).filter({visible: true}).first().click();
      await page.getByRole('heading', {name: section, exact: true}).waitFor({timeout: 30000});
      // Wait for Streamlit to complete the widget rerun before inspecting errors.
      await page.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), {timeout: 30000});
      await assertHealthy(section);
      await page.screenshot({path: path.join(output, section.toLowerCase().replaceAll(' ', '-') + '.png'), fullPage: true});
    }
    await page.setViewportSize({width: 390, height: 844});
    await assertHealthy('mobile');
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    if (consumer) {
      await page.setViewportSize({width: 1440, height: 1000});
      for (const [label, heading] of [['Team analytics', 'Team analytics'], ['Feature validation', 'Feature validation']]) {
        const link = page.locator('[data-testid="stSidebar"]').getByText(label, {exact: true});
        if (await link.count()) {
          await link.click();
          await page.getByRole('heading', {name: heading, exact: true}).waitFor({timeout: 30000});
          await page.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), {timeout: 30000});
          await assertHealthy(label);
          await page.screenshot({path: path.join(output, label.toLowerCase().replaceAll(' ', '-') + '.png'), fullPage: true});
        }
      }
    }
  } finally {
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify({url, reports, failures}, null, 2));
    await browser.close();
  }
  process.stdout.write(JSON.stringify({url, passed: reports.length}) + '\n');
}
main().catch(error => {process.stderr.write(error.stack + '\n'); process.exitCode = 1;});
