const {chromium} = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
(async () => {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1050}});
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const base = process.env.BASE_URL || "http://127.0.0.1:27814/";
  await page.addInitScript(() => {
    const original = window.fetch;
    window.fetch = function(url, options) {
      if (options?.body instanceof FormData) {
        window.sentFields = Object.fromEntries(
          [...options.body.entries()].filter(([,value]) => typeof value === "string")
        );
      }
      return original.call(this, url, options);
    };
  });
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector("#status-dot").classList.contains("online"));
  assert.equal(await page.locator("#submit").isDisabled(), true);
  if (process.env.SCREENSHOTS) {
    fs.mkdirSync(process.env.SCREENSHOTS, {recursive: true});
    await page.screenshot({path: process.env.SCREENSHOTS + "/desktop.png", fullPage: true});
  }
  const data = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 96; canvas.height = 64;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#216a52"; ctx.fillRect(0, 0, 96, 64);
    ctx.fillStyle = "#f0e8c0"; ctx.fillRect(3, 4, 50, 30);
    return canvas.toDataURL("image/png").split(",")[1];
  });
  const file = {name: "synthetic.png", mimeType: "image/png", buffer: Buffer.from(data, "base64")};
  await page.locator("#image-file").setInputFiles(file);
  await page.locator("#preview").waitFor({state: "visible"});
  for (const id of ["x","y","w","h"]) assert.equal(await page.locator("#" + id).inputValue(), "");
  for (const [id,value] of Object.entries({x:3,y:4,w:50,h:30})) {
    await page.locator("#"+id).fill(String(value));
  }
  assert.equal(await page.locator("#submit").isEnabled(), true);
  await page.locator("#canvas").click();
  assert.equal(await page.locator("#x").inputValue(), "3");
  await page.locator("#w").fill("999");
  assert.equal(await page.locator("#submit").isDisabled(), true);
  await page.locator("#w").fill("50");
  const requestPromise = page.waitForRequest(r => r.url().endsWith("/api/v1/search"));
  await page.locator("#submit").click();
  await requestPromise;
  const fields = await page.evaluate(() => window.sentFields);
  for (const [id,value] of Object.entries({x:3,y:4,w:50,h:30})) {
    assert.equal(fields[id], String(value));
  }
  await page.locator("#result-data").waitFor({state:"visible"});
  assert.equal(await page.locator("#matches .match-card").count(), 10);
  assert.equal(await page.locator("#error").isHidden(), true);
  await page.waitForFunction(() => [...document.querySelectorAll("#matches img")].every(img => img.complete && img.naturalWidth > 0));
  await page.locator(".match-photo").first().click();
  assert.equal(await page.locator("#photo-dialog").isVisible(), true);
  await page.locator("#close-photo").click();
  if (process.env.SCREENSHOTS) await page.screenshot({path:process.env.SCREENSHOTS + "/cards-desktop.png",fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  if (process.env.SCREENSHOTS) await page.screenshot({path:process.env.SCREENSHOTS + "/cards-mobile.png",fullPage:true});
  await page.setViewportSize({width:1440,height:1050});
  await page.route("**/api/v1/search", route => route.fulfill({
    status:429,contentType:"application/json",body:JSON.stringify({request_id:"busy-test"})
  }));
  await page.locator("#submit").click();
  await page.locator("#error").waitFor({state:"visible"});
  assert.ok((await page.locator("#error").textContent()).includes("busy-test"));
  assert.equal(await page.locator("#result-data").isHidden(), true);
  await page.unroute("**/api/v1/search");
  await page.locator("#image-file").setInputFiles({
    name:"bad.jpg",mimeType:"image/jpeg",buffer:Buffer.from("bad")
  });
  await page.waitForFunction(() => !document.querySelector("#error").hidden);
  assert.equal(await page.locator("#submit").isDisabled(), true);
  await page.reload();
  await page.waitForFunction(() => document.querySelector("#status-dot").classList.contains("online"));
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  if (process.env.SCREENSHOTS) {
    await page.screenshot({path: process.env.SCREENSHOTS + "/mobile.png", fullPage:true});
  }
  assert.deepEqual(errors, []);
  await browser.close();
  console.log("Frontend browser smoke passed: upload, manual bbox, exact request, results, errors, mobile.");
})().catch(error => { console.error(error); process.exit(1); });
