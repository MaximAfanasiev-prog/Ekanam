const {chromium} = require("playwright");
const assert = require("node:assert/strict");
(async () => {
  const browser = await chromium.launch({headless:true});
  const errors = [];
  const base = process.env.BASE_URL || "http://127.0.0.1:27816/";
  const names = ["x","y","w","h"];
  async function prepare(options) {
    const page = await browser.newPage(options);
    page.on("pageerror", e => errors.push(e.message));
    await page.goto(base);
    await page.waitForFunction(() => document.querySelector("#status-dot").classList.contains("online"));
    const data = await page.evaluate(() => {
      const canvas = document.createElement("canvas");
      canvas.width=1920; canvas.height=1080;
      const ctx=canvas.getContext("2d");
      ctx.fillStyle="#386a65"; ctx.fillRect(0,0,1920,1080);
      return canvas.toDataURL("image/png").split(",")[1];
    });
    await page.locator("#image-file").setInputFiles({name:"bbox.png",mimeType:"image/png",buffer:Buffer.from(data,"base64")});
    await page.locator("#preview").waitFor({state:"visible"});
    await page.locator("#canvas").scrollIntoViewIfNeeded();
    return page;
  }
  async function region(page) {
    const rect = await page.locator("#canvas").boundingBox();
    const scale = Math.min(rect.width/1920,rect.height/1080);
    return {x:rect.x+(rect.width-1920*scale)/2,y:rect.y+(rect.height-1080*scale)/2,w:1920*scale,h:1080*scale};
  }
  async function values(page) {
    return Promise.all(names.map(name => page.locator("#"+name).inputValue()));
  }
  async function check(page, expected) {
    const actual=(await values(page)).map(Number);
    actual.forEach((value,i) => assert.ok(Math.abs(value-expected[i])<=2, actual+" != "+expected));
    assert.equal(await page.locator("#submit").isEnabled(),true);
  }
  async function drag(page, from, to) {
    const r=await region(page);
    await page.mouse.move(r.x+r.w*from[0],r.y+r.h*from[1]);
    await page.mouse.down();
    await page.mouse.move(r.x+r.w*to[0],r.y+r.h*to[1],{steps:8});
    await page.mouse.up();
  }
  const page=await prepare({viewport:{width:1440,height:1050}});
  await drag(page,[0.25,0.25],[0.75,0.75]);
  await check(page,[480,270,960,540]);
  await drag(page,[0.75,0.75],[0.25,0.25]);
  await check(page,[480,270,960,540]);
  const saved=await values(page);
  await page.locator("#canvas").click();
  assert.deepEqual(await values(page),saved);
  const r=await region(page);
  await page.mouse.move(r.x+r.w/2,r.y+r.h/2); await page.mouse.down();
  await page.mouse.move(r.x+r.w*0.9,r.y+r.h*0.9);
  await page.keyboard.press("Escape"); await page.mouse.up();
  assert.deepEqual(await values(page),saved);
  await drag(page,[0.25,0.25],[1.1,1.1]);
  await check(page,[480,270,1440,810]);
  await page.locator("#x").fill("400");
  assert.equal(await page.locator("#x").inputValue(),"400");
  await drag(page,[0.1,0.2],[0.4,0.6]);
  await check(page,[192,216,576,432]);
  await page.evaluate(() => {
    const original=window.fetch;
    window.fetch=function(url,opts) {
      if(opts?.body instanceof FormData) window.sentBBox=Object.fromEntries([...opts.body].filter(([,v])=>typeof v==="string"));
      return original.call(this,url,opts);
    };
  });
  const chosen=await values(page);
  await page.locator("#submit").click();
  await page.locator("#result-data").waitFor({state:"visible"});
  const sent=await page.evaluate(()=>window.sentBBox);
  names.forEach((name,i)=>assert.equal(sent[name],chosen[i]));
  await drag(page,[0.2,0.2],[0.5,0.5]);
  assert.equal(await page.locator("#result-data").isHidden(),true);
  const mobile=await prepare({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
  const m=await region(mobile);
  const cdp=await mobile.context().newCDPSession(mobile);
  const touch=(x,y)=>({x:m.x+m.w*x,y:m.y+m.h*y,id:1});
  await cdp.send("Input.dispatchTouchEvent",{type:"touchStart",touchPoints:[touch(0.2,0.2)]});
  await cdp.send("Input.dispatchTouchEvent",{type:"touchMove",touchPoints:[touch(0.8,0.8)]});
  await cdp.send("Input.dispatchTouchEvent",{type:"touchEnd",touchPoints:[]});
  await check(mobile,[384,216,1152,648]);
  const before=await values(mobile);
  await cdp.send("Input.dispatchTouchEvent",{type:"touchStart",touchPoints:[touch(0.3,0.3)]});
  await cdp.send("Input.dispatchTouchEvent",{type:"touchMove",touchPoints:[touch(0.6,0.6)]});
  await cdp.send("Input.dispatchTouchEvent",{type:"touchCancel",touchPoints:[]});
  assert.deepEqual(await values(mobile),before);
  assert.deepEqual(errors,[]);
  await browser.close();
  console.log("BBox browser smoke passed: scaled and reversed drag, bounds, click, Escape, manual editing, exact HTTP fields, touch and cancellation.");
})().catch(error=>{console.error(error);process.exit(1);});
