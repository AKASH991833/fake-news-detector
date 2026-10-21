module.exports=async(page)=>{
  await page.waitForFunction(()=>document.querySelector('#train-prog')&&/ready|trained/.test(document.querySelector('#train-prog').textContent),{timeout:120000});
  await page.type('#msg','SHOCKING miracle cure big pharma hides from you, doctors are furious');
  await page.click('#analyze');
  await page.waitForSelector('#result:not(.hidden)',{timeout:20000});
  const v1=await page.$eval('#verdict',e=>e.textContent);
  if(!/fake/i.test(v1))throw new Error('fake not detected: '+v1);
  await page.evaluate(()=>{document.getElementById('msg').value='';document.getElementById('prob-num').textContent='—'});
  await page.type('#msg','The city council approved the new transit budget in a 7-2 vote after a lengthy session on Tuesday');
  await page.click('#analyze');
  await page.waitForFunction(()=>document.querySelector('#prob-num').textContent!=='—',{timeout:20000});
  const v2=await page.$eval('#verdict',e=>e.textContent);
  if(!/real/i.test(v2))throw new Error('real not detected: '+v2);
  const acc=await page.$eval('#m-acc',e=>e.textContent);
  return `fake+real detected, CV acc=${acc}`;
};
