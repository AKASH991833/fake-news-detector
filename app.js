/* fake-news-detector — logistic regression on bag-of-words, trained in the browser.
   Right-sized model for a small corpus: one weight per word, fully explainable. */
(function(){
'use strict';
const $=id=>document.getElementById(id);
const STOP=new Set('a an the is are was were i you he she it we they me him her us them my your his their our this that these those and or but if then so of to in on at for with by from as be been have has had do does did will would can could should not no'.split(' '));
const NEGATORS=new Set(['not','never','no','hardly','barely',"didn't","doesn't","don't","wasn't","isn't","couldn't","won't","cant","can't","without"]);
function tokenize(t){return t.toLowerCase().replace(/[^a-z0-9'\s]/g,' ').split(/\s+/).filter(w=>w.length>1)}
function stem(w){if(w.length<=3)return w;
  for(const s of['ingly','edly','ing','ed','ly','es','s'])if(w.endsWith(s)&&w.length-s.length>=3)return w.slice(0,w.length-s.length);
  return w}
// feature extraction: stem + merge negations ("did not work" -> not_work)
function feats(t){
  const ws=tokenize(t),out=[];
  for(let i=0;i<ws.length;i++){
    if(NEGATORS.has(ws[i])&&i+1<ws.length){out.push('not_'+stem(ws[i+1]));i++;continue}
    if(!STOP.has(ws[i]))out.push(stem(ws[i]));
  }
  return out;
}
let VOCAB=[],model=null,trained=false,VOCAB_SIZE=0,metrics=null;
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function seededShuffle(arr,seed){const r=mulberry32(seed);for(let i=arr.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[arr[i],arr[j]]=[arr[j],arr[i]]}return arr}
function buildVocab(){
  const freq={};
  [...POS_TEXTS,...NEG_TEXTS].forEach(t=>new Set(feats(t)).forEach(f=>freq[f]=(freq[f]||0)+1));
  VOCAB=Object.keys(freq).filter(f=>freq[f]>=2); // drop one-off words: they only memorize
  VOCAB_SIZE=VOCAB.length+2;
}
function vectorize(text){
  const v=new Array(VOCAB_SIZE).fill(0);
  feats(text).forEach(f=>{const i=VOCAB.indexOf(f);if(i>=0)v[i+2]=1});
  return {v,words:tokenize(text)};
}
function createModel(){
  const m=tf.sequential();
  m.add(tf.layers.dense({units:1,inputShape:[VOCAB_SIZE],activation:'sigmoid',
    kernelRegularizer:tf.regularizers.l2({l2:0.01})}));
  m.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});
  return m;
}
const EPOCHS=300,BATCH=32,FOLDS=5;
async function fitModel(rows){
  const m=createModel();
  const x=tf.tensor2d(rows.map(d=>vectorize(d.t).v));
  const y=tf.tensor1d(rows.map(d=>d.y));
  await m.fit(x,y,{epochs:EPOCHS,batchSize:BATCH,shuffle:true,verbose:0});
  x.dispose();y.dispose();
  return m;
}
async function predictProbs(m,rows){
  const x=tf.tensor2d(rows.map(d=>vectorize(d.t).v));
  const pv=await m.predict(x).data();
  x.dispose();
  return Array.from(pv);
}
// 5-fold cross-validation: every text scored by a model that never trained on it
async function crossValidate(prog){
  const data=[...POS_TEXTS.map(t=>({t,y:1})),...NEG_TEXTS.map(t=>({t,y:0}))];
  seededShuffle(data,42);
  const oof=new Array(data.length);
  for(let k=0;k<FOLDS;k++){
    const te=data.filter((_,i)=>i%FOLDS===k),tr=data.filter((_,i)=>i%FOLDS!==k);
    const m=await fitModel(tr);
    const pv=await predictProbs(m,te);
    te.forEach((d,i)=>oof[data.indexOf(d)]=pv[i]);
    m.dispose();
    if(prog)prog(`Cross-validating… fold ${k+1}/${FOLDS}`);
  }
  let tp=0,tn=0,fp=0,fn=0;
  oof.forEach((p,i)=>{const c=p>=0.5?1:0,y=data[i].y;if(c===1&&y===1)tp++;else if(c===0&&y===0)tn++;else if(c===1)fp++;else fn++});
  const prec=tp/(tp+fp||1),rec=tp/(tp+fn||1);
  return{acc:(tp+tn)/data.length,prec,rec,f1:2*prec*rec/(prec+rec||1),n:data.length,tp,tn,fp,fn};
}
async function train(prog){
  metrics=await crossValidate(prog);
  const data=[...POS_TEXTS.map(t=>({t,y:1})),...NEG_TEXTS.map(t=>({t,y:0}))];
  model=await fitModel(data);
  trained=true;
  renderMetrics();
}
function renderMetrics(){
  if(!metrics)return;
  $('m-acc').textContent=(metrics.acc*100).toFixed(1)+'%';
  $('m-f1').textContent=(metrics.f1*100).toFixed(1)+'%';
  $('m-n').textContent=metrics.n;
  $('m-conf').innerHTML=`<tr><th></th><th>Predicted +</th><th>Predicted −</th></tr>
  <tr><th>Actually +</th><td class="cm-ok">${metrics.tp}</td><td class="cm-bad">${metrics.fn}</td></tr>
  <tr><th>Actually −</th><td class="cm-bad">${metrics.fp}</td><td class="cm-ok">${metrics.tn}</td></tr>`;
}
// word-level contribution via occlusion: score change when the word is removed
async function explain(text,baseProb){
  const {words}=vectorize(text);
  const contribs=[];
  for(let i=0;i<words.length;i++){
    if(STOP.has(words[i]))continue;
    const w2=words.slice();w2.splice(i,1);
    const {v}=vectorize(w2.join(' '));
    const t=tf.tensor2d([v]);
    const p=(await model.predict(t).data())[0];
    t.dispose();
    contribs.push({w:words[i],d:baseProb-p});
  }
  return contribs;
}
async function analyze(){
  if(!trained){pkToast('Model is still loading, one moment…');return}
  const text=$('msg').value.trim();
  if(!text){pkToast('Type or paste some text first');return}
  const {v}=vectorize(text);
  const t=tf.tensor2d([v]);
  const p=(await model.predict(t).data())[0];
  t.dispose();
  const hasNeg=/\b(not|never|no|hardly|barely|without)\b/.test(text.toLowerCase());
  document.getElementById('result').classList.remove('hidden');
  const label=p>=0.6?['Likely fake','bad']:p<=0.4?['Likely real','ok']:['Unclear','warn'];
  $('verdict').innerHTML=`<span class="pill ${label[1]}">${label[0]}</span>`;
  $('prob-num').textContent=(p*100).toFixed(1)+'%';
  $('prob-meter').style.width=(p*100)+'%';
  $('prob-meter').style.background=p>=0.5?'var(--bad)':'var(--ok)';
  $('prob-note').textContent='Fake-news score — 50% means the model is unsure. Style signals only; always check the source.';
  const cs=await explain(text,p);
  const esc=text.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  const posW=new Map(cs.filter(c=>c.d>0.01).map(c=>[c.w,c.d]));
  const negW=new Map(cs.filter(c=>c.d<-0.01).map(c=>[c.w,-c.d]));
  $('highlighted').innerHTML=esc.split(/\b/).map(part=>{
    const lw=part.toLowerCase();
    if(posW.has(lw))return `<mark class="hl-bad" title="+${posW.get(lw).toFixed(2)}">${part}</mark>`;
    if(negW.has(lw))return `<mark class="hl-ok" title="-${negW.get(lw).toFixed(2)}">${part}</mark>`;
    return part;
  }).join('');
  const topPos=cs.filter(c=>c.d>0.01).sort((a,b)=>b.d-a.d).slice(0,6);
  const topNeg=cs.filter(c=>c.d<-0.01).sort((a,b)=>a.d-b.d).slice(0,6);
  $('pos-words').innerHTML=topPos.map(c=>`<span class="pill bad">${c.w}</span>`).join(' ')||'';
  $('neg-words').innerHTML=topNeg.map(c=>`<span class="pill ok">${c.w}</span>`).join(' ')||'';
}
async function batch(){
  if(!trained){pkToast('Model is still loading, one moment…');return}
  const raw=$('batch-in').value.trim();if(!raw){pkToast('Paste texts, one per line');return}
  const lines=raw.split('\n').map(l=>l.trim()).filter(Boolean);
  const rows=[];
  for(const l of lines){
    const {v}=vectorize(l);const t=tf.tensor2d([v]);
    const p=(await model.predict(t).data())[0];t.dispose();
    rows.push({l,p});
  }
  $('batch-out').classList.remove('hidden');
  $('batch-tbl').innerHTML=rows.map((r,i)=>{
    const lab=r.p>=0.6?'<span class="pill bad">Fake</span>':r.p<=0.4?'<span class="pill ok">Real</span>':'<span class="pill warn">Unclear</span>';
    return `<tr><td>${i+1}</td><td>${r.l.length>80?r.l.slice(0,80)+'…':r.l}</td><td>${lab}</td><td class="mono">${(r.p*100).toFixed(0)}%</td></tr>`}).join('');
  const nP=rows.filter(r=>r.p>=0.6).length,nN=rows.filter(r=>r.p<=0.4).length;
  $('batch-sum').innerHTML=`<span class="pill bad">${nP} fake</span> <span class="pill ok">${nN} real</span> <span class="pill warn">${rows.length-nP-nN} unclear</span>`;
  $('batch-dl').classList.remove('hidden');
  $('batch-dl').onclick=()=>pkDownload('fake-news-results.csv',pkCsv([['headline','verdict','fake_score'],...rows.map(r=>[r.l,r.p>=0.6?'fake':r.p<=0.4?'real':'unclear',r.p.toFixed(3)])]),'text/csv');
}
function readyUI(source){
  trained=true;
  $('stat-vocab').textContent=VOCAB.length.toLocaleString();
  $('stat-texts').textContent=(POS_TEXTS.length+NEG_TEXTS.length).toLocaleString();
  $('stat-params').textContent=model.countParams().toLocaleString();
  $('stat-dim').textContent='LogReg';
  $('train-prog').innerHTML='<span class="pill ok">Model ready</span> '+source;
  renderMetrics();
}
function base64ToBuf(b64){const bin=atob(b64);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u.buffer}
function bufToBase64(buf){const u=new Uint8Array(buf);let s='';for(let i=0;i<u.length;i+=8192)s+=String.fromCharCode.apply(null,u.subarray(i,i+8192));return btoa(s)}
// one-off export path (?export=1): CV metrics + final model on all data -> window.EXPORT
async function exportRun(){
  buildVocab();
  metrics=await crossValidate(t=>{$('train-prog').textContent=t});
  const data=[...POS_TEXTS.map(t=>({t,y:1})),...NEG_TEXTS.map(t=>({t,y:0}))];
  model=await fitModel(data);
  const art=await model.save(tf.io.withSaveHandler(async a=>({modelTopology:a.modelTopology,weightSpecs:a.weightSpecs,weightData:bufToBase64(a.weightData)})));
  art.metrics=metrics;
  art.vocab=VOCAB;
  window.EXPORT=JSON.stringify(art);
  $('train-prog').textContent='EXPORT READY';
}
async function boot(){
  buildVocab();
  if(location.search.includes('export'))return exportRun();
  try{
    const r=await fetch('weights.json');
    if(r.ok){
      const art=await r.json();
      art.weightData=base64ToBuf(art.weightData);
      model=await tf.loadLayersModel(tf.io.fromMemory(art));
      metrics=art.metrics||null;
      readyUI('Pre-trained weights loaded — runs fully offline.');
      return;
    }
  }catch(e){console.warn('weights load failed, training live',e)}
  train(t=>{$('train-prog').textContent=t}).then(()=>readyUI('Trained live in your browser — cross-validated metrics below.')).catch(e=>{$('train-prog').textContent='Model failed: '+e.message;console.error(e)});
}
document.addEventListener('DOMContentLoaded',()=>{
  $('analyze').onclick=analyze;
  $('batch-btn').onclick=batch;
  $('sample').onclick=()=>{const s=['SHOCKING: doctors hate this one weird trick that melts belly fat overnight','The city council approved the new public transit budget in a 7-2 vote after a six-hour session','Miracle cure big pharma doesn\'t want you to know about, doctors are furious','The health ministry confirmed 12 new cases and said contact tracing is underway'];
    $('msg').value=s[Math.floor(Math.random()*s.length)];analyze()};
  const rb=$('retrain');
  if(rb)rb.onclick=async()=>{
    if(rb.disabled)return;rb.disabled=true;
    $('train-prog').textContent='Retraining in your browser…';
    try{await train(t=>{$('train-prog').textContent=t});readyUI('Retrained live in your browser — cross-validated metrics below.')}catch(e){$('train-prog').textContent='Training failed: '+e.message}
    rb.disabled=false;
  };
  boot();
});
})();
