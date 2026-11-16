// Execute the actual page handlers with a small DOM fixture; no browser dependency.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const noop=()=>{},elements=new Map(),callbacks=new Map();let clock=1000,nextId=0;
function element(id){
    if(elements.has(id))return elements.get(id);
    const el={id,hidden:false,disabled:false,value:'',textContent:'',style:{},dataset:{},
        classList:{toggle:noop},options:[{},{},{}],events:{},addEventListener(type,handler){this.events[type]=handler;},setAttribute:noop,
        getBoundingClientRect:()=>({left:0,top:0,width:1160,height:440}),setPointerCapture:noop,
        get parentElement(){return element(id+'-parent');},
        querySelector:()=>element(id+'-option'),replaceChildren(...children){this.children=children;this.options=children;},append(...children){this.children=(this.children||[]).concat(children);},
        getContext:()=>new Proxy({},{get:()=>noop,set:()=>true})};
    elements.set(id,el);return el;
}
for(const id of fs.readFileSync('static/index.html','utf8').matchAll(/id="([^"]+)"/g))element(id[1]);
element('replaySpeed').value='250';element('sceneAnimationMode').value='orbit';element('upAxis').value='auto';
const context={console,document:{getElementById:element,querySelectorAll:()=>[],createElement:()=>element('created-'+(++nextId))},
    window:{devicePixelRatio:1,addEventListener:noop},ResizeObserver:class{observe(){}},
    setTimeout:()=>++nextId,clearTimeout:noop,setInterval:noop,requestAnimationFrame:fn=>{callbacks.set(++nextId,fn);return nextId;},
    cancelAnimationFrame:id=>callbacks.delete(id),performance:{now:()=>clock},
    URL:{createObjectURL:()=>`blob:${++nextId}`,revokeObjectURL:noop}};
vm.createContext(context);
const run=code=>vm.runInContext(code,context);
const app=fs.readFileSync('static/app.js','utf8');
run(app.slice(0,app.indexOf('action(async()=>{const [sys,preset]')));
run(fs.readFileSync('static/novel-view.js','utf8'));
run(fs.readFileSync('static/scene-playback.js','utf8'));
run(fs.readFileSync('static/chart-hover.js','utf8'));
function tick(ms){clock+=ms;run(`tickScenePlayback(${clock})`);}
async function test(){
    // Old runs without 3D history still play a full, multi-angle orbit in tab 1.
    run("activeRun='legacy';currentStatus={gaussians:368426,step:30000};novelRun=activeRun;updateReplayControls()");
    assert.equal(element('playReplay').disabled,false);
    await element('playReplay').onclick();
    assert.equal(run('currentView'),'scene');assert.equal(run('sceneIsAnimating()'),true);
    const before=run('novelPayload()');for(let i=0;i<90;i++)tick(100);
    const after=run('novelPayload()');
    assert.notDeepEqual(before.c2w,after.c2w);assert.equal(after.snapshot_step,null);
    assert.ok(Math.abs(run('yaw')-.45)>3,'camera visits the opposite side');
    // A manual drag pauses automatic movement; releasing resumes from the new pose.
    run('drag=[1,1];yaw=2;pitch=.6');tick(100);assert.equal(run('yaw'),2);
    run('drag=null');tick(100);assert.ok(run('yaw')>2);
    await element('playReplay').onclick();assert.equal(run('sceneIsAnimating()'),false);
    element('replaySlider').value='180';element('replaySlider').oninput();
    assert.equal(run('currentView'),'scene');assert.ok(Math.abs(run('yaw')-Math.PI)<1e-9);
    // Evolution requests only genuine full snapshots and waits until they render.
    run('replayFrames=[{step:0,model:true},{step:500,model:false},{step:1000,model:true}];');
    element('sceneAnimationMode').value='evolution';element('sceneAnimationMode').onchange();
    await element('playReplay').onclick();assert.equal(run('sceneSnapshotStep()'),0);
    for(let i=0;i<15;i++)tick(100);
    assert.equal(run('sceneSnapshotStep()'),0,'do not skip models while loading');
    run('novelDisplayedStep=0;sceneModelShown(0)');for(let i=0;i<11;i++)tick(100);
    assert.equal(run('sceneSnapshotStep()'),1000);assert.equal(run('novelPayload().snapshot_step'),1000);
    assert.equal(run('currentView'),'scene');
    // A direct Step switch pauses animation and keeps the exact same camera.
    const cameraBefore=run('novelPayload()');
    element('sceneStepSelect').value='0';element('sceneStepSelect').onchange();
    assert.equal(run('sceneSnapshotStep()'),0);assert.equal(run('sceneIsAnimating()'),false);
    const cameraAfter=run('novelPayload()');
    assert.deepEqual(cameraAfter.c2w,cameraBefore.c2w);
    assert.equal(cameraAfter.fy/cameraAfter.height,cameraBefore.fy/cameraBefore.height);
    assert.equal(run('currentView'),'scene');
    assert.deepEqual(element('sceneStepSelect').options.map(option=>option.value),['latest','0','1000']);
    // A selected stage remains pinned while orbiting and while new frames arrive.
    element('sceneAnimationMode').value='orbit';element('sceneAnimationMode').onchange();
    assert.equal(run('sceneSnapshotStep()'),0);
    await element('playReplay').onclick();tick(100);
    assert.equal(run('novelPayload().snapshot_step'),0);
    run('replayFrames.push({step:2000,model:true});updateReplayControls()');
    assert.equal(element('sceneStepSelect').value,'0');
    assert.deepEqual(element('sceneStepSelect').options.map(option=>option.value),['latest','0','1000','2000']);
    const choices=element('sceneStepSelect').options;run('updateScenePlaybackControls()');
    assert.equal(element('sceneStepSelect').options,choices,'do not rebuild menu on each animation frame');
    element('sceneStepSelect').value='latest';element('sceneStepSelect').onchange();
    assert.equal(run('sceneSnapshotStep()'),null);assert.equal(run('sceneIsAnimating()'),false);
    element('sceneStepSelect').value='1000';element('sceneStepSelect').onchange();
    // Late responses for a previously selected history frame are discarded.
    let requested;
    context.fetch=async(_url,options)=>{
        requested=JSON.parse(options.body);
        run('sceneFrameIndex=0');
        return {ok:true,blob:async()=>({}),headers:{get:key=>key==='X-Model-Step'?'1000':'1234'}};
    };
    run('novelDirty=true;novelTimer=null');await run('renderNovelFrame()');
    assert.equal(requested.snapshot_step,1000);assert.equal(run('novelDisplayedStep'),0);
    // Changing tabs stops orbit; fixed-render playback retains its original function.
    run("switchView('render')");assert.equal(run('sceneIsAnimating()'),false);
    assert.equal(element('sceneStepSelect').hidden,true);
    assert.equal(element('replaySlider').step,1);
    await element('playReplay').onclick();assert.equal(run('currentView'),'render');
    assert.equal(run('replayPlaying'),true);
    run("switchView('scene')");assert.equal(run('replayPlaying'),false);
    run('sceneGoLive()');assert.equal(run('sceneSnapshotStep()'),null);
    // Quality charts retain fixed-camera comparisons in both display modes.
    run("$('metricMode').value='fixed';fixedMetrics=[{step:0,train_loss:.4,train_psnr:8,train_ssim:.2,val_ssim:.15,train_lpips:.5,val_lpips:.6},{step:1000,train_loss:.1,train_psnr:25,train_ssim:.9,val_ssim:.85,train_lpips:.1,val_lpips:.15}];renderMetrics()");
    assert.equal(element('ssimTrend').textContent,'0.8500');assert.equal(element('lpipsTrend').textContent,'0.1500');
    for(const [id,name,train,val]of [['ssimChart','SSIM','0.9000','0.8500'],['lpipsChart','LPIPS','0.1000','0.1500']]){
        const canvas=element(id);canvas.events.pointermove({clientX:1000,clientY:100});
        assert.equal(canvas.metricHoverStep,1000);assert.equal(canvas.metricTooltip.hidden,false);
        assert.ok(canvas.metricTooltip.children[0].textContent.includes(name));
        assert.ok(canvas.metricTooltip.children[0].textContent.includes('1,000'));
        assert.equal(canvas.metricTooltip.children[1].children[1].textContent,train);
        assert.equal(canvas.metricTooltip.children[2].children[1].textContent,val);
        canvas.events.keydown({key:'ArrowLeft',preventDefault:noop});assert.equal(canvas.metricHoverStep,0);
        canvas.events.pointerleave();assert.equal(canvas.metricTooltip.hidden,true);
    }
    run("$('metricMode').value='train';rawMetrics=[{step:1000,loss:.08,loss_ema:.1,psnr:24}];renderMetrics()");
    assert.equal(element('ssimTrend').textContent,'0.8500');assert.equal(element('lpipsTrend').textContent,'0.1500');
    run('fixedMetrics=[{step:0,train_loss:.4,train_psnr:8}];renderMetrics()');
    assert.equal(element('ssimTrend').textContent,'—');assert.equal(element('lpipsTrend').textContent,'—');
    console.log('PASS: multi-angle playback, exact full-model Step selection, fixed camera comparisons, and late-response handling.');
}
test().catch(error=>{console.error(error);process.exitCode=1;});
