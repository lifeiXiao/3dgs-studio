'use strict';
// Real 3D playback has its own camera and model timeline. It never changes tabs.
let scenePlaybackPlaying=false,scenePlaybackRAF=null,sceneFrameIndex=null,
    sceneLastTick=0,scenePitchBase=0,sceneOrbitPhase=0,sceneShownStep=null,sceneShownAt=0,
    scenePlaybackRun=null,sceneModeChosen=false,sceneStepOptionsKey='';
const sceneModelFrames=()=>replayFrames.filter(frame=>frame.model===true);
const sceneIsAnimating=()=>scenePlaybackPlaying;
const sceneEvolution=()=>$('sceneAnimationMode').value==='evolution';
const sceneSnapshotStep=()=>sceneFrameIndex!==null?sceneModelFrames()[sceneFrameIndex]?.step??null:null;
const sceneAngle=()=>((yaw*180/Math.PI)%360+360)%360;

function updateSceneStepSelector(frames,hasModel){
    const select=$('sceneStepSelect'),key=JSON.stringify([activeRun,frames.map(frame=>frame.step)]);
    select.hidden=false;
    // Keep the open menu stable while the camera animation updates each frame.
    if(key!==sceneStepOptionsKey){
        const choices=[{value:'latest',label:'Step · 最新模型'},...frames.map(frame=>({
            value:String(frame.step),label:`Step ${frame.step.toLocaleString()}${frame.step===0?' · 初始':''}`
        }))];
        select.replaceChildren(...choices.map(choice=>{
            const option=document.createElement('option');option.value=choice.value;option.textContent=choice.label;return option;
        }));
        sceneStepOptionsKey=key;
    }
    const selected=String(sceneSnapshotStep()??'latest');
    if(select.value!==selected)select.value=selected;
    select.disabled=!hasModel||!frames.length;
    select.title=frames.length?'切换真实训练模型；保持当前视角与缩放':'此任务未保存完整三维历史，仅可查看最新模型';
}
function updateScenePlaybackControls(){
    const frames=sceneModelFrames(),hasModel=!!(activeRun&&currentStatus?.gaussians);
    if(scenePlaybackRun!==activeRun){scenePlaybackRun=activeRun;sceneModeChosen=false;sceneFrameIndex=null;}
    if(!sceneModeChosen)$('sceneAnimationMode').value=frames.length>1?'evolution':'orbit';
    const evolution=sceneEvolution();
    updateSceneStepSelector(frames,hasModel);
    $('sceneAnimationMode').hidden=false;
    const option=$('sceneAnimationMode').querySelector('option[value="evolution"]');
    option.disabled=frames.length===0;
    option.textContent=frames.length?`训练演化 + 环绕 · ${frames.length} 帧`:'训练演化 · 此任务无完整历史';
    $('playbackTitle').textContent=evolution?'三维训练演化':'三维动画';
    [...$('replaySpeed').options].forEach((option,i)=>option.textContent=['0.5×','1×','2×'][i]);
    $('playReplay').disabled=!hasModel||(evolution&&!frames.length);
    $('playReplay').textContent=scenePlaybackPlaying?'Ⅱ':'▶';
    $('playReplay').title=evolution?'播放完整历史高斯模型，同时环绕多个视角':'自动环绕完整三维高斯模型，播放中也可拖动和缩放';
    $('replaySlider').disabled=!hasModel||(evolution&&!frames.length);
    $('replaySlider').max=evolution?Math.max(0,frames.length-1):360;
    $('replaySlider').step=evolution?1:.1;
    $('replaySlider').value=evolution?(sceneFrameIndex??Math.max(0,frames.length-1)):sceneAngle();
    $('replaySlider').setAttribute('aria-label',evolution?'完整三维模型训练时间轴':'三维环绕相机角度');
    const requested=sceneSnapshotStep(),actual=novelDisplayedStep;
    $('replayMode').textContent=requested!==null&&actual!==requested?`加载 Step ${requested}…`:evolution?'完整模型 · 自动环绕':scenePlaybackPlaying?'多视角自动环绕':requested!==null?'固定训练阶段 · 可拖动查看':'拖动交互 / 点击播放';
    $('replayStep').textContent=hasModel?`${evolution?'':Math.round(sceneAngle())+'° · '}${actual??currentStatus.step??0} step`:'等待模型';
    $('replayStep').title='显示已实际渲染的模型步数；指标卡片始终显示最新训练评测';
    $('replayLive').textContent='最新模型 ↗';
    $('exportReplay').disabled=replayFrames.length<2;
}
function sceneModelShown(step){
    if(sceneShownStep!==step){sceneShownStep=step;sceneShownAt=performance.now();}
    if(currentView==='scene')updateScenePlaybackControls();
}
function stopScenePlayback(refine=true){
    scenePlaybackPlaying=false;cancelAnimationFrame(scenePlaybackRAF);scenePlaybackRAF=null;
    if(currentView==='scene'){
        $('playReplay').textContent='▶';
        if(refine){updateScenePlaybackControls();scheduleNovelView();}
    }
}
function resetScenePlayback(){
    stopScenePlayback(false);sceneFrameIndex=null;sceneShownStep=null;sceneShownAt=0;sceneOrbitPhase=0;
    $('sceneAnimationMode').value='orbit';
    scenePlaybackRun=null;sceneModeChosen=false;
    sceneStepOptionsKey='';
}
function toggleScenePlayback(){
    if(scenePlaybackPlaying){stopScenePlayback();return;}
    if(!activeRun||!currentStatus?.gaussians)return;
    sceneModeChosen=true;
    stopReplay();replayCursor=null;
    const frames=sceneModelFrames();
    if(sceneEvolution()){
        if(!frames.length){toast('此任务没有完整三维历史；请选择多视角环绕。新训练会自动记录。');return;}
        if(sceneFrameIndex===null||sceneFrameIndex>=frames.length-1)sceneFrameIndex=0;
        sceneShownStep=null;
    }
    novelReferencePose=null;scenePitchBase=pitch;sceneOrbitPhase=0;
    scenePlaybackPlaying=true;sceneLastTick=performance.now();
    updateScenePlaybackControls();scheduleNovelView();
    scenePlaybackRAF=requestAnimationFrame(tickScenePlayback);
}
function tickScenePlayback(now){
    if(!scenePlaybackPlaying)return;
    if(currentView!=='scene'){stopScenePlayback(false);return;}
    const dt=Math.max(0,Math.min((now-sceneLastTick)/1000,.1)),speed=250/+$('replaySpeed').value;
    sceneLastTick=now;
    if(!drag&&!novelInteraction){
        // One orbit in 18 seconds at 1×, with a gentle change in elevation.
        yaw+=dt*Math.PI/9*speed;sceneOrbitPhase+=dt*.28*speed;
        pitch=Math.max(-1.35,Math.min(1.35,scenePitchBase+Math.sin(sceneOrbitPhase)*.12));
    }else scenePitchBase=pitch-Math.sin(sceneOrbitPhase)*.12;
    const frames=sceneModelFrames(),requested=sceneSnapshotStep();
    if(sceneEvolution()&&requested!==null&&sceneShownStep===requested&&now-sceneShownAt>=Math.max(500,+$('replaySpeed').value*4)){
        sceneFrameIndex=(sceneFrameIndex+1)%frames.length;sceneShownStep=null;
    }
    drawScene();updateScenePlaybackControls();
    scenePlaybackRAF=requestAnimationFrame(tickScenePlayback);
}
function scrubScenePlayback(value){
    sceneModeChosen=true;
    stopScenePlayback(false);
    if(sceneEvolution()){
        const frames=sceneModelFrames();if(!frames.length)return;
        sceneFrameIndex=Math.max(0,Math.min(Math.round(value),frames.length-1));sceneShownStep=null;
    }else{yaw=value*Math.PI/180;novelReferencePose=null;}
    updateScenePlaybackControls();drawScene();
}
function sceneGoLive(){
    sceneModeChosen=true;
    stopScenePlayback(false);sceneFrameIndex=null;sceneShownStep=null;
    $('sceneAnimationMode').value='orbit';updateScenePlaybackControls();drawScene();
}
function selectSceneStep(value){
    const frames=sceneModelFrames(),index=frames.findIndex(frame=>String(frame.step)===value);
    if(value!=='latest'&&index<0){updateScenePlaybackControls();return;}
    stopScenePlayback(false);sceneModeChosen=true;replayCursor=null;
    sceneFrameIndex=value==='latest'?null:index;sceneShownStep=null;
    // Change only the model. Preserve yaw, pitch, center, distance, intrinsics,
    // and the exact reference pose so different stages can be compared fairly.
    updateScenePlaybackControls();scheduleNovelView();
}
$('sceneStepSelect').onchange=()=>selectSceneStep($('sceneStepSelect').value);
$('sceneAnimationMode').onchange=()=>{
    sceneModeChosen=true;
    stopScenePlayback(false);if(sceneEvolution()&&sceneFrameIndex===null)sceneFrameIndex=0;sceneShownStep=null;
    updateScenePlaybackControls();scheduleNovelView();
};
updateReplayControls();
