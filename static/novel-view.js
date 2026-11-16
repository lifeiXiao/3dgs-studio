'use strict';
// One in-flight request at a time; dragging never queues stale GPU renders.
let novelFocalScale=.9,novelRun=null,novelInFlight=false,novelDirty=false,novelTimer=null,
    novelVersion=0,novelObjectURL=null,novelInteraction=false,novelReferencePose=null,novelRetryAt=0,novelDisplayedStep=null;
function unorient(p){switch(detectedAxis()){
    case'z':return[p[0],-p[2],p[1]];case'-z':return[p[0],p[2],-p[1]];
    case'x':return[p[1],-p[0],p[2]];case'-x':return[-p[1],p[0],p[2]];
    case'-y':return[p[0],-p[1],-p[2]];default:return p;
}}
function alignReferenceCamera(redraw=true){
    const camera=currentScene.cameras?.[0];if(!camera)return;
    const matrix=camera.matrix,position=orient(matrix.slice(0,3).map(r=>r[3]));
    const back=orient(matrix.slice(0,3).map(r=>r[2]));
    yaw=Math.atan2(back[0],back[2]);pitch=Math.asin(Math.max(-1,Math.min(1,back[1])));
    distance=Math.max(.1,Math.hypot(...position.map((v,k)=>v-center[k])));
    center=position.map((v,k)=>v-back[k]*distance);
    const metadata=dataset?.cameras?.[0];novelFocalScale=metadata?.fy&&metadata?.height?metadata.fy/metadata.height:1.39;
    novelReferencePose=matrix.map(r=>[...r]);novelVersion++;
    if(redraw)drawScene();
}
function novelPayload(){
    const rect=$('viewport').getBoundingClientRect();
    const animating=typeof sceneIsAnimating==='function'&&sceneIsAnimating();
    const scale=Math.min(1,(animating?960:novelInteraction?640:1280)/Math.max(rect.width,1),1200/Math.max(rect.height,1));
    const width=Math.max(32,Math.round(rect.width*scale)),height=Math.max(32,Math.round(rect.height*scale));
    let matrix=novelReferencePose;
    if(!matrix){
        const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
        const right=unorient([cy,0,-sy]),up=unorient([-sp*sy,cp,-sp*cy]),back=unorient([cp*sy,sp,cp*cy]);
        const position=unorient(center.map((v,k)=>v+distance*[cp*sy,sp,cp*cy][k]));
        matrix=[0,1,2].map(k=>[right[k],up[k],back[k],position[k]]);matrix.push([0,0,0,1]);
    }
    return {c2w:matrix,width,height,snapshot_step:typeof sceneSnapshotStep==='function'?sceneSnapshotStep():null,transparent:false,background:[.035,.045,.055],fx:height*novelFocalScale,fy:height*novelFocalScale,cx:width/2,cy:height/2};
}
function scheduleNovelView(){
    if(currentView!=='scene')return;
    $('novelViewStatus').hidden=false;
    if(!activeRun||!currentStatus?.gaussians){$('novelViewStatus').textContent='相机布局 · 训练生成模型后启用真实三维高斯预览';$('novelViewImage').hidden=true;return;}
    if(novelRun!==activeRun){
        novelRun=activeRun;novelReferencePose=null;novelDisplayedStep=null;novelVersion++;$('novelViewImage').hidden=true;
        $('novelViewStatus').textContent='正在加载真实 Gaussian 模型…';alignReferenceCamera(false);
        // Align the grid and the true image to the same camera on the first frame.
        drawScene();return;
    }
    novelDirty=true;novelVersion++;
    if(novelInFlight||novelTimer)return;
    novelTimer=setTimeout(()=>{novelTimer=null;renderNovelFrame();},novelInteraction?60:15);
}
async function renderNovelFrame(){
    if(!novelDirty||novelInFlight||currentView!=='scene'||!activeRun)return;
    if(Date.now()<novelRetryAt){novelTimer=setTimeout(()=>{novelTimer=null;renderNovelFrame();},500);return;}
    const run=activeRun,version=novelVersion,payload=novelPayload();novelDirty=false;novelInFlight=true;
    try{
        const response=await fetch(`/api/runs/${run}/render-view`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        if(!response.ok){let error;try{error=(await response.json()).detail;}catch{error='真实高斯预览暂时不可用';}throw Object.assign(new Error(error),{status:response.status});}
        const blob=await response.blob();
        if(run!==activeRun||currentView!=='scene')return;
        if(payload.snapshot_step!==(typeof sceneSnapshotStep==='function'?sceneSnapshotStep():null)){novelDirty=true;return;}
        if(version!==novelVersion)novelDirty=true;
        const url=URL.createObjectURL(blob),old=novelObjectURL;novelObjectURL=url;
        const image=$('novelViewImage');image.src=url;image.hidden=false;if(old)URL.revokeObjectURL(old);
        const step=+response.headers.get('X-Model-Step'),count=+response.headers.get('X-Gaussians');novelDisplayedStep=step;
        if(typeof sceneModelShown==='function')sceneModelShown(step);
        $('viewLabel').textContent=`3DGS / ${count.toLocaleString()} GAUSSIANS`;
        $('novelViewStatus').textContent=`真实高斯合成 · 模型快照 step ${step}${novelInteraction?' · 交互预览':''}`;
        $('viewHint').textContent='拖动旋转 · 滚轮缩放 · 完整模型 / SH / 不透明度 / 遮挡';novelRetryAt=0;
    }catch(error){
        if(currentView==='scene')$('novelViewStatus').textContent=error.message;
        novelRetryAt=Date.now()+(error.status===429?150:1200);
        if(error.status===429||error.status===503)novelDirty=true;
    }finally{
        novelInFlight=false;
        if(novelDirty&&currentView==='scene'&&!novelTimer)novelTimer=setTimeout(()=>{novelTimer=null;renderNovelFrame();},novelRetryAt>Date.now()?500:15);
    }
}
$('alignReference').onclick=()=>alignReferenceCamera();
const oldReset=$('resetView').onclick;
$('resetView').onclick=()=>{if(currentView==='scene')alignReferenceCamera();else{novelReferencePose=null;oldReset();}};
canvas.addEventListener('pointerdown',()=>{novelInteraction=true;novelReferencePose=null;});
canvas.addEventListener('pointerup',()=>{novelInteraction=false;scheduleNovelView();});
canvas.addEventListener('pointercancel',()=>{novelInteraction=false;scheduleNovelView();});
canvas.addEventListener('wheel',()=>{novelReferencePose=null;novelInteraction=true;clearTimeout(canvas.novelWheelEnd);canvas.novelWheelEnd=setTimeout(()=>{novelInteraction=false;scheduleNovelView();},160);},{passive:true});
$('upAxis').addEventListener('change',()=>{novelReferencePose=null;scheduleNovelView();});
scheduleNovelView();

$('toggleCameras').disabled=currentView==='scene';$('toggleCameras').title='相机线框和网格请在几何检查查看';$('cameraLegend').hidden=currentView==='scene';

$('toggleCameras').addEventListener('click',()=>{if(currentView==='geometry')novelReferencePose=null;});
