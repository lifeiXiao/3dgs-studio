'use strict';
const chartMetricNames={loss:'Loss',psnr:'PSNR',ssim:'SSIM',lpips:'LPIPS · VGG'};
function chartHoverRow(canvas){
    const m=canvas.metricChart;if(!m||canvas.metricHoverX==null)return null;
    const rows=m.rows.filter(row=>Number.isFinite(row[m.key])||Number.isFinite(row['val_'+m.key]));
    if(!rows.length)return null;
    const maxStep=Math.max(m.rows.at(-1)?.step??0,1);
    const step=Math.max(0,Math.min(1,(canvas.metricHoverX-m.l)/(m.w-m.l-m.r)))*maxStep;
    // Select an actual recorded point rather than inventing an interpolated value.
    let lo=0,hi=rows.length-1;
    while(lo<hi){const mid=(lo+hi)>>1;if(rows[mid].step<step)lo=mid+1;else hi=mid;}
    if(lo>0&&Math.abs(rows[lo-1].step-step)<=Math.abs(rows[lo].step-step))lo--;
    return rows[lo];
}
function paintChartHover(canvas){
    const m=canvas.metricChart,row=chartHoverRow(canvas);
    if(!row){if(canvas.metricTooltip)canvas.metricTooltip.hidden=true;return;}
    const ctx=canvas.getContext('2d'),maxStep=Math.max(m.rows.at(-1)?.step??0,1);
    const x=m.l+(m.w-m.l-m.r)*row.step/maxStep;
    ctx.save();ctx.beginPath();ctx.moveTo(x,m.t);ctx.lineTo(x,m.h-m.b);
    ctx.strokeStyle='#d8e8cd88';ctx.lineWidth=1;ctx.setLineDash([3,3]);ctx.stroke();ctx.setLineDash([]);
    const values=[];
    for(const [field,color,label] of [[m.key,m.color,'训练集'],['val_'+m.key,'#b39bec','验证集']]){
        const value=row[field];if(!Number.isFinite(value))continue;
        const y=m.t+(m.h-m.t-m.b)*(1-(value-m.lo)/(m.hi-m.lo));
        ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#101619';ctx.lineWidth=1.5;ctx.stroke();
        values.push({value,color,label});
    }
    ctx.restore();
    let tip=canvas.metricTooltip;
    if(!tip){tip=document.createElement('div');tip.className='chart-tooltip';tip.setAttribute('role','status');canvas.parentElement.append(tip);canvas.metricTooltip=tip;}
    const title=document.createElement('strong');title.textContent=`${chartMetricNames[m.key]} · Step ${row.step.toLocaleString()}`;
    const lines=values.map(item=>{
        const line=document.createElement('div'),label=document.createElement('span'),value=document.createElement('b');
        label.textContent=item.label;label.style.color=item.color;
        value.textContent=item.value.toFixed(m.key==='psnr'?2:m.key==='loss'?5:4)+(m.key==='psnr'?' dB':'');
        line.append(label,value);return line;
    });
    tip.replaceChildren(title,...lines);tip.hidden=false;
    const parent=canvas.parentElement.getBoundingClientRect(),rect=canvas.getBoundingClientRect();
    const width=tip.offsetWidth||190,height=tip.offsetHeight||76;
    let left=rect.left-parent.left+x+12;
    if(left+width>parent.width-4)left=rect.left-parent.left+x-width-12;
    tip.style.left=Math.max(4,Math.min(left,parent.width-width-4))+'px';
    const top=rect.top-parent.top+(canvas.metricHoverY??m.t)+12;
    tip.style.top=Math.max(2,Math.min(top,parent.height-height-2))+'px';
    canvas.metricHoverStep=row.step;
}
for(const id of ['lossChart','psnrChart','ssimChart','lpipsChart']){
    const canvas=$(id);canvas.setAttribute('tabindex','0');canvas.setAttribute('aria-label',`${chartMetricNames[id.replace('Chart','')]} 训练曲线，可悬浮查看步数和指标`);
    const redraw=()=>{const m=canvas.metricChart;if(m)drawChart(canvas,m.rows,m.key,m.color);};
    const move=event=>{const rect=canvas.getBoundingClientRect();canvas.metricHoverX=event.clientX-rect.left;canvas.metricHoverY=event.clientY-rect.top;redraw();};
    canvas.addEventListener('pointermove',move);canvas.addEventListener('pointerdown',move);
    const leave=()=>{canvas.metricHoverX=null;canvas.metricHoverStep=null;if(canvas.metricTooltip)canvas.metricTooltip.hidden=true;redraw();};
    canvas.addEventListener('pointerleave',leave);canvas.addEventListener('blur',leave);
    canvas.addEventListener('keydown',event=>{
        if(event.key==='Escape'){leave();return;}
        if(!['ArrowLeft','ArrowRight'].includes(event.key))return;
        const m=canvas.metricChart;if(!m)return;
        const rows=m.rows.filter(row=>Number.isFinite(row[m.key])||Number.isFinite(row['val_'+m.key]));if(!rows.length)return;
        event.preventDefault();let index=rows.findIndex(row=>row.step===canvas.metricHoverStep);
        index=index<0?0:Math.max(0,Math.min(rows.length-1,index+(event.key==='ArrowRight'?1:-1)));
        canvas.metricHoverX=m.l+(m.w-m.l-m.r)*rows[index].step/Math.max(m.rows.at(-1).step,1);canvas.metricHoverY=m.t;redraw();
    });
}
