const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('static/novel-view.js','utf8');
let axis='z';const context={detectedAxis:()=>axis,novelFocalScale:1.39,novelInteraction:false,novelReferencePose:null,
    center:[.1,.2,.3],distance:4,yaw:.7,pitch:.4,$:()=>({getBoundingClientRect:()=>({width:1200,height:440})})};
vm.createContext(context);vm.runInContext(source.slice(source.indexOf('function unorient'),source.indexOf('function scheduleNovelView')),context);
function orient(p){const[x,y,z]=p;return {z:[x,z,-y],'-z':[x,-z,y],x:[-y,x,z],'-x':[y,-x,z],'-y':[x,-y,-z],y:p}[axis];}
for(axis of ['x','y','z','-x','-y','-z']){
    const point=[.2,.4,.7];const inverse=vm.runInContext(`unorient(${JSON.stringify(orient(point))})`,context);
    point.forEach((v,i)=>assert.ok(Math.abs(v-inverse[i])<1e-10));
    const payload=vm.runInContext('novelPayload()',context),m=payload.c2w;
    const delta=point.map((v,k)=>v-m[k][3]);const camera=[0,1,2].map(k=>delta.reduce((sum,v,i)=>sum+v*m[i][k],0));
    const actual=[payload.fx*camera[0]/-camera[2]+payload.cx,-payload.fy*camera[1]/-camera[2]+payload.cy];
    const p=orient(point).map((v,k)=>v-context.center[k]);
    const cy=Math.cos(context.yaw),sy=Math.sin(context.yaw),cp=Math.cos(context.pitch),sp=Math.sin(context.pitch);
    const u=cy*p[0]-sy*p[2],t=sy*p[0]+cy*p[2],v=cp*p[1]-sp*t,depth=context.distance-(sp*p[1]+cp*t);
    const expected=[600+u*440*1.39/depth,220-v*440*1.39/depth];
    expected.forEach((v,i)=>assert.ok(Math.abs(v-actual[i])<1e-8,axis));
}
console.log('Orbit ↔ camera-to-world ↔ grid projection match for all six up-axis conventions.');
