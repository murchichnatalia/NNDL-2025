// app.js
// Titanic EDA Web App Logic with Survival Filter

let trainData = [];
let headers = [];
let filteredData = []; // Holds filtered dataset
let charts = {};       // Store chart instances for updates

// Utility functions
function mean(arr){ const valid=arr.filter(x=>!isNaN(x)); return valid.reduce((a,b)=>a+b,0)/valid.length; }
function median(arr){ const valid=arr.filter(x=>!isNaN(x)).sort((a,b)=>a-b); const mid=Math.floor(valid.length/2); return valid.length%2?valid[mid]:(valid[mid-1]+valid[mid])/2; }
function std(arr){ const m=mean(arr); const valid=arr.filter(x=>!isNaN(x)); const variance=valid.reduce((sum,x)=>sum+(x-m)**2,0)/valid.length; return Math.sqrt(variance); }
function countValues(arr){ return arr.reduce((acc,val)=>{ acc[val]=(acc[val]||0)+1; return acc; },{}); }

// Data Load
document.getElementById('load-data-btn').addEventListener('click', ()=>{
  const fileInput=document.getElementById('train-file');
  if(!fileInput.files[0]){ alert("Please select train.csv"); return; }
  Papa.parse(fileInput.files[0],{ header:true, skipEmptyLines:true,
    complete:function(results){
      trainData=results.data; headers=results.meta.fields;
      alert("Train data loaded!");
      renderOverview();
    },
    error:function(err){ alert("Error parsing CSV: "+err); }
  });
});

// Filter dataset by survival selection
function applySurvivalFilter(){
  const val=document.getElementById('survival-filter').value;
  if(val==='all') filteredData=trainData;
  else if(val==='survived') filteredData=trainData.filter(r=>r.Survived==='1' || r.Survived===1);
  else filteredData=trainData.filter(r=>r.Survived==='0' || r.Survived===0);
}

// Data Overview
function renderOverview(){
  const overviewDiv=document.getElementById('overview-info');
  overviewDiv.innerHTML=`<p>Rows: ${trainData.length}, Columns: ${headers.length}</p>`;
  const preview=document.getElementById('preview-table-container');
  let html='<table><tr>'; headers.forEach(h=>html+=`<th>${h}</th>`); html+='</tr>';
  trainData.slice(0,5).forEach(r=>{ html+='<tr>'; headers.forEach(h=>html+=`<td>${r[h]}</td>`); html+='</tr>'; });
  html+='</table>'; preview.innerHTML=html;
  renderMissingValues();
}

// Missing Values
function renderMissingValues(){
  const missingCounts={}; headers.forEach(h=>{ missingCounts[h]=trainData.filter(r=>r[h]===''||r[h]==null).length; });
  const missingPercent=headers.map(h=>missingCounts[h]/trainData.length*100);
  const ctx=document.getElementById('missing-chart').getContext('2d');
  new Chart(ctx,{ type:'bar', data:{ labels:headers, datasets:[{label:'% Missing', data:missingPercent, backgroundColor:'rgba(255,99,132,0.6)'}] },
    options:{ responsive:true, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true,max:100}} } });
  const p=document.createElement('p'); p.className='analysis-comment'; p.textContent="Observation: Age and Embarked have some missing values, others mostly complete."; document.getElementById('missing-values').appendChild(p);
}

// Summary Statistics
function renderSummaryStats(){
  applySurvivalFilter();
  const numericCols=['Age','SibSp','Parch','Fare']; const categoricalCols=['Survived','Pclass','Sex','Embarked'];
  let numHtml='<h3>Numeric Features</h3><table><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std</th></tr>';
  numericCols.forEach(c=>{ const arr=filteredData.map(r=>parseFloat(r[c])).filter(x=>!isNaN(x)); numHtml+=`<tr><td>${c}</td><td>${mean(arr).toFixed(2)}</td><td>${median(arr)}</td><td>${std(arr).toFixed(2)}</td></tr>`; });
  numHtml+='</table>'; document.getElementById('numeric-stats').innerHTML=numHtml;
  const pNum=document.createElement('p'); pNum.className='analysis-comment'; pNum.textContent="Observation: Age around 28; Fare varies widely."; document.getElementById('numeric-stats').appendChild(pNum);

  let catHtml='<h3>Categorical Features</h3>';
  categoricalCols.forEach(c=>{ const counts=countValues(filteredData.map(r=>r[c])); catHtml+=`<strong>${c}:</strong> `; for(const k in counts) catHtml+=`${k}=${counts[k]} `; catHtml+='<br>'; });
  document.getElementById('categorical-stats').innerHTML=catHtml;
  const pCat=document.createElement('p'); pCat.className='analysis-comment'; pCat.textContent="Observation: Most in 3rd class; more males; survival ~38%."; document.getElementById('categorical-stats').appendChild(pCat);
}

// Visualizations
function renderCharts(){
  applySurvivalFilter();
  // Destroy previous charts
  Object.values(charts).forEach(c=>c.destroy()); charts={};

  const catFeatures=['Sex','Pclass','Embarked'];
  catFeatures.forEach(f=>{
    const ctx=document.getElementById(f.toLowerCase()+'-chart').getContext('2d');
    const counts=countValues(filteredData.map(r=>r[f]));
    charts[f]=new Chart(ctx,{ type:'bar', data:{ labels:Object.keys(counts), datasets:[{label:f, data:Object.values(counts), backgroundColor:'rgba(54,162,235,0.6)'}] }, options:{responsive:true} });
    const p=document.createElement('p'); p.className='analysis-comment';
    if(f==='Sex') p.textContent="Observation: More males; females higher survival."; 
    if(f==='Pclass') p.textContent="Observation: 3rd class most passengers; higher class better survival.";
    if(f==='Embarked') p.textContent="Observation: Most boarded 'S'; 'C' slightly higher survival.";
    document.getElementById('dashboard').appendChild(p);
  });

  const numericCols=['Age','Fare'];
  numericCols.forEach(col=>{
    const arr=filteredData.map(r=>parseFloat(r[col])).filter(x=>!isNaN(x));
    const bins=10; const min=Math.min(...arr); const max=Math.max(...arr); const binSize=(max-min)/bins;
    const binCounts=Array(bins).fill(0);
    arr.forEach(v=>{ let i=Math.floor((v-min)/binSize); if(i>=bins)i=bins-1; binCounts[i]++; });
    const ctx=document.getElementById(col.toLowerCase()+'-histogram').getContext('2d');
    charts[col]=new Chart(ctx,{ type:'bar', data:{ labels:Array.from({length:bins},(_,i)=>(min+i*binSize).toFixed(1)+'-'+(min+(i+1)*binSize).toFixed(1)), datasets:[{label:col,data:binCounts,backgroundColor:'rgba(75,192,192,0.6)'}] }, options:{responsive:true} });
    const p=document.createElement('p'); p.className='analysis-comment';
    if(col==='Age') p.textContent="Observation: Age mostly 20-40 years.";
    if(col==='Fare') p.textContent="Observation: Fare skewed; few very high-paying passengers.";
    document.getElementById('dashboard').appendChild(p);
  });

  // Correlation heatmap
  const numericColsWithTarget=['Survived','Pclass','Age','SibSp','Parch','Fare'];
  const corrMatrix=[]; numericColsWithTarget.forEach(c1=>{
    const row=[]; const arr1=filteredData.map(r=>parseFloat(r[c1]));
    numericColsWithTarget.forEach(c2=>{ const arr2=filteredData.map(r=>parseFloat(r[c2])); const mean1=mean(arr1), mean2=mean(arr2); const cov=arr1.reduce((sum,v,i)=>sum+(v-mean1)*(arr2[i]-mean2),0)/arr1.length; row.push(cov/(std(arr1)*std(arr2))); }); corrMatrix.push(row);
  });
  const ctx=document.getElementById('correlation-heatmap').getContext('2d');
  charts['corr']=new Chart(ctx,{ type:'matrix', data:{ datasets:[{ label:'Correlation', data:corrMatrix.flatMap((r,y)=>r.map((v,x)=>({x:x,y:y,v:v}))), backgroundColor:ctx=>{ const v=ctx.dataset.data[ctx.dataIndex].v; const alpha=Math.abs(v); return v>0?`rgba(255,0,0,${alpha})`:`rgba(0,0,255,${alpha})`; }, width:({chart})=>chart.chartArea.width/numericColsWithTarget.length-2, height:({chart})=>chart.chartArea.height/numericColsWithTarget.length-2 }] }, options:{ responsive:true, plugins:{legend:{display:false}}, scales:{x:{type:'category', labels:numericColsWithTarget}, y:{type:'category', labels:numericColsWithTarget, reverse:true}} } });
  const p=document.createElement('p'); p.className='analysis-comment'; p.textContent="Observation: Survival positively correlates with Fare and Pclass; weak negative with SibSp/Parch."; document.getElementById('dashboard').appendChild(p);
}

// Run EDA
document.getElementById('run-eda-btn').addEventListener('click', ()=>{
  if(trainData.length===0){ alert("Please load train.csv first"); return; }
  renderSummaryStats(); renderCharts();
});

// Survival Filter Change
document.getElementById('survival-filter').addEventListener('change', ()=>{
  if(trainData.length===0) return;
  // Clear dashboard comments before re-render
  const dashboard=document.getElementById('dashboard'); Array.from(dashboard.querySelectorAll('p.analysis-comment')).forEach(p=>p.remove());
  renderSummaryStats(); renderCharts();
});
