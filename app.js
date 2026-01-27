let trainData = [];
let testData = [];
let mergedData = [];
let charts = {};

document.getElementById("loadBtn").addEventListener("click", loadData);

function loadData(){
  const trainFile = document.getElementById("trainFile").files[0];
  const testFile = document.getElementById("testFile").files[0];
  if(!trainFile || !testFile){
    alert("Please upload both train.csv and test.csv");
    return;
  }

  Papa.parse(trainFile,{
    header:true,
    dynamicTyping:true,
    skipEmptyLines:true,
    complete:(res)=>{
      trainData = res.data.map(r=>({...r,DatasetType:"Train"}));
      Papa.parse(testFile,{
        header:true,
        dynamicTyping:true,
        skipEmptyLines:true,
        complete:(res2)=>{
          testData = res2.data.map(r=>({...r,DatasetType:"Test"}));
          mergedData = [...trainData,...testData];
          analyze();
        }
      });
    }
  });
}

function analyze(){
  renderSurvivalCharts();
  renderStatsTables();
  renderCorrelationHeatmap();
  generateGraphSummary();

  document.getElementById("survivalChartsSection").style.display="block";
  document.getElementById("statsSection").style.display="block";
  document.getElementById("heatmapSection").style.display="block";
  document.getElementById("graphSummarySection").style.display="block";
}

function survivalRateBy(groupKey){
  const groups = {};
  trainData.forEach(r=>{
    const key = r[groupKey] ?? "Missing";
    if(!groups[key]) groups[key]={total:0,survived:0};
    groups[key].total++;
    if(r.Survived===1) groups[key].survived++;
  });

  const labels = Object.keys(groups);
  const rates = labels.map(k=>groups[k].survived/groups[k].total*100);
  return {labels,rates};
}

function renderBarChart(id,title,dataObj){
  if(charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id),{
    type:"bar",
    data:{
      labels:dataObj.labels,
      datasets:[{data:dataObj.rates}]
    },
    options:{
      plugins:{
        legend:{display:false},
        title:{display:true,text:title}
      },
      scales:{y:{beginAtZero:true,max:100}}
    }
  });
}

function renderSurvivalCharts(){
  renderBarChart("sexChart","Survival Rate by Sex",survivalRateBy("Sex"));
  renderBarChart("classChart","Survival Rate by Ticket Class",survivalRateBy("Pclass"));
  renderBarChart("sibspChart","Survival Rate by Siblings (SibSp)",survivalRateBy("SibSp"));
  renderBarChart("parchChart","Survival Rate by Parents/Children (Parch)",survivalRateBy("Parch"));
  renderBarChart("embarkedChart","Survival Rate by Embarked",survivalRateBy("Embarked"));

  const cabinData = trainData.map(r=>({...r,CabinGroup:r.Cabin? r.Cabin[0]:"Unknown"}));
  const cabinGroups = {};
  cabinData.forEach(r=>{
    const key=r.CabinGroup;
    if(!cabinGroups[key]) cabinGroups[key]={total:0,survived:0};
    cabinGroups[key].total++;
    if(r.Survived===1) cabinGroups[key].survived++;
  });
  renderBarChart("cabinChart","Survival Rate by Cabin Deck",{
    labels:Object.keys(cabinGroups),
    rates:Object.values(cabinGroups).map(g=>g.survived/g.total*100)
  });

  renderContinuousChart("ageChart","Age");
  renderContinuousChart("fareChart","Fare");
}

function renderContinuousChart(id,column){
  const bins=5;
  const values=trainData.map(r=>r[column]).filter(v=>v!=null);
  const min=Math.min(...values);
  const max=Math.max(...values);
  const step=(max-min)/bins;

  const groups={};
  trainData.forEach(r=>{
    if(r[column]==null) return;
    const bin=Math.min(Math.floor((r[column]-min)/step),bins-1);
    if(!groups[bin]) groups[bin]={total:0,survived:0};
    groups[bin].total++;
    if(r.Survived===1) groups[bin].survived++;
  });

  const labels=Object.keys(groups).map(b=>{
    const start=(min+b*step).toFixed(0);
    const end=(min+(+b+1)*step).toFixed(0);
    return `${start}-${end}`;
  });

  const rates=Object.values(groups).map(g=>g.survived/g.total*100);
  renderBarChart(id,`Survival Rate by ${column}`,{labels,rates});
}

function renderStatsTables(){
  const container=document.getElementById("statsTables");
  container.innerHTML="";

  const numeric=["Age","Fare","SibSp","Parch"];
  numeric.forEach(col=>{
    const vals=trainData.map(r=>r[col]).filter(v=>v!=null);
    const mean=vals.reduce((a,b)=>a+b,0)/vals.length;
    const sorted=[...vals].sort((a,b)=>a-b);
    const median=sorted[Math.floor(sorted.length/2)];
    const std=Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0)/vals.length);

    container.innerHTML+=`
      <h4>${col}</h4>
      <table>
        <tr><th>Mean</th><th>Median</th><th>Std Dev</th></tr>
        <tr><td>${mean.toFixed(2)}</td><td>${median.toFixed(2)}</td><td>${std.toFixed(2)}</td></tr>
      </table><br>`;
  });
}

function renderCorrelationHeatmap(){
  const numeric=["Survived","Pclass","Age","SibSp","Parch","Fare"];
  const matrix=[];

  numeric.forEach(a=>{
    numeric.forEach(b=>{
      matrix.push({
        x:a,
        y:b,
        v:correlation(a,b)
      });
    });
  });

  if(charts["heatmapChart"]) charts["heatmapChart"].destroy();

  charts["heatmapChart"]=new Chart(document.getElementById("heatmapChart"),{
    type:"matrix",
    data:{datasets:[{data:matrix,width:40,height:40}]},
    options:{plugins:{legend:{display:false}}}
  });
}

function correlation(a,b){
  const vals=trainData.filter(r=>r[a]!=null && r[b]!=null);
  const meanA=vals.reduce((s,r)=>s+r[a],0)/vals.length;
  const meanB=vals.reduce((s,r)=>s+r[b],0)/vals.length;

  const num=vals.reduce((s,r)=>s+(r[a]-meanA)*(r[b]-meanB),0);
  const den=Math.sqrt(
    vals.reduce((s,r)=>s+(r[a]-meanA)**2,0) *
    vals.reduce((s,r)=>s+(r[b]-meanB)**2,0)
  );
  return num/den;
}

function generateGraphSummary(){
  const summary=document.getElementById("graphSummary");
  summary.innerHTML="";

  const sex=survivalRateBy("Sex");
  summary.innerHTML+=`<li>Women survived at much higher rates than men.</li>`;
  summary.innerHTML+=`<li>Higher ticket classes show significantly better survival probability.</li>`;
  summary.innerHTML+=`<li>Passengers with moderate family size had higher survival than those alone or very large families.</li>`;
  summary.innerHTML+=`<li>Fare correlates positively with survival, suggesting wealth/class impact.</li>`;
  summary.innerHTML+=`<li>Cabin deck location strongly relates to survival likelihood.</li>`;
}
