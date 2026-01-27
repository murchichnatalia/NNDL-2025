// Global state
let trainData = [];
let testData = [];
let mergedData = [];
let summaryStats = {};
let insights = [];
let charts = {};

// Column schema (modify here to reuse app for other split datasets)
const numericColumns = ["PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare"];
const categoricalColumns = ["Sex","Embarked","DatasetType"];

document.getElementById("loadBtn").addEventListener("click", loadData);

function loadData() {
  const trainFile = document.getElementById("trainFile").files[0];
  const testFile = document.getElementById("testFile").files[0];
  const message = document.getElementById("uploadMessage");
  message.innerHTML = "";

  if (!trainFile || !testFile) {
    message.innerHTML = "<p class='error'>Please upload both Train and Test CSV files.</p>";
    return;
  }

  Papa.parse(trainFile, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: (results) => {
      trainData = cleanData(results.data, "Train");
      Papa.parse(testFile, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results2) => {
          testData = cleanData(results2.data, "Test");
          mergedData = [...trainData, ...testData];
          onDataLoaded();
        },
        error: () => message.innerHTML = "<p class='error'>Error parsing Test CSV.</p>"
      });
    },
    error: () => message.innerHTML = "<p class='error'>Error parsing Train CSV.</p>"
  });
}

function cleanData(data, type) {
  return data.map(row => {
    let cleaned = {};
    Object.keys(row).forEach(key => {
      const stdKey = key.trim();
      cleaned[stdKey] = row[key];
    });
    cleaned["DatasetType"] = type;
    return cleaned;
  });
}

function onDataLoaded() {
  document.getElementById("uploadMessage").innerHTML =
    `<p>Data successfully loaded. Train: ${trainData.length} rows | Test: ${testData.length} rows | Combined: ${mergedData.length} rows</p>`;

  renderPreview();
  computeSummary();
  renderOverview();
  renderMissingChart();
  renderCharts();
  generateInsights();
  showSections();
}

function renderPreview() {
  const table = document.getElementById("previewTable");
  table.innerHTML = "";
  const preview = mergedData.slice(0,8);
  const headers = Object.keys(preview[0]);

  table.innerHTML += "<tr>" + headers.map(h=>`<th>${h}</th>`).join("") + "</tr>";
  preview.forEach(row => {
    table.innerHTML += "<tr>" + headers.map(h=>`<td>${row[h] ?? ""}</td>`).join("") + "</tr>";
  });
}

function computeSummary() {
  const total = mergedData.length;
  const trainCount = trainData.length;
  const testCount = testData.length;

  const missing = {};
  Object.keys(mergedData[0]).forEach(col => {
    const miss = mergedData.filter(r => r[col] === null || r[col] === "" || r[col] === undefined).length;
    missing[col] = (miss/total)*100;
  });

  const trainSurvival = trainData.filter(r=>r.Survived===1).length/trainData.length*100;

  summaryStats = {
    total, trainCount, testCount,
    missing,
    survivalRate: trainSurvival
  };
}

function renderOverview() {
  const container = document.getElementById("overviewCards");
  container.innerHTML = "";

  const cards = [
    {title: summaryStats.total, label: "Total Passengers"},
    {title: ((summaryStats.trainCount/summaryStats.total)*100).toFixed(1)+"%", label: "Train Data"},
    {title: ((summaryStats.testCount/summaryStats.total)*100).toFixed(1)+"%", label: "Test Data"},
    {title: summaryStats.survivalRate.toFixed(1)+"%", label: "Train Survival Rate"}
  ];

  cards.forEach(c=>{
    container.innerHTML += `
      <div class="card summary-card">
        <h3>${c.title}</h3>
        <p>${c.label}</p>
      </div>`;
  });
}

function renderMissingChart() {
  destroyChart("missingChart");
  const ctx = document.getElementById("missingChart");
  const cols = Object.keys(summaryStats.missing);
  const values = Object.values(summaryStats.missing);

  charts["missingChart"] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: cols,
      datasets: [{
        label: "% Missing",
        data: values
      }]
    },
    options: {
      plugins:{legend:{display:false}},
      scales:{y:{beginAtZero:true,max:100}}
    }
  });

  const maxCol = cols[values.indexOf(Math.max(...values))];
  document.getElementById("missingExplanation").innerText =
    `${maxCol} has the highest missing rate at ${Math.max(...values).toFixed(1)}%.`;
}

function renderCharts() {
  renderGenderSurvival();
  renderClassChart();
  renderAgeHistogram();
  renderFareHistogram();
  renderEmbarkedChart();
}

function renderGenderSurvival() {
  destroyChart("genderSurvivalChart");
  const ctx = document.getElementById("genderSurvivalChart");

  const genders = ["male","female"];
  const rates = genders.map(g=>{
    const group = trainData.filter(r=>r.Sex===g);
    const survived = group.filter(r=>r.Survived===1).length;
    return (survived/group.length)*100;
  });

  charts["genderSurvivalChart"] = new Chart(ctx,{
    type:"bar",
    data:{labels:genders,datasets:[{data:rates}]},
    options:{plugins:{legend:{display:false}},plugins:{title:{display:true,text:"Survival Rate by Gender"}}}
  });
}

function renderClassChart() {
  destroyChart("classChart");
  const ctx = document.getElementById("classChart");
  const classes = [1,2,3];
  const counts = classes.map(c=>mergedData.filter(r=>r.Pclass===c).length);

  charts["classChart"] = new Chart(ctx,{
    type:"bar",
    data:{labels:classes,datasets:[{data:counts}]},
    options:{plugins:{legend:{display:false},title:{display:true,text:"Passenger Class Distribution"}}}
  });
}

function renderAgeHistogram() {
  destroyChart("ageChart");
  const ctx = document.getElementById("ageChart");
  const ages = mergedData.map(r=>r.Age).filter(a=>a!=null);
  const bins = histogram(ages,10);

  charts["ageChart"] = new Chart(ctx,{
    type:"bar",
    data:{labels:bins.labels,datasets:[{data:bins.counts}]},
    options:{plugins:{legend:{display:false},title:{display:true,text:"Age Distribution"}}}
  });
}

function renderFareHistogram() {
  destroyChart("fareChart");
  const ctx = document.getElementById("fareChart");
  const fares = mergedData.map(r=>r.Fare).filter(f=>f!=null).sort((a,b)=>a-b);
  const p99 = fares[Math.floor(feareIndex=0.99*fares.length)];
  const filtered = fares.filter(f=>f<=p99);
  const bins = histogram(filtered,10);

  charts["fareChart"] = new Chart(ctx,{
    type:"bar",
    data:{labels:bins.labels,datasets:[{data:bins.counts}]},
    options:{plugins:{legend:{display:false},title:{display:true,text:"Fare Distribution (≤99th percentile)"}}}
  });
}

function renderEmbarkedChart() {
  destroyChart("embarkedChart");
  const ctx = document.getElementById("embarkedChart");
  const ports = ["C","Q","S"];
  const counts = ports.map(p=>mergedData.filter(r=>r.Embarked===p).length);

  charts["embarkedChart"] = new Chart(ctx,{
    type:"bar",
    data:{labels:ports,datasets:[{data:counts}]},
    options:{plugins:{legend:{display:false},title:{display:true,text:"Embarked Distribution"}}}
  });
}

function histogram(data,bins){
  const min = Math.min(...data);
  const max = Math.max(...data);
  const step = (max-min)/bins;
  let counts = new Array(bins).fill(0);
  data.forEach(val=>{
    let idx = Math.min(Math.floor((val-min)/step),bins-1);
    counts[idx]++;
  });
  let labels = counts.map((_,i)=>`${(min+i*step).toFixed(0)}-${(min+(i+1)*step).toFixed(0)}`);
  return {labels,counts};
}

function generateInsights(){
  insights=[];
  const femaleRate = trainData.filter(r=>r.Sex==="female"&&r.Survived===1).length/
    trainData.filter(r=>r.Sex==="female").length*100;
  const maleRate = trainData.filter(r=>r.Sex==="male"&&r.Survived===1).length/
    trainData.filter(r=>r.Sex==="male").length*100;

  insights.push(`Female survival rate (${femaleRate.toFixed(1)}%) is significantly higher than male survival rate (${maleRate.toFixed(1)}%).`);

  const classCounts = [1,2,3].map(c=>mergedData.filter(r=>r.Pclass===c).length);
  const maxClass = classCounts.indexOf(Math.max(...classCounts))+1;
  insights.push(`Passenger Class ${maxClass} has the highest representation in the dataset.`);

  const maxMissingCol = Object.keys(summaryStats.missing)
    .reduce((a,b)=> summaryStats.missing[a]>summaryStats.missing[b]?a:b);
  insights.push(`${maxMissingCol} contains the highest proportion of missing data.`);

  insights.push(`Overall survival rate in the training dataset is ${summaryStats.survivalRate.toFixed(1)}%.`);

  const list=document.getElementById("insightsList");
  list.innerHTML="";
  insights.forEach(i=>list.innerHTML+=`<li>${i}</li>`);
}

function exportCSV(){
  if(!mergedData.length) return alert("No data loaded.");
  const csv = Papa.unparse(mergedData);
  downloadFile(csv,"merged_titanic.csv","text/csv");
}

function exportJSON(){
  if(!summaryStats.total) return alert("No summary available.");
  downloadFile(JSON.stringify(summaryStats,null,2),"summary.json","application/json");
}

function exportTXT(){
  if(!insights.length) return alert("No insights available.");
  downloadFile(insights.join("\n"),"insights.txt","text/plain");
}

function downloadFile(content,filename,type){
  const blob=new Blob([content],{type});
  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");
  a.href=url;
  a.download=filename;
  a.click();
  URL.revokeObjectURL(url);
}

function destroyChart(id){
  if(charts[id]){
    charts[id].destroy();
  }
}

function showSections(){
  ["overviewSection","missingSection","visualSection","insightsSection","exportSection","previewSection"]
    .forEach(id=>document.getElementById(id).style.display="block");
}
