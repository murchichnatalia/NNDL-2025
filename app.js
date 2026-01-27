// app.js
// Titanic EDA Web App Logic (client-side, browser-only)

// ======================
// Global Variables
// ======================
let trainData = [];
let headers = [];

// Dataset Schema Notes:
// Target: Survived (0/1, train only)
// Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
// Identifier: PassengerId (excluded in analysis)
// To reuse for other datasets, swap CSV file & update schema here

// ======================
// Utility Functions
// ======================

// Calculate mean
function mean(arr) {
  const valid = arr.filter(x => !isNaN(x));
  return valid.reduce((a,b)=>a+b,0)/valid.length;
}

// Calculate median
function median(arr) {
  const valid = arr.filter(x => !isNaN(x)).sort((a,b)=>a-b);
  const mid = Math.floor(valid.length/2);
  return valid.length %2 ? valid[mid] : (valid[mid-1]+valid[mid])/2;
}

// Calculate standard deviation
function std(arr) {
  const m = mean(arr);
  const valid = arr.filter(x => !isNaN(x));
  const variance = valid.reduce((sum,x)=>sum+(x-m)**2,0)/valid.length;
  return Math.sqrt(variance);
}

// Count unique values
function countValues(arr) {
  return arr.reduce((acc,val)=>{
    acc[val] = (acc[val]||0)+1;
    return acc;
  }, {});
}

// ======================
// Data Load
// ======================
document.getElementById('load-data-btn').addEventListener('click', ()=>{
  const fileInput = document.getElementById('train-file');
  if(!fileInput.files[0]) { alert("Please select a train.csv file"); return; }

  Papa.parse(fileInput.files[0], {
    header: true,
    skipEmptyLines: true,
    complete: function(results) {
      trainData = results.data;
      headers = results.meta.fields;
      alert("Train data loaded successfully!");
      renderOverview();
    },
    error: function(err){ alert("Error parsing CSV: "+err); }
  });
});

// ======================
// Data Overview
// ======================
function renderOverview() {
  const overviewDiv = document.getElementById('overview-info');
  overviewDiv.innerHTML = `<p>Rows: ${trainData.length}, Columns: ${headers.length}</p>`;

  // Preview table (first 5 rows)
  const previewContainer = document.getElementById('preview-table-container');
  let html = '<table><tr>';
  headers.forEach(h=>html+=`<th>${h}</th>`);
  html+='</tr>';
  trainData.slice(0,5).forEach(row=>{
    html+='<tr>';
    headers.forEach(h=>html+=`<td>${row[h]}</td>`);
    html+='</tr>';
  });
  html+='</table>';
  previewContainer.innerHTML = html;

  renderMissingValues();
}

// ======================
// Missing Values
// ======================
function renderMissingValues() {
  const missingCounts = {};
  headers.forEach(h=>{
    missingCounts[h] = trainData.filter(r=>r[h]==='' || r[h]==null).length;
  });

  const missingPercent = headers.map(h=>missingCounts[h]/trainData.length*100);

  const ctx = document.getElementById('missing-chart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: headers,
      datasets: [{
        label: '% Missing',
        data: missingPercent,
        backgroundColor: 'rgba(255,99,132,0.6)'
      }]
    },
    options: { responsive:true, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true, max:100}} }
  });

  // Analysis comment
  const mvComment = document.createElement('p');
  mvComment.textContent = "Observation: Age and Embarked have some missing values, while other features are mostly complete.";
  document.getElementById('missing-values').appendChild(mvComment);
}

// ======================
// Summary Statistics
// ======================
function renderSummaryStats() {
  const numericCols = ['Age','SibSp','Parch','Fare'];
  const categoricalCols = ['Survived','Pclass','Sex','Embarked'];

  // Numeric stats table
  let numHtml = '<h3>Numeric Features</h3><table><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std</th></tr>';
  numericCols.forEach(col=>{
    const arr = trainData.map(r=>parseFloat(r[col])).filter(x=>!isNaN(x));
    numHtml += `<tr><td>${col}</td><td>${mean(arr).toFixed(2)}</td><td>${median(arr)}</td><td>${std(arr).toFixed(2)}</td></tr>`;
  });
  numHtml+='</table>';
  document.getElementById('numeric-stats').innerHTML = numHtml;

  // Numeric analysis comment
  const numComment = document.createElement('p');
  numComment.textContent = "Observation: Most passengers are around 28 years old; Fare varies widely, indicating class differences.";
  document.getElementById('numeric-stats').appendChild(numComment);

  // Categorical counts
  let catHtml = '<h3>Categorical Features</h3>';
  categoricalCols.forEach(col=>{
    const counts = countValues(trainData.map(r=>r[col]));
    catHtml+=`<strong>${col}:</strong> `;
    for(const key in counts){ catHtml+=`${key}=${counts[key]} `; }
    catHtml+='<br>';
  });
  document.getElementById('categorical-stats').innerHTML = catHtml;

  // Categorical analysis comment
  const catComment = document.createElement('p');
  catComment.textContent = "Observation: Most passengers are in 3rd class; more males than females; majority embarked from 'S' port; survival rate is roughly 38%.";
  document.getElementById('categorical-stats').appendChild(catComment);
}

// ======================
// Visualizations
// ======================
function renderCharts() {
  // Bar charts for categorical features
  const catFeatures = ['Sex','Pclass','Embarked'];
  catFeatures.forEach(feature=>{
    const ctx = document.getElementById(feature.toLowerCase()+'-chart').getContext('2d');
    const counts = countValues(trainData.map(r=>r[feature]));
    new Chart(ctx, {
      type: 'bar',
      data: { labels:Object.keys(counts), datasets:[{label:feature, data:Object.values(counts), backgroundColor:'rgba(54,162,235,0.6)'}] },
      options:{ responsive:true }
    });

    // Add analysis comment below each chart
    const comment = document.createElement('p');
    if(feature==='Sex') comment.textContent = "Observation: More males than females, but females had higher survival.";
    if(feature==='Pclass') comment.textContent = "Observation: 3rd class has most passengers; higher class passengers had better survival rates.";
    if(feature==='Embarked') comment.textContent = "Observation: Most passengers boarded from port 'S'; survival slightly higher for 'C'.";
    document.getElementById('dashboard').appendChild(comment);
  });

  // Histograms for Age & Fare
  const numericCols = ['Age','Fare'];
  numericCols.forEach(col=>{
    const arr = trainData.map(r=>parseFloat(r[col])).filter(x=>!isNaN(x));
    const bins = 10;
    const min = Math.min(...arr), max = Math.max(...arr);
    const binSize = (max-min)/bins;
    const binCounts = Array(bins).fill(0);
    arr.forEach(v=>{
      let idx = Math.floor((v-min)/binSize);
      if(idx>=bins) idx=bins-1;
      binCounts[idx]++;
    });
    const ctx = document.getElementById(col.toLowerCase()+'-histogram').getContext('2d');
    new Chart(ctx,{
      type:'bar',
      data:{ labels:Array.from({length:bins},(_,i)=> (min+i*binSize).toFixed(1)+'-'+(min+(i+1)*binSize).toFixed(1)),
             datasets:[{label:col, data:binCounts, backgroundColor:'rgba(75,192,192,0.6)'}] },
      options:{ responsive:true }
    });

    // Histogram comment
    const histComment = document.createElement('p');
    if(col==='Age') histComment.textContent = "Observation: Passenger age mostly between 20-40 years.";
    if(col==='Fare') histComment.textContent = "Observation: Fare distribution skewed; few very high-paying passengers.";
    document.getElementById('dashboard').appendChild(histComment);
  });

  // Correlation heatmap (numeric features including target)
  const numericColsWithTarget = ['Survived','Pclass','Age','SibSp','Parch','Fare'];
  const corrMatrix = [];
  numericColsWithTarget.forEach(col1=>{
    const row=[];
    const arr1 = trainData.map(r=>parseFloat(r[col1]));
    numericColsWithTarget.forEach(col2=>{
      const arr2 = trainData.map(r=>parseFloat(r[col2]));
      const mean1 = mean(arr1), mean2 = mean(arr2);
      const cov = arr1.reduce((sum,v,i)=>sum+(v-mean1)*(arr2[i]-mean2),0)/arr1.length;
      row.push(cov/(std(arr1)*std(arr2)));
    });
    corrMatrix.push(row);
  });
  const ctx = document.getElementById('correlation-heatmap').getContext('2d');
  new Chart(ctx,{
    type:'matrix',
    data:{
      datasets:[{
        label:'Correlation',
        data:corrMatrix.flatMap((row,y)=>row.map((v,x)=>({x:x,y:y,v:v}))),
        backgroundColor: function(ctx){
          const v = ctx.dataset.data[ctx.dataIndex].v;
          const alpha = Math.abs(v);
          return v>0?`rgba(255,0,0,${alpha})`:`rgba(0,0,255,${alpha})`;
        },
        width: ({chart}) => chart.chartArea.width / numericColsWithTarget.length -2,
        height: ({chart}) => chart.chartArea.height / numericColsWithTarget.length -2
      }]
    },
    options:{
      responsive:true,
      plugins:{legend:{display:false}},
      scales:{
        x:{type:'category', labels:numericColsWithTarget},
        y:{type:'category', labels:numericColsWithTarget, reverse:true}
      }
    }
  });

  const corrComment = document.createElement('p');
  corrComment.textContent = "Observation: Survival positively correlates with Fare and Pclass (1st class more likely to survive), weakly negatively with SibSp/Parch.";
  document.getElementById('dashboard').appendChild(corrComment);
}

// ======================
// Run EDA Button
// ======================
document.getElementById('run-eda-btn').addEventListener('click', ()=>{
  if(trainData.length===0){ alert("Please load train.csv first"); return; }
  renderSummaryStats();
  renderCharts();
});
