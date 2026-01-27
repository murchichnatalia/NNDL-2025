// ===============================
// Titanic Production EDA Dashboard
// Auto-load CSV from /Week 1/
// ===============================

// IMPORTANT:
// Files must exist at:
// Week 1/train.csv
// Week 1/test.csv
// If deployed on GitHub Pages, path is relative to root.

const TRAIN_URL = "Week 1/train.csv";
const TEST_URL = "Week 1/test.csv";

let mergedData = [];
let charts = {};

// ================= Utilities =================

function mean(arr) {
    return arr.reduce((a,b)=>a+b,0)/arr.length;
}

function median(arr){
    const s=[...arr].sort((a,b)=>a-b);
    const m=Math.floor(s.length/2);
    return s.length%2?s[m]:(s[m-1]+s[m])/2;
}

function std(arr){
    const m=mean(arr);
    return Math.sqrt(mean(arr.map(x=>(x-m)**2)));
}

function destroyChart(id){
    if(charts[id]) charts[id].destroy();
}

// ================= Auto Load =================

window.addEventListener("DOMContentLoaded", () => {
    loadData();
});

async function loadData(){

    try {
        const train = await fetchCsv(TRAIN_URL, "train");
        const test = await fetchCsv(TEST_URL, "test");

        mergedData = [...train, ...test];

        document.getElementById("status").innerText =
            "Data loaded successfully";

        runEDA();

    } catch(err){
        console.error(err);
        document.getElementById("status").innerText =
            "Error loading CSV files. Check paths.";
    }
}

function fetchCsv(url, sourceName){
    return new Promise((resolve,reject)=>{
        Papa.parse(url,{
            download:true,
            header:true,
            dynamicTyping:true,
            skipEmptyLines:true,
            complete:results=>{
                if(!results.data.length){
                    reject("Empty file");
                    return;
                }
                const cleaned = results.data.map(r=>({...r, source:sourceName}));
                resolve(cleaned);
            },
            error:err=>reject(err)
        });
    });
}

// ================= EDA =================

function runEDA(){

    const train = mergedData.filter(d=>d.source==="train");

    // KPI
    const total = train.length;
    const survived = train.filter(d=>d.Survived===1).length;
    const female = train.filter(d=>d.Sex==="female");
    const male = train.filter(d=>d.Sex==="male");

    document.getElementById("totalPassengers").innerText = total;
    document.getElementById("survivalRate").innerText =
        ((survived/total)*100).toFixed(1)+"%";

    document.getElementById("femaleRate").innerText =
        ((female.filter(d=>d.Survived===1).length/female.length)*100).toFixed(1)+"%";

    document.getElementById("maleRate").innerText =
        ((male.filter(d=>d.Survived===1).length/male.length)*100).toFixed(1)+"%";

    survivalByCategory(train,"Sex","sexChart");
    survivalByCategory(train,"Pclass","pclassChart");
    histogram(train,"Age","ageChart");
    histogram(train,"Fare","fareChart");

    renderStats(train);

    /*
    =============================
    FINAL EDA CONCLUSION
    =============================

    1. SEX is the dominant factor.
       Female survival ~70–75%
       Male survival ~15–20%

    2. Pclass second strongest factor.
       1st class survival highest.

    3. Age & Fare moderate effects.

    ==> Main factor of death on Titanic: BEING MALE.
    */
}

// ================= Charts =================

function survivalByCategory(data,feature,canvasId){

    destroyChart(canvasId);

    const groups={};

    data.forEach(r=>{
        if(!groups[r[feature]]) groups[r[feature]]={survived:0,total:0};
        groups[r[feature]].total++;
        if(r.Survived===1) groups[r[feature]].survived++;
    });

    const labels=Object.keys(groups);
    const rates=labels.map(l=>
        (groups[l].survived/groups[l].total*100).toFixed(1)
    );

    charts[canvasId]=new Chart(
        document.getElementById(canvasId),
        {
            type:"bar",
            data:{
                labels:labels,
                datasets:[{
                    label:"Survival Rate (%)",
                    data:rates
                }]
            }
        }
    );
}

function histogram(data,feature,canvasId){

    destroyChart(canvasId);

    const values=data
        .map(d=>d[feature])
        .filter(v=>typeof v==="number"&&!isNaN(v));

    const bins=10;
    const min=Math.min(...values);
    const max=Math.max(...values);
    const step=(max-min)/bins;

    const counts=new Array(bins).fill(0);

    values.forEach(v=>{
        const i=Math.min(Math.floor((v-min)/step),bins-1);
        counts[i]++;
    });

    const labels=counts.map((_,i)=>
        (min+i*step).toFixed(1)
    );

    charts[canvasId]=new Chart(
        document.getElementById(canvasId),
        {
            type:"bar",
            data:{
                labels:labels,
                datasets:[{
                    label:feature+" Distribution",
                    data:counts
                }]
            }
        }
    );
}

// ================= Stats =================

function renderStats(data){

    const numeric=["Age","Fare","SibSp","Parch"];
    let html="<table><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std</th></tr>";

    numeric.forEach(f=>{
        const vals=data
            .map(d=>d[f])
            .filter(v=>typeof v==="number"&&!isNaN(v));

        html+=`<tr>
            <td>${f}</td>
            <td>${mean(vals).toFixed(2)}</td>
            <td>${median(vals).toFixed(2)}</td>
            <td>${std(vals).toFixed(2)}</td>
        </tr>`;
    });

    html+="</table>";

    document.getElementById("statsTable").innerHTML=html;
}
