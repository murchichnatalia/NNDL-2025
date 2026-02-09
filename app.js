// ================= GLOBAL VARIABLES =================
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;

let FEATURE_NAMES = [];
let STANDARDIZATION_STATS = {};

// ================= CONFIG =================
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// =====================================================
// ================= CSV PARSER FIX ====================
// =====================================================
// Fix: handles quoted commas correctly

function parseCSV(text) {
    const rows = [];
    let current = '';
    let inQuotes = false;
    let row = [];

    for (let i = 0; i < text.length; i++) {
        const char = text[i];

        if (char === '"' && text[i + 1] === '"') {
            current += '"';
            i++;
        } else if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            row.push(current);
            current = '';
        } else if ((char === '\n' || char === '\r') && !inQuotes) {
            if (current !== '' || row.length > 0) {
                row.push(current);
                rows.push(row);
                row = [];
                current = '';
            }
        } else {
            current += char;
        }
    }

    if (current !== '' || row.length > 0) {
        row.push(current);
        rows.push(row);
    }

    const headers = rows[0];

    return rows.slice(1).map(r => {
        const obj = {};
        headers.forEach((h, i) => {
            let val = r[i] === '' ? null : r[i];
            if (val !== null && !isNaN(val)) val = parseFloat(val);
            obj[h] = val;
        });
        return obj;
    });
}

// =====================================================
// ================ DATA LOADING =======================
// =====================================================

async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];

    const trainText = await trainFile.text();
    const testText = await testFile.text();

    trainData = parseCSV(trainText);
    testData = parseCSV(testText);

    document.getElementById('data-status').innerHTML =
        `Loaded: ${trainData.length} train rows, ${testData.length} test rows`;

    document.getElementById('inspect-btn').disabled = false;
}

// =====================================================
// ================= INSPECT DATA FIX ==================
// =====================================================

function inspectData() {

    if (!trainData || trainData.length === 0) {
        alert("Load data first.");
        return;
    }

    const previewDiv = document.getElementById('data-preview');
    const statsDiv = document.getElementById('data-stats');
    const chartsDiv = document.getElementById('charts');

    previewDiv.innerHTML = "<h3>First 5 Rows</h3>";
    statsDiv.innerHTML = "";
    chartsDiv.innerHTML = "";

    // ---------- Preview Table ----------
    const table = document.createElement("table");
    const headerRow = document.createElement("tr");

    Object.keys(trainData[0]).forEach(key => {
        const th = document.createElement("th");
        th.textContent = key;
        headerRow.appendChild(th);
    });

    table.appendChild(headerRow);

    trainData.slice(0, 5).forEach(row => {
        const tr = document.createElement("tr");
        Object.values(row).forEach(val => {
            const td = document.createElement("td");
            td.textContent = val ?? "NULL";
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    previewDiv.appendChild(table);

    // ---------- Dataset Shape ----------
    const rows = trainData.length;
    const cols = Object.keys(trainData[0]).length;

    // ---------- Missing Values ----------
    let missingHTML = "<h4>Missing % per Feature</h4><ul>";

    Object.keys(trainData[0]).forEach(feature => {
        const missing = trainData.filter(r => r[feature] === null).length;
        const percent = ((missing / rows) * 100).toFixed(2);
        missingHTML += `<li>${feature}: ${percent}%</li>`;
    });

    missingHTML += "</ul>";

    // ---------- Survival Rate ----------
    const survived = trainData.filter(r => r.Survived === 1).length;
    const survivalRate = ((survived / rows) * 100).toFixed(2);

    statsDiv.innerHTML = `
        <p><strong>Shape:</strong> ${rows} rows × ${cols} columns</p>
        <p><strong>Survival Rate:</strong> ${survivalRate}%</p>
        ${missingHTML}
    `;

    // =====================================================
    // =============== VISUALIZATIONS ======================
    // =====================================================

    // Survival by Sex
    const sexStats = {};
    trainData.forEach(r => {
        if (!sexStats[r.Sex]) sexStats[r.Sex] = { total: 0, survived: 0 };
        sexStats[r.Sex].total++;
        if (r.Survived === 1) sexStats[r.Sex].survived++;
    });

    const sexChartData = Object.keys(sexStats).map(sex => ({
        x: sex,
        y: (sexStats[sex].survived / sexStats[sex].total) * 100
    }));

    tfvis.render.barchart(
        { name: "Survival by Sex" },
        sexChartData,
        { xLabel: "Sex", yLabel: "Survival %" }
    );

    // Survival by Pclass
    const classStats = {};
    trainData.forEach(r => {
        if (!classStats[r.Pclass]) classStats[r.Pclass] = { total: 0, survived: 0 };
        classStats[r.Pclass].total++;
        if (r.Survived === 1) classStats[r.Pclass].survived++;
    });

    const classChartData = Object.keys(classStats).map(c => ({
        x: "Class " + c,
        y: (classStats[c].survived / classStats[c].total) * 100
    }));

    tfvis.render.barchart(
        { name: "Survival by Pclass" },
        classChartData,
        { xLabel: "Class", yLabel: "Survival %" }
    );

    document.getElementById("preprocess-btn").disabled = false;
}

// =====================================================
// ================= MODEL =============================
// =====================================================

function createModel() {

    const inputShape = preprocessedTrainData.features.shape[1];

    model = tf.sequential();

    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [inputShape]
    }));

    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    document.getElementById('model-summary').innerHTML =
        `Input: ${inputShape} → Dense(16) → Dense(1 sigmoid)`;

    document.getElementById('train-btn').disabled = false;
}

// =====================================================
// ================= TRAINING ==========================
// =====================================================

async function trainModel() {

    const split = Math.floor(preprocessedTrainData.features.shape[0]*0.8);

    const trainX = preprocessedTrainData.features.slice([0,0],[split,-1]);
    const trainY = preprocessedTrainData.labels.slice([0],[split]);

    const valX = preprocessedTrainData.features.slice([split,0],[-1,-1]);
    const valY = preprocessedTrainData.labels.slice([split],[-1]);

    validationData = valX;
    validationLabels = valY;

    await model.fit(trainX, trainY, {
        epochs: 50,
        batchSize: 32,
        validationData: [valX,valY],
        callbacks: tfvis.show.fitCallbacks(
            {name:'Training Curves'},
            ['loss','acc','val_loss','val_acc']
        )
    });

    validationPredictions = model.predict(valX);

    document.getElementById('threshold-slider').disabled = false;
    document.getElementById('predict-btn').disabled = false;

    updateMetrics();
    showFeatureImportance();
}

// =====================================================
// ================= EVALUATION FIX ====================
// =====================================================

function updateMetrics(){

    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').innerText = threshold;

    const preds = validationPredictions.dataSync();
    const labels = validationLabels.dataSync();

    let tp=0,tn=0,fp=0,fn=0;

    for(let i=0;i<preds.length;i++){
        const p = preds[i]>=threshold?1:0;
        const y = labels[i];

        if(p===1 && y===1) tp++;
        if(p===0 && y===0) tn++;
        if(p===1 && y===0) fp++;
        if(p===0 && y===1) fn++;
    }

    document.getElementById('confusion-matrix').innerHTML =
    `
    <table>
        <tr><th></th><th>Pred 1</th><th>Pred 0</th></tr>
        <tr><th>True 1</th><td>${tp}</td><td>${fn}</td></tr>
        <tr><th>True 0</th><td>${fp}</td><td>${tn}</td></tr>
    </table>
    `;

    const acc = (tp+tn)/(tp+tn+fp+fn);

    document.getElementById('performance-metrics').innerHTML =
        `Accuracy: ${(acc*100).toFixed(2)}%`;
}

document.getElementById('threshold-slider')
.addEventListener('input', updateMetrics);

// =====================================================
// ============ SIGMOID GATE FEATURE IMPORTANCE =======
// =====================================================

function showFeatureImportance(){

    const weights = model.layers[0].getWeights()[0];
    const w = weights.abs().mean(1).dataSync();

    const sigmoid = x => 1/(1+Math.exp(-x));

    const importance = Array.from(w).map(v => sigmoid(v));

    const data = FEATURE_NAMES.map((name,i)=>({
        x:name,
        y:importance[i]
    }));

    tfvis.render.barchart(
        {name:'Sigmoid Gate Feature Importance'},
        data,
        {xLabel:'Feature', yLabel:'Importance (0-1)'}
    );
}

// =====================================================
// ================= PREDICTION ========================
// =====================================================

function predict(){

    const testTensor = tf.tensor2d(preprocessedTestData.features);

    testPredictions = model.predict(testTensor);

    const preds = testPredictions.dataSync();

    document.getElementById('prediction-output').innerHTML =
        `Predicted ${preds.length} samples`;

    document.getElementById('export-btn').disabled = false;
}

// =====================================================
// ================= EXPORT ============================
// =====================================================

function exportResults(){

    const preds = testPredictions.dataSync();

    let csv = 'PassengerId,Survived\n';

    preprocessedTestData.passengerIds.forEach((id,i)=>{
        csv += `${id},${preds[i]>=0.5?1:0}\n`;
    });

    const blob = new Blob([csv], {type:'text/csv'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'submission.csv';
    link.click();
}

// ================= HELPERS =================
function median(arr){
    arr.sort((a,b)=>a-b);
    const mid = Math.floor(arr.length/2);
    return arr.length%2?arr[mid]:(arr[mid-1]+arr[mid])/2;
}
function mode(arr){
    return arr.sort((a,b)=>
        arr.filter(v=>v===a).length - arr.filter(v=>v===b).length
    ).pop();
}
function std(arr){
    const m = arr.reduce((a,b)=>a+b)/arr.length;
    return Math.sqrt(arr.map(x=>(x-m)**2).reduce((a,b)=>a+b)/arr.length);
}
