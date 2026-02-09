// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;

const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }
    
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading data...';
    
    try {
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error(error);
    }
}

function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Improved CSV Parser to handle commas in quotes (e.g. Names)
function parseCSV(csvText) {
    const lines = csvText.split(/\r?\n/).filter(line => line.trim() !== '');
    const headers = splitCSVLine(lines[0]);
    
    return lines.slice(1).map(line => {
        const values = splitCSVLine(line);
        const obj = {};
        headers.forEach((header, i) => {
            let val = values[i] === undefined || values[i] === '' ? null : values[i];
            if (val !== null && !isNaN(val)) {
                val = parseFloat(val);
            }
            obj[header] = val;
        });
        return obj;
    });
}

function splitCSVLine(line) {
    const result = [];
    let cur = '', inQuotes = false;
    for (let char of line) {
        if (char === '"') inQuotes = !inQuotes;
        else if (char === ',' && !inQuotes) {
            result.push(cur.trim());
            cur = '';
        } else cur += char;
    }
    result.push(cur.trim());
    return result;
}

function inspectData() {
    if (!trainData || trainData.length === 0) return;
    
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    const statsDiv = document.getElementById('data-stats');
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null).length;
        missingInfo += `<li>${feature}: ${(missingCount / trainData.length * 100).toFixed(2)}%</li>`;
    });
    missingInfo += '</ul>';
    
    statsDiv.innerHTML = `<p>Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)</p>${missingInfo}`;
    createVisualizations();
    document.getElementById('preprocess-btn').disabled = false;
}

function createPreviewTable(data) {
    const table = document.createElement('table');
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value !== null ? value : 'NULL';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    return table;
}

function createVisualizations() {
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex && row.Survived !== undefined) {
            if (!survivalBySex[row.Sex]) survivalBySex[row.Sex] = { s: 0, t: 0 };
            survivalBySex[row.Sex].t++;
            if (row.Survived === 1) survivalBySex[row.Sex].s++;
        }
    });
    
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({ x: sex, y: (stats.s / stats.t) * 100 }));
    tfvis.render.barchart({ name: 'Survival Rate by Sex', tab: 'Charts' }, sexData);
}

function preprocessData() {
    const outputDiv = document.getElementById('preprocessing-output');
    try {
        const ages = trainData.map(row => row.Age).filter(a => a !== null);
        const ageMedian = ages.sort((a,b)=>a-b)[Math.floor(ages.length/2)];
        const fares = trainData.map(row => row.Fare).filter(f => f !== null);
        const fareMedian = fares.sort((a,b)=>a-b)[Math.floor(fares.length/2)];
        const embarkedMode = 'S'; 

        preprocessedTrainData = { features: [], labels: [] };
        trainData.forEach(row => {
            preprocessedTrainData.features.push(extractFeatures(row, ageMedian, fareMedian, embarkedMode));
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });

        preprocessedTestData = { features: [], passengerIds: [] };
        testData.forEach(row => {
            preprocessedTestData.features.push(extractFeatures(row, ageMedian, fareMedian, embarkedMode));
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });

        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor2d(preprocessedTrainData.labels, [preprocessedTrainData.labels.length, 1]);

        outputDiv.innerHTML = `<p>Preprocessing completed! Feature shape: ${preprocessedTrainData.features.shape}</p>`;
        document.getElementById('create-model-btn').disabled = false;
    } catch (e) { outputDiv.innerHTML = e.message; }
}

function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    const age = row.Age !== null ? row.Age : ageMedian;
    const fare = row.Fare !== null ? row.Fare : fareMedian;
    const sex = row.Sex === 'female' ? 1 : 0;
    const pclass = [row.Pclass === 1 ? 1 : 0, row.Pclass === 2 ? 1 : 0, row.Pclass === 3 ? 1 : 0];
    
    let features = [age / 80, fare / 500, sex, ...pclass];
    if (document.getElementById('add-family-features').checked) {
        const fam = (row.SibSp || 0) + (row.Parch || 0) + 1;
        features.push(fam / 10, fam === 1 ? 1 : 0);
    }
    return features;
}

// Custom Layer for Sigmoid Gate (Feature Importance)
class SigmoidGate extends tf.layers.Layer {
    constructor() { super({}); }
    build(inputShape) {
        this.weights_gate = this.addWeight('gate_weights', [inputShape[1]], 'float32', tf.initializers.ones());
    }
    call(input) {
        return tf.tidy(() => {
            const mask = tf.sigmoid(this.weights_gate.read());
            return tf.mul(input[0], mask);
        });
    }
    static get className() { return 'SigmoidGate'; }
}
tf.serialization.registerClass(SigmoidGate);

function createModel() {
    const inputShape = preprocessedTrainData.features.shape[1];
    model = tf.sequential();
    
    // Add Sigmoid Gate to analyze feature importance
    model.add(new SigmoidGate({ inputShape: [inputShape] }));
    
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = `<p>Model created with Sigmoid Gate. Total params: ${model.countParams()}</p>`;
    document.getElementById('train-btn').disabled = false;
}

async function trainModel() {
    const statusDiv = document.getElementById('training-status');
    const split = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
    
    const trainX = preprocessedTrainData.features.slice(0, split);
    const trainY = preprocessedTrainData.labels.slice(0, split);
    const valX = preprocessedTrainData.features.slice(split);
    const valY = preprocessedTrainData.labels.slice(split);

    validationData = valX;
    validationLabels = valY;

    await model.fit(trainX, trainY, {
        epochs: 50,
        validationData: [valX, valY],
        callbacks: tfvis.show.fitCallbacks({ name: 'Training' }, ['loss', 'acc', 'val_acc']),
        callbacks: {
            onEpochEnd: (e, logs) => {
                statusDiv.innerHTML = `Epoch ${e+1}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc.toFixed(4)}`;
            }
        }
    });

    validationPredictions = model.predict(validationData);
    document.getElementById('threshold-slider').disabled = false;
    document.getElementById('predict-btn').disabled = false;
    updateMetrics();
}

async function updateMetrics() {
    if (!validationPredictions) return;
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);

    const preds = validationPredictions.dataSync();
    const actuals = validationLabels.dataSync();
    let tp=0, tn=0, fp=0, fn=0;

    for(let i=0; i<preds.length; i++) {
        const p = preds[i] >= threshold ? 1 : 0;
        const a = actuals[i];
        if(p===1 && a===1) tp++; else if(p===0 && a===0) tn++;
        else if(p===1 && a===0) fp++; else fn++;
    }

    document.getElementById('confusion-matrix').innerHTML = `
        <table>
            <tr><td></td><th>Pred 1</th><th>Pred 0</th></tr>
            <tr><th>Act 1</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Act 0</th><td>${fp}</td><td>${tn}</td></tr>
        </table>`;
    
    const acc = (tp+tn)/(tp+tn+fp+fn);
    document.getElementById('performance-metrics').innerHTML = `Accuracy: ${(acc*100).toFixed(2)}%`;
}

async function predict() {
    const testX = tf.tensor2d(preprocessedTestData.features);
    testPredictions = model.predict(testX);
    const results = testPredictions.dataSync();
    
    const output = document.getElementById('prediction-output');
    output.innerHTML = '<h3>Results (First 10)</h3>';
    const table = document.createElement('table');
    table.innerHTML = '<tr><th>ID</th><th>Survived</th><th>Prob</th></tr>';
    
    for(let i=0; i<10; i++) {
        table.innerHTML += `<tr><td>${preprocessedTestData.passengerIds[i]}</td><td>${results[i]>=0.5?1:0}</td><td>${results[i].toFixed(4)}</td></tr>`;
    }
    output.appendChild(table);
    document.getElementById('export-btn').disabled = false;
}

async function exportResults() {
    const preds = testPredictions.dataSync();
    let csv = 'PassengerId,Survived\n';
    preprocessedTestData.passengerIds.forEach((id, i) => {
        csv += `${id},${preds[i] >= 0.5 ? 1 : 0}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'submission.csv';
    a.click();
    document.getElementById('export-status').innerText = 'Exported to submission.csv';
}
