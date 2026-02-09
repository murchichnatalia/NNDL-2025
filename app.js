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
let featureNames = []; // Added to track feature names for importance analysis

// Schema configuration
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];

// 1. FIXED: Robust CSV Parser to handle escaped commas in quotes
function parseCSV(csvText) {
    const lines = csvText.split(/\r?\n/).filter(line => line.trim() !== '');
    const headers = splitCSVLine(lines[0]);
    
    return lines.slice(1).map(line => {
        const values = splitCSVLine(line);
        const obj = {};
        headers.forEach((header, i) => {
            let val = values[i] === undefined || values[i] === '' ? null : values[i].replace(/^"|"$/g, '');
            
            // Convert numerical values
            if (val !== null && !isNaN(val) && val.trim() !== "") {
                obj[header] = parseFloat(val);
            } else {
                obj[header] = val;
            }
        });
        return obj;
    });
}

// Helper to split line by comma but ignore commas inside double quotes
function splitCSVLine(line) {
    const result = [];
    let curVal = "";
    let inQuotes = false;
    for (let char of line) {
        if (char === '"') inQuotes = !inQuotes;
        else if (char === ',' && !inQuotes) {
            result.push(curVal.trim());
            curVal = "";
        } else curVal += char;
    }
    result.push(curVal.trim());
    return result;
}

async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    if (!trainFile || !testFile) {
        alert('Please upload both files.');
        return;
    }
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading...';
    
    try {
        trainData = parseCSV(await readFile(trainFile));
        testData = parseCSV(await readFile(testFile));
        statusDiv.innerHTML = `Loaded: ${trainData.length} train, ${testData.length} test.`;
        document.getElementById('inspect-btn').disabled = false;
    } catch (e) {
        statusDiv.innerHTML = `Error: ${e.message}`;
    }
}

// 2. FIXED: Preprocessing with Feature Name Tracking
function preprocessData() {
    try {
        const ageMedian = calculateMedian(trainData.map(r => r.Age).filter(v => v != null));
        const fareMedian = calculateMedian(trainData.map(r => r.Fare).filter(v => v != null));
        const ageStd = calculateStdDev(trainData.map(r => r.Age).filter(v => v != null)) || 1;
        const fareStd = calculateStdDev(trainData.map(r => r.Fare).filter(v => v != null)) || 1;
        const embarkedMode = calculateMode(trainData.map(r => r.Embarked).filter(v => v != null));

        // Define Feature Names for the Importance Chart
        featureNames = [
            'Standardized Age', 'Standardized Fare', 'SibSp', 'Parch',
            'Class 1', 'Class 2', 'Class 3', 
            'Male', 'Female', 
            'Embarked C', 'Embarked Q', 'Embarked S'
        ];
        if (document.getElementById('add-family-features').checked) {
            featureNames.push('FamilySize', 'IsAlone');
        }

        preprocessedTrainData = { features: [], labels: [] };
        trainData.forEach(row => {
            preprocessedTrainData.features.push(extractFeatures(row, ageMedian, fareMedian, embarkedMode, ageStd, fareStd));
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });

        preprocessedTestData = { features: [], passengerIds: [] };
        testData.forEach(row => {
            preprocessedTestData.features.push(extractFeatures(row, ageMedian, fareMedian, embarkedMode, ageStd, fareStd));
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });

        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);
        
        document.getElementById('preprocessing-output').innerHTML = "Data Ready for Training.";
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        console.error(error);
    }
}

function extractFeatures(row, ageMed, fareMed, embMode, ageStd, fareStd) {
    const age = ( (row.Age || ageMed) - ageMed ) / ageStd;
    const fare = ( (row.Fare || fareMed) - fareMed ) / fareStd;
    const pclass = oneHotEncode(row.Pclass, [1, 2, 3]);
    const sex = oneHotEncode(row.Sex, ['male', 'female']);
    const emb = oneHotEncode(row.Embarked || embMode, ['C', 'Q', 'S']);

    let feats = [age, fare, row.SibSp || 0, row.Parch || 0, ...pclass, ...sex, ...emb];
    if (document.getElementById('add-family-features').checked) {
        const sz = (row.SibSp || 0) + (row.Parch || 0) + 1;
        feats.push(sz, sz === 1 ? 1 : 0);
    }
    return feats;
}

// 3. FIXED & ADDED: Evaluation Logic + Feature Importance (Sigmoid Gate analysis)
async function trainModel() {
    const statusDiv = document.getElementById('training-status');
    const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
    
    const trainX = preprocessedTrainData.features.slice(0, splitIndex);
    const trainY = preprocessedTrainData.labels.slice(0, splitIndex);
    validationData = preprocessedTrainData.features.slice(splitIndex);
    validationLabels = preprocessedTrainData.labels.slice(splitIndex);

    await model.fit(trainX, trainY, {
        epochs: 50,
        batchSize: 32,
        validationData: [validationData, validationLabels],
        callbacks: tfvis.show.fitCallbacks({ name: 'Performance' }, ['loss', 'acc', 'val_loss', 'val_acc'])
    });

    statusDiv.innerHTML = "Training Complete.";
    validationPredictions = model.predict(validationData);
    
    // Trigger the Evaluation Table and Importance Chart
    updateMetrics();
    analyzeImportance(); 
    
    document.getElementById('predict-btn').disabled = false;
    document.getElementById('threshold-slider').disabled = false;
}

// Feature Importance Logic
function analyzeImportance() {
    // We look at the weights of the first layer
    // These act as the "gate" for incoming data
    const weights = model.layers[0].getWeights()[0];
    const absWeights = tf.abs(weights).sum(1); // Sum absolute weights across neurons
    const importanceValues = absWeights.arraySync();
    
    const data = featureNames.map((name, i) => ({
        index: name,
        value: importanceValues[i]
    })).sort((a, b) => b.value - a.value);

    tfvis.render.barchart(
        { name: 'Feature Importance (Input Gate Weights)', tab: 'Evaluation' },
        data,
        { xLabel: 'Feature', yLabel: 'Weight Magnitude' }
    );
}

async function updateMetrics() {
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);

    const preds = validationPredictions.arraySync();
    const actuals = validationLabels.arraySync();

    let tp = 0, tn = 0, fp = 0, fn = 0;
    preds.forEach((p, i) => {
        const pred = p >= threshold ? 1 : 0;
        const act = actuals[i];
        if (pred === 1 && act === 1) tp++;
        else if (pred === 0 && act === 0) tn++;
        else if (pred === 1 && act === 0) fp++;
        else fn++;
    });

    // Ensure this ID exists in your HTML
    const cmDiv = document.getElementById('confusion-matrix');
    if (cmDiv) {
        cmDiv.innerHTML = `
            <table border="1" style="border-collapse: collapse; width: 100%; text-align: center;">
                <tr style="background: #f2f2f2;"><th></th><th>Pred Survive</th><th>Pred Die</th></tr>
                <tr><th>Actual Survive</th><td>${tp} (TP)</td><td>${fn} (FN)</td></tr>
                <tr><th>Actual Die</th><td>${fp} (FP)</td><td>${tn} (TN)</td></tr>
            </table>`;
    }

    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const acc = (tp + tn) / actuals.length;

    document.getElementById('performance-metrics').innerHTML = `
        <ul>
            <li><strong>Accuracy:</strong> ${(acc * 100).toFixed(2)}%</li>
            <li><strong>Precision:</strong> ${precision.toFixed(3)}</li>
            <li><strong>Recall:</strong> ${recall.toFixed(3)}</li>
        </ul>`;
}

// Helper functions (calculateMedian, calculateStdDev, calculateMode, etc. remains same as yours)
