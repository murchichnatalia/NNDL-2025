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

// Schema configuration - change these for different datasets
const TARGET_FEATURE = 'Survived'; // Binary classification target
const ID_FEATURE = 'PassengerId'; // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']; // Numerical features
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']; // Categorical features

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners to buttons
    document.getElementById('load-data-btn').addEventListener('click', loadData);
    document.getElementById('inspect-btn').addEventListener('click', inspectData);
    document.getElementById('preprocess-btn').addEventListener('click', preprocessData);
    document.getElementById('create-model-btn').addEventListener('click', createModel);
    document.getElementById('train-btn').addEventListener('click', trainModel);
    document.getElementById('predict-btn').addEventListener('click', predict);
    document.getElementById('export-btn').addEventListener('click', exportResults);
    
    // Initialize threshold slider
    const thresholdSlider = document.getElementById('threshold-slider');
    thresholdSlider.addEventListener('input', updateMetrics);
});

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
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
        
        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error(error);
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Parse CSV text to array of objects - FIXED COMMA ESCAPE ISSUE
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) return [];
    
    // Parse headers
    const headers = parseCSVLine(lines[0]);
    
    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        const obj = {};
        
        headers.forEach((header, idx) => {
            let value = values[idx] || '';
            // Handle missing values
            if (value === '' || value === 'null' || value === 'NA' || value === 'NaN') {
                obj[header] = null;
            } else {
                // Try to convert to number if possible
                const numValue = parseFloat(value);
                obj[header] = isNaN(numValue) ? value : numValue;
            }
        });
        
        // Ensure all required fields exist
        if (obj['PassengerId'] !== undefined) {
            data.push(obj);
        }
    }
    
    return data;
}

// Helper function to parse CSV line with quotes handling
function parseCSVLine(line) {
    const result = [];
    let inQuotes = false;
    let currentValue = '';
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            result.push(currentValue);
            currentValue = '';
        } else {
            currentValue += char;
        }
    }
    
    // Add last value
    result.push(currentValue);
    
    // Clean quotes from values
    return result.map(value => {
        if (value.startsWith('"') && value.endsWith('"')) {
            return value.substring(1, value.length - 1);
        }
        return value;
    });
}

// Inspect the loaded data
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }
    
    // Show data preview
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';
    
    const columns = Object.keys(trainData[0]);
    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${columns.length} columns`;
    
    // Calculate survival rate
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;
    
    // Calculate missing values percentage for each feature
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    columns.forEach(feature => {
        const missingCount = trainData.filter(row => 
            row[feature] === null || 
            row[feature] === undefined || 
            row[feature] === ''
        ).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li><strong>${feature}</strong>: ${missingPercent}%</li>`;
    });
    missingInfo += '</ul>';
    
    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;
    
    // Create visualizations
    createVisualizations();
    
    // Enable the preprocess button
    document.getElementById('preprocess-btn').disabled = false;
}

// Create a preview table from data
function createPreviewTable(data) {
    const table = document.createElement('table');
    table.style.borderCollapse = 'collapse';
    table.style.width = '100%';
    table.style.margin = '10px 0';
    
    // Create header row
    const headerRow = document.createElement('tr');
    const headers = Object.keys(data[0] || {});
    headers.forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        th.style.border = '1px solid #ddd';
        th.style.padding = '8px';
        th.style.backgroundColor = '#f2f2f2';
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        headers.forEach(key => {
            const td = document.createElement('td');
            const value = row[key];
            if (value === null || value === undefined) {
                td.textContent = 'NULL';
                td.style.color = '#999';
            } else if (typeof value === 'string' && value.length > 30) {
                td.textContent = value.substring(0, 30) + '...';
            } else {
                td.textContent = value;
            }
            td.style.border = '1px solid #ddd';
            td.style.padding = '8px';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// Create visualizations using tfjs-vis
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    // Open tfjs-vis visor
    if (!tfvis.visor().isOpen()) {
        tfvis.visor().open();
    }
    
    // Survival by Sex
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex && row[TARGET_FEATURE] !== undefined && row[TARGET_FEATURE] !== null) {
            if (!survivalBySex[row.Sex]) {
                survivalBySex[row.Sex] = { survived: 0, total: 0 };
            }
            survivalBySex[row.Sex].total++;
            if (row[TARGET_FEATURE] === 1) {
                survivalBySex[row.Sex].survived++;
            }
        }
    });
    
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        sex,
        survivalRate: stats.total > 0 ? (stats.survived / stats.total) * 100 : 0
    }));
    
    if (sexData.length > 0) {
        tfvis.render.barchart(
            { name: 'Survival Rate by Sex', tab: 'Data Analysis' },
            sexData.map(d => ({ x: d.sex, y: d.survivalRate })),
            { 
                xLabel: 'Sex', 
                yLabel: 'Survival Rate (%)',
                width: 400,
                height: 300
            }
        );
    }
    
    // Survival by Pclass
    const survivalByPclass = {};
    trainData.forEach(row => {
        if (row.Pclass !== undefined && row.Pclass !== null && 
            row[TARGET_FEATURE] !== undefined && row[TARGET_FEATURE] !== null) {
            if (!survivalByPclass[row.Pclass]) {
                survivalByPclass[row.Pclass] = { survived: 0, total: 0 };
            }
            survivalByPclass[row.Pclass].total++;
            if (row[TARGET_FEATURE] === 1) {
                survivalByPclass[row.Pclass].survived++;
            }
        }
    });
    
    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
        pclass: `Class ${pclass}`,
        survivalRate: stats.total > 0 ? (stats.survived / stats.total) * 100 : 0
    }));
    
    if (pclassData.length > 0) {
        tfvis.render.barchart(
            { name: 'Survival Rate by Passenger Class', tab: 'Data Analysis' },
            pclassData.map(d => ({ x: d.pclass, y: d.survivalRate })),
            { 
                xLabel: 'Passenger Class', 
                yLabel: 'Survival Rate (%)',
                width: 400,
                height: 300
            }
        );
    }
    
    chartsDiv.innerHTML += '<p>Charts are displayed in the tfjs-vis visor. Click the button in the bottom right to view.</p>';
}

// Preprocess the data
async function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }
    
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';
    
    try {
        // Filter out rows with null target values for training
        const validTrainData = trainData.filter(row => 
            row[TARGET_FEATURE] !== null && row[TARGET_FEATURE] !== undefined
        );
        
        // Calculate imputation values from training data
        const ageValues = validTrainData.map(row => row.Age).filter(age => 
            age !== null && age !== undefined && !isNaN(age)
        );
        const ageMedian = calculateMedian(ageValues) || 28;
        
        const fareValues = validTrainData.map(row => row.Fare).filter(fare => 
            fare !== null && fare !== undefined && !isNaN(fare)
        );
        const fareMedian = calculateMedian(fareValues) || 14.45;
        
        const embarkedValues = validTrainData.map(row => row.Embarked).filter(e => 
            e !== null && e !== undefined && e !== ''
        );
        const embarkedMode = calculateMode(embarkedValues) || 'S';
        
        // Calculate standardization values
        const ageStd = calculateStdDev(ageValues) || 14.5;
        const fareStd = calculateStdDev(fareValues) || 49.7;
        
        // Preprocess training data
        const trainFeatures = [];
        const trainLabels = [];
        
        validTrainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode, ageStd, fareStd);
            if (features) {
                trainFeatures.push(features);
                trainLabels.push(row[TARGET_FEATURE]);
            }
        });
        
        // Preprocess test data
        const testFeatures = [];
        const testPassengerIds = [];
        
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode, ageStd, fareStd);
            if (features) {
                testFeatures.push(features);
                testPassengerIds.push(row[ID_FEATURE]);
            }
        });
        
        // Store preprocessed data
        preprocessedTrainData = {
            features: trainFeatures,
            labels: trainLabels
        };
        
        preprocessedTestData = {
            features: testFeatures,
            passengerIds: testPassengerIds
        };
        
        outputDiv.innerHTML = `
            <p>✓ Preprocessing completed!</p>
            <p><strong>Training samples:</strong> ${trainFeatures.length}</p>
            <p><strong>Number of features:</strong> ${trainFeatures[0] ? trainFeatures[0].length : 0}</p>
            <p><strong>Test samples:</strong> ${testFeatures.length}</p>
            <p><strong>Imputation values:</strong> Age median: ${ageMedian.toFixed(2)}, Fare median: ${fareMedian.toFixed(2)}, Embarked mode: ${embarkedMode}</p>
        `;
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `✗ Error during preprocessing: ${error.message}`;
        console.error(error);
    }
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode, ageStd, fareStd) {
    try {
        // Impute missing values
        const age = (row.Age !== null && row.Age !== undefined && !isNaN(row.Age)) ? row.Age : ageMedian;
        const fare = (row.Fare !== null && row.Fare !== undefined && !isNaN(row.Fare)) ? row.Fare : fareMedian;
        const embarked = (row.Embarked !== null && row.Embarked !== undefined && row.Embarked !== '') ? row.Embarked : embarkedMode;
        
        // Standardize numerical features
        const standardizedAge = (age - ageMedian) / ageStd;
        const standardizedFare = (fare - fareMedian) / fareStd;
        
        // One-hot encode categorical features
        const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]); // Pclass values: 1, 2, 3
        const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
        const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
        
        // Start with numerical features
        let features = [
            standardizedAge,
            standardizedFare,
            row.SibSp || 0,
            row.Parch || 0
        ];
        
        // Add one-hot encoded features
        features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);
        
        // Add optional family features if enabled
        if (document.getElementById('add-family-features').checked) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            features.push(familySize, isAlone);
        }
        
        return features;
    } catch (error) {
        console.warn('Error extracting features for row:', row, error);
        return null;
    }
}

// Calculate median of an array
function calculateMedian(values) {
    if (!values || values.length === 0) return 0;
    
    const sorted = [...values].sort((a, b) => a - b);
    const half = Math.floor(sorted.length / 2);
    
    if (sorted.length % 2 === 0) {
        return (sorted[half - 1] + sorted[half]) / 2;
    }
    
    return sorted[half];
}

// Calculate mode of an array
function calculateMode(values) {
    if (!values || values.length === 0) return 'S'; // Default to Southampton
    
    const frequency = {};
    let maxCount = 0;
    let mode = values[0];
    
    values.forEach(value => {
        frequency[value] = (frequency[value] || 0) + 1;
        if (frequency[value] > maxCount) {
            maxCount = frequency[value];
            mode = value;
        }
    });
    
    return mode;
}

// Calculate standard deviation of an array
function calculateStdDev(values) {
    if (!values || values.length < 2) return 1;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    return Math.sqrt(variance);
}

// One-hot encode a value
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    if (value === null || value === undefined) {
        // Default to first category if value is null
        if (categories.length > 0) encoding[0] = 1;
    } else {
        const index = categories.indexOf(value);
        if (index !== -1) {
            encoding[index] = 1;
        } else if (categories.length > 0) {
            // Default to first category if value not found
            encoding[0] = 1;
        }
    }
    return encoding;
}

// Create the model with sigmoid activation in hidden layer for feature importance
function createModel() {
    if (!preprocessedTrainData || !preprocessedTrainData.features.length) {
        alert('Please preprocess data first.');
        return;
    }
    
    const inputShape = preprocessedTrainData.features[0].length;
    
    // Clear any existing model
    if (model) {
        model.dispose();
    }
    
    // Create a sequential model with sigmoid activation for feature importance analysis
    model = tf.sequential();
    
    // Add layers - Using sigmoid in hidden layer to better understand feature importance
    model.add(tf.layers.dense({
        units: 16,
        activation: 'sigmoid', // Changed to sigmoid for better feature importance interpretation
        inputShape: [inputShape],
        kernelInitializer: 'glorotNormal',
        biasInitializer: 'zeros'
    }));
    
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    // Compile the model
    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    
    let summaryText = '<ul>';
    model.layers.forEach((layer, i) => {
        const layerType = layer.getClassName();
        const units = layer.units || 'N/A';
        const activation = layer.activation ? layer.activation.getClassName() : 'None';
        summaryText += `<li><strong>Layer ${i+1}:</strong> ${layerType} - Units: ${units} - Activation: ${activation} - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p><strong>Total parameters:</strong> ${model.countParams().toLocaleString()}</p>`;
    summaryText += `<p><strong>Input shape:</strong> [${inputShape}]</p>`;
    summaryText += `<p><strong>Note:</strong> Using sigmoid activation in hidden layer for better feature importance interpretation.</p>`;
    summaryDiv.innerHTML += summaryText;
    
    // Enable the train button
    document.getElementById('train-btn').disabled = false;
}

// Train the model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }
    
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';
    
    try {
        // Convert data to tensors
        const featuresTensor = tf.tensor2d(preprocessedTrainData.features);
        const labelsTensor = tf.tensor1d(preprocessedTrainData.labels);
        
        // Split training data into train and validation sets (80/20)
        const splitIndex = Math.floor(featuresTensor.shape[0] * 0.8);
        
        const trainFeatures = featuresTensor.slice([0, 0], [splitIndex, -1]);
        const trainLabels = labelsTensor.slice([0], [splitIndex]);
        
        const valFeatures = featuresTensor.slice([splitIndex, 0], [-1, -1]);
        const valLabels = labelsTensor.slice([splitIndex], [-1]);
        
        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Show tfjs-vis visor for training charts
        if (!tfvis.visor().isOpen()) {
            tfvis.visor().open();
        }
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance', tab: 'Training' },
                ['loss', 'acc', 'val_loss', 'val_acc'],
                { 
                    callbacks: ['onEpochEnd'],
                    height: 300,
                    width: 500
                }
            )
        });
        
        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);
        
        // Enable the threshold slider
        const thresholdSlider = document.getElementById('threshold-slider');
        thresholdSlider.disabled = false;
        
        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;
        
        statusDiv.innerHTML = '<p>✓ Training completed successfully!</p>';
        
        // Calculate initial metrics
        await updateMetrics();
        
        // Clean up tensors
        featuresTensor.dispose();
        labelsTensor.dispose();
        trainFeatures.dispose();
        trainLabels.dispose();
        
    } catch (error) {
        statusDiv.innerHTML = `✗ Error during training: ${error.message}`;
        console.error(error);
    }
}

// Update metrics based on threshold - FIXED EVALUATION TABLE ISSUE
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    try {
        // Get predictions as array
        const predVals = await validationPredictions.array();
        const trueVals = await validationLabels.array();
        
        let tp = 0, tn = 0, fp = 0, fn = 0;
        
        for (let i = 0; i < predVals.length; i++) {
            const prediction = predVals[i][0] >= threshold ? 1 : 0;
            const actual = trueVals[i];
            
            if (prediction === 1 && actual === 1) tp++;
            else if (prediction === 0 && actual === 0) tn++;
            else if (prediction === 1 && actual === 0) fp++;
            else if (prediction === 0 && actual === 1) fn++;
        }
        
        // Update confusion matrix display
        const cmDiv = document.getElementById('confusion-matrix');
        cmDiv.innerHTML = `
            <table style="border-collapse: collapse; margin: 10px 0;">
                <tr>
                    <th style="border: 1px solid #ddd; padding: 8px;"></th>
                    <th style="border: 1px solid #ddd; padding: 8px; background: #e6f7ff;">Predicted Positive</th>
                    <th style="border: 1px solid #ddd; padding: 8px; background: #fff2e6;">Predicted Negative</th>
                </tr>
                <tr>
                    <th style="border: 1px solid #ddd; padding: 8px; background: #e6f7ff;">Actual Positive</th>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background: #d9f7be;">${tp}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background: #ffccc7;">${fn}</td>
                </tr>
                <tr>
                    <th style="border: 1px solid #ddd; padding: 8px; background: #fff2e6;">Actual Negative</th>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background: #ffccc7;">${fp}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center; background: #d9f7be;">${tn}</td>
                </tr>
            </table>
        `;
        
        // Calculate performance metrics
        const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
        const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
        const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
        const accuracy = tp + tn + fp + fn > 0 ? (tp + tn) / (tp + tn + fp + fn) : 0;
        
        // Update performance metrics display
        const metricsDiv = document.getElementById('performance-metrics');
        metricsDiv.innerHTML = `
            <p><strong>Accuracy:</strong> ${(accuracy * 100).toFixed(2)}%</p>
            <p><strong>Precision:</strong> ${precision.toFixed(4)}</p>
            <p><strong>Recall:</strong> ${recall.toFixed(4)}</p>
            <p><strong>F1 Score:</strong> ${f1.toFixed(4)}</p>
        `;
        
        // Calculate and plot ROC curve
        await plotROC(trueVals, predVals.map(p => p[0]));
        
    } catch (error) {
        console.error('Error updating metrics:', error);
    }
}

// Plot ROC curve
async function plotROC(trueLabels, predictions) {
    try {
        // Calculate TPR and FPR for different thresholds
        const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
        const rocData = [];
        
        thresholds.forEach(threshold => {
            let tp = 0, fn = 0, fp = 0, tn = 0;
            
            for (let i = 0; i < predictions.length; i++) {
                const prediction = predictions[i] >= threshold ? 1 : 0;
                const actual = trueLabels[i];
                
                if (actual === 1) {
                    if (prediction === 1) tp++;
                    else fn++;
                } else {
                    if (prediction === 1) fp++;
                    else tn++;
                }
            }
            
            const tpr = tp + fn > 0 ? tp / (tp + fn) : 0;
            const fpr = fp + tn > 0 ? fp / (fp + tn) : 0;
            
            rocData.push({ threshold, fpr, tpr });
        });
        
        // Calculate AUC (approximate using trapezoidal rule)
        let auc = 0;
        for (let i = 1; i < rocData.length; i++) {
            auc += (rocData[i].fpr - rocData[i-1].fpr) * (rocData[i].tpr + rocData[i-1].tpr) / 2;
        }
        
        // Plot ROC curve
        tfvis.render.linechart(
            { name: 'ROC Curve', tab: 'Evaluation' },
            { values: rocData.map(d => ({ x: d.fpr, y: d.tpr })) },
            { 
                xLabel: 'False Positive Rate', 
                yLabel: 'True Positive Rate',
                series: ['ROC Curve'],
                width: 400,
                height: 400
            }
        );
        
        // Add AUC to performance metrics
        const metricsDiv = document.getElementById('performance-metrics');
        metricsDiv.innerHTML += `<p><strong>AUC:</strong> ${auc.toFixed(4)}</p>`;
        
    } catch (error) {
        console.error('Error plotting ROC curve:', error);
    }
}

// Predict on test data - FIXED row[key].toFixed is not a function ERROR
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }
    
    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';
    
    try {
        // Convert test features to tensor
        const testFeaturesTensor = tf.tensor2d(preprocessedTestData.features);
        
        // Make predictions
        const predictionsTensor = model.predict(testFeaturesTensor);
        const predValues = await predictionsTensor.array();
        
        // Store predictions
        testPredictions = {
            tensor: predictionsTensor,
            values: predValues
        };
        
        // Create prediction results - FIXED the toFixed error
        const results = [];
        for (let i = 0; i < preprocessedTestData.passengerIds.length; i++) {
            const id = preprocessedTestData.passengerIds[i];
            const prob = predValues[i][0];
            
            results.push({
                PassengerId: id,
                Survived: prob >= 0.5 ? 1 : 0,
                Probability: prob
            });
        }
        
        // Show first 10 predictions
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        
        outputDiv.innerHTML += `<p>✓ Predictions completed! Total: ${results.length} samples</p>`;
        
        // Enable the export button
        document.getElementById('export-btn').disabled = false;
        
        // Clean up tensor
        testFeaturesTensor.dispose();
        
    } catch (error) {
        outputDiv.innerHTML = `✗ Error during prediction: ${error.message}`;
        console.error(error);
    }
}

// Create prediction table
function createPredictionTable(data) {
    const table = document.createElement('table');
    table.style.borderCollapse = 'collapse';
    table.style.width = '100%';
    table.style.margin = '10px 0';
    
    // Create header row
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        th.style.border = '1px solid #ddd';
        th.style.padding = '8px';
        th.style.backgroundColor = '#f2f2f2';
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        
        // PassengerId
        const tdId = document.createElement('td');
        tdId.textContent = row.PassengerId;
        tdId.style.border = '1px solid #ddd';
        tdId.style.padding = '8px';
        tr.appendChild(tdId);
        
        // Survived
        const tdSurvived = document.createElement('td');
        tdSurvived.textContent = row.Survived;
        tdSurvived.style.border = '1px solid #ddd';
        tdSurvived.style.padding = '8px';
        tr.appendChild(tdSurvived);
        
        // Probability - FIXED: Use toFixed only if it's a number
        const tdProb = document.createElement('td');
        const probValue = row.Probability;
        if (typeof probValue === 'number' && !isNaN(probValue)) {
            tdProb.textContent = probValue.toFixed(4);
        } else {
            tdProb.textContent = 'N/A';
        }
        tdProb.style.border = '1px solid #ddd';
        tdProb.style.padding = '8px';
        tr.appendChild(tdProb);
        
        table.appendChild(tr);
    });
    
    return table;
}

// Export results
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }
    
    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';
    
    try {
        // Create submission CSV (PassengerId, Survived)
        let submissionCSV = 'PassengerId,Survived\n';
        for (let i = 0; i < preprocessedTestData.passengerIds.length; i++) {
            const id = preprocessedTestData.passengerIds[i];
            const prob = testPredictions.values[i][0];
            submissionCSV += `${id},${prob >= 0.5 ? 1 : 0}\n`;
        }
        
        // Create probabilities CSV (PassengerId, Probability)
        let probabilitiesCSV = 'PassengerId,Probability\n';
        for (let i = 0; i < preprocessedTestData.passengerIds.length; i++) {
            const id = preprocessedTestData.passengerIds[i];
            const prob = testPredictions.values[i][0];
            probabilitiesCSV += `${id},${prob.toFixed(6)}\n`;
        }
        
        // Create download links
        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
        submissionLink.download = 'titanic_submission.csv';
        submissionLink.click();
        
        // Small delay before second download
        setTimeout(() => {
            const probabilitiesLink = document.createElement('a');
            probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
            probabilitiesLink.download = 'titanic_probabilities.csv';
            probabilitiesLink.click();
            
            // Try to save model
            try {
                model.save('downloads://titanic-tfjs-model');
                statusDiv.innerHTML = `
                    <p>✓ Export completed successfully!</p>
                    <p>• Downloaded: <strong>titanic_submission.csv</strong> (Kaggle submission format)</p>
                    <p>• Downloaded: <strong>titanic_probabilities.csv</strong> (Prediction probabilities)</p>
                    <p>• Model saved to browser downloads as "titanic-tfjs-model"</p>
                `;
            } catch (saveError) {
                statusDiv.innerHTML = `
                    <p>✓ Export completed!</p>
                    <p>• Downloaded: <strong>titanic_submission.csv</strong> (Kaggle submission format)</p>
                    <p>• Downloaded: <strong>titanic_probabilities.csv</strong> (Prediction probabilities)</p>
                    <p>Note: Model save not supported in this browser</p>
                `;
            }
        }, 100);
        
   
