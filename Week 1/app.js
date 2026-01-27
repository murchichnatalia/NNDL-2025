// app.js
/**
 * Titanic Dataset EDA Explorer
 * A client-side interactive exploratory data analysis tool for the Kaggle Titanic dataset
 * Reusable for other datasets by modifying the schema below
 */

// ============================================
// DATA SCHEMA CONFIGURATION
// ============================================
// TARGET: Survived (0/1, train only)
// FEATURES: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
// IDENTIFIER: PassengerId (excluded from analysis)
// REUSE NOTE: Swap this schema for other datasets by updating:
//   - targetColumn, featureColumns, idColumn
//   - numericFeatures, categoricalFeatures
//   - visualization configurations
const DATA_SCHEMA = {
    targetColumn: 'Survived',  // Binary target variable (train only)
    featureColumns: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    idColumn: 'PassengerId',   // Identifier column (excluded from analysis)
    numericFeatures: ['Age', 'SibSp', 'Parch', 'Fare'],
    categoricalFeatures: ['Pclass', 'Sex', 'Embarked']
};

// Global variables
let mergedData = null;
let trainData = null;
let testData = null;
let charts = {}; // Store Chart.js instances for later updates

// ============================================
// DOM ELEMENT REFERENCES
// ============================================
const dom = {
    // File inputs
    trainFileInput: document.getElementById('train-file'),
    testFileInput: document.getElementById('test-file'),
    trainStatus: document.getElementById('train-status'),
    testStatus: document.getElementById('test-status'),
    
    // Buttons
    loadDataBtn: document.getElementById('load-data-btn'),
    useSampleBtn: document.getElementById('use-sample-btn'),
    togglePreviewBtn: document.getElementById('toggle-preview-btn'),
    runStatsBtn: document.getElementById('run-stats-btn'),
    generateVisualsBtn: document.getElementById('generate-visuals-btn'),
    exportCsvBtn: document.getElementById('export-csv-btn'),
    exportJsonBtn: document.getElementById('export-json-btn'),
    
    // Content sections
    overviewContent: document.getElementById('overview-content'),
    overviewPlaceholder: document.getElementById('overview-placeholder'),
    missingValuesContent: document.getElementById('missing-values-content'),
    missingValuesPlaceholder: document.getElementById('missing-values-placeholder'),
    statsSummaryContent: document.getElementById('stats-summary-content'),
    statsSummaryPlaceholder: document.getElementById('stats-summary-placeholder'),
    visualizationsContent: document.getElementById('visualizations-content'),
    visualizationsPlaceholder: document.getElementById('visualizations-placeholder'),
    
    // Message displays
    dataLoadMessage: document.getElementById('data-load-message'),
    exportMessage: document.getElementById('export-message'),
    
    // Statistics displays
    totalPassengers: document.getElementById('total-passengers'),
    trainSamples: document.getElementById('train-samples'),
    testSamples: document.getElementById('test-samples'),
    featureCount: document.getElementById('feature-count'),
    
    // Data preview
    dataPreviewContainer: document.getElementById('data-preview-container'),
    dataPreview: document.getElementById('data-preview'),
    
    // Charts
    missingValuesChart: document.getElementById('missing-values-chart'),
    sexChart: document.getElementById('sex-chart'),
    pclassChart: document.getElementById('pclass-chart'),
    embarkedChart: document.getElementById('embarked-chart'),
    ageChart: document.getElementById('age-chart'),
    fareChart: document.getElementById('fare-chart'),
    correlationChart: document.getElementById('correlation-chart'),
    
    // Stats containers
    missingValuesTableContainer: document.getElementById('missing-values-table-container'),
    numericStatsContainer: document.getElementById('numeric-stats-container'),
    categoricalStatsContainer: document.getElementById('categorical-stats-container'),
    survivalStatsContainer: document.getElementById('survival-stats-container')
};

// ============================================
// EVENT LISTENERS SETUP
// ============================================
function setupEventListeners() {
    // File input change events
    dom.trainFileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            const file = files[0];
            dom.trainStatus.textContent = `✅ Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            dom.trainStatus.className = 'file-status loaded';
        } else {
            dom.trainStatus.textContent = 'No file selected';
            dom.trainStatus.className = 'file-status';
        }
    });
    
    dom.testFileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            const file = files[0];
            dom.testStatus.textContent = `✅ Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            dom.testStatus.className = 'file-status loaded';
        } else {
            dom.testStatus.textContent = 'No file selected';
            dom.testStatus.className = 'file-status';
        }
    });
    
    // Button click events
    dom.loadDataBtn.addEventListener('click', loadAndMergeData);
    dom.useSampleBtn.addEventListener('click', loadSampleData);
    dom.togglePreviewBtn.addEventListener('click', toggleDataPreview);
    dom.runStatsBtn.addEventListener('click', calculateStatistics);
    dom.generateVisualsBtn.addEventListener('click', generateVisualizations);
    dom.exportCsvBtn.addEventListener('click', exportMergedCSV);
    dom.exportJsonBtn.addEventListener('click', exportJSONSummary);
    
    // Initialize UI state
    updateUIState('initial');
}

// ============================================
// DATA LOADING AND MERGING
// ============================================
/**
 * Load and merge train.csv and test.csv files
 */
async function loadAndMergeData() {
    // Get files properly
    const trainFile = dom.trainFileInput.files ? dom.trainFileInput.files[0] : null;
    const testFile = dom.testFileInput.files ? dom.testFileInput.files[0] : null;
    
    // Validate files are selected
    if (!trainFile || !testFile) {
        showMessage(dom.dataLoadMessage, 'Please select both train.csv and test.csv files.', 'warning');
        return;
    }
    
    // Validate file names (optional but helpful)
    if (!trainFile.name.toLowerCase().includes('train') || !testFile.name.toLowerCase().includes('test')) {
        showMessage(dom.dataLoadMessage, 'Warning: Make sure train.csv is the training file and test.csv is the test file.', 'warning');
    }
    
    showMessage(dom.dataLoadMessage, 'Loading and parsing CSV files...', 'info');
    
    try {
        // Parse train.csv
        const trainPromise = new Promise((resolve, reject) => {
            Papa.parse(trainFile, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                quotes: true, // Handle commas within quotes
                complete: (results) => {
                    if (results.errors && results.errors.length > 0) {
                        console.warn('CSV parse warnings:', results.errors);
                    }
                    
                    if (!results.data || results.data.length === 0) {
                        reject(new Error('Train CSV file is empty or could not be parsed'));
                        return;
                    }
                    
                    // Add source column
                    const dataWithSource = results.data.map(row => ({
                        ...row,
                        source: 'train'
                    }));
                    resolve(dataWithSource);
                },
                error: (error) => {
                    reject(new Error(`Failed to parse train.csv: ${error.message}`));
                }
            });
        });
        
        // Parse test.csv
        const testPromise = new Promise((resolve, reject) => {
            Papa.parse(testFile, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                quotes: true,
                complete: (results) => {
                    if (results.errors && results.errors.length > 0) {
                        console.warn('CSV parse warnings:', results.errors);
                    }
                    
                    if (!results.data || results.data.length === 0) {
                        reject(new Error('Test CSV file is empty or could not be parsed'));
                        return;
                    }
                    
                    // Add source column and missing Survived column
                    const dataWithSource = results.data.map(row => ({
                        ...row,
                        source: 'test',
                        Survived: null // Add Survived column with null for test data
                    }));
                    resolve(dataWithSource);
                },
                error: (error) => {
                    reject(new Error(`Failed to parse test.csv: ${error.message}`));
                }
            });
        });
        
        // Wait for both files to parse
        [trainData, testData] = await Promise.all([trainPromise, testPromise]);
        
        // Merge datasets
        mergedData = [...trainData, ...testData];
        
        // Validate data structure
        if (mergedData.length === 0) {
            throw new Error('No data loaded after merge');
        }
        
        // Check for required columns
        const firstRow = mergedData[0];
        const requiredColumns = [...DATA_SCHEMA.featureColumns, DATA_SCHEMA.idColumn];
        const missingColumns = requiredColumns.filter(col => !(col in firstRow));
        
        if (missingColumns.length > 0) {
            console.warn(`Missing columns: ${missingColumns.join(', ')}. Dataset might have different structure.`);
        }
        
        showMessage(dom.dataLoadMessage, `✅ Successfully loaded and merged ${mergedData.length} records (${trainData.length} train, ${testData.length} test)`, 'success');
        
        // Update UI
        updateUIState('dataLoaded');
        showOverview();
        
    } catch (error) {
        console.error('Error loading data:', error);
        showMessage(dom.dataLoadMessage, `❌ Error: ${error.message}`, 'warning');
        updateUIState('initial');
    }
}

/**
 * Load sample data for demonstration purposes
 */
function loadSampleData() {
    showMessage(dom.dataLoadMessage, 'Loading sample data... This is a demo with limited records.', 'info');
    
    // Create minimal sample data for demonstration
    // In a real scenario, you might want to include actual sample data
    trainData = [
        { PassengerId: 1, Survived: 0, Pclass: 3, Sex: 'male', Age: 22, SibSp: 1, Parch: 0, Fare: 7.25, Embarked: 'S' },
        { PassengerId: 2, Survived: 1, Pclass: 1, Sex: 'female', Age: 38, SibSp: 1, Parch: 0, Fare: 71.28, Embarked: 'C' },
        { PassengerId: 3, Survived: 1, Pclass: 3, Sex: 'female', Age: 26, SibSp: 0, Parch: 0, Fare: 7.92, Embarked: 'S' },
        { PassengerId: 4, Survived: 1, Pclass: 1, Sex: 'female', Age: 35, SibSp: 1, Parch: 0, Fare: 53.1, Embarked: 'S' },
        { PassengerId: 5, Survived: 0, Pclass: 3, Sex: 'male', Age: 35, SibSp: 0, Parch: 0, Fare: 8.05, Embarked: 'S' }
    ].map(row => ({ ...row, source: 'train' }));
    
    testData = [
        { PassengerId: 6, Pclass: 3, Sex: 'male', Age: null, SibSp: 0, Parch: 0, Fare: 8.46, Embarked: 'Q' },
        { PassengerId: 7, Pclass: 1, Sex: 'male', Age: 54, SibSp: 0, Parch: 0, Fare: 51.86, Embarked: 'S' },
        { PassengerId: 8, Pclass: 3, Sex: 'female', Age: 2, SibSp: 3, Parch: 1, Fare: 21.08, Embarked: 'S' }
    ].map(row => ({ ...row, source: 'test', Survived: null }));
    
    mergedData = [...trainData, ...testData];
    
    showMessage(dom.dataLoadMessage, `✅ Loaded ${mergedData.length} sample records (${trainData.length} train, ${testData.length} test)`, 'success');
    
    // Update UI
    updateUIState('dataLoaded');
    showOverview();
}

// ============================================
// UI STATE MANAGEMENT
// ============================================
/**
 * Update UI state based on current data status
 */
function updateUIState(state) {
    switch (state) {
        case 'initial':
            // Disable all analysis buttons initially
            dom.togglePreviewBtn.classList.add('hidden');
            dom.runStatsBtn.classList.add('hidden');
            dom.generateVisualsBtn.classList.add('hidden');
            dom.exportCsvBtn.disabled = true;
            dom.exportJsonBtn.disabled = true;
            break;
            
        case 'dataLoaded':
            // Enable analysis buttons when data is loaded
            dom.togglePreviewBtn.classList.remove('hidden');
            dom.runStatsBtn.classList.remove('hidden');
            dom.generateVisualsBtn.classList.remove('hidden');
            dom.exportCsvBtn.disabled = false;
            dom.exportJsonBtn.disabled = false;
            
            // Automatically run missing values analysis
            setTimeout(() => {
                analyzeMissingValues();
                calculateStatistics();
                generateVisualizations();
            }, 500);
            break;
    }
}

/**
 * Show message with styling
 */
function showMessage(element, text, type = 'info') {
    element.textContent = text;
    element.className = type; // 'info', 'success', 'warning'
    element.classList.remove('hidden');
    
    // Auto-hide info messages after 5 seconds
    if (type === 'info') {
        setTimeout(() => {
            element.classList.add('hidden');
        }, 5000);
    }
}

// ============================================
// OVERVIEW AND DATA PREVIEW
// ============================================
/**
 * Display dataset overview and statistics
 */
function showOverview() {
    if (!mergedData || mergedData.length === 0) return;
    
    // Update statistics
    dom.totalPassengers.textContent = mergedData.length;
    dom.trainSamples.textContent = trainData.length;
    dom.testSamples.textContent = testData.length;
    dom.featureCount.textContent = DATA_SCHEMA.featureColumns.length;
    
    // Show overview content
    dom.overviewContent.classList.remove('hidden');
    dom.overviewPlaceholder.classList.add('hidden');
    
    // Show toggle button
    dom.togglePreviewBtn.classList.remove('hidden');
    
    // Show data preview
    showDataPreview();
}

/**
 * Display a preview of the data (first 10 rows)
 */
function showDataPreview() {
    if (!mergedData || mergedData.length === 0) return;
    
    const previewRows = mergedData.slice(0, 10);
    const columns = ['source', ...DATA_SCHEMA.featureColumns, DATA_SCHEMA.targetColumn].filter(col => 
        col in previewRows[0]
    );
    
    // Create table header
    let html = '<thead><tr>';
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead>';
    
    // Create table body
    html += '<tbody>';
    previewRows.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            // Handle different value types
            if (value === null || value === undefined) {
                html += '<td><em style="color: #999;">null</em></td>';
            } else if (typeof value === 'number') {
                html += `<td>${value.toFixed(2)}</td>`;
            } else {
                html += `<td>${value}</td>`;
            }
        });
        html += '</tr>';
    });
    html += '</tbody>';
    
    dom.dataPreview.innerHTML = html;
    dom.dataPreviewContainer.classList.remove('hidden');
}

/**
 * Toggle data preview visibility
 */
function toggleDataPreview() {
    dom.dataPreviewContainer.classList.toggle('hidden');
}

// ============================================
// MISSING VALUES ANALYSIS
// ============================================
/**
 * Analyze and display missing values
 */
function analyzeMissingValues() {
    if (!mergedData || mergedData.length === 0) return;
    
    // Calculate missing values per column
    const columnsToCheck = [...DATA_SCHEMA.featureColumns, DATA_SCHEMA.targetColumn];
    const missingStats = columnsToCheck.map(col => {
        const missingCount = mergedData.filter(row => 
            row[col] === null || row[col] === undefined || row[col] === '' || 
            (typeof row[col] === 'number' && isNaN(row[col]))
        ).length;
        const missingPercent = (missingCount / mergedData.length) * 100;
        
        return {
            column: col,
            missingCount,
            missingPercent: missingPercent.toFixed(1),
            dataType: DATA_SCHEMA.numericFeatures.includes(col) ? 'numeric' : 
                     DATA_SCHEMA.categoricalFeatures.includes(col) ? 'categorical' : 'other'
        };
    });
    
    // Show missing values content
    dom.missingValuesContent.classList.remove('hidden');
    dom.missingValuesPlaceholder.classList.add('hidden');
    
    // Create bar chart
    createMissingValuesChart(missingStats);
    
    // Create table
    createMissingValuesTable(missingStats);
}

/**
 * Create bar chart for missing values
 */
function createMissingValuesChart(missingStats) {
    // Destroy existing chart if it exists
    if (charts.missingValues) {
        charts.missingValues.destroy();
    }
    
    const ctx = dom.missingValuesChart.getContext('2d');
    charts.missingValues = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: missingStats.map(stat => stat.column),
            datasets: [{
                label: 'Missing Values (%)',
                data: missingStats.map(stat => parseFloat(stat.missingPercent)),
                backgroundColor: missingStats.map(stat => 
                    parseFloat(stat.missingPercent) > 20 ? 'rgba(231, 76, 60, 0.7)' :
                    parseFloat(stat.missingPercent) > 5 ? 'rgba(241, 196, 15, 0.7)' :
                    'rgba(46, 204, 113, 0.7)'
                ),
                borderColor: missingStats.map(stat => 
                    parseFloat(stat.missingPercent) > 20 ? 'rgb(231, 76, 60)' :
                    parseFloat(stat.missingPercent) > 5 ? 'rgb(241, 196, 15)' :
                    'rgb(46, 204, 113)'
                ),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Percentage Missing (%)'
                    }
                },
                x: {
                    ticks: {
                        autoSkip: false,
                        maxRotation: 45
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const stat = missingStats[context.dataIndex];
                            return `${stat.missingPercent}% (${stat.missingCount} of ${mergedData.length} records)`;
                        }
                    }
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Create table for missing values
 */
function createMissingValuesTable(missingStats) {
    let html = '<table class="data-table"><thead><tr>';
    html += '<th>Column</th><th>Type</th><th>Missing Count</th><th>Missing %</th><th>Status</th></tr></thead><tbody>';
    
    missingStats.forEach(stat => {
        const missingPercentNum = parseFloat(stat.missingPercent);
        let statusClass = '';
        let statusText = '';
        
        if (missingPercentNum === 0) {
            statusClass = 'success';
            statusText = 'Complete';
        } else if (missingPercentNum <= 5) {
            statusClass = 'success';
            statusText = 'Good';
        } else if (missingPercentNum <= 20) {
            statusClass = 'warning';
            statusText = 'Moderate';
        } else {
            statusClass = 'warning';
            statusText = 'High';
        }
        
        html += `<tr>
            <td><strong>${stat.column}</strong></td>
            <td>${stat.dataType}</td>
            <td>${stat.missingCount}</td>
            <td>${stat.missingPercent}%</td>
            <td><span class="${statusClass}">${statusText}</span></td>
        </tr>`;
    });
    
    html += '</tbody></table>';
    dom.missingValuesTableContainer.innerHTML = html;
}

// ============================================
// STATISTICAL SUMMARY
// ============================================
/**
 * Calculate and display statistical summaries
 */
function calculateStatistics() {
    if (!mergedData || mergedData.length === 0) return;
    
    // Show stats content
    dom.statsSummaryContent.classList.remove('hidden');
    dom.statsSummaryPlaceholder.classList.add('hidden');
    
    // Calculate numeric statistics
    calculateNumericStats();
    
    // Calculate categorical statistics
    calculateCategoricalStats();
    
    // Calculate survival statistics (train data only)
    calculateSurvivalStats();
}

/**
 * Calculate statistics for numeric features
 */
function calculateNumericStats() {
    const numericStats = {};
    
    DATA_SCHEMA.numericFeatures.forEach(feature => {
        const values = mergedData
            .map(row => row[feature])
            .filter(val => val !== null && val !== undefined && !isNaN(val) && typeof val === 'number');
        
        if (values.length === 0) {
            numericStats[feature] = { count: 0, message: 'No valid numeric values' };
            return;
        }
        
        // Sort for median calculation
        const sortedValues = [...values].sort((a, b) => a - b);
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const median = sortedValues[Math.floor(sortedValues.length / 2)];
        
        // Standard deviation
        const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
        const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
        const std = Math.sqrt(variance);
        
        // Min and max
        const min = Math.min(...values);
        const max = Math.max(...values);
        
        numericStats[feature] = {
            count: values.length,
            missing: mergedData.length - values.length,
            mean: mean.toFixed(2),
            median: median.toFixed(2),
            std: std.toFixed(2),
            min: min.toFixed(2),
            max: max.toFixed(2),
            range: (max - min).toFixed(2)
        };
    });
    
    // Display numeric stats
    let html = '<h3>Numeric Features Summary</h3><table class="data-table"><thead><tr>';
    html += '<th>Feature</th><th>Count</th><th>Missing</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Range</th></tr></thead><tbody>';
    
    Object.entries(numericStats).forEach(([feature, stats]) => {
        if (stats.message) {
            html += `<tr><td><strong>${feature}</strong></td><td colspan="8">${stats.message}</td></tr>`;
        } else {
            html += `<tr>
                <td><strong>${feature}</strong></td>
                <td>${stats.count}</td>
                <td>${stats.missing}</td>
                <td>${stats.mean}</td>
                <td>${stats.median}</td>
                <td>${stats.std}</td>
                <td>${stats.min}</td>
                <td>${stats.max}</td>
                <td>${stats.range}</td>
            </tr>`;
        }
    });
    
    html += '</tbody></table>';
    dom.numericStatsContainer.innerHTML = html;
    dom.numericStatsContainer.classList.remove('hidden');
}

/**
 * Calculate statistics for categorical features
 */
function calculateCategoricalStats() {
    const categoricalStats = {};
    
    DATA_SCHEMA.categoricalFeatures.forEach(feature => {
        const values = mergedData
            .map(row => row[feature])
            .filter(val => val !== null && val !== undefined && val !== '' && !isNaN(val));
        
        if (values.length === 0) {
            categoricalStats[feature] = { count: 0, message: 'No valid categorical values' };
            return;
        }
        
        // Count frequencies
        const frequency = {};
        values.forEach(val => {
            const key = String(val);
            frequency[key] = (frequency[key] || 0) + 1;
        });
        
        // Convert to array and sort by frequency
        const frequencyArray = Object.entries(frequency)
            .map(([value, count]) => ({ value, count, percent: (count / values.length * 100).toFixed(1) }))
            .sort((a, b) => b.count - a.count);
        
        categoricalStats[feature] = {
            count: values.length,
            missing: mergedData.length - values.length,
            unique: frequencyArray.length,
            frequencies: frequencyArray
        };
    });
    
    // Display categorical stats
    let html = '<h3>Categorical Features Summary</h3>';
    
    Object.entries(categoricalStats).forEach(([feature, stats]) => {
        if (stats.message) {
            html += `<div class="stat-box"><h4>${feature}</h4><p>${stats.message}</p></div>`;
        } else {
            html += `<div class="stat-box">
                <h4>${feature}</h4>
                <p><strong>Valid Values:</strong> ${stats.count} | <strong>Missing:</strong> ${stats.missing}</p>
                <p><strong>Unique Categories:</strong> ${stats.unique}</p>
                <table class="data-table" style="margin-top: 10px;">
                    <thead><tr><th>Value</th><th>Count</th><th>%</th></tr></thead>
                    <tbody>`;
            
            stats.frequencies.forEach(freq => {
                html += `<tr>
                    <td>${freq.value}</td>
                    <td>${freq.count}</td>
                    <td>${freq.percent}%</td>
                </tr>`;
            });
            
            html += '</tbody></table></div>';
        }
    });
    
    dom.categoricalStatsContainer.innerHTML = html;
    dom.categoricalStatsContainer.classList.remove('hidden');
}

/**
 * Calculate survival statistics from train data only
 */
function calculateSurvivalStats() {
    if (!trainData || trainData.length === 0) return;
    
    // Filter only train data with valid Survived values
    const validTrainData = trainData.filter(row => 
        row[DATA_SCHEMA.targetColumn] !== null && 
        row[DATA_SCHEMA.targetColumn] !== undefined
    );
    
    if (validTrainData.length === 0) {
        dom.survivalStatsContainer.innerHTML = '<p>No survival data available in train set.</p>';
        dom.survivalStatsContainer.classList.remove('hidden');
        return;
    }
    
    // Overall survival
    const total = validTrainData.length;
    const survived = validTrainData.filter(row => row[DATA_SCHEMA.targetColumn] === 1).length;
    const notSurvived = validTrainData.filter(row => row[DATA_SCHEMA.targetColumn] === 0).length;
    const survivalRate = (survived / total * 100).toFixed(1);
    
    // Survival by sex
    const maleSurvived = validTrainData.filter(row => 
        row.Sex === 'male' && row[DATA_SCHEMA.targetColumn] === 1
    ).length;
    const maleTotal = validTrainData.filter(row => row.Sex === 'male').length;
    const maleRate = maleTotal > 0 ? (maleSurvived / maleTotal * 100).toFixed(1) : 'N/A';
    
    const femaleSurvived = validTrainData.filter(row => 
        row.Sex === 'female' && row[DATA_SCHEMA.targetColumn] === 1
    ).length;
    const femaleTotal = validTrainData.filter(row => row.Sex === 'female').length;
    const femaleRate = femaleTotal > 0 ? (femaleSurvived / femaleTotal * 100).toFixed(1) : 'N/A';
    
    // Survival by class
    const survivalByClass = {};
    [1, 2, 3].forEach(pclass => {
        const classData = validTrainData.filter(row => row.Pclass === pclass);
        const classSurvived = classData.filter(row => row[DATA_SCHEMA.targetColumn] === 1).length;
        survivalByClass[pclass] = {
            total: classData.length,
            survived: classSurvived,
            rate: classData.length > 0 ? (classSurvived / classData.length * 100).toFixed(1) : 'N/A'
        };
    });
    
    // Display survival stats
    let html = `<h3>Survival Analysis (Train Data Only)</h3>
                <div class="stats-grid">
                    <div class="stat-box">
                        <h4>Overall Survival</h4>
                        <div class="stat-value">${survivalRate}%</div>
                        <p>${survived} survived / ${total} total</p>
                    </div>
                    <div class="stat-box">
                        <h4>Male Survival</h4>
                        <div class="stat-value">${maleRate}%</div>
                        <p>${maleSurvived} survived / ${maleTotal} total</p>
                    </div>
                    <div class="stat-box">
                        <h4>Female Survival</h4>
                        <div class="stat-value">${femaleRate}%</div>
                        <p>${femaleSurvived} survived / ${femaleTotal} total</p>
                    </div>
                </div>
                <h4 style="margin-top: 20px;">Survival by Passenger Class</h4>
                <table class="data-table">
                    <thead><tr><th>Class</th><th>Total</th><th>Survived</th><th>Survival Rate</th></tr></thead>
                    <tbody>`;
    
    [1, 2, 3].forEach(pclass => {
        const stats = survivalByClass[pclass];
        html += `<tr>
            <td>${pclass}</td>
            <td>${stats.total}</td>
            <td>${stats.survived}</td>
            <td>${stats.rate}%</td>
        </tr>`;
    });
    
    html += '</tbody></table>';
    
    dom.survivalStatsContainer.innerHTML = html;
    dom.survivalStatsContainer.classList.remove('hidden');
}

// ============================================
// VISUALIZATIONS
// ============================================
/**
 * Generate all visualizations
 */
function generateVisualizations() {
    if (!mergedData || mergedData.length === 0) return;
    
    // Show visualizations content
    dom.visualizationsContent.classList.remove('hidden');
    dom.visualizationsPlaceholder.classList.add('hidden');
    
    // Generate individual visualizations
    createSexChart();
    createPclassChart();
    createEmbarkedChart();
    createAgeHistogram();
    createFareHistogram();
    createCorrelationHeatmap();
}

/**
 * Create chart for Sex distribution
 */
function createSexChart() {
    if (charts.sexChart) charts.sexChart.destroy();
    
    const sexCounts = { male: 0, female: 0 };
    mergedData.forEach(row => {
        if (row.Sex === 'male') sexCounts.male++;
        else if (row.Sex === 'female') sexCounts.female++;
    });
    
    const ctx = dom.sexChart.getContext('2d');
    charts.sexChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Male', 'Female'],
            datasets: [{
                label: 'Passengers',
                data: [sexCounts.male, sexCounts.female],
                backgroundColor: ['rgba(54, 162, 235, 0.7)', 'rgba(255, 99, 132, 0.7)'],
                borderColor: ['rgb(54, 162, 235)', 'rgb(255, 99, 132)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Passenger Gender Distribution'
                }
            }
        }
    });
}

/**
 * Create chart for Pclass distribution
 */
function createPclassChart() {
    if (charts.pclassChart) charts.pclassChart.destroy();
    
    const pclassCounts = { 1: 0, 2: 0, 3: 0 };
    mergedData.forEach(row => {
        if ([1, 2, 3].includes(row.Pclass)) {
            pclassCounts[row.Pclass]++;
        }
    });
    
    const ctx = dom.pclassChart.getContext('2d');
    charts.pclassChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['1st Class', '2nd Class', '3rd Class'],
            datasets: [{
                label: 'Passengers',
                data: [pclassCounts[1], pclassCounts[2], pclassCounts[3]],
                backgroundColor: ['rgba(255, 159, 64, 0.7)', 'rgba(75, 192, 192, 0.7)', 'rgba(153, 102, 255, 0.7)'],
                borderColor: ['rgb(255, 159, 64)', 'rgb(75, 192, 192)', 'rgb(153, 102, 255)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Passenger Class Distribution'
                }
            }
        }
    });
}

/**
 * Create chart for Embarked distribution
 */
function createEmbarkedChart() {
    if (charts.embarkedChart) charts.embarkedChart.destroy();
    
    const embarkedCounts = { C: 0, Q: 0, S: 0, Unknown: 0 };
    mergedData.forEach(row => {
        if (row.Embarked === 'C') embarkedCounts.C++;
        else if (row.Embarked === 'Q') embarkedCounts.Q++;
        else if (row.Embarked === 'S') embarkedCounts.S++;
        else embarkedCounts.Unknown++;
    });
    
    const ctx = dom.embarkedChart.getContext('2d');
    charts.embarkedChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Cherbourg (C)', 'Queenstown (Q)', 'Southampton (S)', 'Unknown'],
            datasets: [{
                label: 'Passengers',
                data: [embarkedCounts.C, embarkedCounts.Q, embarkedCounts.S, embarkedCounts.Unknown],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 205, 86, 0.7)',
                    'rgba(201, 203, 207, 0.7)'
                ],
                borderWidth: 1
            }]
        },
        options
