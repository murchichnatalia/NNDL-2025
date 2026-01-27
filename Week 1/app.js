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
        const file = e.target.files[0];
        if (file) {
            dom.trainStatus.textContent = `✅ Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            dom.trainStatus.className = 'file-status loaded';
        }
    });
    
    dom.testFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            dom.testStatus.textContent = `✅ Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            dom.testStatus.className = 'file-status loaded';
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
    const trainFile = dom.trainFileInput.files[0];
    const testFile = dom.testFileInput.files[0];
    
    // Validate files are selected
    if (!trainFile || !testFile) {
        showMessage(dom.dataLoadMessage, 'Please select both train.csv and test.csv files.', 'warning');
        return;
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
                    if (results.errors.length > 0) {
                        reject(new Error(`Train CSV parse errors: ${results.errors.map(e => e.message).join(', ')}`));
                    } else {
                        // Add source column
                        const dataWithSource = results.data.map(row => ({
                            ...row,
                            source: 'train'
                        }));
                        resolve(dataWithSource);
                    }
                },
                error: (error) => reject(error)
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
                    if (results.errors.length > 0) {
                        reject(new Error(`Test CSV parse errors: ${results.errors.map(e => e.message).join(', ')}`));
                    } else {
                        // Add source column and missing Survived column
                        const dataWithSource = results.data.map(row => ({
                            ...row,
                            source: 'test',
                            Survived: null // Add Survived column with null for test data
                        }));
                        resolve(dataWithSource);
                    }
                },
                error: (error) => reject(error)
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
            throw new Error(`Missing required columns: ${missingColumns.join(', ')}`);
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
            html += `<td>${value === null || value === undefined ? '<em>null</em>' : value}</td>`;
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
            row[col] === null || row[col] === undefined || row[col] === ''
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
                    stat.missingPercent > 20 ? 'rgba(231, 76, 60, 0.7)' :
                    stat.missingPercent > 5 ? 'rgba(241, 196, 15, 0.7)' :
                    'rgba(46, 204, 113, 0.7)'
                ),
                borderColor: missingStats.map(stat => 
                    stat.missingPercent > 20 ? 'rgb(231, 76, 60)' :
                    stat.missingPercent > 5 ? 'rgb(241, 196, 15)' :
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
        let statusClass = 'success';
        let statusText = 'Good';
        
        if (stat.missingPercent > 20) {
            statusClass = 'warning';
            statusText = 'High';
        } else if (stat.missingPercent > 5) {
            statusClass = 'warning';
            statusText = 'Moderate';
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
            .filter(val => val !== null && val !== undefined && !isNaN(val));
        
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
            .filter(val => val !== null && val !== undefined && val !== '');
        
        if (values.length === 0) {
            categoricalStats[feature] = { count: 0, message: 'No valid categorical values' };
            return;
        }
        
        // Count frequencies
        const frequency = {};
        values.forEach(val => {
            frequency[val] = (frequency[val] || 0) + 1;
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
            html += `<div class="stat-box"><h4>${feature}</h
