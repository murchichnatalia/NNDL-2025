/**
 * Titanic EDA Dashboard - Interactive Data Analysis
 * 
 * Data Schema:
 * Target: Survived (0 = Died, 1 = Survived) - only in train.csv
 * Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
 * Identifier: PassengerId (excluded from analysis)
 * 
 * REUSE NOTE: To adapt for other datasets, update:
 * 1. The schema definition above
 * 2. Visualization configurations
 * 3. Feature names in calculations
 */

// Global variables to store data
let trainData = null;
let testData = null;
let mergedData = null;
let charts = {}; // Store chart instances for updates

// DOM elements
const trainFileInput = document.getElementById('trainFile');
const testFileInput = document.getElementById('testFile');
const loadDataBtn = document.getElementById('loadDataBtn');
const runEDABtn = document.getElementById('runEDABtn');
const exportCSVBtn = document.getElementById('exportCSVBtn');
const exportJSONBtn = document.getElementById('exportJSONBtn');

// Initialize the application
function init() {
    // Event listeners
    loadDataBtn.addEventListener('click', loadAndMergeData);
    runEDABtn.addEventListener('click', runFullEDA);
    exportCSVBtn.addEventListener('click', exportMergedCSV);
    exportJSONBtn.addEventListener('click', exportSummaryJSON);
    
    // Disable EDA button initially
    runEDABtn.disabled = true;
    runEDABtn.style.opacity = '0.6';
    
    console.log('Titanic EDA Dashboard initialized');
}

/**
 * Load and merge train and test datasets
 */
async function loadAndMergeData() {
    const trainFile = trainFileInput.files[0];
    const testFile = testFileInput.files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both train.csv and test.csv files');
        return;
    }
    
    try {
        // Parse train data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText, 'train');
        
        // Parse test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText, 'test');
        
        // Merge datasets
        mergedData = mergeDatasets(trainData, testData);
        
        // Update UI
        updateDataOverview();
        runEDABtn.disabled = false;
        runEDABtn.style.opacity = '1';
        
        console.log('Data loaded successfully:', {
            train: trainData.length,
            test: testData.length,
            merged: mergedData.length
        });
        
    } catch (error) {
        alert(`Error loading data: ${error.message}`);
        console.error('Load error:', error);
    }
}

/**
 * Parse CSV file using PapaParse
 */
function parseCSV(csvText, source) {
    return new Promise((resolve, reject) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            quoteChar: '"',
            complete: (results) => {
                // Add source column to identify dataset
                const data = results.data.map(row => ({
                    ...row,
                    Source: source
                }));
                resolve(data);
            },
            error: (error) => {
                reject(error);
            }
        });
    });
}

/**
 * Read file as text
 */
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(e);
        reader.readAsText(file);
    });
}

/**
 * Merge train and test datasets
 */
function mergeDatasets(train, test) {
    // For test data, add Survived as null (missing)
    const testWithSurvived = test.map(row => ({
        ...row,
        Survived: null
    }));
    
    return [...train, ...testWithSurvived];
}

/**
 * Update data overview section
 */
function updateDataOverview() {
    const overviewInfo = document.getElementById('overviewInfo');
    const table = document.getElementById('previewTable');
    const thead = table.querySelector('thead tr');
    const tbody = table.querySelector('tbody');
    
    // Clear previous content
    thead.innerHTML = '';
    tbody.innerHTML = '';
    
    // Update overview info
    overviewInfo.innerHTML = `
        <div class="stats-grid">
            <div class="stat-box">
                <h4>Train Data</h4>
                <p>${trainData.length} passengers with survival labels</p>
            </div>
            <div class="stat-box">
                <h4>Test Data</h4>
                <p>${testData.length} passengers without labels</p>
            </div>
            <div class="stat-box">
                <h4>Merged Data</h4>
                <p>${mergedData.length} total passengers</p>
            </div>
            <div class="stat-box">
                <h4>Features</h4>
                <p>${Object.keys(mergedData[0]).length} columns total</p>
            </div>
        </div>
    `;
    
    // Get column headers (exclude some if needed)
    const columns = Object.keys(mergedData[0]);
    
    // Create table headers
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        thead.appendChild(th);
    });
    
    // Add first 10 rows to preview
    const previewRows = mergedData.slice(0, 10);
    previewRows.forEach(row => {
        const tr = document.createElement('tr');
        columns.forEach(col => {
            const td = document.createElement('td');
            td.textContent = row[col] === null || row[col] === undefined ? 'N/A' : row[col];
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}

/**
 * Run full EDA analysis
 */
function runFullEDA() {
    if (!mergedData) {
        alert('Please load data first');
        return;
    }
    
    // Run all analyses
    analyzeMissingValues();
    calculateStatistics();
    createVisualizations();
    determineKeyFactor();
    
    console.log('Full EDA completed');
}

/**
 * Analyze missing values
 */
function analyzeMissingValues() {
    const missingInfo = document.getElementById('missingInfo');
    const columns = Object.keys(mergedData[0]);
    
    // Calculate missing percentages
    const missingStats = columns.map(col => {
        const total = mergedData.length;
        const missing = mergedData.filter(row => 
            row[col] === null || 
            row[col] === undefined || 
            row[col] === '' || 
            (typeof row[col] === 'number' && isNaN(row[col]))
        ).length;
        
        return {
            column: col,
            missing: missing,
            percentage: (missing / total * 100).toFixed(1)
        };
    });
    
    // Display missing values info
    missingInfo.innerHTML = `
        <h4>Missing Values Summary</h4>
        <p>Total rows: ${mergedData.length}</p>
        <div class="stats-grid">
            ${missingStats.map(stat => `
                <div class="stat-box">
                    <h4>${stat.column}</h4>
                    <p>Missing: ${stat.missing} (${stat.percentage}%)</p>
                </div>
            `).join('')}
        </div>
    `;
    
    // Create bar chart for missing values
    createMissingValuesChart(missingStats);
}

/**
 * Create missing values chart
 */
function createMissingValuesChart(missingStats) {
    const ctx = document.getElementById('missingValuesChart').getContext('2d');
    
    // Destroy existing chart if it exists
    if (charts.missingValues) {
        charts.missingValues.destroy();
    }
    
    const labels = missingStats.map(stat => stat.column);
    const data = missingStats.map(stat => parseFloat(stat.percentage));
    
    charts.missingValues = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Missing Values (%)',
                data: data,
                backgroundColor: data.map(pct => 
                    pct > 50 ? '#e74c3c' : 
                    pct > 20 ? '#f39c12' : 
                    pct > 5 ? '#f1c40f' : '#2ecc71'
                ),
                borderColor: '#34495e',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Percentage of Missing Values by Column'
                },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.parsed.y}% missing`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Percentage (%)'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45
                    }
                }
            }
        }
    });
}

/**
 * Calculate statistical summaries
 */
function calculateStatistics() {
    const statsContainer = document.getElementById('statsContainer');
    
    // Get only train data for survival analysis
    const trainOnly = mergedData.filter(row => row.Source === 'train');
    
    // Numeric columns to analyze
    const numericCols = ['Age', 'Fare', 'SibSp', 'Parch'];
    const categoricalCols = ['Pclass', 'Sex', 'Embarked'];
    
    // Calculate overall statistics
    const numericStats = numericCols.map(col => {
        const values = trainOnly.map(row => row[col]).filter(val => val !== null && !isNaN(val));
        return {
            column: col,
            mean: values.length ? (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2) : 'N/A',
            median: values.length ? calculateMedian(values) : 'N/A',
            std: values.length ? calculateStdDev(values).toFixed(2) : 'N/A',
            min: values.length ? Math.min(...values).toFixed(2) : 'N/A',
            max: values.length ? Math.max(...values).toFixed(2) : 'N/A'
        };
    });
    
    // Calculate survival rates by category
    const survivalBySex = calculateSurvivalRate(trainOnly, 'Sex');
    const survivalByClass = calculateSurvivalRate(trainOnly, 'Pclass');
    const survivalByEmbarked = calculateSurvivalRate(trainOnly, 'Embarked');
    
    // Display statistics
    statsContainer.innerHTML = `
        <div class="stat-box">
            <h4>Overall Survival</h4>
            <p>Total: ${trainOnly.length}</p>
            <p>Survived: ${trainOnly.filter(r => r.Survived === 1).length}</p>
            <p>Died: ${trainOnly.filter(r => r.Survived === 0).length}</p>
            <p>Survival Rate: ${(trainOnly.filter(r => r.Survived === 1).length / trainOnly.length * 100).toFixed(1)}%</p>
        </div>
        
        <div class="stat-box">
            <h4>Survival by Sex</h4>
            ${Object.entries(survivalBySex).map(([sex, rate]) => `
                <p>${sex}: ${(rate * 100).toFixed(1)}% survived</p>
            `).join('')}
        </div>
        
        <div class="stat-box">
            <h4>Survival by Class</h4>
            ${Object.entries(survivalByClass).map(([pclass, rate]) => `
                <p>Class ${pclass}: ${(rate * 100).toFixed(1)}% survived</p>
            `).join('')}
        </div>
        
        <div class="stat-box">
            <h4>Numeric Statistics</h4>
            ${numericStats.map(stat => `
                <p><strong>${stat.column}:</strong> 
                Mean=${stat.mean}, Median=${stat.median}, Std=${stat.std}</p>
            `).join('')}
        </div>
    `;
}

/**
 * Calculate median of an array
 */
function calculateMedian(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0 ? 
        sorted[mid].toFixed(2) : 
        ((sorted[mid - 1] + sorted[mid]) / 2).toFixed(2);
}

/**
 * Calculate standard deviation
 */
function calculateStdDev(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(avgSquaredDiff);
}

/**
 * Calculate survival rate by category
 */
function calculateSurvivalRate(data, column) {
    const groups = {};
    
    // Group data by column value
    data.forEach(row => {
        if (row[column] !== null && row.Survived !== null) {
            const key = row[column];
            if (!groups[key]) {
                groups[key] = { total: 0, survived: 0 };
            }
            groups[key].total++;
            if (row.Survived === 1) {
                groups[key].survived++;
            }
        }
    });
    
    // Calculate rates
    const rates = {};
    Object.entries(groups).forEach(([key, stats]) => {
        rates[key] = stats.survived / stats.total;
    });
    
    return rates;
}

/**
 * Create all visualizations
 */
function createVisualizations() {
    const trainOnly = mergedData.filter(row => row.Source === 'train');
    
    // Create individual charts
    createSexChart(trainOnly);
    createClassChart(trainOnly);
    createAgeChart(trainOnly);
    createFareChart(trainOnly);
    createEmbarkedChart(trainOnly);
    createCorrelationChart(trainOnly);
}

/**
 * Create sex survival chart
 */
function createSexChart(data) {
    const ctx = document.getElementById('sexChart').getContext('2d');
    const survivalRates = calculateSurvivalRate(data, 'Sex');
    
    if (charts.sexChart) charts.sexChart.destroy();
    
    charts.sexChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(survivalRates),
            datasets: [{
                label: 'Survival Rate',
                data: Object.values(survivalRates).map(rate => rate * 100),
                backgroundColor: ['#3498db', '#e74c3c'],
                borderColor: '#2c3e50',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Survival Rate by Gender'
                },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.parsed.y.toFixed(1)}% survived`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Survival Rate (%)'
                    }
                }
            }
        }
    });
}

/**
 * Create class survival chart
 */
function createClassChart(data) {
    const ctx = document.getElementById('classChart').getContext('2d');
    const survivalRates = calculateSurvivalRate(data, 'Pclass');
    
    // Sort by class
    const labels = Object.keys(survivalRates).sort();
    const rates = labels.map(label => survivalRates[label] * 100);
    
    if (charts.classChart) charts.classChart.destroy();
    
    charts.classChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(c => `Class ${c}`),
            datasets: [{
                label: 'Survival Rate',
                data: rates,
                backgroundColor: ['#2ecc71', '#3498db', '#e74c3c'],
                borderColor: '#2c3e50',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Survival Rate by Passenger Class'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Survival Rate (%)'
                    }
                }
            }
        }
    });
}

/**
 * Create age distribution chart
 */
function createAgeChart(data) {
    const ctx = document.getElementById('ageChart').getContext('2d');
    
    // Filter out null ages and separate by survival
    const survivedAges = data.filter(row => row.Survived === 1 && row.Age !== null).map(row => row.Age);
    const diedAges = data.filter(row => row.Survived === 0 && row.Age !== null).map(row => row.Age);
    
    if (charts.ageChart) charts.ageChart.destroy();
    
    charts.ageChart = new Chart(ctx, {
        type: 'histogram',
        data: {
            datasets: [
                {
                    label: 'Survived',
                    data: survivedAges,
                    backgroundColor: 'rgba(46, 204, 113, 0.5)',
                    borderColor: '#27ae60',
                    borderWidth: 1
                },
                {
                    label: 'Died',
                    data: diedAges,
                    backgroundColor: 'rgba(231, 76, 60, 0.5)',
                    borderColor: '#c0392b',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Age Distribution by Survival Status'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Age'
                    },
                    min: 0,
                    max: 80
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * Create fare distribution chart
 */
function createFareChart(data) {
    const ctx = document.getElementById('fareChart').getContext('2d');
    
    const survivedFares = data.filter(row => row.Survived === 1 && row.Fare !== null).map(row => row.Fare);
    const diedFares = data.filter(row => row.Survived === 0 && row.Fare !== null).map(row => row.Fare);
    
    if (charts.fareChart) charts.fareChart.destroy();
    
    charts.fareChart = new Chart(ctx, {
        type: 'histogram',
        data: {
            datasets: [
                {
                    label: 'Survived',
                    data: survivedFares,
                    backgroundColor: 'rgba(46, 204, 113, 0.5)',
                    borderColor: '#27ae60',
                    borderWidth: 1
                },
                {
                    label: 'Died',
                    data: diedFares,
                    backgroundColor: 'rgba(231, 76, 60, 0.5)',
                    borderColor: '#c0392b',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Fare Distribution by Survival Status'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Fare'
                    },
                    min: 0,
                    max: 100
                },
                y: {
                    title: {
                        display: true,
                        text: 'Frequency'
                    },
                    beginAtZero: true
                }
            }
        }
    });
}

/**
 * Create embarked port chart
 */
function createEmbarkedChart(data) {
    const ctx = document.getElementById('embarkedChart').getContext('2d');
    const survivalRates = calculateSurvivalRate(data, 'Embarked');
    
    if (charts.embarkedChart) charts.embarkedChart.destroy();
    
    charts.embarkedChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(survivalRates).map(port => 
                port === 'C' ? 'Cherbourg' : 
                port === 'Q' ? 'Queenstown' : 
                port === 'S' ? 'Southampton' : port
            ),
            datasets: [{
                label: 'Survival Rate',
                data: Object.values(survivalRates).map(rate => rate * 100),
                backgroundColor: ['#9b59b6', '#3498db', '#2ecc71'],
                borderColor: '#2c3e50',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Survival Rate by Embarkation Port'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Survival Rate (%)'
                    }
                }
            }
        }
    });
}

/**
 * Create correlation heatmap
 */
function createCorrelationChart(data) {
    const ctx = document.getElementById('correlationChart').getContext('2d');
    
    // Select numeric features for correlation
    const numericFeatures = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass'];
    const survivalData = data.filter(row => 
        row.Survived !== null && 
        numericFeatures.every(f => row[f] !== null && !isNaN(row[f]))
    );
    
    // Calculate correlation matrix
    const correlations = calculateCorrelationMatrix(survivalData, numericFeatures);
    
    if (charts.correlationChart) charts.correlationChart.destroy();
