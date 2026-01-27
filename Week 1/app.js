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

// Data URLs - UPDATE THESE TO YOUR GITHUB RAW LINKS
const DATA_URLS = {
    train: 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/train.csv',
    test: 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/test.csv'
};

// Initialize the application
function init() {
    // Event listeners
    loadDataBtn.addEventListener('click', loadAndMergeData);
    loadDataBtn.addEventListener('click', loadDataFromGitHub); // New option
    runEDABtn.addEventListener('click', runFullEDA);
    exportCSVBtn.addEventListener('click', exportMergedCSV);
    exportJSONBtn.addEventListener('click', exportSummaryJSON);
    
    // Add auto-load buttons
    addAutoLoadButtons();
    
    // Disable EDA button initially
    runEDABtn.disabled = true;
    runEDABtn.style.opacity = '0.6';
    
    console.log('Titanic EDA Dashboard initialized');
}

/**
 * Add buttons for auto-loading from GitHub
 */
function addAutoLoadButtons() {
    const loadSection = document.querySelector('.button-group');
    const autoLoadHTML = `
        <button class="btn-primary" id="loadFromGitHubBtn">
            🌐 Load from GitHub (Auto)
        </button>
        <button class="btn-secondary" id="loadSampleBtn">
            🧪 Load Sample Data
        </button>
    `;
    
    loadSection.insertAdjacentHTML('afterend', autoLoadHTML);
    
    // Add event listeners for new buttons
    document.getElementById('loadFromGitHubBtn').addEventListener('click', loadDataFromGitHub);
    document.getElementById('loadSampleBtn').addEventListener('click', loadSampleData);
}

/**
 * Load data directly from GitHub URLs
 */
async function loadDataFromGitHub() {
    try {
        showLoading('Loading data from GitHub...');
        
        // Load train data
        const trainResponse = await fetch(DATA_URLS.train);
        if (!trainResponse.ok) throw new Error(`Failed to load train.csv: ${trainResponse.status}`);
        const trainText = await trainResponse.text();
        trainData = await parseCSV(trainText, 'train');
        
        // Load test data
        const testResponse = await fetch(DATA_URLS.test);
        if (!testResponse.ok) throw new Error(`Failed to load test.csv: ${testResponse.status}`);
        const testText = await testResponse.text();
        testData = await parseCSV(testText, 'test');
        
        // Merge datasets
        mergedData = mergeDatasets(trainData, testData);
        
        // Update UI
        updateDataOverview();
        runEDABtn.disabled = false;
        runEDABtn.style.opacity = '1';
        
        hideLoading();
        showSuccess('Data loaded successfully from GitHub!');
        
        console.log('Data loaded from GitHub:', {
            train: trainData.length,
            test: testData.length,
            merged: mergedData.length
        });
        
    } catch (error) {
        hideLoading();
        showError(`Error loading from GitHub: ${error.message}`);
        console.error('GitHub load error:', error);
        
        // Fallback to sample data
        alert('GitHub load failed. Loading sample data instead...');
        loadSampleData();
    }
}

/**
 * Load small sample data for testing
 */
async function loadSampleData() {
    try {
        showLoading('Loading sample data...');
        
        // Create sample data (first 50 rows from each)
        const sampleTrain = `PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath (Lily May Peel)",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S`;
        
        const sampleTest = `PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
892,3,"Kelly, Mr. James",male,34.5,0,0,330911,7.8292,,Q
893,3,"Wilkes, Mrs. James (Ellen Needs)",female,47,1,0,363272,7,,S
894,2,"Myles, Mr. Thomas Francis",male,62,0,0,240276,9.6875,,Q
895,3,"Wirz, Mr. Albert",male,27,0,0,315154,8.6625,,S
896,3,"Hirvonen, Mrs. Alexander (Helga E Lindqvist)",female,22,1,1,3101298,12.2875,,S`;
        
        trainData = await parseCSV(sampleTrain, 'train');
        testData = await parseCSV(sampleTest, 'test');
        mergedData = mergeDatasets(trainData, testData);
        
        updateDataOverview();
        runEDABtn.disabled = false;
        runEDABtn.style.opacity = '1';
        
        hideLoading();
        showSuccess('Sample data loaded! Click "Run Full EDA Analysis"');
        
    } catch (error) {
        hideLoading();
        showError(`Error loading sample: ${error.message}`);
    }
}

/**
 * Load and merge train and test datasets from file inputs
 */
async function loadAndMergeData() {
    const trainFile = trainFileInput.files[0];
    const testFile = testFileInput.files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both train.csv and test.csv files');
        return;
    }
    
    try {
        showLoading('Loading and merging data...');
        
        // Parse train data
        const trainText = await readFile(trainFile);
        trainData = await parseCSV(trainText, 'train');
        
        // Parse test data
        const testText = await readFile(testFile);
        testData = await parseCSV(testText, 'test');
        
        // Merge datasets
        mergedData = mergeDatasets(trainData, testData);
        
        // Update UI
        updateDataOverview();
        runEDABtn.disabled = false;
        runEDABtn.style.opacity = '1';
        
        hideLoading();
        showSuccess('Data loaded successfully! Click "Run Full EDA Analysis"');
        
        console.log('Data loaded successfully:', {
            train: trainData.length,
            test: testData.length,
            merged: mergedData.length
        });
        
    } catch (error) {
        hideLoading();
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
 * Show loading indicator
 */
function showLoading(message) {
    let loadingDiv = document.getElementById('loadingIndicator');
    if (!loadingDiv) {
        loadingDiv = document.createElement('div');
        loadingDiv.id = 'loadingIndicator';
        loadingDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.2);
            z-index: 1000;
            text-align: center;
        `;
        document.body.appendChild(loadingDiv);
    }
    loadingDiv.innerHTML = `
        <div style="font-size: 18px; margin-bottom: 10px;">⏳ ${message}</div>
        <div class="spinner"></div>
        <style>
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    `;
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    const loadingDiv = document.getElementById('loadingIndicator');
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

/**
 * Show success message
 */
function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #2ecc71;
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    successDiv.innerHTML = `✅ ${message}`;
    document.body.appendChild(successDiv);
    
    setTimeout(() => {
        successDiv.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => successDiv.remove(), 300);
    }, 3000);
    
    // Add animation styles
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
}

/**
 * Show error message
 */
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #e74c3c;
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    errorDiv.innerHTML = `❌ ${message}`;
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => errorDiv.remove(), 300);
    }, 5000);
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
    
    // Get column headers
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
    
    showLoading('Running EDA analysis...');
    
    // Run all analyses with slight delay for UI update
    setTimeout(() => {
        try {
            analyzeMissingValues();
            calculateStatistics();
            createVisualizations();
            determineKeyFactor();
            
            hideLoading();
            showSuccess('EDA analysis completed!');
            
            console.log('Full EDA completed');
        } catch (error) {
            hideLoading();
            showError(`Analysis error: ${error.message}`);
            console.error('EDA error:', error);
        }
    }, 100);
}

// ... [ОСТАЛЬНАЯ ЧАСТЬ КОДА БЕЗ ИЗМЕНЕНИЙ - функции analyzeMissingValues, calculateStatistics, createVisualizations и т.д.] ...

/**
 * Determine the most important survival factor
 */
function determineKeyFactor() {
    const trainOnly = mergedData.filter(row => row.Source === 'train');
    const keyFactorElement = document.getElementById('keyFactor');
    const explanationElement = document.getElementById('factorExplanation');
    const detailsElement = document.getElementById('factorDetails');
    
    // Calculate survival rates by different factors
    const sexRate = calculateSurvivalRate(trainOnly, 'Sex');
    const classRate = calculateSurvivalRate(trainOnly, 'Pclass');
    
    // Calculate absolute difference in survival rates
    const sexDiff = sexRate.female && sexRate.male ? 
        Math.abs(sexRate.female - sexRate.male) : 0;
    
    // For class, calculate variance
    const classValues = Object.values(classRate);
    const classDiff = classValues.length ? 
        Math.max(...classValues) - Math.min(...classValues) : 0;
    
    // Determine key factor
    let keyFactor, explanation, details;
    
    if (sexDiff >= classDiff) {
        keyFactor = "GENDER (SEX)";
        explanation = "Being female was the strongest predictor of survival on the Titanic.";
        details = `
            <ul style="margin-top: 10px;">
                <li><strong>Female survival rate:</strong> ${(sexRate.female * 100).toFixed(1)}%</li>
                <li><strong>Male survival rate:</strong> ${(sexRate.male * 100).toFixed(1)}%</li>
                <li><strong>Difference:</strong> ${(sexDiff * 100).toFixed(1)} percentage points</li>
                <li>Women were ${(sexRate.female / sexRate.male).toFixed(1)}x more likely to survive</li>
            </ul>
        `;
    } else {
        keyFactor = "PASSENGER CLASS (PCLASS)";
        explanation = "First-class passengers had significantly higher survival rates.";
        details = `
            <ul style="margin-top: 10px;">
                ${Object.entries(classRate).map(([cls, rate]) => `
                    <li><strong>Class ${cls} survival rate:</strong> ${(rate * 100).toFixed(1)}%</li>
                `).join('')}
                <li><strong>Class gap:</strong> ${(classDiff * 100).toFixed(1)} percentage points</li>
            </ul>
        `;
    }
    
    // Update DOM
    keyFactorElement.textContent = keyFactor;
    explanationElement.textContent = explanation;
    detailsElement.innerHTML = details;
}

/**
 * Export merged data as CSV
 */
function exportMergedCSV() {
    if (!mergedData || mergedData.length === 0) {
        alert('No data to export');
        return;
    }
    
    try {
        const csv = Papa.unparse(mergedData);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', 'titanic_merged_data.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showSuccess('CSV exported successfully!');
    } catch (error) {
        showError(`Export failed: ${error.message}`);
    }
}

/**
 * Export summary as JSON
 */
function exportSummaryJSON() {
    if (!mergedData || mergedData.length === 0) {
        alert('No data to export');
        return;
    }
    
    try {
        const trainOnly = mergedData.filter(row => row.Source === 'train');
        const summary = {
            datasetInfo: {
                totalPassengers: mergedData.length,
                trainPassengers: trainData.length,
                testPassengers: testData.length,
                features: Object.keys(mergedData[0]).length
            },
            survivalStats: {
                totalSurvived: trainOnly.filter(r => r.Survived === 1).length,
                totalDied: trainOnly.filter(r => r.Survived === 0).length,
                survivalRate: (trainOnly.filter(r => r.Survived === 1).length / trainOnly.length * 100).toFixed(1) + '%'
            },
            keyFactors: {
                sex: calculateSurvivalRate(trainOnly, 'Sex'),
                pclass: calculateSurvivalRate(trainOnly, 'Pclass'),
                embarked: calculateSurvivalRate(trainOnly, 'Embarked')
            },
            generatedAt: new Date().toISOString()
        };
        
        const jsonStr = JSON.stringify(summary, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        
        link.setAttribute('href', url);
        link.setAttribute('download', 'titanic_analysis_summary.json');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showSuccess('JSON summary exported!');
    } catch (error) {
        showError(`Export failed: ${error.message}`);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', init);
