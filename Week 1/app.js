// Titanic EDA App - All processing client-side
// To reuse for another split dataset:
// 1. Swap schema columns below
// 2. Adjust numeric/categorical feature lists
// 3. Keep merge logic (add 'source' column)

let mergedData = [];
let summaryResult = {};

const targetColumn = "Survived";
const numericFeatures = ["Age", "Fare", "SibSp", "Parch"];
const categoricalFeatures = ["Sex", "Pclass", "Embarked"];
const identifierColumn = "PassengerId";

// Utility: mean
function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// Utility: median
function median(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 !== 0
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Utility: std deviation
function std(arr) {
    const m = mean(arr);
    return Math.sqrt(mean(arr.map(x => (x - m) ** 2)));
}

// Load and merge
document.getElementById("loadBtn").addEventListener("click", () => {
    const trainFile = document.getElementById("trainFile").files[0];
    const testFile = document.getElementById("testFile").files[0];

    if (!trainFile || !testFile) {
        alert("Please upload both train.csv and test.csv");
        return;
    }

    Papa.parse(trainFile, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function (trainResults) {

            Papa.parse(testFile, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function (testResults) {

                    const trainData = trainResults.data.map(d => ({ ...d, source: "train" }));
                    const testData = testResults.data.map(d => ({ ...d, source: "test" }));

                    mergedData = [...trainData, ...testData];

                    renderOverview();
                    analyzeMissing();
                    runEDA();
                },
                error: function () {
                    alert("Error parsing test.csv");
                }
            });

        },
        error: function () {
            alert("Error parsing train.csv");
        }
    });
});

// Overview
function renderOverview() {
    document.getElementById("overview").innerHTML =
        `<strong>Rows:</strong> ${mergedData.length} |
         <strong>Columns:</strong> ${Object.keys(mergedData[0]).length}`;

    const previewRows = mergedData.slice(0, 5);
    let table = "<table><tr>";
    Object.keys(previewRows[0]).forEach(col => table += `<th>${col}</th>`);
    table += "</tr>";

    previewRows.forEach(row => {
        table += "<tr>";
        Object.values(row).forEach(val => table += `<td>${val}</td>`);
        table += "</tr>";
    });
    table += "</table>";

    document.getElementById("preview").innerHTML = table;
}

// Missing values
function analyzeMissing() {
    const columns = Object.keys(mergedData[0]);
    const missingPercent = columns.map(col => {
        const missing = mergedData.filter(r => !r[col] && r[col] !== 0).length;
        return (missing / mergedData.length) * 100;
    });

    new Chart(document.getElementById("missingChart"), {
        type: "bar",
        data: {
            labels: columns,
            datasets: [{
                label: "% Missing",
                data: missingPercent
            }]
        }
    });
}

// EDA
function runEDA() {

    const trainOnly = mergedData.filter(d => d.source === "train");

    // Numeric stats
    let statsHTML = "<h3>Numeric Summary</h3><table><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std</th></tr>";

    summaryResult.numeric = {};

    numericFeatures.forEach(feature => {
        const values = trainOnly.map(d => d[feature]).filter(v => typeof v === "number");

        if (values.length > 0) {
            const m = mean(values);
            const med = median(values);
            const s = std(values);

            summaryResult.numeric[feature] = { mean: m, median: med, std: s };

            statsHTML += `<tr>
                <td>${feature}</td>
                <td>${m.toFixed(2)}</td>
                <td>${med.toFixed(2)}</td>
                <td>${s.toFixed(2)}</td>
            </tr>`;
        }
    });

    statsHTML += "</table>";

    document.getElementById("stats").innerHTML = statsHTML;

    visualizeSurvivalByCategory(trainOnly, "Sex", "sexChart");
    visualizeSurvivalByCategory(trainOnly, "Pclass", "pclassChart");
    visualizeHistogram(trainOnly, "Age", "ageChart");
    visualizeHistogram(trainOnly, "Fare", "fareChart");

    // EDA Conclusion (based on repeated Kaggle EDA results):
    // 1. Sex is the strongest factor: females survived at much higher rates.
    // 2. Pclass strongly influences survival (1st class higher survival).
    // 3. Age has moderate influence (children higher survival).
    // ==> Main factor of death/survival: SEX.
}

// Survival by categorical
function visualizeSurvivalByCategory(data, feature, canvasId) {

    const groups = {};
    data.forEach(row => {
        const key = row[feature];
        if (!groups[key]) groups[key] = { survived: 0, total: 0 };
        groups[key].total++;
        if (row[targetColumn] === 1) groups[key].survived++;
    });

    const labels = Object.keys(groups);
    const survivalRates = labels.map(l => (groups[l].survived / groups[l].total) * 100);

    new Chart(document.getElementById(canvasId), {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: `Survival Rate (%) by ${feature}`,
                data: survivalRates
            }]
        }
    });
}

// Histogram (simple binning)
function visualizeHistogram(data, feature, canvasId) {

    const values = data.map(d => d[feature]).filter(v => typeof v === "number");
    const bins = 10;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const step = (max - min) / bins;

    const counts = new Array(bins).fill(0);

    values.forEach(v => {
        const index = Math.min(Math.floor((v - min) / step), bins - 1);
        counts[index]++;
    });

    const labels = counts.map((_, i) => `${(min + i * step).toFixed(1)}`);

    new Chart(document.getElementById(canvasId), {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: `${feature} Distribution`,
                data: counts
            }]
        }
    });
}

// Export merged CSV
document.getElementById("exportCsvBtn").addEventListener("click", () => {
    if (mergedData.length === 0) {
        alert("No data loaded.");
        return;
    }

    const csv = Papa.unparse(mergedData);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "merged_titanic.csv";
    link.click();
});

// Export JSON summary
document.getElementById("exportJsonBtn").addEventListener("click", () => {
    if (!summaryResult.numeric) {
        alert("Run EDA first.");
        return;
    }

    const blob = new Blob([JSON.stringify(summaryResult, null, 2)], {
        type: "application/json"
    });

    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "eda_summary.json";
    link.click();
});
