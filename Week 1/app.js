let mergedData = [];
let summaryResult = {};
let charts = {}; // store chart instances to safely destroy before re-creating

// ===== Dataset Schema =====
// To reuse for another dataset:
// 1. Change these feature lists
// 2. Keep merge logic with `source` column

const targetColumn = "Survived";
const numericFeatures = ["Age", "Fare", "SibSp", "Parch"];
const categoricalFeatures = ["Sex", "Pclass", "Embarked"];
const identifierColumn = "PassengerId";

// ================= Utilities =================

function mean(arr) {
    if (!arr.length) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(arr) {
    if (!arr.length) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2
        ? sorted[mid]
        : (sorted[mid - 1] + sorted[mid]) / 2;
}

function std(arr) {
    if (!arr.length) return 0;
    const m = mean(arr);
    return Math.sqrt(mean(arr.map(x => (x - m) ** 2)));
}

function destroyChart(id) {
    if (charts[id]) {
        charts[id].destroy();
    }
}

// ================= Load + Merge =================

document.getElementById("loadBtn").addEventListener("click", () => {

    const trainFile = document.getElementById("trainFile").files[0];
    const testFile = document.getElementById("testFile").files[0];

    if (!trainFile || !testFile) {
        alert("Upload BOTH train.csv and test.csv from Kaggle.");
        return;
    }

    parseFile(trainFile, "train")
        .then(trainData => parseFile(testFile, "test")
            .then(testData => {

                mergedData = [...trainData, ...testData];

                if (!mergedData.length) {
                    alert("Parsed data is empty.");
                    return;
                }

                renderOverview();
                analyzeMissing();
                runEDA();
            })
        )
        .catch(err => {
            console.error(err);
            alert("Error loading CSV files.");
        });
});

function parseFile(file, sourceName) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: function (results) {

                if (!results.data || !results.data.length) {
                    reject("Empty file");
                    return;
                }

                const cleaned = results.data
                    .filter(row => Object.keys(row).length > 1)
                    .map(row => ({ ...row, source: sourceName }));

                resolve(cleaned);
            },
            error: function (err) {
                reject(err);
            }
        });
    });
}

// ================= Overview =================

function renderOverview() {

    const overviewDiv = document.getElementById("overview");
    const previewDiv = document.getElementById("preview");

    overviewDiv.innerHTML =
        `<strong>Rows:</strong> ${mergedData.length} |
         <strong>Columns:</strong> ${Object.keys(mergedData[0]).length}`;

    const previewRows = mergedData.slice(0, 5);

    let table = "<table><tr>";
    Object.keys(previewRows[0]).forEach(col => {
        table += `<th>${col}</th>`;
    });
    table += "</tr>";

    previewRows.forEach(row => {
        table += "<tr>";
        Object.keys(previewRows[0]).forEach(col => {
            table += `<td>${row[col]}</td>`;
        });
        table += "</tr>";
    });

    table += "</table>";

    previewDiv.innerHTML = table;
}

// ================= Missing Values =================

function analyzeMissing() {

    const columns = Object.keys(mergedData[0]);

    const missingPercent = columns.map(col => {
        const missing = mergedData.filter(r =>
            r[col] === null ||
            r[col] === undefined ||
            r[col] === ""
        ).length;

        return (missing / mergedData.length * 100).toFixed(2);
    });

    destroyChart("missingChart");

    charts["missingChart"] = new Chart(
        document.getElementById("missingChart"),
        {
            type: "bar",
            data: {
                labels: columns,
                datasets: [{
                    label: "% Missing",
                    data: missingPercent
                }]
            }
        }
    );
}

// ================= EDA =================

function runEDA() {

    const trainOnly = mergedData.filter(d => d.source === "train");

    if (!trainOnly.length) {
        alert("Train data missing Survived column.");
        return;
    }

    renderNumericStats(trainOnly);

    visualizeSurvivalByCategory(trainOnly, "Sex", "sexChart");
    visualizeSurvivalByCategory(trainOnly, "Pclass", "pclassChart");
    visualizeHistogram(trainOnly, "Age", "ageChart");
    visualizeHistogram(trainOnly, "Fare", "fareChart");

    /*
    ===== EDA CONCLUSION =====
    Based on Titanic dataset analysis:

    1. SEX is the strongest survival factor.
       Females survival rate ~70–75%
       Males survival rate ~15–20%

    2. Pclass is second strongest factor.
       1st class much higher survival.

    3. Age and Fare moderate influence.

    ==> MAIN FACTOR OF DEATH: BEING MALE.
    */
}

// ================= Stats =================

function renderNumericStats(data) {

    let html = "<h3>Numeric Summary (Train Only)</h3>";
    html += "<table><tr><th>Feature</th><th>Mean</th><th>Median</th><th>Std</th></tr>";

    summaryResult.numeric = {};

    numericFeatures.forEach(feature => {

        const values = data
            .map(d => d[feature])
            .filter(v => typeof v === "number" && !isNaN(v));

        if (!values.length) return;

        const m = mean(values);
        const med = median(values);
        const s = std(values);

        summaryResult.numeric[feature] = { mean: m, median: med, std: s };

        html += `<tr>
            <td>${feature}</td>
            <td>${m.toFixed(2)}</td>
            <td>${med.toFixed(2)}</td>
            <td>${s.toFixed(2)}</td>
        </tr>`;
    });

    html += "</table>";

    document.getElementById("stats").innerHTML = html;
}

// ================= Category Survival =================

function visualizeSurvivalByCategory(data, feature, canvasId) {

    destroyChart(canvasId);

    const groups = {};

    data.forEach(row => {

        const key = row[feature];
        if (!groups[key]) {
            groups[key] = { survived: 0, total: 0 };
        }

        groups[key].total++;
        if (row[targetColumn] === 1) {
            groups[key].survived++;
        }
    });

    const labels = Object.keys(groups);
    const survivalRates = labels.map(l =>
        (groups[l].survived / groups[l].total * 100).toFixed(2)
    );

    charts[canvasId] = new Chart(
        document.getElementById(canvasId),
        {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: `Survival Rate (%) by ${feature}`,
                    data: survivalRates
                }]
            }
        }
    );
}

// ================= Histogram =================

function visualizeHistogram(data, feature, canvasId) {

    destroyChart(canvasId);

    const values = data
        .map(d => d[feature])
        .filter(v => typeof v === "number" && !isNaN(v));

    if (!values.length) return;

    const bins = 10;
    const min = Math.min(...values);
    const max = Math.max(...values);

    if (max === min) return;

    const step = (max - min) / bins;
    const counts = new Array(bins).fill(0);

    values.forEach(v => {
        const index = Math.min(
            Math.floor((v - min) / step),
            bins - 1
        );
        counts[index]++;
    });

    const labels = counts.map((_, i) =>
        (min + i * step).toFixed(1)
    );

    charts[canvasId] = new Chart(
        document.getElementById(canvasId),
        {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: `${feature} Distribution`,
                    data: counts
                }]
            }
        }
    );
}

// ================= Export =================

document.getElementById("exportCsvBtn").addEventListener("click", () => {

    if (!mergedData.length) {
        alert("Load data first.");
        return;
    }

    const csv = Papa.unparse(mergedData);
    downloadFile(csv, "merged_titanic.csv", "text/csv");
});

document.getElementById("exportJsonBtn").addEventListener("click", () => {

    if (!summaryResult.numeric) {
        alert("Run EDA first.");
        return;
    }

    const json = JSON.stringify(summaryResult, null, 2);
    downloadFile(json, "eda_summary.json", "application/json");
});

function downloadFile(content, filename, type) {

    const blob = new Blob([content], { type });
    const link = document.createElement("a");

    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
}
