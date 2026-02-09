// 1. Глобальные переменные
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

// Конфигурация схемы
const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';

// 2. ЗАГРУЗКА ДАННЫХ
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Пожалуйста, выберите оба файла: train.csv и test.csv');
        return;
    }
    
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Загрузка данных...';
    
    try {
        // Чтение тренировочных данных
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Чтение тестовых данных
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        statusDiv.innerHTML = `Данные успешно загружены! Train: ${trainData.length}, Test: ${testData.length}`;
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Ошибка: ${error.message}`;
        console.error(error);
    }
}

// Вспомогательная функция чтения (которой не хватало)
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Не удалось прочитать файл'));
        reader.readAsText(file);
    });
}

// Исправленный парсер CSV (игнорирует запятые внутри кавычек в именах)
function parseCSV(csvText) {
    const lines = csvText.split(/\r?\n/).filter(line => line.trim() !== '');
    
    const splitCsvLine = (line) => {
        const result = [];
        let start = 0;
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {
            if (line[i] === '"') inQuotes = !inQuotes;
            else if (line[i] === ',' && !inQuotes) {
                result.push(line.substring(start, i));
                start = i + 1;
            }
        }
        result.push(line.substring(start));
        return result.map(val => {
            val = val.trim();
            if (val.startsWith('"') && val.endsWith('"')) {
                val = val.substring(1, val.length - 1).replace(/""/g, '"');
            }
            return val;
        });
    };

    const headers = splitCsvLine(lines[0]);
    return lines.slice(1).map(line => {
        const values = splitCsvLine(line);
        const obj = {};
        headers.forEach((header, i) => {
            let val = (values[i] === '' || values[i] === undefined) ? null : values[i];
            if (val !== null && !isNaN(val) && val.trim() !== '') {
                obj[header] = parseFloat(val);
            } else {
                obj[header] = val;
            }
        });
        return obj;
    });
}

// 3. ИНСПЕКЦИЯ ДАННЫХ
function inspectData() {
    if (!trainData) return;
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Превью (первые 10 строк)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    const statsDiv = document.getElementById('data-stats');
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    statsDiv.innerHTML = `
        <p>Строк: ${trainData.length}, Столбцов: ${Object.keys(trainData[0]).length}</p>
        <p>Выживаемость: ${survivalCount} (${(survivalCount/trainData.length*100).toFixed(2)}%)</p>
    `;
    
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
        Object.values(row).forEach(val => {
            const td = document.createElement('td');
            td.textContent = val !== null ? val : 'NULL';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    return table;
}

function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<p>Графики доступны в панели tfjs-vis (справа).</p>';
    
    const survivalBySex = {};
    trainData.forEach(row => {
        if (row.Sex) {
            if (!survivalBySex[row.Sex]) survivalBySex[row.Sex] = { survived: 0, total: 0 };
            survivalBySex[row.Sex].total++;
            if (row.Survived === 1) survivalBySex[row.Sex].survived++;
        }
    });
    
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        x: sex, y: (stats.survived / stats.total) * 100
    }));
    
    tfvis.render.barchart({ name: 'Выживаемость по полу', tab: 'Charts' }, sexData);
}

// 4. ПРЕДОБРАБОТКА
function preprocessData() {
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Обработка...';
    
    try {
        const ages = trainData.map(row => row.Age).filter(a => a !== null);
        const ageMedian = calculateMedian(ages);
        const fares = trainData.map(row => row.Fare).filter(f => f !== null);
        const fareMedian = calculateMedian(fares);
        const embarkedMode = calculateMode(trainData.map(row => row.Embarked).filter(e => e !== null));

        const processRows = (data, isTrain) => {
            const features = [];
            const labels = [];
            const ids = [];
            data.forEach(row => {
                const feat = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
                features.push(feat);
                if (isTrain) labels.push(row[TARGET_FEATURE]);
                else ids.push(row[ID_FEATURE]);
            });
            return { features, labels, ids };
        };

        const trainResult = processRows(trainData, true);
        const testResult = processRows(testData, false);

        preprocessedTrainData = {
            features: tf.tensor2d(trainResult.features),
            labels: tf.tensor1d(trainResult.labels)
        };
        preprocessedTestData = {
            features: testResult.features, // Оставляем массивом для последующего перевода в тензор
            passengerIds: testResult.ids
        };

        outputDiv.innerHTML = 'Готово! Данные нормализованы и переведены в тензоры.';
        document.getElementById('create-model-btn').disabled = false;
    } catch (e) {
        outputDiv.innerHTML = `Ошибка: ${e.message}`;
    }
}

function extractFeatures(row, ageMed, fareMed, embMode) {
    const age = row.Age != null ? row.Age : ageMed;
    const fare = row.Fare != null ? row.Fare : fareMed;
    const embarked = row.Embarked != null ? row.Embarked : embMode;
    
    const features = [
        (age - 29) / 14, // Примерная нормализация
        (fare - 32) / 49,
        row.SibSp || 0,
        row.Parch || 0,
        row.Pclass === 1 ? 1 : 0,
        row.Pclass === 2 ? 1 : 0,
        row.Pclass === 3 ? 1 : 0,
        row.Sex === 'male' ? 1 : 0,
        row.Sex === 'female' ? 1 : 0,
        embarked === 'S' ? 1 : 0,
        embarked === 'C' ? 1 : 0,
        embarked === 'Q' ? 1 : 0
    ];

    if (document.getElementById('add-family-features').checked) {
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        features.push(familySize, familySize === 1 ? 1 : 0);
    }
    return features;
}

// 5. МОДЕЛЬ
function createModel() {
    const inputShape = preprocessedTrainData.features.shape[1];
    model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputShape] }));
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    
    document.getElementById('model-summary').innerHTML = 'Модель создана. Слои: 16 -> 8 -> 1';
    document.getElementById('train-btn').disabled = false;
}

// 6. ОБУЧЕНИЕ
async function trainModel() {
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Обучение...';
    
    const total = preprocessedTrainData.features.shape[0];
    const splitIndex = Math.floor(total * 0.8);

    // Правильное разделение тензоров
    const [trainX, valX] = tf.split(preprocessedTrainData.features, [splitIndex, total - splitIndex]);
    const [trainY, valY] = tf.split(preprocessedTrainData.labels, [splitIndex, total - splitIndex]);

    validationData = valX;
    validationLabels = valY;

    await model.fit(trainX, trainY, {
        epochs: 50,
        batchSize: 32,
        validationData: [valX, valY],
        callbacks: tfvis.show.fitCallbacks({ name: 'Обучение' }, ['loss', 'acc', 'val_acc'])
    });

    statusDiv.innerHTML = 'Обучение завершено!';
    validationPredictions = model.predict(validationData);
    document.getElementById('threshold-slider').disabled = false;
    document.getElementById('predict-btn').disabled = false;
    updateMetrics();
}

// 7. МЕТРИКИ
async function updateMetrics() {
    if (!validationPredictions) return;
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);

    const preds = validationPredictions.dataSync();
    const actuals = validationLabels.dataSync();
    
    let tp=0, tn=0, fp=0, fn=0;
    preds.forEach((p, i) => {
        const pred = p >= threshold ? 1 : 0;
        const actual = actuals[i];
        if (pred === 1 && actual === 1) tp++;
        else if (pred === 0 && actual === 0) tn++;
        else if (pred === 1 && actual === 0) fp++;
        else fn++;
    });

    document.getElementById('performance-metrics').innerHTML = `
        <p>Точность (Accuracy): ${((tp+tn)/(tp+tn+fp+fn)*100).toFixed(2)}%</p>
        <p>Верно выживших (TP): ${tp} | Верно погибших (TN): ${tn}</p>
    `;
}

// 8. ПРЕДСКАЗАНИЕ И ЭКСПОРТ
async function predict() {
    const testX = tf.tensor2d(preprocessedTestData.features);
    const results = model.predict(testX).dataSync();
    
    testPredictions = results;
    const out = document.getElementById('prediction-output');
    out.innerHTML = '<h3>Результаты (первые 10)</h3>';
    
    const table = document.createElement('table');
    table.innerHTML = '<tr><th>ID</th><th>Survived</th><th>Prob</th></tr>';
    for(let i=0; i<10; i++) {
        table.innerHTML += `<tr><td>${preprocessedTestData.passengerIds[i]}</td><td>${results[i] >= 0.5 ? 1 : 0}</td><td>${results[i].toFixed(4)}</td></tr>`;
    }
    out.appendChild(table);
    document.getElementById('export-btn').disabled = false;
}

async function exportResults() {
    let csv = 'PassengerId,Survived\n';
    preprocessedTestData.passengerIds.forEach((id, i) => {
        csv += `${id},${testPredictions[i] >= 0.5 ? 1 : 0}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'submission.csv';
    a.click();
}

// Математические помощники
function calculateMedian(arr) {
    if (!arr.length) return 0;
    const s = [...arr].sort((a,b) => a-b);
    const mid = Math.floor(s.length/2);
    return s.length % 2 ? s[mid] : (s[mid-1]+s[mid])/2;
}
function calculateMode(arr) {
    const f = {}; let max=0, m=arr[0];
    arr.forEach(v => { f[v] = (f[v]||0)+1; if(f[v]>max){max=f[v]; m=v;}});
    return m;
}
