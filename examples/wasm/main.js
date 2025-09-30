import init, { 
    DigitClassifier, 
    SentimentAnalyzer, 
    ImageProcessor,
    Benchmark,
    demo 
} from './pkg/torsh_wasm_example.js';

let digitClassifier;
let sentimentAnalyzer;

// Drawing variables
let isDrawing = false;
let drawCanvas;
let drawCtx;
let previewCanvas;
let previewCtx;

// Initialize the WASM module
async function initializeWasm() {
    try {
        await init();
        
        // Initialize models
        digitClassifier = new DigitClassifier();
        sentimentAnalyzer = new SentimentAnalyzer();
        
        // Show model info
        displayModelInfo();
        
        // Hide loading, show content
        document.getElementById('loading').style.display = 'none';
        document.getElementById('main-content').style.display = 'block';
        
        // Set up canvas and event listeners
        setupCanvas();
        setupEventListeners();
        
        // Run demo in console
        demo();
        
        console.log('✅ ToRSh WASM initialized successfully!');
    } catch (error) {
        console.error('Failed to initialize WASM:', error);
        document.getElementById('loading').innerHTML = 
            '<div class="error">Failed to load WASM module: ' + error.message + '</div>';
    }
}

// Display model information
function displayModelInfo() {
    const info = digitClassifier.model_info();
    const infoDiv = document.getElementById('model-info');
    
    infoDiv.innerHTML = `
        <div><strong>Model:</strong> ${info.name}</div>
        <div><strong>Version:</strong> ${info.version}</div>
        <div><strong>Parameters:</strong> ${info.parameters.toLocaleString()}</div>
        <div><strong>Input Shape:</strong> [${info.input_shape.join(', ')}]</div>
        <div><strong>Output Classes:</strong> ${info.output_classes}</div>
        <div><strong>WASM Status:</strong> ✅ Loaded</div>
    `;
}

// Set up drawing canvas
function setupCanvas() {
    drawCanvas = document.getElementById('draw-canvas');
    drawCtx = drawCanvas.getContext('2d');
    previewCanvas = document.getElementById('preview-canvas');
    previewCtx = previewCanvas.getContext('2d');
    
    // Set up drawing style
    drawCtx.fillStyle = 'white';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawCtx.strokeStyle = 'black';
    drawCtx.lineWidth = 15;
    drawCtx.lineCap = 'round';
    drawCtx.lineJoin = 'round';
    
    // Mouse events
    drawCanvas.addEventListener('mousedown', startDrawing);
    drawCanvas.addEventListener('mousemove', draw);
    drawCanvas.addEventListener('mouseup', stopDrawing);
    drawCanvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events for mobile
    drawCanvas.addEventListener('touchstart', handleTouch);
    drawCanvas.addEventListener('touchmove', handleTouch);
    drawCanvas.addEventListener('touchend', stopDrawing);
    
    clearCanvas();
}

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    const rect = drawCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    drawCtx.beginPath();
    drawCtx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = drawCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    drawCtx.lineTo(x, y);
    drawCtx.stroke();
    
    updatePreview();
}

function stopDrawing() {
    isDrawing = false;
}

function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                     e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    drawCanvas.dispatchEvent(mouseEvent);
}

function clearCanvas() {
    drawCtx.fillStyle = 'white';
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    previewCtx.fillStyle = 'white';
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    
    document.getElementById('digit-results').style.display = 'none';
}

function updatePreview() {
    // Draw scaled down version in preview canvas
    previewCtx.fillStyle = 'white';
    previewCtx.fillRect(0, 0, 28, 28);
    previewCtx.drawImage(drawCanvas, 0, 0, 28, 28);
}

// Set up event listeners
function setupEventListeners() {
    // Digit classification
    document.getElementById('predict-button').addEventListener('click', predictDigit);
    document.getElementById('clear-button').addEventListener('click', clearCanvas);
    
    // Sentiment analysis
    document.getElementById('analyze-button').addEventListener('click', analyzeSentiment);
    
    // Benchmark
    document.getElementById('benchmark-button').addEventListener('click', runBenchmark);
}

// Predict digit from canvas
async function predictDigit() {
    try {
        updatePreview();
        
        // Get image data from preview canvas
        const imageData = previewCtx.getImageData(0, 0, 28, 28);
        
        // Convert to grayscale using ImageProcessor
        const grayscale = ImageProcessor.process_canvas_data(
            new Uint8Array(imageData.data),
            28,
            28
        );
        
        // Invert colors (our canvas has white background, MNIST expects black)
        const inverted = grayscale.map(pixel => 1.0 - pixel);
        
        // Make prediction
        const prediction = digitClassifier.predict(inverted);
        
        // Display results
        displayDigitResults(prediction);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction: ' + error);
    }
}

// Display digit classification results
function displayDigitResults(prediction) {
    document.getElementById('digit-results').style.display = 'block';
    document.getElementById('predicted-digit').textContent = prediction.digit;
    document.getElementById('confidence').textContent = (prediction.confidence * 100).toFixed(1);
    
    // Update confidence bar
    const confidenceBar = document.getElementById('confidence-bar');
    confidenceBar.style.width = (prediction.confidence * 100) + '%';
    
    // Display all probabilities
    const probGrid = document.getElementById('probability-grid');
    probGrid.innerHTML = '';
    
    const probs = prediction.probabilities;
    for (let i = 0; i < 10; i++) {
        const probItem = document.createElement('div');
        probItem.className = 'prob-item';
        if (i === prediction.digit) {
            probItem.classList.add('highest');
        }
        probItem.innerHTML = `
            <div style="font-size: 20px;">${i}</div>
            <div>${(probs[i] * 100).toFixed(1)}%</div>
        `;
        probGrid.appendChild(probItem);
    }
}

// Simple tokenizer for sentiment analysis
function tokenize(text) {
    // Very simple tokenization - in practice, use a proper tokenizer
    const words = text.toLowerCase().split(/\s+/);
    const vocab = ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'love', 'hate', 
                   'good', 'bad', 'happy', 'sad', 'great', 'terrible', 'amazing',
                   'awful', 'excellent', 'poor', 'i', 'you', 'it', 'this', 'that'];
    
    // Convert to indices (simple hash)
    const indices = words.map(word => {
        const idx = vocab.indexOf(word);
        return idx >= 0 ? idx : Math.abs(hashCode(word)) % 10000;
    });
    
    return new Uint32Array(indices.slice(0, 20)); // Max 20 tokens
}

function hashCode(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
}

// Analyze sentiment
async function analyzeSentiment() {
    try {
        const text = document.getElementById('sentiment-input').value;
        if (!text.trim()) {
            alert('Please enter some text to analyze');
            return;
        }
        
        // Tokenize text
        const tokens = tokenize(text);
        
        // Analyze sentiment
        const result = sentimentAnalyzer.analyze(tokens);
        
        // Display results
        displaySentimentResults(result);
        
    } catch (error) {
        console.error('Sentiment analysis error:', error);
        alert('Error analyzing sentiment: ' + error);
    }
}

// Display sentiment results
function displaySentimentResults(result) {
    document.getElementById('sentiment-results').style.display = 'block';
    document.getElementById('sentiment-label').textContent = 
        result.label.charAt(0).toUpperCase() + result.label.slice(1);
    
    document.getElementById('negative-score').textContent = 
        (result.negative * 100).toFixed(0) + '%';
    document.getElementById('neutral-score').textContent = 
        (result.neutral * 100).toFixed(0) + '%';
    document.getElementById('positive-score').textContent = 
        (result.positive * 100).toFixed(0) + '%';
}

// Run performance benchmark
async function runBenchmark() {
    const button = document.getElementById('benchmark-button');
    button.disabled = true;
    button.textContent = 'Running...';
    
    try {
        // Run benchmark with different iteration counts
        const iterations = [10, 100, 1000];
        let output = 'Performance Benchmark Results\n';
        output += '============================\n\n';
        
        for (const iter of iterations) {
            const result = Benchmark.inference_speed_test(iter);
            output += `${iter} iterations:\n`;
            output += `  Total time: ${result.total_time_ms.toFixed(2)}ms\n`;
            output += `  Avg inference: ${result.avg_inference_ms.toFixed(3)}ms\n`;
            output += `  Throughput: ${result.throughput.toFixed(0)} inferences/sec\n\n`;
        }
        
        // Display results
        document.getElementById('benchmark-results').style.display = 'block';
        document.getElementById('benchmark-output').textContent = output;
        
    } catch (error) {
        console.error('Benchmark error:', error);
        alert('Error running benchmark: ' + error);
    } finally {
        button.disabled = false;
        button.textContent = 'Run Benchmark';
    }
}

// Initialize when page loads
initializeWasm();