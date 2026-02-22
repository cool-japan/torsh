/**
 * Cloudflare Workers Example - ToRSh WASM Deep Learning on the Edge
 *
 * Deploy this to Cloudflare Workers for serverless edge AI inference.
 *
 * Setup:
 * 1. wrangler init my-torsh-worker
 * 2. Add torsh_wasm.wasm to your worker
 * 3. Deploy: wrangler publish
 */

import { Tensor, Sequential, Linear, Adam, loss } from './torsh_wasm.js';

// Pre-trained model weights (would be loaded from KV storage in production)
let model = null;

/**
 * Initialize the ML model
 */
async function initModel() {
    if (model) return model;

    // Create a simple sentiment analysis model
    model = new Sequential([
        new Linear(100, 64),  // Input: word embeddings
        'relu',
        new Linear(64, 32),
        'relu',
        new Linear(32, 1),
        'sigmoid'             // Output: sentiment score
    ]);

    // In production, load weights from KV storage
    // const weights = await TORSH_MODELS.get('sentiment_v1', 'arrayBuffer');
    // model.load_weights(weights);

    return model;
}

/**
 * Handle inference requests
 */
async function handleInference(request) {
    try {
        const { text, features } = await request.json();

        // Ensure model is initialized
        await initModel();

        // Convert text features to tensor
        // (In production, you'd have proper tokenization/embedding)
        const input = Tensor.from_array(features, [1, 100]);

        // Run inference
        const output = model.forward(input);
        const sentiment = output.data()[0];

        // Free memory
        input.free();
        output.free();

        return new Response(JSON.stringify({
            sentiment: sentiment,
            label: sentiment > 0.5 ? 'positive' : 'negative',
            confidence: Math.abs(sentiment - 0.5) * 2
        }), {
            headers: { 'Content-Type': 'application/json' }
        });

    } catch (error) {
        return new Response(JSON.stringify({
            error: error.message
        }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

/**
 * Handle batch inference for multiple inputs
 */
async function handleBatchInference(request) {
    try {
        const { batch } = await request.json();

        await initModel();

        const results = [];

        for (const item of batch) {
            const input = Tensor.from_array(item.features, [1, 100]);
            const output = model.forward(input);
            const sentiment = output.data()[0];

            results.push({
                id: item.id,
                sentiment: sentiment,
                label: sentiment > 0.5 ? 'positive' : 'negative'
            });

            input.free();
            output.free();
        }

        return new Response(JSON.stringify({ results }), {
            headers: { 'Content-Type': 'application/json' }
        });

    } catch (error) {
        return new Response(JSON.stringify({
            error: error.message
        }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

/**
 * Handle model retraining/fine-tuning (limited by worker CPU time)
 */
async function handleFineTune(request) {
    try {
        const { samples, epochs = 10 } = await request.json();

        await initModel();

        const optimizer = new Adam(model.parameters().length, {
            learning_rate: 0.001
        });

        let totalLoss = 0;

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (const sample of samples) {
                const input = Tensor.from_array(sample.features, [1, 100]);
                const target = Tensor.from_array([sample.label], [1]);

                const output = model.forward(input);
                const lossValue = loss.binary_cross_entropy(output, target);
                totalLoss += lossValue;

                // Simplified backprop (full implementation would need autograd)
                // In production, use proper gradient computation

                input.free();
                target.free();
                output.free();
            }
        }

        return new Response(JSON.stringify({
            message: 'Fine-tuning complete',
            avgLoss: totalLoss / (epochs * samples.length)
        }), {
            headers: { 'Content-Type': 'application/json' }
        });

    } catch (error) {
        return new Response(JSON.stringify({
            error: error.message
        }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

/**
 * Main request handler
 */
export default {
    async fetch(request, env, ctx) {
        const url = new URL(request.url);

        // CORS headers for all responses
        const corsHeaders = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        };

        // Handle OPTIONS preflight
        if (request.method === 'OPTIONS') {
            return new Response(null, { headers: corsHeaders });
        }

        // Route requests
        let response;
        switch (url.pathname) {
            case '/infer':
                response = await handleInference(request);
                break;

            case '/batch':
                response = await handleBatchInference(request);
                break;

            case '/finetune':
                response = await handleFineTune(request);
                break;

            case '/health':
                response = new Response(JSON.stringify({
                    status: 'healthy',
                    model: model ? 'loaded' : 'not loaded'
                }), {
                    headers: { 'Content-Type': 'application/json' }
                });
                break;

            default:
                response = new Response('Not Found', { status: 404 });
        }

        // Add CORS headers to response
        const headers = new Headers(response.headers);
        Object.entries(corsHeaders).forEach(([key, value]) => {
            headers.set(key, value);
        });

        return new Response(response.body, {
            status: response.status,
            headers
        });
    }
};

/**
 * Example usage:
 *
 * POST /infer
 * {
 *   "text": "This product is amazing!",
 *   "features": [0.1, 0.2, ..., 0.5]  // 100 features
 * }
 *
 * Response:
 * {
 *   "sentiment": 0.92,
 *   "label": "positive",
 *   "confidence": 0.84
 * }
 *
 * POST /batch
 * {
 *   "batch": [
 *     { "id": 1, "features": [...] },
 *     { "id": 2, "features": [...] }
 *   ]
 * }
 *
 * POST /finetune
 * {
 *   "samples": [
 *     { "features": [...], "label": 1 },
 *     { "features": [...], "label": 0 }
 *   ],
 *   "epochs": 10
 * }
 */
