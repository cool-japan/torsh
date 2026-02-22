// ToRSh Performance Dashboard JavaScript
// Provides real-time interactivity and WebSocket communication

// Global variables
let websocketConnection = null;
let refreshInterval = null;
let charts = {};
let isFullscreen = false;
let isDarkMode = false;

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    if (typeof WEBSOCKET_URL !== 'undefined' && WEBSOCKET_URL) {
        initializeWebSocket();
    }
});

// Dashboard initialization
function initializeDashboard() {
    console.log('ToRSh Dashboard initializing...');
    updateTimestamp();
    initializeCharts();

    // Start auto-refresh if configured
    if (typeof REFRESH_INTERVAL !== 'undefined' && REFRESH_INTERVAL > 0) {
        refreshInterval = setInterval(refreshDashboard, REFRESH_INTERVAL);
    }
}

// Set up event listeners for dashboard controls
function setupEventListeners() {
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key) {
                case 'r':
                    e.preventDefault();
                    refreshDashboard();
                    break;
                case 'f':
                    e.preventDefault();
                    toggleFullscreen();
                    break;
                case 'd':
                    e.preventDefault();
                    toggleDarkMode();
                    break;
            }
        }
        if (e.key === 'Escape' && isFullscreen) {
            exitFullscreen();
        }
    });

    // Window visibility change
    document.addEventListener('visibilitychange', function() {
        if (document.hidden && refreshInterval) {
            clearInterval(refreshInterval);
        } else if (!document.hidden && typeof REFRESH_INTERVAL !== 'undefined') {
            refreshInterval = setInterval(refreshDashboard, REFRESH_INTERVAL);
        }
    });
}

// Refresh dashboard data
function refreshDashboard() {
    console.log('Refreshing dashboard...');

    // Update timestamp
    updateTimestamp();

    // Add visual feedback
    const refreshButton = document.querySelector('.refresh-button');
    if (refreshButton) {
        refreshButton.style.animation = 'spin 0.5s linear';
        setTimeout(() => {
            refreshButton.style.animation = '';
        }, 500);
    }

    // If WebSocket is available, request fresh data
    if (websocketConnection && websocketConnection.readyState === WebSocket.OPEN) {
        websocketConnection.send(JSON.stringify({
            type: 'refresh_request',
            timestamp: Date.now()
        }));
    } else {
        // Fallback: reload the page
        setTimeout(() => {
            window.location.reload();
        }, 500);
    }
}

// Toggle settings panel
function toggleSettings() {
    let settingsPanel = document.querySelector('.settings-panel');

    if (!settingsPanel) {
        // Create settings panel if it doesn't exist
        settingsPanel = createSettingsPanel();
        document.body.appendChild(settingsPanel);
    }

    settingsPanel.style.display = settingsPanel.style.display === 'none' ? 'block' : 'none';
}

// Create settings panel
function createSettingsPanel() {
    const panel = document.createElement('div');
    panel.className = 'settings-panel';
    panel.style.display = 'none';
    panel.innerHTML = `
        <div class="settings-content">
            <h3>Dashboard Settings</h3>
            <div class="setting-item">
                <label>
                    <input type="checkbox" id="darkModeToggle" ${isDarkMode ? 'checked' : ''}>
                    Dark Mode
                </label>
            </div>
            <div class="setting-item">
                <label>
                    Auto-refresh interval (seconds):
                    <select id="refreshIntervalSelect">
                        <option value="0">Disabled</option>
                        <option value="1000">1s</option>
                        <option value="5000" selected>5s</option>
                        <option value="10000">10s</option>
                        <option value="30000">30s</option>
                    </select>
                </label>
            </div>
            <div class="setting-item">
                <button onclick="exportData()">Export Data</button>
                <button onclick="clearData()">Clear Data</button>
            </div>
            <div class="setting-item">
                <button onclick="toggleSettings()">Close</button>
            </div>
        </div>
    `;

    // Add event listeners
    panel.querySelector('#darkModeToggle').addEventListener('change', toggleDarkMode);
    panel.querySelector('#refreshIntervalSelect').addEventListener('change', updateRefreshInterval);

    return panel;
}

// Toggle fullscreen mode
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().then(() => {
            isFullscreen = true;
            const button = document.querySelector('.fullscreen-button');
            if (button) button.innerHTML = '⬇️ Exit Fullscreen';
        });
    } else {
        exitFullscreen();
    }
}

// Exit fullscreen mode
function exitFullscreen() {
    if (document.exitFullscreen) {
        document.exitFullscreen().then(() => {
            isFullscreen = false;
            const button = document.querySelector('.fullscreen-button');
            if (button) button.innerHTML = '⛶ Fullscreen';
        });
    }
}

// Toggle dark mode
function toggleDarkMode() {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle('dark-mode', isDarkMode);
    localStorage.setItem('dashboardDarkMode', isDarkMode);

    // Update charts if they exist
    Object.values(charts).forEach(chart => {
        if (chart && chart.update) {
            chart.update();
        }
    });
}

// Update refresh interval
function updateRefreshInterval(e) {
    const newInterval = parseInt(e.target.value);

    if (refreshInterval) {
        clearInterval(refreshInterval);
    }

    if (newInterval > 0) {
        refreshInterval = setInterval(refreshDashboard, newInterval);
    }
}

// Initialize charts (placeholder - would integrate with Chart.js or similar)
function initializeCharts() {
    // Performance chart
    const perfCanvas = document.querySelector('#performanceChart');
    if (perfCanvas) {
        charts.performance = createPerformanceChart(perfCanvas);
    }

    // Memory chart
    const memCanvas = document.querySelector('#memoryChart');
    if (memCanvas) {
        charts.memory = createMemoryChart(memCanvas);
    }
}

// Create performance chart (mock implementation)
function createPerformanceChart(canvas) {
    // This would integrate with a real charting library
    const ctx = canvas.getContext('2d');

    // Simple mock chart
    return {
        update: function() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = isDarkMode ? '#fff' : '#333';
            ctx.font = '16px Arial';
            ctx.fillText('Performance Chart', 10, 30);
            ctx.fillText('(Chart.js integration needed)', 10, 50);
        }
    };
}

// Create memory chart (mock implementation)
function createMemoryChart(canvas) {
    // This would integrate with a real charting library
    const ctx = canvas.getContext('2d');

    return {
        update: function() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = isDarkMode ? '#fff' : '#333';
            ctx.font = '16px Arial';
            ctx.fillText('Memory Chart', 10, 30);
            ctx.fillText('(Chart.js integration needed)', 10, 50);
        }
    };
}

// Initialize WebSocket connection
function initializeWebSocket() {
    try {
        websocketConnection = new WebSocket(WEBSOCKET_URL);

        websocketConnection.onopen = function(event) {
            console.log('WebSocket connection established');
            updateConnectionStatus('connected');
        };

        websocketConnection.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        websocketConnection.onerror = function(event) {
            console.error('WebSocket error:', event);
            updateConnectionStatus('error');
        };

        websocketConnection.onclose = function(event) {
            console.log('WebSocket connection closed');
            updateConnectionStatus('disconnected');

            // Attempt to reconnect after 5 seconds
            setTimeout(() => {
                if (websocketConnection.readyState === WebSocket.CLOSED) {
                    initializeWebSocket();
                }
            }, 5000);
        };

    } catch (e) {
        console.error('Failed to create WebSocket connection:', e);
        updateConnectionStatus('error');
    }
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'dashboard_update':
            updateDashboardData(data.data);
            break;
        case 'alert':
            handleAlert(data.alert);
            break;
        case 'performance_metrics':
            updatePerformanceMetrics(data.metrics);
            break;
        case 'memory_metrics':
            updateMemoryMetrics(data.metrics);
            break;
        default:
            console.log('Received unknown message type:', data.type);
    }
}

// Update dashboard data from WebSocket
function updateDashboardData(data) {
    // Update performance metrics
    if (data.performance_metrics) {
        updatePerformanceDisplay(data.performance_metrics);
    }

    // Update memory metrics
    if (data.memory_metrics) {
        updateMemoryDisplay(data.memory_metrics);
    }

    // Update system metrics
    if (data.system_metrics) {
        updateSystemDisplay(data.system_metrics);
    }

    // Update timestamp
    updateTimestamp();
}

// Update performance display
function updatePerformanceDisplay(metrics) {
    const elements = {
        'total-operations': metrics.total_operations,
        'avg-duration': metrics.average_duration_ms?.toFixed(2) + ' ms',
        'ops-per-second': metrics.operations_per_second?.toFixed(1),
        'gflops-per-second': metrics.gflops_per_second?.toFixed(2),
        'cpu-utilization': metrics.cpu_utilization?.toFixed(1) + '%',
        'thread-count': metrics.thread_count
    };

    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element && value !== undefined) {
            element.textContent = value;
        }
    });
}

// Update memory display
function updateMemoryDisplay(metrics) {
    const elements = {
        'current-memory': metrics.current_usage_mb?.toFixed(1) + ' MB',
        'peak-memory': metrics.peak_usage_mb?.toFixed(1) + ' MB',
        'total-allocations': metrics.total_allocations,
        'active-allocations': metrics.active_allocations,
        'fragmentation-ratio': metrics.fragmentation_ratio?.toFixed(3),
        'allocation-rate': metrics.allocation_rate?.toFixed(1) + '/s'
    };

    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element && value !== undefined) {
            element.textContent = value;
        }
    });
}

// Update system display
function updateSystemDisplay(metrics) {
    const elements = {
        'uptime': formatUptime(metrics.uptime_seconds),
        'load-average': metrics.load_average?.toFixed(2),
        'available-memory': metrics.available_memory_mb?.toFixed(1) + ' MB',
        'disk-usage': metrics.disk_usage_percent?.toFixed(1) + '%',
        'network-io': metrics.network_io_mbps?.toFixed(1) + ' MB/s'
    };

    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element && value !== undefined) {
            element.textContent = value;
        }
    });
}

// Handle alerts from WebSocket
function handleAlert(alert) {
    showNotification(alert.message, alert.severity);

    // Add to alerts list
    const alertsList = document.querySelector('.alerts-list');
    if (alertsList) {
        const alertElement = createAlertElement(alert);
        alertsList.insertBefore(alertElement, alertsList.firstChild);

        // Limit to 10 most recent alerts
        while (alertsList.children.length > 10) {
            alertsList.removeChild(alertsList.lastChild);
        }
    }
}

// Create alert element
function createAlertElement(alert) {
    const div = document.createElement('div');
    div.className = `alert alert-${alert.severity.toLowerCase()}`;
    div.innerHTML = `
        <span class="alert-time">${new Date(alert.timestamp * 1000).toLocaleTimeString()}</span>
        <span class="alert-message">${alert.message}</span>
        <button class="alert-dismiss" onclick="this.parentElement.remove()">×</button>
    `;
    return div;
}

// Show notification
function showNotification(message, severity) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${severity.toLowerCase()}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">×</button>
    `;

    // Add to document
    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.parentElement.removeChild(notification);
        }
    }, 5000);
}

// Update connection status indicator
function updateConnectionStatus(status) {
    const indicator = document.querySelector('.status-dot');
    if (indicator) {
        indicator.className = `status-dot ${status}`;
    }

    const statusText = document.querySelector('.status-indicator span:last-child');
    if (statusText) {
        const statusTexts = {
            connected: 'Live',
            disconnected: 'Offline',
            error: 'Error'
        };
        statusText.textContent = statusTexts[status] || status;
    }
}

// Update timestamp
function updateTimestamp() {
    const timestampElement = document.querySelector('.timestamp');
    if (timestampElement) {
        const now = new Date();
        timestampElement.textContent = `Last updated: ${now.toLocaleString()}`;
    }
}

// Format uptime
function formatUptime(seconds) {
    if (!seconds) return '0s';

    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (days > 0) {
        return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

// Export data functionality
function exportData() {
    const data = {
        timestamp: Date.now(),
        dashboard_data: gatherCurrentData(),
        alerts: gatherCurrentAlerts(),
        settings: {
            dark_mode: isDarkMode,
            refresh_interval: refreshInterval
        }
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `torsh_dashboard_${new Date().toISOString().slice(0, 19)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Clear data functionality
function clearData() {
    if (confirm('Are you sure you want to clear all dashboard data?')) {
        // Clear local storage
        localStorage.clear();

        // Clear alerts
        const alertsList = document.querySelector('.alerts-list');
        if (alertsList) {
            alertsList.innerHTML = '';
        }

        // Reset charts
        Object.values(charts).forEach(chart => {
            if (chart && chart.reset) {
                chart.reset();
            }
        });

        showNotification('Dashboard data cleared', 'info');
    }
}

// Gather current dashboard data
function gatherCurrentData() {
    // This would collect current metrics from the DOM
    return {
        performance: {},
        memory: {},
        system: {}
    };
}

// Gather current alerts
function gatherCurrentAlerts() {
    const alerts = [];
    const alertElements = document.querySelectorAll('.alert');

    alertElements.forEach(element => {
        alerts.push({
            message: element.querySelector('.alert-message')?.textContent || '',
            severity: element.className.includes('critical') ? 'critical' :
                     element.className.includes('warning') ? 'warning' : 'info',
            timestamp: Date.now()
        });
    });

    return alerts;
}

// Load saved preferences
function loadPreferences() {
    const darkMode = localStorage.getItem('dashboardDarkMode');
    if (darkMode === 'true') {
        isDarkMode = true;
        document.body.classList.add('dark-mode');
    }
}

// Initialize preferences on load
loadPreferences();

// CSS animations for smooth transitions
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 16px;
        border-radius: 4px;
        color: white;
        font-weight: bold;
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    }

    .notification-info { background: #007bff; }
    .notification-warning { background: #ffc107; color: #333; }
    .notification-critical { background: #dc3545; }

    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }

    .settings-panel {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        z-index: 1000;
        max-width: 400px;
        width: 90%;
    }

    .dark-mode .settings-panel {
        background: #333;
        border-color: #555;
        color: white;
    }

    .setting-item {
        margin: 15px 0;
    }

    .setting-item label {
        display: block;
        margin-bottom: 5px;
    }

    .setting-item input, .setting-item select {
        width: 100%;
        padding: 8px;
        border: 1px solid #ddd;
        border-radius: 4px;
    }

    .dark-mode .setting-item input,
    .dark-mode .setting-item select {
        background: #444;
        border-color: #555;
        color: white;
    }

    .status-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }

    .status-dot.connected { background: #28a745; }
    .status-dot.disconnected { background: #6c757d; }
    .status-dot.error { background: #dc3545; }
    .status-dot.active {
        background: #28a745;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
`;
document.head.appendChild(style);

console.log('ToRSh Dashboard JavaScript loaded successfully');