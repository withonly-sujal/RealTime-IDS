/* ═══════════════════════════════════════════════════════════
   Real-Time IDS Dashboard — JavaScript Controller
   ═══════════════════════════════════════════════════════════ */

(() => {
    'use strict';

    // ── Chart.js Global Defaults (dark theme) ──
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(148, 163, 184, 0.08)';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.pointStyleWidth = 8;
    Chart.defaults.animation.duration = 400;
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;

    // ── Constants ──
    const MAX_TIMELINE_POINTS = 60;
    const MAX_TABLE_ROWS = 100;
    const MAX_LOG_ENTRIES = 250;
    const MAX_TOP_IPS = 8;
    const WS_RECONNECT_DELAY = 3000;

    // ── State ──
    const state = {
        ws: null,
        connected: false,
        charts: {},
        totalPredictions: 0,
        timeline: { labels: [], data: [], colors: [] },
        protocols: { TCP: 0, UDP: 0 },
        probBuckets: new Array(10).fill(0),
        ipCounts: {},
        tableRows: [],
        logEntries: [],
    };


    // ═══════════════════════════════════════════
    //  WEBSOCKET
    // ═══════════════════════════════════════════

    function connectWebSocket() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${location.host}/ws`;

        state.ws = new WebSocket(wsUrl);

        state.ws.onopen = () => {
            state.connected = true;
            setConnectionStatus('connected');
            addLogEntry('info', 'Connected to IDS server');
        };

        state.ws.onclose = () => {
            if (!state.connected) return;   // already handled
            state.connected = false;
            setConnectionStatus('disconnected');
            addLogEntry('error', '⏹ IDS server disconnected');
        };

        state.ws.onerror = () => {
            state.connected = false;
        };

        state.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleEvent(msg);
            } catch (e) {
                console.error('Failed to parse event:', e);
            }
        };
    }

    function handleEvent(event) {
        switch (event.type) {
            case 'prediction':
                handlePrediction(event.data);
                break;
            case 'stats':
                handleStats(event.data);
                break;
            case 'status':
                handleStatus(event.data);
                break;
            case 'error':
                addLogEntry('error', event.data.message);
                break;
        }
    }


    // ═══════════════════════════════════════════
    //  EVENT HANDLERS
    // ═══════════════════════════════════════════

    function handlePrediction(data) {
        state.totalPredictions++;

        const isAttack = data.prediction === 1;
        const time = formatTime(data.timestamp);

        // ── Update timeline data ──
        state.timeline.labels.push(time);
        state.timeline.data.push(data.probability);
        state.timeline.colors.push(isAttack ? '#f43f5e' : '#34d399');

        if (state.timeline.labels.length > MAX_TIMELINE_POINTS) {
            state.timeline.labels.shift();
            state.timeline.data.shift();
            state.timeline.colors.shift();
        }

        // ── Update protocol counts ──
        const proto = data.protocol || 'TCP';
        if (proto in state.protocols) {
            state.protocols[proto]++;
        }

        // ── Update probability histogram ──
        const bucketIdx = Math.min(Math.floor(data.probability * 10), 9);
        state.probBuckets[bucketIdx]++;

        // ── Update IP counts ──
        const srcIp = data.src_ip;
        state.ipCounts[srcIp] = (state.ipCounts[srcIp] || 0) + 1;

        // ── Update charts ──
        updateTimelineChart();
        updateProtocolChart();
        updateProbabilityChart();
        updateIPChart();

        // ── Update UI badge ──
        document.getElementById('timeline-count').textContent =
            `${state.totalPredictions} prediction${state.totalPredictions !== 1 ? 's' : ''}`;
        document.getElementById('detection-count').textContent =
            `${state.totalPredictions} total`;

        // ── Add table row ──
        addTableRow(data, time);

        // ── Add log entry ──
        if (isAttack) {
            addLogEntry('attack',
                `🚨 ATTACK — ${data.src_ip}:${data.src_port} → ${data.dst_ip}:${data.dst_port} [${proto}] Prob: ${data.probability.toFixed(4)}`
            );
        } else {
            addLogEntry('normal',
                `✅ Normal — ${data.src_ip}:${data.src_port} → ${data.dst_ip}:${data.dst_port} [${proto}]`
            );
        }
    }

    function handleStats(data) {
        animateValue('val-packets', data.total_packets);
        animateValue('val-flows', data.active_flows);
        animateValue('val-attacks', data.attacks_detected);
        animateValue('val-normal', data.normal_flows);

        const adr = data.adr !== undefined ? data.adr : 0;
        document.getElementById('val-adr').textContent = adr.toFixed(1);

        if (data.uptime !== undefined) {
            document.getElementById('val-uptime').textContent = formatUptime(data.uptime);
        }
    }

    function handleStatus(data) {
        if (data.status === 'running') {
            setConnectionStatus('running');
            addLogEntry('info', `⚡ ${data.message}`);
        } else if (data.status === 'stopped') {
            setConnectionStatus('disconnected');
            addLogEntry('error', `⏹ ${data.message}`);
        } else if (data.status === 'error') {
            setConnectionStatus('disconnected');
            addLogEntry('error', `❌ ${data.message}`);
        }
    }


    // ═══════════════════════════════════════════
    //  CHARTS
    // ═══════════════════════════════════════════

    function initCharts() {
        // ── Timeline Chart ──
        const tlCtx = document.getElementById('chart-timeline').getContext('2d');
        state.charts.timeline = new Chart(tlCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Attack Probability',
                    data: [],
                    borderColor: '#38bdf8',
                    borderWidth: 2,
                    pointBackgroundColor: [],
                    pointBorderColor: [],
                    pointRadius: 4,
                    pointHoverRadius: 7,
                    fill: true,
                    backgroundColor: createGradient(tlCtx, 'rgba(56, 189, 248, 0.15)', 'rgba(56, 189, 248, 0.01)'),
                    tension: 0.35,
                }, {
                    label: 'Threshold (0.7)',
                    data: [],
                    borderColor: 'rgba(244, 63, 94, 0.4)',
                    borderWidth: 1,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    fill: false,
                }]
            },
            options: {
                scales: {
                    y: { min: 0, max: 1, ticks: { stepSize: 0.2 } },
                    x: { ticks: { maxTicksLimit: 10, maxRotation: 0 } }
                },
                plugins: {
                    legend: { display: true, position: 'top', align: 'end' },
                    tooltip: {
                        backgroundColor: 'rgba(13, 21, 41, 0.95)',
                        borderColor: 'rgba(56, 189, 248, 0.2)',
                        borderWidth: 1,
                        titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
                        bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
                    }
                },
                interaction: { intersect: false, mode: 'index' }
            }
        });

        // ── Protocol Doughnut ──
        const prCtx = document.getElementById('chart-protocol').getContext('2d');
        state.charts.protocol = new Chart(prCtx, {
            type: 'doughnut',
            data: {
                labels: ['TCP', 'UDP'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: ['#38bdf8', '#a78bfa'],
                    borderColor: ['rgba(56, 189, 248, 0.3)', 'rgba(167, 139, 250, 0.3)'],
                    borderWidth: 2,
                    hoverBorderWidth: 3,
                    spacing: 3,
                }]
            },
            options: {
                cutout: '68%',
                plugins: {
                    legend: { position: 'bottom', labels: { padding: 16 } },
                    tooltip: {
                        backgroundColor: 'rgba(13, 21, 41, 0.95)',
                        borderColor: 'rgba(56, 189, 248, 0.2)',
                        borderWidth: 1,
                    }
                }
            }
        });

        // ── Probability Histogram ──
        const pbCtx = document.getElementById('chart-probability').getContext('2d');
        const probLabels = [];
        const probColors = [];
        for (let i = 0; i < 10; i++) {
            probLabels.push(`${(i * 0.1).toFixed(1)}-${((i + 1) * 0.1).toFixed(1)}`);
            // Gradient green → amber → red
            const ratio = i / 9;
            if (ratio < 0.5) {
                probColors.push(`rgba(52, 211, 153, ${0.5 + ratio})`);
            } else if (ratio < 0.7) {
                probColors.push(`rgba(251, 191, 36, ${0.5 + ratio * 0.5})`);
            } else {
                probColors.push(`rgba(244, 63, 94, ${0.4 + ratio * 0.5})`);
            }
        }

        state.charts.probability = new Chart(pbCtx, {
            type: 'bar',
            data: {
                labels: probLabels,
                datasets: [{
                    label: 'Predictions',
                    data: new Array(10).fill(0),
                    backgroundColor: probColors,
                    borderColor: probColors.map(c => c.replace(/[\d.]+\)$/, '0.8)')),
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                scales: {
                    y: { beginAtZero: true, ticks: { precision: 0 } },
                    x: { ticks: { maxRotation: 45, font: { size: 10 } } }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(13, 21, 41, 0.95)',
                        borderColor: 'rgba(56, 189, 248, 0.2)',
                        borderWidth: 1,
                    }
                }
            }
        });

        // ── Top IPs Chart ──
        const ipCtx = document.getElementById('chart-ips').getContext('2d');
        state.charts.ips = new Chart(ipCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Flows',
                    data: [],
                    backgroundColor: 'rgba(56, 189, 248, 0.35)',
                    borderColor: 'rgba(56, 189, 248, 0.6)',
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: { beginAtZero: true, ticks: { precision: 0 } },
                    y: {
                        ticks: {
                            font: { family: "'JetBrains Mono', monospace", size: 11 }
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(13, 21, 41, 0.95)',
                        borderColor: 'rgba(56, 189, 248, 0.2)',
                        borderWidth: 1,
                    }
                }
            }
        });
    }

    function createGradient(ctx, colorTop, colorBottom) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 260);
        gradient.addColorStop(0, colorTop);
        gradient.addColorStop(1, colorBottom);
        return gradient;
    }

    function updateTimelineChart() {
        const chart = state.charts.timeline;
        chart.data.labels = [...state.timeline.labels];
        chart.data.datasets[0].data = [...state.timeline.data];
        chart.data.datasets[0].pointBackgroundColor = [...state.timeline.colors];
        chart.data.datasets[0].pointBorderColor = state.timeline.colors.map(c =>
            c === '#f43f5e' ? 'rgba(244, 63, 94, 0.5)' : 'rgba(52, 211, 153, 0.5)'
        );
        // Threshold line
        chart.data.datasets[1].data = state.timeline.labels.map(() => 0.7);
        chart.update('none');
    }

    function updateProtocolChart() {
        const chart = state.charts.protocol;
        chart.data.datasets[0].data = [state.protocols.TCP, state.protocols.UDP];
        chart.update('none');
    }

    function updateProbabilityChart() {
        const chart = state.charts.probability;
        chart.data.datasets[0].data = [...state.probBuckets];
        chart.update('none');
    }

    function updateIPChart() {
        const chart = state.charts.ips;

        // Sort IPs by count, take top N
        const sorted = Object.entries(state.ipCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, MAX_TOP_IPS);

        chart.data.labels = sorted.map(([ip]) => ip);
        chart.data.datasets[0].data = sorted.map(([, count]) => count);
        chart.update('none');
    }


    // ═══════════════════════════════════════════
    //  TABLE
    // ═══════════════════════════════════════════

    function addTableRow(data, time) {
        const tbody = document.getElementById('detections-body');
        const isAttack = data.prediction === 1;

        // Remove "Waiting..." placeholder
        const emptyRow = tbody.querySelector('.empty-row');
        if (emptyRow) emptyRow.remove();

        // Create row
        const tr = document.createElement('tr');
        tr.className = `row-clickable ${isAttack ? 'row-attack' : 'row-normal'}`;

        if (isAttack) {
            tr.classList.add('attack-glow');
        }

        const probPct = Math.round(data.probability * 100);

        tr.innerHTML = `
            <td class="cell-time">${time}</td>
            <td class="cell-ip">${data.src_ip}<span class="cell-port">:${data.src_port}</span></td>
            <td class="cell-ip">${data.dst_ip}<span class="cell-port">:${data.dst_port}</span></td>
            <td class="cell-proto">${data.protocol}</td>
            <td>
                <span class="status-pill ${isAttack ? 'pill-attack' : 'pill-normal'}">
                    ${isAttack ? '🚨 Attack' : '✅ Normal'}
                </span>
            </td>
            <td>
                <div class="prob-cell">
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill ${isAttack ? 'fill-attack' : 'fill-normal'}"
                             style="width: ${probPct}%"></div>
                    </div>
                    <span class="prob-value ${isAttack ? 'val-attack' : 'val-normal'}">
                        ${data.probability.toFixed(4)}
                    </span>
                </div>
            </td>
        `;

        // Store features for modal
        tr.dataset.features = JSON.stringify(data.features || {});
        tr.dataset.flowKey = data.flow_key || '';
        tr.addEventListener('click', () => showFeatureModal(tr.dataset.flowKey, tr.dataset.features));

        // Insert at top
        tbody.insertBefore(tr, tbody.firstChild);

        // Limit rows
        while (tbody.children.length > MAX_TABLE_ROWS) {
            tbody.removeChild(tbody.lastChild);
        }
    }


    // ═══════════════════════════════════════════
    //  ALERT LOG
    // ═══════════════════════════════════════════

    function addLogEntry(type, message) {
        const log = document.getElementById('alert-log');
        const time = formatTime(Date.now() / 1000);

        const entry = document.createElement('div');
        entry.className = `log-entry log-${type}`;
        entry.innerHTML = `
            <span class="log-time">${time}</span>
            <span class="log-msg">${escapeHtml(message)}</span>
        `;

        log.appendChild(entry);

        // Auto-scroll to bottom
        log.scrollTop = log.scrollHeight;

        // Limit entries
        while (log.children.length > MAX_LOG_ENTRIES) {
            log.removeChild(log.firstChild);
        }
    }

    function clearLog() {
        const log = document.getElementById('alert-log');
        log.innerHTML = '';
        addLogEntry('info', 'Log cleared');
    }


    // ═══════════════════════════════════════════
    //  FEATURE MODAL
    // ═══════════════════════════════════════════

    function showFeatureModal(flowKey, featuresJson) {
        const features = JSON.parse(featuresJson);
        const overlay = document.getElementById('feature-modal-overlay');
        const body = document.getElementById('feature-modal-body');

        let html = `<div class="feature-flow-key" style="margin-bottom:16px;font-family:var(--font-mono);font-size:0.78rem;color:var(--text-muted);word-break:break-all;">${escapeHtml(flowKey)}</div>`;
        html += '<div class="feature-grid">';

        for (const [key, value] of Object.entries(features)) {
            const displayVal = typeof value === 'number' ? value.toFixed(4) : value;
            html += `
                <div class="feature-item">
                    <span class="feature-name">${escapeHtml(key)}</span>
                    <span class="feature-value">${escapeHtml(String(displayVal))}</span>
                </div>
            `;
        }

        html += '</div>';
        body.innerHTML = html;
        overlay.classList.add('active');
    }

    function hideFeatureModal() {
        document.getElementById('feature-modal-overlay').classList.remove('active');
    }


    // ═══════════════════════════════════════════
    //  UI HELPERS
    // ═══════════════════════════════════════════

    function setConnectionStatus(status) {
        const dot = document.getElementById('status-dot');
        const text = document.getElementById('status-text');

        dot.className = 'status-dot';

        switch (status) {
            case 'connected':
            case 'running':
                dot.classList.add('connected');
                text.textContent = 'Connected';
                break;
            case 'disconnected':
                dot.classList.add('disconnected');
                text.textContent = 'Disconnected';
                break;
            default:
                text.textContent = 'Connecting...';
        }
    }

    // Smooth number animation for KPI cards
    const kpiState = {};

    function animateValue(elementId, newValue) {
        const el = document.getElementById(elementId);
        if (!el) return;

        const current = kpiState[elementId] || 0;
        if (current === newValue) return;

        kpiState[elementId] = newValue;

        const start = current;
        const range = newValue - start;
        const duration = 500;
        const startTime = performance.now();

        function tick(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const value = Math.round(start + range * eased);

            el.textContent = value.toLocaleString();

            if (progress < 1) {
                requestAnimationFrame(tick);
            }
        }

        requestAnimationFrame(tick);
    }

    function formatTime(unixTimestamp) {
        const d = new Date(unixTimestamp * 1000);
        return d.toLocaleTimeString('en-GB', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }

    function formatUptime(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    function escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }


    // ═══════════════════════════════════════════
    //  INIT
    // ═══════════════════════════════════════════

    function init() {
        initCharts();
        connectWebSocket();
        initChat();

        // Clear log button
        document.getElementById('btn-clear-log').addEventListener('click', clearLog);

        // Modal close
        document.getElementById('modal-close').addEventListener('click', hideFeatureModal);
        document.getElementById('feature-modal-overlay').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) hideFeatureModal();
        });

        // Escape closes modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') hideFeatureModal();
        });
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // ═══════════════════════════════════════════
    //  AI CHAT
    // ═══════════════════════════════════════════

    let chatSessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    
    function initChat() {
        const fab = document.getElementById('chat-fab');
        const panel = document.getElementById('chat-panel');
        const closeBtn = document.getElementById('chat-close');
        const sendBtn = document.getElementById('chat-send');
        const input = document.getElementById('chat-input');
        
        fab.addEventListener('click', () => {
            panel.classList.toggle('active');
            if (panel.classList.contains('active')) {
                input.focus();
            }
        });
        
        closeBtn.addEventListener('click', () => {
            panel.classList.remove('active');
        });
        
        sendBtn.addEventListener('click', sendChatMessage);
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }
    
    async function sendChatMessage() {
        const input = document.getElementById('chat-input');
        const text = input.value.trim();
        if (!text) return;
        
        // Append user msg
        appendChatMsg('user', text);
        input.value = '';
        
        // Show indicator
        const indicatorId = appendTypingIndicator();
        
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, session_id: chatSessionId })
            });
            
            removeElement(indicatorId);
            
            if (!res.ok) {
                appendChatMsg('ai', 'Error: Server responded with ' + res.status);
                return;
            }
            
            const data = await res.json();
            appendChatMsg('ai', data.reply);
        } catch(err) {
            removeElement(indicatorId);
            appendChatMsg('ai', 'Error: Failed to connect to server.');
        }
    }
    
    function appendChatMsg(role, text) {
        const container = document.getElementById('chat-messages');
        const bubbleWrap = document.createElement('div');
        bubbleWrap.className = `chat-msg msg-${role}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        
        // Render markdown if available, else plain text with line breaks
        if (typeof marked !== 'undefined') {
            bubble.innerHTML = marked.parse(text);
        } else {
            bubble.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');
        }
        
        bubbleWrap.appendChild(bubble);
        container.appendChild(bubbleWrap);
        container.scrollTop = container.scrollHeight;
    }
    
    function appendTypingIndicator() {
        const container = document.getElementById('chat-messages');
        const id = 'typing_' + Date.now();
        
        const bubbleWrap = document.createElement('div');
        bubbleWrap.className = `chat-msg msg-ai`;
        bubbleWrap.id = id;
        
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        
        bubble.appendChild(indicator);
        bubbleWrap.appendChild(bubble);
        container.appendChild(bubbleWrap);
        container.scrollTop = container.scrollHeight;
        
        return id;
    }
    
    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

})();
