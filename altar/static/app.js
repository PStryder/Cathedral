// Cathedral Frontend - Vanilla JS + SSE

const state = {
    currentThreadUid: null,
    currentThreadName: null,
    isStreaming: false,
    consoleCollapsed: false,
    consoleLineCount: 0,
    maxConsoleLines: 200,
    toolsEnabled: false,
    contextInjectionEnabled: true,  // Context injection (RAG/memory context)
    isEditingThreadName: false,
    voiceEnabled: false,  // Voice output (TTS)
    voiceAvailable: false,  // Whether voice is available on server
    micActive: false,  // Whether microphone is currently active
    voiceConnecting: false,  // Whether PersonaPlex connection is in progress
    voiceConnected: false,  // Whether PersonaPlex connection is ready
    // Per-gate enablement
    enabledGates: {
        MemoryGate: false,
        ScriptureGate: false,
        FileSystemGate: false,
        ShellGate: false,
        BrowserGate: false,
        SubAgentGate: false
    }
};

// Event source and connection state
let eventSource = null;
let sseConnected = false;
let lastPollTimestamp = 0;

// DOM Elements
const elements = {
    threadList: document.getElementById('threadList'),
    newThreadBtn: document.getElementById('newThreadBtn'),
    messagesContainer: document.getElementById('messagesContainer'),
    messagesList: document.getElementById('messagesList'),
    emptyState: document.getElementById('emptyState'),
    messageInput: document.getElementById('messageInput'),
    sendBtn: document.getElementById('sendBtn'),
    currentThreadName: document.getElementById('currentThreadName'),
    currentThreadId: document.getElementById('currentThreadId'),
    streamingStatus: document.getElementById('streamingStatus'),
    // Console panel
    consolePanel: document.getElementById('consolePanel'),
    consoleOutput: document.getElementById('consoleOutput'),
    consoleCount: document.getElementById('consoleCount'),
    consoleToggle: document.getElementById('consoleToggle'),
    // Status indicators
    connectionStatus: document.getElementById('connectionStatus'),
    connectionText: document.getElementById('connectionText'),
    // Gate toggles
    gateMemory: document.getElementById('gateMemory'),
    gateScripture: document.getElementById('gateScripture'),
    gateFilesystem: document.getElementById('gateFilesystem'),
    gateShell: document.getElementById('gateShell'),
    gateBrowser: document.getElementById('gateBrowser'),
    gateSubAgent: document.getElementById('gateSubAgent'),
    // Context toggle
    contextToggle: document.getElementById('contextToggle'),
    contextLabel: document.getElementById('contextLabel'),
    // Thread name editing
    threadNameInput: document.getElementById('threadNameInput'),
    // Voice toggle
    voiceToggle: document.getElementById('voiceToggle'),
    voiceLabel: document.getElementById('voiceLabel'),
    ttsAudio: document.getElementById('ttsAudio'),
    // Microphone button
    micButton: document.getElementById('micButton'),
    micIcon: document.getElementById('micIcon'),
    micLabel: document.getElementById('micLabel')
};

// ========== Thread Management ==========

async function loadThreads() {
    try {
        const response = await fetch('/api/threads');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        renderThreadList(data.threads || []);
    } catch (error) {
        console.error('Failed to load threads:', error);
    }
}

function renderThreadList(threads) {
    elements.threadList.innerHTML = '';

    if (threads.length === 0) {
        elements.threadList.innerHTML = `
            <div class="px-3 py-4 text-sm text-cathedral-muted text-center">
                No threads yet
            </div>
        `;
        return;
    }

    threads.forEach(thread => {
        const uid = thread.thread_uid;
        const name = thread.thread_name || 'Unnamed Thread';
        const div = document.createElement('div');
        div.className = `thread-item px-3 py-2 rounded-lg cursor-pointer ${
            uid === state.currentThreadUid ? 'active' : ''
        }`;
        div.dataset.uid = uid;
        div.innerHTML = `
            <div class="text-sm font-medium truncate">${escapeHtml(name)}</div>
            <div class="text-xs text-cathedral-muted mono truncate">${uid.slice(0, 8)}...</div>
        `;
        div.onclick = () => switchThread(uid, name);
        elements.threadList.appendChild(div);
    });
}

async function createNewThread() {
    try {
        Console.info('Creating new thread...');
        const response = await fetch('/api/thread', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_name: null })
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();

        if (data.thread_uid) {
            Console.success(`Thread created: ${data.thread_uid.slice(0, 8)}...`);
            await loadThreads();
            await switchThread(data.thread_uid, 'New Thread');
        }
    } catch (error) {
        console.error('Failed to create thread:', error);
        Console.error(`Failed to create thread: ${error.message}`);
    }
}

async function switchThread(uid, name) {
    // Close any active voice conversation when switching threads
    if (state.micActive) {
        closeVoiceConversation();
        state.micActive = false;
        updateMicButtonUI(false);
    }

    state.currentThreadUid = uid;
    state.currentThreadName = name || 'Unnamed Thread';

    // Update UI
    elements.currentThreadName.textContent = state.currentThreadName;
    elements.currentThreadId.textContent = uid;
    elements.messageInput.disabled = false;
    elements.sendBtn.disabled = false;

    // Update thread list highlighting
    document.querySelectorAll('.thread-item').forEach(item => {
        item.classList.toggle('active', item.dataset.uid === uid);
    });

    // Update mic button state (now that thread is selected)
    updateMicButtonState();

    // If voice is enabled but not connected, pre-connect now that we have a thread
    if (state.voiceEnabled && !state.voiceConnected && !state.voiceConnecting) {
        state.voiceConnecting = true;
        updateMicButtonState();
        preConnectVoice(uid);
    }

    // Load history
    await loadThreadHistory(uid);
}

async function loadThreadHistory(uid) {
    try {
        const response = await fetch(`/api/thread/${uid}/history`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        renderMessages(data.history || []);
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// ========== Message Rendering ==========

function renderMessages(messages) {
    elements.emptyState.classList.add('hidden');
    elements.messagesList.classList.remove('hidden');
    elements.messagesList.innerHTML = '';

    if (messages.length === 0) {
        elements.messagesList.innerHTML = `
            <div class="text-center text-cathedral-muted py-8">
                Start a conversation...
            </div>
        `;
        return;
    }

    messages.forEach(msg => {
        appendMessage(msg.role, msg.content, false);
    });

    scrollToBottom();
}

function appendMessage(role, content, animate = true) {
    // Remove empty state placeholder if present
    const placeholder = elements.messagesList.querySelector('.text-center.text-cathedral-muted');
    if (placeholder) placeholder.remove();

    const div = document.createElement('div');
    div.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'} ${animate ? 'message-enter' : ''}`;

    const bubbleClass = role === 'user'
        ? 'bg-cathedral-user border-blue-800/30'
        : 'bg-cathedral-assistant border-cathedral-border';

    div.innerHTML = `
        <div class="max-w-2xl px-4 py-3 rounded-2xl border ${bubbleClass}">
            <div class="text-xs text-cathedral-muted mb-1 font-medium">
                ${role === 'user' ? 'You' : 'Cathedral'}
            </div>
            <div class="text-sm whitespace-pre-wrap message-content">${escapeHtml(content)}</div>
        </div>
    `;

    elements.messagesList.appendChild(div);
    return div;
}

function createStreamingBubble() {
    const div = document.createElement('div');
    div.className = 'flex justify-start message-enter';
    div.id = 'streaming-bubble';

    div.innerHTML = `
        <div class="max-w-2xl px-4 py-3 rounded-2xl border bg-cathedral-assistant border-cathedral-border">
            <div class="text-xs text-cathedral-muted mb-1 font-medium">Cathedral</div>
            <div class="text-sm whitespace-pre-wrap message-content"></div>
        </div>
    `;

    elements.messagesList.appendChild(div);
    return div.querySelector('.message-content');
}

function scrollToBottom() {
    elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

// ========== Tool Execution Display ==========

// Track active tool calls for updating status
const activeToolCalls = new Map();

function appendToolCallIndicator(toolName) {
    Console.info(`appendToolCallIndicator called for: ${toolName}`);

    const div = document.createElement('div');
    div.className = 'flex justify-center my-2';
    div.dataset.toolCall = toolName;

    const id = `tool-${Date.now()}-${toolName.replace(/[^a-zA-Z0-9]/g, '_')}`;
    div.id = id;

    div.innerHTML = `
        <div style="background: rgba(220, 38, 38, 0.3); border: 2px solid #dc2626; border-radius: 9999px; padding: 6px 12px; display: inline-flex; align-items: center; gap: 8px;">
            <span style="width: 8px; height: 8px; background: #dc2626; border-radius: 50%; animation: pulse 1s infinite;"></span>
            <span style="color: #fca5a5; font-size: 12px; font-family: monospace;">${escapeHtml(toolName)}</span>
        </div>
    `;

    // Insert before streaming bubble if it exists, otherwise append
    const streamingBubble = document.getElementById('streaming-bubble');
    Console.info(`streamingBubble exists: ${!!streamingBubble}, messagesList exists: ${!!elements.messagesList}`);

    if (streamingBubble && elements.messagesList) {
        elements.messagesList.insertBefore(div, streamingBubble);
        Console.success(`Indicator inserted before streaming bubble`);
    } else if (elements.messagesList) {
        elements.messagesList.appendChild(div);
        Console.success(`Indicator appended to messagesList`);
    } else {
        Console.error(`Cannot insert indicator - no messagesList!`);
        return null;
    }

    scrollToBottom();
    activeToolCalls.set(toolName, { id, startTime: Date.now() });

    return div;
}

function updateToolCallIndicator(toolName, success, elapsedMs) {
    const call = activeToolCalls.get(toolName);
    if (!call) return;

    const div = document.getElementById(call.id);
    if (!div) return;

    const elapsed = elapsedMs || (Date.now() - call.startTime);
    const elapsedStr = elapsed >= 1000 ? `${(elapsed / 1000).toFixed(1)}s` : `${elapsed}ms`;

    if (success) {
        div.innerHTML = `
            <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-900/40 border border-emerald-500/50">
                <span class="text-emerald-400">✓</span>
                <span class="text-xs font-medium text-emerald-300 mono">${escapeHtml(toolName)}</span>
                <span class="text-xs text-emerald-400/70">${elapsedStr}</span>
            </div>
        `;
    } else {
        div.innerHTML = `
            <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-900/40 border border-red-500/50">
                <span class="text-red-400">✗</span>
                <span class="text-xs font-medium text-red-300 mono">${escapeHtml(toolName)}</span>
                <span class="text-xs text-red-400/70">${elapsedStr}</span>
            </div>
        `;
    }

    // Remove from tracking
    activeToolCalls.delete(toolName);
}

function appendToolExecution(toolName, args, status = 'calling') {
    const div = document.createElement('div');
    div.className = 'flex justify-start message-enter';
    div.dataset.toolExecution = 'true';

    const statusBadge = status === 'calling'
        ? '<span class="tool-call-badge px-2 py-0.5 rounded text-xs font-medium">Calling...</span>'
        : status === 'success'
        ? '<span class="tool-result-badge px-2 py-0.5 rounded text-xs font-medium">Success</span>'
        : '<span class="tool-error-badge px-2 py-0.5 rounded text-xs font-medium">Failed</span>';

    const argsDisplay = args && Object.keys(args).length > 0
        ? `<div class="mt-2 text-xs text-emerald-300/70 mono">
             ${Object.entries(args).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(', ')}
           </div>`
        : '';

    div.innerHTML = `
        <div class="max-w-2xl px-4 py-3 rounded-2xl tool-execution">
            <div class="flex items-center justify-between mb-1">
                <div class="flex items-center gap-2">
                    <span class="text-emerald-400">&#128295;</span>
                    <span class="text-xs font-medium text-emerald-300">Tool Execution</span>
                </div>
                ${statusBadge}
            </div>
            <div class="mono text-sm text-emerald-200">${escapeHtml(toolName)}</div>
            ${argsDisplay}
        </div>
    `;

    elements.messagesList.appendChild(div);
    scrollToBottom();
    return div;
}

function appendToolResult(toolName, result, success = true) {
    const div = document.createElement('div');
    div.className = 'flex justify-start message-enter';
    div.dataset.toolResult = 'true';

    const badge = success
        ? '<span class="tool-result-badge px-2 py-0.5 rounded text-xs font-medium">Result</span>'
        : '<span class="tool-error-badge px-2 py-0.5 rounded text-xs font-medium">Error</span>';

    const resultText = typeof result === 'object' ? JSON.stringify(result, null, 2) : result;
    const truncated = resultText.length > 500 ? resultText.slice(0, 500) + '...' : resultText;

    div.innerHTML = `
        <div class="max-w-2xl px-4 py-3 rounded-2xl tool-execution">
            <div class="flex items-center justify-between mb-1">
                <div class="flex items-center gap-2">
                    <span class="text-emerald-400">&#128295;</span>
                    <span class="text-xs font-medium text-emerald-300">${escapeHtml(toolName)}</span>
                </div>
                ${badge}
            </div>
            <pre class="mono text-xs text-emerald-200/80 whitespace-pre-wrap max-h-32 overflow-y-auto">${escapeHtml(truncated)}</pre>
        </div>
    `;

    elements.messagesList.appendChild(div);
    scrollToBottom();
    return div;
}

// ========== Chat & Streaming ==========

async function sendMessage() {
    const text = elements.messageInput.value.trim();
    if (!text || !state.currentThreadUid || state.isStreaming) return;

    // Check if it's a command
    const isCommand = text.startsWith('/');
    if (isCommand) {
        Console.info(`Command: ${text.split(' ')[0]}`);
    }

    // Add user message to UI
    appendMessage('user', text);
    elements.messageInput.value = '';
    autoResizeTextarea();
    scrollToBottom();

    // Start streaming
    state.isStreaming = true;
    elements.streamingStatus.classList.remove('hidden');
    elements.messageInput.disabled = true;
    elements.sendBtn.disabled = true;

    // Create streaming bubble
    const contentEl = createStreamingBubble();
    let fullResponse = '';
    const startTime = Date.now();

    // Initialize voice stream if enabled
    if (state.voiceEnabled && state.voiceAvailable) {
        initVoiceStream(state.currentThreadUid);
    }

    try {
        // Use fetch with POST for SSE (EventSource only supports GET)
        // Build list of enabled gates
        const enabledGates = Object.entries(state.enabledGates)
            .filter(([_, enabled]) => enabled)
            .map(([gate, _]) => gate);
        const anyToolsEnabled = enabledGates.length > 0;

        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_input: text,
                thread_uid: state.currentThreadUid,
                enable_tools: anyToolsEnabled,
                enabled_gates: enabledGates,
                enable_context: state.contextInjectionEnabled,
                enable_voice: state.voiceEnabled && state.voiceAvailable
            })
        });

        // Log enabled gates
        if (anyToolsEnabled) {
            Console.tool(`Tools enabled: ${enabledGates.join(', ')}`);
        }

        if (!response.ok) {
            // Check if locked
            if (response.status === 401) {
                const data = await response.json();
                if (data.locked) {
                    Console.security('Session locked - redirecting to unlock');
                    window.location.href = '/lock?redirect=/';
                    return;
                }
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let tokenCount = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Parse SSE events from buffer
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.token) {
                            const token = data.token;

                            // Check for tool markers
                            if (token.includes('[[TOOL:')) {
                                Console.info(`MARKER FOUND: ${token.trim()}`);

                                // Parse and show indicators
                                const markers = token.matchAll(/\[\[TOOL:(START|OK|ERROR):([^\]]+)\]\]/g);
                                for (const m of markers) {
                                    const [, status, toolName] = m;
                                    Console.info(`Parsed: status=${status} tool=${toolName}`);
                                    if (status === 'START') {
                                        appendToolCallIndicator(toolName);
                                    } else {
                                        updateToolCallIndicator(toolName, status === 'OK', 0);
                                    }
                                }

                                // Strip markers, keep other text
                                const clean = token.replace(/\[\[TOOL:[^\]]+\]\]\n?/g, '');
                                if (clean.trim()) {
                                    fullResponse += clean;
                                    contentEl.textContent = fullResponse;
                                    scrollToBottom();
                                }
                            } else {
                                fullResponse += token;
                                contentEl.textContent = fullResponse;
                                scrollToBottom();
                            }
                            tokenCount++;
                        }
                        if (data.done) {
                            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
                            Console.success(`Response complete: ${tokenCount} tokens in ${elapsed}s`);
                        }
                        if (data.error) {
                            contentEl.textContent = `Error: ${data.error}`;
                            Console.error(data.error);
                        }
                    } catch (e) {
                        // Ignore parse errors for incomplete JSON
                    }
                }
            }
        }
    } catch (error) {
        console.error('Streaming error:', error);
        contentEl.textContent = `Error: ${error.message}`;
        Console.error(`Stream error: ${error.message}`);
    } finally {
        state.isStreaming = false;
        elements.streamingStatus.classList.add('hidden');
        elements.messageInput.disabled = false;
        elements.sendBtn.disabled = false;
        elements.messageInput.focus();

        // Remove the streaming bubble ID so it becomes a normal message
        const bubble = document.getElementById('streaming-bubble');
        if (bubble) bubble.removeAttribute('id');
    }
}

// ========== Input Handling ==========

function autoResizeTextarea() {
    const textarea = elements.messageInput;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// ========== Utilities ==========

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ========== Console Panel ==========

function toggleConsole() {
    state.consoleCollapsed = !state.consoleCollapsed;
    if (state.consoleCollapsed) {
        elements.consolePanel.classList.add('collapsed');
        elements.consoleToggle.textContent = '\u25B6'; // right arrow
    } else {
        elements.consolePanel.classList.remove('collapsed');
        elements.consoleToggle.textContent = '\u25BC'; // down arrow
    }
}

function clearConsole() {
    elements.consoleOutput.innerHTML = '';
    state.consoleLineCount = 0;
    updateConsoleCount();
    consolePrint('Console cleared', 'info');
}

function updateConsoleCount() {
    elements.consoleCount.textContent = `(${state.consoleLineCount})`;
}

function consolePrint(message, type = 'info', prefix = null) {
    const line = document.createElement('div');
    line.className = `console-line ${type}`;

    const timestamp = new Date().toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });

    const prefixStr = prefix ? `[${prefix}] ` : '';
    line.innerHTML = `<span class="text-cathedral-muted/50">${timestamp}</span> ${prefixStr}${escapeHtml(message)}`;

    elements.consoleOutput.appendChild(line);
    state.consoleLineCount++;

    // Limit lines
    while (elements.consoleOutput.children.length > state.maxConsoleLines) {
        elements.consoleOutput.removeChild(elements.consoleOutput.firstChild);
        state.consoleLineCount--;
    }

    updateConsoleCount();

    // Auto-scroll if near bottom
    const output = elements.consoleOutput;
    const isNearBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 50;
    if (isNearBottom) {
        output.scrollTop = output.scrollHeight;
    }

    // Expand console if collapsed and important message
    if (state.consoleCollapsed && (type === 'error' || type === 'warning' || type === 'agent')) {
        toggleConsole();
    }
}

// Console API for different event types
const Console = {
    info: (msg) => consolePrint(msg, 'info'),
    success: (msg) => consolePrint(msg, 'success'),
    warning: (msg) => consolePrint(msg, 'warning'),
    error: (msg) => consolePrint(msg, 'error'),
    agent: (id, msg) => consolePrint(msg, 'agent', `agent:${id}`),
    memory: (msg) => consolePrint(msg, 'memory', 'memory'),
    security: (msg) => consolePrint(msg, 'security', 'security'),
    system: (msg) => consolePrint(msg, 'info', 'system'),
    tool: (msg) => consolePrint(msg, 'tool', 'tool')
};

// Make Console globally accessible
window.Console = Console;
window.toggleConsole = toggleConsole;
window.clearConsole = clearConsole;

// ========== Event Source for System Events ==========

function connectEventSource() {
    // Connect to server-sent events for system notifications
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource('/api/events');

    eventSource.onopen = () => {
        Console.system('Connected to event stream');
        elements.connectionStatus.className = 'w-2 h-2 bg-green-500 rounded-full';
        elements.connectionText.textContent = 'Connected';
        sseConnected = true;
    };

    eventSource.onerror = () => {
        elements.connectionStatus.className = 'w-2 h-2 bg-red-500 rounded-full';
        elements.connectionText.textContent = 'Disconnected';
        sseConnected = false;
        // Attempt reconnect after 5s
        setTimeout(connectEventSource, 5000);
    };

    eventSource.addEventListener('agent', (e) => {
        const data = JSON.parse(e.data);
        Console.agent(data.id, data.message);
    });

    eventSource.addEventListener('memory', (e) => {
        const data = JSON.parse(e.data);
        Console.memory(data.message);
    });

    eventSource.addEventListener('security', (e) => {
        const data = JSON.parse(e.data);
        Console.security(data.message);
    });

    eventSource.addEventListener('system', (e) => {
        const data = JSON.parse(e.data);
        Console.system(data.message);
    });

    eventSource.addEventListener('tool', (e) => {
        const data = JSON.parse(e.data);
        const msg = data.message || '';
        // Just log to console panel - indicators now come through stream
        Console.tool(msg);
    });

    // Ignore ping events (keepalive)
    eventSource.addEventListener('ping', () => {});
}

// Poll for agent status updates (fallback if SSE not available)
async function pollAgentStatus() {
    // Skip polling if SSE is connected
    if (sseConnected) return;

    try {
        const res = await fetch(`/api/agents/status?since=${lastPollTimestamp}`);
        if (res.ok) {
            const data = await res.json();
            if (data.updates && data.updates.length > 0) {
                data.updates.forEach(update => {
                    Console.agent(update.id, update.message);
                });
            }
            lastPollTimestamp = data.timestamp || Date.now() / 1000;
        }
    } catch (e) {
        // Silently ignore polling errors
    }
}

// ========== Event Listeners ==========

elements.newThreadBtn.addEventListener('click', createNewThread);
elements.sendBtn.addEventListener('click', sendMessage);
elements.messageInput.addEventListener('keydown', handleKeyDown);
elements.messageInput.addEventListener('input', autoResizeTextarea);

// Gate toggles
const gateElements = [
    elements.gateMemory,
    elements.gateScripture,
    elements.gateFilesystem,
    elements.gateShell,
    elements.gateBrowser,
    elements.gateSubAgent
];

gateElements.forEach(el => {
    if (el) {
        el.addEventListener('change', (e) => {
            const gate = e.target.dataset.gate;
            state.enabledGates[gate] = e.target.checked;

            // Update label style
            const label = e.target.closest('label').querySelector('.gate-label');
            if (label) {
                label.className = e.target.checked
                    ? 'gate-label font-medium'
                    : 'gate-label';
            }

            // Count enabled gates
            const enabledCount = Object.values(state.enabledGates).filter(v => v).length;
            Console.tool(e.target.checked
                ? `${gate} enabled (${enabledCount} gates active)`
                : `${gate} disabled (${enabledCount} gates active)`);
        });
    }
});

// Context toggle
if (elements.contextToggle) {
    elements.contextToggle.addEventListener('change', (e) => {
        state.contextInjectionEnabled = e.target.checked;
        if (elements.contextLabel) {
            elements.contextLabel.className = e.target.checked ? 'font-medium' : '';
        }
        Console.info(e.target.checked
            ? 'Context injection enabled'
            : 'Context injection disabled');
    });
}

// Voice toggle - pre-connects to PersonaPlex when enabled
if (elements.voiceToggle) {
    elements.voiceToggle.addEventListener('change', async (e) => {
        state.voiceEnabled = e.target.checked;
        if (elements.voiceLabel) {
            elements.voiceLabel.className = e.target.checked ? 'voice-label font-medium' : 'voice-label';
        }

        if (e.target.checked) {
            // Voice enabled - pre-connect to PersonaPlex
            Console.info('Voice enabled - connecting to PersonaPlex...');
            state.voiceConnecting = true;
            state.voiceConnected = false;
            updateMicButtonState();

            // Start pre-connection (don't await - let it happen in background)
            preConnectVoice(state.currentThreadUid);
        } else {
            // Voice disabled - close everything
            Console.info('Voice output disabled');
            closeVoiceConversation();
            closeVoiceStream();
            state.voiceConnecting = false;
            state.voiceConnected = false;
        }

        // Update mic button state
        updateMicButtonState();
    });
}

// Microphone button click handler - controls WebSocket lifecycle directly
// OFF → WS closed, ON → WS open + stream active
if (elements.micButton) {
    elements.micButton.addEventListener('click', async () => {
        if (!state.currentThreadUid || !state.voiceAvailable) return;

        // Don't allow click while connecting
        if (state.voiceConnecting) {
            Console.info('Please wait for PersonaPlex connection...');
            return;
        }

        if (state.micActive) {
            // === MIC OFF: Stop audio capture (keep connection for next use) ===
            Console.info('Stopping microphone...');
            stopAudioCapture();
            state.micActive = false;
            updateMicButtonState();
            Console.info('Microphone stopped');
        } else {
            // === MIC ON: Start audio capture ===
            if (!state.voiceConnected) {
                Console.error('Voice not connected - enable voice toggle first');
                return;
            }

            // Request microphone access if needed
            if (!mediaStream) {
                try {
                    mediaStream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            sampleRate: 24000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true,
                            autoGainControl: true,
                        }
                    });
                } catch (e) {
                    Console.error(`Microphone access denied: ${e.message}`);
                    return;
                }
            }

            state.micActive = true;
            startAudioCapture();
            updateMicButtonState();
            playActivationCue();
            Console.success('Voice conversation started - speak into microphone');
        }
    });
}

/**
 * Play a subtle audio cue when mic is activated.
 * Creates a soft "click" using Web Audio API synthesis.
 */
function playActivationCue() {
    try {
        const ctx = initAudioContext();

        // Create a soft click: quick sine burst with fast decay
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, ctx.currentTime);  // A5
        osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.05);

        gain.gain.setValueAtTime(0.15, ctx.currentTime);  // Subtle volume
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.08);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.1);
    } catch (e) {
        // Silently ignore audio cue failures
    }
}

function updateMicButtonState() {
    if (!elements.micButton) return;

    // Enable mic button only when: voice available AND connected AND thread selected
    const canUseMic = state.voiceAvailable && state.voiceConnected && state.currentThreadUid;
    elements.micButton.disabled = !canUseMic && !state.voiceConnecting;

    if (state.voiceConnecting) {
        // Show connecting state
        updateMicButtonUI('connecting');
    } else if (!canUseMic) {
        // Reset to inactive state if disabled
        if (state.micActive) {
            closeVoiceConversation();
            state.micActive = false;
        }
        updateMicButtonUI(false);
    } else if (state.micActive) {
        updateMicButtonUI(true);
    } else {
        // Ready but not active
        updateMicButtonUI('ready');
    }
}

function updateMicButtonUI(mode) {
    if (!elements.micButton) return;

    if (mode === true || mode === 'active') {
        // Active/recording state - red pulsing
        elements.micButton.className = 'flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs transition-colors bg-red-600 hover:bg-red-700 text-white mic-recording';
        if (elements.micIcon) elements.micIcon.innerHTML = '&#128308;'; // Red circle
        if (elements.micLabel) elements.micLabel.textContent = 'Listening...';
        elements.micButton.title = 'Click to stop voice conversation';
    } else if (mode === 'connecting') {
        // Connecting state - yellow/amber
        elements.micButton.className = 'flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs transition-colors bg-yellow-600 text-white cursor-wait';
        if (elements.micIcon) elements.micIcon.innerHTML = '&#8987;'; // Hourglass
        if (elements.micLabel) elements.micLabel.textContent = 'Connecting...';
        elements.micButton.title = 'Connecting to PersonaPlex...';
    } else if (mode === 'ready') {
        // Ready state - green tint to show it's ready
        elements.micButton.className = 'flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs transition-colors bg-green-700 hover:bg-green-600 text-white';
        if (elements.micIcon) elements.micIcon.innerHTML = '&#127908;'; // Microphone
        if (elements.micLabel) elements.micLabel.textContent = 'Mic Ready';
        elements.micButton.title = 'Click to start voice conversation';
    } else {
        // Inactive/disabled state
        elements.micButton.className = 'flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs transition-colors bg-gray-700 hover:bg-gray-600 text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed';
        if (elements.micIcon) elements.micIcon.innerHTML = '&#127908;'; // Microphone
        if (elements.micLabel) elements.micLabel.textContent = 'Mic';
        elements.micButton.title = 'Enable voice to use microphone';
    }
}

// Thread name editing
elements.currentThreadName.addEventListener('click', startEditingThreadName);
elements.threadNameInput.addEventListener('blur', finishEditingThreadName);
elements.threadNameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        finishEditingThreadName();
    } else if (e.key === 'Escape') {
        cancelEditingThreadName();
    }
});

function startEditingThreadName() {
    if (!state.currentThreadUid || state.isEditingThreadName) return;

    state.isEditingThreadName = true;
    elements.currentThreadName.classList.add('hidden');
    elements.threadNameInput.classList.remove('hidden');
    elements.threadNameInput.value = state.currentThreadName || '';
    elements.threadNameInput.focus();
    elements.threadNameInput.select();
}

function cancelEditingThreadName() {
    state.isEditingThreadName = false;
    elements.threadNameInput.classList.add('hidden');
    elements.currentThreadName.classList.remove('hidden');
}

async function finishEditingThreadName() {
    if (!state.isEditingThreadName) return;

    const newName = elements.threadNameInput.value.trim();
    state.isEditingThreadName = false;
    elements.threadNameInput.classList.add('hidden');
    elements.currentThreadName.classList.remove('hidden');

    // Skip if name unchanged or empty
    if (!newName || newName === state.currentThreadName) return;

    try {
        const response = await fetch(`/api/thread/${state.currentThreadUid}/rename`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ thread_name: newName })
        });

        if (response.ok) {
            state.currentThreadName = newName;
            elements.currentThreadName.textContent = newName;
            Console.success(`Thread renamed to "${newName}"`);
            // Refresh thread list to show new name in sidebar
            await loadThreads();
        } else {
            Console.error('Failed to rename thread');
        }
    } catch (error) {
        console.error('Failed to rename thread:', error);
        Console.error(`Rename failed: ${error.message}`);
    }
}

// ========== Voice / TTS ==========

// Audio playback state
let audioContext = null;
let audioQueue = [];
let isPlayingAudio = false;
let voiceSocket = null;

// Gapless playback scheduling
let nextPlaybackTime = 0;          // When next chunk should start
const MIN_BUFFER_CHUNKS = 3;       // Buffer this many chunks before starting
const SCHEDULE_AHEAD_TIME = 0.1;   // Schedule 100ms ahead

function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 24000
        });
    }
    // Resume if suspended (browser policy)
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
    return audioContext;
}

async function checkVoiceStatus() {
    try {
        const res = await fetch('/api/voice/status');
        if (res.ok) {
            const status = await res.json();
            state.voiceAvailable = status.available;

            if (elements.voiceToggle) {
                elements.voiceToggle.disabled = !status.available;
                if (!status.available) {
                    elements.voiceToggle.title = status.error || 'Voice not available';
                }
            }

            if (status.available) {
                Console.info(`Voice available: ${status.provider}`);
                if (status.personaplex_connected) {
                    Console.info('PersonaPlex connected - full-duplex mode available');
                }
            } else {
                Console.info(`Voice unavailable: ${status.error || 'TTS disabled'}`);
            }
        }

        // Also check codec info
        const codecRes = await fetch('/api/voice/codec/info');
        if (codecRes.ok) {
            const codec = await codecRes.json();
            if (codec.available) {
                Console.info(`Opus codec: ${codec.sample_rate}Hz, ${codec.frame_duration_ms}ms frames`);
            }
        }
    } catch (e) {
        // Voice endpoint may not exist
        state.voiceAvailable = false;
    }

    // Update mic button state after voice status is determined
    updateMicButtonState();
}

function initVoiceStream(threadId) {
    if (!state.voiceEnabled || !state.voiceAvailable) return;

    // Close existing connection
    if (voiceSocket) {
        voiceSocket.close();
        voiceSocket = null;
    }

    // Reset audio state
    audioQueue = [];
    isPlayingAudio = false;
    nextPlaybackTime = 0;

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    voiceSocket = new WebSocket(`${protocol}//${location.host}/api/voice/${threadId}`);
    voiceSocket.binaryType = 'arraybuffer';

    let currentMetadata = null;

    voiceSocket.onopen = () => {
        Console.info('Voice stream connected');
    };

    voiceSocket.onmessage = async (event) => {
        if (typeof event.data === 'string') {
            // JSON metadata
            currentMetadata = JSON.parse(event.data);
            if (currentMetadata.error) {
                Console.warning(`Voice error: ${currentMetadata.error}`);
                return;
            }
        } else {
            // Binary audio data
            if (event.data.byteLength > 0) {
                audioQueue.push({
                    data: event.data,
                    metadata: currentMetadata
                });

                if (!isPlayingAudio) {
                    playNextAudioChunk();
                }
            }

            // Check for final chunk
            if (currentMetadata && currentMetadata.is_final) {
                Console.info('Voice stream complete');
            }
        }
    };

    voiceSocket.onerror = (error) => {
        Console.error('Voice stream error');
    };

    voiceSocket.onclose = () => {
        Console.info('Voice stream closed');
        voiceSocket = null;
    };
}

/**
 * Schedule audio chunks for gapless playback.
 * Uses Web Audio API scheduling to avoid gaps between chunks.
 */
function scheduleAudioPlayback() {
    if (audioQueue.length === 0) {
        Console.info('scheduleAudioPlayback: queue empty');
        return;
    }

    const ctx = initAudioContext();
    Console.info(`AudioContext state: ${ctx.state}, sampleRate: ${ctx.sampleRate}`);
    const currentTime = ctx.currentTime;

    // If we fell behind (gap in audio), reset scheduling
    if (nextPlaybackTime < currentTime) {
        nextPlaybackTime = currentTime + 0.01; // Small buffer
    }

    let scheduledCount = 0;
    // Schedule all queued chunks
    while (audioQueue.length > 0) {
        const { data, metadata } = audioQueue.shift();

        try {
            const sampleRate = metadata?.sample_rate || 24000;
            const int16Array = new Int16Array(data);
            const float32Array = new Float32Array(int16Array.length);

            // Check if data is non-zero
            let maxVal = 0;
            for (let i = 0; i < Math.min(100, int16Array.length); i++) {
                maxVal = Math.max(maxVal, Math.abs(int16Array[i]));
            }
            Console.info(`Scheduling chunk: ${int16Array.length} samples at ${sampleRate}Hz, maxVal=${maxVal}`);

            // Convert int16 to float32
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }

            // Create audio buffer
            const audioBuffer = ctx.createBuffer(1, float32Array.length, sampleRate);
            audioBuffer.getChannelData(0).set(float32Array);

            // Schedule the buffer at precise time
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);

            // Play immediately for first chunk, then schedule subsequent ones
            if (scheduledCount === 0) {
                source.start(0);  // Play immediately
                nextPlaybackTime = ctx.currentTime + (audioBuffer.length / sampleRate);
            } else {
                source.start(nextPlaybackTime);
                nextPlaybackTime += audioBuffer.length / sampleRate;
            }
            scheduledCount++;

        } catch (e) {
            Console.error(`Audio schedule error: ${e.message}`);
        }
    }
    Console.info(`Scheduled ${scheduledCount} audio chunks, next play at ${nextPlaybackTime.toFixed(3)}s`);
}

async function playNextAudioChunk() {
    // Wait for minimum buffer before starting playback
    if (!isPlayingAudio && audioQueue.length < MIN_BUFFER_CHUNKS) {
        Console.info(`Buffering audio: ${audioQueue.length}/${MIN_BUFFER_CHUNKS} chunks`);
        return; // Wait for more chunks
    }

    Console.info(`Starting audio playback with ${audioQueue.length} chunks queued`);
    isPlayingAudio = true;
    scheduleAudioPlayback();
}

function closeVoiceStream() {
    if (voiceSocket) {
        voiceSocket.close();
        voiceSocket = null;
    }
    audioQueue = [];
    isPlayingAudio = false;
    nextPlaybackTime = 0;
}

// ========== Full-Duplex Voice Conversation ==========

let conversationSocket = null;
let mediaStream = null;
let audioRecorder = null;
let isRecording = false;
let currentGenerationId = null;

/**
 * Pre-connect to PersonaPlex when voice is enabled.
 * This establishes the WebSocket connection in advance so the mic button
 * is immediately responsive (PersonaPlex handshake takes 30+ seconds).
 */
async function preConnectVoice(threadId) {
    if (!threadId) {
        Console.info('Select a thread to enable voice');
        state.voiceConnecting = false;
        updateMicButtonState();
        return;
    }
    if (!state.voiceAvailable) {
        Console.error('Voice service not available');
        state.voiceConnecting = false;
        updateMicButtonState();
        return;
    }

    // Close existing connection
    if (conversationSocket) {
        conversationSocket.close();
        conversationSocket = null;
    }

    // Reset audio state
    audioQueue = [];
    isPlayingAudio = false;
    nextPlaybackTime = 0;
    currentGenerationId = null;

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    conversationSocket = new WebSocket(`${protocol}//${location.host}/api/voice/conversation/${threadId}`);
    conversationSocket.binaryType = 'arraybuffer';

    conversationSocket.onopen = () => {
        Console.info('Voice WebSocket opened, waiting for PersonaPlex handshake...');
    };

    conversationSocket.onmessage = async (event) => {
        if (typeof event.data === 'string') {
            const data = JSON.parse(event.data);
            handleConversationMessage(data);
        } else {
            // Binary audio data from agent
            if (event.data.byteLength > 0) {
                audioQueue.push({
                    data: event.data,
                    metadata: { sample_rate: 24000, generation_id: currentGenerationId }
                });
                if (!isPlayingAudio) {
                    playNextAudioChunk();
                }
            }
        }
    };

    conversationSocket.onerror = (error) => {
        Console.error('Voice connection error');
        state.voiceConnecting = false;
        state.voiceConnected = false;
        updateMicButtonState();
    };

    conversationSocket.onclose = () => {
        Console.info('Voice connection closed');
        stopAudioCapture();
        conversationSocket = null;
        state.voiceConnecting = false;
        state.voiceConnected = false;
        if (state.micActive) {
            state.micActive = false;
            updateMicButtonUI(false);
        }
        updateMicButtonState();
    };
}

/**
 * Initialize full-duplex voice conversation with PersonaPlex.
 * This enables bidirectional audio streaming with interrupt support.
 */
async function initVoiceConversation(threadId) {
    if (!state.voiceEnabled || !state.voiceAvailable) return false;

    // If not pre-connected, connect now
    if (!conversationSocket || conversationSocket.readyState !== WebSocket.OPEN) {
        await preConnectVoice(threadId);
        // Wait for connection (with timeout)
        for (let i = 0; i < 600; i++) {  // 60 second timeout
            if (state.voiceConnected) break;
            await new Promise(r => setTimeout(r, 100));
        }
        if (!state.voiceConnected) {
            Console.error('Voice connection timeout');
            return false;
        }
    }

    // Request microphone access
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 24000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            }
        });
    } catch (e) {
        Console.error(`Microphone access denied: ${e.message}`);
        return false;
    }

    // Close existing connection
    if (conversationSocket) {
        conversationSocket.close();
        conversationSocket = null;
    }

    // Reset audio state
    audioQueue = [];
    isPlayingAudio = false;
    nextPlaybackTime = 0;
    currentGenerationId = null;

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    conversationSocket = new WebSocket(`${protocol}//${location.host}/api/voice/conversation/${threadId}`);
    conversationSocket.binaryType = 'arraybuffer';

    conversationSocket.onopen = () => {
        Console.info('Voice conversation opened, waiting for PersonaPlex...');
        // Don't start audio capture yet - wait for 'connected' message
    };

    conversationSocket.onmessage = async (event) => {
        if (typeof event.data === 'string') {
            // JSON message
            const data = JSON.parse(event.data);
            handleConversationMessage(data);
        } else {
            // Binary audio data from agent
            if (event.data.byteLength > 0) {
                audioQueue.push({
                    data: event.data,
                    metadata: { sample_rate: 24000, generation_id: currentGenerationId }
                });

                // Always try to schedule - scheduleAudioPlayback handles the queue
                if (!isPlayingAudio) {
                    playNextAudioChunk();
                } else {
                    // Already playing - schedule new chunks immediately
                    scheduleAudioPlayback();
                }
            }
        }
    };

    conversationSocket.onerror = (error) => {
        Console.error('Voice conversation error');
    };

    conversationSocket.onclose = () => {
        Console.info('Voice conversation closed');
        stopAudioCapture();
        conversationSocket = null;
        // Reset mic button state if it was active
        if (state.micActive) {
            state.micActive = false;
            updateMicButtonUI(false);
        }
    };

    return true;
}

// Voice response streaming state
let voiceStreamingBubble = null;
let voiceGenerationId = null;

function getOrCreateVoiceStreamingBubble(generationId) {
    // If generation changed, finalize old bubble and create new one
    if (voiceGenerationId !== generationId) {
        voiceStreamingBubble = null;
        voiceGenerationId = generationId;
    }

    if (!voiceStreamingBubble) {
        const div = document.createElement('div');
        div.className = 'flex justify-start message-enter';
        div.id = 'voice-streaming-bubble';

        div.innerHTML = `
            <div class="max-w-2xl px-4 py-3 rounded-2xl border bg-cathedral-assistant border-orange-800/30">
                <div class="text-xs text-orange-400 mb-1 font-medium">🎙️ PersonaPlex</div>
                <div class="text-sm whitespace-pre-wrap message-content"></div>
            </div>
        `;

        elements.messagesList.appendChild(div);
        voiceStreamingBubble = div.querySelector('.message-content');
        scrollToBottom();
    }

    return voiceStreamingBubble;
}

function handleConversationMessage(data) {
    switch (data.type) {
        case 'connecting':
            Console.info(data.message || 'Connecting to voice service...');
            state.voiceConnecting = true;
            updateMicButtonState();
            break;
        case 'connected':
            Console.success(`Voice ready: ${data.session_id}`);
            state.voiceConnecting = false;
            state.voiceConnected = true;
            updateMicButtonState();
            // Only start audio capture once fully connected AND mic is active
            if (state.micActive && !isRecording) {
                startAudioCapture();
            }
            break;

        case 'text_token':
            // Streaming text from PersonaPlex - display in chat
            const bubble = getOrCreateVoiceStreamingBubble(data.generation_id);
            bubble.textContent = data.buffer || '';
            scrollToBottom();
            break;

        case 'event':
            // Voice event envelope
            const event = data.event;
            if (event.event_type === 'speech_complete') {
                // Agent finished speaking - finalize bubble
                Console.info('Agent: ' + (event.payload?.text || '').slice(0, 50) + '...');
                voiceStreamingBubble = null;  // Next response gets new bubble
            } else if (event.event_type === 'final_transcript') {
                // User finished speaking
                Console.info('User: ' + (event.payload?.text || ''));
            }
            break;

        case 'audio_meta':
            currentGenerationId = data.generation_id;
            break;

        case 'interrupted':
            Console.info(`Interrupt: cancelled ${data.cancelled_generation_id?.slice(0, 8) || 'none'}`);
            // Clear audio queue to stop current playback
            audioQueue = [];
            isPlayingAudio = false;
            nextPlaybackTime = 0;
            currentGenerationId = null;
            break;

        case 'turn':
            if (data.state === 'start') {
                Console.info(`Turn: ${data.source} speaking`);
            }
            break;

        case 'transcript':
            // Real-time transcript update
            break;

        case 'error':
            Console.error(`Voice error: ${data.message}`);
            break;
    }
}

// Audio capture rebuffer state (20ms frames = 480 samples at 24kHz)
const FRAME_SIZE = 480;  // 20ms at 24kHz
let micBuffer = null;    // Float32 ring buffer
let micBufferPos = 0;    // Write position in buffer

function startAudioCapture() {
    if (!mediaStream || isRecording) return;

    const ctx = initAudioContext();

    // Initialize rebuffer (enough for a few frames)
    micBuffer = new Float32Array(FRAME_SIZE * 4);
    micBufferPos = 0;

    // Create audio source from microphone
    const source = ctx.createMediaStreamSource(mediaStream);

    // Create script processor for raw PCM access
    // Using smaller buffer (2048) for more frequent callbacks
    const processor = ctx.createScriptProcessor(2048, 1, 1);

    processor.onaudioprocess = (e) => {
        if (!isRecording || !conversationSocket || conversationSocket.readyState !== WebSocket.OPEN) {
            return;
        }

        // Get PCM data from browser
        const inputData = e.inputBuffer.getChannelData(0);

        // Rebuffer to exactly 20ms frames (480 samples)
        for (let i = 0; i < inputData.length; i++) {
            micBuffer[micBufferPos++] = inputData[i];

            // When we have a complete 20ms frame, send it
            if (micBufferPos >= FRAME_SIZE) {
                // Convert float32 to int16
                const int16Frame = new Int16Array(FRAME_SIZE);
                for (let j = 0; j < FRAME_SIZE; j++) {
                    const s = Math.max(-1, Math.min(1, micBuffer[j]));
                    int16Frame[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }

                // Send exactly 20ms frame to server
                conversationSocket.send(int16Frame.buffer);

                // Reset buffer (move any overflow to start)
                micBufferPos = 0;
            }
        }
    };

    source.connect(processor);
    processor.connect(ctx.destination);

    audioRecorder = { source, processor };
    isRecording = true;

    Console.info('Audio capture started (20ms frame rebuffering)');
}

function stopAudioCapture() {
    if (audioRecorder) {
        audioRecorder.source.disconnect();
        audioRecorder.processor.disconnect();
        audioRecorder = null;
    }

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    // Reset rebuffer state
    micBuffer = null;
    micBufferPos = 0;

    isRecording = false;
    Console.info('Audio capture stopped');
}

function sendInterrupt() {
    if (conversationSocket && conversationSocket.readyState === WebSocket.OPEN) {
        conversationSocket.send(JSON.stringify({ type: 'interrupt' }));
    }
}

function sendSilence() {
    if (conversationSocket && conversationSocket.readyState === WebSocket.OPEN) {
        conversationSocket.send(JSON.stringify({ type: 'silence' }));
    }
}

/**
 * Close voice conversation - ensures clean shutdown of all voice resources.
 * Called when: mic button toggled off, thread switch, voice disabled, page unload.
 */
function closeVoiceConversation() {
    // 1. Stop audio capture first (stops mic streaming)
    stopAudioCapture();

    // 2. Close WebSocket with clean code
    if (conversationSocket) {
        // Send close signal if socket is open
        if (conversationSocket.readyState === WebSocket.OPEN) {
            try {
                conversationSocket.send(JSON.stringify({ type: 'close' }));
            } catch (e) {
                // Ignore send errors during close
            }
        }
        conversationSocket.close(1000, 'User closed');
        conversationSocket = null;
    }

    // 3. Clear all audio state
    audioQueue = [];
    isPlayingAudio = false;
    nextPlaybackTime = 0;
    currentGenerationId = null;

    // 4. Reset mic state flag
    state.micActive = false;
}

// Expose for debugging
window.initVoiceConversation = initVoiceConversation;
window.closeVoiceConversation = closeVoiceConversation;
window.sendInterrupt = sendInterrupt;

// Clean up voice resources on page unload (prevents phantom streams)
window.addEventListener('beforeunload', () => {
    if (state.micActive || conversationSocket) {
        closeVoiceConversation();
    }
    if (voiceSocket) {
        closeVoiceStream();
    }
});

// ========== Initialize ==========

document.addEventListener('DOMContentLoaded', () => {
    // Initialize console
    Console.system('Cathedral initialized');
    Console.info('Console ready - system events will appear here');

    // Load threads
    loadThreads();

    // Connect to event stream for real-time system notifications
    connectEventSource();

    // Start agent status polling as fallback
    setInterval(pollAgentStatus, 10000);

    // Check security status
    checkSecurityStatus();

    // Check voice availability
    checkVoiceStatus();
});

async function checkSecurityStatus() {
    try {
        const res = await fetch('/api/security/status');
        if (res.ok) {
            const status = await res.json();
            if (status.encryption_enabled) {
                if (status.session?.is_locked) {
                    Console.security('Session locked - unlock required');
                } else {
                    Console.security('Session unlocked');
                    const timeout = status.session?.time_until_lock;
                    if (timeout) {
                        Console.security(`Auto-lock in ${Math.ceil(timeout / 60)} minutes`);
                    }
                }
            }
        }
    } catch (e) {
        // Security endpoint may not exist yet
    }
}
