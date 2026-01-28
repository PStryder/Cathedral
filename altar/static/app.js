// Cathedral Frontend - Vanilla JS + SSE

const state = {
    currentThreadUid: null,
    currentThreadName: null,
    isStreaming: false,
    consoleCollapsed: false,
    consoleLineCount: 0,
    maxConsoleLines: 200
};

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
    connectionText: document.getElementById('connectionText')
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

    try {
        // Use fetch with POST for SSE (EventSource only supports GET)
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_input: text,
                thread_uid: state.currentThreadUid
            })
        });

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
                            fullResponse += data.token;
                            contentEl.textContent = fullResponse;
                            scrollToBottom();
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
    system: (msg) => consolePrint(msg, 'info', 'system')
};

// Make Console globally accessible
window.Console = Console;
window.toggleConsole = toggleConsole;
window.clearConsole = clearConsole;

// ========== Event Source for System Events ==========

let eventSource = null;

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

    // Ignore ping events (keepalive)
    eventSource.addEventListener('ping', () => {});
}

// Track last poll time for efficient polling
let lastPollTimestamp = 0;
let sseConnected = false;

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
