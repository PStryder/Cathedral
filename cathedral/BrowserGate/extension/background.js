/**
 * Cathedral Browser Extension - Background Service Worker
 *
 * Manages WebSocket connection and coordinates messaging between
 * content scripts, popup, and Cathedral server.
 */

const CATHEDRAL_WS_URL = 'ws://localhost:8765';
const RECONNECT_INTERVAL = 5000;

let socket = null;
let isConnected = false;
let pendingMessages = [];
let reconnectTimer = null;

// ============================================
// WebSocket Connection Management
// ============================================

function connect() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    return;
  }

  try {
    socket = new WebSocket(CATHEDRAL_WS_URL);

    socket.onopen = () => {
      console.log('[Cathedral] Connected to server');
      isConnected = true;
      updateBadge('connected');

      // Send any pending messages
      while (pendingMessages.length > 0) {
        const msg = pendingMessages.shift();
        sendMessage(msg);
      }

      // Clear reconnect timer
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    socket.onmessage = (event) => {
      try {
        const response = JSON.parse(event.data);
        console.log('[Cathedral] Response:', response);

        // Notify popup if open
        chrome.runtime.sendMessage({
          type: 'server_response',
          data: response
        }).catch(() => {
          // Popup not open, ignore
        });

        // Show notification for important responses
        if (response.success && response.message) {
          showNotification('Cathedral', response.message);
        }
      } catch (e) {
        console.error('[Cathedral] Parse error:', e);
      }
    };

    socket.onclose = () => {
      console.log('[Cathedral] Disconnected');
      isConnected = false;
      socket = null;
      updateBadge('disconnected');
      scheduleReconnect();
    };

    socket.onerror = (error) => {
      console.error('[Cathedral] WebSocket error:', error);
      isConnected = false;
      updateBadge('error');
    };

  } catch (e) {
    console.error('[Cathedral] Connection failed:', e);
    isConnected = false;
    updateBadge('error');
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (reconnectTimer) return;

  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, RECONNECT_INTERVAL);
}

function sendMessage(message) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message));
    return true;
  } else {
    pendingMessages.push(message);
    connect();
    return false;
  }
}

function updateBadge(status) {
  const colors = {
    connected: '#10B981',    // Green
    disconnected: '#6B7280', // Gray
    error: '#EF4444'         // Red
  };

  const text = status === 'connected' ? '' : '!';

  chrome.action.setBadgeBackgroundColor({ color: colors[status] || colors.disconnected });
  chrome.action.setBadgeText({ text });
}

// ============================================
// Context Menu Setup
// ============================================

chrome.runtime.onInstalled.addListener(() => {
  // Create context menus
  chrome.contextMenus.create({
    id: 'send-to-cathedral',
    title: 'Send to Cathedral',
    contexts: ['selection', 'page']
  });

  chrome.contextMenus.create({
    id: 'send-to-scripture',
    title: 'Save as Scripture',
    contexts: ['selection', 'page']
  });

  chrome.contextMenus.create({
    id: 'search-memory',
    title: 'Search Cathedral Memory',
    contexts: ['selection']
  });

  chrome.contextMenus.create({
    id: 'separator',
    type: 'separator',
    contexts: ['selection', 'page']
  });

  chrome.contextMenus.create({
    id: 'send-page',
    title: 'Send Entire Page',
    contexts: ['page']
  });

  console.log('[Cathedral] Context menus created');
});

// ============================================
// Context Menu Handlers
// ============================================

chrome.contextMenus.onClicked.addListener((info, tab) => {
  const action = info.menuItemId;

  if (action === 'separator') return;

  // Get page info
  const pageInfo = {
    url: tab.url,
    title: tab.title
  };

  if (action === 'send-to-cathedral') {
    if (info.selectionText) {
      sendToCathedral(info.selectionText, pageInfo, 'send_to_cathedral');
    } else {
      // Request full page content from content script
      requestPageContent(tab.id, 'send_to_cathedral');
    }
  }
  else if (action === 'send-to-scripture') {
    if (info.selectionText) {
      sendToCathedral(info.selectionText, pageInfo, 'send_to_scripture');
    } else {
      requestPageContent(tab.id, 'send_to_scripture');
    }
  }
  else if (action === 'search-memory') {
    if (info.selectionText) {
      sendToCathedral(info.selectionText, pageInfo, 'search_memory');
    }
  }
  else if (action === 'send-page') {
    requestPageContent(tab.id, 'send_to_cathedral');
  }
});

function sendToCathedral(content, pageInfo, action) {
  const message = {
    type: 'page',
    url: pageInfo.url,
    title: pageInfo.title,
    content: null,
    selection: content,
    action: action,
    timestamp: new Date().toISOString()
  };

  const sent = sendMessage(message);
  showNotification(
    'Cathedral',
    sent ? `Sending to Cathedral...` : 'Queued (connecting...)'
  );
}

function requestPageContent(tabId, action) {
  chrome.tabs.sendMessage(tabId, {
    type: 'get_page_content',
    action: action
  });
}

// ============================================
// Keyboard Commands
// ============================================

chrome.commands.onCommand.addListener((command) => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs[0]) return;

    const tab = tabs[0];

    if (command === 'send-selection') {
      chrome.tabs.sendMessage(tab.id, {
        type: 'get_selection',
        action: 'send_to_cathedral'
      });
    }
    else if (command === 'send-page') {
      chrome.tabs.sendMessage(tab.id, {
        type: 'get_page_content',
        action: 'send_to_cathedral'
      });
    }
  });
});

// ============================================
// Message Handling (from content script & popup)
// ============================================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'page_content') {
    // Content script sending page content
    const wsMessage = {
      type: 'page',
      url: message.url,
      title: message.title,
      content: message.content,
      selection: message.selection,
      action: message.action,
      timestamp: new Date().toISOString()
    };

    sendMessage(wsMessage);
    sendResponse({ success: true });
  }
  else if (message.type === 'get_status') {
    // Popup requesting connection status
    sendResponse({
      connected: isConnected,
      pendingCount: pendingMessages.length
    });
  }
  else if (message.type === 'reconnect') {
    // Popup requesting reconnect
    connect();
    sendResponse({ success: true });
  }
  else if (message.type === 'send_from_popup') {
    // Popup sending content directly
    sendMessage(message.data);
    sendResponse({ success: true });
  }

  return true; // Keep channel open for async response
});

// ============================================
// Notifications
// ============================================

function showNotification(title, message) {
  // Use badge flash instead of notifications (less intrusive)
  chrome.action.setBadgeText({ text: '...' });
  chrome.action.setBadgeBackgroundColor({ color: '#6366F1' });

  setTimeout(() => {
    updateBadge(isConnected ? 'connected' : 'disconnected');
  }, 1500);
}

// ============================================
// Initialize
// ============================================

// Connect on startup
connect();

// Keep service worker alive with periodic ping
setInterval(() => {
  if (isConnected && socket && socket.readyState === WebSocket.OPEN) {
    sendMessage({ type: 'ping', action: 'ping' });
  }
}, 30000);

console.log('[Cathedral] Background service worker started');
