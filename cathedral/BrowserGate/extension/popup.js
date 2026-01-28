/**
 * Cathedral Browser Extension - Popup Script
 */

// ============================================
// DOM Elements
// ============================================

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const pageTitle = document.getElementById('page-title');
const pageUrl = document.getElementById('page-url');
const feedback = document.getElementById('feedback');

const btnSendSelection = document.getElementById('send-selection');
const btnSendPage = document.getElementById('send-page');
const btnSaveScripture = document.getElementById('save-scripture');
const btnSearchMemory = document.getElementById('search-memory');

// ============================================
// Status Management
// ============================================

function updateStatus(connected, pendingCount = 0) {
  statusDot.className = 'status-dot ' + (connected ? 'connected' : '');

  if (connected) {
    statusText.textContent = 'Connected to Cathedral';
  } else if (pendingCount > 0) {
    statusText.textContent = `Reconnecting... (${pendingCount} pending)`;
  } else {
    statusText.textContent = 'Disconnected';
  }
}

function checkStatus() {
  chrome.runtime.sendMessage({ type: 'get_status' }, (response) => {
    if (response) {
      updateStatus(response.connected, response.pendingCount);
    } else {
      updateStatus(false);
    }
  });
}

// ============================================
// Page Info
// ============================================

function loadPageInfo() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs[0]) return;

    const tab = tabs[0];
    pageTitle.textContent = tab.title || 'Unknown';
    pageUrl.textContent = tab.url || '';

    // Check for selection
    chrome.tabs.sendMessage(tab.id, { type: 'get_page_info' }, (response) => {
      if (chrome.runtime.lastError) {
        // Content script not loaded (e.g., chrome:// pages)
        btnSendSelection.disabled = true;
        btnSendPage.disabled = true;
        btnSaveScripture.disabled = true;
        btnSearchMemory.disabled = true;
        showFeedback('Cannot access this page', 'error');
        return;
      }

      if (response && !response.hasSelection) {
        btnSendSelection.textContent = '✎ No Selection';
        btnSearchMemory.disabled = true;
      }
    });
  });
}

// ============================================
// Actions
// ============================================

function sendAction(action, selectionOnly = false) {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs[0]) return;

    const messageType = selectionOnly ? 'get_selection' : 'get_page_content';

    chrome.tabs.sendMessage(tabs[0].id, {
      type: messageType,
      action: action
    }, (response) => {
      if (chrome.runtime.lastError) {
        showFeedback('Cannot access this page', 'error');
        return;
      }

      if (selectionOnly && response && !response.hasSelection) {
        showFeedback('No text selected', 'error');
        return;
      }

      showFeedback('Sent to Cathedral!', 'success');

      // Close popup after brief delay
      setTimeout(() => window.close(), 800);
    });
  });
}

// ============================================
// Feedback
// ============================================

function showFeedback(message, type = 'success') {
  feedback.textContent = message;
  feedback.className = 'feedback show ' + type;

  setTimeout(() => {
    feedback.className = 'feedback';
  }, 3000);
}

// ============================================
// Event Listeners
// ============================================

btnSendSelection.addEventListener('click', () => {
  sendAction('send_to_cathedral', true);
});

btnSendPage.addEventListener('click', () => {
  sendAction('send_to_cathedral', false);
});

btnSaveScripture.addEventListener('click', () => {
  sendAction('send_to_scripture', false);
});

btnSearchMemory.addEventListener('click', () => {
  sendAction('search_memory', true);
});

// Listen for server responses
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'server_response') {
    const response = message.data;
    if (response.success) {
      showFeedback(response.message || 'Success!', 'success');
    } else {
      showFeedback(response.message || 'Error', 'error');
    }
  }
});

// ============================================
// Initialize
// ============================================

checkStatus();
loadPageInfo();

// Refresh status periodically
setInterval(checkStatus, 2000);
