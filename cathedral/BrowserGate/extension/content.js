/**
 * Cathedral Browser Extension - Content Script
 *
 * Runs on every page to capture selections and page content.
 */

// ============================================
// Content Extraction
// ============================================

function getSelectedText() {
  const selection = window.getSelection();
  return selection ? selection.toString().trim() : '';
}

function getPageContent() {
  // Try to get main content, falling back to body
  const mainSelectors = [
    'article',
    'main',
    '[role="main"]',
    '.post-content',
    '.article-content',
    '.entry-content',
    '.content',
    '#content'
  ];

  let contentElement = null;

  for (const selector of mainSelectors) {
    const el = document.querySelector(selector);
    if (el && el.textContent.trim().length > 100) {
      contentElement = el;
      break;
    }
  }

  // Fall back to body
  if (!contentElement) {
    contentElement = document.body;
  }

  // Convert to clean text/markdown
  return extractContent(contentElement);
}

function extractContent(element) {
  // Clone to avoid modifying the page
  const clone = element.cloneNode(true);

  // Remove unwanted elements
  const removeSelectors = [
    'script',
    'style',
    'noscript',
    'nav',
    'header',
    'footer',
    'aside',
    '.sidebar',
    '.navigation',
    '.menu',
    '.advertisement',
    '.ad',
    '[role="navigation"]',
    '[role="banner"]',
    '[role="contentinfo"]',
    '.comments',
    '#comments'
  ];

  removeSelectors.forEach(selector => {
    clone.querySelectorAll(selector).forEach(el => el.remove());
  });

  // Convert to markdown-like format
  return htmlToMarkdown(clone);
}

function htmlToMarkdown(element) {
  let result = '';

  function processNode(node, depth = 0) {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent.replace(/\s+/g, ' ');
      if (text.trim()) {
        result += text;
      }
      return;
    }

    if (node.nodeType !== Node.ELEMENT_NODE) return;

    const tag = node.tagName.toLowerCase();

    // Skip hidden elements
    const style = window.getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden') {
      return;
    }

    // Handle different tags
    switch (tag) {
      case 'h1':
        result += '\n\n# ';
        break;
      case 'h2':
        result += '\n\n## ';
        break;
      case 'h3':
        result += '\n\n### ';
        break;
      case 'h4':
      case 'h5':
      case 'h6':
        result += '\n\n#### ';
        break;
      case 'p':
      case 'div':
        result += '\n\n';
        break;
      case 'br':
        result += '\n';
        return;
      case 'li':
        result += '\n- ';
        break;
      case 'a':
        const href = node.getAttribute('href');
        const text = node.textContent.trim();
        if (href && text && !href.startsWith('#') && !href.startsWith('javascript:')) {
          const absoluteUrl = new URL(href, window.location.href).href;
          result += `[${text}](${absoluteUrl})`;
          return; // Don't process children
        }
        break;
      case 'strong':
      case 'b':
        result += '**';
        break;
      case 'em':
      case 'i':
        result += '*';
        break;
      case 'code':
        result += '`';
        break;
      case 'pre':
        result += '\n\n```\n';
        break;
      case 'blockquote':
        result += '\n\n> ';
        break;
      case 'hr':
        result += '\n\n---\n\n';
        return;
      case 'img':
        const alt = node.getAttribute('alt') || 'image';
        const src = node.getAttribute('src');
        if (src) {
          const absoluteSrc = new URL(src, window.location.href).href;
          result += `![${alt}](${absoluteSrc})`;
        }
        return;
    }

    // Process children
    for (const child of node.childNodes) {
      processNode(child, depth + 1);
    }

    // Closing tags
    switch (tag) {
      case 'h1':
      case 'h2':
      case 'h3':
      case 'h4':
      case 'h5':
      case 'h6':
        result += '\n';
        break;
      case 'strong':
      case 'b':
        result += '**';
        break;
      case 'em':
      case 'i':
        result += '*';
        break;
      case 'code':
        result += '`';
        break;
      case 'pre':
        result += '\n```\n\n';
        break;
    }
  }

  processNode(element);

  // Clean up excessive whitespace
  return result
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]+/g, ' ')
    .trim();
}

// ============================================
// Message Handling
// ============================================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'get_selection') {
    const selection = getSelectedText();

    if (selection) {
      chrome.runtime.sendMessage({
        type: 'page_content',
        url: window.location.href,
        title: document.title,
        content: null,
        selection: selection,
        action: message.action
      });
    }

    sendResponse({ success: true, hasSelection: !!selection });
  }
  else if (message.type === 'get_page_content') {
    const content = getPageContent();
    const selection = getSelectedText();

    chrome.runtime.sendMessage({
      type: 'page_content',
      url: window.location.href,
      title: document.title,
      content: content,
      selection: selection || null,
      action: message.action
    });

    sendResponse({ success: true });
  }
  else if (message.type === 'get_page_info') {
    // Quick page info for popup
    sendResponse({
      url: window.location.href,
      title: document.title,
      hasSelection: !!getSelectedText(),
      contentLength: document.body.textContent.length
    });
  }

  return true;
});

// ============================================
// Optional: Visual Feedback
// ============================================

function showFeedback(message) {
  const existing = document.getElementById('cathedral-feedback');
  if (existing) existing.remove();

  const div = document.createElement('div');
  div.id = 'cathedral-feedback';
  div.textContent = message;
  div.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 14px;
    z-index: 999999;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    animation: cathedral-slide-in 0.3s ease-out;
  `;

  // Add animation
  const style = document.createElement('style');
  style.textContent = `
    @keyframes cathedral-slide-in {
      from { transform: translateX(100%); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
  `;
  document.head.appendChild(style);

  document.body.appendChild(div);

  setTimeout(() => {
    div.style.animation = 'cathedral-slide-in 0.3s ease-out reverse';
    setTimeout(() => div.remove(), 300);
  }, 2000);
}

// Listen for feedback requests
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'show_feedback') {
    showFeedback(message.text);
    sendResponse({ success: true });
  }
  return true;
});

console.log('[Cathedral] Content script loaded');
