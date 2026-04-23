/**
 * 伯乐 Bole — 前端逻辑（增强版）
 *
 * 新增功能：
 *   - 暗色/亮色主题切换（localStorage 持久化）
 *   - 消息一键复制
 *   - 消息时间戳
 *   - 代码语法高亮（highlight.js）
 *   - 代码块一键复制
 *   - 滚动到底部按钮
 */

// ── DOM 元素 ─────────────────────────────────────────────
const $welcomeScreen    = document.getElementById('welcome-screen');
const $chatScreen       = document.getElementById('chat-screen');
const $userIdInput      = document.getElementById('user-id-input');
const $passwordInput    = document.getElementById('password-input');
const $loginError       = document.getElementById('login-error');
const $startBtn         = document.getElementById('start-btn');
const $logoutBtn        = document.getElementById('logout-btn');
const $sessionBadge     = document.getElementById('session-badge');
const $connectionStatus = document.getElementById('connection-status');
const $chatMessages     = document.getElementById('chat-messages');
const $messageInput     = document.getElementById('message-input');
const $sendBtn          = document.getElementById('send-btn');
const $hitlPanel        = document.getElementById('hitl-panel');
const $hitlInstruction  = document.getElementById('hitl-instruction');
const $hitlApprove      = document.getElementById('hitl-approve');
const $hitlReject       = document.getElementById('hitl-reject');
const $chatInputArea    = document.getElementById('chat-input-area');

// 侧边栏
const $sidebar          = document.getElementById('sidebar');
const $sidebarList      = document.getElementById('sidebar-list');
const $sidebarUserId    = document.getElementById('sidebar-user-id');
const $sidebarUserAvatar= document.getElementById('sidebar-user-avatar');
const $btnNewChat       = document.getElementById('btn-new-chat');
const $btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
const $currentSessionTitle = document.getElementById('current-session-title');

// 简历上传
const $btnUpload        = document.getElementById('btn-upload');
const $resumeFileInput  = document.getElementById('resume-file-input');
const $resumePreviewBar = document.getElementById('resume-preview-bar');
const $resumeFilename   = document.getElementById('resume-filename');
const $resumeChars      = document.getElementById('resume-chars');
const $btnSendResume    = document.getElementById('btn-send-resume');
const $btnCancelResume  = document.getElementById('btn-cancel-resume');
const $jdInput          = document.getElementById('jd-input');

// 主题切换
const $btnThemeToggle   = document.getElementById('btn-theme-toggle');
const $themeIconSun     = document.getElementById('theme-icon-sun');
const $themeIconMoon    = document.getElementById('theme-icon-moon');
const $themeLabel       = document.getElementById('theme-label');

// 其他新增
const $copyToast        = document.getElementById('copy-toast');
const $scrollBottomBtn  = document.getElementById('scroll-bottom-btn');

// ── 状态 ─────────────────────────────────────────────────
let ws = null;
let userId = '';
let sessionId = '';
let sessionsCache = [];     // 缓存会话列表
let reconnectAttempts = 0;
const MAX_RECONNECT = 5;
let authToken = '';  // 登录 token

// 流式相关
let streamingBubble = null;
let streamingRawText = '';
let isStreaming = false;

// 简历上传
let pendingResumeText = '';

// ── 工具函数 ─────────────────────────────────────────────

function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        try { return marked.parse(text); } catch { }
    }
    return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

function escapeHtml(text) {
    return String(text || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

let _scrollPending = false;
function scrollToBottom() {
    if (_scrollPending) return;
    _scrollPending = true;
    requestAnimationFrame(() => {
        $chatMessages.scrollTop = $chatMessages.scrollHeight;
        _scrollPending = false;
    });
}

function setConnectionStatus(state) {
    const dot = $connectionStatus.querySelector('.status-dot');
    const text = $connectionStatus.querySelector('.status-text');
    dot.className = 'status-dot';
    switch (state) {
        case 'online':  dot.classList.add('online'); text.textContent = '在线'; break;
        case 'connecting': text.textContent = '连接中...'; break;
        case 'error':   dot.classList.add('error');  text.textContent = '连接断开'; break;
    }
}

function setInputEnabled(enabled) {
    $messageInput.disabled = !enabled;
    $sendBtn.disabled = !enabled || !$messageInput.value.trim();
    if (enabled) $messageInput.focus();
}

function formatRelativeTime(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 60) return '刚刚';
    if (diff < 3600) return `${Math.floor(diff/60)}分钟前`;
    if (diff < 86400) return `${Math.floor(diff/3600)}小时前`;
    if (diff < 86400*30) return `${Math.floor(diff/86400)}天前`;
    return d.toLocaleDateString('zh-CN');
}

function getCurrentTimeStr() {
    const now = new Date();
    const h = String(now.getHours()).padStart(2, '0');
    const m = String(now.getMinutes()).padStart(2, '0');
    return `${h}:${m}`;
}

// ── 暗色模式切换 ─────────────────────────────────────────

function initTheme() {
    const saved = localStorage.getItem('bole-theme') || localStorage.getItem('genai-theme') || 'light';
    applyTheme(saved);
}

function applyTheme(theme) {
    document.body.setAttribute('data-theme', theme);
    if (theme === 'dark') {
        $themeIconSun.style.display = '';
        $themeIconMoon.style.display = 'none';
        $themeLabel.textContent = '亮色模式';
    } else {
        $themeIconSun.style.display = 'none';
        $themeIconMoon.style.display = '';
        $themeLabel.textContent = '暗色模式';
    }
}

function toggleTheme() {
    const current = document.body.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    localStorage.setItem('bole-theme', next);
    applyTheme(next);
}

$btnThemeToggle.addEventListener('click', toggleTheme);
initTheme();

// ── 复制 Toast ───────────────────────────────────────────

let toastTimer = null;
function showCopyToast(text = '✓ 已复制到剪贴板') {
    $copyToast.textContent = text;
    $copyToast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $copyToast.classList.remove('show'), 1800);
}

// ── 代码块增强 ───────────────────────────────────────────

function enhanceCodeBlocks(container) {
    if (!container) return;
    container.querySelectorAll('pre code').forEach(block => {
        // 避免重复处理
        if (block.closest('pre').querySelector('.code-header')) return;

        const pre = block.closest('pre');

        // 检测语言
        let lang = '';
        block.classList.forEach(cls => {
            if (cls.startsWith('language-')) lang = cls.replace('language-', '');
        });

        // 创建头部
        const header = document.createElement('div');
        header.className = 'code-header';

        const langSpan = document.createElement('span');
        langSpan.className = 'code-lang';
        langSpan.textContent = lang || 'code';

        const copyBtn = document.createElement('button');
        copyBtn.className = 'code-copy-btn';
        copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>复制`;
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(block.textContent).then(() => {
                copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>已复制`;
                copyBtn.classList.add('copied');
                showCopyToast();
                setTimeout(() => {
                    copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>复制`;
                    copyBtn.classList.remove('copied');
                }, 2000);
            });
        });

        header.appendChild(langSpan);
        header.appendChild(copyBtn);
        pre.insertBefore(header, pre.firstChild);

        // 应用语法高亮
        if (typeof hljs !== 'undefined') {
            hljs.highlightElement(block);
        }
    });
}

// ── 消息操作按钮（复制） ─────────────────────────────────

function createMsgActions(rawText) {
    const actions = document.createElement('div');
    actions.className = 'msg-actions';

    const copyBtn = document.createElement('button');
    copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>复制`;
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(rawText).then(() => {
            copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>已复制`;
            copyBtn.classList.add('copied');
            showCopyToast();
            setTimeout(() => {
                copyBtn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>复制`;
                copyBtn.classList.remove('copied');
            }, 2000);
        });
    });

    actions.appendChild(copyBtn);
    return actions;
}

// ── 消息渲染 ─────────────────────────────────────────────

function addMessage(role, content) {
    removeThinking();

    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    if (role === 'user') {
        avatar.textContent = '🙋';
    } else if (role === 'error') {
        avatar.textContent = '⚠️';
    } else {
        const avImg = document.createElement('img');
        avImg.src = '/static/bole-avatar.png';
        avImg.alt = '伯乐';
        avImg.style.cssText = 'width:100%;height:100%;border-radius:50%;object-fit:cover;';
        avatar.appendChild(avImg);
    }

    const msgContent = document.createElement('div');
    msgContent.className = 'msg-content';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = role === 'user'
        ? content.replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>')
        : renderMarkdown(content);

    // 时间戳
    const timeEl = document.createElement('div');
    timeEl.className = 'msg-time';
    timeEl.textContent = getCurrentTimeStr();

    msgContent.appendChild(bubble);

    // AI/error 消息增加复制按钮
    if (role === 'ai') {
        msgContent.appendChild(createMsgActions(content));
    }

    msgContent.appendChild(timeEl);

    wrapper.appendChild(avatar);
    wrapper.appendChild(msgContent);
    $chatMessages.appendChild(wrapper);

    // 代码块增强
    if (role !== 'user') {
        enhanceCodeBlocks(bubble);
    }

    scrollToBottom();
}

// ── 流式消息渲染 ─────────────────────────────────────────

let streamingMsgContent = null;
let _streamRenderTimer = null;
let _streamDirty = false;

function createStreamBubble() {
    removeThinking();
    endStream();

    const wrapper = document.createElement('div');
    wrapper.className = 'message ai';
    wrapper.id = 'streaming-message';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    const avImg = document.createElement('img');
    avImg.src = '/static/bole-avatar.png';
    avImg.alt = '伯乐';
    avImg.style.cssText = 'width:100%;height:100%;border-radius:50%;object-fit:cover;';
    avatar.appendChild(avImg);

    const content = document.createElement('div');
    content.className = 'msg-content';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble streaming';
    bubble.innerHTML = '<span class="stream-cursor"></span>';

    content.appendChild(bubble);
    wrapper.appendChild(avatar);
    wrapper.appendChild(content);
    $chatMessages.appendChild(wrapper);

    streamingBubble = bubble;
    streamingMsgContent = content;
    streamingRawText = '';
    isStreaming = true;
    _streamDirty = false;
    scrollToBottom();
}

function _flushStreamRender() {
    if (!streamingBubble || !_streamDirty) return;
    _streamDirty = false;
    streamingBubble.innerHTML = renderMarkdown(streamingRawText) + '<span class="stream-cursor"></span>';
    scrollToBottom();
}

function appendToken(token) {
    if (!streamingBubble || !isStreaming) {
        createStreamBubble();
    }

    streamingRawText += token;
    _streamDirty = true;

    // 节流：每 80ms 渲染一次，避免每 token 都调 marked.parse
    if (!_streamRenderTimer) {
        _streamRenderTimer = setTimeout(() => {
            _streamRenderTimer = null;
            _flushStreamRender();
        }, 80);
    }
}

function endStream() {
    // 清除渲染定时器
    if (_streamRenderTimer) {
        clearTimeout(_streamRenderTimer);
        _streamRenderTimer = null;
    }

    if (streamingBubble && isStreaming) {
        if (streamingRawText) {
            streamingBubble.innerHTML = renderMarkdown(streamingRawText);
            // 增强代码块
            enhanceCodeBlocks(streamingBubble);
        }
        streamingBubble.classList.remove('streaming');
        const cursor = streamingBubble.querySelector('.stream-cursor');
        if (cursor) cursor.remove();

        // 添加操作按钮和时间戳
        if (streamingMsgContent && streamingRawText) {
            streamingMsgContent.appendChild(createMsgActions(streamingRawText));
            const timeEl = document.createElement('div');
            timeEl.className = 'msg-time';
            timeEl.textContent = getCurrentTimeStr();
            streamingMsgContent.appendChild(timeEl);
        }
    }
    streamingBubble = null;
    streamingMsgContent = null;
    streamingRawText = '';
    isStreaming = false;
    _streamDirty = false;

    const el = document.getElementById('streaming-message');
    if (el) el.removeAttribute('id');
}

// ── 进度 & Thinking 指示器 ─────────────────────────────────

function showNodeProgress(label) {
    let el = document.getElementById('thinking-el');
    if (!el) {
        showThinking(label);
        return;
    }
    const textEl = el.querySelector('.thinking-label');
    if (textEl) textEl.textContent = label;
}

function showThinking(label) {
    removeThinking();

    const el = document.createElement('div');
    el.className = 'thinking-indicator';
    el.id = 'thinking-el';

    const av = document.createElement('div');
    av.className = 'msg-avatar';
    av.style.cssText = 'border-radius:50%;width:32px;height:32px;overflow:hidden;flex-shrink:0';
    const avImg = document.createElement('img');
    avImg.src = '/static/bole-avatar.png';
    avImg.alt = '伯乐';
    avImg.style.cssText = 'width:100%;height:100%;object-fit:cover;';
    av.appendChild(avImg);

    const wrap = document.createElement('div');
    wrap.className = 'thinking-wrap';

    const dots = document.createElement('div');
    dots.className = 'thinking-dots';
    dots.innerHTML = '<span></span><span></span><span></span>';

    const labelEl = document.createElement('span');
    labelEl.className = 'thinking-label';
    labelEl.textContent = label || '正在思考...';

    wrap.appendChild(dots);
    wrap.appendChild(labelEl);
    el.appendChild(av);
    el.appendChild(wrap);
    $chatMessages.appendChild(el);
    scrollToBottom();
}

function removeThinking() {
    const el = document.getElementById('thinking-el');
    if (el) el.remove();
}

// ── HITL 面板 ────────────────────────────────────────────

function showHitlPanel(instruction) {
    $hitlInstruction.textContent = instruction || '请检查上方结果，确认是否保存？';
    $hitlPanel.classList.add('active');
    $chatInputArea.style.display = 'none';
    scrollToBottom();
}

function hideHitlPanel() {
    $hitlPanel.classList.remove('active');
    $chatInputArea.style.display = '';
}

function sendHitlDecision(decision) {
    hideHitlPanel();
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'hitl_response', decision }));
    }
}

$hitlApprove.addEventListener('click', () => sendHitlDecision('approve'));
$hitlReject.addEventListener('click', () => sendHitlDecision('reject'));

// ── 侧边栏：会话列表 ─────────────────────────────────────

async function fetchSessions() {
    const resp = await fetch(`/api/users/${encodeURIComponent(userId)}/sessions`, {
        headers: { 'Authorization': `Bearer ${authToken}` },
    });
    if (!resp.ok) throw new Error('加载会话列表失败');
    const data = await resp.json();
    sessionsCache = data.sessions || [];
    return sessionsCache;
}

async function createNewSession(title = '新对话') {
    const resp = await fetch(`/api/users/${encodeURIComponent(userId)}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
        body: JSON.stringify({ title }),
    });
    if (!resp.ok) throw new Error('创建会话失败');
    return resp.json();
}

async function deleteSession(sid) {
    const resp = await fetch(`/api/users/${encodeURIComponent(userId)}/sessions/${encodeURIComponent(sid)}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${authToken}` },
    });
    return resp.ok;
}

function renderSidebar() {
    $sidebarUserId.textContent = userId;
    $sidebarUserAvatar.textContent = (userId[0] || 'U').toUpperCase();

    if (!sessionsCache.length) {
        $sidebarList.innerHTML = '<div class="sidebar-empty">暂无会话，点击上方「新建对话」开始</div>';
        return;
    }

    $sidebarList.innerHTML = sessionsCache.map(s => {
        const active = s.session_id === sessionId ? 'active' : '';
        const title = escapeHtml(s.title || '新对话');
        const time = formatRelativeTime(s.updated_at || s.created_at);
        return `
          <div class="sidebar-item ${active}" data-sid="${s.session_id}">
            <div class="sidebar-item-main">
              <div class="sidebar-item-title">${title}</div>
              <div class="sidebar-item-time">${time}</div>
            </div>
            <button class="sidebar-item-delete" data-sid="${s.session_id}" title="删除">
              <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z"/></svg>
            </button>
          </div>`;
    }).join('');
}

$sidebarList.addEventListener('click', async (e) => {
    const deleteBtn = e.target.closest('.sidebar-item-delete');
    if (deleteBtn) {
        e.stopPropagation();
        const sid = deleteBtn.dataset.sid;
        if (!confirm('确认删除这个会话？')) return;
        const ok = await deleteSession(sid);
        if (!ok) return alert('删除失败');
        await fetchSessions();
        if (sid === sessionId) {
            // 当前会话被删 → 若还有其他会话则切换到第一个，否则新建
            if (sessionsCache.length > 0) {
                await switchSession(sessionsCache[0].session_id, sessionsCache[0].title);
            } else {
                const s = await createNewSession();
                sessionsCache.unshift(s);
                await switchSession(s.session_id, s.title);
            }
        }
        renderSidebar();
        return;
    }

    const item = e.target.closest('.sidebar-item');
    if (item) {
        const sid = item.dataset.sid;
        if (sid === sessionId) return;
        const title = sessionsCache.find(s => s.session_id === sid)?.title || '';
        await switchSession(sid, title);
    }
});

$btnNewChat.addEventListener('click', async () => {
    try {
        const s = await createNewSession();
        sessionsCache.unshift(s);
        await switchSession(s.session_id, s.title);
        renderSidebar();
    } catch (e) {
        alert(e.message);
    }
});

$btnToggleSidebar.addEventListener('click', () => {
    $sidebar.classList.toggle('collapsed');
});

// ── 滚动到底部按钮（rAF 节流） ───────────────────────────

let _scrollCheckPending = false;
$chatMessages.addEventListener('scroll', () => {
    if (_scrollCheckPending) return;
    _scrollCheckPending = true;
    requestAnimationFrame(() => {
        _scrollCheckPending = false;
        const threshold = 200;
        const distToBottom = $chatMessages.scrollHeight - $chatMessages.scrollTop - $chatMessages.clientHeight;
        if (distToBottom > threshold) {
            $scrollBottomBtn.classList.add('visible');
        } else {
            $scrollBottomBtn.classList.remove('visible');
        }
    });
});

$scrollBottomBtn.addEventListener('click', () => {
    $chatMessages.scrollTo({ top: $chatMessages.scrollHeight, behavior: 'smooth' });
});

// ── WebSocket 连接 ───────────────────────────────────────

function closeWebSocket() {
    if (ws) {
        ws.onopen    = null;
        ws.onmessage = null;
        ws.onclose   = null;
        ws.onerror   = null;
        try { ws.close(); } catch {}
        ws = null;
    }
}

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws/${encodeURIComponent(userId)}/${encodeURIComponent(sessionId)}?token=${encodeURIComponent(authToken)}`;

    setConnectionStatus('connecting');
    ws = new WebSocket(url);

    ws.onopen = () => {
        reconnectAttempts = 0;
        setConnectionStatus('online');
        setInputEnabled(true);
    };

    ws.onmessage = (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch { return; }

        switch (data.type) {
            case 'connected':
                console.log('[WS] Connected:', data.user_id, data.session_id);
                break;

            case 'chat_history':
                if (data.messages && data.messages.length > 0) {
                    const wm = document.querySelector('.welcome-message');
                    if (wm) wm.style.display = 'none';

                    data.messages.forEach(msg => addMessage(msg.role, msg.content));

                    const sep = document.createElement('div');
                    sep.className = 'history-separator';
                    sep.innerHTML = '<span>以上为历史对话</span>';
                    $chatMessages.appendChild(sep);
                    scrollToBottom();
                }
                break;

            case 'stream_start':
                showThinking('正在处理...');
                setInputEnabled(false);
                break;

            case 'node_progress':
                if (!isStreaming) {
                    showNodeProgress(data.label);
                }
                break;

            case 'ai_token':
                removeThinking();
                appendToken(data.token);
                break;

            case 'ai_message':
                removeThinking();
                if (isStreaming) {
                    streamingRawText = data.content;
                    streamingBubble.innerHTML = renderMarkdown(data.content) + '<span class="stream-cursor"></span>';
                    scrollToBottom();
                } else {
                    addMessage('ai', data.content);
                }
                break;

            case 'stream_end':
                endStream();
                setInputEnabled(true);
                // 流式结束后刷新侧边栏（标题可能被后端自动改名）
                fetchSessions().then(renderSidebar).catch(() => {});
                break;

            case 'hitl_request':
                removeThinking();
                endStream();
                showHitlPanel(data.instruction);
                break;

            case 'thinking':
                if (data.status) {
                    showThinking();
                    setInputEnabled(false);
                } else {
                    removeThinking();
                    setInputEnabled(true);
                }
                break;

            case 'error':
                removeThinking();
                endStream();
                addMessage('error', data.content);
                setInputEnabled(true);
                break;

            default:
                console.log('Unknown:', data.type);
        }
    };

    ws.onclose = () => {
        setConnectionStatus('error');
        setInputEnabled(false);
        if (reconnectAttempts < MAX_RECONNECT) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 15000);
            console.log(`[WS] Reconnecting in ${delay}ms (#${reconnectAttempts})`);
            setTimeout(connectWebSocket, delay);
        }
    };

    ws.onerror = (err) => console.error('[WS] Error:', err);
}

// ── 切换会话 ─────────────────────────────────────────────

async function switchSession(newSessionId, title = '') {
    closeWebSocket();

    sessionId = newSessionId;
    reconnectAttempts = 0;

    // 清空聊天区
    $chatMessages.innerHTML = `
        <div class="welcome-message">
            <p>🐴 你好！我是<strong>伯乐</strong>，你的 AI 职业助手，可以帮你：</p>
            <div class="quick-actions">
                <button class="quick-btn" data-msg="我想学习生成式AI，请推荐学习路径">📚 学习路径</button>
                <button class="quick-btn" data-msg="我想优化我的简历，请帮我分析">📄 简历优化</button>
                <button class="quick-btn" data-msg="我想准备AI面试，有哪些常见面试题？">🎯 面试准备</button>
                <button class="quick-btn" data-msg="帮我搜索远程AI工程师岗位">🔍 搜索岗位</button>
            </div>
        </div>`;
    hideHitlPanel();
    removeThinking();
    endStream();
    $resumePreviewBar.style.display = 'none';
    pendingResumeText = '';
    $jdInput.value = '';

    $currentSessionTitle.textContent = title || '新对话';
    $sessionBadge.textContent = `${userId} / ${newSessionId.slice(0, 6)}`;
    renderSidebar();

    connectWebSocket();
}

// ── 发送消息 ─────────────────────────────────────────────

function sendMessage() {
    const text = $messageInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    addMessage('user', text);
    ws.send(JSON.stringify({ type: 'message', content: text }));

    $messageInput.value = '';
    $messageInput.style.height = 'auto';
    $sendBtn.disabled = true;

    const wm = document.querySelector('.welcome-message');
    if (wm) wm.style.display = 'none';
}

$sendBtn.addEventListener('click', sendMessage);
$messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
$messageInput.addEventListener('input', () => {
    $messageInput.style.height = 'auto';
    $messageInput.style.height = Math.min($messageInput.scrollHeight, 120) + 'px';
    $sendBtn.disabled = !$messageInput.value.trim();
});

// ── 快捷操作 ─────────────────────────────────────────────
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('quick-btn')) {
        const msg = e.target.dataset.msg;
        if (msg && ws && ws.readyState === WebSocket.OPEN) {
            addMessage('user', msg);
            ws.send(JSON.stringify({ type: 'message', content: msg }));
            const wm = document.querySelector('.welcome-message');
            if (wm) wm.style.display = 'none';
        }
    }
});

// ── 欢迎屏 → 聊天屏 ─────────────────────────────────────

function checkLoginBtnState() {
    $startBtn.disabled = !($userIdInput.value.trim() && $passwordInput.value.trim());
}

$userIdInput.addEventListener('input', checkLoginBtnState);
$passwordInput.addEventListener('input', checkLoginBtnState);
$userIdInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { $passwordInput.focus(); e.preventDefault(); }
});
$passwordInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && $userIdInput.value.trim() && $passwordInput.value.trim()) enterChat();
});
$startBtn.addEventListener('click', enterChat);

async function enterChat() {
    const uid = $userIdInput.value.trim();
    const pwd = $passwordInput.value.trim();
    if (!uid || !pwd) return;

    // 隐藏之前的错误
    $loginError.style.display = 'none';
    $startBtn.disabled = true;

    try {
        // 调用登录 API
        const loginResp = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: uid, password: pwd }),
        });

        if (!loginResp.ok) {
            const err = await loginResp.json().catch(() => ({}));
            throw new Error(err.detail || '登录失败');
        }

        const loginData = await loginResp.json();
        authToken = loginData.token;
        userId = loginData.user_id;

        // 存入 sessionStorage（刷新页面保持登录）
        sessionStorage.setItem('genai-token', authToken);
        sessionStorage.setItem('genai-userId', userId);

        $welcomeScreen.classList.remove('active');
        $chatScreen.classList.add('active');

        await fetchSessions();
        let target;
        if (sessionsCache.length > 0) {
            target = sessionsCache[0];
        } else {
            target = await createNewSession();
            sessionsCache = [target];
        }
        await switchSession(target.session_id, target.title);
    } catch (e) {
        $loginError.textContent = e.message;
        $loginError.style.display = 'block';
        $startBtn.disabled = false;
    }
}

// ── 游客体验 ─────────────────────────────────────────────

const $guestBtn = document.getElementById('guest-btn');
$guestBtn.addEventListener('click', async () => {
    $guestBtn.disabled = true;
    $loginError.style.display = 'none';

    try {
        const resp = await fetch('/api/guest-login', { method: 'POST' });
        if (!resp.ok) throw new Error('游客登录失败');

        const data = await resp.json();
        authToken = data.token;
        userId = data.user_id;

        sessionStorage.setItem('genai-token', authToken);
        sessionStorage.setItem('genai-userId', userId);

        $welcomeScreen.classList.remove('active');
        $chatScreen.classList.add('active');

        const session = await createNewSession();
        sessionsCache = [session];
        await switchSession(session.session_id, session.title);
    } catch (e) {
        $loginError.textContent = e.message;
        $loginError.style.display = 'block';
    } finally {
        $guestBtn.disabled = false;
    }
});

// ── 简历上传 ─────────────────────────────────────────────

$btnUpload.addEventListener('click', () => {
    $resumeFileInput.value = '';
    $resumeFileInput.click();
});

$resumeFileInput.addEventListener('change', async () => {
    const file = $resumeFileInput.files[0];
    if (!file) return;

    $resumeFilename.textContent = file.name;
    $resumeChars.textContent = '解析中...';
    $resumePreviewBar.style.display = 'flex';
    $btnSendResume.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const resp = await fetch('/api/upload-resume', {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || '上传失败');
        }

        const result = await resp.json();
        pendingResumeText = result.text;

        $resumeFilename.textContent = result.filename;
        $resumeChars.textContent = `${result.char_count.toLocaleString()} 字`;
        $btnSendResume.disabled = false;

    } catch (e) {
        $resumePreviewBar.style.display = 'none';
        pendingResumeText = '';
        addMessage('error', `简历上传失败：${e.message}`);
    }
});

$btnSendResume.addEventListener('click', () => {
    if (!pendingResumeText || !ws || ws.readyState !== WebSocket.OPEN) return;

    const filename = $resumeFilename.textContent;
    const jdText = ($jdInput.value || '').trim();

    let userMsg = `请帮我分析并完善这份简历（文件：${filename}）：\n\n【简历内容】\n${pendingResumeText}`;
    if (jdText) {
        userMsg += `\n\n【目标岗位 JD】\n${jdText}`;
    }

    let displayMsg = `📄 已上传简历：${filename}（${$resumeChars.textContent}）`;
    if (jdText) {
        const jdPreview = jdText.length > 60 ? jdText.slice(0, 60) + '…' : jdText;
        displayMsg += `\n📋 岗位JD：${jdPreview}`;
    }
    displayMsg += '\n\n请帮我分析并针对该岗位给出改进建议。';
    addMessage('user', displayMsg);

    ws.send(JSON.stringify({ type: 'message', content: userMsg }));

    $resumePreviewBar.style.display = 'none';
    pendingResumeText = '';
    $jdInput.value = '';

    const wm = document.querySelector('.welcome-message');
    if (wm) wm.style.display = 'none';
});

$btnCancelResume.addEventListener('click', () => {
    $resumePreviewBar.style.display = 'none';
    pendingResumeText = '';
    $jdInput.value = '';
});

// ── 退出 ─────────────────────────────────────────────────

function logout() {
    // 通知后端销毁 token
    if (authToken) {
        fetch('/api/logout', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` },
        }).catch(() => {});
    }
    closeWebSocket();
    endStream();
    $chatScreen.classList.remove('active');
    $welcomeScreen.classList.add('active');
    hideHitlPanel();
    removeThinking();
    $resumePreviewBar.style.display = 'none';
    pendingResumeText = '';
    $jdInput.value = '';
    setConnectionStatus('connecting');
    userId = '';
    sessionId = '';
    sessionsCache = [];
    reconnectAttempts = 0;
    authToken = '';
    sessionStorage.removeItem('genai-token');
    sessionStorage.removeItem('genai-userId');
    $userIdInput.value = '';
    $passwordInput.value = '';
    $loginError.style.display = 'none';
    $startBtn.disabled = true;
}

$logoutBtn.addEventListener('click', logout);

// ── 画像 Modal ────────────────────────────────────────────

const $profileModal   = document.getElementById('profile-modal');
const $btnProfile     = document.getElementById('btn-profile');
const $profileClose   = document.getElementById('profile-modal-close');
const $profileCancel  = document.getElementById('profile-cancel-btn');
const $profileSave    = document.getElementById('profile-save-btn');
const $profileSaveMsg = document.getElementById('profile-save-msg');

// Tag 输入框状态
const profileTags = { skills: [], interests: [] };

function renderTags(field) {
    const container = document.getElementById(`pf-${field}-tags`);
    container.innerHTML = profileTags[field].map(tag =>
        `<span class="profile-tag">${escapeHtml(tag)}<button class="tag-rm" data-field="${field}" data-tag="${escapeHtml(tag)}">×</button></span>`
    ).join('');
}

function addTag(field, raw) {
    raw.split(/[,，\n]/).map(s => s.trim()).filter(Boolean).forEach(tag => {
        if (!profileTags[field].includes(tag)) {
            profileTags[field].push(tag);
        }
    });
    renderTags(field);
}

function removeTag(field, tag) {
    profileTags[field] = profileTags[field].filter(t => t !== tag);
    renderTags(field);
}

// 标签点击删除
['skills', 'interests'].forEach(field => {
    document.getElementById(`pf-${field}-tags`).addEventListener('click', e => {
        const btn = e.target.closest('.tag-rm');
        if (btn) removeTag(field, btn.dataset.tag);
    });

    const input = document.getElementById(`pf-${field}-input`);
    input.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            const val = input.value.trim();
            if (val) { addTag(field, val); input.value = ''; }
        }
    });
    input.addEventListener('blur', () => {
        const val = input.value.trim();
        if (val) { addTag(field, val); input.value = ''; }
    });
});

// 点击遮罩关闭
$profileModal.addEventListener('click', e => {
    if (e.target === $profileModal) closeProfileModal();
});

function openProfileModal() {
    // 先立即显示弹窗，避免等待 API 的卡顿感
    $profileSaveMsg.textContent = '加载中…';
    $profileSaveMsg.style.color = 'var(--text-muted)';
    $profileModal.style.display = 'flex';

    fetch(`/api/users/${encodeURIComponent(userId)}/profile`, {
        headers: { 'Authorization': `Bearer ${authToken}` },
    })
        .then(r => r.json())
        .then(data => {
            const p = data.profile || {};

            document.getElementById('pf-name').value           = p.name || '';
            document.getElementById('pf-target-role').value    = p.target_role || '';
            document.getElementById('pf-skill-level').value    = p.skill_level || '';
            document.getElementById('pf-years').value          = p.years_of_experience ?? '';
            document.getElementById('pf-background').value     = p.background || '';
            document.getElementById('pf-location').value       = p.preferred_location || '';
            document.getElementById('pf-work-type').value      = p.preferred_work_type || '';

            profileTags.skills    = Array.isArray(p.skills)    ? [...p.skills]    : [];
            profileTags.interests = Array.isArray(p.interests) ? [...p.interests] : [];
            renderTags('skills');
            renderTags('interests');

            $profileSaveMsg.textContent = '';
            document.getElementById('pf-name').focus();
        })
        .catch(() => {
            $profileSaveMsg.textContent = '加载失败，请重试';
            $profileSaveMsg.style.color = 'var(--red)';
        });
}

function closeProfileModal() {
    $profileModal.style.display = 'none';
}

async function saveProfile() {
    $profileSave.disabled = true;
    $profileSaveMsg.textContent = '保存中…';

    const yearsRaw = document.getElementById('pf-years').value;
    const profile = {
        name:                 document.getElementById('pf-name').value.trim(),
        target_role:          document.getElementById('pf-target-role').value.trim(),
        skill_level:          document.getElementById('pf-skill-level').value || null,
        years_of_experience:  yearsRaw !== '' ? parseInt(yearsRaw, 10) : null,
        background:           document.getElementById('pf-background').value.trim(),
        preferred_location:   document.getElementById('pf-location').value.trim(),
        preferred_work_type:  document.getElementById('pf-work-type').value || null,
        skills:               [...profileTags.skills],
        interests:            [...profileTags.interests],
    };

    // 去除空值（后端也会过滤，但减少传输）
    Object.keys(profile).forEach(k => {
        if (profile[k] === '' || profile[k] === null) delete profile[k];
    });

    try {
        const resp = await fetch(`/api/users/${encodeURIComponent(userId)}/profile`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${authToken}` },
            body: JSON.stringify(profile),
        });
        if (!resp.ok) throw new Error('保存失败');
        $profileSaveMsg.textContent = '✓ 已保存';
        $profileSaveMsg.style.color = 'var(--green)';
        setTimeout(closeProfileModal, 800);
    } catch (e) {
        $profileSaveMsg.textContent = e.message;
        $profileSaveMsg.style.color = 'var(--red)';
    } finally {
        $profileSave.disabled = false;
    }
}

$btnProfile.addEventListener('click', openProfileModal);
$profileClose.addEventListener('click', closeProfileModal);
$profileCancel.addEventListener('click', closeProfileModal);
$profileSave.addEventListener('click', saveProfile);

// ── 页面刷新自动恢复登录 ─────────────────────────────────
(async function tryAutoRestore() {
    const savedToken = sessionStorage.getItem('genai-token');
    const savedUid   = sessionStorage.getItem('genai-userId');
    if (!savedToken || !savedUid) return;

    // 试探 token 是否仍有效（用 sessions API 做探针）
    try {
        const resp = await fetch(`/api/users/${encodeURIComponent(savedUid)}/sessions`, {
            headers: { 'Authorization': `Bearer ${savedToken}` },
        });
        if (!resp.ok) throw new Error('token expired');

        authToken = savedToken;
        userId = savedUid;
        $welcomeScreen.classList.remove('active');
        $chatScreen.classList.add('active');

        const data = await resp.json();
        sessionsCache = data.sessions || [];
        let target;
        if (sessionsCache.length > 0) {
            target = sessionsCache[0];
        } else {
            target = await createNewSession();
            sessionsCache = [target];
        }
        await switchSession(target.session_id, target.title);
    } catch {
        // token 失效→清理，显示登录页
        sessionStorage.removeItem('genai-token');
        sessionStorage.removeItem('genai-userId');
    }
})();
