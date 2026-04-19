/**
 * GenAI 职业助手 — 前端逻辑（流式输出版）
 *
 * 功能：
 *   1. WebSocket 连接管理 + 自动重连
 *   2. 流式 token 渲染（逐字显示 AI 回复）
 *   3. 节点进度指示（显示当前处理阶段）
 *   4. HITL 审核交互
 *   5. Markdown 渲染 + 自动滚动
 */

// ── DOM 元素 ─────────────────────────────────────────────
const $welcomeScreen    = document.getElementById('welcome-screen');
const $chatScreen       = document.getElementById('chat-screen');
const $userIdInput      = document.getElementById('user-id-input');
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

// 简历上传相关
const $btnUpload        = document.getElementById('btn-upload');
const $resumeFileInput  = document.getElementById('resume-file-input');
const $resumePreviewBar = document.getElementById('resume-preview-bar');
const $resumeFilename   = document.getElementById('resume-filename');
const $resumeChars      = document.getElementById('resume-chars');
const $btnSendResume    = document.getElementById('btn-send-resume');
const $btnCancelResume  = document.getElementById('btn-cancel-resume');
const $jdInput          = document.getElementById('jd-input');

// ── 状态 ─────────────────────────────────────────────────
let ws = null;
let sessionId = '';
let reconnectAttempts = 0;
const MAX_RECONNECT = 5;

// 流式相关
let streamingBubble = null;     // 当前正在流式填充的消息气泡
let streamingRawText = '';      // 流式累积的原始文本
let isStreaming = false;

// 简历上传相关
let pendingResumeText = '';     // 已解析的简历文本，等待用户确认发送

// ── 工具函数 ─────────────────────────────────────────────

function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
        try { return marked.parse(text); } catch { }
    }
    return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        $chatMessages.scrollTop = $chatMessages.scrollHeight;
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

// ── 消息渲染 ─────────────────────────────────────────────

/** 添加一条完整消息（非流式） */
function addMessage(role, content) {
    removeThinking();

    const wrapper = document.createElement('div');
    wrapper.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = role === 'user' ? '🙋' : role === 'error' ? '⚠️' : '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = role === 'user'
        ? content.replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>')
        : renderMarkdown(content);

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    $chatMessages.appendChild(wrapper);
    scrollToBottom();
}

// ── 流式消息渲染 ─────────────────────────────────────────

/** 创建一个空的 AI 消息气泡，准备流式填充 */
function createStreamBubble() {
    removeThinking();
    endStream(); // 结束之前未完成的流

    const wrapper = document.createElement('div');
    wrapper.className = 'message ai';
    wrapper.id = 'streaming-message';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble streaming';
    bubble.innerHTML = '<span class="stream-cursor"></span>';

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);
    $chatMessages.appendChild(wrapper);

    streamingBubble = bubble;
    streamingRawText = '';
    isStreaming = true;
    scrollToBottom();
}

/** 向流式气泡追加 token */
function appendToken(token) {
    if (!streamingBubble || !isStreaming) {
        createStreamBubble();
    }

    streamingRawText += token;

    // 渲染完整 Markdown + 追加光标
    streamingBubble.innerHTML = renderMarkdown(streamingRawText) + '<span class="stream-cursor"></span>';
    scrollToBottom();
}

/** 结束流式输出，移除光标 */
function endStream() {
    if (streamingBubble && isStreaming) {
        // 最终渲染
        if (streamingRawText) {
            streamingBubble.innerHTML = renderMarkdown(streamingRawText);
        }
        streamingBubble.classList.remove('streaming');
        const cursor = streamingBubble.querySelector('.stream-cursor');
        if (cursor) cursor.remove();
    }
    streamingBubble = null;
    streamingRawText = '';
    isStreaming = false;

    // 移除流式 message ID
    const el = document.getElementById('streaming-message');
    if (el) el.removeAttribute('id');
}

// ── 进度 & Thinking 指示器 ─────────────────────────────────

/** 显示节点进度（替换 thinking indicator 的文本） */
function showNodeProgress(label) {
    let el = document.getElementById('thinking-el');
    if (!el) {
        showThinking(label);
        return;
    }
    const textEl = el.querySelector('.thinking-label');
    if (textEl) textEl.textContent = label;
}

/** 显示 thinking indicator */
function showThinking(label) {
    removeThinking();

    const el = document.createElement('div');
    el.className = 'thinking-indicator';
    el.id = 'thinking-el';

    const av = document.createElement('div');
    av.className = 'msg-avatar';
    av.textContent = '🤖';
    av.style.cssText = 'background:linear-gradient(135deg,rgba(129,140,248,.2),rgba(192,132,252,.2));border:1px solid rgba(129,140,248,.3);border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-size:.85rem;flex-shrink:0';

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

/** 移除 thinking indicator */
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

// ── WebSocket 连接 ───────────────────────────────────────

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws/${encodeURIComponent(sessionId)}`;

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
                console.log('[WS] Connected, session:', data.session_id);
                break;

            case 'chat_history':
                // 恢复历史对话 — 隐藏欢迎消息，渲染历史记录
                if (data.messages && data.messages.length > 0) {
                    const wm = document.querySelector('.welcome-message');
                    if (wm) wm.style.display = 'none';

                    data.messages.forEach(msg => {
                        addMessage(msg.role, msg.content);
                    });

                    // 分隔线
                    const sep = document.createElement('div');
                    sep.className = 'history-separator';
                    sep.innerHTML = '<span>以上为历史对话</span>';
                    $chatMessages.appendChild(sep);
                    scrollToBottom();
                }
                break;

            // ── 流式事件 ──────────────────────
            case 'stream_start':
                showThinking('正在处理...');
                setInputEnabled(false);
                break;

            case 'node_progress':
                // 如果已经在流式输出 token，不再显示 thinking（保留流式气泡）
                if (!isStreaming) {
                    showNodeProgress(data.label);
                }
                break;

            case 'ai_token':
                // 收到第一个 token 时移除 thinking，创建流式气泡
                removeThinking();
                appendToken(data.token);
                break;

            case 'ai_message':
                // 完整消息（来自 fallback 或 HITL 回复）
                removeThinking();
                if (isStreaming) {
                    // 如果正在流式中收到完整消息，追加到流式
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
                break;

            // ── HITL 事件 ─────────────────────
            case 'hitl_request':
                removeThinking();
                endStream();
                showHitlPanel(data.instruction);
                break;

            // ── 其他 ─────────────────────────
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

// ── 发送消息 ─────────────────────────────────────────────

function sendMessage() {
    const text = $messageInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    addMessage('user', text);
    ws.send(JSON.stringify({ type: 'message', content: text }));

    $messageInput.value = '';
    $messageInput.style.height = 'auto';
    $sendBtn.disabled = true;

    // 隐藏欢迎消息
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

$userIdInput.addEventListener('input', () => {
    $startBtn.disabled = !$userIdInput.value.trim();
});
$userIdInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && $userIdInput.value.trim()) enterChat();
});
$startBtn.addEventListener('click', enterChat);

function enterChat() {
    sessionId = $userIdInput.value.trim();
    if (!sessionId) return;
    $sessionBadge.textContent = `ID: ${sessionId}`;
    $welcomeScreen.classList.remove('active');
    $chatScreen.classList.add('active');
    connectWebSocket();
}

// ── 简历上传 ─────────────────────────────────────────────

$btnUpload.addEventListener('click', () => {
    $resumeFileInput.value = '';   // 允许重复上传同一文件
    $resumeFileInput.click();
});

$resumeFileInput.addEventListener('change', async () => {
    const file = $resumeFileInput.files[0];
    if (!file) return;

    // 显示上传中状态
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

    // 构建发送给后端的完整消息
    let userMsg = `请帮我分析并完善这份简历（文件：${filename}）：\n\n【简历内容】\n${pendingResumeText}`;
    if (jdText) {
        userMsg += `\n\n【目标岗位 JD】\n${jdText}`;
    }

    // 聊天区显示的摘要（不暴露全文）
    let displayMsg = `📄 已上传简历：${filename}（${$resumeChars.textContent}）`;
    if (jdText) {
        const jdPreview = jdText.length > 60 ? jdText.slice(0, 60) + '…' : jdText;
        displayMsg += `\n📋 岗位JD：${jdPreview}`;
    }
    displayMsg += '\n\n请帮我分析并针对该岗位给出改进建议。';
    addMessage('user', displayMsg);

    ws.send(JSON.stringify({ type: 'message', content: userMsg }));

    // 隐藏面板、清空状态
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
$logoutBtn.addEventListener('click', () => {
    if (ws) ws.close();
    endStream();
    $chatScreen.classList.remove('active');
    $welcomeScreen.classList.add('active');
    $chatMessages.innerHTML = `
        <div class="welcome-message">
            <p>👋 你好！我是 <strong>GenAI 职业助手</strong>，可以帮你：</p>
            <div class="quick-actions">
                <button class="quick-btn" data-msg="我想学习生成式AI，请推荐学习路径">📚 学习 GenAI</button>
                <button class="quick-btn" data-msg="帮我制作一份AI工程师的简历">📄 制作简历</button>
                <button class="quick-btn" data-msg="我想准备AI面试，有哪些常见面试题？">🎯 面试准备</button>
                <button class="quick-btn" data-msg="帮我搜索远程AI工程师岗位">🔍 搜索岗位</button>
            </div>
        </div>`;
    hideHitlPanel();
    removeThinking();
    $resumePreviewBar.style.display = 'none';
    pendingResumeText = '';
    $jdInput.value = '';
    setConnectionStatus('connecting');
    sessionId = '';
    reconnectAttempts = 0;
});
