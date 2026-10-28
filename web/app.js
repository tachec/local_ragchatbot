/* ============================================================
   知源 · 本地知识库助手 —— 前端交互
   纯原生 JS：会话、流式 SSE、Markdown、文本/图片附件、图片搜库、主题
   ============================================================ */
(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    booted: false,
    user: null,
    sessions: [],
    messages: [],
    activeSessionId: null,
    activeSession: null,
    attachments: [],
    docs: { docs: [], total_docs: 0, total_chunks: 0, vector_rows: 0 },
    search: '',
    streaming: false,
    abortController: null,
    liveIndex: null,
    stickToBottom: true,
    theme: document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light',
    dragDepth: 0,
    uploadTimer: null,
  };

  const el = {};
  let liveRaf = null;
  let toastTimer = null;
  let confirmCleanup = null;

  const TYPE_LABEL = { pdf: 'PDF', md: 'Markdown', txt: '文本', image: '图片' };
  const TYPE_ICON = { pdf: 'file', md: 'file', txt: 'file', image: 'image' };
  const ALLOWED_EXT = new Set(['.pdf', '.txt', '.md', '.markdown', '.log', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']);
  const MAX_UPLOAD = 50 * 1024 * 1024;
  const MAX_ATTACHMENTS = 5;
  const MAX_TEXT_ATTACHMENT = 2 * 1024 * 1024;
  const MAX_IMAGE_ATTACHMENT = 8 * 1024 * 1024;
  const TEXT_ATTACHMENT_EXT = new Set([
    '.txt', '.md', '.markdown', '.csv', '.json', '.log', '.yaml', '.yml', '.xml',
    '.html', '.htm', '.css', '.js', '.ts', '.py', '.java', '.c', '.cpp', '.h',
    '.sql', '.ini', '.conf', '.text',
  ]);
  const IMAGE_ATTACHMENT_EXT = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']);

  document.addEventListener('DOMContentLoaded', init);

  function init() {
    cacheEls();
    bindEvents();
    applyTheme(state.theme, false);
    const platform = navigator.userAgentData?.platform || navigator.platform || '';
    if (/Mac|iPhone|iPad/i.test(platform)) {
      const shortcut = $('.shortcut');
      if (shortcut) shortcut.textContent = '⌘ K';
    }
    autoGrow();
    updateSendState();
    loadBootstrap();
  }

  function cacheEls() {
    el.sidebar = $('#sidebar');
    el.sidebarScrim = $('#sidebar-scrim');
    el.menuBtn = $('#menu-btn');
    el.sidebarClose = $('#sidebar-close');
    el.newChatBtn = $('#new-chat-btn');
    el.brandLink = $('#brand-link');
    el.sessionSearch = $('#session-search');
    el.searchClear = $('#search-clear');
    el.sessionList = $('#session-list');
    el.sessionEmpty = $('#session-empty');
    el.sessionCount = $('#session-count');
    el.kbEntry = $('#kb-open-btn');
    el.kbEntryMeta = $('#kb-entry-meta');
    el.userAvatar = $('#user-avatar');
    el.userLabel = $('#user-label');
    el.userId = $('#user-id');
    el.themeSide = $('#theme-btn-side');
    el.themeTop = $('#theme-btn-top');

    el.chatTitle = $('#chat-title');
    el.chatStatus = $('#chat-status');
    el.chatStatusText = $('#chat-status-text');
    el.chatScroll = $('#chat-scroll');
    el.chatColumn = $('.chat-column');
    el.welcome = $('#welcome');
    el.heroMeta = $('#hero-meta');
    el.heroMetaText = $('#hero-meta-text');
    el.suggestions = $('#suggestions');
    el.messages = $('#messages');
    el.composerInput = $('#composer-input');
    el.composer = $('#composer');
    el.sendBtn = $('#send-btn');
    el.stopBtn = $('#stop-btn');
    el.attachmentTray = $('#attachment-tray');
    el.textFileBtn = $('#text-file-btn');
    el.imageFileBtn = $('#image-file-btn');
    el.textFileInput = $('#text-file-input');
    el.imageFileInput = $('#image-file-input');
    el.charCount = $('#char-count');
    el.scrollBottom = $('#scroll-bottom');

    el.kbBtnTop = $('#kb-open-btn-top');
    el.kbBadge = $('#kb-count-badge');
    el.kbScrim = $('#kb-scrim');
    el.kbPanel = $('#kb-panel');
    el.kbClose = $('#kb-close-btn');
    el.dropzone = $('#dropzone');
    el.browseBtn = $('#browse-btn');
    el.fileInput = $('#file-input');
    el.uploadQueue = $('#upload-queue');
    el.docList = $('#doc-list');
    el.docEmpty = $('#doc-empty');
    el.kbListCount = $('#kb-list-count');
    el.statDocs = $('#stat-docs');
    el.statChunks = $('#stat-chunks');
    el.statStatus = $('#stat-status');

    el.toast = $('#toast');
    el.confirmDialog = $('#confirm-dialog');
    el.confirmTitle = $('#confirm-title');
    el.confirmMessage = $('#confirm-message');
    el.confirmOk = $('#confirm-ok');
    el.confirmCancel = $('#confirm-cancel');
    el.sourceDialog = $('#source-dialog');
    el.sourceKicker = $('#source-kicker');
    el.sourceTitle = $('#source-title');
    el.sourceMeta = $('#source-meta');
    el.sourceSnippet = $('#source-snippet');
    el.sourceCloseIcon = $('#source-close-icon');
    el.sourceDone = $('#source-done');
    el.dragOverlay = $('#drag-overlay');
  }

  function bindEvents() {
    el.newChatBtn.addEventListener('click', newChat);
    el.brandLink.addEventListener('click', (e) => { e.preventDefault(); newChat(); });
    el.menuBtn.addEventListener('click', openSidebar);
    el.sidebarClose.addEventListener('click', closeSidebar);
    el.sidebarScrim.addEventListener('click', closeSidebar);

    el.sessionSearch.addEventListener('input', () => {
      state.search = el.sessionSearch.value.trim();
      el.searchClear.hidden = !state.search;
      renderSessions();
    });
    el.sessionSearch.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        el.sessionSearch.value = '';
        state.search = '';
        el.searchClear.hidden = true;
        renderSessions();
        el.sessionSearch.blur();
      }
    });
    el.searchClear.addEventListener('click', () => {
      el.sessionSearch.value = '';
      state.search = '';
      el.searchClear.hidden = true;
      renderSessions();
      el.sessionSearch.focus();
    });

    el.sessionList.addEventListener('click', onSessionClick);
    el.sessionList.addEventListener('keydown', onSessionKeydown);

    el.kbEntry.addEventListener('click', openKb);
    el.kbBtnTop.addEventListener('click', openKb);
    el.kbClose.addEventListener('click', closeKb);
    el.kbScrim.addEventListener('click', closeKb);

    el.browseBtn.addEventListener('click', (e) => { e.stopPropagation(); el.fileInput.click(); });
    el.dropzone.addEventListener('click', () => el.fileInput.click());
    el.dropzone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.fileInput.click(); }
    });
    el.fileInput.addEventListener('change', () => {
      const files = Array.from(el.fileInput.files || []);
      el.fileInput.value = '';
      if (files.length) uploadFiles(files);
    });

    el.textFileBtn.addEventListener('click', () => {
      if (!state.streaming) el.textFileInput.click();
    });
    el.imageFileBtn.addEventListener('click', () => {
      if (!state.streaming) el.imageFileInput.click();
    });
    el.textFileInput.addEventListener('change', async () => {
      const files = Array.from(el.textFileInput.files || []);
      el.textFileInput.value = '';
      if (files.length) await addAttachmentFiles(files, 'text');
    });
    el.imageFileInput.addEventListener('change', async () => {
      const files = Array.from(el.imageFileInput.files || []);
      el.imageFileInput.value = '';
      if (files.length) await addAttachmentFiles(files, 'image');
    });
    el.attachmentTray.addEventListener('click', (e) => {
      const remove = e.target.closest('[data-remove-attachment]');
      if (!remove) return;
      state.attachments = state.attachments.filter((item) => item.id !== remove.dataset.removeAttachment);
      renderAttachmentTray();
      updateSendState();
    });

    el.themeSide.addEventListener('click', toggleTheme);
    el.themeTop.addEventListener('click', toggleTheme);

    el.composerInput.addEventListener('input', () => {
      autoGrow();
      updateSendState();
    });
    el.composerInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        sendMessage();
      }
    });
    el.composer.addEventListener('click', (e) => {
      if (e.target === el.composer) el.composerInput.focus();
    });
    el.sendBtn.addEventListener('click', () => sendMessage());
    el.stopBtn.addEventListener('click', stopStreaming);

    el.chatScroll.addEventListener('scroll', onChatScroll, { passive: true });
    el.scrollBottom.addEventListener('click', () => { state.stickToBottom = true; scrollToBottom(); });

    el.suggestions.addEventListener('click', onSuggestionClick);
    el.messages.addEventListener('click', onMessagesClick);

    el.docList.addEventListener('click', onDocClick);

    el.confirmCancel.addEventListener('click', () => finishConfirm(false));
    el.confirmDialog.addEventListener('cancel', (e) => { e.preventDefault(); finishConfirm(false); });
    el.confirmDialog.addEventListener('click', (e) => { if (e.target === el.confirmDialog) finishConfirm(false); });

    el.sourceCloseIcon.addEventListener('click', () => el.sourceDialog.close());
    el.sourceDone.addEventListener('click', () => el.sourceDialog.close());
    el.sourceDialog.addEventListener('click', (e) => { if (e.target === el.sourceDialog) el.sourceDialog.close(); });

    window.addEventListener('keydown', onGlobalKeydown);
    window.addEventListener('resize', onResize);
    window.addEventListener('dragenter', onDragEnter);
    window.addEventListener('dragover', onDragOver);
    window.addEventListener('dragleave', onDragLeave);
    window.addEventListener('drop', onDrop);
  }

  /* ============================== 数据加载 ============================== */
  async function loadBootstrap() {
    try {
      const res = await fetch('/api/bootstrap', { headers: { Accept: 'application/json' } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      state.booted = true;
      state.user = data.user || { label: '本地访客', short_id: '--------', initial: '访' };
      state.sessions = Array.isArray(data.sessions) ? data.sessions : [];
      state.docs = normalizeDocs(data.docs);
      if (data.active && data.active.session) {
        state.activeSession = data.active.session;
        state.activeSessionId = data.active.session.id;
        state.messages = Array.isArray(data.active.messages) ? data.active.messages : [];
      } else {
        state.activeSession = null;
        state.activeSessionId = null;
        state.messages = [];
      }
      renderUser();
      renderAll();
      scrollToBottom(false);
      if (window.innerWidth > 980) el.composerInput.focus();
    } catch (err) {
      console.error(err);
      showBootError();
    }
  }

  function normalizeDocs(raw) {
    const data = raw || {};
    return {
      docs: Array.isArray(data.docs) ? data.docs : [],
      total_docs: Number(data.total_docs || 0),
      total_chunks: Number(data.total_chunks || 0),
      vector_rows: Number(data.vector_rows || 0),
    };
  }

  function showBootError() {
    state.booted = false;
    el.welcome.classList.remove('hidden');
    const h2 = $('h2', el.welcome);
    const copy = $('.hero-copy', el.welcome);
    if (h2) h2.textContent = '暂时无法连接服务';
    if (copy) copy.textContent = '请确认 Python 服务仍在运行，然后刷新页面重试。';
    el.heroMeta.classList.add('empty');
    el.heroMetaText.textContent = '连接失败';
    el.suggestions.innerHTML = `
      <button class="suggestion" type="button" data-action="retry">
        <span class="suggestion-icon">${icon('refresh')}</span>
        <strong>重新连接</strong>
        <span>刷新并再次尝试</span>
      </button>`;
    el.messages.innerHTML = '';
    renderSessions();
    updateHeader();
  }

  /* ============================== 渲染：用户 / 顶栏 ============================== */
  function renderUser() {
    const user = state.user || {};
    el.userAvatar.textContent = user.initial || '访';
    el.userLabel.textContent = user.label || '本地访客';
    el.userId.textContent = `ID · ${user.short_id || '--------'}`;
  }

  function updateHeader() {
    const title = state.activeSession?.title || '新对话';
    el.chatTitle.textContent = title;
    document.title = title === '新对话' ? '知源 · 本地知识库助手' : `${title} · 知源`;
    el.chatStatus.classList.toggle('busy', state.streaming);
    el.chatStatus.classList.toggle('empty', !state.streaming && state.docs.total_docs === 0);
    if (state.streaming) {
      el.chatStatusText.textContent = '正在生成回答…';
    } else if (state.docs.total_docs > 0) {
      el.chatStatusText.textContent = `知识库就绪 · ${state.docs.total_docs} 份资料`;
    } else {
      el.chatStatusText.textContent = '知识库为空 · 可直接通用问答';
    }
  }

  function renderAll() {
    renderSessions();
    renderDocs();
    renderWelcome();
    renderMessages();
    updateHeader();
    updateSendState();
  }

  /* ============================== 渲染：欢迎页 ============================== */
  function renderWelcome() {
    const hasMessages = state.messages.length > 0;
    el.welcome.classList.toggle('hidden', hasMessages);
    if (hasMessages) return;

    const hasDocs = state.docs.total_docs > 0;
    el.heroMeta.classList.toggle('empty', !hasDocs);
    el.heroMetaText.textContent = hasDocs
      ? `已连接 ${state.docs.total_docs} 份资料 · ${state.docs.total_chunks} 个片段`
      : '知识库为空 · 也可以直接进行通用问答';

    const items = hasDocs
      ? [
          { icon: 'sparkles', title: '总结知识库', desc: '用三句话概括现有资料', task: 'summarize' },
          { icon: 'list', title: '提炼关键结论', desc: '分条列出并标注来源', task: 'key_points' },
          { icon: 'compare', title: '对比不同资料', desc: '梳理相同点与差异', task: 'compare' },
        ]
      : [
          { icon: 'upload', title: '上传第一份资料', desc: 'PDF / Markdown / TXT / 图片', action: 'upload' },
          { icon: 'sparkles', title: '你可以做什么？', desc: '了解知源的能力与用法', prompt: '请介绍一下你的能力和使用方式。' },
          { icon: 'message', title: '试试通用问答', desc: '没有资料也可以先聊聊', prompt: '你好，请简单介绍一下你自己。' },
        ];

    el.suggestions.innerHTML = items.map((item) => {
      // 有 task 的卡片走知识库整体分析接口，不能把标题当普通问题去做 RAG 检索
      const attrs = item.action
        ? `data-action="${item.action}"`
        : item.task
          ? `data-task="${item.task}" data-label="${escapeHtml(item.title)}"`
          : `data-prompt="${escapeHtml(item.prompt)}"`;
      return `
      <button class="suggestion" type="button" ${attrs}>
        <span class="suggestion-icon">${icon(item.icon)}</span>
        <strong>${escapeHtml(item.title)}</strong>
        <span>${escapeHtml(item.desc)}</span>
        ${icon('chevron-right', 'suggestion-arrow')}
      </button>`;
    }).join('');
  }

  function onSuggestionClick(e) {
    const btn = e.target.closest('.suggestion');
    if (!btn) return;
    if (btn.dataset.action === 'retry') {
      window.location.reload();
      return;
    }
    if (btn.dataset.action === 'upload') {
      openKb();
      el.fileInput.click();
      return;
    }
    if (btn.dataset.task) {
      sendMessage(btn.dataset.label || '知识库分析', { task: btn.dataset.task });
      return;
    }
    if (btn.dataset.prompt) sendMessage(btn.dataset.prompt);
  }

  /* ============================== 消息附件 ============================== */
  async function addAttachmentFiles(files, kind) {
    if (state.streaming) return;
    const room = MAX_ATTACHMENTS - state.attachments.length;
    if (room <= 0) {
      toast(`最多同时添加 ${MAX_ATTACHMENTS} 个附件`, 'error');
      return;
    }
    const selected = files.slice(0, room);
    if (files.length > room) toast(`最多同时添加 ${MAX_ATTACHMENTS} 个附件`, 'error');

    for (const file of selected) {
      const ext = extensionOf(file.name);
      try {
        if (kind === 'text') {
          if (!TEXT_ATTACHMENT_EXT.has(ext)) throw new Error('不是支持的文本文件');
          if (file.size > MAX_TEXT_ATTACHMENT) throw new Error('文本文件超过 2 MB');
          const content = (await readTextFile(file)).replace(/\u0000/g, '').trim();
          if (!content) throw new Error('文件内容为空或无法解码');
          addAttachment({
            id: attachmentId(),
            type: 'text',
            name: file.name,
            size: file.size,
            content,
          });
        } else {
          if (!IMAGE_ATTACHMENT_EXT.has(ext)) throw new Error('不是支持的图片格式');
          if (file.size > MAX_IMAGE_ATTACHMENT) throw new Error('图片超过 8 MB');
          const dataUrl = await readFileAsDataURL(file);
          addAttachment({
            id: attachmentId(),
            type: 'image',
            name: file.name,
            size: file.size,
            dataUrl,
          });
        }
      } catch (err) {
        toast(`${file.name}：${err.message || '读取失败'}`, 'error');
      }
    }
    renderAttachmentTray();
    updateSendState();
  }

  function addAttachment(item) {
    const used = state.attachments.reduce((sum, a) => sum + Number(a.size || 0), 0);
    if (used + Number(item.size || 0) > 16 * 1024 * 1024) {
      throw new Error('附件总大小超过 16 MB');
    }
    state.attachments.push(item);
  }

  function renderAttachmentTray() {
    if (!state.attachments.length) {
      el.attachmentTray.hidden = true;
      el.attachmentTray.innerHTML = '';
      return;
    }
    el.attachmentTray.hidden = false;
    el.attachmentTray.innerHTML = state.attachments
      .map((item) => attachmentChipHtml(item, { removable: true }))
      .join('');
  }

  function attachmentChipHtml(item, { removable = false } = {}) {
    const isImage = item.type === 'image';
    const hasError = Boolean(item.error);
    let meta;
    if (hasError) {
      meta = `解析失败：${item.error}`;
    } else if (isImage) {
      meta = item.description || item.preview
        ? `${formatBytes(item.size)} · 已解析，将检索知识库`
        : `${formatBytes(item.size)} · 将检索知识库`;
    } else {
      meta = `${formatBytes(item.size)} · 文本${item.truncated ? ' · 已截断' : ''}`;
    }
    const thumb = isImage && item.dataUrl
      ? `<img class="attachment-thumb" src="${escapeHtml(item.dataUrl)}" alt="">`
      : `<span class="attachment-icon">${icon(isImage ? 'image' : 'file')}</span>`;
    const remove = removable
      ? `<button class="attachment-remove" type="button" data-remove-attachment="${escapeHtml(item.id)}"
                 aria-label="移除附件" title="移除附件">${icon('close')}</button>`
      : '';
    const preview = item.preview || item.description || (item.type === 'text' ? item.content : '');
    return `
      <div class="attachment-chip ${isImage ? 'image' : 'text'}${hasError ? ' has-error' : ''}"
           title="${escapeHtml(String(preview || item.name || '').slice(0, 400))}">
        ${thumb}
        <span class="attachment-copy">
          <span class="attachment-name" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
          <span class="attachment-meta" title="${escapeHtml(meta)}">${escapeHtml(meta)}</span>
        </span>
        ${remove}
      </div>`;
  }

  function clearAttachments() {
    state.attachments = [];
    renderAttachmentTray();
    updateSendState();
  }

  function readTextFile(file) {
    return file.arrayBuffer().then((buffer) => {
      try {
        return new TextDecoder('utf-8', { fatal: true }).decode(buffer);
      } catch (err) {
        try {
          return new TextDecoder('gb18030').decode(buffer);
        } catch (err2) {
          return new TextDecoder('utf-8').decode(buffer);
        }
      }
    });
  }

  function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(new Error('图片读取失败'));
      reader.readAsDataURL(file);
    });
  }

  let attachmentSeq = 0;
  function attachmentId() {
    attachmentSeq += 1;
    if (window.crypto?.randomUUID) return window.crypto.randomUUID();
    return `att-${Date.now()}-${attachmentSeq}`;
  }

  function formatBytes(value) {
    const size = Number(value || 0);
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / 1024 / 1024).toFixed(1)} MB`;
  }

  /* ============================== 渲染：会话列表 ============================== */
  function renderSessions() {
    const query = state.search.toLowerCase();
    const filtered = query
      ? state.sessions.filter((s) => (s.title || '新对话').toLowerCase().includes(query))
      : state.sessions;

    el.sessionCount.textContent = filtered.length ? String(filtered.length) : '';
    el.sessionCount.hidden = filtered.length === 0;
    el.sessionEmpty.hidden = filtered.length > 0;

    if (!filtered.length) {
      const title = el.sessionEmpty.querySelector('span');
      const small = el.sessionEmpty.querySelector('small');
      if (query) {
        if (title) title.textContent = '没有匹配的对话';
        if (small) small.textContent = '试试其他关键词';
      } else {
        if (title) title.textContent = '还没有对话';
        if (small) small.textContent = '发送第一条消息，开始记录';
      }
      el.sessionList.innerHTML = '';
      return;
    }

    const groups = [
      { label: '今天', items: [] },
      { label: '昨天', items: [] },
      { label: '更早', items: [] },
    ];
    const now = new Date();
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
    const yesterdayStart = todayStart - 86400;

    filtered.forEach((s) => {
      const ts = Number(s.updated_at || s.created_at || 0);
      if (ts >= todayStart) groups[0].items.push(s);
      else if (ts >= yesterdayStart) groups[1].items.push(s);
      else groups[2].items.push(s);
    });

    const html = [];
    groups.forEach((group) => {
      if (!group.items.length) return;
      html.push(`<div class="session-group-label">${group.label}</div>`);
      group.items.forEach((s) => {
        const active = s.id === state.activeSessionId;
        html.push(`
          <div class="session-item${active ? ' active' : ''}" data-session-id="${escapeHtml(s.id)}"
               role="button" tabindex="0" aria-current="${active ? 'true' : 'false'}">
            <div class="session-main">
              <span class="session-title" title="${escapeHtml(s.title || '新对话')}">${escapeHtml(s.title || '新对话')}</span>
              <span class="session-time">${formatSessionTime(s.updated_at || s.created_at)}</span>
            </div>
            <button class="session-delete" type="button" data-delete-session="${escapeHtml(s.id)}"
                    aria-label="删除对话" title="删除对话">${icon('trash')}</button>
          </div>`);
      });
    });
    el.sessionList.innerHTML = html.join('');
    if (window.innerWidth > 980 || el.sidebar.classList.contains('open')) {
      requestAnimationFrame(() => {
        const active = el.sessionList.querySelector('.session-item.active');
        if (active) active.scrollIntoView({ block: 'nearest' });
      });
    }
  }

  function onSessionClick(e) {
    const del = e.target.closest('[data-delete-session]');
    if (del) {
      e.stopPropagation();
      deleteSession(del.dataset.deleteSession);
      return;
    }
    const item = e.target.closest('[data-session-id]');
    if (item) selectSession(item.dataset.sessionId);
  }

  function onSessionKeydown(e) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    if (e.target.closest('button')) return;
    const item = e.target.closest('[data-session-id]');
    if (!item) return;
    e.preventDefault();
    selectSession(item.dataset.sessionId);
  }

  function selectSession(id) {
    if (!id || state.streaming) {
      if (state.streaming) toast('请先停止当前回答，再切换对话');
      return;
    }
    if (id === state.activeSessionId && state.messages.length) {
      closeSidebar();
      return;
    }
    fetch(`/api/sessions/${encodeURIComponent(id)}`, { headers: { Accept: 'application/json' } })
      .then(async (res) => {
        if (!res.ok) throw new Error((await safeJson(res))?.detail || '会话加载失败');
        return res.json();
      })
      .then((data) => {
        state.activeSession = data.session;
        state.activeSessionId = data.session.id;
        state.messages = Array.isArray(data.messages) ? data.messages : [];
        state.stickToBottom = true;
        clearAttachments();
        closeSidebar();
        renderAll();
        scrollToBottom(false);
        if (window.innerWidth > 980) el.composerInput.focus();
      })
      .catch((err) => toast(err.message || '会话加载失败', 'error'));
  }

  async function newChat() {
    if (state.streaming) {
      toast('请先停止当前回答，再新建对话');
      return;
    }
    state.activeSession = null;
    state.activeSessionId = null;
    state.messages = [];
    state.liveIndex = null;
    state.stickToBottom = true;
    state.search = '';
    clearAttachments();
    el.sessionSearch.value = '';
    el.searchClear.hidden = true;
    closeSidebar();
    closeKb();
    renderAll();
    history.replaceState(null, '', '/');
    if (window.innerWidth > 980) el.composerInput.focus();
  }

  async function deleteSession(id) {
    if (!id) return;
    const session = state.sessions.find((s) => s.id === id);
    const title = session?.title || '这条对话';
    const ok = await confirmDialog('删除对话', `确定删除「${title}」吗？删除后无法恢复。`, '删除');
    if (!ok) return;
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' });
      const data = await safeJson(res);
      if (!res.ok) throw new Error(data?.detail || '删除失败');
      state.sessions = Array.isArray(data.sessions) ? data.sessions : state.sessions.filter((s) => s.id !== id);
      if (state.activeSessionId === id) {
        state.activeSession = null;
        state.activeSessionId = null;
        state.messages = [];
        clearAttachments();
        renderAll();
      } else {
        renderSessions();
      }
      toast('对话已删除');
    } catch (err) {
      toast(err.message || '删除失败', 'error');
    }
  }

  async function refreshSessions() {
    try {
      const res = await fetch('/api/sessions', { headers: { Accept: 'application/json' } });
      if (!res.ok) return;
      const data = await res.json();
      state.sessions = Array.isArray(data.sessions) ? data.sessions : [];
      if (state.activeSessionId) {
        const active = state.sessions.find((s) => s.id === state.activeSessionId);
        if (active) state.activeSession = { ...state.activeSession, ...active };
      }
      renderSessions();
      updateHeader();
    } catch (err) {
      /* 静默失败：不影响当前对话 */
    }
  }

  /* ============================== 渲染：消息 ============================== */
  function renderMessages() {
    el.messages.innerHTML = state.messages.map((msg, index) => messageHtml(msg, index)).join('');
    if (state.messages.length) el.welcome.classList.add('hidden');
  }

  function messageHtml(msg, index) {
    if (msg.role === 'user') {
      const attachments = Array.isArray(msg.attachments) ? msg.attachments : [];
      const chips = attachments.length
        ? `<div class="user-attachments">${attachments.map((item) => attachmentChipHtml(item)).join('')}</div>`
        : '';
      const text = escapeHtml(msg.content || '').replace(/\n/g, '<br>');
      const bubble = text ? `<div class="user-bubble">${text}</div>` : '';
      return `
        <article class="message user" data-index="${index}">
          <div class="user-stack">${chips}${bubble}</div>
        </article>`;
    }

    const pending = Boolean(msg.pending);
    const hasContent = Boolean(msg.content);
    let body;
    if (!hasContent && pending) {
      body = `<div class="typing"><span class="typing-dots"><i></i><i></i><i></i></span><em>${escapeHtml(msg.status || '正在思考…')}</em></div>`;
    } else {
      const cursor = pending ? '<span class="cursor"></span>' : '';
      body = renderMarkdown(msg.content || '') + cursor;
    }

    const modeNote = msg.mode === 'general'
      ? `<div class="mode-note">${icon('info')}<span>知识库资料不足，以下回答基于模型通用知识，请谨慎采信。</span></div>`
      : '';
    const overviewNote = msg.mode === 'overview'
      ? `<div class="mode-note">${icon('sparkles')}<span>以下回答基于知识库全部资料汇总整理，而非单次检索片段。</span></div>`
      : '';
    const errorNote = msg.errorText
      ? `<div class="mode-note error">${icon('info')}<span>${escapeHtml(msg.errorText)}</span></div>`
      : '';
    const sources = Array.isArray(msg.sources) ? msg.sources : [];
    const imageQuery = Boolean(msg.imageQuery || msg.image_query);
    const imageNote = imageQuery
      ? `<div class="mode-note attachment">${icon('image')}<span>${sources.length
        ? `已根据图片解析结果检索知识库，命中 ${sources.length} 条资料。`
        : '图片已解析，但知识库中未找到足够相关的资料。'}</span></div>`
      : '';
    const attachmentNote = msg.mode === 'attachment' && !imageQuery
      ? `<div class="mode-note attachment">${icon('file')}<span>以下回答结合了你上传的附件内容。</span></div>`
      : '';
    const sourcesHtml = sources.length && (msg.mode || !pending) ? sourceBlock(sources) : '';

    return `
      <article class="message assistant${state.streaming && state.liveIndex === index ? ' live' : ''}" data-index="${index}">
        <div class="assistant-avatar">${icon('logo', 'logo-icon')}</div>
        <div class="assistant-body">
          <div class="assistant-head">
            <span class="assistant-name">知源</span>
            <span class="message-time">${formatClock(msg.created_at)}</span>
            <button class="msg-copy" type="button" data-copy-message="${index}" aria-label="复制回答" title="复制回答">${icon('copy')}</button>
          </div>
          <div class="answer">${body}</div>
          ${errorNote}
          ${modeNote}
          ${overviewNote}
          ${imageNote}
          ${attachmentNote}
          ${sourcesHtml}
          ${!pending && hasContent ? `<div class="answer-actions"><button class="msg-copy" type="button" data-copy-message="${index}" aria-label="复制回答" title="复制回答">${icon('copy')}</button></div>` : ''}
        </div>
      </article>`;
  }

  function sourceBlock(sources) {
    return `
      <div class="sources-block">
        <div class="sources-head">${icon('link')}<span>参考资料</span><span class="sources-count">${sources.length}</span></div>
        <div class="source-grid">
          ${sources.map((s, i) => {
            const page = Number(s.page) > 0 ? `第 ${s.page} 页` : '全文';
            const score = Number(s.score) > 0 ? `相关度 ${Number(s.score).toFixed(2)}` : '';
            const meta = [page, `片段 ${s.chunk_idx ?? 0}`, score].filter(Boolean).join(' · ');
            return `
              <button class="source-card" type="button" data-source-open="${i}">
                <span class="source-num">${s.n || i + 1}</span>
                <span class="source-main">
                  <strong title="${escapeHtml(s.doc_name || '未知文档')}">${escapeHtml(s.doc_name || '未知文档')}</strong>
                  <span>${escapeHtml(meta)}</span>
                </span>
                ${icon('chevron-right')}
              </button>`;
          }).join('')}
        </div>
      </div>`;
  }

  function updateLiveMessage() {
    const index = state.liveIndex;
    if (index == null) return;
    const msg = state.messages[index];
    if (!msg) return;
    const current = el.messages.querySelector(`[data-index="${index}"]`);
    if (!current) { renderMessages(); return; }
    const stick = state.stickToBottom;
    current.outerHTML = messageHtml(msg, index);
    if (stick) scrollToBottom();
  }

  function scheduleLiveRender() {
    if (liveRaf) return;
    liveRaf = requestAnimationFrame(() => {
      liveRaf = null;
      updateLiveMessage();
    });
  }

  function onMessagesClick(e) {
    const copyBtn = e.target.closest('[data-copy-message]');
    if (copyBtn) {
      const msg = state.messages[Number(copyBtn.dataset.copyMessage)];
      if (msg) copyWithFeedback(copyBtn, msg.content || '');
      return;
    }
    const codeCopy = e.target.closest('[data-copy-code]');
    if (codeCopy) {
      const pre = codeCopy.closest('pre');
      const code = pre?.querySelector('code');
      if (code) copyWithFeedback(codeCopy, code.innerText, '复制');
      return;
    }
    const cite = e.target.closest('[data-cite]');
    if (cite) {
      const article = cite.closest('.message');
      const index = Number(article?.dataset.index);
      const n = Number(cite.dataset.cite);
      openSourceByN(index, n);
      return;
    }
    const sourceCard = e.target.closest('[data-source-open]');
    if (sourceCard) {
      const article = sourceCard.closest('.message');
      const index = Number(article?.dataset.index);
      const sourceIndex = Number(sourceCard.dataset.sourceOpen);
      openSource(index, sourceIndex);
    }
  }

  function copyWithFeedback(button, text, restoreLabel) {
    copyText(text).then(() => {
      const old = button.innerHTML;
      button.innerHTML = `${icon('check')}${restoreLabel ? `<span>已复制</span>` : ''}`;
      toast('已复制到剪贴板');
      setTimeout(() => { button.innerHTML = old; }, 1400);
    }).catch(() => toast('复制失败，请手动选择文本', 'error'));
  }

  async function copyText(text) {
    if (!text) return;
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
  }

  /* ============================== 发送 / 流式 ============================== */
  async function sendMessage(override, options = {}) {
    if (!state.booted) {
      toast('服务尚未连接，请刷新页面', 'error');
      return;
    }
    if (state.streaming) return;
    const task = options.task || null;
    const text = String(override ?? el.composerInput.value ?? '').trim();
    // 整体分析任务不使用输入框附件；普通问答照常携带
    const outgoing = task ? [] : state.attachments.slice();
    if (!text && !outgoing.length) return;

    const payloadAttachments = outgoing.map((item) => (
      item.type === 'text'
        ? { type: 'text', name: item.name, size: item.size, content: item.content }
        : { type: 'image', name: item.name, size: item.size, data_url: item.dataUrl }
    ));
    const userAttachments = outgoing.map((item) => ({
      type: item.type,
      name: item.name,
      size: item.size,
      dataUrl: item.dataUrl,
      preview: item.type === 'text' ? String(item.content || '').slice(0, 240) : undefined,
    }));

    el.composerInput.value = '';
    autoGrow();
    if (!task) state.attachments = [];
    renderAttachmentTray();
    state.streaming = true;
    state.stickToBottom = true;
    el.scrollBottom.hidden = true;
    updateSendState();
    updateHeader();

    const now = Date.now() / 1000;
    state.messages.push({ role: 'user', content: text, attachments: userAttachments, created_at: now });
    state.messages.push({
      role: 'assistant',
      content: '',
      sources: [],
      mode: null,
      pending: true,
      status: task
        ? '正在整理知识库资料…'
        : outgoing.length ? '正在解析附件并检索知识库…' : '正在检索知识库…',
      created_at: now,
    });
    state.liveIndex = state.messages.length - 1;
    const pendingTitle = titleFromMessage(text || userAttachments[0]?.name || '新对话');

    // 只追加新消息，避免每次发送都重建整段对话（否则历史消息会闪一下）
    const userIndex = state.messages.length - 2;
    el.welcome.classList.add('hidden');
    el.messages.insertAdjacentHTML(
      'beforeend',
      messageHtml(state.messages[userIndex], userIndex)
        + messageHtml(state.messages[state.liveIndex], state.liveIndex),
    );
    updateHeader();
    updateSendState();
    scrollToBottom(false);

    const controller = new AbortController();
    state.abortController = controller;
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({
          message: text,
          session_id: state.activeSessionId,
          attachments: payloadAttachments,
          task: task || undefined,
        }),
        signal: controller.signal,
      });
      if (!res.ok) {
        const data = await safeJson(res);
        throw new Error(data?.detail || `请求失败（${res.status}）`);
      }
      await readSSE(res, (event) => handleStreamEvent(event, pendingTitle));
    } catch (err) {
      if (err.name === 'AbortError') {
        const msg = state.messages[state.liveIndex];
        if (msg) {
          msg.pending = false;
          msg.status = '';
          if (!msg.content) msg.content = '（已停止生成）';
          msg.errorText = '';
        }
        toast('已停止生成');
      } else {
        console.error(err);
        const msg = state.messages[state.liveIndex];
        if (msg) {
          msg.pending = false;
          msg.status = '';
          if (!msg.content) msg.content = '抱歉，请求失败，请稍后重试。';
          msg.errorText = err.message || '网络请求失败';
        }
        toast(err.message || '请求失败，请稍后重试', 'error');
      }
      updateLiveMessage();
    } finally {
      if (state.liveIndex != null) {
        const msg = state.messages[state.liveIndex];
        if (msg) { msg.pending = false; msg.status = ''; }
        updateLiveMessage();
      }
      state.streaming = false;
      state.abortController = null;
      updateSendState();
      updateHeader();
      if (state.stickToBottom) scrollToBottom();
      state.liveIndex = null;
      await refreshSessions();
      if (window.innerWidth > 980) el.composerInput.focus();
    }
  }

  function handleStreamEvent(event, pendingTitle) {
    if (!event || !event.type) return;
    const index = state.liveIndex;
    if (index == null) return;
    const msg = state.messages[index];
    if (!msg) return;

    if (event.type === 'status') {
      msg.status = event.text || '正在处理…';
      scheduleLiveRender();
      return;
    }

    if (event.type === 'meta') {
      if (event.session_id && !state.activeSessionId) {
        state.activeSessionId = event.session_id;
        state.activeSession = {
          id: event.session_id,
          title: pendingTitle,
          created_at: Date.now() / 1000,
          updated_at: Date.now() / 1000,
          message_count: 2,
        };
        if (!state.sessions.some((s) => s.id === event.session_id)) {
          state.sessions.unshift(state.activeSession);
        }
        renderSessions();
        updateHeader();
      }
      msg.mode = event.mode || null;
      msg.sources = Array.isArray(event.sources) ? event.sources : [];
      msg.imageQuery = Boolean(event.image_query);
      msg.status = '';
      if (Array.isArray(event.attachments)) {
        const userIndex = index - 1;
        const userMsg = state.messages[userIndex];
        if (userMsg?.role === 'user') {
          const previous = Array.isArray(userMsg.attachments) ? userMsg.attachments : [];
          userMsg.attachments = event.attachments.map((item, attIndex) => ({
            ...(previous[attIndex] || {}),
            ...item,
            dataUrl: previous[attIndex]?.dataUrl,
          }));
          const userEl = el.messages.querySelector(`[data-index="${userIndex}"]`);
          if (userEl) userEl.outerHTML = messageHtml(userMsg, userIndex);
          const failed = event.attachments.filter((item) => item.error);
          if (failed.length) {
            toast(`附件解析失败：${failed[0].name}`, 'error');
          }
        }
      }
      scheduleLiveRender();
      return;
    }

    if (event.type === 'delta') {
      msg.content += event.text || '';
      msg.pending = true;
      msg.status = '';
      scheduleLiveRender();
      return;
    }

    if (event.type === 'error') {
      msg.errorText = event.text || '生成失败';
      if (!msg.content) msg.content = event.text || '生成失败，请稍后重试。';
      scheduleLiveRender();
      return;
    }

    if (event.type === 'done') {
      msg.pending = false;
      msg.status = '';
      if (event.session) {
        state.activeSession = event.session;
        state.activeSessionId = event.session.id;
      }
      if (Array.isArray(event.sources)) msg.sources = event.sources;
      if (event.mode) msg.mode = event.mode;
      updateLiveMessage();
      updateHeader();
      if (state.stickToBottom) scrollToBottom();
    }
  }

  async function readSSE(response, onEvent) {
    if (!response.body) throw new Error('浏览器不支持流式响应');
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) >= 0) {
        const chunk = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        for (const rawLine of chunk.split('\n')) {
          const line = rawLine.replace(/\r$/, '');
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload) continue;
          try {
            onEvent(JSON.parse(payload));
          } catch (err) {
            console.warn('无法解析流事件', payload, err);
          }
        }
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) {
      for (const rawLine of buffer.split('\n')) {
        const line = rawLine.replace(/\r$/, '');
        if (!line.startsWith('data:')) continue;
        try { onEvent(JSON.parse(line.slice(5).trim())); } catch (err) { /* ignore */ }
      }
    }
  }

  function stopStreaming() {
    if (state.abortController) state.abortController.abort();
  }

  /* ============================== Markdown 渲染 ============================== */
  function renderMarkdown(src) {
    const text = String(src || '').replace(/\r\n?/g, '\n');
    if (!text.trim()) return '';
    const lines = text.split('\n');
    const out = [];
    let i = 0;
    let codeLang = '';
    let codeLines = [];
    let inCode = false;

    const flushCode = () => {
      const code = codeLines.join('\n');
      out.push(`<pre><div class="code-head"><span>${escapeHtml(codeLang || 'code')}</span><button class="code-copy" type="button" data-copy-code>${icon('copy')}<span>复制</span></button></div><code>${escapeHtml(code)}</code></pre>`);
      codeLines = [];
      codeLang = '';
    };

    while (i < lines.length) {
      const line = lines[i];

      if (/^```/.test(line.trim())) {
        if (inCode) {
          inCode = false;
          flushCode();
        } else {
          inCode = true;
          codeLang = line.trim().slice(3).trim();
        }
        i += 1;
        continue;
      }
      if (inCode) {
        codeLines.push(line);
        i += 1;
        continue;
      }
      if (!line.trim()) { i += 1; continue; }

      let m = line.match(/^(#{1,4})\s+(.+)$/);
      if (m) {
        const level = m[1].length;
        out.push(`<h${level}>${inline(m[2])}</h${level}>`);
        i += 1;
        continue;
      }

      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
        out.push('<hr>');
        i += 1;
        continue;
      }

      if (line.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
        const header = splitTableRow(line);
        const aligns = splitTableRow(lines[i + 1]).map((cell) => {
          const t = cell.trim();
          if (t.startsWith(':') && t.endsWith(':')) return 'center';
          if (t.endsWith(':')) return 'right';
          return 'left';
        });
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
          rows.push(splitTableRow(lines[i]));
          i += 1;
        }
        const th = header.map((cell, idx) => `<th style="text-align:${aligns[idx] || 'left'}">${inline(cell)}</th>`).join('');
        const body = rows.map((row) => `<tr>${header.map((_, idx) => `<td style="text-align:${aligns[idx] || 'left'}">${inline(row[idx] || '')}</td>`).join('')}</tr>`).join('');
        out.push(`<table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table>`);
        continue;
      }

      if (/^>\s?/.test(line)) {
        const quoted = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) {
          quoted.push(lines[i].replace(/^>\s?/, ''));
          i += 1;
        }
        out.push(`<blockquote>${inline(quoted.join('\n')).replace(/\n/g, '<br>')}</blockquote>`);
        continue;
      }

      if (/^\s*([-*+]|\d+\.)\s+/.test(line)) {
        const ordered = /^\s*\d+\./.test(line);
        const items = [];
        while (i < lines.length && /^\s*([-*+]|\d+\.)\s+/.test(lines[i])) {
          items.push(`<li>${inline(lines[i].replace(/^\s*([-*+]|\d+\.)\s+/, ''))}</li>`);
          i += 1;
        }
        out.push(`<${ordered ? 'ol' : 'ul'}>${items.join('')}</${ordered ? 'ol' : 'ul'}>`);
        continue;
      }

      const para = [line];
      i += 1;
      while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) {
        para.push(lines[i]);
        i += 1;
      }
      out.push(`<p>${inline(para.join('\n')).replace(/\n/g, '<br>')}</p>`);
    }

    if (inCode) flushCode();
    return out.join('');
  }

  function isBlockStart(line) {
    return /^```/.test(line.trim())
      || /^(#{1,4})\s+/.test(line)
      || /^>\s?/.test(line)
      || /^\s*([-*+]|\d+\.)\s+/.test(line)
      || /^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)
      || (line.includes('|') && /^\s*\|?[\s:|-]+\|[\s:|-]*\s*$/.test(line) && line.includes('-'));
  }

  function isTableSeparator(line) {
    if (!line || !line.includes('-')) return false;
    const cells = splitTableRow(line);
    return cells.length > 0 && cells.every((cell) => /^\s*:?-{2,}:?\s*$/.test(cell));
  }

  function splitTableRow(line) {
    let text = line.trim();
    if (text.startsWith('|')) text = text.slice(1);
    if (text.endsWith('|')) text = text.slice(0, -1);
    return text.split('|').map((cell) => cell.trim());
  }

  function inline(raw) {
    const codes = [];
    let text = escapeHtml(String(raw ?? ''));
    text = text.replace(/`([^`\n]+)`/g, (match, code) => {
      codes.push(code);
      return `\u0000${codes.length - 1}\u0000`;
    });
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
    text = text.replace(/~~([^~]+)~~/g, '<del>$1</del>');
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    text = text.replace(/【来源\s*(\d+)】/g, '<button class="cite" type="button" data-cite="$1">来源 $1</button>');
    text = text.replace(/【附件[:：]\s*([^】]+)】/g, '<span class="cite file">附件 · $1</span>');
    text = text.replace(/【来源[:：]\s*([^】]+)】/g, '<span class="cite file">来源 · $1</span>');
    text = text.replace(/\u0000(\d+)\u0000/g, (match, index) => `<code>${codes[Number(index)] ?? ''}</code>`);
    return text;
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[char]));
  }

  function icon(name, extraClass = '') {
    return `<svg class="icon ${extraClass}" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
  }

  /* ============================== 知识库 ============================== */
  function openKb() {
    closeSidebar();
    el.kbPanel.classList.add('open');
    el.kbPanel.setAttribute('aria-hidden', 'false');
    el.kbScrim.hidden = false;
    setTimeout(() => el.browseBtn.focus({ preventScroll: true }), 280);
    fetchDocs();
  }

  function closeKb() {
    el.kbPanel.classList.remove('open');
    el.kbPanel.setAttribute('aria-hidden', 'true');
    el.kbScrim.hidden = true;
  }

  async function fetchDocs() {
    try {
      const res = await fetch('/api/docs', { headers: { Accept: 'application/json' } });
      if (!res.ok) return;
      state.docs = normalizeDocs(await res.json());
      renderDocs();
      renderWelcome();
      updateHeader();
    } catch (err) { /* 静默 */ }
  }

  function renderDocs() {
    const docs = state.docs.docs || [];
    const total = state.docs.total_docs || 0;
    const chunks = state.docs.total_chunks || 0;

    el.statDocs.textContent = String(total);
    el.statChunks.textContent = String(chunks);
    el.statStatus.textContent = '就绪';
    el.kbBadge.textContent = String(total);
    el.kbBadge.hidden = total === 0;
    el.kbEntryMeta.textContent = total ? `${total} 份资料 · ${chunks} 个片段` : '还没有资料';
    el.kbListCount.textContent = total ? `${total} 份` : '';
    el.docList.hidden = docs.length === 0;
    el.docEmpty.hidden = docs.length > 0;

    el.docList.innerHTML = docs.map((doc) => {
      const type = String(doc.src_type || 'txt');
      const label = TYPE_LABEL[type] || '文件';
      const iconName = TYPE_ICON[type] || 'file';
      const date = formatShortDate(doc.created_at);
      return `
        <div class="doc-item">
          <span class="doc-icon ${escapeHtml(type)}">${icon(iconName)}</span>
          <div class="doc-info">
            <strong title="${escapeHtml(doc.doc_name)}">${escapeHtml(doc.doc_name)}</strong>
            <span>${label} · ${Number(doc.n_chunks || 0)} 个片段 · ${date}</span>
          </div>
          <button class="doc-delete" type="button" data-delete-doc="${escapeHtml(doc.doc_name)}"
                  aria-label="删除资料" title="删除资料">${icon('trash')}</button>
        </div>`;
    }).join('');
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    openKb();

    const accepted = [];
    const queue = [];
    files.forEach((file, index) => {
      const ext = extensionOf(file.name);
      if (!ALLOWED_EXT.has(ext)) {
        queue.push({ id: `q${Date.now()}-${index}`, name: file.name, state: 'error', text: '不支持的文件类型' });
        return;
      }
      if (file.size > MAX_UPLOAD) {
        queue.push({ id: `q${Date.now()}-${index}`, name: file.name, state: 'error', text: '超过 50 MB' });
        return;
      }
      const item = { id: `q${Date.now()}-${index}`, name: file.name, state: 'uploading', text: '解析并入库…' };
      queue.push(item);
      accepted.push({ file, item });
    });

    if (!accepted.length) {
      renderQueue(queue);
      scheduleQueueClear(queue.length);
      toast('没有可上传的文件', 'error');
      return;
    }

    renderQueue(queue);
    el.dropzone.classList.add('busy');
    const form = new FormData();
    accepted.forEach(({ file }) => form.append('files', file, file.name));

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      const data = await safeJson(res);
      if (!res.ok) throw new Error(data?.detail || '上传失败');
      const results = Array.isArray(data.results) ? data.results : [];
      accepted.forEach(({ item }, index) => {
        const result = results[index];
        if (result && result.ok) {
          item.state = 'done';
          item.text = `${result.chunks} 个片段`;
        } else {
          item.state = 'error';
          item.text = result?.error || '入库失败';
        }
      });
      state.docs = normalizeDocs(data);
      renderDocs();
      renderWelcome();
      updateHeader();
      renderQueue(queue);
      const errors = Array.isArray(data.errors) ? data.errors : [];
      if (errors.length) {
        toast(`${errors.length} 个文件处理失败，请查看列表`, 'error');
      } else {
        toast(`已入库 ${accepted.length} 个文件`);
      }
      scheduleQueueClear(queue.length, 5200);
    } catch (err) {
      accepted.forEach(({ item }) => {
        item.state = 'error';
        item.text = err.message || '上传失败';
      });
      renderQueue(queue);
      scheduleQueueClear(queue.length, 7000);
      toast(err.message || '上传失败，请稍后重试', 'error');
    } finally {
      el.dropzone.classList.remove('busy');
    }
  }

  function renderQueue(queue) {
    el.uploadQueue.hidden = false;
    el.uploadQueue.innerHTML = queue.map((item) => {
      if (item.state === 'uploading') {
        return `<div class="upload-item">
          <span class="spinner"></span>
          <span class="item-name">${escapeHtml(item.name)}</span>
          <span class="item-state">${escapeHtml(item.text)}</span>
        </div>`;
      }
      if (item.state === 'done') {
        return `<div class="upload-item done">
          <span class="ok-mark">${icon('check')}</span>
          <span class="item-name">${escapeHtml(item.name)}</span>
          <span class="item-state">${escapeHtml(item.text)}</span>
        </div>`;
      }
      return `<div class="upload-item error">
        ${icon('info')}
        <span class="item-name">${escapeHtml(item.name)}</span>
        <span class="item-state">${escapeHtml(item.text)}</span>
      </div>`;
    }).join('');
  }

  function scheduleQueueClear(count, delay = 4200) {
    if (state.uploadTimer) clearTimeout(state.uploadTimer);
    state.uploadTimer = setTimeout(() => {
      el.uploadQueue.hidden = true;
      el.uploadQueue.innerHTML = '';
      state.uploadTimer = null;
    }, delay);
  }

  function onDocClick(e) {
    const btn = e.target.closest('[data-delete-doc]');
    if (!btn) return;
    deleteDocument(btn.dataset.deleteDoc);
  }

  async function deleteDocument(name) {
    if (!name) return;
    const ok = await confirmDialog('删除资料', `将从知识库中移除《${name}》及其全部分块，此操作不可撤销。`, '删除');
    if (!ok) return;
    try {
      const res = await fetch(`/api/docs?name=${encodeURIComponent(name)}`, { method: 'DELETE' });
      const data = await safeJson(res);
      if (!res.ok) throw new Error(data?.detail || '删除失败');
      state.docs = normalizeDocs(data);
      renderDocs();
      renderWelcome();
      updateHeader();
      toast(data.message || '资料已删除');
    } catch (err) {
      toast(err.message || '删除失败', 'error');
    }
  }

  /* ============================== 确认弹窗 / 来源弹窗 ============================== */
  function confirmDialog(title, message, okLabel = '确认') {
    if (confirmCleanup) confirmCleanup();
    return new Promise((resolve) => {
      el.confirmTitle.textContent = title;
      el.confirmMessage.textContent = message;
      el.confirmOk.textContent = okLabel;
      let settled = false;
      const done = (value) => {
        if (settled) return;
        settled = true;
        el.confirmDialog.close();
        confirmCleanup = null;
        resolve(value);
      };
      confirmCleanup = () => done(false);
      el.confirmOk.onclick = () => done(true);
      el.confirmCancel.onclick = () => done(false);
      el.confirmDialog.showModal();
    });
  }

  function finishConfirm(value) {
    if (confirmCleanup) confirmCleanup();
    else if (el.confirmDialog.open) el.confirmDialog.close();
  }

  function openSourceByN(messageIndex, n) {
    const msg = state.messages[messageIndex];
    if (!msg || !Array.isArray(msg.sources)) return;
    const idx = msg.sources.findIndex((source) => Number(source.n) === Number(n));
    openSource(messageIndex, idx >= 0 ? idx : n - 1);
  }

  function openSource(messageIndex, sourceIndex) {
    const msg = state.messages[messageIndex];
    const source = msg?.sources?.[sourceIndex];
    if (!source) {
      toast('没有找到这条引用', 'error');
      return;
    }
    const n = source.n || sourceIndex + 1;
    el.sourceKicker.textContent = `REFERENCE ${String(n).padStart(2, '0')}`;
    el.sourceTitle.textContent = source.doc_name || '未知文档';
    const page = Number(source.page) > 0 ? `第 ${source.page} 页` : '全文';
    const score = Number(source.score) > 0 ? `相关度 ${Number(source.score).toFixed(2)}` : '相关度未知';
    el.sourceMeta.innerHTML = `
      <span>${escapeHtml(page)}</span>
      <span>片段 ${Number(source.chunk_idx || 0)}</span>
      <span>${escapeHtml(score)}</span>`;
    el.sourceSnippet.textContent = source.snippet
      ? String(source.snippet)
      : '这条历史引用没有保存原文片段。你可以在知识库中搜索该文档，核对完整内容。';
    el.sourceDialog.showModal();
  }

  /* ============================== 主题 / 滚动 / 输入 ============================== */
  function applyTheme(theme, persist = true) {
    const next = theme === 'dark' ? 'dark' : 'light';
    state.theme = next;
    document.documentElement.dataset.theme = next;
    const isDark = next === 'dark';
    $$('#theme-btn-side use, #theme-btn-top use').forEach((use) => {
      use.setAttribute('href', isDark ? '#i-sun' : '#i-moon');
    });
    const meta = $('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', isDark ? '#151513' : '#f7f7f4');
    if (persist) {
      try { localStorage.setItem('zhiyuan-theme', next); } catch (err) { /* ignore */ }
    }
  }

  function toggleTheme() {
    applyTheme(state.theme === 'dark' ? 'light' : 'dark');
  }

  function onChatScroll() {
    const near = isNearBottom(90);
    state.stickToBottom = near;
    el.scrollBottom.hidden = near || state.messages.length === 0;
  }

  function isNearBottom(threshold = 90) {
    const node = el.chatScroll;
    return node.scrollHeight - node.scrollTop - node.clientHeight < threshold;
  }

  function scrollToBottom(smooth = true) {
    el.chatScroll.scrollTo({ top: el.chatScroll.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
    el.scrollBottom.hidden = true;
  }

  function autoGrow() {
    const input = el.composerInput;
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 196)}px`;
  }

  function updateSendState() {
    const hasText = Boolean(el.composerInput.value.trim());
    const hasAttachments = state.attachments.length > 0;
    el.sendBtn.disabled = state.streaming || (!hasText && !hasAttachments);
    el.stopBtn.hidden = !state.streaming;
    el.textFileBtn.disabled = state.streaming;
    el.imageFileBtn.disabled = state.streaming;
    const len = el.composerInput.value.length;
    el.charCount.textContent = len > 3600 ? `${len}/4000` : '';
    el.charCount.style.color = len > 3900 ? 'var(--danger)' : '';
  }

  /* ============================== 侧栏 / 抽屉 ============================== */
  function openSidebar() {
    el.sidebar.classList.add('open');
    el.sidebarScrim.hidden = false;
    closeKb();
  }

  function closeSidebar() {
    el.sidebar.classList.remove('open');
    el.sidebarScrim.hidden = true;
  }

  function onResize() {
    if (window.innerWidth > 980) {
      closeSidebar();
    }
  }

  /* ============================== 全局快捷键 / 拖拽 ============================== */
  function onGlobalKeydown(e) {
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      newChat();
      return;
    }
    if (e.key === 'Escape') {
      if (el.sourceDialog.open) { el.sourceDialog.close(); return; }
      if (el.confirmDialog.open) { finishConfirm(false); return; }
      if (el.kbPanel.classList.contains('open')) { closeKb(); return; }
      if (el.sidebar.classList.contains('open')) { closeSidebar(); return; }
    }
  }

  function hasFiles(e) {
    return Array.from(e.dataTransfer?.types || []).includes('Files');
  }

  function onDragEnter(e) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    state.dragDepth += 1;
    el.dragOverlay.hidden = false;
  }

  function onDragOver(e) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  }

  function onDragLeave(e) {
    if (!hasFiles(e) && el.dragOverlay.hidden) return;
    state.dragDepth = Math.max(0, state.dragDepth - 1);
    if (state.dragDepth === 0) el.dragOverlay.hidden = true;
  }

  function onDrop(e) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    state.dragDepth = 0;
    el.dragOverlay.hidden = true;
    const files = Array.from(e.dataTransfer?.files || []);
    if (files.length) uploadFiles(files);
  }

  /* ============================== 工具函数 ============================== */
  async function safeJson(res) {
    try { return await res.json(); } catch (err) { return null; }
  }

  function extensionOf(name) {
    const idx = String(name || '').lastIndexOf('.');
    return idx >= 0 ? String(name).slice(idx).toLowerCase() : '';
  }

  function titleFromMessage(text) {
    const clean = String(text || '').replace(/\s+/g, ' ').trim();
    return clean.slice(0, 28) || '新对话';
  }

  function formatClock(ts) {
    const value = Number(ts || 0);
    if (!value) return '';
    const date = new Date(value * 1000);
    return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
  }

  function formatSessionTime(ts) {
    const value = Number(ts || 0);
    if (!value) return '';
    const date = new Date(value * 1000);
    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    if (date.getTime() >= startOfToday) return formatClock(value);
    if (date.getTime() >= startOfToday - 86400000) return '昨天';
    return `${date.getMonth() + 1}月${date.getDate()}日`;
  }

  function formatShortDate(ts) {
    const value = Number(ts || 0);
    if (!value) return '';
    const date = new Date(value * 1000);
    const now = new Date();
    if (date.getFullYear() === now.getFullYear()) {
      return `${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
    }
    return `${date.getFullYear()}/${String(date.getMonth() + 1).padStart(2, '0')}/${String(date.getDate()).padStart(2, '0')}`;
  }

  function toast(message, type = 'info') {
    if (!message) return;
    el.toast.textContent = message;
    el.toast.classList.toggle('error', type === 'error');
    el.toast.classList.add('show');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.toast.classList.remove('show'), 3200);
  }
})();
