import * as pdfjsLib from "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.0.379/pdf.min.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc =
  "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.0.379/pdf.worker.min.mjs";

const PDFJS = pdfjsLib;

export function createPdfViewer(options) {
  const {
    pagesEl,
    titleEl,
    metaEl,
    pageLabelEl,
    emptyEl,
    closeBtn,
    zoomInBtn,
    zoomOutBtn,
    zoomLabelEl,
    layoutEl,
    viewportEl,
  } = options;

  const DEFAULT_SCALE = 1.35;
  const MIN_SCALE = 0.5;
  const MAX_SCALE = 3;
  const ZOOM_STEP = 0.25;

  let pdfDoc = null;
  let pdfUrl = null;
  let scale = DEFAULT_SCALE;
  let activeHighlights = [];
  let pageSlots = new Map();
  let observer = null;
  let scrollRaf = null;

  function updateZoomLabel() {
    zoomLabelEl.textContent = `${Math.round(scale * 100)}%`;
    zoomOutBtn.disabled = scale <= MIN_SCALE + 0.001;
    zoomInBtn.disabled = scale >= MAX_SCALE - 0.001;
  }

  function getVisiblePageNumbers() {
    const viewportRect = viewportEl.getBoundingClientRect();
    const pages = [];

    for (const [pageNum, slot] of pageSlots) {
      const rect = slot.wrap.getBoundingClientRect();
      if (rect.bottom >= viewportRect.top - 600 && rect.top <= viewportRect.bottom + 600) {
        pages.push(pageNum);
      }
    }

    return pages;
  }

  async function applyScale(newScale) {
    const clamped = Math.min(MAX_SCALE, Math.max(MIN_SCALE, newScale));
    if (Math.abs(clamped - scale) < 0.001) {
      return;
    }

    if (!pdfDoc || pageSlots.size === 0) {
      scale = clamped;
      updateZoomLabel();
      return;
    }

    const scrollTop = viewportEl.scrollTop;
    const scrollHeight = viewportEl.scrollHeight;
    const clientHeight = viewportEl.clientHeight;
    const scrollRatio =
      scrollHeight > clientHeight ? scrollTop / (scrollHeight - clientHeight) : 0;

    scale = clamped;
    updateZoomLabel();

    const firstPage = await pdfDoc.getPage(1);
    const pageViewport = firstPage.getViewport({ scale });

    for (const slot of pageSlots.values()) {
      slot.renderTask?.cancel();
      slot.renderTask = null;
      slot.rendered = false;
      slot.viewport = null;
      slot.overlay.innerHTML = "";
      slot.wrap.style.width = `${pageViewport.width}px`;
      slot.wrap.style.minHeight = `${pageViewport.height}px`;
      slot.wrap.dataset.loading = "true";
      slot.canvas.width = 0;
      slot.canvas.height = 0;
    }

    const visiblePages = getVisiblePageNumbers();
    await Promise.all(visiblePages.map((pageNum) => renderPage(pageNum)));

    requestAnimationFrame(() => {
      const newScrollHeight = viewportEl.scrollHeight;
      const newClientHeight = viewportEl.clientHeight;
      if (newScrollHeight > newClientHeight) {
        viewportEl.scrollTop = scrollRatio * (newScrollHeight - newClientHeight);
      }
      updatePageLabel();
    });
  }

  function zoomIn() {
    applyScale(scale + ZOOM_STEP);
  }

  function zoomOut() {
    applyScale(scale - ZOOM_STEP);
  }

  async function loadPdf(url) {
    if (pdfUrl === url && pdfDoc) {
      return pdfDoc;
    }
    pdfUrl = url;
    pdfDoc = await PDFJS.getDocument(url).promise;
    return pdfDoc;
  }

  function drawHighlights(pageNum, viewport, overlay) {
    overlay.innerHTML = "";
    const pageHighlights = activeHighlights.filter((item) => item.page === pageNum);

    for (const item of pageHighlights) {
      const [left, bottom, right, top] = item.rect;
      const [x1, y1, x2, y2] = viewport.convertToViewportRectangle([
        left,
        bottom,
        right,
        top,
      ]);

      const highlight = document.createElement("div");
      highlight.className = "pdf-highlight";
      highlight.style.left = `${Math.min(x1, x2)}px`;
      highlight.style.top = `${Math.min(y1, y2)}px`;
      highlight.style.width = `${Math.abs(x2 - x1)}px`;
      highlight.style.height = `${Math.abs(y2 - y1)}px`;
      overlay.appendChild(highlight);
    }
  }

  function clearDocument() {
    observer?.disconnect();
    observer = null;
    viewportEl.removeEventListener("scroll", onViewportScroll);

    for (const slot of pageSlots.values()) {
      slot.renderTask?.cancel();
    }

    pagesEl.innerHTML = "";
    pageSlots.clear();
  }

  async function renderPage(pageNum) {
    const slot = pageSlots.get(pageNum);
    if (!slot || slot.rendered || !pdfDoc) {
      return;
    }

    if (slot.renderTask) {
      slot.renderTask.cancel();
      slot.renderTask = null;
    }

    const page = await pdfDoc.getPage(pageNum);
    const viewport = page.getViewport({ scale });
    const context = slot.canvas.getContext("2d");

    slot.canvas.width = viewport.width;
    slot.canvas.height = viewport.height;
    slot.overlay.style.width = `${viewport.width}px`;
    slot.overlay.style.height = `${viewport.height}px`;
    slot.wrap.style.width = `${viewport.width}px`;
    slot.wrap.style.minHeight = "";
    slot.wrap.removeAttribute("data-loading");

    slot.renderTask = page.render({ canvasContext: context, viewport });
    await slot.renderTask.promise;
    slot.renderTask = null;
    slot.rendered = true;
    slot.viewport = viewport;

    drawHighlights(pageNum, viewport, slot.overlay);
  }

  function createPageSlots(numPages, pageWidth, pageHeight) {
    for (let pageNum = 1; pageNum <= numPages; pageNum += 1) {
      const wrap = document.createElement("div");
      wrap.className = "pdf-page-wrap";
      wrap.dataset.page = String(pageNum);
      wrap.dataset.loading = "true";
      wrap.style.width = `${pageWidth}px`;
      wrap.style.minHeight = `${pageHeight}px`;

      const canvas = document.createElement("canvas");
      const overlay = document.createElement("div");
      overlay.className = "pdf-overlay";

      wrap.appendChild(canvas);
      wrap.appendChild(overlay);
      pagesEl.appendChild(wrap);

      pageSlots.set(pageNum, {
        wrap,
        canvas,
        overlay,
        rendered: false,
        renderTask: null,
        viewport: null,
      });
    }
  }

  function setupObserver() {
    observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            renderPage(Number(entry.target.dataset.page));
          }
        }
      },
      { root: viewportEl, rootMargin: "600px 0px" },
    );

    for (const slot of pageSlots.values()) {
      observer.observe(slot.wrap);
    }
  }

  function updatePageLabel() {
    if (!pdfDoc || !viewportEl) {
      return;
    }

    const viewportRect = viewportEl.getBoundingClientRect();
    const viewportCenter = viewportRect.top + viewportRect.height / 2;

    let closestPage = 1;
    let closestDist = Infinity;

    for (const [pageNum, slot] of pageSlots) {
      const rect = slot.wrap.getBoundingClientRect();
      if (rect.bottom < viewportRect.top || rect.top > viewportRect.bottom) {
        continue;
      }
      const pageCenter = rect.top + rect.height / 2;
      const dist = Math.abs(pageCenter - viewportCenter);
      if (dist < closestDist) {
        closestDist = dist;
        closestPage = pageNum;
      }
    }

    pageLabelEl.textContent = `Page ${closestPage} / ${pdfDoc.numPages}`;
  }

  function onViewportScroll() {
    if (scrollRaf) {
      return;
    }
    scrollRaf = requestAnimationFrame(() => {
      scrollRaf = null;
      updatePageLabel();
    });
  }

  function scrollToPage(pageNum) {
    const slot = pageSlots.get(pageNum);
    if (!slot) {
      return;
    }
    slot.wrap.scrollIntoView({ block: "start" });
    updatePageLabel();
  }

  async function renderNearbyPages(pageNum) {
    const pages = [pageNum, pageNum - 1, pageNum + 1].filter(
      (num) => num >= 1 && num <= (pdfDoc?.numPages || 0),
    );
    await Promise.all(pages.map((num) => renderPage(num)));
  }

  function refreshHighlights() {
    for (const [pageNum, slot] of pageSlots) {
      if (slot.rendered && slot.viewport) {
        drawHighlights(pageNum, slot.viewport, slot.overlay);
      }
    }
  }

  async function buildDocument() {
    clearDocument();

    const firstPage = await pdfDoc.getPage(1);
    const viewport = firstPage.getViewport({ scale });

    createPageSlots(pdfDoc.numPages, viewport.width, viewport.height);
    setupObserver();
    viewportEl.addEventListener("scroll", onViewportScroll, { passive: true });
    updateZoomLabel();
  }

  function openPanel() {
    layoutEl?.classList.remove("pdf-panel-closed");
  }

  function closePanel() {
    layoutEl?.classList.add("pdf-panel-closed");
    document.querySelectorAll(".source-link.active").forEach((el) => {
      el.classList.remove("active");
    });
  }

  async function showSource(source) {
    if (!source?.pdf_url) {
      return;
    }

    openPanel();
    emptyEl.hidden = true;
    pagesEl.hidden = false;
    titleEl.textContent = source.title || source.procedure_title || "Manual";
    metaEl.textContent = source.page_start
      ? `Pages ${source.page_start}${source.page_end !== source.page_start ? `–${source.page_end}` : ""}`
      : `Page ${source.page_number}`;

    const targetPage = source.page_start || source.page_number || 1;
    activeHighlights = source.highlights || [];

    const sameDocument = pdfUrl === source.pdf_url && pdfDoc && pageSlots.size > 0;

    if (!sameDocument) {
      await loadPdf(source.pdf_url);
      await buildDocument();
    } else {
      refreshHighlights();
    }

    await renderNearbyPages(targetPage);
    scrollToPage(targetPage);
  }

  closeBtn?.addEventListener("click", closePanel);
  zoomInBtn?.addEventListener("click", zoomIn);
  zoomOutBtn?.addEventListener("click", zoomOut);
  updateZoomLabel();

  return { showSource, openPanel, closePanel };
}
