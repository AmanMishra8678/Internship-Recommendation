/**
 * app.js — InternIQ Frontend Logic
 * Handles: form interactions, API calls, resume upload, card rendering,
 * background canvas animation, and score ring animations.
 */

"use strict";

/* ──────────────────────────────────────────────────────────────
   CONSTANTS
   ────────────────────────────────────────────────────────────── */
const API_URL = "https://internship-recommendation-zlwk.onrender.com/api/recommend";
/* ──────────────────────────────────────────────────────────────
   DOM REFERENCES
   ────────────────────────────────────────────────────────────── */
const $ = (sel) => document.querySelector(sel);

const skillsInput       = $("#skills-input");
const interestsInput    = $("#interests-input");
const locationFilter    = $("#location-filter");
const domainFilter      = $("#domain-filter");
const topkSlider        = $("#topk-slider");
const topkDisplay       = $("#topk-display");
const recommendBtn      = $("#recommend-btn");
const uploadZone        = $("#upload-zone");
const fileInput         = $("#file-input");
const uploadStatus      = $("#upload-status");
const extractedWrap     = $("#extracted-skills-wrap");
const extractedPills    = $("#extracted-skills-pills");
const resultsPlaceholder = $("#results-placeholder");
const resultsLoading    = $("#results-loading");
const resultsHeader     = $("#results-header");
const resultsCount      = $("#results-count");
const cardsContainer    = $("#cards-container");
const cardTemplate      = $("#card-template");
const toast             = $("#toast");

/* ──────────────────────────────────────────────────────────────
   STATE
   ────────────────────────────────────────────────────────────── */
let resumeText       = "";   // raw text extracted from uploaded PDF
let uploadedSkills   = [];   // skills detected from resume
let toastTimeout     = null;

/* ──────────────────────────────────────────────────────────────
   BACKGROUND CANVAS — animated particle mesh
   ────────────────────────────────────────────────────────────── */
(function initCanvas() {
  const canvas = $("#bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const PARTICLE_COUNT = 55;
  const CONNECT_DIST   = 160;
  const SPEED          = 0.35;

  let W, H, particles = [];

  class Particle {
    constructor() { this.reset(true); }
    reset(init = false) {
      this.x  = Math.random() * W;
      this.y  = init ? Math.random() * H : -8;
      this.vx = (Math.random() - 0.5) * SPEED;
      this.vy = (Math.random() - 0.5) * SPEED;
      this.r  = Math.random() * 1.5 + 0.5;
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > W) this.vx *= -1;
      if (this.y < 0 || this.y > H) this.vy *= -1;
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(245,166,35,0.45)";
      ctx.fill();
    }
  }

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function createParticles() {
    particles = Array.from({ length: PARTICLE_COUNT }, () => new Particle());
  }

  function drawConnections() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < CONNECT_DIST) {
          const alpha = (1 - dist / CONNECT_DIST) * 0.18;
          ctx.strokeStyle = `rgba(245,166,35,${alpha})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
  }

  function frame() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => { p.update(); p.draw(); });
    drawConnections();
    requestAnimationFrame(frame);
  }

  window.addEventListener("resize", () => { resize(); });
  resize();
  createParticles();
  frame();
})();

/* ──────────────────────────────────────────────────────────────
   LOAD METADATA (filter dropdown options)
   ────────────────────────────────────────────────────────────── */
async function loadMetadata() {
  try {
    const res  = await fetch(`${API_BASE}/api/metadata`);
    const data = await res.json();

    data.locations.forEach(loc => {
      const opt = document.createElement("option");
      opt.value = loc;
      opt.textContent = loc;
      locationFilter.appendChild(opt);
    });

    data.domains.forEach(dom => {
      const opt = document.createElement("option");
      opt.value = dom;
      opt.textContent = dom;
      domainFilter.appendChild(opt);
    });
  } catch (e) {
    console.warn("Could not load metadata — backend may not be running.", e);
  }
}

/* ──────────────────────────────────────────────────────────────
   TOP-K SLIDER
   ────────────────────────────────────────────────────────────── */
topkSlider.addEventListener("input", () => {
  topkDisplay.textContent = topkSlider.value;
});

/* ──────────────────────────────────────────────────────────────
   RESUME UPLOAD
   ────────────────────────────────────────────────────────────── */
uploadZone.addEventListener("click", (e) => {
  if (e.target !== fileInput) fileInput.click();
});

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("drag-over");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFileUpload(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFileUpload(fileInput.files[0]);
});

async function handleFileUpload(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    showToast("Please upload a PDF file.", "error");
    return;
  }

  setUploadStatus("⏳ Parsing résumé…", "");
  uploadZone.classList.remove("has-file");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res  = await fetch(`${API_BASE}/api/upload_resume`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (!res.ok) {
      setUploadStatus(`⚠️ ${data.detail || "Upload failed."}`, "var(--rose)");
      return;
    }

    resumeText     = data.raw_text_preview || "";
    uploadedSkills = data.extracted_skills || [];

    // Pre-fill skills input if empty
    if (!skillsInput.value.trim() && uploadedSkills.length > 0) {
      skillsInput.value = uploadedSkills.join(", ");
    }

    // Show extracted skills pills
    renderExtractedSkills(uploadedSkills);

    setUploadStatus(`✅ ${file.name} — ${data.word_count} words, ${uploadedSkills.length} skills detected`, "var(--green)");
    uploadZone.classList.add("has-file");
    showToast(`Résumé uploaded! ${uploadedSkills.length} skills extracted.`, "success");

  } catch (e) {
    setUploadStatus("⚠️ Could not connect to server.", "var(--rose)");
    showToast("Upload failed — is the backend running?", "error");
  }
}

function setUploadStatus(msg, color) {
  uploadStatus.textContent = msg;
  uploadStatus.style.color = color || "var(--text-2)";
}

function renderExtractedSkills(skills) {
  if (!skills || skills.length === 0) {
    extractedWrap.style.display = "none";
    return;
  }
  extractedPills.innerHTML = "";
  skills.forEach(skill => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = skill;
    extractedPills.appendChild(pill);
  });
  extractedWrap.style.display = "block";
}

/* ──────────────────────────────────────────────────────────────
   RECOMMEND
   ────────────────────────────────────────────────────────────── */
recommendBtn.addEventListener("click", getRecommendations);

async function getRecommendations() {
  const skills    = skillsInput.value.trim();
  const interests = interestsInput.value.trim();

  if (!skills && !interests && !resumeText) {
    showToast("Please enter your skills or interests first!", "error");
    skillsInput.focus();
    return;
  }

  setLoadingState(true);

  const payload = {
    skills,
    interests,
    resume_text: resumeText,
    location: locationFilter.value,
    domain:   domainFilter.value,
    top_k:    parseInt(topkSlider.value, 10),
  };

  try {
    const res  = await fetch(`${API_BASE}/api/recommend`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      showToast(data.detail || "Something went wrong.", "error");
      setLoadingState(false);
      return;
    }

    renderResults(data.recommendations);

  } catch (e) {
    showToast("Cannot reach the server. Is the backend running?", "error");
    setLoadingState(false);
  }
}

/* ──────────────────────────────────────────────────────────────
   RENDER RESULTS
   ────────────────────────────────────────────────────────────── */
function renderResults(recommendations) {
  setLoadingState(false);

  cardsContainer.innerHTML = "";

  if (!recommendations || recommendations.length === 0) {
    showSection("placeholder");
    showToast("No matches found. Try broader skills or remove filters.", "error");
    return;
  }

  showSection("results");
  resultsCount.textContent = `${recommendations.length} result${recommendations.length > 1 ? "s" : ""}`;

  const userSkills = skillsInput.value.toLowerCase().split(/[\s,]+/).filter(Boolean);

  recommendations.forEach((item, idx) => {
    const card = buildCard(item, userSkills, idx);
    cardsContainer.appendChild(card);
  });

  // Trigger score ring animations after cards are in DOM
  requestAnimationFrame(() => animateScoreRings());
}

function buildCard(data, userSkills, idx) {
  const frag = cardTemplate.content.cloneNode(true);
  const card = frag.querySelector(".card");

  card.dataset.grade = data.match_grade || "Fair";
  card.style.animationDelay = `${idx * 0.07}s`;

  card.querySelector(".card-domain").textContent   = data.domain || "";
  card.querySelector(".card-location").textContent = `📍 ${data.location}`;
  card.querySelector(".card-title").textContent    = data.title;
  card.querySelector(".card-company").textContent  = `🏢 ${data.company}`;

  // Badges
  card.querySelector(".badge-duration").textContent = `⏱ ${data.duration}`;
  card.querySelector(".badge-stipend").textContent  = `💰 ${data.stipend}`;
  const gradeBadge = card.querySelector(".badge-grade");
  gradeBadge.textContent        = data.match_grade;
  gradeBadge.dataset.grade      = data.match_grade;

  // Score ring
  const score = data.relevance_score || 0;
  card.querySelector(".score-label").textContent = `${score.toFixed(0)}%`;
  const ringFill = card.querySelector(".ring-fill");
  const circumference = 2 * Math.PI * 15.9;
  ringFill.style.strokeDasharray  = `${circumference}`;
  ringFill.style.strokeDashoffset = `${circumference}`;    // start at 0
  ringFill.dataset.target = circumference - (score / 100) * circumference;

  // Ring colour by grade
  const ringColours = {
    Excellent: "#4ade80",
    Strong:    "#f5a623",
    Good:      "#60a5fa",
    Fair:      "#9c9a96",
    Low:       "#5c5a57",
  };
  ringFill.style.stroke = ringColours[data.match_grade] || "#f5a623";

  // Skills pills
  const skillsWrap = card.querySelector(".skills-pills");
  const requiredSkills = Array.isArray(data.required_skills) ? data.required_skills.map(s => String(s).trim()).filter(Boolean) : (data.required_skills || "").split(",").map(s => s.trim()).filter(Boolean);
  requiredSkills.slice(0, 10).forEach(skill => {
    const pill = document.createElement("span");
    pill.className = "skill-pill";
    const isMatched = userSkills.some(us => skill.toLowerCase().includes(us) || us.includes(skill.toLowerCase()));
    if (isMatched) pill.classList.add("matched");
    pill.textContent = skill;
    skillsWrap.appendChild(pill);
  });

  // Explanation (parse **bold** and *italic*)
  const explanationEl = card.querySelector(".explanation-text");
  explanationEl.innerHTML = parseMarkdown(data.explanation || "No explanation available.");

  return frag;
}

function parseMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

function animateScoreRings() {
  document.querySelectorAll(".ring-fill").forEach(ring => {
    const target = parseFloat(ring.dataset.target || 0);
    setTimeout(() => {
      ring.style.strokeDashoffset = target;
    }, 100);
  });
}

/* ──────────────────────────────────────────────────────────────
   UI STATE HELPERS
   ────────────────────────────────────────────────────────────── */
function setLoadingState(loading) {
  recommendBtn.disabled = loading;
  recommendBtn.querySelector(".btn-text").textContent = loading
    ? "Analysing…"
    : "Get Recommendations";

  if (loading) {
    showSection("loading");
  }
}

function showSection(section) {
  resultsPlaceholder.style.display = section === "placeholder" ? "" : "none";
  resultsLoading.style.display     = section === "loading"     ? "" : "none";
  resultsHeader.style.display      = section === "results"     ? "" : "none";
  cardsContainer.style.display     = section === "results"     ? "" : "none";
}

/* ──────────────────────────────────────────────────────────────
   TOAST NOTIFICATIONS
   ────────────────────────────────────────────────────────────── */
function showToast(message, type = "info") {
  clearTimeout(toastTimeout);
  toast.textContent  = message;
  toast.className    = `toast toast-${type} show`;
  toastTimeout = setTimeout(() => {
    toast.classList.remove("show");
  }, 3500);
}

/* ──────────────────────────────────────────────────────────────
   KEYBOARD SHORTCUT — Ctrl+Enter to recommend
   ────────────────────────────────────────────────────────────── */
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    getRecommendations();
  }
});

/* ──────────────────────────────────────────────────────────────
   INIT
   ────────────────────────────────────────────────────────────── */
loadMetadata();
showSection("placeholder");
