let questions = [];
let currentIndex = 0;
let answers = {};
let skipped = {};

fetch("/static/techmcq/sde_mcq_275_unique.json")
  .then(res => res.json())
  .then(data => {
    questions = shuffleArray(data).slice(0, 10);
    generateSidebar();
    renderQuestion();
  });

function shuffleArray(array) {
  let a = array.slice();
  for (let i = a.length - 1; i > 0; i--) {
    let j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function generateSidebar() {
  const sidebar = document.getElementById("sidebar");
  sidebar.innerHTML = "";
  for (let i = 0; i < questions.length; i++) {
    const btn = document.createElement("button");
    btn.innerText = i + 1;
    btn.onclick = () => {
      currentIndex = i;
      renderQuestion();
    };
    sidebar.appendChild(btn);
  }
}

function renderQuestion() {
  const q = questions[currentIndex];
  document.getElementById("question-number").innerText = `Question ${currentIndex + 1} of ${questions.length}`;
  document.getElementById("question-text").innerText = q.question;

  const optionsDiv = document.getElementById("options");
  optionsDiv.innerHTML = "";

  for (let [key, value] of Object.entries(q.options)) {
    const checked = answers[currentIndex] === key ? "checked" : "";
    optionsDiv.innerHTML += `<label><input type='radio' name='option' value='${key}' ${checked} onchange='selectOption("${key}")'> ${value}</label>`;
  }

  updateSidebarColors();
  updateNavButtons();
}

function selectOption(opt) {
  answers[currentIndex] = opt;
  delete skipped[currentIndex]; // unmark skip if answered
  updateSidebarColors();
}

function updateSidebarColors() {
  const buttons = document.getElementById("sidebar").children;
  for (let i = 0; i < buttons.length; i++) {
    if (answers[i]) {
      buttons[i].style.backgroundColor = "#28a745"; // green
    } else if (skipped[i]) {
      buttons[i].style.backgroundColor = "#dc3545"; // red
    } else {
      buttons[i].style.backgroundColor = "#888";    // gray
    }
  }
}

function nextQuestion() {
  if (currentIndex < questions.length - 1) {
    currentIndex++;
    renderQuestion();
  }
}

function prevQuestion() {
  if (currentIndex > 0) {
    currentIndex--;
    renderQuestion();
  }
}

function skipQuestion() {
  delete answers[currentIndex];
  skipped[currentIndex] = true;
  updateSidebarColors();
  nextQuestion();
}

function updateNavButtons() {
  const nav = document.getElementById("nav-buttons");
  nav.innerHTML = "";

  const prevBtn = document.createElement("button");
  prevBtn.className = "nav";
  prevBtn.innerText = "Previous";
  prevBtn.onclick = prevQuestion;

  const skipBtn = document.createElement("button");
  skipBtn.className = "nav";
  skipBtn.innerText = "Skip";
  skipBtn.onclick = skipQuestion;

  nav.appendChild(prevBtn);
  nav.appendChild(skipBtn);

  if (currentIndex < questions.length - 1) {
    const nextBtn = document.createElement("button");
    nextBtn.className = "nav";
    nextBtn.innerText = "Next";
    nextBtn.onclick = nextQuestion;
    nav.appendChild(nextBtn);
  } else {
    const submitBtn = document.createElement("button");
    submitBtn.className = "nav";
    submitBtn.innerText = "Submit";
    submitBtn.onclick = submitQuiz;
    nav.appendChild(submitBtn);
  }
}

function submitQuiz() {
  const unanswered = questions.filter((_, idx) => !(idx in answers));
  if (unanswered.length > 0) {
    alert(`You have ${unanswered.length} unanswered question(s). Please complete them before submitting.`);
    return;
  }

  let score = 0;
  const resultDiv = document.getElementById("results");
  document.getElementById("quiz-box").style.display = "none";
  resultDiv.innerHTML = `<div id="score"></div><h2>Quiz Results</h2>`;

  questions.forEach((q, idx) => {
    const userAns = answers[idx];
    const correctAns = q.answer;
    const isCorrect = userAns === correctAns;
    if (isCorrect) score++;

    const div = document.createElement("div");
    div.className = "result-question";
    div.innerHTML = `
      <div><strong>Q${idx + 1}:</strong> ${q.question}</div>
      <div>Your Answer: <span class="${isCorrect ? 'correct' : 'wrong'}">${q.options[userAns] || 'Not Answered'}</span></div>
      <div>Correct Answer: <strong>${q.options[correctAns]}</strong></div>
    `;
    resultDiv.appendChild(div);
  });

  document.getElementById("score").innerText = `Your Score: ${score} / ${questions.length}`;
  resultDiv.style.display = "block";

  const retryBtn = document.createElement("button");
  retryBtn.innerText = "Retry Quiz";
  retryBtn.className = "nav";
  retryBtn.style.display = "block";
  retryBtn.style.margin = "2rem auto";
  retryBtn.onclick = () => location.reload();
  resultDiv.appendChild(retryBtn);
}
